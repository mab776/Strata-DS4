# M7a - GPU 1 as a static expert tier + elastic GPU 0 cache (2026-10-10)

Same harness as M2 (`m2run.sh`, ctx 262,144, `--comp-type iq4_nl --icomp-q8`, `--vram-lru --pf-b 0.7 --pcie 0.55
--arena-skip-resident`, chunk 2048 + `--chunk-mmq --chunk-prestage`, greedy, 512 tokens out, CUDA 13 build).
M7a config: `G1=1 GROW=2.0 MARGIN=9.0` = `--gpu1 1 --vram-grow 2.0 --vram-margin 9.0` (both cards visible, PCI bus
order). Host RAM cap MEM 42g (arena cap 36) - 4 GiB LESS than the M2 runs had (46g / 40-41).

## What changed (tools/ds4/ds4_moe.{hpp,cpp}, ds4_generate.cpp)
- **Phase 0, complete ranking.** The profile's train half never routes 80 experts (0.58 GiB); they had no tier and
  were read from NVMe on every use (B8: 538 file reads in 472 passes at "file tier 0"). Now appended to the ranking.
- **GPU 1 tier** (`--gpu1 N`). The ranks right after GPU 0's decode-budget head, as many as fit (2,045 experts =
  14.5 GiB with the chunk buffers; 2,063 without), no arena copies (exclusive tiers). Decode/verify: the 16 KB
  activation goes over, the routed rows come back through pinned memory (no P2P; GPU 1 is PCIe 4.0 x4 behind the
  chipset). Prompt chunks: GPU 1's experts take the sorted entry range [0, n1), GPU 1 computes their rows from the
  chunk's activations, GPU 0 waits on its event (device side) and scatters the rows into c_parts before the mix.
  Same kernels and q8_1 bytes as GPU 0.
- **Elastic GPU 0 cache** (`--vram-grow G`, Strata-GLM's): opened in 64 MiB VMM segments for the decode budget,
  mapped and seeded only up to the prompt budget (`--vram-margin`), grown after the prompt from the arena (which
  keeps that grow set) to leave G GiB. Seed-first: nothing waits for a post-prompt seed from NVMe any more.
- **Prestage halves sized to the busiest layer's arena share** (138 experts -> 1.05 GiB a half instead of 256 x
  (blob + 1 MiB) = 2.6 GiB): the first seed-first + GPU 1 run OOMed in layer 3's chunk graph (the trace showed the
  tier's chunk buffers taking 6.5 GiB of the 8.6 GiB left for the prompt).
- `DS4_CHECK_G1=1`: every GPU-1 decode row and the first GPU-1 expert of every prompt call recomputed on GPU 0 from
  the file's bytes and compared byte for byte. `DS4_VRAM_TRACE=1` now traces chunked prefill too.

## Correctness
- **GPU 1 rows == GPU 0 rows, bit for bit:** decode 24,518 / 24,518 (ppl768 decode loop), prompt 671 / 671
  (ppl768 chunked). A plain on/off ppl A/B cannot show it on this box: without GPU 1 the arena cannot hold every
  non-VRAM expert under the RAM rule, and file-tier experts are computed on the CPU (Q8_K) instead of the GPU (q8_1).
- ppl768 (same ids as M2): decode loop **9.8654** (M2 9.8681), chunked + MMQ **9.8678** (M2 9.8967); llama.cpp 9.9624.
- Needle at 251,199 tokens, 50 % depth: **FOUND** ("falcon-quartz-940" after a 283-char think; prefill 323 tok/s vs
  M2's 267, decode 38.3 tok/s over the 78-token answer). The 10 % / 90 % needles were not re-run: GPU-1 rows are
  byte-identical to GPU 0's, so placement cannot change what attention sees.

## Results (code1)

| depth | M7a prefill | M2 seed-first prefill | M2 one-shot prefill | stack 73 prefill (no draft) | M7a decode | M2 decode (one-shot) | stack 73 decode no draft / w/ drafter |
|---|---|---|---|---|---|---|---|
| 8k | **683** | 617 (decode then 21.0) | 153 | 1011 | **44.4** | 35.7 | 28.0 / 40.8 |
| 127k | **495** | 439 (decode then 23.8) | 210 | 843 | **42.0** | 33.9 | 25.0 / 39.0 |
| 250k | **347** | 256-291 (needles) | 191 | 693 | **40.6** | 29.5 | 22.3 / ~36 |

- **NVMe during the prompt: 0.0 GB** (M2: 12 GB at 8k / 193 GB at 127k seed-first; 76-845 GB one-shot).
  **Decode file reads: 0.00 ms/pass** (M2: 3.3-5.6 one-shot, 21-15 seed-first).
- Decode experts 5.8 / 5.7 ms per token (M2 11.3); attention+router unchanged (13.3 / 14.7 ms) - now 60-65 % of a
  token. GPU 1 serves 14-15 % of lookups at 0.02 ms of host wait per pass (never on the critical path); CPU pool
  0.8 ms (2 % of lookups), PCIe share 2.3 %. Grow after the prompt: +902-929 slots in 120 ms, 0 from the file.
- Host RAM: cgroup peak 37 GiB (M2 43-46), min host avail 15 GiB (M2 10-11). VRAM free at the end 2.0 GiB.
- Prefill at 127k: dense 177 s (attention 175 s - the O(cap x n) prompt attention, 67 % of the prefill), experts
  85 s of which resident MMVQ 59 s (GPU 0's resident experts still run MMVQ, not MMQ). At 250k: dense 571 s
  (attention 567 s = **77 %** of the 738 s), experts 166 s (resident MMVQ 115 s).
- vs stack 73: decode **+59 % / +68 % / +82 %** without its drafter and **+9 % / +8 % / +13 %** against stack 73
  WITH its DSpark drafter; prefill **68 % / 59 % / 50 %** of stack 73's.

## Decode host overhead (A/B in one binary, 8k, DS4_TIER_PROF=1, output ids identical in every arm)
| arm | tier us/layer | sum | gpu launch | layer sync |
|---|---|---|---|---|
| all old paths | 156 | 23 | 14 | 63 (+11 second sync) |
| sum k-outer over the whole 4096-double row | 167 | 37 | 9 | 76 |
| L1-tiled k-outer (512 doubles) | 149 | 19 | 10 | 77 |
| **rows/weights picked once per layer, i-outer** (kept) | **132** | **7** | 8 | 77 |
Kept: the per-layer row pick (7 us), one packed pinned metadata copy per launch, one sync per layer (the readback
queued before it - neutral), the CPU Q8_K quantization only when the pool has work. "Layer sync" (~77 us) is GPU
time (hits ~49 us + PCIe share ~28 us per layer, serial on one stream) - overlapping the PCIe DMA with the hits
kernel is the next decode lever. Run-to-run noise ~1 ms/token while the M4/M5 agents' builds share the CPU.

## What is left / next levers
- **Prefill attention** (175 s at 127k): the dense O(cap x n) flash-attn in chunk mode - upstream territory.
- **Resident experts on MMQ** (59 s at 127k, 3.8 s at 8k): needs a pointer-array MMQ (slots are not strided).
- **M7b drafter**: M2's CUDA verify was slower than plain (18.9-20.6 vs 24.9) because a 2-token pass read 2.2x the
  experts on the CPU; with M7a a pass's experts are ~6 ms, so the MTP draft cost (9 ms on the CPU) dominates -
  MTP experts on GPU 1 is the obvious next step.
- Per-layer host overhead in decode (~130 us/layer: activation, launch, two syncs, D2H, sum): design A's small fixes
  (one sync, vectorised sum, packed metadata) ~0.5-1 ms/token.
