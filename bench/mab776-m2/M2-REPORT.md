# M2 - Strata-DS4 on our UD-IQ2_M vs llama.cpp stack 73 (GO/NO-GO measurement, 2026-10-09)

Engine: Strata-DS4 fork `mab776-main` (upstream 924516a + M1 byte-sized slots + M3 serve graft), `ds4_generate`
built with **CUDA 13.0.2** (`build-ds4-cu13`, sm_120a; upstream warns 12.8 crashes long prompts on sm_120).
GPU 0 only (RTX PRO 5000, 48 GB); GPU 1 idle. Container RAM cap = MemAvailable - 9 GiB (host keeps >= 8 GiB free).
Flags (all runs unless noted): `--ctx 262144 --comp-type iq4_nl --icomp-q8 --slots auto --vram-margin 2.0 --pcie 0.55
--vram-lru --pf-b 0.7 --arena-gib <cap-6> --arena-skip-resident --prefill-chunk 4096 --chunk-prestage --temp 0 --stop 1
--profile ds4routes.bin` (**no `--chunk-mmq`: it gives NaN on our file**, see below). Prompts = the exact
`deepseek-v4/lcpp_depths.py` texts (241/3825/7529 paragraphs + the 3 code tails), tokenized with our tokenizer
(= llama-tokenize b11429, M3 parity), 512 tokens out. Scripts: `make_prompts.py`, `m2run.sh`, `chain-*.sh`, logs in `out/`.

## Results

| depth | stack 73 prefill (no draft / draft) | DS4 prefill (one-shot) | stack 73 decode no draft | stack 73 code decode w/ drafter | DS4 code decode (code1 / 2 / 3, mean) | DS4 peak RSS / cgroup | DS4 peak VRAM | min host avail |
|---|---|---|---|---|---|---|---|---|
| 8k (8,231-8,235) | 1011 / 921 | 143 (141-144) | 28.0 | 40.8 | 32.8 / 35.3 / 38.2 = **35.4** | 44-48 / 45-46 GiB | 46.9 GB | 11.1 GiB |
| 127k (130,088-130,093) | 843 / 793 | 210 (209.9-211.0) | 25.0 | 39.0 | 33.9 / 31.3 / 36.1 = **33.8** | 47.5-47.8 / 43.1-45.0 GiB | 47.0 GB | 10.8 GiB |
| 250k (256,028) | 693 / 630 | 191 (code1) | 22.3 | ~36 | 29.5 (code1) | 48.4 / 45.0 GiB | 47.0 GB | 10.2 GiB |
| server-mode prefill (seed-first, chunk 2048) | | 617 @8k / 439 @127k / 256-291 @251k (needles) | | | (decode 21-24: smaller cache) | | | |

Probes (`--pos-offset`, empty caches, 16-token prompt, 64 tokens, arena 36 GiB, CPU busy with other jobs):
130k 23.3 tok/s, 255k 23.7 tok/s; attention+router 14.7 -> 15.1 ms (flat: sparse top-512 indexer).

Needles at 251k (stack 73: 3/3), server-mode prefill (seed-first, chunk 2048, margin 9), 251,198-251,199 ids:
10 % **FOUND** ("meadow-copper-504", after a 286-char think; prefill 291 tok/s, 865 s); 50 % **FOUND** ("falcon-quartz-940", 341-char think; 267 tok/s, 940 s); 90 % **FOUND** ("willow-cobalt-696"; 256 tok/s, 980 s) = **3/3**.
The first 250k attempt crashed: ggml-cuda's pad kernel puts the key count in gridDim.z (max 65,535) and past ~131k tokens
the compressed cache holds 65,536 rows -> "PAD failed: invalid argument" (exit 139) 10 min into the prefill. Fixed in
7ffee699 (pad the flat buffer past 65,535 keys).
Quality - ppl on 768 ids (BOS + p600 + p3000[:167]), same ids both engines:
llama.cpp b11429 (stack 73 placement, q8_0 KV) **9.9624**; DS4 decode-loop **9.8681 (-0.95 %)**; DS4 chunked w/o MMQ
**9.8737 (-0.89 %)**; DS4 chunked WITH `--chunk-mmq` = **NaN** (bug on our quant types; prestage irrelevant).

## Prefill: disk-bound vs compute-bound (MIMO_CHUNK_PROF + the engine's phase split)
| depth | prefill wall | NVMe waits (chunks + post-prompt cache seed) | dense attention+router | GPU expert kernels (overlapped) | NVMe volume / rate |
|---|---|---|---|---|---|
| 8k code1-3 | 57-58 s | 29.6-30.3 + 15.0-15.4 s = **77-78 %** | 6.0 s (10 %) | 8.4-8.6 s | 76-78 GB at 2.56-2.57 GB/s |
| 127k code1 | 620 s | 323.7 + 15.7 s = **55 %** | 201.5 s (33 %) | 138 s | 845 GB at 2.60 GB/s |
| 250k code1 | 1338 s | 628.0 + 15.5 s = **48 %** | 626.9 s (47 %) | 49 s (MMQ) | 1,654 GB at 2.62 GB/s |
Why: in one-shot mode the VRAM cache is seeded only AFTER the prompt, so each 4096-token chunk re-reads the ~37 GiB
VRAM-destined set (+ the small file tier) from NVMe (O_DIRECT, one blob at a time, ~2.6 GB/s); the dense attention is
flash-attn over the WHOLE compressed cache per chunk (O(cap x n)), which at 127k alone (202 s) exceeds llama.cpp's whole
154 s prefill - that part is an upstream engine limit (sparse prefill attention is not implemented).

## Where the time goes (8k code)
- decode ms/token: attention+router ~13.4, experts 10-14 (VRAM hit 89-90 %, CPU pool 4-5 %, PCIe 5 %), file-tier reads
  1.8-5.6 ms/pass (experts in neither VRAM nor the 39-40 GiB arena: RAM 61.9 GiB < experts 78.1 GiB).
- prefill (one-shot `ds4_generate`): the VRAM cache is only seeded AFTER the prompt, so every 4096-token chunk streams
  ~38 GiB of experts from NVMe at ~2.6 GB/s (O_DIRECT per blob) + the post-prompt cache seed (~15 s).
  A resident server seeds the cache once; see "server-mode prefill" below.

## Server-mode prefill (DS4_SEED_FIRST=1: cache seeded at load, chunks compute cached experts in place)
A resident server keeps the expert cache across requests, so a prompt's chunks compute the VRAM-resident experts in
place and only stream the rest (arena DMA + the file tier). `ds4_generate` defers the seed by design (one-shot), so
`DS4_SEED_FIRST=1` (mab776 hook) reproduces the server case; the chunk buffers then need VRAM beside the cache:
| 8k code1 | prefill | decode | VRAM cache | NVMe during prefill | decode file reads |
|---|---|---|---|---|---|
| one-shot, margin 2.0, chunk 4096 (MMQ) | 153 tok/s | 35.7 tok/s | 5,217 experts (37 GiB) | 77.9 GB + 15 s seed | 3.3 ms/pass |
| seed-first, margin 8, chunk 2048 (MMQ) | **617 tok/s** | 21.0 tok/s | 4,368 experts (31 GiB) | 12.0 GB, no seed | 21.3 ms/pass |
| **127k code1** one-shot, margin 2.0, chunk 4096 (no MMQ) | 210 tok/s | 33.9 tok/s | 5,217 experts | 845 GB + 16 s seed | 3.6 ms/pass |
| **127k code1** seed-first, margin 8.5, chunk 2048 (MMQ) | **439 tok/s** | 23.8 tok/s | 4,297 experts | 193 GB, no seed | 14.7 ms/pass |
(margin 7.5 + chunk 4096 OOMed at the first chunk: its graph alone wanted 4.35 GiB.) The 8 GiB margin sits idle during
decode (7.57 GiB free at the end) and the smaller cache pushes 6 GiB of experts to NVMe -> decode collapses. **M6 needs
an elastic cache** (lend the cache tail to the prompt and take it back: upstream Strata's prompt-path loan /
Strata-GLM's grow_cache + a shrink), not a fixed margin; with it a server gets ~600 tok/s prefill @8k AND ~35 tok/s
decode. At 127k the server-mode prefill is 439 tok/s = 52 % of llama.cpp's 843: dense attention 174 s of 296 s (the
O(cap x n) flash-attn over the whole compressed cache - alone more than llama.cpp's whole 154 s prefill), disk waits
83 s (193 GB at 2.3 GB/s), resident-expert MMVQ kernels 59 s.

## --chunk-mmq NaN: found + fixed (52b44a2c)
On sm_120a llama.cpp's MMQ multiplies MXFP4 FP4 x FP4 and expects FP4 activations; Strata always wrote q8_1 -> NaN in
our two MXFP4-down layers. Default now keeps those 2 layers on MMVQ (ppl 9.8967, -0.66 % vs llama.cpp);
`STRATA_MMQ_FP4=1` = FP4 activations like llama.cpp (ppl 9.9352, -0.27 %). Expert kernels 8.4 s -> 1.7 s @8k, but the
one-shot prefill is disk-bound: 143 -> 153 tok/s.

## Largest arena that keeps >= 8 GiB free (Marc's ask)
MEM 46g / arena 40.5 GiB (5,711 experts, file tier 0 at load; min host avail 10.3 GiB): 8k code1 decode 34.2 tok/s vs
35.7 with arena 39 GiB (noise) - decode still did 4.2 ms/pass of file reads, because with --arena-skip-resident a
VRAM-LRU eviction has no arena copy to fall back on. Prefill 155.6 vs 152.8 (the VRAM set is still read from NVMe per
chunk in one-shot mode). So RAM at today's 61.9 GiB is not the lever; 128 GB is (whole IQ2_M in the arena, no skip).

## Projections (estimates, not measurements - each lever is a planned milestone)
Decode @127k today: 33.8 tok/s = ~29.6 ms/token (attention+router 14.9, experts 11.3 of which file reads 1.9-5.6 ms/pass
and CPU pool ~5, predict/finish/head ~3).
| lever | what it removes | decode @127k (est.) | server prefill @127k (est.) |
|---|---|---|---|
| today (GPU 0, 61.9 GiB) | - | 33.8 | 439 (but needs the cache shrunk -> decode 24) |
| M6 elastic cache (lend the cache tail to the prompt) | the decode/prefill trade-off | ~33.8 | ~440 |
| M7a GPU 1 as a 2nd expert tier (+~1,900 experts, 14 GiB) | file tier -> 0 (VRAM 51 + arena 41 > 78 GiB), fewer CPU misses | ~38-40 | ~550-650 (no NVMe: 83 of 296 s were disk waits) |
| 128 GB RAM (whole IQ2_M in the arena, no --arena-skip-resident) | disk in both paths | ~37-39 | ~550-650 |
| M7b DSpark drafter on GPU 1 (stack 73 got x1.56 at 127k) | - | x1.2-1.5 -> ~45-55 **if** the verify pass gets a sparse per-query attention path (today n>1 passes use the masked O(cap) path, expensive at depth) | - |
| prefill attention (sparse/top-k prefill attention, upstream DS4 is on it) | the 174-202 s O(cap x n) flash-attn @127k | - | toward llama.cpp's 843 |
Bottom line: DS4 already beats stack 73 like-for-like on decode at every measured depth; with the drafter stack 73
(39.0 @127k) stays ahead until M7b; prefill stays behind llama.cpp until the attention work, and is disk-bound today.

## Decision gate (CORRECTED by Marc 2026-10-09 ~23:00, via the DeepSeek-V4-Flash session)
The original gate compared DS4 (GPU 0 only, no drafter) with stack 73 WITH its DSpark drafter on BOTH GPUs - apples to
oranges. Now: compare like for like with stack 73 WITHOUT drafter (decode 28.0 / 25.0 / 22.3, prefill 1011 / 843 / 693
at 8k / 127k / 250k), and report a projection with the M7 levers (DSpark drafter ~x1.45 on stack 73; GPU 1's idle
16 GB as drafter host and/or extra expert tier). Hard stop only for correctness: a failed needle, ppl > 2 % off, or
decode @127k clearly below 25 with no credible M7 path. Otherwise report and Marc decides. M7 = planned milestone after
M6. Prefill is "not final" on this box (disk-bound: GPU 0 only + 61 GiB RAM < 78 GiB of experts); project GPU 1 as an
expert tier and the 128 GB RAM upgrade.

### VERDICT: **GO** (Marc's rule: GO if the 3 needles pass)
- Correctness: needles **3/3** at 251k; ppl -0.3 to -0.9 % vs llama.cpp on the same ids (bar 2 %); coherent code at
  every depth (decoded and read). Two engine bugs found on the way and fixed (chunk-MMQ NaN, PAD past 65,535 keys).
- Like-for-like decode (both without drafter): **+26 % / +35 % / +32 %** at 8k / 127k / 250k (35.4 / 33.8 / 29.5 vs
  28.0 / 25.0 / 22.3). Stack 73 WITH its drafter (40.8 / 39.0 / ~36) stays ahead until M7b.
- Prefill: behind (one-shot 153 / 210 / 191 vs 1011 / 843 / 693; server-mode 617 / 439 / ~270) and NOT final: about half
  is NVMe waits (remove with M7a GPU 1 tier / 128 GB RAM), the other half is the O(cap x n) prefill attention (upstream).
- Next (Marc): M7a GPU 1 expert tier -> re-measure; M7b drafter; prefill attention; M4-M6 on CPU in parallel.
