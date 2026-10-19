# V4.1 overnight speed work (2026-10-15, 23:45 - 07:00)

Engine: upstream Strata-DSV4.1 `ds41_generate`, worktree `strata-ds41/` branch `mab776-v41` (LOCAL commits only, not
pushed).  Test containers only (`ds41-pf` :8332; `ds41-test` :8331 for the tools re-test), GPU 0.  Bench:
`strata-coder/bench/strata_bench.py`, same corpus offsets as the evening's `speed-v41-a.jsonl` (8k = 8,737 tokens,
32k = 31,59x, 127k = 122,96x).  Kit + raw results: `strata-ds41/mab776/pf/` (scripts) and `ram-128gb/v41/pf/` (jsonl,
engine logs, nsys reports `prof/`, ppl logs `ppl/`, `table-final.md`; the 25 GB of logits dumps behind the KL numbers
were deleted after the comparisons).

## Result
| | 8.7k prefill | 32k prefill | 123k prefill | 232k prefill (needles) | decode 8k / 123k |
|---|---|---|---|---|---|
| evening baseline | 42.1 s (207 tok/s) | 81.8 s (386) | 302.3 s (407) | ~635 s (366) | 43.6 / 34.5 |
| tonight's baseline re-run (`n0-base`) | 42.4 s (206) | 82.4 s (384) | - | - | 39.9 / - |
| 04:00 config (`fin2`) | 18.4 s (475) | 22.5 s (1,402) (h1) | 76.7 s (1,602) | 172-177 s (~1,330) | 44.5 / 42.7 |
| **winner** = the served config file (`fin3`, 04:56) | **14.1 s (619)** | **20.1 s (1,575)** (s7) | **72.2 s (1,704)** | **169-173 s (~1,360)** | **53.4 / 39.4** |
| gain vs the evening | **3.0x** (13.5-14.1 s run to run) | **4.1x** | **4.2x** | **3.7x** | **+14 % at 123k** |

Agent turns (8k conversation + a user turn, `pf/fin3.jsonl`): **+300 tokens 7.0 s** (was 18.9 s: the whole
conversation was re-read; 11.6 s in fin2), **+1,500 tokens 9.7 s** (19.4 s; 14.7 s in fin2).  Needles 3/3 at 246k.
Tools: **6/6 and 6/6** on the final paths (s7); 5/6 in fin3 / fin2 / fin-final, each time the after-result case
(answer from a tool result, no further call): fin-final / fin2 on the probe's ASCII "-3" check of a correct answer
written with a Unicode minus, fin3 on a correct "-3 °C" answer that ALSO issued an unrequested `write_file` call -
sampling at the server's T=1.0 (fin3's prompt logits match fin2's at KL 2e-5).  Decode tok/s from 256 sampled tokens is content-dependent
(+-10 %); the greedy screen is the clean comparison.

## Where the time went, and what fixed it
1. **67 % of a long prefill's GPU time was the slow grouped expert kernel** (nsys, `pf/prof/p0-base`): the layer
   prestage laid a layer's arena experts out in id order with other format classes in between, so a class's prestaged
   run was never MMQ-strided and fell back to `native_gu_kernel` / `native_down_kernel` (the decode-window kernel,
   68 ms a call).  `STRATA_PSTAGE_STRIDED=1`: the prestage groups experts by class at the MMQ stride -> one MMQ launch
   per run.  Side effect: the prefill becomes deterministic across restarts (see the gate).
2. **Then the disk**: every 8,192-token batch re-read the routed file tier (~48 GB) from the Samsung 990 EVO at
   2.4 GB/s - 28 of the 8k prompt's 42 s.  Three fixes, all measured:
   - `--prompt-batch 32768` (+ `STRATA_CHUNK_BUDGET_GB=6`): a 32k prompt is ONE batch (the file tier read once), 8k too.
   - **second-drive mirror** (upstream's `--build-mirror`, unused upstream because their file tier was not
     bandwidth-bound): the 56 GiB coldest experts (the whole 5,824-expert file tier + 161) copied to the ROOT drive
     (WD SN7100), `STRATA_BLOB_MIRROR=.../mab776/mirror/V41-GSQ.experts.mirror`, first with `STRATA_BLOB_MIRROR_SHARE=0.5`
     (half of each blob's parts from each drive): reads 2.4 -> 4.9 GB/s - now 0.75 (cause 4).  Built + checked
     byte for byte by the engine in 35 s.
   - **prompt cache** (`STRATA_PROMPT_CACHE_GIB=24`; the previous session's WIP, finished + gated): for a multi-batch
     prompt the free VRAM keeps file-tier experts read by batch 1, so batches 2.. launch them from VRAM.
3. **The attention half** (now the larger part): the per-sub-batch host sync (`STRATA_SPAN_ASYNC=1`), the F32
   concat/pad/cast of every query's 640 keys (`STRATA_KALL16=1` - written in place as F16) and ggml's per-element
   index math in that strided copy (`GGML_CUDA_CPY_ROWS=1`, a row-wise copy kernel: the copy was 23 % of GPU time),
   the indexer's FP4 keys on the fp32 vec kernel (`GGML_LI_WMMA_FP4=1` - the wmma kernel), the stock top-k's one CUB
   call per ROW (`GGML_TOPK_RADIX=1`, our V4 radix select), the expert input's host round trip (`STRATA_FN_DEV=1`),
   lazy grouped-kernel buffers (`STRATA_LAZY_XQ=1`).
4. **The mirror split was half wasted** (late runs, 04:30): `dd` O_DIRECT reads measure the root SN7100 at 6.6 GB/s and
   the models-drive 990 EVO at 3.2 GB/s, but `STRATA_BLOB_MIRROR_SHARE=0.5` gave each drive half of every blob, so the
   slow drive set the pace (4.8 GB/s).  The share is applied in eighths (part p of 8 goes to the mirror when
   floor((p+1)*share) > floor(p*share)), so 0.6 was the same 4/8 split as 0.5.  0.75 = 6 of 8 parts on the SN7100:
   7.5 GB/s; 8.7k 16.7 -> 14.2 s, +300 turn 11.0 -> 8.95 s, +1,500 14.7 -> 11.0 s.  Same bytes (the mirror is checked
   byte for byte), so no numerics change.
5. **The MMQ transcode region was re-allocated on every launch** (nsys `pf/prof/s5-short`, a 475-token prompt: the GPU
   ran kernels for 0.8 s of 7.8 s; the host spent 1.36 s in 762 `cudaMalloc` and 0.9 s in 782 `cudaFree`, each free
   a device-wide wait while the copy engine drained - one layer's window: copies busy 87 of 188 ms).  The PTQ1_0 ->
   Q2_0 copy region `m_tc` is capped at `STRATA_MMQ_TC_GIB=1`, but its size test compared with the uncapped want
   (~1.4 GiB), so it was true on every launch.  `STRATA_TC_KEEP=1` allocates the cap once and keeps it (the sub-run
   sizes stay what the per-launch allocation gave; the MoE products are tile-only, `mmq-deterministic.patch`, so the
   split cannot move a sum): +300 turn 8.95 -> 6.5 s, +1,500 11.0 -> 9.1 s, 8.7k 14.2 -> 13.5 s (123k 76.7 -> 72.2 s
   with both late changes, fin3);
   KL vs fin2 2.1e-5, top-1 1.000 (the same config run twice: 1.6e-5).

## Quality gate
- **KL** (the last 512 prompt rows of logits of the SAME 8,737-token prompt, `STRATA_SERVE_DUMP_TAIL`,
  `tools/v41/logits_kl.py`): two restarts of the SAME baseline config differ by KL 2.5e-2 (top-1 0.990) - the old
  grouped-kernel path is not deterministic across restarts in serve mode.  Every variant: 2.1e-2 vs that baseline;
  variants on the new MMQ path vs EACH OTHER: KL 2e-5..5e-5, top-1 1.000 (reproducible run to run).
- **Perplexity** (one-shot `--ppl`, 4,096 tokens of C++, exact routing, `pf/ppl/`): base 3.8820 (repeat: 3.8820),
  radix top-k 3.8832, strided prestage 3.8842, every path on 3.8849 (+0.07 %), top-1 0.700 -> 0.696.  For scale: the
  dense q4_K requant upstream rejected cost +1.7 %.
- **Tools** (`tools_probe.py`, after QUALITY_DONE, the evening's FUNC_START serve fix): 5/6, was 2/6 - the one FAIL is
  the probe's ASCII check ("-3") on a correct answer that used a Unicode minus ("−3 °C"): functionally 6/6.  On the
  final paths: 6/6 and 6/6 (s7); fin3 5/6 (a correct answer plus an unrequested `write_file` call, sampled at T=1.0).
- LCB-13 (the previous session's run, finished 01:22): V4.1 9/13, V4 Strata 9/13; 11.1 vs 5.7 min per solved task.

## Variants (8k + 32k screens unless "@122963"; prefill s (tok/s); RAM = the container's peak (limit 104 GiB), host = min MemAvailable)
| variant | 8.7k prefill s (tok/s) | 32k / 123k prefill s (tok/s) | decode 8k / long | ctr RAM max | min host avail | KL vs base (top-1) |
|---|---|---|---|---|---|---|
| n0-base | 42.4 (205.9) | 82.4 (383.5) @31592 | 39.9 / 39.0 | 95.38 GiB | 23 GiB | - |
| n0b-base | 44.5 (196.5) | 99.5 (317.5) @31593 | 44.4 / 40.9 | 97.59 GiB | 23 GiB | 2.5e-02 (0.9902) |
| n1-radix | 43.8 (199.3) | 82.1 (384.8) @31593 | 43.0 / 34.5 | 97.35 GiB | 23 GiB | 1.6e-02 (0.9883) |
| n1s-strid | 43.2 (202.3) | - | 35.4 / - | 97.3 GiB | 22 GiB | 2.1e-02 (0.9922) |
| n2-strid | 43.9 (199.1) | 75.4 (419.0) @31593 | 42.5 / 40.7 | 98.04 GiB | 22 GiB | 2.1e-02 (0.9922) |
| n7-kall16 | 42.0 (207.8) | 73.5 (429.9) @31594 | 36.8 / 45.7 | 97.52 GiB | 23 GiB | 2.1e-02 (0.9922) |
| n8-cpfpb | 43.7 (199.8) | 70.8 (446.1) @31594 | 37.8 / 39.7 | 102.7 GiB | 18 GiB | 2.1e-02 (0.9922) |
| n5-pc | 41.7 (209.7) | 56.2 (561.9) @31593 | 38.4 / 36.7 | 101.8 GiB | 18 GiB | 2.1e-02 (0.9922) |
| c3-mir50 | 28.5 (306.7) | 43.1 (733.1) @31594 | 44.7 / 50.3 | 97.38 GiB | 23 GiB | 2.1e-02 (0.9922) |
| c5-mir35 | 34.0 (256.8) | 49.0 (644.3) @31594 | 44.0 / 43.2 | 97.36 GiB | 23 GiB | 2.1e-02 (0.9922) |
| c4-pb16-mir | 19.2 (454.9) | 38.1 (830.2) @31596 | 47.2 / 42.1 | 98.33 GiB | 22 GiB | 2.1e-02 (0.9922) |
| m3-pb16-s1024 | 19.0 (461.0) | 37.9 (833.9) @31597 | 44.1 / 40.4 | 98.39 GiB | 22 GiB | 2.5e-02 (0.9863) |
| c2-pb32-mir | 19.3 (453.1) | 29.8 (1059.1) @31596 | 44.0 / 43.4 | 100.3 GiB | 20 GiB | 2.1e-02 (0.9922) |
| e1-pb32-s1024 | 18.8 (465.1) | 30.7 (1028.6) @31597 | 44.9 / 41.6 | 98.29 GiB | 20 GiB | 2.5e-02 (0.9863) |
| e2-pb32-mir60 | 19.8 (440.6) | 30.1 (1049.1) @31597 | 40.3 / 46.4 | 100.6 GiB | 20 GiB | 2.1e-02 (0.9922) |
| g3-cpyrows | 18.1 (481.9) | 28.5 (1108.8) @31594 | 42.0 / 43.2 | 98.22 GiB | 20 GiB | 2.1e-02 (0.9922) |
| g4-liwmma | 18.9 (462.7) | 27.6 (1146.8) @31594 | 42.0 / 44.1 | 100.4 GiB | 20 GiB | 2.1e-02 (0.9922) |
| f1-best127 | 19.5 (448.4) | 114.9 (1069.8) @122963 | 42.6 / 39.2 | 100.5 GiB | 20 GiB | - |
| g5-best127 | 18.4 (475.6) | 92.6 (1328.1) @122963 | 48.1 / 42.1 | 100.6 GiB | 20 GiB | - |
| fin-final | 17.9 (488.5) | 92.6 (1328.4) @122961 | 45.9 / 43.9 | 100.1 GiB | 20 GiB | 2.1e-02 (0.9922) |
| h1-cpfparts | 16.8 (520.6) | 22.5 (1402.3) @31594 | 44.5 / 47.6 | 102.2 GiB | 15 GiB | 2.1e-02 (0.9922) |
| h2-cpf127 | 17.8 (490.6) | 75.8 (1621.4) @122964 | 43.1 / 45.8 | 100.6 GiB | 15 GiB | - |
| fin2 | 18.4 (474.6) | 76.7 (1602.2) @122961 | 44.5 / 42.7 | 102.0 GiB | 15 GiB | 2.1e-02 (0.9922) |

Notes: n0b-base and n1-radix ran while the mirror was being built (disk contention: their TIMES are not comparable,
their numerics are).  KL "vs base" is vs n0-base, whose own restart noise is 2.5e-2 (n0b).

Late runs (04:30-05:00, after fin2; prefill s, a fresh 8.7k prompt, then `turn_bench`: a fresh 9.6k conversation and
+300 / +1,500 user tokens; reads = the file tier's GB/s from `MIMO_CHUNK_PROF`; all on fin2's config except as noted):
| run | change vs fin2 | 8.7k fresh | 9.6k fresh | +300 turn | +1,500 turn | file-tier reads | KL vs fin2 (top-1) |
|---|---|---|---|---|---|---|---|
| s1-prof | none (+ chunk prof) | 16.7 (525) | 15.3 | 11.0 | 14.7 | 4.8-4.9 GB/s | - |
| s2-sh625 | mirror 5 of 8 parts | 15.7 (555) | 13.2 | 9.9 | 12.5 | 5.5-6.3 | - (same bytes) |
| s2-sh75 | mirror 6 of 8 parts | 14.2 (614) | 12.8 | 8.95 | 11.0 | 7.3-7.7 | - (same bytes) |
| s3-sh875 | mirror 7 of 8 parts | 14.6 (601) | 13.0 | 9.1 | 11.5 | 6.7-7.5 | - (same bytes) |
| s3-sh8125p16 | 13 of 16 parts (CPF/WAVE_PARTS 16) | 13.8 (633) | 12.7 | 9.1 | 11.4 | 7.0-7.3 | - (same bytes) |
| s6-tc | 6 of 8 + `STRATA_TC_KEEP=1` (new binary) | 13.5 (645) | 11.9 | 6.5 | 9.1 | 7.6-7.7 | 2.1e-05 (1.000) |
| fin3 | = the served config file (+ tools, needles) | 14.1 (619) | 11.7 | 7.0 | 9.7 | - | (= s6-tc's paths) |
| s7-dprof | served config + `STRATA_DENSE_PROF=1` (timing hooks only) | 13.1 (668) | 11.4 | 6.5 | 8.9 | - | - |
| s8-dprof | the same, allocator vs launch split | - | 11.8 | 6.5 | 8.8 | - | - |

s7 also gave the winner's **32k: 20.1 s (1,575 tok/s)** and two more tool probes: **6/6 and 6/6** (so fin3's 5/6 was
the after-result case's sampling at T=1.0).

The "KL vs fin2" of s6-tc is the MMQ path's own run-to-run level (fin-final vs fin2, the same config twice:
1.6e-05, top-1 1.000); its fresh 9.6k turn vs fin2's: 2e-10.

## Winner (now in `strata-ds41/mab776/strata-ds41.json`; the evening's = `strata-ds41.json.bak-20261015-overnight`, the 04:00 one (fin2) = `.bak-20261015-share05`)
Args: `--prompt-batch 32768` (the rest unchanged).  A new `"env"` block (the Strata server passes it to the engine;
checked in `/proc/<engine>/environ`, `pf/fin3.env`):
`GGML_TOPK_RADIX=1 STRATA_PSTAGE_STRIDED=1 STRATA_FN_DEV=1 STRATA_SPAN_ASYNC=1 STRATA_KALL16=1 STRATA_LAZY_XQ=1
GGML_CUDA_CPY_ROWS=1 GGML_LI_WMMA_FP4=1 STRATA_PROMPT_CACHE_GIB=24 STRATA_CHUNK_BUDGET_GB=6
STRATA_BLOB_MIRROR=/home/mab776/Documents/portainer/strata-ds41/mab776/mirror/V41-GSQ.experts.mirror
STRATA_BLOB_MIRROR_SHARE=0.75 STRATA_DSPARK_DEPTH=5 STRATA_SERVE_ROLLBACK1=1 STRATA_CHUNK_PF_MIN=2048
STRATA_CPF_PERBLOB=1 STRATA_CPF_PARTS=8 DS41_PF_THREADS=32 STRATA_TC_KEEP=1` (CHUNK_PF_MIN..PF_THREADS: the next
layer's file tier is prefetched part-wise through both drives while this layer runs - 32k 27.6 -> 22.5 s, 123k 92.6 ->
76.7 s; MIRROR_SHARE 0.75 and TC_KEEP: causes 4 and 5 above).
Every one is an opt-in engine path (default = the old one): delete a key to go back.  Engine = `build-ds4-gpu`
(the wrapper's default; = local commit 327d0ddb - the 04:00 binary is `build-ds4-gpu/ds41_generate.bak-20261015-n6`,
commit 178b6191; the branch head be4b4311 only adds the opt-in `STRATA_DENSE_PROF` timing hooks - + the three ggml
patches below, applied to the gitignored
third_party/llama.cpp: `tools/v41/ggml/mab776-topk-radix-select.patch`, `mab776-cpy-rows.patch`,
`mab776-li-wmma-fp4.patch`).  RAM: the container peaks at ~102 GiB of its 104 GiB limit (the two 2.1 GiB pinned prefetch
slabs), host MemAvailable >= 15 GiB.  VRAM: unchanged for decode (the batch buffers are prompt-time only).
- **The mirror file** (56 GiB, root drive, git-excluded) is part of the config: if it is missing the engine logs
  `blob mirror: NOT used` and runs on the models drive alone (the 8k/32k numbers drop back toward c2-style ~1.5x).
  Rebuild (35 s) + byte check: `docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=none --memory 6g -v
  "$G:$G:ro" -v $W:$W strata-ds4:cu13 $W/build-ds4-gpu/ds41_generate -m "$G/V41-GSQ.gguf" --rank "$G/rank.bin"
  --build-mirror $W/mab776/mirror/V41-GSQ.experts.mirror --mirror-gib 56` (then `--check-mirror <file>`;
  G = models/deepseek-v41/gsq, W = the worktree; script `mab776/pf/mirror.sh`).

- **Agent turns** (`pf/fin2.jsonl`, `turn_bench.py`: an 8k conversation, then +300 and +1,500 user tokens):
  without `STRATA_SERVE_ROLLBACK1` EVERY turn re-read the whole conversation (`serve: reuse lcp=9568 ... reused=0`:
  the previous prompt matched up to its last token - V4.1's template opens the generation with `<think>` and renders
  that turn's prefix as `</think>` in the history, so neither the continuation nor the snapshot path matched).  With
  it: +300 tokens = 439 read in 11.6 s (was 18.9 s), +1,500 = 1,891 read in 14.7 s (19.4 s) - 7.0 / 9.7 s with the
  late fixes (fin3).  At a 100k conversation this is ~75 s -> ~12 s a turn.  Numerics (greedy, the same prompts, runs t5-rbA vs t5-rbB):
  turn 0 identical (KL 0), the rolled-back turns vs a full re-read KL 3.0e-3 / 1.2e-3, top-1 0.995 / 0.998 - the
  ordinary incremental-vs-full prefill difference, 8x under the old path's restart noise.
- **Decode**: DSpark depth 5 (greedy, the same tokens, 2 x 384 at 8k): 53.2 tok/s vs 50.5 at depth 4 (2.66 vs 2.42
  tokens/pass); conf 0.5 49.0, depth 5 + conf 0.4 47.9 (more tokens/pass, slower: wider verify passes cost more).
  123k decode 34.5 -> 42.7-45.8 tok/s (the radix top-k: the stock top-k ran one CUB call per row in every verify
  pass).

## Rejected, and why
- `--prefill-chunk 1024` (dense sub-batch 1024 vs 512): 8k 19.0 vs 19.2 s, 32k 37.9 vs 38.1 s - noise, and it moves
  the numerics (KL 2.5e-2, top-1 0.986).
- Chunk prefetch slabs alone (`STRATA_CHUNK_PF_MIN=2048` + `STRATA_CPF_PERBLOB=1`, whole-blob reads from the main
  drive at 1.4 GB/s): +4 % at 32k, -3 % at 8k - superseded by the part-wise version through the mirror (winner).
  Its first build crashed the engine twice (03:00 / 03:03: the per-blob part counters were freed under the running
  readers - fixed in 178b6191).  Each crash wrote a ~107 GB apport core to `/var/lib/apport/coredump` on the ROOT drive
  and filled it (0 bytes free for ~5 min; both cores deleted, every test container now runs `--ulimit core=0`).
  **ds41-test itself was created without `--ulimit core=0`: an engine crash there would do the same.**
- Mirror share 5/8 (0.625), 7/8 (0.875) and 13/16 with 16 parts: 6/8 (0.75) is the balance point of the two drives
  (late-runs table).  The earlier "0.35 / 0.6 worse than 0.5" was wrong: the share is applied in eighths of a blob
  (floor(8 x share) parts on the mirror), so 0.6 IS 0.5 (4 of 8; its 4.7 vs 4.9 GB/s was noise) and 0.35 is 2 of 8.
- `--prompt-batch 16384`: 32k 38.1 s (2 batches) vs 29.8 s at 32768.

## What is left (with a gain guess)
- **Short batch turns** (300-2,000 new tokens) are now 6.5-7 s / 9-10 s.  A +300 turn = the batch ~5.8 s (dense 1.9 +
  experts 3.7) + the VRAM cache re-seed 0.75 s.  Where it goes (nsys `pf/prof/s5-short` + `STRATA_DENSE_PROF`, s7/s8):
  every batch streams the whole arena (~2.2 GiB a layer, 88 GiB a batch = ~2 s of the 46 GB/s copy engine) and reads
  ~26 GB of routed file tier (3.4 s at 7.5 GB/s) whatever its token count; the dense half's 1.9 s is NOT compute (its
  kernels are ~5 ms a layer): each layer's graph launch waits ~45 ms because its small input uploads queue in the copy
  engine behind the next layer's prestage copies.  Reordering cannot win (the copy engine is the bottleneck either way).
  What would: (1) keep the decode cache (~37 GiB = the hottest arena experts) through a short batch - the prestage
  then copies only non-resident experts (~-40 % copy time) and the 0.75 s re-seed goes; needs a VRAM margin for the
  batch buffers (or ExpertCache's segmented shrink/grow, #533) - guess 6.5 -> ~5 s a turn; (2) the next layer's
  file-tier reads started from a predicted routing (decode's spec_read idea, for a batch) so the disk never waits for
  the router - guess another -1 s.  Floor: ~3.4 s (the disk).  The decode-side passes path (STEP_MAX / BATCH_MIN
  raised, t1; + `STRATA_PASS_MULTI`, t2) stays rejected: 21.5 / 19.9 s vs 10.6 s as a batch for 544 tokens.
- **Tensor-core lightning indexer** (our V4 `li_mma` kernel, 32 heads + MXFP4 keys): `GGML_LI_WMMA_FP4` already moved
  the indexer to the wmma kernel; li_mma was 5.6x wmma on V4 - guess +5 % at 127k, more at 250k+.
- **Flash-attn** is now ~25 % of GPU time with one query per KV batch (ncols1 = 1); a 64-heads-as-columns variant could
  double its throughput: guess +8-10 % at long context.
- **GPU 1** stays idle: its x4 Gen4 link (~7 GB/s) makes it useless for prefill streaming; a static expert tier for
  DECODE (our V4 idea) would cut PCIe misses: guess +8-12 % decode, ~a day of work.
- `ds41-test` (stopped; `--config mab776/strata-ds41.json` from the read-only worktree mount, so `docker start ds41-test`
  serves this winner on :8331) was created WITHOUT `--ulimit core=0`: an engine crash there writes a ~107 GB core to
  the root drive.  Recreate it with the flag before relying on it.

## V4 fork (Strata-DS4 M1-M14) features vs ds41 (inventory 2026-10-15 07:45; nothing new ported this morning - Marc's
message arrived at 07:41, so the engine window was ~20 min; checked against strata-ds4/README.mab776.md STATUS + M notes)
| V4 feature (switch) | ds41 | note |
|---|---|---|
| radix top-k (M10, GGML_TOPK_RADIX) | **ported** (tonight) | `mab776-topk-radix-select.patch` |
| warp top-k router (M14, GGML_CUDA_RANK_ARGSORT) | **ported** (upstream) | `tools/v41/ggml/rank-argsort.patch` |
| fused hyper-connection ops (M13, DS4_HC_MIX) | **ported** (upstream) | built with `-DSTRATA_DSV41_HC_MIX=ON`, dsv41-fuse.patch |
| stale-graph unbind fix (M14) | **ported** (upstream) | ds41_dense.cpp `Allo::alloc` |
| fused finish+predict (M14, DS4_FUSE_PREDICT) | **ported** (upstream, default on) | STRATA_NO_FUSE_PREDICT=1 = off |
| chained MTP drafts K=2 (M7c) | **n/a** | ds41 has DSpark (depth 5 tonight) |
| conversation reuse / agent turns read only the new tokens | **ported** (tonight) | `STRATA_SERVE_ROLLBACK1` (+ upstream's snapshot restore) |
| serve fixes (#984 #1624 #1869/#1870 #2026, quoted DSML) | **n/a / in** | ds41 serves through Strata 0.1.43.1 (src-01431) |
| `--pf-b 0` (M14: prefetch hints off, -2..-3 % decode on V4) | **rejected** (morning) | greedy decode 52.3 / 48.5 / 44.5 vs 49.7 / 53.0 / 49.9 tok/s (8k, 8k, 123k) |
| int8 tensor-core lightning indexer (GGML_LI_I8 / li_mma) | **ported** (morning, 7c71c1d2; fp16 version) | `GGML_LI_MMA=1` (prompt batches >= 64 queries): 123k prefill -4..-6 %; not in the served config yet (needles 1 of 3 done) |
| elastic GPU 0 cache kept through prompts (`--vram-margin 9 --vram-grow 2`) | **ported** (morning, 66121bb6) | `STRATA_KEEP_CACHE=2048`: +300 turn 6.50 -> 5.53 s, bit-identical |
| file-tier reads from a predicted routing | **left** | batch form of decode's prefetch; guess -1 s a turn |
| GPU 1 static expert tier (`--gpu1 1`) + MTP head on GPU 1 | **left** | guess +8-12 % decode; ~a day |
| prompt-lookup drafts (`--lookup 7`, M11: copy-heavy edits +42-45 %) | **ported** (morning, 9757f854) | `STRATA_LOOKUP=7`: copy-heavy T=1.0 edits +10 % (0..+24 %) - DSpark already copies well |
| conversation parking (`--park-mib 4096`, M4) | **left** | ds41 keeps only the last prompt's snapshot |
| emit-only compressors + batched HCA (prefill) | **left** (check) | V4-specific kernels; ds41's compressor path differs |
| async decode hand-offs / fused hc gates / expert mix on the card / predict-tier overlap (M11: DS4_ASYNC, DS4_HC_GATES, DS4_DEV_SUM, DS4_OVL) | **left** (partly upstream) | decode-only; each needs a decode profile of ds41 first |
| hash-layer prefetch hint, fp32 acc of F16 decode matmuls, hits routed on the GPU (M12: DS4_PF_HASH, DS4_F16_ACC32, DS4_DEV_ROUTE) | **left** | decode-only |
| lean decode graphs, 64-thread route kernel, pinned prefetch (M13: DS4_LEAN, DS4_ROUTE_PAR, DS4_PF_PINNED) | **left** | decode-only |
| NaN guard (M13, DS4_NAN_GUARD) | **ported** (morning, dc214949) | `STRATA_NAN_GUARD=1`, injection-tested |
| flash-attn with heads as columns | **left** (new idea, not V4) | guess +8-10 % at long context |

## Deploy (07:44-07:52): native Portainer stack 78 `deepseek-v41`
- Kit `portainer/deepseek-v41/` (modeled on deepseek-strata, stack 74): `docker-compose.yml` (reference; Portainer owns
  /data/compose/78), `deploy.sh` (create/update through the API, creds from .portainer-creds), `strata-ds41.json` (the
  fin3 config: same args + env block, paths under /app, port 8080, mirror at /mirror, model id `deepseek-v4-flash` +
  aliases `deepseek-v4-flash-strata` / `deepseek-v4.1-flash` / `deepseek-v4.1-flash-strata-gsq`, allowed hosts + ds4-lcpp),
  `ds41-engine.sh` (the snapshot's engine wrapper), `snapshot.sh` (worktree -> `models/strata-ds41/run`, the previous
  one kept as `run.prev`; VERSION records the tree + the engine's md5, c6d36cc0e22a = the fin3 binary).
- Container `strata-ds41`: image strata-ds4:cu13, `restart: "no"`, GPU 0 only, mem_limit/memswap 104g, oom_score_adj 500,
  ulimits memlock -1 + core 0, ipc host, host :8330 -> 8080, `ai-services-network` alias `ds4-lcpp`, healthcheck /health.
- Verified 07:48 (it starts on creation): healthy 80 s after creation (loaded in 74 s), `blob mirror ... share 0.750`, the
  engine env = the 26 tuned switches; "pong" answered directly on :8330 as `deepseek-v4-flash`, as VS Code's
  `deepseek-v4-flash-strata`, and through llm-router :8340 (OWUI's only OpenAI connection), whose model list now shows the
  V4.1 ids.  Then stopped (`docker stop`, the container this deploy had started): GPUs 15 MiB, mode page `off`.
- Mode page (stack 54, `/home/mab776/mode-toggle/app.py`, backup `app.py.bak-20261015-v41`): DeepSeek tile = llama.cpp |
  Strata | **V4.1** (`DS_VARIANTS["v41"] = "strata-ds41"`, in MANAGED + G1_RIVALS + PROBES); a variant swap stops the
  running DeepSeek engine first, a mode press starts only the selected one, so two never run at once.  The page was
  restarted and shows the button; the saved choice is still `strata` (stack 74), as before.
- **Not done: the start/stop through the tile** - my session's permission classifier refused to switch GPU modes
  ("Interfere With Workloads").  To run it (≈3 min): mode page -> DeepSeek tile -> **V4.1** -> press DeepSeek V4 -> wait
  for the green dot -> All off.
- Update the engine later: build in `strata-ds41` -> `portainer/deepseek-v41/snapshot.sh` -> restart the container
  (mode page Restart).  Rollback: stop, swap `run` / `run.prev` in `models/strata-ds41`, start.

## Morning ports (Marc, 08:00-: "continue porting the V4 features and push to my branch")
Pushed to github.com/mab776/Strata-DS4 branch mab776-v41 after each proven step (Marc allowed it in session).
- **`--pf-b 0` rejected**: greedy 384-token decode 52.3 / 48.5 / 44.5 tok/s vs 49.7 / 53.0 / 49.9 with the default
  prefetch bias (8k, 8k, 123k; the run label made the prompts differ slightly) - the V4 win does not carry over.
- **`STRATA_KEEP_CACHE=<tokens>`** (66121bb6, the V4 fork's elastic cache): the decode cache is opened segmented (VMM);
  a batch of at most N new tokens shrinks it by `STRATA_KEEP_CACHE_FREE_GIB` (10) for its buffers, runs with the rest
  resident (resident experts copied device to device into the MMQ-strided stage halves - launched in place they would
  hit the grouped kernel), then regrows and refills the same experts into the same slots (`ExpertCache::evict_from`,
  new in core).  Greedy agent turns: +300 tokens 6.50 -> 5.53 s, +1,500 9.05 -> 8.06 s, the 0.75 s re-seed gone
  (shrink 25 ms + regrow 220-270 ms); kept turns' tail logits **bit-identical** (KL 0).  A regrow can come back a few
  slots short (26 of ~3,300 once: ggml's pool kept some VRAM) - logged.
- **`STRATA_LOOKUP=7`** (9757f854, the V4 fork's Lookup): when the context's last 3 tokens occurred before and the copy
  starts with DSpark's first draft, the copied block (<= 7) replaces DSpark's chain in the verify pass; ds41's
  spec_accept treats drafts as deterministic proposals, so the distribution is unchanged.  Copy-heavy edits at T=1.0
  (a file + "rename X, output the whole file", fresh prompts): 76.7 -> 84.3 tok/s mean (0 .. +24 %; 90-98 % of copied
  drafts accepted); greedy +2.6 % (DSpark alone already copies at ~81 tok/s - why V4's +42 % does not repeat).  Greedy
  text: the same final code; the reasoning diverged at its 15th token (a wider verify pass rounds differently).
  Side finding: a repeated identical prompt takes the regenerate path, which runs WITHOUT DSpark (plain ~44 tok/s).
- **`STRATA_NAN_GUARD=1`** (dc214949, M13 / upstream #879): a non-finite row of logits ends the request
  `DONE nonfinite` (the 0.1.43.1 server handles it) with none of that pass's tokens, history + snapshot + DSpark window
  dropped; `STRATA_NAN_INJECT=1` test: the request ended nonfinite, its retry re-read from 0 and decoded normally.
- **fin4** (the served config + the three, 602d9fc3, engine `build-ds4-gpu` = dc214949; the fin3 binary is
  `ds41_generate.bak-20261015-tckeep`, the fin3 config `strata-ds41.json.bak-20261015-fin3`): 8.7k 13.5 s (647 tok/s),
  123k 73.3 s (1,678), decode 51.1 / 46.3; turns +300 **5.9 s**, +1,500 **8.5 s**; needles 3/3 (167-173 s); tools 6/6.
  The deploy kit (portainer/deepseek-v41/strata-ds41.json) carries the same switches - stack 78 is not redeployed
  (Marc's call: `deepseek-v41/snapshot.sh`, then restart).
- **`GGML_LI_MMA=1`** (7c71c1d2, `tools/v41/ggml/mab776-li-mma.patch`, applied to the vendored tree after
  mab776-li-wmma-fp4.patch): the V4 fork's fp16 li_mma kernel adapted to V4.1's 32 indexer heads + MXFP4 keys (one
  block = 8 queries x 128 keys, the key tile decoded to fp16 in shared memory once, m16n8k16 MMA per 16-head group).
  Prompt batches of >= `GGML_LI_MMA_MIN_BATCH` (64) queries only: on all passes the 123k decode fell 46.1 -> 40.0
  (a verify pass fills too few of the 8 warps); with the threshold decode is unchanged (greedy 46.1 -> 48.6).  123k
  prefill 72.4 -> 69.6 s and 72.5 -> 68.1 s (dense 49.5 -> 44.4 s); PPL code 3.8843 -> 3.8849, prose 12.2348 ->
  12.2351; KL 2.2e-5, top-1 1.000; tools 6/6; needles: 10 % FOUND (162 s vs 167-173), the 50 / 90 % runs were cut off
  when the GPUs were handed back (09:28).
- **Stopped 09:28 for Marc's work bug** (the GPUs were needed at once; the DeepSeek-V4-Flash session stopped ds41-pf).
  Left as WIP, no GPU used after that:
  - stack 78 still runs the 07:46 snapshot (the fin3 engine + config); `deepseek-v41/snapshot.sh` would deploy fin4
    (cache kept, lookup, NaN guard) - kit config already updated; then restart through the tile and a pong.
  - li_mma: two more needles (50 / 90 % at 246k), then `GGML_LI_MMA=1` into the served config + a rebuild of
    build-ds4-gpu (the vendored tree is already patched).
  - left from the inventory: conversation parking, the GPU 1 expert tier, predicted file-tier reads for batches, the
    M11-M13 decode items (profile ds41 decode first), DSpark on the regenerate path.
  - `ram-128gb/v41/pf/dump` (9.4 GB of this morning's KL dumps, root-owned) still to delete:
    `sudo rm -rf ~/Documents/portainer/ram-128gb/v41/pf/dump` (my session's safety check refused the container rm).

## Evening (Marc relayed, 17:30 -): redeploy, the 500k crash, the GPU 1 expert tier
- **Stack 78 redeployed 17:35** with fin4 + `GGML_LI_MMA=1` + conversation parking (27114ba7, snapshot eb20f598, served
  config 6b29f870; park dir `/home/mab776/v41-engram/park` rw-mounted, RAM 3 / disk 10 GiB, 14 days): health OK 72 s
  after the tile switch, pong through VS Code's id and llm-router.
- **Conversation parking** (`STRATA_PARK=1`, 27114ba7): a request that does not continue the live conversation parks it
  at its snapshot point (snapshot tensors + closed comp / idxk rows, ~0.9 KB a token + 6 MB) and takes the parked
  conversation it extends.  Restored turns bit-identical (KL 0: RAM, disk, after a restart); restore 2.5 ms (RAM) /
  16-20 ms (disk) vs ~16-18 s to re-read 20k tokens.
- **The 500k crash = a segfault in ggml's allocator** (found in the kernel log - the engine log just stopped): a failed
  `ggml_gallocr_reserve` frees the buffer but keeps the failed graph's plan, so the oom_hook retry bound tensors to a
  NULL buffer - the retry path had never worked.  Fix f81f2808: an explicit reserve after a failure, and the hook frees
  the prestage region, then the prompt cache.  Forced test (both released): logits bit-identical; real: needle at
  **505,709 tokens found** (307 s = 1,646 tok/s; decode 46 tok/s), parked 0.39 s, restored 29 ms, follow-up found again.
- **GPU 1 static expert tier** (453d1a1b; `--peer-device 1` + config `"gpu": [0, 1]`): 1,237 experts / 11.4 GiB
  right after GPU 0's head, no arena copy (arena 88 -> 76.5 GiB = the same file tier, -11.5 GiB pinned RAM).

  | | deployed set | GPU 1 tier |
  |---|---|---|
  | decode 8.7k / 123k (256 tok) | 43.4 / 39.9 | **49.4 / 45.4** (+14 %) |
  | prefill 8.7k / 123k | 13.33 / 68.2 s | **12.63 / 66.9 s** |
  | turns +300 / +1,500 | 5.80 / 8.24 s | **5.51 / 7.86 s** |
  | PPL code / prose (exact routing) | 3.8843 / 12.2351 | 3.8849 / 12.2351 |
  | PPL code / prose (served route bias) | 4.2074 / 13.6092 | **4.1086 / 13.2558** |

  GPU 1's decode rows byte-identical to GPU 0's kernel (1,400 / 1,400).  Its prompt share first cost 2-5 % on big
  prompts (the whole 512 MiB x went over its x4 link every layer, GPU 0 waited 5.8 s at 123k): now only the rows its
  entries read go over (gathered on GPU 0 into mapped pinned memory) and its rows are scattered beside GPU 0's passes
  (wait 1.8 s).  The route bias counting GPU 1 as resident is a quality GAIN (a larger favoured set).
- **Finding: the prompt path lands in one of a few numeric states with the memory layout** - the deployed engine with a
  second visible GPU (no tier), or with arena 74 vs 88 GiB, flips between states that differ by KL ~1.9e-4 (top-1 1.0;
  routing near-ties: different prompt-cache hit counts for the same total entries); PPL 3.8843 / 3.8849 / 3.8850
  depending on the binary.  Same config + same binary = bit-identical across restarts.  Cause not found (the li_mma
  kernel looks clean: zero-filled keys past n_kv, masked writes); judge such changes by PPL, not "KL 0 vs reference".
- **DSpark window with the snapshot** (`STRATA_DSPARK_SNAP=1`, b3291139): a regenerate reset the drafter and decoded
  without it (~44 tok/s).  The window (rings + slot positions + the last main_hidden row) is saved at the snapshot and
  restored with it; a pure regenerate samples its first token from the saved logits and redrafts from the last row.
  Regenerate 44.4 / 46.0 -> **75.0 / 80.5 tok/s** (3.56 tokens a pass; the two regenerates identical).  Parked blobs
  carry the window as a trailer: restores 2.6-3.1 ms, no failures.
- **Redeployed 19:46** (stack 78, both GPUs: compose `NVIDIA_VISIBLE_DEVICES=0,1`, config `"gpu": [0, 1]` +
  `--peer-device 1` + `--arena-gib 76.5` + `STRATA_DSPARK_SNAP=1`, served config 897996ff; snapshot = build-g3 md5
  44d90486cc40 = b3291139, run.prev = the 17:35 one): health 72 s after the tile start, pong via VS Code's id and
  llm-router; the candidate run before it: needles 3/3 at 246k (154 / 151 / 156 s), tools 6/6, 8.7k 13.1 s, 123k 67.9 s.
  Mode page labels say GPU 0+1 (app.py.bak-20261015-g1).
- Open: the M11-M13 decode items (profile ds41 decode first); a dynamic GPU 1 slice (expected neutral - Marc asked; see
  the peer reply: a GPU 1 fill is ~1.4 ms over its x4 link vs ~0.19 ms for GPU 0's miss); the numeric-state cause;
  `sudo rm -rf ~/Documents/portainer/ram-128gb/v41/pf/dump` (root-owned KL dumps).

## Late evening (Marc relayed, 20:00 -): dynamic GPU 1 cache, D3 overlap, the route-bias A/B
- **Dynamic GPU 1 cache (Strata-style, `STRATA_GPU1_DYN=1`) loses to the static tier** - Marc asked for it judged under
  domain shift.  Domain test (`mab776/pf/domain_test.py`: greedy 384-token replies to ~7.5k-token prompts of code,
  PowerShell, compose/bash, French, French literature, poetry, then one PS -> FR -> PS session), decode tok/s:

  | | static | static + D3 overlap | dynamic (all of GPU 1) | dynamic (6 GiB static core) |
  |---|---|---|---|---|
  | mean of 9 requests | 53.6 | **57.2** | 51.1 | 48.5 |
  | GPU 1 share, FR literature / poetry | 18.2 / 18.4 % | 18.7 / 18.5 % | 23.5 / 24.5 % | 21.1 / 19.6 % |

  Dynamic does follow the domain (GPU 1's share rises out of domain), but every GPU 0 eviction is pushed down: ~1,000-
  1,500 fills per 384 tokens at ~1.4 ms each over GPU 1's x4 link, competing with its own reads.  Slower in every
  domain but code, and it frees no RAM.  Kept as an opt-in, not served.
- **D3 overlap (`STRATA_D3_OVERLAP=1`)**: the verify pass stages its arena-held PCIe experts on the demand-copy stream
  before the hits kernels.  Verify pass 45.65 -> 38.54 ms at 123k (tier 21.6 -> 15.7 ms); decode 52.1 -> 56.5 (8.7k),
  46.0 -> 47.2 (123k); regenerates 78 -> 82.5.  Under the route bias the greedy text changes: the PCIe misses now pick
  their LRU victims before the prefetched experts do, so the cache - and the bias's resident set - differs.
  **Identity check (Marc asked), exact routing, 9 greedy domain requests**: overlap off vs on = 6/9 identical (the other
  3 diverge at chars 163 / 446 / 580); the SAME config twice (on vs on) = also 6/9 (one diverges at the first token) -
  the served engine is not bit-deterministic run to run (prompt cache / arena-adapt / keep-cache state follows the
  request history), and the overlap's differences sit inside that noise.  Speed in the same exact runs: on 24.2 / 24.2
  vs off 21.3 tok/s (9-request mean, +14 %), 123k 25.8 / 26.2 vs 23.7 (+10 %), turns +300 9.0 / 9.5 vs 10.0 s.
- **Route-bias A/B (Marc: "give me the numbers and I will judge" -> verdict: the bias stays 0.5 / 0.3, final)**.  Same
  build (build-g3 + overlap, no parking), one throwaway container per arm: the 8.7k / 123k bench, agent turns, the 9
  domain prompts, and Marc's real traffic (6 llm-sniffer sessions, 50-60k-token prompts, run blind: metrics only).

  | | **A 0.5 / 0.3 (served)** | B exact 0 / 0 | C 0.2 / 0.1 | D file penalty only 0 / 0.3 |
  |---|---|---|---|---|
  | decode 8.7k / 123k bench | **51.8 / 45.3** | 27.3 / 25.8 | 46.5 / 44.1 | 37.5 / 38.5 |
  | decode, 9 domain prompts (mean) | **60.3** | 24.2 | 44.4 | 38.0 |
  | decode, Marc's real traffic (mean of 6) | **60.4** | 21.8 | 42.4 | 37.9 |
  | experts served from the GPUs (real traffic) | **95.8 %** | 80.0 % | 91.3 % | 84.8 % |
  | file tier (NVMe) share, domain prompts | **0.20 %** | 2.01 % | 0.60 % | 0.60 % |
  | agent turns +300 / +1,500 (wall) | **6.2 / 8.7 s** | 9.0 / 11.6 s | 6.7 / 9.3 s | 6.9 / 9.2 s |
  | prefill 8.7k / 123k | **12.4 / 67.3 s** | 13.4 / 70.3 s | 13.1 / 68.3 s | 12.9 / 68.2 s |

  The VRAM bonus carries most of the speed (D keeps the file penalty and still loses 37 %).  Exact routing also slows
  prompts: without the file penalty, decode pulls NVMe-tier experts into the RAM arena (arena-adapt), so the next prompt
  reads more from disk (a fresh 9.6k prompt 11.1 -> 15.0 s, the KEEP_CACHE regrow 0.23 -> 0.88 s).  The synthetic
  8.7k / 123k bench understates the bias: on Marc's traffic C is -30 % and exact -64 %.  Quality side, from the earlier
  gate: the served bias costs PPL code 3.8849 -> 4.1086, prose 12.2351 -> 13.2558 (teacher-forced, every token biased).
  The per-domain PPL table and LCB-13 were cancelled (Marc: "the price is too high").
- **Served default effort = medium (62)** (Marc): `"reasoning_effort": "medium"` in both config copies (kit backup
  `.bak-20261015-effort`); requests that name an effort keep theirs.  Ships with the next snapshot.
- **Decode items at the served config (21:40 - 22:50)** - nsys of 8 s of greedy decode (`mab776/pf/nsys_decode.sh`, the
  host's nsys mounted into the image): GPU 0 busy 68 %; per ~47 ms verify pass ~4,300 kernels, ~1,200 memcpys, ~270
  stream syncs; the idle time sits after each host readback - router readback -> the tier's first copy ~45 us a layer
  (classify), tier sum -> finish ~26 us, predict -> attention ~21 us, GPU 1's rows ~100 us x 7 a pass.
  - `STRATA_PV_GRAPHS=1` (a9fc5a94, kept multi-token graphs -> CUDA-graph replay): works (plain launches 632k -> 164k
    per 8 s) but decode 103.1 -> 101.4 tok/s - not launch-bound.  Its first version crashed after a keep-cache batch
    (release_pass freed the buffer the kept graphs pointed into) - fixed.  Same commit: a cached graph re-planned after
    an out-of-memory retry now gets a new uid (ggml-cuda would have replayed the stale CUDA graph).
  - `STRATA_ASYNC_FINISH=1` (b99fd746): the tier records an event instead of its sum sync, finish waits on the card,
    GPU 1's rows waited for on the card: token stream identical, verify 40.7 -> 40.5 ms a pass.
  - `GGML_MMVF_F16_MAX=8` (2eaeefeb, the V4 fork's F16 matvec patch): verify -1 %, changes the token stream.
  - The one-shot greedy decode is bit-reproducible run to run (`NSYS=0 nsys_decode.sh`, compare the token md5) - the
    exactness check for decode changes (the server is not: its state follows the request history).
  All opt-in, not served.  The next lever would be routing on the card (V4's design notes): not started.
- **Redeployed 22:52** (stack 78, served config a906469e = the 19:46 one + `STRATA_D3_OVERLAP=1` + `"reasoning_effort":
  "medium"`; snapshot engine build-g4 md5 109ac9e1b09d, run.prev = the 19:46 snapshot).  Gate (cand_run.sh): decode
  54.1 / 47.4 tok/s at 8.7k / 123k (19:46: 49.4 / 45.4), prefill 8.7k 670 tok/s (13.1 s) / 123k 1,806 tok/s (68.1 s),
  turns +300 / +1,500 6.3 / 8.8 s, tools 6/6, needles
  3/3 at 246k, regenerate identical; one-shot token stream identical with the overlap on and off.  Live: health 75 s
  after the tile start, "[strata] thinking level: medium" at startup, pong via VS Code's id and llm-router.
  (Report format from here on, Marc: prefill as tok/s, seconds in brackets.)

## Night 10-16 (Marc relayed: "the overnight run for yourself" - decode GPU-side routing + prefill, equal weight)
All at the served config (route bias 0.5 / 0.3 fixed); every engine change opt-in and bit-exact in the one-shot check
(greedy DSpark decode, token-stream md5, `NSYS=0 mab776/pf/nsys_decode.sh`).  Prefill in tok/s (seconds in brackets).
- **Profile first** (engine counters + nsys): agent turns are NVMe-bound (each +300 turn re-read ~22 GB of routed
  file-tier experts, ~3.2 s, AFTER each layer's router); 8.7k is disk + dense; 123k is dense-bound (44 s) with ~5 s of
  the dense half waiting on Engram rows at each 32K batch.  Decode: GPU 0 busy 68 %; the expert kernels of V4.1's
  formats (gate/up PTQ1_0 47 % / Q2_0 35 % / Q3_K 18 %) cost ~6 / ~12 / ~11.5 us per expert (Q2_0 near 1 TB/s,
  PTQ1_0 ~0.4: its trit decode); and **decode's NVMe-tier experts**: 626 of 14,080 verify calls block ~9 ms on their
  reads (~35 % of a one-shot code decode's wall time; serve requests ~5-10 %).
- **Arena 88 GiB** (was 76.5 since the GPU 1 tier; config): an agent turn reads ~16 instead of ~22 GB of file tier.
- **Engram read-ahead** (`STRATA_ENGRAM_AHEAD=1`, aee0356f): the next batch's rows are read during the current batch
  (+1.6 GB buffer): 123k 1,799 -> 1,907 tok/s (68.3 -> 64.5 s).
- **Kept verify graphs + async tier -> finish hand-off** (`STRATA_PV_GRAPHS=1` + `STRATA_ASYNC_FINISH=1`, a9fc5a94 /
  b99fd746): each alone ~0 (the hand-off only overlaps when the next attention graph needn't be re-planned, which
  drains the stream); together verify 40.1 -> 38.0 ms a pass (one-shot), serve decode +2-7 %.
- **Q3_K split kernels** (`STRATA_SPLIT_FORK=1`, 0d9d210c): Q3_K gate/up -18 % a call, verify -0.7 %.  PTQ1_0's split
  saves nothing (~1 token per expert in a verify pass); the quarter-block PTQ1_0 dot (not bit-exact) only -2 %.
- **Decode read parallelism** (`STRATA_D3_PARTS=128`, was 32): the NVMe-tier wait -12 %, verify -1.2 %.
- **Measured, rejected**: predicted file-tier reads for short batches (dry run `STRATA_CHUNK_PRED=dry`, ca51212e:
  recall 15-26 % top-6, 28-46 % top-12); speculative decode reads (`STRATA_D3_SPEC` 4 / 8: 1 % of blobs, pass +5 %);
  prefetch off (`STRATA_D3_PF=0`: one-shot +3.8 % but a wash in serve - the prefetched experts keep the GPU share up);
  more readers (64 = 32); keeping the NVMe drives awake (APST is not the decode read cost).
- **Memory: arena 88 had no headroom -> arena 83 + park RAM 1 GiB** (served since 02:16, snapshot 9f78a3fc).  The cand7 gate (arena 88) lost
  its engine to the container's OOM killer at the 104 GiB cap during the 50 % needle (232k prompt): anon 95.7 GiB +
  8.0 GiB pinned (cudaMallocHost = shmem: chunk-prefetch slabs, read bounces, Engram buffers) = the cap
  (`journalctl -k`: "Memory cgroup out of memory: Killed process ... ds41_generate").  The cand6 snapshot (00:45) peaked
  only ~0.4 GiB lower.  Rule (relayed): >= 4 GiB margin at ~505k, no cap raise.  `mab776/pf/mem_probe.sh` samples the
  cgroup's anon + shmem + kernel (what the OOM killer counts) every 2 s per phase:

  | arena 83 + park RAM 1 GiB | peak anon + pinned + kernel | margin under 104 GiB |
  |---|---:|---:|
  | 8.7k / 123k bench + turns | 98.5 GiB | 5.5 GiB |
  | needle at 479,953 tokens | 98.7 GiB | 5.3 GiB |
  | three 246k needles after it (parked 480k conversation) | 99.5 GiB | 4.5 GiB |
  | LIVE stack 78, needle at 506,885 tokens (found, 281 s = 1,804 tok/s) | 98.6 GiB | 5.45 GiB |

  Memory hardly grows with context (the 32K-row batch buffers are the step; parked conversations are 0.1-0.5 GB),
  so arena 84 would sit just under 4 GiB.  What the margin costs vs arena 88 (cand7 gate, same binary):

  | | arena 88 (c-g7) | arena 86 (c-g8) | **arena 83 (c-g9)** |
  |---|---:|---:|---:|
  | prefill 8.7k (first request after start) | 740 tok/s (11.8 s) | 712 tok/s (12.3 s) | 601 tok/s (14.5 s) |
  | prefill 9.5k fresh (turn bench, steady) | 876 tok/s (10.92 s) | 874 tok/s (10.95 s) | 867 tok/s (11.03 s) |
  | prefill 123k | 1,913 tok/s (64.3 s) | 1,938 tok/s (63.5 s) | 1,914 tok/s (64.3 s) |
  | +300 turn (439 read): prefill / wall | 4.73 / 5.5 s | 4.90 / 5.6 s | 5.15 / 5.8 s |
  | +1,500 turn (1,891 read): prefill / wall | 6.94 / 7.7 s | 7.12 / 7.7 s | 7.46 / 8.0 s |
  | decode 8.7k / 123k (256 tok) | 56.5 / 56.2 | 54.8 / 51.8 | 58.8 / 51.0 |
  The first 8.7k request pays the bigger file tier (5,172 vs 4,660 experts) while it allocates the prefetch slabs
  (1.96 vs 1.74 GiB each); later fresh prompts do not.
- **Deployed 02:16** (commit 9f78a3fc, pushed; snapshot engine c464ce453e70 = 268f1909's build; run.prev = the 00:45 snapshot b02f7584): the served config = cand7's switches (`STRATA_PV_GRAPHS=1`, `STRATA_ASYNC_FINISH=1`, `STRATA_D3_PARTS=128`) + arena 83 GiB + `STRATA_PARK_RAM_GIB=1`.  Through the mode page: health 75 s, thinking level medium, pong via VS Code's id and llm-router, the 506,885-token needle above.  Gate (c-g9): needles 4/4 (480k + 246k x3), tools 6/6 and the regenerate check from c-g7 (same binary).
- **Agent turns: the layer prestage made the dense half wait; off + a layer-wide file reader = +300 turn -21 %.**  After
  each layer the engine queued the NEXT layer's every non-VRAM arena expert (~1.2 GB, routed or not) as one DMA burst
  (`chunk_pstage_issue`, upstream default 3 GiB) - the dense half's small uploads then queued behind it on the same
  host-to-device copy engine: a +300 turn's dense half took 1.42 s (~31 ms fixed a layer), 0.27 s with the prestage off
  (`STRATA_PRESTAGE_GIB=0`; a fresh 9.5k prompt is even - the burst only moves from the dense half to the expert phase).
  And a turn's file-tier reads were started per format class pass (up to 9 a layer, each with a sync, 32 new threads and
  a tail): the drives give 9.05 GB/s together (O_DIRECT 1.25 MiB random: models drive 2.38, mirror drive 6.67), turns
  read at ~6.7 -> `STRATA_WAVE_LAYER=1` (ec171094) queues a layer's routed file blobs of every class pass on a
  persistent pool before the first pass.  Both exact (one-shot token md5 6c480f98b832 off / reader / reader + prestage
  off; 0 blobs read outside the queue).  Turn prefill, served config at arena 83:

  | | +300 (439 read) | +1,500 (1,891 read) | fresh 9.5k |
  |---|---:|---:|---:|
  | both off (4 runs) | 4.83 s | 7.09 s | 10.47 s |
  | reader on (4 runs) | 4.56 s | 6.75 s | 10.47 s |
  | prestage off (2 runs) | 4.06 s | 6.47 s | 10.52 s |
  | **both** (2 runs + the c-g11 gate) | **3.80 / 3.94 s** | **6.14 / 6.19 s** | 10.51 / 10.88 s |
  | (arena 88, cand7 gate, for reference) | 4.73 s | 6.94 s | 10.92 s |

  The c-g11 gate (arena 83 + both): prefill 8.7k 680 tok/s (12.9 s), 123k 1,996 tok/s (61.6 s); decode 60.8 / 53.4
  tok/s; tools 6/6, effort medium, regenerate identical; needles 2/3 (the 50 % one answered empty after 54 tokens -
  the same needle did that in c-g5 with the prestage on, 2 of 5 runs tonight); margin 4.00 GiB (anon 91.2 + pinned 8.3 +
  kernel 0.55) - so the deploy candidate is arena 82.5 + park RAM 0.5 GiB (c-g12).
- **Deployed 03:20 = the night's end state** (commit cda47f4d, pushed; snapshot engine 9a668dd25b0d = ec171094's build;
  run.prev = the 02:16 snapshot 9f78a3fc): the 02:16 config + `STRATA_WAVE_LAYER=1` + `STRATA_PRESTAGE_GIB=0` + arena
  82.5 GiB + `STRATA_PARK_RAM_GIB=0.5` (parked conversations are 0.1-0.5 GB; disk restore ~20 ms).  Mode page start:
  health 76 s, thinking level medium, pong via VS Code's id and llm-router.  Gate c-g12: needles 4/4 (506,885 tokens in
  260 s = 1,950 tok/s, 244,647 tokens at 10 / 50 / 90 %), margin under the 104 GiB cap 5.80 GiB bench / 5.22 at 505k /
  4.87 at the 246k needles, no OOM events; tools 6/6 + regenerate identical in c-g11 (same binary).

  | | 22:52 deploy (start of the night) | **03:20 deploy** |
  |---|---:|---:|
  | prefill 8.7k | 670 tok/s (13.1 s) | **686 tok/s (12.7 s)** |
  | prefill 123k | 1,806 tok/s (68.1 s) | **2,010 tok/s (61.2 s)** |
  | prefill 506k (needle) | - | 1,950 tok/s (260 s) |
  | +300 agent turn, wall (prefill) | 6.3 s | **4.7 s (4.00 s)** |
  | +1,500 agent turn, wall (prefill) | 8.8 s | **7.1 s (6.35 s)** |
  | decode 8.7k / 123k (256 tok, sampled) | 54.1 / 47.4 tok/s | 55.3-60.8 / 51.4-53.4 tok/s (c-g12 / c-g11) |
  | memory margin at the worst needle | (not measured; arena 76.5) | 4.87 GiB |
- **DSpark confidence threshold** (`STRATA_DSPARK_CONF`, served 0.6; lower = slower on 10-15) at the served config, park
  off, rb_ab.sh: 0.6 / 0.7 / 0.8 = 9-domain mean 63.2 / 59.9 / 58.7 tok/s, 123k 54.6 / 49.1 / 48.6, 8.7k 57.2 / 58.6 /
  54.4, Marc's traffic (6 requests, blind) 61.6 / 63.8 / 64.6 -> no clear winner, stays 0.6.
- **A smaller prestage instead of none** (the served config with `STRATA_PRESTAGE_GIB` 0.5 / 1.0 instead of 0; one run each):
  8.7k 13.2 / 12.3 s (off: 12.7), 123k 61.4 / 61.7 s (61.2), fresh 9.5k 10.87 / 11.06 s (10.97), +300 turn 4.23 / 4.65 s
  (4.00), +1,500 6.57 / 6.76 s (6.35) - the dense half is still blocked (+300: 0.63 / 1.03 s vs 0.27) by more than the
  overlap saves.  Off stays.
- **Two more tries at keeping the prestage without its blocking** (measured, rejected): `CUDA_DEVICE_MAX_CONNECTIONS=32`
  (the streams' hardware queues; prestage on: +300 turn dense half still 1.41 s; prestage off: the same numbers as
  without it) and a prestage PUMP (opt-in `STRATA_PSTAGE_PUMP=n`: the DMAs from a thread on a stream of its own, at most
  n = 2 blobs in flight; token stream identical) - the dense half stayed blocked (9.5k: 6.69 s vs 3.4 prestage off;
  +300: 1.31 vs 0.27), so the cost is NOT the copy queue depth.  Next suspect: VRAM pressure from the 3 GiB region (the
  dense half's graph buffers resizing; `STRATA_DENSE_PROF` counts the resizes) - needs a profile.  The pump code is NOT
  committed (patch: `pf/pstage-pump-REJECTED.patch`).
- **Not done / next**: (1) GPU-side routing for the verify pass - the router readback -> classify -> metadata upload is
  ~50 us of GPU idle a layer (nsys pv0: 6,662 x 49.5 us), but most layers also wait on a PCIe expert DMA, so hiding the
  hits launch alone is worth ~2-4 %, and the full win needs device-driven miss copies (zero-copy from the pinned arena) -
  a project; (2) why the prestage slows the dense half (not the copy queue - see above; VRAM pressure is the next
  suspect): fixing it would overlap the ~3 s of arena DMA of a fresh 8.7k prompt with its dense half (prestage on 8.7k:
  dense 6.1 + experts 5.3 s; off: 3.1 + 8.7 s); (3) `MIMO_CHUNK_STAGE_MIB=512` would free 1 GiB pinned + 1 GiB VRAM (the bounce halves are 2 x 1 GiB,
  a turn's layer uses ~0.5 GB) = arena +1 GiB at the same margin - untested; (4) flash-attn with heads as columns (123k
  dense half 42 s); (5) the models drive itself: 2.38 GB/s for 1.25 MiB random reads (the SN7100 6.67) caps every
  file-tier read that is not mirrored.

## Q2_K DSpark sidecar (2026-10-17 01:13-01:45, unattended) - NOT shipped
IndrasMirror's `V41-MTP-Vision.Q2_K.gguf` (5.66 GB vs 8.42 GB; their claim: 3.23 vs 3.28 drafts/pass, +2.76 GB VRAM for
experts, ~+10 % decode at depth 4). Downloaded with wget -c in a 1 GiB user unit to `models/deepseek-v41/gsq/`, sha256
f4103b81... = the HF LFS hash. Kit: throwaway `ds41-pf` (run.sh, GEN 256, TURNS 1) + one-shot `ds41-nsys` (NSYS=0
nsys_decode.sh, greedy, 1,536 tokens, arena 82.5); configs = the served one + env STRATA_SERVE_DSPARK / STRATA_DSPARK_DEPTH
(`mab776/pf/tmp-cfg/q2k-*`). Order A B C C B A. Raw: `pf/q2k-*`, `pf/nsys/q2k-os-*`.

| | A: 8.42 GB sidecar, depth 5 (live) | B: Q2_K, depth 4 | C: Q2_K, depth 5 |
|---|---|---|---|
| decode 8.7k (2 runs, mean) | 58.0 / 53.0 = **55.5** tok/s | 54.3 / 58.2 = 56.2 (+1 %) | 56.6 / 56.4 = 56.5 (+2 %) |
| decode 123k | 53.7 / 56.0 = **54.9** tok/s | 52.2 / 49.9 = 51.0 (-7 %) | 50.5 / 52.8 = 51.6 (-6 %) |
| drafts accepted / pass (one-shot greedy code) | 3.59 (4.59 tok/pass, verify 41.4 ms) | 3.26 (4.26, 38.7 ms) | 3.48 (4.48, 39.9 ms) |
| serve tokens / pass (sampled, all requests) | 2.26-2.29 | 2.07-2.29 | 2.19-2.24 |
| one-shot decode | 103.8 tok/s | 103.0 | 105.1 |
| prefill 8.7k | 693 tok/s (12.6 s) | 665 tok/s (13.2 s) | 723 tok/s (12.1 s) |
| prefill 123k | 1,997 tok/s (61.5 s) | 2,007 tok/s (61.2 s) | 2,015 tok/s (61.0 s) |
| agent turn +300 / +1,500 | 4.6, 4.5 / 7.1, 6.9 s | 4.6, 4.7 / 6.9, 7.1 s | 4.7, 4.7 / 6.9, 6.9 s |
| DSpark weights on GPU 0 | 4.29 GB | 2.91 GB | 2.91 GB |
| experts: GPU 0 slots / GPU 1 tier | 36.10 GiB of slots / 1,237 | the same 36.10 GiB / 1,237 | the same |
| GPU 0 VRAM used (bench) | 47.5 GB | 47.5 GB | 47.5 GB |
| memory peak anon+pinned+kernel / host avail min | 98.4 GiB (5.6 under 104) / 14.7 GiB | 98.0 / 15.0 GiB | 98.1 / 14.9 GiB |
| **one-shot token md5** | **e84e4be9c62a** (repeat: identical) | **61fe94bb7419** - differs from token 10 | **c3e7ecc14f15** - differs from token 10 (B vs C from token 229) |

**Verdict: not shipped**, on two counts of the rule:
1. **No speed gain**: +1-2 % at 8.7k (inside the ±4 % noise) and -6/-7 % at 123k; turns and prefill unchanged. The 1.4 GB
   the smaller head frees on GPU 0 is not handed to the expert cache in this engine (the slot budget is sized "after the
   dense half" before DSpark loads: 36.10 GiB in every arm, the same VRAM used), so only the lower acceptance shows.
   **(Corrected 10-17 morning: wrong - that line is printed before the drafter loads; the cache is seeded after it and
   the Q2_K arms had +143 experts in VRAM. See "The two DSpark bugs" below.)**
2. **md5 differs** from A (A reproduces itself bit-for-bit): the greedy token stream changes at token 10 with the Q2_K head,
   and depth 4 vs 5 on the same head diverges at token 229. So drafts are not output-neutral in this engine: the verify pass's
   batch shape (how many drafted rows it carries) changes the target model's logits enough to flip a near-tie. Reported as
   a bug (verify not batch-invariant), not investigated.
Live config untouched (stack 78 not started; mode page left OFF / V4.1 as found). The Q2_K file stays on disk for Marc.
If the slot sizing ever counts the freed VRAM, re-test C (the fastest one-shot, 105 tok/s).

## The two DSpark bugs (2026-10-17 08:00-, unattended; Marc: "Unattended")
Kit: one-shot `ds41-nsys` runs (`strata-ds41/mab776/pf/verify_ab.sh` -> `nsys_decode.sh` NSYS=0: greedy, 1,536 tokens after
the 4,096-token C++ prompt, the served engine flags, arena 82.5) on `build-g5` (build-ds4-gpu's cmake config + the ggml
patch below); raw logs + token files `pf/nsys/{vb,bi2,eq,fin,os}-*`. Commit d7635117 (pushed to mab776/Strata-DS4).

**Bug 2 - "the VRAM a smaller drafter frees does not go to experts": not a bug, last night's misread.** The line
`VRAM: 37.79 GiB free after the dense half -> 36.79 GiB of expert slots` is printed at load, BEFORE DSpark loads; the
expert cache is seeded later (`open_cache`, after the drafter, from the VRAM free then - upstream's design). That seed
was not logged; it is now (`VRAM with the drafter loaded: ... experts in VRAM`):

| | live drafter (8.42 GB) | Q2_K drafter (5.66 GB) |
|---|---:|---:|
| VRAM free at the cache seed (one-shot) | 32.60 GiB | 33.89 GiB |
| expert slots -> experts in VRAM | 31.60 GiB -> 3,669 | 32.89 GiB -> **3,812 (+143)** |
| serve, last night's A/B logs: prompt cache | 21.39 GiB (2,318 experts) | 22.68 GiB (2,435) |

So last night's Q2_K arms already had the extra experts; they were not faster because the Q2_K head accepts fewer drafts
(3.26-3.48 vs 3.59 a pass).

**Bug 1 - the draft changed the output: verify passes were not batch-invariant. Fixed (opt-in `GGML_BATCH_INVARIANT=4`).**
Stock: the live drafter at depth 5 vs the same drafter with every draft trimmed (`STRATA_DSPARK_CONF=1.5` = one-row
passes = one-token decode through the verify pass) differ from token 10. ggml runs a 2..8-row call on other kernels than
a 1-row one, and the engine had a second one-token path:
- float weights (the F16 hyper-connection mixes, compressors, indexer; the F32 router): 2..8 rows on the mmf
  tensor-core tiles, 1 row on the vector kernel;
- quantized weights (Q8_0 attention projections, shared expert, logits head - MMVQ): the warp count follows the column
  count (4 warps for 1-4 columns, 2 for 5-8, a small-K variant only at 1);
- 1-row fused epilogues (2..8 rows never fuse);
- flash attention: the stream-K KV split is sized for occupancy, so its split points move with the number of query tiles;
- `step()` (the one-token decode without a drafter, and --serve's plain-decode fallback) runs the routed experts on
  one-token kernels, the verify pass on the batched per-class ones.

Fix: `tools/v41/ggml/mab776-batch-invariant.patch` - with `GGML_BATCH_INVARIANT=C` a <= 8-row matmul / attention computes
every row as a 1-row call would (vector kernel for float weights, the 1-column MMVQ warp count, no 1-row fusions, C KV
blocks per attention output tile; the first try split the PROMPT's attention too and made the prompt non-deterministic,
so prompt batches keep the stock kernels); and ds41_generate decodes a lone token through a one-row verify pass
(`decode_one`: 21.0 ms a pass vs ~21.4 ms a step). Unset = the stock kernels, bit for bit the old binary's paths.

One-shot md5 of the 1,536 greedy tokens ("from N" = first token that differs from the row's reference):

| arm | stock | `=4`, cache as free VRAM gives | `=4`, cache pinned at 31.5 GiB (`STRATA_SEED_GIB`, 3,657 experts) |
|---|---|---|---|
| live drafter, depth 5 | e84e4be9c62a | 4bcc9c9e77b1 (= the one-row arm) | **e24ef0567509** (repeat on the final build: same) |
| live drafter, one-row passes | 85e2c528d276 (from 10) | 4bcc9c9e77b1 (x2) | **e24ef0567509** |
| Q2_K drafter, depth 4 | (61fe94bb7419, last night) | 796292deac66 (from 130; 3,812 experts) | **e24ef0567509** |
| Q2_K drafter, depth 5 | (c3e7ecc14f15, last night) | 796292deac66 (= depth 4) | **e24ef0567509** |
| no drafter | c6061f4a908e (from 10) | f9a75f41ded4 (from 130; = the Q2_K stream to token 181) | **e24ef0567509** (prompt tail through the verify pass, `STRATA_TAIL_PV=1`) |

The no-drafter steps at the pinned cache: prompt in one batch + step() decode = from token 94 (a drafter runs the
prompt's last 128 tokens through 16-row verify passes - it needs their hidden states); tail through the verify pass +
step() = from 130; + `decode_one` = identical.

**What is left: the output follows the VRAM set.** The served route bias favours experts in VRAM and penalizes the file
tier, so a different VRAM set routes some tokens differently: the Q2_K drafter's 143 extra experts change the stream from
token 130, and the cache pinned at 31.5 instead of 31.6 GiB (12 experts fewer) is another stream again. That is the bias
picking experts by where they are, not the drafts: at an equal cache every drafter / depth / no drafter gives the same
tokens. A drafter that frees VRAM can therefore not be both md5-identical to A and use the freed VRAM for experts.

**Gate** (throwaway `ds41-pf`, order A B B A; A = the served config, B = + `GGML_BATCH_INVARIANT=4`, same binary;
raw `pf/g{A,B}{1,2}*`):

| | A1 | B1 | B2 | A2 | A mean -> B mean |
|---|---:|---:|---:|---:|---|
| decode 8.7k (256 tok, sampled) | 57.0 | 56.9 | 55.6 | 58.7 | 57.9 -> 56.3 tok/s (-2.8 %, noise) |
| decode 123k | 53.8 | 52.9 | 60.8 | 57.4 | 55.6 -> 56.9 tok/s (+2.2 %) |
| prefill 8.7k | 698 tok/s (12.5 s) | 692 (12.6 s) | 612 (14.3 s) | 673 (13.0 s) | first request (c-g9 had 601) |
| prefill 123k | 1,953 tok/s (63.0 s) | 1,983 (62.0 s) | 1,981 (62.1 s) | 1,993 (61.7 s) | same |
| fresh 9.5k / +300 / +1,500 turn prefill | 11.09 / 4.06 / 6.33 s | 11.13 / 4.02 / 6.36 s | 10.99 / 3.98 / 6.33 s | 10.99 / 3.95 / 6.34 s | same |

B also: needles **4/4** (479,953 tokens in 248 s = 1,935 tok/s; 232,660 at 10 / 50 / 90 % in ~120 s = 1,939 tok/s),
tools **6/6**, thinking level medium, regenerate identical, memory peak 98.2 / 98.5 / 98.8 GiB (bench / 480k / 246k) =
**margin 5.79 / 5.54 / 5.18 GiB** under the 104 GiB cap, no OOM events (the one `cudaMalloc failed ... 4671 MiB` line is
the long-prompt allocator retry fixed in f81f2808 - the same line shows in earlier runs since 10-15). One-shot greedy verify cost:
37.4 ms a pass at 4.99 tokens/pass vs stock 39.3 ms at 4.59.

On the final binary (md5 54fb9fb30465): `=4` depth 5 = one-row passes = 4bcc9c9e77b1; unset = e84e4be9c62a (the stock
stream, bit for bit).

**Deployed 09:50** (commit 74faff41 = the served config + `GGML_BATCH_INVARIANT=4`, pushed; `snapshot.sh` from build-g5 =
build-ds4-gpu's cmake config; run.prev = the 10-16 03:20 snapshot cda47f4d): PUT stack 78 (live compose = the reference,
unchanged), health 70 s, thinking level medium, the engine process carries the variable, pong direct with VS Code's id
and through llm-router; mode page back OFF (DeepSeek tile still V4.1).

**Q2_K re-test with the fix** (Marc's rule: >= +5 % decode at both depths, turns not slower, md5 identical, tools 6/6,
needles 4/4). A = the deployed config (live drafter, depth 5), B = Q2_K drafter depth 4, C = Q2_K depth 5, all with
`GGML_BATCH_INVARIANT=4`; order A B C C B A, throwaway `ds41-pf`, raw `pf/rq-*`:

| | A (live) | B: Q2_K depth 4 | C: Q2_K depth 5 |
|---|---|---|---|
| experts in VRAM (serve) / prompt cache | 3,746 / 21.39 GiB | **3,891 (+145)** / 22.67 GiB | 3,891 / 22.67 GiB |
| decode 8.7k (2 runs) | 55.6 / 59.8 = **57.7** tok/s | 56.8 / 56.8 = 56.8 (-1.6 %) | 59.7 / 70.0 = 64.9 (+12 %, one 70) |
| decode 123k | 56.2 / 48.1 = **52.2** tok/s | 52.9 / 49.7 = 51.3 (-1.6 %) | 52.6 / 52.0 = 52.3 (+0.3 %) |
| prefill 8.7k / 123k | 711 tok/s (12.3 s) / 1,973 (62.4 s) | 700 (12.5 s) / 2,002 (61.4 s) | 702 (12.5 s) / 2,005 (61.4 s) |
| turn prefill fresh 9.5k / +300 / +1,500 | 11.06 / 4.03 / 6.34 s | 11.16 / 4.08 / 6.39 s | 11.16 / 3.98 / 6.35 s |
| one-shot greedy (free cache) | 4.99 tokens/pass, 37.4 ms, 123.3 tok/s | 3.79, 35.3 ms, 99.2 tok/s | 4.10, 37.1 ms, 102.3 tok/s |
| one-shot md5 | 4bcc9c9e77b1 | 796292deac66 | 796292deac66 (= B) |

**Verdict: not shipped.** No depth reaches +5 % at both depths (C: +12 % at 8.7k on one 70 tok/s run, +0.3 % at 123k);
on the greedy code stream the Q2_K head accepts fewer drafts (4.10 vs 4.99 tokens a pass) and the 145 extra experts do
not make up for it. md5: identical to A at an equal cache (e24ef0567509), different at the cache it frees (the route bias
follows the VRAM set) - depth 4 vs 5 on the Q2_K head now agree. Live config = the 09:50 deploy; mode page left OFF
(DeepSeek tile V4.1). The Q2_K file stays on disk.


## RAM at DDR5-4400 (2026-10-18, was 4000) - live stack 78, same config (74faff41 + GGML_BATCH_INVARIANT=4)
Through the mode page (V4.1, ready in 80 s), the bench of run.sh by hand on :8330 (strata_bench 8000/127000 GEN 256,
turn_bench), 3 rounds (turns 2). Reference = 10-17 morning gate at 4000 (served config + BINV, ds41-pf container).

| | DDR5-4000 (10-17 gate) | DDR5-4400 (3 rounds) | change |
|---|---|---|---|
| decode 8.7k | 56.3 tok/s | 60.7 / 64.3 / 61.9 = **62.3** | **+11 %** |
| decode 123k | 56.9 tok/s | 48.9 / 54.1 / 53.3 = 52.1 | -8 % (wide spread; the 10-17 A arms were 53.7-56.0) |
| prefill 8.7k | ~690 tok/s (12.6 s) | 721 (12.1 s) / 884 (9.9 s) / 885 (9.9 s) | +5..+28 % (first read colder) |
| prefill 123k | ~2,000 tok/s (61.5 s) | 2,039 / 2,046 / 2,045 = **2,043 tok/s (60.2 s)** | +2 % |
| agent turn +300 / +1,500 | 4.0 / 6.3 s (gate; A arms 4.5 / 7.0) | 4.6, 4.6 / 6.7, 6.9 s | in the range of the A arms |
| memory | margin 5.18 GiB | anon+shmem+kernel 98.8 GiB (5.2 under 104), host min 13.2 GiB | same |
No nonfinite / errors in the engine log. Mode page back OFF.

## STRATA_TIER_BATCH_PCIE (upstream Xolotl f7c561b7, 2026-10-18)

Upstream made the env knob a flag (`--tier-batch-pcie`) and said "many-core hosts: try 0". Our engine already reads the
env (`ds41_moe.cpp` `gpu_run_n_ex` = the DSpark verify passes; prompt chunks take another path, so prefill shows 0 cpu in
every arm). A/B on the live engine (build-g5, md5 54fb9fb30465) + live config, throwaway ds41-pf, DDR5-4400,
2 interleaved rounds, results `ram-128gb/v41/pf/tbp-*`:

| arm | decode 8.7k | decode 123k | prefill 8.7k | prefill 123k | turn +300 / +1,500 |
|---|---|---|---|---|---|
| A = 1.0 (live) | 52.4 / 58.9 = **55.7** | 55.0 / 50.7 = **52.9** | 699 / 634 tok/s | 1,958 / 2,030 tok/s | 4.4-4.5 / 6.7-6.8 s |
| B = 0 (CPU pool) | 50.5 / 49.6 = **50.1** (−10 %) | 48.5 / 50.4 = **49.5** (−6 %) | 675 / 692 | 2,028 / 2,026 | 4.3-4.5 / 6.8 s |
| C = 0.5 | 55.0 / 57.2 = **56.1** | 52.9 / 50.0 = **51.5** | 707 / 692 | 2,031 / 2,027 | 4.5-4.6 / 6.8-7.0 s |

Verdict: **keep the default (1.0).** 0 is slower on decode (the pool's ~2 ms per expert stays the critical path even on
16 cores / DDR5); 0.5 = noise. No errors, container peak ~102.8 GiB (page cache included) as before.
