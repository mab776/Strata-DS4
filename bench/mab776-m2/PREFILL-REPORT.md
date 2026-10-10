# Prefill >= stack 73 (Marc 10-10: "prefill first") - 2026-10-10

Target: llama.cpp stack 73 without its drafter, 1011 / 843 / 693 tok/s at 8k / 127k / 250k. Gates: decode >= M7a
(44.4 / 42.0 / 40.6), ppl768 in band, needle at 251k FOUND. Harness: m2run.sh, ctx 262,144, code1 prompts,
`G1=1 GROW=2.0 MARGIN=9.0 CHUNK=2048 MMQ=1 PRESTAGE=1` (+ `MMQ_FP4=1` = env STRATA_MMQ_FP4=1 in the last row).

| step (commit) | 8k | 127k | 251k (needle) | what it fixed |
|---|---|---|---|---|
| M7a (8d24f746) | 683 | 495 | 323 | no NVMe in the prompt (GPU 1 tier, complete ranking, grow) |
| resident experts on MMQ (1202f420) | 760 | 536 | - | GPU 0's resident experts copied into the stage halves by SM kernels, then MMQ (was MMVQ from the slots) |
| batched sparse CSA attention (c2e20ee2) | 810 | 761 | 632 FOUND | one flash-attn over [compressed | raw] per CSA chunk with n_kv_max = SWA + top-k (ggml-cuda's sparse gather); no per-block K concat/cast/pad; the O(n^2) output concat chain gone |
| GPU 1's share on MMQ (c0b946cf) | 968 | 883 | - | GPU 1 gathers its experts into a strided stage, same MMQ products; it had become the expert phase's critical path |
| fused indexer in one call (ba4a7fe2) | 970 | 930 | 723 FOUND | the indexer's 64-query blocks were joined by an O(n^2) concat chain (~8 GB a CSA layer-chunk at 250k) |
| shallow CSA chunks dense in one call (9a555a04) | 973 | - | - | below the sparse threshold too: no per-block K copies |
| **+ STRATA_MMQ_FP4=1** | **1018** | **970** | **752 FOUND** | layers 26 / 42 (MXFP4 down) on Blackwell's FP4 x FP4 MMQ instead of MMVQ - llama.cpp's own arithmetic there |
| stack 73 (llama.cpp b11429, no draft) | 1011 | 843 | 693 | |

- Decode in the same runs: 45.3 (8k), 40.9-42.1 (127k, 304-512 tokens), 39.6-41.3 (251k, 103-269 tokens). Shorter
  answers carry more of the CUDA-graph warm-up; prefill changes do not touch the decode path.
- ppl768: 9.8468-9.9544 across these builds (9.9544 with FP4); it moves ~1 % with summation order alone (sparse /
  dense / MMQ vs MMVQ choices); llama.cpp 9.9624. Needle 50 % at 251k FOUND at every step it was run.
- Rejected: chunk 4096 @margin 11 (842 at 8k), margin 8 (+0.8 %).
- Where the time goes now (127k, FP4): dense 95 s (attention+router 93: the lightning indexer and the projections),
  experts 38 s. 8k: dense 3.7 s, experts 3.8 s; ~185k kernel launches and a rebuilt graph per layer-chunk - host
  overhead is now a visible share.
- **Open choice for Marc:** STRATA_MMQ_FP4=1 is what makes 8k beat stack 73 (968-973 without it). It rounds the
  activations of 2 of 43 layers' down projections to FP4 (llama.cpp does the same on Blackwell); ppl stays below
  llama.cpp's. Recommended ON in the server config; one env var to drop.
