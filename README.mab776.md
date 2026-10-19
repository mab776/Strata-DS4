# Strata-DS4 — mab776 fork: DeepSeek-V4-Flash and V4.1-Flash on one desktop (read this first)

Fork of upstream **[Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl)** (which absorbed Strata-DS4 and
Strata-DSV4.1 on 2026-10-16/18: both engines plus the Strata core and its server in one tree). This fork keeps the two
DeepSeek engines we serve, tuned for one box: Ryzen 9 7950X, 124 GiB DDR5-4400, RTX Pro 5000 48 GB + RTX 5070 Ti 16 GB
(both sm_120), models on NVMe. Branches: **`mab776-main`** = upstream + our changes (the default branch; the stacks build from it) and **`main`** =
an untouched mirror of upstream `Indras-Mirror/Xolotl` `main`, kept only to sync from.

## STATUS (2026-10-19 evening: the Xolotl 87157b2e sync + the dedup pass)
| | V4 — Portainer stack 74 `deepseek-strata` | V4.1 — Portainer stack 78 `deepseek-v41` |
|---|---|---|
| model | DeepSeek-V4-Flash-0731 unsloth UD-Q3_K_M | DeepSeek-V4.1-Flash GSQ (163 GB) + Engram (203 GB) |
| engine | `ds4_serve` (tools/ds4) | `ds41_generate --serve` (tools/v41/engine) |
| live snapshot | tree 3f8c6cfb (tag `mab776-main-x1018-final`), ds4_serve md5 8becae81e296 (the sync brought V4 nothing: not redeployed) | tree fd536a89 (tag `mab776-main-d1019`), ds41_generate md5 3c1373edb6c8 (build-dd); run.prev = 84cd7fec + shim d7a276c0 (tag `mab776-main-s1019`) |
| new on 10-19 | - | **images** (`--vision`, DeepSeek's reference ViT on the CPU, resident, ~2.2 GiB) and **exact routing per request** (model ids ending in `-exact`; the plain ids stay 0.5 / 0.3); evening: **dedup** - prompt lookup and the D3 PCIe-site copies are upstream's code now (diff vs upstream -969 lines) |
| port / alias / model id | :8330 / `ds4-lcpp` / `deepseek-v4-flash` (one of the two runs at a time: mode page DeepSeek tile V4 \| V4.1) | the same |
| prefill 8.7k / 123k | ~940 / ~1,100 tok/s | ~700-715 / ~1,960-2,000 tok/s |
| decode (sampled, 256 tok) | ~50-58 tok/s (MTP K=2, lookup 7) | ~50-60 tok/s (DSpark depth 5); copy-heavy edits 117-144 tok/s (lookup 14); `-exact` 26-31 tok/s |
| RAM | ~77-79 GiB (cap 96) | ~99.5 GiB incl. the image encoder (cap 104, arena 80.0, margin >= 4 GiB at 480k / 246k) |

Exactness: the merged V4.1 engine with `STRATA_TIER_ROWS=8` + `STRATA_GPU1_RESERVE_GIB=0.745` gives the pre-sync one-shot
greedy md5 (4bcc9c9e77b1 at arena 82.5); upstream's 16-row tier calls change the text here. The deployed arena is 80.0
(the encoder's RAM under the 4 GiB rule), so its own md5 differs by residency, not by code. Details: X1018-UPDATE.md "Sync 10-19".
The dedup build gives 4bcc9c9e77b1 too; with upstream's D3 DMA stream on (served) the fast-routing text is deb5da21073c
(deterministic, draft-independent: the stream shifts residency like an arena change); exact routing = the previous 389fbc4c967a.
X1018-UPDATE.md "Dedup 10-19" = what was adopted from upstream, what stays ours and why.

## What this fork adds to upstream
- **V4 (tools/ds4)**: unsloth split GGUFs (`--expert-slots auto`, mixed per-class blobs, `DS4_FORCE_MIXED`), the
  verify-pass activation-row fix (Q3_K_M NaN), GPU 1 as a static expert tier (`--gpu1`), elastic GPU 0 cache per prompt,
  MTP speculative decoding with chained drafts (`--mtp-k`), prompt-lookup drafts (`--lookup`), conversation parking,
  `ds4_serve` (the engine behind Strata's server; `serve/deepseek.py` = the DeepSeek-V4 chat family: DSML tool calls,
  stop ids, effort table), the M11-M14 decode work (the verify mix on the card, GPU-side routing, fused HC ops, lean
  graphs, warp top-k router), the upstream llama.cpp fixes #30103 / #29953 for its ggml tree. History:
  [docs/mab776/V4-NOTES.md](docs/mab776/V4-NOTES.md).
- **V4.1 (tools/v41)**: batch-invariant verify passes (`GGML_BATCH_INVARIANT=4`: the greedy output does not depend on the
  draft) + lone tokens decoded through a one-row verify pass; Q3_K split exact kernels (`STRATA_SPLIT_FORK`); kept verify
  graphs, async finish, 128 decode read parts, the layer-wide file reader (`STRATA_WAVE_LAYER`), conversation parking
  (`STRATA_PARK`, RAM + disk store), keep-cache through short batches (`STRATA_KEEP_CACHE`), the DSpark window kept with the snapshot
  (`STRATA_DSPARK_SNAP`: regenerate +9..+19 % decode), `STRATA_SERVE_LOOKUP_GREEDY` (a greedy cap on upstream's
  prompt lookup `STRATA_SERVE_LOOKUP`: 0 = greedy stays exact), `STRATA_PB_CAP=0` (Xolotl's 24 GB-card prompt-batch cap off), `STRATA_TIER_ROWS` (rows per tier call;
  8 = the pre-76d16570 split, output-identical), GPU 1's static tier on 16-row calls, the FILEG cap per 8 rows of a call,
  the shim's special-`</think>` fix (an answer written straight after a quoted line stayed in the reasoning), the second-drive blob mirror, served
  configs and the bench kit (mab776/pf). Report: [docs/mab776/V41-REPORT.md](docs/mab776/V41-REPORT.md); the x1018
  merge + the 10-18/19 night: [docs/mab776/X1018-UPDATE.md](docs/mab776/X1018-UPDATE.md).
- Kept OFF on purpose: dense Q6_K at load (Marc declined), `STRATA_TIER_BATCH_PCIE` < 1 (lost), the adaptive arena
  (`STRATA_ARENA_REBAL`, no gain here), greedy lookup on V4.1 (a 9-row pass is past the 8 rows batch invariance covers).

## Build (both engines from this tree; Docker, no GPU needed)
Two ggml trees (gitignored, llama.cpp at `third_party/ggml/VERSION.txt` 3cf03257 with each engine's patches applied once):
- `third_party/llama.cpp` = V4.1's: tools/v41/ggml/{upstream-fixes, mmq-deterministic, rank-argsort, kv-fp4,
  dsv41-fake-quant, dsv41-fuse, mab776-cpy-rows, mab776-li-wmma-fp4, mab776-li-mma, mab776-mmvf-f16-batch,
  mab776-topk-radix-select, mab776-batch-invariant}.patch (`patch -p1`);
- `third_party/llama.cpp-ds4` = V4's: tools/ds4/patches/{lightning-indexer-*, zz-dsv4-hc-gates, zzz-dsv4-hc-fused} + the four it
  shares with V4.1 from tools/v41/ggml (top-k radix select, mmvf F16 batch, rank-argsort, upstream-fixes), in the order
  `mab776-build-ds4.sh` lists.

```sh
ENGINE=ds41 BUILD_DIR=build-dd ./mab776-build.sh           # V4.1 -> build-dd/ds41_generate (CUDA 13 container)
ENGINE=ds4 ./mab776-build.sh                               # V4   -> build-ds4-cu13/ds4_serve (+ ds4_generate)
```
`mab776-build-ds4.sh` mounts the V4 ggml tree at the in-container path the V4 binary was always built with.

## Deploy and roll back (Portainer, native stacks)
- V4: `portainer/deepseek-strata/snapshot.sh [REPO] [BUILD_DIR]` (defaults: this worktree, build-ds4-cu13) copies
  ds4_serve, serve/, the tokenizer, the routing profile and ds4-server.json to `models/strata-ds4/run` (previous ->
  `run.prev`), then restart stack 74 (mode page or Portainer API).
- V4.1: `portainer/deepseek-v41/snapshot.sh [WORKTREE] [BUILD_DIR]` -> `models/strata-ds41/run` (+ serve/, the tree's own
  0.1.44 server; previous -> `run.prev`), then restart stack 78.
- Rollback: stop the stack, swap `run` and `run.prev`, start it. Older trees: the `*-final` / `*-pre-x1018` tags below.

## Numbers and how they were measured
- V4: [docs/mab776/V4-NOTES.md](docs/mab776/V4-NOTES.md) (M1-M14, Q3_K_M LCB-13 12/13 vs IQ2_M 9/13).
- V4.1: [docs/mab776/V41-REPORT.md](docs/mab776/V41-REPORT.md) (prefill 3-4x over upstream's defaults, the memory rule,
  the DSpark bugs), [docs/mab776/X1018-UPDATE.md](docs/mab776/X1018-UPDATE.md) (the merge, lookup 14, the arena A/B).
- Exactness harness: one-shot greedy token md5 (`mab776/pf/verify_ab.sh`); V4 is not bit-reproducible across restarts,
  so its check is the greedy probe from one restored state (`bench/mab776-m2/work/m6reg/ksweep2.py`).

## Syncing with upstream
```bash
git fetch xolotl                          # https://github.com/Indras-Mirror/Xolotl
git push mab776 xolotl/main:main          # main = plain mirror, never commit on it
git checkout mab776-main && git merge xolotl/main   # resolve, build both engines, gate, then push mab776-main
```
Rule (Marc, 2026-10-19): our diff vs upstream only shrinks. Where upstream has a feature that does the same job as ours,
adopt upstream's and drop ours (A/B first on the throwaway kit); keep ours only when clearly better, as the smallest
knob / hook on top of upstream's code, with the reason written down.

## Branch history -> tags
| old branch | what it was | tag |
|---|---|---|
| `mab776-main` (before 10-19) | the V4 fork (M1-M14, Q3_K_M) | `mab776-main-pre-x1018` (733ce42a), `mab776-main-final` |
| `mab776-v41` | the V4.1 fork on Strata-DSV4.1 | `mab776-v41-pre-x1018` (74faff41), `mab776-v41-final` |
| `mab776-main-x1018` | V4 merged onto Xolotl f7c561b7 | `mab776-main-x1018-final` (3f8c6cfb) |
| `mab776-v41-x1018` | V4.1 merged onto Xolotl f7c561b7 | `mab776-v41-x1018-final` (026c42f7) |
| `main` (before 10-19) | a stale 10-09 copy of upstream Strata-DS4 | `upstream-main-1009` (924516a0) |
| `main` (10-19 night, 415938a6) | briefly the merged branch; since 10-19 that is `mab776-main` again and `main` mirrors upstream | — |
