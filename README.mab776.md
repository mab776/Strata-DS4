# Strata-DS4 — mab776 fork notes (read this first)

Fork of https://github.com/Indras-Mirror/Strata-DS4 (one developer, no releases, very active) at
https://github.com/mab776/Strata-DS4. Goal: run DeepSeek-V4-Flash-0731 faster than our llama.cpp stack (stack 73
`deepseek-v4`, container `ds4-lcpp`, :8330) using Strata's VRAM hot-expert cache, on OUR unsloth files.

## Repo layout / git
- Local clone: `~/Documents/portainer/strata-ds4`. Branch `mab776-main` = our work; `main` on the fork = upstream mirror.
- `origin` (Indras-Mirror) push URL is DISABLED on purpose. `mab776` remote = the fork. `core.sshCommand` is set
  (`ssh -i ~/.ssh/github_new -o BatchMode=yes -o IdentityAgent=none`, see memory git-ssh-gpg-agent-hang). Author mab776.
- Sync upstream: `git fetch origin && git push mab776 origin/main:refs/heads/main && git rebase origin/main` (on mab776-main).

## Build / run
- `./mab776-build.sh` — throwaway nvidia/cuda:12.8.1-devel container, no GPU, sm_**120a** (plain 120 can't assemble the
  MXFP4 block-scale MMA), full llama.cpp checkout of the pinned commit in `third_party/llama.cpp` (ds4_dense needs
  ggml-impl.h). `BUILD_MEM=32g BUILD_CPUS=16 BUILD_J=14` when the box is idle. Log `build-mab776.log`. ~3-10 min.
- `MODEL=<gguf> SLOTS=5300 ARENA=40 MEM=54g ./mab776-run.sh --ids 0,… -n 64` — GPU 0 only, RAM-capped container
  `ds4-run` (cgroup OOM-kills it instead of freezing the box; replaces upstream's memguard.sh). devel image = libnccl.
- Engine = `build-ds4-gpu/ds4_generate`: token ids in, token ids out (no server yet). Tokenize/decode with
  `tools/strata_tokenizer.py` (needs `regex`: run it in a `python:3.12-slim` container + `pip install regex`; the
  encoder does not parse special tokens — BOS is id 0, prepend it yourself).

## Models (do NOT download upstream's 81 GB Q2 — Marc 10-09: make OUR files work)
- UD-IQ2_M (running in stack 73): `/media/mab776/nvme 2To/models/ollama/models/blobs/sha256-a79d3b8c742f240dc947758bbeb46d266e9328ace6a3b71704ab4bf2a8610d68`
  (84.7 GiB: gate/up IQ2_XXS (1 layer IQ2_S), down IQ3_XXS (2 layers MXFP4)).
- UD-Q3_K_M: `/media/mab776/nvme 2To/models/deepseek-v4-q3km/…-00001-of-00004.gguf` (119.3 GiB: gate/up IQ3_XXS
  (1 layer MXFP4), down native MXFP4). Needs the planned 128 GB RAM upgrade (box has 61 GiB today).
- DSpark drafters: `/media/mab776/nvme 2To/models/deepseek-v4-draft/` (BF16 source + our `dspark-exps-q2_K.gguf`).
- Routing profile: `bench/ds4-2026-10-05/route-probe/ds4routes.bin` (model-level, valid for any 0731 quant).

## What we changed (commit 53de26c on mab776-main)
- geometry: `vocab_size` from `tokenizer.ggml.tokens`, `nextn_predict_layers` = 0 when the keys are absent (unsloth).
- `check_ds4_model`: other dequantizable types accepted; dense tensors converted at load to the type the graphs/fused
  kernels expect (`load_weights(..., want_type)`), routed experts keep their file type (per-layer types supported).
- `load_weights`: F32 sources in the requantize path (F32 has no `to_float`).
- `iq_kernels.cu`: MXFP4 gate/up and IQ3_XXS down added to `STRATA_GU_FMTS` / `STRATA_D_FMTS`.
- `tests/core/ds4_geometry_test` still FAILs 3 author-file-specific asserts on our files (nextn requested 1, output.weight
  Q8_0, shard-1 tensor count) — expected, not engine problems.

## Measured 2026-10-09 (IQ2_M, GPU 0 only, ARENA 40 GiB of 61, SLOTS 5300, short prompt)
- cold: 25.5 tok/s, expert hits 83 %; code continuation warm: **34.5 tok/s**, hits 89 %. Output coherent (fibonacci).
- ms/token: attention+router 15.0 (upstream is attacking this), experts 10.6, predict 1.5, finish 1.1, head 0.5;
  cold run had ~10 ms/pass of file reads (arena too small for 61 GB RAM).
- Baseline llama.cpp stack 73 (both GPUs): no drafter 1011/28.0 @8k, 843/25.0 @127k, 693/22.3 @250k (prefill/decode);
  with DSpark Q2_K drafter ~40 tok/s code @8k-127k, ~36 @250k. LCB 34/39, needles 3/3 @251k. (`deepseek-v4/RESULTS.md`)

## Known gaps / next steps (in order)
1. `--slots auto` oversizes on our files (44.7 GB asked vs 39.4 free): it sizes slots from one layer's expert bytes while
   MXFP4/IQ3_XXS layers are bigger -> size from the max (or per-layer) bytes. Workaround: SLOTS=5300.
2. Depth bench vs llama.cpp at 8k / 127k / 250k: prefill (`--prefill-chunk 4096 --chunk-mmq --chunk-prestage`, needs the
   full arena = no `--arena-skip-resident`; tight on 61 GB) and decode (`--pos-offset N` probe and/or real long prompts via
   `--ids-file`), long-context KV `--comp-type iq4_nl --icomp-q8`, `--ctx 262144`. Record RAM/VRAM too.
3. 250k needles (same haystack as deepseek-v4/needle_ds4.py) to prove long-context correctness.
4. Server: OpenAI chat API + DeepSeek tool calls + reasoning_content (repo has Strata's `serve/`; DS4 isn't wired into
   it yet) so VS Code / OpenHands can use it.
5. Drafter on CUDA (upstream `--mtp` is CPU-tested only, 63 % acceptance) — ideally on GPU 1 so GPU 0 keeps the cache.
6. Q3_K_M once the RAM is in. 7. Mode-page tile + NATIVE Portainer stack, then SWE Verified-mini vs stack 73.

## House rules
- GPUs: ask Marc before taking them (mode page :8500 must be "all off"/DeepSeek); never touch Strata/AuK while he works.
- Never `pkill -f` (kills your own shell) — kill by PID. Cap every big container's RAM. Clean up temp files.
- Parked side idea: our own mixed quant — `deepseek-v4/CUSTOM-QUANT-IDEA.md`.
