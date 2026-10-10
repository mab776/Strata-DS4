# Strata-DS4 — mab776 fork notes (read this first)

Fork of https://github.com/Indras-Mirror/Strata-DS4 (one developer, no releases, very active) at
https://github.com/mab776/Strata-DS4. Goal: run DeepSeek-V4-Flash-0731 faster than our llama.cpp stack (stack 73
`deepseek-v4`, container `ds4-lcpp`, :8330) using Strata's VRAM hot-expert cache, on OUR unsloth files.

## STATUS 2026-10-10 evening - LIVE as Portainer stack 74 `deepseek-strata` (read this, then the session notes)
- **Runs**: container `strata-ds4`, image `strata-ds4:cu13` (portainer/deepseek-strata/Dockerfile: Ubuntu 24.04 +
  Python + libnccl; nothing of the engine baked in), serve/server.py + `ds4_serve` from the DEPLOY SNAPSHOT
  `models/strata-ds4/run` (bind-mounted read-only at /app; the previous one is `run.prev`). A DROP-IN for llama.cpp
  stack 73: host :8330, network alias `ds4-lcpp`, model id `deepseek-v4-flash` (+ alias `deepseek-v4-flash-strata`).
  Mode page :8500 DeepSeek tile: **llama.cpp | Strata** picks the engine (they share the port: never both).
- **Update the engine**: build here (mab776-build.sh, BUILD_DIR=build-ds4-cu13, CUDA 13 image) ->
  `portainer/deepseek-strata/snapshot.sh` (copies ds4_serve, serve/, the tokenizer files, the profile and
  ds4-server.json into run/) -> restart the container (mode page Restart, or `docker restart strata-ds4`).
  Rollback: stop it, swap run/ and run.prev/, start it. Config edits: portainer/deepseek-strata/ds4-server.json, then
  snapshot. Stack edits: portainer/deepseek-strata/docker-compose.yml + deploy.sh (native stack via the API).
- **Engine config** (ds4-server.json): `--gpu1 1` (GPU 1 tier) `--vram-margin 9 --vram-grow 2` (elastic GPU 0 cache,
  shrunk per prompt) `--prefill-chunk 2048 --chunk-mmq --chunk-prestage` + env `STRATA_MMQ_FP4=1` (Marc: keep it)
  `--mtp ... --mtp-device 1` (speculative decoding, the MTP head's experts on GPU 1) `--park-mib 4096` (conversation
  parking) `--arena-gib 41`. ~42 GiB container RAM, mem_limit 50g.
- **Measured through the server**: prefill 1,104-1,362 / 1,007 / 749 tok/s at 8k / 127k / 250k (stack 73: 1011 / 843
  / 693); decode 55.8-59.0 / 49.7 / ~43 tok/s (stack 73 with its drafter 40.8 / 39.0 / ~36); needles 3/3 at 240k;
  parked chat back in 1.3-1.4 s; agent turns read only what they appended.
- **Clients**: OpenHands host profile `DeepSeek`; VS Code block portainer/deepseek-strata/vscode-copilot.txt; Open
  WebUI via **llm-router** (stack `llm-router`, :8340, portainer/llm-router) - OWUI's ONLY OpenAI connection now.
- **Open**: M9 (LCB 39 at 32k t1.0, SWE Verified-mini vs stack 73) = a night session Marc triggers; M10 prefill;
  verify pass at 250k still 1.8x a step; serve/ fixes of Strata 0.1.42 worth taking (#984 thinking budget, #1620,
  #1624 tool order, #1601 cgroup RAM) = a rebase-picture change, ask Marc.

## Repo layout / git
- Local clone: `~/Documents/portainer/strata-ds4`. Branch `mab776-main` = our work; `main` on the fork = upstream mirror.
- `origin` (Indras-Mirror) push URL is DISABLED on purpose. `mab776` remote = the fork. `core.sshCommand` is set
  (`ssh -i ~/.ssh/github_new -o BatchMode=yes -o IdentityAgent=none`, see memory git-ssh-gpg-agent-hang). Author mab776.
- Sync upstream: `git fetch origin && git push mab776 origin/main:refs/heads/main && git merge origin/main` (on mab776-main;
  the branch is published: MERGE, never rebase/force-push). Then rebuild + CPU tests, note the upstream hash.
- Other fetch-only remotes: `niko` ref niko/main = upstream Strata v0.1.41 (fb58e0db), `strata-local` = the LIVE src-0141 tree
  (read/fetch only — never commit/checkout/build in it), `jackwwg83`, `aleksche`, `mobile2machan`, `strata-glm` (see survey).

## Build / run
- `./mab776-build.sh` — throwaway nvidia/cuda:12.8.1-devel container, no GPU, sm_**120a** (plain 120 can't assemble the
  MXFP4 block-scale MMA), full llama.cpp checkout of the pinned commit in `third_party/llama.cpp` (ds4_dense needs
  ggml-impl.h). `BUILD_MEM=32g BUILD_CPUS=16 BUILD_J=14` when the box is idle. Log `build-mab776.log`. ~3-10 min.
- `MODEL=<gguf> SLOTS=5300 ARENA=40 MEM=54g ./mab776-run.sh --ids 0,… -n 64` — GPU 0 only, RAM-capped container
  `ds4-run` (cgroup OOM-kills it instead of freezing the box; replaces upstream's memguard.sh). devel image = libnccl.
- Engine = `build-ds4-gpu/ds4_generate`: token ids in, token ids out (no server yet). Tokenize/decode with
  `mab776-tok.py` (wraps tools/strata_tokenizer.py; needs `regex`: run it in a `python:3.12-slim` container + `pip install regex`; the
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
- GPUs: Marc 10-10: "you are the priority to use the GPU until your task is completed" (the old per-window ask was a
  mistake). Use the mode page (:8500) to free them (DeepSeek/All off); never touch Strata/AuK/GLM while Marc works on
  them; leave DeepSeek on Strata when done so he can test.
- Never `pkill -f` (kills your own shell) — kill by PID. Cap every big container's RAM. Clean up temp files.
- Parked side idea: our own mixed quant — `deepseek-v4/CUSTOM-QUANT-IDEA.md`.

## Fork survey 2026-10-09 (all 1,733 forks of Niko1221/Strata scanned; 488 with own code fetched, ~9.3k unique commits)
No fork is a better ENGINE base for our box than Strata-DS4. Worth borrowing (fetch-only remotes on this clone, MIT):
- `jackwwg83` (github.com/Jackwwg83/Strata-DS, DeepSeek **V4.1**, Linux+NVIDIA, 279 commits, serve on upstream 0.1.40.2):
  the most complete DeepSeek SERVER integration — `serve/deepseek.py`, `serve/fake_ds41_engine.py`,
  `serve/test_deepseek_protocol.py`, snapshots (parking), `--batch`, release-acceptance scripts. Branches feature/ds41-*.
- `aleksche` branch `dev` (github.com/Aleks-Che/Strata, Windows, RTX 5090 + 128 GB, GPU-only expert streaming ~18 tok/s):
  compact `serve/deepseek.py` (115 lines: DeepSeek chat template with thinking + reasoning_effort mapping, strict DSML
  tool-call parser incl. streaming, malformed DSML -> text) + `serve/test_deepseek.py`; docs/deepseek-v4-flash-0731/ (DSpark tuning).
- `mobile2machan` branch `dsv4`: DS4 inside the Strata core, abandoned (~1 tok/s, "does not pay off on a 24 GB card");
  only design notes (docs/DSV4.md: unsloth per-layer type mix, DSpark serve controller).
- Upstream DS4 author's newer Strata-GLM repo has a byte-sized `--slots auto` (our gap #1).
- Not useful for us: Yasei-no-otoko / perronemirko `dsv4_ext` (AMD/Windows HIP), cklxx deepMoE (Vulkan, rejected upstream).
Upstream Strata refused DeepSeek 3x (#310, #479, #943: "out of scope ... external plugin") — no DeepSeek coming from upstream.

## Rebase decision 2026-10-09 (multi-agent dry run; don't redo without a reason)
Keep the DS4 engine on DS4's own v0.1.39 base. Merging DS4 onto v0.1.41: 8 core files / 14 hunks (~310-340 lines) + a hidden
`swiglu_entries_kernel` arity break nvcc found; ggml pin identical (3cf03257); v0.1.41's kernel gains are Qwen-size / AMD / opt-in only
(7950X uses the AVX-512 path anyway); conflicts grow ~1.5 hunks per DS4 day. TAKE the server instead: `git checkout fb58e0db -- serve`
(plain v0.1.41, zero conflicts) + hand-merge tools/strata_tokenizer.py (v0.1.41 piece cache + DS4 `_pre_split`, passes both suites).
Revisit only if DS4 moves base, an upstream kernel is needed for our types, or DS4's pace drops — and ask Marc.

## Session 2026-10-09 night (ultracode handoff: M1 + M3 done, M2 started) - usage limit hit mid-M2
- Upstream DS4 unchanged at 924516a. CPU tests: geometry 1 expected author-file failure; test_ds4_moe 23/23 after
  fixing the TEST (per-layer formats + MXFP4's Q8_0 activation - the engine was right); tokenizer golden 3/3.
- M1 (f9d1dbdf, pushed): byte-sized --slots auto (slot_gib, real 7.19/8.38/9.38 MiB blobs) + margin 2.0 GiB from 64k ctx.
  GPU-checked: 36.99 GiB budget -> 5,217 experts, no OOM, 1.72 GiB VRAM free at the end @ ctx 262144.
- M3 (cc281c92, LOCAL, not pushed - adversarial verify workflow was still running): v0.1.41 serve/ + merged tokenizer;
  parity vs llama-tokenize b11429 = 0 diffs over 1.78M ids (needles 251,198 = stack 73's count). llama-tokenize
  needs --no-escape (unescapes \n by default). Docker default runtime = nvidia: CPU containers need NVIDIA_VISIBLE_DEVICES=none.
- Build with CUDA 13 (BUILD_DIR=build-ds4-cu13, nvidia/cuda:13.0.2): upstream warns CUDA 12.8 crashes long prompts on sm_120.
- M2 so far (bench/mab776-m2/: make_prompts.py, m2run.sh, capture_templates.sh, renders/ = 14 stack-73 renders for M4):
  probes (--pos-offset, cold, MEM 42g/arena 36): 130k 23.3 tok/s, 255k 23.7 (attention+router 14.7 -> 15.1 ms, flat);
  experts ~24-25 ms of which ~15-16 ms = file-tier reads (635 experts in neither VRAM nor the 36 GiB arena) -> next:
  full RAM (MEM = avail-9, arena ~39-40) and maybe --arena-adapt, with the CPU otherwise idle.
  ppl768 (BOS + p600 + p3000[:167]): llama.cpp 9.9624; DS4 decode-loop 9.8681 (-0.95 %), chunked w/o MMQ 9.8737 (-0.89 %).
  ⚠️ --chunk-mmq gives NaN ppl on our file (MMQ path, our types) -> run chunked prefill WITHOUT --chunk-mmq (fix later).
- NEXT: B2-B4 (8k / 127k code prompts, MMQ=0, full RAM) = the go/no-go number; then 250k code + needles x3; then STOP.

## Marc's decisions 2026-10-09 ~23:55 (before leaving the session unattended) - binding for the next steps
- GPU 0+1 may be used whenever the mode page is "off" (also at night), short test runs only (no LCB/SWE nights);
  every run checks first and refuses if anything holds the GPUs; never press the mode page, never touch stacks 67/72/73,
  restore nothing, keep >= 8 GiB host RAM free.
- M2 gate (corrected via the DeepSeek-V4-Flash session): like-for-like vs stack 73 WITHOUT drafter (28.0/25.0/22.3 decode,
  1011/843/693 prefill); hard stop only on correctness. M2 = GO if the 3 needles at 251k pass (ppl and decode already pass).
- Order after M2 (Marc, M7 first): M7a = GPU 1 (16 GB) as an extra expert tier to remove the disk reads (re-measure the M2
  table), then M7b DSpark drafter on GPU 1, then prefill attention; M4 (serve/deepseek.py) and M5 (tools/ds4/ds4_serve.cpp)
  on CPU in parallel. Still ask Marc before M8 (Portainer/mode page) and anything public beyond pushing mab776-main.

## Session end 2026-10-10 ~00:45 (usage limit) - where to pick up
- M2: needles 10 % + 50 % FOUND at 251k (server-mode prefill); 90 % and the 250k one-shot decode run were still running
  in the background chain (`bench/mab776-m2/work/chain-b7b6.sh`, log `bench/mab776-m2/out/chain-b7b6.log`; check
  `out/B7-needle-d90.*` with `detok.py ... --word willow-cobalt-696` and `out/B6-250k-code1-oneshot.log`), then finish
  `bench/mab776-m2/M2-REPORT.md` (verdict: GO if d90 FOUND, per Marc) and commit it with `out/*.log`.
- UNCOMMITTED, verified (1 of 2 verifiers done: 10.8M new oracle ids, 0 diffs): the Unicode-table tokenizer fix =
  tools/strata_tokenizer.py + tools/llama_unicode_classes.py + tools/gen_llama_unicode_classes.py +
  tools/ds4/test_tokenizer_unicode.py + tests/ds4/tokenizer_unicode_golden.json - commit the 5 files TOGETHER (import).
  Then the two M3-verify minors: from_dir must require token_type.json; qwen35 single-pattern fast path.
- M4 / M5 / M7a workflows were STOPPED at the design stage (designs partly done; M7a has 2 designs in its journal).
  Relaunch from the scripts under ~/.claude/projects/*/6a7c0663-*/workflows/scripts/ (m4-deepseek-module, m5-ds4-serve,
  m7a-gpu1-tier-design). M4 must also fix: literal-tag protection cost with 1,283 marks (range-class regex), the
  private-use collision, stop ids {1}, bos_token. M5: /slots SAVE/RESTORE not in the engine (gate in config).

- 00:43: needle 90 % FOUND ("willow-cobalt-696", prefill 256 tok/s) -> needles 3/3 at 251k -> **M2 = GO** (Marc's rule). The 250k one-shot decode run (B6) started 00:43; its log: bench/mab776-m2/out/B6-250k-code1-oneshot.log.

## M4 - DeepSeek-V4 in Strata's server (branch m4-deepseek, spec bench/mab776-m2/work/m4/DESIGN.md)
- `serve/deepseek.py` = the family: `DeepSeekTemplate` (the GGUF's chat template with bos_token, thinking on by default,
  effort none/low/medium/high/xhigh/max -> off/-/-/high/high/max, tools re-wrapped as type=function, tool results in
  call order, text parts joined with "\n", a trailing assistant continued) + `DeepSeekOutputParser` (reasoning_content,
  DSML tool calls whole once valid, malformed DSML = its text, a call inside the thinking only when the output ends
  with it). Selected by the chat template itself (llama.cpp's test); `"architecture": "deepseek4"` only asserts it.
- Server hooks read the family from the template (`getattr`), so Qwen takes the same path as before (checked: 53
  Qwen-visible outputs identical to mab776-main, stop ids still 248044/248046). Fixes for every family: stop ids never
  ordinary pieces (DS4: {1} from special_ids), literal tags marked per request (DS4 1,283 literals: encode_prompt
  683 -> 80 ms at 100k tokens, 1,660 -> 214 ms at 250k, same ids) and a private-use character of the request is never
  rewritten; `--slots` is not read as batch slots for DeepSeek (int("auto") ended the start).
- Config: `tools/ds4/ds4-server.json.example` (refused for DeepSeek: vision, parallel, slot_save_path, vram_elastic,
  expert_profile_save; ignored: effort_position end, tool_call_recovery). Args go to the engine unchanged.
- Tests: `serve/test_deepseek.py` (14 stack-73 captures byte-equal but c06b xhigh by design, parser at every split +
  fuzz, mock-engine calls/#1053/budget/forced tool_choice/Anthropic/Responses, literal tags vs v0.1.41, main() with a
  fake DS4 engine). Real-vocabulary tests: `STRATA_DS4_TOKENIZER=<pack tokenizer dir>`.
- By design vs stack 73: xhigh = high preamble; response_format adds Strata's directive (llama.cpp: grammar); #843
  drops completed empty assistant turns; quoted <think>/control text is text (#537); forced tool_choice = a written
  DSML opening; whole tool calls only (SSE keep-alive every 10 s while one is held).
- DeepSeek serving pieces adapted from Jackwwg83/Strata-DS (MIT) and Aleks-Che/Strata (MIT); tool-result ordering,
  prefill continuation and part joining from ggml-org/llama.cpp (MIT).

## Session 2026-10-10 day - M7a done, M4 merged, M5 running (upstream still 924516a0, nothing to merge)
- **M7a (8d24f746)** = GPU 1 static expert tier (`--gpu1 1`, 2,045 experts / 14.5 GiB, no arena copies) + complete
  ranking (the 80 never-profiled experts) + elastic GPU 0 cache (`--vram-grow 2.0` with `--vram-margin 9.0`: seeded
  at the prompt budget, grown from the arena after the prompt) + prestage halves sized to the busiest layer + a leaner
  decode host path. **NVMe during the prompt: 0.0 GB; decode file reads 0.** GPU-1 rows byte-identical to GPU 0's
  (DS4_CHECK_G1: 24,518/24,518 decode, 671/671 prompt). Run with m2run.sh `G1=1 GROW=2.0 MARGIN=9.0`.
  | depth | prefill tok/s (M2 one-shot / stack 73) | decode tok/s (M2 / stack 73 no draft / w/ drafter) |
  |---|---|---|
  | 8k | **683** (153 / 1011) | **44.4** (35.7 / 28.0 / 40.8) |
  | 127k | **495** (210 / 843) | **42.0** (33.9 / 25.0 / 39.0) |
  | 250k | **347** (191 / 693) | **40.6** (29.5 / 22.3 / ~36) |
  Prefill at depth is now the O(cap x n) prompt attention (67 % at 127k, 77 % at 250k), then GPU 0's resident experts
  on MMVQ. Report: bench/mab776-m2/M7a-REPORT.md.
- **M4 merged (5944ac4d)**, worktree removed. M5 (ds4_serve) still in its worktree strata-ds4-m5 (branch m5-serve).
- **M7b (drafter) needs a decision from Marc** - see the report: DS4's verify loop drafts with the single MTP head
  (DeepSeek-V4-Flash-MTP GGUF, ~5 GB, NOT on disk); stack 73's DSpark file on disk is a different, 3-block "dflash"
  drafter (block of 5, target layers 41-43) the engine does not implement. Decode already beats stack 73 WITH its
  drafter; the user-visible gap is prefill, so prefill attention / resident MMQ may deserve to go first.

## Session 2026-10-10 afternoon - prefill target met, M4/M5 merged, M7b in progress (usage limit)
- **Prefill beats stack 73 at every depth** (bench/mab776-m2/PREFILL-REPORT.md): 1018 / 970 / 752 tok/s at 8k / 127k /
  251k vs 1011 / 843 / 693, needle 251k FOUND, decode unchanged. 8k needs `STRATA_MMQ_FP4=1` (2 MXFP4 layers on FP4 MMQ,
  llama.cpp's arithmetic; ppl768 9.95 vs llama.cpp 9.96) - Marc to confirm; without it 968-973.
- **M5 merged** (f5cf8d79, ds4_serve, CPU gate 34/34). **M6 not started**: port --gpu1/--vram-grow into ds4_serve, CUDA
  build, resident chunk buffers, GPU gates (plan: bench/mab776-m2/work/m6/PLAN.md).
- **M7b (drafter)**: MTP head downloaded (philpax DeepSeek-V4-Flash-MTP-bf16.gguf, sha ok, models/deepseek-v4-draft/mtp/).
  Done: MTP window after chunked prefill (2f636ab5; 96 % draft acceptance at 8k code), verify passes with LRU (12432abb):
  **8k verify 52.5 tok/s vs plain 45.3 (+16 %)**. Next: MTP experts on GPU 1 (draft 4.7 ms, 3.4 of it CPU experts; the
  tier shrinks ~380 experts for 3.2 GB MXFP4) + Marc's sweep (peer msg 10-10): quant BF16/Q8_0/Q4_K/Q2_K x placement (a)
  head on GPU 1 / (b) experts in pinned RAM / (c) GPU 0, net tok/s vs 45/42/41 at 8k/127k/250k; then the 2-token dense
  pass (attention+router 18.6 vs 13.3 ms for one token - profile it with work/prof.sh).

## Session 2026-10-10 evening - M6 + M8 done, M7b MTP on GPU 1, llm-router for Open WebUI (upstream still 924516a0)
- **Strata 0.1.42 check** (released 10-10): engine speedups target top-10 routing on miss-heavy cards (TAIL_SKIP = skip
  low-rank missed experts in verify windows; CPU-measured PCIe share) - nothing to port: DS4 routes top-6 and after M7a
  misses are ~4 % of lookups. serve/ fixes worth taking later (rebase picture -> ask Marc): #984 thinking budget at
  max_tokens, #1620 cancel keeps the restore, #1624 reordered tool set keeps the prompt cache, #1601 cgroup-aware
  parking RAM, #1603 orphaned slots. Stack 67 (Qwen coder) could gain a few % from TAIL_SKIP: separate test.
- **M6 = the live GPU server (7403e309)**: ds4_serve engine 0.2 - M7a flags; the elastic cache shrinks before a prompt's
  chunks (Ds4MoeTier::shrink_cache + ExpertCache::evict_from, ~20 ms) and regrows after (~120 ms, LRU recency kept);
  pinned bounce buffers kept across prompts. Gates through serve/server.py: stream / tools / reasoning / tool round
  trip (cached 299/388) / client hang-up -> cancel / depth 1,227 / 977 / 752 prefill and 46.3 / 43.8 / 43.2 decode
  (plain) / needles 3/3 at 239.8k.
- **Conversation parking** (Marc via peer: A-B-A + agent loop tests): `--park-mib` - Ds4Dense::park/unpark = rings +
  snapshot slots + compressed/indexer rows below the position (~2.9 KB/token: 94 MiB @10k, 273 MiB @93k, ~0.73 GiB
  @262k), LRU under the budget, RAM guard reads the cgroup minus its reclaimable file cache (memory.current alone
  counts the model's page cache and refused every park). A-B-A: A2 reused 92,808 of 92,928, TTFT 1.41 s (vs 88 s
  re-read). Agent loop (bench: work/ tmp scripts agent_loop.py/aba.py): 14 turns 55.7k -> 89.3k, each turn read only
  what it appended. Caveat: an EMPTY assistant message (a turn cut by max_tokens inside its thinking) makes the GGUF
  template drop it and merge the two user messages -> the prompt diverges inside earlier text -> full re-read.
- **M7b in serve**: `--mtp FILE` = verify loop with exact speculative sampling (ds4_sampler.hpp verify_row; chi-square
  tested, 3 mutants killed); CPU gate T8Verify (oracle-driven accept/reject, bit-exact vs plain greedy). Found and fixed:
  2-token passes ran a DENSE masked flash-attn over every compressed row (verify LOST to plain at depth: 33 vs 44 tok/s
  @93k) -> each query gathers its own top-k (DS4_SPARSE_N, bit-identical on CPU). **MTP experts on GPU (d8d626cf)**:
  `--mtp-device N`; sweep (arena 41 - at 38 the tier's giveback fell to NVMe and faked a loss): CPU 4.63 ms/draft,
  51-53 / 46.6 tok/s; GPU 1 1.3-1.5 ms, 55.8-59.0 / 49.7; GPU 0 1.3-1.4 ms, 56.0-57.3 / 51.2 (8k / 127k). GPU 1
  deployed (keeps prefill). Quant sweep of the head's dense part skipped on purpose: it is ~330 MB bf16 per draft
  token (~0.25 ms of a 1.3 ms launch-bound pass) - Q8/Q4 could save <=0.2 ms and cost acceptance.
- **M8 (stack 74)**: native stack via the API (portainer/deepseek-strata: compose, deploy.sh, snapshot.sh, Dockerfile,
  ds4-server.json, vscode-copilot.txt); mode page `ds_variant` (llama.cpp | Strata, app.py backup
  app.py.bak-20261010-dsvariant); off -> DeepSeek (Strata) -> off x3 clean (ready 50-55 s). OpenHands host profile
  `DeepSeek` (a 4-action coding task in 36 s; seed-settings.sh updated - profile names cannot contain "/").
- **llm-router** (Marc 10-10: "DS4 in Open WebUI without breaking it"): portainer/llm-router (stdlib, read-only,
  stack `llm-router`, :8340). OWUI's only OpenAI connection (`http://llm-router:8000/v1`; backup of the old two
  connections: llm-router/owui-openai-config.bak-20261010.json). Lists every coder model (stopped ones "· not
  running"), routes by model id, stopped stack = 503 "start it on the mode page", resolves stack names per request.

## Session 2026-10-10 late - M7c chained drafts (K=2 live), 0731-aligned head A/B, M10 radix-select top-k
- **M7c (6218d874)** - Marc via the DeepSeek session: "did he try drafter at 2-3-4-5 tokens?".  ds4_serve drafts
  K tokens per verify pass: dr[0] from the trunk's state as before, dr[1..K-1] from the MTP block's OWN output (its hc
  streams, one position earlier) fed back as h (`Ds4Dense::mtp_chain_begin`); one pass of K + 1 tokens, rows decided
  in order with the same exact speculative sampling (first rejection -> its row's replacement; all K accepted -> row
  K's token free); the MTP block then reruns over the verified positions, rewriting the speculative rows the chain
  left.  Pass cap 4 -> 8 tokens (kNtMax + the tier's kMaxTok: the rings' slack, so in-place rewinds, is 8 now).
  Drafts' head = last column only, argmax on the device (`mtp_argmax`).  `--mtp-k K`; per request
  `strata_tune: {"mtp_k": K}` (0 = plain decoding - how the sweep timed a step on the same server).  CPU gate 44/44.
  Sweep (deployed config, MXFP4 drafter experts on GPU 1, 512 code tokens, greedy, mean of 2 runs; tmp ksweep.py:
  one prompt per depth, every K restores the same n-1 checkpoint):

  | depth | plain | K=1 | K=2 | K=3 | K=4 | K=5 |
  |---|---|---|---|---|---|---|
  | 8k   | 46.8 | 60.0 | 60.0 | 60.5 | 51.5 | 47.3 |
  | 127k | 44.3 | 53.1 | **56.3** | 53.9 | 47.0 | 44.8 |
  | 248k | 43.2 | 46.8 | **50.4** | 49.0 | - | - |

  Acceptance by draft position (conditional) ~74-78 / 51-55 / 36-41 / 19-30 / 12-13 %; tokens per pass 1.77 / 2.1 /
  2.3 / 2.2 / 2.3; each extra verify token +3.1-4.2 ms (step 21.4 / 22.6 / 23.2 ms) = ~15 % of a step, so K=2 (vLLM's
  k=4 rule does not carry over).  Per-pass d1 acceptance looks lower for K >= 2: a selection effect (bigger passes
  eat the easy runs, so more passes start at hard tokens) - judge by net tok/s.  **Deployed `--mtp-k 1` for now**:
  greedy (T=0) needle runs at 240k looped more with K=2 (11/53 vs 1/45 by 18:00) although K=2's math is clean (below);
  Marc 10-10: decide on the SAMPLED loop rate (what clients use) - trial running, K=2 goes default if it matches.
- **0731 alignment**: philpax's bf16 MTP file = the ORIGINAL V4-Flash head (avlp12's sidecar differs: linears cos
  0.995 / rel 9 % - its base was a 4-bit pack - and hc_head.fn cos 0.89).  Applied r6c_step5000 with
  tools/ds4/mtp_sidecar.py -> `models/deepseek-v4-draft/mtp/DeepSeek-V4-Flash-MTP-bf16-aligned0731-r6c.gguf` (kept;
  sidecars in `models/deepseek-v4-draft/sidecar-0731/`).  Chained acceptance up (d2 57-58 vs 48-55 %, d3 41-45 vs 36 %),
  d1 a bit down (74-76 vs 77-78 %); net decode a wash: 127k K=2 56.1 vs 56.3, K=1 52.1 vs 53.1; 8k K=2 61.2 vs 60.0.
  Kept the original; re-test after the RAM upgrade.
- **M10 radix-select top-k (70f22b3c)**, tools/ds4/patches/ggml-topk-radix-select.patch: ggml-cuda's TOP_K for rows
  >= 2048 columns = one launch, one block per row (12 + 10 + 10-bit histogram passes find the k-th key, early exit,
  one gather pass; ties across the boundary take the lowest indices = a stable sort's set; `GGML_TOPK_RADIX=0` = stock).
  The stock path CUB-sorted every whole row: 0.34-0.36 ms per layer for a verify pass's 2-8 rows at 65,536 compressed
  rows (21 layers per pass!) and 13.5 ms per layer for a 2048-token chunk.  Standalone (exact sets vs a stable-sort
  reference incl. ties / -inf tails / all-equal): 1.30 vs 13.5 ms (2048 x 65536), 0.07 vs 0.35 ms (1-8 rows).
  Prompt chunks use ggml_top_k now (`DS4_PF_ARGSORT=1` = the old argsort).  Same binary, same prompts, stock -> new:

  | depth | prefill | plain | K=1 | K=2 |
  |---|---|---|---|---|
  | 127k | 1,212 -> **1,256** | 44.5 -> 44.8 | 51.2 -> 56.0 | 55.9 -> **63.7** |
  | 250k | 1,053 -> **1,133** | 43.2 -> 43.4 | 45.8 -> 55.7 | 51.8 -> **59.9** |

  The verify pass is now nearly flat in depth (K=1 30.0 / 30.2 ms at 127k / 250k, was 33.6 / 36.9).
- **K=2 numerics, verified** (diagnostic switches, ds4_serve/ds4_moe): `DS4_SERVE_TEST_FORCE=1` + the oracle =
  teacher forcing (every run emits the oracle's tokens, so plain / K=1 / K=2 logits rows compare row for row - no
  trajectory divergence); `DS4_MULTI_SERIAL=1` = the GPU tier runs a verify pass token by token; `DS4_CHECK_MULTI=<l>`
  = layer l's GPU 0 multi-token expert rows recomputed alone and byte-compared.  127k, 384 forced tokens, KL vs a
  plain reference: plain 0.0019-0.0021, K=1 0.0018-0.0021, K=2 0.0018 (no slot pattern); with every 2nd / 3rd draft
  corrupted (rejections): plain 0.0026-0.0031, K=2 0.0025-0.0029; serial tier: same.  => no K=2 state or kernel bug;
  the greedy loop gap is a trajectory effect (temperature 0 at near-ties).
- **Determinism**: the engine is NOT bit-deterministic run to run - two plain decodes from the same checkpoint on the
  same server already differ (the expert tier's placement moves between requests; CPU and GPU expert numerics
  differ).  So outputs cannot be compared by hash across builds; correctness rests on the CPU gate, the kernels'
  standalone exact tests, and needles.
