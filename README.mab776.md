# Strata-DS4 — mab776 fork notes (read this first)

Fork of https://github.com/Indras-Mirror/Strata-DS4 (one developer, no releases, very active) at
https://github.com/mab776/Strata-DS4. Goal: run DeepSeek-V4-Flash-0731 faster than our llama.cpp stack (stack 73
`deepseek-v4`, container `ds4-lcpp`, :8330) using Strata's VRAM hot-expert cache, on OUR unsloth files.

## STATUS 2026-10-13 - Portainer stack 74 `deepseek-strata` (read this, then the session notes)
- **2026-10-17: stack 74 runs UD-Q3_K_M** (Marc; UD-IQ2_M DELETED, stack 73 llama.cpp has no model now). Config portainer/deepseek-strata/ds4-server.json (-m /q3km/..., --arena-gib 76 -> 72.6 used, file tier 0), compose mem_limit 96g + /q3km mount, snapshot 2ee91ae8 (fixes da544a75 + 41128c2c). LCB-13 12/13, 15.6k tokens / 5.6 min per solved (IQ2_M 9/13, 22.5k / 5.7); live check: 8k decode K=2 57 tok/s, prefill 913 tok/s, tools 6/6, ~77 GiB. The IQ2_M numbers below are history.
- **M14 (10-12 night, session notes at the end)**: upstream Strata-DSV4.1 (the DS4 author's renamed repo, remote `dsv41`)
  reviewed; ported: the stale-graph unbind fix (verbatim), the warp top-k router (ggml patch, GGML_CUDA_RANK_ARGSORT=0 =
  old sort, K=2 pass -0.9 % @8k), fused finish+predict (DS4_FUSE_PREDICT=1, opt-in: neutral); deploy config `--pf-b 0`
  (prefetch hints off: K=2 pass -2.1 % @8k / -2.9 % @127k on this box; rollback 0.7).  Not merged (22 files conflict).
- **M13 (10-12 afternoon, session notes at the end)**: upstream 0.1.42 / 0.1.43 / 0.1.43.1 ideas ported (no merge);
  deployed in run/ (88bca4ea, run.prev = M12 eb162164): verify pass K=2 -4.3 / -3.6 / -5.8 % at 8k / 127k / 250k (24.9
  / 25.9 / 26.7 ms), K=1 -5.6 / -3.4 / -5.1 %, prefill unchanged; the NaN guard; serve fixes #984 #1624 #1869/#1870
  #2026 (+ quoted DSML stays text). New env switches (=0 = old path): DS4_HC_MIX, DS4_HC_PRENORM, DS4_LEAN,
  DS4_ROUTE_PAR, DS4_PF_PINNED, DS4_NAN_GUARD; DS4_SH_FOLD=1 opt-in (a loss).
- **Runs**: container `strata-ds4`, image `strata-ds4:cu13` (portainer/deepseek-strata/Dockerfile: Ubuntu 24.04 +
  Python + libnccl; nothing of the engine baked in), serve/server.py + `ds4_serve` from the DEPLOY SNAPSHOT
  `models/strata-ds4/run` (bind-mounted read-only at /app; the previous one is `run.prev`). A DROP-IN for llama.cpp
  stack 73: host :8330, network alias `ds4-lcpp`, model id `deepseek-v4-flash` (+ alias `deepseek-v4-flash-strata`).
  Mode page :8500 DeepSeek tile: **llama.cpp | Strata** picks the engine (they share the port: never both).
- **Update the engine**: build here (mab776-build.sh, BUILD_DIR=build-ds4-cu13, CUDA 13 image; it applies
  tools/ds4/patches/*.patch to ggml) -> `portainer/deepseek-strata/snapshot.sh` (copies ds4_serve, serve/, the
  tokenizer files, the profile and ds4-server.json into run/) -> restart the container (mode page Restart, or
  `docker restart strata-ds4`). Rollback: stop it, swap run/ and run.prev/, start it. Config-only edit: change
  portainer/deepseek-strata/ds4-server.json and copy it to run/ (snapshot.sh would also deploy whatever is built).
- **Engine config** (ds4-server.json): `--gpu1 1` (GPU 1 tier) `--vram-margin 9 --vram-grow 2` (elastic GPU 0 cache,
  shrunk per prompt) `--prefill-chunk 2048 --chunk-mmq --chunk-prestage` + env `STRATA_MMQ_FP4=1` (Marc: keep it)
  `--mtp ... --mtp-device 1 --mtp-k 2` (speculative decoding with two chained drafts for sampled requests; greedy
  requests fall back to one, `--mtp-k-greedy` 1; a request's `strata_tune.mtp_k` wins, 0 = none) `--park-mib 4096`
  (conversation parking) `--arena-gib 41` + env `GGML_LI_I8=1` (the lightning indexer on int8 tensor cores, 10-10
  night; drop the line for the fp16 kernel, no rebuild) + `--lookup 7` (prompt-lookup drafts, sampled requests); the
  engine's own defaults since 10-10/11 night: emit-only compressors + batched HCA (prefill), async decode hand-offs +
  the fused hc gates op + the verify passes' expert mix on the card + the predict/tier overlap (decode) - env
  DS4_PF_EMIT=0 / DS4_PF_HCAB=0 / DS4_ASYNC=0 / DS4_HC_GATES=0 / DS4_DEV_SUM=0 / DS4_OVL=0 / DS4_PF_HASH=0 /
  DS4_F16_ACC32=0 = the old paths - and env `GGML_MMVF_F16_MAX=8` (verify passes keep ggml's F16 vector kernel; drop the
  line for the tensor-core tiles) + `DS4_DEV_ROUTE=1` (M12b: the hits routed from the card; drop it to revert).
  ~42 GiB container RAM, mem_limit 50g.
- **Measured through the server (prefill 10-10 late: int8 indexer + emit-only compressors + batched HCA; decode 10-11)**: prefill
  1,100-1,250 / **1,530** / **1,428** tok/s at 8k / 127k / 250k (morning: 1,297 / 1,192; stack 73: 1011 / 843 / 693);
  decode after M12 (10-11 day, greedy K=2, fresh code, mean of 6) **76.0 / 73.0 / 71.3** (verify pass 25.8 / 26.7 /
  28.0 ms; after M11 73.0 / 71.4 / 67.7; before M11 65.0 / 62.5 / 60.3 at 31.3 / 32.1 / 32.8 ms), K=1 72 / 70 / 68, copy-heavy edits ~100-110
  (prompt lookup), plain 47 / 45 / 43 (stack 73 with its drafter 40.8 / 39.0 / ~36); needles 3/3 at 240k; parked
  chat back in 0.6-1.3 s; agent turns read only what they appended.
- **Clients**: OpenHands host profile `DeepSeek`; VS Code block portainer/deepseek-strata/vscode-copilot.txt; Open
  WebUI via **llm-router** (stack `llm-router`, :8340, portainer/llm-router) - OWUI's ONLY OpenAI connection now.
- **Coding (M9, 10-11/12, session notes at the end)**: LCB 39 at 32k T 1.0 = Strata 30/39 vs stack 73 34/39 pass@1,
  noise (the 6 disagreements resampled: 10/18 vs 12/18); SWE Verified-mini Strata **35/50**; paired first 10 = 8/10 on
  both, Strata 7.4 vs 9.7 min and 26.3k vs 17.4k output tokens per solved issue.
- **Open**: M10 next targets
  (250k profile of the fp16-indexer build, bench/mab776-m2/work/prof/p250k-m10c; done since: the indexer on int8,
  emit-only compressors, batched HCA blocks): expert MMQ ~15 % + the Strata MMQ plumbing ~9 % (copy16 / scatter /
  quantize), dense Q8_0 7 %; M11 DONE (10-10/11 night, session notes below): shipped lookup drafts, async hand-offs,
  fused gates, the expert mix on the card, the F16 vector kernel for verify passes, the predict/tier overlap (-13 % a
  verify pass together); not shipped: early PCIe DMAs (opt-in), tree drafts (a loss); GPU-side expert routing = a
  design (the next decode step); to A/B: the hash-layer predict fix, GGML_PREC_F32 on the F16 decode matmuls; serve/ fixes of Strata 0.1.42 worth taking (#984, #1620,
  #1624, #1601) = a rebase-picture change, ask Marc; after the RAM upgrade: Q3_K_M, the MTP Q2_K expert sweep again,
  the 0731-aligned head again.

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
  Sweep (deployed config, MXFP4 drafter experts on GPU 1, 512 code tokens, greedy, mean of 2 runs; bench/mab776-m2/work/m7c/ksweep.py:
  one prompt per depth, every K restores the same n-1 checkpoint):

  | depth | plain | K=1 | K=2 | K=3 | K=4 | K=5 |
  |---|---|---|---|---|---|---|
  | 8k   | 46.8 | 60.0 | 60.0 | 60.5 | 51.5 | 47.3 |
  | 127k | 44.3 | 53.1 | **56.3** | 53.9 | 47.0 | 44.8 |
  | 248k | 43.2 | 46.8 | **50.4** | 49.0 | - | - |

  Acceptance by draft position (conditional) ~74-78 / 51-55 / 36-41 / 19-30 / 12-13 %; tokens per pass 1.77 / 2.1 /
  2.3 / 2.2 / 2.3; each extra verify token +3.1-4.2 ms (step 21.4 / 22.6 / 23.2 ms) = ~15 % of a step, so K=2 (vLLM's
  k=4 rule does not carry over).  Per-pass d1 acceptance looks lower for K >= 2: a selection effect (bigger passes
  eat the easy runs, so more passes start at hard tokens) - judge by net tok/s.  **K=2 deployed for sampled requests,
  greedy ones fall back to K=1** (2b733950, Marc 10-10): the greedy (T=0) 240k needle looped more with K=2 (all trials
  11/106 vs 3/95; the last clean 60+60 run 4 vs 2) although K=2's math is clean (below); sampled - what clients use
  (OpenHands leaves temperature to the server's 1.0, the VS Code block sends null) - 0/60 loops for both K at T=1.0
  and T=0.6 (needles 58/60 vs 60/60).  `--mtp-k-greedy` (default 1) caps greedy requests that set no mtp_k.
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
- **M10 lean prefill masks (dfbb071c)**: a chunk's CSA layers no longer materialise the [cap, n] visibility mask
  (repeat / add / step / log at F32, then F16 casts: ~6.6 GB per layer-chunk at 250k).  The fused indexer takes an
  F32 [1, n] count mask (ggml patch ggml-lightning-indexer-count-mask.patch + li_mma's epilogue: kv < count -> +0 else
  -inf) and, once every query sees >= 512 blocks, the attention mask is the F16 -inf fill with zeros at the top-k picks.
  Same values: ppl768 9.9034 both.  Through the server: prefill 1,256 -> **1,297** at 127k, 1,133 -> **1,192** at 250k.
  `DS4_PF_LEAN_MASK=0` = the dense masks.
- **One-call HCA chunk attention (`DS4_PF_HCA1=1`, OFF by default)**: the CSA chunks' layout for HCA layers (one
  flash-attn per layer-chunk instead of 32 blocks re-concatenating the compressed K): +~3 % prefill at 250k, but ppl768
  (real text, one deterministic chunk) 9.9034 -> 9.9512 and the 8k filler NLL +0.0005 every run - not shipped until
  the cause is known.  For scale: chunks of 64 instead of 2048 move ppl768 to 10.0335 (code-path numerics, ~1 %), and
  the chunk path itself varies run to run (8k NLL 0.03281-0.03313).
- **Profile of the deployed build** (250k prefill window, p250k-m10c): matmuls 28.8 % (expert MMQ ~15 %, dense Q8_0
  7.3 %), li_mma 20.3 %, flash-attn 15.0 % (CSA sparse 7.2 %, HCA blocked 6.0 %), elementwise 14.3 %, copies 12.7 %
  (HCA's per-block concats 4.6 %, transposes 2.5 %), expert-tier moves 7.4 %, radix top-k 1.5 %.
- **M6 regression suite re-run on the deployed build (10-10 20:07, Marc's "usual tests"; kit in the untracked
  bench/mab776-m2/work/m6reg)**: parking A-B-A: A2 reused 92,808 of 92,955, TTFT 1.27 s (M6 1.41 s), fresh 92.8k read
  in 72 s (M6 88 s); A-B-C-A: A3 reused 92,954 in 1.22 s, B2/C2 re-read (B1/C1 hit max_tokens inside their thinking ->
  empty content -> the template merges the user turns: the M6 caveat); parked 277 MiB at 93k / 98 MiB at 10k, unpark
  19 ms; 14-turn agent loop 55.7k -> 87.7k, every turn read only its append (108-9,294 tokens), TTFT 1.1-1.2 s (7.3-7.8
  s for 9.3k-token file reads), decode 55-65 tok/s; OpenHands "DeepSeek": fib + unittest in 5 actions, 36 s; mode page
  off -> DeepSeek (Strata) ready in 71 s.  All pass.
- **M10b int8 lightning indexer (2f56f400, deployed 10-10 21:08 via env `GGML_LI_I8=1`)**: li_mma::kernel_i8b in
  ggml-lightning-indexer-mma.patch.  The keys are Q8_0 already and an m16n8k32 int8 MMA is 32 dims deep (one Q8_0
  block).  With ONE scale per 128-dim vector on both sides (K's four blocks re-quantized to the key's own scale at tile
  load, Q per head row in registers), the four MMAs accumulate the whole dot product in int32 inside the tensor core and
  the scales apply once per tile: no fp16 dequant, half the shared memory, 256-key tiles.  Standalone (li_test.cu, 1024
  queries x 50k keys, random data = a crowded top-512 boundary; fp16 li_mma 6.06 ms, exact):

  | variant | ms | top-512 overlap vs exact |
  |---|---:|---:|
  | int8, per-32-dim scales (scale epilogue per MMA) | 5.46 | 99.28 % |
  | FP8 e4m3 | 4.34 | 95.5 % (rejected) |
  | **int8, one scale per 128-dim vector (shipped)** | **4.06** | 98.75 % |

  Gate (set by the DeepSeek session, Marc's call; one-shot ds4_generate, deployed config, bench/mab776-m2/out/m10j.log):
  teacher-forced NLL over 128k tokens of real text (the needle haystack's docs + code) fp16 1.64916 / 1.64913, int8
  1.64891 / 1.64871 (0.02 % lower, Marc: OK); needles 3/3 at 251k (d10 / d50 / d90); ppl768 9.9034 both (768 tokens
  never reach the 512-block top-k); 8k filler NLL 0.03323 (fp16 band 0.03281-0.03313, +0.0001 nats); prefill 251k
  1,228-1,235 vs 1,173 tok/s, 128k 731-747 vs 714-723 (ppl mode).  Post-deploy through the server: smoke (plain /
  stream / tools / tool result) OK; prefill **1,337** at 127k (was 1,297) and **1,248** at 250k (was 1,192); decode
  unchanged (K=2 61.7 / 61.7, K=1 54.1 / 55.5); A-B-A: A2 reused 92,808 of 92,850, TTFT 1.21 s.  Rollback = drop
  GGML_LI_I8 from portainer/deepseek-strata/ds4-server.json, copy it to run/, restart.
- **M10d emit-only compressors (8fe661c9, `DS4_PF_EMIT`, default on, EXACT)**: a prompt chunk's compressors made one
  candidate compressed row per token and parked all but each block's last in the dummy slot `comp_max` - an HCA
  layer-chunk gathered [DH, 128, 2048] twice (~512 MB each), transposed, softmaxed and reduced it to keep 16 rows (the
  250k profile's 1,294 transposes, the 128-wide softmaxes, ~7 % of the GPU, per token = at every depth).
  prefill_begin sends only the emitting rows (each block's last token in the chunk + the chunk's last token: its
  partial block) and build_attn sizes the compressors by that count.  CPU gate 45/45 bit-exact incl. chunked prompts
  (mutation: +1 on the emitted rows' RoPE position fails 88 parity subtests); ppl768 9.9034 unchanged on the GPU.
  250k one-shot prefill 1,236 -> **1,348** tok/s (+9 %).
- **M10c batched HCA blocks (8fe661c9, `DS4_PF_HCAB`, default on)**: an HCA chunk's 32 query blocks (each its own K =
  [SWA + RB - 1 window rows | the compressed rows | pad] + mask: ~160 small ops a layer-chunk) are the sequences of ONE
  flash-attn call - windows = an overlapping strided view of k_raw (made one window deep, ne/nb patched after: ggml
  sizes views as if contiguous), the compressed rows (+ pad) repeated per block.  Same keys, masks and order, but not
  bit-identical: ggml-cuda's MMA kernel accumulates V*P in half2 (`T_C_VKQ = tile<16,8,half2>` on Turing+); the
  blocked launches were small enough to be split over CTAs (stream-k, partials combined in F32), the batched one runs
  whole tiles, so a block's whole key range accumulates in fp16 - what llama.cpp's own HCA call does.
  `DS4_CHECK_HCAB=1` (both paths in one run, the blocked output kept, per query row): rel diff 4e-4 at 8k, median
  1.2e-3 / max 3.2e-3 at 250k, uniform over blocks and layers.  ppl768 9.9034 -> 10.0234 - and that is why ppl768 is
  no gate for numerics changes: query blocks of 32 / 128 / one call (identical semantics) give 9.9052 / 9.8735 / 9.9435
  (HCA1: 9.9512).  The gate (Marc's, set by the DeepSeek session): 128k real-text teacher-forced NLL 1.64804 / 1.64974
  / 1.64854 (mean 1.64877; wider spread than the blocked pair 1.64891 / 1.64871, mean 1.64881); 251k (the needle
  haystack, 251k positions) **1.13771 vs blocked 1.13807**; needles 3/3 at 251k (d10 / d50 answers token-identical to
  the int8 runs).  250k one-shot prefill with both: **1,427** tok/s (+15.5 % over the deployed 1,236).  Deployed
  10-10 23:04 (snapshot 2bed102c); post-deploy through the server: smoke OK; prefill **1,530** at 127k (was 1,337)
  and **1,428** at 250k (was 1,248); decode unchanged (K=2 59.7 / 61.2, K=1 53.2 / 56.5); A-B-A: A2 reused 92,808 of
  92,957, TTFT 1.27 s, a fresh 92.8k read 61.6 s (was 70.8).  Rollback: env DS4_PF_EMIT=0 / DS4_PF_HCAB=0.

## Session 2026-10-10 night - M11 decode (Marc's plan via the DeepSeek session: solo, one agent max)
- **Where a step goes** (read-only survey, one agent): a plain step is driven from the host - per layer a predict
  graph (expert prefetch hint), the attention+router graph (165-243 nodes), the expert tier (host placement, GPU 0 hits
  + PCIe on one stream, CPU pool, GPU 1, the weighted sum on the host in double) and the finish graph: 131 graph
  computes and ~300 blocking waits per token, nothing overlapping across layers.  M7a's split: attention+router 13-15
  ms, experts ~6 ms - against a dense-weight read floor of ~5 ms.  ~8,600 kernels a token at ~1.5-2 us each = the
  attention phase: the node count, not the bytes, is the cost.
- **Prompt-lookup drafts (2bed102c + ae329f7f, `--lookup 7`, deployed)**: when the context's last 3 tokens occurred
  earlier (prompt + output), the tokens that followed there are the next pass's drafts instead of the MTP chain.
  Exact for any deterministic draft (verify_row: accept with p(draft), else sample p without it); the MTP block still
  runs over the verified positions.  Policy after 3 A/B rounds (bench/mab776-m2/work/m11/lookab.py, greedy, mtp_k=2
  forced, lookup 0/7 alternating from the same restored checkpoint):
  - the occurrence must match G + 1 tokens back; a match of M drafts M - G + 1 (a copied file: 7);
  - `--lookup-agree 1`: copy only when its first token is the MTP head's first guess (else the chain continues from
    it, no pass lost) - this turned fresh code from -8 % / -3.5 % into +4 %;
  - a lookup pass with less than half right sits the lookup out 1, 2, 4 .. 64 passes;
  - greedy requests without a key: `--lookup-greedy` (default 0, like --mtp-k-greedy); sampled: --lookup; the
    request's `strata_tune.lookup` wins.

  | prompt (decode tok/s) | MTP K=2 | + lookup 7 | |
  |---|---:|---:|---|
  | rewrite a shell script with a rename | 74.6 | **105.7** | +42 %, 5 tokens a pass |
  | edit a JSON config (round 2) | 74.5 | **108.3** | +45 % |
  | fresh code (LRU cache + tests) | 66.3 | 69.0 | +4 % |
  | fresh prose | 58.4 | 60.1 | neutral |
- **Decode profile** (ds4_generate `--pos-offset N` = depth-N graphs over empty caches, no prefill; m11a/m11b windows):

  | depth | tok/s | predict+prefetch | attention+router | experts | finish | head (ms/token) |
  |---|---:|---:|---:|---:|---:|---:|
  | 8k | 43.2 | 1.42 | 13.40 | 6.69 | 1.07 | 0.50 |
  | 127k | 43.0 | 1.40 | 14.30 | 5.84 | 1.05 | 0.50 |
  | 250k | 41.6 | 1.34 | 15.50 | 5.62 | 1.03 | 0.49 |

  Node census (DS4_DBG_GRAPH): 122 / 150 / 176-179 nodes per attention graph (SWA / HCA / CSA layers), 26-36 of them
  free reshapes.  nsys @127k (5 s, node-level graph trace): ~5,100 kernels a token averaging 3.8 us; dense Q8_0
  matvecs ~6 ms of it (the real weight reads), then hundreds of tiny ones a token (522 adds, 351 rms_norms, 345
  q8_1 quantizes, 277 scales, 224 muls, 213 sigmoids); 131 cudaGraphLaunch a token at ~45 us of CPU each, 307
  blocking syncs; GPU 0 busy only 71 % of the window.  One HCA layer = 128 ops in 608 us: host gaps ~45 us (router
  readback -> the tier's first op), ~14 us (inside the tier's launch), ~34 us (tier -> finish), ~17 us (finish ->
  next predict); the predict graph recomputes the next attention's whole hyper-connection prelude (~25 us); the
  router's top-6 of 256 is a full argsort (7.7 us).  Prefetch off (--pf-b 0) broke even in the probe, but there its
  predictions are 17 % useful (65 % after a real prompt) - not a basis to drop it.
- **Async decode hand-offs (633419b4, `DS4_ASYNC`, default on, deployed)**: predict / attention+router launch async
  and their readback (an async copy on the backend stream) is the one wait; finish uploads the routed sums from a
  pinned buffer and launches without waiting, except after a pass's last trunk layer and the MTP block (whatever
  follows a pass - prefill_state_to_decode writes x_state, as the MTP block's finish does - finds the stream idle).
  Adversarial review (one agent): no live race; its two latent points closed (alloc_graph drains before a real
  allocation - the shared decode arena may grow and free its old buffer; ~Impl drains an owned backend).  A first
  version synced a backend ds4_generate had already freed: exit code 139 at exit, fixed (own_backend only).
  Plain decode probe @127k: 43.0 -> 44.25 tok/s.  Through the server (2 rounds, same session, verify pass ms):
  127k K=1 28.05 -> 27.45, K=2 31.4 -> 30.65; 250k K=1 29.2 -> 28.8, K=2 33.05 -> 32.6 = **-1.4 to -2.4 %**.
- **Fused hc gates (6de3a4c8, `DS4_HC_GATES`)**: ggml op GGML_OP_DSV4_HC_GATES (patch ggml-zz-dsv4-hc-gates.patch)
  replaces each hc_pre's 8 gate kernels with one, rounded as those ops (same bits: ppl768 identical on / off).
  -22 decode nodes a layer; plain decode probe +1.2 %; in the server inside one round's noise.
- **Verify passes mix their experts on the card (d9ea28b6, `DS4_DEV_SUM`)**: production decode is verify passes,
  and the tier ended each layer of one by downloading all NE x H expert rows, summing per token in double on the host
  and uploading the sums.  Now the CPU pool's and GPU 1's rows go up into their entry rows, and
  weighted_rows_sum_f64 (double, k order, no FMA, round-to-nearest - the host loop's arithmetic) writes the routed-sum
  tensor directly.  DS4_CHECK_DEVSUM: 77,825 of 77,825 token rows identical.  Server, 2 rounds, verify pass ms off ->
  on: 127k K=1 26.75 -> 26.45, K=2 30.2 -> 29.1 (-3.6 %); 250k K=1 28.25 -> 27.4, K=2 31.85 -> 30.5 (-4.2 %).
- **Lesson**: production decode is VERIFY passes (n = 3 at K=2, up to 8 with lookup) - a one-token gain is worth
  little unless the multi-token path gets it too (a predict -> attention hc_pre reuse was written and reverted for
  that reason: predict only covers token 0).
- **Early PCIe DMAs (`DS4_PCIE_EARLY=1`, opt-in)**: the PCIe share's DMAs (+ LRU admissions) first on their own copy
  stream, under the activation upload / hits kernel / LRU copies (gpu_run and gpu_run_n).  No gain @127k (40.58 /
  40.40 -> 40.61 / 40.63): they then share VRAM bandwidth with the hits kernel.  DS4_CHECK_GPU (1,082 PCIe experts),
  DS4_CHECK_LRU and DS4_CHECK_MULTI (608/608 rows identical) pass with it on.  Retry after the RAM upgrade.
- **Tree drafts (item 3) - not built**: with each extra verify token costing ~15 % of a step, verifying MTP's top-2
  for draft 1 (+ a child each) lifts tokens/pass ~2.18 -> ~2.36 (+8 %) for a 5-token pass (+~22 % pass cost) - a
  loss here.  Lookup drafts are the better use of a longer pass.
- **GPU-side expert routing - DESIGN only (Marc: design tonight unless 1-4 finish early and it is clearly safe)**.
  Today every layer reads the router's ids back (o_span), places the 6 experts on the host, launches the grouped
  kernels with host-built metadata, downloads the parts, sums them on the host in double and uploads the sum: the
  trace shows ~45 + 14 + 34 us of host gaps a layer (~4 ms a token at 127k) besides the GPU work.
  1. A device copy of GPU 0's residency (`slot_of[layer][expert]`, int32, -1 = absent), updated by the host with the
     LRU admissions/evictions it already decides (an H2D of the changed entries before the next use, same stream).
  2. A route kernel at the end of the attention graph: top-6 ids/weights (already on the device) -> the GPU 0 hits'
     grouped metadata (the kernels already read G1Meta from device memory: only the builder moves) + a miss mask.
  3. The hits' gate/up/down kernels and a device weighted sum of their parts (weighted_rows_sum, as prompt chunks do)
     in the same graph -> `routed_sum` partial on the device.
  4. The misses (GPU 1 tier, CPU pool, PCIe) still need the host: an ASYNC readback of the ids + x (24 B + 16 KB)
     queued after the route kernel; the host waits for that alone, dispatches the misses, and their parts are added
     to `routed_sum` on the device (CPU parts H2D, GPU 1 parts peer copy); finish waits on an event.
  5. Layers whose 6 experts are all on GPU 0 run attention -> experts -> finish -> next layer with no host wait at
     all.  Per-expert GPU 0 hit rate ~75 % -> all 6 on GPU 0 in ~18 % of layers; with GPU 1 dispatched from the
     device too (a GPU 1 stream waiting on a GPU 0 event, reading the ids over P2P) ~69 %.
  Expected: ~90 us of host gaps saved on those layers -> ~2.7 ms of a ~22 ms step (~12 %) at 69 %, ~0.7 ms at 18 %.
  Costs/risks: the device sum is float, not the host's double (not bit-identical to today; the CPU gate's
  decode-vs-chunk parity is CPU-only and unaffected), cross-GPU dispatch is new, LRU bookkeeping must stay exact.
  Order: (a) device metadata for GPU 0 hits + device sum + async miss readback, (b) GPU 1 from the device, (c) a
  CUDA graph per layer.
  Two lessons from the windows: a second, overlapping GPU window (an `&` waiter that `ps` didn't show) killed the
  other's runs through the shared container name - gpuwin now holds a flock; and SIGTERM did not stop a window
  script - windows stop through a STOP file checked between runs.

- **F16 matvec for verify passes (a360f164, env `GGML_MMVF_F16_MAX=8`, ggml patch ggml-zz-mmvf-f16-batch.patch,
  deployed 10-11)**: ggml-cuda multiplies an F16 weight by 2-8 columns with mul_mat_f (tensor-core tiles).  A verify
  pass's F16 weights are the small ones the loader makes of our F32/BF16 tensors (the geometry wants F16): the
  hyper-connection mixes (16,384 x 24), the compressors', the indexer's - where the verify trace had
  mul_mat_f<half2> at 11.9 us a call vs 6.8 us for the one-token vector kernel, ~200 calls a pass.  With the env set,
  small F16 weights keep the vector kernel up to 8 columns: per column the one-token decode's own arithmetic (ggml's
  default half2 accumulation for F16 weights; the router is set to F32 and stays so).  Server, 2 rounds, verify pass
  ms off -> on: 127k K=1 26.1 -> 24.9 (-4.6 %), K=2 29.05 -> 28.0 (-3.6 %); 250k K=1 27.45 -> 26.05 (-5.1 %), K=2
  30.5 -> 29.5 (-3.3 %).  Teacher-forced (m7c/forced_test.py at 127k, KL of each logits row vs a plain run): verify
  rows 0.0018-0.0020 off, 0.0020-0.0025 on, plain vs plain 0.0017-0.0020 in both = inside the run-to-run noise
  (M7c: 0.0018-0.0026).  Needles 3/3 at 240k, smoke OK.  Rollback: drop the env line.
  Follow-up worth an A/B: every one-token step accumulates those F16 matmuls in half precision too (ggml's default),
  where llama.cpp runs them from the F32 tensors in F32 - GGML_PREC_F32 on them in the decode graphs should cost
  ~nothing (memory-bound) and might be measurably cleaner.
- **Found, not fixed: predict() never hints the 3 hash-routed layers** - build_predict casts tid2eid's I32 ids to F32
  and predict() reads the floats back as int32 (7.0f -> 1088421888), which prefetch() drops as out of range (only
  expert 0 = 0.0f passes).  One line to fix (read floats), but it moves expert placement on 3 of 43 layers (CPU/GPU
  rows round differently): an A/B of its own.  Marked in predict_pick.
- **A-B-A artifact (not a regression)**: with max_tokens 300 a sampled A1 can spend all of it reasoning; aba.py then
  sends an empty assistant turn, the template collapses it and A2's new question lands INSIDE A's user turn - before
  the turn checkpoint, so nothing is reused (the tell: A2's prompt only +10 tokens, B2 +4).  Gates now use aba2.py
  (max_tokens 1500).
- **DS4_OVL - the last two host waits of a verify layer (M11b, default on)**: after DS4_ASYNC a layer still had the
  card idle twice while the host worked: (1) predict's readback -> the host picked the prefetch -> only then launched
  the attention graph; (2) the expert tier's cudaStreamSynchronize -> only then was finish launched.  Now
  `predict_launch` queues predict + its readback into pinned memory + an event, the attention is queued behind it, and
  attn_router_n's `queued` hook runs `predict_collect` (that event alone) + the prefetch DMAs under the attention; and
  the tier, when it mixed on the card (DS4_DEV_SUM), records `decode_sum_event()` after its last op instead of
  syncing, finish_layer_n waits on it on the card, and the host goes on queuing finish + the next predict + attention.
  Same kernels, same stream order = exact.  What makes it safe (the tier's comment has the list): everything the host
  reuses in layer l+1 is touched only after it observed something on the dense stream ordered after finish(l) - which
  waited on the tier's event: predict_collect's event (d_pf) or the router readback (h_x / h_meta2 / h_w / d_stage /
  LRU slots / parts_n / GPU 1's h_parts); the pass's last finish syncs the dense stream.  One adversarial review
  (agent): no defect; its two hardening points were taken (the event wait is queued first thing in finish_layer_n so
  an error path's dense sync drains the tier too; gpu_run_n's admission skips experts vram_lru already placed, as
  gpu_run does - dormant on our mixed-size layers).  Server A/B, 2 rounds ABBA, fixed prompts, verify pass ms off -> on:
  8k K=1 24.15 -> 23.45 (-2.9 %), K=2 27.43 -> 26.33 (-4.0 %); 127k K=1 25.15 -> 24.25 (-3.6 %), K=2 28.33 -> 27.65
  (-2.4 %); 250k K=1 26.35 -> 25.25 (-4.2 %), K=2 29.48 -> 28.43 (-3.6 %).  Rollback: env DS4_OVL=0.
  (md5s can't prove exactness here: two identical OFF arms gave different texts - restarts are not bit-deterministic,
  as the README says; the teacher-forced test is the numerics gate.)
- **M11 before / after (10-11 05:08-05:39, final_m11.sh, bench/mab776-m2/out/M11-final-before-after.log)**: the live
  server, BEFORE = every M11 decode switch off (DS4_ASYNC=0 DS4_HC_GATES=0 DS4_DEV_SUM=0, no GGML_MMVF_F16_MAX; OVL
  needs async) vs AFTER = the deployed config, ABBA, the same prompts in every arm (fixed nonce), greedy K=2, 512
  tokens of fresh code, lookup 0 and 7 alternating from the same restored state; mean of 4 decodes per cell:

  | decode tok/s (verify pass ms) | 8k | 127k | 250k |
  |---|---:|---:|---:|
  | before M11 | 65.0 (31.3) | 62.5 (32.1) | 60.3 (32.8) |
  | **after M11** | **73.0 (27.1)** | **71.4 (27.7)** | **67.7 (28.6)** |
  | change | +12 % (-13.5 %) | +14 % (-13.5 %) | +12 % (-12.7 %) |
  | after + lookup 7 (this fresh-code prompt) | 75.1 | 71.9 | 69.8 |

  Per item (its own 2-round server A/B, K=2 verify pass): async -3.6 % (8k, 1 round) / -2.4 % / -1.4 %; fused gates
  inside the noise (probe +1.2 %); expert mix on the card - / -3.6 % / -4.2 %; F16 vector kernel - / -3.6 % / -3.3 %;
  OVL -4.0 % / -2.4 % / -3.6 %.  Lookup drafts: edits +42-45 % (short prompts), fresh code +3-4 %, prose neutral.
- **Determinism probe (10-11 05:40)**: the first request after a fresh restart (fixed prompt, greedy) still differs run
  to run with DS4_ASYNC=0 (3/3 different md5s with the other M11 items on) - the engine's known nondeterminism
  (expert placement / CPU-vs-GPU rows), not the async path; the all-off arm matching 2/2 in the before/after run was
  luck.  Validate numerics with the teacher-forced KL test, never with output hashes.

## Session 2026-10-11 day - M12: Marc's three next steps (via the DeepSeek session), order 1 -> 3 -> 2
- **Where a verify layer's time goes now** (nsys of the LIVE server's production passes, 8k, sampled K=2 + lookup;
  bench/mab776-m2/work/prof/S8k-m11, analysis work/m12/layer_phases.py + tier_split.py): GPU 0 busy 77 % (60 % before
  M11).  Per layer, 681 us: attention graph 373 (55 %), expert tier 171 (25 %: GPU 0 expert kernels ~122-152, then ~19
  waiting for CPU / GPU 1 rows), router readback -> first tier op 44 (6.4 %), predict readback -> attention start 50
  (7.3 %: the host's graph launches), finish + predict 32, mix -> finish 4 (M11b).  The big buckets left are the
  attention graph's 150-240 tiny kernels and launch latency, not the expert routing.
- **M12a - the hash-routed layers' prefetch hint (engine default, `DS4_PF_HASH=0` = the old read)**: predict_pick
  reads build_predict's F32-cast tid2eid ids as floats (they were read as int32 -> float bit patterns -> dropped by
  prefetch(): layers 0-2 never prefetched).  A/B, 2 rounds, verify pass ms off -> on: 8k K=1 23.8 -> 23.6, K=2 27.23 ->
  27.05; 127k K=1 24.65 -> 24.35, K=2 27.55 -> 27.55; 250k K=1 25.6 -> 25.65, K=2 28.6 -> 28.48 (noise-level gain;
  it is a bug fix).  out/M12a-pfhash-ab.log.
- **M12c - fp32 accumulation for the decode graphs' F16-weight matmuls (engine default, `DS4_F16_ACC32=0` = half)**:
  alloc_graph sets ggml_prec_set_acc(F32) on every MUL_MAT with an F16 weight in a decode graph (hyper-connection
  mixes, compressors, indexer, MTP eh_proj - the loader makes F16 of our F32 / Q8_0 tensors); ggml-cuda's vector kernel
  otherwise accumulates them in half2 with the activations rounded to half, where llama.cpp multiplies these from the
  GGUF's F32 tensors in F32.  Prompt chunks (own allocator) untouched.  (Flash attention: ggml-cuda ignores the
  precision flag and its vector kernel accumulates in fp32 on NVIDIA anyway - no gap there.)  A/B, 2 rounds, K=2
  verify pass: 27.1 -> 26.55 / 27.5 -> 27.23 / 28.45 -> 28.38 ms at 8k / 127k / 250k - free.  out/M12c-f16acc32-ab.log.
- **M12b - GPU-side expert routing (`DS4_DEV_ROUTE=1` in the deployed config; code default off)**: once the attention
  graph is queued, the hook records an event behind it and queues on the tier's stream: the layer's residency table
  (per expert the device address of its VRAM slot or prefetch staging slot, built from the cache + this layer's
  prefetch, pinned / device copies by layer parity), the fn rows copied out of the router block and quantized, a
  one-thread route kernel (ds4_route_hits: the hits' groups in the host's first-seen order) and the grouped hits
  kernel, which already read its counts on the device - all before the host has the routing.  run_multi then skips
  the x upload and its own hits launch; classification, GPU 1, PCIe, CPU pool, LRU and the mix are unchanged.  Same
  kernel on the same q8_1 rows: exact.  DS4_CHECK_ROUTE=1 compares the card's groups with the host's: 80,384 / 80,384
  layer calls identical (8k + 127k, lookup passes up to 8 tokens; out/M12b-route-check.log).  A/B, 2 rounds, verify
  pass ms off -> on: 8k K=1 23.6 -> 22.85 (-3.2 %), K=2 26.6 -> 26.13 (-1.8 %); 127k K=1 24.35 -> 23.65 (-2.9 %), K=2
  27.6 -> 26.68 (-3.3 %); 250k K=1 25.5 -> 24.65 (-3.3 %), K=2 28.43 -> 27.75 (-2.4 %).  out/M12b-devroute-ab.log.
  Not done (design notes): GPU 1 dispatched from the card (its rows still need the host), PCIe experts decided on the
  card (zero-copy reads), and the attention graph's launch latency (cuStreamWaitValue32 would let the host queue
  finish / predict / attention ahead of the tier's host work).
- **Final gate (all three on)**: teacher-forced KL at 127k vs a plain run 0.0021-0.0024 for every arm (noise band
  0.0017-0.0027), top-1 differs 1-5 / 384; CPU gate 46/46; smoke OK, needles 3/3 at 240k, A-B-A (A2 / A3 / C2 reused 92,808 / 92,963 / 10,158, TTFT 1.3-1.4 s; B2 = the empty-content artifact) - out/M12-gate.log.
- **Incident (10-11 09:49)**: systemd-oomd killed the GNOME Terminal scope running Claude: forced_test.py's analysis
  loaded 7 logit dumps (7 x 198 MB) and did log-softmax in float64 next to the ~42 GiB server with ~9 GiB free.  The
  server and the box survived, the live config was left in test mode (DS4_DUMP_TOKLOGITS) for 30 min.  Fixed: the
  analysis is row by row now (forced_test.py), heavy host-side analysis runs in a memory-capped container / scope.
- **M12 before / after (10-11 10:34-11:04, final_m12.sh, out/M12-final-before-after.log)**: the live server, all three
  M12 switches off vs the deployed config, ABBA, same prompts, greedy, mean of 6 K=2 / 2 K=1 decodes:

  | | 8k | 127k | 250k |
  |---|---:|---:|---:|
  | K=2 decode tok/s before -> after | 75.4 -> **76.0** | 70.5 -> **73.0** | 67.7 -> **71.3** |
  | K=2 verify pass ms | 26.57 -> 25.75 (-3.1 %) | 27.57 -> 26.67 (-3.3 %) | 28.53 -> 27.98 (-1.9 %) |
  | K=1 verify pass ms | 23.75 -> 23.15 (-2.5 %) | 24.40 -> 23.80 (-2.5 %) | 25.45 -> 24.70 (-2.9 %) |

  tok/s moves with each run's draft acceptance (tok/pass 2.02-2.30, restarts are not deterministic); the pass time is
  the engine's own change.

## Session 2026-10-11/12 - M9 coding benchmarks vs stack 73 (Marc via the DeepSeek session; no engine work)
Engine = stack 74 on the deployed M12 config, untouched. Both engines: temperature 1.0, top_p 1.0, no top_k (both
servers then apply top_k 40 / min_p 0.05 by default), the template's default reasoning (thinking on), same harness.
Every host job memory-capped (systemd-run MemoryMax / docker --memory), tasks one at a time, chain = detached user
units. Scripts + tables: `bench/m9/` (m9_chain.sh, m9_resample.sh, m9_resample73.sh, m9_memguard.sh,
m9_release.sh) and `strata-coder/bench/swe/m9_swe.sh` -> `m9_swe2.sh`; SWE overlay `strata-coder/bench/swe/ds4.yaml`.
- **LiveCodeBench** (strata-coder/bench/lcb_eval.py, ab-gsq-20260926, 39 = 24 medium + 15 hard, 32k cap): Strata
  `ds4s-m12-32k-t1` **30/39** (19 + 11; 5 cap hits; mean 15,134 tokens; 2 h 28 of generation) vs stack 73
  `ds4-iq2m-32k-t1` (on file) **34/39** (22 + 12; 3 cap hits; mean 15,602; ~6 h 12). Only 73: abc391_d, abc394_c,
  abc400_d, abc400_g, arc192_e; only Strata: abc394_g. Noise check (Marc): each disagreement sampled 2 more times on
  BOTH engines (`-resample-a/-b`, LCB_IDS): Strata 10/18, stack 73 12/18 -> expected ~32.3 vs 33.0 of 39 = noise;
  abc400_g is hard for both (0/3 vs 1/3). Table: bench/m9/lcb-resample-table.txt. Official pass@1 = first sample.
- **SWE-bench Verified-mini** (mini-swe-agent 2.4.6, offline sandbox --memory 4g --cpus 4, step 250 / 45 min caps,
  graded with 1 worker): Strata all 50 = **35/50** (281 min agent time, 8.0 min + 29.5k output tokens per solved issue,
  median 10.8k tokens a run); stack 73 first 10 only (Marc: 10-12 h for 50). Paired 10: **8/10 both, the same 8**;
  wall 60 vs 77 min, **7.4 vs 9.7 min per solved**, 26.3k vs 17.4k tokens per solved, mean steps 58 vs 44 (Strata
  had 4 runs > 60 steps vs 2). Reference: Qwen Next Strata medium T 0.85 = 37/50, 130 min, 3.5 min + 19.8k tokens per
  solved (summary-temp-1001.md); GLM-5.3 = 12/22. Summary: strata-coder/bench/swe/summary-m9-1011.md.
- **Failure patterns**: 50/50 submitted, 0 step/time-cap hits, 0 malformed tool calls, 0 repeated-action loops,
  0 empty patches; 1 reply spent its 32k tokens without a tool call (sphinx-7748, recovered, unresolved); 6 runs
  > 100 steps (max 150). Stack 73: 1 command timeout, nothing else.
- **Incidents (all harness/host, none engine)**: (1) stack 73's resample first ran while llama.cpp was still loading
  (it answers /v1/models then 503s every chat) -> scripts now wait for a real chat reply; 73's django-12276 that
  overlapped the load was re-run (old one kept as `.503-load`). (2) 20:46: stack 73's llama-server crept from ~44 to
  ~51 GB RSS over its 4 h window (anon ~9 GB, likely its prompt RAM cache); host down to 0.7 GiB -> systemd-oomd killed
  the `claude-portainer` Remote Control tmux scope (restarted 21:07; the relay session died). m9_memguard.sh added.
  (3) 22:21: one sphinx-9320 trajectory ballooned mini-extra to 3.1 GB (a huge command output) and the chain unit's own
  3G cap killed the chain; restarted with each issue in its own 4.5G scope; 9320's re-run was clean (299 MB).
- **End**: GPUs released at 01:08 (Marc 10-11 22:40: mode page -> off after the agent phase, grading is CPU only) -
  stack 74 is STOPPED, not left running; other sessions took the GPUs.

## Session 2026-10-12 afternoon - M13: upstream Strata 0.1.42 / 0.1.43 / 0.1.43.1 ideas (Marc via the portainer-91 session)
Rules: port ideas only (no merge onto v0.1.41+), each change behind an env switch (0 = the old path), basic tests only.
Code: 5cbd45a4 + 63377648 (+ bench kit 88bca4ea). Logs: bench/mab776-m2/work/m13/ (probe.log, chain.log, final.log,
gate.log / gate2.log).
- **Decode graph node cuts (upstream f-143-amdk / f-143-intel, CUDA translation)** - new ggml patch
  tools/ds4/patches/ggml-zzz-dsv4-hc-fused.patch (two ops, CPU forms too; tools/ds4/test_hc_fused.cpp checks them on the
  GPU):
  - `DS4_HC_MIX` (upstream "HC norm from registers"): hc_pre's mixes = one kernel (the 16,384-wide RMS norm on n
    blocks, ~5 us, plus the [16384 x 24] F16 matvec); the scale is applied after the dot, so float rounding only: vs
    the deployed path (GGML_MMVF_F16_MAX=8) <= 2.5e-7 relative.  Teacher-forced KL at 127k vs an HC_MIX=0 reference:
    0.0028-0.0033 vs the restart noise 0.0027-0.0035 (top-1 4-7 / 384 vs 5-8) - inside the noise.  8k ABBA pass
    K=1 -0.7 %, K=2 -2.0 %.  Kept.
  - `DS4_HC_PRENORM`: the stream mean + RMS norm * weight in one kernel, bit-identical (same reduction layout as
    rms_norm_f32<1024>; test_hc_fused: 0 differing floats at n 1..2048).  8k ABBA +0.2 / +0.3 % (noise: one block a
    token is as latency-bound as the norm it absorbs).  Kept (exact; in the measured set).
  - `DS4_LEAN` (upstream "lean verify graph"): no l_last tap copy without the gate's taps, no +0 on the router weight
    sum, a verify pass's compressed-key mask as one -inf fill + one strided copy (was 2(n-1) fills + ~3(n-1) concats).
    Same values.  8k ABBA K=2 -1.2 % (K=1 +1.1 %, n=2, where only the +0 and the tap go).  Kept.
  - `DS4_SH_FOLD` (upstream "moe_hit_add folded into the combine"): the tier's on-card mix adds the shared expert, the
    finish drops its add.  Exact, but 8k ABBA K=1 22.1 -> 22.6 ms, K=2 24.7 -> 24.8: the add moves ONTO the tier's
    critical path.  REJECTED: opt-in (=1).
  - Attention graph nodes per (layer, pass size) variant: 180.8 -> 165.5 (-8.5 %; DS4_DBG_GRAPH census, probe.log).
  - Not ported: host-flag waits folded into copies (our graphs have no host-flag waits), the gfx12 pipeline-window /
    chunk rules (HIP/SYCL).
- **#1737 route-resident swap one warp per token** -> `DS4_ROUTE_PAR`: our M12b route kernel (one thread, a serial walk
  of up to 48 entries with a dependent table read per distinct expert) on 64 threads, every output by counting.
  DS4_CHECK_ROUTE=1: 22,016 / 22,016 layer calls identical to the host's groups.  8k ABBA K=1 -0.7 %, K=2 -1.3 %.
  Kept.  The swap itself (route a near-tie miss to a resident expert) changes the model's routing: not ported (our
  misses are ~4 % of lookups after M7a).
- **STRATA_ADAPT_OVERLAP**: not ported - our LRU admissions are a few us of device copies inside the tier (nsys: D2D
  0.8 us average) and the prefetch DMAs already land under the attention (DS4_OVL); 0.1.43.1 made it opt-in again
  (slower on a 5090 Laptop).
- **PcieAuto**: not ported - our PCIe share is a fixed `--pcie 0.55` of the ~4 % misses; nothing to refine.
- **#1892 NaN guard** -> `DS4_NAN_GUARD` (default on): each pass's logits rows checked on the host (an integer scan of
  the exponent bits, before any of the pass's tokens is emitted); a fire ends the request `DONE ... nonfinite`, drops
  the live state, checkpoints and parked conversations; serve/server.py (upstream #1892's server side) sends a request
  that emitted nothing once more, ends a started one as "length".  DS4_NAN_INJECT=<k>,.. fires on the k-th check; CPU
  gate test test_nonfinite_logits_drop_the_state.  No fire in the live gate.
- **#1712 / #1996 pinned uploads** -> `DS4_PF_PINNED`: a prompt chunk's inputs (embeddings 32 MiB at 2048, masks,
  indices) from one pinned buffer, async on the backend stream.  Prefill unchanged within noise (the uploads were ~0.3 %
  of a chunk).  Kept (exact).  #1996's bounce ring for other pageable async copies: our decode's per-pass input span is
  a few KB, not worth it.
- **#1601 cgroup-aware RAM**: our park guard already read memory.max minus reclaimable file cache; it now also takes
  memory.high and DS4_MEMORY_LIMIT_MIB.  **#1620** (a cancel keeps the restored conversation): already so in ds4_serve
  (`live` is updated per chunk / token, never dropped on cancel).  **#2049 / #2065** (parking vs the watchdog): our
  watchdog ends only an engine silent 90 s with < 0.5 s CPU and < 1 MB I/O; a park / unpark is a ~1 s memcpy (no swap
  in the container) - nothing to do.
- **serve/** (three-way applied, conflicts merged by hand): #1624 (a reordered tool set keeps its first order and the
  prompt cache), #984 (a thinking budget capped below max_tokens + the progress line), #1869 / #1870 (leg timings;
  timings.prompt_n = what the engine read), #1892's server side, #2026 (a quoted </tool_call> in a message is text) -
  extended to DeepSeek: `｜DSML｜` is a type-4 token (matched anywhere), so a user message or tool result quoting DSML
  handed the model call syntax; `<｜DSML｜` / `</｜DSML｜` are now marked as text outside assistant messages (the bare
  marker overlaps itself and would turn off the one-pass tag scan).  #2037 not taken: tool_call_recovery is ignored
  for DeepSeek.  Tests: serve suite 662 OK, CPU serve gate 47 / 47.
- **Before / after (final.log)**, the same binary, every M13 switch off vs the shipped set, ABBA, greedy, same prompts:

  | | 8k | 127k | 250k |
  |---|---:|---:|---:|
  | K=2 verify pass ms | 26.00 -> **24.88** (-4.3 %) | 26.88 -> **25.90** (-3.6 %) | 28.32 -> **26.68** (-5.8 %) |
  | K=1 verify pass ms | 23.35 -> 22.05 (-5.6 %) | 23.80 -> 23.00 (-3.4 %) | 25.25 -> 23.95 (-5.1 %) |
  | K=2 decode tok/s | 76.9 -> **78.7** | 73.8 -> **75.5** | 71.0 -> **74.8** |
  | K=1 decode tok/s | 72.6 -> 73.9 | 70.7 -> 72.7 | 67.0 -> 70.2 |
  | prefill tok/s | 1,300 -> 1,286 | 1,521 -> 1,523 | 1,438 -> 1,433 |

- **Gate** (deployed snapshot, canonical config): smoke OK; needles 3/3 at ~241k (sampled, T 1.0; the haystack now
  comes from strata-data/src-01402 - src-0140 was deleted; deepseek-v4/needle_ds4.py takes NEEDLE_ROOT); A-B-A: parked
  chats back after 0.6-1.4 s (reused 92,787 / 19,272 / 93,008 / 9,772).
- Lessons: two GPU windows must never overlap (the probe's final restore raced the next window's first arm; harmless
  only because the arm's config write came second - verify the engine's /proc environ, as lib.sh's envline does).

## Session 2026-10-12 night - M14: upstream Strata-DSV4.1 review (Marc via the portainer-91 session)
Upstream renamed Strata-DS4 -> **Strata-DSV4.1** (remote `dsv41`, fetch only; 1,700 commits past our merge base
924516a0 incl. their merge of upstream Strata past v0.1.41).  Almost all of it is the new **V4.1 engine**
(`tools/v41/engine/ds41_*`, a fork of our ds4_dense / ds4_moe for DeepSeek-V4.1-Flash: 40 layers x 384 experts,
Engram, per-expert ternary/Q2_0/Q3_K tiers) - `tools/ds4` itself only got REAP pruning, the stale-graph fix and fork
isolation (fork code moved into fork/*.cuh + tools/ds4/cmake/fork.cmake).  Their own docs/ds4/V41_START_HERE.md is the
map; V41_DECODE_OVERLAP_PLAN.md has the attribution.  Same rules as M13: env switch per change, basic tests only.

| idea (upstream commit) | applies to us? | measured / expected | ported? |
|---|---|---|---|
| tier graph capture, STRATA_TIER_GRAPH (66773c2a) | no: built for V4.1's 9 per-expert format classes (3 kernels each); our layer has one class per site and DS4_DEV_ROUTE already queues the hits from the card | upstream: neutral (239 -> 52 launches/token, device busy under them) | no |
| demand copies on a copy stream, STRATA_DMA_OVERLAP (8ad94b23) | barely: our PCIe share is ~1.5 % of a pass | upstream: neutral | no |
| --pf-b sweep, "the prefetch earns its traffic" (5e5f4c16) | yes, but the verdict flips here: their 4090 misses to NVMe, our two cards hold nearly every hot expert | our sweep (6 arms, mirrored): K=2 pass @8k 0 / 0.7 / 1.43 = **23.38** / 23.88 / 24.00 ms, @127k **24.30** / 25.02 / 25.27; K=1 21.20 / 21.45 / 21.55, 22.00 / 22.55 / 22.40; prefill same | config `--pf-b 0` |
| warp top-k / rank argsort, rank-argsort.patch (75009c03) | yes: router top-6 of 256 per layer was a bitonic sort | 8k ABBA: K=2 24.65 -> 24.43 ms (-0.9 %), K=1 22.05 -> 22.10; KL in noise | yes, default on (GGML_CUDA_RANK_ARGSORT=0 = old) |
| fused finish + predict (3d4328cd) | yes, adapted to our async path (the readback rides like predict_launch's) | 8k ABBA: K=1 22.40 -> 22.40, K=2 24.73 -> 24.82: neutral | yes, opt-in DS4_FUSE_PREDICT=1 |
| stale-graph unbind compares recorded buffer pointers (42c77429) | yes: alloc_graph asked a possibly-freed buffer its usage (use after free when the shared arena grows) | correctness | yes, verbatim, no switch |
| x-skip, S4 sync removal, one metadata copy, routed sum on the card, one-sync readbacks (75009c03, 66e63458, a966904c, 59920d80) | already ours (DS4_DEV_ROUTE skips the x upload, DS4_DEV_SUM / DS4_OVL / DS4_ASYNC) | - | n/a |
| REAP soft prune + --arena-adapt (s37) | lossy (+0.3 % ppl, -72..115 % on uncalibrated languages if hard); our hit rate leaves little to gain | upstream: decode +10 % on their box | **no - REJECTED by Marc 2026-10-13: lossy, do not port or trial; don't re-pitch** |
| mmq-deterministic.patch | did not restore prefill exactness upstream either | - | no |
| DSpark harness fix (wo_a FP8 32x32 block scale dropped, 3813b587) | V4.1 only: 2.58 accepted drafts/pass on teacher states, not yet in their engine | - | noted |
| vision encoder (tools/v41/vision, 3813b587) | V4.1 only, Python/CPU, slice 1 of 2 | - | noted |

- **Gates**: CPU serve gate 47 / 47 (stale-graph fix in); teacher-forced KL at 127k vs an M14-off reference: M14 on
  plain / K=1 / K=2 = 0.0025 / 0.0026 / 0.0024, restart noise 0.0023 / 0.0025 / 0.0026 (top-1 differs 7-9 vs 7-12 of
  384).  Logs: bench/mab776-m2/work/m14/chain14.log (scripts lib14.sh, ab14.sh, pfb.sh, kl14.sh, summ14.py).
- **Deploy**: run/ = this build (warp top-k on, fused predict off) + `--pf-b 0`; run.prev = M13 88bca4ea.  No smoke on
  the final binary: the window ended with the mode page All off (00:11:06, Marc's request); it differs from the tested
  binary only by DS4_FUSE_PREDICT's default (the =0 arms ran it).
- **Why the prefetch now loses**: since M7a (GPU 1's static tier) + the elastic GPU 0 cache, almost every routed expert
  is resident; a hint's DMA then competes with the tier for the copy engine and the predict graph is a launch a layer.
  Upstream's verdict holds for a box that misses to NVMe (and will for us on V4.1 if it ever streams).  Unmeasured: long
  agent sessions whose routing drifts - if those decode slower, `--pf-b 0.7` is the one-word rollback.

### V4.1 note (Marc may try DeepSeek-V4.1-Flash after the 128 GB RAM upgrade)
- **Their release** (HF IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored, from dealignai's abliterated FP8): V41-GSQ.gguf
  = 16 parts, **163 GB (152 GiB)** - experts 42.8 % ternary PTQ1_0 / 39.2 % Q2_0 / 18 % Q3_K = 2.25 bpw, dense Q8_0 -
  + V41-MTP-Vision.gguf 8.4 GB (DSpark drafts + vision) + tokenizer/rank.bin.  **Engram is NOT in it**: shards 47/48 of
  the FP8 checkpoint (~203 GB) must be downloaded separately and live on NVMe (read per token at layers 1 and 14).  Not a
  llama.cpp/ollama GGUF (per-expert quant types): their engine only.  Disk: ~375 GB (+163 GB while reassembling the
  parts); the models drive has 657 GB free.
- **Fit after the upgrade** (128 GB = ~119 GiB, minus ~8 GiB kept free and ~10 GiB for the box): ~100 GiB pinned arena +
  GPU 0 ~30 GiB of experts beside dense + KV = ~130 of ~142 GiB experts resident (~90 %); with a GPU 1 tier (~13 GiB) ~97 %.
  Upstream runs it on a 4090 + 90 GB with ~57 % resident at ~26 tok/s decode / ~250 tok/s prefill; ours would stream
  little from NVMe, so decode should land well above that - a guess, not measured: ~35-45 tok/s plain decode.
- **What merging toward V4.1 takes**: a trial merge of dsv41/main into mab776-main conflicts in **22 files (~70 hunks)**:
  serve/ (frontend 15, server 31, tests ~8: their base is upstream Strata past v0.1.41, ours 0.1.40.x + cherry-picks),
  README* (docs), engine 9 (CMakeLists 1, iq_kernels.{cu,hpp} 3, ds4_dense 1, ds4_moe 4, ds4_generate 1, a test 1).  Not
  done: M13's rule was "no merge onto v0.1.41+", and the V4.1 engine is a separate tree (tools/v41) - our tools/ds4 work
  (GPU 1 tier, MTP draft/verify, async/OVL, DEV_SUM/ROUTE, parking, M13 ops) does NOT carry into it by merging; each piece
  would be ported into ds41_* by hand.  Their V4.1 engine today: ONE GPU, no draft/verify (DSpark offline only), served
  by upstream Strata's server through serve_ds41.py (hard-codes ~/AI/Strata; V4.1 effort is numeric 1-100, DSML with a
  leading space, joyai-llm pre-split).  Shortest path when the RAM lands: run their engine as-is on GPU 0 (a test stack),
  measure, then decide whether to port the GPU 1 tier + MTP verify.  To stay mergeable meanwhile: our ggml changes stay
  patch files (theirs: tools/v41/ggml/, ours: tools/ds4/patches/ - the argsort patch = theirs + a 3-line header), our engine
  changes stay env-switched, and a future merge resolves serve/ by taking theirs + re-applying our serve commits.


## 128 GB quants (2026-10-16) - UD-Q4_K_XL and UD-Q3_K_M on stack 74, vs the live IQ2_M
Trial: the 128 GB compose (mem_limit 110g) with a trial snapshot `run.q128` (6c0c1aea + commit da544a75); the live run/,
ds4-server.json and compose were never changed, and stack 74 is back on IQ2_M (50g). Same bench as M13/M14 (ksweep2.py,
greedy, 512 tokens, the m14-ab prompts). LCB-13 skipped (Marc 10-16). Engine fixes needed (da544a75):
- `slots auto` read only shard 1 of a split GGUF ("no layer has routed expert tensors"): the geometry now opens every shard.
- UD-Q4_K_XL is uniform (every expert MXFP4, 12.75 MiB) and `--vram-grow` refused uniform files: env `DS4_FORCE_MIXED=1`
  sends it through the mixed-size (elastic) cache path. Without it (no grow, 2,305 slots) decode was the same or worse.

| | IQ2_M (live, M13) | UD-Q4_K_XL | UD-Q3_K_M |
|---|---|---|---|
| KLD vs full | 0.484 | 0.013 | 0.157 |
| fits? | yes, 42 GiB | yes: arena 98 GiB, 0 experts on NVMe; container 96-101 GiB, host min 10 GiB free | **yes since 41128c2c** (was NaN, see below): arena 72.6 GiB (7,128 experts), 0 on NVMe; container 75.2 GiB anon + 2.4 pinned (RSS 85 incl. model pages), host min 35.8 GiB free |
| experts on GPUs | ~2,000 GPU1 + GPU0 cache | GPU0 2,309 -> 2,841 (grow) + GPU1 868 (10.8 GiB) of 11,008 | GPU0 2,822 -> 3,478 (grow) + GPU1 1,062 (10.8 GiB) |
| decode K=2 8k / 127k | 78.7 / 75.5 tok/s | **42 / 41 tok/s** (pass 49 / 52 ms); K=1 44.6 / 44.4 | **56.2-56.7 / 52.8-53.1 tok/s** (pass 36-39 ms); K=1 53.5 / 54.8 |
| prefill 8k / 127k | 1,286 / 1,523 tok/s | 8.2k **751 tok/s (10.9 s)**, 126k **853 tok/s (148 s)** | 8.2k **903 tok/s (9.1 s)**, 127k **1,052 tok/s (120.7 s)** |
| needle 127k / tools | 3/3 @240k / ok | FOUND / 6/6 | **FOUND (118k tokens) / 6/6** |

Q4_K_XL tuning: arena 95 left 206 experts on NVMe = ~700 MB/s of reads during decode, 26 tok/s; arena 98 (all in RAM)
= 44 tok/s. `--pcie` 0.3 / 0.55 / 0.8 = flat (42-45). Expert hits ~77 % GPU0 + 8.6 % GPU1 + ~15 % CPU/PCIe (IQ2_M ~4 %):
the expert is 1.7x bigger and two thirds of them live in RAM. K=1 is as fast as K=2 here.
**Q3_K_M NaN = fixed (2026-10-16 evening, commit 41128c2c, local).**  Not the mixed formats' kernels: the verify path
(`gpu_run_n`, every 2+-token pass - MTP verify, and the prompt tail that seeds the MTP window) quantized each token's
activation for the CPU pool with the LAYER's format but placed and read the rows `im.f.act_bytes` apart = the LARGEST
layer's layout.  Here that is the one all-MXFP4 layer (Q8_0 rows, 4,352 B) while the IQ3_XXS layers write Q8_K rows
(4,672 B): the rows overlapped, the CPU-computed experts of every multi-token pass read the next token's bytes, the
logits went non-finite and the NaN guard ended each request at its first token.  Single-token decode (offset 0) and
uniform files (IQ2_M all Q8_K, Q4_K_XL all Q8_0) never saw it.  Fix: rows `cpu::kNativeActBytes` apart (as `nact_n` is
allocated and as the core's multi-token path does) - for uniform files only the offsets change.  Found with
`ds4_generate --mtp --verify` on the file: single-token decode and the chunked prompt were clean; with verify passes
every token after the first was id 0; `--pcie 1.0` (no CPU pool) made it coherent -> the CPU pool's multi-token path.
IQ2_M on the fixed engine (same live config, stack 74): 8k K=2 76.0 / 77.6 tok/s (M13 runs 75.8-79.0), prefill 1,229
tok/s, no NaN lines.  Q3_K_M vs Q4_K_XL: KLD 0.157 vs 0.013, but decode ~1.3x and prefill ~1.2x faster and ~20 GiB less RAM (78 vs 96-101 GiB).
Not deployed: stack 74 stays on IQ2_M (live compose, run/ = 6c0c1aea's engine); the fix ships with the next snapshot.sh.
