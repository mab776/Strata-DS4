ultracode

Goal: make our Strata-DS4 fork + our unsloth UD-IQ2_M DeepSeek-V4-Flash-0731 "usable like Strata stack 67 today": one native Portainer stack started from the mode page, OpenAI API with streamed tools + reasoning, 262k context, prefix reuse and parking, at least as fast as llama.cpp stack 73 at depth, quality checked. Stack 73 stays untouched as the fallback.

Read first: ~/Documents/portainer/strata-ds4/README.mab776.md (state, build/run, model paths, fork survey, rebase decision); ~/Documents/portainer/deepseek-v4/RESULTS.md; strata-coder/docker-compose.yml; project-maya/README.mab776.md (how we plugged GLM into Strata's server). Memory: deepseek-v4-flash-ollama-bench, deepseek-v4-custom-quant-idea, strata-coder-stack, glm-trial-strata-stays-off, compare-models-per-solved-task.

Rebase decision (10-09, don't redo it): keep the DS4 engine core on DS4's own v0.1.39 base. Moving it onto v0.1.41 gave 14 conflict hunks in 8 core files plus a hidden swiglu_entries_kernel argument break. The ggml pin is the same, the speedups are Qwen-only, and every DS4 sync (~27 commits/day) would re-fight those conflicts. DO take the server: `git checkout fb58e0db -- serve` (plain v0.1.41) + hand-merge tools/strata_tokenizer.py (0.1.41 piece cache + DS4 _pre_split). Ask me before revisiting.

Borrow before writing (MIT, credit them). Fetch-only remotes already exist:
- jackwwg83: serve/deepseek.py, fake engine, protocol tests, snapshots; DeepSeek V4.1, on a 0.1.40.2 serve base.
- aleksche, branch dev: a compact deepseek.py with a strict DSML parser.
- strata-glm: the DS4 author's newer repo, with a byte-sized --slots auto and an engine --serve mode.
If upstream DS4 ships --serve itself, adopt it.

Each session: `git fetch origin && git merge origin/main` (the branch is published: merge, never rebase or force-push). Rebuild, run the CPU tests (ds4_geometry_test, test_ds4_moe, tokenizer golden), and record the upstream hash.

Milestones (commit each; push ONLY to remote mab776, branch mab776-main):
M1 CPU: --slots auto sized from the real per-layer expert bytes (7.19-8.38 MiB on our file, not 6.75); --vram-margin 2.0 at long ctx.
M3 CPU: serve/ graft + tokenizer, pre=joyai-llm. Ids must equal llama-tokenize (vocab-only, CPU --rm container) on the golden corpus and the 250k haystack.
M2 GPU window = GO/NO-GO.
- Setup: a RAM-capped container that leaves ≥8 GiB free on the host, run with --ctx 262144 --comp-type iq4_nl --icomp-q8 --vram-margin 2.0.
- Measure prefill tok/s, code decode tok/s and peak RAM/VRAM at 8k/127k/250k (lcpp_depths.py prompts). Run needle_ds4.py 3/3 at 251k and a p600 ppl check vs llama.cpp on the same ids.
- Capture stack 73 /apply-template renders of 10 conversations (plain, multi-turn, tools, tool results, history) for M4.
- Report a table against stack 73 (both GPUs): no drafter, prefill 1011/843/693 and decode 28.0/25.0/22.3 tok/s; with the drafter, code decode 40.8/39.0/~36.
- Decision: go if decode at 127k ≥39. Between 25 and 39, go only if I accept that M7 is needed. Stop if decode at 127k <25, if a needle fails, or if ppl is >2% off. Then STOP for my go.
M4 CPU: DeepSeek module (serve/deepseek.py + small hooks).
- Special tokens decode as text (｜DSML｜ must not crash); stop id {1}.
- Tools wrapped as type=function; bos_token; thinking ON by default.
- Effort mapping: none → off, low and medium → on with no preamble, high and xhigh → high, max → max.
- from_json, and streaming DSML → ToolCall, with malformed DSML falling back to text.
- Config: allowed_hosts includes ds4-lcpp + the container name; effort_position start; reasoning_close_retry true; Qwen-only recovery, vision and parallel off.
- Tests on the mock engine: streamed and whole calls, 2 calls, a call inside reasoning, reasoning_content, the #1053 retry. Renders must be byte-equal to the M2 captures.
M5 CPU: tools/ds4/ds4_serve.cpp as a NEW file (don't edit ds4_generate.cpp, upstream rewrites it daily).
- Implements Strata's line protocol (INFO/READY/GEN/PP/T/RESUME/DONE/ERR/STOP/QUIT), the full sampler, sequence reset, ring snapshots, append reuse and a prompt-end checkpoint.
- Pass on mini fixtures (make_mini_gguf.py): output equals ds4_generate token-for-token, reused equals fresh, 20 requests with no reload, STOP → DONE in <1 s.
M6 GPU: the live server, with a resident prefill buffer. Pass:
- 20 requests with RSS within ±1 GiB; a 127k follow-up TTFT ≤2 s; no watchdog trip on prompts over 300 s.
- Stack 73's tool smoke test and a 10-step OpenHands task pass.
- Depth within 5% of M2; needles 3/3; 2-4 parked chats, each resuming in ≤2 s.
M7 (ask first): the drafter on GPU 1 with a sampled verify, or GPU 1 as an extra expert tier. Only if decode misses stack 73's numbers.
M8 (ask first): a native Portainer stack via the API, as a drop-in for stack 73.
- Compose: GPU 0, mem_limit, restart "no", healthcheck start_period ≥300 s.
- Drop-in identity: alias ds4-lcpp, :8330→8080, model id deepseek-v4-flash. /health, /v1/status and /metrics must answer.
- Add a llama.cpp|Strata variant on the mode page's DeepSeek tile, plus VS Code/OpenHands/OWUI entries.
- Pass: off → DeepSeek (Strata) → off, clean, 3 times.
M9 nights: LCB 39 at 32k t1.0 (bar 34/39); SWE Verified-mini vs stack 73, judged per solved task.

Ultracode: run each of M2, M4, M5, M6 and M8 as understand → design → implement → independent adversarial verification. Before M2 you may only do M1 and M3, then report and wait.

Ask me before: every GPU window (the mode page must be all off; GLM or Qwen may be in use; afterwards restore NOTHING), the M2 decision, M7, any mode-page or Portainer change, acting on a changed rebase picture, anything public.

Rules:
- Never commit, check out or build in the live src-0141 tree (remote strata-local); fetch from it only.
- Never touch stacks 67/72/73.
- Don't download the 81 GB upstream Q2. Q3_K_M (on disk) waits for the 128 GB RAM upgrade.
- Never pkill -f. RAM-cap big jobs.
- At the end, update README.mab776.md, memory, the skill and CLAUDE.md, and delete temp files.
