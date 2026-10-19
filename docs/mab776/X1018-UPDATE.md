# Sync 10-19: Xolotl 87157b2e -> per-request routing + image input on V4.1 (stack 78)
Marc's request (relayed, 2026-10-19 morning). Branch layout since this morning: `mab776-main` = upstream + ours (GitHub
default, the stacks build from it), `main` = an untouched mirror of Xolotl `main`.

## 1. The sync
`mab776-main` 84cd7fec (pushed) = merge 954821c6 of `xolotl/main` 87157b2e (per-request routing 43bdd869 / 50870132,
16-row verify as one tier call 76d16570, opt-in D3 DMA stream 0eb8f824, 1017 encoding sync 403e47ec, vision input
4e843f3e, Windows, dashboard) + one fix commit. `main` refreshed = 87157b2e.

Conflicts (4 files): `serve/server.py` (our mtp_k / lookup keys beside upstream's route keys); `ds41_dense.cpp` (our async
span sync kept; the route meter forces a sync); `ds41_generate.cpp` (our STRATA_LOOKUP, parking, GPU 1 stats kept beside
the route-meter line; upstream's own STRATA_SERVE_LOOKUP block stays as in x1018 = ours wins); `ds41_moe.cpp` (our
STRATA_D3_OVERLAP kept, off when STRATA_D3_DMA_STREAM is on, on upstream's banked staging slots).

Fix commit 84cd7fec (V4.1 only):
- upstream's 16-row tier calls aborted our GPU 1 static tier at the first 16-row prompt pass ("native MMVQ requires
  1 <= ncols <= 8"): GPU 1's q8_1 buffer is sized per column and quantized 8 columns a call (what upstream did for GPU 0);
- `STRATA_TIER_ROWS=r` (default 16 = upstream): rows per tier call; 8 = the split before 76d16570;
- the file-tier misses one call runs on the card (FILEG) are capped at 8 per 8 rows OF THAT CALL (upstream's own
  comment); upstream's constant 16 changed what arena_adapt keeps even on <= 8-row calls.

### Exactness (one-shot greedy token md5, 1,536 tokens, served flags, arena 82.5; `mab776/pf/verify_ab.sh`)
| build / knobs | md5 | decode |
|---|---|---|
| live engine (026c42f7) | **4bcc9c9e77b1** | 123.3 tok/s |
| merged, default (16-row calls) | 83ec63b528a2 (differs from token 10) | 117.7 |
| merged, `STRATA_TIER_ROWS=8` | 5c7caef814fb (GPU 1 has 1,236 experts, not 1,237) | 99.1 |
| merged, `STRATA_TIER_ROWS=8` + `STRATA_GPU1_RESERVE_GIB=0.745` + the FILEG fix | **4bcc9c9e77b1 = live** | 123.6 |
| merged, `--vision`, no image | = the same build without it (83ec63b528a2 vs 83ec63b528a2) | |
| exact routing: live / merged 8 rows / merged 16 rows | 389fbc4c967a / **389fbc4c967a** / f9308ef5af17 (from token 831) | 28 tok/s |

So the merge is clean; the served config carries `STRATA_TIER_ROWS=8` + `STRATA_GPU1_RESERVE_GIB=0.745` (the bigger
16-row buffers cost GPU 1 one expert, and with the route bias on the routing follows the resident set).

**76d16570 vs the 16-row exactness gap (report only): it does not close it.** A 16-row tier call is not bit-identical to
two 8-row calls here (exact routing: the text differs from token 831), and the gap itself is in the dense half
(GGML_BATCH_INVARIANT covers 8 rows + the open race in <= 8-row tails), which 76d16570 does not touch. Greedy lookup stays off.

**V4 (stack 74): not redeployed.** Its binary changed only through two upstream build fixes (GGML_CUDA_USE_GRAPHS on
strata_mmq - a Windows layout fix; Windows-compat attributes in the PTQ1 CPU kernel V4 does not use). Greedy probe
(`ksweep2.py`, throwaway, 8k / 127k, K=1 then 3 x K=2): pass 32-40 ms live vs 31-36 ms merged, prefill 934 / 1,099 vs
938 / 1,102 tok/s, decode the same. The texts cannot decide it: here even three K=2 repeats from one restored state give
three md5s (V4's known non-determinism). No V4 feature came with the sync, so the live snapshot stays (3f8c6cfb).

## 2. Routing per request (`-exact`)
Upstream's serve_ds41.py maps a model name ending in `-exact` to route_bias 0 / file penalty 0 for that request (and
`-fast` to 0.5 / 0.3); the engine applies a changed bias again before the decode. Stack 78's config lists
`deepseek-v4-flash-exact` and `deepseek-v4.1-flash-exact` as aliases; the plain ids stay fast (0.5 / 0.3, as live).

Exact vs fast on our box (throwaway, the deploy candidate, sampled 256 tokens; two rounds, ABAB):
| | fast (default ids) | exact (`-exact`) |
|---|---|---|
| 8.7k: prefill / decode | 623-706 tok/s (12.4-14.0 s) / 58-61 tok/s | 758-792 tok/s (11.0-11.5 s) / 28-30 tok/s |
| 123k: prefill / decode | 1,977-1,997 tok/s (61.6-62.2 s) / 50-52 tok/s | 2,003-2,014 tok/s (61.1-61.4 s) / 27-29 tok/s |
| turn +300 / +1,500 (32 tokens out) | 4.8-5.2 / 7.3-7.9 s * | 6.1-6.2 / 8.6-9.0 s |

Exact routing halves the decode (every expert the model picks, so far more PCIe and file-tier reads; one-shot greedy: 28
vs 123 tok/s with DSpark) and adds ~1.2 s to an agent turn; prefill is the same. (* these turns ran with the encoder at
arena 82.5 - see section 3; the deployed config's fast turns are in the gate below.)

Clients: llm-router lists both `-exact` ids (stopped: "not running"; V4 running: "not served by the running stack" -
V4 answers them as its plain model). VS Code block `deepseek-strata/vscode-copilot.txt` has a "DeepSeek V4.1 exact"
entry; OpenHands host has a saved profile `DeepSeek-exact` (`openai/deepseek-v4-flash-exact`, also in
`openhands/host/seed-settings.sh`).

## 3. Image input (V4.1)
- Engine `--vision` + the config's `vision` section: upstream's `tools/v41/vision/ds41_vision.py` running DeepSeek's
  reference ViT + aligner (`vision.py`, `image_processor.py`, MIT, from deepseek-ai/DeepSeek-V4.1-Flash `inference/` @
  314113f8 -> `models/strata-ds41/vision-ref/`, with model.py / kernel.py / engram.py / config.json / LICENSE) on the
  CPU in f32, 16 threads, weights from our `V41-MTP-Vision.gguf`.
- Python: a venv on the models drive, `models/strata-ds41/vision-venv` (torch 2.14.1+cpu, numpy 2.5.4, pillow 13.0;
  built inside strata-ds4:cu13 so it matches the image's python 3.12; ~1 GB); the image is unchanged. Both are mounted
  read-only into stack 78 at their host paths.
- `--vision` alone changes nothing (text md5 above; turns 3.67 / 5.97 s = live).
- Encoder RSS 2.15-2.23 GiB (4.1 GiB peak while it dequantizes at start - before the engine loads). It runs only before
  a prompt's prefill (FIFO: the engine is idle then); during decode it sleeps - no CPU taken from the tier.
- bf16 (DeepSeek's own precision) is broken in upstream's encoder (a dtype mismatch) - f32 as upstream tested.
- RAM: a resident encoder takes its 2.2 GiB from the file cache inside the 104 GiB cap - at arena 82.5 the +300 turn went
  3.7 -> 5.5 s and the margin to 3.06 GiB. "lazy" (start with the first image, stop when idle) costs a ~90 s cold start
  per image (first image 108 s vs 18 s) and its stop logs an "unsupported command" (the elastic VRAM call) - not used.
  Arena 81.0 failed the live gate's RAM rule (3.54 GiB after the full gate: the steady footprint creeps to ~100.5 GiB as
  conversations accumulate) -> rolled back, **arena 80.0**.

Tests (throwaway + live, through the API):
| request | answer | time / prompt tokens |
|---|---|---|
| terminal screenshot: which test failed, expected / got, passed, time | test_route_meter, 0.71 / 0.69, 36 passed / 1 failed, 4.82 s - all exact | 17.6 s / 424 |
| bar chart: every value + the highest | Jan 120, Feb 95, Mar 180, Apr 60, May 240; May - exact | 8.4-9.8 s / 394 |
| fridge photo of the weekly board (French handwriting): the grocery list | 7/7 items; 4 misspelled (Beurre arachiide, Foumange, Gravu, Trisruits) | 29.3 s / 1,071 |
| two images in one message | "Green triangle / Yellow circle" (in order) | 7.6 s / 476 |
| text turn after the chart (same conversation) | 695 (the sum) | 5.1 s / 462 |
| plain text request after images | correct | 3.8 s / 41 |

## 4. Deploy to stack 78 (2026-10-19 afternoon)
Snapshot 84cd7fec (engine 783d89c170fe) + compose with the two read-only mounts (venv, vision-ref); `run.prev` = the 00:00
lookup-14 snapshot 026c42f7 (rollback: stop 78, swap `models/strata-ds41/run` <-> `run.prev`, and PUT the compose from
`deepseek-v41/docker-compose.yml.bak-20261019-sync`).
- **First try, arena 81.0: FAILED the RAM rule** (3.54 GiB after the full gate) -> rolled back to run.prev + the old
  compose, then redeployed at **arena 80.0**.
- Shim fix d7a276c0 copied into the live snapshot afterwards (only `tools/v41/serve/serve_ds41.py`; noted in VERSION), so
  run.prev stayed the rollback.

| gate (live stack 78) | sync, arena 80.0 | before (lookup-14 gate, 00:00) |
|---|---|---|
| one-shot greedy md5 (code) | = pre-sync at arena 82.5 (4bcc9c9e77b1); the deployed arena 80.0 changes residency, so not compared | 4bcc9c9e77b1 |
| prefill 8.7k / 123k | 711 tok/s (12.3 s) / 1,995 tok/s (61.6 s) | 699 tok/s (12.5 s) / 2,028 tok/s (60.6 s) |
| decode 8.7k / 123k (sampled 256) | 54.9 / 49.8 tok/s | ~56 / ~52 |
| turns fresh / +300 / +1,500 | 10.4-11.0 / 4.11-4.28 / 6.43-6.45 s | 10.60 / 3.84 / 6.08 s |
| tools | 5/6 (the Unicode-minus case) then 6/6 x3 | 6/6 |
| images | screenshot, chart, photo, two images, follow-up turn: as in section 3 (live: 8.9-12.8 s) | - |
| needles (fresh, nonce) | 480k FOUND (235 s); 246k x3 FOUND (~120 s each) | 4/4 |
| margin under 104 GiB, encoder loaded | **4.73 GiB** (needles) / 4.94 (bench + turns); host >= 13 GiB free; no OOM | 4.79 |

The turn cost (+0.3 s at +300, +0.35 s at +1,500) is the price of the resident encoder + 2.5 GiB less arena under the
104 GiB cap.

**The empty needle answers (fixed).** Two needles in the first arena-80 runs answered '' - not an engine failure (finish
stop, 54 / 89 tokens, no nonfinite): the model ended its thinking with a quoted line and the special `</think>` token and
wrote the answer straight after it (`...Remember it."</think>falcon-quartz-940`). Strata's frontend rule #537 reads a
`</think>` right after a quote and not followed by a line break as the model QUOTING the tag, so the answer stayed in
reasoning_content. Same rule in the pre-sync snapshot (not a sync regression). Fix d7a276c0 in the V4.1 shim: the server
feeds one token's text at a time and `</think>` is ONE special token (id 128822), so a delta that is exactly the tag gets a
line break after it -> #537 sees the end; a spelled-out tag (several tokens) is still a quote. Unit test: the failing case
gives content '' with the old shim and 'falcon-quartz-940' with the new; the re-gate above found all four.

**Router (stack 75)** rebuilt + redeployed (rollback image `llm-router:pre-20261019`, `router.py.bak-20261019-v41vision`):
images pass only to a stack whose live `/v1/models` entry lists `image` in `architecture.input_modalities` (V4.1 does, V4
doesn't -> the text note), for `auto` and named ids. Live: auto / deepseek-v4-flash / deepseek-v4.1-flash-exact all read
two images through :8340; from inside the open-webui container -> llm-router:8000 an image request answered "May, 240";
with stack 74 running, auto + an image -> "1 image(s) -> text note" and V4 says it can't see it. Open WebUI shows the
image button for every model without a capability override (`vision ?? true`) - the DeepSeek ids have none; the button
itself was not clicked (no Open WebUI login on the box).

**V4 (stack 74)**: not redeployed (section 1).


# MORNING SUMMARY (2026-10-19, overnight unattended)
**Shipped**
- **Stack 78 (V4.1)** = x1018 (our branch merged onto upstream Xolotl) + **lookup drafts up to 14 for sampled requests**:
  tree 026c42f7, ds41_generate 488550b66d99. Copy-heavy edits 117-144 tok/s (was 98-122 at 7, +12..+29 %); greedy
  one-shot md5 unchanged; prefill 8.7k 699 tok/s (12.5 s) / 123k 2,028 tok/s (60.6 s); turns 3.84 / 6.08 s; tools 6/6;
  needles 4/4 (x1018 gate) + 1 (lookup gate); margin >= 4.28 GiB under 104. Greedy lookup stays OFF (not 7): a 9-row
  pass is past the 8 rows batch invariance covers. Rollback: stop 78, swap `models/strata-ds41/run` <-> `run.prev`
  (= b59ac13e, x1018 without lookup 14; the 74faff41 snapshot before it is gone - rebuild from tag mab776-v41-pre-x1018).
- **Stack 74 (V4)** = x1018 (tree 3f8c6cfb, ds4_serve 8becae81e296): the "-12 % at 123k" was 256-token sampling noise -
  a greedy probe from one restored state gives the live speed (K=2 pass 34-37 ms at 8k/127k) and the same text;
  tools 6/6, needles 3/3 at 246k, RAM equal. Rollback: swap `models/strata-ds4/run` <-> `run.prev` (= 2ee91ae8).
- **One branch `main`** on mab776/Strata-DS4 (Xolotl + our V4 + our V4.1 work; builds both engines; README.mab776.md,
  also at .github/README.md). Both binaries from main are output-identical to the live ones (V4.1 one-shot md5 + the
  tokenizer ids; V4 greedy probe) - the stacks were not redeployed.

**Not shipped**: V4 lookup 14 (its native MMVQ kernels take <= 8 columns: the engine aborts; needs a split - and V4
greedy can't be proven exact anyway); the adaptive arena on V4.1 (no gain on turns); lookup 14 for GREEDY V4.1
requests (exactness unproven: a <= 8-row prompt-tail pass is not run-to-run deterministic - an open race to find).

**For Marc**: GitHub's default branch is still `mab776-main` (it points at main's commit); set the default to `main`
(Settings -> General -> Default branch) and then delete `mab776-main` - no gh / token on the box to do it from here.

**Branches / tags on mab776/Strata-DS4**: branches `main` (cd93ebca+) and `mab776-main` (= main, the GitHub default
until switched). Tags: `upstream-main-1009` (924516a0), `mab776-main-pre-x1018` (733ce42a), `mab776-v41-pre-x1018`
(74faff41), `mab776-main-final` = `mab776-main-x1018-final` (3f8c6cfb), `mab776-v41-final` = `mab776-v41-x1018-final`
(026c42f7). Local: `strata-ds4/` on `main` (builds both: `ENGINE=ds4|ds41 ./mab776-build.sh`), `strata-ds41/` detached
at main (keeps the blob mirror + test kit); `xolotl-v4/` and `xolotl-v41/` removed. Mode page OFF, GPUs empty.

# x1018: our V4 + V4.1 branches on upstream Xolotl (2026-10-18)

Upstream Indras-Mirror/Strata-DS4 and Strata-DSV4.1 now both redirect to **Indras-Mirror/Xolotl** (the V4 and V4.1
engines plus Strata's core and server in one tree). Fetched as remote `xolotl` (push disabled) in the strata-ds4
repo; head **f7c561b7** (2026-10-18 05:27).

Nothing live changed: stacks 74/78 and their `run/` snapshots untouched, mode page OFF, GPUs empty at the end.

## Branches (github mab776/Strata-DS4)
| | V4 (stack 74) | V4.1 (stack 78) |
|---|---|---|
| old head (rollback tag) | `mab776-main-pre-x1018` = 733ce42a | `mab776-v41-pre-x1018` = 74faff41 (the live one) |
| new branch | **`mab776-main-x1018`** = 3f8c6cfb | **`mab776-v41-x1018`** = b59ac13e |
| worktree / build | `portainer/xolotl-v4`, `build-x1018/` | `portainer/xolotl-v41`, `build-x1018/` |
| binary md5 | ds4_serve 8becae81e296, ds4_generate c9d4d1b79035 | ds41_generate e43794468f86 |
| live binary | ds4_serve 3b0c64bce42f | ds41_generate 54fb9fb30465 |

`mab776-main` / `mab776-v41` were NOT moved: they are what the stacks deploy from.

## What upstream added since our bases
**V4.1** (base a9f8fbc2, 24 engine commits): the conversation-adaptive arena (`STRATA_ARENA_REBAL`, default ON),
an in-memory park for side requests + partial unpark (`STRATA_SERVE_PARK`, default ON), its own keep-cache
(`STRATA_SERVE_KEEP_CACHE`, opt-in), a port of our lookup drafts up to 14 (`STRATA_SERVE_LOOKUP`), DSpark eviction
during long prompts (`STRATA_SERVE_DSPARK_EVICT`) and re-seed on a repeated prompt, a VRAM-aware prompt-batch
cap + retry, the upload_span memo (bit-identical), lazy prestage, a routing trace + offline simulator, `--tier-batch-pcie`,
dense Q6_K at load (wrapper default only), GSQ v2 quant tooling (act-order GPTQ, ternary/budget alloc, repair),
and the server moved in-tree (Xolotl's own 0.1.44 `serve/`).
**V4** (base 924516a0, 26 engine commits, mostly GLM-driven): REAP pruning / soft prune / skip-file / renorm-skip,
`--pcie auto`, `--vram-pin`, `--arena-admit`, prestage cap, routed-expert LoRA (GLM), s28 de-sync (no gain
upstream), fork-isolated kernel files, plus Strata's server 0.1.41 -> 0.1.44 and the two llama.cpp fixes below.

## Conflicts and how they were resolved
V4.1 (32 hunks): features built on both sides kept side by side under their own envs - parking (ours = RAM/disk
store; xolotl's in-memory park is switched off while ours runs), keep-cache (ours first, renamed `regrow_kept`),
DSpark on a repeated prompt (ours, then theirs); lookup: ours kept, xolotl's copy dropped (same design); our
decode loop + xolotl's `rebalance_tick`; `ExpertCache::evict_from` added twice -> xolotl's (same behaviour for us).
Shim: xolotl's (it now runs the in-tree 0.1.44 server) + our park /health hook. New knob **`STRATA_PB_CAP=0`**:
xolotl's batch cap is calibrated on a 24 GB card and cut our 32,768-column prompt batch to ~6.1K (123k prefill
1,777 tok/s (69.2 s)); the served config in the branch sets it off and `STRATA_ARENA_REBAL=0` (= live behaviour).
GLM-only ggml patch (cuda-graph-max-tokens) not applied.

V4 (30 files): `serve/` = xolotl's 0.1.44 + our server diff since the 0.1.41 graft re-applied (DeepSeek family,
mtp_k/lookup keys, DSML literal tags on their own marks beside 0.1.44's tool tags); engine: our tier kept where it
was measured (M7a GPU 1 tier, M11-M14 verify mix, M6 elastic cache), xolotl's REAP/renorm-skip/pcie-auto/vram-pin
added beside it, GLM LoRA + s28 metadata not wired into our gpu_run; MXFP4 gate/up added to xolotl's fork format
list; duplicate members/targets fixed (incl. an upstream CMake duplicate that fails a configure with tests on);
**llama.cpp #30103 (flash-attn tile race) + #29953 (MMQ out-of-bounds) applied to the V4 ggml tree** (the merged
prefill code needs it; V4.1's tree already had them).

## Smoke tests (throwaway ds41-pf / ds41-test, served configs, arena 82.5 for V4.1)
**V4.1** - one-shot greedy (1,536 tokens): **md5 identical to the live build** (4bcc9c9e77b1 with
`GGML_BATCH_INVARIANT=4` at depth 5 and one-row passes; e84e4be9c62a with it unset), same expert stats, verify
37.3 ms/pass. Serve (cap off, adaptive arena on in that run):

| | live (10-17 gate, A arms) | x1018 |
|---|---|---|
| prefill 8.7k | ~700 tok/s (12.3-13.0 s) | 694 tok/s (12.6 s) |
| prefill 123k | ~1,975 tok/s (61.7-63.0 s) | 2,023 tok/s (60.8 s) |
| turn +300 / +1,500 | 3.95-4.06 / 6.33-6.36 s | 3.86 / 6.17 s |
| decode 8.7k / 123k (sampled) | 55.6-59.8 / 48.1-57.4 | 59.4 / 65.0 |
| tools / effort | 6/6, medium | 6/6, medium |

**V4** - live run/ (ro) vs x1018, A B B A, 256 sampled tokens:

| | live | x1018 |
|---|---|---|
| prefill 8.7k | 937 / 951 tok/s (9.3 / 9.2 s) | 955 / 956 tok/s (9.1 s) |
| prefill 123k | 1,089 / 1,087 tok/s (112.9 / 113.1 s) | 1,095 / 1,093 tok/s (112.2 / 112.5 s) |
| decode ~7-9k (4 runs) | mean 50.6 tok/s | mean 51.1 |
| decode 123k (2 runs) | 54.1 / 55.7 | 45.9 / 50.0 (**-12 %**) |
| K=2 verify pass at 123k | 38.3 / 36.6 ms | 39.2 / 40.4 ms (+6 %) |
| short-prompt K=1 pass | 32.2 ms | 32.2 ms |
| tools, memory | 6/6, 77.5 GiB | 6/6, 77.4 GiB |

V4 greedy output differs across restarts on both builds (known; the flash-attn race fix does not change that), so
no md5 verdict. The long-context verify cost is the open item (suspects: #30103's added barrier, the merged tier).

## Recommendation (Marc decides)
- **V4.1 x1018: worth deploying** - exact vs live, prefill/turns equal or a little better, 0.1.44 server. Deploy
  needs `deepseek-v41/snapshot.sh` to also copy `serve/` (the shim now imports the in-tree server) and the
  served config's `STRATA_PB_CAP=0`; the stack's src-01431 mount becomes unused.
- **V4 x1018: not yet** - long-context decode -12 % in 2/2 runs; nothing in the merge is a V4 speed win. Keep live.

## New things the update makes possible (most promising first; not measured unless said)
1. **V4.1 conversation-adaptive arena** (`STRATA_ARENA_REBAL=1`): swaps cold arena experts for this conversation's
   hot file-tier ones between requests. Applies (5,223 file-tier experts beside the 82.5 GiB arena). Upstream:
   small-turn prefill -10 %, file-tier reads -21..-30 %; our one run with it on: turns -3..-5 % (with other changes).
   Risk: output depends on what is in the arena (the served file penalty) -> not reproducible across histories; no
   extra RAM. Test: turns ABBA, 2 rounds.
2. **Lookup drafts up to 14** (upstream's kVerify 16) in our `STRATA_LOOKUP` (capped at 7 now), V4.1 and V4: upstream
   edit decode 32.0 -> 35.9 tok/s (+12 %). Exact (verified drafts). Small port.
3. **V4.1 GSQ v2 tooling** (act-order GPTQ, `--ternary-cap-at` / `--budget-gib`, v41_repair, fp8 teacher score):
   the tools for our own clean GSQ from the sniffer calibration corpus. Quality, offline, long.
4. **Census-ranked residency for V4** (GLM's `--census`, portable): persistent per-expert decode counts rank VRAM /
   arena instead of the static profile; GLM gate: hit 23.9 -> 34.1 %, decode +6 %. V4's VRAM hit rate is ~0.75-0.79.
   Exact. Medium port.
5. **V4 `--pcie auto`**: learns the CPU-pool vs PCIe split from measured costs; our fixed 0.55 was tuned before the
   RAM went to DDR5-4400. A few % decode at best; greedy text then depends on timing.
6. **Partial resume in our park store** (upstream's partial unpark idea): resume at a >= 3/4 prefix instead of exact /
   one-token rollback - for clients that re-render more than one token. Exact (positional caches). Small-medium.
7. **V4.1 DSpark eviction during long prompts** (`STRATA_SERVE_DSPARK_EVICT=4096`): +4.3 GB VRAM for the prompt cache
   of >32K prompts (upstream 16.8K: +19 %, from a wider batch we already have). Small for us; +2.6 s re-load.
8. `STRATA_PRED_ROUTER` (lighter prefetch-prediction router), `--vram-pin`, `--arena-admit`: low or no expected gain.

Skipped as settled or not for us: dense Q6_K (declined), tier-batch-pcie (lost), route bias (closed), REAP / soft
prune / skip-file (quality cost; V4 Q3_K_M has no file tier), prompt batch 12288 (we run 32768), s28 de-sync.

## Step 1 - V4.1 x1018 DEPLOYED to stack 78 (2026-10-18 22:16, Marc's go)
`deepseek-v41/snapshot.sh` copies `serve/` now (+ optional `SNAP_DEST` for a test copy; backup
`.bak-20261018-x1018`); deploy config + `STRATA_PB_CAP=0` + `STRATA_ARENA_REBAL=0` (backup `.bak-20261018-x1018`).
A test snapshot run exactly as the stack runs it answered first (health 80 s, pong, the four model ids, tools 6/6).
Snapshot = tree b59ac13e, engine e43794468f86; run.prev = the 10-17 09:50 snapshot (74faff41). PUT stack 78 (live
compose unchanged). Branch `mab776-v41` fast-forwarded to b59ac13e and pushed.

| gate (live stack 78) | x1018 | before (10-17 gate) |
|---|---|---|
| one-shot md5 (final binary) | 4bcc9c9e77b1 / e84e4be9c62a (= live) | the same |
| prefill 8.7k / 123k | 730 tok/s (12.0 s) / 2,037 tok/s (60.4 s) | ~700 / ~1,975 tok/s |
| decode 8.7k / 123k | 56.5 / 52.2 tok/s | 55.6-59.8 / 48.1-57.4 |
| turn fresh 9.5k / +300 / +1,500 | 10.63 / 3.89 / 6.10 s | 10.99 / 3.95-4.06 / 6.33 s |
| tools / effort | 6/6, medium | 6/6 |
| needles | 4/4 (479,953 in 243 s = 1,975 tok/s; 232,660 x3 in ~118 s) | 4/4 |
| margin under 104 GiB (bench / 480k / 246k) | 5.16 / 4.89 / **4.28** GiB, no OOM events | 5.79 / 5.54 / 5.18 |

## Step 2 - lookup drafts up to 14 (V4.1): big speed win, exactness NOT proven -> not shipped
Engine: `STRATA_LOOKUP` / `STRATA_LOOKUP_GREEDY` accept up to 14 now (commit 301b74ef, opt-in; served stays 7). A/B
ABBA (throwaway ds41-pf, served config, the arena off), three "copy this file with one rename" edits (4.7-8.9K prompts,
2,000 tokens out) + the 8.7k bench; decode tok/s, mean of 2 runs:

| | cap 7 | cap 14 | change |
|---|---:|---:|---:|
| edit f1, sampled (what serve does) | 97.8 | 126.6 | **+29 %** |
| edit f2, sampled | 122.3 | 144.8 | **+18 %** |
| edit f3, sampled | 107.2 | 120.5 | **+12 %** |
| edits greedy f1 / f2 / f3 | 101.9 / 120.2 / 114.8 | 114.2 / 145.1 / 111.0 | +12 / +21 / -3 % |
| 8.7k bench (sampled, normal text) | 60.2 | 61.9 | noise |

Lookup acceptance 93-98 % on edits. **Why not shipped:** a 14-draft pass carries up to 16 rows and
GGML_BATCH_INVARIANT covers 8, so the output is not proven identical. Two checks, both inconclusive: (1) serve greedy
text differs between two cap-7 servers on 2 of 3 files - serve greedy output is not reproducible across restarts (so
no A/B can prove it there); (2) a one-shot probe (prompt tail in 16-row vs narrower verify passes, `STRATA_TAIL_CHUNK`)
with a 16-row ggml extension (`tools/v41/ggml/mab776-batch-invariant-rows16-UNPROVEN.patch`, not applied): stable,
but the narrower-tail reference itself is not run-to-run deterministic (tail chunks <= 8: two runs, two md5s) - a
second open item (a race in the <= 8-row paths outside decode; production tails use 16 rows, decode stays exact).
To ship it: find that race, then prove 16-row passes = 1-row ones.

## Step 3 - adaptive arena (`STRATA_ARENA_REBAL=1`), report only: no gain here
Turns ABBA, 2 rounds per server (4 x 2 runs per arm), served config, cap 7:

| turn prefill (mean) | arena off | arena on |
|---|---:|---:|
| fresh 9.5k | 11.11 s | 11.25 s |
| +300 | 3.66 s | 3.77 s |
| +1,500 | 5.99 s | 6.01 s |

Within noise or slightly worse: our 82.5 GiB arena + mirror + layer-wide reader already serve the turns' file tier,
and back-to-back requests leave the swap reader little time. Stays off.

## Lookup 14 SHIPPED on V4.1 for sampled requests (2026-10-19 00:00, Marc's decision)
Served config `STRATA_LOOKUP=14` (backup `deepseek-v41/strata-ds41.json.bak-20261019-lk14`; branch commit 026c42f7,
`mab776-v41` fast-forwarded). **Greedy lookup stays OFF** (`STRATA_LOOKUP_GREEDY` unset = 0, as it was live) rather
than the 7 in Marc's message: a greedy lookup pass of 7 drafts + the pass token + a refeed row is 9 rows, one past the
8 rows GGML_BATCH_INVARIANT covers, so 7 is not provably exact either; 0 is. (6 would fit in 8 rows if wanted.)
Snapshot engine 488550b66d99 (tree 026c42f7); run.prev = the 10-18 22:16 snapshot (b59ac13e).

| gate (live stack 78) | lookup 14 | lookup 7 (22:16 gate / A/B) |
|---|---|---|
| greedy one-shot md5 (depth 5 / one-row passes) | 4bcc9c9e77b1 / 4bcc9c9e77b1 | the same |
| sampled edits f1 / f2 / f3 | **116.9 / 144.4 / 124.8 tok/s** | 97.8 / 122.3 / 107.2 |
| prefill 8.7k / 123k | 699 tok/s (12.5 s) / 2,028 tok/s (60.6 s) | 730 (12.0 s) / 2,037 (60.4 s) |
| decode 8.7k / 123k (256 sampled) | 53.1 / 51.0 | 56.5 / 52.2 |
| turns fresh / +300 / +1,500 | 10.60 / 3.84 / 6.08 s | 10.63 / 3.89 / 6.10 s |
| tools | 4/6, then 6/6 x3 re-runs (the two misses = known sampling cases: an extra unrequested call, a Unicode minus) | 6/6 |
| 246k needle / margin | FOUND (232,660 in 120 s) / 4.79 GiB, no OOM events | 4.28 GiB at the 246k needles |

## Step 4 - V4 x1018: the "-12 % at 123k" was noise -> DEPLOYED to stack 74 (2026-10-19 00:12)
The first smoke tests timed 256 SAMPLED tokens per arm, whose acceptance swings with the content. A controlled probe
(`ksweep2.py`: one prompt per depth with a fixed nonce, read once, then greedy K=1 / K=2 decodes of 512 tokens from the
same restored state; throwaway ds41-test) shows no regression - K=2 verify pass ms:

| | live (733ce42a), 2 rounds | x1018 (3f8c6cfb), 2 rounds | x1018 without #30103 (the flash-attn race fix) |
|---|---|---|---|
| 8k | 36.3 34.3 34.3 / 36.4 35.1 34.8 | 36.4 34.7 34.0 / 35.8 34.4 34.4 | 36.1 35.1 34.7 |
| 127k | 39.5 37.1 36.3 / 36.2 36.6 37.3 (mean 37.2) | 36.1 35.0 38.4 / 36.1 38.0 36.4 (mean 36.7) | 36.3 37.2 37.1 |
| 8k K=1 / first K=2 text | md5 e9e07103a2 / 92e8658963 | the same | the same |

So nothing to fix: the race fix costs nothing, the merged tier is the same speed, and the first 8k decodes give the
same text. Deployed: `deepseek-strata/snapshot.sh xolotl-v4 build-x1018` (ds4_serve 8becae81e296, run.prev = the 10-16
22:23 snapshot 2ee91ae8), PUT stack 74 (live compose unchanged); `mab776-main` fast-forwarded to 3f8c6cfb and pushed.

| gate (live stack 74) | x1018 | live before |
|---|---|---|
| probe K=2 pass 8k / 127k | 36.2 34.5 34.1 / 36.3 36.8 35.7 ms | as above |
| prefill 8k / 127k (probe) | 944 / 1,101 tok/s | 943-954 / 1,100-1,102 |
| bench 8.7k / 123k sampled | prefill 914 tok/s (9.5 s) / 1,080 (113.8 s); decode 49.6 / 54.5 | 937-951 / 1,087-1,089; 48.7-48.8 / 54.1-55.7 |
| tools | 6/6 | 6/6 |
| needles 246k | 3/3 (232,634 tokens in ~214 s) | 3/3 |
| memory | 79.0 GiB peak after the 246k needles (cap 96); 77.4 vs 77.5 GiB after the 123k bench in the smoke tests | 77.5 |


## Step 5 - ONE branch `main` (2026-10-19 01:00-01:50)
`main` = mab776-main-x1018 (V4) merged with mab776-v41-x1018 (V4.1), both already on Xolotl f7c561b7 (c89b8d06) + three
V4.1-on-main fixes found by the smoke tests (the shim's tokenizer subclass takes `from_dir`'s arguments, its
`effort_kwargs` / `budget_effort` take the frontend's family-table argument) + the README (cd93ebca). Overlaps:
mab776-build.sh (both sides) -> `mab776-build-ds4.sh` + `mab776-build-ds41.sh` + a dispatcher; one comment in
expert_cache.hpp; each engine's ggml patch set stays its own tree (`third_party/llama.cpp` = V4.1, `third_party/llama.cpp-ds4`
= V4, mounted at the old in-container path so the V4 build is unchanged); V4.1's snapshot.sh now also copies
`tools/llama_unicode_classes.py` (main's tokenizer imports it). Checks of the binaries built from main:

| | V4.1 (ds41_generate 41a2579162ba) | V4 (ds4_serve 4304948a07ca) |
|---|---|---|
| exactness | one-shot md5 4bcc9c9e77b1 (depth 5 + one-row passes) / e84e4be9c62a (invariance off) = live; tokenizer ids identical on 5 texts (12,242 tokens) | greedy probe 8k: e9e07103a2 / 92e8658963 = live (one start gave another text - V4's known restart flip; the repeat matched) |
| speed | 8.7k 736 tok/s (11.9 s), turns 3.73 / 5.95 s (live 3.84 / 6.08), edit 127.3 tok/s | K=2 pass 35-37.5 ms at 8k / 127k (= live), prefill 950 / 1,100 tok/s |
| serve | pong, effort, Anthropic /v1/messages, tools 6/6 + 5/6, the 4 model ids | tools 6/6 |

md5s differ from the live binaries only because each now compiles the other engine's shared kernels (dead code for it).
Tags + branch cleanup + snapshot.sh defaults (both -> strata-ds4/; V4.1 build-x1018), AGENTS.md (backup
.bak-20261019-main), the mab776-portainer skill (backup .bak-20261019-main) and the memory notes updated.
