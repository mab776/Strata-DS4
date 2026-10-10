<h1 align="center">Strata-DS4 · mab776 fork</h1>

<p align="center"><b>DeepSeek-V4-Flash (284B MoE) served from one workstation — an OpenAI-compatible server that out-runs
llama.cpp on the same box at every context length</b><br>
Fork of <a href="https://github.com/Indras-Mirror/Strata-DS4">Indras-Mirror/Strata-DS4</a> (a DeepSeek-V4 engine built on
<a href="https://github.com/Niko1221/Strata">Strata</a>) · Linux · NVIDIA · MIT</p>

> **Status: research code that runs a real daily-use server on one machine.** Everything here was built and measured
> on the PC below, with unsloth's 2-bit GGUF. It serves one request at a time. Other GPUs or quants are untested.
> The engineering log (every measurement, every decision, what was tried and dropped) is in
> [README.mab776.md](README.mab776.md).

## What this fork adds

Upstream Strata-DS4 brought Strata's idea to DeepSeek-V4-Flash: hot experts in a VRAM cache, the rest in pinned RAM,
misses split between the CPU and PCIe. This fork turns that into a server you can point clients at, and makes it fast
at long context:

- **A resident server** (`tools/ds4/ds4_serve.cpp` behind Strata's `serve/server.py`). OpenAI and Anthropic APIs,
  streamed tool calls (DeepSeek's DSML turned into OpenAI `tool_calls`), thinking in `reasoning_content`, and the full
  262,144-token context.
- **Prompt reuse.** Ring checkpoints at turn boundaries, so an agent turn reads only what it appended. A conversation
  you switch away from is parked in RAM (~0.3 GB at 127k) and comes back in ~1.3 s instead of a 90 s re-read.
- **Expert tiers across two GPUs.** GPU 0 has an elastic VRAM cache: it shrinks to make room while a prompt's chunks
  run, then regrows for decoding. GPU 1 holds a static tier. The rest lives in a pinned RAM arena, served by the CPU
  pool and PCIe. Disk reads during decode: none.
- **Speculative decoding** with DeepSeek's own MTP head, using exact speculative sampling (the output distribution is
  unchanged). The head can draft **chained tokens** (K per verify pass; K=2 is the fastest), and its experts run on
  GPU 1.
- **Prefill kernels for long prompts:**
  - chunked prefill (2,048 tokens) on MMQ, with FP4 MMQ for the MXFP4 layers;
  - a tensor-core lightning indexer (`tools/ds4/patches/ggml-lightning-indexer-mma.patch`);
  - a radix-select top-k (`tools/ds4/patches/ggml-topk-radix-select.patch`). The radix select is 10× faster than
    sorting every row, and it also made verify passes almost flat in depth;
  - lean masks for the compressed layers: the indexer takes per-query visible counts
    (`tools/ds4/patches/ggml-lightning-indexer-count-mask.patch`) instead of a dense 65k × 2k visibility mask.

## How fast is it?

**Machine:** RTX Pro 5000 48 GB (GPU 0) + RTX 5070 Ti 16 GB (GPU 1) · Ryzen 9 7950X · 64 GB DDR5 · NVMe.
**Model:** unsloth `DeepSeek-V4-Flash-0731` **UD-IQ2_M** (84.7 GiB).

The baseline is llama.cpp (`full-cuda`) on the same machine with the same GGUF and both GPUs, tuned with
llama-fit-params, plus the DSpark Q2_K drafter (experts in RAM, n = 3).

All numbers are tok/s, measured through the HTTP API: code generation, 512 tokens, greedy.

| context | prefill, Strata-DS4 | prefill, llama.cpp | decode, Strata-DS4 (K=2 / K=1) | decode, llama.cpp |
|---|---:|---:|---:|---:|
| 8k   | 1,100-1,250 | 1,011 | 60-68 / 56-63 | 40.8 |
| 127k | **1,297**   | 843   | **61** / 57    | 39.0 |
| 250k | **1,192**   | 693   | **61** / 57    | ~36 |

- **Without drafts**, decode is 47 / 45 / 43 tok/s at 8k / 127k / 250k. A K=2 verify pass (3 tokens) yields 2.1-2.3
  tokens on average.
- **K=2 for sampled requests, K=1 for greedy ones.** Clients normally sample. In sampled needle runs at 240k (T=1.0
  and T=0.6, 60 per K) neither K ever looped. In greedy (temperature 0) runs, K=2 fell into repetition loops a little
  more often (11/106 vs 3/95). Teacher-forced checks show K=2's logits are as close to plain decoding as plain decoding
  is to itself, with or without rejected drafts. Greedy requests therefore fall back to one draft (`--mtp-k-greedy`).
- **Needle in a haystack:** 3/3 at 240k tokens (depths 10 / 50 / 90 %). One caveat: greedy decoding occasionally
  talks itself into a repetition loop ("But wait… let me check…") after finding the word. The default sampling
  (temperature 1.0, min_p 0.05) avoids it.
- **Draft depth sweep** (K = drafts per pass, before the top-k kernel):

  | context | plain | K=1 | K=2 | K=3 | K=4 | K=5 |
  |---|---:|---:|---:|---:|---:|---:|
  | 127k | 44.3 | 53.1 | 56.3 | 53.9 | 47.0 | 44.8 |

  Every extra verify token costs ~15 % of a decode step (six more experts per layer, partly on the slower tiers), so
  two drafts is the optimum here.
- **Not bit-deterministic.** Which tier computes an expert depends on the cache state, and the CPU's Q8_K and the
  GPU's q8_1 activations round differently. Two runs of the same greedy request can therefore diverge at a near-tie.

## Running it

Requirements: Linux, Docker with the NVIDIA runtime, and an NVIDIA GPU with sm_120 (the build targets `sm_120a`;
other archs need `CMAKE_CUDA_ARCHITECTURES` changed and are untested). You also need ~50 GB of RAM for the server
container, and the GGUF.

```bash
git clone -b mab776-main https://github.com/mab776/Strata-DS4 && cd Strata-DS4
# build ds4_serve in a throwaway CUDA 13 container (no GPU needed); applies tools/ds4/patches/*.patch to ggml
BUILD_DIR=build-ds4-cu13 CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04 BUILD_TARGETS="ds4_serve ds4_generate" \
    ./mab776-build.sh
# then run serve/server.py with an engine config: tools/ds4/ds4-server.json.example = the deployed one
python serve/server.py --engine strata --config ds4-server.json --port 8080
```

The engine flags that matter are all in [tools/ds4/ds4-server.json.example](tools/ds4/ds4-server.json.example):

| flags | what they do |
|---|---|
| `--gpu1 1` | GPU 1 expert tier |
| `--vram-margin 9 --vram-grow 2` | GPU 0's elastic cache |
| `--prefill-chunk 2048 --chunk-mmq --chunk-prestage`, env `STRATA_MMQ_FP4=1` | prefill |
| `--mtp DeepSeek-V4-Flash-MTP-bf16.gguf --mtp-device 1 --mtp-k 2` | speculative decoding (K drafts per pass; greedy requests: `--mtp-k-greedy`, default 1) |
| `--park-mib 4096` | conversation parking |
| `--comp-type iq4_nl --icomp-q8` | compressed KV cache |
| `--arena-gib 41` | pinned RAM arena |

A request can change the draft depth for itself with `"strata_tune": {"mtp_k": K}`, where 0 means plain decoding.

The Portainer stack this runs in (Dockerfile, compose file, snapshot/deploy scripts) is described in the log.

## Verifying it

- **CPU gate** (`tools/ds4/run_serve_gate.sh test`, 44 tests on small synthetic models) covers:
  - reuse, checkpoints and parking;
  - STOP at any moment;
  - the sampler;
  - verify passes with 1-7 chained drafts, accepted or rejected at every position.

  On the CPU backend every one of these is bit-identical to plain one-token decoding. New tests are mutation-checked
  (break the code on purpose; the gate must fail).
- **Server suite:** `tools/ds4/run_serve_gate.sh serve-suite` (645 tests for `serve/`).
- **Kernels:** each patch was checked standalone against a reference before it went in. The indexer kernel was
  compared with the stock kernel. The top-k kernel was checked for exact sets against a stable sort, including ties,
  `-inf` tails and all-equal rows.
- **Upstream's gates** (dense half vs `ds4_ref`, expert tier vs an F32 reference) are described in
  [README.upstream.md](README.upstream.md).

## More reading

- [README.mab776.md](README.mab776.md): the fork's engineering log (milestones M1-M10, every benchmark, rejected
  ideas, how to update and roll back the live server).
- [README.upstream.md](README.upstream.md): upstream Strata-DS4's README (architecture, its RTX 4090 numbers, flag
  reference).
- [README.strata.md](README.strata.md): upstream Strata (the Qwen3.8-Flash-Next engine).
- [docs/ds4/](docs/ds4/): upstream's design notes and findings.

## Credits

- [Indras-Mirror/Strata-DS4](https://github.com/Indras-Mirror/Strata-DS4): the DeepSeek-V4 engine this fork builds on.
- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp), including its DeepSeek-V4 kernels.
- [unsloth](https://huggingface.co/unsloth): the GGUF.
- philpax: the bf16 MTP-head GGUF.
- avlp12: the 0731-aligned MTP sidecar (tested, a wash on speed).
- DeepSeek: DeepSeek-V4-Flash.

License: MIT (see [LICENSE](LICENSE)).
