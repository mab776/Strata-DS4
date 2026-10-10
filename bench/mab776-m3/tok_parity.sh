#!/bin/bash
# mab776 M3: llama-tokenize (b11429 = stack 73's image, CPU, vocab-only) on every INDIR/*.txt -> OUTDIR/<name>.lcpp,
# our tokenizer -> OUTDIR/<name>.ours, then compare. --no-escape: b11429 llama-tokenize UNESCAPES \\n etc. by default.   usage: tok_parity.sh INDIR OUTDIR [GGUF]
set -euo pipefail
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); mkdir -p "$OUT"
M="${3:-/media/mab776/nvme 2To/models/ollama/models/blobs/sha256-a79d3b8c742f240dc947758bbeb46d266e9328ace6a3b71704ab4bf2a8610d68}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=none --memory 4g --memory-swap 4g -u "$(id -u):$(id -g)" -v "$M":/m/ds4.gguf:ro -v "$IN":/in:ro \
  -v "$OUT":/out --entrypoint sh ghcr.io/ggml-org/llama.cpp:full-cuda -c '
  for f in /in/*.txt; do n=$(basename "$f" .txt); x=""; case "$f" in *.nps.txt) x=--no-parse-special;; esac
    /app/llama-tokenize -m /m/ds4.gguf -f "$f" --ids --no-bos --no-escape --log-disable $x > "/out/$n.lcpp" 2>/out/$n.lcpp.err || echo "llama-tokenize failed on $n"
  done'
docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=none --memory 6g -u "$(id -u):$(id -g)" -v "$REPO":/src:ro -v "$M":/m/ds4.gguf:ro -v "$IN":/in:ro \
  -v "$OUT":/out -w /src strata-ds4-pytest:py312 sh -c \
  'python bench/mab776-m3/tok_parity.py ours /m/ds4.gguf /in /out && python bench/mab776-m3/tok_parity.py compare /in /out'
