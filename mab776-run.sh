#!/bin/sh
# mab776: run ds4_generate on GPU 0 in a RAM-capped container (cgroup OOM-kills THIS instead of freezing the box;
# replaces tools/ds4/memguard.sh, which needs a systemd user scope). Usage:
#   MODEL=/path/file.gguf ARENA=40 MEM=54g ./mab776-run.sh [extra ds4_generate args]
# MODEL's directory is mounted read-only at /m. Prints to stdout.
cd "$(dirname "$0")" || exit 1
M="${MODEL:?set MODEL}"; D=$(dirname "$M"); F=$(basename "$M")
exec docker run --rm --name ds4-run --gpus '"device=0"' --ipc host --memory "${MEM:-54g}" --memory-swap "${MEM:-54g}" \
  --ulimit memlock=-1 -v "$PWD":/src:ro -v "$D":/m:ro -w /src nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  build-ds4-gpu/ds4_generate -m "/m/$F" --slots "${SLOTS:-auto}" --pcie 0.55 --vram-lru --pf-b 0.7 \
  --arena-skip-resident --arena-gib "${ARENA:-40}" \
  --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin "$@"
