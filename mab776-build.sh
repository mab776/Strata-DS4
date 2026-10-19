#!/bin/sh
# mab776 (main): the two engines have their own build scripts - ENGINE=ds4 (V4, stack 74: ds4_serve, ggml tree
# third_party/llama.cpp-ds4) or ENGINE=ds41 (V4.1, stack 78: ds41_generate, ggml tree third_party/llama.cpp).
cd "$(dirname "$0")" || exit 1
case "${ENGINE:-}" in
  ds4)  exec sh ./mab776-build-ds4.sh "$@" ;;
  ds41) exec sh ./mab776-build-ds41.sh "$@" ;;
  *) echo "mab776-build.sh: set ENGINE=ds4 (V4: ds4_serve) or ENGINE=ds41 (V4.1: ds41_generate)" >&2; exit 2 ;;
esac
