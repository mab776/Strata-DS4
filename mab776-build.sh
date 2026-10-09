#!/bin/sh
# mab776: build Strata-DS4's ds4_generate (+ header-only geometry test) for sm_120 in a throwaway CUDA 12.8 container,
# WITHOUT a GPU, RAM/CPU-capped so nvcc can't squeeze whatever is serving. Same flags as the README's build-ds4-gpu,
# arch 89 -> 120. ggml = the pinned commit (third_party/ggml/VERSION.txt), fetched by CMake.  Log: build-mab776.log
cd "$(dirname "$0")" || exit 1
docker run --rm --name ds4-build --memory 12g --memory-swap 12g --cpus 8 \
  -v "$PWD":/src -w /src ${CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04} bash -euc "
    trap 'chown -R $(id -u):$(id -g) /src/build-ds4-gpu /src/third_party 2>/dev/null' EXIT
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3 python3-venv git curl g++ >/dev/null
    python3 -m venv /tmp/v && /tmp/v/bin/pip install -q cmake ninja
    export PATH=/tmp/v/bin:\$PATH
    cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    nice -n 19 cmake --build build-ds4-gpu --target ds4_generate ds4_geometry_test -j 4
    echo BUILD_DONE
  " > build-mab776.log 2>&1
