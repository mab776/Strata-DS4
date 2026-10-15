#!/bin/sh
# mab776: build the upstream DeepSeek-V4.1 engine (ds41_generate) AS-IS for sm_120a (both cards: RTX Pro 5000 + 5070 Ti)
# in a throwaway CUDA 13 container, no GPU, RAM/CPU-capped. ggml = third_party/llama.cpp at the pinned commit
# (third_party/ggml/VERSION.txt) with tools/v41/ggml/*.patch applied once by hand (upstream-fixes, mmq-deterministic,
# rank-argsort, kv-fp4, dsv41-fake-quant, dsv41-fuse - all apply cleanly). FQ_OP + HC_MIX on = upstream's opt-in +5 %.
cd "$(dirname "$0")" || exit 1
docker run --rm --name ds41-build -e NVIDIA_VISIBLE_DEVICES=none --memory ${BUILD_MEM:-24g} --memory-swap ${BUILD_MEM:-24g} --cpus ${BUILD_CPUS:-14} \
  -v "$PWD":/src -w /src ${CUDA_IMAGE:-nvidia/cuda:13.0.2-devel-ubuntu24.04} bash -euc "
    trap 'chown -R $(id -u):$(id -g) /src/${BUILD_DIR:-build-ds4-gpu} /src/third_party 2>/dev/null' EXIT
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3 python3-venv git curl g++ >/dev/null
    python3 -m venv /tmp/v && /tmp/v/bin/pip install -q cmake ninja
    export PATH=/tmp/v/bin:\$PATH
    cmake -S . -B ${BUILD_DIR:-build-ds4-gpu} -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
      -DSTRATA_DSV41_FQ_OP=ON -DSTRATA_DSV41_HC_MIX=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    nice -n 19 cmake --build ${BUILD_DIR:-build-ds4-gpu} --target ${BUILD_TARGETS:-ds41_generate test_ds41_fq test_ds41_hcmix test_ds41_moe_gpu} -j ${BUILD_J:-12}
    echo BUILD_DONE
  " > ${BUILD_LOG:-build-mab776.log} 2>&1
