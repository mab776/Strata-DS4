#!/bin/sh
# mab776: build Strata-DS4's ds4_generate (+ header-only geometry test) for sm_120 in a throwaway CUDA 12.8 container,
# WITHOUT a GPU, RAM/CPU-capped so nvcc can't squeeze whatever is serving. Same flags as the README's build-ds4-gpu,
# arch 89 -> 120a (plain 120 can't assemble the MXFP4 block-scale MMA). BUILD_TARGETS overrides the target list, BUILD_DIR the build dir (CUDA 13: BUILD_DIR=build-ds4-cu13 CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04). ggml = full llama.cpp checkout of the pinned commit (third_party/ggml/VERSION.txt) in third_party/llama.cpp.  Log: build-mab776.log
cd "$(dirname "$0")" || exit 1
docker run --rm --name ds4-build -e NVIDIA_VISIBLE_DEVICES=none --memory ${BUILD_MEM:-12g} --memory-swap ${BUILD_MEM:-12g} --cpus ${BUILD_CPUS:-8} \
  -v "$PWD":/src -w /src ${CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04} bash -euc "
    trap 'chown -R $(id -u):$(id -g) /src/${BUILD_DIR:-build-ds4-gpu} /src/third_party 2>/dev/null' EXIT
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3 python3-venv git curl g++ >/dev/null
    python3 -m venv /tmp/v && /tmp/v/bin/pip install -q cmake ninja
    export PATH=/tmp/v/bin:\$PATH
    COMMIT=\$(cut -d' ' -f1 third_party/ggml/VERSION.txt)
    if [ ! -d third_party/llama.cpp ]; then   # full checkout: ds4_dense needs ggml-impl.h (README passes STRATA_GGML_DIR too)
      curl -sL https://github.com/ggml-org/llama.cpp/archive/\$COMMIT.tar.gz | tar -xz -C third_party
      mv third_party/llama.cpp-\$COMMIT third_party/llama.cpp
    fi
    cmake -S . -B ${BUILD_DIR:-build-ds4-gpu} -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    nice -n 19 cmake --build ${BUILD_DIR:-build-ds4-gpu} --target ${BUILD_TARGETS:-ds4_generate ds4_geometry_test test_ds4_moe} -j ${BUILD_J:-4}
    echo BUILD_DONE
  " > ${BUILD_LOG:-build-mab776.log} 2>&1
