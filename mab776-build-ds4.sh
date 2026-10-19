#!/bin/sh
# mab776 (main, 2026-10-19): the V4 engine (stack 74: ds4_serve). Its ggml tree is third_party/llama.cpp-ds4 (GGML_TREE),
# mounted over /src/third_party/llama.cpp so the compile paths - and the binary - are what they were on mab776-main;
# the V4.1 engine (mab776-build-ds41.sh) uses third_party/llama.cpp. Defaults = the live build: CUDA 13, build-ds4-cu13.
# mab776: build Strata-DS4's ds4_generate (+ header-only geometry test) for sm_120 in a throwaway CUDA 12.8 container,
# WITHOUT a GPU, RAM/CPU-capped so nvcc can't squeeze whatever is serving. Same flags as the README's build-ds4-gpu,
# arch 89 -> 120a (plain 120 can't assemble the MXFP4 block-scale MMA). BUILD_TARGETS overrides the target list, BUILD_DIR the build dir (CUDA 13: BUILD_DIR=build-ds4-cu13 CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04). ggml = full llama.cpp checkout of the pinned commit (third_party/ggml/VERSION.txt) in third_party/llama.cpp, plus our patches in tools/ds4/patches (applied once, idempotent).  Log: build-mab776.log
cd "$(dirname "$0")" || exit 1
GT=${GGML_TREE:-third_party/llama.cpp-ds4}; mkdir -p "$GT"
docker run --rm --name ds4-build -e NVIDIA_VISIBLE_DEVICES=none --memory ${BUILD_MEM:-12g} --memory-swap ${BUILD_MEM:-12g} --cpus ${BUILD_CPUS:-8} \
  -v "$PWD":/src -v "$PWD/$GT":/src/third_party/llama.cpp -w /src ${CUDA_IMAGE:-nvidia/cuda:13.0.2-devel-ubuntu24.04} bash -euc "
    trap 'chown -R $(id -u):$(id -g) /src/${BUILD_DIR:-build-ds4-cu13} /src/third_party 2>/dev/null' EXIT
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3 python3-venv git curl g++ >/dev/null
    python3 -m venv /tmp/v && /tmp/v/bin/pip install -q cmake ninja
    export PATH=/tmp/v/bin:\$PATH
    COMMIT=\$(cut -d' ' -f1 third_party/ggml/VERSION.txt)
    if [ ! -f third_party/llama.cpp/CMakeLists.txt ]; then   # full checkout: ds4_dense needs ggml-impl.h (README passes STRATA_GGML_DIR too)
      curl -sL https://github.com/ggml-org/llama.cpp/archive/\$COMMIT.tar.gz | tar -xz -C /tmp
      cp -a /tmp/llama.cpp-\$COMMIT/. third_party/llama.cpp/
    fi
    # mab776: the V4 ggml patches, applied once in this order: V4's own (tools/ds4/patches: the lightning indexer, the
    # hyper-connection ops) + the ones it shares with the V4.1 tree (tools/v41/ggml: top-k radix select - V4 sets
    # GGML_TOPK_RADIX=1 -, the F16 vector kernel for verify passes, upstream's rank-argsort and llama.cpp fixes)
    for p in tools/ds4/patches/ggml-lightning-indexer-count-mask.patch tools/ds4/patches/ggml-lightning-indexer-mma.patch \
             tools/v41/ggml/mab776-topk-radix-select.patch tools/ds4/patches/ggml-zz-dsv4-hc-gates.patch \
             tools/v41/ggml/mab776-mmvf-f16-batch.patch tools/ds4/patches/ggml-zzz-dsv4-hc-fused.patch \
             tools/v41/ggml/rank-argsort.patch tools/v41/ggml/upstream-fixes.patch; do
      [ -f \"\$p\" ] || continue
      if (cd third_party/llama.cpp && git apply --check -R /src/\$p 2>/dev/null); then echo \"ggml patch already in: \$p\";
      elif (cd third_party/llama.cpp && git apply --check /src/\$p 2>/dev/null); then
        (cd third_party/llama.cpp && git apply /src/\$p) && echo \"ggml patch applied: \$p\";
      else echo \"ggml patch neither applies nor reverts alone (a later patch edits next to it - in already?): \$p\"; fi
    done
    cmake -S . -B ${BUILD_DIR:-build-ds4-cu13} -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    nice -n 19 cmake --build ${BUILD_DIR:-build-ds4-cu13} --target ${BUILD_TARGETS:-ds4_serve ds4_generate} -j ${BUILD_J:-4}
    echo BUILD_DONE
  " > ${BUILD_LOG:-build-mab776.log} 2>&1
