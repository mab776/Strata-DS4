#!/bin/sh
# tools/ds4/run_serve_gate.sh - (mab776 M5) build ds4_serve_cpu and run its gate (tools/ds4/test_ds4_serve.py) in
# throwaway containers, CPU only: no GPU is ever visible (NVIDIA_VISIBLE_DEVICES=none), RAM/CPU capped, low priority.
#
#   tools/ds4/run_serve_gate.sh build          CPU tree in build-ds4-cpu (ds4_serve_cpu, ds4_generate_cpu, test_ds4_sampler)
#   tools/ds4/run_serve_gate.sh test [Class]   the gate (all classes, or the named ones), then nothing else
#   tools/ds4/run_serve_gate.sh serve-suite    serve/'s own unittest suite (must stay green)
#
# LCPP  = a llama.cpp checkout of third_party/ggml/VERSION.txt's commit (ds4_dense.cmake needs ggml-impl.h; the fixtures
#         need its gguf-py), mounted read-only at third_party/llama.cpp.  Default: ./third_party/llama.cpp.
# TOOLS = a directory with cmake + ninja (`python3 -m pip install --target $TOOLS cmake ninja`, once): the build then needs
#         no network.  Unset: the build container installs them with pip (network).
# Images: CUDA_IMAGE (default nvidia/cuda:13.0.2-devel-ubuntu24.04, g++ 13) for the build; PYTEST_IMAGE (default
# strata-ds4-pytest:py312: Python 3.12 + numpy, jinja2, regex, psutil) for the tests.
set -eu
cd "$(dirname "$0")/../.."
WT="$PWD"
LCPP="${LCPP:-$WT/third_party/llama.cpp}"
CAPS="-e NVIDIA_VISIBLE_DEVICES=none --cpu-shares 64 -u $(id -u):$(id -g) -e HOME=/tmp"
# the test knobs reach the container when they are set here (binaries, fixtures, the mutation runs' spec overrides)
FWD="-e DS4_SERVE_BIN -e DS4_GEN_BIN -e DS4_SAMPLER_BIN -e DS4_FX -e DS4_TEST_SPEC_K -e DS4_TEST_SPEC_KEEP_ABOVE \
     -e DS4_TEST_SPEC_KEEP_RESET"
case "${1:-test}" in
build)
    if [ -n "${TOOLS:-}" ]; then
        NET="--network none"; TV="-v $TOOLS:/t:ro"; PRE='export PATH=/t/cmake/data/bin:/t/bin:$PATH'
    else
        NET=""; TV=""; PRE='python3 -m venv /tmp/v 2>/dev/null || (apt-get update -qq && apt-get install -y -qq python3-venv >/dev/null && python3 -m venv /tmp/v); /tmp/v/bin/pip install -q cmake ninja; export PATH=/tmp/v/bin:$PATH'
    fi
    # shellcheck disable=SC2086
    docker run --rm --name ds4-m5-build $NET $CAPS --memory 8g --memory-swap 8g --cpus 6 -v "$WT":/src \
        -v "$LCPP":/src/third_party/llama.cpp:ro $TV -w /src --entrypoint bash \
        "${CUDA_IMAGE:-nvidia/cuda:13.0.2-devel-ubuntu24.04}" -c "$PRE
        cmake -S . -B build-ds4-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
            -DSTRATA_GGML_DIR=/src/third_party/llama.cpp >/dev/null
        nice -n 19 cmake --build build-ds4-cpu -j 6 --target ds4_serve_cpu ds4_generate_cpu test_ds4_sampler"
    ;;
test)
    shift || true
    # shellcheck disable=SC2086
    docker run --rm --name ds4-m5-test --network none $CAPS $FWD --memory 4g --memory-swap 4g --cpus 4 -v "$WT":/src \
        -v "$LCPP":/src/third_party/llama.cpp:ro -w /src "${PYTEST_IMAGE:-strata-ds4-pytest:py312}" \
        python3 -W ignore::ResourceWarning tools/ds4/test_ds4_serve.py -v "$@"
    ;;
serve-suite)
    # shellcheck disable=SC2086
    docker run --rm --name ds4-m5-test --network none $CAPS --memory 4g --memory-swap 4g --cpus 4 -v "$WT":/src -w /src \
        "${PYTEST_IMAGE:-strata-ds4-pytest:py312}" python3 -m unittest discover -s serve -t . -p 'test_*.py'
    ;;
*)
    echo "usage: $0 build | test [Class ...] | serve-suite" >&2
    exit 2
    ;;
esac
