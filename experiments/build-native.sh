#!/usr/bin/env bash
# Persistent native development build; source checkout is deliberately caller-owned.
set -euo pipefail
source_dir=${1:?usage: build-native.sh SOURCE BUILD [cpu|cuda]}
build_dir=${2:?usage: build-native.sh SOURCE BUILD [cpu|cuda]}
variant=${3:-cuda}
flags=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=ON)
if command -v ccache >/dev/null; then
    flags+=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
fi
case "$variant" in
    cuda) flags+=(-DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=86;89' -DGGML_CUDA_FA_ALL_QUANTS=OFF) ;;
    cpu) flags+=(-DGGML_CUDA=OFF) ;;
    *) echo 'variant must be cpu or cuda' >&2; exit 2 ;;
esac
cmake -S "$source_dir" -B "$build_dir" "${flags[@]}"
cmake --build "$build_dir" --target llama-server llama-bench test-moe-cache test-moe-cache-model -j "${BUILD_JOBS:-8}"
