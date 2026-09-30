# syntax=docker/dockerfile:1

# Two image variants are produced from this single file, selected by VARIANT:
#   cpu  -> plain Ubuntu, no CUDA anywhere
#   cuda -> nvidia/cuda + NCCL, targeting Ampere/Ada/Hopper
# BuildKit only materialises the base stages that the selected VARIANT actually
# references, so a cpu build never pulls the multi-gigabyte CUDA images.
ARG VARIANT=cuda
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.3.0

# ---------------------------------------------------------------------------
# Build bases. Each variant pins its own toolchain image, installs the deps
# only it needs, and declares the CUDA-related cmake knobs as ENV so that the
# builder stage below can share one cmake invocation between both variants.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS base-cpu
ENV OPT_CUDA=OFF \
    OPT_NCCL=OFF \
    OPT_FA_ALL_QUANTS=OFF \
    OPT_CUDA_ARCHS=""

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base-cuda
ENV OPT_CUDA=ON \
    OPT_NCCL=ON \
    OPT_FA_ALL_QUANTS=ON \
    OPT_CUDA_ARCHS="86;89;90"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Builder. Identical for both variants apart from the OPT_* values inherited
# from the base stage above.
# ---------------------------------------------------------------------------
FROM base-${VARIANT} AS builder

ARG VARIANT
ENV DEBIAN_FRONTEND=noninteractive

# gcc-14 is the floor for -march=znver5; Ubuntu 24.04 ships gcc-13 by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    g++-14 \
    gcc-14 \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Source arrives as a pristine `git archive` tree from the local bare mirror via
# the `ikllama` named context -- no clone, no network. `just` materialises it;
# see docker/bake.hcl. Kept after the apt layer so source churn doesn't refetch
# the toolchain.
COPY --from=ikllama . /app
COPY patches/fastmtp-qwen38-d2t.patch /tmp/fastmtp-qwen38-d2t.patch

ARG IK_LLAMA_EXPERIMENTAL_FASTMTP=0
RUN case "${IK_LLAMA_EXPERIMENTAL_FASTMTP}" in \
      0) echo "Qwen3.8 FastMTP patch disabled" ;; \
      1) git apply --check /tmp/fastmtp-qwen38-d2t.patch && \
         git apply /tmp/fastmtp-qwen38-d2t.patch ;; \
      *) echo "IK_LLAMA_EXPERIMENTAL_FASTMTP must be 0 or 1" >&2; exit 2 ;; \
    esac

ARG IK_LLAMA_SHA=unknown
ARG IK_LLAMA_BUILD_NUMBER=0

# `git archive` strips .git, so upstream's cmake/build-info.cmake finds no repo
# and keeps its `0` / "unknown" defaults -- and those are plain set() calls that
# shadow the cache, so no -D flag can override them (which is why the PKGBUILD's
# -DLLAMA_BUILD_NUMBER has never done anything). Patch the defaults instead.
# common/cmake/build-info-gen-cpp.cmake re-includes this same file at build time,
# so this covers the regeneration path too.
RUN short="$(echo "${IK_LLAMA_SHA}" | cut -c1-7)" && \
    sed -i \
      -e "s/^set(BUILD_NUMBER 0)/set(BUILD_NUMBER ${IK_LLAMA_BUILD_NUMBER})/" \
      -e "s/^set(BUILD_COMMIT \"unknown\")/set(BUILD_COMMIT \"${short}\")/" \
      cmake/build-info.cmake && \
    grep -q "^set(BUILD_NUMBER ${IK_LLAMA_BUILD_NUMBER})$" cmake/build-info.cmake && \
    grep -q "^set(BUILD_COMMIT \"${short}\")$" cmake/build-info.cmake

# Default CUDA_CCVER when the host-compiler version probe returns empty.
RUN sed -i 's/get_flags(${CUDA_CCID} ${CUDA_CCVER})/if(CUDA_CCVER)\nget_flags(${CUDA_CCID} ${CUDA_CCVER})\nelse()\nget_flags(${CUDA_CCID} 13.0.0)\nendif()/g' ggml/src/CMakeLists.txt

RUN cmake -B build -G Ninja \
    -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_LTO=ON \
    -DGGML_RPC=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_OPENSSL=ON \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_AVX512_BF16=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_C_FLAGS="-O3 -march=znver5 -mtune=znver5" \
    -DCMAKE_CXX_FLAGS="-O3 -march=znver5 -mtune=znver5" \
    -DGGML_CUDA="${OPT_CUDA}" \
    -DGGML_NCCL="${OPT_NCCL}" \
    -DGGML_CUDA_FA_ALL_QUANTS="${OPT_FA_ALL_QUANTS}" \
    -DCMAKE_CUDA_ARCHITECTURES="${OPT_CUDA_ARCHS}"

# Per-variant ccache: the two variants share no object hashes, and separate IDs
# keep parallel `bake` runs from serialising on one lock.
RUN --mount=type=cache,id=ccache-${VARIANT},target=/ccache \
    CCACHE_DIR=/ccache cmake --build build --config Release -j "$(nproc)"

# ---------------------------------------------------------------------------
# Runtime bases, again one per variant.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime-cpu

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl2 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Final image.
# ---------------------------------------------------------------------------
FROM runtime-${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libcurl4 \
    libgomp1 \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/bin/llama-server /usr/local/bin/
# Also copy llama-cli in case manual debugging is needed inside the container
COPY --from=builder /app/build/bin/llama-cli /usr/local/bin/

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
