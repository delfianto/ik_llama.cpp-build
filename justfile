# Local build orchestration for ik_llama.cpp.
#
# The bare mirror at .cache/ik_llama.cpp is the single source of truth and is shared
# with makepkg, so
# `just pkg` and the docker builds never clone twice.

mirror := ".cache/ik_llama.cpp"
upstream := "https://github.com/ikawrakow/ik_llama.cpp.git"
srcdir := ".cache/docker/src"
image := "ghcr.io/delfianto/ik_llama.cpp"
fastmtp_patch := "patches/fastmtp-qwen38-d2t.patch"
arch_dir := "arch"
arch_stage := ".cache/build/recipe"
bake_file := "docker/bake.hcl"
compose_file := "docker/compose.example.yml"
ref := env("IK_LLAMA_REF", "main")
force := env("FORCE", "0")
experimental_fastmtp := env("EXPERIMENTAL_FASTMTP", "0")

_default:
    @just --list --unsorted

# Update the local bare mirror from upstream (clones it if missing).
fetch:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "$(dirname "{{ mirror }}")"
    if [[ -d "{{ mirror }}" ]]; then
        git -C "{{ mirror }}" fetch --prune --quiet
    else
        echo "==> cloning mirror (one time, ~131MB)"
        git clone --mirror "{{ upstream }}" "{{ mirror }}"
    fi

# Report whether upstream has moved since each image was built.
check: fetch
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(git -C "{{ mirror }}" rev-parse "{{ ref }}")
    case "{{ experimental_fastmtp }}" in
        0) patch_sha=disabled ;;
        1) patch_sha=$(sha256sum "{{ fastmtp_patch }}" | cut -d' ' -f1) ;;
        *) echo "EXPERIMENTAL_FASTMTP must be 0 or 1" >&2; exit 2 ;;
    esac
    desc=$(git -C "{{ mirror }}" log -1 --format='%cr -- %s' "$sha")
    echo "upstream {{ ref }} @ ${sha:0:12}  ($desc)"
    echo
    stale=0
    for v in cpu cuda; do
        built=$(docker image inspect "{{ image }}:$v" \
            --format '{{{{ index .Config.Labels "org.opencontainers.image.revision" }}' 2>/dev/null || true)
        built_patch=$(docker image inspect "{{ image }}:$v" \
            --format '{{{{ index .Config.Labels "org.opencontainers.image.fastmtp-patch" }}' 2>/dev/null || true)
        if [[ -z "$built" ]]; then
            printf '  %-5s not built\n' "$v:"; stale=1
        elif [[ "$built" == "$sha" && "$built_patch" == "$patch_sha" ]]; then
            printf '  %-5s up to date\n' "$v:"
        elif [[ "$built" == "$sha" ]]; then
            printf '  %-5s FastMTP patch changed\n' "$v:"; stale=1
        else
            behind=$(git -C "{{ mirror }}" rev-list --count "$built..$sha" 2>/dev/null || echo '?')
            printf '  %-5s STALE at %s (%s commits behind)\n' "$v:" "${built:0:12}" "$behind"; stale=1
        fi
    done
    echo
    if (( stale )); then echo "run: just all"; else echo "nothing to do"; fi

# Extract a pristine worktree at the given commit into .cache/docker/src.
_materialize sha:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf "{{ srcdir }}"
    mkdir -p "{{ srcdir }}"
    # `git archive` is byte-deterministic (mtimes come from the commit), so an
    # unchanged SHA re-extracts identically and the COPY layer still cache-hits.
    git -C "{{ mirror }}" archive --format=tar "{{ sha }}" | tar -x -C "{{ srcdir }}"
    # Upstream's own .dockerignore excludes build*/ and **/*.md; it would be
    # applied to the named context and is none of our business.
    rm -f "{{ srcdir }}/.dockerignore"

# Build one variant, skipping if it already matches upstream (FORCE=1 to override).
_build variant: fetch
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(git -C "{{ mirror }}" rev-parse "{{ ref }}")
    case "{{ experimental_fastmtp }}" in
        0) patch_sha=disabled ;;
        1) patch_sha=$(sha256sum "{{ fastmtp_patch }}" | cut -d' ' -f1) ;;
        *) echo "EXPERIMENTAL_FASTMTP must be 0 or 1" >&2; exit 2 ;;
    esac
    built=$(docker image inspect "{{ image }}:{{ variant }}" \
        --format '{{{{ index .Config.Labels "org.opencontainers.image.revision" }}' 2>/dev/null || true)
    built_patch=$(docker image inspect "{{ image }}:{{ variant }}" \
        --format '{{{{ index .Config.Labels "org.opencontainers.image.fastmtp-patch" }}' 2>/dev/null || true)
    if [[ "$built" == "$sha" && "$built_patch" == "$patch_sha" && "{{ force }}" != "1" ]]; then
        echo "==> {{ variant }}: already at ${sha:0:12} -- skipping (FORCE=1 to rebuild)"
        exit 0
    fi
    num=$(git -C "{{ mirror }}" rev-list --count "$sha")
    just _materialize "$sha"
    echo "==> building {{ variant }} @ ${sha:0:12} (build number $num)"
    IK_LLAMA_SHA="$sha" IK_LLAMA_BUILD_NUMBER="$num" IK_LLAMA_PATCH_SHA="$patch_sha" \
        IK_LLAMA_EXPERIMENTAL_FASTMTP="{{ experimental_fastmtp }}" \
        docker buildx bake -f "{{ bake_file }}" "{{ variant }}"

# Build the CPU image.
cpu: (_build "cpu")

# Build the CUDA image.
cuda: (_build "cuda")

# Build both images.
all: cpu cuda

# Stage makepkg's local sources. makepkg only resolves them beside the PKGBUILD,
# so symlinks let Arch and Docker consume the one canonical patch in patches/.
_stage_arch:
    #!/usr/bin/env bash
    set -euo pipefail
    recipe_root="$PWD"
    mkdir -p "{{ arch_stage }}" .cache/pkg
    ln -sfn "$recipe_root/{{ arch_dir }}/PKGBUILD" "{{ arch_stage }}/PKGBUILD"
    ln -sfn "$recipe_root/{{ arch_dir }}/llama.cpp.conf" "{{ arch_stage }}/llama.cpp.conf"
    ln -sfn "$recipe_root/{{ arch_dir }}/llama.cpp.service" "{{ arch_stage }}/llama.cpp.service"
    ln -sfn "$recipe_root/{{ fastmtp_patch }}" "{{ arch_stage }}/fastmtp-qwen38-d2t.patch"

# Build the Arch package with makepkg (reuses the same mirror).
pkg: _stage_arch
    #!/usr/bin/env bash
    set -euo pipefail
    recipe_root="$PWD"
    BUILDDIR="$recipe_root/.cache/build" \
    PKGDEST="$recipe_root/.cache/pkg" \
    SRCDEST="$recipe_root/.cache" \
    EXPERIMENTAL_FASTMTP="{{ experimental_fastmtp }}" \
        makepkg -D "{{ arch_stage }}" -sf --noconfirm

# Build and install the Arch package.
pkg-install: _stage_arch
    #!/usr/bin/env bash
    set -euo pipefail
    recipe_root="$PWD"
    BUILDDIR="$recipe_root/.cache/build" \
    PKGDEST="$recipe_root/.cache/pkg" \
    SRCDEST="$recipe_root/.cache" \
    EXPERIMENTAL_FASTMTP="{{ experimental_fastmtp }}" \
        makepkg -D "{{ arch_stage }}" -sif --noconfirm

# Regenerate arch/.SRCINFO.
srcinfo:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ arch_dir }}"
    makepkg --printsrcinfo > .SRCINFO

# Validate the optional patch against the selected upstream ref.
patch-check: fetch
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(git -C "{{ mirror }}" rev-parse "{{ ref }}")
    mkdir -p .cache/docker
    work=$(mktemp -d .cache/docker/patch-check.XXXXXX)
    trap 'rm -rf "$work"' EXIT
    git -C "{{ mirror }}" archive "$sha" | tar -x -C "$work"
    git -C "$work" apply --check "$PWD/{{ fastmtp_patch }}"
    echo "patch applies to ${sha:0:12}"

# Start the example stack, forwarding arguments to Docker Compose.
compose *args:
    docker compose -f "{{ compose_file }}" {{ args }}

# Push both locally built image variants.
push: all
    docker push "{{ image }}:cpu"
    docker push "{{ image }}:cuda"

# Check local tool availability and patch/package metadata.
doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    for tool in git just docker makepkg cmake; do
        command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
    done
    docker buildx version >/dev/null
    makepkg -D "{{ arch_dir }}" --printsrcinfo >/dev/null
    echo "build tools and package metadata: OK"

# Run metadata, patch, and Dockerfile validation without compiling.
verify: doctor patch-check
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(git -C "{{ mirror }}" rev-parse "{{ ref }}")
    just _materialize "$sha"
    IK_LLAMA_SHA="$sha" \
    IK_LLAMA_BUILD_NUMBER=$(git -C "{{ mirror }}" rev-list --count "$sha") \
    IK_LLAMA_PATCH_SHA=disabled \
        docker buildx bake -f "{{ bake_file }}" --call=check cpu cuda

# Remove disposable materialised source and patch-check trees (keeps the mirror).
clean:
    rm -rf .cache/docker

# Remove all generated build/package output, but retain downloaded sources.
clobber:
    rm -rf .cache/build .cache/docker .cache/pkg
