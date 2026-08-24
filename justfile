# Local build orchestration for ik_llama.cpp.
#
# The bare mirror at ./ik_llama.cpp is the single source of truth and is shared
# with makepkg (PKGBUILD's `source=git+...` populates the same directory), so
# `just pkg` and the docker builds never clone twice.

mirror := "ik_llama.cpp"
upstream := "https://github.com/ikawrakow/ik_llama.cpp.git"
srcdir := ".build/src"
image := "ghcr.io/delfianto/ik_llama.cpp"
fastmtp_patch := "fastmtp-qwen35.patch"
ref := env("IK_LLAMA_REF", "main")
force := env("FORCE", "0")
experimental_fastmtp := env("EXPERIMENTAL_FASTMTP", "0")

_default:
    @just --list --unsorted

# Update the local bare mirror from upstream (clones it if missing).
fetch:
    #!/usr/bin/env bash
    set -euo pipefail
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

# Extract a pristine worktree at the given commit into .build/src.
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
        IK_LLAMA_EXPERIMENTAL_FASTMTP="{{ experimental_fastmtp }}" docker buildx bake "{{ variant }}"

# Build the CPU image.
cpu: (_build "cpu")

# Build the CUDA image.
cuda: (_build "cuda")

# Build both images.
all: cpu cuda

# Build the Arch package with makepkg (reuses the same mirror).
pkg:
    EXPERIMENTAL_FASTMTP="{{ experimental_fastmtp }}" makepkg -sf --noconfirm

# Remove the materialised source tree and build cache (keeps the mirror).
clean:
    rm -rf "{{ srcdir }}"
