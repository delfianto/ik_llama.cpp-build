# Local build orchestration for ik_llama.cpp.
#
# The shallow bare clone at .cache/ik_llama.cpp is the single source of truth and is shared
# with makepkg, so
# `just pkg` and the docker builds never clone twice.

mirror := ".cache/ik_llama.cpp"
source_url := "https://github.com/delfianto/ik_llama.cpp.git"
source_page := "https://github.com/delfianto/ik_llama.cpp"
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

# Check the experiment Python scripts with the locked development tools.
check-python:
    uv run --locked ruff check experiments
    uv run --locked ruff format --check experiments
    uv run --locked basedpyright

# Fetch only the selected fork branch, replacing older full-history caches.
fetch:
    #!/usr/bin/env bash
    set -euo pipefail
    source_ref={{ quote(ref) }}
    git check-ref-format --branch "$source_ref" >/dev/null
    mkdir -p "$(dirname "{{ mirror }}")"
    if [[ ! -d "{{ mirror }}" ]] || [[ $(git -C "{{ mirror }}" rev-parse --is-shallow-repository) != true ]]; then
        work=$(mktemp -d "{{ mirror }}.shallow.XXXXXX")
        trap 'rm -rf "$work"' EXIT
        git clone --bare --depth=1 --single-branch --branch "$source_ref" --no-tags "{{ source_url }}" "$work"
        rm -rf "{{ mirror }}"
        mv "$work" "{{ mirror }}"
    fi
    git -C "{{ mirror }}" remote set-url origin "{{ source_url }}"
    git -C "{{ mirror }}" config remote.origin.fetch "+refs/heads/$source_ref:refs/heads/$source_ref"
    git -C "{{ mirror }}" config remote.origin.tagOpt --no-tags
    git -C "{{ mirror }}" fetch --depth=1 --no-tags --prune origin "+refs/heads/$source_ref:refs/heads/$source_ref"
    git -C "{{ mirror }}" symbolic-ref HEAD "refs/heads/$source_ref"
    while read -r name; do
        [[ "$name" == "refs/heads/$source_ref" ]] || git -C "{{ mirror }}" update-ref -d "$name"
    done < <(git -C "{{ mirror }}" for-each-ref --format='%(refname)')
    git -C "{{ mirror }}" reflog expire --expire=now --all
    git -C "{{ mirror }}" gc --prune=now --quiet

# Report whether the selected fork source differs from each image.
check: fetch
    #!/usr/bin/env bash
    set -euo pipefail
    source_ref={{ quote(ref) }}
    sha=$(git -C "{{ mirror }}" rev-parse "$source_ref")
    case "{{ experimental_fastmtp }}" in
        0) patch_sha=disabled ;;
        1) patch_sha=$(sha256sum "{{ fastmtp_patch }}" | cut -d' ' -f1) ;;
        *) echo "EXPERIMENTAL_FASTMTP must be 0 or 1" >&2; exit 2 ;;
    esac
    desc=$(git -C "{{ mirror }}" log -1 --format='%cr -- %s' "$sha")
    echo "fork $source_ref @ ${sha:0:12}  ($desc)"
    echo
    stale=0
    for v in cpu cuda; do
        built=$(docker image inspect "{{ image }}:$v" \
            --format '{{{{ index .Config.Labels "org.opencontainers.image.revision" }}' 2>/dev/null || true)
        built_patch=$(docker image inspect "{{ image }}:$v" \
            --format '{{{{ index .Config.Labels "org.opencontainers.image.fastmtp-patch" }}' 2>/dev/null || true)
        built_source=$(docker image inspect "{{ image }}:$v" \
            --format '{{{{ index .Config.Labels "org.opencontainers.image.source" }}' 2>/dev/null || true)
        if [[ -z "$built" ]]; then
            printf '  %-5s not built\n' "$v:"; stale=1
        elif [[ "$built" == "$sha" && "$built_patch" == "$patch_sha" && "$built_source" == "{{ source_page }}" ]]; then
            printf '  %-5s up to date\n' "$v:"
        elif [[ "$built_source" != "{{ source_page }}" ]]; then
            printf '  %-5s source repository changed\n' "$v:"; stale=1
        elif [[ "$built" == "$sha" ]]; then
            printf '  %-5s FastMTP patch changed\n' "$v:"; stale=1
        else
            printf '  %-5s STALE at %s\n' "$v:" "${built:0:12}"; stale=1
        fi
    done
    echo
    if (( stale )); then echo "run: just docker"; else echo "nothing to do"; fi

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

# Build one variant, skipping if source and patch match (FORCE=1 to override).
_build variant: fetch
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(git -C "{{ mirror }}" rev-parse {{ quote(ref) }})
    case "{{ experimental_fastmtp }}" in
        0) patch_sha=disabled ;;
        1) patch_sha=$(sha256sum "{{ fastmtp_patch }}" | cut -d' ' -f1) ;;
        *) echo "EXPERIMENTAL_FASTMTP must be 0 or 1" >&2; exit 2 ;;
    esac
    built=$(docker image inspect "{{ image }}:{{ variant }}" \
        --format '{{{{ index .Config.Labels "org.opencontainers.image.revision" }}' 2>/dev/null || true)
    built_patch=$(docker image inspect "{{ image }}:{{ variant }}" \
        --format '{{{{ index .Config.Labels "org.opencontainers.image.fastmtp-patch" }}' 2>/dev/null || true)
    built_source=$(docker image inspect "{{ image }}:{{ variant }}" \
        --format '{{{{ index .Config.Labels "org.opencontainers.image.source" }}' 2>/dev/null || true)
    if [[ "$built" == "$sha" && "$built_patch" == "$patch_sha" && "$built_source" == "{{ source_page }}" && "{{ force }}" != "1" ]]; then
        echo "==> {{ variant }}: already at ${sha:0:12} -- skipping (FORCE=1 to rebuild)"
        exit 0
    fi
    num=$(git -C "{{ mirror }}" show -s --format=%ct "$sha")
    just _materialize "$sha"
    echo "==> building {{ variant }} @ ${sha:0:12} (build number $num)"
    IK_LLAMA_SHA="$sha" IK_LLAMA_BUILD_NUMBER="$num" IK_LLAMA_PATCH_SHA="$patch_sha" \
        IK_LLAMA_EXPERIMENTAL_FASTMTP="{{ experimental_fastmtp }}" \
        docker buildx bake -f "{{ bake_file }}" "{{ variant }}"

# Build Docker images, push local images, or forward arguments to Compose.
[positional-arguments]
docker variant="all" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    shift
    if [[ {{ quote(variant) }} != compose && $# -gt 0 ]]; then
        echo "Only just docker compose accepts additional arguments" >&2
        exit 2
    fi
    case {{ quote(variant) }} in
        all) just _build cpu; just _build cuda ;;
        cpu|cuda) just _build {{ quote(variant) }} ;;
        push)
            docker push "{{ image }}:cpu"
            docker push "{{ image }}:cuda"
            ;;
        compose) docker compose -f "{{ compose_file }}" "$@" ;;
        *) echo "Usage: just docker [cpu|cuda|push|compose [args...]]" >&2; exit 2 ;;
    esac

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

# Export the selected branch for makepkg without its full-history Git source handler.
_archive_arch: fetch _stage_arch
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(git -C "{{ mirror }}" rev-parse {{ quote(ref) }})
    git -C "{{ mirror }}" archive --prefix=ik_llama.cpp/ "$sha" > .cache/ik_llama.cpp.tar
    git -C "{{ mirror }}" show -s --format='%ct %h' "$sha" > .cache/source-version
    ln -sfn "$PWD/.cache/ik_llama.cpp.tar" "{{ arch_stage }}/ik_llama.cpp.tar"
    ln -sfn "$PWD/.cache/source-version" "{{ arch_stage }}/source-version"

# Build the Arch package, or install an existing package (builds if missing).
pkg action="build": fetch _stage_arch
    #!/usr/bin/env bash
    set -euo pipefail
    recipe_root="$PWD"
    case {{ quote(action) }} in
        build) flags=-sf ;;
        install) flags=-si ;;
        *) echo "Usage: just pkg [install]" >&2; exit 2 ;;
    esac
    export BUILDDIR="$recipe_root/.cache/build"
    export PKGDEST="$recipe_root/.cache/pkg"
    export SRCDEST="$recipe_root/.cache"
    export EXPERIMENTAL_FASTMTP="{{ experimental_fastmtp }}"
    if [[ {{ quote(action) }} == install ]]; then
        selected_version=$(git -C "{{ mirror }}" show -s --format='%ct.%h' {{ quote(ref) }})
        recipe_version=$(makepkg -D "{{ arch_stage }}" --printsrcinfo | awk '$1 == "pkgver" { print $3; exit }')
        package_list=$(makepkg -D "{{ arch_stage }}" --packagelist)
        mapfile -t packages <<< "$package_list"
        all_built=1
        for package in "${packages[@]}"; do
            [[ -f "$package" ]] || all_built=0
        done
        if (( all_built )) && [[ "$recipe_version" == "$selected_version" ]]; then
            sudo pacman -U --noconfirm "${packages[@]}"
            exit 0
        fi
    fi
    just _archive_arch
    makepkg -D "{{ arch_stage }}" "$flags" --noconfirm

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
    sha=$(git -C "{{ mirror }}" rev-parse {{ quote(ref) }})
    mkdir -p .cache/docker
    work=$(mktemp -d .cache/docker/patch-check.XXXXXX)
    trap 'rm -rf "$work"' EXIT
    git -C "{{ mirror }}" archive "$sha" | tar -x -C "$work"
    git -C "$work" apply --check "$PWD/{{ fastmtp_patch }}"
    echo "patch applies to ${sha:0:12}"

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
    sha=$(git -C "{{ mirror }}" rev-parse {{ quote(ref) }})
    just _materialize "$sha"
    IK_LLAMA_SHA="$sha" \
    IK_LLAMA_BUILD_NUMBER=$(git -C "{{ mirror }}" show -s --format=%ct "$sha") \
    IK_LLAMA_PATCH_SHA=disabled \
        docker buildx bake -f "{{ bake_file }}" --call=check cpu cuda

# Remove disposable materialised source and patch-check trees (keeps the mirror).
clean:
    rm -rf .cache/docker

# Remove all generated build/package output, but retain downloaded sources.
clobber:
    rm -rf .cache/build .cache/docker .cache/pkg
