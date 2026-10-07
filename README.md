# ik_llama.cpp Build Recipe

Build scripts for [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp), compiled for exactly one machine: this one. Out the other end come an Arch package and two Docker images, all from a single upstream clone.

Nothing here is portable, and that is the entire point. If you want binaries that run anywhere, upstream ships those and they are lovely. These are not those.

## Targets

Two hard constraints are baked in. They are constraints, not achievements:

| | Target | Consequence |
| --- | --- | --- |
| **CPU** | `znver5` (Zen 5) | `SIGILL` on anything older. Not "slower" — *dead*. |
| **GPU** | CUDA SM `86` / `89` / `90` (Ampere, Ada, Hopper) | No kernels for your Pascal card. It will not fall back. It will simply not work. |

Note that "Zen 5" is an **architecture**, not a shopping list. Any Zen 5 part will do. The binary neither knows nor cares which one you bought, and `-march=znver5` has never once been impressed by a model number.

## What the flags actually buy you

The build turns on `-O3 -march=znver5 -mtune=znver5`, the full AVX-512 spread (`VBMI`, `VNNI`, `BF16`), and LTO. All of it applies to **both** image variants — the CPU image is exactly as architecture-locked as the CUDA one, just smaller and worse at matrix multiplication.

Now the part every other README leaves out: **this is worth a few percent, and only in the right places.**

Token generation on CPU is memory-bandwidth-bound. Your cores spend most of their day waiting on RAM, and a compiler flag has no opinion about DDR5 latency. Prompt processing is compute-bound and genuinely enjoys AVX-512, so you will see it there. Everything else is rounding error.

So: real gains, narrow scope, no magic. Compiling `-march=native` has never rescued a machine from its own memory subsystem, and it will not start today. Zen 5 is a perfectly nice place to be — this just declines to leave anything on the table while you are there.

## Building

`just` is the front door. There are two variants, cut from
[`docker/Dockerfile`](docker/Dockerfile) by a `VARIANT` build arg:

| Tag | Runtime base | Unpacked / pull |
| --- | --- | --- |
| `:cuda` | `nvidia/cuda:13.3.0-runtime-ubuntu24.04` | 4.03 GB / 2.94 GB |
| `:cpu` | `ubuntu:24.04` | 0.17 GB / 0.06 GB |

```bash
just            # list recipes
just check      # has upstream moved since each image was built?
just docker cpu        # build ghcr.io/delfianto/ik_llama.cpp:cpu
just docker cuda       # build ghcr.io/delfianto/ik_llama.cpp:cuda
just docker        # both
just pkg        # the Arch package, via makepkg
just pkg install
just patch-check # does the optional FastMTP patch still apply upstream?
just doctor      # check the local toolchain and package metadata
just verify      # doctor + patch applicability + Dockerfile checks
just docker compose up -d
just docker push # push both existing local image variants without rebuilding
```

Everything generated or downloaded stays under `.cache/`: `ik_llama.cpp` is the
shared upstream mirror, `build/` is makepkg's work area, `docker/` is Docker's
materialized source context, and `pkg/` holds finished Arch packages. `just clean`
removes Docker temporaries; `just clobber` removes all build and package output
while retaining the upstream mirror.

### Repository layout

```text
docker/     Dockerfile, Bake definition, entrypoint, Compose example
arch/       PKGBUILD, .SRCINFO, service and default configuration
patches/    shared optional patches and their maintenance notes
justfile    the only build and maintenance entry point
```

BuildKit only builds stages the chosen variant can actually reach, so `just docker cpu` never drags down the CUDA base images. The 24x size gap between the two is almost entirely NVIDIA's runtime plus **1.37 GB of statically-linked device code** — of which `llama-cli` is a ~678 MB near-duplicate of `llama-server`, carried purely so you can poke at things inside the container. Worth it, probably. Nobody has ever checked.

> `docker images` will insist these are 6.97 GB and 231 MB. It is lying — or rather, containerd's image store is counting the compressed blobs *and* the unpacked snapshots and adding them together, which is a bold interpretation of the word "size". The table above is real.

### There is no CI, and that is deliberate

A Zen 5 + LTO + CUDA build on a shared GitHub runner took the better part of forever and produced binaries **the runner itself could not execute**. Paying a stranger's CPU to emit instructions it cannot run, slowly, is a form of performance art this repository has retired. Everything builds on the machine that runs it. A cold `:cuda` build is ~20–25 minutes, and the LTO link is most of the tail.

### One clone, shared with makepkg

`.cache/ik_llama.cpp` is a bare mirror of upstream and the single source
of truth. It is the same clone `makepkg` populates from the PKGBUILD's
`source=git+...`, so `just pkg` and the Docker builds do not each fetch their own
copy.

Each image build runs `git archive <sha>` from that mirror into
`.cache/docker/src`, handed to the Dockerfile as a named context. No network,
no re-clone, no `.git` in the image. `git archive` is byte-deterministic — mtimes
come from the commit — so an unchanged commit re-extracts identically and the
`COPY` layer still cache-hits.

Because the source arrives as a named context, a bare `docker build .` no longer works. Use `just`.

### Tracking upstream

Every image records its commit as an `org.opencontainers.image.revision` label, so it is self-describing and there is no stamp file waiting to quietly desynchronize. `just check` fetches the mirror and diffs upstream against that label:

```console
$ just check
upstream main @ 1fddd12ba861  (18 hours ago -- New op: ggml_sum_rows_ext (#2132))

  cpu:  up to date
  cuda: STALE at bbc7de47c1a2 (37 commits behind)
```

Builds skip themselves when the image already matches upstream. Override with `FORCE=1`, or pin a ref (branch, tag, or full SHA):

```bash
FORCE=1 just docker cuda
IK_LLAMA_REF=bbc7de47 just docker cuda
```

`ccache` is mounted per variant, so bumping upstream a few commits recompiles a small slice rather than the whole tree. The LTO link still runs in full every time, because LTO is like that.

### Publishing

Images are tagged into `ghcr.io/delfianto` but nothing pushes on its own:

```bash
just docker       # build both images first
just docker push  # push the existing local images
```

There is no `:latest`. With two variants it would have to mean one of them, and whichever it meant would be wrong half the time.

## Running it (OpenWebUI sidecar)

Drop-in for any OpenAI-compatible backend. The entrypoint reads:

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_PATH` | — | Required. |
| `MTP_MODEL_PATH` | unset | Loads an external MTP companion with `--model-draft`. |
| `MTP_DRAFT_N` | `3` | MTP lookahead depth (`n_max`). |
| `CTX_SIZE` | `8192` | |
| `N_GPU_LAYERS` | `-1` | Ignored by `:cpu`, which has no GPU to offload to. |
| `THREADS` | `nproc` | |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | |
| `EXTRA_ARGS` | unset | Passed through verbatim, word-split, no quoting. Be nice. |

```yaml
services:
  llama-server-sidecar:
    image: ghcr.io/delfianto/ik_llama.cpp:cuda
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    environment:
      - MODEL_PATH=/models/gemma-4-main-qat.gguf
      - MTP_MODEL_PATH=/models/gemma-4-draft-mtp.gguf
      - CTX_SIZE=8192
      - N_GPU_LAYERS=-1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

For `:cpu`, use the same block minus the `deploy:` section. See
[`docker/compose.example.yml`](docker/compose.example.yml) for a full
stack with OpenWebUI, or run it through `just docker compose up -d`.

Current `ik_llama.cpp` supports normal embedded Qwen3.5/3.8 MTP and standalone
full-vocabulary companions without a downstream patch. HauhauCS FastMTP is a
different compact companion: its output vocabulary is trimmed to 32K and a
`d2t` tensor maps those logits back to the target vocabulary. Upstream does not
consume that tensor yet, so the small compatibility patch remains experimental
and disabled in normal builds. To opt in, build with:

```bash
EXPERIMENTAL_FASTMTP=1 just docker cuda
# Arch package: EXPERIMENTAL_FASTMTP=1 just pkg
```

The flag defaults to `0`. `just check` records whether an image is patched and
rebuilds when the selected mode or patch changes.

The compatibility patch is adapted from the
[HauhauCS Qwen3.8 FastMTP release](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF).
With an opted-in build, pair any compatible target quant with its shared 32K
companion:

```yaml
environment:
  - MODEL_PATH=/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
  - MTP_MODEL_PATH=/models/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf
  - MTP_DRAFT_N=3
  - CTX_SIZE=204800
  - EXTRA_ARGS=--parallel 1 --split-mode none --no-mmap
```

One trap worth naming: on `:cuda` the weights live in VRAM, so the container's RAM footprint is small and a tight memory limit looks fine. Move that same model to `:cpu` and the weights land in host RAM — with `--mlock`, permanently — and the limit you never thought about becomes an OOM kill at load. Size it for the model, not for what the GPU was politely hiding from you.

## Arch package

```bash
just pkg
just pkg install
```

`just pkg install` installs the existing package from `.cache/pkg` when available,
or builds and installs it if missing. `just pkg` rebuilds the package.
Packages land in `.cache/pkg`; makepkg work trees stay in `.cache/build`.
The package shares the source mirror above, so it will not re-clone. Same
architecture constraints apply, which is to say: it will not run on your laptop,
your NAS, or that Xeon you were emotionally attached to in 2019.

The package uses the current `gcc` and `g++` from `PATH`. Pin another installed
toolchain when needed with `aur_llamacpp_cc=gcc-16 aur_llamacpp_cxx=g++-16 just
pkg`.

Because makepkg requires local sources beside its active PKGBUILD, `just pkg`
creates a symlink-only recipe in `.cache/build/recipe`. The symlink targets remain
the canonical files in `arch/` and `patches/`; Docker reads those same files
directly.

See [`patches/README.md`](patches/README.md) for the current patch audit and the MoE
cache work worth watching.
