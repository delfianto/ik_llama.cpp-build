# Upstream and patch watch

Last audited: 2026-10-01 against `ik_llama.cpp` main at `32cddbfcefed`.

## Qwen3.8 FastMTP

The old patch mixed two separate concerns:

1. Loading a predictor-only Qwen3.5-family GGUF as an external MTP companion.
2. Expanding HauhauCS's 32K draft vocabulary into the full target vocabulary
   using its `d2t` tensor.

The first part is upstream now. [ik_llama.cpp PR #2328](https://github.com/ikawrakow/ik_llama.cpp/pull/2328)
added predictor-only Qwen3.5 companion loading, package classification, architecture
checks, and the missing-layer handling. Current main also contains Qwen3.8 Next
runtime and MTP work in [PR #2369](https://github.com/ikawrakow/ik_llama.cpp/pull/2369)
and nearby follow-ups.

The second part is still model specific. The
[HauhauCS model card](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)
states that its compact FastMTP companion has a 32K output and a `d2t` map. Current
`ik_llama.cpp` main does not load or consume `d2t`, so an unpatched binary expects
the full 248,320-token output tensor.

`fastmtp-qwen38-d2t.patch` contains only that remaining delta, and both Docker
and Arch builds consume the same file. It is
optional because ordinary embedded MTP and ordinary full-vocabulary companion
models work without it. Run `just patch-check` after every upstream update. Remove
the patch and the `EXPERIMENTAL_FASTMTP` switch once `d2t` appears in upstream's
Qwen3.5 loader and graph.

## MoE work already in ik_llama.cpp

Before carrying another patch, test the features already merged upstream:

- [`--prefetch-experts` (PR #2101)](https://github.com/ikawrakow/ik_llama.cpp/pull/2101)
  asynchronously faults selected mmap-backed experts into the host page cache.
  This reduces storage/page-fault stalls; it is not a VRAM expert cache.
- [`--offload-only-active-experts` (PR #698)](https://github.com/ikawrakow/ik_llama.cpp/pull/698)
  avoids copying inactive experts for suitable hybrid workloads.
- [`Chunked experts` (PR #2202)](https://github.com/ikawrakow/ik_llama.cpp/pull/2202)
  improves CPU expert execution and is already on main.
- [`GGML_CUDA_NO_PINNED_WEIGHTS` (PR #2444)](https://github.com/ikawrakow/ik_llama.cpp/pull/2444)
  keeps CPU-resident expert weights mmap-backed while preserving pinned staging
  buffers, which can materially reduce host memory pressure.

These are safer tuning targets because upstream owns their interactions with IQK,
fused MoE ops, graph split, MTP, and hot reload.

## VRAM expert cache: promising, not ready to vendor

There is still no merged VRAM hot-expert cache in `ik_llama.cpp`. Two current
`llama.cpp` experiments are worth watching:

- [Adaptive hot-expert cache RFC #24528](https://github.com/ggml-org/llama.cpp/discussions/24528)
  reports roughly 30% uplift in controlled DeepSeek V4 tests and larger gains in
  some user tests. Its implementation is CUDA-heavy, and its original PR was
  closed for review scope. Multi-GPU behavior and warm-up policy remain active
  concerns.
- [Persistent expert pool RFC #28248](https://github.com/ggml-org/llama.cpp/discussions/28248)
  uses existing ops and a slot map, bypasses the pool for large prefill batches,
  and reports up to 84% synthetic decode uplift on a 4090. The current proposal
  still has device-placement limitations and unresolved independent test reports.

Do not drop either branch into this recipe as a long-lived patch yet. `ik_llama.cpp`
has diverged substantially in its scheduler and adds fused MoE, IQK, graph split,
and selective expert offload paths. A useful port needs correctness tests across
CPU-only, single CUDA, multi-GPU graph split, MTP, and both fused/unfused MoE,
plus prompt-processing and steady-state decode benchmarks at equal VRAM use.

The persistent pool design is the better first port candidate: its prefill bypass
and backend-neutral ID remapping give it a smaller maintenance surface. Keep it as
a separate experimental upstream branch until those tests pass; then add it here
as a pinned opt-in ref rather than a growing patch file.

The longer implementation sketch is kept beside this audit in
[`moe-cache.md`](moe-cache.md).

### 2026-10-08 follow-up: persistent pool merged upstream

The persistent pool now has a merged implementation in
[llama.cpp #29887](https://github.com/ggml-org/llama.cpp/pull/29887), extended to
multiple GPUs by [#30112](https://github.com/ggml-org/llama.cpp/pull/30112).
The final version divides one global cache budget among eligible GPUs; its
cache misses upload experts before the current computation. This differs from
the CPU-miss/async-promotion design in `moe-cache.md`.

Adoption in IK is feasible through a manual source port, starting with CUDA and
layer splitting. Neither upstream diff applies directly, and IK needs scheduler
copy hooks, device mapping, fused/repacked MoE handling and lifecycle integration.
See the [detailed adoption audit](moe-cache-30112-audit.md) against IK main at
`67477b6e9b00`. No cache patch is enabled in these build recipes.

The build source is now [our fork](https://github.com/delfianto/ik_llama.cpp).
The [port plan](moe-cache-port-plan.md) uses the merged persistent pool as the
first experiment; [model candidates](../experiments/moe-models.md) and a pinned
download manifest live in `experiments/`. Inference implementation remains pending.

## Fork experiment: persistent MoE decode cache

The manual port is published on
[`delfianto/ik_llama.cpp:moe-cache`](https://github.com/delfianto/ik_llama.cpp/tree/moe-cache).
`moe-cache-v1.patch` is a backup of that source commit, not another automatic
build patch. Select the branch with `IK_LLAMA_REF=moe-cache`.
See [validated support and measurements](../experiments/moe-cache-results.md).
V1 caches single-token decode; multi-token caching is deferred after failing
its full-model numerical gate. The older `moe-cache.md` is a separate design.
