# llama.cpp #30112 adoption audit

Audited 2026-10-08. Build recipe's cached IK source: `c069e89e1020`.
Also inspected current IK main in an isolated checkout at
`67477b6e9b0004f08e954f2922b5dcaab952b333`.
Upstream merged revision: `c811cb8f0ac91b8ac72a32f970bdd45037f20da7`.

## Decision

Adoption is technically feasible as a manual, opt-in source port. Neither
[#30112](https://github.com/ggml-org/llama.cpp/pull/30112) nor its prerequisite
[#29887](https://github.com/ggml-org/llama.cpp/pull/29887) applies directly to IK.
Do the work in an experimental IK source branch, then consume a pinned revision
from this build repository after correctness and performance validation.

This is a persistent GPU expert pool, different from the CPU-miss/async-promotion
design proposed in `moe-cache.md`. Its implementation is substantially simpler,
but a miss introduces a transfer dependency for the current computation.

## What actually merged

#29887 supplies the cache, graph substitutions, configuration and context
lifecycle. #30112 extends that cache to multiple GPUs:

- Each eligible layer uses the cache on its assigned GPU. Only layers whose
  expert matrices are all host-resident weight buffers are registered.
- Layers on the same GPU with matching tensor type, shape and expert stride
  share banks and an LRU. Slot assignment covers gate/up/down together.
- A host expert-ID-to-slot map is looked up with `GET_ROWS`; small-batch expert
  matmuls use the GPU banks and slot IDs instead of the host weights and expert IDs.
- The scheduler weight-copy callback reads routing IDs back to the host,
  synchronizes that read, plans the LRU, uploads missing experts, then uploads
  the slot map. Async API calls do not eliminate the current token's dependency
  on those uploads. Misses are not computed by CPU alongside GPU cache hits.
- Batches above 32 tokens bypass the bank graph. Smaller batches also bypass
  it when the conservative maximum number of selected distinct experts exceeds
  slot capacity. Large-batch expert copying can reuse resident experts through
  device-to-device copies without populating/evicting the decode cache.
- Pipeline parallelism is disabled while the cache is enabled; upstream tensor
  split mode is rejected. The added multi-device test uses layer splitting.
- `--moe-cache-mib N` is a **global** budget in the final merge. It is divided
  among devices with eligible host experts using explicit tensor-split weights,
  otherwise reported free memory, with equal shares when the sum is zero.
  Each device's share is divided among layout groups by host expert bytes.
  Slot rounding, extra padding slots, and uncached groups can leave budget unused.

The PR description still says the budget is per GPU and gives results from that
earlier configuration. Its two-4090 benchmark reports decode improvements of
1.74–1.85x, alongside prompt throughput reductions of 9–27%. Those measurements
establish upstream promise; they do not predict IK performance or validate the
final global-budget configuration. See the
[final cache implementation](https://github.com/ggml-org/llama.cpp/blob/c811cb8f0ac91b8ac72a32f970bdd45037f20da7/src/llama-moe-cache.cpp).

## IK integration work

Paths below refer to IK main at the audited revision.

| Area | Finding and required adaptation |
| --- | --- |
| Scheduler | `ggml/include/ggml-backend.h` has no upstream `ggml_backend_sched_copy_callback` or setter. Introduce a context-owned callback and call it from `ggml_backend_sched_copy_inputs()` in `ggml/src/ggml-backend.cpp`, after required non-weight inputs are ready and before ordinary host-weight copying. A handled slot-map copy must bypass the default copy. |
| Split ordering | IK has a host-weight split boundary heuristic, but the port must verify that selected IDs are computed in an earlier split than the slot lookup. Upstream aborts if they share a split. Preserve synchronization before overwriting bank slots used by earlier queued work. |
| Device mapping | IK uses integer device IDs, `default_layer_device`, and per-layer buffer types rather than upstream's backend-device registry and `model.dev_layer()`. Map each layer to the matching context backend explicitly; adapt memory queries and budget splitting to IK's APIs. |
| Context/configuration | Add cache ownership to `src/llama-context.h`, parameter/default/CLI plumbing in IK's `include/llama.h`, `src/llama-cparams.h`, `src/llama.cpp`, and `common/common.{h,cpp}`. Instantiate before graph reservation, account for VRAM, and release with the context. IK parses these CLI options in `common/common.cpp`, not upstream's `common/arg.cpp`. |
| Graphs | Integrate in `src/llama-build-context.cpp`, especially `llm_build_moe_ffn()` and `llm_build_lora_mm_id()`. Substitute cached weights and slot IDs only for base expert matrix operations. Keep original expert IDs for biases, scales, routing and LoRA adapters; preserve weight-specific Hadamard transformations. |
| Fused MoE | IK defaults to fused up/gate, including `ggml_moe_up_gate_ext()` with biases. Separate fused weights can share the same slot remapping. Fused biases must retain original expert indexing or be cached/remapped consistently. Initially exclude biased fused paths if necessary. |
| Repacked weights | IK's merged tensor is named `ffn_up_gate_exps`; upstream uses `ffn_gate_up_exps`. IK may retain both original matrices and a repacked merged representation. Register the representation actually consumed by the graph after `llama_repack_up_gate_exps()`, rather than blindly caching every non-null field. Verify physical strides and CUDA kernel support for each type. |
| Existing offload paths | IK already selectively copies active experts and prefetches mmap pages. Put the cache callback ahead of ordinary expert uploads and keep large-batch selective copying/prefetch behavior. Avoid duplicate transfers and unnecessary full expert-shaped staging allocations in the cached decode graph. |
| Lifetime | Hot reload currently resets `prev` and `prev_mtp` graphs. It must also invalidate/rebuild cached experts and pointer bindings after affected weights change. Give MTP/draft contexts separate ownership until sharing is explicitly made safe. |

## Recommended implementation scope

1. Port the merged pool design with one CUDA device, ordinary supported expert
   layouts and small decode batches. Retain existing behavior with cache disabled.
   Validate unfused operations first, then separate-weight fused up/gate.
2. Extend to multiple CUDA devices in **layer split** with the final global
   budget semantics. Disable pipeline parallelism while enabled. Initially reject
   IK graph/attention split modes: upstream's layer ownership model does not prove
   safety for IK's sharded experts and concurrent scheduler workers.
3. Add merged/repacked layouts, biased fused paths, hot reload and MTP coverage
   before widening eligibility. Unsupported configurations need a clear fallback
   or rejection, rather than a graph silently interpreting slots as expert IDs.

No new CUDA matmul kernel is inherently required for layouts already supported
by IK: the pool exposes ordinary device tensors to existing kernels. Eligibility
checks must cover the actual matmul/fused operations, not only slot lookup support.

## Validation before adopting in builds

- Compare logits with cache disabled/enabled across empty cache, repeated hits,
  forced evictions, heterogeneous layouts, and capacities near one token's needs.
- Cover CPU-only/disabled operation, single CUDA, two-device layer split,
  unequal and zero tensor-split shares, fully device-resident and mixed-placement
  layers, fused/unfused paths, prefill/decode transitions, and repeated contexts.
- Test batch boundaries around 32 and speculative/MTP multi-token batches;
  test reload after a cache has warmed, once those modes are supported.
- Benchmark cold and steady-state decode plus prompt processing at equal total
  VRAM. Compare against IK's IQK CPU path and active-expert offload, reporting
  hit rate, uploaded bytes, memory use and transfer/synchronization time.

The expected benefit is strongest when host expert computation or repeated
uploads dominate and the routed working set fits the cache. Poor hit rates can
make this transfer-dependent design slower than IK's CPU expert kernels.

## Checks performed and limits

Read both PR diffs and the final merged cache implementation; inspected the
cached IK source and fetched current IK main into an isolated checkout.
`git apply --check` fails for both diffs against current IK, with missing files
and incompatible hunks. No source port, GPU correctness run, or performance
benchmark was performed. Build recipes and runtime defaults are unchanged.
