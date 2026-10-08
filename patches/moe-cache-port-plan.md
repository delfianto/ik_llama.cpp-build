# MoE cache port plan

Status: first experimental decode port implemented and validated, 2026-10-08.
See [implementation and results](../experiments/moe-cache-results.md) for the delivered scope and remaining gates.
Source: [delfianto/ik_llama.cpp](https://github.com/delfianto/ik_llama.cpp),
starting at `67477b6e9b0004f08e954f2922b5dcaab952b333`.
Published branch: [`moe-cache`](https://github.com/delfianto/ik_llama.cpp/tree/moe-cache),
commit `ce36dfeb244b0f979b65e3126e1e44faf892ec19`.

## Objective and design

Port the persistent GPU expert pool introduced by
[llama.cpp #29887](https://github.com/ggml-org/llama.cpp/pull/29887) and extended by
[#30112](https://github.com/ggml-org/llama.cpp/pull/30112), using their merged
implementation at `c811cb8f0ac91b8ac72a32f970bdd45037f20da7` as the reference.
The [adoption audit](moe-cache-30112-audit.md) records source incompatibilities.

Implement a context-owned cache with GPU banks shared across compatible layers,
LRU replacement, and a host expert-ID-to-slot map. Register only eligible
host-resident routed expert weights. Keep original model weights authoritative.
Use existing IK CUDA matmul/fused kernels with cached weight views and slot IDs.
Keep routing, shared experts, scales and biases indexed by original expert IDs
unless a supported path explicitly caches/remaps them too.

A miss uploads weights before the current operation. This first experiment
intentionally follows the merged upstream design; the asynchronous-promotion /
CPU-miss executor in `moe-cache.md` is a separate future experiment.

Configuration: `--moe-cache-mib N`, a nonnegative **total** cache budget,
default zero. Split among eligible CUDA devices using selected tensor-split
weights; otherwise available memory, with equal shares if unavailable. Within
each device, divide among representation-compatible groups by expert bytes.
Exclude devices with no eligible host expert layers before normalizing shares.
Use IK's device ordering rather than assuming scheduler backend indices equal
CUDA device indices. Zero-size shares leave their layers on existing paths.
Reject invalid negative/nonfinite split weights and unusable nonzero budgets.
Report requested budget, actual allocations and skipped groups separately.

## Initial support boundary

- CPU-only inference with the cache disabled behaves as before. Enabling it
  without CUDA is an explicit configuration error.
- Single CUDA and then multi-CUDA layer splitting. Disable pipeline parallelism
  whenever the cache is enabled. Reject graph/attention split modes initially;
  their sharded expert tensors/concurrent workers need a separate design.
- First support contiguous, non-repacked expert matrices in CUDA-supported
  formats. Start with Q4_0; cover heterogeneous Q4_K_M groups next.
- Initial cached graph supports one token. Extend to at most 32 tokens using
  upstream's conservative slot-capacity check. Larger batches use existing
  inference and do not evict the decode working set.
- Initially bypass unsupported biased fused, interleaved/repacked, and adapter
  paths. Log eligibility reasons. Enable each only after its numerical tests.
- Reject MTP/draft operation and hot-swap with a nonzero cache in the first
  implementation. Remove those restrictions in the compatibility phase below.
  Defaults with cache zero retain their existing behavior.

## Implementation sequence

### 1. Isolate source development and capture baselines

Create a persistent writable checkout of the fork, independent of this recipe's
depth-one bare cache and disposable `.cache/docker/src` tree. Use a local
`moe-cache` branch. Keep separate build directories for unmodified
IK, ported IK, and upstream llama.cpp; record full source SHAs and compiler flags.
Do not develop inside `_materialize` output: the next build replaces it.

Use the existing source branch selector once that branch has been published:

```bash
IK_LLAMA_REF=moe-cache just docker cuda
IK_LLAMA_REF=moe-cache just pkg
```

Record current IK cache-zero baselines before changes. No model or image needs
to be downloaded/built as part of this planning step.

### 2. Add scheduler integration and cache core

Add `src/llama-moe-cache.{h,cpp}` and list them in `src/CMakeLists.txt`.
Adapt the upstream LRU/bank logic to IK model fields and backend APIs rather
than copying its model/device-registry calls wholesale.

Add `ggml_backend_sched_copy_callback` and its setter to IK's scheduler API.
Invoke it in `ggml_backend_sched_copy_inputs()` for host weight inputs after
non-weight inputs are available; a handled input skips the ordinary copy.
Preserve the active-expert copy path for unhandled inputs. No global singleton.

Verify split boundaries explicitly: the selected expert IDs must have been
computed before the slot-map callback. Preserve wait/sync before reusing banks
or overwriting slot maps consumed by previously queued kernels. Reserve padding
required by IK's actual CUDA kernels; do not infer all padding requirements from
upstream's one extra slot. Validate expert strides, alignment and allocation size.

Cache metadata tests should cover repeated hits, full cache, eviction, duplicate
IDs, one-token minimum capacity, different layouts, and per-device isolation.
These are substantive correctness tests for the new mutable state.

### 3. Wire single-token graphs and CLI

Add parameter/default propagation through `include/llama.h`,
`src/llama-cparams.h`, `common/common.{h,cpp}`, and `src/llama.cpp`.
Own the cache in `src/llama-context.h`; allocate it after required weight
transformations and before graph reservation. Include cache buffers in memory
reporting and allocation failure handling.

In `src/llama-build-context.cpp`, build the `GET_ROWS` slot lookup on the
owning GPU. In `llm_build_moe_ffn()`, feed cache bank views and slot IDs to
supported expert operations. Keep original IDs for everything else.
Check backend support for the actual cached matmul, not just slot lookup.

Preserve original weight identity when consulting Hadamard rotations or LoRA
metadata in `llm_build_lora_mm_id()`. Initially bypass adapters until verified.
Start with unfused paths, then separate-weight fused up/gate without biases.
Validate Granite and Qwen3 with fixed teacher-forced token sequences. Compare
logits with appropriate floating-point tolerances; investigate greedy divergence,
but do not require bitwise equality between different CPU/CUDA reductions.

### 4. Add layer-split multi-GPU and small batches

Map layers using IK's `default_layer_device`, layer buffer types, and actual
context backends. Keep bank pools and LRU device-local. Test our unequal GPUs
with explicit 4:3, equal, skewed and zero-share splits, plus the default allocation.
The 4:3 ratio is a starting point, not a performance optimum.

Extend cached graphs from one token to batches 2/8/32 when slot capacity permits.
Test batch 33 and small batches exceeding distinct-expert capacity use the
existing path. Add optional D2D reuse of cached experts in the large-batch
active-expert copier after small-batch correctness is established.

### 5. Widen compatibility deliberately

- Register the representation actually used after `llama_repack_up_gate_exps()`.
  IK's `ffn_up_gate_exps` may coexist with original matrices; avoid duplicate
  banks for inactive representations. Verify interleaved physical layouts.
- Handle fused expert biases without mixing original IDs and slot IDs. GPT-OSS
  is a deliberate test of this, not an initial simple-cache benchmark.
- Preserve shared/dense expert execution and auxiliary expert scales. Validate
  DeepSeek-Coder-V2-Lite, GLM-4.7-Flash and LFM2.5 after their loader smoke tests.
- Add hot-reload invalidation before cached execution resumes, rebuilding
  pointer bindings/banks as needed alongside `prev` and `prev_mtp` graphs.
- Give target/draft/MTP contexts separate cache ownership initially; budget all
  context allocations together. Validate multi-token verification and reserve
  graphs before removing the MTP restriction.

### 6. Instrument, benchmark, then consider policy experiments

Collect per-device and per-group hit/miss/eviction counts, occupied slots,
eligible host bytes, actual VRAM, upload bytes, ID-read/sync time, upload time,
and cached expert compute time. Use CUDA event timing for asynchronous work
when needed, with detailed tracing opt-in to avoid distorting normal results.
Report prefill, early decode and steady decode separately.

Only after this reference port works, try second-touch admission, alternative
replacement policies, budget allocation from routing traces, pinned staging,
and asynchronous promotion with CPU fallback. Compare one change at a time.

## Benchmark protocol

Hardware observed 2026-10-08: Ryzen 9 9950X, RTX 4080 16 GiB, RTX 3060 12 GiB,
60 GiB usable host RAM. Other processes were using some RAM/VRAM; capture actual
availability before every run. Record GPU UUIDs and `nvidia-smi topo -m` rather
than assuming equal bandwidth or P2P connectivity.

Start with [Granite and Qwen3 in the model shortlist](../experiments/moe-models.md).
Keep test and control on the same exact GGUF and verify numerical correctness
before interpreting performance. Use fixed tokens for kernel measurements and
real prompts for routing/cache behavior; never compare rates across different
model tokenizers as a direct engine ranking.

Controls: IK CPU-only, IK ordinary hybrid layer split, IK active-expert offload,
ported IK cache zero/nonzero, upstream cache zero/nonzero, and full GPU when the
model fits. For speed comparisons use equal total VRAM limits and report the
actual memory of weights + banks + compute + KV/state. Also retain a same-placement
comparison to isolate the cache's effect. Avoid `-rtr` in the initial common-GGUF
hybrid comparison; test IK-specific repacking/quants separately later.

Initial sweep: cache budgets 0/512/1024/2048/4096 MiB; prompt lengths
512/2048/8192; 256 generated tokens; single GPU then both GPUs; concurrency one;
MTP off. Use identical context/KV types, batch sizes, placement, sampling and
thread counts where applicable. Tune CPU threads once, record the choice, and
hold it fixed. Extend budgets only after checking each GPU's remaining memory.

Use at least three measured repetitions and randomize test/control order.
Separate model-load time and OS page-cache warm-up from expert-cache warm-up.
A fresh inference context clears the expert cache but not the OS page cache;
mark that distinction. Do not reuse server prompt-prefix caching when measuring
prefill. Measure fresh-context latency as well as warmed-context throughput.

Prompts: prose/summary, Python/SQL, math, multilingual (including Indonesian),
and alternating domains to test working-set changes. Use a fixed prompt corpus
and fixed output limits. Record EOS termination and output length; short generations
must not be mistaken for faster completed work. Track TTFT, prompt tokens/sec,
decode tokens/sec, request latency, peak host/VRAM use, uploads and hit rate.

Persist raw JSON/CSV results with source SHA, GGUF SHA256 and repository revision,
build flags, command line, GPU UUIDs, driver, CPU threads and prompt IDs. Record
unsupported combinations as skipped with reasons. No performance target is a
correctness gate: an experiment that regresses still provides useful data.

## Completion gates

1. Source builds on CPU and CUDA with cache zero; existing applicable checks pass.
2. Unit state tests and teacher-forced numerical tests pass with hits and evictions.
3. Multi-device isolation, budget handling and batch bypass are verified.
4. Unsupported configurations fail clearly or use a documented safe fallback.
5. Benchmarks can explain a win or loss using memory, hit-rate and transfer data.

Graph/attention-split caching and concurrent shared caches remain later projects,
not claims made by this port. The original plan remains below; v1 deliberately restricts cached graphs to one token because the multi-token full-model numerical gate failed.
