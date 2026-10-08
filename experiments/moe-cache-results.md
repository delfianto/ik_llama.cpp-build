# Persistent MoE cache experiment

Implemented on [moe-cache](https://github.com/delfianto/ik_llama.cpp/tree/moe-cache),
commit `ce36dfeb244b0f979b65e3126e1e44faf892ec19`, based on IK
`67477b6e9b0004f08e954f2922b5dcaab952b333`. The
[portable patch](../patches/moe-cache-v1.patch) applies cleanly to that base.
Docker/Arch defaults still build the fork's `main`; select the experiment explicitly.

```bash
IK_LLAMA_REF=moe-cache just docker cuda
# runtime example: experts in RAM, separate matrices, 1 GiB total cache
llama-server -m model.gguf -ngl 999 --cpu-moe -no-fug --moe-cache-mib 1024
```

## Delivered support

This is a persistent GPU pool with shared LRU banks across compatible layers,
not a CPU-miss/async-promotion executor. Host GGUF weights remain authoritative.
Every miss uploads the selected expert before its operation. Banks are owned by
one inference context, synchronized on the GPU dependency chain and released
before its scheduler/backends. Expert routing, shared experts and auxiliary scales
retain original IDs; bank matmuls use slot IDs.

`--moe-cache-mib N` and `LLAMA_ARG_MOE_CACHE_MIB` accept a nonnegative **total**
MiB budget; the public context parameter uses bytes and defaults to zero.
Device shares use explicit tensor-split weights, otherwise available GPU memory.
Only devices with eligible layers participate. Actual CUDA padding/alignment and
an extra zeroed expert guard count toward the bank allocation; oversized budgets
stop at the model's complete eligible expert set rather than allocating unused VRAM.
Requested and actual bank memory and per-group hit/miss/eviction/upload counts are logged.
`GGML_MOE_CACHE_TRACE=1` adds ID read/sync and upload timing, with an extra GPU
synchronization that distorts throughput. Cached compute event timing is deferred.

Supported: CUDA none/layer splitting, contiguous separate gate/up/down weights,
canonical Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, K-quants and F16/F32 when the backend
accepts their matmul. Q4_0/Q4_K/Q6_K were exercised on both local GPUs. Biased,
merged/repacked/interleaved, rotated and active-adapter paths bypass the cache.
RPC, effective graph/attention splitting, MTP/draft/speculative decoding and hot
reload are excluded. IK may itself convert an unsupported model's requested graph
split into a layer split; this port checks the resulting effective layout.
CPU builds work with cache zero and reject a nonzero budget.

**V1 caches single-token decode only.** All multi-token batches use existing
inference without updating cache residency. The proposed 2/8/32-token extension
failed the full-model numerical gate: pooled bank dimensions change CUDA MMQ/fusion
paths. Per-operation byte checks and kernel NMSE passed, but that was insufficient
to establish full-model equivalence. No tolerance was relaxed to ship those graphs.
Fixing their kernel selection/numerical behavior is the next port phase.

## Validation

Native CPU and CUDA server/bench builds passed. CPU and CUDA cache-state tests
cover duplicates, repeated hits, selected-hit protection, cross-layer eviction,
overflow without mutation, generation wrap, layout groups, multiple devices,
exact uploaded expert bytes, fused/unfused projections and batch bypass.

The [12-case full-model gate](results/validation/validation.json) passed:
Granite and Qwen3, single GPUs and a 4:3 pair; Granite additionally used equal,
skewed and zero-share splits, graphs disabled and unfused MoE. Tiny budgets and
effective graph split were rejected. Each accepted case consumed fixed teacher-forced
tokens in batches 1/2/8/32/33/64, comparing a cache-zero context with an enabled
context; only batch 1 used the cache. All reported maximum logit NMSE **0** and
zero greedy differences, below the required `5e-4` limit. These comparisons force
CUDA expert computation in both contexts with `-cuda offload-batch-size=0`;
they do not claim bitwise equivalence to CPU quantized kernels.

Both default and experimental-branch `just verify` passed, including package
metadata, existing FastMTP patch applicability, and CPU/CUDA Docker definitions.
New runtime container images were not built. Source/build/model provenance is in
[provenance.json](results/provenance.json) and the per-run JSONL metadata.

## Measurements

Initial measurements are recorded under [results](results/). They use the same
pinned GGUF, 512 real prose prompt tokens, 256 generated tokens, temperature zero,
EOS ignored, 16 CPU threads, concurrency one and three measured requests per case.
The first request has a fresh expert cache; later requests retain it. OS page cache
is not flushed. Native builds use GCC 16.2.1, CUDA 13.4.92 and SM86/89.

These are **same host-weight placement comparisons**, not equal-total-VRAM comparisons.
IK's active control forces CPU-resident experts to execute on CUDA with
`offload-batch-size=0`. Its ordinary hybrid control and upstream's zero-cache
control may execute decode experts on CPU. Full-GPU and CPU-only controls are
reported separately. Generated token sequences can differ between engines and
kernel paths, changing routing locality; raw output hashes are retained.
GPU memory samples include background applications and sample every 0.5 seconds.

The runner also supports prose/code/math/Indonesian/French, prompt lengths
512/2048/8192, streaming TTFT, and the full budget sweep. Initial results are a
small first experiment; the complete multi-domain, long-context and equal-VRAM
campaign remains work to run, alongside broader model compatibility.

### Initial Granite result

Warm decode rates (tokens/s; two requests after one cold request):

| Control / budget | 4080 | 3060 | 4080 + 3060 |
| --- | ---: | ---: | ---: |
| Unmodified IK, ordinary hybrid | 115.8 | 95.6 | 78.0 |
| Unmodified IK, forced GPU expert upload | 52.7 | 14.5 | 38.7 |
| Port, 512 MiB | 393.9 | 164.2 | 248.8 |
| Port, 1024 MiB | 407.1 | 174.1 | 247.1 |
| Port, 2048 MiB | 336.1 | 181.3 | 244.4 |
| Port, 4096 MiB requested | 331.0 | 192.8 | 238.1 |
| Unmodified IK, full GPU | 459.5 | 249.8 | 334.4 |
| Upstream, full GPU | 421.2 | 239.7 | 284.6 |

The 512 MiB port run on the 4080 recorded 99.27% cache hits and approximately
1.86 GiB of uploads over three generations. A separate upstream 512 MiB run
recorded 43.23% hits and 148 GiB of uploads. Their generated output hashes differ;
this workload's routing locality differs, so this is not a clean engine ranking.
For this small model, full residency remains preferable when memory permits.
More bank memory did not monotonically improve the port's throughput.

### Initial Qwen3 port result

| Total cache MiB | 4080 warm decode t/s | 4080 + 3060 warm decode t/s |
| --- | ---: | ---: |
| 0, forced GPU uploads | 21.0 | 14.0 |
| 512 | 21.6 | 9.3 |
| 1024 | 33.3 | 15.3 |
| 2048 | 43.9 | 20.2 |
| 4096 | 82.2 | 41.6 |

On the 4080, 512 MiB produced effectively zero hits; 4096 MiB reached 84.00%
hits. All ten port device/budget cases produced the same output hash. This is
stronger evidence for a cache-capacity effect within IK than a cross-engine
comparison. Warm TTFT stayed around 0.51 seconds on the 4080: prefill uses the
existing path.

### Qwen3 controls

| Control | 4080 warm decode t/s | 4080 + 3060 warm decode t/s |
| --- | ---: | ---: |
| Unmodified IK, ordinary hybrid | 44.9 | 30.8 |
| Unmodified IK, forced GPU expert upload | 20.9 | 13.1 |
| Port, 4096 MiB cache | 82.2 | 41.6 |
| Upstream, ordinary hybrid | 40.7 | 30.9 |
| Upstream, 4096 MiB cache | 67.7 | 39.3 |
| Unmodified IK, full GPU | Not fitted on one GPU | 137.4 |
| Upstream, full GPU | Not fitted on one GPU | 150.4 |

On this initial workload, the port's 4 GiB cache outperformed ordinary IK hybrid
inference on the 4080 by about 1.83×. The two-GPU full-residency control was faster
than either cache implementation. All ten port output hashes match; cross-engine
outputs and routing can differ. These numbers do not establish a general ranking.

### Partial-residency controls

We also tried retaining the final expert layers permanently on the 4080, with
ordinary hybrid scheduling for the remaining layers. This uses roughly similar
expert memory, but is **not an equal-total-VRAM experiment**:

| Model | Permanent expert weight increment | Cache bank allocation | Partial-residency warm t/s | Port warm t/s |
| --- | ---: | ---: | ---: | ---: |
| Granite, `--n-cpu-moe 22` | 552.66 MiB | 509.50 MiB | 169.7 | 393.9 |
| Qwen3, `--n-cpu-moe 36` | 3984.01 MiB | 4094.06 MiB | 50.2 | 82.2 |

The increments come from logged CUDA model-buffer sizes relative to the
CPU-expert placement. KV and logged compute buffers were unchanged in these
controls. Runtime temporary allocations, background activity, and routing still
prevent a strict memory/performance equivalence claim. Granite outputs differ
between these paths; Qwen's comparisons should likewise be read with recorded
output hashes.

The [summary CSV](results/summary.csv), [standalone chart](results/cache-sweep.svg)
and raw JSONL/server logs retain cold and warm timings, command lines, source/model
hashes, memory samples and cache statistics. The failed initial upstream Granite
launches are retained in `granite-upstream.jsonl` and excluded from the summary;
`granite-upstream-v2.jsonl` contains the successful sweep. The full multi-domain,
long-context, isolated-memory campaign and additional model validation are deferred.
