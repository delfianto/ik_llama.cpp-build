# Technical implementation plan: Hybrid MoE VRAM Expert Cache for `ik_llama.cpp`

## 1. Executive design decision

The implementation should **not** be a direct port of either the original `llama.cpp` experiment or thecodacus' fork.

The best design is a synthesis:

1. Take the **execution model** from the later `llama.cpp` RFC:

   * CPU remains the authoritative executor for CPU-resident experts.
   * VRAM-resident experts are opportunistically stolen by CUDA.
   * A cache miss is executed immediately by the normal CPU path.
   * A cache miss may schedule an asynchronous promotion into VRAM for a future token.
   * **Never transfer an expert synchronously just to satisfy the current token.**

2. Take several practical ideas from thecodacus:

   * persistent expert→slot bookkeeping;
   * fixed VRAM slot slabs rather than temporary staging buffers;
   * dedicated H2D copy streams;
   * pinned-host-memory experiments;
   * exploiting the guaranteed relationship between `up`, `gate`, and `down`;
   * sibling expert prefetch after routing is known;
   * simple LRU as a starting point;
   * extensive hit/miss/eviction telemetry.

3. Exploit capabilities that already exist uniquely in `ik_llama.cpp`:

   * `GGML_OP_MOE_FUSED_UP_GATE`;
   * IQK CPU MoE kernels;
   * the `--offload-policy` machinery;
   * existing mmap/page-cache expert prefetch;
   * MTP/speculative multi-token execution;
   * explicit GPU split infrastructure;
   * hot tensor reload and graph invalidation.

The central invariant of the entire project should be:

> **The VRAM cache accelerates already-resident experts. It must never turn a cache miss into a PCIe dependency on the current token.**

That is the critical lesson from the RFC. Earlier approaches moved `MUL_MAT_ID` onto CUDA, making cache misses synchronous PCIe transfers. The RFC explicitly identifies that behavior as the reason several prior attempts regressed badly, and instead keeps `MUL_MAT_ID` logically on CPU while CUDA handles only hit rows. ([GitHub][1])

---

# 2. Desired end-state architecture

The eventual execution hierarchy should look like this:

```text
                         GGUF / mmap
                             │
                             │ existing IK prefetch
                             ▼
                    Linux page cache / RAM
                             │
                             │ asynchronous VRAM promotion
                             ▼
                    ┌──────────────────┐
                    │ VRAM expert cache│
                    └──────────────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
            CACHE HIT                CACHE MISS
                 │                       │
                 ▼                       ▼
           CUDA matvec             IQK CPU matvec
                 │                       │
                 └──────────┬────────────┘
                            ▼
                       merged result
```

This effectively gives IK a three-tier MoE memory hierarchy:

```text
storage
   ↓
page cache / host RAM
   ↓
VRAM working-set cache
```

IK already implements the first promotion mechanism. Its MoE prefetch engine understands selected experts in `MUL_MAT_ID` and `MOE_FUSED_UP_GATE`, operates in 2 MiB chunks, checks residency with `mincore()`, and uses `MADV_POPULATE_READ` to bring mmap pages into both page cache and the process page tables.

The new feature should therefore be conceptually called something like:

```text
MoE VRAM promotion/cache
```

rather than simply "expert prefetch."

---

# 3. What we should keep and reject from thecodacus

Thecodacus' branch is extremely useful because it exposes concrete implementation problems that abstract RFC discussion doesn't.

## Keep

| Idea                                  | Decision                           |
| ------------------------------------- | ---------------------------------- |
| Persistent VRAM slot pool             | Keep                               |
| Expert → slot hash/direct map         | Keep                               |
| Fixed-size expert slots               | Keep, but shape-specific           |
| Dedicated CUDA copy stream            | Keep                               |
| LRU baseline replacement              | Keep                               |
| Cache telemetry                       | Keep and expand                    |
| Async sibling `up/gate/down` prefetch | Keep, later phase                  |
| Pinned host source                    | Benchmark and probably use staging |
| Cache warm-up measurement             | Keep                               |
| Explicit per-tensor identity          | Keep concept, change keying        |

Its cache implementation maintains `host_ptr → slot` and `slot → host_ptr` mappings, plus LRU timestamps, a dedicated CUDA copy stream and persistent device allocation.

That basic data structure is perfectly sensible.

## Reject

### Synchronous/current-request cache filling

Thecodacus' `acquire()` does:

```cpp
cache miss
   ↓
choose LRU
   ↓
cudaMemcpyAsync(expert → VRAM)
   ↓
return slot
   ↓
current CUDA operation consumes slot
```

That means the "async" copy is still logically on the dependency chain of the current operation.

For IK this becomes:

```text
MISS
 ├─ CPU computes this token immediately
 └─ GPU copy happens for a future token
```

not:

```text
MISS
 ↓
copy
 ↓
wait
 ↓
GPU compute
```

This distinction is basically the entire project.

---

### Fake device residency through a special buffer type

Thecodacus introduced a CUDA-MoE-cached host buffer whose `is_host()` deliberately returns false so the scheduler sends the operation into CUDA even though the authoritative expert data is actually host memory.

Ingenious experiment.

Wrong abstraction for IK.

We want the scheduler to continue understanding:

```text
expert tensor = CPU resident
```

because the normal CPU implementation remains the fallback owner.

The VRAM cache should be an **auxiliary acceleration resource**, not tensor placement.

---

### Name-based cache identity

Thecodacus eventually keys caches using tensor names such as:

```text
blk.17.ffn_up_exps.weight
```

This works in that implementation but is unnecessarily fragile.

IK has hot reload, runtime transformations, split tensors, merged tensors and potentially repacked physical layouts.

Use runtime object identity plus generation instead.

---

### Fixed N slots per tensor

Thecodacus notes that 40 layers × three expert matrices × 32 slots can consume roughly 3.1 GiB/device for one model configuration.

Dedicated per-tensor pools eliminate cross-layer contention, but they waste capacity:

```text
layer 3 cache half empty
layer 4 cache full and thrashing
```

yet neither can borrow from the other.

IK should instead use:

```text
logical per-tensor mapping
             +
shared physical shape pools
```

---

# 4. Fundamental runtime model

For every eligible MoE node, split the routed work into:

```text
CPU rows
GPU rows
```

based strictly on residency.

Example with top-8 routing:

```text
selected experts:
[12, 17, 39, 48, 57, 61, 79, 103]

cache:
12  HIT
17  HIT
39  MISS
48  HIT
57  MISS
61  MISS
79  HIT
103 MISS
```

Execution becomes:

```text
CUDA
  experts 12,17,48,79
        │
        ├───────────────┐
                        │
CPU                     │ concurrent
  experts 39,57,61,103  │
        │               │
        └───────────────┘
                │
                ▼
           final output
```

Misses 39/57/61/103 may also generate promotion requests:

```text
CPU compute 39 ────── current token
       │
       └── async fill expert 39 ── future token
```

The next token may then convert that miss into a hit.

---

# 5. Proposed component architecture

I would introduce:

```text
ggml/src/
    ggml-moe-cache.h
    ggml-moe-cache.cpp

ggml/src/ggml-cuda/
    moe-cache.cu
    moe-cache.cuh
```

The generic layer owns:

```text
session
configuration
tensor registration
cache metadata
routing plans
statistics
admission policy
lifecycle/invalidation
```

CUDA owns:

```text
VRAM allocation
copy streams
events
CUDA expert dispatch
activation upload
result collection
slot/device pointers
CUDA-specific eligibility
```

The existing:

```text
ggml-moe-prefetch.cpp
```

should eventually share routing/expert-selection utilities with the new subsystem.

Do **not** put everything into `ggml-cuda.cu`.

That will become miserable to maintain.

---

# 6. Session ownership

Do not implement a global singleton.

The later RFC work specifically removed global singleton state in favor of session-owned state. Its ablation showed that the simplified/session-owned implementation was actually faster than the earlier full mechanism. ([GitHub][1])

Conceptually:

```cpp
struct ggml_moe_cache_session {
    ggml_moe_cache_config config;

    std::vector<device_cache> devices;
    std::unordered_map<tensor_key, tensor_cache_info> tensors;
    std::vector<shape_pool> pools;

    fill_queue fills;

    cache_stats stats;

    uint64_t generation;
};
```

Lifetime should approximately follow:

```text
model/backend scheduler lifetime
```

rather than:

```text
process lifetime
```

This matters for server reloads, multiple contexts, repeated benchmarking and GPU resource cleanup.

---

# 7. Tensor/expert identity

Use something equivalent to:

```cpp
struct expert_key {
    const ggml_tensor * tensor;
    uint32_t            expert_id;
    uint32_t            generation;
};
```

Or internally:

```cpp
tensor_cache_info {
    ggml_tensor * tensor;

    uint64_t generation;

    int n_experts;
    size_t expert_stride;

    shape_pool * pool;

    std::vector<int32_t> expert_to_slot;
};
```

`expert_to_slot[e]` gives:

```text
-1      not resident
>= 0    slot ID
```

A generation number makes stale entries cheap to invalidate.

---

# 8. Physical VRAM pool design

Physical slots should be grouped by compatible representation.

Example key:

```cpp
struct pool_shape {
    ggml_type type;

    int64_t ne0;
    int64_t ne1;

    size_t expert_bytes;

    cuda_device_id device;
};
```

Potential additional fields later:

```text
row layout
repacked/canonical flag
kernel family
alignment requirements
```

Pool:

```cpp
struct shape_pool {
    void * device_base;

    size_t slot_size;
    uint32_t slot_count;

    std::vector<slot_metadata> slots;
};
```

with:

```cpp
struct slot_metadata {
    tensor_cache_info * owner;
    int32_t expert_id;

    uint64_t last_use;

    slot_state state;

    uint32_t pin_count;

    cudaEvent_t fill_done;
};
```

States:

```text
EMPTY
FILLING
RESIDENT
```

Possibly later:

```text
EVICT_PENDING
INVALID
```

MVP does not need them.

---

# 9. Slot safety invariants

These should be asserted aggressively.

A slot must never be evicted while:

```text
FILLING
```

or:

```text
pin_count > 0
```

A CUDA operation probing a resident expert must pin the slot.

Pseudo-flow:

```cpp
slot = probe(expert);

if (slot && slot->state == RESIDENT) {
    slot->pin_count++;
    gpu_plan.add(slot);
}
```

after GPU completion:

```cpp
for (slot : gpu_plan.slots) {
    slot->pin_count--;
}
```

Without pinning, an asynchronous fill worker could theoretically choose an LRU slot whose expert is still being consumed by a CUDA kernel.

That would be spectacularly unpleasant to debug.

---

# 10. Admission policy

Start extremely simple.

## MVP

Use:

```text
second miss → admission
```

A tiny per-expert saturating counter works:

```text
0 unseen
1 seen once
2 eligible for fill
```

Why not fill on every first miss?

Because broad routing can turn PCIe into:

```text
expert streaming simulator 2026
```

Second-touch admission naturally filters one-off experts.

Later RFC experiments found a useful exception: first-miss admission can make sense when the pool can hold the entire relevant expert inventory. ([GitHub][1])

So eventual rule:

```cpp
if (pool_can_hold_all)
    admit_on_first_miss();
else
    admit_on_second_miss();
```

No LFU, TinyLFU or learned predictor in v1.

---

# 11. Replacement policy

Start with LRU.

```text
find EMPTY
else oldest unpinned RESIDENT slot
```

No fancy policy until traces prove one is worthwhile.

Collect enough data so we can later simulate:

```text
LRU
CLOCK
LFU
2Q
TinyLFU
oracle
```

offline.

This is much better than adding live complexity and wondering whether it helped.

---

# 12. Fill queue

Promotion must be explicitly separated from execution.

```cpp
struct fill_request {
    tensor_cache_info * tensor;
    int expert_id;

    const void * host_src;
    size_t bytes;

    shape_pool * pool;
};
```

Fill queue constraints:

```text
max outstanding jobs
max outstanding bytes
max fills/token
```

For example initial experimental values:

```text
8 fills/node
128 queued requests
512 MiB outstanding
```

but expose as internal constants initially rather than CLI knobs.

Queue overflow:

```text
drop fill
```

Never block inference waiting for cache capacity.

---

# 13. H2D copy path

The copy worker should:

```text
pick request
 ↓
revalidate nonresident state
 ↓
select victim
 ↓
mark slot FILLING
 ↓
cudaMemcpyAsync
 ↓
record event
 ↓
when complete:
    install expert→slot mapping
    state = RESIDENT
```

The expert is considered a cache hit **only after the event has completed**.

Never expose a `FILLING` slot as resident.

---

# 14. Pinned memory question

This is one of thecodacus' most useful experiments.

Their cached-buffer implementation allocates host memory using `cudaMallocHost()` specifically to make asynchronous H2D transfers efficient.

IK, however, frequently works directly from mmap-backed model memory.

And:

```text
MADV_POPULATE_READ
```

does not magically turn those pages into CUDA pinned memory.

Therefore benchmark two designs.

### Option A

```text
mmap/page-cache memory
       ↓
cudaMemcpyAsync()
```

CUDA may internally stage pageable memory.

### Option B

Persistent pinned staging ring:

```text
mmap/RAM
   ↓ memcpy
pinned host staging buffer
   ↓ cudaMemcpyAsync
VRAM
```

Counterintuitively, Option B may win because CPU memcpy bandwidth is huge and CUDA gets a genuinely asynchronous DMA source.

Suggested staging ring:

```text
2–4 buffers/device
each >= maximum expert slab
```

Do not assume which wins.

Measure it.

---

# 15. Integration with existing IK page prefetch

This is where an IK implementation can be cleaner than both source projects.

Currently IK already knows how to obtain selected expert ranges:

```cpp
collect_selected_ranges(...)
```

and understands:

```text
MUL_MAT_ID
MOE_FUSED_UP_GATE
```

including `-1` expert sentinels and fused gate/up tensors.

Refactor the common expert extraction logic into something like:

```text
ggml-moe-routing.h/.cpp
```

providing:

```cpp
ggml_moe_collect_selected_ids(node, result);
```

and:

```cpp
ggml_moe_selected_range(tensor, expert_id);
```

Then both systems use the same interpretation of routing.

```text
                    routing IDs
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
RAM/page-cache prefetch       VRAM expert cache
```

This avoids two independent implementations gradually disagreeing about fused tensors or tensor layouts.

---

# 16. Cache miss interaction with disk/page-cache prefetch

A VRAM miss should not do:

```text
wait for mmap page
wait for H2D
GPU
```

Instead:

```text
current token:
    CPU normal path

background:
    ensure pages hot
    then promote expert
```

If the existing prefetch engine has already made those pages resident, great.

If not, the background fill worker may fault them while copying.

Potential later optimization:

```text
VRAM fill request
     ↓
check mmap residency
     ↓
not resident?
   schedule urgent RAM prefetch
     ↓
promotion worker waits OUTSIDE inference path
     ↓
H2D
```

But that is phase 3+, not MVP.

---

# 17. `GGML_OP_MUL_MAT_ID` execution hook

The first functioning prototype should support only ordinary:

```text
GGML_OP_MUL_MAT_ID
```

Conditions:

```text
CUDA enabled
CPU-resident expert tensor
n_tokens == 1
supported CUDA quant
cache enabled
pool allocated
```

The CPU kernel thread 0 does approximately:

```cpp
plan = cache_plan(node);

if (!plan.gpu_rows.empty()) {
    cuda_dispatch(plan.gpu_rows);
}

cpu_execute(plan.cpu_rows);

cuda_collect(plan);

merge_outputs();
```

Important:

```text
CPU workers must begin misses immediately.
```

Do not plan everything, dispatch CUDA, wait, and then start CPU.

The whole point is:

```text
GPU bandwidth/computation
       overlaps
CPU memory bandwidth/computation
```

---

# 18. Failure semantics

This should be deliberately paranoid.

## Probe failure

```text
everything CPU
```

## CUDA dispatch failure before launch

Restore GPU rows into CPU work list:

```text
everything CPU
```

## CUDA execution/collect failure

CPU recompute stolen rows.

Correctness beats cleverness.

This gives the optimization the valuable property:

```text
cache failure → degraded performance
not
cache failure → bad logits
```

---

# 19. Conventional scheduler offload interaction

IK already has an explicit offload policy.

Its existing policy recognizes:

```text
GGML_OP_MUL_MAT       26
GGML_OP_MUL_MAT_ID    27
GGML_OP_MOE_FUSED...  29
```

and allows disabling them individually.

For cache-eligible decode nodes, ordinary GPU offload and expert-cache interception must not compete.

Policy should become approximately:

```cpp
if (moe_cache_session &&
    node_is_decode &&
    host_expert_tensor &&
    cache_eligible(node)) {

    keep node CPU-owned;
}
else {
    existing_offload_policy();
}
```

Prompt processing remains governed by normal scheduler decisions.

---

# 20. Do not touch PP initially

The RFC specifically keeps prefill untouched because prompt routing is much broader and can thrash the cache. ([GitHub][1])

Phase 1 eligibility:

```text
n_tokens == 1
```

Eventually:

```text
n_tokens <= 8
```

for MTP/speculation.

But:

```text
large PP batch
```

should remain stock IK.

This cleanly separates:

```text
PP optimization = existing GPU offload + prefetch
TG optimization = persistent VRAM cache
```

---

# 21. Fused MoE should be Phase 2, but treated as strategically important

IK has a massive advantage here.

Its fused operation turns:

```text
up
gate
activation
```

into one MoE operator and can additionally fuse the following multiply on CUDA. Existing IK measurements showed about 7% TG improvement and much larger PP improvement on the test hardware.

Meanwhile the newer RFC ablation is extremely revealing:

```text
cache only            ~ +8%
fusion contribution   ~ +9%
redirect              ~ +2%
backfill              ~ +1%
```

and packed dispatch pushed the minimal implementation beyond the older full implementation. ([GitHub][1])

So cache residency and fused dispatch are **roughly equally important**.

This means we should not think:

```text
cache first
fusion optional someday
```

but rather:

```text
MVP proves architecture

MVP+1 integrates MOE_FUSED_UP_GATE immediately
```

---

# 22. Fused gate/up residency rules

For:

```text
MOE_FUSED_UP_GATE
```

initial rule:

```text
GPU row only if BOTH required expert slabs are resident.
```

Example:

| Expert | Gate | Up   | Execution |
| ------ | ---- | ---- | --------- |
| 17     | hit  | hit  | GPU       |
| 22     | hit  | miss | CPU       |
| 31     | miss | hit  | CPU       |
| 48     | miss | miss | CPU       |

Do not initially split one expert across CPU and GPU because one half happens to be resident.

That produces annoying synchronization with very questionable gain.

The later RFC also found pair-aware eviction unnecessary because half-resident pairs were extremely rare in their measurements. ([GitHub][1])

---

# 23. Thecodacus sibling prefetch: adapt, don't copy

This is perhaps the best idea from that fork.

Its observation:

```text
router selected expert E
```

means:

```text
up[E]
gate[E]
down[E]
```

are all known dependencies.

Its branch therefore starts asynchronous acquisition of sibling matrices once one is encountered.

But in our architecture, this becomes:

```text
routing decision known
      │
      ├── execute current hits
      ├── CPU compute current misses
      │
      └── enqueue future promotions:
            gate[E]
            up[E]
            down[E]
```

No waits.

For IK fused gate/up:

```text
MOE_FUSED_UP_GATE
       │
       └──── background promote down[E]
```

is especially attractive.

By the time:

```text
MUL_MAT_ID(down)
```

runs, `down[E]` may already be resident.

This is true prefetch because the transfer overlaps useful work rather than blocking the requesting operation.

---

# 24. Packed dispatch

Do not launch one CUDA copy/kernel per hit expert.

Construct one packed dispatch descriptor.

Something like:

```cpp
struct gpu_row_desc {
    uint32_t slot;
    uint16_t token;
    uint16_t route;
};
```

Pack:

```text
activation data
routing/slot metadata
```

into as few H2D transfers as possible.

The RFC's newer Nsight work found two H2D copies per dispatch and removing that extra transfer eliminated **40,596 H2D operations** in matched traces, raising performance further. ([GitHub][1])

Small control transfers matter because MoE decode consists of enormous numbers of tiny dispatches.

---

# 25. Reuse existing CUDA indirect-matmul infrastructure

Do not invent a completely separate matrix multiplication implementation.

IK already has CUDA MoE/indirect matmul machinery.

The new kernel interface conceptually needs to replace:

```text
logical expert ID → tensor base + expert stride
```

with:

```text
logical expert ID → cache slot address
```

The actual quantized GEMV machinery should remain shared.

Ideal eventual interface:

```cpp
launch_cached_moe_matvec(
    slot_base,
    slot_stride,
    slot_ids,
    activations,
    routing,
    output,
    ...
);
```

not synthetic temporary `ggml_tensor` construction if avoidable.

---

# 26. Row-interleaved quantization is the largest IK-specific trap

The offload-policy documentation explicitly notes that row-interleaved formats such as the `_R4` / `_R8` families cannot use current CUDA GEMM/GEMV paths, and `-rtr` effectively disables those GPU operations.

Therefore the first version must state:

```text
MoE VRAM cache incompatible with -rtr
```

or more precisely:

```text
Only cache experts whose current physical representation is CUDA-consumable.
```

This is not a minor detail.

If enabling the cache forces the user to give up a much faster IQK CPU representation, the cache can win against the wrong baseline while losing in actual use.

---

# 27. Benchmarking must therefore use three arms

For every meaningful benchmark:

### Arm A — best normal IK

```text
-rtr or other normal CPU optimization enabled
cache off
```

### Arm B — cache-compatible baseline

```text
same physical expert layout required by cache
cache off
```

### Arm C — cache enabled

```text
identical physical layout to B
cache on
```

Interpretation:

```text
C - B = cache mechanism gain

C - A = actual user benefit
```

This is absolutely mandatory.

---

# 28. Future solution for RTR

Long-term, IK could do something genuinely interesting:

```text
GGUF canonical representation
          │
          ▼
RAM runtime representation
     row-interleaved / IQK
          │
          ├──────── CPU
          │
          └── background transcode
                    ↓
              CUDA-native expert
                    ↓
                  VRAM
```

This would effectively give different memory tiers different physical weight encodings.

That is ambitious, but conceptually sound.

Do not attempt it in MVP.

---

# 29. Cache allocation policy

Start with explicit user size:

```text
--moe-vram-cache 8192
```

meaning:

```text
8192 MiB total cache budget
```

not "N slots/tensor."

Only after manual sizing is proven should we add:

```text
--moe-vram-cache auto
```

Automatic sizing should prioritize:

1. dense layers;
2. required compute buffers;
3. KV cache;
4. safety reserve;
5. MoE cache gets genuinely unused capacity.

The RFC author explicitly notes that the desirable regime is after all dense layers fit on GPU and enough VRAM remains for a meaningful fraction—roughly 20–30% in his hardware observations—of the MoE working set. ([GitHub][1])

---

# 30. Multi-GPU: defer deliberately

MVP:

```text
one CUDA device
```

Do not immediately distribute one cache over multiple GPUs.

First establish:

```text
CPU + one GPU cache
```

with unquestionably correct semantics.

Then multi-GPU can use:

```text
device-local pools
```

and stable tensor/layer assignment.

Possible later rule:

```text
layer N cache owner = GPU already executing dense part of layer N
```

avoiding unnecessary inter-device traffic.

Much later:

```text
NVLink-aware peer cache
```

might be fascinating, but that's another research project.

---

# 31. NUMA considerations

For multi-GPU systems, future fill workers should ideally know:

```text
GPU ↔ NUMA node affinity
```

A cache attached to GPU 3 should preferentially receive H2D traffic from RAM local to GPU 3's PCIe root complex.

Not MVP.

But structure the API so `device_cache` is explicit from day one rather than hidden global state.

That keeps NUMA-aware scheduling possible later.

---

# 32. MTP/speculative decoding

MVP:

```text
n_tokens == 1
```

Phase 3:

```text
n_tokens <= 8
```

The newer RFC results show this is potentially huge. Their cache combined very well with speculative/MTP verification workloads; one GLM-5.2 result went from 17.80 to 29.35 t/s with cache + MTP in the tested configuration. ([GitHub][1])

IK is particularly well positioned here because it already actively develops MTP paths.

Once stable, verification batches may become an even better cache target than single-token decoding.

---

# 33. Hot reload integration

IK's tensor reload system can:

```text
change tensor backend
change tensor type
change buffers
change split topology
detach/reattach storage
invalidate CUDA graphs
```

and explicitly supports MoE sibling consistency.

Therefore any cached expert bytes can become stale.

MVP rule:

> **Any successful tensor hot reload flushes the entire MoE VRAM cache session.**

Simple and safe.

Later:

```cpp
moe_cache_invalidate(base, size);
```

or:

```cpp
moe_cache_invalidate(tensor);
```

can surgically invalidate affected entries.

But full flush is more than good enough initially.

---

# 34. CUDA graph considerations

Cached slot addresses should remain stable.

That argues strongly for:

```text
one fixed slab allocation
```

and mutable:

```text
slot index/control buffers
```

rather than individual `cudaMalloc()` per expert.

Avoid capturing:

```text
expert 42 currently lives at random pointer X
```

inside a CUDA graph.

Prefer:

```text
pool_base
slot_size
slot_indices[]
```

where only `slot_indices[]` changes.

---

# 35. Telemetry

This project needs excellent telemetry from day one.

At minimum:

```text
cache probes
hits
misses
hit rate

GPU rows
CPU rows

fills requested
fills admitted
fills completed
fills dropped

bytes promoted
evictions

requested cache size
actual cache size
slot count
resident experts
coverage

H2D weight bytes
H2D control bytes
D2H result bytes
```

Fused:

```text
fused probes
full-pair hits
gate-only hits
up-only hits
pair misses
```

Timing:

```text
probe time
planning time
CUDA dispatch
CPU work
CUDA collect
fill worker time
```

---

# 36. Routing trace mode

Add optional debugging output:

```text
layer,token,expert,event
```

Example:

```text
17,441,38,miss
17,442,38,fill
17,445,38,hit
```

From this we can calculate offline:

```text
frequency
reuse distance
working-set size
LRU miss curve
admission behavior
eviction lifetime
```

Reuse distance will tell us vastly more than just "expert 38 was popular."

---

# 37. Useful derived metrics

### Fill amplification

```text
H2D expert bytes / RAM bytes avoided
```

If this exceeds ~1 badly, we're moving too much data for too few hits.

### Reuse factor

```text
GPU cache hits generated per expert fill
```

A useful cached expert should ideally produce multiple hits before eviction.

### Effective cache coverage

```text
resident expert bytes / total eligible expert bytes
```

### Avoided host bandwidth

Approximate:

```text
sum(expert bytes for GPU-hit rows)
```

This lets us compare performance gain directly against reduced DDR traffic.

---

# 38. Benchmark matrix

For each major model:

```text
cache size:
0
2 GiB
4 GiB
8 GiB
12 GiB
16 GiB
...
```

Report:

```text
actual allocated capacity
expert coverage
steady hit rate
TG
TTFT
PP
```

Do not report merely:

```text
--cache 16GB
```

because allocation may be capped or fragmented.

The RFC's independent testers also found that larger pools need substantially longer warm-up; otherwise an under-warmed large cache misleadingly looks saturated. ([GitHub][1])

---

# 39. Warm-up protocol

Measure separately:

```text
cold TG
100-token TG
500-token TG
1000-token TG
settled TG
```

Plot or report:

```text
hit rate vs token number
```

Cache warm-up itself is part of runtime behavior.

---

# 40. Existing page-cache prefetch benchmark matrix

Since IK already has a prefetch mechanism, test four configurations:

| RAM prefetch | VRAM cache |
| ------------ | ---------- |
| OFF          | OFF        |
| ON           | OFF        |
| OFF          | ON         |
| ON           | ON         |

And ideally:

```text
mmap
no-mmap
```

Otherwise we cannot tell whether VRAM caching and disk/page-cache prefetch reinforce or fight each other.

---

# 41. Profiling requirements

Use Nsight Systems to verify the intended shape:

```text
CPU expert work ██████████████
GPU hit work       ████████
H2D fill             ███
```

overlapping.

The bad profile is:

```text
H2D ████
WAIT     ------
GPU            ████
CPU                ████
```

If we see that, we accidentally recreated the original synchronous streaming problem.

---

# 42. Correctness tests

Required from first prototype:

```text
cache disabled == baseline exact path
0% hit == baseline
100% miss == baseline
forced eviction
fill failure
cuda allocation failure
queue saturation
reload flush
model unload
multiple context lifecycle
```

Numerical:

```text
CPU baseline logits
vs
mixed CPU/GPU logits
```

Expect normal backend floating-point differences, but perplexity and model behavior must remain equivalent.

---

# 43. Phase-by-phase implementation roadmap

## Phase 0 — instrumentation only

No GPU cache yet.

Add routing statistics:

```text
expert frequency
reuse distance
per-layer working set
```

Use real IK workloads to answer:

> Is the cache idea actually worthwhile for the models we care about?

Deliverable:

```text
--moe-expert-trace
```

or debug environment variable.

---

## Phase 1 — minimal one-GPU cache

Support:

```text
CUDA
one device
MUL_MAT_ID
n_tokens = 1
no RTR
manual MiB budget
```

Cache:

```text
fixed shape pools
expert_to_slot
LRU
slot pinning
second-miss admission
```

Execution:

```text
hit GPU
miss normal IQK CPU
```

Fills asynchronous.

No:

```text
fused op
multi-GPU
auto-fit
sibling prefetch
MTP
```

Success criterion:

```text
0% hit → near exact baseline performance
meaningful hit rate → measurable positive TG
```

---

# 44. Phase 2 — fused gate/up

Add:

```text
GGML_OP_MOE_FUSED_UP_GATE
```

A row goes GPU only when all required cached expert data is available.

At this point also implement packed dispatch seriously.

This should be considered part of the first truly performance-relevant implementation because the RFC's later ablation showed fusion contributes approximately as much as caching itself. ([GitHub][1])

---

# 45. Phase 2.5 — sibling promotion

Use routing to enqueue:

```text
up[E]
gate[E]
down[E]
```

intelligently.

For fused gate/up specifically:

```text
fused gate/up executes
      │
      └── immediately enqueue down[E]
```

Measure:

```text
down cache hit rate
PCIe fill amplification
TG
```

Only retain it if it materially helps.

---

# 46. Phase 3 — speculative/MTP

Allow:

```text
n_tokens <= 8
```

with:

```text
<=64 total routed rows
```

or similarly conservative bounds.

Pack many routed rows into one dispatch.

This is where IK could potentially outperform a simple direct RFC port substantially.

---

# 47. Phase 4 — multi-GPU

Create:

```text
device_cache[GPU]
```

and assign pools deterministically.

First avoid P2P entirely.

Host:

```text
RAM → GPU-local pool
```

Once stable, evaluate:

```text
GPU cache peer access
NVLink
P2P
```

as separate research.

---

# 48. Phase 5 — automatic budget

Only after enough real benchmark data exists.

Possible policy:

```text
free VRAM
 - compute reserve
 - graph reserve
 - KV estimate
 - safety margin
 = candidate expert budget
```

Then divide capacity according to discovered pool shapes.

No automatic heuristics before we understand the manual curves.

---

# 49. Phase 6 — RTR/transcoding research

Potentially:

```text
IQK-optimized RAM
      ↓ background conversion
CUDA-optimized VRAM
```

This may be one of the most interesting IK-specific extensions, but it should sit on top of a proven architecture.

---

# 50. Suggested public API

Something along these lines:

```cpp
struct ggml_moe_cache_api {
    bool (*eligible)(
        ggml_moe_cache_session *,
        const ggml_tensor *);

    void * (*begin)(
        ggml_moe_cache_session *,
        const ggml_tensor *);

    bool (*plan)(
        void * node_state,
        const int32_t * ids,
        int n_ids);

    bool (*dispatch)(
        void * node_state);

    bool (*collect)(
        void * node_state);

    void (*end)(
        void * node_state);

    void (*invalidate)(
        ggml_moe_cache_session *,
        const void * base,
        size_t size);
};
```

The exact API can be simpler in IK than upstream llama.cpp because we do not necessarily need backend-pluggability immediately.

But preserving the sequence:

```text
begin
plan
dispatch
collect
end
```

is useful.

It explicitly encodes failure recovery boundaries.

---

# 51. Suggested execution pseudo-code

For ordinary `MUL_MAT_ID`:

```cpp
auto * cs = moe_cache_begin(node);

if (!cs) {
    iqk_mul_mat_id_normal(node);
    return;
}

for (row : routed_rows) {
    auto resident = moe_cache_probe(cs, row.expert);

    if (resident) {
        cs->gpu_rows.push_back(row);
        pin(resident.slot);
    } else {
        cs->cpu_rows.push_back(row);
        maybe_enqueue_fill(row.expert);
    }
}

bool gpu_started = false;

if (!cs->gpu_rows.empty()) {
    gpu_started = moe_cache_dispatch(cs);
    if (!gpu_started) {
        cs->cpu_rows.insert(
            cs->cpu_rows.end(),
            cs->gpu_rows.begin(),
            cs->gpu_rows.end());
        cs->gpu_rows.clear();
    }
}

iqk_compute_rows(cs->cpu_rows);

if (gpu_started) {
    if (!moe_cache_collect(cs)) {
        iqk_compute_rows(cs->gpu_rows);
    }
}

moe_cache_end(cs);
```

That is the architecture in one page.

---

# 52. What I would explicitly not implement

For the initial project, avoid:

```text
persistent disk hot-set profiles
learned routing prediction
per-model cache files
LFU/TinyLFU
partial-expert caching
GPU resident up→down activation handoff
dynamic EWMA bailouts
pageable-vs-pinned automatic heuristics
cross-GPU expert borrowing
partial layer tensor parallelism
```

The RFC's later work is encouraging here: redirect/handoff was only around a 2% isolated contribution and backfill around 1%, while the simple cache and fusion supplied most of the gain. ([GitHub][1])

So there is very little justification for dragging all that machinery into the first IK implementation.

---

# 53. Recommended initial CLI

I would begin with exactly two options:

```text
--moe-vram-cache N
```

MiB, default:

```text
0
```

and perhaps:

```text
--moe-vram-cache-debug
```

Everything else stays internal.

Eventually:

```text
--moe-vram-cache auto
```

can become possible.

But don't expose replacement policy, admission threshold, fill count, stream priority, etc. unless measurements demonstrate that users actually need those controls.

---

# 54. Recommended initial diagnostics

At model load:

```text
moe-vram-cache:
  device            = CUDA0
  budget requested  = 8192 MiB
  budget allocated  = 7936 MiB
  eligible tensors  = 183
  eligible experts  = 11264
  cache slots       = 3472
  coverage          = 30.8 %
  admission         = second-touch
```

At run end:

```text
moe-vram-cache:
  probes        = 182304
  hits          = 127811
  misses        = 54493
  hit rate      = 70.11 %

  gpu rows      = 127811
  cpu rows      = 54493

  fills         = 8124
  dropped       = 32
  evictions     = 4652

  promoted      = 18.7 GiB
  avoided RAM   = 103.5 GiB
  reuse/fill    = 5.53
```

That alone will make experimental development dramatically easier.

---

# 55. Minimum viable patch definition

If I were drawing a hard boundary around the first patch, it would be:

```text
FEATURE:
    persistent VRAM cache for CPU-resident MoE experts

BACKEND:
    CUDA only

NODE:
    GGML_OP_MUL_MAT_ID

TOKENS:
    exactly 1

GPU:
    one device

CACHE:
    explicit MiB budget
    shape-specific fixed slot pools
    direct expert→slot table
    LRU
    slot pinning
    second-touch admission

EXECUTION:
    resident row → CUDA
    nonresident row → existing IQK CPU path

FILLS:
    asynchronous
    bounded queue
    never waited upon by current inference op

LAYOUT:
    CUDA-compatible quantizations only
    no RTR

FAILURE:
    all paths recover to CPU execution

RELOAD:
    successful hot reload flushes cache

PREFETCH:
    existing IK page-cache prefetch unchanged

METRICS:
    comprehensive
```

That is small enough to debug but contains the essential architecture.

---

# 56. The first optimization immediately after MVP

Not multi-GPU.

Not fancy caching.

Not automatic fitting.

It should be:

```text
GGML_OP_MOE_FUSED_UP_GATE
        +
packed CUDA dispatch
```

because the available measurements strongly imply that dispatch efficiency is approximately as important as residency itself.

The settled RFC ablation went:

```text
cache off                     81.55
cache only                    88.12
cache + fusion                96.26
cache + fusion + redirect     98.45
old full implementation       99.26
new packed minimal core      100.21
```

on its stated Qwen3.6-35B test configuration. ([GitHub][1])

That is a beautiful engineering result because it says:

> **Less machinery, better dispatch architecture, higher performance.**

Exactly the direction I would take in IK.

---

# 57. Proposed development branch structure

Rather than one giant experimental commit:

```text
ik-moe-cache/01-routing-telemetry
ik-moe-cache/02-cache-core
ik-moe-cache/03-cuda-slots
ik-moe-cache/04-hybrid-mul-mat-id
ik-moe-cache/05-async-fill
ik-moe-cache/06-fused-up-gate
ik-moe-cache/07-sibling-prefetch
ik-moe-cache/08-mtp
```

Each stage should compile and ideally benchmark independently.

That makes bisecting a performance regression enormously easier.

---

# 58. Final architecture

The mature IK design I would aim for is therefore:

```text
                            MoE router
                                │
                                ▼
                       selected expert IDs
                                │
               ┌────────────────┼─────────────────┐
               │                │                 │
               ▼                ▼                 ▼
        RAM prefetch       cache probe       sibling predictor
         (existing)          runtime            (exact dependency,
               │                │                not ML prediction)
               │       ┌────────┴────────┐        │
               │       │                 │        │
               │      HIT               MISS      │
               │       │                 │        │
               │       ▼                 ▼        │
               │   CUDA queue         IQK CPU     │
               │       │                 │        │
               │       │                 ├────────┘
               │       │                 │
               │       │            async promotion
               │       │                 │
               │       │                 ▼
               │       │           VRAM shape pool
               │       │
               └───────┴──────────────┐
                                      ▼
                              final MoE output
```

The CPU remains the safety net.

VRAM becomes a dynamically learned working set.

PCIe transfers are speculative/promotional rather than mandatory.

Existing IK CPU performance remains valuable instead of being bypassed.

And IK's fused MoE, page prefetch, MTP and IQK machinery all become complementary pieces rather than things the new subsystem has to replace.

That, I think, is the right architectural interpretation of both experiments: **take thecodacus' practical caching/prefetch observations, but put them inside the later RFC's much safer hybrid execution model.** The resulting IK implementation should behave less like "GPU offloading experts" and more like a real heterogeneous memory/cache hierarchy.

[1]: https://github.com/ggml-org/llama.cpp/discussions/24528 "https://github.com/ggml-org/llama.cpp/discussions/24528"
