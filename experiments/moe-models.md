# MoE model playground

Researched 2026-10-08 for Ryzen 9950X + RTX 4080 16 GiB + RTX 3060 12 GiB,
with 60 GiB usable host RAM. These are candidates, not tested compatibility or
performance claims. Model sizes below are downloaded GGUF size in decimal GB,
not total inference memory. Dense/shared weights, cache banks, KV/state,
compute buffers and existing GPU applications need additional room.

Start with **Granite** to debug the cache and **Qwen3-30B-A3B** to see whether it
helps a practical hybrid workload. Force host experts plus deliberately small
cache budgets for cache tests; tiny models otherwise fit entirely on GPU and
would hide the behavior we want to measure.

| Model | Routed experts / active per MoE layer | Initial file | GB | Experiment |
| --- | --- | --- | ---: | --- |
| Granite 3.1 3B-A800M Instruct | 40 / 8 | Q4_K_M | 2.02 | Quick numerical tests, forced evictions, tiny-cache overhead; possible CPU sidecar. |
| Qwen3-30B-A3B-Instruct-2507 | 128 / 8 | Q4_0, then Q4_K_M | 17.63 / 18.63 | First substantial target: conventional attention, many small experts, practical coding/chat prompts. |
| DeepSeek-Coder-V2-Lite-Instruct | 64 / 6, plus 2 shared | Q4_K_M | 10.36 | MLA and shared-expert handling; Python/SQL domain locality. |
| GPT-OSS-20B | 32 / 4 | native MXFP4 | 12.11 | Different weight format and biased/fused experts; compare SWA/context effects after bias support. |
| LFM2.5-8B-A1B | 32 / 4 | Q4_0, then Q4_K_M | 4.84 / 5.16 | Small hybrid conv/attention model; low-overhead routing/cache tests and a useful assistant candidate. |
| GLM-4.7-Flash | 64 / 4, plus 1 shared | Q4_0, then Q4_K_M | 17.39 / 18.47 | Another practical coding model, MLA/shared experts; architecture portability check. |
| Qwen3.5-35B-A3B | 256 / 8 | Q4_0, then Q4_K_M | 20.84 / 22.29 | Hybrid DeltaNet/attention, larger routing space; later MTP compatibility and cache interaction. |

Expert counts come from the original model configurations, not active parameter
counts in model names. An “A3B” model still stores all its experts. An available
GGUF and an IK architecture entry do not establish that this exact revision loads
correctly: run a cache-disabled loader/logit smoke test first. GPT-OSS and LFM2.5
are later compatibility targets because their expert bias/layout paths need review.
OLMoE was considered but is not an initial candidate: the inspected IK architecture
registry has no `olmoe` entry, so it would risk bundling another model port into this one.

## Sources and exact download candidates

- Granite: [IBM model card](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-instruct),
  [GGUF files](https://huggingface.co/bartowski/granite-3.1-3b-a800m-instruct-GGUF).
- Qwen3: [Qwen model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507),
  [GGUF files](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF).
- DeepSeek: [model card](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct),
  [GGUF files](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF).
- GPT-OSS: [model card](https://huggingface.co/openai/gpt-oss-20b),
  [GGUF files](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF).
- LFM2.5: [Liquid AI model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B),
  [official GGUF files](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF).
- GLM: [Z.ai model card](https://huggingface.co/zai-org/GLM-4.7-Flash),
  [GGUF files](https://huggingface.co/bartowski/zai-org_GLM-4.7-Flash-GGUF).
- Qwen3.5: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B),
  [GGUF files](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF).

The [machine-readable shortlist](moe-models.json) pins repository revisions and
exact filenames for the initial file of each model. Read the repository's own
license/model card when choosing models. Granite and Qwen3 were downloaded; exact sizes and SHA256 hashes are recorded in the manifest. Other candidates remain unvalidated.

## Experiments to try in order

1. **Make it thrash:** Granite with a cache just above one token's expert set,
   then larger pools. Check repeated requests, cross-layer borrowing, evictions
   and whether bookkeeping costs more than CPU compute on this small model.
2. **Find the knee:** Qwen3 on 4080 alone, then 4080 + 3060, with experts in RAM.
   Sweep 512 MiB to 4 GiB total cache and plot decode rate against hit rate and
   uploaded bytes. Compare the two-GPU full-residency control when it fits.
3. **Change domains mid-stream:** alternate coding, prose and Indonesian prompts;
   test whether warm experts remain useful or the cache churns between requests.
4. **Change architecture:** DeepSeek/GLM for MLA/shared experts, then GPT-OSS for
   MXFP4/SWA/bias paths and LFM2.5 for its small hybrid model.
5. **Add speculation:** Qwen3.5 with MTP disabled first, then enabled after the
   cache supports verification batches and separate context budgeting. Measure
   accepted output tokens/sec as well as raw target/draft work and hit rates.

Detailed controls, correctness gates and result fields are in the
[port and benchmark plan](../patches/moe-cache-port-plan.md).
