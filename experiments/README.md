# MoE cache playground

The build recipe consumes our fork. The source experiment lives on
[`moe-cache`](https://github.com/delfianto/ik_llama.cpp/tree/moe-cache).
A reviewable, portable backup is [moe-cache-v1.patch](../patches/moe-cache-v1.patch).
The [results and support boundary](moe-cache-results.md) describe what actually passed.

Install the Python runtime and development dependencies with `uv sync --locked`.
Run `just check-python` for Ruff lint/format checks and basedpyright type checks.
Use `uv run python experiments/SCRIPT.py ...` to run a script in that environment;
Matplotlib is required by the summarizer. Tool settings live in the root
`pyproject.toml`, and `uv.lock` pins the resolved dependencies.

Keep source development outside the disposable Docker source export. For example:

```bash
git clone --branch moe-cache https://github.com/delfianto/ik_llama.cpp.git .cache/experiments/ik_llama.cpp
CUDAFLAGS=--allow-unsupported-compiler CUDACXX=/opt/cuda/bin/nvcc \
  experiments/build-native.sh .cache/experiments/ik_llama.cpp .cache/experiments/build-cuda cuda
ctest --test-dir .cache/experiments/build-cuda -R '^test-moe-cache$' --output-on-failure
```

The compiler override above is needed by this machine's GCC 16 / CUDA 13.4 combination.
The Docker toolchain uses its own compiler. For model downloads, use the pinned
[manifest](moe-models.json). The ranged downloader requires the expected file size:

```bash
python experiments/download-model.py PINNED_HF_RESOLVE_URL .cache/experiments/models/model.gguf --bytes EXACT_BYTES
```

Check the final SHA256 against the manifest. Its range journal records completed
chunks; keep the journal with the partially downloaded file when resuming.

Run the model numerical gate on these two GPUs:

```bash
python experiments/validate-moe.py \
  --binary .cache/experiments/build-cuda/bin/test-moe-cache-model \
  --model-dir .cache/experiments/models --output .cache/experiments/validation
```

Run sequential benchmarks with fixed real tokens, 256 generated tokens, three
repetitions and opt-in streaming TTFT measurement:

```bash
python experiments/benchmark-moe.py --engine ik \
  --server .cache/experiments/build-cuda/bin/llama-server \
  --source .cache/experiments/ik_llama.cpp \
  --model .cache/experiments/models/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf \
  --output .cache/experiments/results/qwen.jsonl --stream-ttft
```

Defaults sweep budgets 0/512/1024/2048/4096 MiB, lengths 512/2048/8192, both GPUs
individually and as a 4:3 layer split, and prose/code/math/multilingual prompts.
This is a substantial sweep. For a quick check use `--lengths 512 --domains prose`.
Run engines sequentially to avoid GPU contention; use distinct output paths.

`--engine ik-baseline` needs an unmodified checkout/build and budget zero.
`--engine upstream` needs llama.cpp at `c811cb8f0ac91b8ac72a32f970bdd45037f20da7`.
Upstream's CLI cannot force the same IK active-expert GPU policy, so its zero-cache
hybrid control must be interpreted with that scheduling difference in mind.
`--placement active` forces IK GPU expert computation with RAM-backed weights;
`hybrid` keeps each engine's ordinary offload heuristics; `full` and `cpu` are
cache-zero residency controls. Hold source, GGUF, KV types and command flags fixed.

Each case starts a new server/context. The first request is labelled cold and
subsequent requests warm/domain-mixed. This does not flush the OS page cache.
Model loading, request latency, prompt/decode rates, output length, source SHA,
compiler flags, GGUF hash, memory samples and server cache counters are retained.
Memory samples include other GPU processes and are sampled every 0.5 seconds;
they are not exact allocator peaks. These runs compare the same weight placement,
not equal total VRAM. Use separately specified placements to make that comparison.

Use `--prompt-style chat --save-output` to render each server's chat template and
retain generated text. Repeated material is removed from the middle to preserve
role delimiters and the assistant prefix at the exact target token count.
Completion-style and chat-style results are different workloads; compare them
separately. Output hashes/previews help spot routing changes or repetitive output.

```bash
python experiments/summarize-moe.py --results experiments/results
```

This produces `summary.csv` and standalone `cache-sweep.svg` / `cache-sweep.png`. The chart shows
only the initial prose sweeps; each warm mean uses two requests after one cold
request. Raw JSONL retains all three measurements and every command.
