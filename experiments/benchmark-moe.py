#!/usr/bin/env python3
"""Sequential real-prompt MoE cache sweep; every case gets a fresh server/context.

Example: python experiments/benchmark-moe.py --engine ik --server PATH \
  --source PATH --model PATH --output experiments/results/run.jsonl
Equal placement is measured here. Equal-total-VRAM placement requires separate
runs with explicit --extra arguments, because engines allocate different buffers.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict


class BenchmarkArgs(argparse.Namespace):
    engine: str
    server: Path
    source: Path
    model: Path
    output: Path
    append: bool
    placement: str
    budgets: str
    lengths: str
    devices: str
    domains: str
    repetitions: int
    generate: int
    threads: int
    port: int
    prompt_style: str
    save_output: bool
    stream_ttft: bool
    extra: list[str]


class MemorySample(TypedDict):
    at: float
    gpu: str
    rss_kib: str


# HTTP endpoints and JSONL records have heterogeneous, server-specific fields.
JSONObject = dict[str, Any]


PROMPTS = {
    "prose": "Summarize this proposal: reuse frequently selected experts across tokens, track misses and test changes with controlled experiments. Discuss tradeoffs. ",
    "code": "Write Python and SQL to group orders by customer, count distinct products, handle missing data, and add an index. Explain time complexity. ",
    "math": "Solve a probability problem with independent events. Derive the formula and show numerical examples. Explain why the assumptions matter. ",
    "multilingual": "Jelaskan cara kerja cache dan pengaruhnya pada kecepatan inferensi. Compare it with an LRU cache. Expliquez aussi les compromis en français. ",
}


def command_output(command: Sequence[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def request(
    base: str, path: str, payload: JSONObject | None = None, timeout: float = 120
) -> JSONObject:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def completion(
    base: str, payload: JSONObject, stream: bool
) -> tuple[JSONObject, float | None]:
    if not stream:
        return request(base, "/completion", payload, timeout=1800), None
    payload = dict(payload, stream=True)
    req = urllib.request.Request(
        base + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = None
    content: list[str] = []
    final: JSONObject = {}
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("content"):
                if first is None:
                    first = time.monotonic() - start
                content.append(event["content"])
            final = event
    final["content"] = "".join(content)
    return final, first


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def monitor_memory(
    pid: int, stop: threading.Event, samples: list[MemorySample]
) -> None:
    while not stop.is_set():
        try:
            samples.append(
                {
                    "at": time.time(),
                    "gpu": command_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used",
                            "--format=csv,noheader,nounits",
                        ]
                    ),
                    "rss_kib": command_output(["ps", "-o", "rss=", "-p", str(pid)]),
                }
            )
        except subprocess.CalledProcessError:
            pass  # The process may exit between memory samples.
        stop.wait(0.5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["ik", "ik-baseline", "upstream"], required=True)
    p.add_argument("--server", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--append",
        action="store_true",
        help="explicitly append a new sweep to an existing result file",
    )
    p.add_argument(
        "--placement", choices=["active", "hybrid", "full", "cpu"], default="active"
    )
    p.add_argument("--budgets", default="0,512,1024,2048,4096")
    p.add_argument("--lengths", default="512,2048,8192")
    p.add_argument("--devices", default="0;1;0,1")
    p.add_argument("--domains", default="prose,code,math,multilingual")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--generate", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--port", type=int, default=18981)
    p.add_argument(
        "--prompt-style", choices=["completion", "chat"], default="completion"
    )
    p.add_argument("--save-output", action="store_true")
    p.add_argument(
        "--stream-ttft",
        action="store_true",
        help="measure time to first nonempty generated SSE chunk",
    )
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = BenchmarkArgs()
    p.parse_args(namespace=args)
    if args.output.exists() and args.output.stat().st_size and not args.append:
        p.error("result file already exists; choose a new path or --append")
    budgets = [int(x) for x in args.budgets.split(",")]
    if args.placement in ("full", "cpu") and any(budgets):
        p.error("full/CPU controls require budget zero")
    if args.engine == "ik-baseline" and any(budgets):
        p.error("baseline supports budget zero only")
    lengths = [int(x) for x in args.lengths.split(",")]
    domains = args.domains.split(",")
    cases = list(itertools.product(args.devices.split(";"), budgets))
    random.Random(42).shuffle(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_cache = args.server.resolve().parent.parent / "CMakeCache.txt"
    build_flags = (
        [
            line
            for line in build_cache.read_text().splitlines()
            if line.startswith(
                (
                    "CMAKE_BUILD_TYPE:",
                    "CMAKE_CUDA_ARCHITECTURES:",
                    "CMAKE_CUDA_FLAGS:",
                    "GGML_NATIVE:",
                    "GGML_CUDA_FA_ALL_QUANTS:",
                    "CMAKE_CXX_COMPILER:",
                )
            )
        ]
        if build_cache.exists()
        else []
    )
    metadata = {
        "runner_sha256": sha256(Path(__file__)),
        "recipe_sha": command_output(["git", "rev-parse", "HEAD"]),
        "recipe_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "build_flags": build_flags,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": args.engine,
        "source_sha": command_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"]
        ),
        "source_dirty": bool(
            command_output(["git", "-C", str(args.source), "status", "--porcelain"])
        ),
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "protocol": "same-placement; fresh context per case; OS page cache uncontrolled; server warmup disabled",
        "hardware": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv",
            ]
        ),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "cpu": command_output(["lscpu"]),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    failures = 0
    with args.output.open("a") as results:
        results.write(json.dumps({"metadata": metadata}) + "\n")
        results.flush()
        for case_index, (devices, budget) in enumerate(cases):
            # Some server HTTP stacks cannot rebind a socket still in TIME_WAIT.
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", args.port + case_index))
                except OSError:
                    probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            tag = f"{args.engine}-gpu{devices.replace(',', '_')}-cache{budget}"
            log = args.output.with_name(args.output.stem + "-" + tag + ".txt")
            command = [
                str(args.server.resolve()),
                "-m",
                str(args.model.resolve()),
                "-c",
                str(max(lengths) + args.generate + 256),
                "-ngl",
                "0" if args.placement == "cpu" else "999",
                "-t",
                str(args.threads),
                "-b",
                "512",
                "-ub",
                "512",
                "--parallel",
                "1",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-warmup",
            ]
            if args.placement in ("active", "hybrid"):
                command += ["--cpu-moe"]
            if args.engine == "upstream":
                command += ["-lv", "4"]
            if args.engine != "upstream":
                command += ["-no-fug"]
                if args.placement == "active":
                    command += ["-cuda", "offload-batch-size=0"]
            if args.engine != "ik-baseline":
                command += ["--moe-cache-mib", str(budget)]
            if "," in devices:
                command += ["-sm", "layer", "-ts", "4,3"]
            command += args.extra
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = devices
            env.pop("LLAMA_ARG_MOE_CACHE_MIB", None)
            samples: list[MemorySample] = []
            stop = threading.Event()

            started = time.monotonic()
            with log.open("w") as stderr:
                server = subprocess.Popen(
                    command, env=env, stdout=stderr, stderr=stderr
                )
                watcher = threading.Thread(
                    target=monitor_memory, args=(server.pid, stop, samples), daemon=True
                )
                watcher.start()
                try:
                    deadline = time.monotonic() + 600
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(
                                f"server exited {server.returncode}: {log}"
                            )
                        try:
                            if (
                                request(base, "/health", timeout=2).get("status")
                                == "ok"
                            ):
                                break
                        except (urllib.error.URLError, TimeoutError):
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError("model load timed out")
                        time.sleep(0.5)
                    load_seconds = time.monotonic() - started
                    jobs = list(
                        itertools.product(range(args.repetitions), lengths, domains)
                    )
                    random.Random(42).shuffle(jobs)
                    for index, (rep, length, domain) in enumerate(jobs):
                        text = PROMPTS[domain] * (length // 16 + 100)
                        if args.prompt_style == "chat":
                            text = request(
                                base,
                                "/apply-template",
                                {"messages": [{"role": "user", "content": text}]},
                            )["prompt"]
                        full_tokens = request(
                            base, "/tokenize", {"content": text, "add_special": True}
                        )["tokens"]
                        # Remove repeated material from the middle to keep chat role delimiters and the assistant prefix.
                        tokens = (
                            full_tokens[: length // 2]
                            + full_tokens[-(length - length // 2) :]
                            if args.prompt_style == "chat"
                            else full_tokens[:length]
                        )
                        if len(tokens) != length:
                            raise RuntimeError("prompt corpus too short")
                        before = len(samples)
                        begin = time.monotonic()
                        response, ttft = completion(
                            base,
                            {
                                "prompt": tokens,
                                "n_predict": args.generate,
                                "temperature": 0,
                                "seed": 42,
                                "ignore_eos": True,
                                "cache_prompt": False,
                            },
                            args.stream_ttft,
                        )
                        row: JSONObject = {
                            "case": tag,
                            "devices": devices,
                            "budget_mib": budget,
                            "command": command,
                            "load_seconds": load_seconds,
                            "context_state": "cold"
                            if index == 0
                            else "warm/domain-mixed",
                            "repetition": rep,
                            "domain": domain,
                            "prompt_tokens": length,
                            "prompt_sha256": hashlib.sha256(
                                json.dumps(tokens).encode()
                            ).hexdigest(),
                            "latency_seconds": time.monotonic() - begin,
                            "ttft_seconds": ttft,
                            "timings": response.get("timings"),
                            "tokens_predicted": response.get("tokens_predicted"),
                            "output_preview": response.get("content", "")[:400],
                            "output_characters": len(response.get("content", "")),
                            "stop_type": response.get("stop_type"),
                            "memory_samples": samples[before:],
                            "output_sha256": hashlib.sha256(
                                response.get("content", "").encode()
                            ).hexdigest(),
                        }
                        if args.save_output:
                            row["output"] = response.get("content", "")
                        results.write(json.dumps(row) + "\n")
                        results.flush()
                        print(tag, rep, length, domain, row["timings"], flush=True)
                except Exception as error:
                    failures += 1
                    results.write(
                        json.dumps(
                            {"case": tag, "error": str(error), "command": command}
                        )
                        + "\n"
                    )
                    results.flush()
                    print(tag, error, flush=True)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
                    stop.set()
                    watcher.join(timeout=2)
                    results.write(
                        json.dumps(
                            {
                                "case": tag,
                                "all_memory_samples": samples,
                                "log": str(log),
                            }
                        )
                        + "\n"
                    )
                    results.flush()

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
