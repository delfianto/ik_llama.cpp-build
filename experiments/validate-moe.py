#!/usr/bin/env python3
"""Run the full-model cache gate across the two local CUDA GPUs and splits."""

import argparse
import json
import os
import subprocess
from pathlib import Path


class ValidationArgs(argparse.Namespace):
    binary: Path
    model_dir: Path
    output: Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = ValidationArgs()
    p.parse_args(namespace=args)
    args.output.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, str, str, int, list[str], int]] = [
        ("granite-gpu0", "0", "granite", 512, [], 256),
        ("granite-gpu1", "1", "granite", 512, [], 256),
        ("granite-pair4-3", "0,1", "granite", 512, ["-sm", "layer", "-ts", "4,3"], 256),
        ("granite-equal", "0,1", "granite", 512, ["-sm", "layer", "-ts", "1,1"], 64),
        ("granite-skew", "0,1", "granite", 512, ["-sm", "layer", "-ts", "1,4"], 64),
        (
            "granite-zero-share",
            "0,1",
            "granite",
            512,
            ["-sm", "layer", "-ts", "1,0"],
            64,
        ),
        (
            "granite-graphs-off",
            "0",
            "granite",
            512,
            ["-cuda", "offload-batch-size=0,graphs=0"],
            64,
        ),
        ("granite-unfused", "0", "granite", 512, ["-no-fmoe"], 64),
        ("qwen-gpu0", "0", "qwen", 1024, [], 256),
        ("qwen-pair4-3", "0,1", "qwen", 1024, ["-sm", "layer", "-ts", "4,3"], 256),
        ("tiny-budget", "0", "granite", 1, [], 16),
        ("graph-split-rejected", "0,1", "qwen", 512, ["-sm", "graph"], 16),
    ]
    names = {
        "granite": "granite-3.1-3b-a800m-instruct-Q4_K_M.gguf",
        "qwen": "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf",
    }
    results = []
    for name, devices, model, budget, extra, count in cases:
        command = [
            str(args.binary.resolve()),
            "-m",
            str((args.model_dir / names[model]).resolve()),
            "-ngl",
            "999",
            "--cpu-moe",
            "-no-fug",
            "--moe-cache-mib",
            str(budget),
            "-cuda",
            "offload-batch-size=0",
            "-t",
            "16",
            "-n",
            str(count),
        ] + extra
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices
        env.pop("LLAMA_ARG_MOE_CACHE_MIB", None)
        run = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=600, check=False
        )
        log = args.output / (name + ".txt")
        log.write_text(run.stderr + run.stdout)
        result = next(
            (
                json.loads(line)
                for line in reversed(run.stdout.splitlines())
                if line.startswith("{")
            ),
            None,
        )
        rejected = name in ("tiny-budget", "graph-split-rejected")
        reason = (
            "too small to hold the experts of one token"
            if name == "tiny-budget"
            else "only CUDA none/layer split modes"
        )
        passed = (
            (run.returncode != 0 and reason in run.stderr)
            if rejected
            else run.returncode == 0 and result and result["passed"]
        )
        row = {
            "case": name,
            "devices": devices,
            "command": command,
            "returncode": run.returncode,
            "result": result,
            "expected_rejection": rejected,
            "passed": bool(passed),
            "log": str(log),
        }
        results.append(row)
        (args.output / "validation.json").write_text(
            json.dumps(
                {"threshold_nmse": 5e-4, "cache_batch_limit": 1, "cases": results},
                indent=2,
            )
            + "\n"
        )
        print(name, "PASS" if passed else "FAIL", result, flush=True)
    if not all(row["passed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
