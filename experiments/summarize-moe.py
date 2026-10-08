#!/usr/bin/env python3
"""Create auditable summary CSV and a standalone SVG from recorded JSONL runs."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


class SummaryArgs(argparse.Namespace):
    results: Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("experiments/results"))
    args = SummaryArgs()
    p.parse_args(namespace=args)
    records: list[dict[str, Any]] = []
    for file in sorted(args.results.glob("*.jsonl")):
        metadata: dict[str, Any] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        content = file.read_text()
        lines = content.splitlines()
        if content and not content.endswith("\n"):
            lines = lines[:-1]
        for line in lines:
            row = json.loads(line)
            if "metadata" in row:
                metadata = row["metadata"]
                continue
            if row.get("timings"):
                grouped.setdefault(row["case"], []).append(row)
        for case, samples in grouped.items():
            warm = [row for row in samples if row["context_state"] != "cold"]
            cold = [row for row in samples if row["context_state"] == "cold"]

            def mean(rows: list[dict[str, Any]], key: str) -> float | None:
                values = [row["timings"][key] for row in rows]
                return statistics.mean(values) if values else None

            def timing(rows: list[dict[str, Any]], key: str) -> float | None:
                values = [
                    float(value) for row in rows if (value := row.get(key)) is not None
                ]
                return statistics.mean(values) if values else None

            first = samples[0]
            log_paths = [
                args.results / (file.stem + "-" + case + suffix)
                for suffix in (".txt", ".log")
            ]
            text = next((f.read_text() for f in log_paths if f.exists()), "")
            ik_stats = re.findall(
                r"group hits=(\d+) misses=(\d+) uploaded_bytes=(\d+)", text
            )
            upstream_stats = re.findall(
                r"llama_moe_cache: ubatch\s*<=\s*8: hits = (\d+), misses = (\d+).*uploaded = ([0-9.]+) MiB",
                text,
            )
            hits = misses = uploaded = 0
            for h, m, size in ik_stats:
                hits += int(h)
                misses += int(m)
                uploaded += int(size)
            if not ik_stats:
                for h, m, size in upstream_stats:
                    hits += int(h)
                    misses += int(m)
                    uploaded += float(size) * 1024 * 1024
            bank = sum(
                float(size)
                for size in re.findall(r"MoE cache size =\s*([0-9.]+) MiB", text)
            )
            row = {
                "run": file.stem,
                "case": case,
                "engine": metadata.get("engine"),
                "model": Path(metadata.get("model", "")).name,
                "source_sha": metadata.get("source_sha"),
                "placement": metadata.get("args", {}).get("placement", "active"),
                "devices": first["devices"],
                "budget_mib": first["budget_mib"],
                "bank_mib": bank if bank else None,
                "requests": len(samples),
                "cold_requests": len(cold),
                "warm_requests": len(warm),
                "cold_decode_ts": mean(cold, "predicted_per_second"),
                "warm_decode_ts": mean(warm, "predicted_per_second"),
                "warm_prefill_ts": mean(warm, "prompt_per_second"),
                "cold_ttft_seconds": timing(cold, "ttft_seconds"),
                "warm_ttft_seconds": timing(warm, "ttft_seconds"),
                "warm_request_seconds": timing(warm, "latency_seconds"),
                "hits": hits or None,
                "misses": misses or None,
                "hit_rate": hits / (hits + misses) if hits + misses else None,
                "uploaded_mib": uploaded / 1024 / 1024 if hits + misses else None,
                "log_stats_available": bool(ik_stats or upstream_stats),
            }
            records.append(row)
    if not records:
        raise SystemExit("No successful requests found")
    with (args.results / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # This figure shows only repeated-prose sweeps. Domain-mixed/chat runs stay in CSV/raw data.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for mi, (prefix, title) in enumerate(
        [("granite", "Granite 3.1 3B-A800M Q4_K_M"), ("qwen", "Qwen3-30B-A3B Q4_0")]
    ):
        for gi, (device, gpu) in enumerate(
            [("0", "RTX 4080"), ("1", "RTX 3060"), ("0,1", "4080 + 3060 (4:3)")]
        ):
            ax = axes[mi, gi]
            for name, label, color in [
                (prefix + "-ik", "IK cache port", "#2878b5"),
                (
                    prefix + "-upstream-v2"
                    if prefix == "granite"
                    else prefix + "-upstream",
                    "llama.cpp cache",
                    "#d96a28",
                ),
            ]:
                rows = sorted(
                    [
                        r
                        for r in records
                        if r["run"] == name
                        and r["devices"] == device
                        and r["budget_mib"] > 0
                    ],
                    key=lambda r: r["budget_mib"],
                )
                if rows:
                    ax.plot(
                        [r["budget_mib"] for r in rows],
                        [r["warm_decode_ts"] for r in rows],
                        marker="o",
                        label=label,
                        color=color,
                    )
            for suffix, label, style, color in [
                ("baseline-hybrid", "IK ordinary hybrid", "--", "#666666"),
                ("baseline-full", "IK full GPU", ":", "#339966"),
                ("upstream-full", "llama.cpp full GPU", "-.", "#aa55aa"),
            ]:
                rows = [
                    r
                    for r in records
                    if r["run"] == prefix + "-" + suffix and r["devices"] == device
                ]
                if rows:
                    ax.axhline(
                        rows[0]["warm_decode_ts"],
                        linestyle=style,
                        color=color,
                        label=label,
                    )
            ax.set_title(title + "\n" + gpu, fontsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.set_xticks([512, 1024, 2048, 4096], ["512", "1024", "2048", "4096"])
            if gi == 0:
                ax.set_ylabel("Warm decode tokens/s")
            if mi == 1:
                ax.set_xlabel("Requested total cache MiB")
            if not ax.has_data():
                ax.text(0.5, 0.5, "Not measured", transform=ax.transAxes, ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(
        "Initial repeated-prose experiment: 512 prompt tokens + 256 output tokens\nWarm mean: two requests following one cold request; weight placement and engine kernels differ",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(args.results / "cache-sweep.svg")
    fig.savefig(args.results / "cache-sweep.png", dpi=140)
    print(f"Wrote {len(records)} summary rows and {args.results}/cache-sweep.svg")


if __name__ == "__main__":
    main()
