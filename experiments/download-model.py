#!/usr/bin/env python3
"""Download a pinned manifest model with checked HTTP ranges and resumable chunks."""

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path


class DownloadArgs(argparse.Namespace):
    url: str
    output: Path
    bytes: int
    workers: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = DownloadArgs()
    parser.parse_args(namespace=args)
    chunk = 128 * 1024 * 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = args.output.with_suffix(args.output.suffix + ".ranges.json")
    done: set[int] = set()
    if state.exists():
        if not args.output.exists():
            raise SystemExit("Range journal exists but the data file is missing")
        saved = json.loads(state.read_text())
        if not isinstance(saved, dict) or (
            saved.get("url"),
            saved.get("bytes"),
            saved.get("chunk"),
        ) != (args.url, args.bytes, chunk):
            raise SystemExit(
                "Range journal does not match this URL/size/chunk; use a new output path"
            )
        if args.output.stat().st_size != args.bytes:
            raise SystemExit("Partial file size does not match the range journal")
        done = set(saved["done"])
    fd = os.open(args.output, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, args.bytes)

    def fetch(index: int) -> int:
        start = index * chunk
        stop = min(start + chunk, args.bytes) - 1
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    args.url, headers={"Range": f"bytes={start}-{stop}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    expected = f"bytes {start}-{stop}/{args.bytes}"
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != expected
                    ):
                        raise RuntimeError(
                            f"Unexpected range response: {response.status}, {response.headers.get('Content-Range')}"
                        )
                    offset = start
                    while data := response.read(4 * 1024 * 1024):
                        if offset + len(data) > stop + 1:
                            raise RuntimeError("Response exceeded requested range")
                        written = 0
                        while written < len(data):
                            written += os.pwrite(fd, data[written:], offset + written)
                        offset += len(data)
                    if offset != stop + 1:
                        raise RuntimeError("Truncated range")
                return index
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Download retries exhausted")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [
                pool.submit(fetch, i)
                for i in range((args.bytes + chunk - 1) // chunk)
                if i not in done
            ]
            for future in concurrent.futures.as_completed(pending):
                done.add(future.result())
                temporary = state.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "url": args.url,
                            "bytes": args.bytes,
                            "chunk": chunk,
                            "done": sorted(done),
                        }
                    )
                )
                temporary.replace(state)
                print(
                    f"{len(done)}/{(args.bytes + chunk - 1) // chunk} ranges",
                    flush=True,
                )
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
