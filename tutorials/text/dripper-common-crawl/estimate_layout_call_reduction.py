# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Estimate Dripper LLM-call reduction from global host/layout grouping.

This script is deliberately CPU-only.  It scans one or more host-clustered
manifest parquet files and estimates how many LLM representative calls would be
required if pages were grouped globally by:

* full URL host
* full URL host + a cheap URL-shape signature

The URL-shape signature is a proxy for the later DOM-layout clustering stage.
It is not a replacement for llm-webkit's DBSCAN DOM clustering, but it gives a
fast upper-bound sanity check on whether large call reduction is plausible.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Dripper representative-call reduction")
    parser.add_argument("--input", required=True, help="Manifest parquet file, directory, or glob")
    parser.add_argument("--output", required=True, help="Output JSON metrics path")
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all matching files")
    parser.add_argument("--workers", type=int, default=1, help="Number of manifest files to scan concurrently")
    parser.add_argument(
        "--host-bucket-groups",
        default=None,
        help="Optional comma/range filter over host_bucket_group values in file names, e.g. 0,7,10-19.",
    )
    parser.add_argument(
        "--representative-min-group-pages",
        default="2,4,8,16",
        help="Comma-separated group-size thresholds for call-ratio estimates.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_files < 0:
        raise ValueError("--max-files must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    return args


def main() -> int:
    args = parse_args()
    manifest_files = resolve_manifest_files(args.input, parse_int_ranges(args.host_bucket_groups))
    if args.max_files:
        manifest_files = manifest_files[: args.max_files]
    if not manifest_files:
        raise FileNotFoundError(f"No manifest parquet files matched {args.input!r}")

    thresholds = sorted({int(value) for value in args.representative_min_group_pages.split(",") if value.strip()})
    if any(value <= 1 for value in thresholds):
        raise ValueError("--representative-min-group-pages values must be greater than 1")

    total_rows = 0
    total_bytes = 0
    total_hosts = 0
    total_url_shape_groups = 0
    host_size_hist: Counter[int] = Counter()
    url_shape_size_hist: Counter[int] = Counter()
    file_metrics: list[dict[str, Any]] = []

    for file_index, path, file_result in iter_manifest_results(
        manifest_files,
        batch_size=args.batch_size,
        workers=args.workers,
    ):
        file_metrics.append(file_result)
        total_rows += file_result["rows"]
        total_bytes += file_result["bytes"]
        total_hosts += file_result["hosts"]
        total_url_shape_groups += file_result["host_url_shape_groups"]
        host_size_hist.update({int(k): int(v) for k, v in file_result["host_size_hist"].items()})
        url_shape_size_hist.update({int(k): int(v) for k, v in file_result["host_url_shape_size_hist"].items()})

    metrics = {
        "input": args.input,
        "files": [str(path) for path in manifest_files],
        "file_count": len(manifest_files),
        "bytes": total_bytes,
        "rows": total_rows,
        "hosts": total_hosts,
        "host_url_shape_groups": total_url_shape_groups,
        "host_call_ratio": safe_ratio(total_hosts, total_rows),
        "host_reduction_factor": safe_ratio(total_rows, total_hosts),
        "host_url_shape_call_ratio": safe_ratio(total_url_shape_groups, total_rows),
        "host_url_shape_reduction_factor": safe_ratio(total_rows, total_url_shape_groups),
        "host_size_quantiles": histogram_quantiles(host_size_hist),
        "host_url_shape_size_quantiles": histogram_quantiles(url_shape_size_hist),
        "host_size_buckets": size_buckets(host_size_hist),
        "host_url_shape_size_buckets": size_buckets(url_shape_size_hist),
        "representative_min_group_pages": thresholds,
        "representative_call_estimates": {
            str(threshold): representative_call_metrics(url_shape_size_hist, total_rows, threshold)
            for threshold in thresholds
        },
        "file_metrics": file_metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print("CALL_REDUCTION_ESTIMATE_BEGIN")
    print(json.dumps({k: v for k, v in metrics.items() if k != "file_metrics"}, indent=2, sort_keys=True))
    print("CALL_REDUCTION_ESTIMATE_END")
    print(f"OUTPUT={output_path}")
    return 0


def iter_manifest_results(
    manifest_files: list[Path],
    *,
    batch_size: int,
    workers: int,
) -> Iterable[tuple[int, Path, dict[str, Any]]]:
    worker_count = min(workers, len(manifest_files))
    if worker_count <= 1:
        for file_index, path in enumerate(manifest_files):
            print(f"ESTIMATE_FILE_BEGIN index={file_index} path={path}", flush=True)
            result = scan_manifest_file(path, batch_size=batch_size)
            print_file_result(file_index, result)
            yield file_index, path, result
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for file_index, path in enumerate(manifest_files):
            print(f"ESTIMATE_FILE_BEGIN index={file_index} path={path}", flush=True)
            futures[executor.submit(scan_manifest_file, path, batch_size=batch_size)] = (file_index, path)
        for future in as_completed(futures):
            file_index, path = futures[future]
            result = future.result()
            print_file_result(file_index, result)
            yield file_index, path, result


def print_file_result(file_index: int, file_result: dict[str, Any]) -> None:
    print(
        "ESTIMATE_FILE_END "
        f"index={file_index} rows={file_result['rows']} hosts={file_result['hosts']} "
        f"host_url_shape_groups={file_result['host_url_shape_groups']} "
        f"shape_reduction={file_result['host_url_shape_reduction_factor']:.3f}",
        flush=True,
    )


def scan_manifest_file(path: Path, *, batch_size: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    schema_names = set(parquet_file.schema_arrow.names)
    missing = sorted({"url", "url_host_name"}.difference(schema_names))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    host_counts: Counter[str] = Counter()
    host_shape_counts: Counter[int] = Counter()
    rows = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["url", "url_host_name"], use_threads=True):
        data = batch.to_pydict()
        urls = data["url"]
        hosts = data["url_host_name"]
        rows += len(urls)
        for url_value, host_value in zip(urls, hosts, strict=True):
            host = normalize_host(host_value)
            if not host:
                continue
            host_counts[host] += 1
            shape = url_shape_key(url_value)
            host_shape_counts[stable_group_hash(host, shape)] += 1

    host_hist = Counter(host_counts.values())
    shape_hist = Counter(host_shape_counts.values())
    host_shape_groups = len(host_shape_counts)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "hosts": len(host_counts),
        "host_url_shape_groups": host_shape_groups,
        "host_call_ratio": safe_ratio(len(host_counts), rows),
        "host_reduction_factor": safe_ratio(rows, len(host_counts)),
        "host_url_shape_call_ratio": safe_ratio(host_shape_groups, rows),
        "host_url_shape_reduction_factor": safe_ratio(rows, host_shape_groups),
        "host_size_quantiles": histogram_quantiles(host_hist),
        "host_url_shape_size_quantiles": histogram_quantiles(shape_hist),
        "host_size_buckets": size_buckets(host_hist),
        "host_url_shape_size_buckets": size_buckets(shape_hist),
        "host_size_hist": dict(host_hist),
        "host_url_shape_size_hist": dict(shape_hist),
    }


def url_shape_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        if not parsed.hostname and "://" not in text:
            parsed = urlparse(f"//{text}")
    except ValueError:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    query_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [normalize_url_path_segment(segment) for segment in raw_segments]
    return f"path={'/'.join(normalized_segments)}|q={query_keys}"


def normalize_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        segment, extension = segment.rsplit(".", 1)
        suffix = f".{extension}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def normalize_host(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower().rstrip(".")
    if not text:
        return ""
    try:
        return text.encode("idna").decode("ascii")
    except UnicodeError:
        return text


def stable_group_hash(host: str, shape: str) -> int:
    try:
        import xxhash

        digest = xxhash.xxh64_intdigest(host)
        digest = xxhash.xxh64_intdigest(shape, seed=digest)
        return int(digest)
    except ModuleNotFoundError:
        import hashlib

        payload = f"{host}\0{shape}".encode("utf-8", errors="ignore")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False)


def representative_call_metrics(
    group_size_hist: Counter[int], rows: int, min_group_pages: int
) -> dict[str, float | int]:
    calls = 0
    saved_pages = 0
    propagated_groups = 0
    propagated_pages = 0
    for size, count in group_size_hist.items():
        if size >= min_group_pages:
            calls += count
            saved_pages += (size - 1) * count
            propagated_groups += count
            propagated_pages += size * count
        else:
            calls += size * count
    return {
        "calls": int(calls),
        "call_ratio": safe_ratio(calls, rows),
        "reduction_factor": safe_ratio(rows, calls),
        "saved_pages": int(saved_pages),
        "saved_page_ratio": safe_ratio(saved_pages, rows),
        "propagated_groups": int(propagated_groups),
        "propagated_pages": int(propagated_pages),
        "propagated_page_ratio": safe_ratio(propagated_pages, rows),
    }


def histogram_quantiles(hist: Counter[int]) -> dict[str, float | int]:
    total = sum(hist.values())
    if total == 0:
        return {"count": 0}
    targets = {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    out: dict[str, float | int] = {"count": int(total), "mean": weighted_mean(hist), "max": max(hist)}
    seen = 0
    pending = sorted(targets.items(), key=lambda item: item[1])
    pending_index = 0
    for size, count in sorted(hist.items()):
        seen += count
        while pending_index < len(pending) and seen >= math.ceil(total * pending[pending_index][1]):
            out[pending[pending_index][0]] = int(size)
            pending_index += 1
    return out


def weighted_mean(hist: Counter[int]) -> float:
    total = sum(hist.values())
    if not total:
        return 0.0
    return sum(size * count for size, count in hist.items()) / total


def size_buckets(hist: Counter[int]) -> dict[str, dict[str, int]]:
    buckets = {
        "1": (1, 1),
        "2-3": (2, 3),
        "4-7": (4, 7),
        "8-15": (8, 15),
        "16-31": (16, 31),
        "32-63": (32, 63),
        "64-127": (64, 127),
        "128-255": (128, 255),
        "256+": (256, None),
    }
    out = {name: {"groups": 0, "pages": 0} for name in buckets}
    for size, count in hist.items():
        for name, (start, end) in buckets.items():
            if size >= start and (end is None or size <= end):
                out[name]["groups"] += count
                out[name]["pages"] += size * count
                break
    return out


def resolve_manifest_files(input_value: str, host_bucket_groups: set[int] | None) -> list[Path]:
    if any(char in input_value for char in "*?["):
        paths = [Path(path) for path in glob(input_value)]
    else:
        path = Path(input_value)
        if path.is_dir():
            paths = sorted(path.glob("host_bucket_group=*.parquet"))
            if not paths:
                paths = sorted(path.glob("host_bucket_group=*/*.parquet"))
        else:
            paths = [path]
    files = [path for path in paths if path.suffix == ".parquet" and not path.name.startswith("_")]
    if host_bucket_groups is not None:
        files = [path for path in files if host_bucket_group_from_path(path) in host_bucket_groups]
    return sorted(files)


def host_bucket_group_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"host_bucket_group=(\d+)", part)
        if match:
            return int(match.group(1))
    match = re.search(r"host_bucket_group=(\d+)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer host_bucket_group from path: {path}")


def parse_int_ranges(value: str | None) -> set[int] | None:
    if not value:
        return None
    numbers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid range: {part}")
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    return numbers


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
