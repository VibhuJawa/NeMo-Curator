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

"""Reduce host-bucketed CC index shards into host-clustered manifests."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd

from build_host_clustered_manifest import parse_host_buckets

OUTPUT_COLUMNS = [
    "url",
    "url_host_name",
    "host_bucket",
    "content_mime_type",
    "content_mime_detected",
    "content_languages",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]
REQUIRED_COLUMNS = ["url", "url_host_name", "host_bucket", "warc_filename", "warc_record_offset", "warc_record_length"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce host-bucketed CC index shards into host-clustered manifests")
    parser.add_argument("--input-shards", required=True, help="Shard directory, parquet file, or glob")
    parser.add_argument("--output", required=True, help="Output parquet path for single mode, or output directory for per-group")
    parser.add_argument("--output-mode", choices=["single", "per-group"], default="single")
    parser.add_argument("--max-pages", type=int, default=8192, help="Global page cap for single mode. Use 0 for no cap.")
    parser.add_argument("--min-host-pages", type=int, default=8)
    parser.add_argument("--max-pages-per-host", type=int, default=64, help="Use 0 for no per-host cap")
    parser.add_argument("--max-hosts", type=int, default=0, help="0 means choose enough top hosts for single mode or all hosts")
    parser.add_argument("--host-bucket-groups", default=None, help="Optional comma/range filter over host_bucket_group values")
    args = parser.parse_args()
    if args.max_pages < 0:
        raise ValueError("--max-pages must be non-negative")
    if args.min_host_pages < 1:
        raise ValueError("--min-host-pages must be positive")
    if args.max_pages_per_host < 0:
        raise ValueError("--max-pages-per-host must be non-negative")
    if args.max_hosts < 0:
        raise ValueError("--max-hosts must be non-negative")
    if args.output_mode == "per-group" and args.max_pages > 0:
        raise ValueError("--output-mode per-group requires --max-pages 0; otherwise the cap is ambiguous")
    return args


def main() -> int:
    args = parse_args()
    host_bucket_groups = parse_host_buckets(args.host_bucket_groups)
    shard_files = resolve_shard_files(args.input_shards, host_bucket_groups)
    if not shard_files:
        raise FileNotFoundError(f"No shard parquet files matched {args.input_shards!r}")

    if args.output_mode == "single":
        selected, metrics = build_single_manifest(args, shard_files)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(output_path, index=False)
        metrics["output"] = str(output_path)
        metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    else:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics = build_per_group_manifests(args, shard_files, output_path)
        metrics["output"] = str(output_path)
        metrics_suffix = sanitize_metrics_suffix(args.host_bucket_groups or "all")
        metrics_path = output_path / f"_metrics_{metrics_suffix}.json"

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print("HOST_CLUSTERED_REDUCE_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("HOST_CLUSTERED_REDUCE_METRICS_END")
    return 0


def build_single_manifest(args: argparse.Namespace, shard_files: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = count_hosts(shard_files)
    if not counts:
        raise RuntimeError("No rows found in host-bucketed shards")

    requested_hosts = args.max_hosts
    if requested_hosts == 0 and args.max_pages > 0 and args.max_pages_per_host > 0:
        requested_hosts = math.ceil(args.max_pages / args.max_pages_per_host) + 16
    eligible_hosts = select_eligible_hosts(counts, min_host_pages=args.min_host_pages, max_hosts=requested_hosts)
    if not eligible_hosts:
        raise RuntimeError(f"No host had at least {args.min_host_pages} page(s)")

    selected = select_manifest_rows(
        shard_files,
        eligible_hosts,
        max_pages=args.max_pages,
        max_pages_per_host=args.max_pages_per_host,
    )
    if selected.empty:
        raise RuntimeError("No rows selected from host-bucketed shards")

    selected = sort_manifest(selected)
    if args.max_pages > 0:
        selected = selected.head(args.max_pages)
    metrics = make_metrics(
        shard_files,
        selected,
        mode="single",
        counted_hosts=len(counts),
        eligible_hosts=len(eligible_hosts),
        min_host_pages=args.min_host_pages,
        max_pages_per_host=args.max_pages_per_host,
    )
    return selected, metrics


def build_per_group_manifests(args: argparse.Namespace, shard_files: list[Path], output_dir: Path) -> dict[str, Any]:
    files_by_group: dict[int, list[Path]] = {}
    for path in shard_files:
        group = host_bucket_group_from_path(path)
        files_by_group.setdefault(group, []).append(path)

    group_metrics: list[dict[str, Any]] = []
    total_rows = 0
    total_hosts = 0
    for group, files in sorted(files_by_group.items()):
        counts = count_hosts(files)
        eligible_hosts = select_eligible_hosts(counts, min_host_pages=args.min_host_pages, max_hosts=args.max_hosts)
        if not eligible_hosts:
            group_metrics.append(
                {
                    "host_bucket_group": group,
                    "input_files": len(files),
                    "counted_hosts": len(counts),
                    "eligible_hosts": 0,
                    "selected_rows": 0,
                    "output": None,
                }
            )
            continue

        selected = select_manifest_rows(
            files,
            eligible_hosts,
            max_pages=0,
            max_pages_per_host=args.max_pages_per_host,
        )
        selected = sort_manifest(selected)
        group_path = output_dir / f"host_bucket_group={group}.parquet"
        selected.to_parquet(group_path, index=False)
        selected_hosts = int(selected["url_host_name"].nunique()) if not selected.empty else 0
        total_rows += len(selected)
        total_hosts += selected_hosts
        group_metrics.append(
            {
                "host_bucket_group": group,
                "input_files": len(files),
                "counted_hosts": len(counts),
                "eligible_hosts": len(eligible_hosts),
                "selected_rows": len(selected),
                "selected_hosts": selected_hosts,
                "output": str(group_path),
            }
        )

    return {
        "mode": "per-group",
        "input_files": len(shard_files),
        "groups": len(files_by_group),
        "selected_rows": total_rows,
        "selected_hosts": total_hosts,
        "group_metrics": group_metrics,
        "min_host_pages": args.min_host_pages,
        "max_pages_per_host": args.max_pages_per_host,
    }


def count_hosts(shard_files: Iterable[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in shard_files:
        df = pd.read_parquet(path, columns=["url_host_name"])
        counts.update(df["url_host_name"].dropna().astype(str).tolist())
    return counts


def select_eligible_hosts(counts: Counter[str], *, min_host_pages: int, max_hosts: int) -> set[str]:
    hosts = [host for host, count in counts.most_common() if count >= min_host_pages]
    if max_hosts > 0:
        hosts = hosts[:max_hosts]
    return set(hosts)


def select_manifest_rows(
    shard_files: Iterable[Path],
    eligible_hosts: set[str],
    *,
    max_pages: int,
    max_pages_per_host: int,
) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    host_selected: Counter[str] = Counter()
    selected_count = 0

    for path in shard_files:
        df = read_manifest_shard(path)
        df = df[df["url_host_name"].isin(eligible_hosts)]
        if df.empty:
            continue
        df = sort_manifest(df)

        if max_pages_per_host > 0:
            keep_parts: list[pd.DataFrame] = []
            for host, host_df in df.groupby("url_host_name", sort=False):
                remaining_for_host = max_pages_per_host - host_selected[host]
                if remaining_for_host <= 0:
                    continue
                kept = host_df.head(remaining_for_host)
                host_selected[host] += len(kept)
                keep_parts.append(kept)
            if not keep_parts:
                continue
            df = pd.concat(keep_parts, ignore_index=True)

        if max_pages > 0:
            remaining = max_pages - selected_count
            if remaining <= 0:
                break
            df = df.head(remaining)

        selected_count += len(df)
        selected_frames.append(df)
        if max_pages > 0 and selected_count >= max_pages:
            break

    if not selected_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(selected_frames, ignore_index=True)


def read_manifest_shard(path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        columns = pq.read_schema(path).names
    except ModuleNotFoundError:
        columns = pd.read_parquet(path).columns.tolist()
    missing = sorted(set(REQUIRED_COLUMNS).difference(columns))
    if missing:
        raise ValueError(f"Shard {path} is missing required columns: {missing}")
    keep_columns = [column for column in OUTPUT_COLUMNS if column in columns]
    return pd.read_parquet(path, columns=keep_columns)


def sort_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        ["host_bucket", "url_host_name", "url", "warc_filename", "warc_record_offset"],
        kind="stable",
    ).reset_index(drop=True)


def make_metrics(
    shard_files: list[Path],
    selected: pd.DataFrame,
    *,
    mode: str,
    counted_hosts: int,
    eligible_hosts: int,
    min_host_pages: int,
    max_pages_per_host: int,
) -> dict[str, Any]:
    host_counts = selected.groupby("url_host_name").size()
    return {
        "mode": mode,
        "input_files": len(shard_files),
        "host_bucket_groups": sorted({host_bucket_group_from_path(path) for path in shard_files}),
        "counted_hosts": counted_hosts,
        "eligible_hosts": eligible_hosts,
        "selected_rows": len(selected),
        "selected_hosts": int(selected["url_host_name"].nunique()),
        "min_host_pages": min_host_pages,
        "max_pages_per_host": max_pages_per_host,
        "p50_selected_host_pages": float(host_counts.quantile(0.5)),
        "p95_selected_host_pages": float(host_counts.quantile(0.95)),
        "max_selected_host_pages": int(host_counts.max()),
    }


def resolve_shard_files(input_shards: str, host_bucket_groups: set[int] | None) -> list[Path]:
    if any(char in input_shards for char in "*?["):
        paths = [Path(path) for path in glob(input_shards)]
    else:
        path = Path(input_shards)
        if path.is_dir():
            paths = sorted(path.glob("host_bucket_group=*/*.parquet"))
            if not paths:
                paths = sorted(path.glob("host_bucket_group=*.parquet"))
        else:
            paths = [path]
    shard_files = sorted(path for path in paths if path.suffix == ".parquet")
    if host_bucket_groups is not None:
        shard_files = [path for path in shard_files if host_bucket_group_from_path(path) in host_bucket_groups]
    return shard_files


def host_bucket_group_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"host_bucket_group=(\d+)", part)
        if match:
            return int(match.group(1))
    match = re.search(r"host_bucket_group=(\d+)", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer host_bucket_group from path: {path}")


def sanitize_metrics_suffix(value: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return suffix.strip("_") or "all"


if __name__ == "__main__":
    raise SystemExit(main())
