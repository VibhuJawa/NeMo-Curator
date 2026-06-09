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

"""Materialize the WARC-row sample selected by a prompt-dedup estimate.

The prompt-dedup estimator can spend most of its time fetching and preprocessing
HTML. This helper reuses the completed estimate JSON, replays the deterministic
host-row selection, and writes a GPU-runnable manifest with WARC byte-range
columns. It is intended for follow-up A/B runs against the exact same selected
host sample.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from estimate_prompt_dedup_call_reduction import (
    REQUIRED_WARC_COLUMNS,
    parse_int_ranges,
    resolve_manifest_files,
    select_manifest_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a GPU-runnable manifest from a prompt-dedup estimate JSON")
    parser.add_argument("--estimate-json", required=True, help="Completed prompt_dedup_estimate.json path")
    parser.add_argument("--output", required=True, help="Output parquet manifest path")
    parser.add_argument("--input", default=None, help="Override source manifest dir/file/glob from the estimate JSON")
    parser.add_argument("--host-bucket-groups", default=None, help="Override host_bucket_group filter from the estimate JSON")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size; 0 uses the estimate JSON value")
    parser.add_argument("--max-files", type=int, default=-1, help="Override max files; -1 uses the estimate JSON value")
    parser.add_argument("--max-pages", type=int, default=0, help="Override max pages; 0 uses the estimate JSON value")
    parser.add_argument(
        "--max-pages-per-host",
        type=int,
        default=0,
        help="Override max pages per host; 0 uses the estimate JSON value",
    )
    parser.add_argument(
        "--select-max-rows",
        type=int,
        default=-1,
        help="Override row scan cap; -1 uses the estimate JSON value",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=-1,
        help="Expected output rows; -1 uses candidate_rows from the estimate JSON, 0 disables the check",
    )
    args = parser.parse_args()
    if args.batch_size < 0:
        raise ValueError("--batch-size must be non-negative")
    if args.max_files < -1:
        raise ValueError("--max-files must be -1 or non-negative")
    if args.max_pages < 0:
        raise ValueError("--max-pages must be non-negative")
    if args.max_pages_per_host < 0:
        raise ValueError("--max-pages-per-host must be non-negative")
    if args.select_max_rows < -1:
        raise ValueError("--select-max-rows must be -1 or non-negative")
    if args.expected_rows < -1:
        raise ValueError("--expected-rows must be -1 or non-negative")
    return args


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    estimate = json.loads(Path(args.estimate_json).read_text(encoding="utf-8"))
    estimate_args = estimate.get("args", {})
    selected_hosts = [str(item["host"]) for item in estimate.get("selected_hosts", []) if item.get("host")]
    if not selected_hosts:
        raise ValueError(f"No selected_hosts found in {args.estimate_json}")

    input_path = args.input or str(estimate.get("input") or "")
    if not input_path:
        raise ValueError("--input was not provided and the estimate JSON has no input field")

    host_bucket_groups = args.host_bucket_groups
    if host_bucket_groups is None:
        host_bucket_groups = estimate_args.get("host_bucket_groups")
    batch_size = args.batch_size or int(estimate_args.get("batch_size") or 131072)
    max_files = args.max_files if args.max_files >= 0 else int(estimate_args.get("max_files") or 0)
    max_pages = args.max_pages or int(estimate_args.get("max_pages") or estimate.get("candidate_rows") or 0)
    max_pages_per_host = args.max_pages_per_host or int(estimate_args.get("max_pages_per_host") or 512)
    select_max_rows = (
        args.select_max_rows if args.select_max_rows >= 0 else int(estimate_args.get("select_max_rows") or 0)
    )
    expected_rows = args.expected_rows if args.expected_rows >= 0 else int(estimate.get("candidate_rows") or 0)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    if max_pages_per_host <= 0:
        raise ValueError("max_pages_per_host must be positive")

    manifest_files = resolve_manifest_files(input_path, parse_int_ranges(host_bucket_groups))
    if max_files:
        manifest_files = manifest_files[:max_files]
    if not manifest_files:
        raise FileNotFoundError(f"No manifest parquet files matched {input_path!r}")

    print(
        "PROMPT_DEDUP_SAMPLE_MANIFEST_INPUT "
        f"files={len(manifest_files)} selected_hosts={len(selected_hosts)} max_pages={max_pages} "
        f"max_pages_per_host={max_pages_per_host}",
        flush=True,
    )
    sample_df, selection_stats = select_manifest_rows(
        manifest_files,
        selected_hosts=selected_hosts,
        batch_size=batch_size,
        max_pages=max_pages,
        max_pages_per_host=max_pages_per_host,
        max_rows=select_max_rows,
    )
    if sample_df.empty:
        raise RuntimeError("Selected no rows while materializing prompt-dedup sample manifest")
    missing = sorted(set(REQUIRED_WARC_COLUMNS).difference(sample_df.columns))
    if missing:
        raise RuntimeError(f"Output manifest is missing required WARC columns: {missing}")
    if expected_rows and len(sample_df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} selected rows from estimate JSON, got {len(sample_df)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(output_path, index=False)
    metrics = {
        "estimate_json": str(args.estimate_json),
        "input": input_path,
        "output": str(output_path),
        "rows": int(len(sample_df)),
        "hosts": int(sample_df["url_host_name"].nunique()) if "url_host_name" in sample_df.columns else 0,
        "files": [str(path) for path in manifest_files],
        "file_count": len(manifest_files),
        "selected_hosts": selected_hosts,
        "selection_stats": selection_stats,
        "args": {
            "batch_size": batch_size,
            "max_files": max_files,
            "host_bucket_groups": host_bucket_groups,
            "max_pages": max_pages,
            "max_pages_per_host": max_pages_per_host,
            "select_max_rows": select_max_rows,
            "expected_rows": expected_rows,
        },
        "timings_s": {"total_s": time.perf_counter() - started},
    }
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print("PROMPT_DEDUP_SAMPLE_MANIFEST_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("PROMPT_DEDUP_SAMPLE_MANIFEST_END")
    print(f"OUTPUT={output_path}")
    print(f"METRICS={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
