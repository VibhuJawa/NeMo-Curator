#!/usr/bin/env python3
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
"""Compute token-level F1 between streaming-pipeline output and standalone Dripper baseline.

Usage:
  python3 eval_f1_vs_standalone.py \
    --pipeline-output /lustre/.../dripper_streaming_f1_comparison_20260616 \
    --baseline        /lustre/.../pipeline_standalone_output_03/baseline_merged.parquet \
    [--out-csv        /lustre/.../f1_scores.csv]
    [--sample         1000]   # only evaluate first N matched URLs
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lower-cased."""
    return re.findall(r"\w+", text.lower())


def _token_f1(pred: str, ref: str) -> float:
    """Token-overlap F1 between two strings (standard QA-style metric)."""
    pred_tokens = Counter(_tokenize(pred or ""))
    ref_tokens = Counter(_tokenize(ref or ""))
    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_tokens.values())
    recall = common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _load_pipeline_output(output_dir: str) -> dict[str, str]:
    """Load url → dripper_content from all parquet shards in output_dir."""
    import pyarrow.parquet as pq

    files = sorted(
        glob.glob(os.path.join(output_dir, "*.parquet")) + glob.glob(os.path.join(output_dir, "**", "*.parquet"))
    )
    if not files:
        sys.exit(f"ERROR: no parquet files found in {output_dir}")

    url_to_content: dict[str, str] = {}
    for f in files:
        schema_names = pq.read_schema(f).names
        if "url" not in schema_names or "dripper_content" not in schema_names:
            continue
        tbl = pq.read_table(f, columns=["url", "dripper_content"])
        for url, content in zip(tbl["url"].to_pylist(), tbl["dripper_content"].to_pylist(), strict=False):
            if url and url not in url_to_content:
                url_to_content[url] = content or ""
    return url_to_content


def _load_baseline(baseline_path: str) -> dict[str, str]:
    import pyarrow.parquet as pq

    tbl = pq.read_table(baseline_path, columns=["url", "dripper_content"])
    return {
        url: (content or "")
        for url, content in zip(tbl["url"].to_pylist(), tbl["dripper_content"].to_pylist(), strict=False)
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = (len(values) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    return values[lo] + (idx - lo) * (values[hi] - values[lo])


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Evaluate F1 vs standalone Dripper baseline")
    parser.add_argument("--pipeline-output", required=True, help="Dir containing output parquet shards")
    parser.add_argument("--baseline", required=True, help="Standalone baseline parquet (url, dripper_content)")
    parser.add_argument("--out-csv", default="", help="Optional path to write per-URL F1 CSV")
    parser.add_argument("--sample", type=int, default=0, help="Evaluate only first N matched URLs (0=all)")
    args = parser.parse_args()

    print(f"Loading pipeline output from {args.pipeline_output} ...")
    pipeline = _load_pipeline_output(args.pipeline_output)
    print(f"  pipeline urls: {len(pipeline)}")

    print(f"Loading baseline from {args.baseline} ...")
    baseline = _load_baseline(args.baseline)
    print(f"  baseline urls: {len(baseline)}")

    matched_urls = [u for u in baseline if u in pipeline]
    print(f"  matched urls: {len(matched_urls)} ({100 * len(matched_urls) / len(baseline):.1f}% of baseline)")

    if args.sample > 0:
        matched_urls = matched_urls[: args.sample]
        print(f"  sampling first {len(matched_urls)} URLs")

    f1_scores: list[float] = []
    zero_f1 = 0
    csv_rows: list[str] = []

    for url in matched_urls:
        pred = pipeline[url]
        ref = baseline[url]
        f1 = _token_f1(pred, ref)
        f1_scores.append(f1)
        if f1 == 0.0:
            zero_f1 += 1
        if args.out_csv:
            csv_rows.append(f"{f1:.4f},{json.dumps(url)}")

    n = len(f1_scores)
    mean_f1 = sum(f1_scores) / n if n else float("nan")
    median_f1 = _percentile(f1_scores, 50)

    print()
    print("=" * 60)
    print(f"  Matched URLs evaluated : {n}")
    print(f"  Mean F1                : {mean_f1:.4f}")
    print(f"  Median F1              : {median_f1:.4f}")
    print(f"  P10 F1                 : {_percentile(f1_scores, 10):.4f}")
    print(f"  P25 F1                 : {_percentile(f1_scores, 25):.4f}")
    print(f"  P75 F1                 : {_percentile(f1_scores, 75):.4f}")
    print(f"  P90 F1                 : {_percentile(f1_scores, 90):.4f}")
    print(f"  F1 == 0.0              : {zero_f1} ({100 * zero_f1 / n:.1f}%)")
    buckets = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    print("  F1 distribution:")
    for lo, hi in itertools.pairwise(buckets):
        count = sum(1 for s in f1_scores if lo <= s < hi)
        bar = "#" * int(40 * count / n)
        print(f"    [{lo:.2f},{hi:.2f}): {count:6d} ({100 * count / n:5.1f}%)  {bar}")
    perfect = sum(1 for s in f1_scores if s == 1.0)
    print(f"    [1.00,1.00]: {perfect:6d} ({100 * perfect / n:5.1f}%)")
    print("=" * 60)

    if args.out_csv and csv_rows:
        Path(args.out_csv).write_text("f1,url\n" + "\n".join(csv_rows) + "\n")
        print(f"Per-URL scores written to {args.out_csv}")


if __name__ == "__main__":
    main()
