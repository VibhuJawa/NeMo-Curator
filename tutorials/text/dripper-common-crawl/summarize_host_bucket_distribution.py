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

"""Summarize row-count distribution across host hash buckets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger


def _parquet_files(path: Path) -> list[Path]:
    root = path / "parquet" if (path / "parquet").is_dir() else path
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}")
    return files


def _count_file(path: str, num_buckets: int, batch_size: int) -> np.ndarray:
    counts = np.zeros(num_buckets, dtype=np.int64)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["host_bucket"], batch_size=batch_size):
        arr = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        counts += np.bincount(arr, minlength=num_buckets)[:num_buckets]
    return counts


def _summarize(counts: np.ndarray) -> dict:
    total = int(counts.sum())
    mean = float(counts.mean())
    std = float(counts.std())
    sorted_counts = np.sort(counts)
    quantiles = {
        f"p{q}": int(np.percentile(counts, q, method="nearest"))
        for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    }
    top_idx = np.argsort(counts)[::-1][:20]
    bottom_idx = np.argsort(counts)[:20]
    return {
        "total_rows": total,
        "num_buckets": int(len(counts)),
        "nonempty_buckets": int(np.count_nonzero(counts)),
        "empty_buckets": int((counts == 0).sum()),
        "mean_rows_per_bucket": mean,
        "std_rows_per_bucket": std,
        "coefficient_of_variation": float(std / mean) if mean else math.nan,
        "min_rows": int(sorted_counts[0]) if len(sorted_counts) else 0,
        "max_rows": int(sorted_counts[-1]) if len(sorted_counts) else 0,
        "max_to_mean": float(sorted_counts[-1] / mean) if mean else math.nan,
        "quantiles": quantiles,
        "top_buckets": [
            {
                "host_bucket": int(i),
                "host_bucket_label": f"{int(i):05d}",
                "rows": int(counts[i]),
                "pct_total": float(counts[i] / total * 100) if total else 0.0,
                "ratio_to_mean": float(counts[i] / mean) if mean else math.nan,
            }
            for i in top_idx
        ],
        "bottom_buckets": [
            {
                "host_bucket": int(i),
                "host_bucket_label": f"{int(i):05d}",
                "rows": int(counts[i]),
                "pct_total": float(counts[i] / total * 100) if total else 0.0,
                "ratio_to_mean": float(counts[i] / mean) if mean else math.nan,
            }
            for i in bottom_idx
        ],
    }


def run(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    files = _parquet_files(Path(args.input))
    logger.info("Scanning {} parquet files from {}", len(files), args.input)

    total_counts = np.zeros(args.num_buckets, dtype=np.int64)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_count_file, str(path), args.num_buckets, args.batch_size): path
            for path in files
        }
        for future in as_completed(futures):
            total_counts += future.result()
            done += 1
            if done % args.log_every == 0 or done == len(files):
                logger.info("Scanned {:,}/{:,} files", done, len(files))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts_df = pd.DataFrame(
        {
            "host_bucket": np.arange(args.num_buckets, dtype=np.int64),
            "host_bucket_label": [f"{i:05d}" for i in range(args.num_buckets)],
            "rows": total_counts,
        }
    )
    total = int(total_counts.sum())
    mean = float(total_counts.mean())
    counts_df["pct_total"] = counts_df["rows"] / max(total, 1) * 100
    counts_df["ratio_to_mean"] = counts_df["rows"] / mean if mean else np.nan
    counts_df.to_csv(output / "host_bucket_counts.csv", index=False)

    summary = _summarize(total_counts)
    summary.update(
        {
            "input": str(Path(args.input)),
            "parquet_files": len(files),
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
            "counts_csv": str(output / "host_bucket_counts.csv"),
        }
    )
    (output / "host_bucket_distribution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    logger.info("Wrote {}", output / "host_bucket_counts.csv")
    logger.info("Wrote {}", output / "host_bucket_distribution_summary.json")
    logger.info("Summary:\n{}", json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Host-bucketed dataset root or parquet dir")
    parser.add_argument("--output", required=True, help="Directory for CSV/JSON distribution outputs")
    parser.add_argument("--num-buckets", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
