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

"""Map CC URL Index rows into host-bucketed parquet shards.

This is the scalable first phase for whole-snapshot host clustering:
each Slurm CPU job reads a subset of CC index parquet parts once, filters to
HTML response rows, computes full-host and xxhash host buckets, and writes
partitioned shards under ``host_bucket_group=<N>/``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_host_clustered_manifest import (
    iter_filtered_batches,
    parse_host_buckets,
    resolve_input_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build host-bucketed CC index shard files")
    parser.add_argument("--cc-index-path", required=True, help="Directory, parquet file, or glob for CC URL Index parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", required=True, help="Stable ID for output file names, e.g. part range or Slurm array ID")
    parser.add_argument("--host-bucket-mod", type=int, default=10000)
    parser.add_argument("--host-bucket-group-size", type=int, default=100)
    parser.add_argument("--host-buckets", default=None, help="Optional comma/range host-bucket filter")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--max-index-rows", type=int, default=0)
    parser.add_argument("--status", type=int, default=200)
    parser.add_argument("--html-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    if args.host_bucket_mod <= 0:
        raise ValueError("--host-bucket-mod must be positive")
    if args.host_bucket_group_size <= 0:
        raise ValueError("--host-bucket-group-size must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_index_rows < 0:
        raise ValueError("--max-index-rows must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    input_paths = resolve_input_paths(args.cc_index_path)
    host_buckets = parse_host_buckets(args.host_buckets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_hosts: set[str] = set()
    batch_count = 0
    tables_by_group: dict[int, list[pa.Table]] = defaultdict(list)
    for batch in iter_filtered_batches(args, input_paths, host_buckets):
        if batch.empty:
            continue
        batch = batch.copy()
        batch["host_bucket_group"] = (batch["host_bucket"] // args.host_bucket_group_size).astype("int64")
        total_rows += len(batch)
        total_hosts.update(batch["url_host_name"].unique().tolist())
        for group, group_df in batch.groupby("host_bucket_group", sort=False):
            tables_by_group[int(group)].append(pa.Table.from_pandas(group_df, preserve_index=False))
        batch_count += 1

    written_files = write_group_tables(tables_by_group, output_dir, source_id=args.source_id)
    metrics = {
        "input_paths": input_paths,
        "source_id": args.source_id,
        "rows": total_rows,
        "hosts": len(total_hosts),
        "batches": batch_count,
        "written_files": len(written_files),
        "output_dir": str(output_dir),
        "host_bucket_mod": args.host_bucket_mod,
        "host_bucket_group_size": args.host_bucket_group_size,
    }
    metrics_path = output_dir / f"{args.source_id}.metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print("HOST_BUCKET_SHARDS_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("HOST_BUCKET_SHARDS_METRICS_END")
    return 0


def write_group_tables(
    tables_by_group: dict[int, list[pa.Table]],
    output_dir: Path,
    *,
    source_id: str,
) -> list[str]:
    written_files: list[str] = []
    for group, tables in sorted(tables_by_group.items()):
        if not tables:
            continue
        group_dir = output_dir / f"host_bucket_group={group}"
        group_dir.mkdir(parents=True, exist_ok=True)
        output_path = group_dir / f"{source_id}.parquet"
        table = pa.concat_tables(tables, promote_options="default") if len(tables) > 1 else tables[0]
        pq.write_table(table, output_path)
        written_files.append(str(output_path))
    return written_files


if __name__ == "__main__":
    raise SystemExit(main())
