#!/usr/bin/env python3
"""
reorganize_host_buckets.py

For one host_bucket_group (0-99):
  - Read all chunk_*.parquet files
  - Group by host_bucket (each group has 100 distinct bucket IDs)
  - Sort each bucket's pages by url_host_name
  - Write one parquet per host_bucket → output_dir/host_bucket=NNNN.parquet

Run as: python3 reorganize_host_buckets.py <group_id>

Slurm: submit 100 jobs, one per group, each writing 100 output files.
Total output: 10,000 parquet files, one per host_bucket, sorted by hostname.
"""

import glob
import sys
import time
from pathlib import Path

import pandas as pd

_LOG_EVERY = 50  # log progress every N chunks read
_ARGV_GROUP_IDX = 2  # sys.argv index for group_id argument
_ARGV_INPUT_IDX = 3  # sys.argv index for optional input_dir argument

if len(sys.argv) < _ARGV_GROUP_IDX:
    print(f"Usage: {sys.argv[0]} <group_id> [input_dir] [output_dir]", file=sys.stderr)
    sys.exit(1)

GROUP_ID = int(sys.argv[1])
INPUT_BASE = (
    sys.argv[_ARGV_GROUP_IDX]
    if len(sys.argv) > _ARGV_GROUP_IDX
    else (
        "/lustre/fsw/portfolios/llmservice/users/vjawa/"
        "nemo_curator_dripper_host_bucket_map_20260608_003146/host_bucket_shards"
    )
)
OUTPUT_DIR = (
    sys.argv[_ARGV_INPUT_IDX]
    if len(sys.argv) > _ARGV_INPUT_IDX
    else ("/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_sorted_host_buckets_20260611")
)

group_dir = f"{INPUT_BASE}/host_bucket_group={GROUP_ID}"
chunk_files = sorted(glob.glob(f"{group_dir}/chunk_*.parquet"))

if not chunk_files:
    print(f"ERROR: no chunks found in {group_dir}", file=sys.stderr)
    sys.exit(1)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

t0 = time.perf_counter()
print(f"[group {GROUP_ID:3d}] reading {len(chunk_files)} chunks from {group_dir}")

dfs = []
for i, cf in enumerate(chunk_files):
    dfs.append(pd.read_parquet(cf))
    if (i + 1) % _LOG_EVERY == 0:
        elapsed = time.perf_counter() - t0
        print(f"[group {GROUP_ID:3d}]   read {i + 1}/{len(chunk_files)} chunks  ({elapsed:.1f}s)")

df = pd.concat(dfs, ignore_index=True)
del dfs

read_time = time.perf_counter() - t0
print(f"[group {GROUP_ID:3d}] loaded {len(df):,} rows in {read_time:.1f}s")
print(f"[group {GROUP_ID:3d}] host_bucket range: {df['host_bucket'].min()} – {df['host_bucket'].max()}")
print(f"[group {GROUP_ID:3d}] unique host_buckets: {df['host_bucket'].nunique()}")
print(f"[group {GROUP_ID:3d}] unique hostnames: {df['url_host_name'].nunique():,}")

# Sort once by (host_bucket, url_host_name) — all pages from same host are contiguous
df = df.sort_values(["host_bucket", "url_host_name"], kind="stable").reset_index(drop=True)

sort_time = time.perf_counter() - t0 - read_time
print(f"[group {GROUP_ID:3d}] sorted in {sort_time:.1f}s")

# Write one parquet per host_bucket
buckets_written = 0
for bucket_id, bucket_df in df.groupby("host_bucket", sort=False):
    out_path = f"{OUTPUT_DIR}/host_bucket={bucket_id:04d}.parquet"
    bucket_df.reset_index(drop=True).to_parquet(out_path, index=False, compression="snappy")
    buckets_written += 1

total = time.perf_counter() - t0
print(f"[group {GROUP_ID:3d}] wrote {buckets_written} host_bucket files in {total:.1f}s total")
print(f"[group {GROUP_ID:3d}] output: {OUTPUT_DIR}/host_bucket={{0–9999}}.parquet")
