#!/usr/bin/env python3
# Re-shard a Dripper plan-output dir into N balanced shards WITHOUT splitting any
# layout cluster across shards.
#
# Why: Phase-2 layout propagation is BLOCK-LOCAL -- a cluster's representative and all
# its members must land in the same Phase-2 block (= one input shard) or the members
# cannot propagate and fall back to individual LLM calls. The plan stage's
# `repartition(output_shards)` does a ROW SHUFFLE that splits ~half the clusters across
# shards, breaking propagation AND creating size imbalance (one giant shard = serial
# long-pole that idles the GPUs during its CPU finalize).
#
# This re-shards by UNIT (a whole layout cluster = one atomic unit; each standalone /
# unclustered row = a size-1 unit), greedily bin-packing units into N balanced shards.
# Result: clusters intact (correct propagation) + even shard sizes (CPU finalize on one
# block overlaps GPU inference on another).
#
# Usage: rebalance_plan_shards.py <src_dir> <dst_dir> [n_shards=24] [cluster_col]
import collections
import glob
import heapq
import os
import sys

import pyarrow as pa  # noqa: F401  (kept for type clarity / future use)
import pyarrow.parquet as pq

src = sys.argv[1]
dst = sys.argv[2]
n_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 24  # noqa: PLR2004
cluster_col = sys.argv[4] if len(sys.argv) > 4 else "dripper_layout_cluster"  # noqa: PLR2004

files = sorted(glob.glob(os.path.join(src, "*.parquet")))
if not files:
    msg = f"no parquet shards in {src}"
    raise SystemExit(msg)

# Pass 1 (light): cluster id per global row, in `files` order.
clusters: list[str] = []
for f in files:
    col = pq.read_table(f, columns=[cluster_col]).column(0).to_pylist()
    clusters.extend(str(c or "") for c in col)
n_rows = len(clusters)

# Build units: each non-empty cluster -> its global row indices (atomic);
# each empty-cluster (standalone/unclustered) row -> its own size-1 unit.
unit_idxs: dict[str, list[int]] = collections.defaultdict(list)
solo: list[int] = []
for i, c in enumerate(clusters):
    if c:
        unit_idxs[c].append(i)
    else:
        solo.append(i)
units: list[list[int]] = list(unit_idxs.values()) + [[i] for i in solo]
units.sort(key=len, reverse=True)  # largest-first for good greedy packing

# Greedy bin-pack into n_shards by row count (min-heap of current bin sizes).
heap = [(0, b) for b in range(n_shards)]
heapq.heapify(heap)
bin_of = [0] * n_rows
bin_sz = [0] * n_shards
for idxs in units:
    size, b = heapq.heappop(heap)
    for i in idxs:
        bin_of[i] = b
    bin_sz[b] += len(idxs)
    heapq.heappush(heap, (size + len(idxs), b))

# Pass 2 (stream): route each input shard's rows to per-bin writers (one input
# shard resident at a time -> peak memory ~= largest input shard).
os.makedirs(dst, exist_ok=True)
schema = pq.read_schema(files[0])
writers: dict[int, pq.ParquetWriter] = {}
g = 0
for f in files:
    # Stream one ROW GROUP at a time (not the whole shard) -- a shard's full HTML payload
    # can exceed the login node's per-process memory cap; row groups are ~hundreds of rows.
    pf = pq.ParquetFile(f)
    for rgi in range(pf.num_row_groups):
        t = pf.read_row_group(rgi)
        m = t.num_rows
        by_bin: dict[int, list[int]] = collections.defaultdict(list)
        for li in range(m):
            by_bin[bin_of[g + li]].append(li)
        for b, lis in by_bin.items():
            sub = t.take(lis)
            w = writers.get(b)
            if w is None:
                w = writers[b] = pq.ParquetWriter(os.path.join(dst, f"shard_{b:04d}.parquet"), schema)
            w.write_table(sub)
        g += m
for w in writers.values():
    w.close()

nz = [s for s in bin_sz if s > 0]
print(f"input_rows={n_rows} units={len(units)} biggest_unit={len(units[0])}")
print(f"shards_written={len(writers)} max_shard={max(nz)} min_shard={min(nz)} mean={sum(nz) // len(nz)}")
