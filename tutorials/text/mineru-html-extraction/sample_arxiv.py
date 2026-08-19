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
"""
Cut a deterministic sample out of the arXiv LaTeXML HTML corpus, for the pipeline to read.

Sampling lives here rather than in ``run_pipeline.py`` on purpose. That pipeline is
Vibhu's, and the fewer reasons it has to know about arXiv tiers the better; it wants a
parquet with an HTML column and a URL column, which is exactly what this writes. It also
means ``--limit`` keeps its own meaning — it truncates *per reader partition*, so it is a
throttle rather than a sample size, and asking it for "exactly 5 documents" would quietly
give you more.

    python sample_arxiv.py --input sample.parquet --out sample-5.parquet \
        --limit 5 --tier A --tier B

Selection is round-robin over the tiers, seeded, and the chosen row indices are written
to a JSON sidecar so the same sample can be rebuilt without this file agreeing with
itself later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Carried through so a row can be sliced by what the converter thought of it. `status`
# and `tier` are LaTeXML's own verdict; without them a truncated conversion and a bad
# extraction look identical downstream.
CARRY = ("arxiv_id", "url", "status", "tier", "era", "n_math", "n_section", "n_img")


def deterministic_order(count: int, salt: str) -> list[int]:
    return sorted(range(count), key=lambda i: hashlib.sha256(f"{salt}/{i}".encode()).digest())


def choose(frame: pd.DataFrame, limit: int, tiers: list[str], seed: int) -> list[int]:
    """Round-robin over the tiers, so a 90/10 split does not sample as all-majority."""
    if tiers:
        frame = frame[frame["tier"].astype(str).isin(tiers)]

    groups: dict[str, list[int]] = {}
    for tier, rows in frame.groupby(frame["tier"].astype(str)).groups.items():
        ordered = sorted(rows.tolist())
        groups[str(tier)] = [ordered[i] for i in deterministic_order(len(ordered), f"{seed}/{tier}")]

    picked: list[int] = []
    position = 0
    while len(picked) < limit and any(len(rows) > position for rows in groups.values()):
        for tier in sorted(groups):
            if position < len(groups[tier]) and len(picked) < limit:
                picked.append(groups[tier][position])
        position += 1
    return picked


def read_rows(path: Path, wanted: list[int], columns: list[str]) -> pd.DataFrame:
    """Only the row groups the chosen rows fall in, single-threaded.

    `use_threads=False` is not a style choice: pyarrow's allocator reserves ~1 GiB of
    address space per worker thread it touches, and this login node has a hard 8 GB
    `ulimit -v`.
    """
    file = pq.ParquetFile(str(path))
    present = [c for c in columns if c in file.schema_arrow.names]
    target = set(wanted)
    frames: list[pd.DataFrame] = []
    start = 0
    for group in range(file.metadata.num_row_groups):
        rows = file.metadata.row_group(group).num_rows
        hits = [i for i in range(start, start + rows) if i in target]
        if hits:
            chunk = file.read_row_groups([group], columns=present, use_threads=False).to_pandas()
            picked = chunk.iloc[[i - start for i in hits]].copy()
            picked.index = pd.Index(hits)
            frames.append(picked)
        start += rows

    if not frames:
        return pd.DataFrame(columns=present)
    combined = pd.concat(frames)
    rank = {row: at for at, row in enumerate(wanted)}
    return combined.assign(_rank=[rank[i] for i in combined.index]).sort_values("_rank").drop(columns=["_rank"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="Parquet file to write")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tier", action="append", help="Keep only these tiers; repeatable")
    ap.add_argument("--html-field", default="html")
    ap.add_argument("--row-group-size", type=int, default=64)
    args = ap.parse_args()

    source = Path(args.input).resolve()
    file = pq.ParquetFile(str(source))
    index = file.read(columns=["arxiv_id", "tier"], use_threads=False).to_pandas()

    wanted = choose(index, args.limit, args.tier or [], args.seed)
    frame = read_rows(source, wanted, [*CARRY, args.html_field])

    # A rejected conversion carries a null html column; keeping it would spend a sample
    # slot on a document there is nothing to extract from.
    before = len(frame)
    frame = frame[frame[args.html_field].astype("string").fillna("").str.strip() != ""]
    dropped = before - len(frame)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Small row groups so a reader can take one document without paying for the file:
    # the delivered corpus is one row group per shard, where a single document costs
    # 1.79 s and 1.44 GB of peak RSS against 24 ms re-chunked.
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), out, row_group_size=args.row_group_size)

    sidecar = out.with_suffix(".selection.json")
    sidecar.write_text(
        json.dumps(
            {
                "source": str(source),
                "rule": f"round-robin over tiers {'/'.join(args.tier or ['*'])}, seed {args.seed}",
                "seed": args.seed,
                "limit": args.limit,
                "rowIndices": wanted,
                "droppedEmptyHtml": dropped,
                "arxivIds": frame["arxiv_id"].tolist(),
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    print(f"  {len(frame)} document(s) -> {out}  ({dropped} dropped for empty html)")
    print(f"  tiers: {frame['tier'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
