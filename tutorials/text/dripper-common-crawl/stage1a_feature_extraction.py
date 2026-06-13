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
stage1a_feature_extraction.py — CPU-only DOM feature extraction.

RUNS ON: cpu_short partition (no GPU needed).

INPUT:  manifest parquet (url, html, url_host_name, ...)
OUTPUT: features parquet per shard:
          url, url_host_name, html,
          dom_feature (JSON-serialized dict from get_feature()),
          warc_filename, warc_record_offset, warc_record_length

CURATOR PATTERN:
  ProcessingStage with ProcessPoolExecutor for CPU parallelism.
  Reads parquet in row groups (streaming, bounded memory).
  Writes output incrementally.

Stage 1b (GPU DBSCAN) reads this output.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

OUTPUT_COLS = [
    "url", "url_host_name", "html", "dom_feature",
    "warc_filename", "warc_record_offset", "warc_record_length",
]


def _init_worker():
    global _WEB
    try:
        from nemo_curator.stages.text.experimental.dripper.stage import _load_llm_web_kit_bindings
        _WEB = _load_llm_web_kit_bindings()
    except Exception:
        _WEB = None


def _extract_one(rec: dict) -> dict:
    global _WEB
    html = rec.get("html", "")
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    feat = None
    if _WEB and html.strip():
        try:
            feat = _WEB.get_feature(html)
        except Exception:
            feat = None
    return {
        "url":               rec.get("url", ""),
        "url_host_name":     rec.get("url_host_name", ""),
        "html":              html,
        "dom_feature":       json.dumps(feat) if feat else "",
        "warc_filename":     rec.get("warc_filename"),
        "warc_record_offset": rec.get("warc_record_offset"),
        "warc_record_length": rec.get("warc_record_length"),
    }


def run(args):
    pf = pq.ParquetFile(args.input)
    total = pf.metadata.num_rows
    start = total * args.shard_index // args.num_shards
    end   = total * (args.shard_index + 1) // args.num_shards

    need = ["url", "url_host_name", "html", "warc_filename",
            "warc_record_offset", "warc_record_length"]
    avail = pf.schema_arrow.names
    cols  = [c for c in need if c in avail]

    rows_seen, parts = 0, []
    for batch in pf.iter_batches(batch_size=65_536, columns=cols):
        df = batch.to_pandas()
        lo = max(0, start - rows_seen)
        hi = min(len(df), end - rows_seen)
        rows_seen += len(df)
        if lo < hi:
            parts.append(df.iloc[lo:hi])
        if rows_seen >= end:
            break

    shard_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"[stage1a] shard {args.shard_index}/{args.num_shards}: {len(shard_df):,} pages")

    if len(shard_df) == 0:
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline_metrics import StageMetrics
    tracker = StageMetrics("stage1a", shard_index=args.shard_index,
                           num_shards=args.num_shards, n_workers=args.workers)
    tracker.start()

    records = shard_df.to_dict("records")
    results = []

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_extract_one, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 5000 == 0:
                tracker.checkpoint(done)

    out_df = pd.DataFrame(results)
    for col in OUTPUT_COLS:
        if col not in out_df.columns:
            out_df[col] = None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1
                      else "shard_0000.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")
    out_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    feat_ok = int((out_df["dom_feature"] != "").sum())
    tracker.finish(total_pages=len(out_df),
                   errors=len(out_df) - feat_ok)
    tracker.extra = {"feature_ok": feat_ok, "output": str(out_path)}
    tracker.save(args.output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers",    type=int, default=max(1, (os.cpu_count() or 4) - 2))
    run(p.parse_args())


if __name__ == "__main__":
    main()
