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
  ProcessingStage[DocumentBatch, DocumentBatch] via RayActorPoolExecutor.
  Ray spawns floor(available_cpus / resources.cpus) actors; each loads the
  webkit bindings once in setup() and loops over rows in process() — no
  nested ProcessPoolExecutor.

Stage 1b (GPU DBSCAN) reads this output.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "html",
    "dom_feature",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


@dataclass(kw_only=True)
class DOMFeatureExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU stage: calls get_feature() per row via llm_web_kit bindings.

    Ray spawns one actor per Resources(cpus=4.0) block. Each actor loads the
    heavy C++ bindings once in setup() and processes DocumentBatch tasks via a
    plain list-comp in process() — no nested ProcessPoolExecutor.
    """

    name: str = "DOMFeatureExtractionStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0))
    html_col: str = "html"
    feature_col: str = "dom_feature"
    _web: Any = field(init=False, repr=False, default=None)

    def setup(self, worker_metadata=None) -> None:
        from nemo_curator.stages.text.experimental.dripper.stage import _load_llm_web_kit_bindings

        try:
            self._web = _load_llm_web_kit_bindings()
        except Exception as exc:
            print(f"[stage1a] WARNING: bindings unavailable: {exc}", flush=True)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        web = self._web

        def _extract(html: Any) -> str:
            if isinstance(html, bytes):
                html = html.decode("utf-8", errors="replace")
            if web and isinstance(html, str) and html.strip():
                try:
                    return json.dumps(web.get_feature(html))
                except Exception:
                    pass
            return ""

        df[self.feature_col] = [_extract(h) for h in df[self.html_col]]
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def run(args):
    pf = pq.ParquetFile(args.input)
    total = pf.metadata.num_rows
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards

    need = ["url", "url_host_name", "html", "warc_filename", "warc_record_offset", "warc_record_length"]
    cols = [c for c in need if c in pf.schema_arrow.names]

    rows_seen, parts = 0, []
    for batch in pf.iter_batches(batch_size=65_536, columns=cols):
        df_b = batch.to_pandas()
        lo, hi = max(0, start - rows_seen), min(len(df_b), end - rows_seen)
        rows_seen += len(df_b)
        if lo < hi:
            parts.append(df_b.iloc[lo:hi])
        if rows_seen >= end:
            break

    shard_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    print(f"[stage1a] shard {args.shard_index}/{args.num_shards}: {len(shard_df):,} pages", flush=True)
    if len(shard_df) == 0:
        return

    from pipeline_metrics import StageMetrics

    tracker = StageMetrics(
        "stage1a", shard_index=args.shard_index, num_shards=args.num_shards, n_workers=args.cpus_per_actor
    )
    tracker.start()

    # One DocumentBatch task per actor-sized chunk; Ray scheduler assigns actors.
    chunk = max(1, len(shard_df) // max(1, args.num_actors))
    tasks = [
        DocumentBatch(dataset_name="stage1a", data=shard_df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(shard_df), chunk)
    ]

    pipeline = Pipeline(name="stage1a")
    pipeline.add_stage(DOMFeatureExtractionStage(resources=Resources(cpus=args.cpus_per_actor)))
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []

    out_df = (
        pd.concat(
            [t.to_pandas() for t in result_tasks if hasattr(t, "to_pandas")],
            ignore_index=True,
        )
        if result_tasks
        else pd.DataFrame(columns=OUTPUT_COLS)
    )
    for col in OUTPUT_COLS:
        if col not in out_df.columns:
            out_df[col] = None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "shard_0000.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")
    out_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    feat_ok = int((out_df["dom_feature"].astype(str) != "").sum())
    tracker.finish(total_pages=len(out_df), errors=len(out_df) - feat_ok)
    tracker.extra = {"feature_ok": feat_ok, "output": str(out_path)}
    tracker.save(args.output)
    print(f"[stage1a] feature_ok={feat_ok}/{len(out_df)}  output → {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--cpus-per-actor",
        type=int,
        default=4,
        help="CPUs per Ray actor; Ray spawns total_cpus / cpus_per_actor actors",
    )
    p.add_argument(
        "--num-actors",
        type=int,
        default=max(1, (os.cpu_count() or 16) // 4),
        help="Hint for task chunk count (actual actor count set by Ray scheduler)",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
