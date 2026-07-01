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

"""Stage 1a: CPU-only DOM feature extraction via llm_web_kit get_feature()."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import ray
from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.experimental.dripper._html_compression import (
    HTML_CHARS_COL,
    HTML_ZLIB_COL,
    coerce_html_text,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

OUTPUT_COLS = [
    "snapshot",
    "record_id",
    "url",
    "url_host_name",
    "host_hash64",
    "host_bucket",
    "host_bucket_label",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
    "dom_feature",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "source_manifest_file",
]


def _init_ray_from_slurm() -> None:
    if ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
        return
    ray_kwargs = {"ignore_reinit_error": True, "num_gpus": 0}
    if os.environ.get("RAY_TMPDIR"):
        ray_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
    if os.environ.get("RAY_OBJECT_STORE_MEMORY_BYTES"):
        ray_kwargs["object_store_memory"] = int(os.environ["RAY_OBJECT_STORE_MEMORY_BYTES"])
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        ray_kwargs["num_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    ray.init(**ray_kwargs)
    resources = ray.cluster_resources()
    logger.info(
        "Ray resources: CPUs={} memory={:.2f}GiB object_store_memory={:.2f}GiB",
        resources.get("CPU", 0),
        float(resources.get("memory", 0)) / (1024**3),
        float(resources.get("object_store_memory", 0)) / (1024**3),
    )


def _record_id(src: dict) -> str:
    parts = [src.get("warc_filename"), src.get("warc_record_offset"), src.get("warc_record_length")]
    if all(part is not None and str(part) for part in parts):
        return "|".join(str(part) for part in parts)
    return str(src.get("record_id") or src.get("url") or "")


class DOMFeatureExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "DOMFeatureExtractionStage"

    def __init__(self, cpus_per_actor: int = 4) -> None:
        super().__init__()
        self.resources = Resources(cpus=float(cpus_per_actor))
        self._web = None

    def setup(self, _worker_metadata: object = None) -> None:
        from nemo_curator.stages.text.experimental.dripper.stage import _load_llm_web_kit_bindings

        self._web = _load_llm_web_kit_bindings()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()
        if HTML_ZLIB_COL not in df.columns:
            raise ValueError(f"Input batch is missing required HTML column: {HTML_ZLIB_COL!r}")
        html_values = df[HTML_ZLIB_COL].tolist()

        def _extract(html: object) -> str:
            html = coerce_html_text(html)
            if not isinstance(html, str) or not html.strip():
                return ""
            try:
                return json.dumps(self._web.get_feature(html))
            except Exception:
                return ""

        df["dom_feature"] = [_extract(h) for h in html_values]
        if "record_id" not in df.columns:
            df["record_id"] = [_record_id(row) for row in df.to_dict("records")]
        if HTML_CHARS_COL not in df.columns:
            df[HTML_CHARS_COL] = [len(coerce_html_text(h)) for h in html_values]
        df = df.drop(columns=["html"], errors="ignore")
        return DocumentBatch(dataset_name=batch.dataset_name, data=df)


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        if exact.exists():
            inp = exact
        else:
            candidates = sorted(inp.glob("*.parquet"))
            if not candidates:
                raise FileNotFoundError(f"No parquet files in {args.input}")
            inp = candidates[0]

    pf = pq.ParquetFile(str(inp))
    total = pf.metadata.num_rows
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards
    schema_names = set(pf.schema_arrow.names)
    if HTML_ZLIB_COL not in schema_names:
        raise ValueError(f"{inp} is missing required HTML column: {HTML_ZLIB_COL!r}")
    if "host_bucket" not in schema_names:
        raise ValueError(f"{inp} is missing required Stage 1a input column: 'host_bucket'")
    html_cols = [HTML_ZLIB_COL]
    need = [
        "snapshot",
        "record_id",
        "url",
        "url_host_name",
        "host_hash64",
        "host_bucket",
        "host_bucket_label",
        *html_cols,
        HTML_CHARS_COL,
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
        "source_manifest_file",
    ]
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
    if "host_bucket_label" not in shard_df.columns and "host_bucket" in shard_df.columns:
        shard_df["host_bucket_label"] = shard_df["host_bucket"].map(lambda value: f"{int(value):05d}")
    if "record_id" not in shard_df.columns and not shard_df.empty:
        shard_df["record_id"] = [_record_id(row) for row in shard_df.to_dict("records")]
    logger.info("shard {}/{}: {:,} pages", args.shard_index, args.num_shards, len(shard_df))
    if len(shard_df) == 0:
        return

    available_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    n_actors = max(1, available_cpus // max(1, args.cpus_per_actor))
    chunk = max(1, len(shard_df) // n_actors)
    tasks = [
        DocumentBatch(dataset_name="stage1a", data=shard_df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(shard_df), chunk)
    ]
    stage = DOMFeatureExtractionStage(cpus_per_actor=args.cpus_per_actor)
    pipeline = Pipeline(name="stage1a")
    pipeline.add_stage(stage)
    _init_ray_from_slurm()
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []

    out_df = (
        pd.concat([t.to_pandas() for t in result_tasks if hasattr(t, "to_pandas")], ignore_index=True)
        if result_tasks
        else pd.DataFrame(columns=OUTPUT_COLS)
    )
    for col in OUTPUT_COLS:
        if col not in out_df.columns:
            out_df[col] = None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_df["host_bucket"] = pd.to_numeric(out_df["host_bucket"], errors="raise").astype("int64")
    out_df["host_bucket_label"] = out_df["host_bucket"].map(lambda value: f"{int(value):05d}")
    sort_cols = [
        col
        for col in ("host_bucket", "url_host_name", "url", "warc_filename", "warc_record_offset", "record_id")
        if col in out_df.columns
    ]
    out_df = out_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    written = []
    for label, bucket_df in out_df.groupby("host_bucket_label", sort=True):
        out_path = out / f"host_bucket_{label}.parquet"
        tmp = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
        bucket_df[OUTPUT_COLS].to_parquet(str(tmp), index=False, compression="zstd")
        tmp.rename(out_path)
        written.append({"host_bucket_label": str(label), "rows": int(len(bucket_df)), "path": str(out_path)})
    summary_path = out / "_stage1a_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "input": str(inp),
                "output": str(out),
                "rows": int(len(out_df)),
                "bucket_files": len(written),
                "buckets": written,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    feat_ok = int((out_df["dom_feature"].astype(str) != "").sum())
    logger.info(
        "feature_ok={}/{}  bucket_files={}  summary -> {}",
        feat_ok,
        len(out_df),
        len(written),
        summary_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cpus-per-actor", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
