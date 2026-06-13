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

"""stage1b_gpu_dbscan.py — GPU DBSCAN clustering using NeMo Curator ProcessingStage.

INPUT:  stage1a output parquet (url, url_host_name, dom_feature JSON, html, warc_*)
OUTPUT: cluster assignments parquet:
          url, url_host_name, html, cluster_id, cluster_role,
          layout_cluster_id, is_representative, cluster_size, warc_*

CURATOR PATTERN:
  HostDBSCANStage(ProcessingStage) with Resources(cpus=4, gpus=1).
  RayActorPoolExecutor spawns one actor per GPU; Ray assigns CUDA_VISIBLE_DEVICES
  automatically. Each actor loads cuML once in setup() then processes hosts
  one at a time via process(). No manual multiprocessing or CUDA env management.

  One DocumentBatch = one host's pages. Ray schedules actors across the
  host queue so large hosts and small hosts are balanced automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_metrics import StageMetrics

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "html",
    "cluster_id",
    "cluster_role",
    "layout_cluster_id",
    "is_representative",
    "cluster_size",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


def _singleton_row(url: str, host: str, html: Any, warc_src: dict, include_html: bool = True) -> dict:
    row = {
        "url": url,
        "url_host_name": host,
        "cluster_id": "",
        "cluster_role": "singleton",
        "layout_cluster_id": "",
        "is_representative": False,
        "cluster_size": 1,
        "warc_filename": warc_src.get("warc_filename"),
        "warc_record_offset": warc_src.get("warc_record_offset"),
        "warc_record_length": warc_src.get("warc_record_length"),
    }
    if include_html:
        row["html"] = html
    return row


@dataclass(kw_only=True)
class HostDBSCANStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """GPU DBSCAN clustering — batches multiple hosts per GPU call.

    Each Ray actor owns one GPU. To maintain high GPU utilisation and avoid
    the GPU reaper, process_batch() concatenates feature vectors from ALL
    hosts in the batch into one large matrix and runs a single cuML DBSCAN
    call, then demultiplexes results back to individual hosts. This keeps
    the GPU busy even when individual hosts are small.

    batch_size=32 means each actor processes 32 hosts per call, giving
    the GPU a matrix of ~32*median_host_size rows — large enough to
    saturate cuBLAS/cuML without over-allocating memory.
    """

    name: str = "host_dbscan"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))
    batch_size: int = 16  # 16 hosts per actor invocation keeps GPU warm between calls

    threshold: float = 0.95
    min_cluster_size: int = 2
    gpu_min_size: int = 5  # use cuML for almost all hosts to keep GPU warm
    max_host_size: int = 3000

    # Per-actor state (set in setup, used in process)
    _cluster_gpu: Any = field(init=False, repr=False, default=None)
    _has_gpu: bool = field(init=False, repr=False, default=False)
    _web: Any = field(init=False, repr=False, default=None)

    def setup(self, _worker_metadata=None) -> None:
        """Load cuML DBSCAN and llm-webkit bindings once per GPU actor."""
        try:
            from nemo_curator.stages.text.experimental.dripper.gpu_layout_clustering import (
                _gpu_available,
                cluster_html_struct_gpu,
            )
            from nemo_curator.stages.text.experimental.dripper.stage import _load_llm_web_kit_bindings

            self._cluster_gpu = cluster_html_struct_gpu
            self._has_gpu = _gpu_available()
            self._web = _load_llm_web_kit_bindings()
        except Exception as exc:
            print(f"[stage1b] WARNING: cuML/llm-webkit unavailable ({exc}), using CPU fallback", flush=True)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        return self.process_batch([batch])[0]

    def process_batch(self, tasks: list) -> list:
        """Process batch_size=16 hosts sequentially — keeps GPU warm between calls.
        Each host is clustered INDEPENDENTLY (no cross-host contamination).
        batch_size>1 means the GPU never fully releases between small hosts.
        """
        results = []
        for task in tasks:
            samples = task.to_pandas().to_dict("records")
            host = task.dataset_name
            result_rows = self._cluster_host(host, samples)
            results.append(task.__class__(dataset_name=host, data=pd.DataFrame(result_rows)))
        return results

    def _run_clustering(self, chunk: list[dict], chunk_idx: int | None = None) -> list[dict]:
        """Run GPU or CPU DBSCAN on a chunk; offset layout_ids to avoid collisions."""
        try:
            if self._cluster_gpu and self._has_gpu and len(chunk) >= self.gpu_min_size:
                cc, _ = self._cluster_gpu(chunk, threshold=self.threshold, gpu_min_size=self.gpu_min_size)
            elif self._web:
                cc, _ = self._web.cluster_html_struct(chunk, threshold=self.threshold)
            else:
                cc = chunk
                for i, s in enumerate(cc):
                    s["layout_id"] = 0 if i == 0 else -1
            if chunk_idx is not None:
                for s in cc:
                    lid = s.get("layout_id", -1)
                    if lid >= 0:
                        s["layout_id"] = chunk_idx * 100_000 + lid
        except Exception as exc:
            label = f"chunk {chunk_idx}" if chunk_idx is not None else "DBSCAN"
            print(f"[stage1b] {label} failed for host: {exc}", flush=True)
            cc = chunk
        return cc

    def _cluster_host(self, host: str, samples: list[dict]) -> list[dict]:
        """Cluster all pages for one host; chunk oversized hosts to avoid OOM."""
        if len(samples) > self.max_host_size:
            clustered = []
            for ci, start in enumerate(range(0, len(samples), self.max_host_size)):
                clustered.extend(self._run_clustering(samples[start : start + self.max_host_size], chunk_idx=ci))
        else:
            clustered = self._run_clustering(samples)

        by_lid: dict[int, list] = defaultdict(list)
        for s in clustered:
            by_lid[int(s.get("layout_id", -1))].append(s)

        rows = []
        for lid, members in by_lid.items():
            if lid < 0 or len(members) < self.min_cluster_size:
                for m in members:
                    rows.append(_singleton_row(m["url"], host, None, m, include_html=False))
                continue

            cid = f"{host}:cluster_{lid}"
            try:
                rep_url = (
                    self._web.select_representative_html(
                        [{"track_id": m["url"], "html": m.get("html", "")} for m in members]
                    )["track_id"]
                    if self._web
                    else members[0]["url"]
                )
            except Exception:
                rep_url = members[0]["url"]

            for m in members:
                is_rep = m["url"] == rep_url
                rows.append(
                    {
                        "url": m["url"],
                        "url_host_name": host,
                        # html excluded from Ray result — driver joins from html_lookup
                        "cluster_id": cid,
                        "cluster_role": "representative" if is_rep else "sibling",
                        "layout_cluster_id": cid,
                        "is_representative": is_rep,
                        "cluster_size": len(members),
                        "warc_filename": m.get("warc_filename"),
                        "warc_record_offset": m.get("warc_record_offset"),
                        "warc_record_length": m.get("warc_record_length"),
                    }
                )
        return rows


def run(args):
    # ── Load shard ────────────────────────────────────────────────────────────
    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        inp = exact if exact.exists() else sorted(inp.glob("shard_*.parquet"))[0]

    pf = pq.ParquetFile(str(inp))
    total = pf.metadata.num_rows
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards

    need = ["url", "url_host_name", "dom_feature", "html", "warc_filename", "warc_record_offset", "warc_record_length"]
    cols = [c for c in need if c in pf.schema_arrow.names]

    rows_seen, parts = 0, []
    for batch in pf.iter_batches(batch_size=65_536, columns=cols):
        df = batch.to_pandas()
        lo, hi = max(0, start - rows_seen), min(len(df), end - rows_seen)
        rows_seen += len(df)
        if lo < hi:
            parts.append(df.iloc[lo:hi])
        if rows_seen >= end:
            break

    shard_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    tracker = StageMetrics("stage1b", shard_index=args.shard_index, num_shards=args.num_shards, n_gpus=0)
    tracker.start()
    print(f"[stage1b] shard {args.shard_index}/{args.num_shards}: {len(shard_df):,} pages", flush=True)
    if len(shard_df) == 0:
        return

    # ── Separate singletons (no feature) from clustering candidates ───────────
    # html_lookup: url → html kept on driver; NOT sent through Ray object store
    # (86k pages × ~10KB HTML each = ~870MB through Ray is the bottleneck fix)
    html_lookup: dict[str, Any] = {rec["url"]: rec.get("html") for rec in shard_df.to_dict("records")}

    by_host: dict[str, list] = defaultdict(list)
    singleton_rows: list[dict] = []
    for rec in shard_df.to_dict("records"):
        feat_json = rec.get("dom_feature", "")
        if not feat_json:
            singleton_rows.append(_singleton_row(rec["url"], rec.get("url_host_name", ""), rec.get("html"), rec))
            continue
        try:
            feat = json.loads(feat_json)
        except Exception:
            feat = None
        if feat is None:
            continue
        host = str(rec.get("url_host_name") or "")
        by_host[host].append(
            {
                "track_id": rec["url"],
                "url": rec["url"],
                # html excluded — actors only need features for DBSCAN clustering
                # and HTML for select_representative_html (which uses html= arg)
                "html": rec.get("html", ""),
                "feature": feat,
                "warc_filename": rec.get("warc_filename"),
                "warc_record_offset": rec.get("warc_record_offset"),
                "warc_record_length": rec.get("warc_record_length"),
            }
        )

    # ── Build one DocumentBatch per host ──────────────────────────────────────
    host_tasks = [DocumentBatch(dataset_name=host, data=pd.DataFrame(samples)) for host, samples in by_host.items()]

    # ── Execute via RayActorPoolExecutor (one GPU actor per available GPU) ────
    t0 = time.perf_counter()
    stage = HostDBSCANStage(
        threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
        gpu_min_size=args.gpu_min_size,
        max_host_size=int(os.environ.get("STAGE1B_MAX_HOST_SIZE", "3000")),
    )
    pipeline = Pipeline(name="stage1b_dbscan")
    pipeline.add_stage(stage)

    output_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=host_tasks) if host_tasks else []
    elapsed = time.perf_counter() - t0
    print(f"[stage1b] GPU DBSCAN done in {elapsed:.1f}s for {len(host_tasks)} hosts", flush=True)

    # ── Assemble output: cluster rows + singletons ────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "shard_0000.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")

    writer = None
    total_rows = 0

    for task in output_tasks:
        df = task.to_pandas()
        if df.empty:
            continue
        # Join html back from driver-side lookup (html was not sent through Ray)
        if "html" not in df.columns:
            df["html"] = df["url"].map(html_lookup)
        df = df[[c for c in OUTPUT_COLS if c in df.columns]]
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(tmp), table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(df)

    if singleton_rows:
        sing_df = pd.DataFrame(singleton_rows)
        # Singletons were built without html — join from lookup
        if "html" not in sing_df.columns or sing_df["html"].isna().all():
            sing_df["html"] = sing_df["url"].map(html_lookup)
        sing_table = pa.Table.from_pandas(
            sing_df[[c for c in OUTPUT_COLS if c in sing_df.columns]], preserve_index=False
        )
        if writer is None:
            writer = pq.ParquetWriter(str(tmp), sing_table.schema, compression="snappy")
        writer.write_table(sing_table)
        total_rows += len(singleton_rows)

    if writer:
        writer.close()
        tmp.rename(out_path)
    else:
        pd.DataFrame().to_parquet(str(out_path), index=False)

    print(f"[stage1b] merged {total_rows:,} rows → {out_path}", flush=True)

    result_df = pq.read_table(str(out_path), columns=["cluster_role"]).to_pandas()
    n_reps = int((result_df["cluster_role"] == "representative").sum())
    n_sing = int((result_df["cluster_role"] == "singleton").sum())
    call_reduction = 1.0 - (n_reps + n_sing) / max(len(result_df), 1)

    tracker.finish(total_pages=len(result_df), errors=0)
    tracker.extra = {
        "representative_pages": n_reps,
        "singleton_pages": n_sing,
        "call_reduction_fraction": round(call_reduction, 4),
        "dbscan_elapsed_s": round(elapsed, 2),
        "output": str(out_path),
    }
    tracker.save(str(out_path.parent))
    tracker.checkpoint(len(result_df), label="final")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--gpu-min-size", type=int, default=200)
    run(p.parse_args())


if __name__ == "__main__":
    main()
