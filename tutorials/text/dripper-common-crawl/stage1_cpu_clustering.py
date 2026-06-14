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
stage1_cpu_clustering.py — Curator-native Stage 1: DOM clustering with fan-out/fan-in.

PIPELINE DESIGN
───────────────
Uses NeMo Curator's ProcessingStage + RayDataExecutor + IS_FANOUT_STAGE flag.
Three-stage pipeline:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                  Stage 1 Curator Pipeline                           │
    │                                                                     │
    │  ┌──────────────────────────────────────────────────┐              │
    │  │  FAN-OUT: HostPartitionStage                      │              │
    │  │  1 shard DocumentBatch → N host DocumentBatches   │              │
    │  │  IS_FANOUT_STAGE=True → repartition(1 per block)  │              │
    │  │  All N host blocks now flow independently         │              │
    │  └──────────────────┬───────────────────────────────┘              │
    │                     │ N independent blocks (one per host)           │
    │                     │                                               │
    │  ┌──────────────────▼───────────────────────────────┐              │
    │  │  GPU DBSCAN: DripperHTMLLayoutClusteringStage     │              │
    │  │  IS_ACTOR_STAGE=True (setup() override)           │              │
    │  │  resources=Resources(cpus=4.0, gpus=1.0)          │              │
    │  │  → RayDataExecutor spawns 1 actor per GPU         │              │
    │  │  → All N_GPU actors run concurrently              │              │
    │  │  → GPU DBSCAN via _load_llm_web_kit_bindings()    │              │
    │  │    (substitutes cluster_html_struct_gpu = cuML)   │              │
    │  └──────────────────┬───────────────────────────────┘              │
    │                     │ N processed blocks (layout_id assigned)       │
    │                     │                                               │
    │  ┌──────────────────▼───────────────────────────────┐              │
    │  │  FAN-IN: RepresentativeSelectionStage             │              │
    │  │  N host blocks → select 1 rep per cluster        │              │
    │  │  + add cluster_role, is_representative columns   │              │
    │  │  (still N blocks — merge at driver below)        │              │
    │  └──────────────────────────────────────────────────┘              │
    │                     │ N output blocks                               │
    │                     ▼                                               │
    │  Driver: concat N output tasks → write shard parquet               │
    └─────────────────────────────────────────────────────────────────────┘

CURATOR ACTOR PATTERN
──────────────────────
  IS_FANOUT_STAGE: after FAN-OUT stage, Ray Data calls
    repartition(target_num_rows_per_block=1)
    → each host group becomes its own block
    → actors pick up one host block at a time (no cross-host data leakage)

  IS_ACTOR_STAGE: DripperHTMLLayoutClusteringStage overrides setup()
    → RayDataExecutor creates one Ray actor per GPU
    → Heavy state (llm_web_kit bindings, cuML context) loaded once per actor
    → Actors held warm across blocks (no re-initialization per host)

SCALING
───────
  Horizontal (across Slurm nodes): --array=0-79, one Ray cluster per task.
    Each task independently processes 1/80 of the input host_buckets.
    xxhash bucketing guarantees all pages from same host → same task.

  Vertical (within node, N GPUs): RayDataExecutor auto-scales to N actors
    (N = available GPUs in the Ray cluster). All N GPUs run concurrently,
    each actor processes one host block at a time from the shared queue.

  Memory: bounded by block size (~1 host × ~235K pages × feature vectors).
    Input parquet read in row groups → never fully loaded into RAM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_LAYOUT_ID_COL = "dripper_layout_id"  # Curator's internal clustering output col

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "html",
    "cluster_id",  # "host:layout_id_suffix" | "" for singletons
    "cluster_role",  # "representative" | "sibling" | "singleton"
    "layout_cluster_id",  # legacy alias = cluster_id (Stage 3 compat)
    "is_representative",  # bool
    "cluster_size",  # int
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — FAN-OUT: 1 shard → N host-granular blocks
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class HostPartitionFanOutStage:
    """FAN-OUT: splits one shard DocumentBatch into N per-host DocumentBatches.

    IS_FANOUT_STAGE=True tells RayDataExecutor to call
      dataset.repartition(target_num_rows_per_block=1)
    after this stage, so each host group becomes its own independent Ray block.
    All subsequent stages process one host at a time — no cross-host leakage.

    Why fan-out here:
      DBSCAN is per-host. Each host must be fully present in one block so the
      actor sees all pages and can compute the N×N cosine similarity matrix.
      domain_complete sharding at task-creation time guarantees same-host pages
      land in same shard, but within a shard there may be 1000+ hosts. Splitting
      now lets all N GPU actors work in parallel on different hosts.
    """

    name: str = "HostPartitionFanOutStage"
    host_col: str = "url_host_name"
    min_host_pages: int = 1

    def ray_stage_spec(self) -> dict:
        from nemo_curator.backends.utils import RayStageSpecKeys

        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup(self, _worker_metadata: object = None) -> None:
        pass  # stateless — no setup needed

    def process(self, batch: object) -> list:  # returns list[DocumentBatch]
        """Split one DocumentBatch into N per-host DocumentBatches."""
        from nemo_curator.tasks import DocumentBatch

        df = batch.to_pandas() if hasattr(batch, "to_pandas") else batch
        if self.host_col not in df.columns:
            from urllib.parse import urlparse

            df = df.copy()
            df[self.host_col] = df["url"].map(lambda u: urlparse(str(u)).hostname or "")

        host_batches = []
        for host, host_df in df.groupby(self.host_col, sort=False):
            if len(host_df) < self.min_host_pages:
                continue
            host_batches.append(
                DocumentBatch(
                    task_id=f"host_{host}",
                    dataset_name=getattr(batch, "dataset_name", "stage1"),
                    data=host_df.reset_index(drop=True),
                )
            )

        logger.debug("FanOut: shard → %d host batches", len(host_batches))
        return host_batches


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — GPU DBSCAN: DripperHTMLLayoutClusteringStage (existing Curator stage)
# ─────────────────────────────────────────────────────────────────────────────
# Used directly from nemo_curator.stages.text.experimental.dripper.stage.
# Key properties:
#   - overrides setup() → IS_ACTOR_STAGE=True
#   - setup() calls _load_llm_web_kit_bindings() which substitutes
#     cluster_html_struct_gpu (cuML) for llm-webkit's CPU cluster_html_struct
#   - RayDataExecutor creates one actor per GPU (Resources(cpus=4, gpus=1))
#   - Each actor processes one host block at a time
#   - Output: adds _LAYOUT_ID_COL (stable SHA-1 hash per cluster)


# ─────────────────────────────────────────────────────────────────────────────
# Stage C — FAN-IN prep: representative selection per host cluster
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class RepresentativeSelectionStage:
    """FAN-IN prep: for each layout cluster in a host block, select 1 representative.

    Runs after DripperHTMLLayoutClusteringStage (which assigned layout_ids).
    Adds cluster_role, is_representative, cluster_size columns needed by Stage 2.

    The actual fan-in (merging N host blocks → 1 shard) happens at the driver
    after pipeline.run() returns — Curator's collect + concat pattern.

    Why this is still N→N (not N→1):
      The driver-level fan-in (concat) is more efficient than a Ray-level merge
      because the merged result fits easily in driver memory (cluster assignments
      are small compared to raw HTML). Keeping N blocks through the pipeline
      maximizes parallelism up to this point.
    """

    name: str = "RepresentativeSelectionStage"
    html_col: str = "html"
    host_col: str = "url_host_name"
    min_cluster_size: int = 2

    _web_bindings: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def setup(self, _worker_metadata: object = None) -> None:
        """Load llm_web_kit bindings once per actor (triggers IS_ACTOR_STAGE)."""
        if self._initialized:
            return
        from nemo_curator.stages.text.experimental.dripper.stage import (
            _load_llm_web_kit_bindings,
        )

        self._web_bindings = _load_llm_web_kit_bindings()
        self._initialized = True

    def process(self, batch: object) -> object:
        """Add representative role columns to one host block."""
        if not self._initialized:
            self.setup()

        from nemo_curator.tasks import DocumentBatch

        df = batch.to_pandas() if hasattr(batch, "to_pandas") else batch
        df = self._assign_roles(df)
        return DocumentBatch(
            task_id=getattr(batch, "task_id", ""),
            dataset_name=getattr(batch, "dataset_name", "stage1"),
            data=df,
        )

    def _assign_roles(self, df: pd.DataFrame) -> pd.DataFrame:
        cluster_id_col = [""] * len(df)
        cluster_role_col = ["singleton"] * len(df)
        is_rep_col = [False] * len(df)
        cluster_size_col = [1] * len(df)

        if _LAYOUT_ID_COL not in df.columns:
            df["cluster_id"] = cluster_id_col
            df["cluster_role"] = cluster_role_col
            df["layout_cluster_id"] = cluster_id_col
            df["is_representative"] = is_rep_col
            df["cluster_size"] = cluster_size_col
            return df

        layout_ids = df[_LAYOUT_ID_COL].fillna("").tolist()
        by_lid: dict[str, list[int]] = defaultdict(list)
        for i, lid in enumerate(layout_ids):
            if lid:
                by_lid[lid].append(i)

        for lid, indices in by_lid.items():
            if len(indices) < self.min_cluster_size:
                continue  # leave as singletons

            candidates = [{"track_id": str(i), "html": str(df.iloc[i].get(self.html_col, "") or "")} for i in indices]
            try:
                rep = self._web_bindings.select_representative_html(candidates)
                rep_idx = int(rep["track_id"]) if rep else indices[0]
            except Exception:
                rep_idx = indices[0]

            host = str(df.iloc[indices[0]].get(self.host_col, ""))
            cid = f"{host}:{lid[:12]}"

            for i in indices:
                is_rep = i == rep_idx
                cluster_id_col[i] = cid
                cluster_role_col[i] = "representative" if is_rep else "sibling"
                is_rep_col[i] = is_rep
                cluster_size_col[i] = len(indices)

        df["cluster_id"] = cluster_id_col
        df["cluster_role"] = cluster_role_col
        df["layout_cluster_id"] = cluster_id_col
        df["is_representative"] = is_rep_col
        df["cluster_size"] = cluster_size_col
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Curator ProcessingStage wrappers (adds .inputs/.outputs/.batch_size/.resources)
# ─────────────────────────────────────────────────────────────────────────────


def _make_fanout_stage(host_col: str, min_host_pages: int) -> object:
    """Wrap HostPartitionFanOutStage as a Curator ProcessingStage."""
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.tasks import DocumentBatch

    inner = HostPartitionFanOutStage(host_col=host_col, min_host_pages=min_host_pages)

    @dataclass(kw_only=True)
    class _FanOutStage(ProcessingStage):
        name: str = "HostPartitionFanOutStage"
        resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
        batch_size: int = 1

        def inputs(self) -> tuple:
            return ["data"], ["url", host_col, "html"]

        def outputs(self) -> tuple:
            return ["data"], ["url", host_col, "html"]

        def ray_stage_spec(self) -> dict:
            from nemo_curator.backends.utils import RayStageSpecKeys

            return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

        def process(self, batch: DocumentBatch) -> list:
            return inner.process(batch)

    return _FanOutStage()


def _make_repsel_stage(html_col: str, host_col: str, min_cluster_size: int) -> object:
    """Wrap RepresentativeSelectionStage as a Curator ProcessingStage."""
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.tasks import DocumentBatch

    inner = RepresentativeSelectionStage(
        html_col=html_col,
        host_col=host_col,
        min_cluster_size=min_cluster_size,
    )

    @dataclass(kw_only=True)
    class _RepSelStage(ProcessingStage):
        name: str = "RepresentativeSelectionStage"
        # setup() override → IS_ACTOR_STAGE automatically
        resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))
        batch_size: int = 1

        def inputs(self) -> tuple:
            return ["data"], ["url", host_col, _LAYOUT_ID_COL]

        def outputs(self) -> tuple:
            return ["data"], ["cluster_id", "cluster_role", "is_representative", "cluster_size"]

        def setup(self, _worker_metadata: object = None) -> None:
            inner.setup()

        def process(self, batch: DocumentBatch) -> DocumentBatch:
            return inner.process(batch)

    return _RepSelStage()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline runner
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Stage1Config:
    """Groups run_stage1 parameters to avoid PLR0913 (too-many-arguments)."""

    input_path: str
    output_dir: str
    shard_index: int
    num_shards: int
    threshold: float
    min_cluster_size: int
    max_host_pages: int


def _load_shard(cfg: Stage1Config) -> pd.DataFrame:
    """Stream-read the shard slice from the input parquet."""
    pf = pq.ParquetFile(cfg.input_path)
    total_rows = pf.metadata.num_rows
    shard_start = total_rows * cfg.shard_index // cfg.num_shards
    shard_end = total_rows * (cfg.shard_index + 1) // cfg.num_shards
    need_cols = ["url", "url_host_name", "html", "warc_filename", "warc_record_offset", "warc_record_length"]
    read_cols = [c for c in need_cols if c in pf.schema_arrow.names]
    rows_seen, shard_parts = 0, []
    for batch in pf.iter_batches(batch_size=65_536, columns=read_cols):
        batch_df = batch.to_pandas()
        lo = max(0, shard_start - rows_seen)
        hi = min(len(batch_df), shard_end - rows_seen)
        rows_seen += len(batch_df)
        if lo < hi:
            shard_parts.append(batch_df.iloc[lo:hi])
        if rows_seen >= shard_end:
            break
    return pd.concat(shard_parts, ignore_index=True) if shard_parts else pd.DataFrame()


def _write_shard_result(result_df: pd.DataFrame, cfg: Stage1Config, n_gpus: int, elapsed: float) -> dict:
    """Ensure output columns, write parquet, compute and return metrics dict."""
    for col in OUTPUT_COLS:
        if col not in result_df.columns:
            result_df[col] = None
    out_cols = [c for c in OUTPUT_COLS if c in result_df.columns]
    result_df = result_df[out_cols]

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard_{cfg.shard_index:04d}.parquet" if cfg.num_shards > 1 else "shard_0000.parquet"
    out_path = out_dir / shard_name

    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    n_reps = int((result_df.get("cluster_role", pd.Series(dtype=str)) == "representative").sum())
    n_sing = int((result_df.get("cluster_role", pd.Series(dtype=str)) == "singleton").sum())
    call_reduction = 1.0 - (n_reps + n_sing) / max(len(result_df), 1)

    metrics = {
        "shard_index": cfg.shard_index,
        "num_shards": cfg.num_shards,
        "total_pages": len(result_df),
        "representative_pages": n_reps,
        "singleton_pages": n_sing,
        "call_reduction_fraction": call_reduction,
        "n_gpu_actors": max(1, n_gpus),
        "elapsed_s": elapsed,
        "pages_per_s": len(result_df) / max(elapsed, 1),
        "output_path": str(out_path),
    }
    metrics_path = out_path.with_name(f"metrics_shard_{cfg.shard_index:04d}.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info(
        "Stage 1 shard %d: %d pages | reps=%d | singletons=%d | call_reduction=%.1f%% | %.0f pages/s | %d GPU actors",
        cfg.shard_index,
        len(result_df),
        n_reps,
        n_sing,
        call_reduction * 100,
        metrics["pages_per_s"],
        metrics["n_gpu_actors"],
    )
    return metrics


def run_stage1(cfg: Stage1Config) -> dict:
    """Run Stage 1 via Curator's Pipeline + RayDataExecutor.

    Pipeline: FanOut → GPU DBSCAN → RepresentativeSelection → (driver fan-in)
    """
    import ray

    from nemo_curator.backends.ray_data.executor import RayDataExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.experimental.dripper.stage import (
        DripperHTMLLayoutClusteringStage,
    )
    from nemo_curator.tasks import DocumentBatch

    # ── 1. Init Ray ───────────────────────────────────────────────────────────
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""}},
    )
    n_gpus = int(ray.available_resources().get("GPU", 0))
    logger.info("Ray cluster: GPUs=%d CPUs=%d", n_gpus, int(ray.available_resources().get("CPU", 1)))

    # ── 2. Load shard from input parquet (streaming row-group reads) ──────────
    shard_df = _load_shard(cfg)
    logger.info(
        "Shard %d/%d: %d pages, %d unique hosts",
        cfg.shard_index,
        cfg.num_shards,
        len(shard_df),
        shard_df["url_host_name"].nunique() if "url_host_name" in shard_df.columns else 0,
    )

    if len(shard_df) == 0:
        return {"shard_index": cfg.shard_index, "total_pages": 0, "skipped": True}

    # ── 3. Create initial tasks (domain-complete: one task per host bucket) ───
    # Sort by host so same-host pages are contiguous, then create one task
    # per large-enough host group. This is the pre-fan-out grouping that ensures
    # the FanOut stage receives well-formed host groups.
    shard_df = shard_df.sort_values("url_host_name").reset_index(drop=True)
    initial_tasks = [DocumentBatch(task_id="shard_input", dataset_name="stage1", data=shard_df)]

    # ── 4. Build Curator pipeline: FanOut → DBSCAN → RepSel ──────────────────
    pipeline = Pipeline(
        name="stage1_dom_clustering",
        description="Stage 1: host fan-out → GPU DBSCAN → representative selection",
    )

    # Stage A: FAN-OUT — 1 shard → N host blocks
    pipeline.add_stage(_make_fanout_stage(host_col="url_host_name", min_host_pages=1))

    # Stage B: GPU DBSCAN — DripperHTMLLayoutClusteringStage
    # setup() override → actor mode → 1 actor per GPU, all GPUs concurrent
    pipeline.add_stage(
        DripperHTMLLayoutClusteringStage(
            html_col="html",
            url_col="url",
            host_col="url_host_name",
            layout_id_col=_LAYOUT_ID_COL,
            layout_cluster_threshold=cfg.threshold,
            layout_template_min_cluster_size=cfg.min_cluster_size,
            layout_template_max_exact_host_pages=cfg.max_host_pages,
            worker_count=max(1, n_gpus) if n_gpus > 0 else None,
        )
    )

    # Stage C: Representative selection — IS_ACTOR_STAGE (setup() override)
    pipeline.add_stage(
        _make_repsel_stage(
            html_col="html",
            host_col="url_host_name",
            min_cluster_size=cfg.min_cluster_size,
        )
    )

    # ── 5. Execute pipeline ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    output_tasks = pipeline.run(
        executor=RayDataExecutor(),
        initial_tasks=initial_tasks,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Pipeline executed: %d output tasks in %.1fs", len(output_tasks), elapsed)

    # ── 6. FAN-IN: driver-level merge of N host blocks → 1 shard output ──────
    # N host DocumentBatch tasks → concat → single shard DataFrame
    result_dfs = [t.to_pandas() for t in output_tasks]
    result_df = pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()
    logger.info("Fan-in: merged %d host batches → %d rows", len(result_dfs), len(result_df))

    # ── 7. Write output and compute metrics ───────────────────────────────────
    metrics = _write_shard_result(result_df, cfg, n_gpus, elapsed)

    ray.shutdown()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Stage 1: Curator fan-out/GPU-DBSCAN/fan-in DOM clustering")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-host-pages", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    # Idempotency check
    out_dir = Path(args.output)
    out_path = out_dir / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "shard_0000.parquet")
    if out_path.exists():
        try:
            n = pq.ParquetFile(str(out_path)).metadata.num_rows
            if n > 0:
                logger.info("Output already complete (%d rows) — skipping", n)
                return 0
        except Exception:
            logger.debug("Existing output unreadable — will re-run the stage")  # fall through

    metrics = run_stage1(
        Stage1Config(
            input_path=args.input,
            output_dir=args.output,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            threshold=args.threshold,
            min_cluster_size=args.min_cluster_size,
            max_host_pages=args.max_host_pages,
        )
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
