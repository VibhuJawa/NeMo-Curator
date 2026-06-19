#!/usr/bin/env python3  # noqa: EXE001
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
"""Dripper CPU-only pipeline — WARC fetch through Plan stage, no vLLM.

Runs the first 6 stages of the full pipeline to validate CPU stage correctness
without the 3-4 minute vLLM startup penalty:

  WARC fetch → parse → group → preprocess → cluster → plan → write

Output parquet contains intermediate columns (_dripper_needs_llm,
dripper_layout_cluster, dripper_layout_representative, etc.) useful for
inspecting clustering and planning behaviour.

Usage (Slurm):
  srun python pipeline_cpu_only.py --slurm \\
    --manifest-path /lustre/.../shard_0001.parquet \\
    --output-dir /lustre/.../output_cpu

Usage (local):
  python pipeline_cpu_only.py \\
    --manifest-path /path/to/manifest.parquet \\
    --output-dir /tmp/output_cpu \\
    --max-rows 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCReader
from nemo_curator.stages.text.download.common_crawl.warc_parse import WARCParseStage
from nemo_curator.stages.text.experimental.dripper.stages.clustering import DripperHTMLLayoutClusteringStage
from nemo_curator.stages.text.experimental.dripper.stages.grouping import HostDomainGroupingStage
from nemo_curator.stages.text.experimental.dripper.stages.layout_plan import DripperHTMLLayoutPlanStage
from nemo_curator.stages.text.experimental.dripper.stages.preprocess import DripperHTMLPreprocessStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import EmptyTask

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dripper CPU-only pipeline (no vLLM)")

    # --- logging ---
    p.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (DEBUG shows per-batch progress)",
    )

    # --- phase split (avoids the GPU-idle watchdog: cluster on GPU, plan on a CPU partition) ---
    p.add_argument(
        "--cluster-only",
        action="store_true",
        help="Phase 1a: run group->WARC->parse->preprocess->cluster only (no plan stage); write clustered parquet",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Phase 1b: input is a CLUSTERED parquet; run ONLY the LayoutPlan stage on CPU "
        "(uses precomputed --layout-id-col, no re-clustering, no GPU)",
    )
    p.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Phase A (CPU): run group->WARC->parse->preprocess and precompute the llm-webkit "
        "DOM feature (feature_only) into _dripper_layout_feature; write the preprocessed+feature "
        "parquet. No clustering, no GPU. Phase B (--cluster-from-preprocessed) consumes it.",
    )
    p.add_argument(
        "--cluster-from-preprocessed",
        action="store_true",
        help="Phase B (GPU): input is the Phase-A preprocessed+feature parquet; run ONLY "
        "group->cluster, consuming the precomputed _dripper_layout_feature column so no CPU "
        "feature extraction runs on the GPU node. Write the clustered parquet.",
    )
    p.add_argument(
        "--layout-id-col",
        default="dripper_layout_id",
        help="Precomputed layout-id column the plan stage consumes (skips CPU re-clustering)",
    )

    # --- I/O ---
    p.add_argument(
        "--manifest-path", required=True, help="Parquet manifest (Phase 1a) or clustered parquet (Phase 1b)"
    )
    p.add_argument("--output-dir", required=True, help="Output directory for plan-stage output")
    p.add_argument("--output-shards", type=int, default=8, help="Output shards after compaction")
    p.add_argument("--max-rows", type=int, default=0, help="Limit rows for smoke testing (0 = all)")
    p.add_argument(
        "--use-s3",
        action="store_true",
        default=None,
        help="Fetch WARCs from the S3/PBSS endpoint instead of the Common Crawl HTTPS endpoint, "
        "which rate-limits/403s under concurrency. Default None = respect the CC_USE_S3 env var "
        "(set with the PBSS creds + endpoint). Strongly prefer S3 for any real run.",
    )

    # --- Ray ---
    p.add_argument("--slurm", action="store_true", help="Use SlurmRayClient")
    p.add_argument("--ray-num-cpus", type=int, default=None)
    p.add_argument(
        "--ray-num-gpus", type=int, default=None, help="Expose GPUs to Ray (needed for cuML GPU clustering)"
    )
    p.add_argument(
        "--cluster-gpus",
        type=float,
        default=0.0,
        help="GPUs for the clustering stage (>0 enables cuML GPU clustering)",
    )
    p.add_argument(
        "--object-store-memory-gb",
        type=float,
        default=None,
        help=(
            "Ray object-store size in GB. Default (None) lets Ray pick (~30%% of RAM, often capped ~36 GiB), "
            "which OVERFLOWS on a mega-host's single feature block (e.g. tgcom24's 20k rows) and wedges the "
            "write stage. Set high on big-RAM CPU nodes (e.g. 150 on a 235 GB node)."
        ),
    )
    p.add_argument("--ray-temp-dir", default="/tmp/ray")  # noqa: S108
    p.add_argument("--ray-port", type=int, default=6379)

    # --- pipeline ---
    p.add_argument("--prompt-version", default="short_compact")
    p.add_argument("--min-rows-per-batch", type=int, default=1000)
    p.add_argument(
        "--max-rows-per-batch",
        type=int,
        default=None,
        help=(
            "Phase A balance: split a host with more rows than this into balanced ~equal "
            "preprocess batches so a mega-host (tgcom24 20k) spreads across actors instead of "
            "one (43-min serial tail -> ~4 min). The finalize then re-groups each host WHOLE "
            "(single-row-group shard) for Phase B clustering. Unset = one batch per host."
        ),
    )
    p.add_argument("--warc-max-workers", type=int, default=64)
    p.add_argument("--worker-count", type=int, default=None)

    # --- layout clustering ---
    p.add_argument("--layout-cluster-threshold", type=float, default=0.95)
    p.add_argument("--layout-template-min-cluster-size", type=int, default=2)
    p.add_argument("--layout-template-max-selected-item-ratio", type=float, default=0.50)
    p.add_argument("--layout-template-validation-rows", type=int, default=2)
    p.add_argument("--layout-template-validation-min-content-f1", type=float, default=0.98)
    p.add_argument("--layout-template-validation-signature-mode", default="none")
    p.add_argument("--layout-template-large-cluster-validation-rows", type=int, default=0)
    p.add_argument("--layout-template-large-cluster-min-size", type=int, default=0)
    p.add_argument("--layout-template-representative-candidates", type=int, default=1)
    p.add_argument("--layout-template-feature-source", default="raw_html")
    p.add_argument("--layout-template-propagation-target", default="raw_html")
    p.add_argument("--layout-template-propagation-content-source", default="converted")
    p.add_argument("--layout-page-signature-mode", default="none")
    p.add_argument("--layout-exact-query-value-keys", default="entityid,id")
    p.add_argument("--layout-template-prompt-dedup-fallback-min-fraction", type=float, default=0.0)
    p.add_argument("--layout-template-min-saved-call-pages", type=int, default=0)
    p.add_argument(
        "--layout-template-max-propagation-group-pages",
        type=int,
        default=0,
        help="Split layout groups larger than this into balanced sub-clusters "
        "(distributes mega-hosts like tgcom24 across Phase-2 shards). 0=off.",
    )
    p.add_argument("--layout-template-propagation-concurrency", type=int, default=1)
    p.add_argument("--dynamic-classid-similarity-threshold", type=float, default=0.85)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _read_unified_large(raw_dir: Path) -> pa.Table:  # noqa: F821
    """Read all parquet shards in raw_dir as one Arrow table, robust to cross-shard schema drift.

    PERMISSIVE schema unification promotes a column that is all-null in some shards (Arrow ``null``
    type) but typed (e.g. double) in others, so ``to_table()`` doesn't fail with "Unsupported cast
    from double to null" (tgcom24's data triggers this; the no-tgcom set did not). Then widen 32-bit
    string/binary columns to their 64-bit ``large_`` variants so a mega-host's html never overflows
    Arrow's 2 GB offset limit when taken/concatenated here or by any downstream stage.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    dset = ds.dataset(str(raw_dir), format="parquet")
    unified = pa.unify_schemas([frag.physical_schema for frag in dset.get_fragments()], promote_options="permissive")
    table = ds.dataset(str(raw_dir), format="parquet", schema=unified).to_table()
    return table.cast(
        pa.schema(
            [
                field.with_type(pa.large_string())
                if field.type == pa.string()
                else field.with_type(pa.large_binary())
                if field.type == pa.binary()
                else field
                for field in table.schema
            ]
        )
    )


def _consolidate_by_host(raw_dir: Path, output_dir: Path, host_col: str, min_rows_per_batch: int) -> int:
    """Re-group balanced (split) Phase A shards so each host is WHOLE in one shard.

    Phase A's `--max-rows-per-batch` split fragments a mega-host across shards for parallel
    preprocess. Phase B clusters PER HOST and needs all of a host's pages together, so here
    (on the head node, after the Ray run) we read the fragmented shards, group rows by host,
    and write host-whole shards: each big host (>= min_rows) gets its own shard; small hosts
    are bin-packed. Each shard is written as a SINGLE row group (`row_group_size = len`) so
    Ray reads it as exactly one block -- a big host is never re-split by row-group blocking,
    so Phase B's per-block HostDomainGrouping sees it whole. Returns shards written.

    Uses pyarrow (columnar read + per-host ``take``) -- faster and lighter than a pandas
    concat + groupby: Arrow string arrays avoid pandas' object-dtype blow-up, and ``take``
    gathers one shard at a time rather than building a full reordered copy of the table.
    """
    import pyarrow.parquet as pq

    table = _read_unified_large(raw_dir)
    # Build host -> row indices in one pass over just the host column (cheap), preserving
    # first-seen host order; each host's rows are then gathered with a single take().
    idx_by_host: dict[str, list[int]] = {}
    for i, host in enumerate(table.column(host_col).to_pylist()):
        idx_by_host.setdefault(str(host), []).append(i)

    def _write(indices: list[int], idx: int) -> None:
        pq.write_table(
            table.take(indices), output_dir / f"host_{idx:05d}.parquet", row_group_size=max(1, len(indices))
        )

    shard_i = 0
    small_idx: list[int] = []
    small_rows = 0
    for indices in idx_by_host.values():
        if len(indices) >= min_rows_per_batch:
            _write(indices, shard_i)  # big host -> its own WHOLE shard
            shard_i += 1
        else:
            small_idx.extend(indices)
            small_rows += len(indices)
            if small_rows >= min_rows_per_batch:
                _write(small_idx, shard_i)  # bin-packed small hosts (each whole)
                shard_i += 1
                small_idx, small_rows = [], 0
    if small_idx:
        _write(small_idx, shard_i)
        shard_i += 1
    return shard_i


def _rebalance_by_cluster(  # noqa: C901, PLR0912, PLR0915
    raw_dir: Path, output_dir: Path, cluster_col: str, n_shards: int
) -> int:
    """Re-shard the plan output into balanced shards, splitting mega-clusters into sub-clusters.

    Phase-2 propagation is BLOCK-LOCAL: a cluster's representative + its members must land in one
    Phase-2 block (= one input shard) or members cannot propagate. So we bin-pack WHOLE clusters
    into ``n_shards`` size-balanced bins (each unclustered row a size-1 unit). BUT a single dominant
    layout can be a huge cluster (tgcom24 ~= 15k pages); kept atomic it forces one multi-GB straggler
    shard whose Phase-2 postprocess serializes while the others idle. So any cluster larger than the
    target shard size is split into balanced sub-clusters, each given its OWN representative (any
    member works -- same layout), preserving block-local propagation while balancing the shards.
    Done here on the full table (not per-batch), so a cross-batch cluster splits cleanly. Returns
    shards written.
    """
    import heapq

    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    table = _read_unified_large(raw_dir)
    # Units to bin-pack: whole clusters, BUT split any cluster larger than the target shard size
    # into balanced sub-clusters so one dominant layout (tgcom24 = a single ~15k-page cluster) can't
    # force a 2.8 GB straggler shard that serializes Phase-2 postprocess. Each sub-cluster stays
    # block-local: it gets its OWN representative (any member is valid -- they all share the layout),
    # so the per-block finalize is unchanged; cost is a few extra rep LLM calls per mega-cluster
    # (negligible) and a SMALLER blast radius per rep (more F1-robust). Done here on the full table,
    # so a cross-batch cluster splits cleanly -- the per-batch plan-stage cap could not (sub-ids
    # collided across batches and re-merged).
    from nemo_curator.stages.text.experimental.dripper.stages._types import (
        _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL,
        _DRIPPER_NEEDS_LLM_COL,
    )
    from nemo_curator.utils.grouping import split_into_n_chunks

    _rep_col = "dripper_layout_representative"
    cluster_vals = table.column(cluster_col).to_pylist()
    units: dict[str, list[int]] = {}
    solo: list[int] = []
    for i, c in enumerate(cluster_vals):
        cs = str(c or "")
        if cs:
            units.setdefault(cs, []).append(i)
        else:
            solo.append(i)

    target = max(1, -(-table.num_rows // n_shards))  # ceil(rows / n_shards) = even shard size
    # Only materialize the row-level rep/needs/pending columns (and a mutable cluster-id copy) when a
    # cluster actually exceeds the target shard size -- the common shard has none, so skip the work.
    _can_split = any(len(idxs) > target for idxs in units.values()) and all(
        c in table.column_names for c in (_rep_col, _DRIPPER_NEEDS_LLM_COL, _DRIPPER_LAYOUT_PENDING_PROPAGATION_COL)
    )
    new_cluster = list(cluster_vals) if _can_split else cluster_vals
    rep = table.column(_rep_col).to_pylist() if _can_split else []
    needs = table.column(_DRIPPER_NEEDS_LLM_COL).to_pylist() if _can_split else []
    pend = table.column(_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL).to_pylist() if _can_split else []

    unit_list: list[list[int]] = []
    n_split = 0
    for cid, idxs in units.items():
        if not _can_split or len(idxs) <= target:
            unit_list.append(idxs)
            continue
        for k, sub in enumerate(split_into_n_chunks(idxs, -(-len(idxs) // target))):
            for j in sub:
                new_cluster[j] = f"{cid}#r{k:03d}"  # distinct sub-cluster id -> own finalize group
            if not any(rep[j] for j in sub):  # this sub-cluster lost the original rep -> mint one
                r = sub[0]
                rep[r], needs[r], pend[r] = True, True, False
            unit_list.append(sub)
        n_split += 1
    unit_list.extend([i] for i in solo)
    unit_list.sort(key=len, reverse=True)

    if _can_split and n_split:  # write back re-ided clusters + re-designated reps
        for col, vals in (
            (cluster_col, new_cluster),
            (_rep_col, rep),
            (_DRIPPER_NEEDS_LLM_COL, needs),
            (_DRIPPER_LAYOUT_PENDING_PROPAGATION_COL, pend),
        ):
            i = table.schema.get_field_index(col)
            table = table.set_column(i, table.schema.field(col), pa.array(vals, type=table.schema.field(col).type))
        logger.info("Rebalance split {} mega-cluster(s) (> {} rows) into balanced sub-clusters", n_split, target)

    # Greedy largest-first bin-pack into n_shards (min-heap on current bin size).
    bins: list[list[int]] = [[] for _ in range(n_shards)]
    heap = [(0, b) for b in range(n_shards)]
    heapq.heapify(heap)
    for unit in unit_list:
        size, b = heapq.heappop(heap)
        bins[b].extend(unit)
        heapq.heappush(heap, (size + len(unit), b))

    shard_i = 0
    for idxs in bins:
        if not idxs:
            continue
        pq.write_table(table.take(idxs), output_dir / f"shard_{shard_i:05d}.parquet", row_group_size=max(1, len(idxs)))
        shard_i += 1
    return shard_i


def _log_code_provenance() -> None:
    """Log the import path + content hash of the running nemo_curator dripper code so every job log
    proves which code actually executed -- catches a stale editable install or a failed rsync that
    would otherwise silently run old code (the failure mode that wasted hours debugging tgcom24)."""
    import hashlib

    import nemo_curator
    from nemo_curator.stages.text.experimental.dripper.stages import clustering, grouping, preprocess

    def _sha(path: str) -> str:
        try:
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()[:12]
        except OSError:
            return "??"

    files = " ".join(f"{Path(m.__file__).name}={_sha(m.__file__)}" for m in (clustering, grouping, preprocess))
    logger.info(
        "CODE PROVENANCE | nemo_curator={} | {} | _GPU_CLUSTER_MAX_N={}",
        nemo_curator.__file__,
        files,
        clustering._GPU_CLUSTER_MAX_N,
    )


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    started = time.monotonic()

    # ---- Ray init (no vLLM; GPUs optional for cuML clustering) ---------------
    ray_client_kwargs: dict = {"ray_temp_dir": args.ray_temp_dir}
    if args.ray_num_cpus:
        ray_client_kwargs["num_cpus"] = args.ray_num_cpus
    if args.ray_num_gpus:
        ray_client_kwargs["num_gpus"] = args.ray_num_gpus
    if args.object_store_memory_gb:
        # int bytes; raise the default cap so a mega-host's single block fits in plasma
        # instead of overflowing -> backpressure -> wedged write stage.
        ray_client_kwargs["object_store_memory"] = int(args.object_store_memory_gb * 1_000_000_000)
    if args.slurm:
        ray_client_kwargs["ray_port"] = args.ray_port

    ray_client = SlurmRayClient(**ray_client_kwargs) if args.slurm else RayClient(**ray_client_kwargs)
    ray_client.start()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    logger.info("CPU-only pipeline starting")
    logger.info("Output dir : {}", output_dir)
    _log_code_provenance()

    try:
        # ---- read manifest --------------------------------------------------
        manifest_path = args.manifest_path
        if args.max_rows > 0:
            import pandas as pd

            if os.path.isdir(manifest_path):
                files = sorted(glob.glob(os.path.join(manifest_path, "**/*.parquet"), recursive=True))
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            else:
                df = pd.read_parquet(manifest_path)
            df = df.head(args.max_rows)
            trimmed_path = str(raw_dir / "_manifest_trimmed.parquet")
            df.to_parquet(trimmed_path, index=False)
            manifest_path = trimmed_path
            logger.info("Trimmed manifest to {} rows → {}", args.max_rows, trimmed_path)

        # ---- build CPU-only stage list --------------------------------------
        shared_kwargs = {"html_col": "html", "url_col": "url"}

        # Fixed actor-pool sizing. When a stage's num_workers() is None, Ray Data
        # defaults actor concurrency to the AUTOSCALING tuple (1, max_actors): it
        # starts with a single actor and only ramps up under sustained backpressure,
        # so a streaming CPU workload never saturates the box (we measured CPU
        # peaking ~25/64 with a long single-actor tail). Pin a FIXED pool instead so
        # every actor is live from the first block.
        #
        # The pool must SHARE the CPU budget: Phase A runs two CPU actor stages
        # concurrently (extractor/preprocess + feature extraction) plus the upstream
        # WARC-fetch/parse/group task stages, all on the same cores. A fixed actor
        # pool reserves cpus for its whole lifetime, so pinning each stage to #CPUs
        # would over-reserve the box and starve the I/O tasks (deadlock risk). Split
        # ~7/8 of the cores across the concurrent CPU actor stages and leave the rest
        # for the I/O tasks (WARC fetch is I/O-bound: a few cpu slots back many fetch
        # threads). --worker-count overrides this CPU pool (the "extractor
        # concurrency" knob); GPU stages always size to #GPUs.
        ray_cpus = args.ray_num_cpus or os.cpu_count() or 1
        # Max number of CPU actor stages alive at once in any phase (preprocess +
        # feature extraction in Phase A); single-stage phases simply under-fill.
        _max_concurrent_cpu_actor_stages = 2
        _cpu_headroom = max(1, ray_cpus // 8)
        cpu_pool = args.worker_count or max(1, (ray_cpus - _cpu_headroom) // _max_concurrent_cpu_actor_stages)

        def actor_workers(gpus: float) -> int | None:
            if gpus > 0:
                # Fixed GPU actor pool sized to PACK the GPUs: floor(#GPUs / gpus_per_actor).
                # With cluster_gpus < 1 this runs several clustering actors on ONE GPU so their
                # (I/O-bound: read a host-whole shard, then a ~0.1s GPU cluster) work overlaps and
                # the GPU stays busy instead of idling between bursts. 1 GPU @ 1.0 -> 1 actor.
                return max(1, int((args.ray_num_gpus or 1) / gpus))
            return cpu_pool

        template_kwargs = {
            "layout_cluster_threshold": args.layout_cluster_threshold,
            "layout_template_min_cluster_size": args.layout_template_min_cluster_size,
            "layout_template_max_selected_item_ratio": args.layout_template_max_selected_item_ratio,
            "layout_template_validation_rows": args.layout_template_validation_rows,
            "layout_template_validation_min_content_f1": args.layout_template_validation_min_content_f1,
            "layout_template_validation_signature_mode": args.layout_template_validation_signature_mode,
            "layout_template_large_cluster_validation_rows": args.layout_template_large_cluster_validation_rows,
            "layout_template_large_cluster_min_size": args.layout_template_large_cluster_min_size,
            "layout_template_representative_candidates": args.layout_template_representative_candidates,
            "layout_template_feature_source": args.layout_template_feature_source,
            "layout_template_propagation_target": args.layout_template_propagation_target,
            "layout_template_propagation_content_source": args.layout_template_propagation_content_source,
            "layout_page_signature_mode": args.layout_page_signature_mode,
            "layout_exact_query_value_keys": args.layout_exact_query_value_keys or None,
            "layout_template_prompt_dedup_fallback_min_fraction": args.layout_template_prompt_dedup_fallback_min_fraction,
            "layout_template_min_saved_call_pages": args.layout_template_min_saved_call_pages,
            "layout_template_max_propagation_group_pages": args.layout_template_max_propagation_group_pages,
            "layout_template_propagation_concurrency": args.layout_template_propagation_concurrency,
            "dynamic_classid_similarity_threshold": args.dynamic_classid_similarity_threshold,
            "worker_count": actor_workers(0.0),  # plan stage runs on CPU
        }

        plan_stage = DripperHTMLLayoutPlanStage(
            **shared_kwargs,
            **template_kwargs,
            host_col="url_host_name",
            # Consume the precomputed GPU layout id instead of re-clustering every host on
            # CPU (web_bindings O(n^2)) -- the prior ~20-min/host stall that tripped the
            # GPU-idle watchdog. See _build_precomputed_layout_group_plans.
            layout_id_col=args.layout_id_col,
        )

        # --- WARC manifest reader (default / cluster-only / preprocess-only) ---------
        # Group by host FIRST (url_host_name is already in the manifest, no WARC needed): the
        # StreamingRepartition barrier then blocks only on the fast parquet read, and
        # WARC fetch -> parse -> preprocess STREAM per host-block (parallel across blocks)
        # instead of fetching all rows up front.
        def warc_reader() -> ParquetReader:
            return ParquetReader(
                file_paths=manifest_path,
                fields=["url_host_name", "url", "warc_filename", "warc_record_offset", "warc_record_length"],
            )

        def warc_preprocess_chain() -> list:
            return [
                HostDomainGroupingStage(
                    host_domain_col="url_host_name",
                    min_rows_per_batch=args.min_rows_per_batch,
                    max_rows_per_batch=args.max_rows_per_batch,  # Phase A: split big hosts for balance
                ),
                CommonCrawlWARCReader(
                    warc_filename_col="warc_filename",
                    warc_record_offset_col="warc_record_offset",
                    warc_record_length_col="warc_record_length",
                    binary_content_col="binary_content",
                    use_s3=args.use_s3,
                    max_workers=args.warc_max_workers,
                ),
                WARCParseStage(binary_content_col="binary_content", html_col="html"),
                DripperHTMLPreprocessStage(
                    **shared_kwargs,
                    prompt_version=args.prompt_version,
                    worker_count=actor_workers(0.0),  # CPU extractor: fixed pool = #CPUs
                ),
            ]

        def cluster_stage() -> DripperHTMLLayoutClusteringStage:
            return DripperHTMLLayoutClusteringStage(
                **shared_kwargs,
                host_col="url_host_name",  # cluster per host (NOT per unique url) so large hosts hit the GPU path
                layout_cluster_threshold=args.layout_cluster_threshold,
                layout_template_min_cluster_size=args.layout_template_min_cluster_size,
                layout_page_signature_mode=args.layout_page_signature_mode,
                layout_exact_query_value_keys=args.layout_exact_query_value_keys or None,
                layout_feature_source=args.layout_template_feature_source,
                worker_count=actor_workers(args.cluster_gpus),  # GPU clustering: pool sized to #GPUs
                resources=Resources(cpus=1.0, gpus=args.cluster_gpus),  # GPU cuML clustering when >0
            )

        if args.plan_only:
            # Phase 1b (CPU partition): input is a CLUSTERED parquet (Phase 1a output). Read all
            # columns and run ONLY the plan stage -- no group/WARC/parse/preprocess/cluster, no GPU.
            reader = ParquetReader(file_paths=manifest_path)
            stages = [plan_stage]
        elif args.preprocess_only:
            # Phase A (CPU partition): WARC -> parse -> preprocess, then precompute the llm-webkit
            # DOM feature (feature_only=True) into _dripper_layout_feature. No clustering, no GPU --
            # all CPU-heavy work lives here so the GPU phase (Phase B) stays busy on its watchdog.
            reader = warc_reader()
            stages = [
                *warc_preprocess_chain(),
                DripperHTMLLayoutClusteringStage(
                    **shared_kwargs,
                    host_col="url_host_name",
                    feature_only=True,
                    layout_feature_col="_dripper_layout_feature",
                    worker_count=actor_workers(0.0),  # CPU feature extract: fixed pool = #CPUs
                    resources=Resources(cpus=1.0, gpus=0.0),  # CPU only -- no GPU on Phase A
                ),
            ]
        elif args.cluster_from_preprocessed:
            # Phase B (GPU partition): input is the Phase-A preprocessed+feature parquet. Read ALL
            # columns (url_host_name, html, preprocess cols, _dripper_layout_feature) and run ONLY
            # group -> cluster. The clustering stage consumes the precomputed feature column, so no
            # CPU feature extraction runs on the GPU node (keeps GPU util above the watchdog floor).
            reader = ParquetReader(file_paths=manifest_path)
            stages = [
                HostDomainGroupingStage(
                    host_domain_col="url_host_name",
                    min_rows_per_batch=args.min_rows_per_batch,
                    # Split mega-hosts into <=max_rows_per_batch sub-blocks here too (not just Phase A):
                    # a consolidated mega-host (tgcom24 ~20k) would otherwise arrive as ONE block and
                    # stall the clustering's per-row sample build / overflow plasma. Deterministic
                    # _stable_layout_id merges identical layouts across sub-blocks -> no dedup loss.
                    max_rows_per_batch=args.max_rows_per_batch,
                ),
                cluster_stage(),
            ]
        elif args.cluster_only:
            # Phase 1a (GPU): group->WARC->parse->preprocess->cluster; write the clustered parquet.
            reader = warc_reader()
            stages = [*warc_preprocess_chain(), cluster_stage()]
        else:
            # Combined default path: full WARC->parse->preprocess->cluster then the plan stage
            # (now consuming the precomputed layout id).
            reader = warc_reader()
            stages = [*warc_preprocess_chain(), cluster_stage(), plan_stage]
        writer = ParquetWriter(path=str(raw_dir))

        pipeline = Pipeline(name="dripper-cpu-only")
        pipeline.add_stage(reader)
        for stage in stages:
            pipeline.add_stage(stage)
        pipeline.add_stage(writer)

        executor = RayDataExecutor()
        pipeline_start = time.monotonic()
        pipeline.run(executor=executor, initial_tasks=[EmptyTask()])
        pipeline_elapsed = time.monotonic() - pipeline_start
        logger.info("CPU pipeline done in {:.1f}s", pipeline_elapsed)

        # ---- compaction -----------------------------------------------------
        # Drop the raw WARC bytes (binary_content) before the Ray repartition: parse already
        # consumed them and they are dead weight downstream (~250 KB/row). Carrying them through
        # the repartition shuffle is what OOM-killed the Ray GCS on large runs. Compaction is
        # best-effort: on failure keep the _raw shards as the usable output (Phase 1b reads them).
        logger.info("Compacting {} → {} shards", raw_dir, args.output_shards)
        import ray as _ray

        compact_start = time.monotonic()
        if args.preprocess_only:
            # Phase A output feeds Phase B clustering, which clusters PER HOST -- a host must stay
            # WHOLE in one shard. DO NOT repartition (a Ray shuffle would scatter a big host across
            # shards and Phase B would cluster it in chunks).
            if args.max_rows_per_batch:
                # Big hosts were SPLIT into balanced batches for parallel preprocess, so they are now
                # fragmented across shards -> re-group each host whole (single-row-group shards) before
                # Phase B. Runs on the head node (binary_content already dropped, so data is modest).
                n_shards = _consolidate_by_host(raw_dir, output_dir, "url_host_name", args.min_rows_per_batch)
                shutil.rmtree(raw_dir, ignore_errors=True)
                compact_elapsed = time.monotonic() - compact_start
                logger.info("Phase A: consolidated split shards into {} host-whole shards", n_shards)
            else:
                # No split -> HostDomainGrouping already emitted one shard per host-group; just move.
                moved = 0
                for f in sorted(raw_dir.glob("*.parquet")):
                    shutil.move(str(f), str(output_dir / f.name))
                    moved += 1
                shutil.rmtree(raw_dir, ignore_errors=True)
                compact_elapsed = time.monotonic() - compact_start
                logger.info("Phase A: moved {} host-grouped shards to {} (no shuffle)", moved, output_dir)
        else:
            try:
                import glob as _glob

                import pyarrow.parquet as _pq

                _shard0 = next(iter(sorted(_glob.glob(str(raw_dir / "*.parquet")))), None)
                _has_cluster = _shard0 is not None and "dripper_layout_cluster" in _pq.read_schema(_shard0).names
                if _has_cluster:
                    # Plan output: bin-pack WHOLE layout clusters into balanced shards so Phase-2
                    # block-local propagation works (cluster intact) AND blocks are size-balanced.
                    # Folds the separate offline rebalance into the plan stage; Phase 2 reads this
                    # directly. Runs on the full-RAM compute node (no Ray shuffle that split clusters).
                    n_shards = _rebalance_by_cluster(raw_dir, output_dir, "dripper_layout_cluster", args.output_shards)
                    shutil.rmtree(raw_dir, ignore_errors=True)
                    compact_elapsed = time.monotonic() - compact_start
                    logger.info("Plan: rebalanced into {} cluster-intact balanced shards", n_shards)
                else:
                    compact_ds = _ray.data.read_parquet(str(raw_dir))
                    drop_cols = [c for c in ("binary_content",) if c in compact_ds.schema().names]
                    if drop_cols:
                        compact_ds = compact_ds.drop_columns(drop_cols)
                    compact_ds.repartition(args.output_shards).write_parquet(str(output_dir))
                    compact_elapsed = time.monotonic() - compact_start
                    logger.info("Compaction done in {:.1f}s", compact_elapsed)
                    shutil.rmtree(raw_dir)
            except Exception as exc:  # noqa: BLE001
                compact_elapsed = time.monotonic() - compact_start
                logger.warning("Compaction failed ({}); leaving _raw shards as the output at {}", exc, raw_dir)

        # ---- metrics --------------------------------------------------------
        total_elapsed = time.monotonic() - started
        metrics = {
            "mode": "cpu_only",
            "stages": ["warc_reader", "warc_parse", "host_group", "preprocess", "cluster", "plan"],
            "pipeline_elapsed_s": round(pipeline_elapsed, 2),
            "compaction_elapsed_s": round(compact_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "output_dir": str(output_dir),
            "output_shards": args.output_shards,
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Metrics: {}", metrics)

    finally:
        try:  # noqa: SIM105
            ray_client.stop()
        except Exception:  # noqa: BLE001, S110
            pass


if __name__ == "__main__":
    main()
