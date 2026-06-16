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

    # --- I/O ---
    p.add_argument("--manifest-path", required=True, help="Parquet manifest with WARC coordinates")
    p.add_argument("--output-dir", required=True, help="Output directory for plan-stage output")
    p.add_argument("--output-shards", type=int, default=8, help="Output shards after compaction")
    p.add_argument("--max-rows", type=int, default=0, help="Limit rows for smoke testing (0 = all)")
    p.add_argument("--use-s3", action="store_true", help="Fetch WARCs from S3 (default: HTTPS)")

    # --- Ray ---
    p.add_argument("--slurm", action="store_true", help="Use SlurmRayClient")
    p.add_argument("--ray-num-cpus", type=int, default=None)
    p.add_argument("--ray-temp-dir", default="/tmp/ray")  # noqa: S108
    p.add_argument("--ray-port", type=int, default=6379)

    # --- pipeline ---
    p.add_argument("--prompt-version", default="short_compact")
    p.add_argument("--min-rows-per-batch", type=int, default=1000)
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
    p.add_argument("--layout-template-failed-host-fallback-signature-mode", default="none")
    p.add_argument("--layout-template-failed-layout-fallback-signature-mode", default="none")
    p.add_argument("--layout-template-host-single-cluster-min-pages", type=int, default=0)
    p.add_argument("--layout-template-host-single-cluster-max-pages", type=int, default=0)
    p.add_argument("--layout-template-max-exact-host-pages", type=int, default=0)
    p.add_argument(
        "--layout-template-large-host-mode",
        default="standalone",
        choices=["standalone", "feature_hash", "dom_path_hash"],
    )
    p.add_argument("--layout-template-prompt-dedup-fallback-min-fraction", type=float, default=0.0)
    p.add_argument("--layout-template-min-saved-call-pages", type=int, default=0)
    p.add_argument("--layout-template-propagation-concurrency", type=int, default=1)
    p.add_argument("--dynamic-classid-similarity-threshold", type=float, default=0.85)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    started = time.monotonic()

    # ---- Ray init (CPU only — no num_gpus) ----------------------------------
    ray_client_kwargs: dict = {"ray_temp_dir": args.ray_temp_dir}
    if args.ray_num_cpus:
        ray_client_kwargs["num_cpus"] = args.ray_num_cpus
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
        template_kwargs = {
            "layout_cluster_threshold": args.layout_cluster_threshold,
            "layout_template_min_cluster_size": args.layout_template_min_cluster_size,
            "layout_template_max_exact_host_pages": args.layout_template_max_exact_host_pages,
            "layout_template_large_host_mode": args.layout_template_large_host_mode,
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
            "layout_template_failed_host_fallback_signature_mode": args.layout_template_failed_host_fallback_signature_mode,
            "layout_template_failed_layout_fallback_signature_mode": args.layout_template_failed_layout_fallback_signature_mode,
            "layout_template_host_single_cluster_min_pages": args.layout_template_host_single_cluster_min_pages,
            "layout_template_host_single_cluster_max_pages": args.layout_template_host_single_cluster_max_pages,
            "layout_template_prompt_dedup_fallback_min_fraction": args.layout_template_prompt_dedup_fallback_min_fraction,
            "layout_template_min_saved_call_pages": args.layout_template_min_saved_call_pages,
            "layout_template_propagation_concurrency": args.layout_template_propagation_concurrency,
            "dynamic_classid_similarity_threshold": args.dynamic_classid_similarity_threshold,
            "worker_count": args.worker_count,
        }

        reader = ParquetReader(
            file_paths=manifest_path,
            fields=["url_host_name", "url", "warc_filename", "warc_record_offset", "warc_record_length"],
        )
        stages = [
            CommonCrawlWARCReader(
                warc_filename_col="warc_filename",
                warc_record_offset_col="warc_record_offset",
                warc_record_length_col="warc_record_length",
                binary_content_col="binary_content",
                use_s3=args.use_s3,
                max_workers=args.warc_max_workers,
            ),
            WARCParseStage(binary_content_col="binary_content", html_col="html"),
            HostDomainGroupingStage(
                host_domain_col="url_host_name",
                min_rows_per_batch=args.min_rows_per_batch,
            ),
            DripperHTMLPreprocessStage(
                **shared_kwargs,
                prompt_version=args.prompt_version,
                worker_count=args.worker_count,
            ),
            DripperHTMLLayoutClusteringStage(
                **shared_kwargs,
                layout_cluster_threshold=args.layout_cluster_threshold,
                layout_template_min_cluster_size=args.layout_template_min_cluster_size,
                layout_page_signature_mode=args.layout_page_signature_mode,
                layout_exact_query_value_keys=args.layout_exact_query_value_keys or None,
                layout_template_max_exact_host_pages=args.layout_template_max_exact_host_pages,
                layout_template_large_host_mode=args.layout_template_large_host_mode,
                layout_feature_source=args.layout_template_feature_source,
                worker_count=args.worker_count,
                resources=Resources(cpus=1.0, gpus=0.0),  # CPU-only: no GPU required
            ),
            DripperHTMLLayoutPlanStage(**shared_kwargs, **template_kwargs),
        ]
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
        logger.info("Compacting {} → {} shards", raw_dir, args.output_shards)
        import ray as _ray

        compact_start = time.monotonic()
        _ray.data.read_parquet(str(raw_dir)).repartition(args.output_shards).write_parquet(str(output_dir))
        compact_elapsed = time.monotonic() - compact_start
        logger.info("Compaction done in {:.1f}s", compact_elapsed)
        shutil.rmtree(raw_dir)

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
