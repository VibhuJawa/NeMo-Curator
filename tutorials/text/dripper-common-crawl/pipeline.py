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
"""Dripper Common Crawl streaming pipeline — self-contained Slurm entry point.

Starts vLLM via Ray Serve, runs the full streaming pipeline (WARC fetch → parse →
group → preprocess → cluster → plan → infer × 2 → finalize → postprocess), then
compacts output shards.

Usage (Slurm, called via srun):
  srun python pipeline.py --slurm \\
    --manifest-path /lustre/.../shard_0001.parquet \\
    --output-dir /lustre/.../output

Usage (local smoke test):
  python pipeline.py \\
    --manifest-path /path/to/manifest.parquet \\
    --output-dir /tmp/output \\
    --max-rows 200 --replicas 1
"""  # noqa: RUF002

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.core.serve import InferenceServer
from nemo_curator.core.serve.ray_serve.config import RayServeModelConfig
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.experimental.dripper.pipeline import DripperCommonCrawlPipeline
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import EmptyTask

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: PLR0915
    p = argparse.ArgumentParser(description="Dripper Common Crawl streaming pipeline")

    # --- logging ---
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (DEBUG shows per-batch stage entry, INFO shows setup milestones)",
    )

    # --- I/O ---
    p.add_argument("--manifest-path", help="Parquet manifest with WARC coordinates (full pipeline / Phase 1)")
    p.add_argument(
        "--input-parquet",
        help="Phase 2: precomputed clustering output (html + prompts + layout_id). Runs inference-only.",
    )
    p.add_argument("--output-dir", required=True, help="Output directory for extracted content")
    p.add_argument("--output-shards", type=int, default=24, help="Output shards after compaction")
    p.add_argument("--max-rows", type=int, default=0, help="Limit rows for smoke testing (0 = all)")
    p.add_argument(
        "--use-s3",
        action="store_true",
        default=bool(os.environ.get("CC_USE_S3")),
        help="Fetch WARCs from S3 (default: HTTPS; auto-enabled via CC_USE_S3 env var)",
    )

    # --- Ray ---
    p.add_argument("--slurm", action="store_true", help="Use SlurmRayClient (set when called via srun)")
    p.add_argument("--ray-num-cpus", type=int, default=None)
    p.add_argument("--ray-num-gpus", type=int, default=None)
    p.add_argument("--ray-temp-dir", default="/tmp/ray")  # noqa: S108
    p.add_argument("--ray-port", type=int, default=6379)

    # --- vLLM server ---
    p.add_argument("--model-identifier", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--replicas", type=int, default=8)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--enable-prefix-caching", action="store_true", default=True)
    p.add_argument("--no-enable-prefix-caching", dest="enable_prefix_caching", action="store_false")
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--server-health-check-timeout-s", type=int, default=300)

    # --- generation ---
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-concurrent-requests", type=int, default=64)
    p.add_argument("--disable-thinking", action="store_true", default=True)
    p.add_argument("--no-disable-thinking", dest="disable_thinking", action="store_false")

    # --- pipeline ---
    p.add_argument("--prompt-version", default="short_compact")
    p.add_argument("--output-format", default="mm_md")
    p.add_argument("--fallback", default="trafilatura", choices=["trafilatura", "bypass", "empty"])
    p.add_argument("--min-rows-per-batch", type=int, default=1000)
    p.add_argument("--warc-max-workers", type=int, default=64)

    # --- layout clustering ---
    p.add_argument("--layout-cluster-threshold", type=float, default=0.95)
    p.add_argument("--layout-template-min-cluster-size", type=int, default=2)
    p.add_argument("--layout-template-max-selected-item-ratio", type=float, default=0.50)
    p.add_argument("--layout-template-validation-rows", type=int, default=2)
    p.add_argument("--layout-template-validation-min-content-f1", type=float, default=0.98)
    p.add_argument(
        "--layout-template-validation-aggregation",
        default="min",
        choices=["min", "mean", "median"],
    )
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
    p.add_argument("--layout-template-propagation-concurrency", type=int, default=1)
    p.add_argument("--dynamic-classid-similarity-threshold", type=float, default=0.85)
    p.add_argument("--worker-count", type=int, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def _build_inference_server(args: argparse.Namespace) -> InferenceServer:
    engine_kwargs: dict = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.max_num_seqs:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs

    model_cfg = RayServeModelConfig(
        model_identifier=args.model_identifier,
        deployment_config={"num_replicas": args.replicas},
        engine_kwargs=engine_kwargs,
    )
    return InferenceServer(
        models=[model_cfg],
        port=args.server_port,
        health_check_timeout_s=args.server_health_check_timeout_s,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    started = time.monotonic()

    # ---- Ray init ----------------------------------------------------------
    ray_client_kwargs: dict = {"ray_temp_dir": args.ray_temp_dir}
    if args.ray_num_cpus:
        ray_client_kwargs["num_cpus"] = args.ray_num_cpus
    if args.ray_num_gpus:
        ray_client_kwargs["num_gpus"] = args.ray_num_gpus
    if args.slurm:
        ray_client_kwargs["ray_port"] = args.ray_port

    ray_client = SlurmRayClient(**ray_client_kwargs) if args.slurm else RayClient(**ray_client_kwargs)
    ray_client.start()
    # On Slurm worker nodes (SLURM_NODEID > 0), start() never returns.
    # Only the head node continues below.

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    logger.info("Output dir : {}", output_dir)
    logger.info("Raw dir    : {}", raw_dir)

    # ---- vLLM server -------------------------------------------------------
    server = _build_inference_server(args)
    try:
        server.start()
        api_url = server.endpoint
        logger.info("Inference server ready at {}", api_url)

        # ---- client --------------------------------------------------------
        client = AsyncOpenAIClient(
            max_concurrent_requests=args.max_concurrent_requests,
            base_url=api_url,
            api_key=os.environ.get("INFERENCE_API_KEY", "EMPTY"),
        )

        # disable_thinking was previously parsed but never wired -- the hunyuan model
        # then emitted <think> reasoning on every call (often un-closed, consuming all
        # max_tokens -> empty extraction + garbage template propagation). Pass it via
        # vLLM chat_template_kwargs so the server applies the no-think chat template.
        generation_config = GenerationConfig(
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=0.0,
            extra_kwargs=(
                {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if args.disable_thinking else None
            ),
        )

        # ---- read manifest -------------------------------------------------
        manifest_path = args.manifest_path
        if args.max_rows > 0 and not args.input_parquet:
            import glob

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
            logger.info("Smoke test: trimmed manifest to {} rows → {}", args.max_rows, trimmed_path)

        # ---- pipeline -------------------------------------------------------
        if args.input_parquet:
            # Phase 2: read precomputed clustering output (all columns: html, prompts, layout_id, ...)
            reader = ParquetReader(file_paths=args.input_parquet)
        else:
            reader = ParquetReader(
                file_paths=manifest_path,
                fields=["url_host_name", "url", "warc_filename", "warc_record_offset", "warc_record_length"],
            )
        dripper = DripperCommonCrawlPipeline(
            client=client,
            model_name=args.model_identifier,
            generation_config=generation_config,
            prompt_version=args.prompt_version,
            output_format=args.output_format,
            fallback=args.fallback,
            min_rows_per_batch=args.min_rows_per_batch,
            max_concurrent_requests=args.max_concurrent_requests,
            health_check=False,  # server already verified above
            warc_max_workers=args.warc_max_workers,
            use_s3=args.use_s3,
            layout_cluster_threshold=args.layout_cluster_threshold,
            layout_template_min_cluster_size=args.layout_template_min_cluster_size,
            layout_template_max_selected_item_ratio=args.layout_template_max_selected_item_ratio,
            layout_template_validation_rows=args.layout_template_validation_rows,
            layout_template_validation_min_content_f1=args.layout_template_validation_min_content_f1,
            layout_template_validation_aggregation=args.layout_template_validation_aggregation,
            layout_template_validation_signature_mode=args.layout_template_validation_signature_mode,
            layout_template_large_cluster_validation_rows=args.layout_template_large_cluster_validation_rows,
            layout_template_large_cluster_min_size=args.layout_template_large_cluster_min_size,
            layout_template_representative_candidates=args.layout_template_representative_candidates,
            layout_template_feature_source=args.layout_template_feature_source,
            layout_template_propagation_target=args.layout_template_propagation_target,
            layout_template_propagation_content_source=args.layout_template_propagation_content_source,
            layout_page_signature_mode=args.layout_page_signature_mode,
            layout_exact_query_value_keys=args.layout_exact_query_value_keys or None,
            layout_template_prompt_dedup_fallback_min_fraction=args.layout_template_prompt_dedup_fallback_min_fraction,
            layout_template_min_saved_call_pages=args.layout_template_min_saved_call_pages,
            layout_template_propagation_concurrency=args.layout_template_propagation_concurrency,
            dynamic_classid_similarity_threshold=args.dynamic_classid_similarity_threshold,
            worker_count=args.worker_count,
            host_domain_col="url_host_name",
            inference_only=bool(args.input_parquet),
        )
        writer = ParquetWriter(path=str(raw_dir))

        pipeline = Pipeline(name="dripper-common-crawl")
        pipeline.add_stage(reader)
        pipeline.add_stage(dripper)
        pipeline.add_stage(writer)

        # RayDataExecutor required when Ray Serve (vLLM) is active
        executor = RayDataExecutor()
        pipeline_start = time.monotonic()
        pipeline.run(executor=executor, initial_tasks=[EmptyTask()])
        pipeline_elapsed = time.monotonic() - pipeline_start
        logger.info("Pipeline done in {:.1f}s", pipeline_elapsed)

        # ---- compaction -----------------------------------------------------
        logger.info("Compacting {} → {} shards at {}", raw_dir, args.output_shards, output_dir)
        import ray as _ray

        compact_start = time.monotonic()
        _ray.data.read_parquet(str(raw_dir)).repartition(args.output_shards).write_parquet(str(output_dir))
        compact_elapsed = time.monotonic() - compact_start
        logger.info("Compaction done in {:.1f}s", compact_elapsed)
        shutil.rmtree(raw_dir)

        # ---- metrics --------------------------------------------------------
        total_elapsed = time.monotonic() - started
        metrics = {
            "pipeline_elapsed_s": round(pipeline_elapsed, 2),
            "compaction_elapsed_s": round(compact_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "output_dir": str(output_dir),
            "output_shards": args.output_shards,
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        logger.info("Metrics: {}", metrics_path)
        logger.info("Pipeline metrics: {}", metrics)

    finally:
        try:  # noqa: SIM105
            server.stop()
        except Exception:  # noqa: BLE001, S110
            pass
        try:  # noqa: SIM105
            ray_client.stop()
        except Exception:  # noqa: BLE001, S110
            pass


if __name__ == "__main__":
    main()
