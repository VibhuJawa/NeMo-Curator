# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run MinerU-HTML with an external endpoint or a Curator-managed Dynamo server."""

# ruff: noqa: EM101, EM102

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from loguru import logger
from mineru_snapshot import (
    new_attempt_directory,
    preflight_input,
    publish_attempt,
    read_published_result,
    scan_parquet,
    select_work_unit,
    validate_output,
    verify_snapshot,
)
from utils import setup_executor, write_benchmark_results

from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.html_extraction import DEFAULT_MODEL, STATUS_FIELD, MinerUHtmlExtractor
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def build_server(args: argparse.Namespace) -> InferenceServer:
    engine_kwargs: dict = {
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "generation_config": "vllm",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": args.enable_prefix_caching,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
    }
    if args.kv_cache_dtype != "auto":
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.cache_dir:
        engine_kwargs["download_dir"] = args.cache_dir
    if args.cudagraph_mode:
        engine_kwargs["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    packages = []
    if args.speculative_tokens:
        packages.append(args.arctic_inference_spec)
        engine_kwargs["speculative_config"] = {
            "method": "suffix",
            "num_speculative_tokens": args.speculative_tokens,
            "suffix_decoding_max_spec_factor": 2.0,
            "suffix_decoding_max_cached_requests": args.suffix_max_cached_requests,
        }
    runtime_env: dict = {
        "env_vars": {
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "USE_TORCH": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    }
    if packages and not args.reuse_driver_environment:
        runtime_env["uv"] = {"packages": packages}
    return InferenceServer(
        models=[
            DynamoVLLMModelConfig(
                model_identifier=args.model,
                num_replicas=args.num_replicas,
                engine_kwargs=engine_kwargs,
                runtime_env=runtime_env,
                reuse_driver_environment=args.reuse_driver_environment,
            )
        ],
        backend=DynamoServerConfig(),
        port=args.server_port,
        health_check_timeout_s=args.server_timeout_s,
    )


def build_pipeline(args: argparse.Namespace, input_paths: list[str], output_dir: Path, base_url: str) -> Pipeline:
    reader = ParquetReader(
        file_paths=input_paths,
        files_per_partition=args.files_per_partition,
        fields=[args.html_field, args.url_field] if args.url_field else [args.html_field],
        read_kwargs={"dtype_backend": "numpy_nullable"},
    )
    extractor = MinerUHtmlExtractor(
        base_url=base_url,
        served_model_name=args.model if args.server_mode == "managed" else args.served_model_name,
        server_concurrency=args.server_concurrency,
        html_field=args.html_field,
        html_compression=args.html_compression,
        url_field=args.url_field or None,
        text_field=args.text_field,
        model_identifier=args.model,
        max_model_len=args.max_model_len,
        structured_outputs=args.structured_outputs,
        output_format=args.output_format,
        fallback=args.fallback,
        simplify_workers=args.simplify_workers,
        inference_workers=args.inference_workers,
        extract_workers=args.extract_workers,
        chat_template_mode=args.chat_template_mode,
        cache_dir=args.cache_dir,
    )
    pipeline = Pipeline(name="mineru_html", description="Extract main content from Common Crawl HTML")
    pipeline.add_stage(reader)
    pipeline.add_stage(extractor)
    pipeline.add_stage(ParquetWriter(path=str(output_dir), mode="error"))
    return pipeline


def output_metrics(output_dir: Path, args: argparse.Namespace) -> dict:
    stats = scan_parquet(
        sorted(output_dir.rglob("*.parquet")),
        url_field=args.url_field,
        text_field=args.text_field,
        status_field=STATUS_FIELD,
        required_fields={args.url_field, args.text_field, STATUS_FIELD},
    )
    rows = stats["num_rows"]
    counts = stats["status_counts"]
    return {
        **stats,
        "num_output_files": stats["num_files"],
        "num_documents_processed": rows,
        "num_documents_written": rows,
        "status_ok_rate": counts.get("ok", 0) / rows if rows else 0.0,
        "nonempty_rate": stats["num_documents_with_text"] / rows if rows else 0.0,
        "extraction_rate": stats["num_documents_substantive"] / rows if rows else 0.0,
        "convert_error_rate": counts.get("convert_error", 0) / rows if rows else 0.0,
    }


def run_benchmark(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0912, PLR0915
    manifest_sha256 = unit = input_stats = None
    if args.work_unit_manifest:
        unit, manifest_sha256 = select_work_unit(args.work_unit_manifest, args.work_unit_index)
        published = read_published_result(unit, manifest_sha256)
        if published:
            metrics = published["validation"] | {
                "is_success": True,
                "skipped_published_work_unit": True,
                "published": True,
                "num_documents_processed": published["validation"]["num_rows"],
            }
            return {"params": vars(args) | {"work_unit": asdict(unit)}, "metrics": metrics, "tasks": []}
        input_paths = list(unit.input_paths)
        final_output = Path(unit.output_path)
        input_stats = preflight_input(unit, html_field=args.html_field, url_field=args.url_field)
        output_dir = new_attempt_directory(unit)
    else:
        input_paths = [str(Path(args.input_path).resolve())]
        final_output = Path(args.output_path).resolve()
        output_dir = final_output

    if args.server_mode == "external" and not args.server_url:
        raise ValueError("--server-url is required with --server-mode=external")
    executor = setup_executor(args.executor)
    results = []
    server_startup_s = 0.0
    pipeline_elapsed_s = 0.0
    success = False
    if args.reuse_driver_environment:
        expected = {"vllm": "0.26.", "ai-dynamo": "1.4."}
        mismatched = {
            name: package_version(name)
            for name, prefix in expected.items()
            if not (package_version(name) or "").startswith(prefix)
        }
        if mismatched:
            raise RuntimeError(f"managed baseline requires {expected}; installed versions: {mismatched}")
        if args.speculative_tokens and package_version("arctic-inference") is None:
            raise RuntimeError("suffix decoding requires arctic-inference in the driver environment")
    server = build_server(args) if args.server_mode == "managed" else None
    try:
        if server:
            started = time.perf_counter()
            server.start()
            server_startup_s = time.perf_counter() - started
            base_url = server.endpoint.removesuffix("/v1")
            logger.info(f"Dynamo is healthy after {server_startup_s:.1f}s at {server.endpoint}")
        else:
            base_url = args.server_url
        pipeline = build_pipeline(args, input_paths, output_dir, base_url)
        started = time.perf_counter()
        results = pipeline.run(executor, initial_tasks=None)
        pipeline_elapsed_s = time.perf_counter() - started
        success = True
    except Exception:
        logger.exception("MinerU work unit failed")
    finally:
        if server:
            server.stop()

    metrics: dict = {
        "is_success": success,
        "server_startup_s": server_startup_s,
        "time_taken_s": pipeline_elapsed_s,
        "attempt_output_path": str(output_dir),
        "final_output_path": str(final_output),
    }
    try:
        if success and unit:
            validation = validate_output(
                unit,
                output_dir,
                input_stats,
                url_field=args.url_field,
                text_field=args.text_field,
                status_field=STATUS_FIELD,
                min_status_ok_rate=args.min_status_ok_rate,
                min_nonempty_rate=args.min_nonempty_rate,
                max_convert_error_rate=args.max_convert_error_rate,
            )
            success = validation["verification_passed"]
            metrics.update(validation)
            if success:
                publish_attempt(unit, output_dir, manifest_sha256, validation)
                metrics["published"] = True
        elif success:
            metrics.update(output_metrics(output_dir, args))
    except Exception as e:  # validation failures must still leave inspectable metrics
        logger.exception("MinerU output validation failed")
        success = False
        metrics.update({"verification_passed": False, "verification_errors": [str(e)]})
    metrics["is_success"] = success
    rows = int(metrics.get("num_rows", metrics.get("num_documents_processed", 0)))
    metrics["num_documents_processed"] = rows
    metrics["throughput_docs_per_sec"] = rows / pipeline_elapsed_s if pipeline_elapsed_s else 0.0
    params = vars(args) | {
        "input_paths": input_paths,
        "output_path": str(final_output),
        "runtime_versions": {
            name: package_version(name) for name in ("nemo-curator", "vllm", "ai-dynamo", "ray", "torch")
        },
    }
    if unit:
        params["work_unit"] = asdict(unit)
        params["manifest_sha256"] = manifest_sha256
        params["input_stats"] = input_stats
    return {"params": params, "metrics": metrics, "tasks": results or []}


def parse_args() -> argparse.Namespace:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--work-unit-manifest")
    mode.add_argument("--verify-snapshot", metavar="MANIFEST")
    parser.add_argument("--work-unit-index", type=int, default=0)
    parser.add_argument("--snapshot-success-path")
    parser.add_argument("--input-path")
    parser.add_argument("--output-path")
    parser.add_argument("--html-field", default="content")
    parser.add_argument("--html-compression", default="none", choices=["none", "zstd"])
    parser.add_argument("--url-field", default="url")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--structured-outputs", default="none", choices=["none", "per_request"])
    parser.add_argument("--output-format", default="mm_md", choices=["mm_md", "md", "json", "txt", "none"])
    parser.add_argument("--fallback", default="trafilatura", choices=["trafilatura", "bypass", "empty"])
    parser.add_argument("--chat-template-mode", default="single", choices=["single", "upstream_double"])
    parser.add_argument("--simplify-workers", type=int, default=8)
    parser.add_argument("--inference-workers", type=int, default=32)
    parser.add_argument("--extract-workers", type=int, default=24)
    parser.add_argument("--server-concurrency", type=int, default=256)
    parser.add_argument("--cache-dir")
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data", "ray_actors"])
    parser.add_argument("--server-mode", default="managed", choices=["managed", "external"])
    parser.add_argument("--server-url")
    parser.add_argument("--served-model-name", default="mineru")
    parser.add_argument("--reuse-driver-environment", action="store_true")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-timeout-s", type=int, default=2400)
    parser.add_argument("--num-replicas", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--kv-cache-dtype", default="fp8", choices=["auto", "fp8"])
    parser.add_argument("--speculative-tokens", type=int, default=16)
    parser.add_argument("--suffix-max-cached-requests", type=int, default=10000)
    parser.add_argument("--arctic-inference-spec", default="arctic-inference")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudagraph-mode", choices=["NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE"])
    parser.add_argument("--min-status-ok-rate", type=float, default=0.95)
    parser.add_argument("--min-nonempty-rate", type=float, default=0.95)
    parser.add_argument("--max-convert-error-rate", type=float, default=0.02)
    args = parser.parse_args()
    if args.verify_snapshot:
        if not args.snapshot_success_path:
            parser.error("--verify-snapshot requires --snapshot-success-path")
    elif not args.work_unit_manifest and not (args.input_path and args.output_path):
        parser.error("provide --work-unit-manifest or both --input-path and --output-path")
    return args


def main() -> int:
    os.umask(0o022)
    args = parse_args()
    if args.verify_snapshot:
        metrics = verify_snapshot(args.verify_snapshot, args.snapshot_success_path)
        results = {"params": vars(args), "metrics": metrics, "tasks": []}
    else:
        results = run_benchmark(args)
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"].get("verification_passed", results["metrics"].get("is_success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
