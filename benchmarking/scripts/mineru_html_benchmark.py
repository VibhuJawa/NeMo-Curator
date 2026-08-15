# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run MinerU-HTML on Parquet input or a natively sharded Common Crawl snapshot."""

# ruff: noqa: EM101, EM102

from __future__ import annotations

import argparse
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from loguru import logger
from mineru_snapshot import scan_parquet, verify_native_snapshot
from mineru_text_accuracy import evaluate_text_accuracy
from utils import setup_executor, write_benchmark_results

from nemo_curator.backends.failed_task_markers import failed_task_manifest_exists
from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download.common_crawl import (
    CommonCrawlWARCDownloadAndReadStage,
    CommonCrawlWARCManifestSourceStage,
)
from nemo_curator.stages.text.download.common_crawl.download import CommonCrawlWARCDownloader
from nemo_curator.stages.text.html_extraction import DEFAULT_MODEL, STATUS_FIELD, MinerUHtmlExtractor
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.utils import TaskPerfUtils


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


def build_extractor(args: argparse.Namespace, base_url: str) -> MinerUHtmlExtractor:
    return MinerUHtmlExtractor(
        base_url=base_url,
        served_model_name=args.model if args.server_mode == "managed" else args.served_model_name,
        server_concurrency=args.server_concurrency,
        html_field=args.html_field,
        html_compression=args.html_compression,
        url_field=args.url_field or None,
        text_field=args.text_field,
        boilerplate_text_field=args.boilerplate_text_field,
        llm_output_field=args.llm_output_field,
        model_identifier=args.model,
        cutoff_length=args.cutoff_length,
        max_model_len=args.max_model_len,
        structured_outputs=args.structured_outputs,
        output_format=args.output_format,
        fallback=args.fallback,
        simplify_workers=args.simplify_workers,
        inference_workers=args.inference_workers,
        extract_workers=args.extract_workers,
        chat_template_mode=args.chat_template_mode,
        cache_dir=args.cache_dir,
        drop_html_field=args.drop_html_field,
    )


def build_parquet_pipeline(args: argparse.Namespace, output_dir: Path, base_url: str) -> Pipeline:
    fields = None if args.preserve_input_fields else [args.html_field] + ([args.url_field] if args.url_field else [])
    pipeline = Pipeline(name="mineru_html", description="Extract main content from HTML")
    pipeline.add_stage(
        ParquetReader(
            file_paths=[str(Path(args.input_path).resolve())],
            files_per_partition=args.files_per_partition,
            fields=fields,
            read_kwargs={"dtype_backend": "numpy_nullable"},
        )
    )
    pipeline.add_stage(build_extractor(args, base_url))
    pipeline.add_stage(ParquetWriter(path=str(output_dir), mode="error"))
    return pipeline


def build_snapshot_pipeline(args: argparse.Namespace, output_dir: Path, base_url: str) -> Pipeline:
    key_prefix = "" if args.cc_s3_key_prefix == "-" else args.cc_s3_key_prefix
    endpoint = None if args.cc_s3_endpoint_url == "-" else args.cc_s3_endpoint_url
    logger.info(
        "Common Crawl transport={} endpoint={} bucket={} strip_prefix={} downloads={} multipart={}",
        args.cc_transport,
        endpoint,
        args.cc_s3_bucket,
        key_prefix,
        args.download_workers,
        f"{args.cc_s5cmd_concurrency}x{args.cc_s5cmd_part_size_mb}MiB",
    )
    downloader = CommonCrawlWARCDownloader(
        download_dir=args.download_dir,
        use_aws_to_download=args.cc_transport == "s3",
        s3_bucket=args.cc_s3_bucket,
        s3_key_prefix=key_prefix,
        s3_endpoint_url=endpoint,
        s5cmd_concurrency=args.cc_s5cmd_concurrency,
        s5cmd_part_size_mb=args.cc_s5cmd_part_size_mb,
    )
    pipeline = Pipeline(name="mineru_common_crawl", description=f"MinerU extraction for CC-MAIN-{args.snapshot}")
    pipeline.add_stage(CommonCrawlWARCManifestSourceStage(args.warc_manifest))
    pipeline.add_stage(
        CommonCrawlWARCDownloadAndReadStage(
            downloader,
            content_field=args.html_field,
            compression=args.html_compression,
            workers_per_node=args.download_workers,
            records_per_batch=args.warc_records_per_batch,
        )
    )
    pipeline.add_stage(build_extractor(args, base_url))
    # Every WARC chunk maps to one deterministic Parquet file. Array tasks share
    # this directory; retries overwrite only files derived from the same source
    # WARC and chunk index.
    pipeline.add_stage(ParquetWriter(path=str(output_dir), mode="ignore"))
    return pipeline


def output_metrics(paths: list[Path], args: argparse.Namespace) -> dict:
    stats = scan_parquet(
        paths,
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


def _written_paths(tasks: list) -> list[Path]:
    return [Path(path) for task in tasks if isinstance(task, FileGroupTask) for path in task.data]


def vllm_performance_metrics(tasks: list) -> dict[str, float | int]:
    """Return aggregate serving rate over the real cross-worker request window."""
    stage = TaskPerfUtils.collect_stage_metrics(tasks).get("mineru_html_server_inference", {})
    requests = stage.get("custom.requests")
    starts = stage.get("custom.request_window_start_s")
    ends = stage.get("custom.request_window_end_s")
    if requests is None or starts is None or ends is None or not len(requests):
        return {
            "vllm_requests": 0,
            "vllm_inference_time_s": 0.0,
            "vllm_docs_per_sec": 0.0,
        }

    request_count = int(requests.sum())
    inference_time_s = float(ends.max() - starts.min())
    return {
        "vllm_requests": request_count,
        "vllm_inference_time_s": inference_time_s,
        "vllm_docs_per_sec": request_count / inference_time_s if inference_time_s else 0.0,
    }


def _validate_runtime(args: argparse.Namespace) -> None:
    if not args.reuse_driver_environment:
        return
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


def run_benchmark(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_path).resolve()
    if args.server_mode == "external" and not args.server_url:
        raise ValueError("--server-url is required with --server-mode=external")
    _validate_runtime(args)
    executor = setup_executor(args.executor)
    server = build_server(args) if args.server_mode == "managed" else None
    results = []
    server_startup_s = pipeline_elapsed_s = 0.0
    success = False
    try:
        if server:
            started = time.perf_counter()
            server.start()
            server_startup_s = time.perf_counter() - started
            base_url = server.endpoint.removesuffix("/v1")
            logger.info(f"Dynamo is healthy after {server_startup_s:.1f}s at {server.endpoint}")
        else:
            base_url = args.server_url
        pipeline = (
            build_snapshot_pipeline(args, output_dir, base_url)
            if args.warc_manifest
            else build_parquet_pipeline(args, output_dir, base_url)
        )
        started = time.perf_counter()
        results = pipeline.run(
            executor,
            initial_tasks=None,
            checkpoint_path=args.checkpoint_path if args.warc_manifest else None,
        )
        pipeline_elapsed_s = time.perf_counter() - started
        success = not failed_task_manifest_exists()
    except Exception:
        logger.exception("MinerU pipeline failed")
    finally:
        if server:
            server.stop()

    metrics: dict = {
        "is_success": success,
        "verification_passed": success,
        "server_startup_s": server_startup_s,
        "time_taken_s": pipeline_elapsed_s,
        "output_path": str(output_dir),
    }
    try:
        paths = _written_paths(results or [])
        if not args.warc_manifest:
            paths = sorted(output_dir.rglob("*.parquet"))
        if paths:
            metrics.update(output_metrics(paths, args))
        else:
            metrics.update({"num_output_files": 0, "num_documents_processed": 0, "num_documents_written": 0})
        metrics.update(vllm_performance_metrics(results or []))
        if success and args.accuracy_reference_path:
            metrics.update(
                evaluate_text_accuracy(
                    output_path=output_dir,
                    reference_path=Path(args.accuracy_reference_path),
                    url_field=args.url_field,
                    text_field=args.text_field,
                )
            )
    except Exception as e:
        logger.exception("MinerU output validation failed")
        success = False
        metrics.update({"verification_passed": False, "verification_errors": [str(e)]})
    metrics["is_success"] = success
    rows = int(metrics.get("num_documents_processed", 0))
    metrics["throughput_docs_per_sec"] = rows / pipeline_elapsed_s if pipeline_elapsed_s else 0.0
    params = vars(args) | {
        "runtime_versions": {
            name: package_version(name) for name in ("nemo-curator", "vllm", "ai-dynamo", "ray", "torch")
        }
    }
    return {"params": params, "metrics": metrics, "tasks": results or []}


def parse_args() -> argparse.Namespace:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-path")
    source.add_argument("--warc-manifest")
    parser.add_argument("--snapshot", help="Main crawl snapshot in YYYY-WW form")
    parser.add_argument("--verify-snapshot", action="store_true")
    parser.add_argument("--output-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--snapshot-success-path")
    parser.add_argument("--download-dir")
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--warc-records-per-batch", type=int, default=1024)
    parser.add_argument("--quality-sample-files", type=int, default=1024)
    parser.add_argument("--cc-transport", default="s3", choices=["https", "s3"])
    parser.add_argument("--cc-s3-bucket", default="crawl-data")
    parser.add_argument("--cc-s3-key-prefix", default="crawl-data/")
    parser.add_argument("--cc-s3-endpoint-url", default="https://pdx.s8k.io")
    parser.add_argument("--cc-s5cmd-concurrency", type=int, default=8)
    parser.add_argument("--cc-s5cmd-part-size-mb", type=int, default=256)
    parser.add_argument("--html-field", default="content")
    parser.add_argument("--html-compression", default="none", choices=["none", "zstd"])
    parser.add_argument("--drop-html-field", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--url-field", default="url")
    parser.add_argument("--preserve-input-fields", action="store_true")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--boilerplate-text-field")
    parser.add_argument("--llm-output-field")
    parser.add_argument("--accuracy-reference-path")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cutoff-length", type=int, default=500)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--structured-outputs", default="none", choices=["none", "per_request"])
    parser.add_argument("--output-format", default="mm_md", choices=["mm_md", "md", "json", "txt", "none"])
    parser.add_argument("--fallback", default="trafilatura", choices=["trafilatura", "bypass", "empty"])
    parser.add_argument("--chat-template-mode", default="single", choices=["single", "upstream_double"])
    parser.add_argument("--simplify-workers", type=int, default=32)
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
        required = (args.warc_manifest, args.snapshot_success_path, args.output_path, args.checkpoint_path)
    elif args.warc_manifest:
        required = (args.snapshot, args.output_path, args.checkpoint_path, args.download_dir)
        if args.html_compression != "zstd":
            parser.error("snapshot mode requires --html-compression=zstd")
    else:
        required = (args.input_path, args.output_path)
    if not all(required):
        parser.error("the selected mode is missing required path arguments")
    return args


def main() -> int:
    os.umask(0o022)
    args = parse_args()
    if args.verify_snapshot:
        expected_num_warcs = sum(1 for line in Path(args.warc_manifest).read_text().splitlines() if line.strip())
        metrics = verify_native_snapshot(
            output_path=Path(args.output_path),
            checkpoint_path=Path(args.checkpoint_path),
            success_path=Path(args.snapshot_success_path),
            expected_num_warcs=expected_num_warcs,
            url_field=args.url_field,
            text_field=args.text_field,
            status_field=STATUS_FIELD,
            min_status_ok_rate=args.min_status_ok_rate,
            min_nonempty_rate=args.min_nonempty_rate,
            max_convert_error_rate=args.max_convert_error_rate,
            quality_sample_files=args.quality_sample_files,
        )
        results = {"params": vars(args), "metrics": metrics, "tasks": []}
    else:
        results = run_benchmark(args)
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"].get("verification_passed", results["metrics"].get("is_success")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
