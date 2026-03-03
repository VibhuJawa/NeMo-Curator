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

"""Benchmark: read parquet (interleaved or OBELICS), optional filter, write 3 formats via Ray."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from loguru import logger
from utils import (
    collect_lance_output_metrics,
    collect_parquet_output_metrics,
    collect_webdataset_output_metrics,
    setup_executor,
    write_benchmark_results,
)

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import (
    InterleavedLanceWriterStage,
    InterleavedParquetReader,
    InterleavedParquetWriterStage,
    InterleavedWebdatasetWriterStage,
)
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.utils import TaskPerfUtils


@dataclass
class _MatchedSampleFilterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Keep only samples where every image row has match_status == 'matched'.

    Also drops samples that have zero image rows (text/metadata-only documents).
    """

    name: str = "matched_sample_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        df = task.to_pandas()
        image_df = df[df["modality"] == "image"]

        sids_with_images = set(image_df["sample_id"].unique())
        unmatched_sids = set(
            image_df.loc[image_df["match_status"] != "matched", "sample_id"].unique()
        )
        keep_sids = sids_with_images - unmatched_sids

        all_sids = df["sample_id"].nunique()
        logger.info(
            "MatchedFilter: {} total samples, {} with images, {} fully matched",
            all_sids, len(sids_with_images), len(keep_sids),
        )

        if not keep_sids:
            return None

        filtered = df[df["sample_id"].isin(keep_sids)].reset_index(drop=True)
        table = pa.Table.from_pandas(filtered, preserve_index=False)
        return InterleavedBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class ObelicsReaderStage(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Read raw OBELICS parquet and convert to InterleavedBatch on the fly.

    OBELICS schema: texts (list[str|None]), images (list[str|None]), general_metadata (str).
    Converts to interleaved rows: sample_id, position, modality, content_type,
    text_content, source_ref, binary_content, metadata_json, materialize_error.
    """

    name: str = "obelics_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @staticmethod
    def _to_list(val: object) -> list:
        if val is None:
            return []
        try:
            return list(val)
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _extract_doc_url(general_metadata: object) -> str:
        if not general_metadata or (isinstance(general_metadata, float) and pd.isna(general_metadata)):
            return ""
        try:
            gm = json.loads(general_metadata) if isinstance(general_metadata, str) else general_metadata
            return gm.get("url", "") if isinstance(gm, dict) else ""
        except (json.JSONDecodeError, TypeError, AttributeError):
            return ""

    def _convert_document(self, record: pd.Series, idx: int, file_path: str, rows: list[dict[str, Any]]) -> None:
        texts = self._to_list(record.get("texts"))
        images = self._to_list(record.get("images"))
        general_metadata = record.get("general_metadata", "")

        doc_url = self._extract_doc_url(general_metadata)
        sample_id = hashlib.sha256(f"{file_path}_{idx}_{doc_url}".encode()).hexdigest()[:16]

        max_len = max(len(texts), len(images))
        position = 0
        for i in range(max_len):
            text_val = texts[i] if i < len(texts) else None
            image_val = images[i] if i < len(images) else None

            if text_val is not None and not (isinstance(text_val, float) and pd.isna(text_val)):
                rows.append({
                    "sample_id": sample_id,
                    "position": position,
                    "modality": "text",
                    "content_type": "text/plain",
                    "text_content": str(text_val),
                    "binary_content": None,
                    "source_ref": None,
                    "metadata_json": None,
                    "materialize_error": None,
                })
                position += 1
            elif image_val is not None and not (isinstance(image_val, float) and pd.isna(image_val)):
                source_ref = json.dumps({
                    "path": str(image_val), "member": None, "byte_offset": None, "byte_size": None,
                })
                rows.append({
                    "sample_id": sample_id,
                    "position": position,
                    "modality": "image",
                    "content_type": "image/jpeg",
                    "text_content": None,
                    "binary_content": None,
                    "source_ref": source_ref,
                    "metadata_json": None,
                    "materialize_error": None,
                })
                position += 1

        gm_str = str(general_metadata) if general_metadata and not (isinstance(general_metadata, float) and pd.isna(general_metadata)) else ""
        if gm_str:
            rows.append({
                "sample_id": sample_id,
                "position": -1,
                "modality": "metadata",
                "content_type": None,
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "metadata_json": gm_str,
                "materialize_error": None,
            })

    def process(self, task: FileGroupTask) -> InterleavedBatch | list[InterleavedBatch]:
        rows: list[dict[str, Any]] = []
        for file_path in task.data:
            df = pd.read_parquet(file_path)
            for idx, record in df.iterrows():
                self._convert_document(record, idx, file_path, rows)

        if not rows:
            msg = f"No data read from OBELICS parquet files in task {task.task_id}"
            raise ValueError(msg)

        from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

        result_df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(result_df, preserve_index=False)
        table = table.cast(INTERLEAVED_SCHEMA)

        return InterleavedBatch(
            task_id=f"{task.task_id}_obelics",
            dataset_name=task.dataset_name,
            data=table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _build_storage_options(args: argparse.Namespace) -> dict[str, Any]:
    if not args.s3_access_key:
        return {}
    opts: dict[str, Any] = {
        "key": args.s3_access_key,
        "secret": args.s3_secret_key,
    }
    client_kwargs: dict[str, str] = {}
    if args.s3_endpoint:
        client_kwargs["endpoint_url"] = args.s3_endpoint
    if args.s3_region:
        client_kwargs["region_name"] = args.s3_region
    if client_kwargs:
        opts["client_kwargs"] = client_kwargs
    return opts


def _resolve_input_paths(args: argparse.Namespace) -> list[str]:
    if args.input_path:
        p = Path(args.input_path).absolute()
        if not p.exists():
            msg = f"Input path does not exist: {p}"
            raise FileNotFoundError(msg)
        return [str(p)]

    paths: list[str] = []
    for i in range(args.num_buckets):
        bucket_dir = Path(args.input_base_path) / f"domain_bucket={i}"
        if not bucket_dir.is_dir():
            logger.warning("Bucket directory does not exist, skipping: {}", bucket_dir)
            continue
        paths.append(str(bucket_dir))
    if not paths:
        msg = f"No bucket directories found under {args.input_base_path} for num_buckets={args.num_buckets}"
        raise FileNotFoundError(msg)
    logger.info("Resolved {} bucket path(s): {}", len(paths), paths)
    return paths


def _create_writer(
    fmt: str, args: argparse.Namespace, storage_options: dict[str, Any],
) -> ProcessingStage:
    write_kwargs: dict[str, Any] = {}
    if storage_options:
        write_kwargs["storage_options"] = storage_options
    path = str(Path(args.output_path) / fmt)

    common = {
        "path": path,
        "materialize_on_write": args.materialize_on_write,
        "write_kwargs": write_kwargs,
        "on_materialize_error": args.on_materialize_error,
        "mode": args.mode,
    }

    writer_cls = {
        "parquet": InterleavedParquetWriterStage,
        "webdataset": InterleavedWebdatasetWriterStage,
        "lance": InterleavedLanceWriterStage,
    }.get(fmt)

    if writer_cls is None:
        msg = f"Unknown format: {fmt}"
        raise ValueError(msg)

    return writer_cls(**common)


_METRICS_COLLECTORS = {
    "parquet": collect_parquet_output_metrics,
    "webdataset": collect_webdataset_output_metrics,
    "lance": collect_lance_output_metrics,
}


def run_format_benchmark(
    fmt: str,
    args: argparse.Namespace,
    input_paths: list[str],
    storage_options: dict[str, Any],
) -> dict[str, Any]:
    executor = setup_executor(args.executor)
    output_sub = Path(args.output_path) / fmt
    output_sub.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        name=f"parquet_to_{fmt}_benchmark",
        description=f"Benchmark: parquet -> {fmt}",
    )

    if args.reader_type == "obelics":
        from nemo_curator.stages.file_partitioning import FilePartitioningStage

        pipeline.add_stage(
            FilePartitioningStage(
                file_paths=input_paths,
                files_per_partition=args.files_per_partition,
                file_extensions=[".parquet"],
            )
        )
        pipeline.add_stage(ObelicsReaderStage())
    else:
        pipeline.add_stage(
            InterleavedParquetReader(
                file_paths=input_paths,
                files_per_partition=args.files_per_partition,
                max_batch_bytes=args.output_max_batch_bytes,
                read_kwargs={},
            )
        )

    if args.use_filter:
        pipeline.add_stage(_MatchedSampleFilterStage())

    pipeline.add_stage(_create_writer(fmt, args, storage_options))

    logger.info("=== Running {} pipeline ===", fmt)
    logger.info("Pipeline:\n{}", pipeline.describe())

    start = time.perf_counter()
    output_tasks: list = []
    success = False
    try:
        output_tasks = pipeline.run(executor)
        success = True
    except Exception as e:
        logger.error("{} pipeline failed: {}", fmt, e)
        logger.debug(traceback.format_exc())

    elapsed = time.perf_counter() - start
    logger.info("{} pipeline finished in {:.1f}s (success={})", fmt, elapsed, success)

    collector = _METRICS_COLLECTORS.get(fmt)
    output_metrics: dict[str, Any] = {}
    if collector and success:
        try:
            output_metrics = collector(output_sub)
        except Exception as e:
            logger.warning("Failed to collect {} output metrics: {}", fmt, e)

    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    return {
        "format": fmt,
        "is_success": success,
        "time_taken_s": elapsed,
        **task_metrics,
        **{f"{fmt}_{k}": v for k, v in output_metrics.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal parquet -> multi-format benchmark (Ray)")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--input-path", type=str, default=None, help="Flat parquet directory (alternative to --input-base-path)")
    parser.add_argument("--input-base-path", type=str, default=None, help="Base path for domain_bucket=N subdirs")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-buckets", type=int, default=1)
    parser.add_argument("--formats", type=str, default="parquet,webdataset,lance")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument("--materialize-on-write", action="store_true", dest="materialize_on_write")
    parser.add_argument("--no-materialize-on-write", action="store_false", dest="materialize_on_write")
    parser.add_argument(
        "--on-materialize-error", type=str, default="warn",
        choices=["error", "warn", "drop_row", "drop_sample"],
    )
    parser.add_argument("--mode", type=str, default="overwrite", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--reader-type", type=str, default="interleaved", choices=["interleaved", "obelics"])
    parser.add_argument("--use-filter", action="store_true", dest="use_filter")
    parser.add_argument("--no-filter", action="store_false", dest="use_filter")
    parser.add_argument("--s3-access-key", type=str, default=None)
    parser.add_argument("--s3-secret-key", type=str, default=None)
    parser.add_argument("--s3-endpoint", type=str, default=None)
    parser.add_argument("--s3-region", type=str, default=None)
    parser.set_defaults(materialize_on_write=True, use_filter=False)
    args = parser.parse_args()

    if not args.input_path and not args.input_base_path:
        logger.error("Either --input-path or --input-base-path must be provided")
        return 1

    formats = [f.strip() for f in args.formats.split(",")]
    valid_formats = {"parquet", "webdataset", "lance"}
    for fmt in formats:
        if fmt not in valid_formats:
            logger.error("Invalid format '{}'. Valid: {}", fmt, valid_formats)
            return 1

    input_paths = _resolve_input_paths(args)
    storage_options = _build_storage_options(args)

    ray_client = RayClient()
    ray_client.start()

    all_metrics: dict[str, Any] = {
        "formats_requested": formats,
        "input_paths": input_paths,
        "reader_type": args.reader_type,
    }
    all_success = True

    try:
        for fmt in formats:
            fmt_result = run_format_benchmark(fmt, args, input_paths, storage_options)
            all_metrics[fmt] = fmt_result
            if not fmt_result["is_success"]:
                all_success = False
    except Exception as e:
        logger.error("Benchmark crashed: {}", e)
        logger.debug(traceback.format_exc())
        all_success = False
    finally:
        results = {
            "params": {
                "executor": args.executor,
                "input_path": args.input_path,
                "input_base_path": args.input_base_path,
                "output_path": args.output_path,
                "formats": formats,
                "reader_type": args.reader_type,
                "use_filter": args.use_filter,
                "files_per_partition": args.files_per_partition,
                "output_max_batch_bytes": args.output_max_batch_bytes,
                "materialize_on_write": args.materialize_on_write,
                "on_materialize_error": args.on_materialize_error,
                "mode": args.mode,
            },
            "metrics": {
                "is_success": all_success,
                **all_metrics,
            },
            "tasks": [],
        }
        write_benchmark_results(results, args.benchmark_results_path)
        ray_client.stop()

    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
