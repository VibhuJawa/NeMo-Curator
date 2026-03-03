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

"""Symmetric benchmark: Reader -> [optional Filter] -> Writer for all 3 output formats.

Supports WDS and Parquet readers, AspectRatioFilter, and Parquet/WebDataset/Lance writers.
Each format runs as a separate pipeline execution for independent timing.
Does NOT manage Ray -- the benchmarking framework (run.py) handles that.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from utils import (
    collect_lance_output_metrics,
    collect_parquet_output_metrics,
    collect_webdataset_output_metrics,
    setup_executor,
    write_benchmark_results,
)

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import (
    InterleavedLanceWriterStage,
    InterleavedParquetReader,
    InterleavedParquetWriterStage,
    InterleavedWebdatasetWriterStage,
    WebdatasetReader,
)
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage
from nemo_curator.tasks.utils import TaskPerfUtils

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage

_WRITER_CLASSES: dict[str, type] = {
    "parquet": InterleavedParquetWriterStage,
    "webdataset": InterleavedWebdatasetWriterStage,
    "lance": InterleavedLanceWriterStage,
}

_METRICS_COLLECTORS = {
    "parquet": collect_parquet_output_metrics,
    "webdataset": collect_webdataset_output_metrics,
    "lance": collect_lance_output_metrics,
}


def _create_writer(fmt: str, output_base: str, args: argparse.Namespace) -> ProcessingStage:
    path = str(Path(output_base) / fmt)
    writer_cls = _WRITER_CLASSES.get(fmt)
    if writer_cls is None:
        msg = f"Unknown format: {fmt}"
        raise ValueError(msg)
    return writer_cls(
        path=path,
        materialize_on_write=args.materialize_on_write,
        on_materialize_error=args.on_materialize_error,
        mode=args.mode,
    )


def _build_pipeline(fmt: str, args: argparse.Namespace) -> Pipeline:
    reader_label = args.reader_type
    filter_label = "filter" if args.use_filter else "nofilter"
    pipeline = Pipeline(
        name=f"symmetric_{reader_label}_{filter_label}_{fmt}",
        description=f"Symmetric benchmark: {reader_label} -> {filter_label} -> {fmt}",
    )

    if args.reader_type == "wds":
        pipeline.add_stage(
            WebdatasetReader(
                source_id_field=args.source_id_field,
                file_paths=args.input_path,
                files_per_partition=args.files_per_partition,
                materialize_on_read=args.materialize_on_read,
            )
        )
    else:
        pipeline.add_stage(
            InterleavedParquetReader(
                file_paths=args.input_path,
                files_per_partition=args.files_per_partition,
            )
        )

    if args.use_filter:
        pipeline.add_stage(InterleavedAspectRatioFilterStage(
            drop_invalid_rows=True, min_aspect_ratio=1.0, max_aspect_ratio=2.0,
        ))

    pipeline.add_stage(_create_writer(fmt, args.output_path, args))
    return pipeline


def run_format(fmt: str, args: argparse.Namespace) -> dict[str, Any]:
    executor = setup_executor(args.executor)
    output_sub = Path(args.output_path) / fmt
    output_sub.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(fmt, args)
    logger.info("=== Running {} pipeline ===", fmt)
    logger.info("Pipeline:\n{}", pipeline.describe())

    start = time.perf_counter()
    output_tasks: list = []
    success = False
    try:
        output_tasks = pipeline.run(executor)
        success = True
    except Exception as exc:
        logger.error("{} pipeline failed: {}", fmt, exc)
        logger.debug(traceback.format_exc())

    elapsed = time.perf_counter() - start
    logger.info("{} pipeline finished in {:.1f}s (success={})", fmt, elapsed, success)

    output_metrics: dict[str, Any] = {}
    collector = _METRICS_COLLECTORS.get(fmt)
    if collector and success:
        try:
            output_metrics = collector(output_sub)
        except Exception as exc:
            logger.warning("Failed to collect {} output metrics: {}", fmt, exc)

    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    return {
        "format": fmt,
        "is_success": success,
        "time_taken_s": elapsed,
        **task_metrics,
        **{f"{fmt}_{k}": v for k, v in output_metrics.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Symmetric interleaved benchmark: Reader -> [Filter] -> Writer x3")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--reader-type", type=str, required=True, choices=["wds", "parquet"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--formats", type=str, default="parquet,webdataset,lance")
    parser.add_argument("--use-filter", action="store_true", dest="use_filter")
    parser.add_argument("--no-filter", action="store_false", dest="use_filter")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--materialize-on-write", action="store_true", dest="materialize_on_write")
    parser.add_argument("--no-materialize-on-write", action="store_false", dest="materialize_on_write")
    parser.add_argument("--materialize-on-read", action="store_true", dest="materialize_on_read")
    parser.add_argument("--no-materialize-on-read", action="store_false", dest="materialize_on_read")
    parser.add_argument("--on-materialize-error", type=str, default="warn", choices=["error", "warn", "drop_row", "drop_sample"])
    parser.add_argument("--mode", type=str, default="overwrite", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--source-id-field", type=str, default="pdf_name")
    parser.set_defaults(materialize_on_write=False, materialize_on_read=False, use_filter=True)
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",")]
    valid_formats = set(_WRITER_CLASSES.keys())
    for fmt in formats:
        if fmt not in valid_formats:
            logger.error("Invalid format '{}'. Valid: {}", fmt, valid_formats)
            return 1

    all_metrics: dict[str, Any] = {
        "reader_type": args.reader_type,
        "use_filter": args.use_filter,
        "materialize_on_write": args.materialize_on_write,
        "formats_requested": formats,
    }
    all_success = True

    for fmt in formats:
        fmt_result = run_format(fmt, args)
        all_metrics[fmt] = fmt_result
        if not fmt_result["is_success"]:
            all_success = False

    results = {
        "params": {
            "executor": args.executor,
            "reader_type": args.reader_type,
            "input_path": args.input_path,
            "output_path": args.output_path,
            "formats": formats,
            "use_filter": args.use_filter,
            "files_per_partition": args.files_per_partition,
            "materialize_on_write": args.materialize_on_write,
            "materialize_on_read": args.materialize_on_read,
            "on_materialize_error": args.on_materialize_error,
            "mode": args.mode,
            "source_id_field": args.source_id_field,
        },
        "metrics": {
            "is_success": all_success,
            **all_metrics,
        },
        "tasks": [],
    }
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
