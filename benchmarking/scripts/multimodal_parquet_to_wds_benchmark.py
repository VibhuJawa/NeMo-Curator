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

"""Benchmark for multimodal Parquet-to-WebDataset conversion."""

import argparse
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
from loguru import logger
from utils import collect_webdataset_output_metrics, setup_executor, write_benchmark_results

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.multimodal.io import MultimodalParquetReader, MultimodalWebdatasetWriterStage
from nemo_curator.tasks import MultiBatchTask
from nemo_curator.tasks.utils import TaskPerfUtils


@dataclass
class _S3SourceRefFilterStage(ProcessingStage[MultiBatchTask, MultiBatchTask]):
    """Drop samples where any image row has a non-S3 source_ref (benchmark-only helper)."""

    name: str = "s3_source_ref_filter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: MultiBatchTask) -> MultiBatchTask | None:
        df = task.to_pandas()
        image_df = df[df["modality"] == "image"].copy()
        ref_strs = image_df["source_ref"].fillna("").astype(str)
        non_s3 = ref_strs.str.contains('"path":', na=False) & ~ref_strs.str.contains('"path": "s3://', na=False)
        bad_sids = set(image_df.loc[non_s3, "sample_id"].unique())

        if not bad_sids:
            return task

        filtered = df[~df["sample_id"].isin(bad_sids)].reset_index(drop=True)
        logger.info(f"S3 filter: dropped {len(bad_sids)} samples with non-S3 image refs, {len(filtered)} rows remain")
        if filtered.empty:
            return None

        table = pa.Table.from_pandas(filtered, preserve_index=False)
        return MultiBatchTask(
            task_id=task.task_id, dataset_name=task.dataset_name,
            data=table, _metadata=task._metadata, _stage_perf=task._stage_perf,
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


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    storage_options = _build_storage_options(args)
    write_kwargs: dict[str, Any] = {}
    if storage_options:
        write_kwargs["storage_options"] = storage_options
    pipeline = Pipeline(
        name="multimodal_parquet_to_wds_benchmark",
        description="Benchmark: Multimodal parquet to WebDataset tar shards",
    )
    pipeline.add_stage(
        MultimodalParquetReader(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs={},
        )
    )
    if args.filter_s3_only:
        pipeline.add_stage(_S3SourceRefFilterStage())
    pipeline.add_stage(
        MultimodalWebdatasetWriterStage(
            path=args.output_path,
            materialize_on_write=args.materialize_on_write,
            on_materialize_error=args.on_materialize_error,
            write_kwargs=write_kwargs,
            mode=args.mode,
        )
    )
    return pipeline


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    executor = setup_executor(args.executor)
    input_path = str(Path(args.input_path).absolute())
    output_path = Path(args.output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    output_tasks = []
    success = False
    try:
        pipeline = create_pipeline(args)
        logger.info("Pipeline:\n{}", pipeline.describe())
        output_tasks = pipeline.run(executor)
        success = True
    except Exception as e:
        logger.error("Benchmark failed: {}", e)
        logger.debug(traceback.format_exc())

    elapsed = time.perf_counter() - start
    metrics_start = time.perf_counter()
    output_metrics = collect_webdataset_output_metrics(output_path)
    metrics_elapsed = time.perf_counter() - metrics_start
    logger.info("collect_webdataset_output_metrics took {:.3f}s", metrics_elapsed)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    samples = output_metrics.get("num_samples", 0)
    return {
        "params": {
            "executor": args.executor,
            "input_path": input_path,
            "output_path": str(output_path),
            "files_per_partition": args.files_per_partition,
            "output_max_batch_bytes": args.output_max_batch_bytes,
            "materialize_on_write": args.materialize_on_write,
            "on_materialize_error": args.on_materialize_error,
            "mode": args.mode,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": elapsed,
            "throughput_samples_per_sec": (samples / elapsed) if elapsed > 0 else 0.0,
            **task_metrics,
            **output_metrics,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal Parquet-to-WebDataset benchmark")
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument("--materialize-on-write", action="store_true", dest="materialize_on_write")
    parser.add_argument("--no-materialize-on-write", action="store_false", dest="materialize_on_write")
    parser.add_argument(
        "--on-materialize-error", type=str, default="error", choices=["error", "warn", "drop_row", "drop_sample"],
    )
    parser.add_argument("--mode", type=str, default="overwrite", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--filter-s3-only", action="store_true", default=False)
    parser.add_argument("--s3-access-key", type=str, default=None)
    parser.add_argument("--s3-secret-key", type=str, default=None)
    parser.add_argument("--s3-endpoint", type=str, default=None)
    parser.add_argument("--s3-region", type=str, default=None)
    parser.set_defaults(materialize_on_write=False)
    args = parser.parse_args()

    ray_client = RayClient()
    ray_client.start()
    try:
        results = run_benchmark(args)
    except Exception as e:
        logger.error("Benchmark crashed: {}", e)
        logger.debug(traceback.format_exc())
        results = {
            "params": vars(args),
            "metrics": {"is_success": False},
            "tasks": [],
        }
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
        ray_client.stop()

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
