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

"""Stage 3b-d: merge retry LLM content back into Stage 3 bucket outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, FileGroupTask

from stage3_cpu_propagation import OUTPUT_SCHEMA

STAGE3_PREFIX = "stage3_host_bucket_"
RETRY_PREFIX = "stage3b_retry_host_bucket_"


def _init_ray_from_slurm() -> None:
    if ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
        return
    ray_kwargs: dict[str, object] = {"ignore_reinit_error": True, "num_gpus": 0}
    if os.environ.get("RAY_TMPDIR"):
        ray_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        ray_kwargs["num_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    ray.init(**ray_kwargs)


def _bucket_label(path: Path) -> str:
    if not path.name.startswith(STAGE3_PREFIX) or path.suffix != ".parquet":
        raise ValueError(f"{path} is not a stage3_host_bucket_*.parquet file")
    return path.stem.removeprefix(STAGE3_PREFIX)


def _retry_path(retry_dir: Path, label: str) -> Path:
    return retry_dir / f"{RETRY_PREFIX}{label}.parquet"


def _output_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"stage3_final_host_bucket_{label}.parquet"


def _discover_tasks(stage3_dir: Path, retry_dir: Path, output_dir: Path) -> list[FileGroupTask]:
    files = [stage3_dir] if stage3_dir.is_file() else sorted(stage3_dir.glob("stage3_host_bucket_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No stage3_host_bucket_*.parquet files found in {stage3_dir}")

    tasks: list[FileGroupTask] = []
    for stage3_path in files:
        label = _bucket_label(stage3_path)
        tasks.append(
            FileGroupTask(
                dataset_name=f"stage3b_merge_{label}",
                data=[str(stage3_path)],
                reader_config={
                    "bucket_label": label,
                    "stage3_path": str(stage3_path),
                    "retry_path": str(_retry_path(retry_dir, label)),
                    "output_path": str(_output_path(output_dir, label)),
                    "input_rows": pq.read_metadata(stage3_path).num_rows,
                },
            )
        )
    return tasks


def _as_str(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp_{os.getpid()}.parquet")
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    pq.write_table(table, str(tmp), compression="zstd")
    tmp.rename(path)


class RetryMergeStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    name = "stage3b_retry_merge"

    def __init__(self, task_cpus: int) -> None:
        super().__init__()
        self.resources = Resources(cpus=float(task_cpus))

    def process(self, task: FileGroupTask) -> DocumentBatch:
        label = str(task.reader_config["bucket_label"])
        stage3_path = Path(task.reader_config["stage3_path"])
        retry_path = Path(task.reader_config["retry_path"])
        output_path = Path(task.reader_config["output_path"])
        t0 = time.perf_counter()

        stage3_df = pq.read_table(stage3_path).to_pandas()
        retry_df = pq.read_table(retry_path).to_pandas() if retry_path.exists() else pd.DataFrame()
        retry_rows = int(len(retry_df))
        retry_success: dict[tuple[str, str], dict[str, Any]] = {}
        retry_ok_rows = 0
        if not retry_df.empty:
            for row in retry_df.to_dict("records"):
                is_ok = _as_str(row.get("stage3b_status")) == "ok"
                content = _as_str(row.get("dripper_content"))
                if is_ok and len(content.strip()) > 5:
                    retry_success[(_as_str(row.get("record_id")), _as_str(row.get("cluster_id")))] = row
                    retry_ok_rows += 1

        output_rows: list[dict[str, Any]] = []
        updated_rows = 0
        for row in stage3_df.to_dict("records"):
            out = {field.name: row.get(field.name) for field in OUTPUT_SCHEMA}
            key = (_as_str(row.get("record_id")), _as_str(row.get("cluster_id")))
            retry = retry_success.get(key)
            if retry is not None:
                out["dripper_content"] = _as_str(retry.get("dripper_content"))
                out["dripper_html"] = _as_str(retry.get("dripper_html"))
                out["dripper_error"] = ""
                out["dripper_time_s"] = _as_float(retry.get("inference_time_s"))
                out["propagation_success"] = True
                out["propagation_method"] = "llm_retry"
                updated_rows += 1
            output_rows.append(out)

        _write_output(output_path, output_rows)
        final_success = int(sum(1 for row in output_rows if bool(row.get("propagation_success"))))
        metrics = {
            "bucket_label": label,
            "stage3_path": str(stage3_path),
            "retry_path": str(retry_path),
            "output_path": str(output_path),
            "input_rows": int(len(stage3_df)),
            "retry_rows": retry_rows,
            "retry_ok_rows": int(retry_ok_rows),
            "updated_rows": int(updated_rows),
            "output_rows": int(len(output_rows)),
            "final_success_rows": final_success,
            "final_error_rows": int(len(output_rows) - final_success),
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
        logger.info(
            "stage3b-d bucket={} rows={} retry={} updated={} final_success={} -> {}",
            label,
            len(output_rows),
            retry_rows,
            updated_rows,
            final_success,
            output_path,
        )
        return DocumentBatch(dataset_name=task.dataset_name, data=pd.DataFrame([metrics]))


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _discover_tasks(Path(args.stage3), Path(args.retry_outputs), output_dir)
    if args.shard_index or args.num_shards != 1:
        tasks = tasks[len(tasks) * args.shard_index // args.num_shards : len(tasks) * (args.shard_index + 1) // args.num_shards]
    if not tasks:
        raise RuntimeError(f"No Stage 3b merge tasks for shard {args.shard_index}/{args.num_shards}")

    logger.info("Stage 3b-d scheduling {} merge task(s)", len(tasks))
    _init_ray_from_slurm()
    stage = RetryMergeStage(task_cpus=args.task_cpus)
    pipeline = Pipeline(name="stage3b_retry_merge")
    pipeline.add_stage(stage)
    t0 = time.perf_counter()
    result_tasks = pipeline.run(executor=RayDataExecutor(), initial_tasks=tasks) or []
    elapsed = time.perf_counter() - t0

    metrics: list[dict[str, Any]] = []
    for task in result_tasks:
        if hasattr(task, "to_pandas"):
            metrics.extend(task.to_pandas().to_dict("records"))

    summary = {
        "stage3": str(Path(args.stage3)),
        "retry_outputs": str(Path(args.retry_outputs)),
        "output": str(output_dir),
        "elapsed_s": round(elapsed, 3),
        "bucket_tasks": len(tasks),
        "completed_buckets": len(metrics),
        "input_rows": int(sum(item.get("input_rows", 0) for item in metrics)),
        "retry_rows": int(sum(item.get("retry_rows", 0) for item in metrics)),
        "retry_ok_rows": int(sum(item.get("retry_ok_rows", 0) for item in metrics)),
        "updated_rows": int(sum(item.get("updated_rows", 0) for item in metrics)),
        "output_rows": int(sum(item.get("output_rows", 0) for item in metrics)),
        "final_success_rows": int(sum(item.get("final_success_rows", 0) for item in metrics)),
        "final_error_rows": int(sum(item.get("final_error_rows", 0) for item in metrics)),
        "buckets": metrics,
    }
    summary_path = output_dir / "_stage3b_merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Stage 3b-d done in {:.1f}s rows={} updated={} final_success={} final_errors={} summary={}",
        elapsed,
        summary["output_rows"],
        summary["updated_rows"],
        summary["final_success_rows"],
        summary["final_error_rows"],
        summary_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage3", required=True, help="Stage 3a output directory")
    parser.add_argument("--retry-outputs", required=True, help="Stage 3b-c retry output directory")
    parser.add_argument("--output", required=True, help="Final merged output directory")
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--task-cpus", type=int, default=int(os.environ.get("TASK_CPUS", "1")))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
