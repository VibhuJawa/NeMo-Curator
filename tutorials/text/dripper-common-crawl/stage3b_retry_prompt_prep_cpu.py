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

"""Stage 3b-a: CPU prompt preparation for sibling retry rows."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.experimental.dripper import DripperHTMLPreprocessStage
from nemo_curator.stages.text.experimental.dripper._html_compression import HTML_CHARS_COL, HTML_ZLIB_COL
from nemo_curator.tasks import DocumentBatch, FileGroupTask

from stage2a_prompt_prep_cpu import (
    ITEM_COUNT_COL,
    OUTPUT_SCHEMA,
    PREPROCESS_STATUS_COL,
    PROMPT_CHARS_COL,
    _empty_table,
    _normalize_output,
)

RETRY_FILE_RE = re.compile(r"^retry_host_bucket_(?P<label>\d+)(?:[_.-].*)?\.parquet$")
READ_COLS = [
    "record_id",
    "url",
    "url_host_name",
    "host_hash64",
    "host_bucket",
    "host_bucket_label",
    "cluster_id",
    "cluster_role",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "source_manifest_file",
]


def _init_ray_from_slurm() -> None:
    if ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
        return
    ray_kwargs: dict[str, object] = {"ignore_reinit_error": True, "num_gpus": 0}
    if os.environ.get("RAY_TMPDIR"):
        ray_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        ray_kwargs["num_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    ray.init(**ray_kwargs)


def _retry_label(path: Path) -> str:
    match = RETRY_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"{path} is not a retry_host_bucket_*.parquet file")
    return match.group("label")


def _discover_inputs(input_path: Path, output_dir: Path) -> list[FileGroupTask]:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("retry_host_bucket_*.parquet"))
    files = [path for path in files if ".tmp" not in path.name]
    if not files:
        raise FileNotFoundError(f"No retry_host_bucket_*.parquet files found in {input_path}")

    tasks: list[FileGroupTask] = []
    for path in files:
        label = _retry_label(path)
        output_path = output_dir / f"prompt_retry_host_bucket_{label}.parquet"
        tasks.append(
            FileGroupTask(
                dataset_name=f"stage3b_retry_prompt_{label}",
                data=[str(path)],
                reader_config={
                    "bucket_label": label,
                    "output_path": str(output_path),
                    "input_rows": pq.read_metadata(path).num_rows,
                },
            )
        )
    return tasks


class RetryPromptPrepStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    name: str = "stage3b_retry_prompt_prep"

    def __init__(self, cpus_per_actor: int) -> None:
        super().__init__()
        self.resources = Resources(cpus=float(cpus_per_actor))
        self._prep: DripperHTMLPreprocessStage | None = None

    def setup(self, _worker_metadata: object = None) -> None:
        self._prep = DripperHTMLPreprocessStage(html_col=HTML_ZLIB_COL, url_col="url", worker_count=1)
        self._prep.setup()

    def process(self, task: FileGroupTask) -> DocumentBatch:
        if self._prep is None:
            self.setup()
        assert self._prep is not None

        input_path = Path(task.data[0])
        output_path = Path(task.reader_config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(f".tmp_{os.getpid()}.parquet")
        tmp.unlink(missing_ok=True)
        t0 = time.perf_counter()

        pf = pq.ParquetFile(input_path)
        schema_names = set(pf.schema_arrow.names)
        required = [HTML_ZLIB_COL, "cluster_role", "record_id", "cluster_id"]
        missing = [col for col in required if col not in schema_names]
        if missing:
            raise ValueError(f"{input_path} is missing required Stage 3b retry columns: {missing}")

        cols = [col for col in READ_COLS if col in schema_names]
        df = pf.read(columns=cols).to_pandas()
        input_rows = len(df)
        if not df.empty and not df["cluster_role"].astype(str).eq("sibling").all():
            raise ValueError(f"{input_path} contains non-sibling retry rows")

        if df.empty:
            pq.write_table(_empty_table(), str(tmp), compression="zstd")
            tmp.rename(output_path)
            metrics = self._metrics(input_path, output_path, input_rows, 0, 0, 0, 0, 0, time.perf_counter() - t0)
            return DocumentBatch(dataset_name=task.dataset_name, data=pd.DataFrame([metrics]))

        out = self._prep.process(DocumentBatch(dataset_name=task.dataset_name, data=df)).to_pandas()
        out = _normalize_output(out)
        out = out[[field.name for field in OUTPUT_SCHEMA]]
        pq.write_table(pa.Table.from_pandas(out, schema=OUTPUT_SCHEMA, preserve_index=False), str(tmp), compression="zstd")
        tmp.rename(output_path)

        status_counts = out[PREPROCESS_STATUS_COL].value_counts().to_dict()
        ok_rows = int(status_counts.get("ok", 0))
        prompt_chars = int(pd.to_numeric(out[PROMPT_CHARS_COL], errors="coerce").fillna(0).sum())
        error_rows = int(len(out) - ok_rows)
        metrics = self._metrics(
            input_path,
            output_path,
            input_rows,
            len(out),
            ok_rows,
            error_rows,
            prompt_chars,
            int(out[ITEM_COUNT_COL].max()) if len(out) else 0,
            time.perf_counter() - t0,
        )
        metrics.update({f"status_{key}": int(value) for key, value in status_counts.items()})
        logger.info(
            "stage3b-a {} rows={} ok={} errors={} prompt_chars={} -> {}",
            input_path.name,
            len(out),
            ok_rows,
            error_rows,
            prompt_chars,
            output_path,
        )
        return DocumentBatch(dataset_name=task.dataset_name, data=pd.DataFrame([metrics]))

    @staticmethod
    def _metrics(
        input_path: Path,
        output_path: Path,
        input_rows: int,
        retry_rows: int,
        ok_rows: int,
        error_rows: int,
        prompt_chars: int,
        max_item_count: int,
        elapsed_s: float,
    ) -> dict[str, Any]:
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_rows": int(input_rows),
            "retry_rows": int(retry_rows),
            "ok_rows": int(ok_rows),
            "error_rows": int(error_rows),
            "prompt_chars": int(prompt_chars),
            "max_item_count": int(max_item_count),
            "elapsed_s": round(elapsed_s, 3),
        }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _discover_inputs(Path(args.input), output_dir)
    if args.shard_index or args.num_shards != 1:
        tasks = tasks[len(tasks) * args.shard_index // args.num_shards : len(tasks) * (args.shard_index + 1) // args.num_shards]
    if not tasks:
        raise RuntimeError(f"No Stage 3b retry prompt tasks for shard {args.shard_index}/{args.num_shards}")

    logger.info("Stage 3b-a scheduling {} retry file(s)", len(tasks))
    _init_ray_from_slurm()
    stage = RetryPromptPrepStage(cpus_per_actor=args.cpus_per_actor)
    pipeline = Pipeline(name="stage3b_retry_prompt_prep")
    pipeline.add_stage(stage)
    t0 = time.perf_counter()
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []
    elapsed = time.perf_counter() - t0

    metrics: list[dict[str, Any]] = []
    for task in result_tasks:
        if hasattr(task, "to_pandas"):
            metrics.extend(task.to_pandas().to_dict("records"))

    summary = {
        "input": str(Path(args.input)),
        "output": str(output_dir),
        "elapsed_s": round(elapsed, 3),
        "input_files": len(tasks),
        "completed_files": len(metrics),
        "input_rows": int(sum(item.get("input_rows", 0) for item in metrics)),
        "retry_rows": int(sum(item.get("retry_rows", 0) for item in metrics)),
        "ok_rows": int(sum(item.get("ok_rows", 0) for item in metrics)),
        "error_rows": int(sum(item.get("error_rows", 0) for item in metrics)),
        "prompt_chars": int(sum(item.get("prompt_chars", 0) for item in metrics)),
        "prompt_shards": metrics,
    }
    summary_path = output_dir / "_stage3b_prompt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Stage 3b-a done in {:.1f}s retry_rows={} ok={} errors={} summary={}",
        elapsed,
        summary["retry_rows"],
        summary["ok_rows"],
        summary["error_rows"],
        summary_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Stage 3a stage3b_retry_input directory or one retry parquet file")
    parser.add_argument("--output", required=True, help="Stage 3b prompt output directory")
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cpus-per-actor", type=int, default=int(os.environ.get("CPUS_PER_ACTOR", "2")))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
