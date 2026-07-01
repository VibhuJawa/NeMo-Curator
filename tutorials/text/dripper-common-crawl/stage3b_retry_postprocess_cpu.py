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

"""Stage 3b-c: CPU postprocess for retry LLM responses."""

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

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.experimental.dripper._html_compression import HTML_CHARS_COL, HTML_ZLIB_COL, get_html_from_row
from nemo_curator.stages.text.experimental.dripper.stage import (
    _coerce_optional_str,
    _load_mineru_html_bindings,
    _sanitize_case_output_html,
)
from nemo_curator.tasks import DocumentBatch, FileGroupTask

PREPROCESS_STATUS_COL = "stage2a_status"
PREPROCESS_ERROR_COL = "stage2a_error"
LLM_STATUS_COL = "stage2b_status"
LLM_ERROR_COL = "stage2b_error"
POSTPROCESS_STATUS_COL = "stage3b_status"
POSTPROCESS_ERROR_COL = "stage3b_error"
OUTPUT_FORMAT = "mm_md"

OUTPUT_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("url", pa.string()),
        ("url_host_name", pa.string()),
        ("cluster_id", pa.string()),
        ("cluster_role", pa.string()),
        ("host_bucket", pa.int64()),
        ("host_bucket_label", pa.string()),
        ("warc_filename", pa.string()),
        ("warc_record_offset", pa.int64()),
        ("warc_record_length", pa.int64()),
        ("source_manifest_file", pa.string()),
        ("llm_response", pa.string()),
        ("dripper_content", pa.string()),
        ("dripper_html", pa.string()),
        ("dripper_error", pa.string()),
        ("inference_time_s", pa.float64()),
        (PREPROCESS_STATUS_COL, pa.string()),
        (PREPROCESS_ERROR_COL, pa.string()),
        (LLM_STATUS_COL, pa.string()),
        (LLM_ERROR_COL, pa.string()),
        (POSTPROCESS_STATUS_COL, pa.string()),
        (POSTPROCESS_ERROR_COL, pa.string()),
    ]
)


def _init_ray_from_slurm() -> None:
    if ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
        return
    ray_kwargs: dict[str, object] = {"ignore_reinit_error": True, "num_gpus": 0}
    if os.environ.get("RAY_TMPDIR"):
        ray_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        ray_kwargs["num_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    ray.init(**ray_kwargs)


def _response_path(response_dir: Path, prompt_path: Path) -> Path:
    stem = prompt_path.stem.removeprefix("prompt_")
    return response_dir / f"response_{stem}.parquet"


def _output_path(output_dir: Path, prompt_path: Path) -> Path:
    stem = prompt_path.stem.removeprefix("prompt_")
    return output_dir / f"stage3b_{stem}.parquet"


def _prompt_files(input_path: Path) -> list[Path]:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("prompt_*.parquet"))
    files = [path for path in files if ".tmp" not in path.name]
    if not files:
        raise FileNotFoundError(f"No prompt_*.parquet files found in {input_path}")
    return files


def _discover_tasks(prompt_dir: Path, response_dir: Path, output_dir: Path) -> list[FileGroupTask]:
    tasks: list[FileGroupTask] = []
    for prompt_path in _prompt_files(prompt_dir):
        response_path = _response_path(response_dir, prompt_path)
        tasks.append(
            FileGroupTask(
                dataset_name=prompt_path.stem,
                data=[str(prompt_path)],
                reader_config={
                    "prompt_path": str(prompt_path),
                    "response_path": str(response_path),
                    "output_path": str(_output_path(output_dir, prompt_path)),
                    "input_rows": pq.read_metadata(prompt_path).num_rows,
                },
            )
        )
    return tasks


def _empty_table() -> pa.Table:
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in OUTPUT_SCHEMA], schema=OUTPUT_SCHEMA)


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: object) -> str:
    return "" if value is None else str(value)


def _record_id(row: dict[str, Any]) -> str:
    return _as_str(row.get("record_id"))


class RetryResponsePostprocessStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    name: str = "stage3b_retry_postprocess"

    def __init__(self, cpus_per_actor: int) -> None:
        super().__init__()
        self.resources = Resources(cpus=float(cpus_per_actor))
        self._bindings = None

    def setup(self, _worker_metadata: object = None) -> None:
        self._bindings = _load_mineru_html_bindings()

    def process(self, task: FileGroupTask) -> DocumentBatch:
        if self._bindings is None:
            self.setup()
        assert self._bindings is not None

        prompt_path = Path(task.reader_config["prompt_path"])
        response_path = Path(task.reader_config["response_path"])
        output_path = Path(task.reader_config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(f".tmp_{os.getpid()}.parquet")
        tmp.unlink(missing_ok=True)
        t0 = time.perf_counter()

        prompt_df = pq.ParquetFile(prompt_path).read().to_pandas()
        if response_path.exists():
            response_df = pq.ParquetFile(response_path).read().to_pandas()
        else:
            response_df = pd.DataFrame(columns=["record_id", "cluster_id"])
        merged = self._merge(prompt_df, response_df)
        output_rows = [self._process_row(row) for row in merged.to_dict("records")]
        table = pa.Table.from_pylist(output_rows, schema=OUTPUT_SCHEMA) if output_rows else _empty_table()
        pq.write_table(table, str(tmp), compression="zstd")
        tmp.rename(output_path)

        status_counts = pd.Series([row[POSTPROCESS_STATUS_COL] for row in output_rows]).value_counts().to_dict()
        metrics = {
            "prompt_path": str(prompt_path),
            "response_path": str(response_path),
            "output_path": str(output_path),
            "input_rows": int(len(prompt_df)),
            "response_rows": int(len(response_df)),
            "output_rows": int(len(output_rows)),
            "ok_rows": int(status_counts.get("ok", 0)),
            "error_rows": int(len(output_rows) - status_counts.get("ok", 0)),
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
        metrics.update({f"status_{key}": int(value) for key, value in status_counts.items()})
        logger.info(
            "stage3b-c {} rows={} ok={} errors={} -> {}",
            prompt_path.name,
            metrics["output_rows"],
            metrics["ok_rows"],
            metrics["error_rows"],
            output_path,
        )
        return DocumentBatch(dataset_name=task.dataset_name, data=pd.DataFrame([metrics]))

    @staticmethod
    def _merge(prompt_df: pd.DataFrame, response_df: pd.DataFrame) -> pd.DataFrame:
        if "record_id" not in prompt_df.columns:
            prompt_df = prompt_df.copy()
            prompt_df["record_id"] = [_record_id(row) for row in prompt_df.to_dict("records")]
        if response_df.empty:
            out = prompt_df.copy()
            for col in ("llm_response", "inference_time_s", LLM_STATUS_COL, LLM_ERROR_COL):
                out[col] = ""
            return out
        response_cols = ["record_id", "cluster_id", "llm_response", "inference_time_s", LLM_STATUS_COL, LLM_ERROR_COL]
        response_cols = [col for col in response_cols if col in response_df.columns]
        response_df = response_df[response_cols].drop_duplicates(["record_id", "cluster_id"], keep="last")
        return prompt_df.merge(response_df, on=["record_id", "cluster_id"], how="left", suffixes=("", "_response"))

    def _process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        base = _base_output(row)
        prep_status = _as_str(row.get(PREPROCESS_STATUS_COL))
        if prep_status != "ok":
            error = _as_str(row.get(PREPROCESS_ERROR_COL)) or "Stage 3b-a did not produce a prompt"
            return {**base, "dripper_error": error, POSTPROCESS_STATUS_COL: prep_status or "stage3b_prompt_error", POSTPROCESS_ERROR_COL: error}
        llm_status = _as_str(row.get(LLM_STATUS_COL))
        if not llm_status or llm_status == "nan":
            error = "missing Stage 3b-b response"
            return {**base, "dripper_error": error, POSTPROCESS_STATUS_COL: "missing_response", POSTPROCESS_ERROR_COL: error}
        if llm_status != "ok":
            error = _as_str(row.get(LLM_ERROR_COL)) or "Stage 3b-b response error"
            return {**base, "dripper_error": error, POSTPROCESS_STATUS_COL: "stage3b_llm_error", POSTPROCESS_ERROR_COL: error}
        response = _as_str(row.get("llm_response"))
        if not response.strip():
            error = "empty Dripper response"
            return {**base, "dripper_error": error, POSTPROCESS_STATUS_COL: "empty_response", POSTPROCESS_ERROR_COL: error}

        try:
            html = get_html_from_row(row)
            if not html.strip():
                raise ValueError("empty HTML input")
            url = _coerce_optional_str(row.get("url"))
            case = self._bindings.case_cls(self._bindings.input_cls(raw_html=html, url=url))
            simplified_html = _as_str(row.get("dripper_simplified_html"))
            mapped_html = _as_str(row.get("dripper_mapped_html"))
            if simplified_html or mapped_html:
                case.process_data = self._bindings.process_data_cls(simpled_html=simplified_html, map_html=mapped_html)
            case.generate_output = self._bindings.generate_output_cls(response=response)
            case = self._bindings.parse_result(case)
            case = self._bindings.extract_main_html_single(case)
            _sanitize_case_output_html(case)
            case = self._bindings.convert2content(case, output_format=OUTPUT_FORMAT)
            output_data = getattr(case, "output_data", None)
            main_html = getattr(output_data, "main_html", "") if output_data is not None else ""
            main_content = getattr(output_data, "main_content", "") if output_data is not None else ""
            if main_content is None:
                main_content = ""
        except Exception as exc:  # noqa: BLE001 - row-level model/HTML failures are explicit output errors
            error = str(exc)
            return {**base, "dripper_error": error, POSTPROCESS_STATUS_COL: "postprocess_error", POSTPROCESS_ERROR_COL: error}

        return {
            **base,
            "dripper_content": _as_str(main_content),
            "dripper_html": _as_str(main_html),
            POSTPROCESS_STATUS_COL: "ok",
            POSTPROCESS_ERROR_COL: "",
        }


def _base_output(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": _record_id(row),
        "url": _as_str(row.get("url")),
        "url_host_name": _as_str(row.get("url_host_name")),
        "cluster_id": _as_str(row.get("cluster_id")),
        "cluster_role": _as_str(row.get("cluster_role")),
        "host_bucket": _as_int(row.get("host_bucket")),
        "host_bucket_label": _as_str(row.get("host_bucket_label")),
        "warc_filename": _as_str(row.get("warc_filename")),
        "warc_record_offset": _as_int(row.get("warc_record_offset")),
        "warc_record_length": _as_int(row.get("warc_record_length")),
        "source_manifest_file": _as_str(row.get("source_manifest_file")),
        "llm_response": _as_str(row.get("llm_response")),
        "dripper_content": "",
        "dripper_html": "",
        "dripper_error": "",
        "inference_time_s": _as_float(row.get("inference_time_s")),
        PREPROCESS_STATUS_COL: _as_str(row.get(PREPROCESS_STATUS_COL)),
        PREPROCESS_ERROR_COL: _as_str(row.get(PREPROCESS_ERROR_COL)),
        LLM_STATUS_COL: _as_str(row.get(LLM_STATUS_COL)),
        LLM_ERROR_COL: _as_str(row.get(LLM_ERROR_COL)),
        POSTPROCESS_STATUS_COL: "",
        POSTPROCESS_ERROR_COL: "",
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _discover_tasks(Path(args.prompts), Path(args.responses), output_dir)
    if args.shard_index or args.num_shards != 1:
        tasks = tasks[len(tasks) * args.shard_index // args.num_shards : len(tasks) * (args.shard_index + 1) // args.num_shards]
    if not tasks:
        raise RuntimeError(f"No Stage 3b-c tasks for shard {args.shard_index}/{args.num_shards}")

    logger.info("Stage 3b-c scheduling {} prompt shard(s)", len(tasks))
    _init_ray_from_slurm()
    stage = RetryResponsePostprocessStage(cpus_per_actor=args.cpus_per_actor)
    pipeline = Pipeline(name="stage3b_retry_postprocess")
    pipeline.add_stage(stage)
    t0 = time.perf_counter()
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []
    elapsed = time.perf_counter() - t0
    metrics: list[dict[str, Any]] = []
    for task in result_tasks:
        if hasattr(task, "to_pandas"):
            metrics.extend(task.to_pandas().to_dict("records"))

    summary = {
        "prompts": str(Path(args.prompts)),
        "responses": str(Path(args.responses)),
        "output": str(output_dir),
        "elapsed_s": round(elapsed, 3),
        "prompt_shards": len(tasks),
        "completed_shards": len(metrics),
        "input_rows": int(sum(item.get("input_rows", 0) for item in metrics)),
        "response_rows": int(sum(item.get("response_rows", 0) for item in metrics)),
        "output_rows": int(sum(item.get("output_rows", 0) for item in metrics)),
        "ok_rows": int(sum(item.get("ok_rows", 0) for item in metrics)),
        "error_rows": int(sum(item.get("error_rows", 0) for item in metrics)),
        "shards": metrics,
    }
    summary_path = output_dir / "_stage3b_postprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Stage 3b-c done in {:.1f}s rows={} ok={} errors={} summary={}",
        elapsed,
        summary["output_rows"],
        summary["ok_rows"],
        summary["error_rows"],
        summary_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prompts", required=True, help="Stage 3b-a prompt directory")
    parser.add_argument("--responses", required=True, help="Stage 3b-b response directory")
    parser.add_argument("--output", required=True, help="Stage 3b-c output directory")
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
