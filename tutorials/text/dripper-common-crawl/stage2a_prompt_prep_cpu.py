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

"""Stage 2a: CPU prompt preparation for Dripper representative/singleton rows."""

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

PROMPT_COL = "prompt"
ITEM_COUNT_COL = "item_count"
REQUEST_MAX_TOKENS_COL = "request_max_tokens"
PREPROCESS_STATUS_COL = "stage2a_status"
PREPROCESS_ERROR_COL = "stage2a_error"
PROMPT_CHARS_COL = "prompt_chars"
NEEDS_LLM_COL = "_dripper_needs_llm"
PRIMARY_ERROR_COL = "_dripper_primary_error"
EMPTY_INPUT_COL = "_dripper_empty_input"
BUCKET_FILE_RE = re.compile(r"^host_bucket_(?P<label>\d+)(?:[_.-].*)?\.parquet$")

OUTPUT_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("url", pa.string()),
        ("url_host_name", pa.string()),
        ("host_hash64", pa.string()),
        ("host_bucket", pa.int64()),
        ("host_bucket_label", pa.string()),
        ("cluster_id", pa.string()),
        ("cluster_role", pa.string()),
        (HTML_ZLIB_COL, pa.binary()),
        (HTML_CHARS_COL, pa.int64()),
        ("warc_filename", pa.string()),
        ("warc_record_offset", pa.int64()),
        ("warc_record_length", pa.int64()),
        ("source_manifest_file", pa.string()),
        ("dripper_simplified_html", pa.string()),
        ("dripper_mapped_html", pa.string()),
        (PROMPT_COL, pa.string()),
        (ITEM_COUNT_COL, pa.int64()),
        ("dripper_preprocess_time_s", pa.float64()),
        (REQUEST_MAX_TOKENS_COL, pa.int64()),
        (PROMPT_CHARS_COL, pa.int64()),
        (NEEDS_LLM_COL, pa.bool_()),
        (PRIMARY_ERROR_COL, pa.string()),
        (EMPTY_INPUT_COL, pa.bool_()),
        (PREPROCESS_STATUS_COL, pa.string()),
        (PREPROCESS_ERROR_COL, pa.string()),
    ]
)

READ_COLS = [
    "record_id",
    "url",
    "url_host_name",
    "host_hash64",
    "host_bucket",
    "host_bucket_label",
    "cluster_id",
    "cluster_role",
    "is_representative",
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


def _record_id(row: dict[str, Any]) -> str:
    parts = [row.get("warc_filename"), row.get("warc_record_offset"), row.get("warc_record_length")]
    if all(part is not None and str(part) for part in parts):
        return "|".join(str(part) for part in parts)
    return str(row.get("record_id") or row.get("url") or "")


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _empty_table() -> pa.Table:
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in OUTPUT_SCHEMA], schema=OUTPUT_SCHEMA)


def _bucket_label_from_path(path: Path) -> str:
    match = BUCKET_FILE_RE.match(path.name)
    return match.group("label") if match else path.stem


def _discover_inputs(input_path: Path, output_dir: Path) -> list[FileGroupTask]:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("host_bucket_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No host_bucket_*.parquet files found in {input_path}")

    tasks: list[FileGroupTask] = []
    for path in files:
        label = _bucket_label_from_path(path)
        output_path = output_dir / f"prompt_{path.stem}.parquet"
        tasks.append(
            FileGroupTask(
                dataset_name=f"stage2a_{label}",
                data=[str(path)],
                reader_config={
                    "bucket_label": label,
                    "output_path": str(output_path),
                    "input_rows": pq.read_metadata(path).num_rows,
                },
            )
        )
    return tasks


class PromptPrepStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    name: str = "stage2a_prompt_prep"

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
        if HTML_ZLIB_COL not in schema_names:
            raise ValueError(f"{input_path} is missing required HTML column: {HTML_ZLIB_COL!r}")
        if "cluster_role" not in schema_names:
            raise ValueError(f"{input_path} is missing required Stage 1b column: 'cluster_role'")

        cols = [col for col in READ_COLS if col in schema_names]
        df = pf.read(columns=cols).to_pandas()
        input_rows = len(df)
        df = df[df["cluster_role"].isin(["representative", "singleton"])].reset_index(drop=True)
        llm_candidate_rows = len(df)

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
        errors = int(len(out) - ok_rows)
        metrics = self._metrics(
            input_path,
            output_path,
            input_rows,
            llm_candidate_rows,
            ok_rows,
            errors,
            prompt_chars,
            int(out[ITEM_COUNT_COL].max()) if len(out) else 0,
            time.perf_counter() - t0,
        )
        metrics.update({f"status_{key}": int(value) for key, value in status_counts.items()})
        logger.info(
            "stage2a {} rows={} ok={} errors={} prompt_chars={} -> {}",
            input_path.name,
            llm_candidate_rows,
            ok_rows,
            errors,
            prompt_chars,
            output_path,
        )
        return DocumentBatch(dataset_name=task.dataset_name, data=pd.DataFrame([metrics]))

    @staticmethod
    def _metrics(
        input_path: Path,
        output_path: Path,
        input_rows: int,
        llm_candidate_rows: int,
        ok_rows: int,
        errors: int,
        prompt_chars: int,
        max_item_count: int,
        elapsed_s: float,
    ) -> dict[str, Any]:
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_rows": int(input_rows),
            "llm_candidate_rows": int(llm_candidate_rows),
            "ok_rows": int(ok_rows),
            "error_rows": int(errors),
            "prompt_chars": int(prompt_chars),
            "max_item_count": int(max_item_count),
            "elapsed_s": round(elapsed_s, 3),
        }


def _normalize_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prompt" not in out.columns:
        out[PROMPT_COL] = out.get("_dripper_prompt", pd.Series("", index=out.index)).fillna("").astype(str)
    if ITEM_COUNT_COL not in out.columns:
        out[ITEM_COUNT_COL] = pd.to_numeric(out.get("dripper_item_count", 0), errors="coerce").fillna(0).astype("int64")
    if REQUEST_MAX_TOKENS_COL not in out.columns:
        out[REQUEST_MAX_TOKENS_COL] = (
            pd.to_numeric(out.get("dripper_request_max_tokens", 0), errors="coerce").fillna(0).astype("int64")
        )
    if "record_id" not in out.columns:
        out["record_id"] = [_record_id(row) for row in out.to_dict("records")]
    if NEEDS_LLM_COL not in out.columns:
        out[NEEDS_LLM_COL] = out[PROMPT_COL].astype(str).str.len() > 0
    if PRIMARY_ERROR_COL not in out.columns:
        out[PRIMARY_ERROR_COL] = ""
    if EMPTY_INPUT_COL not in out.columns:
        out[EMPTY_INPUT_COL] = pd.to_numeric(out.get(HTML_CHARS_COL, 0), errors="coerce").fillna(0).le(0)

    status: list[str] = []
    errors: list[str] = []
    prompt_chars: list[int] = []
    for row in out.to_dict("records"):
        prompt = str(row.get(PROMPT_COL) or "")
        primary_error = str(row.get(PRIMARY_ERROR_COL) or "")
        warning = str(row.get("dripper_warning") or "")
        prompt_chars.append(len(prompt))
        if _as_bool(row.get(EMPTY_INPUT_COL)):
            status.append("empty_html")
            errors.append(primary_error or warning or "empty HTML input")
        elif primary_error:
            status.append("preprocess_error")
            errors.append(primary_error)
        elif not _as_bool(row.get(NEEDS_LLM_COL)):
            status.append("no_item_ids")
            errors.append(warning or "no _item_id attributes after simplification")
        elif len(prompt) <= 10:
            status.append("empty_prompt")
            errors.append(warning or "empty Dripper prompt")
        else:
            status.append("ok")
            errors.append("")

    out[PROMPT_CHARS_COL] = prompt_chars
    out[PREPROCESS_STATUS_COL] = status
    out[PREPROCESS_ERROR_COL] = errors

    for col in OUTPUT_SCHEMA.names:
        if col not in out.columns:
            out[col] = None
    out["record_id"] = [_record_id(row) for row in out.to_dict("records")]
    out["host_bucket"] = [_as_int(value) for value in out["host_bucket"].tolist()]
    out[HTML_CHARS_COL] = [_as_int(value) for value in out[HTML_CHARS_COL].tolist()]
    out["warc_record_offset"] = [_as_int(value) for value in out["warc_record_offset"].tolist()]
    out["warc_record_length"] = [_as_int(value) for value in out["warc_record_length"].tolist()]
    out[ITEM_COUNT_COL] = [_as_int(value) for value in out[ITEM_COUNT_COL].tolist()]
    out[REQUEST_MAX_TOKENS_COL] = [_as_int(value) for value in out[REQUEST_MAX_TOKENS_COL].tolist()]
    out[PROMPT_CHARS_COL] = [_as_int(value) for value in out[PROMPT_CHARS_COL].tolist()]
    out["dripper_preprocess_time_s"] = [_as_float(value) for value in out["dripper_preprocess_time_s"].tolist()]
    out[NEEDS_LLM_COL] = [_as_bool(value) for value in out[NEEDS_LLM_COL].tolist()]
    out[EMPTY_INPUT_COL] = [_as_bool(value) for value in out[EMPTY_INPUT_COL].tolist()]
    for col in (
        "record_id",
        "url",
        "url_host_name",
        "host_hash64",
        "host_bucket_label",
        "cluster_id",
        "cluster_role",
        "warc_filename",
        "source_manifest_file",
        "dripper_simplified_html",
        "dripper_mapped_html",
        PROMPT_COL,
        PRIMARY_ERROR_COL,
        PREPROCESS_STATUS_COL,
        PREPROCESS_ERROR_COL,
    ):
        out[col] = [_as_str_or_none(value) or "" for value in out[col].tolist()]
    return out


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _discover_inputs(Path(args.input), output_dir)
    if args.shard_index or args.num_shards != 1:
        tasks = tasks[len(tasks) * args.shard_index // args.num_shards : len(tasks) * (args.shard_index + 1) // args.num_shards]
    if not tasks:
        raise RuntimeError(f"No Stage 2a tasks for shard {args.shard_index}/{args.num_shards}")

    logger.info("Stage 2a scheduling {} bucket file(s)", len(tasks))
    _init_ray_from_slurm()
    stage = PromptPrepStage(cpus_per_actor=args.cpus_per_actor)
    pipeline = Pipeline(name="stage2a_prompt_prep")
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
        "llm_candidate_rows": int(sum(item.get("llm_candidate_rows", 0) for item in metrics)),
        "ok_rows": int(sum(item.get("ok_rows", 0) for item in metrics)),
        "error_rows": int(sum(item.get("error_rows", 0) for item in metrics)),
        "prompt_chars": int(sum(item.get("prompt_chars", 0) for item in metrics)),
        "prompt_shards": metrics,
    }
    summary_path = output_dir / "_stage2a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Stage 2a done in {:.1f}s candidate_rows={} ok={} errors={} summary={}",
        elapsed,
        summary["llm_candidate_rows"],
        summary["ok_rows"],
        summary["error_rows"],
        summary_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Stage 1b output directory containing host_bucket_*.parquet")
    parser.add_argument("--output", required=True, help="Stage 2a prompt output directory")
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
