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

"""Stage 3: CPU layout-template propagation over host-bucket parquet files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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
from nemo_curator.stages.text.experimental.dripper._html_compression import HTML_CHARS_COL, HTML_ZLIB_COL
from nemo_curator.stages.text.experimental.dripper._mapping_serialization import parse_mapping_data
from nemo_curator.stages.text.experimental.dripper.propagation_stage import (
    _PropagationConfig,
    _StaticTrustConfig,
    _cluster_static_trustworthy,
    _run_content_convert,
    _run_lbp,
    _sibling_propagate,
)
from nemo_curator.stages.text.experimental.dripper.stage import _load_mineru_html_bindings
from nemo_curator.tasks import DocumentBatch, FileGroupTask

STAGE1B_COLUMNS = [
    "record_id",
    "url",
    "url_host_name",
    "host_bucket",
    "host_bucket_label",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
    "cluster_id",
    "cluster_role",
    "is_representative",
    "cluster_size",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "source_manifest_file",
]

STAGE2C_COLUMNS = [
    "record_id",
    "cluster_id",
    "cluster_role",
    "dripper_content",
    "dripper_html",
    "dripper_error",
    "mapping_json",
    "mapping_error",
    "inference_time_s",
    "stage2c_status",
    "stage2c_error",
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("url", pa.string()),
        ("url_host_name", pa.string()),
        ("host_bucket", pa.int64()),
        ("host_bucket_label", pa.string()),
        ("cluster_id", pa.string()),
        ("cluster_role", pa.string()),
        ("cluster_size", pa.int64()),
        ("warc_filename", pa.string()),
        ("warc_record_offset", pa.int64()),
        ("warc_record_length", pa.int64()),
        ("source_manifest_file", pa.string()),
        ("dripper_content", pa.string()),
        ("dripper_html", pa.string()),
        ("dripper_error", pa.string()),
        ("dripper_time_s", pa.float64()),
        ("propagation_success", pa.bool_()),
        ("propagation_method", pa.string()),
    ]
)

RETRY_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("url", pa.string()),
        ("url_host_name", pa.string()),
        ("host_bucket", pa.int64()),
        ("host_bucket_label", pa.string()),
        ("cluster_id", pa.string()),
        ("cluster_role", pa.string()),
        ("cluster_size", pa.int64()),
        (HTML_ZLIB_COL, pa.binary()),
        (HTML_CHARS_COL, pa.int64()),
        ("warc_filename", pa.string()),
        ("warc_record_offset", pa.int64()),
        ("warc_record_length", pa.int64()),
        ("source_manifest_file", pa.string()),
        ("stage3_error", pa.string()),
        ("stage3_method", pa.string()),
    ]
)

BUCKET_PREFIX = "host_bucket_"
STAGE2C_PREFIX = "stage2c_host_bucket_"
RETRY_DIR = "stage3b_retry_input"


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
    if not path.name.startswith(BUCKET_PREFIX) or path.suffix != ".parquet":
        raise ValueError(f"{path} is not a Stage 1b host bucket parquet file")
    return path.stem.removeprefix(BUCKET_PREFIX)


def _stage2c_path(stage2c_dir: Path, label: str) -> Path:
    return stage2c_dir / f"{STAGE2C_PREFIX}{label}.parquet"


def _output_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"stage3_host_bucket_{label}.parquet"


def _retry_output_path(output_dir: Path, label: str) -> Path:
    return output_dir / RETRY_DIR / f"retry_host_bucket_{label}.parquet"


def _require_columns(path: Path, required: list[str]) -> None:
    names = set(pq.read_schema(path).names)
    missing = [col for col in required if col not in names]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _discover_tasks(stage1b_dir: Path, stage2c_dir: Path, output_dir: Path) -> list[FileGroupTask]:
    files = sorted(stage1b_dir.glob("host_bucket_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No host_bucket_*.parquet files found in {stage1b_dir}")

    tasks: list[FileGroupTask] = []
    for manifest_path in files:
        label = _bucket_label(manifest_path)
        template_path = _stage2c_path(stage2c_dir, label)
        if not template_path.exists():
            raise FileNotFoundError(f"Missing Stage 2c template file for bucket {label}: {template_path}")
        tasks.append(
            FileGroupTask(
                dataset_name=f"stage3_host_bucket_{label}",
                data=[str(manifest_path)],
                reader_config={
                    "bucket_label": label,
                    "manifest_path": str(manifest_path),
                    "template_path": str(template_path),
                    "output_path": str(_output_path(output_dir, label)),
                    "retry_output_path": str(_retry_output_path(output_dir, label)),
                    "input_rows": pq.read_metadata(manifest_path).num_rows,
                    "template_rows": pq.read_metadata(template_path).num_rows,
                },
            )
        )
    return tasks


def _as_str(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _as_int(value: object, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _base_output(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": _as_str(row.get("record_id")),
        "url": _as_str(row.get("url")),
        "url_host_name": _as_str(row.get("url_host_name")),
        "host_bucket": _as_int(row.get("host_bucket")),
        "host_bucket_label": _as_str(row.get("host_bucket_label")),
        "cluster_id": _as_str(row.get("cluster_id")),
        "cluster_role": _as_str(row.get("cluster_role")),
        "cluster_size": _as_int(row.get("cluster_size")),
        "warc_filename": _as_str(row.get("warc_filename")),
        "warc_record_offset": _as_int(row.get("warc_record_offset")),
        "warc_record_length": _as_int(row.get("warc_record_length")),
        "source_manifest_file": _as_str(row.get("source_manifest_file")),
        "dripper_content": "",
        "dripper_html": "",
        "dripper_error": "",
        "dripper_time_s": 0.0,
        "propagation_success": False,
        "propagation_method": "",
    }


def _retry_output(row: dict[str, Any], stage3_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": _as_str(row.get("record_id")),
        "url": _as_str(row.get("url")),
        "url_host_name": _as_str(row.get("url_host_name")),
        "host_bucket": _as_int(row.get("host_bucket")),
        "host_bucket_label": _as_str(row.get("host_bucket_label")),
        "cluster_id": _as_str(row.get("cluster_id")),
        "cluster_role": _as_str(row.get("cluster_role")),
        "cluster_size": _as_int(row.get("cluster_size")),
        HTML_ZLIB_COL: row.get(HTML_ZLIB_COL),
        HTML_CHARS_COL: _as_int(row.get(HTML_CHARS_COL)),
        "warc_filename": _as_str(row.get("warc_filename")),
        "warc_record_offset": _as_int(row.get("warc_record_offset")),
        "warc_record_length": _as_int(row.get("warc_record_length")),
        "source_manifest_file": _as_str(row.get("source_manifest_file")),
        "stage3_error": _as_str(stage3_row.get("dripper_error")),
        "stage3_method": _as_str(stage3_row.get("propagation_method")),
    }


def _write_output(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp_{os.getpid()}.parquet")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, str(tmp), compression="zstd")
    tmp.rename(path)


class BucketPropagationStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    name = "stage3_bucket_propagation"

    def __init__(self, task_cpus: int) -> None:
        super().__init__()
        self.resources = Resources(cpus=float(task_cpus))

    def _lbp_fn(self, parser_cache: dict[str, Any]) -> Any:
        params = {
            "more_noise_enable": True,
            "dynamic_classid_similarity_threshold": 0.85,
        }

        def _lbp(html: str, mapping_data: dict[str, Any], dynamic: bool = True) -> tuple[str, str]:
            return _run_lbp(params, html, mapping_data, dynamic, parser_cache)

        return _lbp

    def _content_fn(self) -> Any:
        def _content(main_html: str, url: str) -> tuple[str, str]:
            return _run_content_convert(_load_mineru_html_bindings(), main_html, url)

        return _content

    def _propagation_config(self, parser_cache: dict[str, Any]) -> _PropagationConfig:
        return _PropagationConfig(
            lbp_fn=self._lbp_fn(parser_cache),
            content_fn=self._content_fn(),
            min_ratio=0.25,
            max_ratio=4.0,
        )

    def _static_config(self, parser_cache: dict[str, Any], static_trust: dict[str, bool]) -> _StaticTrustConfig:
        return _StaticTrustConfig(
            memo=static_trust,
            lbp_fn=self._lbp_fn(parser_cache),
            content_fn=self._content_fn(),
            threshold=0.97,
        )

    def process(self, task: FileGroupTask) -> DocumentBatch:
        manifest_path = Path(task.reader_config["manifest_path"])
        template_path = Path(task.reader_config["template_path"])
        output_path = Path(task.reader_config["output_path"])
        retry_output_path = Path(task.reader_config["retry_output_path"])
        label = str(task.reader_config["bucket_label"])
        t0 = time.perf_counter()

        _require_columns(manifest_path, STAGE1B_COLUMNS)
        _require_columns(template_path, STAGE2C_COLUMNS)
        manifest_df = pq.read_table(manifest_path, columns=STAGE1B_COLUMNS).to_pandas()
        template_df = pq.read_table(template_path, columns=STAGE2C_COLUMNS).to_pandas()

        template_by_record = {
            (str(row["record_id"]), str(row["cluster_id"])): row
            for row in template_df.to_dict("records")
        }
        mapping_by_cluster: dict[str, dict[str, Any]] = {}
        mapping_parse_errors: dict[str, str] = {}
        for row in template_df.to_dict("records"):
            if _as_str(row.get("cluster_role")) != "representative":
                continue
            mapping_json = _as_str(row.get("mapping_json"))
            cluster_id = _as_str(row.get("cluster_id"))
            if not mapping_json:
                continue
            try:
                mapping_data = parse_mapping_data(mapping_json)
            except Exception as exc:  # noqa: BLE001
                mapping_parse_errors[cluster_id] = f"mapping_parse_error={exc!s:.200}"
                continue
            if mapping_data is None:
                mapping_parse_errors[cluster_id] = "mapping_parse_error=expected_pickle_b64"
                continue
            mapping_by_cluster[cluster_id] = mapping_data

        rows: list[dict[str, Any]] = []
        retry_rows: list[dict[str, Any]] = []
        methods: Counter[str] = Counter()
        errors: Counter[str] = Counter()
        role_counts: Counter[str] = Counter()
        success_count = 0
        static_trust: dict[str, bool] = {}

        manifest_records = manifest_df.to_dict("records")
        siblings_by_cluster: dict[str, list[dict[str, Any]]] = {}
        for row in manifest_records:
            role = _as_str(row.get("cluster_role"))
            role_counts[role] += 1
            if role == "sibling":
                siblings_by_cluster.setdefault(_as_str(row.get("cluster_id")), []).append(row)

        for row in manifest_records:
            out = _base_output(row)
            role = out["cluster_role"]
            if role in {"representative", "singleton"}:
                template = template_by_record.get((out["record_id"], out["cluster_id"]))
                if template is None:
                    out["dripper_error"] = "missing_stage2c_row"
                    out["propagation_method"] = "llm_missing"
                else:
                    out["dripper_content"] = _as_str(template.get("dripper_content"))
                    out["dripper_html"] = _as_str(template.get("dripper_html"))
                    out["dripper_error"] = _as_str(template.get("dripper_error") or template.get("stage2c_error"))
                    out["dripper_time_s"] = _as_float(template.get("inference_time_s"))
                    out["propagation_success"] = _as_str(template.get("stage2c_status")) == "ok" and len(
                        out["dripper_content"]
                    ) > 5
                    out["propagation_method"] = f"llm_{role}"
                rows.append(out)
                methods[out["propagation_method"]] += 1
                if out["propagation_success"]:
                    success_count += 1
                elif out["dripper_error"]:
                    errors[out["dripper_error"][:160]] += 1
                continue

            if role != "sibling":
                out["dripper_error"] = f"unknown_cluster_role={role}"
                out["propagation_method"] = "failed"
                rows.append(out)
                methods[out["propagation_method"]] += 1
                errors[out["dripper_error"]] += 1
                continue

            cluster_id = out["cluster_id"]
            mapping_data = mapping_by_cluster.get(cluster_id)
            if mapping_data is None:
                out["dripper_error"] = mapping_parse_errors.get(cluster_id, f"no_mapping_data_for_cluster={cluster_id}")
                out["propagation_method"] = "failed"
                rows.append(out)
                retry_rows.append(_retry_output(row, out))
                methods[out["propagation_method"]] += 1
                errors[out["dripper_error"][:160]] += 1
                continue

            parser_cache: dict[str, Any] = {}
            prop_cfg = self._propagation_config(parser_cache)
            trust_cfg = self._static_config(parser_cache, static_trust)
            use_static = _cluster_static_trustworthy(
                cluster_id,
                siblings_by_cluster.get(cluster_id, [])[:3],
                mapping_data,
                trust_cfg,
            )
            row_t0 = time.perf_counter()
            main_html, content, error, method = _sibling_propagate(row, mapping_data, use_static, prop_cfg)
            out["dripper_html"] = main_html
            out["dripper_content"] = content
            out["dripper_error"] = error
            out["dripper_time_s"] = time.perf_counter() - row_t0
            out["propagation_success"] = bool(content.strip()) and not error
            out["propagation_method"] = method if out["propagation_success"] else "failed"
            rows.append(out)
            methods[out["propagation_method"]] += 1
            if out["propagation_success"]:
                success_count += 1
            else:
                retry_rows.append(_retry_output(row, out))
                errors[(error or "empty_content")[:160]] += 1

        _write_output(output_path, rows, OUTPUT_SCHEMA)
        _write_output(retry_output_path, retry_rows, RETRY_SCHEMA)
        elapsed = time.perf_counter() - t0
        metrics: dict[str, Any] = {
            "bucket_label": label,
            "manifest_path": str(manifest_path),
            "template_path": str(template_path),
            "output_path": str(output_path),
            "retry_output_path": str(retry_output_path),
            "input_rows": int(len(manifest_df)),
            "template_rows": int(len(template_df)),
            "output_rows": int(len(rows)),
            "retry_rows": int(len(retry_rows)),
            "success_rows": int(success_count),
            "error_rows": int(len(rows) - success_count),
            "mapping_clusters": int(len(mapping_by_cluster)),
            "elapsed_s": round(elapsed, 3),
        }
        metrics.update({f"role_{key}": int(value) for key, value in role_counts.items()})
        metrics.update({f"method_{key}": int(value) for key, value in methods.items()})
        for i, (error, count) in enumerate(errors.most_common(5), start=1):
            metrics[f"top_error_{i}"] = error
            metrics[f"top_error_{i}_count"] = int(count)

        logger.info(
            "stage3 bucket={} rows={} success={} retry={} methods={} elapsed={:.1f}s -> {}",
            label,
            len(rows),
            success_count,
            len(retry_rows),
            dict(methods),
            elapsed,
            output_path,
        )
        return DocumentBatch(dataset_name=task.dataset_name, data=pd.DataFrame([metrics]))


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _discover_tasks(Path(args.cluster_manifest), Path(args.inference_results), output_dir)
    if args.shard_index or args.num_shards != 1:
        tasks = tasks[len(tasks) * args.shard_index // args.num_shards : len(tasks) * (args.shard_index + 1) // args.num_shards]
    if not tasks:
        raise RuntimeError(f"No Stage 3 tasks for shard {args.shard_index}/{args.num_shards}")

    logger.info("Stage 3 scheduling {} host bucket task(s)", len(tasks))
    _init_ray_from_slurm()
    stage = BucketPropagationStage(task_cpus=args.task_cpus)
    pipeline = Pipeline(name="stage3_bucket_propagation")
    pipeline.add_stage(stage)
    t0 = time.perf_counter()
    result_tasks = pipeline.run(executor=RayDataExecutor(), initial_tasks=tasks) or []
    elapsed = time.perf_counter() - t0

    metrics: list[dict[str, Any]] = []
    for task in result_tasks:
        if hasattr(task, "to_pandas"):
            metrics.extend(task.to_pandas().to_dict("records"))

    summary: dict[str, Any] = {
        "cluster_manifest": str(Path(args.cluster_manifest)),
        "inference_results": str(Path(args.inference_results)),
        "output": str(output_dir),
        "elapsed_s": round(elapsed, 3),
        "bucket_tasks": len(tasks),
        "completed_buckets": len(metrics),
        "input_rows": int(sum(item.get("input_rows", 0) for item in metrics)),
        "template_rows": int(sum(item.get("template_rows", 0) for item in metrics)),
        "output_rows": int(sum(item.get("output_rows", 0) for item in metrics)),
        "retry_rows": int(sum(item.get("retry_rows", 0) for item in metrics)),
        "success_rows": int(sum(item.get("success_rows", 0) for item in metrics)),
        "error_rows": int(sum(item.get("error_rows", 0) for item in metrics)),
        "buckets": metrics,
    }
    counter_prefixes = ("role_", "method_")
    for prefix in counter_prefixes:
        keys = sorted({key for item in metrics for key in item if key.startswith(prefix)})
        for key in keys:
            summary[key] = int(sum(item.get(key, 0) for item in metrics))

    summary_path = output_dir / "_stage3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Stage 3 done in {:.1f}s rows={} success={} retry={} errors={} summary={}",
        elapsed,
        summary["output_rows"],
        summary["success_rows"],
        summary["retry_rows"],
        summary["error_rows"],
        summary_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cluster-manifest", required=True, help="Stage 1b host-bucket output directory")
    parser.add_argument("--inference-results", required=True, help="Stage 2c host-bucket template directory")
    parser.add_argument("--output-dir", required=True, help="Stage 3 output directory")
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
