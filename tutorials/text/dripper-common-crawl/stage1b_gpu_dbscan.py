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

"""Stage 1b: streaming GPU DBSCAN over host-bucket parquet files.

The driver schedules small bucket metadata tasks. Each GPU actor opens the
bucket parquet file(s), streams rows grouped by url_host_name, clusters one
host at a time, and writes its own bucket output file.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
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
from nemo_curator.stages.text.experimental.dripper._html_compression import HTML_CHARS_COL, HTML_ZLIB_COL
from nemo_curator.tasks import DocumentBatch, FileGroupTask

OUTPUT_COLS = [
    "snapshot",
    "record_id",
    "url",
    "url_host_name",
    "host_hash64",
    "host_bucket",
    "host_bucket_label",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
    "cluster_id",
    "cluster_role",
    "layout_cluster_id",
    "is_representative",
    "cluster_size",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "source_manifest_file",
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("snapshot", pa.string()),
        ("record_id", pa.string()),
        ("url", pa.string()),
        ("url_host_name", pa.string()),
        ("host_hash64", pa.string()),
        ("host_bucket", pa.int64()),
        ("host_bucket_label", pa.string()),
        (HTML_ZLIB_COL, pa.binary()),
        (HTML_CHARS_COL, pa.int64()),
        ("cluster_id", pa.string()),
        ("cluster_role", pa.string()),
        ("layout_cluster_id", pa.string()),
        ("is_representative", pa.bool_()),
        ("cluster_size", pa.int64()),
        ("warc_filename", pa.string()),
        ("warc_record_offset", pa.int64()),
        ("warc_record_length", pa.int64()),
        ("source_manifest_file", pa.string()),
    ]
)

REQUIRED_INPUT_COLS = [
    "record_id",
    "url",
    "url_host_name",
    "host_bucket",
    "host_bucket_label",
    "dom_feature",
    HTML_ZLIB_COL,
    HTML_CHARS_COL,
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]
OPTIONAL_INPUT_COLS = ["snapshot", "host_hash64", "source_manifest_file"]
READ_COLS = [*OPTIONAL_INPUT_COLS, *REQUIRED_INPUT_COLS]
BUCKET_FILE_RE = re.compile(r"^host_bucket_(?P<label>\d+)(?:[_.-].*)?\.parquet$")


def _empty_output_table() -> pa.Table:
    arrays = [pa.array([], type=field.type) for field in OUTPUT_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=OUTPUT_SCHEMA)


def _slurm_gpu_count() -> int:
    for name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_GPUS"):
        raw = os.environ.get(name, "")
        match = re.search(r"\d+", raw)
        if match:
            return int(match.group(0))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and visible not in {"NoDevFiles", "-1"}:
        return len([item for item in visible.split(",") if item.strip()])
    return 0


def _init_ray_from_slurm() -> None:
    if ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
        return
    ray_kwargs: dict[str, object] = {"ignore_reinit_error": True}
    if os.environ.get("RAY_TMPDIR"):
        ray_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        ray_kwargs["num_cpus"] = int(os.environ["SLURM_CPUS_PER_TASK"])
    gpus = _slurm_gpu_count()
    if gpus:
        ray_kwargs["num_gpus"] = gpus
    ray.init(**ray_kwargs)


def _bucket_label_from_path(path: Path) -> str:
    match = BUCKET_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"{path} is not a Stage 1a host-bucket parquet file")
    return match.group("label")


def _discover_bucket_inputs(input_path: Path, output_dir: Path) -> list[FileGroupTask]:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("host_bucket_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No host_bucket_*.parquet files found in {input_path}. "
            "Run Stage 1a with host-bucket output before Stage 1b."
        )

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        label = _bucket_label_from_path(path)
        grouped[label].append(path)

    tasks: list[FileGroupTask] = []
    for label, paths in sorted(grouped.items()):
        input_rows = 0
        for path in paths:
            input_rows += pq.read_metadata(path).num_rows
        tasks.append(
            FileGroupTask(
                dataset_name=f"host_bucket_{label}",
                data=[str(path) for path in sorted(paths)],
                reader_config={
                    "bucket_label": label,
                    "output_path": str(output_dir / f"host_bucket_{label}.parquet"),
                    "input_rows": input_rows,
                },
            )
        )
    return tasks


def _record_id(src: dict[str, Any]) -> str:
    parts = [src.get("warc_filename"), src.get("warc_record_offset"), src.get("warc_record_length")]
    if all(part is not None and str(part) for part in parts):
        return "|".join(str(part) for part in parts)
    return str(src.get("record_id") or src.get("url") or "")


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _representative_index(members: list[dict[str, Any]]) -> int:
    sizes: list[tuple[int, int]] = []
    for i, member in enumerate(members):
        sizes.append((i, _as_int(member.get(HTML_CHARS_COL))))
    median_size = sorted(size for _, size in sizes)[len(sizes) // 2]
    return min(sizes, key=lambda item: (abs(item[1] - median_size), item[0]))[0]


def _output_row(
    src: dict[str, Any],
    *,
    host: str,
    cluster_id: str,
    role: str,
    is_representative: bool,
    cluster_size: int,
) -> dict[str, Any]:
    return {
        "snapshot": _as_str_or_none(src.get("snapshot")),
        "record_id": _record_id(src),
        "url": str(src.get("url") or ""),
        "url_host_name": host,
        "host_hash64": _as_str_or_none(src.get("host_hash64")),
        "host_bucket": _as_int(src.get("host_bucket")),
        "host_bucket_label": str(src.get("host_bucket_label") or ""),
        HTML_ZLIB_COL: src.get(HTML_ZLIB_COL),
        HTML_CHARS_COL: _as_int(src.get(HTML_CHARS_COL)),
        "cluster_id": cluster_id,
        "cluster_role": role,
        "layout_cluster_id": cluster_id,
        "is_representative": is_representative,
        "cluster_size": cluster_size,
        "warc_filename": _as_str_or_none(src.get("warc_filename")),
        "warc_record_offset": _as_int(src.get("warc_record_offset")),
        "warc_record_length": _as_int(src.get("warc_record_length")),
        "source_manifest_file": _as_str_or_none(src.get("source_manifest_file")),
    }


def _singleton_row(src: dict[str, Any], host: str) -> dict[str, Any]:
    key = _record_id(src) or str(src.get("url") or "")
    digest = hashlib.blake2b(f"{host}\0{key}".encode("utf-8", errors="replace"), digest_size=8).hexdigest()
    cluster_id = f"{host}:singleton_{digest}"
    return _output_row(src, host=host, cluster_id=cluster_id, role="singleton", is_representative=True, cluster_size=1)


def _iter_records(batch: pa.RecordBatch) -> list[dict[str, Any]]:
    columns = batch.to_pydict()
    names = list(columns)
    return [{name: columns[name][i] for name in names} for i in range(batch.num_rows)]


@dataclass(kw_only=True)
class HostBucketDBSCANStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Cluster one host bucket per task, streaming host groups inside the actor."""

    name: str = "host_bucket_dbscan"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))
    threshold: float = 0.95
    min_cluster_size: int = 2
    max_host_size: int = 3000
    read_batch_size: int = 8192
    _cluster_gpu: Any = field(init=False, repr=False, default=None)

    def setup(self, _worker_metadata: object = None) -> None:
        faulthandler.enable(all_threads=True)
        import cuml.cluster  # noqa: F401
        import cupy as cp

        from nemo_curator.stages.text.experimental.dripper.gpu_layout_clustering import cluster_html_struct_gpu

        _ = cp.cuda.Device(0).compute_capability
        self._cluster_gpu = cluster_html_struct_gpu
        logger.info(
            "actor setup: CUDA_VISIBLE_DEVICES={} threshold={} max_host_size={}",
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            self.threshold,
            self.max_host_size,
        )

    def process(self, task: FileGroupTask) -> DocumentBatch:
        bucket_label = str(task.reader_config["bucket_label"])
        input_paths = [Path(path) for path in task.data]
        output_path = Path(task.reader_config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(f".tmp_{os.getpid()}.parquet")
        tmp.unlink(missing_ok=True)

        metrics: dict[str, Any] = {
            "bucket_label": bucket_label,
            "input_paths": ",".join(str(path) for path in input_paths),
            "output_path": str(output_path),
            "input_rows": 0,
            "output_rows": 0,
            "hosts": 0,
            "clusters": 0,
            "representatives": 0,
            "siblings": 0,
            "singletons": 0,
            "empty_feature_rows": 0,
            "invalid_feature_rows": 0,
            "max_host_rows": 0,
        }
        t0 = time.perf_counter()
        writer: pq.ParquetWriter | None = None

        def write_rows(rows: list[dict[str, Any]]) -> None:
            nonlocal writer
            if not rows:
                return
            table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), OUTPUT_SCHEMA, compression="zstd")
            writer.write_table(table)
            metrics["output_rows"] += len(rows)

        current_host: str | None = None
        current_rows: list[dict[str, Any]] = []
        closed_hosts: set[str] = set()

        try:
            for path in input_paths:
                logger.info("bucket {} reading {}", bucket_label, path)
                pf = pq.ParquetFile(path)
                schema_names = set(pf.schema_arrow.names)
                missing = [col for col in REQUIRED_INPUT_COLS if col not in schema_names]
                if missing:
                    raise ValueError(f"{path} is missing required Stage 1b input columns: {missing}")
                cols = [col for col in READ_COLS if col in schema_names]
                for batch in pf.iter_batches(batch_size=self.read_batch_size, columns=cols):
                    for rec in _iter_records(batch):
                        metrics["input_rows"] += 1
                        host = str(rec.get("url_host_name") or "")
                        if not host:
                            raise ValueError(f"{path} has an empty url_host_name in bucket {bucket_label}")
                        rec_label = str(rec.get("host_bucket_label") or "")
                        if rec_label and rec_label != bucket_label:
                            raise ValueError(f"{path} contains host_bucket_label={rec_label}, expected {bucket_label}")
                        if current_host is None:
                            current_host = host
                        elif host != current_host:
                            write_rows(self._cluster_host(current_host, current_rows, metrics))
                            closed_hosts.add(current_host)
                            if host in closed_hosts:
                                raise ValueError(
                                    f"{path} is not grouped by url_host_name: host {host!r} reappeared"
                                )
                            current_host = host
                            current_rows = []
                        current_rows.append(rec)

            if current_host is not None:
                write_rows(self._cluster_host(current_host, current_rows, metrics))
            if writer is None:
                pq.write_table(_empty_output_table(), str(tmp), compression="zstd")
            else:
                writer.close()
                writer = None
            tmp.rename(output_path)
        finally:
            if writer is not None:
                writer.close()
            tmp.unlink(missing_ok=True)

        metrics["elapsed_s"] = round(time.perf_counter() - t0, 3)
        logger.info(
            "bucket {} done rows={} hosts={} clusters={} reps={} singletons={} elapsed={:.1f}s",
            bucket_label,
            metrics["output_rows"],
            metrics["hosts"],
            metrics["clusters"],
            metrics["representatives"],
            metrics["singletons"],
            metrics["elapsed_s"],
        )
        return DocumentBatch(dataset_name=f"host_bucket_{bucket_label}", data=pd.DataFrame([metrics]))

    def _cluster_host(self, host: str, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> list[dict[str, Any]]:
        metrics["hosts"] += 1
        metrics["max_host_rows"] = max(metrics["max_host_rows"], len(rows))
        valid_samples: list[dict[str, Any]] = []
        output_rows: list[dict[str, Any]] = []

        for rec in rows:
            feature_json = str(rec.get("dom_feature") or "")
            if not feature_json:
                output_rows.append(_singleton_row(rec, host))
                metrics["empty_feature_rows"] += 1
                metrics["singletons"] += 1
                continue
            try:
                feature = json.loads(feature_json)
            except json.JSONDecodeError:
                output_rows.append(_singleton_row(rec, host))
                metrics["invalid_feature_rows"] += 1
                metrics["singletons"] += 1
                continue
            if (
                not isinstance(feature, dict)
                or not isinstance(feature.get("tags"), dict)
                or not isinstance(feature.get("attrs"), dict)
            ):
                output_rows.append(_singleton_row(rec, host))
                metrics["invalid_feature_rows"] += 1
                metrics["singletons"] += 1
                continue
            sample = dict(rec)
            sample["track_id"] = rec["url"]
            sample["feature"] = feature
            valid_samples.append(sample)

        if len(valid_samples) < self.min_cluster_size:
            for sample in valid_samples:
                output_rows.append(_singleton_row(sample, host))
                metrics["singletons"] += 1
            return output_rows

        clustered: list[dict[str, Any]] = []
        for chunk_idx, start in enumerate(range(0, len(valid_samples), self.max_host_size)):
            chunk = [dict(sample) for sample in valid_samples[start : start + self.max_host_size]]
            cc, _ = self._cluster_gpu(chunk, threshold=self.threshold)
            if chunk_idx:
                for sample in cc:
                    layout_id = int(sample.get("layout_id", -1))
                    if layout_id >= 0:
                        sample["layout_id"] = chunk_idx * 100_000 + layout_id
            clustered.extend(cc)

        by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in clustered:
            by_layout[int(sample.get("layout_id", -1))].append(sample)

        for layout_id, members in by_layout.items():
            if layout_id < 0 or len(members) < self.min_cluster_size:
                for member in members:
                    output_rows.append(_singleton_row(member, host))
                    metrics["singletons"] += 1
                continue

            cluster_id = f"{host}:cluster_{layout_id}"
            rep_idx = _representative_index(members)
            metrics["clusters"] += 1
            for i, member in enumerate(members):
                is_rep = i == rep_idx
                role = "representative" if is_rep else "sibling"
                output_rows.append(
                    _output_row(
                        member,
                        host=host,
                        cluster_id=cluster_id,
                        role=role,
                        is_representative=is_rep,
                        cluster_size=len(members),
                    )
                )
                metrics["representatives" if is_rep else "siblings"] += 1
        return output_rows


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _discover_bucket_inputs(input_path, output_dir)
    if args.shard_index or args.num_shards != 1:
        tasks = tasks[len(tasks) * args.shard_index // args.num_shards : len(tasks) * (args.shard_index + 1) // args.num_shards]
    if not tasks:
        raise RuntimeError(f"No Stage 1b bucket tasks for shard {args.shard_index}/{args.num_shards}")

    logger.info(
        "Stage 1b scheduling {} host-bucket task(s), {:,} input rows",
        len(tasks),
        sum(task.reader_config.get("input_rows", 0) for task in tasks),
    )
    stage = HostBucketDBSCANStage(
        threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
        max_host_size=args.max_host_size,
        read_batch_size=args.read_batch_size,
        resources=Resources(cpus=float(args.cpus_per_actor), gpus=1.0),
    )
    pipeline = Pipeline(name="stage1b_dbscan")
    pipeline.add_stage(stage)
    _init_ray_from_slurm()
    t0 = time.perf_counter()
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []
    elapsed = time.perf_counter() - t0
    metrics = []
    for task in result_tasks:
        if hasattr(task, "to_pandas"):
            metrics.extend(task.to_pandas().to_dict("records"))
    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "elapsed_s": round(elapsed, 3),
        "bucket_tasks": len(tasks),
        "completed_buckets": len(metrics),
        "input_rows": int(sum(item.get("input_rows", 0) for item in metrics)),
        "output_rows": int(sum(item.get("output_rows", 0) for item in metrics)),
        "hosts": int(sum(item.get("hosts", 0) for item in metrics)),
        "clusters": int(sum(item.get("clusters", 0) for item in metrics)),
        "representatives": int(sum(item.get("representatives", 0) for item in metrics)),
        "siblings": int(sum(item.get("siblings", 0) for item in metrics)),
        "singletons": int(sum(item.get("singletons", 0) for item in metrics)),
        "empty_feature_rows": int(sum(item.get("empty_feature_rows", 0) for item in metrics)),
        "invalid_feature_rows": int(sum(item.get("invalid_feature_rows", 0) for item in metrics)),
        "max_host_rows": int(max((item.get("max_host_rows", 0) for item in metrics), default=0)),
        "buckets": metrics,
    }
    summary_path = output_dir / "_stage1b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Stage 1b done in {:.1f}s rows={} clusters={} reps={} singletons={} summary={}",
        elapsed,
        summary["output_rows"],
        summary["clusters"],
        summary["representatives"],
        summary["singletons"],
        summary_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Stage 1a output containing host_bucket_*.parquet files")
    parser.add_argument("--output", required=True, help="Output directory for Stage 1b host_bucket_*.parquet files")
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-host-size", type=int, default=int(os.environ.get("STAGE1B_MAX_HOST_SIZE", "3000")))
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--cpus-per-actor", type=int, default=4)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
