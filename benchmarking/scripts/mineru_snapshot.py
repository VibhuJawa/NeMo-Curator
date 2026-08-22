# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Snapshot-level Parquet validation for the native Common Crawl pipeline."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.backends.slurm_array import find_slurm_array_retries

if TYPE_CHECKING:
    from pathlib import Path

MIN_SUBSTANTIVE_CHARS = 200


def scan_parquet(
    paths: list[Path],
    *,
    url_field: str,
    required_fields: set[str],
    text_field: str | None = None,
    status_field: str | None = None,
) -> dict[str, Any]:
    if not paths:
        msg = "no Parquet files found"
        raise ValueError(msg)
    rows = with_text = substantive = total_chars = 0
    statuses: dict[str, int] = {}
    for path in paths:
        pf = pq.ParquetFile(path)
        missing = required_fields - set(pf.schema_arrow.names)
        if missing:
            msg = f"{path} is missing fields: {', '.join(sorted(missing))}"
            raise ValueError(msg)
        columns = [url_field]
        if text_field:
            columns.append(text_field)
        if status_field:
            columns.append(status_field)
        for batch in pf.iter_batches(batch_size=8192, columns=columns):
            rows += batch.num_rows
            if text_field:
                lengths = pc.fill_null(pc.utf8_length(batch.column(text_field)), 0)
                with_text += int(pc.sum(pc.cast(pc.greater(lengths, 0), "int64")).as_py() or 0)
                substantive += int(
                    pc.sum(pc.cast(pc.greater_equal(lengths, MIN_SUBSTANTIVE_CHARS), "int64")).as_py() or 0
                )
                total_chars += int(pc.sum(lengths).as_py() or 0)
            if status_field:
                for item in pc.value_counts(batch.column(status_field)).to_pylist():
                    key = str(item["values"] if item["values"] is not None else "unknown")
                    statuses[key] = statuses.get(key, 0) + int(item["counts"])
    return {
        "num_files": len(paths),
        "num_rows": rows,
        "num_documents_with_text": with_text,
        "num_documents_substantive": substantive,
        "total_text_chars": total_chars,
        "status_counts": statuses,
    }


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o644)
    os.replace(temporary, path)


def _sample_paths(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    return [paths[index * len(paths) // limit] for index in range(limit)]


def _parquet_rows(path: Path, required: set[str]) -> int:
    parquet = pq.ParquetFile(path)
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        msg = f"missing fields: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    return parquet.metadata.num_rows


def verify_native_snapshot(  # noqa: C901, PLR0913
    *,
    output_path: Path,
    checkpoint_path: Path,
    success_path: Path,
    expected_num_warcs: int,
    url_field: str,
    text_field: str,
    status_field: str,
    min_status_ok_rate: float,
    min_nonempty_rate: float,
    max_convert_error_rate: float,
    quality_sample_files: int = 1024,
) -> dict[str, Any]:
    """Verify array completeness, every Parquet footer, and a quality sample."""
    errors = []
    retry_plan = find_slurm_array_retries(checkpoint_path)
    if retry_plan is None:
        errors.append("checkpoint has no Slurm-array run configuration")
        missing_shards: list[int] = []
    else:
        missing_shards = list(retry_plan.shard_indices)
        if missing_shards:
            errors.append(f"{len(missing_shards)} Slurm-array shards are incomplete")

    paths = sorted(output_path.rglob("*.parquet"))
    if len(paths) < expected_num_warcs:
        errors.append(f"output files {len(paths)} < expected WARC files {expected_num_warcs}")

    total_rows = 0
    required = {url_field, text_field, status_field}
    invalid_files: dict[str, str] = {}
    for path in paths:
        try:
            total_rows += _parquet_rows(path, required)
        except (OSError, ValueError) as e:
            invalid_files[str(path)] = str(e)
    if invalid_files:
        errors.append(f"{len(invalid_files)} Parquet files have invalid footer/schema")

    sample = _sample_paths(paths, quality_sample_files)
    quality = (
        scan_parquet(
            sample,
            url_field=url_field,
            text_field=text_field,
            status_field=status_field,
            required_fields=required,
        )
        if sample
        else {"num_rows": 0, "status_counts": {}, "num_documents_with_text": 0}
    )
    sampled_rows = quality["num_rows"]
    counts = quality["status_counts"]
    rates = {
        "sampled_status_ok_rate": counts.get("ok", 0) / sampled_rows if sampled_rows else 0.0,
        "sampled_nonempty_rate": quality["num_documents_with_text"] / sampled_rows if sampled_rows else 0.0,
        "sampled_convert_error_rate": counts.get("convert_error", 0) / sampled_rows if sampled_rows else 0.0,
    }
    if rates["sampled_status_ok_rate"] < min_status_ok_rate:
        errors.append(f"sampled status_ok_rate {rates['sampled_status_ok_rate']:.6f} < {min_status_ok_rate}")
    if rates["sampled_nonempty_rate"] < min_nonempty_rate:
        errors.append(f"sampled nonempty_rate {rates['sampled_nonempty_rate']:.6f} < {min_nonempty_rate}")
    if rates["sampled_convert_error_rate"] > max_convert_error_rate:
        errors.append(
            f"sampled convert_error_rate {rates['sampled_convert_error_rate']:.6f} > {max_convert_error_rate}"
        )

    result = {
        "verification_passed": not errors,
        "expected_num_warcs": expected_num_warcs,
        "num_output_files": len(paths),
        "num_documents_processed": total_rows,
        "num_quality_sample_files": len(sample),
        "num_quality_sample_rows": sampled_rows,
        "missing_shard_indices": missing_shards,
        "invalid_files": invalid_files,
        "verification_errors": errors,
        **rates,
    }
    if not errors:
        _atomic_json(success_path, result | {"created_at": datetime.now(UTC).isoformat()})
    return result
