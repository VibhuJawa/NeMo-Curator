# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Manifest, validation, and atomic-publish helpers for snapshot extraction."""

# ruff: noqa: EM101, EM102

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

SUCCESS_FILE = "_SUCCESS.json"
MIN_SUBSTANTIVE_CHARS = 200
_MODULUS = 1 << 128


@dataclass(frozen=True)
class WorkUnit:
    snapshot_id: str
    work_unit_id: str
    input_paths: tuple[str, ...]
    output_path: str
    expected_rows: int
    expected_input_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkUnit:
        required = {"snapshot_id", "work_unit_id", "input_paths", "output_path", "expected_rows"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"work unit is missing: {', '.join(sorted(missing))}")
        unit = cls(
            snapshot_id=str(data["snapshot_id"]),
            work_unit_id=str(data["work_unit_id"]),
            input_paths=tuple(map(str, data["input_paths"])),
            output_path=str(data["output_path"]),
            expected_rows=int(data["expected_rows"]),
            expected_input_bytes=(
                int(data["expected_input_bytes"]) if data.get("expected_input_bytes") is not None else None
            ),
        )
        if not unit.snapshot_id or not unit.work_unit_id or not unit.input_paths or unit.expected_rows < 1:
            raise ValueError("snapshot_id, work_unit_id, input_paths, and positive expected_rows are required")
        paths = (*unit.input_paths, unit.output_path)
        if any("://" in path or not Path(path).is_absolute() for path in paths):
            raise ValueError("snapshot manifests currently require absolute local filesystem paths")
        return unit


def load_manifest(path: str | Path) -> tuple[list[WorkUnit], str]:
    manifest = Path(path).resolve()
    raw = manifest.read_bytes()
    units = [WorkUnit.from_dict(json.loads(line)) for line in raw.splitlines() if line.strip()]
    if not units:
        raise ValueError(f"manifest has no work units: {manifest}")
    snapshots = {unit.snapshot_id for unit in units}
    ids = [unit.work_unit_id for unit in units]
    outputs = [unit.output_path for unit in units]
    inputs = [path for unit in units for path in unit.input_paths]
    if len(snapshots) != 1:
        raise ValueError("one manifest must contain exactly one snapshot_id")
    for label, values in (("work_unit_id", ids), ("output_path", outputs), ("input path", inputs)):
        if len(values) != len(set(values)):
            raise ValueError(f"manifest contains a duplicate {label}")
    return units, hashlib.sha256(raw).hexdigest()


def select_work_unit(path: str | Path, index: int) -> tuple[WorkUnit, str]:
    units, digest = load_manifest(path)
    if index < 0 or index >= len(units):
        raise IndexError(f"work-unit index {index} is outside [0, {len(units)})")
    return units[index], digest


class _MultisetFingerprint:
    """Order-independent, multiplicity-sensitive 384-bit digest accumulator."""

    def __init__(self) -> None:
        self.count = self.xor = self.total = self.squares = 0

    def add(self, value: Any) -> None:  # noqa: ANN401
        if value is None:
            raw = b"n"
        elif isinstance(value, bytes):
            raw = b"b" + value
        else:
            raw = b"s" + str(value).encode("utf-8", errors="surrogatepass")
        item = int.from_bytes(hashlib.blake2b(raw, digest_size=16).digest(), "little")
        self.count += 1
        self.xor ^= item
        self.total = (self.total + item) % _MODULUS
        self.squares = (self.squares + item * item) % _MODULUS

    def value(self) -> str:
        return f"{self.count}:{self.xor:032x}:{self.total:032x}:{self.squares:032x}"


def scan_parquet(  # noqa: C901
    paths: list[Path],
    *,
    url_field: str,
    required_fields: set[str],
    text_field: str | None = None,
    status_field: str | None = None,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("no Parquet files found")
    rows = with_text = substantive = total_chars = 0
    statuses: dict[str, int] = {}
    fingerprint = _MultisetFingerprint()
    for path in paths:
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        missing = required_fields - names
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
        columns = [url_field]
        if text_field:
            columns.append(text_field)
        if status_field:
            columns.append(status_field)
        for batch in pf.iter_batches(batch_size=8192, columns=columns):
            rows += batch.num_rows
            for value in batch.column(url_field).to_pylist():
                fingerprint.add(value)
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
        "url_multiset_fingerprint": fingerprint.value(),
        "num_documents_with_text": with_text,
        "num_documents_substantive": substantive,
        "total_text_chars": total_chars,
        "status_counts": statuses,
    }


def preflight_input(unit: WorkUnit, *, html_field: str, url_field: str) -> dict[str, Any]:
    paths = [Path(path) for path in unit.input_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input Parquet files: {missing[:5]}")
    stats = scan_parquet(paths, url_field=url_field, required_fields={html_field, url_field})
    stats["total_bytes"] = sum(path.stat().st_size for path in paths)
    if stats["num_rows"] != unit.expected_rows:
        raise ValueError(f"input row count {stats['num_rows']} != manifest {unit.expected_rows}")
    if unit.expected_input_bytes is not None and stats["total_bytes"] != unit.expected_input_bytes:
        raise ValueError(f"input bytes {stats['total_bytes']} != manifest {unit.expected_input_bytes}")
    return stats


def validate_output(  # noqa: PLR0913
    unit: WorkUnit,
    output_dir: Path,
    input_stats: dict[str, Any],
    *,
    url_field: str,
    text_field: str,
    status_field: str,
    min_status_ok_rate: float,
    min_nonempty_rate: float,
    max_convert_error_rate: float,
) -> dict[str, Any]:
    stats = scan_parquet(
        sorted(output_dir.rglob("*.parquet")),
        url_field=url_field,
        text_field=text_field,
        status_field=status_field,
        required_fields={url_field, text_field, status_field},
    )
    rows = stats["num_rows"]
    counts = stats["status_counts"]
    rates = {
        "status_ok_rate": counts.get("ok", 0) / rows if rows else 0.0,
        "nonempty_rate": stats["num_documents_with_text"] / rows if rows else 0.0,
        "extraction_rate": stats["num_documents_substantive"] / rows if rows else 0.0,
        "convert_error_rate": counts.get("convert_error", 0) / rows if rows else 0.0,
    }
    errors = []
    if rows != unit.expected_rows:
        errors.append(f"output rows {rows} != expected {unit.expected_rows}")
    if stats["url_multiset_fingerprint"] != input_stats["url_multiset_fingerprint"]:
        errors.append("output URL multiset does not match input")
    if rates["status_ok_rate"] < min_status_ok_rate:
        errors.append(f"status_ok_rate {rates['status_ok_rate']:.6f} < {min_status_ok_rate}")
    if rates["nonempty_rate"] < min_nonempty_rate:
        errors.append(f"nonempty_rate {rates['nonempty_rate']:.6f} < {min_nonempty_rate}")
    if rates["convert_error_rate"] > max_convert_error_rate:
        errors.append(f"convert_error_rate {rates['convert_error_rate']:.6f} > {max_convert_error_rate}")
    return {**stats, **rates, "verification_errors": errors, "verification_passed": not errors}


def read_published_result(unit: WorkUnit, manifest_sha256: str) -> dict[str, Any] | None:
    output = Path(unit.output_path)
    if not output.exists():
        return None
    marker = output / SUCCESS_FILE
    if not marker.is_file():
        raise FileExistsError(f"output exists without {SUCCESS_FILE}: {output}")
    data = json.loads(marker.read_text())
    if (
        data.get("manifest_sha256") != manifest_sha256
        or WorkUnit.from_dict(data.get("work_unit", {})) != unit
        or not data.get("validation", {}).get("verification_passed")
    ):
        raise ValueError(f"published result does not match this manifest: {marker}")
    return data


def new_attempt_directory(unit: WorkUnit) -> Path:
    output = Path(unit.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = os.getenv("SLURM_JOB_ID", "local") + "-" + uuid.uuid4().hex[:10]
    return output.parent / f".{output.name}.attempt-{identity}"


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o644)
    os.replace(temporary, path)


def _make_world_readable(root: Path) -> None:
    root.chmod(0o755)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def publish_attempt(
    unit: WorkUnit,
    attempt: Path,
    manifest_sha256: str,
    validation: dict[str, Any],
) -> dict[str, Any]:
    if not validation.get("verification_passed"):
        raise ValueError("refusing to publish an output that failed validation")
    output = Path(unit.output_path)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    marker = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": manifest_sha256,
        "work_unit": asdict(unit),
        "validation": validation,
        "slurm": {
            key: os.getenv(key)
            for key in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID", "SLURMD_NODENAME")
            if os.getenv(key)
        },
    }
    _atomic_json(attempt / SUCCESS_FILE, marker)
    _make_world_readable(attempt)
    os.replace(attempt, output)
    return marker


def verify_snapshot(manifest_path: str | Path, success_path: str | Path) -> dict[str, Any]:
    units, digest = load_manifest(manifest_path)
    missing: list[str] = []
    invalid: dict[str, str] = {}
    rows = 0
    for unit in units:
        try:
            marker = read_published_result(unit, digest)
            if marker is None:
                missing.append(unit.work_unit_id)
            else:
                output = Path(unit.output_path)
                parquet_files = sorted(output.rglob("*.parquet"))
                footer_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)
                expected = marker["validation"]
                if len(parquet_files) != int(expected["num_files"]) or footer_rows != int(expected["num_rows"]):
                    invalid[unit.work_unit_id] = "published Parquet footers no longer match the success record"
                    continue
                rows += footer_rows
        except (OSError, ValueError) as e:
            invalid[unit.work_unit_id] = str(e)
    expected_rows = sum(unit.expected_rows for unit in units)
    passed = not missing and not invalid and rows == expected_rows
    result = {
        "verification_passed": passed,
        "snapshot_id": units[0].snapshot_id,
        "manifest_sha256": digest,
        "num_work_units": len(units),
        "num_missing_work_units": len(missing),
        "num_invalid_work_units": len(invalid),
        "num_documents_processed": rows,
        "expected_rows": expected_rows,
        "missing_work_unit_ids": missing,
        "invalid_work_units": invalid,
    }
    if passed:
        _atomic_json(Path(success_path), result | {"created_at": datetime.now(UTC).isoformat()})
    return result
