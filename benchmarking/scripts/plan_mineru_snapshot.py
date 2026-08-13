# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Create stable, file-aligned MinerU work units for one Common Crawl snapshot."""

# ruff: noqa: EM102

from __future__ import annotations

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq


def inspect(path: Path, required_fields: set[str]) -> tuple[Path, int, int]:
    parquet = pq.ParquetFile(path)
    missing = required_fields - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return path, parquet.metadata.num_rows, path.stat().st_size


def plan(  # noqa: PLR0913
    input_path: Path,
    output_root: Path,
    *,
    snapshot_id: str,
    target_rows: int,
    workers: int,
    required_fields: set[str],
) -> list[dict]:
    paths = sorted(input_path.rglob("*.parquet")) if input_path.is_dir() else [input_path]
    if not paths:
        raise ValueError(f"no Parquet files found under {input_path}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda path: inspect(path, required_fields), paths))
    oversized = [(path, rows) for path, rows, _ in records if rows > target_rows]
    if oversized:
        examples = ", ".join(f"{path} ({rows:,} rows)" for path, rows in oversized[:3])
        raise ValueError(f"input files exceed --target-rows; repartition them first: {examples}")

    groups: list[list[tuple[Path, int, int]]] = []
    current: list[tuple[Path, int, int]] = []
    current_rows = 0
    for record in records:
        if current and current_rows + record[1] > target_rows:
            groups.append(current)
            current, current_rows = [], 0
        current.append(record)
        current_rows += record[1]
    if current:
        groups.append(current)

    units = []
    for index, group in enumerate(groups):
        unit_id = f"{index:06d}"
        units.append(
            {
                "snapshot_id": snapshot_id,
                "work_unit_id": unit_id,
                "input_paths": [str(path.resolve()) for path, _, _ in group],
                "output_path": str((output_root / f"work-unit-{unit_id}").resolve()),
                "expected_rows": sum(rows for _, rows, _ in group),
                "expected_input_bytes": sum(size for _, _, size in group),
            }
        )
    return units


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument(
        "--target-rows",
        type=int,
        default=1_800_000,
        help="Rows per work unit. 1.8M targets about 3h at the measured 1M-corpus rate.",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--html-field", default="content")
    parser.add_argument("--url-field", default="url")
    args = parser.parse_args()
    if args.target_rows < 1 or args.workers < 1:
        parser.error("--target-rows and --workers must be positive")
    manifest = args.manifest_path.resolve()
    if manifest.exists():
        parser.error(f"refusing to replace existing manifest: {manifest}")
    units = plan(
        args.input_path.resolve(),
        args.output_root.resolve(),
        snapshot_id=args.snapshot_id,
        target_rows=args.target_rows,
        workers=args.workers,
        required_fields={args.html_field, args.url_field},
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text("".join(json.dumps(unit, sort_keys=True) + "\n" for unit in units))
    os.replace(temporary, manifest)
    total_rows = sum(unit["expected_rows"] for unit in units)
    print(f"planned {total_rows:,} rows in {len(units):,} work units: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
