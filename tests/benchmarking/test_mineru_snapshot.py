# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking" / "scripts"))

from mineru_snapshot import (
    load_manifest,
    new_attempt_directory,
    preflight_input,
    publish_attempt,
    read_published_result,
    select_work_unit,
    validate_output,
    verify_snapshot,
)
from plan_mineru_snapshot import plan


def _write_parquet(path: Path, urls: list[str], *, output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"url": urls, "content": [b"html"] * len(urls)}
    if output:
        data |= {"text": ["x" * 300] * len(urls), "_mineru_status": ["ok"] * len(urls)}
    pd.DataFrame(data).to_parquet(path, index=False)


def _manifest(tmp_path: Path, input_paths: list[Path], expected_rows: int) -> Path:
    path = tmp_path / "manifest.jsonl"
    unit = {
        "snapshot_id": "CC-MAIN-TEST",
        "work_unit_id": "000000",
        "input_paths": [str(item) for item in input_paths],
        "output_path": str(tmp_path / "published" / "work-unit-000000"),
        "expected_rows": expected_rows,
        "expected_input_bytes": sum(item.stat().st_size for item in input_paths),
    }
    path.write_text(json.dumps(unit) + "\n")
    return path


def test_manifest_rejects_duplicate_input_files(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_parquet(source, ["a"])
    manifest = _manifest(tmp_path, [source], 1)
    manifest.write_text(manifest.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate work_unit_id"):
        load_manifest(manifest)


def test_work_unit_validation_is_order_independent_and_publishes_atomically(tmp_path: Path) -> None:
    first, second = tmp_path / "in-0.parquet", tmp_path / "in-1.parquet"
    _write_parquet(first, ["a", "b"])
    _write_parquet(second, ["c"])
    manifest = _manifest(tmp_path, [first, second], 3)
    unit, digest = select_work_unit(manifest, 0)
    input_stats = preflight_input(unit, html_field="content", url_field="url")

    attempt = new_attempt_directory(unit)
    _write_parquet(attempt / "part.parquet", ["c", "a", "b"], output=True)
    validation = validate_output(
        unit,
        attempt,
        input_stats,
        url_field="url",
        text_field="text",
        status_field="_mineru_status",
        min_status_ok_rate=0.95,
        min_nonempty_rate=0.95,
        max_convert_error_rate=0.02,
    )
    assert validation["verification_passed"]
    publish_attempt(unit, attempt, digest, validation)
    assert not attempt.exists()
    assert read_published_result(unit, digest)["validation"]["num_rows"] == 3


def test_validation_rejects_a_duplicate_and_missing_url(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_parquet(source, ["a", "b"])
    manifest = _manifest(tmp_path, [source], 2)
    unit, _ = select_work_unit(manifest, 0)
    input_stats = preflight_input(unit, html_field="content", url_field="url")
    attempt = tmp_path / "attempt"
    _write_parquet(attempt / "part.parquet", ["a", "a"], output=True)
    validation = validate_output(
        unit,
        attempt,
        input_stats,
        url_field="url",
        text_field="text",
        status_field="_mineru_status",
        min_status_ok_rate=0.95,
        min_nonempty_rate=0.95,
        max_convert_error_rate=0.02,
    )
    assert not validation["verification_passed"]
    assert "URL multiset" in " ".join(validation["verification_errors"])


def test_snapshot_verifier_requires_every_published_unit(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _write_parquet(source, ["a"])
    manifest = _manifest(tmp_path, [source], 1)
    success_path = tmp_path / "SNAPSHOT_SUCCESS.json"
    assert not verify_snapshot(manifest, success_path)["verification_passed"]
    assert not success_path.exists()

    unit, digest = select_work_unit(manifest, 0)
    input_stats = preflight_input(unit, html_field="content", url_field="url")
    attempt = new_attempt_directory(unit)
    _write_parquet(attempt / "part.parquet", ["a"], output=True)
    validation = validate_output(
        unit,
        attempt,
        input_stats,
        url_field="url",
        text_field="text",
        status_field="_mineru_status",
        min_status_ok_rate=0.95,
        min_nonempty_rate=0.95,
        max_convert_error_rate=0.02,
    )
    publish_attempt(unit, attempt, digest, validation)
    assert verify_snapshot(manifest, success_path)["verification_passed"]
    assert success_path.is_file()


def test_planner_uses_file_boundaries_and_stable_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_parquet(source / "b.parquet", ["c", "d"])
    _write_parquet(source / "a.parquet", ["a", "b"])
    units = plan(
        source,
        tmp_path / "out",
        snapshot_id="snapshot",
        target_rows=3,
        workers=2,
        required_fields={"content", "url"},
    )
    assert [unit["expected_rows"] for unit in units] == [2, 2]
    assert units[0]["input_paths"][0].endswith("a.parquet")


def test_planner_rejects_a_file_larger_than_one_work_unit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_parquet(source / "large.parquet", ["a", "b", "c", "d"])

    with pytest.raises(ValueError, match="repartition them first"):
        plan(
            source,
            tmp_path / "out",
            snapshot_id="snapshot",
            target_rows=3,
            workers=1,
            required_fields={"content", "url"},
        )
