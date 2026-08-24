# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking" / "scripts"))

from mineru_snapshot import verify_native_snapshot

from nemo_curator.backends.slurm_array import build_slurm_array_completion_manifest


def _write_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "url": ["https://example.com"],
            "text": ["x" * 300],
            "_mineru_status": ["ok"],
        }
    ).to_parquet(path, index=False)


def _verify(tmp_path: Path, *, expected_num_warcs: int = 2) -> dict:
    return verify_native_snapshot(
        output_path=tmp_path / "output",
        checkpoint_path=tmp_path / "checkpoint",
        success_path=tmp_path / "SNAPSHOT_SUCCESS.json",
        expected_num_warcs=expected_num_warcs,
        url_field="url",
        text_field="text",
        status_field="_mineru_status",
        min_status_ok_rate=0.95,
        min_nonempty_rate=0.95,
        max_convert_error_rate=0.02,
        quality_sample_files=1,
    )


def test_native_snapshot_verifies_completion_footers_and_quality(tmp_path: Path) -> None:
    _write_output(tmp_path / "output" / "a.parquet")
    _write_output(tmp_path / "output" / "b.parquet")
    for shard in range(2):
        build_slurm_array_completion_manifest(tmp_path / "checkpoint", shard, 2, 0).mark_completed()

    result = _verify(tmp_path)

    assert result["verification_passed"]
    assert result["num_output_files"] == 2
    assert result["num_documents_processed"] == 2
    assert (tmp_path / "SNAPSHOT_SUCCESS.json").is_file()


def test_native_snapshot_rejects_an_incomplete_shard(tmp_path: Path) -> None:
    _write_output(tmp_path / "output" / "a.parquet")
    _write_output(tmp_path / "output" / "b.parquet")
    build_slurm_array_completion_manifest(tmp_path / "checkpoint", 0, 2, 0).mark_completed()

    result = _verify(tmp_path)

    assert not result["verification_passed"]
    assert result["missing_shard_indices"] == [1]


def test_native_snapshot_rejects_wrong_output_file_count(tmp_path: Path) -> None:
    _write_output(tmp_path / "output" / "a.parquet")
    build_slurm_array_completion_manifest(tmp_path / "checkpoint", 0, 1, 0).mark_completed()

    result = _verify(tmp_path, expected_num_warcs=2)

    assert not result["verification_passed"]
    assert "output files 1 < expected WARC files 2" in result["verification_errors"]


def test_native_snapshot_accepts_multiple_chunks_per_warc(tmp_path: Path) -> None:
    for name in ("a-0.parquet", "a-1.parquet", "b-0.parquet"):
        _write_output(tmp_path / "output" / name)
    for shard in range(2):
        build_slurm_array_completion_manifest(tmp_path / "checkpoint", shard, 2, 0).mark_completed()

    result = _verify(tmp_path)

    assert result["verification_passed"]
    assert result["num_output_files"] == 3
