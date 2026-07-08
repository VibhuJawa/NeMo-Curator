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

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pytest

from benchmarking.scripts import gpu_lance_remote_sequential_ceiling as ceiling


class _FakeScanner:
    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches

    def to_batches(self) -> Iterator[pa.RecordBatch]:
        return iter(self._batches)


class _FakeFragment:
    fragment_id = 7

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches

    def scanner(self, *, columns: list[str], batch_size: int, batch_readahead: int) -> _FakeScanner:
        assert columns == ["image"]
        assert batch_size == 2
        assert batch_readahead == 1
        return _FakeScanner(self._batches)


def _parse_args(*extra: str) -> argparse.Namespace:
    return ceiling.build_parser().parse_args(
        [
            "--dataset-uri",
            "s3://bucket/dataset",
            "--storage-options-file",
            "storage-options.json",
            "--output",
            "report.json",
            *extra,
        ]
    )


def test_evenly_spaced_ordinals_include_manifest_ends() -> None:
    assert ceiling._evenly_spaced_ordinals(10, 4) == [0, 3, 6, 9]
    assert ceiling._evenly_spaced_ordinals(5, 5) == [0, 1, 2, 3, 4]


def test_fragment_id_parser_sorts_and_rejects_duplicates() -> None:
    assert ceiling._fragment_ids("9, 2,5") == [2, 5, 9]
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        ceiling._fragment_ids("2,2")


def test_reader_concurrency_keeps_default_sweep_and_accepts_sensitivity_values() -> None:
    assert ceiling.DEFAULT_READER_CONCURRENCY == (1, 4, 8, 16)
    assert _parse_args().reader_concurrency == []
    assert _parse_args(
        "--reader-concurrency",
        "32",
        "--reader-concurrency",
        "64",
    ).reader_concurrency == [32, 64]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_reader_concurrency_rejects_nonpositive_values(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        _parse_args("--reader-concurrency", value)

    assert raised.value.code == 2
    assert "value must be greater than zero" in capsys.readouterr().err


def test_projection_axes_have_identical_image_payload_column() -> None:
    args = argparse.Namespace(
        image_column="image",
        url_column="url",
        md5_column="md5",
        width_column="width",
        height_column="height",
        projection=[],
    )

    assert ceiling._projection_axes(args) == {
        "image_only": ["image"],
        "image_url": ["image", "url"],
        "full": ["url", "image", "md5", "width", "height"],
    }


def test_scan_fragment_streams_binary_batches() -> None:
    fragment = _FakeFragment(
        [
            pa.record_batch({"image": pa.array([b"abc", None], type=pa.large_binary())}),
            pa.record_batch({"image": pa.array([b"de"], type=pa.large_binary())}),
        ]
    )

    result = ceiling._scan_fragment(
        fragment,
        ceiling._ScanSettings(
            image_column="image",
            projection_columns=["image"],
            batch_rows=2,
            batch_readahead=1,
        ),
        expected_rows=3,
    )

    assert result["fragment_id"] == 7
    assert result["scanned_rows"] == 3
    assert result["logical_payload_bytes"] == 5
    assert result["null_payloads"] == 1
    assert result["arrow_batches"] == 2
    assert result["row_count_correct"] is True


def test_storage_options_reject_credentials_and_load_nonsecret_tuning(tmp_path: Path) -> None:
    options_path = tmp_path / "storage.json"
    options_path.write_text(json.dumps({"secret_access_key": "do-not-load", "region": "test"}))

    with pytest.raises(ValueError, match="process environment"):
        ceiling._load_storage_options(options_path)

    options_path.write_text(json.dumps({"endpoint": "https://object.test", "region": "test"}))
    assert ceiling._load_storage_options(options_path) == {
        "endpoint": "https://object.test",
        "region": "test",
    }


def test_atomic_json_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    measurements = [
        {
            "reader_concurrency": 4,
            "projection": "image_only",
            "status": "measured",
            "elapsed_seconds": 2.0,
            "fetch_seconds": 2.0,
            "images_per_second": 10.0,
            "logical_payload_mib_per_second": 20.0,
            "physical_read_mib_per_second": 30.0,
            "lance_read_bytes": 100.0,
            "lance_read_iops": 5.0,
            "average_physical_read_bytes": 20.0,
            "physical_reads_per_image": 0.5,
            "physical_to_logical_byte_ratio": 1.5,
        },
        {
            "reader_concurrency": 4,
            "projection": "image_only",
            "status": "measured",
            "elapsed_seconds": 4.0,
            "fetch_seconds": 4.0,
            "images_per_second": 5.0,
            "logical_payload_mib_per_second": 10.0,
            "physical_read_mib_per_second": 15.0,
            "lance_read_bytes": 100.0,
            "lance_read_iops": 5.0,
            "average_physical_read_bytes": 20.0,
            "physical_reads_per_image": 0.5,
            "physical_to_logical_byte_ratio": 1.5,
        },
    ]
    summary = ceiling._summarize(measurements)
    ceiling._atomic_write_json(output, {"summary": summary})

    assert (
        json.loads(output.read_text())["summary"]["image_only"]["4"]["metrics"]["images_per_second"]["median"] == 7.5
    )
    assert list(tmp_path.glob(".report.json.tmp-*")) == []
