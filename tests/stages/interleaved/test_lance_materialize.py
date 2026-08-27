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

from pathlib import Path

import lance
import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.lance import InterleavedLanceMaterializeStage
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

_ROW_ID_COLUMN = "image_row_id"


def _image_table(path: Path, count: int, *, stable_row_ids: bool = True, rows_per_file: int = 4) -> lance.LanceDataset:
    table = pa.table(
        {
            "url": [f"https://example/{i}.jpg" for i in range(count)],
            "image": pa.array([f"payload-{i}".encode() for i in range(count)], type=pa.large_binary()),
        }
    )
    lance.write_dataset(
        table,
        str(path),
        mode="overwrite",
        max_rows_per_file=rows_per_file,
        enable_stable_row_ids=stable_row_ids,
    )
    return lance.dataset(str(path))


def _batch(row_ids: list[int | None]) -> InterleavedBatch:
    schema = INTERLEAVED_SCHEMA.append(pa.field(_ROW_ID_COLUMN, pa.int64(), nullable=True))
    rows = [
        {
            "sample_id": f"s{index}",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            _ROW_ID_COLUMN: row_id,
        }
        for index, row_id in enumerate(row_ids)
    ]
    return InterleavedBatch(dataset_name="d", data=pa.Table.from_pylist(rows, schema=schema))


def _stage(path: Path, **overrides: object) -> InterleavedLanceMaterializeStage:
    stage = InterleavedLanceMaterializeStage(uri=str(path), version=1, row_id_column=_ROW_ID_COLUMN, **overrides)
    stage.setup()
    return stage


def test_fills_payloads_for_scattered_row_ids(tmp_path: Path) -> None:
    """The core case: ids scattered across fragments, each row gets its own payload."""
    _image_table(tmp_path / "img", 20)
    stage = _stage(tmp_path / "img")
    try:
        out = stage.process(_batch([17, 3, 11, 0])).to_pyarrow()
    finally:
        stage.teardown()

    assert out["binary_content"].to_pylist() == [b"payload-17", b"payload-3", b"payload-11", b"payload-0"]


def test_a_missing_row_id_does_not_shift_other_payloads(tmp_path: Path) -> None:
    """The dangerous case.

    ``_take_rows`` silently returns fewer rows when an id is absent. Mapping that
    positionally would attach payload-9 to the row that asked for the missing id and
    shift everything after it, with no error.
    """
    dataset = _image_table(tmp_path / "img", 10)
    missing = dataset.count_rows() + 500
    stage = _stage(tmp_path / "img")
    try:
        out = stage.process(_batch([2, missing, 9])).to_pyarrow()
    finally:
        stage.teardown()

    assert out["binary_content"].to_pylist() == [b"payload-2", None, b"payload-9"]


def test_null_row_ids_are_left_alone(tmp_path: Path) -> None:
    _image_table(tmp_path / "img", 10)
    stage = _stage(tmp_path / "img")
    try:
        out = stage.process(_batch([4, None, 6])).to_pyarrow()
    finally:
        stage.teardown()

    assert out["binary_content"].to_pylist() == [b"payload-4", None, b"payload-6"]


def test_duplicate_row_ids_are_fetched_once_and_filled_everywhere(tmp_path: Path) -> None:
    _image_table(tmp_path / "img", 10)
    stage = _stage(tmp_path / "img")
    try:
        out = stage.process(_batch([5, 5, 5])).to_pyarrow()
    finally:
        stage.teardown()

    assert out["binary_content"].to_pylist() == [b"payload-5"] * 3


def test_chunking_across_many_takes_preserves_mapping(tmp_path: Path) -> None:
    """Small take_batch_size forces many chunks; every row must still match its id."""
    _image_table(tmp_path / "img", 200, rows_per_file=16)
    row_ids = list(range(199, -1, -3))
    stage = _stage(tmp_path / "img", take_batch_size=7, io_threads=4)
    try:
        out = stage.process(_batch(row_ids)).to_pyarrow()
    finally:
        stage.teardown()

    assert out["binary_content"].to_pylist() == [f"payload-{i}".encode() for i in row_ids]


def test_setup_rejects_a_table_without_stable_row_ids(tmp_path: Path) -> None:
    """Global ids are meaningless without stable row ids; fail rather than corrupt."""
    _image_table(tmp_path / "img", 10, stable_row_ids=False)
    stage = InterleavedLanceMaterializeStage(uri=str(tmp_path / "img"), version=1, row_id_column=_ROW_ID_COLUMN)
    with pytest.raises(ValueError, match="stable row ids"):
        stage.setup()


def test_setup_rejects_a_missing_binary_column(tmp_path: Path) -> None:
    _image_table(tmp_path / "img", 10)
    stage = InterleavedLanceMaterializeStage(
        uri=str(tmp_path / "img"), version=1, row_id_column=_ROW_ID_COLUMN, binary_column="absent"
    )
    with pytest.raises(ValueError, match="absent"):
        stage.setup()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"version": 0}, "version"),
        ({"row_id_column": ""}, "row_id_column"),
        ({"io_threads": 0}, "io_threads"),
        ({"take_batch_size": -1}, "take_batch_size"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict, match: str) -> None:
    base = {"uri": "s3://bucket/table", "version": 1, "row_id_column": _ROW_ID_COLUMN}
    with pytest.raises(ValueError, match=match):
        InterleavedLanceMaterializeStage(**{**base, **kwargs})


def test_already_materialized_rows_are_not_refetched(tmp_path: Path) -> None:
    _image_table(tmp_path / "img", 10)
    stage = _stage(tmp_path / "img")
    try:
        first = stage.process(_batch([1, 2])).to_pyarrow()
        # Feed the filled batch back; nothing is pending, so the table is returned as is.
        second = stage.process(InterleavedBatch(dataset_name="d", data=first)).to_pyarrow()
    finally:
        stage.teardown()

    assert second["binary_content"].to_pylist() == [b"payload-1", b"payload-2"]
