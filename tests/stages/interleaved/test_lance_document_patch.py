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

from __future__ import annotations

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.lance_document_patch import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    LANCE_ROWADDR,
    SAMPLE_ID,
    STABLE_ROW_ID,
    apply_payload_part,
    split_interleaved_by_actual_bytes,
)


def _document(*, addresses: list[int] | None = None, binary: list[bytes | None] | None = None) -> pa.Table:
    addresses = [100, 101, 102, 103] if addresses is None else addresses
    columns: dict[str, pa.Array] = {
        SAMPLE_ID: pa.array([f"sample-{index}" for index in range(len(addresses))], type=pa.string()),
        LANCE_ROWADDR: pa.array(addresses, type=pa.uint64()),
    }
    if binary is not None:
        columns["binary_content"] = pa.array(binary, type=pa.binary())
    return pa.table(columns)


def _payload_part(
    *,
    addresses: list[int] | None = None,
    positions: list[int] | None = None,
    stable_ids: list[int] | None = None,
    images: list[bytes | None] | None = None,
) -> pa.Table:
    addresses = [103, 101] if addresses is None else addresses
    positions = [3, 1] if positions is None else positions
    stable_ids = [7, 7] if stable_ids is None else stable_ids
    images = [b"image-d", b"image-b"] if images is None else images
    return pa.table(
        {
            DOCUMENT_ROWADDR: pa.array(addresses, type=pa.uint64()),
            DOCUMENT_POSITION: pa.array(positions, type=pa.uint64()),
            STABLE_ROW_ID: pa.array(stable_ids, type=pa.uint64()),
            "image": pa.array(images, type=pa.binary()),
            "width": pa.array([640 + index for index in range(len(addresses))], type=pa.uint32()),
        }
    )


def _interleaved(sample_ids: list[str], payload_sizes: list[int]) -> pa.Table:
    return pa.table(
        {
            SAMPLE_ID: pa.array(sample_ids, type=pa.string()),
            "position": pa.array(range(len(sample_ids)), type=pa.int32()),
            "binary_content": pa.array([b"x" * size for size in payload_sizes], type=pa.large_binary()),
        }
    )


def test_apply_payload_part_preserves_order_and_allows_duplicate_stable_ids() -> None:
    document = _document()
    part = _payload_part()

    result = apply_payload_part(
        document,
        part,
        {"image": "binary_content", "width": "reference_width"},
        "overwrite",
    )

    assert result[SAMPLE_ID].to_pylist() == document[SAMPLE_ID].to_pylist()
    assert result[LANCE_ROWADDR].to_pylist() == [100, 101, 102, 103]
    assert result["binary_content"].to_pylist() == [None, b"image-b", None, b"image-d"]
    assert result["reference_width"].to_pylist() == [None, 641, None, 640]
    assert result.num_rows == document.num_rows


def test_apply_payload_part_repeatedly_fills_disjoint_parts() -> None:
    document = _document(binary=[b"keep", None, None, None])
    first = _payload_part(
        addresses=[101],
        positions=[1],
        stable_ids=[5],
        images=[b"first"],
    )
    second = _payload_part(
        addresses=[103],
        positions=[3],
        stable_ids=[8],
        images=[b"second"],
    )

    after_first = apply_payload_part(document, first, {"image": "binary_content"}, "fill_null")
    result = apply_payload_part(after_first, second, {"image": "binary_content"}, "fill_null")

    assert result["binary_content"].to_pylist() == [b"keep", b"first", None, b"second"]
    assert result[LANCE_ROWADDR].to_pylist() == document[LANCE_ROWADDR].to_pylist()


@pytest.mark.parametrize(
    ("document", "part", "message"),
    [
        (_document(addresses=[100, 100]), _payload_part(addresses=[100], positions=[0], stable_ids=[1], images=[b"x"]), "document _rowaddr values must be unique"),
        (_document(), _payload_part(addresses=[101, 101], positions=[1, 2]), "document_rowaddr values must be unique"),
        (_document(), _payload_part(addresses=[101, 103], positions=[1, 1]), "document_position values must be unique"),
        (_document(), _payload_part(addresses=[999], positions=[1], stable_ids=[1], images=[b"x"]), "outside the document"),
    ],
)
def test_apply_payload_part_rejects_duplicate_or_foreign_coordinates(
    document: pa.Table,
    part: pa.Table,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_payload_part(document, part, {"image": "binary_content"}, "overwrite")


def test_apply_payload_part_rejects_coordinate_and_destination_type_errors() -> None:
    wrong_coordinates = _payload_part().set_column(
        0,
        DOCUMENT_ROWADDR,
        pa.array([103, 101], type=pa.int64()),
    )
    with pytest.raises(TypeError, match="must have uint64 type"):
        apply_payload_part(_document(), wrong_coordinates, {"image": "binary_content"}, "overwrite")

    wrong_destination = _document().append_column("binary_content", pa.array(["", "", "", ""]))
    with pytest.raises(TypeError, match="payload source has type"):
        apply_payload_part(wrong_destination, _payload_part(), {"image": "binary_content"}, "overwrite")


def test_apply_payload_part_rejects_invalid_mapping_policy_and_existing_destination() -> None:
    part = _payload_part()
    with pytest.raises(ValueError, match="missing image source"):
        apply_payload_part(_document(), part, {"missing": "binary_content"}, "overwrite")
    with pytest.raises(ValueError, match="more than one source"):
        apply_payload_part(
            _document(),
            part,
            {"image": "payload", "width": "payload"},
            "overwrite",
        )
    with pytest.raises(ValueError, match="Unsupported existing_column_policy"):
        apply_payload_part(_document(), part, {"image": "binary_content"}, "invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="already contains destination"):
        apply_payload_part(
            _document(binary=[None, None, None, None]),
            part,
            {"image": "binary_content"},
            "error",
        )


def test_split_interleaved_packs_contiguous_samples_without_splitting_them() -> None:
    table = _interleaved(
        ["sample-a", "sample-a", "sample-b", "sample-c", "sample-c"],
        [4, 5, 3, 2, 2],
    )
    target_bytes = table.slice(0, 3).nbytes

    result = split_interleaved_by_actual_bytes(table, target_bytes)

    assert len(result.patches) == 2
    assert result.patches[0][SAMPLE_ID].to_pylist() == ["sample-a", "sample-a", "sample-b"]
    assert result.patches[1][SAMPLE_ID].to_pylist() == ["sample-c", "sample-c"]
    assert all(patch.nbytes <= target_bytes for patch in result.patches)
    assert result.oversized_samples == ()
    assert pa.concat_tables(result.patches).equals(table)


def test_split_interleaved_isolates_and_reports_one_oversized_sample() -> None:
    table = _interleaved(["small-a", "large", "large", "small-c"], [1, 128, 128, 1])
    target_bytes = max(table.slice(0, 1).nbytes, table.slice(3, 1).nbytes) + 1

    result = split_interleaved_by_actual_bytes(table, target_bytes)

    assert [patch[SAMPLE_ID].to_pylist() for patch in result.patches] == [
        ["small-a"],
        ["large", "large"],
        ["small-c"],
    ]
    assert len(result.oversized_samples) == 1
    oversized = result.oversized_samples[0]
    assert oversized.patch_index == 1
    assert oversized.start_row == 1
    assert oversized.row_count == 2
    assert oversized.actual_bytes == result.patches[1].nbytes
    assert oversized.actual_bytes > target_bytes
    assert oversized.sample_id.as_py() == "large"
    assert result.patches[0].nbytes <= target_bytes
    assert result.patches[2].nbytes <= target_bytes
    assert pa.concat_tables(result.patches).equals(table)


def test_split_interleaved_is_deterministic_and_handles_empty_tables() -> None:
    table = _interleaved(["a", "b", "c"], [5, 5, 5])
    target_bytes = table.slice(0, 2).nbytes

    first = split_interleaved_by_actual_bytes(table, target_bytes)
    second = split_interleaved_by_actual_bytes(table, target_bytes)
    empty = split_interleaved_by_actual_bytes(table.slice(0, 0), target_bytes)

    assert len(first.patches) == len(second.patches)
    assert all(left.equals(right) for left, right in zip(first.patches, second.patches, strict=True))
    assert first.oversized_samples == second.oversized_samples
    assert empty.patches == ()
    assert empty.oversized_samples == ()


@pytest.mark.parametrize(
    ("table", "target_bytes", "exception", "message"),
    [
        (pa.table({"payload": [b"x"]}), 10, ValueError, "missing required column"),
        (pa.table({SAMPLE_ID: pa.array([1], type=pa.int64())}), 10, TypeError, "must have string"),
        (pa.table({SAMPLE_ID: pa.array([None], type=pa.string())}), 10, ValueError, "must not contain nulls"),
        (_interleaved(["a", "b", "a"], [1, 1, 1]), 1000, ValueError, "exactly one contiguous"),
        (_interleaved(["a"], [1]), 0, ValueError, "positive integer"),
        (_interleaved(["a"], [1]), True, ValueError, "positive integer"),
    ],
)
def test_split_interleaved_rejects_schema_order_and_target_errors(
    table: pa.Table,
    target_bytes: int,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        split_interleaved_by_actual_bytes(table, target_bytes)
