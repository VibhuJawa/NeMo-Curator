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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.io.reader import InterleavedParquetReader
from nemo_curator.stages.interleaved.io.readers.base import _resolve_schema
from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.stages.interleaved.utils.schema import align_table, reconcile_schema
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .conftest import make_row


def _write_synthetic_parquet(path: Path, num_samples: int = 3) -> str:
    rows = []
    for i in range(num_samples):
        sid = f"sample_{i}"
        rows.append(make_row(sample_id=sid, position=-1, modality="metadata"))
        rows.append(make_row(sample_id=sid, position=0, modality="text", text_content=f"Hello from sample {i}"))
        rows.append(make_row(sample_id=sid, position=1, modality="image"))
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    file_path = str(path / "test.parquet")
    pq.write_table(table, file_path)
    return file_path


def test_read_basic(tmp_path: Path) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage().process(task)
    assert isinstance(result, InterleavedBatch)
    assert result.count() == 9
    assert result.count(modality="metadata") == 3
    assert result.count(modality="text") == 3
    assert result.count(modality="image") == 3


def test_read_with_fields_subset(tmp_path: Path) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage(fields=["sample_id", "position", "modality"]).process(task)
    assert isinstance(result, InterleavedBatch)
    cols = set(result.to_pyarrow().column_names)
    assert {"sample_id", "position", "modality"} <= cols


def test_read_missing_columns_filled_as_null(tmp_path: Path) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage(fields=["sample_id", "position", "modality", "nonexistent_col"]).process(
        task
    )
    table = result.to_pyarrow()
    assert "nonexistent_col" in table.column_names
    assert table.column("nonexistent_col").null_count == table.num_rows


def test_reconcile_schema_large_string_compat() -> None:
    inferred = pa.schema(
        [
            pa.field("sample_id", pa.large_string()),
            pa.field("position", pa.int32()),
            pa.field("modality", pa.large_string()),
            pa.field("extra_col", pa.float64()),
        ]
    )
    reconciled = reconcile_schema(inferred)
    assert reconciled.field("sample_id").type == pa.large_string()
    assert reconciled.field("modality").type == pa.large_string()
    assert reconciled.field("position").type == pa.int32()
    assert reconciled.field("extra_col").type == pa.float64()


def test_reconcile_schema_preserves_small_types() -> None:
    inferred = pa.schema(
        [
            pa.field("sample_id", pa.string()),
            pa.field("position", pa.int32()),
            pa.field("modality", pa.string()),
        ]
    )
    reconciled = reconcile_schema(inferred)
    assert reconciled.field("sample_id").type == pa.string()
    assert reconciled.field("position").type == pa.int32()


# --- _resolve_schema ---


def test_resolve_schema_overrides_preserve_nullable() -> None:
    """schema_overrides must preserve nullable=False for reserved INTERLEAVED_SCHEMA columns."""
    import pyarrow as pa

    # sample_id is nullable=False in INTERLEAVED_SCHEMA; override only the type
    result = _resolve_schema(schema=None, overrides={"sample_id": pa.large_string()})
    assert result is not None
    assert result.field("sample_id").type == pa.large_string()
    assert result.field("sample_id").nullable is False  # preserved, not hardcoded True


def test_resolve_schema_overrides_passthrough_column_is_nullable() -> None:
    """schema_overrides for a new (passthrough) column defaults to nullable=True."""
    import pyarrow as pa

    result = _resolve_schema(schema=None, overrides={"custom_score": pa.float32()})
    assert result is not None
    assert result.field("custom_score").type == pa.float32()
    assert result.field("custom_score").nullable is True


def test_resolve_schema_both_none_raises() -> None:
    """_resolve_schema must raise ValueError when both schema and overrides are None."""
    with pytest.raises(ValueError, match="At least one of schema= or schema_overrides= must be provided"):
        _resolve_schema(schema=None, overrides=None)


# --- align_table ---


def test_align_table_passthrough_overflow_raises() -> None:
    """align_table must raise (not silently truncate) when a passthrough column
    cast would overflow — safe=True is used for non-reserved columns."""
    # int64 value that overflows int32 (> 2^31-1); safe=False would silently truncate
    overflow_val = 2**31
    table = pa.table({"custom_count": pa.array([overflow_val], type=pa.int64())})
    target = pa.schema([pa.field("custom_count", pa.int32())])
    with pytest.raises(pa.lib.ArrowInvalid):
        align_table(table, target)


def test_align_table_passthrough_safe_upcast_succeeds() -> None:
    """align_table must successfully upcast a passthrough column (string→large_string)."""
    table = pa.table({"custom_tag": pa.array(["tag1", "tag2"], type=pa.string())})
    target = pa.schema([pa.field("custom_tag", pa.large_string())])
    result = align_table(table, target)
    assert result.schema.field("custom_tag").type == pa.large_string()
    assert result.column("custom_tag").to_pylist() == ["tag1", "tag2"]


def test_align_table_reserved_large_string_preserved() -> None:
    """Reserved columns must not be downcast — reconcile_schema upgrades string→large_string."""
    # sample_id is a reserved column; if inferred as string, reconcile_schema keeps it as string
    # (it only avoids large→small; small stays small for reserved cols too)
    table = pa.table(
        {
            "sample_id": pa.array(["s1"], type=pa.large_string()),
            "position": pa.array([0], type=pa.int32()),
            "modality": pa.array(["text"], type=pa.large_string()),
        }
    )
    target = pa.schema(
        [
            pa.field("sample_id", pa.large_string()),
            pa.field("position", pa.int32()),
            pa.field("modality", pa.large_string()),
        ]
    )
    result = align_table(table, target)
    assert result.schema.field("sample_id").type == pa.large_string()
    assert result.column("sample_id").to_pylist() == ["s1"]


def test_align_table_null_fills_missing_columns() -> None:
    """Columns present in target but absent from table are filled with typed nulls."""
    table = pa.table({"sample_id": pa.array(["s1"], type=pa.large_string())})
    target = pa.schema(
        [
            pa.field("sample_id", pa.large_string()),
            pa.field("custom_score", pa.float32()),
        ]
    )
    result = align_table(table, target)
    assert result.schema.field("custom_score").type == pa.float32()
    assert result.column("custom_score").null_count == 1


def test_read_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pq1 = _write_synthetic_parquet(tmp_path / "a", num_samples=2)
    pq2 = _write_synthetic_parquet(tmp_path / "b", num_samples=3)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq1, pq2], _metadata={"source_files": [pq1, pq2]})
    result = InterleavedParquetReaderStage().process(task)
    assert isinstance(result, InterleavedBatch)
    assert result.count() == 15


@pytest.mark.parametrize(
    ("schema", "schema_overrides", "expected_cols"),
    [
        pytest.param(
            pa.schema(
                [
                    pa.field("sample_id", pa.string()),
                    pa.field("position", pa.int32()),
                    pa.field("modality", pa.string()),
                ]
            ),
            None,
            ["sample_id", "position", "modality"],
            id="explicit_schema_drops_extra_cols",
        ),
        pytest.param(
            None,
            {"sample_id": pa.large_string()},
            None,  # all INTERLEAVED_SCHEMA cols present; check sample_id type instead
            id="schema_overrides_applied",
        ),
    ],
)
def test_explicit_schema_aligns_table(
    tmp_path: Path,
    schema: pa.Schema | None,
    schema_overrides: dict | None,
    expected_cols: list[str] | None,
) -> None:
    pq_path = _write_synthetic_parquet(tmp_path)
    task = FileGroupTask(task_id="t0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]})
    result = InterleavedParquetReaderStage(schema=schema, schema_overrides=schema_overrides).process(task)
    assert isinstance(result, InterleavedBatch)
    table = result.to_pyarrow()
    if expected_cols is not None:
        assert table.column_names == expected_cols
    else:
        assert table.schema.field("sample_id").type == pa.large_string()


def test_interleaved_parquet_reader_decompose(tmp_path: Path) -> None:
    reader = InterleavedParquetReader(file_paths=str(tmp_path))
    stages = reader.decompose()
    assert len(stages) == 2
    assert isinstance(stages[0], FilePartitioningStage)
    assert isinstance(stages[1], InterleavedParquetReaderStage)
