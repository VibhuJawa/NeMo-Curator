from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal import ParquetMultimodalReader
from nemo_curator.stages.multimodal.io.readers.base import BaseMultimodalReaderStage
from nemo_curator.stages.multimodal.io.readers.parquet import ParquetMultimodalReaderStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypeAlias

    from nemo_curator.tasks import MultimodalBatch

    ReaderOut: TypeAlias = MultimodalBatch | list[MultimodalBatch]


def _process(
    data_path: Path,
    *,
    reader: ParquetMultimodalReaderStage | None = None,
) -> ReaderOut:
    stage = reader or ParquetMultimodalReaderStage()
    data_task = FileGroupTask(task_id="data", dataset_name="ds", data=[str(data_path)])
    return stage.process(data_task)


def _write_data_parquet(path: Path) -> None:
    table = pa.table(
        {
            "sample_id": ["docA", "docA", "docB"],
            "position": [0, 1, 0],
            "modality": ["text", "image", "text"],
            "content_type": ["text/plain", "image/jpeg", "application/json"],
            "text_content": ["caption-a", None, '{"caption":"b"}'],
            "binary_content": [None, b"img-a", None],
            "element_metadata_json": [None, None, None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard-0", "shard-0", "shard-1"],
            "content_path": [None, "s3://bucket/shard-0.tar", None],
            "content_key": [None, "docA.000001.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    pq.write_table(table, path)


def test_parquet_multimodal_reader_stage_reads_data_rows(tmp_path: Path) -> None:
    data_path = tmp_path / "batch.parquet"
    _write_data_parquet(data_path)
    out = _process(data_path)
    rows = out.data.to_pylist()
    assert [row["sample_id"] for row in rows] == ["docA", "docA", "docB"]
    assert [row["position"] for row in rows] == [0, 1, 0]
    assert [row["modality"] for row in rows] == ["text", "image", "text"]


def test_parquet_multimodal_reader_composite_decomposes_like_other_readers() -> None:
    stage = ParquetMultimodalReader(
        file_paths="data/mm/a.parquet",
        files_per_partition=2,
        max_batch_bytes=128,
    )
    decomposed = stage.decompose()
    assert len(decomposed) == 2
    assert isinstance(decomposed[0], FilePartitioningStage)
    assert isinstance(decomposed[1], ParquetMultimodalReaderStage)
    assert decomposed[0].file_extensions == [".parquet"]
    assert decomposed[1].max_batch_bytes == 128


def test_parquet_multimodal_reader_accepts_directory_or_prefix_string_input() -> None:
    stage = ParquetMultimodalReader(file_paths="data/mm", files_per_partition=1)
    decomposed = stage.decompose()
    assert isinstance(decomposed[0], FilePartitioningStage)
    assert decomposed[0].file_paths == "data/mm"


def test_parquet_multimodal_reader_stage_preserves_extra_columns_from_data(tmp_path: Path) -> None:
    data_path = tmp_path / "extra_cols.parquet"
    table = pa.table(
        {
            "sample_id": ["docA"],
            "position": [0],
            "modality": ["text"],
            "content_type": ["text/plain"],
            "text_content": ["caption"],
            "binary_content": [None],
            "element_metadata_json": [None],
            "source_id": ["src"],
            "source_shard": ["shard-0"],
            "content_path": [None],
            "content_key": [None],
            "aesthetic_score": [0.9],
        }
    )
    pq.write_table(table, data_path)
    out = _process(data_path)
    assert set(MULTIMODAL_SCHEMA.names).issubset(set(out.data.column_names))
    assert "aesthetic_score" in out.data.column_names
    assert out.data.num_rows == 1


def test_parquet_multimodal_reader_stage_columns_must_include_required_schema_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "columns_missing_required.parquet"
    _write_data_parquet(data_path)
    with pytest.raises(ValueError, match="must include all multimodal required columns"):
        _process(data_path, reader=ParquetMultimodalReaderStage(columns=["sample_id", "position"]))


def test_parquet_multimodal_reader_stage_reads_with_selected_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "columns_selected.parquet"
    _write_data_parquet(data_path)

    out = _process(
        data_path,
        reader=ParquetMultimodalReaderStage(
            columns=list(MULTIMODAL_SCHEMA.names),
        ),
    )
    assert out.data.column_names == MULTIMODAL_SCHEMA.names


def test_parquet_multimodal_reader_stage_backfills_missing_element_metadata_column(tmp_path: Path) -> None:
    data_path = tmp_path / "legacy_rows.parquet"
    legacy_table = pa.table(
        {
            "sample_id": ["docA"],
            "position": [0],
            "modality": ["text"],
            "content_type": ["text/plain"],
            "text_content": ["caption-a"],
            "binary_content": [None],
            "source_id": ["src"],
            "source_shard": ["shard-0"],
            "content_path": [None],
            "content_key": [None],
        }
    )
    pq.write_table(legacy_table, data_path)
    out = _process(data_path)
    assert "element_metadata_json" in out.data.column_names
    row = out.data.to_pylist()[0]
    assert row["element_metadata_json"] is None


def test_parquet_multimodal_reader_preserves_metadata_rows_in_main_table(tmp_path: Path) -> None:
    data_path = tmp_path / "metadata_rows.parquet"
    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc"],
            "position": [-1, 0, 1],
            "modality": ["metadata", "text", "image"],
            "content_type": [
                "interleaved",
                "text/plain",
                "image/jpeg",
            ],
            "text_content": ['{"src":"x"}', "caption", None],
            "binary_content": [None, None, b"img"],
            "element_metadata_json": ['{"src":"x"}', None, None],
            "source_id": ["src"] * 3,
            "source_shard": ["shard"] * 3,
            "content_path": [None] * 3,
            "content_key": [None, None, "doc.1.jpg"],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    pq.write_table(table, data_path)

    out = _process(data_path)
    rows = sorted(out.data.to_pylist(), key=lambda row: int(row["position"]))
    assert [row["modality"] for row in rows] == ["metadata", "text", "image"]


def test_base_multimodal_reader_text_row_allows_source_id_override() -> None:
    class _Reader(BaseMultimodalReaderStage):
        def read_data(self, data_path: str) -> pa.Table:
            _ = data_path
            return self._empty_data_table()

    reader = _Reader()
    row = reader._text_row(
        sid="sample-1",
        position=0,
        source_shard="shard-0",
        content_type="text/plain",
        text_content="caption",
        source_id="original-source",
    )
    assert row["source_id"] == "original-source"
