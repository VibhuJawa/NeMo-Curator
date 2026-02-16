from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal import ParquetMultimodalReader
from nemo_curator.stages.multimodal.io.readers.parquet import ParquetMultimodalReaderStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.multimodal import METADATA_SCHEMA, MULTIMODAL_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path


def _write_data_parquet(path: Path) -> None:
    table = pa.table(
        {
            "sample_id": ["docA", "docA", "docB"],
            "position": [0, 1, 0],
            "modality": ["text", "image", "text"],
            "content_type": ["text/plain", "image/jpeg", "application/json"],
            "text_content": ["caption-a", None, '{"caption":"b"}'],
            "binary_content": [None, b"img-a", None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard-0", "shard-0", "shard-1"],
            "content_path": [None, "s3://bucket/shard-0.tar", None],
            "content_key": [None, "docA.000001.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    pq.write_table(table, path)


def test_parquet_multimodal_reader_stage_reads_data_and_sidecar(tmp_path: Path) -> None:
    data_path = tmp_path / "batch.parquet"
    _write_data_parquet(data_path)
    metadata_path = tmp_path / "batch.metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["docA", "docB"],
                "sample_type": ["pair", "single"],
                "metadata_json": ['{"src":"a"}', '{"src":"b"}'],
            },
            schema=METADATA_SCHEMA,
        ),
        metadata_path,
    )

    data_task = FileGroupTask(task_id="p0_data", dataset_name="ds", data=[str(data_path)])
    metadata_task = FileGroupTask(task_id="p0_meta", dataset_name="ds", data=[str(metadata_path)])
    out = ParquetMultimodalReaderStage().process((data_task, metadata_task))
    rows = out.data.to_pylist()
    assert [row["sample_id"] for row in rows] == ["docA", "docA", "docB"]
    assert [row["position"] for row in rows] == [0, 1, 0]
    assert [row["modality"] for row in rows] == ["text", "image", "text"]
    metadata_by_id = {str(row["sample_id"]): str(row["metadata_json"]) for row in out.metadata_index.to_pylist()}
    assert metadata_by_id == {"docA": '{"src":"a"}', "docB": '{"src":"b"}'}


def test_parquet_multimodal_reader_stage_skips_metadata_when_none_provided(tmp_path: Path) -> None:
    data_path = tmp_path / "batch_no_sidecar.parquet"
    _write_data_parquet(data_path)
    data_task = FileGroupTask(task_id="p1_data", dataset_name="ds", data=[str(data_path)])
    out = ParquetMultimodalReaderStage().process((data_task, None))
    assert out.metadata_index is not None
    metadata_by_id = {str(row["sample_id"]): row["metadata_json"] for row in out.metadata_index.to_pylist()}
    assert metadata_by_id == {"docA": None, "docB": None}


def test_parquet_multimodal_reader_stage_skips_missing_metadata_files(tmp_path: Path) -> None:
    data_path = tmp_path / "batch_missing_meta.parquet"
    _write_data_parquet(data_path)
    missing_metadata_path = tmp_path / "missing.metadata.parquet"
    data_task = FileGroupTask(task_id="p_missing_data", dataset_name="ds", data=[str(data_path)])
    metadata_task = FileGroupTask(task_id="p_missing_meta", dataset_name="ds", data=[str(missing_metadata_path)])
    out = ParquetMultimodalReaderStage().process((data_task, metadata_task))
    metadata_by_id = {str(row["sample_id"]): row["metadata_json"] for row in out.metadata_index.to_pylist()}
    assert metadata_by_id == {"docA": None, "docB": None}


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


def test_parquet_multimodal_reader_rejects_non_file_string_input() -> None:
    with pytest.raises(ValueError, match=r"must point to a \.parquet file"):
        ParquetMultimodalReader(
            file_paths="data/mm",
        )


def test_parquet_multimodal_reader_stage_accepts_tuple_of_data_and_metadata_filetasks(tmp_path: Path) -> None:
    data_path = tmp_path / "tuple_data.parquet"
    metadata_path = tmp_path / "tuple_data.metadata.parquet"
    _write_data_parquet(data_path)
    pq.write_table(
        pa.table(
            {
                "sample_id": ["docA", "docB"],
                "sample_type": ["pair", "single"],
                "metadata_json": ['{"src":"tuple-a"}', '{"src":"tuple-b"}'],
            },
            schema=METADATA_SCHEMA,
        ),
        metadata_path,
    )

    data_task = FileGroupTask(task_id="tuple_data", dataset_name="ds", data=[str(data_path)])
    metadata_task = FileGroupTask(task_id="tuple_meta", dataset_name="ds", data=[str(metadata_path)])

    out = ParquetMultimodalReaderStage().process((data_task, metadata_task))
    metadata_by_id = {str(row["sample_id"]): str(row["metadata_json"]) for row in out.metadata_index.to_pylist()}
    assert metadata_by_id == {"docA": '{"src":"tuple-a"}', "docB": '{"src":"tuple-b"}'}


def test_parquet_multimodal_reader_stage_skips_metadata_when_tuple_metadata_task_empty(tmp_path: Path) -> None:
    data_path = tmp_path / "tuple_mismatch_data.parquet"
    _write_data_parquet(data_path)
    data_task = FileGroupTask(task_id="tuple_data", dataset_name="ds", data=[str(data_path)])
    metadata_task = FileGroupTask(task_id="tuple_meta", dataset_name="ds", data=[])

    out = ParquetMultimodalReaderStage().process((data_task, metadata_task))
    metadata_by_id = {str(row["sample_id"]): row["metadata_json"] for row in out.metadata_index.to_pylist()}
    assert metadata_by_id == {"docA": None, "docB": None}


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
            "source_id": ["src"],
            "source_shard": ["shard-0"],
            "content_path": [None],
            "content_key": [None],
            "aesthetic_score": [0.9],
        }
    )
    pq.write_table(table, data_path)
    out = ParquetMultimodalReaderStage().process((FileGroupTask(task_id="d", dataset_name="ds", data=[str(data_path)]), None))
    assert set(MULTIMODAL_SCHEMA.names).issubset(set(out.data.column_names))
    assert "aesthetic_score" in out.data.column_names
    assert out.data.num_rows == 1


def test_parquet_multimodal_reader_stage_columns_must_include_required_schema_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "columns_missing_required.parquet"
    _write_data_parquet(data_path)
    with pytest.raises(ValueError, match="must include all multimodal required columns"):
        ParquetMultimodalReaderStage(columns=["sample_id", "position"]).process(
            (FileGroupTask(task_id="d", dataset_name="ds", data=[str(data_path)]), None)
        )


def test_parquet_multimodal_reader_stage_reads_with_selected_columns(tmp_path: Path) -> None:
    data_path = tmp_path / "columns_selected.parquet"
    _write_data_parquet(data_path)
    metadata_path = tmp_path / "columns_selected.metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["docA", "docB"],
                "sample_type": ["pair", "single"],
                "metadata_json": ['{"src":"a"}', '{"src":"b"}'],
                "extra_meta": ["x", "y"],
            }
        ),
        metadata_path,
    )

    out = ParquetMultimodalReaderStage(
        columns=list(MULTIMODAL_SCHEMA.names),
        metadata_columns=["sample_id", "metadata_json"],
    ).process(
        (
            FileGroupTask(task_id="d", dataset_name="ds", data=[str(data_path)]),
            FileGroupTask(task_id="m", dataset_name="ds", data=[str(metadata_path)]),
        )
    )
    assert out.data.column_names == MULTIMODAL_SCHEMA.names
    assert set(METADATA_SCHEMA.names).issubset(set(out.metadata_index.column_names))
    metadata_rows = {str(row["sample_id"]): row["metadata_json"] for row in out.metadata_index.to_pylist()}
    assert metadata_rows == {"docA": '{"src":"a"}', "docB": '{"src":"b"}'}
