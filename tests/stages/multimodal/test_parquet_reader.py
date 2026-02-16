from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal import (
    ParquetMultimodalReader,
    ParquetMultimodalReaderStage,
)
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

    out = ParquetMultimodalReaderStage(
        metadata_paths_by_data_path={str(data_path): str(metadata_path)}
    ).process(FileGroupTask(task_id="p0", dataset_name="ds", data=[str(data_path)]))
    rows = out.data.to_pylist()
    assert [row["sample_id"] for row in rows] == ["docA", "docA", "docB"]
    assert [row["position"] for row in rows] == [0, 1, 0]
    assert [row["modality"] for row in rows] == ["text", "image", "text"]
    metadata_by_id = {str(row["sample_id"]): str(row["metadata_json"]) for row in out.metadata_index.to_pylist()}
    assert metadata_by_id == {"docA": '{"src":"a"}', "docB": '{"src":"b"}'}


def test_parquet_multimodal_reader_stage_requires_metadata_mapping(tmp_path: Path) -> None:
    data_path = tmp_path / "batch_no_sidecar.parquet"
    _write_data_parquet(data_path)
    with pytest.raises(ValueError, match="No metadata parquet path configured for source"):
        ParquetMultimodalReaderStage().process(FileGroupTask(task_id="p1", dataset_name="ds", data=[str(data_path)]))


def test_parquet_multimodal_reader_stage_raises_when_explicit_metadata_path_missing(tmp_path: Path) -> None:
    data_path = tmp_path / "batch_missing_meta.parquet"
    _write_data_parquet(data_path)
    missing_metadata_path = tmp_path / "missing.metadata.parquet"
    with pytest.raises(FileNotFoundError, match="Metadata parquet file does not exist"):
        ParquetMultimodalReaderStage(
            metadata_paths_by_data_path={str(data_path): str(missing_metadata_path)}
        ).process(FileGroupTask(task_id="p_missing", dataset_name="ds", data=[str(data_path)]))


def test_parquet_multimodal_reader_composite_decomposes_like_other_readers() -> None:
    stage = ParquetMultimodalReader(
        file_paths="data/mm/a.parquet",
        metadata_file_paths="data/mm/a.metadata.parquet",
        files_per_partition=2,
        max_batch_bytes=128,
    )
    decomposed = stage.decompose()
    assert len(decomposed) == 2
    assert isinstance(decomposed[0], FilePartitioningStage)
    assert isinstance(decomposed[1], ParquetMultimodalReaderStage)
    assert decomposed[0].file_extensions == [".parquet"]
    assert decomposed[1].max_batch_bytes == 128
    assert decomposed[1].metadata_paths_by_data_path == {"data/mm/a.parquet": "data/mm/a.metadata.parquet"}


def test_parquet_multimodal_reader_rejects_misaligned_metadata_path_inputs() -> None:
    with pytest.raises(TypeError, match="must be a list when file_paths is a list"):
        ParquetMultimodalReader(
            file_paths=["a.parquet", "b.parquet"],
            metadata_file_paths="a.metadata.parquet",
        )

    with pytest.raises(ValueError, match="length must match file_paths length"):
        ParquetMultimodalReader(
            file_paths=["a.parquet", "b.parquet"],
            metadata_file_paths=["a.metadata.parquet"],
        )


def test_parquet_multimodal_reader_rejects_non_file_string_input() -> None:
    with pytest.raises(ValueError, match=r"must point to a \.parquet file"):
        ParquetMultimodalReader(
            file_paths="data/mm",
            metadata_file_paths="data/mm.metadata.parquet",
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

    out = ParquetMultimodalReaderStage(metadata_paths_by_data_path={}).process((data_task, metadata_task))
    metadata_by_id = {str(row["sample_id"]): str(row["metadata_json"]) for row in out.metadata_index.to_pylist()}
    assert metadata_by_id == {"docA": '{"src":"tuple-a"}', "docB": '{"src":"tuple-b"}'}


def test_parquet_multimodal_reader_stage_rejects_mismatched_tuple_filetasks(tmp_path: Path) -> None:
    data_path = tmp_path / "tuple_mismatch_data.parquet"
    _write_data_parquet(data_path)
    data_task = FileGroupTask(task_id="tuple_data", dataset_name="ds", data=[str(data_path)])
    metadata_task = FileGroupTask(task_id="tuple_meta", dataset_name="ds", data=[])

    with pytest.raises(ValueError, match="must have matching lengths"):
        ParquetMultimodalReaderStage(metadata_paths_by_data_path={}).process((data_task, metadata_task))
