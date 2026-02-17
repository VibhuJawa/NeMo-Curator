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
from nemo_curator.tasks.multimodal import METADATA_SCHEMA, MULTIMODAL_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import TypeAlias

    from nemo_curator.tasks import MultimodalBatch

    ReaderOut: TypeAlias = MultimodalBatch | list[MultimodalBatch]


def _process(
    data_path: Path,
    metadata_path: Path | None = None,
    *,
    reader: ParquetMultimodalReaderStage | None = None,
    metadata_task_data: list[str] | None = None,
) -> ReaderOut:
    stage = reader or ParquetMultimodalReaderStage()
    data_task = FileGroupTask(task_id="data", dataset_name="ds", data=[str(data_path)])
    if metadata_task_data is not None:
        metadata_task = FileGroupTask(task_id="meta", dataset_name="ds", data=metadata_task_data)
        return stage.process((data_task, metadata_task))
    if metadata_path is None:
        return stage.process((data_task, None))
    metadata_task = FileGroupTask(task_id="meta", dataset_name="ds", data=[str(metadata_path)])
    return stage.process((data_task, metadata_task))


def _metadata_by_id(batch: MultimodalBatch) -> dict[str, str | None]:
    return {str(row["sample_id"]): row["metadata_json"] for row in batch.metadata_index.to_pylist()}


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


def test_parquet_multimodal_reader_stage_reads_data_and_sidecar(tmp_path: Path) -> None:
    data_path = tmp_path / "batch.parquet"
    _write_data_parquet(data_path)
    metadata_path = tmp_path / "batch.metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["docA", "docB"],
                "sample_type": ["custom_type_a", "custom_type_b"],
                "metadata_json": ['{"src":"a"}', '{"src":"b"}'],
            },
            schema=METADATA_SCHEMA,
        ),
        metadata_path,
    )

    out = _process(data_path, metadata_path)
    rows = out.data.to_pylist()
    assert [row["sample_id"] for row in rows] == ["docA", "docA", "docB"]
    assert [row["position"] for row in rows] == [0, 1, 0]
    assert [row["modality"] for row in rows] == ["text", "image", "text"]
    assert _metadata_by_id(out) == {"docA": '{"src":"a"}', "docB": '{"src":"b"}'}


@pytest.mark.parametrize(
    ("metadata_path_factory", "metadata_task_data"),
    [
        (None, None),
        (lambda p: p / "missing.metadata.parquet", None),
        (None, []),
    ],
)
def test_parquet_multimodal_reader_stage_metadata_fallback_paths(
    tmp_path: Path,
    metadata_path_factory: Callable[[Path], Path] | None,
    metadata_task_data: list[str] | None,
) -> None:
    data_path = tmp_path / "batch_missing_or_none_meta.parquet"
    _write_data_parquet(data_path)
    metadata_path = metadata_path_factory(tmp_path) if metadata_path_factory is not None else None
    out = _process(data_path, metadata_path, metadata_task_data=metadata_task_data)
    assert out.metadata_index is not None
    assert _metadata_by_id(out) == {"docA": None, "docB": None}


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
    metadata_path = tmp_path / "columns_selected.metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["docA", "docB"],
                "sample_type": ["custom_type_a", "custom_type_b"],
                "metadata_json": ['{"src":"a"}', '{"src":"b"}'],
                "extra_meta": ["x", "y"],
            }
        ),
        metadata_path,
    )

    out = _process(
        data_path,
        metadata_path,
        reader=ParquetMultimodalReaderStage(
        columns=list(MULTIMODAL_SCHEMA.names),
        metadata_columns=["sample_id", "metadata_json"],
        ),
    )
    assert out.data.column_names == MULTIMODAL_SCHEMA.names
    assert set(METADATA_SCHEMA.names).issubset(set(out.metadata_index.column_names))
    metadata_rows = {str(row["sample_id"]): row["metadata_json"] for row in out.metadata_index.to_pylist()}
    assert metadata_rows == {"docA": '{"src":"a"}', "docB": '{"src":"b"}'}


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


def test_parquet_multimodal_reader_infers_sample_type_for_multimodality_patterns(tmp_path: Path) -> None:
    data_path = tmp_path / "sample_types.parquet"
    table = pa.table(
        {
            "sample_id": ["txt", "txt", "img", "img", "mix", "mix", "mix"],
            "position": [0, 1, 0, 1, 0, 1, 2],
            "modality": ["text", "text", "image", "image", "text", "image", "text"],
            "content_type": [
                "text/plain",
                "text/plain",
                "image/jpeg",
                "image/jpeg",
                "text/plain",
                "image/jpeg",
                "text/plain",
            ],
            "text_content": ["a", "b", None, None, "x", None, "y"],
            "binary_content": [None, None, b"i0", b"i1", None, b"im", None],
            "element_metadata_json": [None, None, None, None, None, None, None],
            "source_id": ["src"] * 7,
            "source_shard": ["shard"] * 7,
            "content_path": [None] * 7,
            "content_key": [None, None, "img.0.jpg", "img.1.jpg", None, "mix.1.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    pq.write_table(table, data_path)

    out = _process(data_path)
    sample_type_by_id = {str(row["sample_id"]): str(row["sample_type"]) for row in out.metadata_index.to_pylist()}
    assert sample_type_by_id == {
        "img": "multi_image",
        "mix": "interleaved",
        "txt": "multi_text",
    }


def test_parquet_multimodal_reader_infers_two_row_text_image_as_interleaved(tmp_path: Path) -> None:
    data_path = tmp_path / "sample_type_pair_retired.parquet"
    table = pa.table(
        {
            "sample_id": ["doc", "doc"],
            "position": [0, 1],
            "modality": ["text", "image"],
            "content_type": ["text/plain", "image/jpeg"],
            "text_content": ["caption", None],
            "binary_content": [None, b"img"],
            "element_metadata_json": [None, None],
            "source_id": ["src", "src"],
            "source_shard": ["shard", "shard"],
            "content_path": [None, None],
            "content_key": [None, "doc.1.jpg"],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    pq.write_table(table, data_path)
    out = _process(data_path)
    metadata_rows = out.metadata_index.to_pylist()
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["sample_type"] == "interleaved"


def test_base_multimodal_reader_text_row_allows_source_id_override() -> None:
    class _Reader(BaseMultimodalReaderStage):
        def read_data(self, data_path: str, metadata_path: str | None) -> tuple[pa.Table, pa.Table]:
            _ = data_path, metadata_path
            return self._empty_data_table(), self._empty_metadata_table()

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
