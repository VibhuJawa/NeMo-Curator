from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.multimodal import (
    MultimodalWriterStage,
    WebDatasetReaderStage,
)
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA, MultimodalBatch


def _sample_task(task_id: str = "t0") -> MultimodalBatch:
    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc"],
            "position": [0, 1, 2],
            "modality": ["text", "image", "text"],
            "content_type": ["text/plain", "image/jpeg", "text/plain"],
            "text_content": ["alpha", None, "omega"],
            "binary_content": [None, b"img", None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard", "shard", "shard"],
            "content_path": [None, None, None],
            "content_key": [None, "doc.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    return MultimodalBatch(task_id=task_id, dataset_name="ds", data=table)


def _read_output_rows(out: Path, output_format: str) -> list[dict[str, object]]:
    if output_format == "parquet":
        return pq.read_table(out).to_pylist()
    with pa.memory_map(str(out), "r") as source:
        return pa.ipc.open_file(source).read_all().to_pylist()


def _webdataset_task(task_id: str) -> MultimodalBatch:
    return MultimodalBatch(
        task_id=task_id,
        dataset_name="ds",
        data=pa.table(
            {
                "sample_id": ["doc", "doc"],
                "position": [0, 1],
                "modality": ["text", "image"],
                "content_type": ["text/plain", "image/jpeg"],
                "text_content": ["caption", None],
                "binary_content": [None, b"jpg-bytes"],
                "source_id": ["src", "src"],
                "source_shard": ["shard", "shard"],
                "content_path": [None, None],
                "content_key": [None, "doc.jpg"],
            },
            schema=MULTIMODAL_SCHEMA,
        ),
    )


def _read_tar_members(out: Path) -> tuple[list[str], dict[str, bytes]]:
    with tarfile.open(out, "r") as tf:
        names = [m.name for m in tf.getmembers()]
        return names, {m.name: tf.extractfile(m).read() for m in tf.getmembers()}


@pytest.mark.parametrize(
    ("writer_kwargs", "output_format"),
    [
        ({"output_path": "out.parquet"}, "parquet"),
        ({"output_path": "out.arrow", "output_format": "arrow"}, "arrow"),
    ],
)
def test_writer_roundtrip_formats(
    tmp_path: Path,
    writer_kwargs: dict[str, str],
    output_format: str,
) -> None:
    kwargs = {
        key: str(tmp_path / value) if key == "output_path" else value
        for key, value in writer_kwargs.items()
    }
    stage = MultimodalWriterStage(**kwargs)
    result = stage.process(_sample_task(task_id="t0"))

    assert len(result.data) == 2
    output_file = Path(result.data[0])
    metadata_file = Path(result.data[1])
    assert output_file.exists()
    assert metadata_file.exists()
    assert result._metadata["data_output_path"] == str(output_file)
    assert result._metadata["metadata_output_path"] == str(metadata_file)

    rows = sorted(_read_output_rows(output_file, output_format), key=lambda r: int(r["position"]))
    assert set(rows[0]) == set(MULTIMODAL_SCHEMA.names)
    assert {r["modality"] for r in rows} == {"image", "text"}
    assert [r["text_content"] for r in rows] == ["alpha", None, "omega"]
    assert [r["content_key"] for r in rows] == [None, "doc.jpg", None]

    metadata_rows = (
        pq.read_table(metadata_file).to_pylist()
        if metadata_file.suffix == ".parquet"
        else _read_output_rows(metadata_file, "arrow")
    )
    assert metadata_rows == []


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"output_path": "out.any", "output_format": "csv"}, "Unsupported output_format"),
        ({}, "requires output_path"),
    ],
)
def test_writer_validation_errors(tmp_path: Path, kwargs: dict[str, str], error_match: str) -> None:
    resolved = {
        key: str(tmp_path / value) if key == "output_path" else value
        for key, value in kwargs.items()
    }
    with pytest.raises(ValueError, match=error_match):
        MultimodalWriterStage(**resolved)


def test_writer_rejects_invalid_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        MultimodalWriterStage(
            output_path=str(tmp_path / "out.parquet"),
            output_format="parquet",
            mode="ignore",  # type: ignore[arg-type]
        )


def test_writer_outputs_are_isolated_per_task(tmp_path: Path) -> None:
    stage = MultimodalWriterStage(output_path=str(tmp_path / "out.parquet"), output_format="parquet")
    t0 = stage.process(_sample_task(task_id="task-0"))
    t1 = stage.process(_sample_task(task_id="task-1"))
    assert t0.data[0] != t1.data[0]
    assert t0.data[1] != t1.data[1]
    assert Path(t0.data[0]).exists()
    assert Path(t0.data[1]).exists()
    assert Path(t1.data[0]).exists()
    assert Path(t1.data[1]).exists()


@pytest.mark.parametrize(("name", "output_format"), [("out.parquet", "parquet"), ("out.arrow", "arrow")])
def test_writer_always_writes_metadata_index_when_present(tmp_path: Path, name: str, output_format: str) -> None:
    task = _sample_task(task_id="meta")
    task.metadata_index = pa.table(
        {
            "sample_id": ["s1", "s0"],
            "sample_type": ["doc", "img"],
            "metadata_json": ['{"k":1}', '{"k":2}'],
        }
    )

    result = MultimodalWriterStage(output_path=str(tmp_path / name), output_format=output_format).process(task)
    metadata_file = Path(result.data[1])
    assert metadata_file.exists()
    rows = (
        pq.read_table(metadata_file).to_pylist()
        if metadata_file.suffix == ".parquet"
        else _read_output_rows(metadata_file, "arrow")
    )
    assert [r["sample_id"] for r in rows] == ["s0", "s1"]


def test_webdataset_writer_writes_tar_members(tmp_path: Path) -> None:
    out = tmp_path / "out.tar"
    stage = MultimodalWriterStage(output_path=str(out), output_format="webdataset")
    result = stage.process(_webdataset_task(task_id="t2"))
    assert len(result.data) == 2
    output_file = Path(result.data[0])
    metadata_file = Path(result.data[1])
    assert metadata_file.suffix == ".parquet"
    assert metadata_file.exists()
    names, members = _read_tar_members(output_file)
    assert names == ["doc.000000.txt", "doc.000001.jpg"]
    assert members["doc.000000.txt"] == b"caption"
    assert members["doc.000001.jpg"] == b"jpg-bytes"


def test_webdataset_roundtrip_from_real_tar_file(tmp_path: Path) -> None:
    in_tar = tmp_path / "in.tar"
    out_tar = tmp_path / "out.tar"
    expected = {
        "docA.000000.txt": b"hello",
        "docA.000001.jpg": b"jpg-bytes",
        "docB.000000.json": b'{"caption":"json"}',
        "docB.000001.png": b"png-bytes",
    }

    with tarfile.open(in_tar, "w") as tf:
        for name, payload in {
            "docA.txt": b"hello",
            "docA.jpg": b"jpg-bytes",
            "docB.json": b'{"caption":"json"}',
            "docB.png": b"png-bytes",
        }.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))

    task = FileGroupTask(task_id="t4", dataset_name="ds", data=[str(in_tar)])
    batch = WebDatasetReaderStage(load_binary=False, sample_format="auto").process(task)
    result = MultimodalWriterStage(output_path=str(out_tar), output_format="webdataset").process(batch)

    names, members = _read_tar_members(Path(result.data[0]))
    assert names == sorted(expected)
    assert members == expected


def test_webdataset_writer_allows_same_suffix_by_position(tmp_path: Path) -> None:
    out = tmp_path / "dup.tar"
    table = pa.table(
        {
            "sample_id": ["doc", "doc"],
            "position": [0, 1],
            "modality": ["text", "text"],
            "content_type": ["text/plain", "text/plain"],
            "text_content": ["a", "b"],
            "binary_content": [None, None],
            "source_id": ["src", "src"],
            "source_shard": ["shard", "shard"],
            "content_path": [None, None],
            "content_key": [None, None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultimodalBatch(task_id="t5", dataset_name="ds", data=table)
    result = MultimodalWriterStage(output_path=str(out), output_format="webdataset").process(task)
    names, members = _read_tar_members(Path(result.data[0]))
    assert names == ["doc.000000.txt", "doc.000001.txt"]
    assert members["doc.000000.txt"] == b"a"
    assert members["doc.000001.txt"] == b"b"
