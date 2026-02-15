from __future__ import annotations

import tarfile
from io import BytesIO
from typing import TYPE_CHECKING

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.multimodal import (
    MetadataWriterStage,
    MultimodalWriterStage,
    WebDatasetReaderStage,
)
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA, MultimodalBatch

if TYPE_CHECKING:
    from pathlib import Path


def _sample_task() -> MultimodalBatch:
    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc"],
            "position": [0, 1, 2],
            "modality": ["text", "image", "text"],
            "content_type": ["text/plain", "image/jpeg", "text/plain"],
            "text_content": ["alpha", None, "omega"],
            "binary_content": [None, None, None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard", "shard", "shard"],
            "content_path": [None, "/path/to/shard.tar", None],
            "content_key": [None, "doc.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    return MultimodalBatch(task_id="t0", dataset_name="ds", data=table)


def _read_output_rows(out: Path, output_format: str) -> list[dict[str, object]]:
    if output_format == "parquet":
        return pq.read_table(out).to_pylist()
    if output_format == "arrow":
        with pa.memory_map(str(out), "r") as source:
            return pa.ipc.open_file(source).read_all().to_pylist()
    return lance.dataset(str(out)).to_table().to_pylist()


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
    ("name", "writer_kwargs", "output_format"),
    [
        ("out.parquet", {"output_parquet": "out.parquet"}, "parquet"),
        ("out.arrow", {"output_path": "out.arrow", "output_format": "arrow"}, "arrow"),
        ("out.lance", {"output_path": "out.lance", "output_format": "lance"}, "lance"),
    ],
)
def test_writer_roundtrip_formats(
    tmp_path: Path,
    name: str,
    writer_kwargs: dict[str, str],
    output_format: str,
) -> None:
    out = tmp_path / name
    kwargs = {
        key: str(tmp_path / value) if key in {"output_path", "output_parquet"} else value
        for key, value in writer_kwargs.items()
    }
    stage = MultimodalWriterStage(**kwargs)
    result = stage.process(_sample_task())

    assert result.data == [str(out)]
    rows = sorted(_read_output_rows(out, output_format), key=lambda r: int(r["position"]))
    assert {r["modality"] for r in rows} == {"image", "text"}
    assert [r["text"] for r in rows] == ["alpha", None, "omega"]


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"output_path": "out.any", "output_format": "csv"}, "Unsupported output_format"),
        ({}, "requires output_path"),
    ],
)
def test_writer_validation_errors(tmp_path: Path, kwargs: dict[str, str], error_match: str) -> None:
    resolved = {
        key: str(tmp_path / value) if key in {"output_path", "output_parquet"} else value
        for key, value in kwargs.items()
    }
    with pytest.raises(ValueError, match=error_match):
        MultimodalWriterStage(**resolved)


@pytest.mark.parametrize(("name", "output_format"), [("metadata.parquet", "parquet"), ("metadata.lance", "lance")])
def test_metadata_writer_writes_metadata_index(tmp_path: Path, name: str, output_format: str) -> None:
    out = tmp_path / name
    task = _sample_task()
    task.metadata_index = pa.table(
        {
            "sample_id": ["s1", "s0"],
            "sample_type": ["doc", "img"],
            "metadata_json": ['{"k":1}', '{"k":2}'],
        }
    )

    result = MetadataWriterStage(output_path=str(out), output_format=output_format).process(task)
    assert result.data == [str(out)]
    rows = pq.read_table(out).to_pylist() if output_format == "parquet" else lance.dataset(str(out)).to_table().to_pylist()
    assert [r["sample_id"] for r in rows] == ["s0", "s1"]


def test_metadata_writer_requires_metadata_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metadata_index"):
        MetadataWriterStage(output_path=str(tmp_path / "metadata.parquet"), output_format="parquet").process(_sample_task())


def test_webdataset_writer_writes_tar_members(tmp_path: Path) -> None:
    name = "out.tar"
    out = tmp_path / name
    stage = MultimodalWriterStage(output_path=str(out), output_format="webdataset")
    result = stage.process(_webdataset_task(task_id="t2"))
    assert result.data == [str(out)]
    names, members = _read_tar_members(out)
    assert names == ["doc.txt", "doc.jpg"]
    assert members["doc.txt"] == b"caption"
    assert members["doc.jpg"] == b"jpg-bytes"


def test_webdataset_roundtrip_from_real_tar_file(tmp_path: Path) -> None:
    in_tar = tmp_path / "in.tar"
    out_tar = tmp_path / "out.tar"
    expected = {
        "docA.txt": b"hello",
        "docA.jpg": b"jpg-bytes",
        "docB.json": b'{"caption":"json"}',
        "docB.png": b"png-bytes",
    }

    with tarfile.open(in_tar, "w") as tf:
        for name, payload in expected.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))

    task = FileGroupTask(task_id="t4", dataset_name="ds", data=[str(in_tar)])
    batch = WebDatasetReaderStage(load_binary=False).process(task)
    result = MultimodalWriterStage(output_path=str(out_tar), output_format="webdataset").process(batch)
    assert result.data == [str(out_tar)]

    names, members = _read_tar_members(out_tar)
    assert names == ["docA.txt", "docA.jpg", "docB.json", "docB.png"]
    assert members == expected


def test_webdataset_writer_rejects_duplicate_suffix_per_sample(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="Duplicate webdataset suffix"):
        MultimodalWriterStage(output_path=str(out), output_format="webdataset").process(task)
