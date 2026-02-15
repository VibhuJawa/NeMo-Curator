from __future__ import annotations

import tarfile
from io import BytesIO
from typing import TYPE_CHECKING

import pyarrow as pa
from PIL import Image

from nemo_curator.stages.multimodal import WebDatasetReaderStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA, MultimodalBatch

if TYPE_CHECKING:
    from pathlib import Path


def _write_members(tar_path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))


def _dummy_jpeg(color: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (1, 1), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_reader_sets_content_type_from_image_extension(tmp_path: Path) -> None:
    tar_path = tmp_path / "content_types.tar"
    _write_members(
        tar_path,
        {
            "a.jpg": b"jpg",
            "a.txt": b"caption a",
            "b.png": b"png",
            "b.txt": b"caption b",
            "c.tiff": b"tiff",
            "c.txt": b"caption c",
            "d.json": b'{"caption": "json"}',
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False).process(task)

    by_id = {str(r["sample_id"]): str(r["content_type"]) for r in out.data.to_pylist() if r["modality"] == "image"}
    assert by_id == {"a": "image/jpeg", "b": "image/png", "c": "image/tiff"}
    text_by_id = {str(r["sample_id"]): str(r["content_type"]) for r in out.data.to_pylist() if r["modality"] == "text"}
    assert text_by_id["d"] == "application/json"


def test_reader_preserves_indexed_positions_for_interleaved_members(tmp_path: Path) -> None:
    tar_path = tmp_path / "indexed.tar"
    _write_members(
        tar_path,
        {
            "doc_1.000.txt": b"alpha",
            "doc_1.001.jpg": b"jpg",
            "doc_1.002.txt": b"omega",
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False).process(task)

    rows = sorted(out.data.to_pylist(), key=lambda r: int(r["position"]))
    assert [r["sample_id"] for r in rows] == ["doc_1", "doc_1", "doc_1"]
    assert [r["position"] for r in rows] == [0, 1, 2]
    assert [r["modality"] for r in rows] == ["text", "image", "text"]


def test_materialize_and_dematerialize_roundtrip(tmp_path: Path) -> None:
    tar_path = tmp_path / "shard-0.tar"
    file_path = tmp_path / "img.jpg"
    tar_jpeg_1 = _dummy_jpeg((255, 0, 0))
    tar_jpeg_2 = _dummy_jpeg((0, 255, 0))
    file_jpeg = _dummy_jpeg((0, 0, 255))
    _write_members(
        tar_path,
        {
            "doc_1.000.jpg": tar_jpeg_1,
            "doc_2.000.jpg": tar_jpeg_2,
        },
    )
    file_path.write_bytes(file_jpeg)

    table = pa.table(
        {
            "sample_id": ["doc_1", "doc_2", "doc_3", "doc_4"],
            "position": [0, 0, 0, 0],
            "modality": ["image", "image", "image", "image"],
            "content_type": ["image/jpeg", "image/jpeg", "image/jpeg", "image/jpeg"],
            "text_content": [None, None, None, None],
            "binary_content": [None, None, None, None],
            "source_id": ["s1", "s2", "s3", "s4"],
            "source_shard": ["shard-0", "shard-0", "files", "files"],
            "content_path": [str(tar_path), str(tar_path), str(file_path), str(file_path)],
            "content_key": ["doc_1.000.jpg", "doc_2.000.jpg", None, None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    batch = MultimodalBatch(task_id="t0", dataset_name="ds", data=table)
    materialized = batch.materialize(modality="image")
    assert materialized.data["binary_content"].to_pylist() == [
        tar_jpeg_1,
        tar_jpeg_2,
        file_jpeg,
        file_jpeg,
    ]
    assert materialized.is_lazy is False

    dematerialized = materialized.dematerialize(modality="image")
    assert dematerialized.data["binary_content"].to_pylist() == [None, None, None, None]
