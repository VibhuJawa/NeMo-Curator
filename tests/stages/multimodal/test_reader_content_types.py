from __future__ import annotations

import json
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING

import fsspec
import pyarrow as pa
import pytest
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
    out = WebDatasetReaderStage(load_binary=False, sample_format="auto").process(task)

    by_id = {str(r["sample_id"]): str(r["content_type"]) for r in out.data.to_pylist() if r["modality"] == "image"}
    assert by_id == {"a": "image/jpeg", "b": "image/png", "c": "image/tiff"}
    text_by_id = {str(r["sample_id"]): str(r["content_type"]) for r in out.data.to_pylist() if r["modality"] == "text"}
    assert text_by_id["d"] == "application/json"


def test_reader_can_split_output_batches_by_max_batch_bytes(tmp_path: Path) -> None:
    tar_a = tmp_path / "a.tar"
    tar_b = tmp_path / "b.tar"
    _write_members(tar_a, {"a.txt": b"alpha"})
    _write_members(tar_b, {"b.txt": b"beta"})

    task = FileGroupTask(task_id="split", dataset_name="ds", data=[str(tar_a), str(tar_b)])
    out = WebDatasetReaderStage(load_binary=False, max_batch_bytes=1).process(task)
    assert isinstance(out, list)
    assert len(out) == 2
    assert [batch.task_id for batch in out] == ["split.part_00000", "split.part_00001"]
    first_rows = out[0].data.to_pylist()
    second_rows = out[1].data.to_pylist()
    assert {row["sample_id"] for row in first_rows} == {"a"}
    assert {row["sample_id"] for row in second_rows} == {"b"}


def test_reader_rejects_non_positive_max_batch_bytes() -> None:
    with pytest.raises(ValueError, match="max_batch_bytes must be > 0"):
        WebDatasetReaderStage(max_batch_bytes=0)


def test_reader_split_keeps_all_rows_for_sample_in_single_batch(tmp_path: Path) -> None:
    tar_a = tmp_path / "a.tar"
    tar_b = tmp_path / "b.tar"
    _write_members(
        tar_a,
        {
            "doc.000.jpg": b"img-a",
            "other.txt": b"text-other",
        },
    )
    _write_members(
        tar_b,
        {
            "doc.001.txt": b"text-doc",
        },
    )

    task = FileGroupTask(task_id="nosplit", dataset_name="ds", data=[str(tar_a), str(tar_b)])
    out = WebDatasetReaderStage(load_binary=False, sample_format="interleaved", max_batch_bytes=1).process(task)
    assert isinstance(out, list)
    batches_with_doc = [
        batch for batch in out if any(str(row["sample_id"]) == "doc" for row in batch.data.to_pylist())
    ]
    assert len(batches_with_doc) == 1
    doc_rows = [row for row in batches_with_doc[0].data.to_pylist() if str(row["sample_id"]) == "doc"]
    assert len(doc_rows) == 2


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
    out = WebDatasetReaderStage(load_binary=False, sample_format="interleaved").process(task)

    rows = sorted(out.data.to_pylist(), key=lambda r: int(r["position"]))
    assert [r["sample_id"] for r in rows] == ["doc_1", "doc_1", "doc_1"]
    assert [r["position"] for r in rows] == [0, 1, 2]
    assert [r["modality"] for r in rows] == ["text", "image", "text"]


def test_reader_populates_metadata_index_sample_type(tmp_path: Path) -> None:
    tar_path = tmp_path / "meta.tar"
    _write_members(
        tar_path,
        {
            "pair.jpg": b"img",
            "pair.txt": b"caption",
            "single.txt": b"hello",
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False).process(task)
    metadata = {str(row["sample_id"]): str(row["sample_type"]) for row in out.metadata_index.to_pylist()}
    assert metadata["pair"] == "pair"
    assert metadata["single"] == "single"


def test_reader_preserves_directory_components_in_sample_id(tmp_path: Path) -> None:
    tar_path = tmp_path / "dirs.tar"
    _write_members(
        tar_path,
        {
            "a/doc.jpg": b"img-a",
            "a/doc.txt": b"text-a",
            "b/doc.jpg": b"img-b",
            "b/doc.txt": b"text-b",
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, sample_format="simple").process(task)
    sample_ids = sorted({str(r["sample_id"]) for r in out.data.to_pylist()})
    assert sample_ids == ["a/doc", "b/doc"]


def test_reader_sample_format_simple_assigns_pair_positions(tmp_path: Path) -> None:
    tar_path = tmp_path / "simple.tar"
    _write_members(
        tar_path,
        {
            "sample.jpg": b"img",
            "sample.txt": b"caption",
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, sample_format="simple").process(task)
    rows = {(str(r["modality"]), str(r["sample_id"])): int(r["position"]) for r in out.data.to_pylist()}
    assert rows[("image", "sample")] == 0
    assert rows[("text", "sample")] == 1


def test_reader_error_handling_raise_on_invalid_utf8_member(tmp_path: Path) -> None:
    tar_path = tmp_path / "invalid-utf8.tar"
    _write_members(tar_path, {"bad.txt": b"\xff\xfe"})

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    with pytest.raises(UnicodeDecodeError):
        WebDatasetReaderStage(load_binary=False, error_handling="raise").process(task)


def test_reader_error_handling_skip_on_invalid_utf8_member(tmp_path: Path) -> None:
    tar_path = tmp_path / "invalid-utf8-skip.tar"
    _write_members(
        tar_path,
        {
            "bad.txt": b"\xff\xfe",
            "ok.txt": b"hello",
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, error_handling="skip").process(task)
    rows = out.data.to_pylist()
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "ok"


def test_reader_rejects_unknown_binary_content_type_raise(tmp_path: Path) -> None:
    tar_path = tmp_path / "unknown-bin-raise.tar"
    _write_members(
        tar_path,
        {
            "doc.bin": b"\x00\x01",
            "doc.txt": b"caption",
        },
    )
    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    with pytest.raises(ValueError, match="Unsupported content_type"):
        WebDatasetReaderStage(load_binary=False, error_handling="raise").process(task)


def test_reader_skips_unknown_binary_content_type_with_skip_policy(tmp_path: Path) -> None:
    tar_path = tmp_path / "unknown-bin-skip.tar"
    _write_members(
        tar_path,
        {
            "doc.bin": b"\x00\x01",
            "doc.txt": b"caption",
        },
    )
    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, error_handling="skip").process(task)
    rows = out.data.to_pylist()
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "doc"
    assert rows[0]["modality"] == "text"


def test_reader_interleaved_json_requires_content_key_raise(tmp_path: Path) -> None:
    tar_path = tmp_path / "interleaved-key-required-raise.tar"
    _write_members(
        tar_path,
        {
            "bundle.json": (
                b'{"sample_id":"doc1","segments":[{"modality":"text","text":"caption"},{"modality":"image"}]}'
            ),
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    with pytest.raises(ValueError, match="must include non-empty string 'content_key'"):
        WebDatasetReaderStage(load_binary=False, sample_format="interleaved", error_handling="raise").process(task)


def test_reader_interleaved_json_requires_content_key_skip(tmp_path: Path) -> None:
    tar_path = tmp_path / "interleaved-key-required-skip.tar"
    _write_members(
        tar_path,
        {
            "bad.json": (
                b'{"sample_id":"doc1","segments":[{"modality":"text","text":"caption"},{"modality":"image"}]}'
            ),
            "ok.txt": b"fallback",
        },
    )

    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, sample_format="interleaved", error_handling="skip").process(task)
    rows = out.data.to_pylist()
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "ok"


def test_reader_interleaved_json_supports_custom_field_map(tmp_path: Path) -> None:
    tar_path = tmp_path / "interleaved-custom-map.tar"
    _write_members(
        tar_path,
        {
            "mapped.json": (
                b'{"sid":"doc1","chunks":[{"kind":"text","body":"caption"},{"kind":"image","path":"doc1.000.jpg"}]}'
            ),
        },
    )
    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(
        load_binary=False,
        sample_format="interleaved",
        interleaved_field_map={
            "sample_id": "sid",
            "segments": "chunks",
            "modality": "kind",
            "text": "body",
            "content_key": "path",
        },
    ).process(task)

    rows = sorted(out.data.to_pylist(), key=lambda r: int(r["position"]))
    assert [r["modality"] for r in rows] == ["text", "image"]
    assert rows[0]["text_content"] == "caption"
    assert rows[1]["content_key"] == "doc1.000.jpg"


def test_reader_metadata_policy_first_json_wins(tmp_path: Path) -> None:
    tar_path = tmp_path / "metadata-first.tar"
    _write_members(
        tar_path,
        {
            "doc.json": b'{"a":1}',
            "doc.001.json": b'{"a":2}',
        },
    )
    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, sample_format="auto").process(task)
    metadata_by_id = {str(r["sample_id"]): str(r["metadata_json"]) for r in out.metadata_index.to_pylist()}
    assert json.loads(metadata_by_id["doc"]) == {"a": 1}


def test_reader_accepts_partial_interleaved_field_map_overrides() -> None:
    stage = WebDatasetReaderStage(interleaved_field_map={"sample_id": "sid", "segments": "chunks"})
    assert stage.interleaved_field_map["sample_id"] == "sid"
    assert stage.interleaved_field_map["segments"] == "chunks"
    assert stage.interleaved_field_map["modality"] == "modality"
    assert stage.interleaved_field_map["text"] == "text"
    assert stage.interleaved_field_map["content_key"] == "content_key"


def test_reader_default_interleaved_field_map_returns_copy() -> None:
    defaults = WebDatasetReaderStage.default_interleaved_field_map()
    defaults["sample_id"] = "mutated"
    fresh = WebDatasetReaderStage.default_interleaved_field_map()
    assert fresh["sample_id"] == "sample_id"


def test_reader_rejects_unknown_interleaved_field_map_keys() -> None:
    with pytest.raises(ValueError, match="interleaved_field_map has unknown keys"):
        WebDatasetReaderStage(interleaved_field_map={"unknown_key": "value"})


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


def test_materialize_supports_fsspec_memory_paths() -> None:
    fs = fsspec.filesystem("memory")
    fs.makedirs("unit", exist_ok=True)
    payload = b"memory-bytes"
    with fs.open("unit/blob.bin", "wb") as f:
        f.write(payload)

    table = pa.table(
        {
            "sample_id": ["doc"],
            "position": [0],
            "modality": ["image"],
            "content_type": ["application/octet-stream"],
            "text_content": [None],
            "binary_content": [None],
            "source_id": ["src"],
            "source_shard": ["shard"],
            "content_path": ["memory://unit/blob.bin"],
            "content_key": [None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    batch = MultimodalBatch(task_id="mem", dataset_name="ds", data=table)
    out = batch.materialize(modality="image")
    assert out.data["binary_content"].to_pylist() == [payload]


def test_reader_propagates_storage_options_to_batch_metadata(tmp_path: Path) -> None:
    tar_path = tmp_path / "meta.tar"
    _write_members(tar_path, {"doc.txt": b"hello"})
    opts = {"anon": True}
    out = WebDatasetReaderStage(load_binary=False, storage_options=opts).process(
        FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    )
    assert out._metadata["storage_options"] == opts


def test_materialize_from_memory_tar_path_with_explicit_storage_options() -> None:
    fs = fsspec.filesystem("memory")
    fs.makedirs("unit", exist_ok=True)
    payload = b"tar-image-bytes"
    with fs.open("unit/shard.tar", "wb") as raw, tarfile.open(fileobj=raw, mode="w") as tf:
        info = tarfile.TarInfo(name="doc.000.jpg")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

    table = pa.table(
        {
            "sample_id": ["doc"],
            "position": [0],
            "modality": ["image"],
            "content_type": ["image/jpeg"],
            "text_content": [None],
            "binary_content": [None],
            "source_id": ["src"],
            "source_shard": ["shard"],
            "content_path": ["memory://unit/shard.tar"],
            "content_key": ["doc.000.jpg"],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    batch = MultimodalBatch(task_id="t0", dataset_name="ds", data=table)
    out = batch.materialize(modality="image", storage_options={})
    assert out.data["binary_content"].to_pylist() == [payload]
