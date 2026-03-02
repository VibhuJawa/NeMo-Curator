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

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.stages.multimodal.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask


def _as_df(task_or_tasks: MultiBatchTask | list[MultiBatchTask]) -> pd.DataFrame:
    task = task_or_tasks[0] if isinstance(task_or_tasks, list) else task_or_tasks
    return task.to_pandas()


def _write_tar_sample(
    tar_path: Path,
    payload: dict[str, object],
    *,
    json_name: str = "sample.json",
    image_name: str = "image.jpg",
    image_bytes: bytes = b"abc",
) -> None:
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name=json_name)
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))
        img_info = tarfile.TarInfo(name=image_name)
        img_info.size = len(image_bytes)
        tf.addfile(img_info, BytesIO(image_bytes))


def _task_for_tar(tar_path: Path, task_id: str) -> FileGroupTask:
    return FileGroupTask(
        task_id=task_id,
        dataset_name="custom_dataset",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )


def test_reader_supports_custom_field_mapping(tmp_path: Path) -> None:
    tar_path = tmp_path / "alt-shard-00000.tar"
    payload = {
        "doc_id": "doc-custom",
        "source_doc": "custom.pdf",
        "captions": ["a", "b"],
        "frames": ["custom-image.jpg"],
        "primary_image": "custom-image.jpg",
        "p_hash": "abc123",
    }
    image_bytes = b"custom-image-bytes"
    _write_tar_sample(
        tar_path,
        payload,
        json_name="sample-xyz.meta.json",
        image_name="custom-image.jpg",
        image_bytes=image_bytes,
    )
    task = _task_for_tar(tar_path, "file_group_custom")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
        materialize_on_read=True,
        fields=("p_hash",),
    )
    df = _as_df(reader.process(task))
    assert ((df["sample_id"] == "doc-custom") & (df["modality"] == "metadata")).any()
    text_rows = df[df["modality"] == "text"]
    assert text_rows["text_content"].tolist() == ["a", "b"]
    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1
    assert image_rows.iloc[0]["binary_content"] == image_bytes
    assert "p_hash" in df.columns
    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["p_hash"] == "abc123"
    assert pd.isna(image_rows.iloc[0]["p_hash"])


def test_reader_reads_all_fields_by_default(tmp_path: Path) -> None:
    tar_path = tmp_path / "all-fields.tar"
    payload = {
        "doc_id": "doc-all",
        "source_doc": "all.pdf",
        "captions": ["hello"],
        "frames": ["image.jpg"],
        "primary_image": "image.jpg",
        "p_hash": "phash-1",
        "score": 0.91,
        "aux": {"page": 3},
    }
    _write_tar_sample(tar_path, payload, json_name="sample.meta.json")
    task = _task_for_tar(tar_path, "all_fields")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
    )
    df = _as_df(reader.process(task))
    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["p_hash"] == "phash-1"
    assert meta_row["score"] == 0.91
    assert meta_row["aux"] == json.dumps({"page": 3}, ensure_ascii=True)
    image_row = df[df["modality"] == "image"].iloc[0]
    assert pd.isna(image_row["p_hash"])
    assert "captions" not in df.columns
    assert "frames" not in df.columns


def test_reader_uses_resolved_content_key_for_content_type(tmp_path: Path) -> None:
    tar_path = tmp_path / "content-type-resolve.tar"
    payload = {
        "doc_id": "doc-ct",
        "source_doc": "ct.pdf",
        "captions": ["hello"],
        "frames": ["token.png"],
        "primary_image": "fallback.jpg",
    }
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name="sample.meta.json")
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))
        png_info = tarfile.TarInfo(name="token.png")
        png_info.size = 3
        tf.addfile(png_info, BytesIO(b"png"))
        jpg_info = tarfile.TarInfo(name="fallback.jpg")
        jpg_info.size = 3
        tf.addfile(jpg_info, BytesIO(b"jpg"))

    task = _task_for_tar(tar_path, "content_type_resolve")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
    )
    df = _as_df(reader.process(task))
    image_row = df[df["modality"] == "image"].iloc[0]
    assert image_row["content_type"] == "image/png"


def test_reader_image_tokens_with_frame_index(tmp_path: Path) -> None:
    """Non-None tokens get frame_index and resolve to default TIFF. None tokens are skipped."""
    tar_path = tmp_path / "sub-image-shard.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["text1", "text2", "text3"],
        "images": [None, "page_0_image_15", "page_1_image_22"],
    }
    _write_tar_sample(tar_path, payload, json_name="sample.json", image_name="doc.pdf.tiff", image_bytes=b"TIFF_DATA")
    task = _task_for_tar(tar_path, "sub_image_test")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name", sample_id_field="pdf_name", image_extensions=(".tiff",),
    )
    df = _as_df(reader.process(task))

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 2, "None image tokens should be skipped"

    assert image_rows.iloc[0]["position"] == 1, "First non-None image at interleaved position 1"
    assert image_rows.iloc[1]["position"] == 2, "Second non-None image at interleaved position 2"

    refs = [MultiBatchTask.parse_source_ref(v) for v in image_rows["source_ref"].tolist()]

    assert refs[0]["member"] == "doc.pdf.tiff", "Non-matching string should resolve to default TIFF"
    assert refs[0]["frame_index"] == 0, "First non-None token gets frame_index=0"

    assert refs[1]["member"] == "doc.pdf.tiff"
    assert refs[1]["frame_index"] == 1, "Second non-None token gets frame_index=1"

    text_rows = df[df["modality"] == "text"]
    assert len(text_rows) == 3
    assert text_rows["position"].tolist() == [0, 1, 2], "All text entries are strings so positions 0,1,2"


def test_reader_interleaved_positions_do_not_overlap(tmp_path: Path) -> None:
    """Parallel texts/images arrays with None placeholders produce non-overlapping positions."""
    tar_path = tmp_path / "interleaved-shard.tar"
    payload = {
        "pdf_name": "interleaved.pdf",
        "texts": ["intro text", None, "middle text", None, "conclusion"],
        "images": [None, "page_img", None, "chart_img", None],
    }
    _write_tar_sample(tar_path, payload, image_name="interleaved.pdf.jpg", image_bytes=b"\xff\xd8\xff")
    task = _task_for_tar(tar_path, "interleaved_test")
    reader = WebdatasetReaderStage(source_id_field="pdf_name", sample_id_field="pdf_name")
    df = _as_df(reader.process(task))

    text_rows = df[df["modality"] == "text"].sort_values("position")
    image_rows = df[df["modality"] == "image"].sort_values("position")

    assert text_rows["position"].tolist() == [0, 2, 4]
    assert text_rows["text_content"].tolist() == ["intro text", "middle text", "conclusion"]

    assert image_rows["position"].tolist() == [1, 3]

    text_positions = set(text_rows["position"].tolist())
    image_positions = set(image_rows["position"].tolist())
    assert text_positions.isdisjoint(image_positions), "Text and image positions must not overlap"

    all_positions = sorted(text_positions | image_positions)
    assert all_positions == [0, 1, 2, 3, 4], "Interleaved positions should cover 0..N-1 without gaps"


def test_reader_empty_output_schema_includes_requested_passthrough_fields(tmp_path: Path) -> None:
    tar_path = tmp_path / "empty-no-json.tar"
    with tarfile.open(tar_path, "w") as tf:
        img_info = tarfile.TarInfo(name="image.jpg")
        img_info.size = 3
        tf.addfile(img_info, BytesIO(b"abc"))

    task = _task_for_tar(tar_path, "empty_schema")
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=("p_hash",))
    df = _as_df(reader.process(task))
    assert "p_hash" in df.columns


@pytest.mark.parametrize(
    ("task_id", "fields", "error_pattern"),
    [
        ("missing_key", ("p_hash",), "fields not found in source sample"),
        ("reserved_key", ("sample_id",), "fields contains reserved keys"),
    ],
)
def test_reader_fields_validation_errors(
    tmp_path: Path, task_id: str, fields: tuple[str, ...], error_pattern: str
) -> None:
    tar_path = tmp_path / f"{task_id}.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, task_id)
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=fields)
    with pytest.raises(ValueError, match=error_pattern):
        _ = reader.process(task)


def test_reader_per_image_fields_distributed_to_image_rows(tmp_path: Path) -> None:
    """per_image_fields lists are distributed 1:1 to non-None image rows."""
    tar_path = tmp_path / "per-image.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello", None, "world"],
        "images": [None, "img_token", None],
        "image_metadata": [{"height": 100, "width": 200}],
    }
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, "per_image")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        per_image_fields=("image_metadata",),
    )
    df = _as_df(reader.process(task))

    assert "image_metadata" in df.columns

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1
    assert image_rows.iloc[0]["image_metadata"] == json.dumps({"height": 100, "width": 200})

    text_rows = df[df["modality"] == "text"]
    assert all(pd.isna(v) for v in text_rows["image_metadata"])

    meta_rows = df[df["modality"] == "metadata"]
    assert all(pd.isna(v) for v in meta_rows["image_metadata"])


def test_reader_per_text_fields_distributed_to_text_rows(tmp_path: Path) -> None:
    """per_text_fields lists are distributed 1:1 to non-None text rows."""
    tar_path = tmp_path / "per-text.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello", None, "world"],
        "images": [None, "img_token", None],
        "text_scores": [0.95, 0.42],
    }
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, "per_text")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        per_text_fields=("text_scores",),
    )
    df = _as_df(reader.process(task))

    assert "text_scores" in df.columns

    text_rows = df[df["modality"] == "text"].sort_values("position")
    assert len(text_rows) == 2
    assert text_rows.iloc[0]["text_scores"] == 0.95
    assert text_rows.iloc[1]["text_scores"] == 0.42

    image_rows = df[df["modality"] == "image"]
    assert all(pd.isna(v) for v in image_rows["text_scores"])

    meta_rows = df[df["modality"] == "metadata"]
    assert all(pd.isna(v) for v in meta_rows["text_scores"])


def test_reader_per_image_and_per_text_fields_together(tmp_path: Path) -> None:
    """Both per_image_fields and per_text_fields work correctly in the same reader."""
    tar_path = tmp_path / "both-per-modality.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["intro", None, "conclusion"],
        "images": [None, "page_img", None],
        "image_metadata": [{"page": 1, "width": 640}],
        "text_lang": ["en", "fr"],
        "url": "https://example.com",
    }
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, "both_per_modality")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        per_image_fields=("image_metadata",),
        per_text_fields=("text_lang",),
    )
    df = _as_df(reader.process(task))

    image_rows = df[df["modality"] == "image"]
    assert image_rows.iloc[0]["image_metadata"] == json.dumps({"page": 1, "width": 640})
    assert pd.isna(image_rows.iloc[0]["text_lang"])

    text_rows = df[df["modality"] == "text"].sort_values("position")
    assert text_rows.iloc[0]["text_lang"] == "en"
    assert text_rows.iloc[1]["text_lang"] == "fr"
    assert all(pd.isna(v) for v in text_rows["image_metadata"])

    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["url"] == "https://example.com"
    assert pd.isna(meta_row["image_metadata"])
    assert pd.isna(meta_row["text_lang"])


def test_reader_per_modality_fields_excluded_from_sample_passthrough(tmp_path: Path) -> None:
    """Fields in per_image_fields/per_text_fields must not appear on the metadata row."""
    tar_path = tmp_path / "exclude-passthrough.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["text"],
        "images": [],
        "image_metadata": [],
        "text_scores": [],
        "url": "https://example.com",
    }
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, "exclude_pt")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        per_image_fields=("image_metadata",),
        per_text_fields=("text_scores",),
    )
    df = _as_df(reader.process(task))

    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["url"] == "https://example.com"
    assert pd.isna(meta_row.get("image_metadata"))
    assert pd.isna(meta_row.get("text_scores"))


def test_reader_raises_on_non_list_per_modality_field(tmp_path: Path) -> None:
    """A per-modality field that is not a list in the source sample must raise ValueError."""
    tar_path = tmp_path / "non-list-field.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello"],
        "images": [],
        "image_metadata": "not-a-list",
    }
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, "non_list_field")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        per_image_fields=("image_metadata",),
    )
    with pytest.raises(ValueError, match="must be a list"):
        reader.process(task)
