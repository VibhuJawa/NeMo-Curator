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
from pathlib import Path

import pyarrow as pa

from nemo_curator.stages.multimodal.io.writers.webdataset import MultimodalWebdatasetWriterStage
from nemo_curator.tasks import MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

from .test_data_gen import generate_jpeg_bytes


def _make_task(num_samples: int = 2) -> MultiBatchTask:
    rows = []
    for i in range(num_samples):
        sid = f"sample_{i:03d}"
        img_bytes = generate_jpeg_bytes(seed=i)
        metadata_json = json.dumps({
            "pdf_name": f"doc_{i}.pdf",
            "texts": [f"text_{i}_0", f"text_{i}_1"],
            "images": [f"{sid}.jpg"],
            "score": 0.5 + i * 0.1,
        })
        rows.extend([
            {
                "sample_id": sid, "position": -1, "modality": "metadata",
                "content_type": "application/json", "text_content": None,
                "binary_content": None, "source_ref": None,
                "metadata_json": metadata_json, "materialize_error": None,
            },
            {
                "sample_id": sid, "position": 0, "modality": "text",
                "content_type": "text/plain", "text_content": f"text_{i}_0",
                "binary_content": None, "source_ref": None,
                "metadata_json": None, "materialize_error": None,
            },
            {
                "sample_id": sid, "position": 1, "modality": "text",
                "content_type": "text/plain", "text_content": f"text_{i}_1",
                "binary_content": None, "source_ref": None,
                "metadata_json": None, "materialize_error": None,
            },
            {
                "sample_id": sid, "position": 0, "modality": "image",
                "content_type": "image/jpeg", "text_content": None,
                "binary_content": img_bytes, "source_ref": None,
                "metadata_json": None, "materialize_error": None,
            },
        ])

    table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
    return MultiBatchTask(
        task_id="test_task",
        dataset_name="test_dataset",
        data=table,
        _metadata={"source_files": ["test.parquet"]},
    )


def test_writer_produces_valid_tar(tmp_path: Path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalWebdatasetWriterStage(
        path=str(out_dir), materialize_on_write=False, mode="overwrite",
    )
    result = writer.process(_make_task(num_samples=2))

    assert len(result.data) == 1
    tar_path = result.data[0]
    assert tar_path.endswith(".tar")

    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        names = {m.name for m in members}
        assert len(names) >= 4  # 2 samples * (json + jpg)

        json_members = [n for n in names if n.endswith(".json")]
        assert len(json_members) == 2


def test_writer_preserves_text_content(tmp_path: Path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalWebdatasetWriterStage(
        path=str(out_dir), materialize_on_write=False, mode="overwrite",
    )
    writer.process(_make_task(num_samples=1))

    tar_files = list(out_dir.glob("*.tar"))
    assert len(tar_files) == 1

    with tarfile.open(tar_files[0], "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json"):
                payload = json.load(tf.extractfile(member))
                assert "texts" in payload
                assert payload["texts"] == ["text_0_0", "text_0_1"]
                assert payload["images"] == ["0.jpg", None]


def test_writer_preserves_image_bytes(tmp_path: Path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalWebdatasetWriterStage(
        path=str(out_dir), materialize_on_write=False, mode="overwrite",
    )
    writer.process(_make_task(num_samples=1))

    expected_bytes = generate_jpeg_bytes(seed=0)
    tar_files = list(out_dir.glob("*.tar"))
    with tarfile.open(tar_files[0], "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".jpg"):
                actual_bytes = tf.extractfile(member).read()
                assert actual_bytes == expected_bytes


def test_writer_handles_no_binary_content(tmp_path: Path):
    rows = [
        {
            "sample_id": "s1", "position": -1, "modality": "metadata",
            "content_type": "application/json", "text_content": None,
            "binary_content": None, "source_ref": None,
            "metadata_json": json.dumps({"texts": ["hi"], "images": [None]}),
            "materialize_error": None,
        },
        {
            "sample_id": "s1", "position": 0, "modality": "text",
            "content_type": "text/plain", "text_content": "hi",
            "binary_content": None, "source_ref": None,
            "metadata_json": None, "materialize_error": None,
        },
        {
            "sample_id": "s1", "position": 0, "modality": "image",
            "content_type": "image/jpeg", "text_content": None,
            "binary_content": None, "source_ref": None,
            "metadata_json": None, "materialize_error": None,
        },
    ]
    table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
    task = MultiBatchTask(
        task_id="no_binary", dataset_name="test", data=table,
        _metadata={"source_files": ["x"]},
    )

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalWebdatasetWriterStage(
        path=str(out_dir), materialize_on_write=False, mode="overwrite",
    )
    writer.process(task)

    tar_files = list(out_dir.glob("*.tar"))
    with tarfile.open(tar_files[0], "r") as tf:
        names = {m.name for m in tf.getmembers() if m.isfile()}
        jpg_members = [n for n in names if n.endswith(".jpg")]
        assert len(jpg_members) == 0

        json_members = [n for n in names if n.endswith(".json")]
        assert len(json_members) == 1
        for m in tf.getmembers():
            if m.name.endswith(".json"):
                payload = json.load(tf.extractfile(m))
                assert payload["images"] == [None]


def test_writer_preserves_extra_columns(tmp_path: Path):
    """Extra (non-schema) columns must be round-tripped in _row_extra / _metadata_extra."""
    import pandas as pd

    img_bytes = generate_jpeg_bytes(seed=0)
    df = pd.DataFrame([
        {
            "sample_id": "s1", "position": -1, "modality": "metadata",
            "content_type": "application/json", "text_content": None,
            "binary_content": None, "source_ref": None,
            "metadata_json": json.dumps({"url": "https://example.com"}),
            "materialize_error": None,
            "nv_width": None, "nv_height": None, "match_status": None,
            "custom_score": 0.95,
        },
        {
            "sample_id": "s1", "position": 0, "modality": "image",
            "content_type": "image/jpeg", "text_content": None,
            "binary_content": img_bytes, "source_ref": None,
            "metadata_json": None, "materialize_error": None,
            "nv_width": 100, "nv_height": 80, "match_status": "matched",
            "custom_score": None,
        },
        {
            "sample_id": "s1", "position": 1, "modality": "text",
            "content_type": "text/plain", "text_content": "caption",
            "binary_content": None, "source_ref": None,
            "metadata_json": None, "materialize_error": None,
            "nv_width": None, "nv_height": None, "match_status": None,
            "custom_score": 0.7,
        },
    ])
    task = MultiBatchTask(
        task_id="extra_cols", dataset_name="test", data=df,
        _metadata={"source_files": ["test.parquet"]},
    )

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalWebdatasetWriterStage(
        path=str(out_dir), materialize_on_write=False, mode="overwrite",
    )
    writer.process(task)

    tar_files = list(out_dir.glob("*.tar"))
    with tarfile.open(tar_files[0], "r") as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".json"):
                continue
            payload = json.load(tf.extractfile(m))

            assert "_row_extra" in payload, "Missing _row_extra in JSON payload"
            row_extra = payload["_row_extra"]
            assert "text" in row_extra, "Missing text key in _row_extra"
            assert "image" in row_extra, "Missing image key in _row_extra"
            assert "metadata" in row_extra, "Missing metadata key in _row_extra"

            assert len(row_extra["text"]) == 2, f"Expected 2 text entries, got {len(row_extra['text'])}"
            assert len(row_extra["image"]) == 2, f"Expected 2 image entries, got {len(row_extra['image'])}"

            # pos 0: image row has nv_width=100, text row has nothing special
            img_extra = row_extra["image"][0]
            assert img_extra is not None
            assert img_extra["nv_width"] == 100
            assert img_extra["nv_height"] == 80
            assert img_extra["match_status"] == "matched"

            # pos 1: text row has custom_score=0.7, no image
            txt_extra = row_extra["text"][1]
            assert txt_extra is not None
            assert txt_extra["nv_width"] is None
            assert txt_extra["custom_score"] == 0.7
            assert row_extra["image"][1] is None

            # metadata row
            meta_extra = row_extra["metadata"]
            assert meta_extra["custom_score"] == 0.95
            assert meta_extra["match_status"] is None
