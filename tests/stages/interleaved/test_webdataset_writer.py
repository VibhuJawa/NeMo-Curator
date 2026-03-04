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

from nemo_curator.stages.interleaved.io.writers.webdataset import InterleavedWebdatasetWriterStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


def _make_batch(num_samples: int = 2) -> InterleavedBatch:
    rows = []
    for i in range(num_samples):
        sid = f"sample_{i}"
        rows.append({"sample_id": sid, "position": -1, "modality": "metadata", "content_type": "application/json",
                      "text_content": None, "binary_content": None, "source_ref": None, "materialize_error": None})
        rows.append({"sample_id": sid, "position": 0, "modality": "text", "content_type": "text/plain",
                      "text_content": f"Hello {i}", "binary_content": None,
                      "source_ref": None, "materialize_error": None})
        rows.append({"sample_id": sid, "position": 1, "modality": "image", "content_type": "image/jpeg",
                      "text_content": None, "binary_content": b"fake-jpeg-bytes",
                      "source_ref": None, "materialize_error": None})
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id="test_batch", dataset_name="test", data=table,
                            _metadata={"source_files": ["test.tar"]})


def test_write_creates_tar(tmp_path: Path) -> None:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(_make_batch())
    assert isinstance(result, FileGroupTask)
    assert len(result.data) == 1
    assert result.data[0].endswith(".tar")
    assert Path(result.data[0]).exists()


def test_tar_contains_json_and_image_members(tmp_path: Path) -> None:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(_make_batch(num_samples=2))
    with tarfile.open(result.data[0], "r") as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    json_members = [n for n in names if n.endswith(".json")]
    image_members = [n for n in names if not n.endswith(".json")]
    assert len(json_members) == 2
    assert len(image_members) == 2


def test_tar_json_has_texts_and_images(tmp_path: Path) -> None:
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(_make_batch(num_samples=1))
    with tarfile.open(result.data[0], "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".json"):
                payload = json.load(tf.extractfile(member))
                assert "texts" in payload
                assert "images" in payload
                assert payload["texts"] == ["Hello 0", None]
                assert isinstance(payload["images"], list)


def test_write_no_binary_still_records_image_positions(tmp_path: Path) -> None:
    rows = [
        {"sample_id": "s0", "position": -1, "modality": "metadata", "content_type": "application/json",
         "text_content": None, "binary_content": None, "source_ref": None, "materialize_error": None},
        {"sample_id": "s0", "position": 0, "modality": "image", "content_type": "image/png",
         "text_content": None, "binary_content": None, "source_ref": None, "materialize_error": None},
    ]
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    batch = InterleavedBatch(task_id="t", dataset_name="test", data=table, _metadata={"source_files": ["x.tar"]})
    out_dir = tmp_path / "wds_out"
    out_dir.mkdir()
    writer = InterleavedWebdatasetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    result = writer.process(batch)
    with tarfile.open(result.data[0], "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        json_members = [m for m in members if m.name.endswith(".json")]
        image_members = [m for m in members if not m.name.endswith(".json")]
        assert len(json_members) == 1
        assert len(image_members) == 0
        payload = json.load(tf.extractfile(json_members[0]))
        assert payload["images"] == ["0.png"]
