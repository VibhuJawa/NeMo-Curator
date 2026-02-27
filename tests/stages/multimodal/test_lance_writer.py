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
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa

from nemo_curator.stages.multimodal.io.writers.lance import MultimodalLanceWriterStage
from nemo_curator.tasks import MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA


def _source_ref(content_path: str, content_key: str | None) -> str:
    return json.dumps(
        {"path": content_path, "member": content_key, "byte_offset": None, "byte_size": None}
    )


def _make_task(rows: list[dict], source_files: list[str] | None = None) -> MultiBatchTask:
    table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
    metadata = {"source_files": source_files} if source_files else {}
    return MultiBatchTask(
        task_id="lance_test",
        dataset_name="test_dataset",
        data=table,
        _metadata=metadata,
    )


def test_lance_writer_roundtrip_text_only(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "metadata_json": json.dumps({"url": "https://example.com"}),
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "Hello world",
            "binary_content": None,
            "source_ref": None,
            "metadata_json": None,
            "materialize_error": None,
        },
    ]
    task = _make_task(rows, source_files=["fake.parquet"])

    writer = MultimodalLanceWriterStage(
        path=str(tmp_path / "out_lance"),
        materialize_on_write=False,
        mode="overwrite",
    )
    result = writer.process(task)
    assert len(result.data) == 1

    ds = lance.dataset(result.data[0])
    df_out = ds.to_table().to_pandas()
    assert len(df_out) == 2
    assert set(df_out["sample_id"]) == {"s1"}
    assert df_out.loc[df_out["modality"] == "text", "text_content"].iloc[0] == "Hello world"


def test_lance_writer_roundtrip_with_binary(tmp_path: Path) -> None:
    image_bytes = b"fake-jpeg-content-12345"
    raw_path = tmp_path / "test_image.jpg"
    raw_path.write_bytes(image_bytes)

    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "metadata_json": json.dumps({"url": "https://example.com"}),
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": _source_ref(str(raw_path), None),
            "metadata_json": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 1,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "Caption for image",
            "binary_content": None,
            "source_ref": None,
            "metadata_json": None,
            "materialize_error": None,
        },
    ]
    task = _make_task(rows, source_files=["fake.parquet"])

    writer = MultimodalLanceWriterStage(
        path=str(tmp_path / "out_lance_bin"),
        materialize_on_write=True,
        mode="overwrite",
    )
    result = writer.process(task)
    assert len(result.data) == 1
    assert result._metadata["format"] == "lance"

    ds = lance.dataset(result.data[0])
    df_out = ds.to_table().to_pandas()
    assert len(df_out) == 3
    assert df_out["sample_id"].nunique() == 1

    image_row = df_out[df_out["modality"] == "image"].iloc[0]
    assert image_row["binary_content"] == image_bytes
    assert pd.isna(image_row["materialize_error"])

    text_row = df_out[df_out["modality"] == "text"].iloc[0]
    assert text_row["text_content"] == "Caption for image"


def test_lance_writer_multiple_samples(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "First sample",
            "binary_content": None,
            "source_ref": None,
            "metadata_json": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s2",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "Second sample",
            "binary_content": None,
            "source_ref": None,
            "metadata_json": None,
            "materialize_error": None,
        },
    ]
    task = _make_task(rows, source_files=["fake.parquet"])

    writer = MultimodalLanceWriterStage(
        path=str(tmp_path / "out_lance_multi"),
        materialize_on_write=False,
        mode="overwrite",
    )
    result = writer.process(task)

    ds = lance.dataset(result.data[0])
    df_out = ds.to_table().to_pandas()
    assert len(df_out) == 2
    assert set(df_out["sample_id"]) == {"s1", "s2"}
