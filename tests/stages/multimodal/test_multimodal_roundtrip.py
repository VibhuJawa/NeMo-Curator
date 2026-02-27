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

from nemo_curator.stages.multimodal.io.readers.parquet import MultimodalParquetReaderStage
from nemo_curator.stages.multimodal.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.stages.multimodal.io.writers.tabular import MultimodalParquetWriterStage
from nemo_curator.stages.multimodal.io.writers.webdataset import MultimodalWebdatasetWriterStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask

from .test_data_gen import build_mint1t_tar, build_multimodal_parquet


def test_parquet_to_wds_to_parquet(tmp_path: Path):
    pq_path = build_multimodal_parquet(tmp_path / "input", num_samples=3, materialized=True)

    reader = MultimodalParquetReaderStage()
    task = FileGroupTask(
        task_id="fg_0", dataset_name="test", data=[pq_path], _metadata={"source_files": [pq_path]},
    )
    batch = reader.process(task)
    assert isinstance(batch, MultiBatchTask)
    original_df = batch.to_pandas()

    wds_dir = tmp_path / "wds_output"
    wds_dir.mkdir()
    wds_writer = MultimodalWebdatasetWriterStage(
        path=str(wds_dir), materialize_on_write=False, mode="overwrite",
    )
    wds_result = wds_writer.process(batch)
    tar_path = wds_result.data[0]

    wds_reader = WebdatasetReaderStage(source_id_field="pdf_name", materialize_on_read=True)
    wds_task = FileGroupTask(
        task_id="fg_wds", dataset_name="test", data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    wds_batch = wds_reader.process(wds_task)
    if isinstance(wds_batch, list):
        wds_batch = wds_batch[0]

    pq_dir2 = tmp_path / "pq_output"
    pq_dir2.mkdir()
    pq_writer = MultimodalParquetWriterStage(
        path=str(pq_dir2), materialize_on_write=False, mode="overwrite",
    )
    pq_writer.process(wds_batch)

    reader2 = MultimodalParquetReaderStage()
    pq_files = list(pq_dir2.glob("*.parquet"))
    assert len(pq_files) >= 1
    task2 = FileGroupTask(
        task_id="fg_2", dataset_name="test", data=[str(p) for p in pq_files],
        _metadata={"source_files": [str(p) for p in pq_files]},
    )
    final_batch = reader2.process(task2)
    final_df = final_batch.to_pandas()

    orig_samples = set(original_df["sample_id"].unique())
    final_samples = set(final_df["sample_id"].unique())
    assert len(final_samples) == len(orig_samples)

    for sid in orig_samples:
        orig_mask = (original_df["sample_id"] == sid) & (original_df["modality"] == "text")
        orig_texts = sorted(original_df[orig_mask]["text_content"].tolist())
        final_mask = (final_df["sample_id"] == sid) & (final_df["modality"] == "text")
        final_texts = sorted(final_df[final_mask]["text_content"].dropna().tolist())
        assert orig_texts == final_texts, f"Text mismatch for {sid}"


def test_wds_to_parquet_to_wds(tmp_path: Path):
    tar_path = build_mint1t_tar(tmp_path / "input")

    wds_reader = WebdatasetReaderStage(source_id_field="pdf_name", materialize_on_read=True)
    task = FileGroupTask(
        task_id="fg_0", dataset_name="test", data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    batch = wds_reader.process(task)
    if isinstance(batch, list):
        batch = batch[0]
    original_df = batch.to_pandas()

    pq_dir = tmp_path / "pq_output"
    pq_dir.mkdir()
    pq_writer = MultimodalParquetWriterStage(
        path=str(pq_dir), materialize_on_write=False, mode="overwrite",
    )
    pq_writer.process(batch)

    pq_reader = MultimodalParquetReaderStage()
    pq_files = list(pq_dir.glob("*.parquet"))
    pq_task = FileGroupTask(
        task_id="fg_pq", dataset_name="test", data=[str(p) for p in pq_files],
        _metadata={"source_files": [str(p) for p in pq_files]},
    )
    pq_batch = pq_reader.process(pq_task)
    if isinstance(pq_batch, list):
        pq_batch = pq_batch[0]

    wds_dir = tmp_path / "wds_output"
    wds_dir.mkdir()
    wds_writer = MultimodalWebdatasetWriterStage(
        path=str(wds_dir), materialize_on_write=False, mode="overwrite",
    )
    wds_writer.process(pq_batch)

    tar_files = list(wds_dir.glob("*.tar"))
    assert len(tar_files) >= 1

    with tarfile.open(tar_files[0], "r") as tf:
        json_members = [m for m in tf.getmembers() if m.name.endswith(".json")]
        assert len(json_members) == 2

        for member in json_members:
            payload = json.load(tf.extractfile(member))
            assert "texts" in payload
            assert "images" in payload
            assert isinstance(payload["texts"], list)

    orig_image_rows = original_df[original_df["modality"] == "image"]
    materialized_count = orig_image_rows["binary_content"].notna().sum()

    with tarfile.open(tar_files[0], "r") as tf:
        img_members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
        assert len(img_members) == materialized_count
