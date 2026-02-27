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

from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.stages.multimodal.io.readers.parquet import MultimodalParquetReaderStage
from nemo_curator.stages.multimodal.io.writers.tabular import MultimodalParquetWriterStage
from nemo_curator.stages.multimodal.io.writers.webdataset import MultimodalWebdatasetWriterStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask

from .test_data_gen import build_bad_source_ref_parquet


def _read_bad_parquet(tmp_path: Path) -> MultiBatchTask:
    pq_path = build_bad_source_ref_parquet(tmp_path / "input", num_samples=3)
    reader = MultimodalParquetReaderStage()
    task = FileGroupTask(
        task_id="fg_bad", dataset_name="test", data=[pq_path],
        _metadata={"source_files": [pq_path]},
    )
    return reader.process(task)


def test_error_mode_raises_on_materialize_failure(tmp_path: Path):
    batch = _read_bad_parquet(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalParquetWriterStage(
        path=str(out_dir), materialize_on_write=True,
        on_materialize_error="error", mode="overwrite",
    )
    with pytest.raises(RuntimeError, match="failed to materialize"):
        writer.process(batch)


def test_drop_row_mode_removes_failed_image_rows(tmp_path: Path):
    batch = _read_bad_parquet(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalParquetWriterStage(
        path=str(out_dir), materialize_on_write=True,
        on_materialize_error="drop_row", mode="overwrite",
    )
    result = writer.process(batch)
    written = pd.read_parquet(result.data[0])
    image_rows = written[written["modality"] == "image"]
    assert len(image_rows) == 0


def test_drop_sample_mode_removes_entire_samples(tmp_path: Path):
    batch = _read_bad_parquet(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalParquetWriterStage(
        path=str(out_dir), materialize_on_write=True,
        on_materialize_error="drop_sample", mode="overwrite",
    )
    result = writer.process(batch)
    written = pd.read_parquet(result.data[0])
    assert len(written) == 0


def test_wds_writer_error_mode(tmp_path: Path):
    batch = _read_bad_parquet(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalWebdatasetWriterStage(
        path=str(out_dir), materialize_on_write=True,
        on_materialize_error="error", mode="overwrite",
    )
    with pytest.raises(RuntimeError, match="failed to materialize"):
        writer.process(batch)


def test_wds_writer_drop_row_mode(tmp_path: Path):
    batch = _read_bad_parquet(tmp_path)
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    writer = MultimodalWebdatasetWriterStage(
        path=str(out_dir), materialize_on_write=True,
        on_materialize_error="drop_row", mode="overwrite",
    )
    result = writer.process(batch)
    assert result.data[0].endswith(".tar")
