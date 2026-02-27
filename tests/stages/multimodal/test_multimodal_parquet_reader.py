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

import pytest

from nemo_curator.stages.multimodal.io.readers.parquet import MultimodalParquetReaderStage
from nemo_curator.tasks import FileGroupTask, MultiBatchTask

from .test_data_gen import build_multimodal_parquet


def _make_task(parquet_path: str) -> FileGroupTask:
    return FileGroupTask(
        task_id="file_group_0",
        dataset_name="test_dataset",
        data=[parquet_path],
        _metadata={"source_files": [parquet_path]},
    )


def test_reader_reads_materialized_parquet(tmp_path: Path):
    pq_path = build_multimodal_parquet(tmp_path, num_samples=3, materialized=True)
    reader = MultimodalParquetReaderStage()
    result = reader.process(_make_task(pq_path))

    assert isinstance(result, MultiBatchTask)
    df = result.to_pandas()
    assert len(df) == 12  # 3 samples * 4 rows each (metadata + 2 text + 1 image)
    assert set(df["sample_id"].unique()) == {"sample_000", "sample_001", "sample_002"}
    assert set(df["modality"].unique()) == {"metadata", "text", "image"}

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 3
    for _, row in image_rows.iterrows():
        assert isinstance(row["binary_content"], bytes)
        assert len(row["binary_content"]) > 0


def test_reader_reads_non_materialized_parquet(tmp_path: Path):
    pq_path = build_multimodal_parquet(
        tmp_path, num_samples=2, materialized=False, image_dir=tmp_path / "images",
    )
    reader = MultimodalParquetReaderStage()
    result = reader.process(_make_task(pq_path))

    df = result.to_pandas()
    assert len(df) == 8  # 2 samples * 4 rows
    image_rows = df[df["modality"] == "image"]
    assert image_rows["binary_content"].isna().all()
    assert image_rows["source_ref"].notna().all()


def test_reader_preserves_sample_integrity(tmp_path: Path):
    pq_path = build_multimodal_parquet(tmp_path, num_samples=5, materialized=True)
    reader = MultimodalParquetReaderStage()
    result = reader.process(_make_task(pq_path))

    df = result.to_pandas()
    for sid in df["sample_id"].unique():
        group = df[df["sample_id"] == sid]
        assert (group["modality"] == "metadata").sum() == 1
        assert (group["modality"] == "text").sum() == 2
        assert (group["modality"] == "image").sum() == 1


def test_reader_with_fields_filter(tmp_path: Path):
    pq_path = build_multimodal_parquet(tmp_path, num_samples=2, materialized=True)
    reader = MultimodalParquetReaderStage(
        fields=["sample_id", "modality", "position", "text_content"],
    )
    result = reader.process(_make_task(pq_path))

    df = result.to_pandas()
    assert "text_content" in df.columns
    assert "binary_content" not in df.columns


def test_reader_raises_on_empty_parquet(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

    empty_path = tmp_path / "empty.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA), str(empty_path))
    reader = MultimodalParquetReaderStage()
    with pytest.raises(ValueError, match="No data read"):
        reader.process(_make_task(str(empty_path)))
