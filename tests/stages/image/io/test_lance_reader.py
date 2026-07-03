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

import pyarrow as pa
import pytest

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.image.io.lance_reader import (
    LANCE_ROW_OFFSET_COLUMN,
    LanceImageReader,
    LanceImageSlice,
    LanceImageSlicePartitioningStage,
    LanceImageSliceReaderStage,
)
from nemo_curator.tasks import EmptyTask
from nemo_curator.utils.lance import LANCE_FRAGID_COLUMN

lance = pytest.importorskip("lance")


def _write_lance_images(path: Path, prefix: str = "image") -> None:
    table = pa.table(
        {
            "url": [f"https://example.com/{i}" for i in range(10)],
            "image": [f"{prefix}-{i}".encode() for i in range(10)],
            "width": list(range(100, 110)),
        }
    )
    lance.write_dataset(table, str(path), mode="create", max_rows_per_file=3, max_rows_per_group=3)


def test_partitioning_interleaves_fragment_slices_and_pins_version(tmp_path: Path) -> None:
    dataset_path = tmp_path / "images.lance"
    _write_lance_images(dataset_path)

    tasks = LanceImageSlicePartitioningStage(path=str(dataset_path), rows_per_slice=2, slices_per_partition=3).process(
        EmptyTask
    )

    assert [task.num_items for task in tasks] == [6, 3, 1]
    assert [(item.fragment_id, item.row_offset, item.row_count) for item in tasks[0].data] == [
        (0, 0, 2),
        (1, 0, 2),
        (2, 0, 2),
    ]
    assert [(item.fragment_id, item.row_offset, item.row_count) for item in tasks[1].data] == [
        (3, 0, 1),
        (0, 2, 1),
        (1, 2, 1),
    ]
    assert tasks[0]._metadata["lance"]["version"] == lance.dataset(str(dataset_path)).version
    assert set(tasks[0]._metadata["lance"]) == {"path", "version"}
    assert tasks[0].get_deterministic_id() != tasks[1].get_deterministic_id()


def test_reader_projects_images_adds_coordinates_and_reuses_session(tmp_path: Path) -> None:
    dataset_path = tmp_path / "images.lance"
    _write_lance_images(dataset_path)
    tasks = LanceImageSlicePartitioningStage(path=str(dataset_path), rows_per_slice=2, slices_per_partition=3).process(
        EmptyTask
    )
    reader = LanceImageSliceReaderStage(path=str(dataset_path), reader_threads=2)

    first = reader.process(tasks[0])
    opened_dataset = reader._dataset
    second = reader.process(tasks[1])

    assert reader._dataset is opened_dataset
    assert first.to_pyarrow().column_names == ["image", LANCE_FRAGID_COLUMN, LANCE_ROW_OFFSET_COLUMN]
    assert first.to_pyarrow()["image"].to_pylist() == [
        b"image-0",
        b"image-1",
        b"image-3",
        b"image-4",
        b"image-6",
        b"image-7",
    ]
    assert first.to_pyarrow()[LANCE_FRAGID_COLUMN].to_pylist() == [0, 0, 1, 1, 2, 2]
    assert first.to_pyarrow()[LANCE_ROW_OFFSET_COLUMN].to_pylist() == [0, 1, 0, 1, 0, 1]
    assert "schema" in first._metadata["lance"]
    assert second.num_items == 3
    assert reader.ray_stage_spec()[RayStageSpecKeys.IS_ACTOR_STAGE] is True
    assert reader.runtime_env["env_vars"] == {"LANCE_CPU_THREADS": "16", "LANCE_IO_THREADS": "64"}


def test_reader_uses_pinned_version(tmp_path: Path) -> None:
    dataset_path = tmp_path / "images.lance"
    _write_lance_images(dataset_path, prefix="old")
    task = LanceImageSlicePartitioningStage(
        path=str(dataset_path), rows_per_slice=2, slices_per_partition=1, fragment_ids=[0]
    ).process(EmptyTask)[0]

    replacement = pa.table({"url": ["https://new.example"], "image": [b"new"], "width": [1]})
    lance.write_dataset(replacement, str(dataset_path), mode="overwrite", max_rows_per_file=1)

    batch = LanceImageSliceReaderStage(path=str(dataset_path), reader_threads=1, include_coordinates=False).process(
        task
    )
    assert batch.to_pyarrow()["image"].to_pylist() == [b"old-0", b"old-1"]


def test_reader_validates_ranges_fragments_and_scanner_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="row_count"):
        LanceImageSlice(fragment_id=0, row_offset=0, row_count=0)
    with pytest.raises(ValueError, match="reader_threads"):
        LanceImageSliceReaderStage(path="dataset", reader_threads=0)
    with pytest.raises(ValueError, match="lance_cpu_threads"):
        LanceImageSliceReaderStage(path="dataset", lance_cpu_threads=0)

    dataset_path = tmp_path / "images.lance"
    _write_lance_images(dataset_path)
    with pytest.raises(ValueError, match="requested fragment ids"):
        LanceImageSlicePartitioningStage(path=str(dataset_path), fragment_ids=[999]).process(EmptyTask)

    task = LanceImageSlicePartitioningStage(path=str(dataset_path), slices_per_partition=1).process(EmptyTask)[0]
    with pytest.raises(ValueError, match="managed by Lance image slices"):
        LanceImageSliceReaderStage(path=str(dataset_path), read_kwargs={"filter": "width > 100"}).process(task)
    with pytest.raises(ValueError, match="path mismatch"):
        LanceImageSliceReaderStage(path="different.lance").process(task)


def test_composite_uses_tuned_defaults_and_fields_override() -> None:
    partitioner, reader = LanceImageReader(
        path="s3://bucket/images.lance",
        fields=["image", "url"],
        read_kwargs={"storage_options": {"region": "us-west-2"}},
    ).decompose()

    assert isinstance(partitioner, LanceImageSlicePartitioningStage)
    assert partitioner.rows_per_slice == 100
    assert partitioner.slices_per_partition == 40
    assert isinstance(reader, LanceImageSliceReaderStage)
    assert reader.fields == ["image", "url"]
    assert reader.reader_threads == 32
    assert reader.lance_cpu_threads == 16
    assert reader.lance_io_threads == 64
    assert reader.resources.cpus == 32
