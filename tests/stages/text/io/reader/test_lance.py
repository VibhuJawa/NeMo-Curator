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

from nemo_curator.stages.text.io.reader.lance import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    LancePartitioningStage,
    LanceReader,
    LanceReaderStage,
)
from nemo_curator.tasks import EmptyTask

pytest.importorskip("lance")


def _write_lance_dataset(path: Path) -> None:
    import lance

    table = pa.table(
        {
            "snapshot_id": ["CC-MAIN-2025-26", "CC-MAIN-2025-18", "CC-MAIN-2025-26", "CC-MAIN-2025-26"],
            "url": ["https://a.example", "https://b.example", "https://c.example", "https://d.example"],
            "text": ["alpha one", "beta two", "gamma three", "delta four"],
            "content_zlib": lance.blob_array([b"html-a", b"html-b", b"html-c", b"html-d"]),
        },
        schema=pa.schema(
            [
                pa.field("snapshot_id", pa.string()),
                pa.field("url", pa.string()),
                pa.field("text", pa.string()),
                lance.blob_field("content_zlib"),
            ]
        ),
    )
    lance.write_dataset(
        table,
        str(path),
        mode="create",
        max_rows_per_file=2,
        max_rows_per_group=2,
        data_storage_version="2.2",
    )


def test_lance_reader_partitions_by_fragment_and_returns_pyarrow_with_metadata(tmp_path: Path):
    dataset_path = tmp_path / "docs.lance"
    _write_lance_dataset(dataset_path)

    partitioner = LancePartitioningStage(path=str(dataset_path), fragments_per_partition=1)
    read_tasks = partitioner.process(EmptyTask)

    assert len(read_tasks) == 2
    assert {fragment_id for task in read_tasks for fragment_id in task.data} == {0, 1}

    reader = LanceReaderStage(
        path=str(dataset_path),
        fields=["snapshot_id", "url", "content_zlib"],
        include_lance_metadata=True,
        read_kwargs={
            "filter": "snapshot_id = 'CC-MAIN-2025-26'",
            "scanner_options": {"batch_size": 2},
        },
    )

    batches = [reader.process(task) for task in read_tasks]
    batches = [batch for batch in batches if batch]

    assert batches
    assert all(isinstance(batch.data, pa.Table) for batch in batches)
    seen_fragments: set[int] = set()
    for batch in batches:
        table = batch.to_pyarrow()
        assert LANCE_ROWADDR_COLUMN in table.column_names
        assert LANCE_FRAGID_COLUMN in table.column_names
        assert "content_zlib" in table.column_names
        assert table.schema.field("content_zlib").type.extension_name == "lance.blob.v2"
        fragids = {int(value) for value in table[LANCE_FRAGID_COLUMN].combine_chunks().to_pylist()}
        assert not seen_fragments.intersection(fragids)
        seen_fragments.update(fragids)

    assert seen_fragments == {0, 1}


def test_lance_reader_fields_override_read_kwargs_columns():
    reader = LanceReader(
        path="example.lance",
        fields=["a", "b"],
        read_kwargs={"columns": ["ignored"], "filter": "a IS NOT NULL"},
    )
    _, reader_stage = reader.decompose()

    assert reader_stage.fields == ["a", "b"]


def test_lance_reader_uses_read_kwargs_columns_when_fields_are_absent(tmp_path: Path):
    dataset_path = tmp_path / "docs.lance"
    _write_lance_dataset(dataset_path)
    task = LancePartitioningStage(path=str(dataset_path)).process(EmptyTask)[0]

    batch = LanceReaderStage(
        path=str(dataset_path),
        read_kwargs={"columns": ["url"]},
        include_lance_metadata=False,
    ).process(task)

    assert batch is not None
    assert batch.to_pyarrow().column_names == ["url"]


def test_lance_reader_returns_none_for_empty_filtered_fragment(tmp_path: Path):
    dataset_path = tmp_path / "docs.lance"
    _write_lance_dataset(dataset_path)
    task = LancePartitioningStage(path=str(dataset_path), fragments_per_partition=1).process(EmptyTask)[0]

    batch = LanceReaderStage(
        path=str(dataset_path),
        read_kwargs={"filter": "snapshot_id = 'DOES-NOT-EXIST'"},
    ).process(task)

    assert batch is None


def test_lance_reader_pins_partitioned_dataset_version(tmp_path: Path):
    import lance

    dataset_path = tmp_path / "docs.lance"
    lance.write_dataset(pa.table({"text": ["old"]}), str(dataset_path), mode="create", max_rows_per_file=1)
    task = LancePartitioningStage(path=str(dataset_path)).process(EmptyTask)[0]
    lance.write_dataset(pa.table({"text": ["new"]}), str(dataset_path), mode="overwrite", max_rows_per_file=1)

    batch = LanceReaderStage(path=str(dataset_path), fields=["text"], include_lance_metadata=False).process(task)

    assert batch is not None
    assert batch.to_pyarrow()["text"].to_pylist() == ["old"]


def test_lance_read_task_has_stable_content_id(tmp_path: Path):
    dataset_path = tmp_path / "docs.lance"
    _write_lance_dataset(dataset_path)
    task_a = LancePartitioningStage(path=str(dataset_path), fragments_per_partition=1).process(EmptyTask)[0]
    task_b = LancePartitioningStage(path=str(dataset_path), fragments_per_partition=1).process(EmptyTask)[0]

    assert task_a.get_deterministic_id() == task_b.get_deterministic_id()
