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

from nemo_curator.stages.interleaved import (
    InterleavedLanceReader,
    InterleavedLanceReaderStage,
    LanceColumnFetchStage,
    LanceDatasetConfig,
    LanceIndexCacheConfig,
)
from nemo_curator.stages.interleaved.lance import fragment_row_id_starts, group_row_ids_by_fragment
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage
from nemo_curator.stages.text.io.writer import LanceWriter, commit_lance_checkpoint
from nemo_curator.tasks import EmptyTask, InterleavedBatch

lance = pytest.importorskip("lance")
pytest.importorskip("lance_ray")


def _reference_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("url", pa.string(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("md5", pa.string(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("metadata", pa.struct([pa.field("format", pa.string()), pa.field("animated", pa.bool_())])),
            pa.field("tags", pa.list_(pa.string())),
        ]
    )


def _write_reference(path: Path, *, duplicate_key: bool = False) -> int:
    urls = ["https://a.example/image", "https://b.example/image"]
    if duplicate_key:
        urls[1] = urls[0]
    table = pa.Table.from_pylist(
        [
            {
                "url": urls[0],
                "image": b"image-a",
                "md5": "md5-a",
                "width": 100,
                "metadata": {"format": "JPEG", "animated": False},
                "tags": ["photo", "web"],
            },
            {
                "url": urls[1],
                "image": b"image-b",
                "md5": "md5-b",
                "width": 200,
                "metadata": {"format": "PNG", "animated": True},
                "tags": ["graphic"],
            },
        ],
        schema=_reference_schema(),
    )
    lance.write_dataset(
        table,
        str(path),
        mode="create",
        data_storage_version="2.2",
        enable_stable_row_ids=True,
    )
    dataset = lance.dataset(str(path))
    dataset.create_scalar_index("url", "BTREE", name="url_btree")
    return lance.dataset(str(path)).version


def _config(path: Path, version: int) -> tuple[LanceDatasetConfig, LanceIndexCacheConfig]:
    return (
        LanceDatasetConfig(uri=str(path), version=version, key_column="url", index_name="url_btree"),
        LanceIndexCacheConfig(
            mirror_path=str(path),
            prewarm=False,
            index_cache_size_bytes=1024**2,
            metadata_cache_size_bytes=1024**2,
        ),
    )


def _batch(keys: list[str | None], **extra: pa.Array) -> InterleavedBatch:
    count = len(keys)
    columns: dict[str, pa.Array] = {
        "sample_id": pa.array([f"sample-{index}" for index in range(count)]),
        "position": pa.array(list(range(count)), type=pa.int32()),
        "modality": pa.array(["image"] * count),
        "source_ref": pa.array(keys, type=pa.string()),
    }
    columns.update(extra)
    return InterleavedBatch(dataset_name="interleaved", data=pa.table(columns))


def test_lance_column_fetch_presence_only_and_duplicate_input_keys(tmp_path: Path) -> None:
    path = tmp_path / "reference.lance"
    dataset, cache = _config(path, _write_reference(path))
    stage = LanceColumnFetchStage(dataset=dataset, index_cache=cache, columns={}, presence_column="image_present")

    output = stage.process(
        _batch(["https://a.example/image", "https://missing.example/image", "https://a.example/image", None, ""])
    ).to_pyarrow()
    stage.teardown()

    assert output["image_present"].to_pylist() == [True, False, True, None, None]
    assert output["source_ref"].to_pylist() == [
        "https://a.example/image",
        "https://missing.example/image",
        "https://a.example/image",
        None,
        "",
    ]


def test_lance_column_fetch_batches_tasks_into_one_deduplicated_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "reference.lance"
    dataset, cache = _config(path, _write_reference(path))
    stage = LanceColumnFetchStage(dataset=dataset, index_cache=cache, columns={}, presence_column="image_present")
    stage.setup()
    assert stage._fetcher is not None

    fetch_calls: list[list[object]] = []
    original_fetch = stage._fetcher.fetch

    def record_fetch(keys: list[object]):  # noqa: ANN202
        fetch_calls.append(list(keys))
        return original_fetch(keys)

    monkeypatch.setattr(stage._fetcher, "fetch", record_fetch)
    outputs = stage.process_batch(
        [
            _batch(["https://a.example/image", "https://missing.example/image"]),
            _batch(["https://a.example/image", "https://b.example/image"]),
        ]
    )
    stage.teardown()

    assert fetch_calls == [["https://a.example/image", "https://missing.example/image", "https://b.example/image"]]
    assert outputs[0].to_pyarrow()["image_present"].to_pylist() == [True, False]
    assert outputs[1].to_pyarrow()["image_present"].to_pylist() == [True, True]


def test_lance_column_fetch_projects_binary_scalar_struct_and_list(tmp_path: Path) -> None:
    path = tmp_path / "reference.lance"
    dataset, cache = _config(path, _write_reference(path))
    stage = LanceColumnFetchStage(
        dataset=dataset,
        index_cache=cache,
        columns={
            "image": "binary_content",
            "width": "reference_width",
            "metadata": "reference_metadata",
            "tags": "reference_tags",
        },
        presence_column="image_present",
        fetch_batch_size=1,
    )

    output = stage.process(_batch(["https://b.example/image", "https://a.example/image"])).to_pyarrow()
    stage.teardown()

    assert output["binary_content"].to_pylist() == [b"image-b", b"image-a"]
    assert output["reference_width"].to_pylist() == [200, 100]
    assert output["reference_metadata"].to_pylist() == [
        {"format": "PNG", "animated": True},
        {"format": "JPEG", "animated": False},
    ]
    assert output["reference_tags"].to_pylist() == [["graphic"], ["photo", "web"]]
    assert output["image_present"].to_pylist() == [True, True]


def test_lance_column_fetch_collision_policies_and_presence_short_circuit(tmp_path: Path) -> None:
    path = tmp_path / "reference.lance"
    dataset, cache = _config(path, _write_reference(path))
    task = _batch(
        ["https://a.example/image", "https://b.example/image"],
        reference_md5=pa.array([None, "existing"], type=pa.string()),
        image_present=pa.array([True, False], type=pa.bool_()),
    )

    error_stage = LanceColumnFetchStage(
        dataset=dataset,
        index_cache=cache,
        columns={"md5": "reference_md5"},
        presence_column="image_present",
    )
    with pytest.raises(ValueError, match="already exist"):
        error_stage.process(task)
    error_stage.teardown()

    fill_stage = LanceColumnFetchStage(
        dataset=dataset,
        index_cache=cache,
        columns={"md5": "reference_md5"},
        presence_column="image_present",
        existing_column_policy="fill_null",
    )
    filled = fill_stage.process(task).to_pyarrow()
    fill_stage.teardown()
    assert filled["reference_md5"].to_pylist() == ["md5-a", "existing"]
    assert filled["image_present"].to_pylist() == [True, False]

    overwrite_stage = LanceColumnFetchStage(
        dataset=dataset,
        index_cache=cache,
        columns={"md5": "reference_md5"},
        presence_column="image_present",
        existing_column_policy="overwrite",
    )
    overwritten = overwrite_stage.process(task).to_pyarrow()
    overwrite_stage.teardown()
    assert overwritten["reference_md5"].to_pylist() == ["md5-a", "existing"]


def test_lance_column_fetch_missing_and_duplicate_reference_policies(tmp_path: Path) -> None:
    path = tmp_path / "reference.lance"
    dataset, cache = _config(path, _write_reference(path))
    stage = LanceColumnFetchStage(
        dataset=dataset,
        index_cache=cache,
        columns={"md5": "reference_md5"},
        presence_column="image_present",
        missing_key_policy="error",
    )
    with pytest.raises(KeyError, match="not found"):
        stage.process(_batch(["https://missing.example/image"]))
    stage.teardown()

    duplicate_path = tmp_path / "duplicates.lance"
    duplicate_dataset, duplicate_cache = _config(duplicate_path, _write_reference(duplicate_path, duplicate_key=True))
    duplicate_stage = LanceColumnFetchStage(
        dataset=duplicate_dataset,
        index_cache=duplicate_cache,
        columns={"md5": "reference_md5"},
        presence_column="image_present",
    )
    with pytest.raises(ValueError, match="Multiple Lance rows"):
        duplicate_stage.process(_batch(["https://a.example/image"]))
    duplicate_stage.teardown()


def test_lance_column_fetch_validates_configuration_and_types(tmp_path: Path) -> None:
    path = tmp_path / "reference.lance"
    dataset, cache = _config(path, _write_reference(path))
    with pytest.raises(ValueError, match="only when presence_column"):
        LanceColumnFetchStage(dataset=dataset, index_cache=cache, columns={}, presence_column=None)
    with pytest.raises(ValueError, match="distinct destination"):
        LanceColumnFetchStage(
            dataset=dataset,
            index_cache=cache,
            columns={"md5": "same", "width": "same"},
            presence_column="image_present",
        )
    missing = LanceColumnFetchStage(
        dataset=dataset,
        index_cache=cache,
        columns={"unknown": "output"},
        presence_column="image_present",
    )
    with pytest.raises(ValueError, match="do not exist"):
        missing.setup()

    stage = LanceColumnFetchStage(
        dataset=dataset,
        index_cache=cache,
        columns={"width": "reference_width"},
        presence_column="image_present",
        existing_column_policy="fill_null",
    )
    with pytest.raises(TypeError, match="has type string"):
        stage.process(
            _batch(
                ["https://a.example/image"],
                reference_width=pa.array(["wrong"], type=pa.string()),
            )
        )
    stage.teardown()


def _write_sharded(path: Path, rows: int, rows_per_file: int) -> int:
    table = pa.table(
        {
            "url": pa.array([f"https://example/{index:03d}" for index in range(rows)]),
            "width": pa.array(list(range(rows)), type=pa.int32()),
        }
    )
    lance.write_dataset(
        table,
        str(path),
        mode="create",
        data_storage_version="2.2",
        enable_stable_row_ids=True,
        max_rows_per_file=rows_per_file,
    )
    lance.dataset(str(path)).create_scalar_index("url", "BTREE", name="url_btree")
    return lance.dataset(str(path)).version


def test_group_row_ids_by_fragment_splits_at_fragment_boundaries() -> None:
    starts = [0, 4, 9]
    assert group_row_ids_by_fragment([10, 3, 4, 0, 8, 9], starts) == {0: [0, 3], 1: [4, 8], 2: [9, 10]}
    assert group_row_ids_by_fragment([], starts) == {}
    assert group_row_ids_by_fragment([7, 1], [0]) == {0: [1, 7]}


def test_lance_column_fetch_fragment_affinity_matches_default_and_reports_locality(tmp_path: Path) -> None:
    path = tmp_path / "sharded.lance"
    version = _write_sharded(path, rows=12, rows_per_file=4)
    dataset, cache = _config(path, version)
    assert fragment_row_id_starts(lance.dataset(str(path), version=version)) == [0, 4, 8]

    keys = ["https://example/000", "https://example/005", "https://example/011"]
    outputs: dict[bool, pa.Table] = {}
    for affinity in (False, True):
        stage = LanceColumnFetchStage(
            dataset=dataset,
            index_cache=cache,
            columns={"width": "reference_width"},
            presence_column="image_present",
            fetch_batch_size=1,
            fragment_affinity=affinity,
        )
        outputs[affinity] = stage.process(_batch(keys)).to_pyarrow()
        metrics = stage._consume_custom_metrics()
        stage.teardown()

        assert "lance_gets_per_image" in metrics
        assert ("lance_images_per_file_open" in metrics) is affinity
        if affinity:
            # One key per fragment file, every fragment opened for the first time.
            assert metrics["lance_fragments_touched"] == 3.0
            assert metrics["lance_fragment_first_opens"] == 3.0
            assert metrics["lance_images_per_file_open"] == 1.0

    assert outputs[True].equals(outputs[False])
    assert outputs[True]["reference_width"].to_pylist() == [0, 5, 11]


def test_interleaved_lance_reader_and_writer_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "interleaved.lance"
    commit_path = tmp_path / "writer-commit"
    task = _batch(["https://a.example/image", None])
    task._set_task_id("0", "interleaved")
    schema = task.to_pyarrow().schema

    LanceWriter(
        path=str(output_path),
        commit_path=str(commit_path),
        schema=schema,
        write_kwargs={"data_storage_version": "2.2"},
    ).process(task)
    version = commit_lance_checkpoint(str(output_path), str(commit_path))

    read_task = LancePartitioningStage(
        path=str(output_path), read_kwargs={"version": version}, fragments_per_partition=1
    ).process(EmptyTask)[0]
    batch = InterleavedLanceReaderStage(
        path=str(output_path),
        read_kwargs={"version": version},
        include_lance_metadata=False,
    ).process(read_task)

    assert isinstance(batch, InterleavedBatch)
    assert batch.to_pyarrow().equals(task.to_pyarrow())
    _, reader_stage = InterleavedLanceReader(path=str(output_path), read_kwargs={"version": version}).decompose()
    assert isinstance(reader_stage, InterleavedLanceReaderStage)
