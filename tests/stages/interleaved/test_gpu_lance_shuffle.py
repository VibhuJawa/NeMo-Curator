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

from __future__ import annotations

import builtins
import importlib.util
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import nemo_curator.stages.interleaved as interleaved_module
from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.ray_actor_pool import executor as executor_module
from nemo_curator.backends.ray_actor_pool.adapter import RayActorPoolStageAdapter
from nemo_curator.backends.ray_actor_pool.executor import (
    RayActorPoolExecutor,
    _iter_rank_task_windows,
    _tasks_are_object_refs,
)
from nemo_curator.backends.ray_actor_pool.shuffle_adapter import ShuffleStageAdapter
from nemo_curator.backends.ray_actor_pool.utils import create_named_ray_actor_pool_stage_adapter
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved import GpuLanceShuffleFetchStage
from nemo_curator.stages.interleaved import gpu_lance_shuffle_actor as actor_module
from nemo_curator.stages.interleaved.gpu_lance_shuffle import (
    _normalise_index_shards,
    _resolve_fetch_window_bytes,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.lance import LanceReadTask
from nemo_curator.tasks import InterleavedBatch

if TYPE_CHECKING:
    from types import ModuleType


def _stage(**overrides: object) -> GpuLanceShuffleFetchStage:
    kwargs: dict[str, object] = {
        "image_uri": "s3://bucket/images.lance",
        "image_version": 7,
        "index_shards": {0: "index-0.parquet", 1: ["index-1a.parquet", "index-1b.parquet"]},
        "index_manifest_uri": "s3://bucket/index-manifest.json",
        "index_manifest_sha256": "0" * 64,
    }
    kwargs.update(overrides)
    return GpuLanceShuffleFetchStage(**kwargs)


def _interleaved_output(dataset_name: str = "documents") -> InterleavedBatch:
    return InterleavedBatch(
        dataset_name=dataset_name,
        data=pa.table(
            {
                "sample_id": ["sample-0"],
                "position": pa.array([0], type=pa.int32()),
                "modality": ["image"],
                "binary_content": pa.array([b"payload"], type=pa.large_binary()),
            }
        ),
    )


class _FakeShuffleActor:
    inserted: list[list[LanceReadTask]]
    finished: bool
    cleaned: bool
    output: list[InterleavedBatch]

    def __init__(self) -> None:
        self.inserted = []
        self.finished = False
        self.cleaned = False
        self.output = [_interleaved_output()]

    def read_and_insert_tasks(self, tasks: list[LanceReadTask]) -> None:
        self.inserted.append(tasks)

    def insert_finished(self) -> None:
        self.finished = True

    def resolve_return_and_fetch(self) -> list[InterleavedBatch]:
        return self.output

    def cleanup(self) -> None:
        self.cleaned = True


class _FakeImageDataset:
    def __init__(self, *, returned_rows: int | None = None) -> None:
        self.returned_rows = returned_rows
        self.calls: list[tuple[list[int], list[str]]] = []

    def _take_rows(self, stable_ids: list[int], *, columns: list[str]) -> pa.Table:
        self.calls.append((stable_ids, columns))
        rows = len(stable_ids) if self.returned_rows is None else self.returned_rows
        return pa.table({"image": pa.array([b"payload"] * rows, type=pa.binary())})


class _EchoRefStage(ProcessingStage[LanceReadTask, LanceReadTask]):
    name = "echo_ref_probe"
    resources = Resources(cpus=1.0)
    batch_size = 2

    def process(self, task: LanceReadTask) -> LanceReadTask:
        return task

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []


class _ImmediateRemoteMethod:
    def __init__(self, function: object) -> None:
        self._function = function

    def remote(self, *args: object, **kwargs: object) -> object:
        return self._function(*args, **kwargs)  # type: ignore[operator]


class _ImmediateGenerator:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def __iter__(self) -> object:
        return iter(self._values)


class _OrderedOutputStage:
    is_source_stage = False

    def __init__(self, outputs: list[InterleavedBatch]) -> None:
        self.outputs = outputs

    def extract_and_write(self) -> list[InterleavedBatch]:
        return self.outputs


class _FakeWindowRemoteActor:
    def __init__(self, rank: int, events: list[tuple[str, int, int]]) -> None:
        self.rank = rank
        self.events = events
        self.current: list[LanceReadTask] = []
        self.insert_sizes: list[int] = []
        self.read_and_insert = _ImmediateRemoteMethod(self._read_and_insert)
        self.insert_finished = _ImmediateRemoteMethod(self._insert_finished)
        self.extract_and_write = _ImmediateRemoteMethod(self._extract_and_write)
        self.extract_and_write_streaming = _ImmediateRemoteMethod(self._extract_and_write_streaming)

    def _read_and_insert(self, *, tasks: list[LanceReadTask]) -> list[LanceReadTask]:
        self.current = tasks
        self.insert_sizes.append(len(tasks))
        self.events.append(("read", self.rank, len(tasks)))
        return tasks

    def _insert_finished(self) -> None:
        self.events.append(("finish", self.rank, len(self.current)))

    def _extract_and_write(self) -> list[InterleavedBatch]:
        self.events.append(("extract", self.rank, len(self.current)))
        outputs = [_interleaved_output(task.dataset_name) for task in self.current]
        self.current = []
        return outputs

    def _extract_and_write_streaming(self) -> _ImmediateGenerator:
        return _ImmediateGenerator(list(self._extract_and_write()))


def test_gpu_lance_shuffle_is_exported_and_configures_actor() -> None:
    stage = _stage(
        image_columns={"image": "binary_content", "width": "image_width"},
        stable_row_id_output_column="image_stable_row_id",
        document_projection=["sample_id", "position", "modality"],
        fetch_task_window=4,
    )

    assert stage.index_shards == (
        ("index-0.parquet",),
        ("index-1a.parquet", "index-1b.parquet"),
    )
    assert stage.actor_kwargs["total_nparts"] == 2
    assert stage.actor_kwargs["index_manifest_uri"] == "s3://bucket/index-manifest.json"
    assert stage.actor_kwargs["index_manifest_sha256"] == "0" * 64
    assert stage.actor_kwargs["fetch_task_window"] == 4
    assert stage._shuffle_task_window_size == 4
    assert stage.actor_kwargs["fetch_window_bytes"] == 1024**3
    assert stage.actor_kwargs["estimated_payload_bytes_per_row"] == 128 * 1024
    assert stage.actor_kwargs["fetch_batch_size"] == 1024
    assert stage.actor_kwargs["max_pending_takes"] == 16
    assert stage.actor_kwargs["coordinate_plan_output_path"] is None
    assert stage.actor_kwargs["image_columns"] == {"image": "binary_content", "width": "image_width"}
    assert "index_image_rowaddr_column" not in stage.actor_kwargs
    assert "fragment_take_batch_size" not in stage.actor_kwargs
    assert "io_threads" not in stage.actor_kwargs
    assert stage.outputs() == (
        ["data"],
        ["binary_content", "image_width", "image_stable_row_id"],
    )
    assert stage.ray_stage_spec() == {RayStageSpecKeys.IS_SHUFFLE_STAGE: True}
    assert stage.num_workers() == len(stage.index_shards)
    assert stage.resources.gpus == 1.0
    assert stage.is_resumable is False
    assert interleaved_module.GpuLanceShuffleFetchStage is GpuLanceShuffleFetchStage
    assert "GpuLanceShuffleFetchStage" in interleaved_module.__all__


def test_gpu_lance_shuffle_configures_coordinate_plan_mode(tmp_path: Path) -> None:
    stage = _stage(coordinate_plan_output_path=str(tmp_path / "plans"))

    assert stage.actor_kwargs["coordinate_plan_output_path"] == str(tmp_path / "plans")
    assert stage.outputs() == ([], [])


@pytest.mark.parametrize(
    ("nranks", "total_nparts", "expected"),
    [
        (1, 1, True),
        (1, 2, False),
        (2, 2, False),
        (2, 4, False),
    ],
)
def test_replicated_sidecar_compatibility_is_single_rank_single_partition_only(
    nranks: int,
    total_nparts: int,
    expected: bool,
) -> None:
    assert actor_module._allow_single_partition_replicated_sidecar(nranks, total_nparts) is expected


def test_replicated_sidecar_loader_uses_persistent_segmented_mapper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeMapper:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(actor_module, "_GpuExactKeyMapper", FakeMapper)
    storage_options = {"endpoint": "https://object-store.invalid"}

    mapper = actor_module._load_segmented_replicated_index(
        ("segment-0.parquet", "segment-1.parquet"),
        "source_url",
        "global_id",
        storage_options,
        17,
    )
    storage_options["endpoint"] = "mutated"

    assert isinstance(mapper, FakeMapper)
    assert calls == [
        {
            "reference_files": ("segment-0.parquet", "segment-1.parquet"),
            "reference_key_column": "source_url",
            "reference_row_id_column": "global_id",
            "storage_options": {"endpoint": "https://object-store.invalid"},
            "expected_reference_rows": 17,
            "load_factor": 0.5,
        }
    ]


@pytest.mark.gpu
def test_replicated_sidecar_segmented_mapper_preserves_exact_cross_segment_mapping(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(
        pa.table(
            {
                "url": pa.array(["alpha", "gamma"]),
                "stable_row_id": pa.array([0, 2], type=pa.uint64()),
            }
        ),
        first,
    )
    pq.write_table(
        pa.table(
            {
                "url": pa.array(["beta", "delta"]),
                "stable_row_id": pa.array([1, 3], type=pa.uint64()),
            }
        ),
        second,
    )
    mapper = actor_module._load_segmented_replicated_index(
        (str(first), str(second)),
        "url",
        "stable_row_id",
        {},
        4,
    )
    try:
        result = mapper.map(pa.array(["delta", "alpha", "missing", "beta"]))
    finally:
        mapper.close()

    assert result.matched.tolist() == [True, True, False, True]
    assert result.row_ids.tolist() == [3, 0, 0, 1]


@pytest.mark.gpu
def test_replicated_sidecar_segmented_mapper_rejects_duplicate_probe_across_segments(tmp_path: Path) -> None:
    paths = []
    for stable_row_id in range(2):
        path = tmp_path / f"duplicate-{stable_row_id}.parquet"
        pq.write_table(
            pa.table(
                {
                    "url": pa.array(["duplicate"]),
                    "stable_row_id": pa.array([stable_row_id], type=pa.uint64()),
                }
            ),
            path,
        )
        paths.append(str(path))
    mapper = actor_module._load_segmented_replicated_index(
        tuple(paths),
        "url",
        "stable_row_id",
        {},
        2,
    )
    try:
        with pytest.raises(ValueError, match="duplicate keys across segments"):
            mapper.map(pa.array(["duplicate"]))
    finally:
        mapper.close()


def test_coordinate_plan_table_restores_document_order() -> None:
    coordinates = pa.table(
        {
            "origin_rank": pa.array([0, 0, 0], type=pa.int32()),
            "origin_slot": pa.array([2, 2, 2], type=pa.uint64()),
            "document_rowaddr": pa.array([103, 101, 102], type=pa.uint64()),
            "document_position": pa.array([2, 0, 1], type=pa.uint64()),
            "stable_row_id": pa.array([9, 7, None], type=pa.uint64()),
        }
    )

    plan = actor_module._coordinate_plan_table(coordinates, allow_missing=True)

    assert plan.column_names == ["document_rowaddr", "document_position", "stable_row_id"]
    assert plan["document_rowaddr"].to_pylist() == [101, 102, 103]
    assert plan["document_position"].to_pylist() == [0, 1, 2]
    assert plan["stable_row_id"].to_pylist() == [7, None, 9]


def test_private_take_deduplicates_sorts_and_attaches_stable_ids() -> None:
    coordinates = pa.table(
        {
            "stable_row_id": pa.array([9, 2, None, 9, 5, 2], type=pa.uint64()),
        }
    )
    dataset = _FakeImageDataset()

    stable_ids = actor_module._sorted_unique_stable_ids(coordinates)
    payloads = actor_module._take_rows_by_stable_id(dataset, stable_ids, ["image"])

    assert stable_ids == [2, 5, 9]
    assert dataset.calls == [([2, 5, 9], ["image"])]
    assert payloads.column_names == ["image", "__nemo_fetched_stable_row_id"]
    assert payloads["__nemo_fetched_stable_row_id"].to_pylist() == [2, 5, 9]


def test_private_take_rejects_incomplete_payload_fetch() -> None:
    dataset = _FakeImageDataset(returned_rows=1)

    with pytest.raises(RuntimeError, match="returned 1 rows for 2 stable IDs"):
        actor_module._take_rows_by_stable_id(dataset, [2, 5], ["image"])


def test_private_take_chunks_sorted_ids_by_estimated_bytes() -> None:
    stable_ids = list(range(7))

    chunks = actor_module._stable_id_fetch_chunks(
        stable_ids,
        fetch_batch_size=3,
        fetch_window_bytes=256 * 1024**2,
        estimated_payload_bytes_per_row=128 * 1024**2,
    )

    assert chunks == [[0, 1], [2, 3], [4, 5], [6]]


def test_private_take_chunks_keep_large_window_separate_from_take_size() -> None:
    stable_ids = list(range(7))

    chunks = actor_module._stable_id_fetch_chunks(
        stable_ids,
        fetch_batch_size=3,
        fetch_window_bytes=1024**3,
        estimated_payload_bytes_per_row=1,
    )

    assert chunks == [[0, 1, 2], [3, 4, 5], [6]]


def test_private_take_chunks_run_bounded_and_reassemble_in_order() -> None:
    dataset = _FakeImageDataset()
    chunks = [[0, 1], [2, 3], [4]]

    with ThreadPoolExecutor(max_workers=2) as executor:
        tables, peak_pending = actor_module._take_stable_id_chunks(
            dataset,
            chunks,
            ["image"],
            executor,
            max_pending_takes=2,
        )

    assert peak_pending == 2
    assert [table["__nemo_fetched_stable_row_id"].to_pylist() for table in tables] == chunks
    assert sorted(call[0] for call in dataset.calls) == chunks


def test_payload_window_bound_rejects_estimated_and_actual_fanout_overshoot() -> None:
    assert (
        actor_module._validate_payload_window_bound(
            logical_requests=4,
            unique_payloads=2,
            estimated_payload_bytes_per_row=10,
            unique_payload_bytes=18,
            fetch_window_bytes=40,
        )
        == 36
    )
    with pytest.raises(MemoryError, match="Payload window estimate"):
        actor_module._validate_payload_window_bound(
            logical_requests=5,
            unique_payloads=5,
            estimated_payload_bytes_per_row=10,
            unique_payload_bytes=None,
            fetch_window_bytes=40,
        )
    with pytest.raises(MemoryError, match="duplicate fan-out"):
        actor_module._validate_payload_window_bound(
            logical_requests=4,
            unique_payloads=2,
            estimated_payload_bytes_per_row=5,
            unique_payload_bytes=22,
            fetch_window_bytes=40,
        )


def test_private_take_metrics_report_coalescing_and_physical_io() -> None:
    metrics = {"private_take_target_bytes": 16.0}

    actor_module._update_private_take_metrics(
        metrics,
        actor_module._PrivateTakeMeasurement(
            logical_requests=6,
            unique_payloads=3,
            estimated_bytes_by_take=(12, 6),
            actual_bytes_by_take=(10, 20),
            read_bytes=120,
            read_iops=3,
            seconds=2.0,
            peak_pending_takes=2,
        ),
    )

    assert metrics["private_take_calls"] == 2
    assert metrics["unique_payloads"] == 3
    assert metrics["logical_duplicate_requests"] == 3
    assert metrics["logical_duplicate_fanout"] == 2
    assert metrics["unique_payloads_per_private_take"] == 1.5
    assert metrics["sparse_calls_avoided"] == 1
    assert metrics["max_estimated_private_take_bytes"] == 12
    assert metrics["max_actual_private_take_bytes"] == 20
    assert metrics["max_private_take_target_overshoot_bytes"] == 4
    assert metrics["actual_to_estimated_payload_ratio"] == pytest.approx(30 / 18)
    assert metrics["payload_estimation_error_bytes"] == 12
    assert metrics["average_physical_read_bytes"] == 40
    assert metrics["physical_reads_per_unique_payload"] == 1
    assert metrics["read_amplification"] == 4
    assert metrics["payload_bytes_per_second"] == 15
    assert metrics["physical_read_bytes_per_second"] == 60
    assert metrics["physical_read_operations_per_second"] == 1.5
    assert metrics["max_pending_private_takes"] == 2


def test_arrow_reconstruction_preserves_order_and_duplicate_fanout_after_fake_collective() -> None:
    document = pa.table(
        {
            "sample_id": ["row-10", "row-11", "row-12", "row-13", "row-14"],
            "_rowaddr": pa.array([10, 11, 12, 13, 14], type=pa.uint64()),
        }
    )
    # Simulate rank-returned coordinates from multiple owner partitions.  They
    # are deliberately out of document order and stable ID 5 has fan-out two.
    coordinates = pa.table(
        {
            "document_rowaddr": pa.array([13, 10, 12, 11], type=pa.uint64()),
            "stable_row_id": pa.array([7, 5, None, 5], type=pa.uint64()),
        }
    )
    payloads = pa.table(
        {
            "image": pa.array([b"five", b"seven"], type=pa.binary()),
            "__nemo_fetched_stable_row_id": pa.array([5, 7], type=pa.uint64()),
        }
    )

    result = actor_module._apply_payloads_to_document(
        document,
        coordinates,
        payloads,
        image_columns={"image": "binary_content"},
        stable_row_id_output_column="image_stable_id",
        existing_column_policy="overwrite",
    )

    assert result.column_names == ["sample_id", "binary_content", "image_stable_id"]
    assert result["sample_id"].to_pylist() == ["row-10", "row-11", "row-12", "row-13", "row-14"]
    assert result["binary_content"].to_pylist() == [b"five", b"five", None, b"seven", None]
    assert result["image_stable_id"].to_pylist() == [5, 5, None, 7, None]


def test_arrow_reconstruction_fails_when_private_take_omits_resolved_payload() -> None:
    document = pa.table({"_rowaddr": pa.array([10], type=pa.uint64())})
    coordinates = pa.table(
        {
            "document_rowaddr": pa.array([10], type=pa.uint64()),
            "stable_row_id": pa.array([7], type=pa.uint64()),
        }
    )
    payloads = pa.table(
        {
            "image": pa.array([], type=pa.binary()),
            "__nemo_fetched_stable_row_id": pa.array([], type=pa.uint64()),
        }
    )

    with pytest.raises(RuntimeError, match="omitted resolved stable IDs"):
        actor_module._apply_payloads_to_document(
            document,
            coordinates,
            payloads,
            image_columns={"image": "binary_content"},
            stable_row_id_output_column=None,
            existing_column_policy="overwrite",
        )


def test_rank_task_windows_are_bounded_contiguous_and_include_empty_ranks() -> None:
    tasks = [LanceReadTask(dataset_name=f"document-{index}", data=[index + 1]) for index in range(7)]

    windows = list(_iter_rank_task_windows(tasks, nranks=3, task_window_size=2))

    assert [[task.dataset_name for task in rank] for rank in windows[0]] == [
        ["document-0", "document-1"],
        ["document-2", "document-3"],
        ["document-4", "document-5"],
    ]
    assert [[task.dataset_name for task in rank] for rank in windows[1]] == [["document-6"], [], []]


def test_task_ref_detection_rejects_mixed_driver_state() -> None:
    refs = [executor_module.ray.ObjectRef.from_random(), executor_module.ray.ObjectRef.from_random()]

    assert _tasks_are_object_refs(refs)
    assert not _tasks_are_object_refs([LanceReadTask(dataset_name="documents", data=[1])])
    with pytest.raises(TypeError, match="must not mix"):
        _tasks_are_object_refs([refs[0], LanceReadTask(dataset_name="documents", data=[1])])


def test_streamed_task_refs_survive_cleanup_and_feed_downstream_actor(shared_ray_client: None) -> None:
    del shared_ray_client
    ray = executor_module.ray
    stage = _EchoRefStage()
    actor_class = create_named_ray_actor_pool_stage_adapter(stage, RayActorPoolStageAdapter)
    actor = actor_class.remote(stage)
    inputs = [LanceReadTask(dataset_name=f"document-{index}", data=[index + 1]) for index in range(2)]
    generator = actor.process_batch_from_refs.remote([ray.put(task) for task in inputs])

    output_refs = list(generator)
    ray.kill(actor)
    downstream = actor_class.remote(_EchoRefStage())
    downstream_refs = RayActorPoolExecutor(show_progress=False)._process_stage_with_pool(
        executor_module.ActorPool([downstream]),
        _EchoRefStage(),
        output_refs,
    )
    ray.kill(downstream)
    outputs = ray.get(downstream_refs)

    assert len(output_refs) == 2
    assert len(downstream_refs) == 2
    assert [output.dataset_name for output in outputs] == [task.dataset_name for task in inputs]


def test_windowed_shuffle_executor_delivers_bounded_windows_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [LanceReadTask(dataset_name=f"document-{index}", data=[index + 1]) for index in range(7)]
    events: list[tuple[str, int, int]] = []
    actors = [_FakeWindowRemoteActor(rank, events) for rank in range(3)]
    executor = RayActorPoolExecutor(show_progress=False)
    monkeypatch.setattr(executor_module.ray, "get", lambda values: values)

    outputs = executor._process_windowed_shuffle_stage_with_rapidsmpf_actors(
        actors,  # type: ignore[arg-type]
        tasks,
        task_window_size=2,
    )

    assert [output.dataset_name for output in outputs] == [task.dataset_name for task in tasks]
    assert [actor.insert_sizes for actor in actors] == [[2, 1], [2, 0], [2, 0]]
    assert all(size <= 2 for actor in actors for size in actor.insert_sizes)
    assert events == [
        ("read", 0, 2),
        ("read", 1, 2),
        ("read", 2, 2),
        ("finish", 0, 2),
        ("finish", 1, 2),
        ("finish", 2, 2),
        ("extract", 0, 2),
        ("extract", 1, 2),
        ("extract", 2, 2),
        ("read", 0, 1),
        ("read", 1, 0),
        ("read", 2, 0),
        ("finish", 0, 1),
        ("finish", 1, 0),
        ("finish", 2, 0),
        ("extract", 0, 1),
        ("extract", 1, 0),
        ("extract", 2, 0),
    ]


def test_ordered_shuffle_adapter_assigns_unique_parent_task_ids_and_lineage() -> None:
    inputs = [LanceReadTask(dataset_name=f"document-{index}", data=[index]) for index in range(3)]
    for index, task in enumerate(inputs):
        task.task_id = f"source_{index}"
        task._source_id = f"partition-{index}"
    stage = _OrderedOutputStage([_interleaved_output(task.dataset_name) for task in inputs])
    adapter_class = ShuffleStageAdapter.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    adapter = object.__new__(adapter_class)
    BaseStageAdapter.__init__(adapter, stage)  # type: ignore[arg-type]
    adapter._preserve_ordered_window_identity = True
    adapter._ordered_window_inputs = inputs
    adapter._ordered_window_open = True

    outputs = adapter.extract_and_write()

    assert [task.task_id for task in outputs] == ["source_0_0", "source_1_0", "source_2_0"]
    assert [task._source_id for task in outputs] == ["partition-0", "partition-1", "partition-2"]
    assert len({f"{task.task_id}.parquet" for task in outputs}) == len(outputs)
    assert adapter._ordered_window_open is False
    assert adapter._ordered_window_inputs == []


def test_shuffle_adapter_unwraps_rapidsmpf_root_address() -> None:
    class FakeActor:
        def setup_root(self) -> tuple[int, bytes]:
            return 3, b"root-address"

    adapter_class = ShuffleStageAdapter.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    adapter = object.__new__(adapter_class)
    adapter.stage = type("FakeStage", (), {"_actor_obj": FakeActor()})()
    adapter.root_address = None

    assert adapter.setup_root() == b"root-address"
    assert adapter.root_address == b"root-address"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("256MiB", 256 * 1024**2),
        ("1GiB", 1024**3),
        ("4GiB", 4 * 1024**3),
        (256 * 1024**2, 256 * 1024**2),
    ],
)
def test_fetch_window_byte_profiles(value: str | int, expected: int) -> None:
    assert _resolve_fetch_window_bytes(value) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["512MiB", 0, 512 * 1024**2, True])
def test_fetch_window_byte_profiles_reject_other_values(value: object) -> None:
    with pytest.raises(ValueError, match="fetch_window_bytes must be"):
        _stage(fetch_window_bytes=value)


@pytest.mark.parametrize(
    ("removed_argument", "value"),
    [
        ("index_image_rowaddr_column", "image_rowaddr"),
        ("fragment_take_batch_size", 1024),
        ("io_threads", 4),
    ],
)
def test_gpu_lance_shuffle_rejects_removed_row_address_options(removed_argument: str, value: object) -> None:
    with pytest.raises(TypeError, match=removed_argument):
        _stage(**{removed_argument: value})


@pytest.mark.parametrize(
    ("shards", "match"),
    [
        ([], "must not be empty"),
        ({1: "index.parquet"}, "contiguous partition IDs"),
        ([[]], "at least one non-empty path"),
        (["same.parquet", "same.parquet"], "must not reuse"),
    ],
)
def test_normalise_index_shards_rejects_invalid_layouts(
    shards: list[str | list[str]] | dict[int, str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _normalise_index_shards(shards)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"image_uri": ""}, "image_uri must not be empty"),
        (
            {"image_uri": "s3://dummy-user:dummy-pass@bucket/images?dummy-token=value#fragment"},
            "userinfo",
        ),
        ({"image_version": 0}, "image_version must be greater than zero"),
        ({"index_manifest_uri": ""}, "index_manifest_uri"),
        (
            {"index_manifest_uri": "s3://dummy-user:dummy-pass@bucket/index?dummy-token=value#fragment"},
            "userinfo",
        ),
        ({"index_manifest_sha256": ""}, "index_manifest_uri"),
        ({"image_columns": {}}, "image_columns must not be empty"),
        ({"image_columns": {"image": "payload", "width": "payload"}}, "destination names must be unique"),
        ({"image_columns": {"_rowaddr": "payload"}}, "internal coordinate columns"),
        ({"stable_row_id_output_column": ""}, "must not be empty"),
        ({"stable_row_id_output_column": "   "}, "must not be empty"),
        ({"stable_row_id_output_column": "_rowaddr"}, "internal reconstruction column"),
        (
            {"stable_row_id_output_column": "__nemo_fetched_stable_row_id"},
            "internal reconstruction column",
        ),
        (
            {"image_columns": {"image": "image_stable_id"}, "stable_row_id_output_column": "image_stable_id"},
            "must not collide",
        ),
        ({"existing_column_policy": "append"}, "Unsupported existing_column_policy"),
        ({"missing_key_policy": "drop"}, "Unsupported missing_key_policy"),
        ({"fetch_task_window": 0}, "must be positive"),
        ({"estimated_payload_bytes_per_row": 0}, "must be positive"),
        ({"fetch_batch_size": 0}, "must be positive"),
        ({"max_pending_takes": 0}, "must be positive"),
        ({"coordinate_plan_output_path": "relative/plans"}, "must be an absolute"),
        ({"document_projection": ["sample_id", "position"]}, "omits required"),
        (
            {"document_projection": ["sample_id", "position", "modality", "modality"]},
            "must not contain duplicate",
        ),
    ],
)
def test_gpu_lance_shuffle_validates_configuration(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _stage(**overrides)


def test_gpu_lance_shuffle_dispatches_collective_lifecycle() -> None:
    stage = _stage()
    actor = _FakeShuffleActor()
    stage._actor_obj = actor
    tasks = [
        LanceReadTask(
            dataset_name="documents",
            data=[3, 4],
            _metadata={"lance": {"path": "documents", "version": 2}},
        )
    ]

    assert stage.read_and_insert_batch(tasks) is tasks
    assert actor.inserted == [tasks]
    stage.insert_finished()
    assert actor.finished
    assert stage.extract_and_write() is actor.output
    stage.teardown()
    assert actor.cleaned


def test_gpu_lance_shuffle_rejects_noncollective_execution() -> None:
    stage = _stage()
    task = LanceReadTask(dataset_name="documents", data=[1])

    with pytest.raises(RuntimeError, match="actor is not initialized"):
        stage.insert_finished()
    with pytest.raises(NotImplementedError, match="RAPIDS-MPF shuffle lifecycle"):
        stage.process(task)
    with pytest.raises(NotImplementedError, match="collective shuffle stage"):
        stage.process_batch([task])
    stage._actor_obj = _FakeShuffleActor()
    with pytest.raises(TypeError, match="Expected LanceReadTask"):
        stage.read_and_insert_batch([_interleaved_output()])  # type: ignore[list-item]


def test_gpu_actor_module_import_does_not_load_optional_gpu_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    optional_roots = {"cudf", "lance", "rapidsmpf", "rmm"}
    imported_optional: list[str] = []
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name.partition(".")[0] in optional_roots:
            imported_optional.append(name)
            msg = f"optional dependency imported during CPU module load: {name}"
            raise AssertionError(msg)
        return real_import(name, globals_, locals_, fromlist, level)

    probe_name = "nemo_curator.stages.interleaved._gpu_lance_shuffle_actor_cpu_probe"
    spec = importlib.util.spec_from_file_location(probe_name, Path(actor_module.__file__))
    assert spec is not None
    assert spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, probe_name, probe)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec.loader.exec_module(probe)

    assert imported_optional == []
    assert hasattr(probe, "GpuLanceShuffleActor")


def test_gpu_lance_extra_pins_rapids_2606_and_conflicts_with_deduplication_stack() -> None:
    project = tomllib.loads((Path(__file__).resolve().parents[3] / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(project["project"]["optional-dependencies"]["gpu_lance_cuda12"])
    deduplication_dependencies = set(project["project"]["optional-dependencies"]["deduplication_cuda12"])

    assert "cudf-cu12==26.6.*" in dependencies
    assert "cupy-cuda12x>=14.1.1,<15" in dependencies
    assert "rapidsmpf-cu12==26.6.*" in dependencies
    assert "lance-ray[gpu]==0.5.0" in dependencies
    assert all("deduplication_cuda12" not in dependency for dependency in dependencies)
    assert "cudf-cu12==25.10.*" in deduplication_dependencies
    assert "rapidsmpf-cu12==25.10.*" in deduplication_dependencies
    assert project["tool"]["uv"]["sources"]["lance-ray"]["rev"] == ("ad7631238644899103225bbbe6409232ba2dd7ee")

    conflicts = {frozenset(entry["extra"] for entry in group) for group in project["tool"]["uv"]["conflicts"]}
    expected_conflicts = {
        frozenset(("gpu_lance_cuda12", extra))
        for extra in ("deduplication_cuda12", "image_cuda12", "math_cuda12", "text_cuda12", "all")
    }
    assert expected_conflicts <= conflicts
