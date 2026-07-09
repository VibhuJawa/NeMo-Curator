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

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved import (
    GpuLanceColumnFetchStage,
    LanceColumnFetchStage,
    LanceDatasetConfig,
    LanceIndexCacheConfig,
    gpu_lance,
)
from nemo_curator.stages.interleaved.gpu_key_lookup import (
    _build_sidecar_contract_bytes,
    _stable_global_ordinal_manifest_sha256,
)
from nemo_curator.stages.interleaved.lance import (
    _bounded_parallel_map,
    _bounded_parallel_map_iter,
    _LanceColumnFetcher,
    _plan_adaptive_locality_reads,
    _plan_private_takes,
    _StableGlobalOrdinalManifest,
    _validate_stable_global_ordinal_manifest,
)
from nemo_curator.tasks import InterleavedBatch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

lance = pytest.importorskip("lance")


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("url", pa.string(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
        ]
    )


def _rows(values: list[tuple[str, int]]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"url": key, "image": f"image-{key}".encode(), "width": width} for key, width in values],
        schema=_schema(),
    )


def _write_fragmented_dataset(path: Path) -> int:
    fragments = [
        [("a", 100), ("b", 200)],
        [("c", 300), ("d", 400), ("e", 500)],
        [("f", 600)],
    ]
    for index, values in enumerate(fragments):
        options = {"enable_stable_row_ids": True} if index == 0 else {}
        lance.write_dataset(
            _rows(values),
            str(path),
            mode="create" if index == 0 else "append",
            data_storage_version="2.2",
            **options,
        )
    dataset = lance.dataset(str(path))
    dataset.create_scalar_index("url", "BTREE", name="url_btree")
    return lance.dataset(str(path)).version


def _dataset_config(path: Path, version: int) -> LanceDatasetConfig:
    return LanceDatasetConfig(uri=str(path), version=version, key_column="url", index_name="url_btree")


def _index_cache(path: Path) -> LanceIndexCacheConfig:
    del path
    return LanceIndexCacheConfig(
        prewarm=False,
        index_cache_size_bytes=1024**2,
        metadata_cache_size_bytes=1024**2,
    )


def _batch(keys: list[str]) -> InterleavedBatch:
    return InterleavedBatch(
        dataset_name="documents",
        data=pa.table(
            {
                "sample_id": [f"sample-{index}" for index in range(len(keys))],
                "position": pa.array(range(len(keys)), type=pa.int32()),
                "modality": ["image"] * len(keys),
                "source_ref": keys,
            }
        ),
    )


def _cpu_stage(path: Path, version: int) -> LanceColumnFetchStage:
    return LanceColumnFetchStage(
        dataset=_dataset_config(path, version),
        index_cache=_index_cache(path),
        columns={"image": "binary_content", "width": "reference_width"},
        presence_column="image_present",
        fetch_batch_size=2,
        max_pending_takes=2,
    )


class _FakeMapper:
    def __init__(  # noqa: PLR0913
        self,
        reference_files: list[str],
        reference_key_column: str,
        reference_row_id_column: str,
        storage_options: dict[str, str],
        expected_reference_rows: int,
        load_factor: float,
    ) -> None:
        del storage_options, load_factor
        table = pq.read_table(reference_files, columns=[reference_key_column, reference_row_id_column])
        self.reference_type = table.schema.field(reference_key_column).type
        self.reference_rows = table.num_rows
        if self.reference_rows != expected_reference_rows:
            msg = "unexpected reference row count"
            raise ValueError(msg)
        self.load_seconds = 0.0
        self.build_seconds = 0.0
        self.gpu_bytes = 0
        self.gpu_total_bytes = 0
        self.closed = False
        self._row_ids = dict(
            zip(
                table[reference_key_column].combine_chunks().to_pylist(),
                table[reference_row_id_column].combine_chunks().to_pylist(),
                strict=True,
            )
        )

    def map(self, keys: pa.Array) -> SimpleNamespace:
        values = keys.to_pylist()
        return SimpleNamespace(
            matched=np.array([value in self._row_ids for value in values], dtype=np.bool_),
            row_ids=np.array([self._row_ids.get(value, 0) for value in values], dtype=np.uint64),
            transfer_seconds=0.0,
            probe_seconds=0.0,
            search_seconds=0.0,
            gather_seconds=0.0,
        )

    def close(self) -> None:
        self.closed = True


def _write_sidecar(path: Path, sidecar: Path) -> None:
    table = lance.dataset(str(path)).scanner(columns=["url"], with_row_id=True).to_table()
    pq.write_table(
        pa.table(
            {
                "url": table["url"].combine_chunks(),
                "stable_row_id": table["_rowid"].combine_chunks().cast(pa.uint64()),
            }
        ).sort_by([("url", "ascending")]),
        sidecar,
    )


def _write_sidecar_contract(path: Path, version: int, sidecar: Path) -> tuple[Path, str]:
    dataset = lance.dataset(str(path), version=version)
    manifest = _validate_stable_global_ordinal_manifest(dataset)
    fragment_digest = _stable_global_ordinal_manifest_sha256(str(path), version, manifest)
    raw_contract, contract_digest = _build_sidecar_contract_bytes(
        dataset=dataset,
        dataset_uri=str(path),
        dataset_version=version,
        fragment_manifest_sha256=fragment_digest,
        total_rows=manifest.total_rows,
        key_column="url",
        row_id_column="stable_row_id",
        layout="replicated_sorted",
        partition_files=((str(sidecar),),),
        storage_options={},
    )
    contract = sidecar.with_suffix(".manifest.json")
    contract.write_bytes(raw_contract)
    return contract, contract_digest


def test_private_stable_ordinal_fetch_preserves_order_and_reports_sparse_metrics(tmp_path: Path) -> None:
    path = tmp_path / "fragmented.lance"
    stage = _cpu_stage(path, _write_fragmented_dataset(path))

    output = stage.process(_batch(["f", "e", "a", "d"])).to_pyarrow()
    metrics = stage._consume_custom_metrics()
    stage.teardown()

    assert output["source_ref"].to_pylist() == ["f", "e", "a", "d"]
    assert output["binary_content"].to_pylist() == [b"image-f", b"image-e", b"image-a", b"image-d"]
    assert output["reference_width"].to_pylist() == [600, 500, 100, 400]
    assert output["image_present"].to_pylist() == [True, True, True, True]
    assert metrics["private_take_calls"] == 2.0
    assert metrics["private_take_rows"] == 4.0
    assert metrics["rows_per_private_take"] == 2.0
    assert metrics["max_pending_private_takes"] == 2.0
    assert metrics["coordinate_density"] == pytest.approx(4 / 6)
    assert metrics["sparse_calls_avoided"] == 2.0
    assert metrics["logical_payload_requests"] == 4.0
    assert metrics["unique_payloads"] == 4.0
    assert metrics["duplicate_fanout"] == 1.0
    assert metrics["average_physical_read_bytes"] >= 0.0
    assert metrics["physical_reads_per_payload"] >= 0.0
    assert metrics["physical_read_operations_per_second"] == pytest.approx(
        metrics["lance_read_iops"] / metrics["private_take_seconds"]
    )
    assert metrics["read_amplification"] >= 0.0
    assert metrics["stage_windows"] == 1.0


def test_private_stable_ordinal_backend_is_shared_by_cpu_and_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "fragmented.lance"
    version = _write_fragmented_dataset(path)
    sidecar = tmp_path / "stable-row-ids.parquet"
    _write_sidecar(path, sidecar)
    contract, contract_digest = _write_sidecar_contract(path, version, sidecar)
    monkeypatch.setattr(gpu_lance, "_GpuExactKeyMapper", _FakeMapper)

    cpu_stage = _cpu_stage(path, version)
    gpu_stage = GpuLanceColumnFetchStage(
        dataset=_dataset_config(path, version),
        index_cache=_index_cache(path),
        reference_files=[str(sidecar)],
        reference_manifest_uri=str(contract),
        reference_manifest_sha256=contract_digest,
        expected_reference_rows=6,
        columns={"image": "binary_content", "width": "reference_width"},
        presence_column="image_present",
        fetch_batch_size=2,
        max_pending_takes=2,
    )
    task = _batch(["e", "a", "f", "b", "e"])

    cpu_output = cpu_stage.process(task).to_pyarrow()
    gpu_output = gpu_stage.process(task).to_pyarrow()
    cpu_metrics = cpu_stage._consume_custom_metrics()
    gpu_metrics = gpu_stage._consume_custom_metrics()
    cpu_stage.teardown()
    gpu_stage.teardown()

    assert gpu_output.equals(cpu_output)
    for name in (
        "private_take_calls",
        "private_take_rows",
        "rows_per_private_take",
        "sparse_calls_avoided",
        "logical_payload_requests",
        "unique_payloads",
        "logical_duplicate_requests",
        "duplicate_fanout",
    ):
        assert gpu_metrics[name] == cpu_metrics[name]
    assert cpu_metrics["logical_payload_requests"] == 5.0
    assert cpu_metrics["unique_payloads"] == 4.0
    assert cpu_metrics["logical_duplicate_requests"] == 1.0
    assert cpu_metrics["duplicate_fanout"] == 1.25


def test_stable_global_ordinal_fetch_rejects_deletions(tmp_path: Path) -> None:
    path = tmp_path / "deleted.lance"
    _write_fragmented_dataset(path)
    lance.dataset(str(path)).delete("url = 'c'")
    version = lance.dataset(str(path)).version
    stage = _cpu_stage(path, version)

    with pytest.raises(ValueError, match="append-only dataset without deletions"):
        stage.setup()

    assert stage._fetcher is None


def test_stable_global_ordinal_fetch_rejects_out_of_range_ids(tmp_path: Path) -> None:
    path = tmp_path / "fragmented.lance"
    stage = _cpu_stage(path, _write_fragmented_dataset(path))
    stage.setup()
    assert stage._fetcher is not None

    with pytest.raises(ValueError, match=r"outside global-ordinal range \[0, 6\)"):
        stage._fetcher._take_rows([-1])
    with pytest.raises(ValueError, match=r"outside global-ordinal range \[0, 6\)"):
        stage._fetcher._take_rows([6])
    stage.teardown()


def test_private_take_plan_sorts_deduplicates_and_reports_density() -> None:
    plan = _plan_private_takes([5, 1, 5, 0, 3], total_rows=6, fetch_batch_size=2)

    assert plan.row_ids == (0, 1, 3, 5)
    assert plan.batches == ((0, 1), (3, 5))
    assert plan.coordinate_density == pytest.approx(4 / 6)


def test_adaptive_locality_plan_selects_sparse_range_and_fragment_reads() -> None:
    manifest = _StableGlobalOrdinalManifest(
        fragment_starts=(0, 10, 20),
        fragment_rows=(10, 10, 10),
        total_rows=30,
    )

    plan = _plan_adaptive_locality_reads(
        [0, 2, 10, 11, 13, *range(20, 28)],
        manifest=manifest,
        fetch_batch_size=2,
        payload_read_mode="adaptive_unmeasured",
        medium_density_threshold=0.3,
        high_density_threshold=0.8,
        max_coalesced_range_gap=0,
    )

    assert [operation.strategy for operation in plan.operations] == [
        "take_rows",
        "take_scan_ranges",
        "take_scan_fragment",
    ]
    assert plan.operations[0].row_ids == (0, 2)
    assert plan.operations[1].ranges == ((10, 12), (13, 14))
    assert plan.operations[2].ranges == ((20, 30),)
    assert plan.sparse_fragments == 1
    assert plan.range_fragments == 1
    assert plan.sequential_fragments == 1
    assert plan.take_scan_ranges == 3
    assert plan.planned_scan_rows == 13
    assert plan.range_overread_rows == 2


def test_adaptive_locality_fetch_preserves_rows_and_reports_strategy_metrics(tmp_path: Path) -> None:
    path = tmp_path / "fragmented.lance"
    stage = LanceColumnFetchStage(
        dataset=_dataset_config(path, _write_fragmented_dataset(path)),
        index_cache=_index_cache(path),
        columns={"image": "binary_content"},
        presence_column="image_present",
        fetch_batch_size=2,
        max_pending_takes=2,
        payload_read_mode="adaptive_unmeasured",
        medium_density_threshold=0.5,
        high_density_threshold=0.9,
    )

    output = stage.process(_batch(["f", "c", "a"])).to_pyarrow()
    metrics = stage._consume_custom_metrics()
    stage.teardown()

    assert output["binary_content"].to_pylist() == [b"image-f", b"image-c", b"image-a"]
    assert metrics["strategy_sparse_fragments"] == 1.0
    assert metrics["strategy_range_fragments"] == 1.0
    assert metrics["strategy_sequential_fragments"] == 1.0
    assert metrics["take_rows_calls"] == 1.0
    assert metrics["take_scan_calls"] == 2.0
    assert metrics["take_scan_ranges"] == 2.0


@pytest.mark.parametrize(
    ("validate_payload_keys", "expected_projection"),
    [
        (False, ["image", "width"]),
        (True, ["url", "image", "width"]),
    ],
)
def test_payload_projection_adds_key_only_for_opt_in_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validate_payload_keys: bool,
    expected_projection: list[str],
) -> None:
    path = tmp_path / "fragmented.lance"
    stage = _cpu_stage(path, _write_fragmented_dataset(path))
    stage.validate_payload_keys = validate_payload_keys
    stage.setup()
    assert stage._fetcher is not None
    dataset_type = type(stage._fetcher.remote_dataset)
    original_take_rows = dataset_type._take_rows
    projections: list[list[str]] = []

    def record_projection(
        dataset: object,
        row_ids: list[int],
        *,
        columns: list[str],
    ) -> pa.Table:
        projections.append(columns)
        return original_take_rows(dataset, row_ids, columns=columns)

    monkeypatch.setattr(dataset_type, "_take_rows", record_projection)
    output = stage.process(_batch(["a"])).to_pyarrow()
    stage.teardown()

    assert output["binary_content"].to_pylist() == [b"image-a"]
    assert projections == [expected_projection]


def test_bounded_parallel_map_preserves_order_and_pending_limit() -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs, peak_pending = _bounded_parallel_map(executor, lambda value: value * 2, list(range(7)), 2)

    assert outputs == [0, 2, 4, 6, 8, 10, 12]
    assert peak_pending == 2


def test_bounded_parallel_map_iterator_is_lazy_ordered_and_bounded() -> None:
    class _ManualExecutor:
        def __init__(self) -> None:
            self.items: list[int] = []
            self.futures: list[Future[int]] = []

        def submit(self, _function: object, item: int) -> Future[int]:
            future: Future[int] = Future()
            self.items.append(item)
            self.futures.append(future)
            return future

    pulled: list[int] = []

    def items() -> Iterator[int]:
        for item in range(5):
            pulled.append(item)
            yield item

    executor = _ManualExecutor()
    iterator = _bounded_parallel_map_iter(executor, lambda value: value * 10, items(), 2)

    assert pulled == [0, 1]
    assert iterator.peak_pending == 2
    executor.futures[1].set_result(10)
    executor.futures[0].set_result(0)
    assert next(iterator) == 0
    assert pulled == [0, 1, 2]
    assert next(iterator) == 10
    assert pulled == [0, 1, 2, 3]

    executor.futures[3].set_result(30)
    executor.futures[2].set_result(20)
    assert next(iterator) == 20
    assert pulled == [0, 1, 2, 3, 4]
    assert next(iterator) == 30

    executor.futures[4].set_result(40)
    assert next(iterator) == 40
    with pytest.raises(StopIteration):
        next(iterator)

    assert executor.items == [0, 1, 2, 3, 4]
    assert iterator.peak_pending == 2


def test_bounded_parallel_map_iterator_cancels_pending_futures_on_error() -> None:
    class _ManualExecutor:
        def __init__(self) -> None:
            self.futures: list[Future[int]] = []

        def submit(self, _function: object, _item: int) -> Future[int]:
            future: Future[int] = Future()
            self.futures.append(future)
            return future

    executor = _ManualExecutor()
    iterator = _bounded_parallel_map_iter(executor, lambda value: value, range(5), 3)
    executor.futures[0].set_exception(RuntimeError("failed read"))

    with pytest.raises(RuntimeError, match="failed read"):
        next(iterator)

    assert executor.futures[1].cancelled()
    assert executor.futures[2].cancelled()
    assert iterator.peak_pending == 3


def test_bounded_parallel_map_iterator_drains_running_futures_on_error() -> None:
    barrier = threading.Barrier(2)
    second_finished = threading.Event()

    def execute(value: int) -> int:
        barrier.wait(timeout=5)
        if value == 0:
            msg = "failed read"
            raise RuntimeError(msg)
        second_finished.set()
        return value

    with ThreadPoolExecutor(max_workers=2) as executor:
        iterator = _bounded_parallel_map_iter(executor, execute, range(2), 2)
        with pytest.raises(RuntimeError, match="failed read"):
            next(iterator)

    assert second_finished.is_set()


def test_mirror_lookup_and_remote_payload_io_stats_are_isolated() -> None:
    class _StatsDataset:
        def __init__(self, stats: list[tuple[int, int]]) -> None:
            self._stats = iter(stats)
            self.calls = 0

        def io_stats_incremental(self) -> SimpleNamespace:
            self.calls += 1
            read_bytes, read_iops = next(self._stats)
            return SimpleNamespace(read_bytes=read_bytes, read_iops=read_iops)

    fetcher = object.__new__(_LanceColumnFetcher)
    fetcher.remote_dataset = _StatsDataset([(900, 9), (0, 0), (600, 6)])
    fetcher.index_dataset = _StatsDataset([(800, 8), (120, 3)])
    fetcher.columns = {}
    fetcher.validate_payload_keys = False
    fetcher._resolve_row_ids = lambda _keys: ({"key": 0}, {})
    fetcher._take_rows = lambda _row_ids: (
        [],
        {"private_take_rows": 1.0, "private_take_seconds": 2.0},
    )

    result = fetcher.fetch(["key"])

    assert result.lookup_metrics["lookup_read_bytes"] == 120.0
    assert result.lookup_metrics["lookup_read_iops"] == 3.0
    assert result.lookup_metrics["physical_read_operations_per_second"] == 3.0
    assert result.read_bytes == 600
    assert result.read_iops == 6
    assert fetcher.index_dataset.calls == 2
    assert fetcher.remote_dataset.calls == 3


def test_stable_global_ordinal_manifest_requires_contiguous_fragment_ids() -> None:
    fragments = [
        SimpleNamespace(
            fragment_id=0,
            physical_rows=2,
            num_deletions=0,
            deletion_file=lambda: None,
            metadata=SimpleNamespace(physical_rows=2, deletion_file=None),
        ),
        SimpleNamespace(
            fragment_id=2,
            physical_rows=1,
            num_deletions=0,
            deletion_file=lambda: None,
            metadata=SimpleNamespace(physical_rows=1, deletion_file=None),
        ),
    ]
    dataset = SimpleNamespace(get_fragments=lambda: fragments, count_rows=lambda: 3)

    with pytest.raises(ValueError, match="contiguous manifest-order fragment IDs"):
        _validate_stable_global_ordinal_manifest(dataset)


def test_stable_global_ordinal_manifest_requires_matching_physical_totals() -> None:
    fragment = SimpleNamespace(
        fragment_id=0,
        physical_rows=2,
        num_deletions=0,
        deletion_file=lambda: None,
        metadata=SimpleNamespace(physical_rows=2, deletion_file=None),
    )
    dataset = SimpleNamespace(get_fragments=lambda: [fragment], count_rows=lambda: 3)

    with pytest.raises(ValueError, match="complete physical-row coverage"):
        _validate_stable_global_ordinal_manifest(dataset)


@pytest.mark.parametrize("removed_argument", ["row_id_layout", "io_threads"])
def test_lance_column_fetch_rejects_removed_compatibility_options(
    tmp_path: Path,
    removed_argument: str,
) -> None:
    path = tmp_path / "fragmented.lance"
    version = _write_fragmented_dataset(path)
    kwargs = {
        "dataset": _dataset_config(path, version),
        "index_cache": _index_cache(path),
        "columns": {"image": "binary_content"},
        "presence_column": "image_present",
        removed_argument: "fragment_ordinal" if removed_argument == "row_id_layout" else 16,
    }

    with pytest.raises(TypeError, match=removed_argument):
        LanceColumnFetchStage(**kwargs)  # type: ignore[arg-type]


def test_lance_column_fetch_uses_provisional_medium_take_defaults(tmp_path: Path) -> None:
    path = tmp_path / "fragmented.lance"
    stage = LanceColumnFetchStage(
        dataset=_dataset_config(path, _write_fragmented_dataset(path)),
        index_cache=_index_cache(path),
        columns={"image": "binary_content"},
        presence_column="image_present",
    )

    assert stage.fetch_batch_size == 1_024
    assert stage.max_pending_takes == 16
    assert stage.payload_read_mode == "sparse"
    assert stage.medium_density_threshold == 0.25
    assert stage.high_density_threshold == 0.75
    assert stage.max_coalesced_range_gap == 0
    assert stage.take_scan_batch_readahead == 16
    assert stage.validate_payload_keys is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"payload_read_mode": "automatic"}, "Unsupported payload_read_mode"),
        ({"medium_density_threshold": 0.8}, "Density thresholds"),
        ({"high_density_threshold": 1.1}, "Density thresholds"),
        ({"max_coalesced_range_gap": -1}, "nonnegative"),
        ({"take_scan_batch_readahead": 0}, "greater than 0"),
    ],
)
def test_lance_column_fetch_validates_adaptive_locality_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    kwargs = {
        "dataset": LanceDatasetConfig(
            uri="unused.lance",
            version=1,
            key_column="url",
            index_name="url_btree",
        ),
        "columns": {"image": "binary_content"},
        "presence_column": "image_present",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        LanceColumnFetchStage(**kwargs)  # type: ignore[arg-type]
