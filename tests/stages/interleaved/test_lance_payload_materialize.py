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
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
    lance_coordinate_plan_schema,
)
from nemo_curator.stages.interleaved.lance_payload_materialize import materialize_lance_payload_to_spool
from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


_PAYLOAD_SCHEMA = pa.schema(
    [
        pa.field("image", pa.large_binary(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
    ]
)
_SPOOL_SCHEMA = pa.schema(
    [
        pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
        pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
        pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
        *_PAYLOAD_SCHEMA,
    ],
    metadata={b"purpose": b"stable-id-materialize-test"},
)


def _coordinate_plan(stable_ids: list[int | None]) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(range(100, 100 + len(stable_ids)), type=pa.uint64()),
            pa.array(range(len(stable_ids)), type=pa.uint64()),
            pa.array(stable_ids, type=pa.uint64()),
        ],
        schema=lance_coordinate_plan_schema(allow_missing=True),
    )


def _payload_spool(tmp_path: Path, name: str = "payload") -> PayloadSpool:
    return PayloadSpool(
        tmp_path / name,
        _SPOOL_SCHEMA,
        target_bytes=4096,
        bucket_rows=100,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
    )


def test_materialize_replenishes_from_completion_order_and_preserves_coordinates(tmp_path: Path) -> None:
    class _SyntheticDataset:
        def __init__(self) -> None:
            self.table = pa.Table.from_arrays(
                [
                    pa.array([b"one", b"two", b"three", b"four", b"five", b"six"], type=pa.large_binary()),
                    pa.array([10, 20, 30, 40, 50, 60], type=pa.int32()),
                ],
                schema=_PAYLOAD_SCHEMA,
            )
            self.first_started = threading.Event()
            self.third_appended = threading.Event()
            self.lock = threading.Lock()
            self.requests: list[tuple[int, ...]] = []
            self.completion_order: list[tuple[int, ...]] = []
            self.actual_payload_bytes = 0

        def _take_rows(self, row_ids: list[int], *, columns: list[str]) -> pa.Table:
            request = tuple(row_ids)
            with self.lock:
                self.requests.append(request)
            if request == (1, 2):
                self.first_started.set()
                if not self.third_appended.wait(timeout=5):
                    msg = "third payload chunk was not replenished and appended"
                    raise TimeoutError(msg)
            if request == (3, 4) and not self.first_started.wait(timeout=5):
                msg = "first payload chunk did not start"
                raise TimeoutError(msg)
            indices = pa.array([row_id - 1 for row_id in row_ids], type=pa.int64())
            table = self.table.take(indices).select(columns)
            with self.lock:
                self.completion_order.append(request)
                self.actual_payload_bytes += sum(table[name].nbytes for name in columns)
            return table

    class _RecordingPayloadSpool(PayloadSpool):
        def __init__(self, root: Path) -> None:
            super().__init__(
                root,
                _SPOOL_SCHEMA,
                target_bytes=4096,
                bucket_rows=100,
                stable_id_column=STABLE_ROW_ID,
                document_position_column=DOCUMENT_POSITION,
            )
            self.appended: list[pa.Table] = []

        def append(self, table: pa.Table) -> None:
            self.appended.append(table)
            super().append(table)
            if table[STABLE_ROW_ID].to_pylist() == [5, 6]:
                dataset.third_appended.set()

    coordinate_plan = _coordinate_plan([5, 1, None, 4, 1, 6, 2, 3])
    dataset = _SyntheticDataset()
    spool = _RecordingPayloadSpool(tmp_path / "payload")

    with ThreadPoolExecutor(max_workers=2) as executor:
        metrics = materialize_lance_payload_to_spool(
            dataset,
            coordinate_plan,
            ("image", "width"),
            spool,
            executor,
            fetch_batch_size=2,
            max_pending=2,
        )

    assert dataset.completion_order == [(3, 4), (5, 6), (1, 2)]
    assert sorted(dataset.requests) == [(1, 2), (3, 4), (5, 6)]
    assert [table[STABLE_ROW_ID].to_pylist() for table in spool.appended] == [[4, 3], [5, 6], [1, 1, 2]]
    assert [table[DOCUMENT_POSITION].to_pylist() for table in spool.appended] == [[3, 7], [0, 5], [1, 4, 6]]
    assert all(
        table[DOCUMENT_POSITION].to_pylist() == sorted(table[DOCUMENT_POSITION].to_pylist())
        for table in spool.appended
    )

    output = spool.read_all()
    assert output.schema.equals(_SPOOL_SCHEMA, check_metadata=True)
    output = output.sort_by([(DOCUMENT_POSITION, "ascending")])
    assert output[DOCUMENT_ROWADDR].to_pylist() == [100, 101, 103, 104, 105, 106, 107]
    assert output[DOCUMENT_POSITION].to_pylist() == [0, 1, 3, 4, 5, 6, 7]
    assert output[STABLE_ROW_ID].to_pylist() == [5, 1, 4, 1, 6, 2, 3]
    assert output["image"].to_pylist() == [b"five", b"one", b"four", b"one", b"six", b"two", b"three"]
    assert output["width"].to_pylist() == [50, 10, 40, 10, 60, 20, 30]

    private_take_call_seconds_sum = metrics.pop("private_take_call_seconds_sum")
    private_take_execution_envelope_seconds = metrics.pop("private_take_execution_envelope_seconds")
    assert private_take_call_seconds_sum > 0
    assert private_take_execution_envelope_seconds > 0
    assert metrics == {
        "logical_rows": 7,
        "unique_rows": 6,
        "null_rows_skipped": 1,
        "duplicate_fanout": pytest.approx(7 / 6),
        "take_calls": 3,
        "take_rows": 6,
        "scatter_input_rows": 7,
        "peak_pending": 2,
        "peak_retained_batches": 2,
        "completion_rounds": 3,
        "completions_ahead_of_earlier_pending": 2,
        "sparse_calls_avoided": 3,
        "actual_payload_bytes": dataset.actual_payload_bytes,
        "spooled_payload_bytes": sum(table["image"].nbytes + table["width"].nbytes for table in spool.appended),
        "spool_arrow_bytes": spool.finish().total_arrow_nbytes,
    }
    spool.cleanup()


def test_materialize_rejects_incorrect_fetch_rows_and_cancels_pending(tmp_path: Path) -> None:
    class _WrongRowDataset:
        def __init__(self) -> None:
            self.calls: list[list[int]] = []

        def _take_rows(self, row_ids: list[int], *, columns: list[str]) -> pa.Table:
            self.calls.append(row_ids)
            return pa.Table.from_arrays(
                [pa.array([b"wrong"], type=pa.large_binary()), pa.array([99], type=pa.int32())],
                schema=_PAYLOAD_SCHEMA,
            ).select(columns)

    class _ManualExecutor:
        def __init__(self) -> None:
            self.futures: list[Future[object]] = []

        def submit(
            self,
            function: Callable[[tuple[int, ...]], object],
            item: tuple[int, ...],
        ) -> Future[object]:
            future: Future[object] = Future()
            execute = not self.futures
            self.futures.append(future)
            if execute:
                try:
                    future.set_result(function(item))
                except Exception as exc:  # noqa: BLE001
                    future.set_exception(exc)
            return future

    dataset = _WrongRowDataset()
    executor = _ManualExecutor()
    spool = _payload_spool(tmp_path)

    with pytest.raises(RuntimeError, match="returned 1 rows for 2 stable IDs"):
        materialize_lance_payload_to_spool(
            dataset,
            _coordinate_plan([1, 2, 3, 4]),
            ("image", "width"),
            spool,
            executor,
            fetch_batch_size=2,
            max_pending=2,
        )

    assert dataset.calls == [[1, 2]]
    assert len(executor.futures) == 2
    assert executor.futures[1].cancelled()
    with pytest.raises(RuntimeError, match="finish must be called"):
        spool.iter_tables()
    spool.cleanup()


def test_materialize_bounds_duplicate_fanout_before_spool_append(tmp_path: Path) -> None:
    class _OneRowDataset:
        def _take_rows(self, row_ids: list[int], *, columns: list[str]) -> pa.Table:
            assert row_ids == [1]
            return pa.Table.from_arrays(
                [pa.array([b"x" * 64], type=pa.large_binary()), pa.array([10], type=pa.int32())],
                schema=_PAYLOAD_SCHEMA,
            ).select(columns)

    class _RecordingSpool(PayloadSpool):
        def __init__(self, root: Path) -> None:
            super().__init__(
                root,
                _SPOOL_SCHEMA,
                target_bytes=128,
                bucket_rows=100,
                stable_id_column=STABLE_ROW_ID,
                document_position_column=DOCUMENT_POSITION,
            )
            self.append_sizes: list[int] = []

        def append(self, table: pa.Table) -> None:
            self.append_sizes.append(table.nbytes)
            super().append(table)

    spool = _RecordingSpool(tmp_path / "fanout")
    with ThreadPoolExecutor(max_workers=1) as executor:
        metrics = materialize_lance_payload_to_spool(
            _OneRowDataset(),
            _coordinate_plan([1] * 20),
            ("image", "width"),
            spool,
            executor,
            fetch_batch_size=1024,
            max_pending=1,
        )

    assert len(spool.append_sizes) > 1
    assert all(size <= spool.target_bytes for size in spool.append_sizes)
    assert metrics["logical_rows"] == 20
    assert metrics["unique_rows"] == 1
    assert metrics["scatter_input_rows"] == 20
    spool.cleanup()
