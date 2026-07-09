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

"""Stream stable-ID Lance payloads into an attempt-local payload spool."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
    validate_lance_coordinate_plan,
)
from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from concurrent.futures import ThreadPoolExecutor


class _LanceDatasetLike(Protocol):
    def _take_rows(self, row_ids: list[int], *, columns: list[str]) -> pa.Table: ...


@dataclass(frozen=True)
class _CoordinateFetchChunk:
    row_ids: tuple[int, ...]
    coordinates: pa.Table


@dataclass(frozen=True)
class _FetchedPayloadBatch:
    table: pa.Table
    coordinates: pa.Table


@dataclass
class _CompletionSchedulerStats:
    peak_pending: int = 0
    peak_retained_batches: int = 0
    completion_rounds: int = 0
    completions_ahead_of_earlier_pending: int = 0


def _completion_order_map_iter(
    executor: ThreadPoolExecutor,
    function: Callable[[_CoordinateFetchChunk], _FetchedPayloadBatch],
    items: Iterable[_CoordinateFetchChunk],
    max_pending: int,
    stats: _CompletionSchedulerStats,
) -> Iterator[_FetchedPayloadBatch]:
    """Yield ready fetches without letting an earlier slow request block refill."""

    item_iterator = iter(items)
    pending = {}
    ready: deque[_FetchedPayloadBatch] = deque()
    input_exhausted = False
    next_sequence = 0

    def record_retained_batches(delivering: int = 0) -> None:
        retained_batches = len(pending) + len(ready) + delivering
        if retained_batches > max_pending:
            msg = f"completion scheduler retained {retained_batches} batches with max_pending={max_pending}"
            raise RuntimeError(msg)
        stats.peak_retained_batches = max(stats.peak_retained_batches, retained_batches)

    def fill_pending() -> None:
        nonlocal input_exhausted, next_sequence
        while not input_exhausted and len(pending) + len(ready) < max_pending:
            try:
                item = next(item_iterator)
            except StopIteration:
                input_exhausted = True
                break
            pending[executor.submit(function, item)] = next_sequence
            next_sequence += 1
        stats.peak_pending = max(stats.peak_pending, len(pending))
        record_retained_batches()

    try:
        fill_pending()
        while pending or ready:
            if not ready:
                completed, not_completed = wait(tuple(pending), return_when=FIRST_COMPLETED)
                del not_completed
                stats.completion_rounds += 1
                completed_with_sequence = sorted(
                    ((pending.pop(future), future) for future in completed),
                    key=lambda item: item[0],
                )
                earlier_pending = tuple(pending.values())
                for sequence, future in completed_with_sequence:
                    if any(pending_sequence < sequence for pending_sequence in earlier_pending):
                        stats.completions_ahead_of_earlier_pending += 1
                    ready.append(future.result())
                del completed, completed_with_sequence, earlier_pending, future
                record_retained_batches()
            result = ready.popleft()
            record_retained_batches(delivering=1)
            yield result
            del result
            fill_pending()
    finally:
        running = tuple(future for future in pending if not future.cancel())
        wait(running)
        pending.clear()
        ready.clear()


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _validate_inputs(  # noqa: C901
    coordinate_plan: pa.Table,
    image_source_columns: Sequence[str],
    spool: PayloadSpool,
) -> tuple[str, ...]:
    if not isinstance(coordinate_plan, pa.Table):
        msg = "coordinate_plan must be a pyarrow.Table"
        raise TypeError(msg)
    stable_field_index = coordinate_plan.schema.get_field_index(STABLE_ROW_ID)
    if stable_field_index < 0:
        msg = f"coordinate plan is missing {STABLE_ROW_ID!r}"
        raise ValueError(msg)
    stable_field = coordinate_plan.schema.field(stable_field_index)
    validate_lance_coordinate_plan(
        coordinate_plan,
        missing_key_policy="null" if stable_field.nullable else "error",
    )

    if isinstance(image_source_columns, (str, bytes)):
        msg = "image_source_columns must be a sequence of column names"
        raise TypeError(msg)
    source_columns = tuple(image_source_columns)
    if not source_columns or any(not isinstance(name, str) or not name for name in source_columns):
        msg = "image_source_columns must contain non-empty strings"
        raise ValueError(msg)
    if len(set(source_columns)) != len(source_columns):
        msg = "image_source_columns must not contain duplicates"
        raise ValueError(msg)
    coordinate_columns = (DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID)
    if set(source_columns) & set(coordinate_columns):
        msg = "image_source_columns must not collide with coordinate columns"
        raise ValueError(msg)

    if not isinstance(spool, PayloadSpool):
        msg = "spool must be a PayloadSpool"
        raise TypeError(msg)
    if spool.stable_id_column != STABLE_ROW_ID or spool.document_position_column != DOCUMENT_POSITION:
        msg = "payload spool coordinate column names do not match the coordinate plan"
        raise ValueError(msg)
    expected_columns = [*coordinate_columns, *source_columns]
    if spool.schema.names != expected_columns:
        msg = f"payload spool columns are {spool.schema.names}; expected {expected_columns}"
        raise ValueError(msg)
    for name in coordinate_columns:
        field = spool.schema.field(name)
        if field.type != pa.uint64() or field.nullable:
            msg = f"payload spool coordinate column {name!r} must be non-nullable uint64"
            raise TypeError(msg)
    return source_columns


def _sorted_coordinate_runs(coordinate_plan: pa.Table) -> tuple[pa.Table, pa.Array, pa.Array]:
    non_null_coordinates = coordinate_plan.filter(pc.is_valid(coordinate_plan[STABLE_ROW_ID]))
    sort_indices = pc.sort_indices(
        non_null_coordinates,
        sort_keys=[
            (STABLE_ROW_ID, "ascending"),
            (DOCUMENT_POSITION, "ascending"),
        ],
    )
    sorted_coordinates = non_null_coordinates.take(sort_indices)
    encoded_ids = pc.run_end_encode(sorted_coordinates[STABLE_ROW_ID].combine_chunks())
    return sorted_coordinates, encoded_ids.values, encoded_ids.run_ends


def _iter_coordinate_chunks(
    sorted_coordinates: pa.Table,
    stable_ids: pa.Array,
    run_ends: pa.Array,
    fetch_batch_size: int,
) -> Iterator[_CoordinateFetchChunk]:
    for stable_id_offset in range(0, len(stable_ids), fetch_batch_size):
        stable_id_stop = min(stable_id_offset + fetch_batch_size, len(stable_ids))
        coordinate_start = 0 if stable_id_offset == 0 else int(run_ends[stable_id_offset - 1].as_py())
        coordinate_stop = int(run_ends[stable_id_stop - 1].as_py())
        row_ids = tuple(
            int(value) for value in stable_ids.slice(stable_id_offset, stable_id_stop - stable_id_offset).to_pylist()
        )
        yield _CoordinateFetchChunk(
            row_ids=row_ids,
            coordinates=sorted_coordinates.slice(
                coordinate_start,
                coordinate_stop - coordinate_start,
            ),
        )


def _scatter_payload_batch(
    coordinate_plan: pa.Table,
    payload: pa.Table,
    source_columns: tuple[str, ...],
    output_schema: pa.Schema,
) -> pa.Table:
    payload_indices = pc.index_in(
        coordinate_plan[STABLE_ROW_ID],
        value_set=payload[STABLE_ROW_ID],
    )
    matched = pc.is_valid(payload_indices)
    coordinates = coordinate_plan.filter(matched)
    take_indices = pc.filter(payload_indices, matched)
    scattered_payload = payload.take(take_indices)
    if scattered_payload.num_rows != coordinates.num_rows:
        msg = "payload scatter row conservation failed"
        raise RuntimeError(msg)
    stable_ids_match = pc.all(
        pc.equal(
            coordinates[STABLE_ROW_ID],
            scattered_payload[STABLE_ROW_ID],
        )
    ).as_py()
    if stable_ids_match is not True:
        msg = "payload scatter produced mismatched stable row IDs"
        raise RuntimeError(msg)

    arrays_by_name: dict[str, pa.ChunkedArray] = {
        DOCUMENT_ROWADDR: coordinates[DOCUMENT_ROWADDR],
        DOCUMENT_POSITION: coordinates[DOCUMENT_POSITION],
        STABLE_ROW_ID: scattered_payload[STABLE_ROW_ID],
    }
    arrays_by_name.update({name: scattered_payload[name] for name in source_columns})
    return pa.Table.from_arrays(
        [arrays_by_name[field.name].combine_chunks() for field in output_schema],
        schema=output_schema,
    ).take(pc.sort_indices(coordinates, sort_keys=[(DOCUMENT_POSITION, "ascending")]))


def _scatter_rows_per_table(
    payload: pa.Table,
    source_columns: tuple[str, ...],
    target_bytes: int,
) -> int:
    """Bound pre-spool fan-out using the largest fetched payload row."""
    if payload.num_rows == 0:
        return 1
    largest_row_bytes = max(
        3 * 8 + sum(payload[name].slice(row_index, 1).nbytes for name in source_columns)
        for row_index in range(payload.num_rows)
    )
    return max(1, target_bytes // max(1, largest_row_bytes))


def materialize_lance_payload_to_spool(  # noqa: PLR0913, PLR0915
    dataset: _LanceDatasetLike,
    coordinate_plan: pa.Table,
    image_source_columns: Sequence[str],
    spool: PayloadSpool,
    executor: ThreadPoolExecutor,
    *,
    fetch_batch_size: int,
    max_pending: int,
) -> dict[str, int | float]:
    """Fetch each unique non-null stable ID once and spool every occurrence."""

    fetch_batch_size = _require_positive_integer(fetch_batch_size, "fetch_batch_size")
    max_pending = _require_positive_integer(max_pending, "max_pending")
    source_columns = _validate_inputs(coordinate_plan, image_source_columns, spool)
    sorted_coordinates, stable_ids, run_ends = _sorted_coordinate_runs(coordinate_plan)
    expected_payload_schema = pa.schema([spool.schema.field(name) for name in source_columns])
    timing_lock = threading.Lock()
    private_take_timings: list[tuple[float, float]] = []

    def read_chunk(chunk: _CoordinateFetchChunk) -> _FetchedPayloadBatch:
        started = time.perf_counter()
        try:
            table = dataset._take_rows(list(chunk.row_ids), columns=list(source_columns))
        finally:
            finished = time.perf_counter()
            with timing_lock:
                private_take_timings.append((started, finished))
        if not isinstance(table, pa.Table):
            msg = "private Lance _take_rows must return a pyarrow.Table"
            raise TypeError(msg)
        if table.num_rows != len(chunk.row_ids):
            msg = f"private Lance _take_rows returned {table.num_rows} rows for {len(chunk.row_ids)} stable IDs"
            raise RuntimeError(msg)
        if table.column_names != list(source_columns) or not table.schema.equals(
            expected_payload_schema,
            check_metadata=False,
        ):
            msg = "private Lance _take_rows returned an unexpected payload schema"
            raise TypeError(msg)
        stable_id_array = pa.array(chunk.row_ids, type=pa.uint64())
        attached = table.append_column(
            pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
            stable_id_array,
        )
        return _FetchedPayloadBatch(table=attached, coordinates=chunk.coordinates)

    chunks = _iter_coordinate_chunks(
        sorted_coordinates,
        stable_ids,
        run_ends,
        fetch_batch_size,
    )
    scheduler_stats = _CompletionSchedulerStats()
    iterator = _completion_order_map_iter(
        executor,
        read_chunk,
        chunks,
        max_pending,
        scheduler_stats,
    )
    take_calls = 0
    take_rows = 0
    scatter_input_rows = 0
    actual_payload_bytes = 0
    spooled_payload_bytes = 0
    try:
        for fetched_batch in iterator:
            fetched = fetched_batch.table
            take_calls += 1
            take_rows += fetched.num_rows
            actual_payload_bytes += sum(fetched[name].nbytes for name in source_columns)
            rows_per_scatter = _scatter_rows_per_table(
                fetched,
                source_columns,
                spool.target_bytes,
            )
            for offset in range(0, fetched_batch.coordinates.num_rows, rows_per_scatter):
                coordinate_slice = fetched_batch.coordinates.slice(offset, rows_per_scatter)
                scatter_input_rows += coordinate_slice.num_rows
                scattered = _scatter_payload_batch(
                    coordinate_slice,
                    fetched,
                    source_columns,
                    spool.schema,
                )
                spooled_payload_bytes += sum(scattered[name].nbytes for name in source_columns)
                spool.append(scattered)
                del coordinate_slice, scattered
            del fetched, fetched_batch
        manifest = spool.finish()
    except BaseException:
        iterator.close()
        raise

    logical_rows = sorted_coordinates.num_rows
    unique_rows = len(stable_ids)
    if scatter_input_rows != logical_rows:
        msg = f"payload scatter row conservation failed: expected {logical_rows}, processed {scatter_input_rows}"
        raise RuntimeError(msg)
    if manifest.total_rows != logical_rows:
        msg = f"payload spool row conservation failed: expected {logical_rows}, wrote {manifest.total_rows}"
        raise RuntimeError(msg)
    private_take_call_seconds_sum = sum(stop - start for start, stop in private_take_timings)
    private_take_execution_envelope_seconds = (
        max(stop for _, stop in private_take_timings) - min(start for start, _ in private_take_timings)
        if private_take_timings
        else 0.0
    )
    return {
        "logical_rows": logical_rows,
        "unique_rows": unique_rows,
        "null_rows_skipped": coordinate_plan[STABLE_ROW_ID].null_count,
        "duplicate_fanout": logical_rows / unique_rows if unique_rows else 0.0,
        "take_calls": take_calls,
        "take_rows": take_rows,
        "scatter_input_rows": scatter_input_rows,
        "peak_pending": scheduler_stats.peak_pending,
        "peak_retained_batches": scheduler_stats.peak_retained_batches,
        "completion_rounds": scheduler_stats.completion_rounds,
        "completions_ahead_of_earlier_pending": scheduler_stats.completions_ahead_of_earlier_pending,
        "sparse_calls_avoided": max(0, unique_rows - take_calls),
        "private_take_call_seconds_sum": private_take_call_seconds_sum,
        "private_take_execution_envelope_seconds": private_take_execution_envelope_seconds,
        "actual_payload_bytes": actual_payload_bytes,
        "spooled_payload_bytes": spooled_payload_bytes,
        "spool_arrow_bytes": manifest.total_arrow_nbytes,
    }
