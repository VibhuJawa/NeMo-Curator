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
    from collections.abc import Iterator, Sequence


class _StableIdPayloadStreamer(Protocol):
    last_metrics: dict[str, int | float | bool]

    def close(self) -> None: ...

    def iter_stable_row_ids(
        self,
        values: pa.Array | pa.ChunkedArray | pa.Table,
        *,
        stable_row_id_column: str = STABLE_ROW_ID,
    ) -> Iterator[pa.Table]: ...


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


def _lower_bound_uint64(values: pa.Array, target: int) -> int:
    """Locate one uint64 value without materializing the full Arrow array."""
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if int(values[middle].as_py()) < target:
            low = middle + 1
        else:
            high = middle
    return low


def _locate_requested_interval(requested: pa.Array, returned: pa.Array) -> tuple[int, int]:
    """Map one sorted operation result to its exact requested-ID interval."""
    first = int(returned[0].as_py())
    start = _lower_bound_uint64(requested, first)
    if start == len(requested) or int(requested[start].as_py()) != first:
        msg = f"Lance stable-ID payload stream returned unknown stable row ID {first}"
        raise RuntimeError(msg)
    stop = start + len(returned)
    if stop > len(requested) or not returned.equals(requested.slice(start, len(returned))):
        msg = "Lance stable-ID payload batch is not one contiguous interval of requested stable row IDs"
        raise RuntimeError(msg)
    return start, stop


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


def materialize_lance_payload_to_spool(  # noqa: C901, PLR0912, PLR0915
    payload_streamer: _StableIdPayloadStreamer,
    coordinate_plan: pa.Table,
    image_source_columns: Sequence[str],
    spool: PayloadSpool,
) -> dict[str, int | float | bool]:
    """Consume unique stable-ID payload batches and spool every occurrence."""

    source_columns = _validate_inputs(coordinate_plan, image_source_columns, spool)
    sorted_coordinates, stable_ids, run_ends = _sorted_coordinate_runs(coordinate_plan)
    expected_payload_schema = pa.schema(
        [
            pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
            *(spool.schema.field(name) for name in source_columns),
        ]
    )
    iterator = payload_streamer.iter_stable_row_ids(stable_ids)
    covered_unique_rows = bytearray(len(stable_ids))
    unique_rows_covered = 0
    payload_batches = 0
    scatter_input_rows = 0
    actual_payload_bytes = 0
    spooled_payload_bytes = 0
    try:
        for fetched in iterator:
            if not isinstance(fetched, pa.Table):
                msg = "Lance stable-ID payload streamer must yield pyarrow.Table batches"
                raise TypeError(msg)
            if fetched.num_rows <= 0:
                msg = "Lance stable-ID payload streamer yielded an empty batch"
                raise RuntimeError(msg)
            if not fetched.schema.equals(expected_payload_schema, check_metadata=True):
                msg = f"Lance stable-ID payload schema is {fetched.schema}; expected {expected_payload_schema}"
                raise TypeError(msg)

            returned_stable_ids = fetched[STABLE_ROW_ID].combine_chunks()
            unique_row_start, unique_row_stop = _locate_requested_interval(stable_ids, returned_stable_ids)
            if covered_unique_rows.find(b"\x01", unique_row_start, unique_row_stop) >= 0:
                msg = "Lance stable-ID payload stream returned an overlapping or duplicate stable row-ID interval"
                raise RuntimeError(msg)

            coordinate_start = 0 if unique_row_start == 0 else int(run_ends[unique_row_start - 1].as_py())
            coordinate_stop = int(run_ends[unique_row_stop - 1].as_py())
            batch_coordinates = sorted_coordinates.slice(
                coordinate_start,
                coordinate_stop - coordinate_start,
            )
            payload_batches += 1
            actual_payload_bytes += sum(fetched[name].nbytes for name in source_columns)
            rows_per_scatter = _scatter_rows_per_table(
                fetched,
                source_columns,
                spool.target_bytes,
            )
            for offset in range(0, batch_coordinates.num_rows, rows_per_scatter):
                coordinate_slice = batch_coordinates.slice(offset, rows_per_scatter)
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
            covered_unique_rows[unique_row_start:unique_row_stop] = b"\x01" * fetched.num_rows
            unique_rows_covered += fetched.num_rows
            del batch_coordinates, fetched
    finally:
        iterator.close()

    logical_rows = sorted_coordinates.num_rows
    unique_rows = len(stable_ids)
    if unique_rows_covered != unique_rows:
        first_missing_index = covered_unique_rows.find(b"\x00")
        first_missing_id = int(stable_ids[first_missing_index].as_py()) if first_missing_index >= 0 else None
        msg = (
            f"Lance stable-ID payload stream covered {unique_rows_covered} unique rows; expected {unique_rows}; "
            f"first missing stable row ID is {first_missing_id}"
        )
        raise RuntimeError(msg)
    if scatter_input_rows != logical_rows:
        msg = f"payload scatter row conservation failed: expected {logical_rows}, processed {scatter_input_rows}"
        raise RuntimeError(msg)

    reader_metrics = dict(payload_streamer.last_metrics)
    if reader_metrics.get("stream_complete") is not True:
        msg = "Lance stable-ID payload stream did not publish complete final metrics"
        raise RuntimeError(msg)

    def require_metric_int(name: str) -> int:
        value = reader_metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"Lance stable-ID payload metric {name!r} must be an integer"
            raise TypeError(msg)
        return value

    def require_metric_number(name: str) -> float:
        value = reader_metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = f"Lance stable-ID payload metric {name!r} must be numeric"
            raise TypeError(msg)
        return float(value)

    def require_metric_true(name: str) -> None:
        if reader_metrics.get(name) is not True:
            msg = f"Lance stable-ID payload metric {name!r} must be true"
            raise RuntimeError(msg)

    metric_input_rows = require_metric_int("input_stable_rows")
    metric_output_rows = require_metric_int("stream_output_rows")
    metric_take_rows = require_metric_int("payload_take_rows")
    metric_planned_batches = require_metric_int("payload_batches_planned")
    metric_batches = require_metric_int("payload_batches_emitted")
    take_calls = require_metric_int("payload_read_calls")
    metric_payload_bytes = require_metric_int("payload_bytes")
    expected_counts = (
        unique_rows,
        unique_rows,
        unique_rows,
        payload_batches,
        payload_batches,
        actual_payload_bytes,
    )
    actual_counts = (
        metric_input_rows,
        metric_output_rows,
        metric_take_rows,
        metric_planned_batches,
        metric_batches,
        metric_payload_bytes,
    )
    if actual_counts != expected_counts or take_calls != payload_batches:
        msg = f"Lance stable-ID payload metrics do not reconcile: actual={actual_counts}, expected={expected_counts}"
        raise RuntimeError(msg)

    require_metric_true("completion_order_output")
    require_metric_true("batch_stable_ids_sorted")
    require_metric_true("exact_operation_coverage")
    reordered_batches = require_metric_int("completion_order_reordered_batches")
    if reordered_batches < 0 or reordered_batches > payload_batches:
        msg = "Lance stable-ID payload completion-order metric is outside the emitted-batch range"
        raise RuntimeError(msg)

    peak_in_flight = require_metric_int("peak_in_flight_payload_reads")
    peak_running = require_metric_int("peak_running_payload_reads")
    peak_ready = require_metric_int("peak_ready_payload_batches")
    peak_producer_retained = require_metric_int("peak_producer_retained_payload_batches")
    peak_total_retained = require_metric_int("peak_total_retained_payload_batches")
    retained_upper_bound = require_metric_int("retained_payload_batch_upper_bound")
    consumer_held_limit = require_metric_int("consumer_held_payload_batch_limit")
    if consumer_held_limit != 1 or retained_upper_bound <= consumer_held_limit or retained_upper_bound % 2 != 1:
        msg = "Lance stable-ID payload retention-bound metrics are invalid"
        raise RuntimeError(msg)
    pending_limit = (retained_upper_bound - 1) // 2
    retention_counts = (
        peak_in_flight,
        peak_running,
        peak_ready,
        peak_producer_retained,
        peak_total_retained,
    )
    if any(value < 0 for value in retention_counts) or (
        peak_in_flight > pending_limit
        or peak_running > pending_limit
        or peak_ready > pending_limit
        or peak_producer_retained > 2 * pending_limit
        or peak_total_retained > retained_upper_bound
    ):
        msg = "Lance stable-ID payload retention metrics exceed the configured internal bound"
        raise RuntimeError(msg)
    if (
        require_metric_int("max_pending_payload_reads") != peak_in_flight
        or require_metric_int("max_retained_payload_batches") != peak_total_retained
    ):
        msg = "Lance stable-ID payload compatibility retention metrics do not reconcile"
        raise RuntimeError(msg)

    manifest = spool.finish()
    if manifest.total_rows != logical_rows:
        msg = f"payload spool row conservation failed: expected {logical_rows}, wrote {manifest.total_rows}"
        raise RuntimeError(msg)
    return {
        **reader_metrics,
        "logical_rows": logical_rows,
        "unique_rows": unique_rows,
        "null_rows_skipped": coordinate_plan[STABLE_ROW_ID].null_count,
        "duplicate_fanout": logical_rows / unique_rows if unique_rows else 0.0,
        "take_calls": take_calls,
        "take_rows": metric_take_rows,
        "scatter_input_rows": scatter_input_rows,
        "peak_pending": peak_in_flight,
        "peak_retained_batches": peak_total_retained,
        "sparse_calls_avoided": require_metric_int("sparse_calls_avoided"),
        "private_take_call_seconds_sum": require_metric_number("payload_read_call_sum_seconds"),
        "private_take_execution_envelope_seconds": require_metric_number("payload_read_envelope_seconds"),
        "actual_payload_bytes": actual_payload_bytes,
        "spooled_payload_bytes": spooled_payload_bytes,
        "spool_arrow_bytes": manifest.total_arrow_nbytes,
    }
