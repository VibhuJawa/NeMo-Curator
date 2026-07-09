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
from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool, PayloadSpoolCoordinator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


_MEMBER_INDEX = "__payload_member_index"
_GROUPED_COORDINATE_VALUE_BYTES_PER_ROW = 4 + 3 * 8
_GROUPED_SORT_INDEX_BYTES_PER_ROW = 8
_GROUPED_RUN_END_BYTES_PER_ROW = 2 * 8
_GROUPED_COVERAGE_BYTES_PER_ROW = 1
_GROUPED_DERIVED_VALIDITY_FIELD_COUNT = 8


@dataclass(frozen=True)
class GroupedPayloadPlanMetrics:
    """Artifact-local metrics from one member of a shared payload fetch."""

    logical_rows: int
    unique_rows: int
    null_rows_skipped: int
    scatter_input_rows: int
    spooled_payload_bytes: int
    payload_batches_contributed: int
    spool_arrow_bytes: int
    payload_spool_files: int
    payload_spool_oversized_rows: int
    payload_spool_peak_active_bytes: int

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "shared_fetch_group": True,
            "stream_complete": True,
            "completion_order_output": True,
            "batch_stable_ids_sorted": True,
            "exact_operation_coverage": True,
            "logical_rows": self.logical_rows,
            "unique_rows": self.unique_rows,
            "null_rows_skipped": self.null_rows_skipped,
            "scatter_input_rows": self.scatter_input_rows,
            "duplicate_fanout": self.logical_rows / self.unique_rows if self.unique_rows else 0.0,
            "spooled_payload_bytes": self.spooled_payload_bytes,
            "payload_batches_contributed": self.payload_batches_contributed,
            "spool_arrow_bytes": self.spool_arrow_bytes,
            "payload_spool_arrow_bytes": self.spool_arrow_bytes,
            "payload_spool_files": self.payload_spool_files,
            "payload_spool_oversized_rows": self.payload_spool_oversized_rows,
            "payload_spool_peak_active_bytes": self.payload_spool_peak_active_bytes,
        }


@dataclass(frozen=True)
class GroupedPayloadMaterializeResult:
    """One global reader result and positional artifact-local spool metrics."""

    fetch_metrics: dict[str, int | float | bool]
    plan_metrics: tuple[GroupedPayloadPlanMetrics, ...]


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


def estimate_grouped_coordinate_workspace_bytes(coordinate_plans: Sequence[pa.Table]) -> int:
    """Conservatively bound the fixed-width grouped sort workspace.

    The bound includes every resident input plan, filtered non-null copies,
    member IDs, uint64 sort indices, the sorted fixed-width copy, worst-case
    run-end arrays, coverage bytes, and validity-bit allowance. It bounds the
    retained Arrow queue; opaque temporary scratch inside Arrow kernels is
    reported separately through process RSS rather than mislabeled as exact.
    """

    if isinstance(coordinate_plans, (str, bytes)):
        msg = "coordinate_plans must be a sequence of pyarrow tables"
        raise TypeError(msg)
    total_rows = 0
    resident_plan_bytes = 0
    filtered_copy_bytes = 0
    for plan in coordinate_plans:
        if not isinstance(plan, pa.Table):
            msg = "coordinate_plans must contain only pyarrow tables"
            raise TypeError(msg)
        for name in (DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID):
            index = plan.schema.get_field_index(name)
            if index < 0:
                msg = f"coordinate plan is missing {name!r}"
                raise ValueError(msg)
            if plan.schema.field(index).type != pa.uint64():
                msg = f"coordinate plan field {name!r} must be uint64"
                raise TypeError(msg)
        rows = plan.num_rows - plan[STABLE_ROW_ID].null_count
        total_rows += rows
        resident_plan_bytes += plan.nbytes
        if plan[STABLE_ROW_ID].null_count:
            filtered_copy_bytes += 3 * 8 * rows + 3 * ((rows + 7) // 8)

    derived_bytes = total_rows * (
        4
        + _GROUPED_SORT_INDEX_BYTES_PER_ROW
        + _GROUPED_COORDINATE_VALUE_BYTES_PER_ROW
        + _GROUPED_RUN_END_BYTES_PER_ROW
        + _GROUPED_COVERAGE_BYTES_PER_ROW
    )
    validity_bytes = _GROUPED_DERIVED_VALIDITY_FIELD_COUNT * ((total_rows + 7) // 8)
    return resident_plan_bytes + filtered_copy_bytes + derived_bytes + validity_bytes


def _grouped_sorted_coordinate_runs(
    coordinate_plans: Sequence[pa.Table],
) -> tuple[pa.Table, pa.Array, pa.Array, tuple[int, ...], tuple[int, ...], int]:
    schema = pa.schema(
        [
            pa.field(_MEMBER_INDEX, pa.uint32(), nullable=False),
            pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
            pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
            pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
        ]
    )
    tables: list[pa.Table] = []
    logical_rows: list[int] = []
    unique_rows: list[int] = []
    resident_plan_bytes = sum(plan.nbytes for plan in coordinate_plans)
    allocated_grouped_input_bytes = 0
    for member_index, plan in enumerate(coordinate_plans):
        coordinates = plan.select([DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID])
        if coordinates[STABLE_ROW_ID].null_count:
            coordinates = coordinates.filter(pc.is_valid(coordinates[STABLE_ROW_ID]))
            allocated_grouped_input_bytes += coordinates.nbytes
        logical_rows.append(coordinates.num_rows)
        unique_rows.append(int(pc.count_distinct(coordinates[STABLE_ROW_ID], mode="only_valid").as_py()))
        member = pa.repeat(pa.scalar(member_index, type=pa.uint32()), coordinates.num_rows)
        allocated_grouped_input_bytes += member.nbytes
        table = pa.Table.from_arrays(
            [
                member,
                coordinates[DOCUMENT_ROWADDR].combine_chunks(),
                coordinates[DOCUMENT_POSITION].combine_chunks(),
                coordinates[STABLE_ROW_ID].combine_chunks(),
            ],
            schema=schema,
        )
        tables.append(table)

    grouped = pa.concat_tables(tables) if tables else pa.Table.from_batches([], schema=schema)
    sort_indices = pc.sort_indices(
        grouped,
        sort_keys=[
            (STABLE_ROW_ID, "ascending"),
            (_MEMBER_INDEX, "ascending"),
            (DOCUMENT_POSITION, "ascending"),
        ],
    )
    sorted_coordinates = grouped.take(sort_indices)
    encoded_ids = pc.run_end_encode(sorted_coordinates[STABLE_ROW_ID].combine_chunks())
    workspace_bytes = (
        resident_plan_bytes
        + allocated_grouped_input_bytes
        + sort_indices.nbytes
        + sorted_coordinates.nbytes
        + encoded_ids.values.nbytes
        + encoded_ids.run_ends.nbytes
        + len(encoded_ids.values)
    )
    return (
        sorted_coordinates,
        encoded_ids.values,
        encoded_ids.run_ends,
        tuple(logical_rows),
        tuple(unique_rows),
        workspace_bytes,
    )


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


def _scatter_group_payload_batch(
    coordinates: pa.Table,
    payload: pa.Table,
    source_columns: tuple[str, ...],
    output_schema: pa.Schema,
) -> pa.Table:
    coordinate_order = pc.sort_indices(
        coordinates,
        sort_keys=[(_MEMBER_INDEX, "ascending"), (DOCUMENT_POSITION, "ascending")],
    )
    sorted_coordinates = coordinates.take(coordinate_order)
    payload_indices = pc.index_in(sorted_coordinates[STABLE_ROW_ID], value_set=payload[STABLE_ROW_ID])
    if payload_indices.null_count:
        msg = "grouped payload scatter could not match every stable row ID"
        raise RuntimeError(msg)
    scattered_payload = payload.take(payload_indices)
    stable_ids_match = pc.all(pc.equal(sorted_coordinates[STABLE_ROW_ID], scattered_payload[STABLE_ROW_ID])).as_py()
    if stable_ids_match is not True:
        msg = "grouped payload scatter produced mismatched stable row IDs"
        raise RuntimeError(msg)
    grouped_schema = pa.schema([pa.field(_MEMBER_INDEX, pa.uint32(), nullable=False), *output_schema])
    arrays_by_name: dict[str, pa.ChunkedArray] = {
        _MEMBER_INDEX: sorted_coordinates[_MEMBER_INDEX],
        DOCUMENT_ROWADDR: sorted_coordinates[DOCUMENT_ROWADDR],
        DOCUMENT_POSITION: sorted_coordinates[DOCUMENT_POSITION],
        STABLE_ROW_ID: scattered_payload[STABLE_ROW_ID],
    }
    arrays_by_name.update({name: scattered_payload[name] for name in source_columns})
    return pa.Table.from_arrays(
        [arrays_by_name[field.name].combine_chunks() for field in grouped_schema],
        schema=grouped_schema,
    )


def _iter_member_tables(
    table: pa.Table, member_count: int, output_schema: pa.Schema
) -> Iterator[tuple[int, pa.Table]]:
    if table.num_rows == 0:
        return
    encoded = pc.run_end_encode(table[_MEMBER_INDEX].combine_chunks())
    start = 0
    for member_scalar, run_end_scalar in zip(encoded.values, encoded.run_ends, strict=True):
        member_index = int(member_scalar.as_py())
        if member_index < 0 or member_index >= member_count:
            msg = f"grouped payload scatter returned invalid member index {member_index}"
            raise RuntimeError(msg)
        stop = int(run_end_scalar.as_py())
        member_table = table.slice(start, stop - start).select(output_schema.names)
        member_table = member_table.replace_schema_metadata(output_schema.metadata)
        if not member_table.schema.equals(output_schema, check_metadata=True):
            msg = "grouped payload member schema changed during zero-copy slicing"
            raise RuntimeError(msg)
        yield member_index, member_table
        start = stop
    if start != table.num_rows:
        msg = "grouped payload member runs did not conserve scattered rows"
        raise RuntimeError(msg)


def _largest_scatter_row_bytes(payload: pa.Table, source_columns: tuple[str, ...]) -> int:
    if payload.num_rows == 0:
        return 1
    return max(
        4 * 8 + sum(payload[name].slice(row_index, 1).nbytes for name in source_columns)
        for row_index in range(payload.num_rows)
    )


def _scatter_rows_per_table(
    payload: pa.Table,
    source_columns: tuple[str, ...],
    target_bytes: int,
) -> int:
    """Bound pre-spool fan-out using the largest fetched payload row."""

    return max(1, target_bytes // _largest_scatter_row_bytes(payload, source_columns))


def _reconcile_reader_metrics(  # noqa: C901, PLR0915
    payload_streamer: _StableIdPayloadStreamer,
    *,
    unique_rows: int,
    payload_batches: int,
    actual_payload_bytes: int,
) -> dict[str, int | float | bool]:
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

    sparse_calls_avoided = require_metric_int("sparse_calls_avoided")
    if sparse_calls_avoided != max(unique_rows - take_calls, 0):
        msg = "Lance stable-ID payload sparse-call metrics do not reconcile"
        raise RuntimeError(msg)
    return {
        **reader_metrics,
        "take_calls": take_calls,
        "take_rows": metric_take_rows,
        "peak_pending": peak_in_flight,
        "peak_retained_batches": peak_total_retained,
        "sparse_calls_avoided": sparse_calls_avoided,
        "private_take_call_seconds_sum": require_metric_number("payload_read_call_sum_seconds"),
        "private_take_execution_envelope_seconds": require_metric_number("payload_read_envelope_seconds"),
        "actual_payload_bytes": actual_payload_bytes,
    }


def materialize_lance_payload_to_spool(  # noqa: PLR0915
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

    reader_metrics = _reconcile_reader_metrics(
        payload_streamer,
        unique_rows=unique_rows,
        payload_batches=payload_batches,
        actual_payload_bytes=actual_payload_bytes,
    )

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
        "scatter_input_rows": scatter_input_rows,
        "spooled_payload_bytes": spooled_payload_bytes,
        "spool_arrow_bytes": manifest.total_arrow_nbytes,
    }


def materialize_lance_payload_group_to_spools(  # noqa: C901, PLR0912, PLR0913, PLR0915
    payload_streamer: _StableIdPayloadStreamer,
    coordinate_plans: Sequence[pa.Table],
    image_source_columns: Sequence[str],
    spools: Sequence[PayloadSpool],
    *,
    shared_spool_budget_bytes: int,
    max_coordinate_workspace_bytes: int,
) -> GroupedPayloadMaterializeResult:
    """Fetch the global stable-ID union once and scatter into positional spools."""

    plans = tuple(coordinate_plans)
    output_spools = tuple(spools)
    if not plans:
        msg = "coordinate_plans must contain at least one plan"
        raise ValueError(msg)
    if len(plans) != len(output_spools):
        msg = "coordinate_plans and spools must have the same length"
        raise ValueError(msg)
    if isinstance(shared_spool_budget_bytes, bool) or not isinstance(shared_spool_budget_bytes, int):
        msg = "shared_spool_budget_bytes must be an integer"
        raise TypeError(msg)
    if shared_spool_budget_bytes <= 0:
        msg = "shared_spool_budget_bytes must be positive"
        raise ValueError(msg)
    if isinstance(max_coordinate_workspace_bytes, bool) or not isinstance(max_coordinate_workspace_bytes, int):
        msg = "max_coordinate_workspace_bytes must be an integer"
        raise TypeError(msg)
    if max_coordinate_workspace_bytes <= 0:
        msg = "max_coordinate_workspace_bytes must be positive"
        raise ValueError(msg)

    source_columns: tuple[str, ...] | None = None
    output_schema: pa.Schema | None = None
    for plan, spool in zip(plans, output_spools, strict=True):
        validated_columns = _validate_inputs(plan, image_source_columns, spool)
        if source_columns is None:
            source_columns = validated_columns
            output_schema = spool.schema
        elif validated_columns != source_columns or not spool.schema.equals(output_schema, check_metadata=True):
            msg = "grouped payload spools must use one payload schema and source-column mapping"
            raise ValueError(msg)
    if source_columns is None or output_schema is None:  # pragma: no cover - plans is nonempty
        msg = "grouped payload input validation did not produce a schema"
        raise RuntimeError(msg)
    if len({id(spool) for spool in output_spools}) != len(output_spools):
        msg = "grouped payload spools must be distinct objects"
        raise ValueError(msg)

    coordinate_workspace_estimated_bytes = estimate_grouped_coordinate_workspace_bytes(plans)
    if coordinate_workspace_estimated_bytes > max_coordinate_workspace_bytes:
        msg = (
            "grouped coordinate workspace estimate exceeds max_coordinate_workspace_bytes: "
            f"estimated={coordinate_workspace_estimated_bytes}, maximum={max_coordinate_workspace_bytes}"
        )
        raise MemoryError(msg)
    coordinator = PayloadSpoolCoordinator(shared_spool_budget_bytes)
    for spool in output_spools:
        coordinator.register(spool)

    (
        sorted_coordinates,
        stable_ids,
        run_ends,
        plan_logical_rows,
        plan_unique_rows,
        coordinate_workspace_bytes,
    ) = _grouped_sorted_coordinate_runs(plans)
    if coordinate_workspace_bytes > coordinate_workspace_estimated_bytes:
        msg = (
            "grouped coordinate workspace exceeded its conservative estimate: "
            f"actual={coordinate_workspace_bytes}, estimated={coordinate_workspace_estimated_bytes}"
        )
        raise RuntimeError(msg)
    expected_payload_schema = pa.schema(
        [
            pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
            *(output_schema.field(name) for name in source_columns),
        ]
    )
    iterator = payload_streamer.iter_stable_row_ids(stable_ids)
    covered_unique_rows = bytearray(len(stable_ids))
    unique_rows_covered = 0
    payload_batches = 0
    scatter_input_rows = [0] * len(plans)
    spooled_payload_bytes = [0] * len(plans)
    contributed_batches = [0] * len(plans)
    actual_payload_bytes = 0
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
            batch_coordinates = sorted_coordinates.slice(coordinate_start, coordinate_stop - coordinate_start)
            payload_batches += 1
            actual_payload_bytes += sum(fetched[name].nbytes for name in source_columns)
            largest_scatter_row_bytes = _largest_scatter_row_bytes(fetched, source_columns)
            batch_members: set[int] = set()
            offset = 0
            while offset < batch_coordinates.num_rows:
                available_scatter_bytes = shared_spool_budget_bytes - coordinator.active_bytes
                if available_scatter_bytes < largest_scatter_row_bytes and coordinator.active_bytes:
                    coordinator.flush()
                    available_scatter_bytes = shared_spool_budget_bytes
                rows_per_scatter = max(1, available_scatter_bytes // largest_scatter_row_bytes)
                coordinate_slice = batch_coordinates.slice(offset, rows_per_scatter)
                scattered = _scatter_group_payload_batch(
                    coordinate_slice,
                    fetched,
                    source_columns,
                    output_schema,
                )
                if scattered.nbytes > shared_spool_budget_bytes and scattered.num_rows > 1:
                    msg = "grouped payload scatter exceeded the shared byte budget"
                    raise RuntimeError(msg)
                if coordinator.active_bytes and coordinator.active_bytes + scattered.nbytes > (
                    shared_spool_budget_bytes
                ):
                    coordinator.flush()
                for member_index, member_table in _iter_member_tables(scattered, len(plans), output_schema):
                    batch_members.add(member_index)
                    scatter_input_rows[member_index] += member_table.num_rows
                    spooled_payload_bytes[member_index] += sum(member_table[name].nbytes for name in source_columns)
                    coordinator.append(output_spools[member_index], member_table)
                offset += coordinate_slice.num_rows
                del coordinate_slice, scattered
            for member_index in batch_members:
                contributed_batches[member_index] += 1
            covered_unique_rows[unique_row_start:unique_row_stop] = b"\x01" * fetched.num_rows
            unique_rows_covered += fetched.num_rows
            del batch_coordinates, fetched
    finally:
        iterator.close()

    global_unique_rows = len(stable_ids)
    global_logical_rows = sorted_coordinates.num_rows
    if unique_rows_covered != global_unique_rows:
        first_missing_index = covered_unique_rows.find(b"\x00")
        first_missing_id = int(stable_ids[first_missing_index].as_py()) if first_missing_index >= 0 else None
        msg = (
            f"Lance stable-ID payload stream covered {unique_rows_covered} unique rows; "
            f"expected {global_unique_rows}; first missing stable row ID is {first_missing_id}"
        )
        raise RuntimeError(msg)
    if tuple(scatter_input_rows) != plan_logical_rows:
        msg = (
            f"grouped payload scatter rows do not reconcile: actual={tuple(scatter_input_rows)}, "
            f"expected={plan_logical_rows}"
        )
        raise RuntimeError(msg)

    fetch_metrics = _reconcile_reader_metrics(
        payload_streamer,
        unique_rows=global_unique_rows,
        payload_batches=payload_batches,
        actual_payload_bytes=actual_payload_bytes,
    )
    coordinator.flush()
    manifests = tuple(spool.finish() for spool in output_spools)
    for member_index, (manifest, expected_rows) in enumerate(zip(manifests, plan_logical_rows, strict=True)):
        if manifest.total_rows != expected_rows:
            msg = (
                f"payload spool {member_index} row conservation failed: "
                f"expected {expected_rows}, wrote {manifest.total_rows}"
            )
            raise RuntimeError(msg)

    sum_plan_unique_rows = sum(plan_unique_rows)
    shared_spool_peak_active_bytes = max(
        coordinator.peak_active_bytes,
        *(manifest.peak_active_bytes for manifest in manifests),
    )
    shared_spool_peak_bounded_active_bytes = max(
        coordinator.peak_active_bytes,
        *(manifest.peak_bounded_active_bytes for manifest in manifests),
    )
    fetch_metrics.update(
        {
            "logical_rows": global_logical_rows,
            "unique_rows": global_unique_rows,
            "sum_plan_unique_rows": sum_plan_unique_rows,
            "cross_plan_unique_ids_coalesced": sum_plan_unique_rows - global_unique_rows,
            "duplicate_fanout": global_logical_rows / global_unique_rows if global_unique_rows else 0.0,
            "scatter_input_rows": sum(scatter_input_rows),
            "spooled_payload_bytes": sum(spooled_payload_bytes),
            "spool_arrow_bytes": sum(manifest.total_arrow_nbytes for manifest in manifests),
            "payload_spool_files": sum(len(manifest.files) for manifest in manifests),
            "payload_spool_oversized_rows": sum(len(manifest.oversized_rows) for manifest in manifests),
            "shared_spool_budget_bytes": shared_spool_budget_bytes,
            "shared_spool_peak_active_bytes": shared_spool_peak_active_bytes,
            "shared_spool_peak_bounded_active_bytes": shared_spool_peak_bounded_active_bytes,
            "coordinate_member_count": len(plans),
            "coordinate_queue_rows": global_logical_rows,
            "coordinate_workspace_bytes": coordinate_workspace_bytes,
            "coordinate_workspace_estimated_bytes": coordinate_workspace_estimated_bytes,
            "max_coordinate_workspace_bytes": max_coordinate_workspace_bytes,
        }
    )
    plan_metrics = tuple(
        GroupedPayloadPlanMetrics(
            logical_rows=plan_logical_rows[index],
            unique_rows=plan_unique_rows[index],
            null_rows_skipped=plans[index][STABLE_ROW_ID].null_count,
            scatter_input_rows=scatter_input_rows[index],
            spooled_payload_bytes=spooled_payload_bytes[index],
            payload_batches_contributed=contributed_batches[index],
            spool_arrow_bytes=manifests[index].total_arrow_nbytes,
            payload_spool_files=len(manifests[index].files),
            payload_spool_oversized_rows=len(manifests[index].oversized_rows),
            payload_spool_peak_active_bytes=manifests[index].peak_active_bytes,
        )
        for index in range(len(plans))
    )
    return GroupedPayloadMaterializeResult(fetch_metrics=fetch_metrics, plan_metrics=plan_metrics)
