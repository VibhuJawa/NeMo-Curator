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

"""Pure-Arrow reconstruction and byte-bounded document patch helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc

DOCUMENT_ROWADDR = "document_rowaddr"
DOCUMENT_POSITION = "document_position"
STABLE_ROW_ID = "stable_row_id"
LANCE_ROWADDR = "_rowaddr"
SAMPLE_ID = "sample_id"

ExistingColumnPolicy = Literal["error", "fill_null", "overwrite"]

_COORDINATE_COLUMNS = (DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID)


@dataclass(frozen=True)
class OversizedInterleavedSample:
    """One sample that cannot fit within the requested Arrow-byte target."""

    patch_index: int
    start_row: int
    row_count: int
    actual_bytes: int
    sample_id: pa.Scalar


@dataclass(frozen=True)
class InterleavedPatchSplit:
    """Contiguous Arrow patches and explicit oversized-sample reports."""

    patches: tuple[pa.Table, ...]
    oversized_samples: tuple[OversizedInterleavedSample, ...]


def _require_table(value: object, name: str) -> pa.Table:
    if not isinstance(value, pa.Table):
        msg = f"{name} must be a pyarrow.Table, got {type(value).__name__}"
        raise TypeError(msg)
    if len(set(value.column_names)) != len(value.column_names):
        msg = f"{name} must not contain duplicate column names"
        raise ValueError(msg)
    return value


def _require_uint64_column(table: pa.Table, name: str, table_name: str) -> pa.ChunkedArray:
    if name not in table.column_names:
        msg = f"{table_name} is missing required column {name!r}"
        raise ValueError(msg)
    column = table[name]
    if column.type != pa.uint64():
        msg = f"{table_name} column {name!r} must have uint64 type, got {column.type}"
        raise TypeError(msg)
    if column.null_count:
        msg = f"{table_name} column {name!r} must not contain nulls"
        raise ValueError(msg)
    return column


def _require_unique(values: pa.ChunkedArray, name: str) -> None:
    unique = int(pc.count_distinct(values, mode="only_valid").as_py())
    if unique != len(values):
        msg = f"{name} values must be unique"
        raise ValueError(msg)


def _validate_image_columns(
    payload_part: pa.Table,
    image_columns: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(image_columns, Mapping):
        msg = "image_columns must be a source-to-destination mapping"
        raise TypeError(msg)
    pairs = tuple(image_columns.items())
    if not pairs:
        msg = "image_columns must not be empty"
        raise ValueError(msg)
    destinations: set[str] = set()
    for source, destination in pairs:
        if not isinstance(source, str) or not source:
            msg = "image_columns source names must be non-empty strings"
            raise TypeError(msg)
        if not isinstance(destination, str) or not destination:
            msg = "image_columns destination names must be non-empty strings"
            raise TypeError(msg)
        if source not in payload_part.column_names:
            msg = f"payload_part is missing image source column {source!r}"
            raise ValueError(msg)
        if destination == LANCE_ROWADDR:
            msg = f"image_columns must not overwrite internal document column {LANCE_ROWADDR!r}"
            raise ValueError(msg)
        if destination in destinations:
            msg = f"image_columns maps more than one source to destination {destination!r}"
            raise ValueError(msg)
        destinations.add(destination)
    return pairs


def _validate_existing_policy(existing_column_policy: str) -> ExistingColumnPolicy:
    if existing_column_policy not in {"error", "fill_null", "overwrite"}:
        msg = f"Unsupported existing_column_policy: {existing_column_policy!r}"
        raise ValueError(msg)
    return existing_column_policy


def _set_payload_column(  # noqa: PLR0913
    table: pa.Table,
    *,
    destination: str,
    projected: pa.Array | pa.ChunkedArray,
    matched: pa.Array | pa.ChunkedArray,
    source_field: pa.Field,
    existing_column_policy: ExistingColumnPolicy,
) -> pa.Table:
    column_index = table.schema.get_field_index(destination)
    if column_index < 0:
        field = pa.field(destination, source_field.type, nullable=True, metadata=source_field.metadata)
        return table.append_column(field, projected)
    if existing_column_policy == "error":
        msg = f"Document already contains destination column {destination!r}"
        raise ValueError(msg)

    existing_field = table.schema.field(column_index)
    existing = table.column(column_index)
    if existing.type != projected.type:
        msg = (
            f"Document destination column {destination!r} has type {existing.type}; "
            f"payload source has type {projected.type}"
        )
        raise TypeError(msg)
    replace = pc.and_(matched, pc.is_null(existing)) if existing_column_policy == "fill_null" else matched
    values = pc.if_else(replace, projected, existing)
    if not existing_field.nullable and values.null_count:
        msg = f"Payload patch would insert nulls into non-nullable destination column {destination!r}"
        raise TypeError(msg)
    return table.set_column(column_index, existing_field, values)


def apply_payload_part(
    document: pa.Table,
    payload_part: pa.Table,
    image_columns: Mapping[str, str],
    existing_column_policy: ExistingColumnPolicy,
) -> pa.Table:
    """Scatter one payload-spool part into a document without changing row order.

    Duplicate ``stable_row_id`` values are valid fan-out.  Document row
    addresses and part coordinates must be unique, and every part address must
    belong to ``document``.  ``_rowaddr`` remains in the returned table so
    additional disjoint parts can be applied safely.
    """
    document = _require_table(document, "document")
    payload_part = _require_table(payload_part, "payload_part")
    policy = _validate_existing_policy(existing_column_policy)
    pairs = _validate_image_columns(payload_part, image_columns)

    document_addresses = _require_uint64_column(document, LANCE_ROWADDR, "document")
    _require_unique(document_addresses, f"document {LANCE_ROWADDR}")
    part_addresses = _require_uint64_column(payload_part, DOCUMENT_ROWADDR, "payload_part")
    part_positions = _require_uint64_column(payload_part, DOCUMENT_POSITION, "payload_part")
    _require_uint64_column(payload_part, STABLE_ROW_ID, "payload_part")
    _require_unique(part_addresses, f"payload_part {DOCUMENT_ROWADDR}")
    _require_unique(part_positions, f"payload_part {DOCUMENT_POSITION}")

    if payload_part.num_rows == 0:
        return document

    part_document_indices = pc.index_in(part_addresses, value_set=document_addresses)
    if part_document_indices.null_count:
        msg = "payload_part contains document_rowaddr values outside the document"
        raise ValueError(msg)

    payload_indices = pc.index_in(document_addresses, value_set=part_addresses)
    matched = pc.is_valid(payload_indices)
    result = document
    for source, destination in pairs:
        projected = pc.take(payload_part[source], payload_indices, boundscheck=False)
        result = _set_payload_column(
            result,
            destination=destination,
            projected=projected,
            matched=matched,
            source_field=payload_part.schema.field(source),
            existing_column_policy=policy,
        )
    return result


def _require_sample_ids(table: pa.Table) -> pa.Array:
    if SAMPLE_ID not in table.column_names:
        msg = f"table is missing required column {SAMPLE_ID!r}"
        raise ValueError(msg)
    sample_ids = table[SAMPLE_ID].combine_chunks()
    if not (pa.types.is_string(sample_ids.type) or pa.types.is_large_string(sample_ids.type)):
        msg = f"table column {SAMPLE_ID!r} must have string or large_string type, got {sample_ids.type}"
        raise TypeError(msg)
    if sample_ids.null_count:
        msg = f"table column {SAMPLE_ID!r} must not contain nulls"
        raise ValueError(msg)
    return sample_ids


def _sample_ranges(sample_ids: pa.Array) -> tuple[tuple[int, int], ...]:
    if len(sample_ids) == 0:
        return ()
    changes = pc.not_equal(sample_ids.slice(1), sample_ids.slice(0, len(sample_ids) - 1))
    change_indices = pc.indices_nonzero(changes)
    boundaries = [0]
    boundaries.extend(int(change_indices[index].as_py()) + 1 for index in range(len(change_indices)))
    boundaries.append(len(sample_ids))
    ranges = tuple(pairwise(boundaries))
    unique_samples = int(pc.count_distinct(sample_ids, mode="only_valid").as_py())
    if unique_samples != len(ranges):
        msg = "Each sample_id must occupy exactly one contiguous row range"
        raise ValueError(msg)
    return ranges


def split_interleaved_by_actual_bytes(table: pa.Table, target_bytes: int) -> InterleavedPatchSplit:
    """Greedily split contiguous samples using each candidate Arrow slice's bytes.

    Normal patches do not exceed ``target_bytes``.  A sample that exceeds the
    target is isolated in its own patch and reported instead of being split.
    """
    table = _require_table(table, "table")
    if isinstance(target_bytes, bool) or not isinstance(target_bytes, int) or target_bytes <= 0:
        msg = "target_bytes must be a positive integer"
        raise ValueError(msg)
    sample_ids = _require_sample_ids(table)
    sample_ranges = _sample_ranges(sample_ids)
    if not sample_ranges:
        return InterleavedPatchSplit(patches=(), oversized_samples=())

    patches: list[pa.Table] = []
    oversized_samples: list[OversizedInterleavedSample] = []
    pending_start: int | None = None
    pending_end = 0

    def flush_pending() -> None:
        nonlocal pending_start, pending_end
        if pending_start is not None:
            patches.append(table.slice(pending_start, pending_end - pending_start))
            pending_start = None

    for sample_start, sample_end in sample_ranges:
        sample = table.slice(sample_start, sample_end - sample_start)
        if sample.nbytes > target_bytes:
            flush_pending()
            patch_index = len(patches)
            patches.append(sample)
            oversized_samples.append(
                OversizedInterleavedSample(
                    patch_index=patch_index,
                    start_row=sample_start,
                    row_count=sample.num_rows,
                    actual_bytes=sample.nbytes,
                    sample_id=sample_ids[sample_start],
                )
            )
            continue

        if pending_start is None:
            pending_start = sample_start
            pending_end = sample_end
            continue
        candidate = table.slice(pending_start, sample_end - pending_start)
        if candidate.nbytes <= target_bytes:
            pending_end = sample_end
        else:
            flush_pending()
            pending_start = sample_start
            pending_end = sample_end

    flush_pending()
    return InterleavedPatchSplit(
        patches=tuple(patches),
        oversized_samples=tuple(oversized_samples),
    )
