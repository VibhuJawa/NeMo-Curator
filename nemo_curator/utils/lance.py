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

import os
import threading
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc

LANCE_ROWADDR_COLUMN = "__lance_rowaddr"
LANCE_ROWID_COLUMN = "__lance_rowid"
LANCE_FRAGID_COLUMN = "__lance_fragid"

# Opening a Lance dataset is not free: the reader prefetches per-page metadata for
# the columns it touches, and that cost is paid again on every fresh open. Reads are
# also usually scattered over many fragments, so a worker that re-opens per task
# re-pays it constantly. Holding one dataset handle and one Session per process --
# so the metadata cache survives between tasks -- was measured to cut requests per
# image from 3.520 to 1.065 on a 50 TB table.
_LANCE_SESSION: lance.Session | None = None
_LANCE_DATASETS: dict[tuple[Any, ...], lance.LanceDataset] = {}
_LANCE_LOCK = threading.Lock()

#: Metadata cache for the shared session. Must be large enough to hold the fragment
#: working set, or handles are retained while their metadata is evicted and the
#: saving disappears. Override with ``NEMO_CURATOR_LANCE_METADATA_CACHE_BYTES``.
DEFAULT_METADATA_CACHE_BYTES = 4 * 1024**3


def _cache_key(path: str, dataset_kwargs: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return a hashable identity for a *pinned* dataset, or ``None`` if uncacheable.

    Only pinned versions are cacheable. Without a version Lance resolves to whatever
    is latest, so a cached handle could silently serve a stale snapshot.
    """
    if dataset_kwargs.get("version") is None:
        return None
    try:
        return (path, *sorted((key, repr(value)) for key, value in dataset_kwargs.items()))
    except TypeError:  # pragma: no cover - unhashable kwargs are simply not cached
        return None


def open_lance_dataset(path: str, **dataset_kwargs: Any) -> lance.LanceDataset:  # noqa: ANN401 - passthrough to lance.dataset
    """Open *path*, reusing this process's handle when the version is pinned.

    Callers may open the same dataset for every task; this keeps that pattern
    correct while paying the open cost once per worker.
    """
    key = _cache_key(path, dataset_kwargs)
    if key is None:
        return lance.dataset(path, **dataset_kwargs)

    with _LANCE_LOCK:
        cached = _LANCE_DATASETS.get(key)
        if cached is not None:
            return cached

        global _LANCE_SESSION  # noqa: PLW0603 - one session per worker process
        if _LANCE_SESSION is None:
            cache_bytes = int(os.environ.get("NEMO_CURATOR_LANCE_METADATA_CACHE_BYTES", DEFAULT_METADATA_CACHE_BYTES))
            _LANCE_SESSION = lance.Session(metadata_cache_size_bytes=cache_bytes)

        dataset = lance.dataset(path, session=_LANCE_SESSION, **dataset_kwargs)
        _LANCE_DATASETS[key] = dataset
        return dataset


def clear_lance_dataset_cache() -> None:
    """Drop cached handles and the shared session. Intended for tests."""
    global _LANCE_SESSION  # noqa: PLW0603 - mirrors open_lance_dataset
    with _LANCE_LOCK:
        _LANCE_DATASETS.clear()
        _LANCE_SESSION = None


def add_lance_metadata_columns(table: pa.Table) -> pa.Table:
    missing = [name for name in ("_rowid", "_rowaddr") if name not in table.column_names]
    if missing:
        msg = f"Lance scanner did not return {missing}; include_lance_metadata requires row ids and addresses"
        raise ValueError(msg)

    renamed = {
        "_rowid": LANCE_ROWID_COLUMN,
        "_rowaddr": LANCE_ROWADDR_COLUMN,
    }
    table = table.rename_columns([renamed.get(name, name) for name in table.column_names])
    row_addresses = table[LANCE_ROWADDR_COLUMN].combine_chunks().cast(pa.uint64())
    fragment_ids = pc.shift_right(row_addresses, pa.scalar(32, type=pa.uint64())).cast(pa.uint64())
    return table.append_column(LANCE_FRAGID_COLUMN, fragment_ids)


def materialize_lance_blob_columns(dataset: lance.LanceDataset, table: pa.Table) -> pa.Table:
    """Replace scanned Blob v2 descriptors with binary payloads."""
    row_addresses = [int(value) for value in table["_rowaddr"].combine_chunks().to_pylist()]
    for field in dataset.schema:
        column_index = table.schema.get_field_index(field.name)
        if column_index < 0 or getattr(field.type, "extension_name", None) != "lance.blob.v2":
            continue
        # read_blobs may omit nulls, so align returned payloads to scanned rows by address.
        payloads_by_address = dict(dataset.read_blobs(field.name, addresses=row_addresses))
        payloads = pa.array(
            [payloads_by_address.get(row_address) for row_address in row_addresses], type=pa.large_binary()
        )
        output_field = pa.field(field.name, pa.large_binary(), nullable=field.nullable)
        table = table.set_column(column_index, output_field, payloads)
    return table


def encode_lance_blob_columns(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Rebuild Lance Blob v2 arrays from materialized binary columns."""
    for field in schema:
        column_index = table.schema.get_field_index(field.name)
        if column_index < 0 or getattr(field.type, "extension_name", None) != "lance.blob.v2":
            continue
        column = table.column(column_index).combine_chunks()
        if column.type != field.type:
            table = table.set_column(column_index, field, lance.blob_array(column.to_pylist()))
    return table
