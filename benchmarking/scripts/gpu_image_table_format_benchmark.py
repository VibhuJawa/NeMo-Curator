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

"""Single-GPU Lance versus cuDF-Parquet image workflow benchmark.

Preparation selects one representative source Lance fragment and materializes
the *entire* fragment as matched Lance and Parquet inputs. Preparation is never
included in measured trials. One persistent actor warms both formats, then each
measured arm reads the same deterministic fraction of physical row offsets,
decodes on CUDA, calculates variance-of-Laplacian on CUDA, filters on CUDA, and
writes the surviving rows in the source format.

The format reader, GPU heuristic, and writer are fused into one Curator stage.
This is intentional: a Ray stage boundary would serialize a GPU dataframe and
turn the benchmark into a Ray object-store benchmark. Sub-phase timings remain
available in the output metrics.
"""

# Benchmark-local integration code intentionally uses dynamic optional-library
# types and a single fused stage with explicit sub-phase instrumentation.
# ruff: noqa: ANN401, C901, EM101, EM102, PLR0912, PLR0913, PLR0915

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit, urlunsplit

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable


DEFAULT_SOURCE_URI = (
    "s3://mm-nemo-curator/lance_dbs/mint_1t_html_images/47f4e65f452f20ffca8b205a/stable_row_ids/dataset"
)
DEFAULT_SOURCE_VERSION = 4
FORMAT_VERSION = "gpu-image-table-format-v2"
IMAGE_COLUMN = "image"
URL_COLUMN = "url"
SIZE_COLUMN = "image_size_bytes"
FRAGMENT_COLUMN = "__benchmark_source_fragment_id"
OFFSET_COLUMN = "__benchmark_source_row_offset"
OUTPUT_COLUMNS = (FRAGMENT_COLUMN, OFFSET_COLUMN, URL_COLUMN, SIZE_COLUMN, IMAGE_COLUMN)
ARMS_PER_TRIAL = 2
DEFAULT_SAMPLE_FRACTIONS = (0.10, 0.20, 0.40, 0.80, 1.00)


@dataclass(frozen=True)
class StorageConfig:
    """Non-secret S3 routing options; credentials remain in the environment."""

    endpoint: str | None = None
    region: str | None = None
    dm_storage_location: str | None = None

    def apply_environment(self) -> None:
        """Load the selected Data Mover identity without exposing its secrets."""

        if not self.dm_storage_location:
            return
        import yaml

        locations_path = Path.home() / ".config" / "datamover" / "storage_locations"
        with locations_path.open(encoding="utf-8") as stream:
            locations = yaml.safe_load(stream)
        try:
            secret = locations[self.dm_storage_location]["secrets"]["local"]
        except KeyError as error:
            raise KeyError(
                f"Data Mover location {self.dm_storage_location!r} is not configured in {locations_path}"
            ) from error
        endpoint = self.endpoint or os.environ.get("AWS_ENDPOINT_URL_S3")
        region = self.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        updates = {
            "AWS_ACCESS_KEY_ID": str(secret["access_key_id"]),
            "AWS_SECRET_ACCESS_KEY": str(secret["secret_access_key"]),
            "AWS_EC2_METADATA_DISABLED": "true",
        }
        if endpoint:
            updates.update({"AWS_ENDPOINT": endpoint, "AWS_ENDPOINT_URL": endpoint, "AWS_ENDPOINT_URL_S3": endpoint})
        if region:
            updates.update({"AWS_DEFAULT_REGION": region, "AWS_REGION": region})
        os.environ.update(updates)
        os.environ.pop("AWS_SESSION_TOKEN", None)
        os.environ.pop("AWS_SECURITY_TOKEN", None)

    def lance_options(self) -> dict[str, str] | None:
        options: dict[str, str] = {}
        if self.endpoint:
            options["endpoint"] = self.endpoint
        if self.region:
            options["aws_region"] = self.region
        options.update(
            {
                "virtual_hosted_style_request": "false",
                "request_timeout": "300s",
                "connect_timeout": "30s",
                "client_max_retries": "20",
            }
        )
        return options or None

    def fsspec_options(self) -> dict[str, Any] | None:
        client_kwargs: dict[str, str] = {}
        if self.endpoint:
            client_kwargs["endpoint_url"] = self.endpoint
        if self.region:
            client_kwargs["region_name"] = self.region
        if not client_kwargs:
            return None
        return {
            "client_kwargs": client_kwargs,
            "config_kwargs": {
                "s3": {"addressing_style": "path"},
                "retries": {"max_attempts": 10, "mode": "adaptive"},
            },
        }


def uri_join(root: str, *parts: str) -> str:
    """Join local paths and object-store URIs without losing the URI scheme."""

    parsed = urlsplit(root)
    cleaned = [part.strip("/") for part in parts if part.strip("/")]
    if parsed.scheme:
        path = "/".join([parsed.path.rstrip("/"), *cleaned])
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))
    return str(Path(root).joinpath(*cleaned))


def _lance_options_for(uri: str, storage: StorageConfig) -> dict[str, str] | None:
    return storage.lance_options() if urlsplit(uri).scheme in {"s3", "s3a"} else None


def _fsspec_options_for(uri: str, storage: StorageConfig) -> dict[str, Any] | None:
    return storage.fsspec_options() if urlsplit(uri).scheme in {"s3", "s3a"} else None


def sample_row_offsets(row_count: int, fraction: float, seed: str, fragment_id: int) -> list[int]:
    """Return stable sorted offsets sampled without replacement."""

    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must be in (0, 1]")
    sample_count = min(row_count, max(1, math.ceil(row_count * fraction)))
    rng = random.Random(f"{seed}:{fragment_id}:{row_count}")  # noqa: S311 - reproducibility, not security
    return sorted(rng.sample(range(row_count), sample_count))


def parse_sample_fractions(value: str) -> tuple[float, ...]:
    """Parse a unique, increasing comma-separated fraction schedule."""

    try:
        fractions = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("sample fractions must be comma-separated numbers") from error
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("sample fractions must all be in (0, 1]")
    if len(set(fractions)) != len(fractions):
        raise ValueError("sample fractions must be unique")
    return tuple(sorted(fractions))


def fraction_label(fraction: float) -> str:
    """Return a stable path/metric label for a sample fraction."""

    return f"{round(fraction * 100):03d}pct"


def choose_representative_fragment(fragment_rows: Iterable[tuple[int, int]]) -> tuple[int, int]:
    """Choose the fragment nearest the median physical row count."""

    return choose_representative_fragments(fragment_rows, 1)[0]


def choose_representative_fragments(
    fragment_rows: Iterable[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    """Choose a stable cohort of fragments nearest the median row count."""

    rows = sorted((int(fragment_id), int(row_count)) for fragment_id, row_count in fragment_rows)
    if not rows:
        raise ValueError("Lance dataset has no fragments")
    if count <= 0:
        raise ValueError("fragment count must be positive")
    if count > len(rows):
        raise ValueError(f"Requested {count} fragments but the dataset has only {len(rows)}")
    if any(row_count <= 0 for _, row_count in rows):
        raise ValueError("Lance dataset contains an empty fragment")
    median_rows = statistics.median(row_count for _, row_count in rows)
    return sorted(rows, key=lambda item: (abs(item[1] - median_rows), item[0]))[:count]


def selected_parquet_row_groups(offsets: Iterable[int], row_count: int, row_group_size: int) -> tuple[list[int], int]:
    """Return touched row groups and their total logical row count."""

    if row_count <= 0 or row_group_size <= 0:
        raise ValueError("row_count and row_group_size must be positive")
    normalized = sorted({int(offset) for offset in offsets})
    if not normalized or normalized[0] < 0 or normalized[-1] >= row_count:
        raise ValueError("offsets must be non-empty and within the fragment")
    groups = sorted({offset // row_group_size for offset in normalized})
    touched_rows = sum(min(row_group_size, row_count - group * row_group_size) for group in groups)
    return groups, touched_rows


def trial_arm_order(
    trial: int,
) -> tuple[Literal["parquet"], Literal["lance"]] | tuple[Literal["lance"], Literal["parquet"]]:
    """Alternate order to prevent a fixed warm-cache advantage."""

    return ("parquet", "lance") if trial % 2 == 0 else ("lance", "parquet")


def _fragment_row_count(fragment: Any) -> int:
    metadata = getattr(fragment, "metadata", None)
    for owner in (metadata, fragment):
        for name in ("physical_rows", "num_rows"):
            value = getattr(owner, name, None)
            if value is not None:
                return int(value() if callable(value) else value)
    return int(fragment.count_rows())


def _uri_fs(uri: str, storage: StorageConfig) -> tuple[Any, str]:
    from fsspec.core import url_to_fs

    storage.apply_environment()
    return url_to_fs(uri, **(_fsspec_options_for(uri, storage) or {}))


def _uri_exists(uri: str, storage: StorageConfig) -> bool:
    fs, path = _uri_fs(uri, storage)
    return bool(fs.exists(path))


def _write_json(uri: str, payload: dict[str, Any], storage: StorageConfig) -> None:
    fs, path = _uri_fs(uri, storage)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _ensure_uri_parent(uri: str, storage: StorageConfig) -> None:
    fs, path = _uri_fs(uri, storage)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)


def _read_json(uri: str, storage: StorageConfig) -> dict[str, Any]:
    fs, path = _uri_fs(uri, storage)
    with fs.open(path) as stream:
        return dict(json.load(stream))


def _write_parquet(table: pa.Table, uri: str, row_group_size: int, storage: StorageConfig) -> None:
    fs, path = _uri_fs(uri, storage)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as stream:
        pq.write_table(
            table,
            stream,
            compression={column: "none" if column == IMAGE_COLUMN else "zstd" for column in table.column_names},
            use_dictionary=False,
            column_encoding={IMAGE_COLUMN: "PLAIN"},
            row_group_size=row_group_size,
        )


def _validate_parquet_image_encoding(uri: str, storage: StorageConfig) -> dict[str, list[str]]:
    """Require uncompressed, non-dictionary PLAIN image pages."""

    fs, path = _uri_fs(uri, storage)
    with fs.open(path, "rb") as stream:
        metadata = pq.ParquetFile(stream).metadata
        image_columns = [
            metadata.row_group(row_group).column(column)
            for row_group in range(metadata.num_row_groups)
            for column in range(metadata.num_columns)
            if metadata.row_group(row_group).column(column).path_in_schema == IMAGE_COLUMN
        ]
    if not image_columns:
        raise RuntimeError(f"Parquet output {uri} has no {IMAGE_COLUMN!r} column metadata")

    compressions = sorted({column.compression for column in image_columns})
    encodings = sorted({encoding for column in image_columns for encoding in column.encodings})
    if compressions != ["UNCOMPRESSED"]:
        raise RuntimeError(f"Parquet {IMAGE_COLUMN!r} compression is {compressions}; expected UNCOMPRESSED")
    if "PLAIN" not in encodings or any("DICTIONARY" in encoding for encoding in encodings):
        raise RuntimeError(f"Parquet {IMAGE_COLUMN!r} encodings are {encodings}; expected PLAIN without dictionary")
    return {"compressions": compressions, "encodings": encodings}


def _payload_digest(table: pa.Table) -> str:
    digest = hashlib.sha256()
    for offset, payload in zip(table[OFFSET_COLUMN].to_pylist(), table[IMAGE_COLUMN].to_pylist(), strict=True):
        digest.update(int(offset).to_bytes(8, byteorder="little", signed=False))
        digest.update(hashlib.sha256(payload or b"").digest())
    return digest.hexdigest()


def _normalize_fragment_table(table: pa.Table, fragment_id: int) -> pa.Table:
    """Create a format-neutral, guaranteed schema for both prepared inputs."""

    if IMAGE_COLUMN not in table.column_names:
        raise ValueError(f"Source table has no {IMAGE_COLUMN!r} column")
    payloads = table[IMAGE_COLUMN].combine_chunks().to_pylist()
    row_count = len(payloads)
    urls = table[URL_COLUMN].combine_chunks() if URL_COLUMN in table.column_names else pa.nulls(row_count, pa.string())
    if SIZE_COLUMN in table.column_names:
        sizes = table[SIZE_COLUMN].combine_chunks().cast(pa.int64())
    else:
        sizes = pa.array([len(payload) if payload is not None else 0 for payload in payloads], type=pa.int64())
    return pa.table(
        {
            FRAGMENT_COLUMN: pa.array([fragment_id] * row_count, type=pa.int64()),
            OFFSET_COLUMN: pa.array(range(row_count), type=pa.int64()),
            URL_COLUMN: urls.cast(pa.string()),
            SIZE_COLUMN: sizes,
            IMAGE_COLUMN: pa.array(payloads, type=pa.large_binary()),
        }
    )


def _scan_source_fragment(dataset: Any, fragment: Any) -> pa.Table:
    """Read and materialize a source fragment during the unmeasured setup."""

    available = set(dataset.schema.names)
    columns = [column for column in (IMAGE_COLUMN, URL_COLUMN, SIZE_COLUMN) if column in available]
    blob_columns = [
        field.name
        for field in dataset.schema
        if field.name in columns and getattr(field.type, "extension_name", None) == "lance.blob.v2"
    ]
    scanner_kwargs: dict[str, Any] = {"columns": columns}
    if blob_columns:
        scanner_kwargs["with_row_address"] = True
    table = fragment.scanner(**scanner_kwargs).to_table()
    if blob_columns:
        from nemo_curator.utils.lance import materialize_lance_blob_columns

        table = materialize_lance_blob_columns(dataset, table)
        table = table.drop_columns(["_rowaddr"])
    return table


def prepare_matched_inputs(
    *,
    source_uri: str,
    source_version: int,
    working_root: str,
    fragment_id: int | None,
    fragment_count: int,
    row_group_size: int,
    storage: StorageConfig,
    reuse_prepared: bool,
) -> dict[str, Any]:
    """Materialize one complete source fragment into matched format inputs."""

    import lance

    storage.apply_environment()
    marker_uri = uri_join(working_root, "prepared", "preparation.json")
    if _uri_exists(marker_uri, storage):
        marker = _read_json(marker_uri, storage)
        expected = {
            "format_version": FORMAT_VERSION,
            "source_uri": source_uri,
            "source_version": source_version,
            "requested_fragment_id": fragment_id,
            "row_group_size": row_group_size,
        }
        if not reuse_prepared:
            raise RuntimeError(f"Prepared inputs already exist at {marker_uri}; pass --reuse-prepared")
        if any(marker.get(key) != value for key, value in expected.items()) or marker.get("fragment_count", 1) != fragment_count:
            raise RuntimeError(f"Prepared input identity differs at {marker_uri}")
        return marker

    parquet_uri = uri_join(working_root, "prepared", "parquet", "fragment.parquet")
    lance_uri = uri_join(working_root, "prepared", "lance", "fragment.lance")
    if _uri_exists(parquet_uri, storage) or _uri_exists(lance_uri, storage):
        raise RuntimeError(f"Refusing incomplete prepared inputs below {uri_join(working_root, 'prepared')}")

    started = time.perf_counter()
    dataset = lance.dataset(
        source_uri, version=source_version, storage_options=_lance_options_for(source_uri, storage)
    )
    fragments = list(dataset.get_fragments())
    fragment_rows = [(int(item.fragment_id), _fragment_row_count(item)) for item in fragments]
    if fragment_count <= 0:
        raise ValueError("fragment_count must be positive")
    if fragment_id is not None and fragment_count != 1:
        raise ValueError("fragment_id and fragment_count cannot be combined for a multi-fragment cohort")
    if fragment_id is not None:
        selected = next((item for item in fragment_rows if item[0] == fragment_id), None)
        if selected is None:
            raise ValueError(f"Source Lance dataset has no fragment {fragment_id}")
        selected_rows_metadata = [selected]
    else:
        selected_rows_metadata = choose_representative_fragments(fragment_rows, fragment_count)

    tables: list[pa.Table] = []
    for selected_id, selected_rows in selected_rows_metadata:
        fragment = dataset.get_fragment(int(selected_id))
        if fragment is None:
            raise RuntimeError(f"Source Lance fragment {selected_id} disappeared")
        fragment_table = _normalize_fragment_table(_scan_source_fragment(dataset, fragment), int(selected_id))
        if fragment_table.num_rows != selected_rows:
            raise RuntimeError(f"Fragment row count changed: metadata={selected_rows}, scan={fragment_table.num_rows}")
        tables.append(fragment_table)

    table = pa.concat_tables(tables)
    table = table.set_column(
        table.schema.get_field_index(OFFSET_COLUMN),
        pa.field(OFFSET_COLUMN, pa.int64(), nullable=False),
        pa.array(range(table.num_rows), type=pa.int64()),
    )
    selected_ids = [int(item[0]) for item in selected_rows_metadata]
    selected_rows = sum(int(item[1]) for item in selected_rows_metadata)
    if table.num_rows != selected_rows:
        raise RuntimeError(f"Cohort row count changed: metadata={selected_rows}, scan={table.num_rows}")
    payload_digest = _payload_digest(table)
    _write_parquet(table, parquet_uri, row_group_size, storage)
    parquet_image_encoding = _validate_parquet_image_encoding(parquet_uri, storage)

    prepared_lance = lance.write_dataset(
        table,
        lance_uri,
        mode="create",
        schema=table.schema,
        max_rows_per_file=selected_rows,
        max_rows_per_group=row_group_size,
        data_storage_version="2.2",
        storage_options=_lance_options_for(lance_uri, storage),
    )
    prepared_fragments = list(prepared_lance.get_fragments())
    if len(prepared_fragments) != 1 or prepared_lance.count_rows() != selected_rows:
        raise RuntimeError("Prepared Lance input does not contain exactly one full fragment")

    marker = {
        "format_version": FORMAT_VERSION,
        "source_uri": source_uri,
        "source_version": source_version,
        "requested_fragment_id": fragment_id,
        "selected_source_fragment_id": selected_ids[0],
        "selected_source_fragment_ids": selected_ids,
        "fragment_count": fragment_count,
        "sample_identity": int(hashlib.sha256(",".join(map(str, selected_ids)).encode()).hexdigest()[:12], 16),
        "prepared_lance_fragment_id": int(prepared_fragments[0].fragment_id),
        "prepared_lance_version": int(prepared_lance.version),
        "row_count": selected_rows,
        "row_group_size": row_group_size,
        "parquet_uri": parquet_uri,
        "lance_uri": lance_uri,
        "payload_digest": payload_digest,
        "parquet_image_encoding": parquet_image_encoding,
        "preparation_seconds": time.perf_counter() - started,
    }
    _write_json(marker_uri, marker, storage)
    logger.info(
        "Prepared source fragment cohort {} ({} rows) as matched Lance and Parquet inputs in {:.2f}s",
        selected_ids,
        selected_rows,
        marker["preparation_seconds"],
    )
    return marker


def _materialize_taken_blobs(dataset: Any, fragment: Any, offsets: list[int], table: pa.Table) -> pa.Table:
    """Replace any Lance blob descriptors returned by a random take."""

    addresses = [(int(fragment.fragment_id) << 32) | offset for offset in offsets]
    for schema_field in dataset.schema:
        column_index = table.schema.get_field_index(schema_field.name)
        if column_index < 0 or getattr(schema_field.type, "extension_name", None) != "lance.blob.v2":
            continue
        payload_by_address = dict(dataset.read_blobs(schema_field.name, addresses=addresses))
        payloads = pa.array([payload_by_address.get(address) for address in addresses], type=pa.large_binary())
        table = table.set_column(
            column_index,
            pa.field(schema_field.name, pa.large_binary(), nullable=True),
            payloads,
        )
    # Some Lance versions already materialize blobs in take(); normalize the type.
    image_index = table.schema.get_field_index(IMAGE_COLUMN)
    if image_index >= 0 and isinstance(table.schema.field(image_index).type, pa.ExtensionType):
        table = table.set_column(
            image_index,
            pa.field(IMAGE_COLUMN, pa.large_binary()),
            pa.array(table[IMAGE_COLUMN].to_pylist(), type=pa.large_binary()),
        )
    return table


def _read_lance_selection(task_data: dict[str, Any], storage: StorageConfig) -> tuple[pa.Table, float]:
    """Read one prepared Lance selection for the actor's prefetch thread."""

    import lance

    started = time.perf_counter()
    input_uri = str(task_data["input_uri"])
    dataset = lance.dataset(
        input_uri,
        version=int(task_data["input_version"]),
        storage_options=_lance_options_for(input_uri, storage),
    )
    fragment = dataset.get_fragment(int(task_data["fragment_id"]))
    if fragment is None:
        raise RuntimeError(f"Prepared Lance fragment {task_data['fragment_id']} disappeared")
    offsets = [int(value) for value in task_data["offsets"]]
    table = fragment.take(offsets, columns=list(OUTPUT_COLUMNS))
    table = _materialize_taken_blobs(dataset, fragment, offsets, table)
    return table, time.perf_counter() - started


def _torch_to_cupy(tensor: Any) -> Any:
    import cupy as cp

    try:
        return cp.from_dlpack(tensor)
    except AttributeError:  # CuPy < 10 compatibility
        import torch

        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))


def _cudf_binary_to_host(series: Any) -> list[bytes]:
    """Copy a cuDF binary/string column to host without UTF-8 decoding.

    cuDF represents Parquet BYTE_ARRAY columns with its GPU string column. Its
    public ``to_arrow`` path assumes UTF-8, so arbitrary JPEG bytes must be
    reconstructed from the device-resident offsets and character buffers.
    """

    import cupy as cp

    if int(series.null_count):
        raise ValueError(f"{IMAGE_COLUMN!r} contains null image payloads")
    if len(series) == 0:
        return []

    column = series._column
    offsets_column = column.to_pylibcudf().child(0)
    raw_offsets = cp.asarray(offsets_column.data())
    offset_width = raw_offsets.nbytes // offsets_column.size()
    if offset_width == cp.dtype(cp.int32).itemsize:
        offset_dtype = cp.int32
    elif offset_width == cp.dtype(cp.int64).itemsize:
        offset_dtype = cp.int64
    else:
        raise RuntimeError(f"Unexpected cuDF string offset width: {offset_width}")

    row_start = int(column.offset)
    offsets = cp.asnumpy(raw_offsets.view(offset_dtype)[row_start : row_start + len(series) + 1])
    char_start = int(offsets[0])
    char_stop = int(offsets[-1])
    chars = cp.asnumpy(cp.asarray(column.data)[char_start:char_stop]).tobytes()
    return [chars[int(start) - char_start : int(stop) - char_start] for start, stop in pairwise(offsets)]


@dataclass
class _PendingPinnedBatch:
    """One in-flight encoded-byte copy in the double-buffered decoder queue."""

    start_row: int
    stop_row: int
    offsets: list[int]
    char_start: int
    char_stop: int
    host_buffer: Any
    copy_started: Any
    copy_finished: Any


class _PinnedDoubleBufferedBinaryQueue:
    """Issue encoded-byte D2H copies ahead of CUDA decode on two pinned buffers.

    The cuDF column and its offsets remain device-resident. Only the current
    decoder batch is exposed as CPU tensor views, because TorchVision 0.25
    requires encoded JPEG input on the host. The next batch's D2H transfer runs
    on a separate CUDA copy stream while the current batch is decoded/scored.
    """

    def __init__(self, series: Any, batch_rows: int, copy_stream: Any | None = None) -> None:
        import cupy as cp
        import torch

        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        if int(series.null_count):
            raise ValueError(f"{IMAGE_COLUMN!r} contains null image payloads")
        if len(series) == 0:
            raise ValueError("encoded image series must not be empty")

        column = series._column
        offsets_column = column.to_pylibcudf().child(0)
        raw_offsets = cp.asarray(offsets_column.data())
        offset_width = raw_offsets.nbytes // offsets_column.size()
        if offset_width == cp.dtype(cp.int32).itemsize:
            offset_dtype = cp.int32
        elif offset_width == cp.dtype(cp.int64).itemsize:
            offset_dtype = cp.int64
        else:
            raise RuntimeError(f"Unexpected cuDF string offset width: {offset_width}")

        self._chars = torch.from_dlpack(cp.asarray(column.data).view(cp.uint8))
        self._offsets = torch.from_dlpack(raw_offsets.view(offset_dtype))
        self._row_start = int(column.offset)
        self._rows = len(series)
        self._batch_rows = int(batch_rows)
        self._copy_stream = copy_stream or torch.cuda.Stream()
        self._buffers: list[Any | None] = [None, None]
        self._buffer_capacities = [0, 0]

    def _host_offsets(self, start_row: int, stop_row: int) -> tuple[list[int], float]:
        started = time.perf_counter()
        offsets = self._offsets[self._row_start + start_row : self._row_start + stop_row + 1].cpu().tolist()
        return [int(value) for value in offsets], time.perf_counter() - started

    def _submit(self, start_row: int, stop_row: int, slot: int) -> tuple[_PendingPinnedBatch, float, float]:
        import torch

        offsets, metadata_s = self._host_offsets(start_row, stop_row)
        char_start, char_stop = offsets[0], offsets[-1]
        byte_count = char_stop - char_start
        if byte_count <= 0:
            raise ValueError("encoded batch contains no bytes")
        if self._buffer_capacities[slot] < byte_count:
            allocation_started = time.perf_counter()
            self._buffers[slot] = torch.empty(byte_count, dtype=torch.uint8, pin_memory=True)
            self._buffer_capacities[slot] = byte_count
            allocation_s = time.perf_counter() - allocation_started
        else:
            allocation_s = 0.0

        copy_started = torch.cuda.Event(enable_timing=True)
        copy_finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._copy_stream):
            copy_started.record(self._copy_stream)
            self._buffers[slot][:byte_count].copy_(
                self._chars[char_start:char_stop],
                non_blocking=True,
            )
            copy_finished.record(self._copy_stream)
        return (
            _PendingPinnedBatch(
                start_row=start_row,
                stop_row=stop_row,
                offsets=offsets,
                char_start=char_start,
                char_stop=char_stop,
                host_buffer=self._buffers[slot],
                copy_started=copy_started,
                copy_finished=copy_finished,
            ),
            metadata_s,
            allocation_s,
        )

    def _consume(self, pending: _PendingPinnedBatch) -> tuple[list[Any], dict[str, float]]:
        started = time.perf_counter()
        pending.copy_finished.synchronize()
        wait_s = time.perf_counter() - started
        copy_gpu_s = float(pending.copy_started.elapsed_time(pending.copy_finished)) / 1000.0
        encoded = [
            pending.host_buffer.narrow(
                0,
                int(start) - pending.char_start,
                int(stop) - int(start),
            )
            for start, stop in pairwise(pending.offsets)
        ]
        return encoded, {
            "encoded_device_to_host_s": copy_gpu_s,
            "pinned_d2h_gpu_s": copy_gpu_s,
            "pinned_d2h_wait_s": wait_s,
            "pinned_d2h_bytes": float(pending.char_stop - pending.char_start),
            "pinned_d2h_batches": 1.0,
        }

    def __iter__(self):
        current_slot = 0
        pending, metadata_s, allocation_s = self._submit(0, min(self._rows, self._batch_rows), current_slot)
        while True:
            next_start = pending.stop_row
            next_pending: _PendingPinnedBatch | None = None
            next_metadata_s = 0.0
            next_allocation_s = 0.0
            next_slot = 1 - current_slot
            if next_start < self._rows:
                next_pending, next_metadata_s, next_allocation_s = self._submit(
                    next_start,
                    min(self._rows, next_start + self._batch_rows),
                    next_slot,
                )
            encoded, metrics = self._consume(pending)
            metrics["pinned_d2h_metadata_s"] = metadata_s
            metrics["pinned_d2h_allocation_s"] = allocation_s
            yield pending.start_row, encoded, metrics
            if next_pending is None:
                break
            pending = next_pending
            metadata_s = next_metadata_s
            allocation_s = next_allocation_s
            current_slot = next_slot


def _decode_cudf_and_score(
    series: Any,
    batch_size: int,
    resize: int,
    kernel: Any,
    copy_stream: Any | None = None,
) -> tuple[Any, list[int], dict[str, float]]:
    """Decode a device-resident cuDF column with queued pinned D2H batches."""

    import torch
    from torch.nn import functional
    from torchvision.io import ImageReadMode, decode_jpeg

    valid_positions: list[int] = []
    score_chunks: list[Any] = []
    metrics = {
        "encoded_tensor_setup_s": 0.0,
        # The queued path transfers encoded bytes through pinned host buffers
        # directly into the decoder; retain this legacy metric at zero so the
        # existing matrix summarizer remains compatible.
        "encoded_arrow_to_python_s": 0.0,
        "decode_wall_s": 0.0,
        "decode_gpu_s": 0.0,
        "heuristic_wall_s": 0.0,
        "heuristic_gpu_s": 0.0,
        "decode_failures": 0.0,
        "compressed_input_bytes": 0.0,
        "pinned_d2h_gpu_s": 0.0,
        "pinned_d2h_wait_s": 0.0,
        "pinned_d2h_metadata_s": 0.0,
        "pinned_d2h_allocation_s": 0.0,
        "pinned_d2h_batches": 0.0,
    }

    def decode_individually(encoded: list[Any], base: int) -> tuple[list[Any], list[int]]:
        decoded: list[Any] = []
        positions: list[int] = []
        for index, item in enumerate(encoded):
            try:
                decoded.append(decode_jpeg(item, mode=ImageReadMode.RGB, device="cuda"))
                positions.append(base + index)
            except RuntimeError:
                metrics["decode_failures"] += 1.0
        return decoded, positions

    for start, encoded, d2h_metrics in _PinnedDoubleBufferedBinaryQueue(series, batch_size, copy_stream):
        for name, value in d2h_metrics.items():
            metrics[name] = metrics.get(name, 0.0) + float(value)
        metrics["compressed_input_bytes"] += float(d2h_metrics["pinned_d2h_bytes"])
        setup_started = time.perf_counter()

        def batch_decode(encoded_batch: list[Any] = encoded, batch_start: int = start) -> tuple[list[Any], list[int]]:
            try:
                output = decode_jpeg(encoded_batch, mode=ImageReadMode.RGB, device="cuda")
                decoded = [output] if isinstance(output, torch.Tensor) else list(output)
                return decoded, list(range(batch_start, batch_start + len(decoded)))
            except RuntimeError:
                return decode_individually(encoded_batch, batch_start)

        metrics["encoded_tensor_setup_s"] += time.perf_counter() - setup_started
        (decoded, positions), wall_s, gpu_s = _cuda_elapsed(batch_decode)
        metrics["decode_wall_s"] += wall_s
        metrics["decode_gpu_s"] += gpu_s
        if not decoded:
            continue

        def calculate_scores(decoded_batch: list[Any] = decoded) -> Any:
            resized = torch.stack(
                [
                    functional.interpolate(
                        image.unsqueeze(0).float(),
                        size=(resize, resize),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    for image in decoded_batch
                ]
            ).div_(255.0)
            luma = resized[:, 0:1] * 0.299 + resized[:, 1:2] * 0.587 + resized[:, 2:3] * 0.114
            laplacian = functional.conv2d(luma, kernel, padding=1)
            return laplacian.var(dim=(1, 2, 3), correction=0)

        scores, wall_s, gpu_s = _cuda_elapsed(calculate_scores)
        metrics["heuristic_wall_s"] += wall_s
        metrics["heuristic_gpu_s"] += gpu_s
        valid_positions.extend(positions)
        score_chunks.append(scores)

    if not score_chunks:
        raise RuntimeError("No sampled images could be decoded on CUDA")
    return torch.cat(score_chunks), valid_positions, metrics


def _arrow_large_binary_as_large_string(table: pa.Table) -> pa.Table:
    """Reinterpret large_binary buffers so cuDF can copy them to GPU.

    cuDF 26.6 accepts Arrow large_string but rejects Arrow large_binary in
    ``DataFrame.from_arrow``. This is a zero-copy Arrow-side type view; nobody
    decodes the invalid UTF-8 JPEG payloads, and storage remains large_binary.
    """

    image_index = table.schema.get_field_index(IMAGE_COLUMN)
    field = table.schema.field(image_index)
    if field.type != pa.large_binary():
        raise TypeError(f"Expected Lance {IMAGE_COLUMN!r} to be large_binary; got {field.type}")
    chunked_binary = table[IMAGE_COLUMN]
    binary = chunked_binary.chunk(0) if chunked_binary.num_chunks == 1 else chunked_binary.combine_chunks()
    string_view = pa.Array.from_buffers(
        pa.large_string(),
        len(binary),
        binary.buffers(),
        null_count=binary.null_count,
    )
    return table.set_column(
        image_index,
        pa.field(IMAGE_COLUMN, pa.large_string(), nullable=field.nullable),
        string_view,
    )


def _cudf_to_arrow_large_binary(dataframe: Any, payloads: list[bytes] | None = None) -> pa.Table:
    """Convert a cuDF frame to Arrow while retaining JPEG as large_binary."""

    columns = list(dataframe.columns)
    if IMAGE_COLUMN in columns:
        image_index = columns.index(IMAGE_COLUMN)
        payloads = _cudf_binary_to_host(dataframe[IMAGE_COLUMN])
        table = dataframe.drop(columns=[IMAGE_COLUMN]).to_arrow()
    else:
        if payloads is None:
            raise ValueError("payloads are required when the cuDF frame omits the binary image column")
        image_index = min(OUTPUT_COLUMNS.index(IMAGE_COLUMN), len(columns))
        table = dataframe.to_arrow()
    return table.add_column(
        image_index,
        pa.field(IMAGE_COLUMN, pa.large_binary(), nullable=False),
        pa.array(payloads, type=pa.large_binary()),
    )


def _cuda_elapsed(operation: Any) -> tuple[Any, float, float]:
    """Run an operation and report synchronized wall and CUDA-event seconds."""

    import torch

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    result = operation()
    end_event.record()
    end_event.synchronize()
    return result, time.perf_counter() - wall_started, float(start_event.elapsed_time(end_event)) / 1000.0


def _decode_and_score(
    payloads: list[bytes], batch_size: int, resize: int, kernel: Any | None = None
) -> tuple[Any, list[int], dict[str, float]]:
    """Decode JPEG payloads and calculate variance-of-Laplacian on CUDA."""

    import torch
    from torch.nn import functional
    from torchvision.io import ImageReadMode, decode_jpeg

    if kernel is None:
        kernel = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]], device="cuda")
    valid_positions: list[int] = []
    score_chunks: list[Any] = []
    metrics = {
        "encoded_tensor_setup_s": 0.0,
        "decode_wall_s": 0.0,
        "decode_gpu_s": 0.0,
        "heuristic_wall_s": 0.0,
        "heuristic_gpu_s": 0.0,
        "decode_failures": 0.0,
    }

    def decode_individually(encoded: list[Any], base: int) -> tuple[list[Any], list[int]]:
        decoded: list[Any] = []
        positions: list[int] = []
        for index, item in enumerate(encoded):
            try:
                decoded.append(decode_jpeg(item, mode=ImageReadMode.RGB, device="cuda"))
                positions.append(base + index)
            except RuntimeError:
                metrics["decode_failures"] += 1.0
        return decoded, positions

    for start in range(0, len(payloads), batch_size):
        host_batch = payloads[start : start + batch_size]
        setup_started = time.perf_counter()
        encoded = [torch.frombuffer(bytearray(payload), dtype=torch.uint8) for payload in host_batch]
        metrics["encoded_tensor_setup_s"] += time.perf_counter() - setup_started

        def batch_decode(encoded_batch: list[Any] = encoded, batch_start: int = start) -> tuple[list[Any], list[int]]:
            try:
                output = decode_jpeg(encoded_batch, mode=ImageReadMode.RGB, device="cuda")
                decoded = [output] if isinstance(output, torch.Tensor) else list(output)
                return decoded, list(range(batch_start, batch_start + len(decoded)))
            except RuntimeError:
                return decode_individually(encoded_batch, batch_start)

        (decoded, positions), wall_s, gpu_s = _cuda_elapsed(batch_decode)
        metrics["decode_wall_s"] += wall_s
        metrics["decode_gpu_s"] += gpu_s
        if not decoded:
            continue

        def calculate_scores(decoded_batch: list[Any] = decoded) -> Any:
            resized = torch.stack(
                [
                    functional.interpolate(
                        image.unsqueeze(0).float(),
                        size=(resize, resize),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    for image in decoded_batch
                ]
            ).div_(255.0)
            luma = resized[:, 0:1] * 0.299 + resized[:, 1:2] * 0.587 + resized[:, 2:3] * 0.114
            laplacian = functional.conv2d(luma, kernel, padding=1)
            return laplacian.var(dim=(1, 2, 3), correction=0)

        scores, wall_s, gpu_s = _cuda_elapsed(calculate_scores)
        metrics["heuristic_wall_s"] += wall_s
        metrics["heuristic_gpu_s"] += gpu_s
        valid_positions.extend(positions)
        score_chunks.append(scores)

    if not score_chunks:
        raise RuntimeError("No sampled images could be decoded on CUDA")
    return torch.cat(score_chunks), valid_positions, metrics


def _make_curator_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Define benchmark-local Curator types lazily for metadata-only tests."""

    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.tasks import DocumentBatch, Task

    @dataclass
    class ImageFormatTask(Task[dict[str, Any]]):
        data: dict[str, Any] = field(default_factory=dict)

        @property
        def num_items(self) -> int:
            return int(self.data.get("sample_rows", 0))

        def validate(self) -> bool:
            return self.data.get("arm") in {"lance", "parquet"} and self.num_items > 0

        def get_deterministic_id(self) -> str:
            identity = {key: self.data[key] for key in sorted(self.data) if key != "output_uri"}
            return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @dataclass
    class ImageFormatResultTask(Task[dict[str, Any]]):
        data: dict[str, Any] = field(default_factory=dict)

        @property
        def num_items(self) -> int:
            return int(self.data.get("sample_rows", 0))

        def validate(self) -> bool:
            return bool(self.data.get("is_success")) and self.num_items > 0

    @dataclass
    class GpuImageFormatWorkflowStage(ProcessingStage[ImageFormatTask, ImageFormatResultTask]):
        storage: StorageConfig
        batch_rows: int = 256
        resize: int = 64
        blur_threshold: float = 0.10
        lance_prefetch_plan: tuple[dict[str, Any], ...] = ()
        name: str = "gpu_image_format_workflow"
        resources: Resources = field(default_factory=lambda: Resources(cpus=4, gpus=1))
        batch_size: int = 1
        _blur_kernel: Any = field(default=None, init=False, repr=False, compare=False)
        _d2h_copy_stream: Any = field(default=None, init=False, repr=False, compare=False)
        _lance_prefetch_executor: Any = field(default=None, init=False, repr=False, compare=False)
        _lance_prefetch_futures: dict[str, Future[tuple[pa.Table, float]]] = field(
            default_factory=dict,
            init=False,
            repr=False,
            compare=False,
        )
        _lance_prefetch_index: int = field(default=0, init=False, repr=False, compare=False)
        _runtime_setup_s: float = field(default=0.0, init=False, repr=False, compare=False)
        _processed_tasks: int = field(default=0, init=False, repr=False, compare=False)

        def setup(self, _worker_metadata: Any = None) -> None:
            """Initialize CUDA libraries once, outside every task timing."""

            started = time.perf_counter()
            import cudf
            import cupy as cp
            import torch
            import torchvision  # noqa: F401 - force extension initialization

            if not torch.cuda.is_available():
                raise RuntimeError("The image-format workflow was scheduled without a CUDA device")
            self.storage.apply_environment()
            torch.cuda.init()
            torch.zeros(1, device="cuda").sum()
            cp.zeros(1, dtype=cp.uint8).sum()
            cudf.Series(cp.zeros(1, dtype=cp.int8)).sum()
            self._blur_kernel = torch.tensor(
                [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
                device="cuda",
            )
            self._d2h_copy_stream = torch.cuda.Stream()
            self._lance_prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lance-prefetch")
            torch.cuda.synchronize()
            self._runtime_setup_s = time.perf_counter() - started

        @staticmethod
        def _prefetch_key(task_data: dict[str, Any]) -> str:
            return str(task_data["prefetch_key"])

        def _submit_lance_prefetch(self, task_data: dict[str, Any]) -> None:
            key = self._prefetch_key(task_data)
            if key in self._lance_prefetch_futures:
                return
            if self._lance_prefetch_executor is None:
                raise RuntimeError("Lance prefetch executor was not initialized")
            self._lance_prefetch_futures[key] = self._lance_prefetch_executor.submit(
                _read_lance_selection,
                task_data,
                self.storage,
            )

        def _take_lance_selection(self, task_data: dict[str, Any]) -> tuple[pa.Table, float, float]:
            """Resolve the current Lance read and queue the next one immediately."""

            key = self._prefetch_key(task_data)
            expected_key = (
                self._prefetch_key(self.lance_prefetch_plan[self._lance_prefetch_index])
                if self._lance_prefetch_index < len(self.lance_prefetch_plan)
                else None
            )
            if expected_key != key:
                raise RuntimeError(
                    "Lance task order changed; async prefetch would be invalid: "
                    f"expected {expected_key!r}, received {key!r}"
                )
            submit_started = time.perf_counter()
            self._submit_lance_prefetch(task_data)
            future = self._lance_prefetch_futures.pop(key)
            table, source_read_s = future.result()
            wait_s = time.perf_counter() - submit_started
            self._lance_prefetch_index += 1
            if self._lance_prefetch_index < len(self.lance_prefetch_plan):
                self._submit_lance_prefetch(self.lance_prefetch_plan[self._lance_prefetch_index])
            return table, source_read_s, wait_s

        def teardown(self) -> None:
            if self._lance_prefetch_executor is not None:
                self._lance_prefetch_executor.shutdown(wait=True, cancel_futures=True)
                self._lance_prefetch_executor = None

        def process(self, task: ImageFormatTask) -> ImageFormatResultTask:
            import cudf
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("The image-format workflow was scheduled without a CUDA device")
            self.storage.apply_environment()
            arm = str(task.data["arm"])
            offsets = [int(value) for value in task.data["offsets"]]
            expected_columns = list(OUTPUT_COLUMNS)
            metrics: dict[str, float] = {
                "sample_rows": float(len(offsets)),
                "source_rows": float(task.data["source_rows"]),
                "compressed_input_bytes": 0.0,
                "runtime_setup_s": self._runtime_setup_s,
                "actor_task_index": float(self._processed_tasks),
            }
            self._processed_tasks += 1
            e2e_started = time.perf_counter()

            if arm == "parquet":
                read_started = time.perf_counter()
                row_groups = [int(value) for value in task.data["row_groups"]]
                dataframe = cudf.read_parquet(
                    [str(task.data["input_uri"])],
                    columns=expected_columns,
                    row_groups=[row_groups],
                    storage_options=_fsspec_options_for(str(task.data["input_uri"]), self.storage),
                )
                metrics["source_read_s"] = time.perf_counter() - read_started
                metrics["source_rows_read"] = float(task.data["parquet_touched_rows"])
                selection_started = time.perf_counter()
                dataframe = dataframe[dataframe[OFFSET_COLUMN].isin(offsets)].sort_values(OFFSET_COLUMN)
                dataframe = dataframe.reset_index(drop=True)
                metrics["row_select_s"] = time.perf_counter() - selection_started
                metrics["arrow_to_device_s"] = 0.0
                metrics["prefetch_wait_s"] = 0.0
                metrics["prefetch_source_read_s"] = 0.0
            else:
                table, prefetch_source_read_s, prefetch_wait_s = self._take_lance_selection(task.data)
                metrics["source_read_s"] = prefetch_source_read_s
                metrics["prefetch_source_read_s"] = prefetch_source_read_s
                metrics["prefetch_wait_s"] = prefetch_wait_s
                metrics["source_rows_read"] = float(len(offsets))
                transfer_started = time.perf_counter()
                dataframe = cudf.DataFrame.from_arrow(_arrow_large_binary_as_large_string(table))
                metrics["arrow_to_device_s"] = time.perf_counter() - transfer_started
                metrics["row_select_s"] = 0.0

            if len(dataframe) != len(offsets):
                raise RuntimeError(f"{arm} selected {len(dataframe)} rows; expected {len(offsets)}")
            scores, valid_positions, gpu_metrics = _decode_cudf_and_score(
                dataframe[IMAGE_COLUMN],
                self.batch_rows,
                self.resize,
                self._blur_kernel,
                self._d2h_copy_stream,
            )
            metrics.update(gpu_metrics)

            gather_started = time.perf_counter()
            if len(valid_positions) != len(dataframe):
                dataframe = dataframe.iloc[valid_positions].reset_index(drop=True)
            dataframe["blur_score"] = cudf.Series(_torch_to_cupy(scores))
            keep_mask = scores >= self.blur_threshold
            kept = dataframe[cudf.Series(_torch_to_cupy(keep_mask))].reset_index(drop=True)
            torch.cuda.synchronize()
            metrics["filter_gather_s"] = time.perf_counter() - gather_started
            metrics["decoded_rows"] = float(len(valid_positions))
            metrics["output_rows"] = float(len(kept))
            metrics["filtered_rows"] = float(len(valid_positions) - len(kept))

            output_uri = str(task.data["output_uri"])
            _ensure_uri_parent(output_uri, self.storage)
            if arm == "parquet":
                write_started = time.perf_counter()
                kept.to_parquet(
                    output_uri,
                    compression="zstd",
                    index=False,
                    use_dictionary=False,
                    skip_compression={IMAGE_COLUMN},
                    column_encoding={IMAGE_COLUMN: "PLAIN"},
                    output_as_binary={IMAGE_COLUMN},
                    # Bound cuDF's device-side Parquet writer staging buffers
                    # for the larger cohort; this does not move image payloads
                    # to CPU and keeps the encoded column uncompressed/plain.
                    row_group_size_rows=self.batch_rows,
                    row_group_size_bytes=64 * 1024 * 1024,
                    max_page_size_bytes=16 * 1024 * 1024,
                    storage_options=_fsspec_options_for(output_uri, self.storage),
                )
                metrics["writer_data_s"] = time.perf_counter() - write_started
                metrics["writer_prepare_s"] = 0.0
                metrics["writer_commit_s"] = 0.0
                output_version = 0
            else:
                prepare_started = time.perf_counter()
                output_table = _cudf_to_arrow_large_binary(kept)
                output_schema = output_table.schema
                batch = DocumentBatch(dataset_name=task.dataset_name, data=output_table)
                batch._set_task_id("", task.task_id or "image-format-benchmark")
                commit_uri = str(task.data["commit_uri"])
                metrics["writer_prepare_s"] = time.perf_counter() - prepare_started

                from nemo_curator.stages.text.io.writer import LanceWriter, commit_lance_checkpoint

                write_kwargs: dict[str, Any] = {
                    "data_storage_version": "2.2",
                    "max_rows_per_file": max(1, len(output_table)),
                    "max_rows_per_group": max(1, min(len(output_table), self.batch_rows)),
                }
                dataset_storage_options = _lance_options_for(output_uri, self.storage)
                checkpoint_storage_options = _fsspec_options_for(commit_uri, self.storage)
                if dataset_storage_options:
                    write_kwargs["storage_options"] = dataset_storage_options
                if checkpoint_storage_options:
                    write_kwargs["checkpoint_storage_options"] = checkpoint_storage_options
                write_started = time.perf_counter()
                LanceWriter(
                    path=output_uri,
                    commit_path=commit_uri,
                    schema=output_schema,
                    write_kwargs=write_kwargs,
                    mode="create",
                ).process(batch)
                metrics["writer_data_s"] = time.perf_counter() - write_started
                commit_started = time.perf_counter()
                output_version = commit_lance_checkpoint(
                    output_uri,
                    commit_uri,
                    dataset_storage_options=dataset_storage_options,
                    checkpoint_storage_options=checkpoint_storage_options,
                )
                metrics["writer_commit_s"] = time.perf_counter() - commit_started

            summary_started = time.perf_counter()
            kept_offsets = [int(value) for value in kept[OFFSET_COLUMN].to_arrow().to_pylist()]
            score_values = [float(value) for value in kept["blur_score"].to_arrow().to_pylist()]
            digest = hashlib.sha256(",".join(map(str, kept_offsets)).encode()).hexdigest()
            metrics["result_summary_to_host_s"] = time.perf_counter() - summary_started
            metrics["end_to_end_s"] = time.perf_counter() - e2e_started
            metrics["read_amplification_rows"] = metrics["source_rows_read"] / metrics["sample_rows"]
            self._log_metrics(metrics)
            result = {
                "is_success": True,
                "arm": arm,
                "trial": int(task.data["trial"]),
                "warmup": bool(task.data["warmup"]),
                "sample_fraction": float(task.data["sample_fraction"]),
                "sample_rows": len(offsets),
                "output_rows": len(kept),
                "decode_failures": int(metrics["decode_failures"]),
                "kept_offset_digest": digest,
                "score_sum": sum(score_values),
                "output_uri": output_uri,
                "output_version": int(output_version),
                **metrics,
            }
            return ImageFormatResultTask(
                dataset_name=task.dataset_name,
                data=result,
                _metadata={"arm": arm, "trial": task.data["trial"]},
                _stage_perf=task._stage_perf,
            )

    return ImageFormatTask, ImageFormatResultTask, GpuImageFormatWorkflowStage


def _runtime_versions() -> dict[str, str]:
    packages = ("nemo-curator", "ray", "pylance", "lance-ray", "pyarrow", "cudf-cu12", "torch", "torchvision")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _summarize_arm(trials: list[dict[str, Any]], arm: str) -> dict[str, float]:
    arm_trials = [trial for trial in trials if trial["arm"] == arm]
    summary: dict[str, float] = {}
    metric_names = (
        "end_to_end_s",
        "source_read_s",
        "prefetch_source_read_s",
        "prefetch_wait_s",
        "row_select_s",
        "arrow_to_device_s",
        "encoded_tensor_setup_s",
        "encoded_device_to_host_s",
        "encoded_arrow_to_python_s",
        "pinned_d2h_gpu_s",
        "pinned_d2h_wait_s",
        "pinned_d2h_metadata_s",
        "pinned_d2h_allocation_s",
        "pinned_d2h_batches",
        "decode_wall_s",
        "decode_gpu_s",
        "heuristic_wall_s",
        "heuristic_gpu_s",
        "filter_gather_s",
        "writer_prepare_s",
        "writer_data_s",
        "writer_commit_s",
        "read_amplification_rows",
    )
    for metric in metric_names:
        values = [float(trial[metric]) for trial in arm_trials]
        summary[f"{arm}_{metric}_mean"] = statistics.fmean(values)
        summary[f"{arm}_{metric}_min"] = min(values)
        summary[f"{arm}_{metric}_max"] = max(values)
        summary[f"{arm}_{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    rows = statistics.fmean(float(trial["sample_rows"]) for trial in arm_trials)
    summary[f"{arm}_images_per_second_mean"] = rows / summary[f"{arm}_end_to_end_s_mean"]
    summary[f"{arm}_output_rows_mean"] = statistics.fmean(float(trial["output_rows"]) for trial in arm_trials)
    summary[f"{arm}_decode_failures_mean"] = statistics.fmean(float(trial["decode_failures"]) for trial in arm_trials)
    return summary


def _make_workflow_task(
    *,
    task_type: type[Any],
    arm: Literal["lance", "parquet"],
    trial: int,
    warmup: bool,
    sample_fraction: float,
    preparation: dict[str, Any],
    offsets: list[int],
    row_groups: list[int],
    touched_rows: int,
    output_root: str,
) -> Any:
    run_name = "warmup" if warmup else f"trial-{trial}"
    fraction_root = uri_join(output_root, f"fraction-{fraction_label(sample_fraction)}")
    output_uri = uri_join(
        fraction_root,
        run_name,
        arm,
        "output.parquet" if arm == "parquet" else "output.lance",
    )
    commit_uri = uri_join(fraction_root, run_name, arm, "lance-commit")
    task_data = {
        "arm": arm,
        "trial": trial,
        "warmup": warmup,
        "sample_fraction": sample_fraction,
        "prefetch_key": f"{arm}:{trial}:{fraction_label(sample_fraction)}:{int(warmup)}",
        "sample_rows": len(offsets),
        "source_rows": int(preparation["row_count"]),
        "offsets": offsets,
        "row_groups": row_groups,
        "parquet_touched_rows": touched_rows,
        "input_uri": preparation[f"{arm}_uri"],
        "input_version": int(preparation.get("prepared_lance_version", 0)),
        "fragment_id": int(preparation.get("prepared_lance_fragment_id", 0)),
        "output_uri": output_uri,
        "commit_uri": commit_uri,
    }
    return task_type(dataset_name="mint_1t_html_images_fragment", data=task_data)


def _run_schedule(
    *,
    preparation: dict[str, Any],
    selections: dict[float, tuple[list[int], list[int], int]],
    output_root: str,
    storage: StorageConfig,
    batch_rows: int,
    resize: int,
    blur_threshold: float,
    trials: int,
) -> tuple[list[dict[str, Any]], float]:
    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.pipeline import Pipeline

    image_format_task_type, _, workflow_stage_type = _make_curator_types()
    warmup_fraction = min(selections)
    warmup_offsets, warmup_row_groups, warmup_touched_rows = selections[warmup_fraction]
    warmup_common = {
        "task_type": image_format_task_type,
        "preparation": preparation,
        "sample_fraction": warmup_fraction,
        "offsets": warmup_offsets,
        "row_groups": warmup_row_groups,
        "touched_rows": warmup_touched_rows,
        "output_root": output_root,
    }
    initial_tasks = [
        _make_workflow_task(arm="parquet", trial=-1, warmup=True, **warmup_common),
        _make_workflow_task(arm="lance", trial=-1, warmup=True, **warmup_common),
    ]
    for sample_fraction, (offsets, row_groups, touched_rows) in selections.items():
        common = {
            "task_type": image_format_task_type,
            "preparation": preparation,
            "sample_fraction": sample_fraction,
            "offsets": offsets,
            "row_groups": row_groups,
            "touched_rows": touched_rows,
            "output_root": output_root,
        }
        for trial in range(trials):
            initial_tasks.extend(
                _make_workflow_task(arm=arm, trial=trial, warmup=False, **common)
                for arm in trial_arm_order(trial)
            )

    lance_prefetch_plan = tuple(
        task.data for task in initial_tasks if task.data["arm"] == "lance"
    )

    stage = workflow_stage_type(
        storage=storage,
        batch_rows=batch_rows,
        resize=resize,
        blur_threshold=blur_threshold,
        lance_prefetch_plan=lance_prefetch_plan,
        name="persistent_gpu_image_workflow",
    ).with_(num_workers=1, batch_size=1)
    pipeline = Pipeline(
        name="gpu_image_table_persistent_actor_schedule",
        description="Format read -> CUDA JPEG decode -> CUDA blur filter -> same-format output",
        stages=[stage],
    )
    logger.info(
        "Running one persistent GPU actor for {} warmup and {} measured tasks across {} sample fractions",
        ARMS_PER_TRIAL,
        len(selections) * trials * ARMS_PER_TRIAL,
        len(selections),
    )
    pipeline_started = time.perf_counter()
    output_tasks = pipeline.run(executor=RayDataExecutor(), initial_tasks=initial_tasks) or []
    pipeline_wall_s = time.perf_counter() - pipeline_started
    if len(output_tasks) != len(initial_tasks):
        raise RuntimeError(f"Persistent schedule returned {len(output_tasks)} tasks; expected {len(initial_tasks)}")
    results = [dict(task.data) for task in output_tasks]
    if sorted(int(result["actor_task_index"]) for result in results) != list(range(len(initial_tasks))):
        raise RuntimeError("Persistent actor task indexes are not contiguous; actor reuse was not preserved")
    for result in results:
        if result["arm"] == "parquet":
            result["parquet_image_encoding"] = _validate_parquet_image_encoding(result["output_uri"], storage)
    return results, pipeline_wall_s


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    storage = StorageConfig(
        endpoint=args.s3_endpoint,
        region=args.s3_region,
        dm_storage_location=args.dm_storage_location,
    )
    preparation = prepare_matched_inputs(
        source_uri=args.source_lance_uri,
        source_version=args.source_version,
        working_root=args.working_root,
        fragment_id=args.fragment_id,
        fragment_count=args.fragment_count,
        row_group_size=args.parquet_row_group_rows,
        storage=storage,
        reuse_prepared=args.reuse_prepared,
    )
    selections: dict[float, tuple[list[int], list[int], int]] = {}
    for sample_fraction in args.sample_fractions:
        offsets = sample_row_offsets(
            int(preparation["row_count"]),
            sample_fraction,
            args.sample_seed,
            int(preparation.get("sample_identity", preparation["selected_source_fragment_id"])),
        )
        row_groups, touched_rows = selected_parquet_row_groups(
            offsets, int(preparation["row_count"]), int(preparation["row_group_size"])
        )
        selections[sample_fraction] = (offsets, row_groups, touched_rows)
    schedule_results, pipeline_wall_s = _run_schedule(
        preparation=preparation,
        selections=selections,
        output_root=args.output_root,
        storage=storage,
        batch_rows=args.batch_rows,
        resize=args.resize,
        blur_threshold=args.blur_threshold,
        trials=args.trials,
    )
    warmups = [result for result in schedule_results if result["warmup"]]
    trials = [result for result in schedule_results if not result["warmup"]]
    if len(warmups) != ARMS_PER_TRIAL or {result["arm"] for result in warmups} != {"parquet", "lance"}:
        raise RuntimeError("Persistent schedule did not return exactly one warmup per arm")

    parity_groups: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for result in trials:
        key = (float(result["sample_fraction"]), int(result["trial"]))
        parity_groups.setdefault(key, []).append(result)
    parity_by_fraction = {
        sample_fraction: all(
            len(parity_groups.get((sample_fraction, trial), [])) == ARMS_PER_TRIAL
            and parity_groups[(sample_fraction, trial)][0]["sample_rows"]
            == parity_groups[(sample_fraction, trial)][1]["sample_rows"]
            and parity_groups[(sample_fraction, trial)][0]["output_rows"]
            == parity_groups[(sample_fraction, trial)][1]["output_rows"]
            and parity_groups[(sample_fraction, trial)][0]["kept_offset_digest"]
            == parity_groups[(sample_fraction, trial)][1]["kept_offset_digest"]
            and math.isclose(
                float(parity_groups[(sample_fraction, trial)][0]["score_sum"]),
                float(parity_groups[(sample_fraction, trial)][1]["score_sum"]),
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            for trial in range(args.trials)
        )
        for sample_fraction in selections
    }
    parity_valid = all(parity_by_fraction.values()) and all(
        len(pair) == ARMS_PER_TRIAL
        for pair in parity_groups.values()
    )
    metrics = {
        "is_success": parity_valid,
        "output_parity_valid": parity_valid,
        "source_fragment_rows": int(preparation["row_count"]),
        "runtime_setup_s": float(schedule_results[0]["runtime_setup_s"]),
        "pipeline_wall_s": pipeline_wall_s,
        "warmup_end_to_end_s_total": sum(float(result["end_to_end_s"]) for result in warmups),
        "measured_end_to_end_s_total": sum(float(result["end_to_end_s"]) for result in trials),
    }
    fraction_summaries: dict[str, dict[str, Any]] = {}
    for sample_fraction, (offsets, row_groups, touched_rows) in selections.items():
        label = fraction_label(sample_fraction)
        fraction_trials = [trial for trial in trials if float(trial["sample_fraction"]) == sample_fraction]
        summary: dict[str, Any] = {
            "sample_fraction": sample_fraction,
            "sample_rows": len(offsets),
            "sample_fraction_actual": len(offsets) / int(preparation["row_count"]),
            "parquet_row_groups_touched": len(row_groups),
            "parquet_rows_touched": touched_rows,
            "output_parity_valid": parity_by_fraction[sample_fraction],
            **_summarize_arm(fraction_trials, "parquet"),
            **_summarize_arm(fraction_trials, "lance"),
        }
        summary["lance_over_parquet_end_to_end_ratio"] = (
            summary["lance_end_to_end_s_mean"] / summary["parquet_end_to_end_s_mean"]
        )
        summary["lance_over_parquet_source_read_ratio"] = (
            summary["lance_source_read_s_mean"] / summary["parquet_source_read_s_mean"]
        )
        fraction_summaries[label] = summary
        for metric_name, value in summary.items():
            if isinstance(value, (bool, int, float)):
                metrics[f"{label}_{metric_name}"] = value
    metrics["pipeline_non_task_overhead_s"] = pipeline_wall_s - sum(
        float(result["end_to_end_s"]) for result in schedule_results
    )
    Path(args.benchmark_results_path).mkdir(parents=True, exist_ok=True)
    Path(args.benchmark_results_path, "trials.json").write_text(json.dumps(trials, indent=2, sort_keys=True) + "\n")
    Path(args.benchmark_results_path, "fraction_summaries.json").write_text(
        json.dumps(fraction_summaries, indent=2, sort_keys=True) + "\n"
    )
    return {
        "params": {
            "format_version": FORMAT_VERSION,
            "source_lance_uri": args.source_lance_uri,
            "source_version": args.source_version,
            "selected_source_fragment_id": preparation["selected_source_fragment_id"],
            "selected_source_fragment_ids": preparation.get("selected_source_fragment_ids", [preparation["selected_source_fragment_id"]]),
            "fragment_count": preparation.get("fragment_count", 1),
            "prepared_lance_version": preparation["prepared_lance_version"],
            "working_root": args.working_root,
            "output_root": args.output_root,
            "storage_label": args.storage_label,
            "sample_fractions": list(args.sample_fractions),
            "sample_seed": args.sample_seed,
            "batch_rows": args.batch_rows,
            "resize": args.resize,
            "blur_threshold": args.blur_threshold,
            "parquet_row_group_rows": args.parquet_row_group_rows,
            "trials": args.trials,
            "s3_endpoint": args.s3_endpoint,
            "s3_region": args.s3_region,
            "dm_storage_location": args.dm_storage_location,
            "runtime_versions": _runtime_versions(),
            "gpu_residency_contract": {
                "gpu_resident": "cuDF encoded image bytes and metadata, decoded pixels, blur scores, filter mask",
                "host_boundaries": (
                    "Lance Arrow reader/writer APIs plus an explicit encoded-image D2H decoder copy; "
                    "torchvision 0.25 rejects CUDA encoded JPEG tensors"
                ),
            },
        },
        "metrics": metrics,
        # The benchmark emits its fully instrumented per-arm records in
        # trials.json. Local task classes are intentionally not persisted: the
        # generic runner can aggregate an empty task list while keeping the
        # benchmark's explicit metrics intact.
        "tasks": [],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    parser.add_argument("--source-lance-uri", default=DEFAULT_SOURCE_URI)
    parser.add_argument("--source-version", type=int, default=DEFAULT_SOURCE_VERSION)
    parser.add_argument(
        "--working-root", required=True, help="NVMe, Weka, or S3 root for matched full-fragment inputs"
    )
    parser.add_argument("--output-root", required=True, help="NVMe, Weka, or S3 root for measured outputs")
    parser.add_argument("--fragment-id", type=int, help="Explicit source fragment; default chooses median row count")
    parser.add_argument(
        "--fragment-count",
        type=int,
        default=1,
        help="Number of representative source fragments to combine into the prepared cohort",
    )
    parser.add_argument(
        "--sample-fractions",
        default=",".join(str(value) for value in DEFAULT_SAMPLE_FRACTIONS),
        help="Comma-separated fractions run through one persistent GPU actor",
    )
    parser.add_argument("--storage-label", default="unspecified")
    parser.add_argument("--sample-seed", default="mint-image-format-single-fragment-v1")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--parquet-row-group-rows", type=int, default=1024)
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--blur-threshold", type=float, default=0.10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--s3-endpoint", default=os.environ.get("AWS_ENDPOINT_URL_S3"))
    parser.add_argument("--s3-region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    parser.add_argument(
        "--dm-storage-location",
        default=os.environ.get("CURATOR_DM_STORAGE_LOCATION", "pdx-multimodal"),
        help="Data Mover location whose local credentials are loaded inside the driver and Ray worker",
    )
    args = parser.parse_args()
    if (
        args.batch_rows <= 0
        or args.parquet_row_group_rows <= 0
        or args.resize <= 0
        or args.trials <= 0
        or args.fragment_count <= 0
    ):
        parser.error("batch sizes, resize, trials, and fragment count must be positive")
    try:
        args.sample_fractions = parse_sample_fractions(args.sample_fractions)
    except ValueError as error:
        parser.error(str(error))
    if args.blur_threshold < 0:
        parser.error("--blur-threshold must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    results: dict[str, Any]
    try:
        results = run_benchmark(args)
    except Exception as error:
        logger.error("GPU image table format benchmark failed: {}", error)
        logger.debug(traceback.format_exc())
        results = {
            "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "metrics": {"is_success": False, "error_type": type(error).__name__, "error": str(error)},
            "tasks": [],
        }

    from utils import write_benchmark_results

    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"].get("is_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
