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

"""Persistent GPU exact-key membership for row-wise interleaved batches."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import resource
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.uri import validate_credential_free_uri_identity

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from nemo_curator.stages.interleaved.lance import _StableGlobalOrdinalManifest


_GPU_LANCE_SIDECAR_FORMAT = "nemo-curator-gpu-lance-sidecar-v2"
_MPF_HASH_ALGORITHM = "cudf::hash_id::HASH_MURMUR3"
_MPF_HASH_IMPLEMENTATION = "rapidsmpf.integrations.cudf.partition.partition_and_pack"
_MPF_HASH_SEED = 0
_SHA256_HEX_LENGTH = 64
_STABLE_ID_COVERAGE_DTYPE = "uint32"


@dataclass(frozen=True)
class _SidecarFileIdentity:
    path: str
    partition_id: int
    ordinal: int
    rows: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class _MpfHashPartitioningContract:
    algorithm: str
    implementation: str
    libcudf_version: str
    rapidsmpf_version: str
    seed: int

    def to_payload(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "implementation": self.implementation,
            "libcudf_version": self.libcudf_version,
            "rapidsmpf_version": self.rapidsmpf_version,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class _GpuLanceSidecarContract:
    dataset_uri: str
    dataset_version: int
    fragment_manifest_sha256: str
    key_stable_ordinal_sha256: str
    total_rows: int
    key_column: str
    row_id_column: str
    layout: str
    partition_count: int
    stable_id_min: int
    stable_id_max: int
    files: tuple[_SidecarFileIdentity, ...]
    partitioning: _MpfHashPartitioningContract | None


class _GpuHashSeries(Protocol):
    def __mod__(self, other: int) -> _GpuHashSeries: ...

    def __ne__(self, other: object) -> _GpuHashSeries: ...

    def sum(self) -> int: ...


class _GpuHashFrame(Protocol):
    def __getitem__(self, key: list[str]) -> _GpuHashFrame: ...

    def hash_values(self, *, method: str, seed: int) -> _GpuHashSeries: ...


class _IdentityScanner(Protocol):
    def to_batches(self) -> Iterable[pa.RecordBatch]: ...


class _IdentityDataset(Protocol):
    schema: pa.Schema

    def scanner(self, **kwargs: object) -> _IdentityScanner: ...


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _stable_global_ordinal_manifest_sha256(
    dataset_uri: str,
    dataset_version: int,
    manifest: _StableGlobalOrdinalManifest,
) -> str:
    """Fingerprint the exact pinned fragment order used by global ordinals."""
    validate_credential_free_uri_identity(dataset_uri, "Lance dataset URI")
    payload = {
        "dataset_uri": dataset_uri,
        "dataset_version": dataset_version,
        "fragment_rows": list(manifest.fragment_rows),
        "format": "nemo-curator-stable-global-ordinal-v1",
        "total_rows": manifest.total_rows,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _offset_safe_key_take_source(keys: pa.ChunkedArray, key_type: pa.DataType) -> pa.Array:
    """Combine keys with 64-bit offsets when a string take can exceed 2 GiB."""
    take_type = pa.large_string() if pa.types.is_string(key_type) else key_type
    return keys.cast(take_type).combine_chunks()


def _sorted_sidecar_identity_sha256(
    identity: pa.Table,
    *,
    key_column: str,
    row_id_column: str,
    key_field: pa.Field,
    total_rows: int,
) -> str:
    """Hash a sidecar in stable-ordinal order using bounded selected batches."""
    from nemo_curator.stages.interleaved.lance import _KEY_STABLE_ORDINAL_FORMAT, _MIRROR_IDENTITY_BATCH_ROWS

    order = pc.sort_indices(identity, sort_keys=[(row_id_column, "ascending")])
    row_id_take_source = identity[row_id_column].combine_chunks()
    key_take_source = _offset_safe_key_take_source(identity[key_column], key_field.type)
    identity_schema = pa.schema(
        [
            pa.field("stable_global_ordinal", pa.uint64(), nullable=False),
            key_field,
        ]
    )
    digest = hashlib.sha256()
    digest.update(_KEY_STABLE_ORDINAL_FORMAT.encode())
    schema_bytes = identity_schema.serialize().to_pybytes()
    digest.update(len(schema_bytes).to_bytes(8, "little"))
    digest.update(schema_bytes)

    offset = 0
    while offset < total_rows:
        batch_rows = min(_MIRROR_IDENTITY_BATCH_ROWS, total_rows - offset)
        batch_order = order.slice(offset, batch_rows)
        row_ids = pc.take(row_id_take_source, batch_order).cast(pa.uint64())
        expected_ids = pa.array(range(offset, offset + batch_rows), type=pa.uint64())
        if not bool(pc.all(pc.equal(row_ids, expected_ids)).as_py()):
            msg = f"Sidecar key identity is not an exact stable-global-ordinal permutation at row {offset}"
            raise ValueError(msg)
        keys = pc.take(key_take_source, batch_order)
        if keys.type != key_field.type:
            keys = keys.cast(key_field.type)
        identity_batch = pa.RecordBatch.from_arrays([row_ids, keys], schema=identity_schema)
        batch_bytes = identity_batch.serialize().to_pybytes()
        digest.update(batch_rows.to_bytes(8, "little"))
        digest.update(len(batch_bytes).to_bytes(8, "little"))
        digest.update(batch_bytes)
        offset += batch_rows
    return digest.hexdigest()


def _sidecar_key_stable_ordinal_sha256(  # noqa: PLR0913
    dataset: _IdentityDataset,
    *,
    key_column: str,
    row_id_column: str,
    partition_files: Sequence[Sequence[str]],
    storage_options: Mapping[str, str],
    total_rows: int,
) -> str:
    """Prove that every sidecar key names the same pinned Lance ordinal."""
    from nemo_curator.stages.interleaved.lance import _key_stable_ordinal_sha256

    key_field = dataset.schema.field(key_column)
    tables: list[pa.Table] = []
    for paths in partition_files:
        for path in paths:
            with fsspec.open(path, "rb", **dict(storage_options)) as stream:
                table = pq.read_table(stream, columns=[row_id_column, key_column])
            if table.schema.field(row_id_column).type != pa.uint64():
                msg = f"Sidecar row-ID column in {path} must have type uint64"
                raise TypeError(msg)
            if table.schema.field(key_column).type != key_field.type:
                msg = (
                    f"Sidecar key column in {path} has type {table.schema.field(key_column).type}; "
                    f"pinned Lance column has type {key_field.type}"
                )
                raise TypeError(msg)
            if table[key_column].null_count:
                msg = f"Sidecar key column contains nulls: {path}"
                raise ValueError(msg)
            tables.append(table)
    if not tables:
        msg = "Sidecar key identity requires at least one Parquet file"
        raise ValueError(msg)

    identity = pa.concat_tables(tables)
    if identity.num_rows != total_rows:
        msg = f"Sidecar key identity contains {identity.num_rows} rows; expected {total_rows}"
        raise ValueError(msg)
    sidecar_digest = _sorted_sidecar_identity_sha256(
        identity,
        key_column=key_column,
        row_id_column=row_id_column,
        key_field=key_field,
        total_rows=total_rows,
    )
    dataset_digest = _key_stable_ordinal_sha256(dataset, key_column, total_rows=total_rows)
    if sidecar_digest != dataset_digest:
        msg = "Sidecar key-to-stable-ordinal identity does not match the pinned Lance dataset"
        raise ValueError(msg)
    return dataset_digest


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        msg = f"{label} must be a lowercase SHA-256 hex digest"
        raise ValueError(msg)
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        msg = f"{label} must be a lowercase SHA-256 hex digest"
        raise ValueError(msg) from exc
    if value != value.lower():
        msg = f"{label} must be a lowercase SHA-256 hex digest"
        raise ValueError(msg)
    return value


def _require_contract_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{label} must be an integer greater than or equal to {minimum}"
        raise ValueError(msg)
    return value


def _read_bytes(path: str, storage_options: Mapping[str, str]) -> bytes:
    with fsspec.open(path, "rb", **dict(storage_options)) as stream:
        return stream.read()


def _file_sha256_and_size(path: str, storage_options: Mapping[str, str]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with fsspec.open(path, "rb", **dict(storage_options)) as stream:
        while chunk := stream.read(16 * 1024**2):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _parquet_rows(path: str, storage_options: Mapping[str, str]) -> int:
    with fsspec.open(path, "rb", **dict(storage_options)) as stream:
        return pq.read_metadata(stream).num_rows


def _runtime_mpf_hash_partitioning_contract() -> _MpfHashPartitioningContract:
    """Describe the exact RAPIDS-MPF/libcudf hash implementation in this process."""
    try:
        rapidsmpf_version = importlib.metadata.version("rapidsmpf-cu12")
        libcudf_version = importlib.metadata.version("libcudf-cu12")
    except importlib.metadata.PackageNotFoundError as exc:
        msg = "Hash-partitioned GPU Lance sidecars require rapidsmpf-cu12 and libcudf-cu12"
        raise ImportError(msg) from exc
    return _MpfHashPartitioningContract(
        algorithm=_MPF_HASH_ALGORITHM,
        implementation=_MPF_HASH_IMPLEMENTATION,
        libcudf_version=libcudf_version,
        rapidsmpf_version=rapidsmpf_version,
        seed=_MPF_HASH_SEED,
    )


def _mpf_partition_ids(frame: _GpuHashFrame, key_column: str, partition_count: int) -> _GpuHashSeries:
    """Return the partition IDs used by RAPIDS-MPF ``partition_and_pack``."""
    if partition_count <= 0:
        msg = "partition_count must be greater than zero"
        raise ValueError(msg)
    # The pinned RAPIDS-MPF implementation calls cudf::hash_partition with
    # HASH_MURMUR3 and DEFAULT_HASH_SEED.  Explicit seed=0 keeps uint32 modulo semantics.
    hashes = frame[[key_column]].hash_values(method="murmur3", seed=_MPF_HASH_SEED)
    return hashes % partition_count


def _validate_mpf_partition_ownership(
    frame: _GpuHashFrame,
    *,
    key_column: str,
    expected_partition: int,
    partition_count: int,
) -> None:
    """Fail unless every key has the exact partition RAPIDS-MPF will assign."""
    if not 0 <= expected_partition < partition_count:
        msg = f"expected_partition must be in [0, {partition_count}), got {expected_partition}"
        raise ValueError(msg)
    partition_ids = _mpf_partition_ids(frame, key_column, partition_count)
    misplaced = int((partition_ids != expected_partition).sum())
    if misplaced:
        msg = (
            f"Hash sidecar partition {expected_partition} contains {misplaced} keys owned by another "
            f"RAPIDS-MPF partition out of {partition_count}"
        )
        raise ValueError(msg)


def _validate_hash_partitioned_sidecars(
    *,
    partition_files: Sequence[Sequence[str]],
    key_column: str,
    storage_options: Mapping[str, str],
) -> _MpfHashPartitioningContract:
    """Prove global key uniqueness and exact MPF ownership for hash sidecars."""
    try:
        import cudf
    except ImportError as exc:
        msg = "Building a hash-partitioned GPU Lance sidecar contract requires cudf"
        raise ImportError(msg) from exc

    contract = _runtime_mpf_hash_partitioning_contract()
    partition_count = len(partition_files)
    for partition_id, paths in enumerate(partition_files):
        frame = cudf.read_parquet(
            list(paths),
            columns=[key_column],
            storage_options=dict(storage_options) or None,
        )
        if len(frame) == 0:
            msg = f"Hash sidecar partition {partition_id} is empty"
            raise ValueError(msg)
        if frame[key_column].isnull().any():
            msg = f"Hash sidecar partition {partition_id} contains null keys"
            raise ValueError(msg)
        if frame[key_column].duplicated().any():
            msg = f"Hash sidecar partition {partition_id} contains duplicate keys"
            raise ValueError(msg)
        _validate_mpf_partition_ownership(
            frame,
            key_column=key_column,
            expected_partition=partition_id,
            partition_count=partition_count,
        )
    # A key has one deterministic MPF owner.  Correct ownership for every row,
    # plus uniqueness inside each owner, proves uniqueness across all shards.
    return contract


def _build_sidecar_contract_bytes(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    dataset: _IdentityDataset,
    dataset_uri: str,
    dataset_version: int,
    fragment_manifest_sha256: str,
    total_rows: int,
    key_column: str,
    row_id_column: str,
    layout: str,
    partition_files: Sequence[Sequence[str]],
    storage_options: Mapping[str, str],
) -> tuple[bytes, str]:
    """Build a deterministic contract after proving exact stable-ID coverage."""
    validate_credential_free_uri_identity(dataset_uri, "Lance dataset URI")
    for paths in partition_files:
        for path in paths:
            validate_credential_free_uri_identity(path, "sidecar file URI")
    if total_rows <= 0:
        msg = "total_rows must be greater than zero"
        raise ValueError(msg)
    if layout not in {"replicated_sorted", "hash_partitioned"}:
        msg = f"Unsupported sidecar layout: {layout!r}"
        raise ValueError(msg)
    if layout == "replicated_sorted" and len(partition_files) != 1:
        msg = "replicated_sorted layout requires exactly one partition"
        raise ValueError(msg)
    partitioning = (
        _validate_hash_partitioned_sidecars(
            partition_files=partition_files,
            key_column=key_column,
            storage_options=storage_options,
        )
        if layout == "hash_partitioned"
        else None
    )
    covered = np.zeros(total_rows, dtype=np.bool_)
    files: list[dict[str, object]] = []
    observed_rows = 0
    observed_min: int | None = None
    observed_max: int | None = None
    for partition_id, paths in enumerate(partition_files):
        if not paths:
            msg = f"Sidecar partition {partition_id} is empty"
            raise ValueError(msg)
        for ordinal, path in enumerate(paths):
            sha256, size_bytes = _file_sha256_and_size(path, storage_options)
            with fsspec.open(path, "rb", **dict(storage_options)) as stream:
                parquet_file = pq.ParquetFile(stream)
                schema = parquet_file.schema_arrow
                missing = sorted({key_column, row_id_column} - set(schema.names))
                if missing:
                    msg = f"Sidecar file {path} is missing columns: {missing}"
                    raise ValueError(msg)
                row_id_type = schema.field(row_id_column).type
                if row_id_type != pa.uint64():
                    msg = f"Sidecar row-ID column has type {row_id_type} in {path}; expected uint64"
                    raise TypeError(msg)
                file_rows = parquet_file.metadata.num_rows
                if file_rows <= 0:
                    msg = f"Sidecar file is empty: {path}"
                    raise ValueError(msg)
                for batch in parquet_file.iter_batches(batch_size=1_048_576, columns=[row_id_column]):
                    row_ids = batch.column(0)
                    if row_ids.null_count:
                        msg = f"Sidecar row-ID column contains nulls: {path}"
                        raise ValueError(msg)
                    values = row_ids.to_numpy(zero_copy_only=False)
                    batch_min = int(values.min())
                    batch_max = int(values.max())
                    if batch_min < 0 or batch_max >= total_rows:
                        msg = (
                            f"Sidecar stable IDs in {path} span [{batch_min}, {batch_max}]; "
                            f"expected values in [0, {total_rows})"
                        )
                        raise ValueError(msg)
                    unique = np.unique(values)
                    if len(unique) != len(values) or bool(covered[unique].any()):
                        msg = "Sidecar stable IDs are not globally unique"
                        raise ValueError(msg)
                    covered[unique] = True
                    observed_min = batch_min if observed_min is None else min(observed_min, batch_min)
                    observed_max = batch_max if observed_max is None else max(observed_max, batch_max)
            files.append(
                {
                    "ordinal": ordinal,
                    "partition_id": partition_id,
                    "path": path,
                    "rows": file_rows,
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                }
            )
            observed_rows += file_rows
    if observed_rows != total_rows or not bool(covered.all()):
        msg = (
            "Sidecar stable IDs do not exactly cover the pinned global-ordinal range: "
            f"observed_rows={observed_rows}, total_rows={total_rows}"
        )
        raise ValueError(msg)
    if observed_min != 0 or observed_max != total_rows - 1:
        msg = f"Sidecar stable-ID bounds are [{observed_min}, {observed_max}]; expected [0, {total_rows - 1}]"
        raise ValueError(msg)
    key_stable_ordinal_sha256 = _sidecar_key_stable_ordinal_sha256(
        dataset,
        key_column=key_column,
        row_id_column=row_id_column,
        partition_files=partition_files,
        storage_options=storage_options,
        total_rows=total_rows,
    )
    payload = {
        "dataset_uri": dataset_uri,
        "dataset_version": dataset_version,
        "files": files,
        "format": _GPU_LANCE_SIDECAR_FORMAT,
        "fragment_manifest_sha256": _require_sha256(
            fragment_manifest_sha256,
            "stable global-ordinal manifest SHA-256",
        ),
        "key_column": key_column,
        "key_stable_ordinal_sha256": key_stable_ordinal_sha256,
        "layout": layout,
        "partition_count": len(partition_files),
        "row_id_column": row_id_column,
        "stable_id_max": total_rows - 1,
        "stable_id_min": 0,
        "total_rows": total_rows,
    }
    if partitioning is not None:
        payload["partitioning"] = partitioning.to_payload()
    raw_manifest = _canonical_json_bytes(payload)
    return raw_manifest, hashlib.sha256(raw_manifest).hexdigest()


def _load_and_validate_sidecar_contract(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    manifest_uri: str,
    manifest_sha256: str,
    dataset_uri: str,
    dataset_version: int,
    fragment_manifest_sha256: str,
    total_rows: int,
    key_column: str,
    row_id_column: str,
    layout: str,
    partition_files: Sequence[Sequence[str]],
    storage_options: Mapping[str, str],
    actual_files: Sequence[Sequence[str]] | None = None,
    actual_storage_options: Mapping[str, str] | None = None,
    verify_file_coordinates: set[tuple[int, int]] | None = None,
) -> _GpuLanceSidecarContract:
    """Verify a caller-pinned sidecar manifest and every referenced Parquet file."""
    validate_credential_free_uri_identity(manifest_uri, "sidecar manifest URI")
    validate_credential_free_uri_identity(dataset_uri, "Lance dataset URI")
    for files in (partition_files, actual_files or ()):
        for paths in files:
            for path in paths:
                validate_credential_free_uri_identity(path, "sidecar file URI")
    expected_manifest_sha256 = _require_sha256(manifest_sha256, "sidecar manifest SHA-256")
    raw_manifest = _read_bytes(manifest_uri, storage_options)
    actual_manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        msg = f"Sidecar manifest SHA-256 is {actual_manifest_sha256}; expected {expected_manifest_sha256}"
        raise ValueError(msg)
    try:
        payload = json.loads(raw_manifest)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        msg = f"Sidecar manifest is not valid UTF-8 JSON: {manifest_uri}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict) or raw_manifest != _canonical_json_bytes(payload):
        msg = "Sidecar manifest must use canonical sorted compact JSON with one trailing newline"
        raise ValueError(msg)

    required_keys = {
        "dataset_uri",
        "dataset_version",
        "files",
        "format",
        "fragment_manifest_sha256",
        "key_column",
        "key_stable_ordinal_sha256",
        "layout",
        "partition_count",
        "row_id_column",
        "stable_id_max",
        "stable_id_min",
        "total_rows",
    }
    if layout == "hash_partitioned":
        required_keys.add("partitioning")
    if set(payload) != required_keys:
        msg = f"Sidecar manifest keys differ from the v2 contract: {sorted(set(payload) ^ required_keys)}"
        raise ValueError(msg)
    validate_credential_free_uri_identity(payload["dataset_uri"], "sidecar manifest dataset URI")
    key_stable_ordinal_sha256 = _require_sha256(
        payload["key_stable_ordinal_sha256"],
        "sidecar key-to-stable-ordinal SHA-256",
    )
    partitioning = _runtime_mpf_hash_partitioning_contract() if layout == "hash_partitioned" else None
    expected_scalars = {
        "format": _GPU_LANCE_SIDECAR_FORMAT,
        "dataset_uri": dataset_uri,
        "dataset_version": dataset_version,
        "fragment_manifest_sha256": _require_sha256(
            fragment_manifest_sha256,
            "stable global-ordinal manifest SHA-256",
        ),
        "total_rows": total_rows,
        "key_column": key_column,
        "row_id_column": row_id_column,
        "layout": layout,
        "partition_count": len(partition_files),
        "stable_id_min": 0,
        "stable_id_max": total_rows - 1,
    }
    if partitioning is not None:
        expected_scalars["partitioning"] = partitioning.to_payload()
    for name, expected in expected_scalars.items():
        if payload.get(name) != expected:
            msg = f"Sidecar manifest {name}={payload.get(name)!r}; expected {expected!r}"
            raise ValueError(msg)

    manifest_files = payload["files"]
    if not isinstance(manifest_files, list):
        msg = "Sidecar manifest files must be a list"
        raise TypeError(msg)
    expected_coordinates = [
        (partition_id, ordinal, path)
        for partition_id, paths in enumerate(partition_files)
        for ordinal, path in enumerate(paths)
    ]
    resolved_actual_files = partition_files if actual_files is None else actual_files
    resolved_actual_storage_options = storage_options if actual_storage_options is None else actual_storage_options
    actual_coordinates = [
        (partition_id, ordinal, path)
        for partition_id, paths in enumerate(resolved_actual_files)
        for ordinal, path in enumerate(paths)
    ]
    if [(partition, ordinal) for partition, ordinal, _ in actual_coordinates] != [
        (partition, ordinal) for partition, ordinal, _ in expected_coordinates
    ]:
        msg = "Actual sidecar file layout differs from the manifest partition layout"
        raise ValueError(msg)
    if len(manifest_files) != len(expected_coordinates):
        msg = f"Sidecar manifest contains {len(manifest_files)} files; expected {len(expected_coordinates)}"
        raise ValueError(msg)

    identities: list[_SidecarFileIdentity] = []
    total_file_rows = 0
    required_file_keys = {"ordinal", "partition_id", "path", "rows", "sha256", "size_bytes"}
    for raw_identity, expected_coordinate, actual_coordinate in zip(
        manifest_files,
        expected_coordinates,
        actual_coordinates,
        strict=True,
    ):
        if not isinstance(raw_identity, dict) or set(raw_identity) != required_file_keys:
            msg = "Each sidecar file entry must contain exactly path/partition/ordinal/rows/size/SHA-256"
            raise ValueError(msg)
        validate_credential_free_uri_identity(raw_identity["path"], "sidecar manifest file URI")
        partition_id, ordinal, source_path = expected_coordinate
        _, _, actual_path = actual_coordinate
        if (
            raw_identity["path"] != source_path
            or raw_identity["partition_id"] != partition_id
            or raw_identity["ordinal"] != ordinal
        ):
            msg = (
                "Sidecar manifest file order does not match configured partitions: "
                f"entry={raw_identity!r}, expected=({partition_id}, {ordinal}, {source_path!r})"
            )
            raise ValueError(msg)
        rows = _require_contract_int(raw_identity["rows"], "sidecar file rows", minimum=1)
        size_bytes = _require_contract_int(raw_identity["size_bytes"], "sidecar file size", minimum=1)
        sha256 = _require_sha256(raw_identity["sha256"], "sidecar file SHA-256")
        if verify_file_coordinates is None or (partition_id, ordinal) in verify_file_coordinates:
            actual_sha256, actual_size = _file_sha256_and_size(actual_path, resolved_actual_storage_options)
            if (actual_sha256, actual_size) != (sha256, size_bytes):
                msg = (
                    f"Sidecar file identity mismatch for {source_path}: "
                    f"sha256={actual_sha256}, size_bytes={actual_size}; "
                    f"expected sha256={sha256}, size_bytes={size_bytes}"
                )
                raise ValueError(msg)
            parquet_rows = _parquet_rows(actual_path, resolved_actual_storage_options)
            if parquet_rows != rows:
                msg = f"Sidecar file {source_path} contains {parquet_rows} rows; manifest declares {rows}"
                raise ValueError(msg)
        identities.append(
            _SidecarFileIdentity(
                path=source_path,
                partition_id=partition_id,
                ordinal=ordinal,
                rows=rows,
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )
        total_file_rows += rows
    if total_file_rows != total_rows:
        msg = f"Sidecar files declare {total_file_rows} rows; pinned Lance manifest has {total_rows}"
        raise ValueError(msg)
    return _GpuLanceSidecarContract(
        dataset_uri=dataset_uri,
        dataset_version=dataset_version,
        fragment_manifest_sha256=fragment_manifest_sha256,
        key_stable_ordinal_sha256=key_stable_ordinal_sha256,
        total_rows=total_rows,
        key_column=key_column,
        row_id_column=row_id_column,
        layout=layout,
        partition_count=len(partition_files),
        stable_id_min=0,
        stable_id_max=total_rows - 1,
        files=tuple(identities),
        partitioning=partitioning,
    )


@dataclass(frozen=True)
class _GpuMatchResult:
    matched: np.ndarray
    transfer_seconds: float
    probe_seconds: float
    gather_seconds: float


@dataclass(frozen=True)
class _GpuMapResult:
    matched: np.ndarray
    row_ids: np.ndarray
    transfer_seconds: float
    probe_seconds: float
    search_seconds: float
    gather_seconds: float


class _GpuExactKeyMatcher:
    """Own persistent RAPIDS filtered joins for immutable reference segments."""

    def __init__(  # noqa: PLR0915
        self,
        reference_files: Sequence[str],
        reference_key_column: str,
        storage_options: dict[str, str],
        expected_reference_rows: int | None,
        load_factor: float,
    ) -> None:
        try:
            import cudf
            import cupy as cp
            import pylibcudf as plc
            from pylibcudf.join import FilteredJoin
            from pylibcudf.types import NullEquality
        except ImportError as exc:  # pragma: no cover - exercised only without the optional GPU dependency
            msg = "GpuExactKeyLookupStage requires cudf-cu12==26.6.*"
            raise ImportError(msg) from exc

        self._cp = cp
        self._plc = plc
        self._frames: list[Any] = []
        self._build_tables: list[Any] = []
        self._joins: list[Any] = []
        reference_type: pa.DataType | None = None

        free_before, total_memory = cp.cuda.runtime.memGetInfo()
        load_started = time.perf_counter()
        build_seconds = 0.0
        reference_rows = 0
        for path in reference_files:
            frame = cudf.read_parquet(
                path,
                columns=[reference_key_column],
                storage_options=storage_options or None,
            )
            if len(frame) == 0:
                msg = f"Reference key segment is empty: {path}"
                raise ValueError(msg)
            frame_type = frame[reference_key_column].head(1).to_arrow().type
            if reference_type is None:
                reference_type = frame_type
            elif frame_type != reference_type:
                msg = f"Reference key column has type {frame_type} in {path}; expected {reference_type}"
                raise TypeError(msg)
            if frame[reference_key_column].null_count:
                msg = f"Reference key column {reference_key_column!r} contains nulls in {path}"
                raise ValueError(msg)
            build_table = frame[[reference_key_column]].to_pylibcudf()[0]
            build_started = time.perf_counter()
            join = FilteredJoin(build_table, NullEquality.UNEQUAL, load_factor)
            cp.cuda.runtime.deviceSynchronize()
            build_seconds += time.perf_counter() - build_started
            reference_rows += len(frame)
            # libcudf's filtered_join stores a view of the build table. Retain
            # both owners for the complete lifetime of the join object.
            self._frames.append(frame)
            self._build_tables.append(build_table)
            self._joins.append(join)

        if expected_reference_rows is not None and reference_rows != expected_reference_rows:
            msg = f"Reference sidecars contain {reference_rows} rows; expected {expected_reference_rows}"
            raise ValueError(msg)
        free_after, _ = cp.cuda.runtime.memGetInfo()
        self.reference_rows = reference_rows
        self.load_seconds = time.perf_counter() - load_started - build_seconds
        self.build_seconds = build_seconds
        self.gpu_bytes = free_before - free_after
        self.gpu_total_bytes = total_memory
        if reference_type is None:  # pragma: no cover - reference_files is validated as non-empty
            msg = "GPU reference loading did not discover a key type"
            raise RuntimeError(msg)
        self.reference_type = reference_type

    def match(self, keys: pa.Array) -> _GpuMatchResult:
        if not len(keys):
            return _GpuMatchResult(np.zeros(0, dtype=np.bool_), 0.0, 0.0, 0.0)

        transfer_started = time.perf_counter()
        probe = self._plc.Table([self._plc.Column.from_arrow(keys)])
        self._cp.cuda.runtime.deviceSynchronize()
        transfer_seconds = time.perf_counter() - transfer_started

        probe_started = time.perf_counter()
        gather_maps = [join.semi_join(probe) for join in self._joins]
        self._cp.cuda.runtime.deviceSynchronize()
        probe_seconds = time.perf_counter() - probe_started

        gather_started = time.perf_counter()
        matched = np.zeros(len(keys), dtype=np.bool_)
        for gather_map in gather_maps:
            indices = gather_map.to_arrow().to_numpy(zero_copy_only=False)
            matched[indices] = True
        gather_seconds = time.perf_counter() - gather_started
        return _GpuMatchResult(matched, transfer_seconds, probe_seconds, gather_seconds)

    def close(self) -> None:
        self._joins.clear()
        self._build_tables.clear()
        self._frames.clear()


class _GpuExactKeyMapper:
    """Own persistent GPU key indices and map probes to stable Lance row IDs."""

    def __init__(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        reference_files: Sequence[str],
        reference_key_column: str,
        reference_row_id_column: str,
        storage_options: dict[str, str],
        expected_reference_rows: int,
        load_factor: float,
    ) -> None:
        try:
            import cudf
            import cupy as cp
            import pylibcudf as plc
            from pylibcudf.join import FilteredJoin
            from pylibcudf.types import NullEquality, NullOrder, Order
        except ImportError as exc:  # pragma: no cover - exercised only without the optional GPU dependency
            msg = "GpuLanceColumnFetchStage requires cudf-cu12==26.6.*"
            raise ImportError(msg) from exc

        self._cp = cp
        self._plc = plc
        self._null_order = NullOrder.AFTER
        self._order = Order.ASCENDING
        self._frames: list[Any] = []
        self._key_tables: list[Any] = []
        self._row_id_tables: list[Any] = []
        self._joins: list[Any] = []
        reference_type: pa.DataType | None = None

        free_before, total_memory = cp.cuda.runtime.memGetInfo()
        load_started = time.perf_counter()
        build_seconds = 0.0
        reference_rows = 0
        # CuPy scatter-add does not support uint8 accumulators.
        stable_id_coverage = cp.zeros(
            expected_reference_rows,
            dtype=_STABLE_ID_COVERAGE_DTYPE,
        )
        for path in reference_files:
            frame = cudf.read_parquet(
                path,
                columns=[reference_key_column, reference_row_id_column],
                storage_options=storage_options or None,
            )
            if len(frame) == 0:
                msg = f"Reference key segment is empty: {path}"
                raise ValueError(msg)
            frame_type = frame[reference_key_column].head(1).to_arrow().type
            if reference_type is None:
                reference_type = frame_type
            elif frame_type != reference_type:
                msg = f"Reference key column has type {frame_type} in {path}; expected {reference_type}"
                raise TypeError(msg)
            if frame[reference_key_column].null_count:
                msg = f"Reference key column {reference_key_column!r} contains nulls in {path}"
                raise ValueError(msg)
            row_id_type = frame[reference_row_id_column].head(1).to_arrow().type
            if row_id_type != pa.uint64():
                msg = f"Reference row-ID column has type {row_id_type} in {path}; expected uint64"
                raise TypeError(msg)
            if frame[reference_row_id_column].null_count:
                msg = f"Reference row-ID column {reference_row_id_column!r} contains nulls in {path}"
                raise ValueError(msg)
            segment_min = int(frame[reference_row_id_column].min())
            segment_max = int(frame[reference_row_id_column].max())
            if segment_min < 0 or segment_max >= expected_reference_rows:
                msg = (
                    f"Reference stable IDs in {path} span [{segment_min}, {segment_max}]; "
                    f"expected values in [0, {expected_reference_rows})"
                )
                raise ValueError(msg)
            cp.add.at(stable_id_coverage, frame[reference_row_id_column].values, 1)

            key_table = frame[[reference_key_column]].to_pylibcudf()[0]
            if not plc.sorting.is_sorted(
                key_table,
                [self._order],
                [self._null_order],
            ):
                msg = f"Reference key segment is not sorted by {reference_key_column!r}: {path}"
                raise ValueError(msg)
            row_id_table = frame[[reference_row_id_column]].to_pylibcudf()[0]
            build_started = time.perf_counter()
            join = FilteredJoin(key_table, NullEquality.UNEQUAL, load_factor)
            cp.cuda.runtime.deviceSynchronize()
            build_seconds += time.perf_counter() - build_started
            reference_rows += len(frame)
            self._frames.append(frame)
            self._key_tables.append(key_table)
            self._row_id_tables.append(row_id_table)
            self._joins.append(join)

        if reference_rows != expected_reference_rows:
            msg = f"Reference key segments contain {reference_rows} rows; expected {expected_reference_rows}"
            raise ValueError(msg)
        cp.cuda.runtime.deviceSynchronize()
        if not bool(cp.all(stable_id_coverage == 1)):
            msg = (
                "Reference stable IDs must be a duplicate-free permutation of the pinned "
                f"global-ordinal range [0, {expected_reference_rows})"
            )
            raise ValueError(msg)
        del stable_id_coverage
        free_after, _ = cp.cuda.runtime.memGetInfo()
        self.reference_rows = reference_rows
        self.load_seconds = time.perf_counter() - load_started - build_seconds
        self.build_seconds = build_seconds
        self.gpu_bytes = free_before - free_after
        self.gpu_total_bytes = total_memory
        if reference_type is None:  # pragma: no cover - reference_files is validated as non-empty
            msg = "GPU reference loading did not discover a key type"
            raise RuntimeError(msg)
        self.reference_type = reference_type

    def map(self, keys: pa.Array) -> _GpuMapResult:
        if not len(keys):
            return _GpuMapResult(
                np.zeros(0, dtype=np.bool_),
                np.zeros(0, dtype=np.uint64),
                0.0,
                0.0,
                0.0,
                0.0,
            )

        transfer_started = time.perf_counter()
        probe = self._plc.Table([self._plc.Column.from_arrow(keys)])
        self._cp.cuda.runtime.deviceSynchronize()
        transfer_seconds = time.perf_counter() - transfer_started

        probe_started = time.perf_counter()
        gather_maps = [join.semi_join(probe) for join in self._joins]
        self._cp.cuda.runtime.deviceSynchronize()
        probe_seconds = time.perf_counter() - probe_started

        search_started = time.perf_counter()
        located: list[tuple[Any, Any, Any, Any]] = []
        for key_table, row_id_table, gather_map in zip(
            self._key_tables,
            self._row_id_tables,
            gather_maps,
            strict=True,
        ):
            if gather_map.size() == 0:
                continue
            matched_keys = self._plc.copying.gather(
                probe,
                gather_map,
                self._plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
            lower = self._plc.search.lower_bound(
                key_table,
                matched_keys,
                [self._order],
                [self._null_order],
            )
            upper = self._plc.search.upper_bound(
                key_table,
                matched_keys,
                [self._order],
                [self._null_order],
            )
            mapped_rows = self._plc.copying.gather(
                row_id_table,
                lower,
                self._plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
            located.append((gather_map, lower, upper, mapped_rows))
        self._cp.cuda.runtime.deviceSynchronize()
        search_seconds = time.perf_counter() - search_started

        gather_started = time.perf_counter()
        matched = np.zeros(len(keys), dtype=np.bool_)
        row_ids = np.zeros(len(keys), dtype=np.uint64)
        for gather_map, lower, upper, mapped_rows in located:
            probe_indices = gather_map.to_arrow().to_numpy(zero_copy_only=False)
            lower_indices = lower.to_arrow().to_numpy(zero_copy_only=False)
            upper_indices = upper.to_arrow().to_numpy(zero_copy_only=False)
            if np.any(upper_indices - lower_indices != 1):
                msg = "Reference key index contains duplicate keys within one segment"
                raise ValueError(msg)
            if np.any(matched[probe_indices]):
                msg = "Reference key index contains duplicate keys across segments"
                raise ValueError(msg)
            mapped_ids = mapped_rows.columns()[0].to_arrow().to_numpy(zero_copy_only=False)
            matched[probe_indices] = True
            row_ids[probe_indices] = mapped_ids
        gather_seconds = time.perf_counter() - gather_started
        return _GpuMapResult(
            matched,
            row_ids,
            transfer_seconds,
            probe_seconds,
            search_seconds,
            gather_seconds,
        )

    def close(self) -> None:
        self._joins.clear()
        self._row_id_tables.clear()
        self._key_tables.clear()
        self._frames.clear()


@dataclass
class GpuExactKeyLookupStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Add exact-key presence to an ``InterleavedBatch`` using one GPU.

    Input data consists of two independent pieces:

    * Every task is an ``InterleavedBatch`` containing ``input_key_column``.
      For MINT-1T HTML, image rows store their exact image URL in
      ``source_ref`` while text and metadata rows store null.
    * ``reference_files`` is an immutable collection of Parquet files whose
      ``reference_key_column`` values form the exact membership set. For MINT,
      these are the URLs present in the pinned image Lance table.

    The output is a new ``InterleavedBatch`` with the same rows, order,
    metadata, and unrelated columns, plus a nullable boolean
    ``presence_column``. A non-null, non-empty input key maps to ``True`` when
    it exists in any reference segment and ``False`` otherwise. Null and empty
    string keys are not queried and map to null presence.

    Each actor loads the reference segments once in ``setup()`` and builds one
    persistent ``pylibcudf.join.FilteredJoin`` per segment. Multiple Curator
    tasks can be coalesced with ``ProcessingStage.with_(batch_size=...)`` and
    are probed as one Arrow array; task boundaries and row order are restored
    before returning. The stage performs membership only: it does not fetch
    payload columns, mutate the reference dataset, or persist reference row
    IDs.
    """

    reference_files: list[str]
    reference_key_column: str
    input_key_column: str = "source_ref"
    presence_column: str = "image_present"
    storage_options: dict[str, str] = field(default_factory=dict)
    expected_reference_rows: int | None = None
    load_factor: float = 0.5
    name: str = "gpu_exact_key_lookup"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    _matcher: _GpuExactKeyMatcher | None = field(default=None, init=False, repr=False)
    _setup_metrics_pending: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reference_files = list(self.reference_files)
        self.storage_options = dict(self.storage_options or {})
        if not self.reference_files:
            msg = "reference_files must not be empty"
            raise ValueError(msg)
        if len(set(self.reference_files)) != len(self.reference_files):
            msg = "reference_files must not contain duplicates"
            raise ValueError(msg)
        if not self.reference_key_column or not self.input_key_column or not self.presence_column:
            msg = "reference_key_column, input_key_column, and presence_column must not be empty"
            raise ValueError(msg)
        if self.expected_reference_rows is not None and self.expected_reference_rows <= 0:
            msg = "expected_reference_rows must be greater than zero"
            raise ValueError(msg)
        if not 0.0 < self.load_factor <= 1.0:
            msg = "load_factor must be in the interval (0, 1]"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_key_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.presence_column]

    def setup(self, _worker_metadata: object | None = None) -> None:
        self._matcher = _GpuExactKeyMatcher(
            self.reference_files,
            self.reference_key_column,
            self.storage_options,
            self.expected_reference_rows,
            self.load_factor,
        )
        self._setup_metrics_pending = True

    def teardown(self) -> None:
        if self._matcher is not None:
            self._matcher.close()
            self._matcher = None

    def _ensure_matcher(self) -> _GpuExactKeyMatcher:
        if self._matcher is None:
            self.setup()
        if self._matcher is None:  # pragma: no cover - setup returns a matcher or raises
            msg = "GPU exact-key matcher setup did not initialize the worker"
            raise RuntimeError(msg)
        return self._matcher

    def _validate_table(self, table: pa.Table, reference_type: pa.DataType) -> None:
        if self.input_key_column not in table.column_names:
            msg = f"Input key column {self.input_key_column!r} does not exist"
            raise ValueError(msg)
        if self.presence_column in table.column_names:
            msg = f"Presence column {self.presence_column!r} already exists"
            raise ValueError(msg)
        input_type = table.schema.field(self.input_key_column).type
        both_string = (pa.types.is_string(input_type) or pa.types.is_large_string(input_type)) and (
            pa.types.is_string(reference_type) or pa.types.is_large_string(reference_type)
        )
        if input_type != reference_type and not both_string:
            msg = f"Input key column has type {input_type}; reference key column has type {reference_type}"
            raise TypeError(msg)

    @staticmethod
    def _eligible_mask(keys: pa.Array) -> pa.BooleanArray:
        eligible = pc.is_valid(keys)
        if pa.types.is_string(keys.type) or pa.types.is_large_string(keys.type):
            eligible = pc.and_kleene(eligible, pc.not_equal(keys, ""))
        return pc.fill_null(eligible, False)

    def _process_tasks(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        if len(tasks) == 0:
            return []
        matcher = self._ensure_matcher()

        tables = [task.to_pyarrow() for task in tasks]
        for table in tables:
            self._validate_table(table, matcher.reference_type)
        key_arrays = [table[self.input_key_column].combine_chunks() for table in tables]
        combined_keys = pa.concat_arrays(key_arrays)
        eligible = self._eligible_mask(combined_keys)
        eligible_indices = pc.indices_nonzero(eligible)
        eligible_keys = pc.take(combined_keys, eligible_indices)

        match_result = matcher.match(eligible_keys)
        presence_values = np.zeros(len(combined_keys), dtype=np.bool_)
        presence_valid = eligible.to_numpy(zero_copy_only=False)
        presence_values[eligible_indices.to_numpy(zero_copy_only=False)] = match_result.matched
        presence = pa.array(presence_values, mask=~presence_valid, type=pa.bool_())

        outputs: list[InterleavedBatch] = []
        offset = 0
        for task, table in zip(tasks, tables, strict=True):
            task_presence = presence.slice(offset, table.num_rows)
            result = table.append_column(self.presence_column, task_presence)
            outputs.append(
                InterleavedBatch(
                    dataset_name=task.dataset_name,
                    data=result,
                    _metadata=task._metadata,
                    _stage_perf=task._stage_perf,
                )
            )
            offset += table.num_rows

        found = int(match_result.matched.sum())
        metrics = {
            "input_tasks": float(len(tasks)),
            "input_rows": float(len(combined_keys)),
            "eligible_keys": float(len(eligible_keys)),
            "found_keys": float(found),
            "missing_keys": float(len(eligible_keys) - found),
            "gpu_key_transfer_seconds": match_result.transfer_seconds,
            "gpu_key_probe_seconds": match_result.probe_seconds,
            "gpu_result_gather_seconds": match_result.gather_seconds,
            "peak_rss_bytes": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        }
        if self._setup_metrics_pending:
            metrics.update(
                {
                    "reference_rows": float(matcher.reference_rows),
                    "gpu_reference_load_seconds": matcher.load_seconds,
                    "gpu_hash_build_seconds": matcher.build_seconds,
                    "gpu_reference_bytes": float(matcher.gpu_bytes),
                    "gpu_total_bytes": float(matcher.gpu_total_bytes),
                }
            )
            self._setup_metrics_pending = False
        self._log_metrics(metrics)
        return outputs

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return self._process_tasks([task])[0]

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        """Probe one coalesced key array and preserve task boundaries."""
        return self._process_tasks(tasks)
