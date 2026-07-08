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

"""Indexed Lance column fetches for row-wise interleaved batches."""

from __future__ import annotations

import hashlib
import json
import operator
import os
import resource
import shutil
import tempfile
import time
from bisect import bisect_right
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage, LanceReaderStage, LanceReadTask
from nemo_curator.tasks import EmptyTask, InterleavedBatch
from nemo_curator.utils.uri import validate_credential_free_uri_identity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata
    from nemo_curator.stages.text.io.reader.base import ReaderOutput

_ROW_ID_COLUMN = "_rowid"
_NODE_READY_FILE = ".nemo_curator_lance_index_ready.json"
_MIRROR_CONTRACT_FORMAT = "nemo-curator-lance-index-mirror-v1"
_KEY_STABLE_ORDINAL_FORMAT = "nemo-curator-key-stable-ordinal-v1"
_MIRROR_IDENTITY_BATCH_ROWS = 65_536
_SHA256_HEX_LENGTH = 64

ExistingColumnPolicy = Literal["error", "fill_null", "overwrite"]
MissingKeyPolicy = Literal["mark", "error"]
PayloadReadMode = Literal["sparse", "adaptive_unmeasured"]
PrivateReadStrategy = Literal["take_rows", "take_scan_ranges", "take_scan_fragment"]

_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")


@dataclass(frozen=True)
class LanceDatasetConfig:
    """Identity and indexed-key configuration for a pinned Lance dataset."""

    uri: str
    version: int
    key_column: str
    index_name: str
    storage_options: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uri:
            msg = "uri must not be empty"
            raise ValueError(msg)
        validate_credential_free_uri_identity(self.uri, "Lance dataset URI")
        if self.version <= 0:
            msg = "version must be greater than 0"
            raise ValueError(msg)
        if not self.key_column:
            msg = "key_column must not be empty"
            raise ValueError(msg)
        if not self.index_name:
            msg = "index_name must not be empty"
            raise ValueError(msg)
        object.__setattr__(self, "storage_options", dict(self.storage_options or {}))


@dataclass(frozen=True)
class LanceIndexMirrorContract:
    """Caller-pinned identity for an exact local mirror of a Lance index."""

    remote_uri: str
    remote_version: int
    remote_fragment_manifest_sha256: str
    mirror_uri: str
    mirror_version: int
    key_column: str
    key_stable_ordinal_sha256: str
    index_name: str
    index_artifacts_sha256: str

    def __post_init__(self) -> None:
        for name in ("remote_uri", "mirror_uri", "key_column", "index_name"):
            if not getattr(self, name):
                msg = f"{name} must not be empty"
                raise ValueError(msg)
        validate_credential_free_uri_identity(self.remote_uri, "remote Lance URI")
        validate_credential_free_uri_identity(self.mirror_uri, "mirror Lance URI")
        for name in ("remote_version", "mirror_version"):
            if getattr(self, name) <= 0:
                msg = f"{name} must be greater than 0"
                raise ValueError(msg)
        for name in (
            "remote_fragment_manifest_sha256",
            "key_stable_ordinal_sha256",
            "index_artifacts_sha256",
        ):
            value = getattr(self, name)
            if len(value) != _SHA256_HEX_LENGTH or any(character not in "0123456789abcdef" for character in value):
                msg = f"{name} must be a lowercase SHA-256 digest"
                raise ValueError(msg)

    def as_dict(self) -> dict[str, object]:
        """Return the canonical, secret-free contract representation."""
        return {
            "format": _MIRROR_CONTRACT_FORMAT,
            "remote_uri": self.remote_uri,
            "remote_version": self.remote_version,
            "remote_fragment_manifest_sha256": self.remote_fragment_manifest_sha256,
            "mirror_uri": self.mirror_uri,
            "mirror_version": self.mirror_version,
            "key_column": self.key_column,
            "key_stable_ordinal_sha256": self.key_stable_ordinal_sha256,
            "index_name": self.index_name,
            "index_artifacts_sha256": self.index_artifacts_sha256,
        }

    def identity_sha256(self) -> str:
        """Hash every pinned field for cache paths and ready markers."""
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def validate_for(self, dataset: LanceDatasetConfig, mirror_path: str) -> None:
        """Fail when a contract is applied to any other dataset or mirror."""
        expected = {
            "remote_uri": dataset.uri,
            "remote_version": dataset.version,
            "mirror_uri": mirror_path,
            "key_column": dataset.key_column,
            "index_name": dataset.index_name,
        }
        mismatches = {
            name: (getattr(self, name), value) for name, value in expected.items() if getattr(self, name) != value
        }
        if mismatches:
            msg = f"Lance index mirror contract does not match the requested dataset: {mismatches}"
            raise ValueError(msg)


@dataclass(frozen=True)
class LanceIndexCacheConfig:
    """Worker cache and optional contract-pinned local Lance index mirror."""

    mirror_path: str | None = None
    mirror_contract: LanceIndexMirrorContract | None = None
    copy_to_node_local: bool = False
    node_local_root: str = "/local/lance-indexes"
    prewarm: bool = True
    index_cache_size_bytes: int = 32 * 1024**3
    metadata_cache_size_bytes: int = 1024**3

    def __post_init__(self) -> None:
        if bool(self.mirror_path) != bool(self.mirror_contract):
            msg = "mirror_path and mirror_contract must be configured together"
            raise ValueError(msg)
        if self.mirror_contract is not None and self.mirror_contract.mirror_uri != self.mirror_path:
            msg = "mirror_contract.mirror_uri must exactly match mirror_path"
            raise ValueError(msg)
        if self.copy_to_node_local and self.mirror_contract is None:
            msg = "copy_to_node_local requires a contract-pinned mirror"
            raise ValueError(msg)
        if self.copy_to_node_local and not self.node_local_root:
            msg = "node_local_root must not be empty"
            raise ValueError(msg)
        if self.index_cache_size_bytes <= 0:
            msg = "index_cache_size_bytes must be greater than 0"
            raise ValueError(msg)
        if self.metadata_cache_size_bytes <= 0:
            msg = "metadata_cache_size_bytes must be greater than 0"
            raise ValueError(msg)

    def node_local_path(self, dataset: LanceDatasetConfig) -> Path:
        if self.mirror_contract is None or self.mirror_path is None:
            msg = "node_local_path requires a contract-pinned mirror"
            raise ValueError(msg)
        self.mirror_contract.validate_for(dataset, self.mirror_path)
        identity = (
            f"{dataset.uri}\n{dataset.version}\n{dataset.index_name}\n{self.mirror_contract.identity_sha256()}"
        ).encode()
        cache_id = hashlib.sha256(identity).hexdigest()[:24]
        return Path(self.node_local_root) / cache_id / "dataset"

    def resolved_index_uri(self, dataset: LanceDatasetConfig) -> str:
        if self.copy_to_node_local:
            return str(self.node_local_path(dataset))
        return self.mirror_path or dataset.uri

    def resolved_index_version(self, dataset: LanceDatasetConfig) -> int:
        if self.mirror_contract is not None:
            return self.mirror_contract.mirror_version
        return dataset.version


@dataclass
class _FetchResult:
    key_to_row_id: dict[object, int]
    payload: pa.Table
    payload_row_ids: pa.Array
    lookup_seconds: float
    fetch_seconds: float
    fetched_bytes_by_column: dict[str, int]
    read_bytes: int = 0
    read_iops: int = 0
    lookup_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreparedFetchTask:
    task: InterleavedBatch
    table: pa.Table
    keys: list[object]
    requested_keys: list[object]
    requested_mask: pa.Array


class _FragmentMetadataProtocol(Protocol):
    physical_rows: int
    deletion_file: object | None

    def to_json(self) -> dict[str, object]: ...


class _FragmentProtocol(Protocol):
    fragment_id: int
    physical_rows: int
    num_deletions: int
    metadata: _FragmentMetadataProtocol

    def deletion_file(self) -> object | None: ...


class _ManifestDatasetProtocol(Protocol):
    def get_fragments(self) -> Iterable[_FragmentProtocol]: ...

    def count_rows(self) -> int: ...


class _MirrorScannerProtocol(Protocol):
    def to_batches(self) -> Iterable[pa.RecordBatch]: ...


class _IOStatsProtocol(Protocol):
    read_bytes: int
    read_iops: int


class _IOStatsDatasetProtocol(Protocol):
    def io_stats_incremental(self) -> _IOStatsProtocol: ...


class _IndexDescriptionProtocol(Protocol):
    name: str
    field_names: Sequence[str]


class _MirrorDatasetProtocol(_ManifestDatasetProtocol, Protocol):
    schema: pa.Schema
    has_stable_row_ids: bool

    def scanner(self, **kwargs: object) -> _MirrorScannerProtocol: ...

    def describe_indices(self) -> Iterable[_IndexDescriptionProtocol]: ...


@dataclass(frozen=True)
class _StableGlobalOrdinalManifest:
    """Validated append-only manifest backing stable global ordinals."""

    fragment_starts: tuple[int, ...]
    fragment_rows: tuple[int, ...]
    total_rows: int


@dataclass(frozen=True)
class _PrivateTakePlan:
    """Pure sparse plan for sorted global ordinals and bounded private takes."""

    row_ids: tuple[int, ...]
    batches: tuple[tuple[int, ...], ...]
    coordinate_density: float


@dataclass(frozen=True)
class _PrivateReadOperation:
    """One private payload call and the requested IDs it must return."""

    strategy: PrivateReadStrategy
    row_ids: tuple[int, ...]
    ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _PrivateReadBatchResult:
    """Projected Arrow rows paired with their stable global ordinals."""

    table: pa.Table
    row_ids: tuple[int, ...]


@dataclass(frozen=True)
class _AdaptiveLocalityPlan:
    """Executable private-read plan with per-fragment strategy telemetry."""

    row_ids: tuple[int, ...]
    operations: tuple[_PrivateReadOperation, ...]
    coordinate_density: float
    sparse_fragments: int
    range_fragments: int
    sequential_fragments: int
    take_scan_ranges: int
    planned_scan_rows: int
    range_overread_rows: int


def _require_integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        msg = f"{label} must be an integer, got bool"
        raise TypeError(msg)
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        msg = f"{label} must be an integer, got {type(value).__name__}"
        raise TypeError(msg) from exc
    if positive and resolved <= 0:
        msg = f"{label} must be greater than zero, got {resolved}"
        raise ValueError(msg)
    return resolved


def _validate_stable_global_ordinal_manifest(
    dataset: _ManifestDatasetProtocol,
) -> _StableGlobalOrdinalManifest:
    """Validate the one supported stable global-ordinal dataset contract."""
    fragments = list(dataset.get_fragments())
    if not fragments:
        msg = "Stable global-ordinal fetches require at least one Lance fragment"
        raise ValueError(msg)

    fragment_starts: list[int] = []
    fragment_rows: list[int] = []
    physical_rows_total = 0
    metadata_rows_total = 0
    for position, fragment in enumerate(fragments):
        fragment_id = _require_integer(fragment.fragment_id, f"fragment {position} ID")
        if fragment_id != position:
            msg = (
                "Stable global-ordinal fetches require contiguous manifest-order fragment IDs; "
                f"position {position} has fragment ID {fragment_id}"
            )
            raise ValueError(msg)

        physical_rows = _require_integer(
            fragment.physical_rows,
            f"fragment {fragment_id} physical_rows",
            positive=True,
        )
        metadata_physical_rows = _require_integer(
            fragment.metadata.physical_rows,
            f"fragment {fragment_id} metadata physical_rows",
            positive=True,
        )
        if metadata_physical_rows != physical_rows:
            msg = (
                f"Fragment {fragment_id} reports physical_rows={physical_rows}, "
                f"metadata physical_rows={metadata_physical_rows}"
            )
            raise ValueError(msg)

        num_deletions = _require_integer(fragment.num_deletions, f"fragment {fragment_id} num_deletions")
        if num_deletions != 0 or fragment.deletion_file() is not None or fragment.metadata.deletion_file is not None:
            msg = (
                "Stable global-ordinal fetches require an append-only dataset without deletions; "
                f"fragment {fragment_id} has num_deletions={num_deletions}"
            )
            raise ValueError(msg)

        fragment_starts.append(physical_rows_total)
        fragment_rows.append(physical_rows)
        physical_rows_total += physical_rows
        metadata_rows_total += metadata_physical_rows

    dataset_rows = _require_integer(dataset.count_rows(), "Lance dataset row count", positive=True)
    if physical_rows_total != metadata_rows_total or physical_rows_total != dataset_rows:
        msg = (
            "Stable global-ordinal fetches require complete physical-row coverage; "
            f"physical_rows={physical_rows_total}, metadata_rows={metadata_rows_total}, "
            f"dataset_rows={dataset_rows}"
        )
        raise ValueError(msg)
    return _StableGlobalOrdinalManifest(
        fragment_starts=tuple(fragment_starts),
        fragment_rows=tuple(fragment_rows),
        total_rows=dataset_rows,
    )


def _exact_fragment_manifest_sha256(dataset: _MirrorDatasetProtocol) -> str:
    """Hash ordered fragment metadata after validating stable ordinals."""
    manifest = _validate_stable_global_ordinal_manifest(dataset)
    fragments: list[dict[str, object]] = []
    for fragment in dataset.get_fragments():
        metadata = fragment.metadata.to_json()
        if not isinstance(metadata, dict):
            msg = "Lance fragment metadata must serialize to a JSON object"
            raise TypeError(msg)
        fragments.append(metadata)
    encoded = json.dumps(
        {
            "format": _MIRROR_CONTRACT_FORMAT,
            "fragment_starts": manifest.fragment_starts,
            "fragment_rows": manifest.fragment_rows,
            "total_rows": manifest.total_rows,
            "fragments": fragments,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _key_stable_ordinal_sha256(
    dataset: _MirrorDatasetProtocol,
    key_column: str,
    *,
    total_rows: int,
) -> str:
    """Hash the logical Arrow stream of key-to-stable-ordinal assignments."""
    identity_schema = pa.schema(
        [
            pa.field("stable_global_ordinal", pa.uint64(), nullable=False),
            dataset.schema.field(key_column),
        ]
    )
    digest = hashlib.sha256()
    digest.update(_KEY_STABLE_ORDINAL_FORMAT.encode())
    schema_bytes = identity_schema.serialize().to_pybytes()
    digest.update(len(schema_bytes).to_bytes(8, "little"))
    digest.update(schema_bytes)

    offset = 0
    scanner = dataset.scanner(
        columns=[key_column],
        with_row_id=True,
        scan_in_order=True,
        batch_size=_MIRROR_IDENTITY_BATCH_ROWS,
        strict_batch_size=True,
    )
    for batch in scanner.to_batches():
        if _ROW_ID_COLUMN not in batch.schema.names:
            msg = "Lance mirror identity scan did not return stable row IDs"
            raise RuntimeError(msg)
        row_ids = batch.column(batch.schema.get_field_index(_ROW_ID_COLUMN)).cast(pa.uint64())
        expected_ids = pa.array(range(offset, offset + batch.num_rows), type=pa.uint64())
        if not bool(pc.all(pc.equal(row_ids, expected_ids)).as_py()):
            msg = f"Lance mirror identity scan is not in stable global-ordinal order at row {offset}"
            raise ValueError(msg)
        keys = batch.column(batch.schema.get_field_index(key_column))
        identity_batch = pa.RecordBatch.from_arrays([row_ids, keys], schema=identity_schema)
        batch_bytes = identity_batch.serialize().to_pybytes()
        digest.update(batch.num_rows.to_bytes(8, "little"))
        digest.update(len(batch_bytes).to_bytes(8, "little"))
        digest.update(batch_bytes)
        offset += batch.num_rows
    if offset != total_rows:
        msg = f"Lance mirror identity scan returned {offset} rows; expected {total_rows}"
        raise ValueError(msg)
    return digest.hexdigest()


def _local_index_artifacts_sha256(dataset_uri: str) -> str:
    """Hash every physical index artifact in a local Lance mirror."""
    if "://" in dataset_uri:
        msg = "Contract-pinned Lance index mirrors must use a local filesystem path"
        raise ValueError(msg)
    index_root = Path(dataset_uri) / "_indices"
    if not index_root.is_dir():
        msg = f"Lance index artifact directory does not exist: {index_root}"
        raise FileNotFoundError(msg)
    files = sorted(path for path in index_root.rglob("*") if path.is_file())
    if not files:
        msg = f"Lance index artifact directory is empty: {index_root}"
        raise ValueError(msg)

    digest = hashlib.sha256()
    digest.update(_MIRROR_CONTRACT_FORMAT.encode())
    for path in files:
        if path.is_symlink():
            msg = f"Lance index artifacts must not contain symlinks: {path}"
            raise ValueError(msg)
        relative_path = path.relative_to(index_root).as_posix().encode()
        size = path.stat().st_size
        digest.update(len(relative_path).to_bytes(8, "little"))
        digest.update(relative_path)
        digest.update(size.to_bytes(8, "little"))
        bytes_read = 0
        with path.open("rb") as artifact:
            while chunk := artifact.read(8 * 1024**2):
                digest.update(chunk)
                bytes_read += len(chunk)
        if bytes_read != size:
            msg = f"Lance index artifact changed while hashing: {path}"
            raise RuntimeError(msg)
    return digest.hexdigest()


def _validate_mirror_index_surface(
    remote_dataset: _MirrorDatasetProtocol,
    mirror_dataset: _MirrorDatasetProtocol,
    dataset_config: LanceDatasetConfig,
) -> None:
    key_column = dataset_config.key_column
    if key_column not in mirror_dataset.schema.names:
        msg = f"Lance key column {key_column!r} is missing from the index mirror"
        raise ValueError(msg)
    if mirror_dataset.schema.field(key_column).type != remote_dataset.schema.field(key_column).type:
        msg = "Index mirror and remote Lance key column types do not match"
        raise TypeError(msg)
    indices = {index.name: index for index in mirror_dataset.describe_indices()}
    index = indices.get(dataset_config.index_name)
    if index is None:
        msg = f"Lance index {dataset_config.index_name!r} does not exist in the mirror"
        raise ValueError(msg)
    if key_column not in index.field_names:
        msg = f"Lance index {dataset_config.index_name!r} does not cover {key_column!r}"
        raise ValueError(msg)
    if not mirror_dataset.has_stable_row_ids:
        msg = "LanceColumnFetchStage requires stable row IDs in its index mirror"
        raise ValueError(msg)


def build_lance_index_mirror_contract(
    dataset: LanceDatasetConfig,
    *,
    mirror_uri: str,
    mirror_version: int,
) -> LanceIndexMirrorContract:
    """Build an offline contract after proving an exact local mirror identity."""
    import lance

    if mirror_version <= 0:
        msg = "mirror_version must be greater than 0"
        raise ValueError(msg)
    remote_dataset = lance.dataset(
        dataset.uri,
        version=dataset.version,
        storage_options=dataset.storage_options or None,
    )
    mirror_dataset = lance.dataset(mirror_uri, version=mirror_version)
    _validate_mirror_index_surface(remote_dataset, mirror_dataset, dataset)
    remote_manifest = _validate_stable_global_ordinal_manifest(remote_dataset)
    mirror_manifest = _validate_stable_global_ordinal_manifest(mirror_dataset)
    remote_fragment_digest = _exact_fragment_manifest_sha256(remote_dataset)
    mirror_fragment_digest = _exact_fragment_manifest_sha256(mirror_dataset)
    if mirror_fragment_digest != remote_fragment_digest:
        msg = "Lance index mirror does not preserve the remote exact fragment manifest"
        raise ValueError(msg)
    remote_key_digest = _key_stable_ordinal_sha256(
        remote_dataset,
        dataset.key_column,
        total_rows=remote_manifest.total_rows,
    )
    mirror_key_digest = _key_stable_ordinal_sha256(
        mirror_dataset,
        dataset.key_column,
        total_rows=mirror_manifest.total_rows,
    )
    if mirror_key_digest != remote_key_digest:
        msg = "Lance index mirror key-to-stable-ordinal identity does not match the remote dataset"
        raise ValueError(msg)
    return LanceIndexMirrorContract(
        remote_uri=dataset.uri,
        remote_version=dataset.version,
        remote_fragment_manifest_sha256=remote_fragment_digest,
        mirror_uri=mirror_uri,
        mirror_version=mirror_version,
        key_column=dataset.key_column,
        key_stable_ordinal_sha256=remote_key_digest,
        index_name=dataset.index_name,
        index_artifacts_sha256=_local_index_artifacts_sha256(mirror_uri),
    )


def _validate_index_mirror_contract(
    dataset_config: LanceDatasetConfig,
    index_cache: LanceIndexCacheConfig,
    remote_dataset: _MirrorDatasetProtocol,
    mirror_dataset: _MirrorDatasetProtocol,
    resolved_mirror_uri: str,
) -> None:
    """Verify every pinned mirror field before prewarm or key lookup."""
    contract = index_cache.mirror_contract
    mirror_path = index_cache.mirror_path
    if contract is None or mirror_path is None:
        return
    contract.validate_for(dataset_config, mirror_path)
    remote_fragment_digest = _exact_fragment_manifest_sha256(remote_dataset)
    if remote_fragment_digest != contract.remote_fragment_manifest_sha256:
        msg = "Remote Lance exact fragment manifest does not match the pinned mirror contract"
        raise ValueError(msg)
    mirror_manifest = _validate_stable_global_ordinal_manifest(mirror_dataset)
    mirror_fragment_digest = _exact_fragment_manifest_sha256(mirror_dataset)
    if mirror_fragment_digest != contract.remote_fragment_manifest_sha256:
        msg = "Lance index mirror does not preserve the pinned remote exact fragment manifest"
        raise ValueError(msg)
    mirror_key_digest = _key_stable_ordinal_sha256(
        mirror_dataset,
        dataset_config.key_column,
        total_rows=mirror_manifest.total_rows,
    )
    if mirror_key_digest != contract.key_stable_ordinal_sha256:
        msg = "Lance index mirror key-to-stable-ordinal identity does not match its pinned contract"
        raise ValueError(msg)
    artifact_digest = _local_index_artifacts_sha256(resolved_mirror_uri)
    if artifact_digest != contract.index_artifacts_sha256:
        msg = "Lance index mirror artifacts do not match their pinned contract"
        raise ValueError(msg)


def _node_ready_payload(
    dataset: LanceDatasetConfig,
    index_cache: LanceIndexCacheConfig,
) -> dict[str, object]:
    contract = index_cache.mirror_contract
    mirror_path = index_cache.mirror_path
    if contract is None or mirror_path is None:
        msg = "Node-local Lance mirrors require a pinned mirror contract"
        raise ValueError(msg)
    contract.validate_for(dataset, mirror_path)
    return {
        "format": _MIRROR_CONTRACT_FORMAT,
        "remote_uri": dataset.uri,
        "remote_version": dataset.version,
        "mirror_uri": mirror_path,
        "mirror_version": contract.mirror_version,
        "mirror_contract_sha256": contract.identity_sha256(),
        "mirror_contract": contract.as_dict(),
    }


def _require_node_ready_marker(
    ready: Path,
    dataset: LanceDatasetConfig,
    index_cache: LanceIndexCacheConfig,
) -> None:
    expected = _node_ready_payload(dataset, index_cache)
    try:
        actual = json.loads(ready.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        msg = f"Node-local Lance index mirror is not ready: {ready.parent}"
        raise RuntimeError(msg) from exc
    if actual != expected:
        msg = f"Node-local Lance index mirror has a stale ready marker: {ready}"
        raise RuntimeError(msg)


def _plan_private_takes(
    row_ids: Sequence[object],
    *,
    total_rows: int,
    fetch_batch_size: int,
) -> _PrivateTakePlan:
    """Sort, deduplicate, validate, and batch stable global ordinals."""
    # Keep this measured default independent from the opt-in adaptive planner.
    if fetch_batch_size <= 0:
        msg = "fetch_batch_size must be greater than 0"
        raise ValueError(msg)
    if total_rows <= 0:
        msg = "total_rows must be greater than 0"
        raise ValueError(msg)

    unique_row_ids: set[int] = set()
    for raw_row_id in row_ids:
        row_id = _require_integer(raw_row_id, "stable row ID")
        if row_id < 0 or row_id >= total_rows:
            msg = f"Stable row ID {row_id} is outside global-ordinal range [0, {total_rows})"
            raise ValueError(msg)
        unique_row_ids.add(row_id)
    sorted_row_ids = tuple(sorted(unique_row_ids))
    batches = tuple(
        sorted_row_ids[start : start + fetch_batch_size] for start in range(0, len(sorted_row_ids), fetch_batch_size)
    )
    return _PrivateTakePlan(
        row_ids=sorted_row_ids,
        batches=batches,
        coordinate_density=len(sorted_row_ids) / total_rows,
    )


def _coalesce_ordinal_ranges(
    row_ids: Sequence[int],
    *,
    max_gap_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Coalesce sorted global ordinals into half-open private scan ranges."""
    if max_gap_rows < 0:
        msg = "max_gap_rows must be nonnegative"
        raise ValueError(msg)
    if not row_ids:
        return ()

    ranges: list[tuple[int, int]] = []
    start = row_ids[0]
    stop = start + 1
    for row_id in row_ids[1:]:
        if row_id - stop <= max_gap_rows:
            stop = row_id + 1
        else:
            ranges.append((start, stop))
            start = row_id
            stop = row_id + 1
    ranges.append((start, stop))
    return tuple(ranges)


def _validate_adaptive_locality_config(
    *,
    payload_read_mode: PayloadReadMode,
    medium_density_threshold: float,
    high_density_threshold: float,
    max_coalesced_range_gap: int,
    take_scan_batch_readahead: int,
) -> None:
    """Validate the explicitly provisional adaptive locality controls."""
    if payload_read_mode not in {"sparse", "adaptive_unmeasured"}:
        msg = f"Unsupported payload_read_mode: {payload_read_mode}"
        raise ValueError(msg)
    if not 0.0 < medium_density_threshold < high_density_threshold <= 1.0:
        msg = "Density thresholds must satisfy 0 < medium < high <= 1"
        raise ValueError(msg)
    if max_coalesced_range_gap < 0:
        msg = "max_coalesced_range_gap must be nonnegative"
        raise ValueError(msg)
    if take_scan_batch_readahead <= 0:
        msg = "take_scan_batch_readahead must be greater than 0"
        raise ValueError(msg)


def _plan_adaptive_locality_reads(  # noqa: PLR0913
    row_ids: Sequence[object],
    *,
    manifest: _StableGlobalOrdinalManifest,
    fetch_batch_size: int,
    payload_read_mode: PayloadReadMode,
    medium_density_threshold: float,
    high_density_threshold: float,
    max_coalesced_range_gap: int,
) -> _AdaptiveLocalityPlan:
    """Plan sparse takes or opt-in, unmeasured per-fragment range scans."""
    sparse_plan = _plan_private_takes(
        row_ids,
        total_rows=manifest.total_rows,
        fetch_batch_size=fetch_batch_size,
    )
    grouped: dict[int, list[int]] = {}
    for row_id in sparse_plan.row_ids:
        fragment_index = bisect_right(manifest.fragment_starts, row_id) - 1
        if fragment_index < 0:
            msg = f"Stable row ID {row_id} does not map to a manifest fragment"
            raise ValueError(msg)
        grouped.setdefault(fragment_index, []).append(row_id)

    if payload_read_mode == "sparse":
        operations = tuple(_PrivateReadOperation(strategy="take_rows", row_ids=batch) for batch in sparse_plan.batches)
        return _AdaptiveLocalityPlan(
            row_ids=sparse_plan.row_ids,
            operations=operations,
            coordinate_density=sparse_plan.coordinate_density,
            sparse_fragments=len(grouped),
            range_fragments=0,
            sequential_fragments=0,
            take_scan_ranges=0,
            planned_scan_rows=0,
            range_overread_rows=0,
        )

    operations_list: list[_PrivateReadOperation] = []
    sparse_fragments = 0
    range_fragments = 0
    sequential_fragments = 0
    take_scan_ranges = 0
    planned_scan_rows = 0
    range_requested_rows = 0
    for fragment_index, fragment_row_ids in sorted(grouped.items()):
        fragment_rows = manifest.fragment_rows[fragment_index]
        fragment_start = manifest.fragment_starts[fragment_index]
        fragment_stop = fragment_start + fragment_rows
        density = len(fragment_row_ids) / fragment_rows
        requested = tuple(fragment_row_ids)
        if density >= high_density_threshold:
            ranges = ((fragment_start, fragment_stop),)
            operations_list.append(
                _PrivateReadOperation(
                    strategy="take_scan_fragment",
                    row_ids=requested,
                    ranges=ranges,
                )
            )
            sequential_fragments += 1
        elif density >= medium_density_threshold:
            ranges = _coalesce_ordinal_ranges(
                fragment_row_ids,
                max_gap_rows=max_coalesced_range_gap,
            )
            operations_list.append(
                _PrivateReadOperation(
                    strategy="take_scan_ranges",
                    row_ids=requested,
                    ranges=ranges,
                )
            )
            range_fragments += 1
        else:
            operations_list.extend(
                _PrivateReadOperation(strategy="take_rows", row_ids=requested[start : start + fetch_batch_size])
                for start in range(0, len(requested), fetch_batch_size)
            )
            sparse_fragments += 1
            continue
        take_scan_ranges += len(ranges)
        planned_scan_rows += sum(stop - start for start, stop in ranges)
        range_requested_rows += len(requested)

    return _AdaptiveLocalityPlan(
        row_ids=sparse_plan.row_ids,
        operations=tuple(operations_list),
        coordinate_density=sparse_plan.coordinate_density,
        sparse_fragments=sparse_fragments,
        range_fragments=range_fragments,
        sequential_fragments=sequential_fragments,
        take_scan_ranges=take_scan_ranges,
        planned_scan_rows=planned_scan_rows,
        range_overread_rows=planned_scan_rows - range_requested_rows,
    )


def _bounded_parallel_map(
    executor: ThreadPoolExecutor,
    function: Callable[[_InputT], _OutputT],
    items: Sequence[_InputT],
    max_pending: int,
) -> tuple[list[_OutputT], int]:
    """Map in input order while bounding all submitted, unfinished work."""
    if max_pending <= 0:
        msg = "max_pending must be greater than 0"
        raise ValueError(msg)

    pending: dict[Future[_OutputT], int] = {}
    completed: dict[int, _OutputT] = {}
    next_index = 0
    peak_pending = 0

    def fill_pending() -> None:
        nonlocal next_index, peak_pending
        while next_index < len(items) and len(pending) < max_pending:
            pending[executor.submit(function, items[next_index])] = next_index
            next_index += 1
        peak_pending = max(peak_pending, len(pending))

    fill_pending()
    try:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed[pending.pop(future)] = future.result()
            fill_pending()
    except Exception:
        for future in pending:
            future.cancel()
        raise
    return [completed[index] for index in range(len(items))], peak_pending


class _LancePayloadFetcher:
    """Persistent worker-local executor for private stable-ordinal takes."""

    def __init__(  # noqa: PLR0913
        self,
        dataset_config: LanceDatasetConfig,
        index_cache: LanceIndexCacheConfig,
        columns: dict[str, str],
        fetch_batch_size: int,
        max_pending_takes: int,
        payload_read_mode: PayloadReadMode,
        medium_density_threshold: float,
        high_density_threshold: float,
        max_coalesced_range_gap: int,
        take_scan_batch_readahead: int,
        validate_payload_keys: bool,
    ) -> None:
        import lance

        self.config = dataset_config
        self.index_cache = index_cache
        self.columns = columns
        self.fetch_batch_size = fetch_batch_size
        self.max_pending_takes = max_pending_takes
        self.payload_read_mode = payload_read_mode
        self.medium_density_threshold = medium_density_threshold
        self.high_density_threshold = high_density_threshold
        self.max_coalesced_range_gap = max_coalesced_range_gap
        self.take_scan_batch_readahead = take_scan_batch_readahead
        self.validate_payload_keys = validate_payload_keys
        self.session = lance.Session(
            index_cache_size_bytes=index_cache.index_cache_size_bytes,
            metadata_cache_size_bytes=index_cache.metadata_cache_size_bytes,
        )
        self.remote_dataset = lance.dataset(
            dataset_config.uri,
            version=dataset_config.version,
            storage_options=dataset_config.storage_options or None,
            session=self.session,
        )
        self._validate_remote_dataset()
        self.manifest = _validate_stable_global_ordinal_manifest(self.remote_dataset)
        self.prewarm_seconds = 0.0
        self.executor = ThreadPoolExecutor(
            max_workers=max_pending_takes,
            thread_name_prefix="lance-private-take",
        )

    @property
    def key_type(self) -> pa.DataType:
        return self.remote_dataset.schema.field(self.config.key_column).type

    @property
    def source_types(self) -> dict[str, pa.DataType]:
        return {source: self.remote_dataset.schema.field(source).type for source in self.columns}

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.remote_dataset = None
        self.session = None

    def _validate_remote_dataset(self) -> None:
        key_column = self.config.key_column
        remote_schema = self.remote_dataset.schema
        if key_column not in remote_schema.names:
            msg = f"Lance key column {key_column!r} is missing"
            raise ValueError(msg)

        missing = sorted(set(self.columns) - set(remote_schema.names))
        if missing:
            msg = f"Requested Lance columns do not exist: {missing}"
            raise ValueError(msg)
        if not self.remote_dataset.has_stable_row_ids:
            msg = "LanceColumnFetchStage requires stable row IDs"
            raise ValueError(msg)

    def _resolve_row_ids(self, keys: list[object]) -> tuple[dict[object, int], dict[str, float]]:
        raise NotImplementedError

    def _lookup_io_dataset(self) -> _IOStatsDatasetProtocol:
        """Return the Lance dataset whose object store performs key lookup."""
        return self.remote_dataset

    def _reset_io_stats_before_lookup(self, lookup_io_dataset: _IOStatsDatasetProtocol) -> None:
        self.remote_dataset.io_stats_incremental()
        if lookup_io_dataset is not self.remote_dataset:
            lookup_io_dataset.io_stats_incremental()

    def _take_rows(self, row_ids: list[int]) -> tuple[list[_PrivateReadBatchResult], dict[str, float]]:
        plan = _plan_adaptive_locality_reads(
            row_ids,
            manifest=self.manifest,
            fetch_batch_size=self.fetch_batch_size,
            payload_read_mode=self.payload_read_mode,
            medium_density_threshold=self.medium_density_threshold,
            high_density_threshold=self.high_density_threshold,
            max_coalesced_range_gap=self.max_coalesced_range_gap,
        )
        projected = list(self.columns)
        if self.validate_payload_keys:
            projected.insert(0, self.config.key_column)
        projected = list(dict.fromkeys(projected))

        def read_operation(operation: _PrivateReadOperation) -> _PrivateReadBatchResult:
            if operation.strategy == "take_rows":
                table = self.remote_dataset._take_rows(list(operation.row_ids), columns=projected)
                if table.num_rows != len(operation.row_ids):
                    msg = (
                        f"Private Lance take returned {table.num_rows} rows "
                        f"for {len(operation.row_ids)} stable row IDs"
                    )
                    raise RuntimeError(msg)
                return _PrivateReadBatchResult(table=table, row_ids=operation.row_ids)

            batches = list(
                self.remote_dataset._ds.take_scan(
                    list(operation.ranges),
                    columns=projected,
                    batch_readahead=self.take_scan_batch_readahead,
                )
            )
            if len(batches) != len(operation.ranges):
                msg = f"Private Lance take_scan returned {len(batches)} batches for {len(operation.ranges)} ranges"
                raise RuntimeError(msg)
            requested = set(operation.row_ids)
            filtered_tables: list[pa.Table] = []
            filtered_row_ids: list[int] = []
            for (start, stop), batch in zip(operation.ranges, batches, strict=True):
                if batch.num_rows != stop - start:
                    msg = f"Private Lance take_scan range [{start}, {stop}) returned {batch.num_rows} rows"
                    raise RuntimeError(msg)
                range_ids = list(range(start, stop))
                mask = pa.array([row_id in requested for row_id in range_ids], type=pa.bool_())
                filtered_tables.append(pa.Table.from_batches([batch]).filter(mask))
                filtered_row_ids.extend(row_id for row_id in range_ids if row_id in requested)
            table = pa.concat_tables(filtered_tables) if len(filtered_tables) > 1 else filtered_tables[0]
            if table.num_rows != len(operation.row_ids):
                msg = (
                    f"Private Lance take_scan returned {table.num_rows} requested rows "
                    f"for {len(operation.row_ids)} stable row IDs"
                )
                raise RuntimeError(msg)
            return _PrivateReadBatchResult(table=table, row_ids=tuple(filtered_row_ids))

        private_take_started = time.perf_counter()
        if projected:
            tables, peak_pending = _bounded_parallel_map(
                self.executor,
                read_operation,
                plan.operations,
                self.max_pending_takes,
            )
        else:
            tables, peak_pending = [], 0
        private_take_seconds = time.perf_counter() - private_take_started if projected else 0.0
        take_calls = len(plan.operations) if projected else 0
        take_rows_calls = sum(operation.strategy == "take_rows" for operation in plan.operations) if projected else 0
        take_scan_calls = take_calls - take_rows_calls
        metrics = {
            "private_take_calls": float(take_calls),
            "private_take_rows": float(len(plan.row_ids)),
            "private_take_seconds": private_take_seconds,
            "rows_per_private_take": len(plan.row_ids) / take_calls if take_calls else 0.0,
            "max_pending_private_takes": float(peak_pending),
            "coordinate_density": plan.coordinate_density,
            "coordinate_duplicate_fanout": len(row_ids) / len(plan.row_ids) if plan.row_ids else 0.0,
            "sparse_calls_avoided": float(max(0, len(plan.row_ids) - take_calls)),
            "strategy_sparse_fragments": float(plan.sparse_fragments),
            "strategy_range_fragments": float(plan.range_fragments),
            "strategy_sequential_fragments": float(plan.sequential_fragments),
            "take_rows_calls": float(take_rows_calls),
            "take_scan_calls": float(take_scan_calls),
            "take_scan_ranges": float(plan.take_scan_ranges if projected else 0),
            "planned_scan_rows": float(plan.planned_scan_rows if projected else 0),
            "range_overread_rows": float(plan.range_overread_rows if projected else 0),
        }
        return tables, metrics

    def _empty_payload_table(self) -> pa.Table:
        return pa.table({source: pa.array([], type=source_type) for source, source_type in self.source_types.items()})

    def _assemble_arrow_payload(
        self,
        tables: list[_PrivateReadBatchResult],
        key_to_row_id: dict[object, int],
        expected_payload_rows: int,
    ) -> tuple[pa.Table, pa.Array, dict[str, int]]:
        """Validate private reads while retaining projected payloads in Arrow."""
        fetched_bytes = dict.fromkeys(self.columns, 0)
        key_by_row_id = {row_id: key for key, row_id in key_to_row_id.items()}
        returned_row_ids: list[int] = []
        payload_tables: list[pa.Table] = []
        for result in tables:
            table = result.table
            for source in self.columns:
                fetched_bytes[source] += table[source].nbytes
            key_values = (
                table[self.config.key_column].combine_chunks().to_pylist() if self.validate_payload_keys else None
            )
            for row_index, row_id in enumerate(result.row_ids):
                if key_values is not None and key_values[row_index] != key_by_row_id[row_id]:
                    msg = (
                        f"Stable row ID {row_id} returned key {key_values[row_index]!r}; "
                        f"expected {key_by_row_id[row_id]!r}"
                    )
                    raise RuntimeError(msg)
            returned_row_ids.extend(result.row_ids)
            payload_tables.append(table)

        if len(set(returned_row_ids)) != len(returned_row_ids):
            msg = "Private Lance reads returned a stable row ID more than once"
            raise ValueError(msg)
        if expected_payload_rows:
            missing_row_ids = set(key_to_row_id.values()) - set(returned_row_ids)
            if missing_row_ids:
                msg = f"Private Lance reads omitted stable row IDs: {sorted(missing_row_ids)[:5]}"
                raise RuntimeError(msg)
        payload = (
            pa.concat_tables(payload_tables)
            if len(payload_tables) > 1
            else payload_tables[0]
            if payload_tables
            else self._empty_payload_table()
        )
        return payload, pa.array(returned_row_ids, type=pa.uint64()), fetched_bytes

    def fetch(self, keys: list[object]) -> _FetchResult:
        if not keys:
            return _FetchResult(
                key_to_row_id={},
                payload=self._empty_payload_table(),
                payload_row_ids=pa.array([], type=pa.uint64()),
                lookup_seconds=0.0,
                fetch_seconds=0.0,
                fetched_bytes_by_column={},
                lookup_metrics={
                    "private_take_calls": 0.0,
                    "private_take_rows": 0.0,
                    "private_take_seconds": 0.0,
                    "rows_per_private_take": 0.0,
                    "max_pending_private_takes": 0.0,
                    "coordinate_density": 0.0,
                    "coordinate_duplicate_fanout": 0.0,
                    "sparse_calls_avoided": 0.0,
                    "strategy_sparse_fragments": 0.0,
                    "strategy_range_fragments": 0.0,
                    "strategy_sequential_fragments": 0.0,
                    "take_rows_calls": 0.0,
                    "take_scan_calls": 0.0,
                    "take_scan_ranges": 0.0,
                    "planned_scan_rows": 0.0,
                    "range_overread_rows": 0.0,
                    "lookup_read_bytes": 0.0,
                    "lookup_read_iops": 0.0,
                    "average_physical_read_bytes": 0.0,
                    "physical_reads_per_payload": 0.0,
                    "physical_read_operations_per_second": 0.0,
                    "read_amplification": 0.0,
                },
            )

        lookup_io_dataset = self._lookup_io_dataset()
        self._reset_io_stats_before_lookup(lookup_io_dataset)
        lookup_started = time.perf_counter()
        key_to_row_id, lookup_metrics = self._resolve_row_ids(keys)
        lookup_seconds = time.perf_counter() - lookup_started
        lookup_io_stats = lookup_io_dataset.io_stats_incremental()
        lookup_metrics.update(
            {
                "lookup_read_bytes": float(lookup_io_stats.read_bytes),
                "lookup_read_iops": float(lookup_io_stats.read_iops),
            }
        )
        if len(set(key_to_row_id.values())) != len(key_to_row_id):
            msg = "Lance key resolver maps multiple keys to one stable row ID"
            raise ValueError(msg)

        # Index and payload datasets may use distinct object stores (for
        # example, a local mirror backed by remote payloads).  Reset the remote
        # counters immediately before the private payload reads.
        self.remote_dataset.io_stats_incremental()
        fetch_started = time.perf_counter()
        tables, fetch_metrics = self._take_rows(list(key_to_row_id.values()))
        fetch_seconds = time.perf_counter() - fetch_started

        returned_rows = sum(result.table.num_rows for result in tables)
        expected_rows = int(fetch_metrics["private_take_rows"])
        expected_payload_rows = expected_rows if self.columns or self.validate_payload_keys else 0
        if returned_rows != expected_payload_rows:
            msg = f"Lance payload fetch returned {returned_rows} rows for {expected_payload_rows} stable row IDs"
            raise RuntimeError(msg)

        payload, payload_row_ids, fetched_bytes = self._assemble_arrow_payload(
            tables,
            key_to_row_id,
            expected_payload_rows,
        )

        io_stats = self.remote_dataset.io_stats_incremental()
        fetched_payload_bytes = sum(fetched_bytes.values())
        fetch_metrics.update(
            {
                "average_physical_read_bytes": (
                    int(io_stats.read_bytes) / int(io_stats.read_iops) if io_stats.read_iops else 0.0
                ),
                "physical_reads_per_payload": (int(io_stats.read_iops) / expected_rows if expected_rows else 0.0),
                "physical_read_operations_per_second": (
                    int(io_stats.read_iops) / float(fetch_metrics["private_take_seconds"])
                    if fetch_metrics["private_take_seconds"]
                    else 0.0
                ),
                "read_amplification": (
                    int(io_stats.read_bytes) / fetched_payload_bytes if fetched_payload_bytes else 0.0
                ),
            }
        )
        return _FetchResult(
            key_to_row_id=key_to_row_id,
            payload=payload,
            payload_row_ids=payload_row_ids,
            lookup_seconds=lookup_seconds,
            fetch_seconds=fetch_seconds,
            fetched_bytes_by_column=fetched_bytes,
            read_bytes=int(io_stats.read_bytes),
            read_iops=int(io_stats.read_iops),
            lookup_metrics={**lookup_metrics, **fetch_metrics},
        )


class _LanceColumnFetcher(_LancePayloadFetcher):
    """Resolve keys with a Lance scalar index, then fetch stable row IDs."""

    def __init__(  # noqa: PLR0913
        self,
        dataset_config: LanceDatasetConfig,
        index_cache: LanceIndexCacheConfig,
        columns: dict[str, str],
        lookup_batch_size: int,
        fetch_batch_size: int,
        max_pending_takes: int,
        payload_read_mode: PayloadReadMode,
        medium_density_threshold: float,
        high_density_threshold: float,
        max_coalesced_range_gap: int,
        take_scan_batch_readahead: int,
        validate_payload_keys: bool,
    ) -> None:
        import lance

        super().__init__(
            dataset_config,
            index_cache,
            columns,
            fetch_batch_size,
            max_pending_takes,
            payload_read_mode,
            medium_density_threshold,
            high_density_threshold,
            max_coalesced_range_gap,
            take_scan_batch_readahead,
            validate_payload_keys,
        )
        self.lookup_batch_size = lookup_batch_size
        self.index_uri = index_cache.resolved_index_uri(dataset_config)
        index_options: dict[str, Any] = {
            "version": index_cache.resolved_index_version(dataset_config),
            "session": self.session,
        }
        if self.index_uri == dataset_config.uri and dataset_config.storage_options:
            index_options["storage_options"] = dataset_config.storage_options
        self.index_dataset = None
        try:
            self.index_dataset = lance.dataset(self.index_uri, **index_options)
            self._validate_index_dataset()
            if index_cache.prewarm:
                started = time.perf_counter()
                self.index_dataset.prewarm_index(dataset_config.index_name)
                self.prewarm_seconds = time.perf_counter() - started
        except BaseException:
            self.index_dataset = None
            super().close()
            raise

    def close(self) -> None:
        self.index_dataset = None
        super().close()

    def _validate_index_dataset(self) -> None:
        _validate_mirror_index_surface(self.remote_dataset, self.index_dataset, self.config)
        _validate_index_mirror_contract(
            self.config,
            self.index_cache,
            self.remote_dataset,
            self.index_dataset,
            self.index_uri,
        )

    def _lookup_io_dataset(self) -> _IOStatsDatasetProtocol:
        if self.index_dataset is None:
            msg = "Lance index dataset is closed"
            raise RuntimeError(msg)
        return self.index_dataset

    def _resolve_row_ids(self, keys: list[object]) -> tuple[dict[object, int], dict[str, float]]:
        key_to_row_id: dict[object, int] = {}
        requested = set(keys)
        for start in range(0, len(keys), self.lookup_batch_size):
            key_chunk = keys[start : start + self.lookup_batch_size]
            key_array = pa.array(key_chunk, type=self.key_type, from_pandas=True)
            table = self.index_dataset.scanner(
                columns=[self.config.key_column],
                filter=pc.field(self.config.key_column).isin(key_array),
                prefilter=True,
                with_row_id=True,
                use_scalar_index=True,
                fast_search=False,
            ).to_table()
            if _ROW_ID_COLUMN not in table.column_names:
                msg = "Lance index lookup did not return stable row IDs"
                raise RuntimeError(msg)
            returned_keys = table[self.config.key_column].combine_chunks().to_pylist()
            returned_row_ids = table[_ROW_ID_COLUMN].combine_chunks().to_pylist()
            for key, row_id in zip(returned_keys, returned_row_ids, strict=True):
                if key not in requested:
                    msg = f"Lance index returned unexpected key {key!r}"
                    raise RuntimeError(msg)
                if key in key_to_row_id:
                    msg = f"Multiple Lance rows matched key {key!r}"
                    raise ValueError(msg)
                key_to_row_id[key] = int(row_id)
        return key_to_row_id, {}


@dataclass
class LanceColumnFetchStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Fetch selected columns from a pinned Lance table using exact indexed keys.

    The pinned dataset must expose stable global-ordinal row IDs backed by
    contiguous manifest-order fragments without deletions. Resolved IDs are
    sorted and deduplicated. The measured ``payload_read_mode="sparse"``
    default fetches only through private ``dataset._take_rows`` calls;
    ``fetch_batch_size`` bounds each take while ``max_pending_takes`` bounds
    submitted and running work. The opt-in
    ``"adaptive_unmeasured"`` mode uses inner private ``_ds.take_scan`` for
    configured medium/high per-fragment densities; its thresholds are
    provisional and carry no speedup claim. Its tracked objective is fewer
    sparse payload calls, reported through ``sparse_calls_avoided`` and the
    per-strategy operation metrics. Payload reads project only
    ``columns`` unless ``validate_payload_keys`` explicitly adds the key column
    for an out-of-band correctness run. ``index_cache.mirror_path`` is accepted
    only with a caller-pinned ``mirror_contract``; setup verifies its exact
    fragment layout, key-to-stable-ordinal Arrow identity, and index artifacts
    before prewarm or lookup.
    """

    dataset: LanceDatasetConfig
    index_cache: LanceIndexCacheConfig = field(default_factory=LanceIndexCacheConfig)
    input_key_column: str = "source_ref"
    columns: dict[str, str] = field(default_factory=dict)
    presence_column: str | None = None
    existing_column_policy: ExistingColumnPolicy = "error"
    missing_key_policy: MissingKeyPolicy = "mark"
    lookup_batch_size: int = 2_000
    fetch_batch_size: int = 1_024
    max_pending_takes: int = 16
    payload_read_mode: PayloadReadMode = "sparse"
    medium_density_threshold: float = 0.25
    high_density_threshold: float = 0.75
    max_coalesced_range_gap: int = 0
    take_scan_batch_readahead: int = 16
    validate_payload_keys: bool = False
    name: str = "lance_column_fetch"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _fetcher: _LanceColumnFetcher | None = field(default=None, init=False, repr=False)
    _prewarm_metric_pending: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.columns = dict(self.columns or {})
        if not self.input_key_column:
            msg = "input_key_column must not be empty"
            raise ValueError(msg)
        if not self.columns and not self.presence_column:
            msg = "columns may be empty only when presence_column is configured"
            raise ValueError(msg)
        if len(set(self.columns.values())) != len(self.columns):
            msg = "Each Lance source column must map to a distinct destination column"
            raise ValueError(msg)
        if self.presence_column in self.columns.values():
            msg = "presence_column must not also be a projected destination column"
            raise ValueError(msg)
        if self.existing_column_policy not in {"error", "fill_null", "overwrite"}:
            msg = f"Unsupported existing_column_policy: {self.existing_column_policy}"
            raise ValueError(msg)
        if self.missing_key_policy not in {"mark", "error"}:
            msg = f"Unsupported missing_key_policy: {self.missing_key_policy}"
            raise ValueError(msg)
        if self.missing_key_policy == "mark" and not self.presence_column:
            msg = "missing_key_policy='mark' requires presence_column"
            raise ValueError(msg)
        _validate_adaptive_locality_config(
            payload_read_mode=self.payload_read_mode,
            medium_density_threshold=self.medium_density_threshold,
            high_density_threshold=self.high_density_threshold,
            max_coalesced_range_gap=self.max_coalesced_range_gap,
            take_scan_batch_readahead=self.take_scan_batch_readahead,
        )
        for name, value in {
            "lookup_batch_size": self.lookup_batch_size,
            "fetch_batch_size": self.fetch_batch_size,
            "max_pending_takes": self.max_pending_takes,
        }.items():
            if value <= 0:
                msg = f"{name} must be greater than 0"
                raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_columns = list(self.columns.values())
        if self.presence_column:
            output_columns.append(self.presence_column)
        return ["data"], output_columns

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if not self.index_cache.copy_to_node_local:
            return
        source = Path(self.index_cache.mirror_path or "")
        if not source.is_dir():
            msg = f"Lance index mirror does not exist: {source}"
            raise FileNotFoundError(msg)
        target = self.index_cache.node_local_path(self.dataset)
        ready = target / _NODE_READY_FILE
        if ready.is_file():
            _require_node_ready_marker(ready, self.dataset, self.index_cache)
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        shutil.rmtree(temporary)
        try:
            shutil.copytree(source, temporary)
            (temporary / _NODE_READY_FILE).write_text(
                json.dumps(
                    _node_ready_payload(self.dataset, self.index_cache),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                os.rename(temporary, target)
            except FileExistsError:
                _require_node_ready_marker(ready, self.dataset, self.index_cache)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.index_cache.copy_to_node_local:
            ready = self.index_cache.node_local_path(self.dataset) / _NODE_READY_FILE
            _require_node_ready_marker(ready, self.dataset, self.index_cache)
        self._fetcher = _LanceColumnFetcher(
            self.dataset,
            self.index_cache,
            self.columns,
            self.lookup_batch_size,
            self.fetch_batch_size,
            self.max_pending_takes,
            self.payload_read_mode,
            self.medium_density_threshold,
            self.high_density_threshold,
            self.max_coalesced_range_gap,
            self.take_scan_batch_readahead,
            self.validate_payload_keys,
        )
        self._prewarm_metric_pending = self._fetcher.prewarm_seconds

    def teardown(self) -> None:
        if self._fetcher is not None:
            self._fetcher.close()
            self._fetcher = None

    def _ensure_fetcher(self) -> _LanceColumnFetcher:
        if self._fetcher is None:
            self.setup()
        if self._fetcher is None:  # pragma: no cover - setup either returns a fetcher or raises
            msg = "Lance column fetcher setup did not initialize the worker"
            raise RuntimeError(msg)
        return self._fetcher

    def _validate_input_table(self, table: pa.Table, source_types: dict[str, pa.DataType]) -> None:
        if self.input_key_column not in table.column_names:
            msg = f"Input key column {self.input_key_column!r} does not exist"
            raise ValueError(msg)
        collisions = sorted(set(self.columns.values()) & set(table.column_names))
        if collisions and self.existing_column_policy == "error":
            msg = f"Projected destination columns already exist: {collisions}"
            raise ValueError(msg)
        for source, destination in self.columns.items():
            if destination in table.column_names and table.schema.field(destination).type != source_types[source]:
                msg = (
                    f"Destination column {destination!r} has type {table.schema.field(destination).type}; "
                    f"Lance column {source!r} has type {source_types[source]}"
                )
                raise TypeError(msg)
        if self.presence_column in table.column_names and not pa.types.is_boolean(
            table.schema.field(self.presence_column).type
        ):
            msg = f"Presence column {self.presence_column!r} must have boolean type"
            raise TypeError(msg)

    @staticmethod
    def _key_is_present(key: object) -> bool:
        return key is not None and (not isinstance(key, str) or key != "")

    def _requested_keys(
        self,
        table: pa.Table,
        keys: list[object],
        presence: list[bool | None] | None,
    ) -> tuple[list[object], pa.Array]:
        destination_validity = {
            destination: pc.is_valid(table[destination]).to_pylist()
            for destination in self.columns.values()
            if destination in table.column_names
        }
        requested: list[object] = []
        requested_mask: list[bool] = []
        for index, key in enumerate(keys):
            if not self._key_is_present(key) or (presence is not None and presence[index] is False):
                requested_mask.append(False)
                continue
            if self.existing_column_policy == "fill_null" and self.columns:
                all_populated = all(
                    destination in destination_validity and destination_validity[destination][index]
                    for destination in self.columns.values()
                )
                presence_populated = presence is None or presence[index] is not None
                if all_populated and presence_populated:
                    requested_mask.append(False)
                    continue
            elif not self.columns and presence is not None and presence[index] is not None:
                requested_mask.append(False)
                continue
            try:
                hash(key)
            except TypeError as exc:
                msg = f"Input Lance key is not hashable: {key!r}"
                raise TypeError(msg) from exc
            requested.append(key)
            requested_mask.append(True)
        return requested, pa.array(requested_mask, type=pa.bool_())

    def _apply_projection(
        self,
        table: pa.Table,
        keys: list[object],
        requested_mask: pa.Array,
        fetch_result: _FetchResult,
    ) -> pa.Table:
        all_row_ids = pa.array(
            [fetch_result.key_to_row_id.get(key) for key in keys],
            type=pa.uint64(),
            from_pandas=True,
        )
        requested_row_ids = pc.if_else(
            requested_mask,
            all_row_ids,
            pa.nulls(table.num_rows, type=pa.uint64()),
        )
        payload_offsets = pc.index_in(
            requested_row_ids,
            value_set=fetch_result.payload_row_ids,
        )
        matched = pc.and_(pc.is_valid(payload_offsets), requested_mask)
        result = table
        for source, destination in self.columns.items():
            projected = pc.take(
                fetch_result.payload[source],
                payload_offsets,
                boundscheck=False,
            )
            if destination in result.column_names:
                existing = result[destination]
                replace = (
                    pc.and_(matched, pc.is_null(existing)) if self.existing_column_policy == "fill_null" else matched
                )
                projected = pc.if_else(replace, projected, existing)
            column_index = result.schema.get_field_index(destination)
            if column_index >= 0:
                result = result.set_column(column_index, destination, projected)
            else:
                result = result.append_column(destination, projected)
        return result

    def _apply_presence(
        self,
        table: pa.Table,
        keys: list[object],
        requested_mask: pa.Array,
        found_keys: set[object],
    ) -> pa.Table:
        if not self.presence_column:
            return table
        resolved = pa.array([key in found_keys for key in keys], type=pa.bool_())
        existing = (
            table[self.presence_column]
            if self.presence_column in table.column_names
            else pa.nulls(table.num_rows, type=pa.bool_())
        )
        presence = pc.if_else(requested_mask, resolved, existing)
        column_index = table.schema.get_field_index(self.presence_column)
        if column_index >= 0:
            return table.set_column(column_index, self.presence_column, presence)
        return table.append_column(self.presence_column, presence)

    def _prepare_task(
        self,
        task: InterleavedBatch,
        fetcher: _LanceColumnFetcher,
        source_types: dict[str, pa.DataType],
    ) -> _PreparedFetchTask:
        table = task.to_pyarrow()
        self._validate_input_table(table, source_types)
        if table.schema.field(self.input_key_column).type != fetcher.key_type:
            msg = (
                f"Input key column has type {table.schema.field(self.input_key_column).type}; "
                f"Lance key column has type {fetcher.key_type}"
            )
            raise TypeError(msg)

        keys = table[self.input_key_column].combine_chunks().to_pylist()
        presence = (
            table[self.presence_column].combine_chunks().to_pylist()
            if self.presence_column and self.presence_column in table.column_names
            else None
        )
        requested_keys, requested_mask = self._requested_keys(table, keys, presence)
        return _PreparedFetchTask(
            task=task,
            table=table,
            keys=keys,
            requested_keys=requested_keys,
            requested_mask=requested_mask,
        )

    def _process_tasks(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        if len(tasks) == 0:
            return []

        fetcher = self._ensure_fetcher()
        source_types = fetcher.source_types
        prepared = [self._prepare_task(task, fetcher, source_types) for task in tasks]

        # An actor owns one Lance session. Fold the Curator task batch into one
        # deterministic lookup so the B-tree is traversed serially and keys
        # shared by adjacent partitions are resolved and fetched only once.
        requested_keys = list(dict.fromkeys(key for prepared_task in prepared for key in prepared_task.requested_keys))
        fetch_result = fetcher.fetch(requested_keys)
        found_keys = set(fetch_result.key_to_row_id)
        missing_keys = [key for key in requested_keys if key not in found_keys]
        logical_payload_requests = sum(
            key in found_keys for prepared_task in prepared for key in prepared_task.requested_keys
        )
        unique_payloads = len(fetch_result.key_to_row_id)
        if missing_keys and self.missing_key_policy == "error":
            sample = ", ".join(repr(key) for key in missing_keys[:5])
            msg = f"{len(missing_keys)} Lance keys were not found; examples: {sample}"
            raise KeyError(msg)

        outputs: list[InterleavedBatch] = []
        for prepared_task in prepared:
            result = self._apply_projection(
                prepared_task.table,
                prepared_task.keys,
                prepared_task.requested_mask,
                fetch_result,
            )
            result = self._apply_presence(
                result,
                prepared_task.keys,
                prepared_task.requested_mask,
                found_keys,
            )
            outputs.append(
                InterleavedBatch(
                    dataset_name=prepared_task.task.dataset_name,
                    data=result,
                    _metadata=prepared_task.task._metadata,
                    _stage_perf=prepared_task.task._stage_perf,
                )
            )

        metrics = {
            "stage_windows": 1.0,
            "input_tasks": float(len(prepared)),
            "input_rows": float(sum(prepared_task.table.num_rows for prepared_task in prepared)),
            "requested_unique_keys": float(len(requested_keys)),
            "found_unique_keys": float(len(fetch_result.key_to_row_id)),
            "missing_unique_keys": float(len(missing_keys)),
            "logical_payload_requests": float(logical_payload_requests),
            "unique_payloads": float(unique_payloads),
            "logical_duplicate_requests": float(max(0, logical_payload_requests - unique_payloads)),
            "duplicate_fanout": logical_payload_requests / unique_payloads if unique_payloads else 0.0,
            "lance_lookup_seconds": fetch_result.lookup_seconds,
            "lance_fetch_seconds": fetch_result.fetch_seconds,
            "lance_fetched_bytes": float(sum(fetch_result.fetched_bytes_by_column.values())),
            "lance_read_bytes": float(fetch_result.read_bytes),
            "lance_read_iops": float(fetch_result.read_iops),
            "peak_rss_bytes": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        }
        for source, value in fetch_result.fetched_bytes_by_column.items():
            metrics[f"lance_fetched_{source}_bytes"] = float(value)
        metrics.update(fetch_result.lookup_metrics)
        if self._prewarm_metric_pending is not None:
            metrics["lance_index_prewarm_seconds"] = self._prewarm_metric_pending
            self._prewarm_metric_pending = None
        self._log_metrics(metrics)
        return outputs

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return self._process_tasks([task])[0]

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        """Resolve and fetch one deduplicated key set for a Curator task batch."""
        return self._process_tasks(tasks)


@dataclass
class InterleavedLanceReaderStage(LanceReaderStage):
    """Read Lance fragments into validated ``InterleavedBatch`` objects."""

    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.fields is not None:
            missing = sorted(InterleavedBatch.REQUIRED_COLUMNS - set(self.fields))
            if missing:
                msg = f"Interleaved Lance fields omit required columns: {missing}"
                raise ValueError(msg)

    def process(self, task: LanceReadTask) -> InterleavedBatch:
        output: ReaderOutput = self.read_task(task, self._effective_read_kwargs(), self.fields)
        self._validate_result(task, output.data)
        batch = InterleavedBatch(
            dataset_name=task.dataset_name,
            data=output.data,
            _metadata=self._output_metadata(task, output),
            _stage_perf=task._stage_perf,
        )
        if batch.to_pyarrow().num_rows and not batch.validate():
            msg = f"Lance fragment task {task.task_id} is not a valid InterleavedBatch"
            raise ValueError(msg)
        return batch


@dataclass
class InterleavedLanceReader(CompositeStage[EmptyTask, InterleavedBatch]):
    """Partition and read a Lance dataset as row-wise interleaved batches."""

    path: str
    fragments_per_partition: int = 1
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    name: str = "interleaved_lance_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        return [
            LancePartitioningStage(
                path=self.path,
                fragments_per_partition=self.fragments_per_partition,
                fragment_ids=self.fragment_ids,
                read_kwargs=self.read_kwargs,
            ),
            InterleavedLanceReaderStage(
                path=self.path,
                fields=self.fields,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]
