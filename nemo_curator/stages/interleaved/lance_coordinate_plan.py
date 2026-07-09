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

"""Durable contract for one document fragment's resolved image coordinates."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.tasks.tasks import Task
from nemo_curator.utils.atomic_io import fsync_directory, write_json_atomically_if_absent
from nemo_curator.utils.uri import validate_credential_free_uri_identity

DOCUMENT_ROWADDR = "document_rowaddr"
DOCUMENT_POSITION = "document_position"
STABLE_ROW_ID = "stable_row_id"

_ARTIFACT_KIND = "lance_coordinate_plan"
_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
MissingKeyPolicy = Literal["error", "null"]


def lance_coordinate_plan_schema(*, allow_missing: bool = False) -> pa.Schema:
    """Return the exact Arrow schema for the requested missing-key policy."""
    if not isinstance(allow_missing, bool):
        msg = "allow_missing must be a boolean"
        raise TypeError(msg)
    return pa.schema(
        [
            pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
            pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
            pa.field(STABLE_ROW_ID, pa.uint64(), nullable=allow_missing),
        ]
    )


LANCE_COORDINATE_PLAN_SCHEMA = lance_coordinate_plan_schema()


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{name} must be a nonnegative integer"
        raise ValueError(msg)
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        msg = f"{name} must be a lowercase SHA-256 digest"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class _CoordinatePlanArtifactIdentity:
    """Immutable dataset and sidecar identity for a coordinate plan."""

    document_uri: str
    document_version: int
    image_uri: str
    image_version: int
    fragment_id: int
    sidecar_manifest_sha256: str
    fragment_manifest_sha256: str
    missing_key_policy: MissingKeyPolicy = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.document_uri, str) or not self.document_uri:
            msg = "document_uri must be a non-empty string"
            raise ValueError(msg)
        if not isinstance(self.image_uri, str) or not self.image_uri:
            msg = "image_uri must be a non-empty string"
            raise ValueError(msg)
        validate_credential_free_uri_identity(self.document_uri, "document Lance URI")
        validate_credential_free_uri_identity(self.image_uri, "image Lance URI")
        _require_positive_integer(self.document_version, "document_version")
        _require_positive_integer(self.image_version, "image_version")
        _require_nonnegative_integer(self.fragment_id, "fragment_id")
        _require_sha256(self.sidecar_manifest_sha256, "sidecar_manifest_sha256")
        _require_sha256(self.fragment_manifest_sha256, "fragment_manifest_sha256")
        if self.missing_key_policy not in {"error", "null"}:
            msg = f"Unsupported missing_key_policy: {self.missing_key_policy!r}"
            raise ValueError(msg)

    def as_manifest_fields(self) -> dict[str, object]:
        return {
            "document": {
                "uri": self.document_uri,
                "version": self.document_version,
                "fragment_id": self.fragment_id,
            },
            "image": {"uri": self.image_uri, "version": self.image_version},
            "sidecar_manifest_sha256": self.sidecar_manifest_sha256,
            "fragment_manifest_sha256": self.fragment_manifest_sha256,
            "missing_key_policy": self.missing_key_policy,
        }


@dataclass(frozen=True)
class CoordinatePlanIdentity:
    """Caller-facing identity independent of the missing-key policy."""

    document_uri: str
    document_version: int
    image_uri: str
    image_version: int
    fragment_id: int
    sidecar_manifest_sha256: str
    fragment_manifest_sha256: str

    def _with_missing_policy(self, allow_missing: bool) -> _CoordinatePlanArtifactIdentity:
        if not isinstance(allow_missing, bool):
            msg = "allow_missing must be a boolean"
            raise TypeError(msg)
        return _CoordinatePlanArtifactIdentity(
            document_uri=self.document_uri,
            document_version=self.document_version,
            image_uri=self.image_uri,
            image_version=self.image_version,
            fragment_id=self.fragment_id,
            sidecar_manifest_sha256=self.sidecar_manifest_sha256,
            fragment_manifest_sha256=self.fragment_manifest_sha256,
            missing_key_policy="null" if allow_missing else "error",
        )

    def __post_init__(self) -> None:
        self._with_missing_policy(False)

    def identity_sha256(self) -> str:
        fields = self._with_missing_policy(False).as_manifest_fields()
        fields.pop("missing_key_policy")
        return hashlib.sha256(_canonical_json_bytes(fields)).hexdigest()


@dataclass
class LanceCoordinatePlanTask(Task[str]):
    """Lightweight task naming one validated Parquet/manifest pair."""

    data: str = ""
    manifest_path: str = ""
    source_identity_sha256: str = ""

    @property
    def num_items(self) -> int:
        return 1

    def validate(self) -> bool:
        if not isinstance(self.data, str) or not self.data:
            return False
        if not isinstance(self.manifest_path, str) or not self.manifest_path or self.data == self.manifest_path:
            return False
        return not self.source_identity_sha256 or (
            isinstance(self.source_identity_sha256, str)
            and _SHA256_PATTERN.fullmatch(self.source_identity_sha256) is not None
        )

    def get_deterministic_id(self) -> str:
        if self.source_identity_sha256:
            return self.source_identity_sha256
        material = f"{self.data}\n{self.manifest_path}".encode()
        return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class LanceCoordinatePlanStats:
    """Validated logical counts retained in the durable manifest."""

    rows: int
    unique_document_rowaddrs: int
    unique_document_positions: int
    non_null_stable_row_ids: int
    unique_stable_row_ids: int
    duplicate_stable_row_id_occurrences: int
    null_stable_row_ids: int
    document_position_min: int | None
    document_position_max: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "rows": self.rows,
            "unique_document_rowaddrs": self.unique_document_rowaddrs,
            "unique_document_positions": self.unique_document_positions,
            "non_null_stable_row_ids": self.non_null_stable_row_ids,
            "unique_stable_row_ids": self.unique_stable_row_ids,
            "duplicate_stable_row_id_occurrences": self.duplicate_stable_row_id_occurrences,
            "null_stable_row_ids": self.null_stable_row_ids,
            "document_position_min": self.document_position_min,
            "document_position_max": self.document_position_max,
        }


@dataclass(frozen=True)
class LanceCoordinatePlanArtifact:
    """One validated coordinate artifact and its publication state."""

    parquet_path: Path
    manifest_path: Path
    table: pa.Table
    manifest: dict[str, object]
    adopted: bool


def _schema_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    return [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema]


def _schema_for_policy(missing_key_policy: MissingKeyPolicy) -> pa.Schema:
    if missing_key_policy not in {"error", "null"}:
        msg = f"Unsupported missing_key_policy: {missing_key_policy!r}"
        raise ValueError(msg)
    return lance_coordinate_plan_schema(allow_missing=missing_key_policy == "null")


def _canonicalize_uint64_array(values: pa.ChunkedArray) -> pa.Array:
    contiguous = values.combine_chunks()
    if contiguous.null_count == 0:
        return pa.Array.from_buffers(
            pa.uint64(),
            len(contiguous),
            [None, contiguous.buffers()[1]],
            null_count=0,
        )
    filled = pc.fill_null(contiguous, pa.scalar(0, type=pa.uint64()))
    return pc.if_else(
        pc.is_valid(contiguous),
        filled,
        pa.scalar(None, type=pa.uint64()),
    )


def _canonicalize_and_validate(
    table: pa.Table,
    missing_key_policy: MissingKeyPolicy,
) -> tuple[pa.Table, LanceCoordinatePlanStats]:
    if not isinstance(table, pa.Table):
        msg = f"Coordinate plan must be a pyarrow.Table, got {type(table).__name__}"
        raise TypeError(msg)
    expected_schema = _schema_for_policy(missing_key_policy)
    if not table.schema.equals(expected_schema, check_metadata=False):
        msg = f"Coordinate plan schema is {table.schema}; expected {expected_schema}"
        raise TypeError(msg)

    canonical = pa.Table.from_arrays(
        [_canonicalize_uint64_array(table[field.name]) for field in expected_schema],
        schema=expected_schema,
    )
    row_addresses = canonical[DOCUMENT_ROWADDR].chunk(0)
    positions = canonical[DOCUMENT_POSITION].chunk(0)
    stable_ids = canonical[STABLE_ROW_ID].chunk(0)
    if row_addresses.null_count:
        msg = "document_rowaddr must not contain nulls"
        raise ValueError(msg)
    if positions.null_count:
        msg = "document_position must not contain nulls"
        raise ValueError(msg)

    unique_rowaddrs = int(pc.count_distinct(row_addresses, mode="only_valid").as_py())
    if unique_rowaddrs != canonical.num_rows:
        msg = "document_rowaddr values must be unique"
        raise ValueError(msg)
    unique_positions = int(pc.count_distinct(positions, mode="only_valid").as_py())
    if unique_positions != canonical.num_rows:
        msg = "document_position values must be unique"
        raise ValueError(msg)
    if canonical.num_rows > 1:
        strictly_increasing = pc.all(pc.less(positions.slice(0, canonical.num_rows - 1), positions.slice(1))).as_py()
        if strictly_increasing is not True:
            msg = "Coordinate plan rows must be strictly ordered by document_position"
            raise ValueError(msg)

    null_stable_ids = stable_ids.null_count
    if null_stable_ids and missing_key_policy != "null":
        msg = "stable_row_id contains nulls but missing_key_policy is not 'null'"
        raise ValueError(msg)
    non_null_stable_ids = canonical.num_rows - null_stable_ids
    unique_stable_ids = int(pc.count_distinct(stable_ids, mode="only_valid").as_py())
    stats = LanceCoordinatePlanStats(
        rows=canonical.num_rows,
        unique_document_rowaddrs=unique_rowaddrs,
        unique_document_positions=unique_positions,
        non_null_stable_row_ids=non_null_stable_ids,
        unique_stable_row_ids=unique_stable_ids,
        duplicate_stable_row_id_occurrences=non_null_stable_ids - unique_stable_ids,
        null_stable_row_ids=null_stable_ids,
        document_position_min=(int(positions[0].as_py()) if canonical.num_rows else None),
        document_position_max=(int(positions[-1].as_py()) if canonical.num_rows else None),
    )
    return canonical, stats


def validate_lance_coordinate_plan(
    table: pa.Table,
    *,
    missing_key_policy: MissingKeyPolicy = "error",
) -> LanceCoordinatePlanStats:
    """Validate schema, deterministic row order, uniqueness, and null policy."""
    _, stats = _canonicalize_and_validate(table, missing_key_policy)
    return stats


def canonical_lance_coordinate_plan_ipc_bytes(
    table: pa.Table,
    *,
    missing_key_policy: MissingKeyPolicy = "error",
) -> bytes:
    """Serialize a validated, single-chunk table to canonical Arrow IPC bytes."""
    canonical, _ = _canonicalize_and_validate(table, missing_key_policy)
    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(compression=None, use_legacy_format=False)
    with pa.ipc.new_stream(sink, canonical.schema, options=options) as writer:
        writer.write_table(canonical, max_chunksize=max(1, canonical.num_rows))
    return sink.getvalue().to_pybytes()


def lance_coordinate_plan_sha256(
    table: pa.Table,
    *,
    missing_key_policy: MissingKeyPolicy = "error",
) -> str:
    """Hash canonical Arrow IPC bytes, independent of input chunk boundaries."""
    raw = canonical_lance_coordinate_plan_ipc_bytes(table, missing_key_policy=missing_key_policy)
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _manifest_payload(
    *,
    identity: _CoordinatePlanArtifactIdentity,
    stats: LanceCoordinatePlanStats,
    coordinate_sha256: str,
    parquet_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_kind": _ARTIFACT_KIND,
        **identity.as_manifest_fields(),
        "coordinates": {
            **stats.as_dict(),
            "schema": _schema_manifest(_schema_for_policy(identity.missing_key_policy)),
            "canonical_ipc_sha256": coordinate_sha256,
        },
        "parquet": {
            "filename": parquet_path.name,
            "size_bytes": parquet_path.stat().st_size,
            "sha256": _file_sha256(parquet_path),
        },
    }


def _identity_from_manifest(manifest: Mapping[str, object]) -> _CoordinatePlanArtifactIdentity:
    try:
        document = manifest["document"]
        image = manifest["image"]
    except KeyError as exc:
        msg = "Coordinate plan manifest contains an invalid identity"
        raise ValueError(msg) from exc
    if not isinstance(document, Mapping) or not isinstance(image, Mapping):
        msg = "Coordinate plan manifest contains an invalid identity"
        raise TypeError(msg)
    try:
        return _CoordinatePlanArtifactIdentity(
            document_uri=document["uri"],
            document_version=document["version"],
            image_uri=image["uri"],
            image_version=image["version"],
            fragment_id=document["fragment_id"],
            sidecar_manifest_sha256=manifest["sidecar_manifest_sha256"],
            fragment_manifest_sha256=manifest["fragment_manifest_sha256"],
            missing_key_policy=manifest["missing_key_policy"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = "Coordinate plan manifest contains an invalid identity"
        raise ValueError(msg) from exc


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Coordinate plan manifest is unreadable: {path}"
        raise ValueError(msg) from exc
    if not isinstance(value, dict):
        msg = f"Coordinate plan manifest must contain a JSON object: {path}"
        raise TypeError(msg)
    return value


def _require_regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        msg = f"{name} must be a regular, non-symlink file: {path}"
        raise ValueError(msg)


def validate_existing_lance_coordinate_plan(
    parquet_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_identity: _CoordinatePlanArtifactIdentity | None = None,
) -> LanceCoordinatePlanArtifact:
    """Recompute and validate every logical and physical manifest field."""
    parquet = Path(parquet_path)
    manifest_file = Path(manifest_path)
    _require_regular_file(parquet, "Coordinate plan Parquet")
    _require_regular_file(manifest_file, "Coordinate plan manifest")
    manifest = _read_manifest(manifest_file)
    identity = _identity_from_manifest(manifest)
    if expected_identity is not None and identity != expected_identity:
        msg = "Existing coordinate plan identity does not match the requested identity"
        raise ValueError(msg)

    try:
        table = pq.read_table(parquet)
    except (OSError, pa.ArrowException) as exc:
        msg = f"Coordinate plan Parquet is unreadable: {parquet}"
        raise ValueError(msg) from exc
    canonical, stats = _canonicalize_and_validate(table, identity.missing_key_policy)
    coordinate_sha256 = hashlib.sha256(
        canonical_lance_coordinate_plan_ipc_bytes(canonical, missing_key_policy=identity.missing_key_policy)
    ).hexdigest()
    rebuilt = _manifest_payload(
        identity=identity,
        stats=stats,
        coordinate_sha256=coordinate_sha256,
        parquet_path=parquet,
    )
    if _canonical_json_bytes(manifest) != _canonical_json_bytes(rebuilt):
        msg = "Coordinate plan manifest does not reconcile with the stored Parquet artifact"
        raise ValueError(msg)
    return LanceCoordinatePlanArtifact(
        parquet_path=parquet,
        manifest_path=manifest_file,
        table=canonical,
        manifest=rebuilt,
        adopted=True,
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _validate_partial_parquet(
    parquet_path: Path,
    expected_table: pa.Table,
    missing_key_policy: MissingKeyPolicy,
) -> None:
    _require_regular_file(parquet_path, "Coordinate plan Parquet")
    try:
        existing = pq.read_table(parquet_path)
    except (OSError, pa.ArrowException) as exc:
        msg = f"Partial coordinate plan Parquet is unreadable: {parquet_path}"
        raise ValueError(msg) from exc
    canonical, _ = _canonicalize_and_validate(existing, missing_key_policy)
    if not canonical.equals(expected_table, check_metadata=True):
        msg = "Partial coordinate plan Parquet content does not match the requested table"
        raise ValueError(msg)


def publish_lance_coordinate_plan(
    table: pa.Table,
    parquet_path: str | Path,
    manifest_path: str | Path,
    *,
    identity: _CoordinatePlanArtifactIdentity,
) -> LanceCoordinatePlanArtifact:
    """Publish Parquet first and its validating JSON commit marker last.

    A complete matching pair is adopted after full revalidation. Any partial,
    conflicting, or corrupt existing state fails closed and is never replaced.
    """
    parquet = Path(parquet_path)
    manifest_file = Path(manifest_path)
    if parquet == manifest_file:
        msg = "parquet_path and manifest_path must be different"
        raise ValueError(msg)
    canonical, stats = _canonicalize_and_validate(table, identity.missing_key_policy)
    coordinate_sha256 = hashlib.sha256(
        canonical_lance_coordinate_plan_ipc_bytes(canonical, missing_key_policy=identity.missing_key_policy)
    ).hexdigest()

    parquet_exists = parquet.exists() or parquet.is_symlink()
    manifest_exists = manifest_file.exists() or manifest_file.is_symlink()
    if manifest_exists:
        if not parquet_exists:
            msg = "Coordinate plan manifest exists without its Parquet artifact"
            raise RuntimeError(msg)
        adopted = validate_existing_lance_coordinate_plan(
            parquet,
            manifest_file,
            expected_identity=identity,
        )
        existing_digest = adopted.manifest["coordinates"]
        if (
            not isinstance(existing_digest, Mapping)
            or existing_digest.get("canonical_ipc_sha256") != coordinate_sha256
        ):
            msg = "Existing coordinate plan content does not match the requested table"
            raise ValueError(msg)
        return adopted
    recovered_partial = parquet_exists
    if parquet_exists:
        _validate_partial_parquet(parquet, canonical, identity.missing_key_policy)

    if not parquet_exists:
        parquet.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{parquet.name}.",
            suffix=".tmp",
            dir=parquet.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            pq.write_table(
                canonical,
                temporary,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
            )
            _fsync_file(temporary)
            try:
                os.link(temporary, parquet)
            except FileExistsError:
                recovered_partial = True
            fsync_directory(parquet.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        _validate_partial_parquet(parquet, canonical, identity.missing_key_policy)

    manifest = _manifest_payload(
        identity=identity,
        stats=stats,
        coordinate_sha256=coordinate_sha256,
        parquet_path=parquet,
    )
    created = write_json_atomically_if_absent(
        manifest_file,
        manifest,
        separators=(",", ":"),
        sort_keys=True,
    )
    if not created:
        adopted = validate_existing_lance_coordinate_plan(
            parquet,
            manifest_file,
            expected_identity=identity,
        )
        existing_coordinates = adopted.manifest.get("coordinates")
        if not isinstance(existing_coordinates, Mapping) or (
            existing_coordinates.get("canonical_ipc_sha256") != coordinate_sha256
        ):
            msg = "Concurrent coordinate plan publication contains different coordinates"
            raise ValueError(msg)
        return adopted
    return LanceCoordinatePlanArtifact(
        parquet_path=parquet,
        manifest_path=manifest_file,
        table=canonical,
        manifest=manifest,
        adopted=recovered_partial,
    )


def publish_coordinate_plan(
    root: Path,
    table: pa.Table,
    identity: CoordinatePlanIdentity,
    *,
    allow_missing: bool = False,
) -> LanceCoordinatePlanTask:
    """Publish one deterministic coordinate artifact and return its task."""
    if not isinstance(root, Path):
        msg = f"root must be a pathlib.Path, got {type(root).__name__}"
        raise TypeError(msg)
    artifact_identity = identity._with_missing_policy(allow_missing)
    root = root.absolute()
    if root.is_symlink():
        msg = f"Coordinate plan root must not be a symlink: {root}"
        raise ValueError(msg)
    stem = f"fragment-{identity.fragment_id:08d}-{identity.identity_sha256()[:16]}"
    parquet_path = root / f"{stem}.parquet"
    manifest_path = root / f"{stem}.manifest.json"
    artifact = publish_lance_coordinate_plan(
        table,
        parquet_path,
        manifest_path,
        identity=artifact_identity,
    )
    coordinates = artifact.manifest["coordinates"]
    if not isinstance(coordinates, Mapping):  # pragma: no cover - constructed or validated above
        msg = "Published coordinate manifest has an invalid coordinates section"
        raise TypeError(msg)
    return LanceCoordinatePlanTask(
        dataset_name=identity.document_uri,
        data=str(artifact.parquet_path),
        manifest_path=str(artifact.manifest_path),
        _metadata={
            "source_files": [str(artifact.parquet_path), str(artifact.manifest_path)],
            "lance_coordinate_plan": {
                "fragment_id": identity.fragment_id,
                "rows": coordinates["rows"],
                "canonical_ipc_sha256": coordinates["canonical_ipc_sha256"],
                "adopted": artifact.adopted,
            },
        },
    )


def load_coordinate_plan(
    task_or_path: LanceCoordinatePlanTask | str | Path,
    manifest_path: str | Path | None = None,
    *,
    expected_identity: CoordinatePlanIdentity | None = None,
    allow_missing: bool | None = None,
) -> tuple[pa.Table, dict[str, object]]:
    """Fail closed on the artifact pair, then return Arrow data and manifest."""
    if isinstance(task_or_path, LanceCoordinatePlanTask):
        if manifest_path is not None:
            msg = "manifest_path must not be provided with LanceCoordinatePlanTask"
            raise ValueError(msg)
        parquet = Path(task_or_path.data)
        manifest_file = Path(task_or_path.manifest_path)
    else:
        parquet = Path(task_or_path)
        manifest_file = Path(manifest_path) if manifest_path is not None else parquet.with_suffix(".manifest.json")

    artifact = validate_existing_lance_coordinate_plan(parquet, manifest_file)
    artifact_identity = _identity_from_manifest(artifact.manifest)
    if expected_identity is not None:
        expected_without_policy = expected_identity._with_missing_policy(
            artifact_identity.missing_key_policy == "null"
        )
        if artifact_identity != expected_without_policy:
            msg = "Coordinate plan identity does not match expected_identity"
            raise ValueError(msg)
    if allow_missing is not None:
        if not isinstance(allow_missing, bool):
            msg = "allow_missing must be a boolean or None"
            raise TypeError(msg)
        actual_allow_missing = artifact_identity.missing_key_policy == "null"
        if actual_allow_missing != allow_missing:
            msg = "Coordinate plan missing-key policy does not match allow_missing"
            raise ValueError(msg)
    return artifact.table, artifact.manifest
