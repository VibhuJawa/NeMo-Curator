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

"""Durable, deterministic Parquet artifacts for one Lance payload patch."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.atomic_io import fsync_directory, write_json_atomically_if_absent
from nemo_curator.utils.uri import validate_credential_free_uri_identity

_ARTIFACT_KIND = "lance_payload_patch"
_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_HASH_CHUNK_BYTES = 1024 * 1024


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _schema_bytes(schema: pa.Schema) -> bytes:
    return schema.serialize().to_pybytes()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(_schema_bytes(schema)).hexdigest()


def _schema_manifest(schema: pa.Schema) -> dict[str, object]:
    return {
        "ipc_base64": base64.b64encode(_schema_bytes(schema)).decode("ascii"),
        "sha256": _schema_sha256(schema),
        "fields": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema],
    }


def _schema_from_manifest(value: object) -> pa.Schema:
    if not isinstance(value, Mapping):
        msg = "Patch manifest schema section is invalid"
        raise TypeError(msg)
    encoded = value.get("ipc_base64")
    if not isinstance(encoded, str):
        msg = "Patch manifest schema encoding is invalid"
        raise TypeError(msg)
    try:
        schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(encoded, validate=True)))
    except (ValueError, pa.ArrowException) as exc:
        msg = "Patch manifest schema encoding is invalid"
        raise ValueError(msg) from exc
    if _canonical_json_bytes(_schema_manifest(schema)) != _canonical_json_bytes(dict(value)):
        msg = "Patch manifest schema does not reconcile with its encoded Arrow schema"
        raise ValueError(msg)
    return schema


def _part_name(ordinal: int) -> str:
    return f"part-{ordinal:08d}.parquet"


@dataclass(frozen=True)
class LancePatchArtifactIdentity:
    """Immutable source identity and expected conservation totals."""

    document_uri: str
    document_version: int
    image_uri: str
    image_version: int
    fragment_id: int
    coordinate_plan_sha256: str
    patch_config_sha256: str
    expected_rows: int

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
        _require_sha256(self.coordinate_plan_sha256, "coordinate_plan_sha256")
        _require_sha256(self.patch_config_sha256, "patch_config_sha256")
        _require_positive_integer(self.expected_rows, "expected_rows")

    def as_manifest_fields(self) -> dict[str, object]:
        return {
            "document": {
                "uri": self.document_uri,
                "version": self.document_version,
                "fragment_id": self.fragment_id,
            },
            "image": {"uri": self.image_uri, "version": self.image_version},
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "patch_config_sha256": self.patch_config_sha256,
        }


@dataclass(frozen=True)
class LancePatchPart:
    """Validated metadata for one ordered Parquet patch part."""

    ordinal: int
    path: Path
    row_start: int
    row_stop: int
    document_position_min: int
    document_position_max: int
    size_bytes: int
    sha256: str
    schema_sha256: str

    @property
    def rows(self) -> int:
        return self.row_stop - self.row_start

    def as_manifest(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "filename": self.path.name,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "rows": self.rows,
            "document_position_min": self.document_position_min,
            "document_position_max": self.document_position_max,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
        }


@dataclass(frozen=True)
class LancePatchArtifact:
    """One fully validated manifest and its ordered patch parts."""

    root: Path
    manifest_path: Path
    identity: LancePatchArtifactIdentity
    schema: pa.Schema
    document_position_column: str
    oversized_sample_count: int
    parts: tuple[LancePatchPart, ...]
    manifest: dict[str, object]
    manifest_sha256: str
    adopted: bool


def _identity_from_manifest(manifest: Mapping[str, object]) -> LancePatchArtifactIdentity:
    document = manifest.get("document")
    image = manifest.get("image")
    if not isinstance(document, Mapping) or not isinstance(image, Mapping):
        msg = "Patch manifest dataset identity is invalid"
        raise TypeError(msg)
    try:
        return LancePatchArtifactIdentity(
            document_uri=document["uri"],
            document_version=document["version"],
            image_uri=image["uri"],
            image_version=image["version"],
            fragment_id=document["fragment_id"],
            coordinate_plan_sha256=manifest["coordinate_plan_sha256"],
            patch_config_sha256=manifest["patch_config_sha256"],
            expected_rows=manifest["total_rows"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = "Patch manifest dataset identity is invalid"
        raise ValueError(msg) from exc


def _validate_position_column(schema: pa.Schema, column: str) -> None:
    if not isinstance(column, str) or not column:
        msg = "document_position_column must be a non-empty string"
        raise ValueError(msg)
    index = schema.get_field_index(column)
    if index < 0:
        msg = f"Patch schema is missing document-position column {column!r}"
        raise ValueError(msg)
    field = schema.field(index)
    if field.type != pa.uint64() or field.nullable:
        msg = f"Patch document-position column {column!r} must be non-nullable uint64"
        raise TypeError(msg)


def _position_interval(
    table: pa.Table,
    column: str,
    *,
    previous_maximum: int | None,
) -> tuple[int, int]:
    positions = table[column].combine_chunks()
    if positions.null_count:
        msg = "Patch document positions must not contain nulls"
        raise ValueError(msg)
    minimum = int(positions[0].as_py())
    maximum = int(positions[-1].as_py())
    if table.num_rows > 1:
        increasing = pc.all(pc.less(positions.slice(0, table.num_rows - 1), positions.slice(1))).as_py()
        if increasing is not True:
            msg = "Patch rows must be strictly ordered by document_position"
            raise ValueError(msg)
    if previous_maximum is not None and minimum <= previous_maximum:
        msg = "Patch parts must be strictly ordered by document_position"
        raise ValueError(msg)
    return minimum, maximum


def _manifest_payload(
    identity: LancePatchArtifactIdentity,
    schema: pa.Schema,
    document_position_column: str,
    parts: tuple[LancePatchPart, ...],
    oversized_sample_count: int,
) -> dict[str, object]:
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        **identity.as_manifest_fields(),
        "oversized_sample_count": oversized_sample_count,
        "document_position_column": document_position_column,
        "schema": _schema_manifest(schema),
        "parts": [part.as_manifest() for part in parts],
        "total_rows": sum(part.rows for part in parts),
        "total_size_bytes": sum(part.size_bytes for part in parts),
    }


def _require_regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        msg = f"{name} must be a regular, non-symlink file: {path}"
        raise ValueError(msg)


def _read_manifest(path: Path) -> dict[str, object]:
    _require_regular_file(path, "Patch manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Patch manifest is unreadable: {path}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = "Patch manifest must contain a JSON object"
        raise TypeError(msg)
    return payload


def _part_from_table(  # noqa: PLR0913
    path: Path,
    table: pa.Table,
    *,
    ordinal: int,
    row_start: int,
    document_position_column: str,
    previous_maximum: int | None,
    schema_sha256: str,
) -> LancePatchPart:
    minimum, maximum = _position_interval(
        table,
        document_position_column,
        previous_maximum=previous_maximum,
    )
    return LancePatchPart(
        ordinal=ordinal,
        path=path,
        row_start=row_start,
        row_stop=row_start + table.num_rows,
        document_position_min=minimum,
        document_position_max=maximum,
        size_bytes=path.stat().st_size,
        sha256=_file_sha256(path),
        schema_sha256=schema_sha256,
    )


def validate_lance_patch_artifact(  # noqa: C901, PLR0912, PLR0915
    root: str | Path,
    *,
    expected_identity: LancePatchArtifactIdentity | None = None,
    expected_schema: pa.Schema | None = None,
    expected_oversized_sample_count: int | None = None,
) -> LancePatchArtifact:
    """Recompute and fail closed on every manifest, part, and ordering field."""

    artifact_root = Path(root).absolute()
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        msg = f"Patch artifact root must be a regular directory: {artifact_root}"
        raise ValueError(msg)
    manifest_path = artifact_root / _MANIFEST_NAME
    manifest = _read_manifest(manifest_path)
    if manifest.get("artifact_kind") != _ARTIFACT_KIND or manifest.get("schema_version") != _SCHEMA_VERSION:
        msg = "Patch manifest kind or schema version is invalid"
        raise ValueError(msg)
    identity = _identity_from_manifest(manifest)
    if expected_identity is not None and identity != expected_identity:
        msg = "Existing patch artifact identity does not match the requested identity"
        raise ValueError(msg)
    oversized_sample_count = _require_nonnegative_integer(
        manifest.get("oversized_sample_count"),
        "manifest oversized_sample_count",
    )
    if oversized_sample_count > identity.expected_rows:
        msg = "Patch manifest oversized_sample_count exceeds total_rows"
        raise ValueError(msg)
    if expected_oversized_sample_count is not None:
        expected_oversized_sample_count = _require_nonnegative_integer(
            expected_oversized_sample_count,
            "expected_oversized_sample_count",
        )
        if oversized_sample_count != expected_oversized_sample_count:
            msg = "Existing patch artifact oversized-sample count does not match the request"
            raise ValueError(msg)
    schema = _schema_from_manifest(manifest.get("schema"))
    if expected_schema is not None and not schema.equals(expected_schema, check_metadata=True):
        msg = "Existing patch artifact schema does not match the requested schema"
        raise TypeError(msg)
    position_column = manifest.get("document_position_column")
    if not isinstance(position_column, str):
        msg = "Patch manifest document_position_column is invalid"
        raise TypeError(msg)
    _validate_position_column(schema, position_column)
    raw_parts = manifest.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        msg = "Patch manifest parts must be a non-empty list"
        raise ValueError(msg)

    schema_digest = _schema_sha256(schema)
    parts: list[LancePatchPart] = []
    row_start = 0
    previous_maximum: int | None = None
    for ordinal, raw in enumerate(raw_parts):
        if not isinstance(raw, Mapping):
            msg = "Patch manifest contains an invalid part record"
            raise TypeError(msg)
        filename = raw.get("filename")
        if filename != _part_name(ordinal):
            msg = "Patch manifest part names or ordinals are not deterministic"
            raise ValueError(msg)
        path = artifact_root / filename
        _require_regular_file(path, "Patch Parquet part")
        if path.stat().st_size != raw.get("size_bytes"):
            msg = f"Patch part size does not match manifest: {filename}"
            raise ValueError(msg)
        if _file_sha256(path) != raw.get("sha256"):
            msg = f"Patch part SHA-256 does not match manifest: {filename}"
            raise ValueError(msg)
        try:
            table = pq.read_table(path)
        except (OSError, pa.ArrowException) as exc:
            msg = f"Patch Parquet part is unreadable: {filename}"
            raise ValueError(msg) from exc
        if not table.schema.equals(schema, check_metadata=True):
            msg = f"Patch part schema does not match manifest: {filename}"
            raise TypeError(msg)
        rebuilt = _part_from_table(
            path,
            table,
            ordinal=ordinal,
            row_start=row_start,
            document_position_column=position_column,
            previous_maximum=previous_maximum,
            schema_sha256=schema_digest,
        )
        if _canonical_json_bytes(rebuilt.as_manifest()) != _canonical_json_bytes(dict(raw)):
            msg = f"Patch part metadata does not reconcile with Parquet: {filename}"
            raise ValueError(msg)
        parts.append(rebuilt)
        row_start = rebuilt.row_stop
        previous_maximum = rebuilt.document_position_max

    expected_names = {_MANIFEST_NAME, *(part.path.name for part in parts)}
    actual_names = {entry.name for entry in artifact_root.iterdir()}
    if actual_names != expected_names:
        msg = "Patch artifact contains partial, temporary, or unexpected files"
        raise RuntimeError(msg)
    rebuilt_manifest = _manifest_payload(
        identity,
        schema,
        position_column,
        tuple(parts),
        oversized_sample_count,
    )
    if _canonical_json_bytes(manifest) != _canonical_json_bytes(rebuilt_manifest):
        msg = "Patch manifest does not reconcile with stored Parquet parts"
        raise ValueError(msg)
    if row_start != identity.expected_rows:
        msg = "Patch artifact row conservation does not match expected_rows"
        raise ValueError(msg)
    if parts[0].document_position_min != 0 or parts[-1].document_position_max != identity.expected_rows - 1:
        msg = "Patch artifact document positions do not cover exactly 0..expected_rows-1"
        raise ValueError(msg)
    content = _canonical_json_bytes(rebuilt_manifest)
    return LancePatchArtifact(
        root=artifact_root,
        manifest_path=manifest_path,
        identity=identity,
        schema=schema,
        document_position_column=position_column,
        oversized_sample_count=oversized_sample_count,
        parts=tuple(parts),
        manifest=rebuilt_manifest,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        adopted=True,
    )


def _publish_parquet_part(table: pa.Table, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            row_group_size=table.num_rows,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _task_from_artifact(artifact: LancePatchArtifact) -> FileGroupTask:
    patch_paths = [str(part.path) for part in artifact.parts]
    metadata = {
        "source_files": [*patch_paths, str(artifact.manifest_path)],
        "format": "parquet",
        "lance_patch_artifact": {
            "manifest_path": str(artifact.manifest_path),
            "manifest_sha256": artifact.manifest_sha256,
            "coordinate_plan_sha256": artifact.identity.coordinate_plan_sha256,
            "patch_config_sha256": artifact.identity.patch_config_sha256,
            "document_version": artifact.identity.document_version,
            "image_version": artifact.identity.image_version,
            "fragment_id": artifact.identity.fragment_id,
            "rows": artifact.identity.expected_rows,
            "size_bytes": sum(part.size_bytes for part in artifact.parts),
            "oversized_sample_count": artifact.oversized_sample_count,
            "adopted": artifact.adopted,
        },
    }
    return FileGroupTask(
        dataset_name=artifact.identity.document_uri,
        data=patch_paths,
        _metadata=metadata,
    )


class LancePatchArtifactWriter:
    """Publish already-bounded ordered Arrow tables without retaining them."""

    def __init__(
        self,
        root: str | Path,
        schema: pa.Schema,
        identity: LancePatchArtifactIdentity,
        *,
        document_position_column: str = "document_position",
        expected_oversized_sample_count: int | None = None,
    ) -> None:
        if not isinstance(schema, pa.Schema):
            msg = "schema must be a pyarrow.Schema"
            raise TypeError(msg)
        if not isinstance(identity, LancePatchArtifactIdentity):
            msg = "identity must be a LancePatchArtifactIdentity"
            raise TypeError(msg)
        _validate_position_column(schema, document_position_column)
        self.root = Path(root).absolute()
        self.schema = schema
        self.identity = identity
        self.document_position_column = document_position_column
        self.expected_oversized_sample_count = (
            None
            if expected_oversized_sample_count is None
            else _require_nonnegative_integer(
                expected_oversized_sample_count,
                "expected_oversized_sample_count",
            )
        )
        self._schema_sha256 = _schema_sha256(schema)
        self._parts: list[LancePatchPart] = []
        self._rows = 0
        self._oversized_sample_count = 0
        self._last_document_position: int | None = None
        self._artifact: LancePatchArtifact | None = None
        self._task: FileGroupTask | None = None

        if self.root.exists() or self.root.is_symlink():
            if self.root.is_symlink() or not self.root.is_dir():
                msg = f"Patch output root is not a regular directory: {self.root}"
                raise ValueError(msg)
            manifest_path = self.root / _MANIFEST_NAME
            if not manifest_path.exists() and not manifest_path.is_symlink():
                msg = "Patch artifact publication is partial; a complete manifest is required for adoption"
                raise RuntimeError(msg)
            self._artifact = validate_lance_patch_artifact(
                self.root,
                expected_identity=identity,
                expected_schema=schema,
                expected_oversized_sample_count=self.expected_oversized_sample_count,
            )
            if self._artifact.document_position_column != document_position_column:
                msg = "Existing patch artifact document-position column does not match the request"
                raise ValueError(msg)
            self._task = _task_from_artifact(self._artifact)
            return

        self.root.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.root.mkdir(exist_ok=False)
        except FileExistsError:
            msg = "Patch output root appeared concurrently; retry adoption explicitly"
            raise FileExistsError(msg) from None
        fsync_directory(self.root.parent)

    @property
    def adopted(self) -> bool:
        return self._artifact is not None and self._artifact.adopted

    def append(self, table: pa.Table, *, oversized_sample_count: int = 0) -> LancePatchPart:
        """Validate and immediately publish one bounded contiguous part."""

        if self._artifact is not None:
            msg = "Patch artifact is already complete; call finish() to use the adopted output"
            raise RuntimeError(msg)
        if not isinstance(table, pa.Table):
            msg = "Patch append requires a pyarrow.Table"
            raise TypeError(msg)
        if not table.schema.equals(self.schema, check_metadata=True):
            msg = "Patch append schema does not match the configured schema"
            raise TypeError(msg)
        if table.num_rows <= 0:
            msg = "Patch append tables must contain at least one row"
            raise ValueError(msg)
        oversized_sample_count = _require_nonnegative_integer(
            oversized_sample_count,
            "oversized_sample_count",
        )
        if oversized_sample_count > table.num_rows:
            msg = "oversized_sample_count must not exceed the appended table row count"
            raise ValueError(msg)
        if self._rows + table.num_rows > self.identity.expected_rows:
            msg = "Patch append would exceed expected_rows"
            raise ValueError(msg)
        if (self.root / _MANIFEST_NAME).exists() or (self.root / _MANIFEST_NAME).is_symlink():
            msg = "Patch manifest appeared before writer finalization"
            raise FileExistsError(msg)
        minimum, maximum = _position_interval(
            table,
            self.document_position_column,
            previous_maximum=self._last_document_position,
        )
        expected_minimum = self._rows
        expected_maximum = self._rows + table.num_rows - 1
        if minimum != expected_minimum or maximum != expected_maximum:
            msg = "Patch document positions must cover exactly 0..expected_rows-1"
            raise ValueError(msg)
        ordinal = len(self._parts)
        path = self.root / _part_name(ordinal)
        _publish_parquet_part(table, path)
        part = LancePatchPart(
            ordinal=ordinal,
            path=path,
            row_start=self._rows,
            row_stop=self._rows + table.num_rows,
            document_position_min=minimum,
            document_position_max=maximum,
            size_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
            schema_sha256=self._schema_sha256,
        )
        self._parts.append(part)
        self._rows = part.row_stop
        self._oversized_sample_count += oversized_sample_count
        self._last_document_position = part.document_position_max
        return part

    def finish(self) -> FileGroupTask:
        """Publish the validating manifest last and return one patch file task."""

        if self._task is not None:
            return self._task
        if self._rows != self.identity.expected_rows:
            msg = f"Patch row conservation failed: wrote {self._rows}, expected {self.identity.expected_rows}"
            raise RuntimeError(msg)
        if (
            self.expected_oversized_sample_count is not None
            and self._oversized_sample_count != self.expected_oversized_sample_count
        ):
            msg = "Patch oversized-sample conservation does not match the expected count"
            raise RuntimeError(msg)
        parts = tuple(self._parts)
        manifest = _manifest_payload(
            self.identity,
            self.schema,
            self.document_position_column,
            parts,
            self._oversized_sample_count,
        )
        manifest_path = self.root / _MANIFEST_NAME
        created = write_json_atomically_if_absent(
            manifest_path,
            manifest,
            separators=(",", ":"),
            sort_keys=True,
        )
        if not created:
            msg = "Patch manifest already exists; refusing to overwrite concurrent output"
            raise FileExistsError(msg)
        validated = validate_lance_patch_artifact(
            self.root,
            expected_identity=self.identity,
            expected_schema=self.schema,
            expected_oversized_sample_count=self._oversized_sample_count,
        )
        self._artifact = replace(validated, adopted=False)
        self._task = _task_from_artifact(self._artifact)
        return self._task
