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

"""Durable Arrow payload overlays for one pinned Lance document fragment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pyarrow as pa

from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
    lance_coordinate_plan_schema,
    lance_coordinate_plan_sha256,
)
from nemo_curator.stages.interleaved.lance_payload_spool import (
    PayloadSpoolManifest,
    PayloadSpoolReader,
)
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.atomic_io import fsync_directory, write_json_atomically_if_absent
from nemo_curator.utils.uri import validate_credential_free_uri_identity

_ARTIFACT_KIND = "lance_payload_overlay"
_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_PAYLOAD_DIRECTORY = "payload"
_PAYLOAD_MANIFEST = "manifest.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping"
        raise TypeError(msg)
    return value


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def normalize_image_columns(value: Mapping[str, str] | Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Return one deterministic, validated source-to-destination mapping."""

    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    if not items:
        msg = "image_columns must not be empty"
        raise ValueError(msg)
    if any(
        not isinstance(source, str) or not source or not isinstance(destination, str) or not destination
        for source, destination in items
    ):
        msg = "image_columns must contain non-empty string names"
        raise ValueError(msg)
    sources = [source for source, _ in items]
    destinations = [destination for _, destination in items]
    if len(set(sources)) != len(sources) or len(set(destinations)) != len(destinations):
        msg = "image_columns source and destination names must be unique"
        raise ValueError(msg)
    return tuple(sorted(items))


@dataclass(frozen=True)
class LancePayloadOverlayIdentity:
    """Pinned inputs and exact row-conservation totals for one overlay."""

    document_uri: str
    document_version: int
    image_uri: str
    image_version: int
    fragment_id: int
    coordinate_plan_sha256: str
    coordinate_manifest_sha256: str
    payload_coordinate_sha256: str
    sidecar_manifest_sha256: str
    fragment_manifest_sha256: str
    overlay_config_sha256: str
    expected_document_rows: int
    expected_coordinate_rows: int
    expected_logical_rows: int
    expected_unique_rows: int
    expected_null_rows: int

    def __post_init__(self) -> None:
        for value, name in ((self.document_uri, "document_uri"), (self.image_uri, "image_uri")):
            if not isinstance(value, str) or not value:
                msg = f"{name} must be a non-empty string"
                raise ValueError(msg)
            validate_credential_free_uri_identity(value, name)
        _require_positive_integer(self.document_version, "document_version")
        _require_positive_integer(self.image_version, "image_version")
        _require_nonnegative_integer(self.fragment_id, "fragment_id")
        for value, name in (
            (self.coordinate_plan_sha256, "coordinate_plan_sha256"),
            (self.coordinate_manifest_sha256, "coordinate_manifest_sha256"),
            (self.payload_coordinate_sha256, "payload_coordinate_sha256"),
            (self.sidecar_manifest_sha256, "sidecar_manifest_sha256"),
            (self.fragment_manifest_sha256, "fragment_manifest_sha256"),
            (self.overlay_config_sha256, "overlay_config_sha256"),
        ):
            _require_sha256(value, name)
        _require_positive_integer(self.expected_document_rows, "expected_document_rows")
        _require_nonnegative_integer(self.expected_coordinate_rows, "expected_coordinate_rows")
        _require_nonnegative_integer(self.expected_logical_rows, "expected_logical_rows")
        _require_nonnegative_integer(self.expected_unique_rows, "expected_unique_rows")
        _require_nonnegative_integer(self.expected_null_rows, "expected_null_rows")
        if self.expected_coordinate_rows != self.expected_logical_rows + self.expected_null_rows:
            msg = "expected coordinate rows must equal logical plus null rows"
            raise ValueError(msg)
        if self.expected_unique_rows > self.expected_logical_rows:
            msg = "expected unique rows must not exceed logical rows"
            raise ValueError(msg)
        if self.expected_coordinate_rows > self.expected_document_rows:
            msg = "expected coordinate rows must not exceed document rows"
            raise ValueError(msg)

    @property
    def expected_duplicate_occurrences(self) -> int:
        return self.expected_logical_rows - self.expected_unique_rows

    def as_manifest_fields(self) -> dict[str, object]:
        return {
            "document": {
                "uri": self.document_uri,
                "version": self.document_version,
                "fragment_id": self.fragment_id,
                "rows": self.expected_document_rows,
            },
            "image": {"uri": self.image_uri, "version": self.image_version},
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "coordinate_manifest_sha256": self.coordinate_manifest_sha256,
            "payload_coordinate_sha256": self.payload_coordinate_sha256,
            "sidecar_manifest_sha256": self.sidecar_manifest_sha256,
            "fragment_manifest_sha256": self.fragment_manifest_sha256,
            "overlay_config_sha256": self.overlay_config_sha256,
            "counts": {
                "coordinate_rows": self.expected_coordinate_rows,
                "logical_rows": self.expected_logical_rows,
                "unique_rows": self.expected_unique_rows,
                "duplicate_occurrences": self.expected_duplicate_occurrences,
                "null_rows": self.expected_null_rows,
            },
        }


@dataclass(frozen=True)
class LancePayloadOverlayArtifact:
    """One validated overlay publication."""

    root: Path
    manifest_path: Path
    identity: LancePayloadOverlayIdentity
    image_columns: tuple[tuple[str, str], ...]
    payload: PayloadSpoolManifest
    producer_metrics: dict[str, int | float | bool]
    manifest: dict[str, object]
    manifest_sha256: str
    adopted: bool
    payload_verified: bool

    def iter_tables(self) -> Iterator[pa.Table]:
        """Stream hash- and schema-validated Arrow parts under the byte target."""

        return PayloadSpoolReader(self.payload).iter_tables()


@dataclass
class LancePayloadOverlayTask(FileGroupTask):
    """One validated overlay with a content-stable Curator source identity."""

    manifest_path: str = ""
    source_identity_sha256: str = ""
    data: list[str] = field(default_factory=list)

    def validate(self) -> bool:
        if not isinstance(self.data, list) or any(not isinstance(path, str) or not path for path in self.data):
            return False
        if not isinstance(self.manifest_path, str) or not self.manifest_path:
            return False
        return (
            isinstance(self.source_identity_sha256, str)
            and _SHA256_PATTERN.fullmatch(self.source_identity_sha256) is not None
        )

    def get_deterministic_id(self) -> str:
        return self.source_identity_sha256


def lance_payload_overlay_config_sha256(
    image_columns: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    payload_schema: pa.Schema,
    payload_window_bytes: int,
    bucket_rows: int,
) -> str:
    """Hash output columns and physical overlay layout settings."""

    columns = normalize_image_columns(image_columns)
    if not isinstance(payload_schema, pa.Schema):
        msg = "payload_schema must be a pyarrow.Schema"
        raise TypeError(msg)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "image_columns": [{"source": source, "destination": destination} for source, destination in columns],
        "payload_schema_sha256": _schema_sha256(payload_schema),
        "payload_window_bytes": _require_positive_integer(payload_window_bytes, "payload_window_bytes"),
        "bucket_rows": _require_positive_integer(bucket_rows, "bucket_rows"),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def lance_payload_overlay_root(
    output_root: str | Path,
    identity: LancePayloadOverlayIdentity,
) -> Path:
    """Return the deterministic publication directory for one overlay."""

    material = _canonical_json_bytes(identity.as_manifest_fields())
    digest = hashlib.sha256(material).hexdigest()[:16]
    return Path(output_root) / f"fragment-{identity.fragment_id:08d}-overlay-{digest}"


def lance_payload_overlay_source_identity_sha256(identity: LancePayloadOverlayIdentity) -> str:
    """Return the stable source-task identity for checkpointed consumers."""

    return hashlib.sha256(_canonical_json_bytes(identity.as_manifest_fields())).hexdigest()


def payload_coordinate_sha256(table: pa.Table) -> str:
    """Hash the non-null coordinate projection in document order."""

    if not isinstance(table, pa.Table):
        msg = "payload coordinates require a pyarrow.Table"
        raise TypeError(msg)
    missing = [name for name in (DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID) if name not in table.column_names]
    if missing:
        msg = f"payload coordinates are missing columns: {missing}"
        raise ValueError(msg)
    projected = table.select([DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID])
    if projected[STABLE_ROW_ID].null_count:
        projected = projected.filter(projected[STABLE_ROW_ID].is_valid())
    if projected.num_rows > 1:
        projected = projected.sort_by([(DOCUMENT_POSITION, "ascending")])
    schema = lance_coordinate_plan_schema(allow_missing=False)
    canonical = pa.Table.from_arrays(
        [projected[field.name].combine_chunks() for field in schema],
        schema=schema,
    )
    return lance_coordinate_plan_sha256(canonical, missing_key_policy="error")


def _identity_from_manifest(manifest: Mapping[str, object]) -> LancePayloadOverlayIdentity:
    try:
        document = _require_mapping(manifest["document"], "payload overlay document identity")
        image = _require_mapping(manifest["image"], "payload overlay image identity")
        counts = _require_mapping(manifest["counts"], "payload overlay counts")
        identity = LancePayloadOverlayIdentity(
            document_uri=document["uri"],
            document_version=document["version"],
            image_uri=image["uri"],
            image_version=image["version"],
            fragment_id=document["fragment_id"],
            coordinate_plan_sha256=manifest["coordinate_plan_sha256"],
            coordinate_manifest_sha256=manifest["coordinate_manifest_sha256"],
            payload_coordinate_sha256=manifest["payload_coordinate_sha256"],
            sidecar_manifest_sha256=manifest["sidecar_manifest_sha256"],
            fragment_manifest_sha256=manifest["fragment_manifest_sha256"],
            overlay_config_sha256=manifest["overlay_config_sha256"],
            expected_document_rows=document["rows"],
            expected_coordinate_rows=counts["coordinate_rows"],
            expected_logical_rows=counts["logical_rows"],
            expected_unique_rows=counts["unique_rows"],
            expected_null_rows=counts["null_rows"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = "payload overlay manifest identity is invalid"
        raise ValueError(msg) from exc
    if counts.get("duplicate_occurrences") != identity.expected_duplicate_occurrences:
        msg = "payload overlay duplicate count is inconsistent"
        raise ValueError(msg)
    return identity


def _image_columns_from_manifest(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        msg = "payload overlay image_columns must be a list"
        raise TypeError(msg)
    try:
        columns = [(item["source"], item["destination"]) for item in value if isinstance(item, Mapping)]
    except KeyError as exc:
        msg = "payload overlay image_columns are invalid"
        raise ValueError(msg) from exc
    if len(columns) != len(value):
        msg = "payload overlay image_columns are invalid"
        raise ValueError(msg)
    normalized = normalize_image_columns(columns)
    if tuple(columns) != normalized:
        msg = "payload overlay image_columns are not in deterministic order"
        raise ValueError(msg)
    return normalized


def _manifest_payload(
    identity: LancePayloadOverlayIdentity,
    image_columns: tuple[tuple[str, str], ...],
    payload: PayloadSpoolManifest,
    producer_metrics: Mapping[str, int | float | bool],
) -> dict[str, object]:
    return {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        **identity.as_manifest_fields(),
        "image_columns": [{"source": source, "destination": destination} for source, destination in image_columns],
        "payload": {
            "directory": _PAYLOAD_DIRECTORY,
            "manifest": f"{_PAYLOAD_DIRECTORY}/{_PAYLOAD_MANIFEST}",
            "manifest_sha256": payload.sha256,
            "schema_sha256": _schema_sha256(payload.schema),
            "target_bytes": payload.target_bytes,
            "bucket_rows": payload.bucket_rows,
            "sync_mode": payload.sync_mode,
            "rows": payload.total_rows,
            "arrow_nbytes": payload.total_arrow_nbytes,
            "files": len(payload.files),
            "oversized_rows": len(payload.oversized_rows),
        },
        "producer_metrics": dict(sorted(producer_metrics.items())),
    }


def _normalize_producer_metrics(value: Mapping[str, object] | None) -> dict[str, int | float | bool]:
    metrics: dict[str, int | float | bool] = {}
    for name, raw in (value or {}).items():
        if not isinstance(name, str) or not name:
            msg = "payload overlay producer metric names must be non-empty strings"
            raise ValueError(msg)
        if isinstance(raw, bool | int) or (isinstance(raw, float) and math.isfinite(raw)):
            metrics[name] = raw
        else:
            msg = f"payload overlay producer metric {name!r} must be a finite JSON number or boolean"
            raise TypeError(msg)
    return dict(sorted(metrics.items()))


def _metric_integer(metrics: Mapping[str, int | float | bool], name: str) -> int:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"payload overlay producer metric {name!r} must be a nonnegative integer"
        raise ValueError(msg)
    return value


def _metric_true(metrics: Mapping[str, int | float | bool], name: str) -> None:
    if metrics.get(name) is not True:
        msg = f"payload overlay producer metric {name!r} must be true"
        raise ValueError(msg)


def _validate_producer_metrics(
    metrics: Mapping[str, int | float | bool],
    identity: LancePayloadOverlayIdentity,
    payload: PayloadSpoolManifest,
) -> None:
    for name in ("stream_complete", "completion_order_output", "batch_stable_ids_sorted", "exact_operation_coverage"):
        _metric_true(metrics, name)
    expected_counts = {
        "logical_rows": identity.expected_logical_rows,
        "unique_rows": identity.expected_unique_rows,
        "null_rows_skipped": identity.expected_null_rows,
        "scatter_input_rows": identity.expected_logical_rows,
        "input_stable_rows": identity.expected_unique_rows,
        "stream_output_rows": identity.expected_unique_rows,
        "payload_take_rows": identity.expected_unique_rows,
        "take_rows": identity.expected_unique_rows,
        "spool_arrow_bytes": payload.total_arrow_nbytes,
        "payload_spool_arrow_bytes": payload.total_arrow_nbytes,
        "payload_spool_files": len(payload.files),
        "payload_spool_oversized_rows": len(payload.oversized_rows),
    }
    for name, expected in expected_counts.items():
        if _metric_integer(metrics, name) != expected:
            msg = f"payload overlay producer metric {name!r} does not reconcile with the artifact"
            raise ValueError(msg)
    planned = _metric_integer(metrics, "payload_batches_planned")
    emitted = _metric_integer(metrics, "payload_batches_emitted")
    read_calls = _metric_integer(metrics, "payload_read_calls")
    take_calls = _metric_integer(metrics, "take_calls")
    if len({planned, emitted, read_calls, take_calls}) != 1:
        msg = "payload overlay producer take-call metrics do not reconcile"
        raise ValueError(msg)
    if _metric_integer(metrics, "sparse_calls_avoided") != identity.expected_unique_rows - take_calls:
        msg = "payload overlay producer sparse-call accounting is invalid"
        raise ValueError(msg)
    payload_bytes = _metric_integer(metrics, "payload_bytes")
    actual_payload_bytes = _metric_integer(metrics, "actual_payload_bytes")
    spooled_payload_bytes = _metric_integer(metrics, "spooled_payload_bytes")
    if payload_bytes != actual_payload_bytes or spooled_payload_bytes < actual_payload_bytes:
        msg = "payload overlay producer payload-byte accounting is invalid"
        raise ValueError(msg)
    _metric_integer(metrics, "lance_read_iops")
    _metric_integer(metrics, "lance_read_bytes")


def _require_regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        msg = f"{name} must be a regular, non-symlink file: {path}"
        raise ValueError(msg)


def _validate_inventory(root: Path, payload: PayloadSpoolManifest) -> None:
    expected_root = {_MANIFEST_NAME, _PAYLOAD_DIRECTORY}
    if {item.name for item in root.iterdir()} != expected_root:
        msg = "payload overlay root contains partial or unexpected entries"
        raise RuntimeError(msg)
    payload_root = root / _PAYLOAD_DIRECTORY
    if payload_root.is_symlink() or not payload_root.is_dir():
        msg = "payload overlay payload directory must be a regular directory"
        raise ValueError(msg)
    expected_payload = {_PAYLOAD_MANIFEST, *(item.path.name for item in payload.files)}
    if {item.name for item in payload_root.iterdir()} != expected_payload:
        msg = "payload overlay payload directory contains partial or unexpected entries"
        raise RuntimeError(msg)
    for record in payload.files:
        _require_regular_file(record.path, "payload overlay Arrow part")
        if record.path.stat().st_size != record.file_bytes:
            msg = f"payload overlay Arrow part size mismatch: {record.path.name}"
            raise ValueError(msg)


def validate_lance_payload_overlay(  # noqa: C901, PLR0912, PLR0915
    root: str | Path,
    *,
    expected_identity: LancePayloadOverlayIdentity | None = None,
    expected_image_columns: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    verify_payload: bool = True,
) -> LancePayloadOverlayArtifact:
    """Validate identity, inventory, hashes, Arrow schema, buckets, and coordinates."""

    artifact_root = Path(root).absolute()
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        msg = f"payload overlay root must be a regular directory: {artifact_root}"
        raise ValueError(msg)
    manifest_path = artifact_root / _MANIFEST_NAME
    _require_regular_file(manifest_path, "payload overlay manifest")
    content = manifest_path.read_bytes()
    try:
        manifest = json.loads(content)
    except json.JSONDecodeError as exc:
        msg = "payload overlay manifest is not valid JSON"
        raise ValueError(msg) from exc
    if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != content:
        msg = "payload overlay manifest must be canonical JSON"
        raise ValueError(msg)
    if manifest.get("artifact_kind") != _ARTIFACT_KIND or manifest.get("schema_version") != _SCHEMA_VERSION:
        msg = "payload overlay manifest kind or schema version is invalid"
        raise ValueError(msg)
    identity = _identity_from_manifest(manifest)
    if expected_identity is not None and identity != expected_identity:
        msg = "existing payload overlay identity does not match the request"
        raise ValueError(msg)
    image_columns = _image_columns_from_manifest(manifest.get("image_columns"))
    if expected_image_columns is not None and image_columns != normalize_image_columns(expected_image_columns):
        msg = "existing payload overlay image columns do not match the request"
        raise ValueError(msg)
    raw_producer_metrics = manifest.get("producer_metrics")
    if not isinstance(raw_producer_metrics, Mapping):
        msg = "payload overlay producer_metrics section is invalid"
        raise TypeError(msg)
    producer_metrics = _normalize_producer_metrics(raw_producer_metrics)
    if dict(raw_producer_metrics) != producer_metrics:
        msg = "payload overlay producer_metrics are not in deterministic order"
        raise ValueError(msg)

    raw_payload = manifest.get("payload")
    if not isinstance(raw_payload, Mapping):
        msg = "payload overlay payload section is invalid"
        raise TypeError(msg)
    if (
        raw_payload.get("directory") != _PAYLOAD_DIRECTORY
        or raw_payload.get("manifest") != f"{_PAYLOAD_DIRECTORY}/{_PAYLOAD_MANIFEST}"
    ):
        msg = "payload overlay payload paths are invalid"
        raise ValueError(msg)
    payload_manifest_path = artifact_root / _PAYLOAD_DIRECTORY / _PAYLOAD_MANIFEST
    _require_regular_file(payload_manifest_path, "payload spool manifest")
    if _file_sha256(payload_manifest_path) != _require_sha256(
        raw_payload.get("manifest_sha256"), "payload manifest_sha256"
    ):
        msg = "payload overlay inner manifest SHA-256 mismatch"
        raise ValueError(msg)
    reader = PayloadSpoolReader(payload_manifest_path)
    payload = reader.manifest
    expected_payload_fields = {
        "directory": _PAYLOAD_DIRECTORY,
        "manifest": f"{_PAYLOAD_DIRECTORY}/{_PAYLOAD_MANIFEST}",
        "manifest_sha256": payload.sha256,
        "schema_sha256": _schema_sha256(payload.schema),
        "target_bytes": payload.target_bytes,
        "bucket_rows": payload.bucket_rows,
        "sync_mode": payload.sync_mode,
        "rows": payload.total_rows,
        "arrow_nbytes": payload.total_arrow_nbytes,
        "files": len(payload.files),
        "oversized_rows": len(payload.oversized_rows),
    }
    if dict(raw_payload) != expected_payload_fields:
        msg = "payload overlay payload metadata does not reconcile with its spool manifest"
        raise ValueError(msg)
    if payload.sync_mode != "fsync":
        msg = "durable payload overlays require fsync payload publication"
        raise ValueError(msg)
    if payload.total_rows != identity.expected_logical_rows:
        msg = "payload overlay row conservation does not match expected logical rows"
        raise ValueError(msg)
    required_columns = {DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID, *(source for source, _ in image_columns)}
    if set(payload.schema.names) != required_columns or len(payload.schema.names) != len(required_columns):
        msg = "payload overlay schema does not match its coordinate and image columns"
        raise TypeError(msg)
    if payload.stable_id_column != STABLE_ROW_ID or payload.document_position_column != DOCUMENT_POSITION:
        msg = "payload overlay coordinate-column metadata is invalid"
        raise ValueError(msg)
    expected_config = lance_payload_overlay_config_sha256(
        image_columns,
        payload_schema=payload.schema,
        payload_window_bytes=payload.target_bytes,
        bucket_rows=payload.bucket_rows,
    )
    if identity.overlay_config_sha256 != expected_config:
        msg = "payload overlay configuration digest is invalid"
        raise ValueError(msg)
    _validate_producer_metrics(producer_metrics, identity, payload)
    _validate_inventory(artifact_root, payload)

    if verify_payload:
        coordinate_tables: list[pa.Table] = []
        rows = 0
        for table in reader.iter_tables():
            rows += table.num_rows
            coordinate_tables.append(table.select([DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID]))
        if rows != identity.expected_logical_rows:
            msg = "payload overlay validated row count does not match expected logical rows"
            raise ValueError(msg)
        if coordinate_tables:
            coordinates = pa.concat_tables(coordinate_tables)
        else:
            coordinates = pa.Table.from_arrays(
                [pa.array([], type=pa.uint64()) for _ in range(3)],
                names=[DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID],
            )
        try:
            coordinate_digest = payload_coordinate_sha256(coordinates)
        except (TypeError, ValueError) as exc:
            msg = "payload overlay coordinates do not match the pinned coordinate plan"
            raise ValueError(msg) from exc
        if coordinate_digest != identity.payload_coordinate_sha256:
            msg = "payload overlay coordinates do not match the pinned coordinate plan"
            raise ValueError(msg)

    return LancePayloadOverlayArtifact(
        root=artifact_root,
        manifest_path=manifest_path,
        identity=identity,
        image_columns=image_columns,
        payload=payload,
        producer_metrics=producer_metrics,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        adopted=True,
        payload_verified=verify_payload,
    )


def publish_lance_payload_overlay(  # noqa: PLR0913
    attempt_root: str | Path,
    final_root: str | Path,
    *,
    identity: LancePayloadOverlayIdentity,
    image_columns: Mapping[str, str] | Sequence[tuple[str, str]],
    payload: PayloadSpoolManifest,
    producer_metrics: Mapping[str, object],
) -> LancePayloadOverlayArtifact:
    """Write the outer manifest last, then atomically publish one directory."""

    attempt = Path(attempt_root).absolute()
    final = Path(final_root).absolute()
    columns = normalize_image_columns(image_columns)
    normalized_metrics = _normalize_producer_metrics(producer_metrics)
    if attempt.is_symlink() or not attempt.is_dir():
        msg = "payload overlay attempt root must be a regular directory"
        raise ValueError(msg)
    if payload.root.absolute() != attempt / _PAYLOAD_DIRECTORY:
        msg = "payload spool must be located beneath the overlay attempt root"
        raise ValueError(msg)
    if payload.sync_mode != "fsync":
        msg = "durable payload overlays require an fsync payload spool"
        raise ValueError(msg)
    if payload.total_rows != identity.expected_logical_rows:
        msg = "payload spool rows do not match the overlay identity"
        raise ValueError(msg)
    if final.exists() or final.is_symlink():
        msg = f"payload overlay output already exists: {final}"
        raise FileExistsError(msg)
    manifest = _manifest_payload(identity, columns, payload, normalized_metrics)
    manifest_path = attempt / _MANIFEST_NAME
    if not write_json_atomically_if_absent(
        manifest_path,
        manifest,
        separators=(",", ":"),
        sort_keys=True,
    ):
        msg = "payload overlay manifest appeared before publication"
        raise FileExistsError(msg)
    fsync_directory(attempt)
    validate_lance_payload_overlay(
        attempt,
        expected_identity=identity,
        expected_image_columns=columns,
        verify_payload=True,
    )
    os.rename(attempt, final)
    fsync_directory(final.parent)
    artifact = validate_lance_payload_overlay(
        final,
        expected_identity=identity,
        expected_image_columns=columns,
        verify_payload=False,
    )
    return replace(artifact, adopted=False, payload_verified=True)


def lance_payload_overlay_task(
    artifact: LancePayloadOverlayArtifact,
    *,
    metadata: Mapping[str, object] | None = None,
) -> LancePayloadOverlayTask:
    """Expose Arrow parts and the durable overlay identity as a Curator task."""

    if not artifact.payload_verified:
        msg = "a payload overlay task requires full payload verification"
        raise ValueError(msg)

    paths = [str(record.path) for record in artifact.payload.files]
    output_metadata = dict(metadata or {})
    output_metadata["source_files"] = [*paths, str(artifact.payload.path), str(artifact.manifest_path)]
    output_metadata["format"] = "arrow"
    output_metadata["lance_payload_overlay"] = {
        "manifest_path": str(artifact.manifest_path),
        "manifest_sha256": artifact.manifest_sha256,
        "payload_manifest_path": str(artifact.payload.path),
        "payload_manifest_sha256": artifact.payload.sha256,
        "coordinate_plan_sha256": artifact.identity.coordinate_plan_sha256,
        "coordinate_manifest_sha256": artifact.identity.coordinate_manifest_sha256,
        "payload_coordinate_sha256": artifact.identity.payload_coordinate_sha256,
        "overlay_config_sha256": artifact.identity.overlay_config_sha256,
        "document_version": artifact.identity.document_version,
        "image_version": artifact.identity.image_version,
        "fragment_id": artifact.identity.fragment_id,
        "document_rows": artifact.identity.expected_document_rows,
        "coordinate_rows": artifact.identity.expected_coordinate_rows,
        "logical_rows": artifact.identity.expected_logical_rows,
        "unique_rows": artifact.identity.expected_unique_rows,
        "duplicate_occurrences": artifact.identity.expected_duplicate_occurrences,
        "null_rows": artifact.identity.expected_null_rows,
        "payload_arrow_nbytes": artifact.payload.total_arrow_nbytes,
        "payload_files": len(artifact.payload.files),
        "adopted": artifact.adopted,
        "producer_metrics": dict(artifact.producer_metrics),
    }
    return LancePayloadOverlayTask(
        dataset_name=artifact.identity.document_uri,
        data=paths,
        manifest_path=str(artifact.manifest_path),
        source_identity_sha256=lance_payload_overlay_source_identity_sha256(artifact.identity),
        _metadata=output_metadata,
    )
