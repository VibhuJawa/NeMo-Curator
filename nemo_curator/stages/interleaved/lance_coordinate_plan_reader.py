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

"""Checkpointable source for durable Lance coordinate-plan artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    CoordinatePlanIdentity,
    LanceCoordinatePlanArtifact,
    LanceCoordinatePlanTask,
    MissingKeyPolicy,
    validate_existing_lance_coordinate_plan,
)
from nemo_curator.tasks import EmptyTask
from nemo_curator.utils.uri import validate_credential_free_uri_identity

_PARQUET_SUFFIX = ".parquet"
_MANIFEST_SUFFIX = ".manifest.json"
_PLAN_STEM_PATTERN = re.compile(r"fragment-(?P<fragment_id>[0-9]{8,})-(?P<identity_prefix>[0-9a-f]{16})")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_optional_positive_integer(value: int | None, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        msg = f"{name} must be a positive integer or None"
        raise ValueError(msg)


def _require_optional_sha256(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None):
        msg = f"{name} must be a lowercase SHA-256 digest or None"
        raise ValueError(msg)


def _normalize_expected_fragment_ids(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "expected_fragment_ids must be a sequence of nonnegative integers or None"
        raise TypeError(msg)
    fragment_ids = tuple(value)
    if any(
        isinstance(fragment_id, bool) or not isinstance(fragment_id, int) or fragment_id < 0
        for fragment_id in fragment_ids
    ):
        msg = "expected_fragment_ids must contain only nonnegative integers"
        raise ValueError(msg)
    if len(set(fragment_ids)) != len(fragment_ids):
        msg = "expected_fragment_ids must not contain duplicates"
        raise ValueError(msg)
    return tuple(sorted(fragment_ids))


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"Validated coordinate plan manifest {name} section is invalid"
        raise TypeError(msg)
    return value


def _identity_from_artifact(artifact: LanceCoordinatePlanArtifact) -> tuple[CoordinatePlanIdentity, MissingKeyPolicy]:
    manifest = artifact.manifest
    document = _require_mapping(manifest.get("document"), "document")
    image = _require_mapping(manifest.get("image"), "image")
    try:
        identity = CoordinatePlanIdentity(
            document_uri=document["uri"],
            document_version=document["version"],
            image_uri=image["uri"],
            image_version=image["version"],
            fragment_id=document["fragment_id"],
            sidecar_manifest_sha256=manifest["sidecar_manifest_sha256"],
            fragment_manifest_sha256=manifest["fragment_manifest_sha256"],
        )
        missing_key_policy = manifest["missing_key_policy"]
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - durable validator rejects this first
        msg = "Validated coordinate plan manifest identity is invalid"
        raise ValueError(msg) from exc
    if missing_key_policy not in {"error", "null"}:  # pragma: no cover - durable validator rejects this first
        msg = "Validated coordinate plan manifest missing-key policy is invalid"
        raise ValueError(msg)
    return identity, missing_key_policy


def _source_identity_sha256(
    artifact: LanceCoordinatePlanArtifact,
    identity: CoordinatePlanIdentity,
    missing_key_policy: MissingKeyPolicy,
) -> str:
    coordinates = _require_mapping(artifact.manifest.get("coordinates"), "coordinates")
    coordinate_sha256 = coordinates.get("canonical_ipc_sha256")
    if not isinstance(coordinate_sha256, str) or _SHA256_PATTERN.fullmatch(coordinate_sha256) is None:
        msg = "Validated coordinate plan coordinate digest is invalid"
        raise ValueError(msg)
    material = {
        "coordinate_sha256": coordinate_sha256,
        "identity_sha256": identity.identity_sha256(),
        "missing_key_policy": missing_key_policy,
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class LanceCoordinatePlanReader(ProcessingStage[EmptyTask, LanceCoordinatePlanTask]):
    """Enumerate and validate one shared coordinate-plan publication root.

    ``expected_fragment_ids`` pins publication completeness when supplied;
    ``None`` retains unpinned inventory discovery for programmatic inspection.
    """

    plan_root: str
    document_uri: str | None = None
    document_version: int | None = None
    image_uri: str | None = None
    image_version: int | None = None
    sidecar_manifest_sha256: str | None = None
    fragment_manifest_sha256: str | None = None
    missing_key_policy: MissingKeyPolicy | None = None
    expected_fragment_ids: Sequence[int] | None = field(default=None, repr=False)
    name: str = "lance_coordinate_plan_reader"

    is_source_stage = True
    is_resumable = True

    def __post_init__(self) -> None:
        if not isinstance(self.plan_root, str) or not self.plan_root:
            msg = "plan_root must be a non-empty absolute local filesystem path"
            raise ValueError(msg)
        root = Path(self.plan_root)
        if not root.is_absolute():
            msg = "plan_root must be an absolute local filesystem path"
            raise ValueError(msg)
        self.plan_root = str(root)
        for value, name in ((self.document_uri, "document_uri"), (self.image_uri, "image_uri")):
            if value is not None:
                if not isinstance(value, str) or not value:
                    msg = f"{name} must be a non-empty string or None"
                    raise ValueError(msg)
                validate_credential_free_uri_identity(value, name)
        _require_optional_positive_integer(self.document_version, "document_version")
        _require_optional_positive_integer(self.image_version, "image_version")
        _require_optional_sha256(self.sidecar_manifest_sha256, "sidecar_manifest_sha256")
        _require_optional_sha256(self.fragment_manifest_sha256, "fragment_manifest_sha256")
        if self.missing_key_policy not in {None, "error", "null"}:
            msg = f"Unsupported missing_key_policy: {self.missing_key_policy!r}"
            raise ValueError(msg)
        self.expected_fragment_ids = _normalize_expected_fragment_ids(self.expected_fragment_ids)

    def ray_stage_spec(self) -> dict[str, object]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def num_workers(self) -> int | None:
        return 1

    def _inventory(self) -> tuple[tuple[str, ...], dict[str, Path], dict[str, Path]]:
        root = Path(self.plan_root)
        if root.is_symlink() or not root.is_dir():
            msg = f"plan_root must be an existing regular directory, not a symlink: {root}"
            raise ValueError(msg)
        entries = sorted(root.iterdir(), key=lambda path: path.name)
        parquet: dict[str, Path] = {}
        manifests: dict[str, Path] = {}
        for path in entries:
            if path.is_symlink() or not path.is_file():
                msg = f"Coordinate plan root contains a stray non-regular entry: {path}"
                raise ValueError(msg)
            if path.name.endswith(_MANIFEST_SUFFIX):
                stem = path.name[: -len(_MANIFEST_SUFFIX)]
                destination = manifests
            elif path.name.endswith(_PARQUET_SUFFIX):
                stem = path.name[: -len(_PARQUET_SUFFIX)]
                destination = parquet
            else:
                msg = f"Coordinate plan root contains a stray artifact: {path}"
                raise ValueError(msg)
            if _PLAN_STEM_PATTERN.fullmatch(stem) is None:
                msg = f"Coordinate plan artifact has an invalid filename: {path}"
                raise ValueError(msg)
            if stem in destination:  # pragma: no cover - directory entries cannot share a filename
                msg = f"Coordinate plan root contains a duplicate artifact stem: {stem}"
                raise ValueError(msg)
            destination[stem] = path
        names = tuple(path.name for path in entries)
        return names, parquet, manifests

    def _validate_pins(self, identity: CoordinatePlanIdentity, missing_key_policy: MissingKeyPolicy) -> None:
        expected = {
            "document_uri": self.document_uri,
            "document_version": self.document_version,
            "image_uri": self.image_uri,
            "image_version": self.image_version,
            "sidecar_manifest_sha256": self.sidecar_manifest_sha256,
            "fragment_manifest_sha256": self.fragment_manifest_sha256,
        }
        for name, value in expected.items():
            if value is not None and getattr(identity, name) != value:
                msg = f"Coordinate plan {name} does not match the reader pin"
                raise ValueError(msg)
        if self.missing_key_policy is not None and missing_key_policy != self.missing_key_policy:
            msg = "Coordinate plan missing_key_policy does not match the reader pin"
            raise ValueError(msg)

    def _task_for_pair(self, stem: str, parquet: Path, manifest: Path) -> tuple[int, LanceCoordinatePlanTask]:
        artifact = validate_existing_lance_coordinate_plan(parquet, manifest)
        identity, missing_key_policy = _identity_from_artifact(artifact)
        filename_match = _PLAN_STEM_PATTERN.fullmatch(stem)
        if filename_match is None:  # pragma: no cover - inventory validates this first
            msg = f"Coordinate plan artifact has an invalid stem: {stem}"
            raise ValueError(msg)
        filename_fragment = int(filename_match.group("fragment_id"))
        if filename_fragment != identity.fragment_id:
            msg = "Coordinate plan filename fragment does not match its manifest identity"
            raise ValueError(msg)
        if filename_match.group("identity_prefix") != identity.identity_sha256()[:16]:
            msg = "Coordinate plan filename identity prefix does not match its manifest identity"
            raise ValueError(msg)
        self._validate_pins(identity, missing_key_policy)
        coordinates = _require_mapping(artifact.manifest.get("coordinates"), "coordinates")
        source_identity_sha256 = _source_identity_sha256(artifact, identity, missing_key_policy)
        task = LanceCoordinatePlanTask(
            dataset_name=identity.document_uri,
            data=str(artifact.parquet_path),
            manifest_path=str(artifact.manifest_path),
            source_identity_sha256=source_identity_sha256,
            _metadata={
                "source_files": [str(artifact.parquet_path), str(artifact.manifest_path)],
                "lance_coordinate_plan": {
                    "plan_root": self.plan_root,
                    "fragment_id": identity.fragment_id,
                    "rows": coordinates["rows"],
                    "canonical_ipc_sha256": coordinates["canonical_ipc_sha256"],
                    "source_identity_sha256": source_identity_sha256,
                    "document_uri": identity.document_uri,
                    "document_version": identity.document_version,
                    "image_uri": identity.image_uri,
                    "image_version": identity.image_version,
                    "sidecar_manifest_sha256": identity.sidecar_manifest_sha256,
                    "fragment_manifest_sha256": identity.fragment_manifest_sha256,
                    "missing_key_policy": missing_key_policy,
                },
            },
        )
        return identity.fragment_id, task

    def process(self, task: EmptyTask) -> list[LanceCoordinatePlanTask]:
        if not isinstance(task, EmptyTask):
            msg = f"Expected EmptyTask, got {type(task).__name__}"
            raise TypeError(msg)
        inventory_before, parquet, manifests = self._inventory()
        partial_parquet = sorted(set(parquet) - set(manifests))
        partial_manifests = sorted(set(manifests) - set(parquet))
        if partial_parquet or partial_manifests:
            msg = (
                "Coordinate plan root contains partial artifact pairs: "
                f"Parquet-only={partial_parquet[:10]}, manifest-only={partial_manifests[:10]}"
            )
            raise ValueError(msg)

        by_fragment: dict[int, LanceCoordinatePlanTask] = {}
        for stem in sorted(parquet):
            fragment_id, plan_task = self._task_for_pair(stem, parquet[stem], manifests[stem])
            if fragment_id in by_fragment:
                msg = f"Coordinate plan root contains duplicate document fragment {fragment_id}"
                raise ValueError(msg)
            by_fragment[fragment_id] = plan_task

        actual_fragment_ids = tuple(sorted(by_fragment))
        if self.expected_fragment_ids is not None and actual_fragment_ids != self.expected_fragment_ids:
            expected = set(self.expected_fragment_ids)
            actual = set(actual_fragment_ids)
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            msg = (
                "Coordinate plan root fragment inventory does not match expected_fragment_ids: "
                f"missing={missing[:10]} ({len(missing)} total), extra={extra[:10]} ({len(extra)} total)"
            )
            raise ValueError(msg)

        inventory_after, _, _ = self._inventory()
        if inventory_after != inventory_before:
            msg = "Coordinate plan root changed while it was being validated"
            raise RuntimeError(msg)
        return [by_fragment[fragment_id] for fragment_id in actual_fragment_ids]
