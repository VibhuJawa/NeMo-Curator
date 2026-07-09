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

"""Checkpointable source for validated Lance payload overlays."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.lance_payload_overlay import (
    LancePayloadOverlayArtifact,
    LancePayloadOverlayIdentity,
    LancePayloadOverlayTask,
    lance_payload_overlay_root,
    lance_payload_overlay_source_identity_sha256,
    lance_payload_overlay_task,
    normalize_image_columns,
    validate_lance_payload_overlay,
)
from nemo_curator.tasks import EmptyTask
from nemo_curator.utils.uri import validate_credential_free_uri_identity

_ARTIFACT_PATTERN = re.compile(r"fragment-(?P<fragment_id>[0-9]{8,})-overlay-(?P<identity>[0-9a-f]{16})")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _optional_positive_integer(value: int | None, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        msg = f"{name} must be a positive integer or None"
        raise ValueError(msg)


def _optional_sha256(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None):
        msg = f"{name} must be a lowercase SHA-256 digest or None"
        raise ValueError(msg)


def _expected_fragments(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        msg = "expected_fragment_ids must be a sequence of nonnegative integers or None"
        raise TypeError(msg)
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        msg = "expected_fragment_ids must contain only nonnegative integers"
        raise ValueError(msg)
    if len(set(result)) != len(result):
        msg = "expected_fragment_ids must not contain duplicates"
        raise ValueError(msg)
    return tuple(sorted(result))


@dataclass
class LancePayloadOverlayReader(ProcessingStage[EmptyTask, LancePayloadOverlayTask]):
    """Enumerate overlays only after complete identity and payload validation."""

    overlay_root: str
    document_uri: str | None = None
    document_version: int | None = None
    image_uri: str | None = None
    image_version: int | None = None
    sidecar_manifest_sha256: str | None = None
    fragment_manifest_sha256: str | None = None
    overlay_config_sha256: str | None = None
    image_columns: Mapping[str, str] | Sequence[tuple[str, str]] | None = field(default=None, repr=False)
    expected_fragment_ids: Sequence[int] | None = field(default=None, repr=False)
    name: str = "lance_payload_overlay_reader"

    is_source_stage = True
    is_resumable = True

    def __post_init__(self) -> None:
        if not isinstance(self.overlay_root, str) or not self.overlay_root:
            msg = "overlay_root must be a non-empty absolute filesystem path"
            raise ValueError(msg)
        root = Path(self.overlay_root)
        if not root.is_absolute():
            msg = "overlay_root must be an absolute filesystem path"
            raise ValueError(msg)
        self.overlay_root = str(root)
        for value, name in ((self.document_uri, "document_uri"), (self.image_uri, "image_uri")):
            if value is not None:
                if not isinstance(value, str) or not value:
                    msg = f"{name} must be a non-empty string or None"
                    raise ValueError(msg)
                validate_credential_free_uri_identity(value, name)
        _optional_positive_integer(self.document_version, "document_version")
        _optional_positive_integer(self.image_version, "image_version")
        _optional_sha256(self.sidecar_manifest_sha256, "sidecar_manifest_sha256")
        _optional_sha256(self.fragment_manifest_sha256, "fragment_manifest_sha256")
        _optional_sha256(self.overlay_config_sha256, "overlay_config_sha256")
        self.image_columns = None if self.image_columns is None else normalize_image_columns(self.image_columns)
        self.expected_fragment_ids = _expected_fragments(self.expected_fragment_ids)

    def ray_stage_spec(self) -> dict[str, object]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def num_workers(self) -> int | None:
        return 1

    def _inventory(self) -> tuple[tuple[str, ...], dict[str, Path]]:
        root = Path(self.overlay_root)
        if root.is_symlink() or not root.is_dir():
            msg = f"overlay_root must be an existing regular directory, not a symlink: {root}"
            raise ValueError(msg)
        names: list[str] = []
        artifacts: dict[str, Path] = {}
        locks: set[str] = set()
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            names.append(path.name)
            match = _ARTIFACT_PATTERN.fullmatch(path.name)
            if match is not None:
                if path.is_symlink() or not path.is_dir():
                    msg = f"payload overlay entry must be a regular directory: {path}"
                    raise ValueError(msg)
                artifacts[path.name] = path
                continue
            if path.name.startswith(".") and path.name.endswith(".lock"):
                artifact_name = path.name[1:-5]
                if _ARTIFACT_PATTERN.fullmatch(artifact_name) is None or path.is_symlink() or not path.is_file():
                    msg = f"payload overlay lock entry is invalid: {path}"
                    raise ValueError(msg)
                locks.add(artifact_name)
                continue
            msg = f"payload overlay root contains a stray entry: {path}"
            raise ValueError(msg)
        orphan_locks = sorted(locks - set(artifacts))
        if orphan_locks:
            msg = f"payload overlay root contains orphan locks: {orphan_locks[:10]}"
            raise ValueError(msg)
        return tuple(names), artifacts

    def _validate_pins(self, identity: LancePayloadOverlayIdentity) -> None:
        expected = {
            "document_uri": self.document_uri,
            "document_version": self.document_version,
            "image_uri": self.image_uri,
            "image_version": self.image_version,
            "sidecar_manifest_sha256": self.sidecar_manifest_sha256,
            "fragment_manifest_sha256": self.fragment_manifest_sha256,
            "overlay_config_sha256": self.overlay_config_sha256,
        }
        for name, value in expected.items():
            if value is not None and getattr(identity, name) != value:
                msg = f"payload overlay {name} does not match the reader pin"
                raise ValueError(msg)

    def _task_for_artifact(self, path: Path) -> tuple[int, LancePayloadOverlayTask]:
        artifact: LancePayloadOverlayArtifact = validate_lance_payload_overlay(
            path,
            expected_image_columns=self.image_columns,
            verify_payload=True,
        )
        identity = artifact.identity
        self._validate_pins(identity)
        expected_name = lance_payload_overlay_root(path.parent, identity).name
        if path.name != expected_name:
            msg = "payload overlay directory name does not match its manifest identity"
            raise ValueError(msg)
        match = _ARTIFACT_PATTERN.fullmatch(path.name)
        if match is None or int(match.group("fragment_id")) != identity.fragment_id:
            msg = "payload overlay directory fragment does not match its manifest identity"
            raise ValueError(msg)
        source_identity = lance_payload_overlay_source_identity_sha256(identity)
        task = lance_payload_overlay_task(
            artifact,
            metadata={
                "lance_payload_overlay_source": {
                    "overlay_root": self.overlay_root,
                    "fragment_id": identity.fragment_id,
                    "source_identity_sha256": source_identity,
                    "payload_verified": True,
                }
            },
        )
        return identity.fragment_id, task

    def process(self, task: EmptyTask) -> list[LancePayloadOverlayTask]:
        if not isinstance(task, EmptyTask):
            msg = f"Expected EmptyTask, got {type(task).__name__}"
            raise TypeError(msg)
        inventory_before, artifacts = self._inventory()
        by_fragment: dict[int, LancePayloadOverlayTask] = {}
        for name in sorted(artifacts):
            fragment_id, overlay_task = self._task_for_artifact(artifacts[name])
            if fragment_id in by_fragment:
                msg = f"payload overlay root contains duplicate document fragment {fragment_id}"
                raise ValueError(msg)
            by_fragment[fragment_id] = overlay_task
        actual_fragment_ids = tuple(sorted(by_fragment))
        if self.expected_fragment_ids is not None and actual_fragment_ids != self.expected_fragment_ids:
            expected = set(self.expected_fragment_ids)
            actual = set(actual_fragment_ids)
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            msg = (
                "payload overlay root fragment inventory does not match expected_fragment_ids: "
                f"missing={missing[:10]} ({len(missing)} total), extra={extra[:10]} ({len(extra)} total)"
            )
            raise ValueError(msg)
        inventory_after, _ = self._inventory()
        if inventory_after != inventory_before:
            msg = "payload overlay root changed while it was being validated"
            raise RuntimeError(msg)
        return [by_fragment[fragment_id] for fragment_id in actual_fragment_ids]
