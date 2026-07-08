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

"""Schema-v3 contract for terminal GPU Lance cluster telemetry."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

TELEMETRY_NODE_VALIDATION_SCHEMA_VERSION = 2
TELEMETRY_NODE_VALIDATION_ARTIFACT_KIND = "gpu_lance_saturation_node_telemetry_validation"
TELEMETRY_CLUSTER_VALIDATION_SCHEMA_VERSION = 3
TELEMETRY_CLUSTER_VALIDATION_ARTIFACT_KIND = "gpu_lance_saturation_cluster_telemetry_validation"
_SHA256_HEX_LENGTH = 64


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _short_hostname(value: str) -> str:
    return value.split(".", maxsplit=1)[0]


def _load_json_object(path: Path, label: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{label} is unreadable: {type(exc).__name__}: {exc}"]
    if not isinstance(value, Mapping):
        return None, [f"{label} must contain a JSON object"]
    return value, []


def _artifact_identity_failures(identity: object, path: Path, label: str) -> list[str]:
    if not isinstance(identity, Mapping):
        return [f"{label} identity is missing or invalid"]
    failures = []
    if identity.get("path") != path.name:
        failures.append(f"{label}.path is {identity.get('path')!r}; expected {path.name!r}")
    if not path.is_file():
        failures.append(f"{label} artifact is missing")
        return failures
    if identity.get("bytes") != path.stat().st_size:
        failures.append(f"{label}.bytes does not match the artifact")
    digest = identity.get("sha256")
    if not _valid_sha256(digest):
        failures.append(f"{label}.sha256 is missing or invalid")
    elif digest != _sha256_file(path):
        failures.append(f"{label}.sha256 does not match the artifact")
    return failures


def _node_marker_failures(
    marker: Mapping[str, Any],
    *,
    node_id: int,
    hostname: str,
    raw_stream_path: Path,
) -> list[str]:
    failures = []
    expected_fields = {
        "schema_version": TELEMETRY_NODE_VALIDATION_SCHEMA_VERSION,
        "artifact_kind": TELEMETRY_NODE_VALIDATION_ARTIFACT_KIND,
        "terminal": True,
        "node_id": node_id,
        "hostname": hostname,
        "status": "passed",
        "failures": [],
    }
    for name, expected in expected_fields.items():
        if marker.get(name) != expected:
            failures.append(f"{name} is {marker.get(name)!r}; expected {expected!r}")
    raw_identity = marker.get("telemetry_artifact")
    if not isinstance(raw_identity, Mapping):
        failures.append("telemetry_artifact identity is missing or invalid")
    else:
        if raw_identity.get("path") != raw_stream_path.name:
            failures.append(
                f"telemetry_artifact.path is {raw_identity.get('path')!r}; expected {raw_stream_path.name!r}"
            )
        digest = raw_identity.get("sha256")
        if not _valid_sha256(digest):
            failures.append("telemetry_artifact.sha256 is missing or invalid")
        elif raw_stream_path.is_file() and digest != _sha256_file(raw_stream_path):
            failures.append("telemetry_artifact.sha256 does not match the raw stream")
    return failures


def validate_cluster_telemetry_contract(  # noqa: C901, PLR0912, PLR0915
    telemetry: Mapping[str, Any],
    output_dir: Path,
    *,
    expected_node_count: int,
    repeat_count: int,
) -> list[str]:
    """Validate semantic coverage and every raw-stream/marker identity."""

    failures = []
    expected_node_ids = {str(node_id) for node_id in range(expected_node_count)}
    expected_top_level = {
        "schema_version": TELEMETRY_CLUSTER_VALIDATION_SCHEMA_VERSION,
        "artifact_kind": TELEMETRY_CLUSTER_VALIDATION_ARTIFACT_KIND,
        "terminal": True,
        "status": "passed",
        "failures": [],
        "required_steady_state_coverage": True,
        "required_steady_repeat_count": repeat_count,
        "missing": [],
        "unexpected": [],
        "missing_terminal_markers": [],
        "unexpected_terminal_markers": [],
    }
    for name, expected in expected_top_level.items():
        if telemetry.get(name) != expected:
            failures.append(f"{name} is {telemetry.get(name)!r}; expected {expected!r}")

    mappings = {
        "expected_nodes": telemetry.get("expected_nodes"),
        "nodes": telemetry.get("nodes"),
        "terminal_markers": telemetry.get("terminal_markers"),
        "node_artifacts": telemetry.get("node_artifacts"),
    }
    invalid_mapping_shape = False
    for name, value in mappings.items():
        if not isinstance(value, Mapping):
            failures.append(f"{name} is missing or invalid")
            invalid_mapping_shape = True
        elif set(value) != expected_node_ids:
            failures.append(f"{name} node IDs are {sorted(map(str, value))}; expected {sorted(expected_node_ids)}")
            invalid_mapping_shape = True
    if invalid_mapping_shape:
        return failures

    expected_nodes = cast("Mapping[str, Any]", mappings["expected_nodes"])
    nodes = cast("Mapping[str, Any]", mappings["nodes"])
    terminal_markers = cast("Mapping[str, Any]", mappings["terminal_markers"])
    node_artifacts = cast("Mapping[str, Any]", mappings["node_artifacts"])
    required_phases = [f"steady_repeat_{repeat_index}" for repeat_index in range(repeat_count)]
    telemetry_dir = output_dir / "telemetry"

    for node_id_text in sorted(expected_node_ids, key=int):
        node_id = int(node_id_text)
        hostname = expected_nodes[node_id_text]
        if not isinstance(hostname, str) or not hostname:
            failures.append(f"expected_nodes[{node_id_text}] is invalid")
            hostname = ""
        else:
            hostname = _short_hostname(hostname)

        validation = nodes[node_id_text]
        if not isinstance(validation, Mapping):
            failures.append(f"nodes[{node_id_text}] is invalid")
        else:
            expected_node_fields = {
                "status": "passed",
                "failures": [],
                "expected_node_id": node_id,
                "expected_hostname": hostname,
                "observed_hostname": hostname,
                "steady_state_observed": True,
                "required_steady_phases": required_phases,
                "missing_steady_phases": [],
            }
            for name, expected in expected_node_fields.items():
                if validation.get(name) != expected:
                    failures.append(f"nodes[{node_id_text}].{name} is {validation.get(name)!r}; expected {expected!r}")

        raw_path = telemetry_dir / f"node_{node_id:04d}.jsonl"
        marker_path = telemetry_dir / f"node_{node_id:04d}.validation.json"
        identities = node_artifacts[node_id_text]
        if not isinstance(identities, Mapping):
            failures.append(f"node_artifacts[{node_id_text}] is invalid")
        else:
            failures.extend(
                _artifact_identity_failures(
                    identities.get("raw_stream"), raw_path, f"node_artifacts[{node_id_text}].raw_stream"
                )
            )
            failures.extend(
                _artifact_identity_failures(
                    identities.get("terminal_marker"),
                    marker_path,
                    f"node_artifacts[{node_id_text}].terminal_marker",
                )
            )

        embedded_marker = terminal_markers[node_id_text]
        if not isinstance(embedded_marker, Mapping):
            failures.append(f"terminal_markers[{node_id_text}] is invalid")
            continue
        marker, marker_load_failures = _load_json_object(marker_path, f"node {node_id_text} terminal marker")
        failures.extend(marker_load_failures)
        if marker is not None and marker != embedded_marker:
            failures.append(f"terminal_markers[{node_id_text}] does not match its digest-bound artifact")
        failures.extend(
            f"terminal_markers[{node_id_text}]: {failure}"
            for failure in _node_marker_failures(
                embedded_marker,
                node_id=node_id,
                hostname=hostname,
                raw_stream_path=raw_path,
            )
        )
    return failures


def validate_legacy_cluster_telemetry_contract(  # noqa: C901, PLR0912
    telemetry: Mapping[str, Any],
    *,
    expected_node_count: int,
) -> list[str]:
    """Validate the bounded pre-marker summary used by eligibility schema v2."""

    failures = []
    if telemetry.get("schema_version") not in {None, 2}:
        failures.append(f"legacy schema_version is {telemetry.get('schema_version')!r}; expected absent or 2")
    expected_fields = {
        "status": "passed",
        "failures": [],
        "required_steady_state_coverage": True,
        "missing": [],
        "unexpected": [],
    }
    for name, expected in expected_fields.items():
        if telemetry.get(name) != expected:
            failures.append(f"{name} is {telemetry.get(name)!r}; expected {expected!r}")

    expected_node_ids = {str(node_id) for node_id in range(expected_node_count)}
    expected_nodes = telemetry.get("expected_nodes")
    nodes = telemetry.get("nodes")
    for name, value in (("expected_nodes", expected_nodes), ("nodes", nodes)):
        if not isinstance(value, Mapping):
            failures.append(f"{name} is missing or invalid")
        elif set(value) != expected_node_ids:
            failures.append(f"{name} node IDs are {sorted(map(str, value))}; expected {sorted(expected_node_ids)}")
    if not isinstance(expected_nodes, Mapping) or set(expected_nodes) != expected_node_ids:
        return failures
    if not isinstance(nodes, Mapping) or set(nodes) != expected_node_ids:
        return failures

    for node_id_text in sorted(expected_node_ids, key=int):
        node_id = int(node_id_text)
        hostname = expected_nodes[node_id_text]
        if not isinstance(hostname, str) or not hostname:
            failures.append(f"expected_nodes[{node_id_text}] is invalid")
            hostname = ""
        else:
            hostname = _short_hostname(hostname)
        validation = nodes[node_id_text]
        if not isinstance(validation, Mapping):
            failures.append(f"nodes[{node_id_text}] is invalid")
            continue
        expected_node_fields = {
            "status": "passed",
            "failures": [],
            "expected_node_id": node_id,
            "expected_hostname": hostname,
            "observed_hostname": hostname,
            "steady_state_observed": True,
        }
        for name, expected in expected_node_fields.items():
            if validation.get(name) != expected:
                failures.append(f"nodes[{node_id_text}].{name} is {validation.get(name)!r}; expected {expected!r}")
    return failures
