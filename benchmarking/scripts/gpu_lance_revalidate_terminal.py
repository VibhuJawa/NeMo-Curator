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

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, PLR2004, TRY004
"""Revalidate an immutable eligibility-v1/v2 run into a new terminal-v3 family.

This reads only benchmark metadata and existing telemetry. It never opens the
Lance dataset or fetches payloads, and it publishes into a new output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarking.scripts import gpu_lance_saturation_runner as runner
from benchmarking.scripts.gpu_lance_telemetry_contract import validate_legacy_cluster_telemetry_contract
from nemo_curator.utils.uri import validate_credential_free_uri_identity

_SECRET_OPTION_PARTS = ("access_key", "secret", "token", "password", "credential")


def _json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _source_artifact_failures(source_dir: Path, eligibility: Mapping[str, Any]) -> list[str]:
    failures = []
    if eligibility.get("schema_version") not in {1, 2}:
        failures.append("source eligibility schema_version must be 1 or 2")
    if eligibility.get("artifact_kind") != "gpu_lance_saturation_terminal_eligibility":
        failures.append("source eligibility artifact_kind is invalid")
    if eligibility.get("terminal") is not True:
        failures.append("source eligibility is not terminal")
    artifacts = eligibility.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return [*failures, "source eligibility artifacts is missing or invalid"]
    for name, filename in (
        ("benchmark", "benchmark.json"),
        ("run_identity", "run_identity.json"),
        ("telemetry_validation", "telemetry_validation.json"),
    ):
        identity = artifacts.get(name)
        path = source_dir / filename
        if not isinstance(identity, Mapping):
            failures.append(f"source eligibility {name} identity is missing")
        elif identity.get("path") != filename or identity.get("sha256") != runner._sha256_file(path):
            failures.append(f"source eligibility {name} identity does not match {filename}")
    return failures


def _append_uri_failure(failures: list[str], value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        failures.append(f"{label} is missing or invalid")
        return
    try:
        validate_credential_free_uri_identity(value, label)
    except ValueError as exc:
        failures.append(str(exc))


def _append_storage_options_failure(failures: list[str], value: str, label: str) -> None:
    if value.startswith("@"):
        if not value[1:] or Path(value[1:]).name != value[1:]:
            failures.append(f"{label} may reference only an @basename")
        return
    if value == "<redacted-json>":
        return
    if value.startswith("<redacted-json keys=") and value.endswith(">"):
        keys = value.removeprefix("<redacted-json keys=").removesuffix(">").split(",")
    else:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            failures.append(f"{label} must be credential-free JSON, a redacted placeholder, or @basename")
            return
        if not isinstance(payload, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in payload.items()
        ):
            failures.append(f"{label} must be a JSON object with string keys and values")
            return
        keys = list(payload)
        if payload:
            failures.append(f"{label} must not contain non-empty inline storage options; use an @basename")
    secret_keys = sorted(key for key in keys if any(part in key.casefold() for part in _SECRET_OPTION_PARTS))
    if secret_keys:
        failures.append(f"{label} contains credential-like option keys {secret_keys}")


def _append_option_key_failures(failures: list[str], value: object, label: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        failures.append(f"{label} must be an array of option-key names")
        return
    if not all(isinstance(key, str) for key in value):
        failures.append(f"{label} contains a non-string option key")
        return
    secret_keys = sorted(key for key in value if any(part in key.casefold() for part in _SECRET_OPTION_PARTS))
    if secret_keys:
        failures.append(f"{label} contains credential-like option keys {secret_keys}")


def _source_uri_failures(
    report: Mapping[str, Any], identity: Mapping[str, Any], *, source_schema_version: int
) -> list[str]:
    failures = []
    report_dataset = report.get("dataset")
    identity_dataset = identity.get("dataset")
    configuration = report.get("configuration")
    if not isinstance(report_dataset, Mapping):
        return ["source benchmark dataset identity is missing or invalid"]
    if not isinstance(configuration, Mapping):
        return ["source benchmark configuration is missing or invalid"]
    _append_uri_failure(failures, report_dataset.get("uri"), "benchmark dataset URI")
    if isinstance(identity_dataset, Mapping):
        _append_uri_failure(failures, identity_dataset.get("uri"), "run identity dataset URI")
    elif source_schema_version >= 2:
        failures.append("source run identity dataset identity is missing or invalid")
    _append_uri_failure(
        failures,
        configuration.get("reference_manifest_uri"),
        "benchmark reference manifest URI",
    )
    _append_uri_failure(
        failures,
        identity.get("reference_manifest_uri"),
        "run identity reference manifest URI",
    )
    index_mirror = configuration.get("index_mirror")
    if index_mirror is not None:
        _append_uri_failure(failures, index_mirror, "benchmark index mirror URI")
    for value, label in (
        (report_dataset.get("storage_option_keys", []), "benchmark dataset storage_option_keys"),
        (configuration.get("reference_storage_option_keys", []), "benchmark reference_storage_option_keys"),
        (identity.get("storage_option_keys", []), "run identity storage_option_keys"),
        (identity.get("reference_storage_option_keys", []), "run identity reference_storage_option_keys"),
    ):
        _append_option_key_failures(failures, value, label)
    for field in ("reference_files", "reference_glob"):
        values = configuration.get(field)
        if values is None:
            continue
        if not isinstance(values, Sequence) or isinstance(values, str | bytes | bytearray):
            failures.append(f"benchmark {field} must be an array of URI identities")
            continue
        for ordinal, value in enumerate(values):
            _append_uri_failure(failures, value, f"benchmark {field}[{ordinal}]")

    command = identity.get("benchmark_command")
    if not isinstance(command, str) or not command:
        failures.append("run identity benchmark_command is missing or invalid")
        return failures
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        failures.append(f"run identity benchmark_command is invalid: {type(exc).__name__}")
        return failures
    uri_flags = {"--image-lance-uri", "--reference-manifest-uri", "--reference-glob"}
    storage_flags = {"--storage-options-json", "--reference-storage-options-json"}
    for ordinal, token in enumerate(tokens):
        if token in uri_flags:
            if ordinal + 1 >= len(tokens):
                failures.append(f"run identity benchmark_command {token} has no value")
            else:
                _append_uri_failure(failures, tokens[ordinal + 1], f"run identity benchmark_command {token}")
        else:
            for flag in uri_flags:
                prefix = f"{flag}="
                if token.startswith(prefix):
                    _append_uri_failure(failures, token[len(prefix) :], f"run identity benchmark_command {flag}")
        if token in storage_flags:
            if ordinal + 1 >= len(tokens):
                failures.append(f"run identity benchmark_command {token} has no value")
            else:
                _append_storage_options_failure(
                    failures,
                    tokens[ordinal + 1],
                    f"run identity benchmark_command {token}",
                )
        else:
            for flag in storage_flags:
                prefix = f"{flag}="
                if token.startswith(prefix):
                    _append_storage_options_failure(
                        failures,
                        token[len(prefix) :],
                        f"run identity benchmark_command {flag}",
                    )
    return failures


def _source_contract(
    source_dir: Path,
) -> tuple[int, str, runner.SaturationGeometry, int, int, Mapping[str, Any]]:
    report = _json_object(source_dir / "benchmark.json", "source benchmark")
    identity = _json_object(source_dir / "run_identity.json", "source run identity")
    telemetry = _json_object(source_dir / "telemetry_validation.json", "source telemetry validation")
    eligibility = _json_object(source_dir / "eligibility.json", "source eligibility")
    source_schema_version = eligibility.get("schema_version")
    if source_schema_version not in {1, 2}:
        raise RuntimeError("source eligibility schema_version must be 1 or 2")
    failures = _source_artifact_failures(source_dir, eligibility)
    failures.extend(_source_uri_failures(report, identity, source_schema_version=source_schema_version))

    geometry_value = identity.get("geometry")
    policy = identity.get("benchmark_policy")
    configuration = report.get("configuration")
    if not isinstance(geometry_value, Mapping):
        raise RuntimeError("source run identity geometry is missing or invalid")
    if not isinstance(policy, Mapping):
        raise RuntimeError("source run identity benchmark_policy is missing or invalid")
    if not isinstance(configuration, Mapping):
        raise RuntimeError("source benchmark configuration is missing or invalid")
    geometry = runner.SaturationGeometry(
        nodes=_positive_int(geometry_value.get("nodes"), "geometry.nodes"),
        waves=_positive_int(geometry_value.get("waves"), "geometry.waves"),
    )
    arm = policy.get("arm")
    if arm not in runner.SUPPORTED_ARMS:
        raise RuntimeError(f"source benchmark arm is invalid: {arm!r}")
    repeat_count = _positive_int(configuration.get("repeat_count"), "configuration.repeat_count")
    warmup_count = configuration.get("warmup_count")
    if isinstance(warmup_count, bool) or not isinstance(warmup_count, int) or warmup_count < 0:
        raise RuntimeError("configuration.warmup_count must be a nonnegative integer")

    failures.extend(
        f"legacy telemetry: {failure}"
        for failure in validate_legacy_cluster_telemetry_contract(
            telemetry,
            expected_node_count=geometry.nodes,
        )
    )
    benchmark_validation = runner.validate_benchmark_report(
        source_dir / "benchmark.json",
        arm,
        geometry,
        repeat_count,
        warmup_count,
    )
    failures.extend(f"benchmark: {failure}" for failure in benchmark_validation["failures"])
    if failures:
        raise RuntimeError(f"source terminal-v2 family failed offline validation: {failures}")
    return source_schema_version, arm, geometry, repeat_count, warmup_count, telemetry


def _terminal_summary(path: Path) -> Mapping[str, Any]:
    records, failures = runner._read_telemetry_records(path)
    if failures or not records or records[-1].get("record_type") != "summary":
        raise RuntimeError(f"raw telemetry lacks a valid terminal summary: {failures}")
    return records[-1]


def _passed_collector_process(marker_path: Path) -> dict[str, Any]:
    marker = _json_object(marker_path, f"legacy node marker {marker_path.name}")
    collector = marker.get("collector_process")
    if (
        marker.get("status") != "passed"
        or not isinstance(collector, Mapping)
        or collector.get("status") != "passed"
        or collector.get("returncode") != 0
        or collector.get("failures") != []
    ):
        raise RuntimeError(f"legacy node marker {marker_path.name} lacks a clean collector-process pass")
    return dict(collector)


def _publish_v3_family(  # noqa: PLR0913
    source_dir: Path,
    staging_dir: Path,
    *,
    source_schema_version: int,
    arm: str,
    geometry: runner.SaturationGeometry,
    repeat_count: int,
    warmup_count: int,
    legacy_telemetry: Mapping[str, Any],
) -> None:
    shutil.copy2(source_dir / "benchmark.json", staging_dir / "benchmark.json")
    if source_schema_version == 2:
        shutil.copy2(source_dir / "run_identity.json", staging_dir / "run_identity.json")
    else:
        source_identity_path = staging_dir / "source_run_identity.json"
        shutil.copy2(source_dir / "run_identity.json", source_identity_path)
        source_identity = dict(_json_object(source_identity_path, "source run identity"))
        report = _json_object(staging_dir / "benchmark.json", "source benchmark")
        dataset = report.get("dataset")
        manifest = report.get("manifest")
        if not isinstance(dataset, Mapping) or not isinstance(manifest, Mapping):
            raise RuntimeError("source benchmark dataset/manifest identity is missing")
        source_identity.update(
            {
                "schema_version": runner.RUN_IDENTITY_SCHEMA_VERSION,
                "evidence_class": geometry.evidence_class,
                "dataset": {"uri": dataset.get("uri"), "version": dataset.get("version")},
                "manifest": {"digest_sha256": manifest.get("digest_sha256")},
                "legacy_source_identity": {
                    "path": source_identity_path.name,
                    "sha256": runner._sha256_file(source_identity_path),
                },
            }
        )
        runner._atomic_json(staging_dir / "run_identity.json", source_identity)
    telemetry_dir = staging_dir / "telemetry"
    telemetry_dir.mkdir()
    expected_nodes_value = legacy_telemetry.get("expected_nodes")
    if not isinstance(expected_nodes_value, Mapping):
        raise RuntimeError("legacy telemetry expected_nodes is missing or invalid")
    expected_nodes = {int(node_id): str(hostname) for node_id, hostname in expected_nodes_value.items()}
    storage_axis = _json_object(source_dir / "run_identity.json", "source run identity").get("storage_axis")
    if not isinstance(storage_axis, str) or not storage_axis:
        raise RuntimeError("source run identity storage_axis is missing")

    intervals = set()
    for node_id, hostname in expected_nodes.items():
        source_raw = source_dir / "telemetry" / f"node_{node_id:04d}.jsonl"
        source_marker = source_dir / "telemetry" / f"node_{node_id:04d}.validation.json"
        destination_raw = telemetry_dir / source_raw.name
        shutil.copy2(source_raw, destination_raw)
        summary = _terminal_summary(destination_raw)
        interval = summary.get("interval_seconds")
        if not isinstance(interval, int | float) or isinstance(interval, bool) or interval <= 0:
            raise RuntimeError(f"raw telemetry {source_raw.name} has an invalid interval")
        intervals.add(float(interval))
        artifact_validation = runner.validate_telemetry_artifact(
            destination_raw,
            runner.TelemetryValidationSpec(
                node_id=node_id,
                hostname=hostname,
                gpu_count=runner.ACTORS_PER_NODE,
                interval_seconds=float(interval),
                required_steady_repeat_count=repeat_count,
                storage_axis=storage_axis,
            ),
        )
        if artifact_validation["status"] != "passed":
            raise RuntimeError(
                f"raw telemetry {source_raw.name} failed revalidation: {artifact_validation['failures']}"
            )
        artifact_validation["path"] = f"telemetry/{destination_raw.name}"
        collector_process = _passed_collector_process(source_marker)
        collector_process["output"] = f"telemetry/{destination_raw.name}"
        node_marker = {
            "schema_version": runner.TELEMETRY_SCHEMA_VERSION,
            "artifact_kind": runner.TELEMETRY_NODE_VALIDATION_ARTIFACT_KIND,
            "terminal": True,
            "node_id": node_id,
            "hostname": hostname,
            "status": "passed",
            "failures": [],
            "telemetry_artifact": {
                "path": destination_raw.name,
                "sha256": runner._sha256_file(destination_raw),
            },
            "collector_process": collector_process,
            "artifact": artifact_validation,
        }
        runner._atomic_json(telemetry_dir / f"node_{node_id:04d}.validation.json", node_marker)
    if len(intervals) != 1:
        raise RuntimeError(f"raw telemetry intervals disagree across nodes: {sorted(intervals)}")

    cluster_validation = runner.validate_telemetry_cluster(
        telemetry_dir,
        runner.TelemetryClusterSpec(
            nodes=expected_nodes,
            report_path=staging_dir / "benchmark.json",
            arm=arm,
            gpu_count=runner.ACTORS_PER_NODE,
            interval_seconds=intervals.pop(),
            wait_seconds=0,
            storage_axis=storage_axis,
            repeat_count=repeat_count,
        ),
    )
    if cluster_validation["status"] != "passed":
        raise RuntimeError(f"cluster telemetry failed revalidation: {cluster_validation['failures']}")
    for node_id_text, node_validation in cluster_validation["nodes"].items():
        node_validation["path"] = f"telemetry/node_{int(node_id_text):04d}.jsonl"
    runner._atomic_json(staging_dir / "telemetry_validation.json", cluster_validation)
    benchmark_validation = runner.validate_benchmark_report(
        staging_dir / "benchmark.json",
        arm,
        geometry,
        repeat_count,
        warmup_count,
    )
    runner._atomic_json(staging_dir / "validation.json", benchmark_validation)
    eligibility = runner.build_terminal_eligibility(
        staging_dir,
        arm=arm,
        geometry=geometry,
        repeat_count=repeat_count,
        warmup_count=warmup_count,
    )
    if eligibility["status"] != "eligible":
        raise RuntimeError(f"terminal-v3 eligibility failed: {eligibility['failures']}")
    runner._atomic_json(staging_dir / "eligibility.json", eligibility)


def revalidate_terminal_family(source_dir: Path, output_dir: Path) -> Mapping[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise RuntimeError("output directory must differ from the immutable source directory")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_schema_version, arm, geometry, repeat_count, warmup_count, legacy_telemetry = _source_contract(source_dir)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _publish_v3_family(
            source_dir,
            staging_dir,
            source_schema_version=source_schema_version,
            arm=arm,
            geometry=geometry,
            repeat_count=repeat_count,
            warmup_count=warmup_count,
            legacy_telemetry=legacy_telemetry,
        )
        os.replace(staging_dir, output_dir)
    finally:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(staging_dir)
    return {
        "status": "eligible",
        "source_schema_version": source_schema_version,
        "output_schema_version": runner.ELIGIBILITY_SCHEMA_VERSION,
        "arm": arm,
        "nodes": geometry.nodes,
        "waves": geometry.waves,
        "output_dir": str(output_dir),
        "payload_reads": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-dir", required=True, type=Path, help="Immutable terminal eligibility-v2 directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="New terminal eligibility-v3 directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = revalidate_terminal_family(args.source_dir, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
