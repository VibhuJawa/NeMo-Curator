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

"""Run the corrected per-node GPU Lance weak-scaling geometry under Slurm.

Launch this program once per allocated node with ``srun --ntasks-per-node=1``.
``SlurmRayClient`` creates one cluster for the exclusive allocation; only the
head process invokes the existing benchmark. Worker processes host Ray actors
and block until the head tears the cluster down.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.utils.uri import redact_uri_identity, validate_credential_free_uri_identity

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, TextIO

    from nemo_curator.core.client import SlurmRayClient

TASK_ROWS = 256
TASKS_PER_ACTOR = 64
ACTORS_PER_NODE = 8
SUPPORTED_ARMS = ("lance_ray_gpu_actor", "ray_data_persistent_gpu_actor")
DEFAULT_IMAGE_VERSION = 4
_SECRET_OPTION_PARTS = ("access_key", "secret", "token", "password", "credential")
TELEMETRY_SCHEMA_VERSION = 2
RUN_IDENTITY_SCHEMA_VERSION = 2
ELIGIBILITY_SCHEMA_VERSION = 2
MINIMUM_REPEAT_COUNT = 2
MINIMUM_DELTA_SAMPLE_COUNT = 2
_MAX_PERCENT = 100.0
_SHA256_HEX_LENGTH = 64
_SHA256_BYTES = _SHA256_HEX_LENGTH // 2
_BENCHMARK_QUERY_ORDINAL = "_benchmark_query_ordinal"
LOCALITY_SENSITIVITY = "locality_sensitivity"
PRIMARY_SATURATION = "primary_saturation"
LOCALITY_SENSITIVITY_WAVES = (1, 2)
PRIMARY_SATURATION_WAVES = (4, 8)
SUPPORTED_WAVES = (*LOCALITY_SENSITIVITY_WAVES, *PRIMARY_SATURATION_WAVES)


@dataclass(frozen=True)
class SaturationGeometry:
    """Weak-scaling work held constant per GPU and per node."""

    nodes: int
    waves: int
    actors_per_node: int = ACTORS_PER_NODE
    tasks_per_actor: int = TASKS_PER_ACTOR
    task_rows: int = TASK_ROWS

    def __post_init__(self) -> None:
        if self.nodes <= 0 or self.actors_per_node <= 0 or self.tasks_per_actor <= 0 or self.task_rows <= 0:
            msg = "nodes, actors_per_node, tasks_per_actor, and task_rows must be positive"
            raise ValueError(msg)
        if self.waves not in SUPPORTED_WAVES:
            msg = "waves must be 1, 2, 4, or 8"
            raise ValueError(msg)
        if self.tasks_per_actor % self.waves:
            msg = "tasks_per_actor must be divisible by waves"
            raise ValueError(msg)

    @property
    def actor_count(self) -> int:
        return self.nodes * self.actors_per_node

    @property
    def target_tasks(self) -> int:
        return self.actor_count * self.tasks_per_actor

    @property
    def target_rows(self) -> int:
        return self.target_tasks * self.task_rows

    @property
    def coalesce_tasks(self) -> int:
        return self.tasks_per_actor // self.waves

    @property
    def actor_batch_rows(self) -> int:
        return self.coalesce_tasks * self.task_rows

    @property
    def expected_actor_calls(self) -> int:
        return self.actor_count * self.waves

    @property
    def evidence_class(self) -> str:
        if self.waves in PRIMARY_SATURATION_WAVES:
            return PRIMARY_SATURATION
        return LOCALITY_SENSITIVITY


@dataclass(frozen=True)
class TelemetryHandle:
    """One node-local collector and its atomically published stream."""

    process: subprocess.Popen[str]
    log_stream: TextIO
    output: Path
    node_id: int
    hostname: str


@dataclass(frozen=True)
class TelemetryValidationSpec:
    """Expected identity and sampling contract for one telemetry stream."""

    node_id: int
    hostname: str
    gpu_count: int
    interval_seconds: float
    required_steady_repeat_count: int
    storage_axis: str


@dataclass(frozen=True)
class TelemetryClusterSpec:
    """Expected allocation-wide telemetry contract."""

    nodes: Mapping[int, str]
    report_path: Path
    arm: str
    gpu_count: int
    interval_seconds: float
    wait_seconds: float
    storage_axis: str
    repeat_count: int


@dataclass(frozen=True)
class TelemetryRunContext:
    """Allocation context needed when a collector is finalized."""

    node_id: int
    hostname: str
    nodes: Mapping[int, str]
    report_path: Path
    output_dir: Path
    arm: str
    interval_seconds: float
    storage_axis: str
    geometry: SaturationGeometry
    warmup_count: int
    repeat_count: int


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be greater than zero"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        msg = "value must be nonnegative"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        msg = "value must be greater than zero"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def validate_remaining_slurm_time(
    *,
    minimum_remaining_seconds: int | None,
    allocation_end_epoch: float | None,
    now_epoch: float | None = None,
) -> dict[str, float | int]:
    """Reject a live Slurm launch that cannot honor its caller-set time floor."""

    if minimum_remaining_seconds is None or minimum_remaining_seconds <= 0:
        msg = "live Slurm runs require --minimum-remaining-slurm-seconds"
        raise ValueError(msg)
    if allocation_end_epoch is None or not math.isfinite(allocation_end_epoch):
        msg = (
            "live Slurm runs require --allocation-end-epoch or numeric SLURM_JOB_END_TIME "
            "to enforce the remaining-time guard"
        )
        raise ValueError(msg)
    now = time.time() if now_epoch is None else now_epoch
    remaining = allocation_end_epoch - now
    if remaining < minimum_remaining_seconds:
        msg = (
            f"Slurm allocation has {remaining:.0f}s remaining; "
            f"caller requires at least {minimum_remaining_seconds}s before starting the benchmark"
        )
        raise RuntimeError(msg)
    return {
        "checked_at_epoch": now,
        "allocation_end_epoch": allocation_end_epoch,
        "remaining_seconds": remaining,
        "minimum_remaining_seconds": minimum_remaining_seconds,
    }


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH or value != value.lower():
        return False
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        return False
    return len(decoded) == _SHA256_BYTES


def _sha256(value: str) -> str:
    if not _valid_sha256(value):
        msg = "value must be a lowercase SHA-256 hex digest"
        raise argparse.ArgumentTypeError(msg)
    return value


def _json_options(value: str) -> dict[str, str]:
    if not value:
        return {}
    raw = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "storage options JSON must contain an object"
        raise TypeError(msg)
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in parsed.items()):
        msg = "storage options JSON keys and values must be strings"
        raise TypeError(msg)
    return parsed


def _normalize_json_argument(value: str) -> str:
    if not value.startswith("@"):
        return value
    return f"@{Path(value[1:]).resolve()}"


def _reject_secret_storage_options(options: Mapping[str, str]) -> None:
    secret_keys = sorted(key for key in options if any(part in key.casefold() for part in _SECRET_OPTION_PARTS))
    if secret_keys:
        msg = (
            f"storage options contain credential-like keys {secret_keys}; "
            "load credentials through the process environment instead"
        )
        raise ValueError(msg)


def _redact_uri_for_identity(value: str) -> str:
    """Match the benchmark report's credential-free URI identity."""

    return redact_uri_identity(value)


def _credential_free_uri(value: str) -> str:
    try:
        return validate_credential_free_uri_identity(value, "URI identity")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _query_manifest_digest(path: Path) -> str:
    """Reproduce the benchmark harness digest for an unsliced query manifest."""

    table = pq.read_table(path)
    table = table.append_column(
        _BENCHMARK_QUERY_ORDINAL,
        pa.array(range(table.num_rows), type=pa.int64()),
    )
    expected_columns = []
    for aliases in (("expected_md5", "md5"), ("expected_width", "width"), ("expected_height", "height")):
        column = next((name for name in aliases if name in table.column_names), None)
        if column is not None:
            expected_columns.append(column)
    selected = table.select(["source_ref", _BENCHMARK_QUERY_ORDINAL, *expected_columns])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, selected.schema) as writer:
        writer.write_table(selected)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def load_manifest_metadata(  # noqa: C901, PLR0912, PLR0915
    manifest_dir: Path, geometry: SaturationGeometry
) -> dict[str, Any]:
    """Load and strictly validate the generated saturation geometry."""

    metadata_path = manifest_dir / "manifest.json"
    manifest_path = manifest_dir / "manifest.parquet"
    if not metadata_path.is_file() or not manifest_path.is_file():
        msg = f"manifest directory must contain manifest.json and manifest.parquet: {manifest_dir}"
        raise FileNotFoundError(msg)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        msg = "manifest.json must contain an object"
        raise TypeError(msg)
    expected = {
        "task_rows": geometry.task_rows,
        "target_tasks": geometry.target_tasks,
        "target_rows": geometry.target_rows,
        "actor_count": geometry.actor_count,
        "tasks_per_actor": geometry.tasks_per_actor,
        "rows_per_actor": geometry.tasks_per_actor * geometry.task_rows,
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        msg = f"manifest geometry does not match allocation: {json.dumps(mismatches, sort_keys=True)}"
        raise ValueError(msg)
    actor_files = sorted((manifest_dir / "actors").glob("actor_*.parquet"))
    if len(actor_files) != geometry.actor_count:
        msg = f"manifest has {len(actor_files)} actor shards; expected {geometry.actor_count}"
        raise ValueError(msg)

    document = metadata.get("document")
    if (
        metadata.get("schema_version") != 1
        or not isinstance(document, Mapping)
        or not isinstance(document.get("uri"), str)
        or not document["uri"]
        or not isinstance(document.get("version"), int)
        or document["version"] <= 0
    ):
        msg = "manifest source identity is missing or invalid"
        raise ValueError(msg)
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        msg = "manifest.json files must contain generator-recorded identities"
        raise TypeError(msg)
    expected_paths = [Path("manifest.parquet"), *(Path("actors") / path.name for path in actor_files)]
    if set(files) != {str(path) for path in expected_paths}:
        msg = "manifest.json file inventory differs from the generated saturation files"
        raise ValueError(msg)
    for relative_path in expected_paths:
        path = manifest_dir / relative_path
        identity = files[str(relative_path)]
        if not isinstance(identity, Mapping):
            msg = f"manifest file identity must be an object: {relative_path}"
            raise TypeError(msg)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if identity.get("bytes") != path.stat().st_size or identity.get("sha256") != digest:
            msg = f"manifest file identity mismatch: {relative_path}"
            raise ValueError(msg)
        rows = pq.read_metadata(path).num_rows
        expected_identity = {"bytes": path.stat().st_size, "rows": rows, "sha256": digest}
        if dict(identity) != expected_identity:
            msg = f"manifest file identity mismatch: {relative_path}"
            raise ValueError(msg)
        expected_rows = (
            geometry.target_rows
            if relative_path.name == "manifest.parquet"
            else (geometry.tasks_per_actor * geometry.task_rows)
        )
        if rows != expected_rows:
            msg = f"{relative_path} contains {rows} rows; expected {expected_rows}"
            raise ValueError(msg)

    schema = pq.read_schema(manifest_path)
    if metadata.get("schema") != str(schema):
        msg = "manifest schema differs from the generator-recorded schema"
        raise ValueError(msg)
    schema_metadata = schema.metadata or {}
    expected_schema_metadata = {
        b"document_uri": document["uri"].encode(),
        b"document_version": str(document["version"]).encode(),
        b"task_rows": str(geometry.task_rows).encode(),
        b"actor_count": str(geometry.actor_count).encode(),
    }
    mismatched_schema_metadata = {
        key.decode(): {"actual": schema_metadata.get(key), "expected": value}
        for key, value in expected_schema_metadata.items()
        if schema_metadata.get(key) != value
    }
    if mismatched_schema_metadata:
        msg = f"manifest schema source identity mismatch: {mismatched_schema_metadata}"
        raise ValueError(msg)
    return metadata


def build_benchmark_command(
    args: argparse.Namespace,
    geometry: SaturationGeometry,
    *,
    ray_address: str,
    report_path: Path,
) -> list[str]:
    """Build the existing harness command without executing it."""

    repo_root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(repo_root / "benchmarking/scripts/gpu_lance_column_fetch_benchmark.py"),
        "--query-manifest",
        str((args.manifest_dir / "manifest.parquet").resolve()),
        "--image-lance-uri",
        args.image_lance_uri,
        "--image-lance-version",
        str(args.image_lance_version),
        "--storage-options-json",
        _normalize_json_argument(args.storage_options_json),
        "--reference-storage-options-json",
        _normalize_json_argument(args.reference_storage_options_json),
        "--reference-manifest-uri",
        args.reference_manifest_uri,
        "--reference-manifest-sha256",
        args.reference_manifest_sha256,
        "--expected-reference-rows",
        str(args.expected_reference_rows),
        "--task-rows",
        str(geometry.task_rows),
        "--coalesce-tasks",
        str(geometry.coalesce_tasks),
        "--fetch-batch-size",
        str(args.fetch_batch_size),
        "--max-lookup-bytes-mib",
        str(args.max_lookup_bytes_mib),
        "--max-pending-fetch-batches",
        str(args.max_pending_fetch_batches),
        "--io-threads",
        str(args.io_threads_per_actor),
        "--ray-address",
        ray_address,
        "--ray-gpu-actors",
        str(geometry.actor_count),
        "--actor-warmup-rows",
        str(args.actor_warmup_rows),
        "--warmup-count",
        str(args.warmup_count),
        "--repeat-count",
        str(args.repeat_count),
        "--arm",
        args.arm,
        "--evidence-class",
        geometry.evidence_class,
        "--output",
        str(report_path.resolve()),
    ]
    for pattern in args.reference_glob:
        command.extend(["--reference-glob", pattern])
    if args.payload_projection in {"image_only", "image_url"}:
        command.extend(["--md5-column", "", "--width-column", "", "--height-column", ""])
    if args.payload_projection in {"image_url", "full"}:
        command.append("--validate-payload-keys")
    if args.copy_reference_to_node_local:
        command.extend(
            [
                "--copy-reference-to-node-local",
                "--reference-node-local-root",
                args.reference_node_local_root,
            ]
        )
    return command


def sanitized_command(command: Sequence[str]) -> str:
    """Return a shell-rendered command with secret-bearing values redacted."""

    redacted: list[str] = []
    hide_next = False
    redact_uri_next = False
    for value in command:
        if hide_next:
            if value.startswith("@"):
                redacted.append(f"@{Path(value[1:]).name}")
            else:
                try:
                    keys = sorted(json.loads(value))
                    redacted.append(f"<redacted-json keys={','.join(keys)}>")
                except (json.JSONDecodeError, TypeError):
                    redacted.append("<redacted-json>")
            hide_next = False
            continue
        if redact_uri_next:
            redacted.append(redact_uri_identity(value))
            redact_uri_next = False
            continue
        redacted.append(value)
        hide_next = value in {"--storage-options-json", "--reference-storage-options-json"}
        redact_uri_next = value in {"--image-lance-uri", "--reference-manifest-uri", "--reference-glob"}
    return shlex.join(redacted)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _short_hostname(value: str) -> str:
    return value.split(".", maxsplit=1)[0]


def _allocated_node_identities(nodes: int) -> dict[int, str]:
    """Resolve the scheduler's ordered node identities for telemetry validation."""

    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        if nodes != 1:
            msg = "SLURM_JOB_NODELIST is required to validate multi-node telemetry"
            raise RuntimeError(msg)
        return {0: _short_hostname(socket.gethostname())}
    scontrol = shutil.which("scontrol")
    if scontrol is None:
        msg = "scontrol is required to resolve allocated node identities"
        raise RuntimeError(msg)
    try:
        completed = subprocess.run(  # noqa: S603
            [scontrol, "show", "hostnames", nodelist],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"failed to resolve allocated node identities: {type(exc).__name__}: {exc}"
        raise RuntimeError(msg) from exc
    hostnames = [_short_hostname(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode:
        msg = f"scontrol show hostnames failed with exit code {completed.returncode}"
        raise RuntimeError(msg)
    if len(hostnames) != nodes or len(set(hostnames)) != nodes:
        msg = f"resolved allocated nodes {hostnames}; expected {nodes} unique node identities"
        raise RuntimeError(msg)
    return dict(enumerate(hostnames))


def _telemetry_command(
    args: argparse.Namespace,
    report_path: Path,
    output: Path,
    *,
    node_id: int,
    hostname: str,
) -> list[str]:
    script = Path(__file__).with_name("gpu_lance_saturation_telemetry.py")
    command = [
        sys.executable,
        str(script),
        "--output",
        str(output),
        "--benchmark-report",
        str(report_path),
        "--arm",
        args.arm,
        "--warmup-count",
        str(args.warmup_count),
        "--repeat-count",
        str(args.repeat_count),
        "--interval-seconds",
        str(args.telemetry_interval_seconds),
        "--node-id",
        str(node_id),
        "--expected-hostname",
        hostname,
    ]
    for path in args.filesystem_path:
        command.extend(["--filesystem-path", str(path)])
    return command


def _start_telemetry(
    args: argparse.Namespace,
    report_path: Path,
    *,
    node_id: int,
    hostname: str,
) -> TelemetryHandle:
    telemetry_dir = args.output_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    output = telemetry_dir / f"node_{node_id:04d}.jsonl"
    if output.exists():
        msg = f"telemetry output already exists: {output}"
        raise FileExistsError(msg)
    log_stream = (telemetry_dir / f"node_{node_id:04d}.stderr.log").open("x", encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603
        _telemetry_command(args, report_path, output, node_id=node_id, hostname=hostname),
        stdout=subprocess.DEVNULL,
        stderr=log_stream,
        text=True,
    )
    return TelemetryHandle(process=process, log_stream=log_stream, output=output, node_id=node_id, hostname=hostname)


def _stop_telemetry(handle: TelemetryHandle) -> dict[str, Any]:
    process = handle.process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    handle.log_stream.close()
    returncode = process.returncode
    return {
        "status": "passed" if returncode == 0 else "failed",
        "returncode": returncode,
        "output": str(handle.output),
        "node_id": handle.node_id,
        "hostname": handle.hostname,
        "failures": [] if returncode == 0 else [f"telemetry collector exited with code {returncode}"],
    }


def _finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _plausible_gpu(gpu: object) -> bool:
    if not isinstance(gpu, Mapping):
        return False
    utilization = gpu.get("utilization.gpu")
    used = gpu.get("memory.used")
    total = gpu.get("memory.total")
    return (
        _finite_number(utilization)
        and 0 <= float(utilization) <= _MAX_PERCENT
        and _finite_number(used)
        and _finite_number(total)
        and 0 <= float(used) <= float(total)
        and float(total) > 0
    )


def _plausible_counter_map(value: object, required: set[str]) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(counters, Mapping)
            and required <= set(counters)
            and all(isinstance(counters[name], int) and counters[name] >= 0 for name in required)
            for counters in value.values()
        )
    )


def _plausible_sample(sample: Mapping[str, Any], expected_gpu_count: int) -> bool:
    cpu = sample.get("cpu")
    gpus = sample.get("gpus")
    network = sample.get("network")
    block_devices = sample.get("block_devices")
    filesystems = sample.get("filesystems")
    errors = sample.get("errors")
    return (
        isinstance(errors, list)
        and not errors
        and isinstance(cpu, Mapping)
        and isinstance(cpu.get("logical_cpus"), int)
        and cpu["logical_cpus"] > 0
        and _plausible_counter_map(network, {"receive_bytes", "transmit_bytes"})
        and _plausible_counter_map(block_devices, {"reads_completed", "sectors_read"})
        and _plausible_counter_map(filesystems, {"total_bytes", "free_bytes"})
        and isinstance(gpus, list)
        and len(gpus) == expected_gpu_count
        and all(_plausible_gpu(gpu) for gpu in gpus)
    )


def _read_telemetry_records(path: Path) -> tuple[list[Mapping[str, Any]], list[str]]:
    records: list[Mapping[str, Any]] = []
    failures: list[str] = []
    if not path.is_file():
        failures.append(f"missing telemetry artifact {path.name}")
        return records, failures
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            record = json.loads(line)
            if isinstance(record, Mapping):
                records.append(record)
            else:
                failures.append(f"line {line_number} is not a JSON object")
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid telemetry JSONL: {type(exc).__name__}: {exc}")
    return records, failures


def _record_structure(
    records: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], list[str]]:
    failures: list[str] = []
    samples = [record for record in records if record.get("record_type") == "sample"]
    summaries = [record for record in records if record.get("record_type") == "summary"]
    unexpected = [
        record.get("record_type") for record in records if record.get("record_type") not in {"sample", "summary"}
    ]
    if unexpected:
        failures.append(f"unexpected telemetry record types: {unexpected}")
    if not samples:
        failures.append("telemetry stream contains zero samples")
    if len(summaries) != 1:
        failures.append(f"telemetry stream contains {len(summaries)} summaries; expected exactly one")
    summary = summaries[0] if len(summaries) == 1 else {}
    if records and (not summaries or records[-1] is not summary):
        failures.append("terminal telemetry summary is missing or is not the last record")
    return samples, summary, failures


def _summary_failures(  # noqa: C901
    summary: Mapping[str, Any],
    samples: list[Mapping[str, Any]],
    phases: Counter[str],
    spec: TelemetryValidationSpec,
) -> list[str]:
    failures: list[str] = []
    if summary.get("status") != "complete":
        failures.append(f"telemetry summary status is {summary.get('status')!r}; expected 'complete'")
    if summary.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
        failures.append(
            f"telemetry summary schema_version is {summary.get('schema_version')}; expected {TELEMETRY_SCHEMA_VERSION}"
        )
    if summary.get("node_id") != spec.node_id:
        failures.append(f"telemetry summary node_id is {summary.get('node_id')}; expected {spec.node_id}")
    if _short_hostname(str(summary.get("hostname", ""))) != spec.hostname:
        failures.append(f"telemetry summary hostname is {summary.get('hostname')!r}; expected {spec.hostname!r}")
    if summary.get("sample_count") != len(samples):
        failures.append(f"telemetry summary sample_count is {summary.get('sample_count')}; observed {len(samples)}")
    summary_interval = summary.get("interval_seconds")
    if not _finite_number(summary_interval) or not math.isclose(
        float(summary_interval), spec.interval_seconds, rel_tol=1e-9, abs_tol=1e-9
    ):
        failures.append(f"telemetry summary interval is {summary_interval}; expected {spec.interval_seconds}")
    summary_started = summary.get("started_timestamp_epoch")
    summary_finished = summary.get("finished_timestamp_epoch")
    summary_duration = summary.get("duration_seconds")
    if not _finite_number(summary_started) or not _finite_number(summary_finished):
        failures.append("telemetry summary has invalid start/finish timestamps")
    elif float(summary_finished) < float(summary_started):
        failures.append("telemetry summary finishes before it starts")
    if not _finite_number(summary_duration) or float(summary_duration) < 0:
        failures.append(f"telemetry summary has invalid duration_seconds {summary_duration!r}")
    if summary.get("phase_counts") != dict(sorted(phases.items())):
        failures.append("telemetry summary phase_counts do not match sample records")
    return failures


def _sample_observations(  # noqa: C901, PLR0912, PLR0915
    samples: list[Mapping[str, Any]], spec: TelemetryValidationSpec
) -> dict[str, Any]:
    failures: list[str] = []
    timestamps: list[float] = []
    monotonic_times: list[float] = []
    phases: Counter[str] = Counter()
    plausible_samples = 0
    for sample_index, sample in enumerate(samples):
        if sample.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
            failures.append(
                f"sample {sample_index} schema_version is {sample.get('schema_version')}; "
                f"expected {TELEMETRY_SCHEMA_VERSION}"
            )
        if sample.get("node_id") != spec.node_id:
            failures.append(f"sample {sample_index} node_id is {sample.get('node_id')}; expected {spec.node_id}")
        if _short_hostname(str(sample.get("hostname", ""))) != spec.hostname:
            failures.append(
                f"sample {sample_index} hostname is {sample.get('hostname')!r}; expected {spec.hostname!r}"
            )
        timestamp = sample.get("timestamp_epoch")
        monotonic = sample.get("monotonic_seconds")
        if _finite_number(timestamp) and float(timestamp) > 0:
            timestamps.append(float(timestamp))
        else:
            failures.append(f"sample {sample_index} has invalid timestamp_epoch {timestamp!r}")
        if _finite_number(monotonic) and float(monotonic) >= 0:
            monotonic_times.append(float(monotonic))
        else:
            failures.append(f"sample {sample_index} has invalid monotonic_seconds {monotonic!r}")
        phases[str(sample.get("phase", ""))] += 1
        plausible_samples += int(_plausible_sample(sample, spec.gpu_count))
    if timestamps != sorted(timestamps) or monotonic_times != sorted(monotonic_times):
        failures.append("telemetry sample times are not monotonic")
    max_gap = max((second - first for first, second in pairwise(monotonic_times)), default=None)
    allowed_gap = max(spec.interval_seconds * 4, spec.interval_seconds + 5)
    if max_gap is not None and max_gap > allowed_gap:
        failures.append(f"telemetry maximum sample gap is {max_gap:.3f}s; expected at most {allowed_gap:.3f}s")
    if samples and plausible_samples != len(samples):
        failures.append(
            f"only {plausible_samples} of {len(samples)} telemetry samples contain complete plausible host/GPU/storage metrics"
        )
    required_steady_phases = tuple(f"steady_repeat_{repeat}" for repeat in range(spec.required_steady_repeat_count))
    observed_steady_phases = {phase for phase in phases if phase.startswith("steady_repeat_")}
    missing_steady_phases = [phase for phase in required_steady_phases if phases[phase] < MINIMUM_DELTA_SAMPLE_COUNT]
    unexpected_steady_phases = sorted(observed_steady_phases - set(required_steady_phases))
    if missing_steady_phases:
        failures.append(
            "telemetry requires at least two samples in every configured steady repeat; "
            f"missing or undersampled phases: {missing_steady_phases}"
        )
    if unexpected_steady_phases:
        failures.append(f"telemetry contains unexpected steady repeat phases: {unexpected_steady_phases}")

    network_receive_delta = 0
    block_read_sector_delta = 0
    steady_delta_sample_count = 0
    network_receive_deltas_by_phase: dict[str, int] = {}
    block_read_sector_deltas_by_phase: dict[str, int] = {}
    for phase in required_steady_phases:
        phase_samples = [sample for sample in samples if sample.get("phase") == phase]
        if len(phase_samples) < MINIMUM_DELTA_SAMPLE_COUNT:
            continue
        steady_delta_sample_count += len(phase_samples)
        phase_network_delta = sum(
            int(counters.get("receive_bytes", 0))
            for interface, counters in phase_samples[-1].get("network", {}).items()
            if interface != "lo"
        ) - sum(
            int(counters.get("receive_bytes", 0))
            for interface, counters in phase_samples[0].get("network", {}).items()
            if interface != "lo"
        )
        phase_block_delta = sum(
            int(counters.get("sectors_read", 0)) for counters in phase_samples[-1].get("block_devices", {}).values()
        ) - sum(
            int(counters.get("sectors_read", 0)) for counters in phase_samples[0].get("block_devices", {}).values()
        )
        if phase_network_delta < 0 or phase_block_delta < 0:
            failures.append(f"telemetry counters decreased within {phase}")
        if spec.storage_axis in {"remote_s3", "lustre"} and phase_network_delta <= 0:
            failures.append(
                f"{spec.storage_axis} telemetry has no positive non-loopback receive-byte delta in {phase}"
            )
        if spec.storage_axis == "node_local_nvme" and phase_block_delta <= 0:
            failures.append(f"node-local NVMe telemetry has no positive block-read delta in {phase}")
        network_receive_deltas_by_phase[phase] = phase_network_delta
        block_read_sector_deltas_by_phase[phase] = phase_block_delta
        network_receive_delta += phase_network_delta
        block_read_sector_delta += phase_block_delta

    covered_steady_state = not missing_steady_phases and not unexpected_steady_phases
    return {
        "failures": failures,
        "timestamps": timestamps,
        "monotonic_times": monotonic_times,
        "phases": phases,
        "plausible_samples": plausible_samples,
        "max_gap": max_gap,
        "covered_steady_state": covered_steady_state,
        "required_steady_phases": list(required_steady_phases),
        "missing_steady_phases": missing_steady_phases,
        "steady_delta_sample_count": steady_delta_sample_count,
        "network_receive_bytes_delta_by_phase": network_receive_deltas_by_phase,
        "block_read_sectors_delta_by_phase": block_read_sector_deltas_by_phase,
        "network_receive_bytes_delta": network_receive_delta,
        "block_read_sectors_delta": block_read_sector_delta,
    }


def _interval_failures(summary: Mapping[str, Any], observations: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    timestamps = observations["timestamps"]
    monotonic_times = observations["monotonic_times"]
    summary_started = summary.get("started_timestamp_epoch")
    summary_finished = summary.get("finished_timestamp_epoch")
    if (
        timestamps
        and _finite_number(summary_started)
        and _finite_number(summary_finished)
        and (timestamps[0] + 1 < float(summary_started) or timestamps[-1] > float(summary_finished) + 1)
    ):
        failures.append("telemetry samples fall outside the summary collection interval")
    last_sample_monotonic = summary.get("last_sample_monotonic_seconds")
    if monotonic_times and (
        not _finite_number(last_sample_monotonic)
        or not math.isclose(float(last_sample_monotonic), monotonic_times[-1], rel_tol=1e-9, abs_tol=1e-9)
    ):
        failures.append("telemetry summary last sample time does not match the stream")
    return failures


def validate_telemetry_artifact(path: Path, spec: TelemetryValidationSpec) -> dict[str, Any]:
    """Validate one atomic JSONL stream, including its terminal summary."""

    records, failures = _read_telemetry_records(path)
    samples, summary, structure_failures = _record_structure(records)
    failures.extend(structure_failures)
    observations = _sample_observations(samples, spec)
    failures.extend(observations["failures"])
    failures.extend(_summary_failures(summary, samples, observations["phases"], spec))
    failures.extend(_interval_failures(summary, observations))

    return {
        "status": "passed" if not failures else "failed",
        "path": str(path),
        "expected_node_id": spec.node_id,
        "expected_hostname": spec.hostname,
        "observed_hostname": summary.get("hostname"),
        "sample_count": len(samples),
        "plausible_sample_count": observations["plausible_samples"],
        "max_sample_gap_seconds": observations["max_gap"],
        "steady_state_observed": observations["covered_steady_state"],
        "required_steady_phases": observations["required_steady_phases"],
        "missing_steady_phases": observations["missing_steady_phases"],
        "steady_delta_sample_count": observations["steady_delta_sample_count"],
        "storage_axis": spec.storage_axis,
        "network_receive_bytes_delta_by_phase": observations["network_receive_bytes_delta_by_phase"],
        "block_read_sectors_delta_by_phase": observations["block_read_sectors_delta_by_phase"],
        "network_receive_bytes_delta": observations["network_receive_bytes_delta"],
        "block_read_sectors_delta": observations["block_read_sectors_delta"],
        "failures": failures,
    }


def validate_telemetry_cluster(telemetry_dir: Path, spec: TelemetryClusterSpec) -> dict[str, Any]:
    """Wait for and validate exactly one complete stream per allocated node."""

    expected_paths = {node_id: telemetry_dir / f"node_{node_id:04d}.jsonl" for node_id in spec.nodes}
    deadline = time.monotonic() + spec.wait_seconds
    while not all(path.is_file() for path in expected_paths.values()) and time.monotonic() < deadline:
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    actual_paths = set(telemetry_dir.glob("node_*.jsonl"))
    missing = sorted(path.name for path in expected_paths.values() if path not in actual_paths)
    unexpected = sorted(path.name for path in actual_paths if path not in set(expected_paths.values()))
    nodes = {
        str(node_id): validate_telemetry_artifact(
            path,
            TelemetryValidationSpec(
                node_id=node_id,
                hostname=_short_hostname(spec.nodes[node_id]),
                gpu_count=spec.gpu_count,
                interval_seconds=spec.interval_seconds,
                required_steady_repeat_count=spec.repeat_count,
                storage_axis=spec.storage_axis,
            ),
        )
        for node_id, path in expected_paths.items()
    }
    failures = [f"missing telemetry artifacts: {missing}"] if missing else []
    if unexpected:
        failures.append(f"unexpected telemetry artifacts: {unexpected}")
    for node_id, validation in nodes.items():
        failures.extend(f"node {node_id}: {failure}" for failure in validation["failures"])
    observed_hostnames = [validation.get("observed_hostname") for validation in nodes.values()]
    if all(observed_hostnames) and len(set(observed_hostnames)) != len(observed_hostnames):
        failures.append(f"telemetry node hostnames are not unique: {observed_hostnames}")
    return {
        "status": "passed" if not failures else "failed",
        "expected_nodes": {str(node_id): hostname for node_id, hostname in spec.nodes.items()},
        "required_steady_state_coverage": spec.repeat_count > 0,
        "required_steady_repeat_count": spec.repeat_count,
        "nodes": nodes,
        "missing": missing,
        "unexpected": unexpected,
        "failures": failures,
    }


def _configuration_failures(configuration: Mapping[str, Any], geometry: SaturationGeometry) -> list[str]:
    expected_configuration = {
        "ray_actor_pool_size": geometry.actor_count,
        "ray_actor_input_blocks": geometry.target_tasks,
        "ray_actor_input_block_rows": geometry.task_rows,
        "ray_actor_coalesce_tasks": geometry.coalesce_tasks,
        "ray_actor_target_batch_rows": geometry.actor_batch_rows,
        "rows_per_coalesced_fetch": geometry.actor_batch_rows,
    }
    return [
        f"configuration {name} is {configuration.get(name)}; expected {expected}"
        for name, expected in expected_configuration.items()
        if configuration.get(name) != expected
    ]


def _persistent_pool_failures(arm_result: Mapping[str, Any], geometry: SaturationGeometry) -> list[str]:
    cold_setup = arm_result.get("cold_setup")
    if not isinstance(cold_setup, Mapping):
        return ["arm cold_setup is missing or invalid"]
    metrics = cold_setup.get("backend_metrics")
    if not isinstance(metrics, Mapping):
        return ["arm cold_setup backend_metrics is missing or invalid"]
    failures = []
    if metrics.get("persistent_actor_pool") is not True:
        failures.append("cold setup does not attest a persistent actor pool")
    if metrics.get("persistent_actor_count") != geometry.actor_count:
        failures.append(
            f"cold setup persistent_actor_count is {metrics.get('persistent_actor_count')}; "
            f"expected {geometry.actor_count}"
        )
    return failures


def _warmup_failures(warmup: Mapping[str, Any], warmup_index: int) -> list[str]:
    failures = []
    correctness = warmup.get("correctness")
    if (
        warmup.get("status") != "completed"
        or not isinstance(correctness, Mapping)
        or correctness.get("correct") is not True
    ):
        failures.append(f"warmup {warmup_index} did not pass correctness")
    wall_seconds = warmup.get("wall_seconds")
    if not _finite_number(wall_seconds) or float(wall_seconds) <= 0:
        failures.append(f"warmup {warmup_index} wall_seconds must be finite and positive; got {wall_seconds!r}")
    digest = correctness.get("output_digest_sha256") if isinstance(correctness, Mapping) else None
    if not _valid_sha256(digest):
        failures.append(f"warmup {warmup_index} correctness digest is missing or invalid")
    return failures


def _metric_number(
    mapping: Mapping[str, Any],
    name: str,
    *,
    repeat_index: int,
    failures: list[str],
    positive: bool = False,
) -> float | None:
    value = mapping.get(name)
    valid = _finite_number(value) and (float(value) > 0 if positive else float(value) >= 0)
    if not valid:
        qualifier = "positive" if positive else "nonnegative"
        failures.append(f"repeat {repeat_index} metric {name} must be finite and {qualifier}; got {value!r}")
        return None
    return float(value)


def _require_metric_close(
    *,
    actual: float | None,
    expected: float | None,
    name: str,
    repeat_index: int,
    failures: list[str],
) -> None:
    if actual is None or expected is None:
        return
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
        failures.append(
            f"repeat {repeat_index} metric {name}={actual} does not reconcile with additive counters ({expected})"
        )


def _repeat_timing_failures(
    repeat: Mapping[str, Any],
    repeat_index: int,
    geometry: SaturationGeometry,
    *,
    require_zero_setup: bool,
) -> list[str]:
    failures: list[str] = []
    wall_seconds = _metric_number(
        repeat,
        "wall_seconds",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    process_seconds = _metric_number(
        repeat,
        "warm_process_seconds",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    cold_setup = _metric_number(
        repeat,
        "cold_setup_seconds",
        repeat_index=repeat_index,
        failures=failures,
    )
    internal_warmup = _metric_number(
        repeat,
        "internal_warmup_seconds",
        repeat_index=repeat_index,
        failures=failures,
    )
    if wall_seconds is not None and process_seconds is not None and process_seconds > wall_seconds:
        failures.append(
            f"repeat {repeat_index} warm_process_seconds={process_seconds} exceeds wall_seconds={wall_seconds}"
        )
    if require_zero_setup:
        for name, value in (("cold_setup_seconds", cold_setup), ("internal_warmup_seconds", internal_warmup)):
            if value is not None and not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-9):
                failures.append(f"repeat {repeat_index} {name}={value}; expected 0 for steady-state timing")

    metrics = repeat.get("backend_metrics")
    backend_process_seconds = (
        _metric_number(
            metrics,
            "process_seconds",
            repeat_index=repeat_index,
            failures=failures,
            positive=True,
        )
        if isinstance(metrics, Mapping)
        else None
    )
    _require_metric_close(
        actual=process_seconds,
        expected=backend_process_seconds,
        name="warm_process_seconds",
        repeat_index=repeat_index,
        failures=failures,
    )

    images_per_second = _metric_number(
        repeat,
        "images_per_second",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    payload_rate = _metric_number(
        repeat,
        "payload_mib_per_second",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    payload_bytes = _metric_number(
        repeat,
        "payload_bytes",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    correctness = repeat.get("correctness")
    correctness_payload_bytes = (
        _metric_number(
            correctness,
            "payload_bytes",
            repeat_index=repeat_index,
            failures=failures,
            positive=True,
        )
        if isinstance(correctness, Mapping)
        else None
    )
    if not isinstance(correctness, Mapping):
        failures.append(f"repeat {repeat_index} correctness is missing or invalid")
    _require_metric_close(
        actual=payload_bytes,
        expected=correctness_payload_bytes,
        name="payload_bytes",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=images_per_second,
        expected=(geometry.target_rows / process_seconds if process_seconds is not None else None),
        name="images_per_second",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=payload_rate,
        expected=(
            payload_bytes / (1024**2 * process_seconds)
            if payload_bytes is not None and process_seconds is not None
            else None
        ),
        name="payload_mib_per_second",
        repeat_index=repeat_index,
        failures=failures,
    )
    return failures


def _io_reconciliation_failures(
    repeat: Mapping[str, Any],
    repeat_index: int,
    geometry: SaturationGeometry,
    configuration: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    metrics_value = repeat.get("backend_metrics")
    if not isinstance(metrics_value, Mapping):
        return [f"repeat {repeat_index} backend_metrics is missing or invalid"]
    metrics = metrics_value
    read_bytes = _metric_number(
        repeat, "lance_read_bytes", repeat_index=repeat_index, failures=failures, positive=True
    )
    read_iops = _metric_number(repeat, "lance_read_iops", repeat_index=repeat_index, failures=failures, positive=True)
    fetched_bytes = _metric_number(
        metrics,
        "lance_fetched_bytes",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    average_read_bytes = _metric_number(
        metrics,
        "average_physical_read_bytes",
        repeat_index=repeat_index,
        failures=failures,
    )
    read_amplification = _metric_number(
        metrics,
        "read_amplification",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=average_read_bytes,
        expected=(read_bytes / read_iops if read_bytes is not None and read_iops is not None else None),
        name="average_physical_read_bytes",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=read_amplification,
        expected=(read_bytes / fetched_bytes if read_bytes is not None and fetched_bytes is not None else None),
        name="read_amplification",
        repeat_index=repeat_index,
        failures=failures,
    )

    payload_calls = _metric_number(
        metrics,
        "payload_take_calls",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    payload_rows = _metric_number(
        metrics,
        "payload_take_rows",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    rows_per_take = _metric_number(
        metrics,
        "rows_per_payload_take",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    sparse_calls_avoided = _metric_number(
        metrics,
        "sparse_calls_avoided",
        repeat_index=repeat_index,
        failures=failures,
    )
    take_rows_calls = _metric_number(
        metrics,
        "take_rows_calls",
        repeat_index=repeat_index,
        failures=failures,
    )
    take_scan_calls = _metric_number(
        metrics,
        "take_scan_calls",
        repeat_index=repeat_index,
        failures=failures,
    )
    strategy_sparse = _metric_number(
        metrics,
        "strategy_sparse_fragments",
        repeat_index=repeat_index,
        failures=failures,
    )
    strategy_range = _metric_number(
        metrics,
        "strategy_range_fragments",
        repeat_index=repeat_index,
        failures=failures,
    )
    strategy_sequential = _metric_number(
        metrics,
        "strategy_sequential_fragments",
        repeat_index=repeat_index,
        failures=failures,
    )
    planned_scan_rows = _metric_number(
        metrics,
        "planned_scan_rows",
        repeat_index=repeat_index,
        failures=failures,
    )
    range_overread_rows = _metric_number(
        metrics,
        "range_overread_rows",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=rows_per_take,
        expected=(payload_rows / payload_calls if payload_rows is not None and payload_calls is not None else None),
        name="rows_per_payload_take",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=sparse_calls_avoided,
        expected=(payload_rows - payload_calls if payload_rows is not None and payload_calls is not None else None),
        name="sparse_calls_avoided",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=payload_calls,
        expected=(
            take_rows_calls + take_scan_calls if take_rows_calls is not None and take_scan_calls is not None else None
        ),
        name="payload_take_calls",
        repeat_index=repeat_index,
        failures=failures,
    )
    found_unique = _metric_number(
        metrics,
        "found_unique_keys",
        repeat_index=repeat_index,
        failures=failures,
        positive=True,
    )
    duplicate_queries = _metric_number(
        metrics,
        "duplicate_queries_coalesced",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=payload_rows,
        expected=found_unique,
        name="payload_take_rows",
        repeat_index=repeat_index,
        failures=failures,
    )
    _require_metric_close(
        actual=(
            found_unique + duplicate_queries if found_unique is not None and duplicate_queries is not None else None
        ),
        expected=float(geometry.target_rows),
        name="found_unique_keys+duplicate_queries_coalesced",
        repeat_index=repeat_index,
        failures=failures,
    )

    payload_read_mode = configuration.get("payload_read_mode")
    if payload_read_mode != "sparse":
        failures.append(
            f"repeat {repeat_index} saturation payload_read_mode must be 'sparse'; got {payload_read_mode!r}"
        )
    elif any(
        value not in {0.0, None}
        for value in (take_scan_calls, strategy_range, strategy_sequential, planned_scan_rows, range_overread_rows)
    ):
        failures.append(f"repeat {repeat_index} sparse policy reported range/sequential scan activity")
    if strategy_sparse is not None and strategy_sparse <= 0:
        failures.append(f"repeat {repeat_index} sparse policy reported no sparse fragments")
    return failures


def _repeat_failures(
    repeat: Mapping[str, Any],
    repeat_index: int,
    geometry: SaturationGeometry,
    configuration: Mapping[str, Any],
    *,
    require_zero_setup: bool,
) -> list[str]:
    failures: list[str] = []
    metrics = repeat.get("backend_metrics") or {}
    correctness = repeat.get("correctness")
    if (
        repeat.get("status") != "completed"
        or not isinstance(correctness, Mapping)
        or correctness.get("correct") is not True
    ):
        failures.append(f"repeat {repeat_index} did not pass correctness")
    failures.extend(
        _repeat_timing_failures(
            repeat,
            repeat_index,
            geometry,
            require_zero_setup=require_zero_setup,
        )
    )
    used = metrics.get("ray_gpu_actors_used")
    if used != geometry.actor_count:
        failures.append(f"repeat {repeat_index} used {used} actors; expected {geometry.actor_count}")
    input_blocks = metrics.get("ray_input_blocks")
    if input_blocks != geometry.target_tasks:
        failures.append(
            f"repeat {repeat_index} consumed {input_blocks} input blocks; expected {geometry.target_tasks}"
        )
    actor_calls = repeat.get("fetch_calls")
    if actor_calls != geometry.expected_actor_calls:
        failures.append(
            f"repeat {repeat_index} made {actor_calls} actor calls; expected {geometry.expected_actor_calls}"
        )
    failures.extend(_io_reconciliation_failures(repeat, repeat_index, geometry, configuration))
    return failures


def _warmup_collection_failures(
    warmups: Sequence[Any], warmup_count: int, configuration: Mapping[str, Any]
) -> list[str]:
    failures = []
    if configuration.get("warmup_count") != warmup_count:
        failures.append(f"configuration warmup_count is {configuration.get('warmup_count')}; expected {warmup_count}")
    if len(warmups) != warmup_count:
        failures.append(f"warmup count is {len(warmups)}; expected {warmup_count}")
    for warmup_index, warmup in enumerate(warmups):
        if not isinstance(warmup, Mapping):
            failures.append(f"warmup {warmup_index} is not an object")
        else:
            failures.extend(_warmup_failures(warmup, warmup_index))
    return failures


def _correctness_digest_failures(warmups: Sequence[Any], repeats: Sequence[Any]) -> list[str]:
    repeat_digests = {
        (repeat.get("correctness") or {}).get("output_digest_sha256")
        for repeat in repeats
        if isinstance(repeat, Mapping)
    }
    failures = []
    if len(repeat_digests) != 1 or not all(_valid_sha256(digest) for digest in repeat_digests):
        failures.append("repeat correctness digests are missing or unstable")
    warmup_digests = {
        (warmup.get("correctness") or {}).get("output_digest_sha256")
        for warmup in warmups
        if isinstance(warmup, Mapping)
    }
    if warmups and (
        None in warmup_digests
        or len(warmup_digests) != 1
        or len(repeat_digests) != 1
        or warmup_digests != repeat_digests
    ):
        failures.append("warmup and repeat correctness digests are missing or unstable")
    return failures


def validate_benchmark_report(
    report_path: Path,
    arm: str,
    geometry: SaturationGeometry,
    repeat_count: int,
    warmup_count: int,
) -> dict[str, Any]:
    """Fail unless warmup and repeats satisfy the saturation contract."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    arm_result = report.get("arms", {}).get(arm, {})
    configuration = report.get("configuration") or {}
    failures: list[str] = []
    if repeat_count < MINIMUM_REPEAT_COUNT:
        failures.append(f"requested repeat count is {repeat_count}; expected at least {MINIMUM_REPEAT_COUNT}")
    if report.get("status") != "completed":
        failures.append(f"report status is {report.get('status')!r}")
    if arm_result.get("status") != "completed":
        failures.append(f"arm status is {arm_result.get('status')!r}")
    warmups = arm_result.get("warmups") or []
    repeats = arm_result.get("repeats") or []
    if configuration.get("repeat_count") != repeat_count:
        failures.append(f"configuration repeat_count is {configuration.get('repeat_count')}; expected {repeat_count}")
    failures.extend(_warmup_collection_failures(warmups, warmup_count, configuration))
    if len(repeats) != repeat_count:
        failures.append(f"repeat count is {len(repeats)}; expected {repeat_count}")
    failures.extend(_configuration_failures(configuration, geometry))
    require_persistent_pool = arm == "lance_ray_gpu_actor"
    if require_persistent_pool:
        failures.extend(_persistent_pool_failures(arm_result, geometry))
    for repeat_index, repeat in enumerate(repeats):
        if not isinstance(repeat, Mapping):
            failures.append(f"repeat {repeat_index} is not an object")
            continue
        failures.extend(
            _repeat_failures(
                repeat,
                repeat_index,
                geometry,
                configuration,
                require_zero_setup=require_persistent_pool,
            )
        )
    failures.extend(_correctness_digest_failures(warmups, repeats))
    observed = {
        "ray_actor_stage_windows": [repeat.get("fetch_calls") for repeat in repeats],
        "private_take_calls": [(repeat.get("backend_metrics") or {}).get("payload_take_calls") for repeat in repeats],
        "fragment_strategy_calls": [
            (repeat.get("backend_metrics") or {}).get("fragment_take_calls") for repeat in repeats
        ],
        "physical_io_tracker_reads": [repeat.get("lance_read_iops") for repeat in repeats],
    }
    return {
        "status": "passed" if not failures else "failed",
        "evidence_class": geometry.evidence_class,
        "arm": arm,
        "nodes": geometry.nodes,
        "actor_count": geometry.actor_count,
        "tasks_per_actor": geometry.tasks_per_actor,
        "waves": geometry.waves,
        "expected_actor_calls": geometry.expected_actor_calls,
        "warmup_count": warmup_count,
        "observed": observed,
        "metric_definitions": {
            "ray_actor_stage_windows": "Harness fetch_calls for Ray actor arms: len(process_values).",
            "private_take_calls": "Sum of backend payload_take_calls; one stage window can issue several.",
            "fragment_strategy_calls": "Public fragment strategy calls when that exploratory path is active.",
            "physical_io_tracker_reads": "Lance IOTracker read-operation counter, not Python API calls.",
        },
        "target_rows": geometry.target_rows,
        "failures": failures,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> tuple[Mapping[str, Any] | None, list[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{label} is unreadable: {type(exc).__name__}: {exc}"]
    if not isinstance(value, Mapping):
        return None, [f"{label} must contain a JSON object"]
    return value, []


def _terminal_identity_failures(  # noqa: C901, PLR0912, PLR0913, PLR0915
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    arm: str,
    geometry: SaturationGeometry,
    repeat_count: int,
    warmup_count: int,
) -> list[str]:
    failures: list[str] = []
    identity_schema_version = identity.get("schema_version")
    if identity_schema_version != RUN_IDENTITY_SCHEMA_VERSION:
        failures.append(
            f"run identity schema_version is {identity_schema_version!r}; expected {RUN_IDENTITY_SCHEMA_VERSION}"
        )
    identity_evidence_class = identity.get("evidence_class")
    if identity_evidence_class != geometry.evidence_class:
        failures.append(
            f"run identity evidence_class is {identity_evidence_class!r}; expected {geometry.evidence_class!r}"
        )
    expected_geometry = {
        "nodes": geometry.nodes,
        "actors_per_node": geometry.actors_per_node,
        "actor_count": geometry.actor_count,
        "tasks_per_actor": geometry.tasks_per_actor,
        "task_rows": geometry.task_rows,
        "target_rows": geometry.target_rows,
        "waves": geometry.waves,
        "coalesce_tasks": geometry.coalesce_tasks,
        "actor_batch_rows": geometry.actor_batch_rows,
    }
    recorded_geometry = identity.get("geometry")
    if not isinstance(recorded_geometry, Mapping):
        failures.append("run identity geometry is missing or invalid")
        recorded_geometry = {}
    for name, expected in expected_geometry.items():
        if recorded_geometry.get(name) != expected:
            failures.append(f"run identity geometry.{name} is {recorded_geometry.get(name)!r}; expected {expected}")

    configuration = report.get("configuration")
    dataset = report.get("dataset")
    report_manifest = report.get("manifest")
    environment = report.get("environment")
    if not isinstance(configuration, Mapping):
        failures.append("benchmark configuration is missing or invalid")
        configuration = {}
    if not isinstance(dataset, Mapping):
        failures.append("benchmark dataset identity is missing or invalid")
        dataset = {}
    if not isinstance(report_manifest, Mapping):
        failures.append("benchmark manifest identity is missing or invalid")
        report_manifest = {}
    if not isinstance(environment, Mapping):
        failures.append("benchmark environment identity is missing or invalid")
        environment = {}
    if report.get("evidence_class") != geometry.evidence_class:
        failures.append(
            f"benchmark evidence_class is {report.get('evidence_class')!r}; expected {geometry.evidence_class!r}"
        )
    packages = environment.get("packages")
    if not isinstance(environment.get("python"), str) or not environment.get("python"):
        failures.append("benchmark environment lacks Python runtime identity")
    if not isinstance(environment.get("platform"), str) or not environment.get("platform"):
        failures.append("benchmark environment lacks platform identity")
    required_packages = ("nemo-curator", "lance-ray", "pyarrow", "pylance", "ray")
    if not isinstance(packages, Mapping):
        failures.append("benchmark environment package identity is missing or invalid")
    else:
        missing_packages = [
            package
            for package in required_packages
            if not isinstance(packages.get(package), str) or not packages[package]
        ]
        if missing_packages:
            failures.append(f"benchmark environment lacks package/code identity for {missing_packages}")

    recorded_dataset = identity.get("dataset")
    if not isinstance(recorded_dataset, Mapping):
        failures.append("run identity dataset is missing or invalid")
        recorded_dataset = {}
    dataset_uri = dataset.get("uri")
    dataset_version = dataset.get("version")
    if not isinstance(dataset_uri, str) or not dataset_uri:
        failures.append("benchmark dataset URI is missing or invalid")
    if not isinstance(dataset_version, int) or isinstance(dataset_version, bool) or dataset_version <= 0:
        failures.append("benchmark dataset version is missing or invalid")
    if not isinstance(recorded_dataset.get("uri"), str) or not recorded_dataset.get("uri"):
        failures.append("run identity dataset.uri is missing or invalid")
    recorded_version = recorded_dataset.get("version")
    if not isinstance(recorded_version, int) or isinstance(recorded_version, bool) or recorded_version <= 0:
        failures.append("run identity dataset.version is missing or invalid")
    expected_dataset = {"uri": dataset_uri, "version": dataset_version}
    for name, expected in expected_dataset.items():
        if recorded_dataset.get(name) != expected:
            failures.append(f"run identity dataset.{name} disagrees with benchmark ({expected!r})")

    recorded_manifest = identity.get("manifest")
    if not isinstance(recorded_manifest, Mapping):
        failures.append("run identity manifest is missing or invalid")
        recorded_manifest = {}
    manifest_digest = recorded_manifest.get("digest_sha256")
    if not _valid_sha256(manifest_digest):
        failures.append("run identity manifest.digest_sha256 is missing or invalid")
    if not _valid_sha256(report_manifest.get("digest_sha256")):
        failures.append("benchmark manifest.digest_sha256 is missing or invalid")
    if manifest_digest != report_manifest.get("digest_sha256"):
        failures.append("run identity manifest digest disagrees with benchmark")

    source_columns = dataset.get("source_columns")
    projection = recorded_geometry.get("payload_projection")
    expected_columns = {
        "image_only": {"image"},
        "image_url": {"image"},
        "full": {"image", "md5", "width", "height"},
    }
    if projection not in expected_columns:
        failures.append(f"run identity payload_projection is {projection!r}")
    elif not isinstance(source_columns, Mapping) or set(source_columns) != expected_columns[projection]:
        failures.append(f"benchmark source_columns={source_columns!r} do not match payload_projection={projection!r}")
    expected_validation = projection in {"image_url", "full"}
    if configuration.get("validate_payload_keys") is not expected_validation:
        failures.append(
            f"validate_payload_keys={configuration.get('validate_payload_keys')!r}; expected {expected_validation}"
        )

    for name in (
        "payload_read_mode",
        "io_threads",
        "max_lookup_bytes",
        "max_pending_fetch_batches",
        "take_scan_batch_readahead",
        "copy_index_to_node_local",
        "index_mirror",
    ):
        if name not in configuration:
            failures.append(f"benchmark configuration is missing compatibility field {name}")
    if configuration.get("payload_read_mode") != "sparse":
        failures.append(
            f"benchmark payload_read_mode is {configuration.get('payload_read_mode')!r}; expected 'sparse'"
        )

    sidecar_uri = identity.get("reference_manifest_uri")
    sidecar_sha256 = identity.get("reference_manifest_sha256")
    if not isinstance(sidecar_uri, str) or not sidecar_uri:
        failures.append("run identity reference_manifest_uri is missing")
    if not _valid_sha256(sidecar_sha256):
        failures.append("run identity reference_manifest_sha256 is missing or invalid")
    if configuration.get("reference_manifest_uri") != sidecar_uri:
        failures.append("benchmark and run identity reference_manifest_uri disagree")
    if configuration.get("reference_manifest_sha256") != sidecar_sha256:
        failures.append("benchmark and run identity reference_manifest_sha256 disagree")

    benchmark_policy = identity.get("benchmark_policy")
    if not isinstance(benchmark_policy, Mapping):
        failures.append("run identity benchmark_policy is missing or invalid")
    else:
        expected_policy = {
            "arm": arm,
            "repeat_count": repeat_count,
            "warmup_count": warmup_count,
            "payload_read_mode": "sparse",
            "io_threads_per_actor": configuration.get("io_threads"),
            "max_pending_fetch_batches": configuration.get("max_pending_fetch_batches"),
            "validate_payload_keys": configuration.get("validate_payload_keys"),
            "copy_reference_to_node_local": configuration.get("copy_index_to_node_local"),
        }
        for name, expected in expected_policy.items():
            if benchmark_policy.get(name) != expected:
                failures.append(f"run identity benchmark_policy.{name} disagrees with benchmark ({expected!r})")

    slurm_job_id = identity.get("slurm_job_id")
    allocation_guard = identity.get("allocation_time_guard")
    if geometry.nodes > 1 and not slurm_job_id:
        failures.append("multi-node run identity lacks slurm_job_id")
    if slurm_job_id and not isinstance(allocation_guard, Mapping):
        failures.append("Slurm run identity lacks the pre-benchmark allocation_time_guard")
    return failures


def build_terminal_eligibility(
    output_dir: Path,
    *,
    arm: str,
    geometry: SaturationGeometry,
    repeat_count: int,
    warmup_count: int,
) -> dict[str, Any]:
    """Join benchmark, telemetry, and run identity into one fail-closed verdict."""

    report_path = output_dir / "benchmark.json"
    identity_path = output_dir / "run_identity.json"
    telemetry_path = output_dir / "telemetry_validation.json"
    report, failures = _load_json_object(report_path, "benchmark artifact")
    identity, identity_load_failures = _load_json_object(identity_path, "run identity")
    telemetry, telemetry_load_failures = _load_json_object(telemetry_path, "telemetry validation")
    failures.extend(identity_load_failures)
    failures.extend(telemetry_load_failures)

    benchmark_validation: dict[str, Any]
    if report is None:
        benchmark_validation = {
            "status": "failed",
            "evidence_class": geometry.evidence_class,
            "failures": ["benchmark artifact unavailable"],
        }
    else:
        try:
            benchmark_validation = validate_benchmark_report(
                report_path,
                arm,
                geometry,
                repeat_count,
                warmup_count,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            benchmark_validation = {
                "status": "failed",
                "evidence_class": geometry.evidence_class,
                "failures": [f"benchmark validation raised {type(exc).__name__}: {exc}"],
            }
        failures.extend(f"benchmark: {failure}" for failure in benchmark_validation["failures"])

    identity_failures: list[str] = []
    if identity is not None and report is not None:
        identity_failures = _terminal_identity_failures(
            identity,
            report,
            arm=arm,
            geometry=geometry,
            repeat_count=repeat_count,
            warmup_count=warmup_count,
        )
    failures.extend(f"identity: {failure}" for failure in identity_failures)
    if telemetry is not None and telemetry.get("status") != "passed":
        failures.append(f"telemetry: status is {telemetry.get('status')!r}; expected 'passed'")

    artifacts = {}
    for name, path in (
        ("benchmark", report_path),
        ("run_identity", identity_path),
        ("telemetry_validation", telemetry_path),
    ):
        if path.is_file():
            artifacts[name] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    status = "eligible" if not failures else "ineligible"
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "artifact_kind": "gpu_lance_saturation_terminal_eligibility",
        "status": status,
        "terminal": True,
        "evidence_class": geometry.evidence_class,
        "generated_at_epoch": time.time(),
        "policy": {
            "evidence_class": geometry.evidence_class,
            "primary_saturation_waves": list(PRIMARY_SATURATION_WAVES),
            "locality_sensitivity_waves": list(LOCALITY_SENSITIVITY_WAVES),
            "minimum_repeat_count": MINIMUM_REPEAT_COUNT,
            "telemetry_pass_is_not_benchmark_eligibility": True,
            "requires_benchmark_validation": True,
            "requires_telemetry_validation": True,
            "requires_run_identity_validation": True,
        },
        "benchmark_validation": benchmark_validation,
        "telemetry_validation_status": telemetry.get("status") if telemetry is not None else "missing",
        "identity_validation": {
            "status": "passed" if not identity_failures and identity is not None else "failed",
            "failures": identity_failures,
        },
        "artifacts": artifacts,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nodes", type=_positive, help="Defaults to SLURM_NNODES")
    parser.add_argument("--waves", type=_positive, choices=SUPPORTED_WAVES, default=8)
    parser.add_argument("--arm", choices=SUPPORTED_ARMS, default="lance_ray_gpu_actor")
    parser.add_argument("--image-lance-uri", required=True, type=_credential_free_uri)
    parser.add_argument("--image-lance-version", type=_positive, default=DEFAULT_IMAGE_VERSION)
    parser.add_argument("--storage-options-json", default="{}", help="Inline JSON object or @path")
    parser.add_argument("--reference-storage-options-json", default="{}", help="Inline JSON object or @path")
    parser.add_argument("--reference-manifest-uri", required=True, type=_credential_free_uri)
    parser.add_argument("--reference-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--reference-glob", action="append", required=True)
    parser.add_argument("--expected-reference-rows", type=_positive, required=True)
    parser.add_argument("--copy-reference-to-node-local", action="store_true")
    parser.add_argument("--reference-node-local-root", default="/local/nemo-curator/gpu-lance-indexes")
    parser.add_argument("--max-lookup-bytes-mib", type=_positive, default=256)
    parser.add_argument(
        "--payload-projection",
        choices=("image_only", "image_url", "full"),
        default="image_only",
    )
    parser.add_argument("--fetch-batch-size", type=_positive, default=1024)
    parser.add_argument("--max-pending-fetch-batches", type=_positive, default=16)
    parser.add_argument("--io-threads-per-actor", type=_positive, default=4)
    parser.add_argument("--ray-cpus-per-node", type=_positive, default=64)
    parser.add_argument("--lance-cpu-threads", type=_positive, default=32)
    parser.add_argument("--lance-io-threads", type=_positive, default=64)
    parser.add_argument("--actor-warmup-rows", type=_positive, default=128)
    parser.add_argument("--warmup-count", type=_nonnegative, default=1)
    parser.add_argument("--repeat-count", type=_positive, default=3)
    parser.add_argument(
        "--minimum-remaining-slurm-seconds",
        type=_positive,
        help="Required for live Slurm runs; include setup plus every requested repeat",
    )
    parser.add_argument(
        "--allocation-end-epoch",
        type=_positive_float,
        help="Slurm allocation end as Unix epoch; defaults to numeric SLURM_JOB_END_TIME",
    )
    parser.add_argument("--telemetry-interval-seconds", type=_positive_float, default=5.0)
    parser.add_argument(
        "--storage-axis",
        choices=("remote_s3", "lustre", "node_local_nvme"),
        default="remote_s3",
    )
    parser.add_argument("--filesystem-path", action="append", type=Path, default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_nodes(args: argparse.Namespace) -> int:
    allocated = os.environ.get("SLURM_NNODES")
    nodes = args.nodes if args.nodes is not None else int(allocated or "0")
    if nodes <= 0:
        msg = "--nodes is required outside a Slurm allocation"
        raise ValueError(msg)
    if allocated is not None and int(allocated) != nodes:
        msg = f"--nodes={nodes} does not match SLURM_NNODES={allocated}"
        raise ValueError(msg)
    return nodes


def _run_head(args: argparse.Namespace, geometry: SaturationGeometry, ray_address: str) -> int:
    validate_credential_free_uri_identity(args.image_lance_uri, "image Lance URI")
    validate_credential_free_uri_identity(args.reference_manifest_uri, "reference manifest URI")
    for pattern in args.reference_glob:
        validate_credential_free_uri_identity(pattern, "reference sidecar glob")
    report_path = args.output_dir / "benchmark.json"
    if report_path.exists():
        msg = f"benchmark output already exists: {report_path}"
        raise FileExistsError(msg)
    command = build_benchmark_command(args, geometry, ray_address=ray_address, report_path=report_path)
    query_manifest_path = (args.manifest_dir / "manifest.parquet").resolve()
    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        "evidence_class": geometry.evidence_class,
        "geometry": {
            "nodes": geometry.nodes,
            "actors_per_node": geometry.actors_per_node,
            "actor_count": geometry.actor_count,
            "tasks_per_actor": geometry.tasks_per_actor,
            "task_rows": geometry.task_rows,
            "target_rows": geometry.target_rows,
            "waves": geometry.waves,
            "coalesce_tasks": geometry.coalesce_tasks,
            "actor_batch_rows": geometry.actor_batch_rows,
            "private_fetch_batch_size": args.fetch_batch_size,
            "max_pending_fetch_batches": args.max_pending_fetch_batches,
            "payload_projection": args.payload_projection,
        },
        "dataset": {
            "uri": _redact_uri_for_identity(args.image_lance_uri),
            "version": args.image_lance_version,
        },
        "manifest": {
            "path": str(query_manifest_path),
            "digest_sha256": _query_manifest_digest(query_manifest_path),
        },
        "benchmark_command": sanitized_command(command),
        "storage_option_keys": sorted(_json_options(args.storage_options_json)),
        "reference_storage_option_keys": sorted(_json_options(args.reference_storage_options_json)),
        "reference_manifest_uri": _redact_uri_for_identity(args.reference_manifest_uri),
        "reference_manifest_sha256": args.reference_manifest_sha256,
        "storage_axis": args.storage_axis,
        "manifest_metadata": str((args.manifest_dir / "manifest.json").resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "allocation_time_guard": args.allocation_time_guard,
        "benchmark_policy": {
            "arm": args.arm,
            "repeat_count": args.repeat_count,
            "warmup_count": args.warmup_count,
            "payload_read_mode": "sparse",
            "io_threads_per_actor": args.io_threads_per_actor,
            "max_lookup_bytes_mib": args.max_lookup_bytes_mib,
            "max_pending_fetch_batches": args.max_pending_fetch_batches,
            "validate_payload_keys": args.payload_projection in {"image_url", "full"},
            "copy_reference_to_node_local": args.copy_reference_to_node_local,
        },
    }
    _atomic_json(args.output_dir / "run_identity.json", identity)
    stdout_path = args.output_dir / "benchmark.stdout.log"
    stderr_path = args.output_dir / "benchmark.stderr.log"
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode:
        return completed.returncode
    validation = validate_benchmark_report(
        report_path,
        args.arm,
        geometry,
        args.repeat_count,
        args.warmup_count,
    )
    _atomic_json(args.output_dir / "validation.json", validation)
    if validation["status"] != "passed":
        print(json.dumps(validation, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(validation, sort_keys=True))
    return 0


def _finalize_telemetry(handle: TelemetryHandle, context: TelemetryRunContext) -> None:
    if handle.process.poll() is None:
        time.sleep(min(context.interval_seconds, 2.0))
    process_validation = _stop_telemetry(handle)
    artifact_validation = validate_telemetry_artifact(
        handle.output,
        TelemetryValidationSpec(
            node_id=context.node_id,
            hostname=context.hostname,
            gpu_count=ACTORS_PER_NODE,
            interval_seconds=context.interval_seconds,
            required_steady_repeat_count=context.repeat_count,
            storage_axis=context.storage_axis,
        ),
    )
    local_validation = {
        "status": (
            "passed"
            if process_validation["status"] == "passed" and artifact_validation["status"] == "passed"
            else "failed"
        ),
        "collector_process": process_validation,
        "artifact": artifact_validation,
    }
    _atomic_json(
        context.output_dir / "telemetry" / f"node_{context.node_id:04d}.validation.json",
        local_validation,
    )
    failures = [*process_validation["failures"], *artifact_validation["failures"]]
    if context.node_id == 0:
        cluster_validation = validate_telemetry_cluster(
            context.output_dir / "telemetry",
            TelemetryClusterSpec(
                nodes=context.nodes,
                report_path=context.report_path,
                arm=context.arm,
                gpu_count=ACTORS_PER_NODE,
                interval_seconds=context.interval_seconds,
                wait_seconds=max(30.0, context.interval_seconds * 4),
                storage_axis=context.storage_axis,
                repeat_count=context.repeat_count,
            ),
        )
        _atomic_json(context.output_dir / "telemetry_validation.json", cluster_validation)
        failures.extend(cluster_validation["failures"])
        eligibility = build_terminal_eligibility(
            context.output_dir,
            arm=context.arm,
            geometry=context.geometry,
            repeat_count=context.repeat_count,
            warmup_count=context.warmup_count,
        )
        _atomic_json(context.output_dir / "eligibility.json", eligibility)
        failures.extend(eligibility["failures"])
    if failures:
        print(json.dumps({"status": "failed", "telemetry_failures": failures}), file=sys.stderr)
        msg = "saturation telemetry validation failed"
        raise RuntimeError(msg)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    args = build_parser().parse_args(argv)
    if args.repeat_count < MINIMUM_REPEAT_COUNT:
        msg = f"saturation eligibility requires at least {MINIMUM_REPEAT_COUNT} repeats"
        raise ValueError(msg)
    args.manifest_dir = args.manifest_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    nodes = _resolve_nodes(args)
    geometry = SaturationGeometry(nodes=nodes, waves=args.waves)
    validate_credential_free_uri_identity(args.image_lance_uri, "image Lance URI")
    validate_credential_free_uri_identity(args.reference_manifest_uri, "reference manifest URI")
    for pattern in args.reference_glob:
        validate_credential_free_uri_identity(pattern, "reference sidecar glob")
    load_manifest_metadata(args.manifest_dir, geometry)
    _reject_secret_storage_options(_json_options(args.storage_options_json))
    _reject_secret_storage_options(_json_options(args.reference_storage_options_json))
    if not args.filesystem_path:
        msg = "at least one --filesystem-path is required for storage telemetry"
        raise ValueError(msg)
    report_path = args.output_dir / "benchmark.json"
    command = build_benchmark_command(args, geometry, ray_address="<slurm-ray-address>", report_path=report_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "evidence_class": geometry.evidence_class,
                    "geometry": {
                        "nodes": nodes,
                        "actors": geometry.actor_count,
                        "tasks": geometry.target_tasks,
                        "rows": geometry.target_rows,
                        "waves": geometry.waves,
                        "actor_calls": geometry.expected_actor_calls,
                    },
                    "command": sanitized_command(command),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    node_id = int(os.environ.get("SLURM_NODEID", "0"))
    if nodes > 1 and "SLURM_JOB_ID" not in os.environ:
        msg = "multi-node saturation runs require an exclusive Slurm allocation"
        raise RuntimeError(msg)
    args.allocation_time_guard = None
    if "SLURM_JOB_ID" in os.environ:
        raw_end_epoch = args.allocation_end_epoch
        if raw_end_epoch is None and os.environ.get("SLURM_JOB_END_TIME"):
            try:
                raw_end_epoch = float(os.environ["SLURM_JOB_END_TIME"])
            except ValueError as exc:
                msg = "SLURM_JOB_END_TIME must be a Unix epoch when --allocation-end-epoch is absent"
                raise ValueError(msg) from exc
        args.allocation_time_guard = validate_remaining_slurm_time(
            minimum_remaining_seconds=args.minimum_remaining_slurm_seconds,
            allocation_end_epoch=raw_end_epoch,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LANCE_CPU_THREADS", str(args.lance_cpu_threads))
    os.environ.setdefault("LANCE_IO_THREADS", str(args.lance_io_threads))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("RAY_PORT_BROADCAST_DIR", str(args.output_dir / ".ray_ports"))

    expected_nodes = _allocated_node_identities(nodes)
    expected_hostname = expected_nodes.get(node_id)
    if expected_hostname is None:
        msg = f"SLURM_NODEID={node_id} is outside the allocated node range 0..{nodes - 1}"
        raise RuntimeError(msg)
    observed_hostname = _short_hostname(socket.gethostname())
    if observed_hostname != expected_hostname:
        msg = f"SLURM_NODEID={node_id} runs on {observed_hostname!r}; expected {expected_hostname!r}"
        raise RuntimeError(msg)

    telemetry: TelemetryHandle | None = None
    client: SlurmRayClient | None = None
    telemetry_context = TelemetryRunContext(
        node_id=node_id,
        hostname=expected_hostname,
        nodes=expected_nodes,
        report_path=report_path,
        output_dir=args.output_dir,
        arm=args.arm,
        interval_seconds=args.telemetry_interval_seconds,
        storage_axis=args.storage_axis,
        geometry=geometry,
        warmup_count=args.warmup_count,
        repeat_count=args.repeat_count,
    )
    try:
        telemetry = _start_telemetry(
            args,
            report_path,
            node_id=node_id,
            hostname=expected_hostname,
        )
        from nemo_curator.core.client import SlurmRayClient

        ray_temp = str(
            Path(tempfile.gettempdir()) / f"gpu-lance-saturation-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        )
        client = SlurmRayClient(
            ray_temp_dir=ray_temp,
            include_dashboard=False,
            num_gpus=ACTORS_PER_NODE,
            num_cpus=args.ray_cpus_per_node,
            worker_connect_timeout_s=600,
            cleanup_on_start=True,
        )
        client.start()
        if node_id != 0:
            return 0
        return _run_head(args, geometry, os.environ["RAY_ADDRESS"])
    finally:
        if node_id == 0 and client is not None:
            client.stop()
        if telemetry is not None:
            _finalize_telemetry(telemetry, telemetry_context)


if __name__ == "__main__":
    raise SystemExit(main())
