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

"""Periodically capture host and GPU telemetry for a Lance saturation run."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

SCHEMA_VERSION = 2
_STOP = threading.Event()
_GPU_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "pcie.link.gen.current",
    "pcie.link.width.current",
)
_MIN_DISKSTAT_FIELDS = 14


@dataclass(frozen=True)
class SampleConfig:
    """Inputs that stay fixed for a telemetry process."""

    started: float
    report_path: Path | None
    arm: str
    warmups: int
    repeats: int
    filesystem_paths: Sequence[Path]
    node_id: int
    hostname: str


def _signal_stop(_signum: int, _frame: object) -> None:
    _STOP.set()


def derive_phase(report: Mapping[str, Any] | None, arm: str, warmups: int, repeats: int) -> str:
    """Classify setup, warmup, and steady work from the atomic benchmark report."""

    phase = "cluster_setup"
    if report:
        arm_result = (report.get("arms") or {}).get(arm)
        phase = "benchmark_setup"
        if isinstance(arm_result, Mapping):
            status = str(arm_result.get("status", "pending"))
            terminal = {"setup_failed", "warmup_failed", "warmup_incorrect", "run_failed", "incorrect", "skipped"}
            completed_warmups = len(arm_result.get("warmups") or [])
            completed_repeats = len(arm_result.get("repeats") or [])
            if status in terminal:
                phase = f"terminal_{status}"
            elif arm_result.get("cold_setup") is None:
                phase = "benchmark_setup"
            elif completed_warmups < warmups:
                phase = f"warmup_{completed_warmups}"
            elif completed_repeats < repeats:
                phase = f"steady_repeat_{completed_repeats}"
            else:
                phase = "complete" if status == "completed" else "benchmark_finalize"
    return phase


def _read_report(path: Path | None) -> tuple[Mapping[str, Any] | None, str | None]:
    if path is None or not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return None, "benchmark report is not a JSON object"
    return payload, None


def _read_proc_stat() -> dict[str, int]:
    with Path("/proc/stat").open(encoding="utf-8") as stream:
        values = stream.readline().split()
    if not values or values[0] != "cpu":
        msg = "/proc/stat does not begin with aggregate cpu counters"
        raise RuntimeError(msg)
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    return {name: int(value) for name, value in zip(names, values[1:], strict=False)}


def _cpu_delta(previous: Mapping[str, int] | None, current: Mapping[str, int]) -> dict[str, float | int | list[float]]:
    load = list(os.getloadavg())
    result: dict[str, float | int | list[float]] = {
        "logical_cpus": os.cpu_count() or 0,
        "load_average": load,
    }
    if previous is None:
        return result
    delta = {name: max(0, current.get(name, 0) - previous.get(name, 0)) for name in current}
    total = sum(delta.values())
    if total:
        idle = delta.get("idle", 0)
        iowait = delta.get("iowait", 0)
        result.update(
            {
                "busy_percent": 100.0 * (total - idle - iowait) / total,
                "iowait_percent": 100.0 * iowait / total,
                "idle_percent": 100.0 * idle / total,
            }
        )
    return result


def _network_counters() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with Path("/proc/net/dev").open(encoding="utf-8") as stream:
        for line in list(stream)[2:]:
            interface, raw = line.split(":", 1)
            values = raw.split()
            result[interface.strip()] = {
                "receive_bytes": int(values[0]),
                "receive_packets": int(values[1]),
                "receive_drops": int(values[3]),
                "transmit_bytes": int(values[8]),
                "transmit_packets": int(values[9]),
                "transmit_drops": int(values[11]),
            }
    return result


def _disk_counters() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with Path("/proc/diskstats").open(encoding="utf-8") as stream:
        for line in stream:
            values = line.split()
            if len(values) < _MIN_DISKSTAT_FIELDS:
                continue
            device = values[2]
            if device.startswith(("loop", "ram", "fd")):
                continue
            result[device] = {
                "reads_completed": int(values[3]),
                "sectors_read": int(values[5]),
                "read_milliseconds": int(values[6]),
                "writes_completed": int(values[7]),
                "sectors_written": int(values[9]),
                "write_milliseconds": int(values[10]),
                "io_in_progress": int(values[11]),
                "io_milliseconds": int(values[12]),
            }
    return result


def _filesystem_stats(paths: Sequence[Path]) -> dict[str, dict[str, int] | dict[str, str]]:
    result: dict[str, dict[str, int] | dict[str, str]] = {}
    for path in paths:
        try:
            stats = os.statvfs(path)
        except OSError as exc:
            result[str(path)] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        result[str(path)] = {
            "block_size": stats.f_frsize,
            "total_bytes": stats.f_blocks * stats.f_frsize,
            "available_bytes": stats.f_bavail * stats.f_frsize,
            "free_bytes": stats.f_bfree * stats.f_frsize,
            "files": stats.f_files,
            "free_files": stats.f_ffree,
        }
    return result


def _parse_gpu_value(value: str) -> str | float | int | None:
    stripped = value.strip()
    if stripped in {"[N/A]", "N/A", "Not Supported", ""}:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    return int(number) if number.is_integer() else number


def _gpu_stats() -> tuple[list[dict[str, Any]], str | None]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return [], "nvidia-smi is unavailable"
    command = [
        binary,
        f"--query-gpu={','.join(_GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)  # noqa: S603
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(_GPU_FIELDS):
            return rows, f"nvidia-smi returned {len(values)} values; expected {len(_GPU_FIELDS)}"
        rows.append({name: _parse_gpu_value(value) for name, value in zip(_GPU_FIELDS, values, strict=True)})
    return rows, None


def _sample(
    config: SampleConfig,
    previous_cpu: Mapping[str, int] | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    errors: list[str] = []
    report, report_error = _read_report(config.report_path)
    if report_error:
        errors.append(f"report: {report_error}")
    try:
        cpu = _read_proc_stat()
        cpu_metrics = _cpu_delta(previous_cpu, cpu)
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(f"cpu: {type(exc).__name__}: {exc}")
        cpu = {}
        cpu_metrics = {}
    try:
        network = _network_counters()
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"network: {type(exc).__name__}: {exc}")
        network = {}
    try:
        disks = _disk_counters()
    except (OSError, ValueError) as exc:
        errors.append(f"disks: {type(exc).__name__}: {exc}")
        disks = {}
    gpus, gpu_error = _gpu_stats()
    if gpu_error:
        errors.append(f"gpu: {gpu_error}")
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "sample",
            "timestamp_epoch": time.time(),
            "monotonic_seconds": time.monotonic() - config.started,
            "node_id": config.node_id,
            "hostname": config.hostname,
            "phase": derive_phase(report, config.arm, config.warmups, config.repeats),
            "cpu": cpu_metrics,
            "gpus": gpus,
            "network": network,
            "block_devices": disks,
            "filesystems": _filesystem_stats(config.filesystem_paths),
            "errors": errors,
        },
        cpu,
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--arm", default="lance_ray_gpu_actor")
    parser.add_argument("--warmup-count", type=_nonnegative, default=1)
    parser.add_argument("--repeat-count", type=_nonnegative, default=3)
    parser.add_argument("--interval-seconds", type=_positive_float, default=5.0)
    parser.add_argument("--node-id", type=_nonnegative, required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--filesystem-path", action="append", type=Path, default=[])
    parser.add_argument("--sample-count", type=_nonnegative, help="Stop after this many samples; 0 means no samples")
    return parser


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        msg = f"telemetry output already exists: {output}"
        raise FileExistsError(msg)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    _STOP.clear()
    hostname = socket.gethostname().split(".", maxsplit=1)[0]
    expected_hostname = args.expected_hostname.split(".", maxsplit=1)[0]
    if hostname != expected_hostname:
        msg = f"collector hostname {hostname!r} does not match allocated node {expected_hostname!r}"
        raise RuntimeError(msg)
    started_epoch = time.time()
    started = time.monotonic()
    config = SampleConfig(
        started=started,
        report_path=args.benchmark_report,
        arm=args.arm,
        warmups=args.warmup_count,
        repeats=args.repeat_count,
        filesystem_paths=tuple(args.filesystem_path),
        node_id=args.node_id,
        hostname=hostname,
    )
    previous_cpu: Mapping[str, int] | None = None
    samples = 0
    phases: Counter[str] = Counter()
    last_sample_monotonic: float | None = None
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            while not _STOP.is_set() and (args.sample_count is None or samples < args.sample_count):
                sample, previous_cpu = _sample(config, previous_cpu)
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                samples += 1
                phases[str(sample["phase"])] += 1
                last_sample_monotonic = float(sample["monotonic_seconds"])
                if not _STOP.is_set() and (args.sample_count is None or samples < args.sample_count):
                    _STOP.wait(args.interval_seconds)
            # A SIGTERM from the runner means the benchmark has finished.  Take
            # one final sample so the stream proves it lived through report
            # publication instead of ending at an arbitrary polling boundary.
            if _STOP.is_set():
                sample, previous_cpu = _sample(config, previous_cpu)
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                samples += 1
                phases[str(sample["phase"])] += 1
                last_sample_monotonic = float(sample["monotonic_seconds"])
            finished_epoch = time.time()
            finished_monotonic = time.monotonic() - started
            summary = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "summary",
                "status": "complete" if samples else "incomplete",
                "node_id": args.node_id,
                "hostname": hostname,
                "sample_count": samples,
                "phase_counts": dict(sorted(phases.items())),
                "started_timestamp_epoch": started_epoch,
                "finished_timestamp_epoch": finished_epoch,
                "duration_seconds": finished_monotonic,
                "last_sample_monotonic_seconds": last_sample_monotonic,
                "interval_seconds": args.interval_seconds,
            }
            stream.write(json.dumps(summary, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return 0 if samples else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGINT, _signal_stop)
    signal.signal(signal.SIGTERM, _signal_stop)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
