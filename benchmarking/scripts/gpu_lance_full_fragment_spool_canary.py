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

"""Run a real stable-ID payload fetch through the bounded Arrow spool path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import resource
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.stages.interleaved.lance import _validate_stable_global_ordinal_manifest
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    STABLE_ROW_ID,
    lance_coordinate_plan_schema,
)
from nemo_curator.stages.interleaved.lance_payload_materialize import materialize_lance_payload_to_spool
from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool
from nemo_curator.utils.uri import validate_credential_free_uri_identity

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_MIB = 1024**2
_PROC_NET_COUNTERS = 16


@dataclass(frozen=True)
class _TelemetrySnapshot:
    usage: resource.struct_rusage
    process_io: dict[str, int] | None
    node_network: dict[str, int] | None


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--stable-id-column", default=STABLE_ROW_ID)
    parser.add_argument("--image-lance-uri", required=True)
    parser.add_argument("--image-lance-version", required=True, type=_positive)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--storage-options-file", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=_positive)
    parser.add_argument("--fetch-batch-size", type=_positive, default=4096)
    parser.add_argument("--max-pending", type=_positive, default=16)
    parser.add_argument("--target-bytes", type=_positive, default=1024**3)
    parser.add_argument("--bucket-rows", type=_positive, default=131_072)
    parser.add_argument("--metadata-cache-mib", type=_positive, default=512)
    parser.add_argument("--spool-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--keep-spool", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * _MIB):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_storage_options(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_bytes()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in parsed.items()
    ):
        msg = "storage options must be a JSON object with string keys and values"
        raise TypeError(msg)
    sensitive_tokens = ("access_key", "secret", "token", "credential", "password")
    if any(any(token in key.lower() for token in sensitive_tokens) for key in parsed):
        msg = "storage options file must not contain credentials"
        raise ValueError(msg)
    return parsed, hashlib.sha256(raw).hexdigest()


def _load_stable_ids(path: Path, column: str, expected_rows: int) -> pa.Array:
    table = pq.read_table(path, columns=[column])
    stable_ids = table[column].combine_chunks()
    if stable_ids.type != pa.uint64() or stable_ids.null_count:
        msg = "stable IDs must be non-null uint64"
        raise TypeError(msg)
    if len(stable_ids) != expected_rows:
        msg = f"coordinate row count is {len(stable_ids)}; expected {expected_rows}"
        raise ValueError(msg)
    if len(stable_ids) > 1:
        strictly_increasing = pc.all(pc.greater(stable_ids.slice(1), stable_ids.slice(0, len(stable_ids) - 1))).as_py()
        if strictly_increasing is not True:
            msg = "stable IDs must be strictly increasing"
            raise ValueError(msg)
    return stable_ids


def _coordinate_plan(stable_ids: pa.Array) -> pa.Table:
    positions = pa.array(range(len(stable_ids)), type=pa.uint64())
    return pa.Table.from_arrays(
        [positions, positions, stable_ids],
        schema=lance_coordinate_plan_schema(),
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _io_value(stats: object, name: str) -> int:
    value = getattr(stats, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"Lance I/O statistic {name!r} is not a nonnegative integer"
        raise TypeError(msg)
    return value


def _read_process_io_bytes() -> dict[str, int] | None:
    try:
        values = {}
        with Path("/proc/self/io").open(encoding="utf-8") as stream:
            for line in stream:
                name, separator, raw_value = line.partition(":")
                if separator and name in {"read_bytes", "write_bytes"}:
                    values[name] = int(raw_value.strip())
    except (OSError, ValueError):
        return None
    if values.keys() != {"read_bytes", "write_bytes"} or any(value < 0 for value in values.values()):
        return None
    return values


def _read_node_network_bytes() -> dict[str, int] | None:
    try:
        receive_bytes = 0
        transmit_bytes = 0
        interface_count = 0
        with Path("/proc/net/dev").open(encoding="utf-8") as stream:
            for line in list(stream)[2:]:
                interface, separator, raw_counters = line.partition(":")
                if not separator or interface.strip() == "lo":
                    continue
                counters = raw_counters.split()
                if len(counters) < _PROC_NET_COUNTERS:
                    return None
                receive_bytes += int(counters[0])
                transmit_bytes += int(counters[8])
                interface_count += 1
    except (OSError, ValueError):
        return None
    return {
        "receive_bytes": receive_bytes,
        "transmit_bytes": transmit_bytes,
        "interface_count": interface_count,
    }


def _counter_delta(before: dict[str, int] | None, after: dict[str, int] | None, name: str) -> int | None:
    if before is None or after is None:
        return None
    value = after[name] - before[name]
    return value if value >= 0 else None


def _capture_before_materialize() -> _TelemetrySnapshot:
    process_io = _read_process_io_bytes()
    node_network = _read_node_network_bytes()
    return _TelemetrySnapshot(resource.getrusage(resource.RUSAGE_SELF), process_io, node_network)


def _capture_after_materialize() -> _TelemetrySnapshot:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return _TelemetrySnapshot(usage, _read_process_io_bytes(), _read_node_network_bytes())


def _materialize_telemetry(
    before: _TelemetrySnapshot,
    after: _TelemetrySnapshot,
    materialize_seconds: float,
) -> dict[str, object]:
    user_cpu_seconds = max(0.0, after.usage.ru_utime - before.usage.ru_utime)
    system_cpu_seconds = max(0.0, after.usage.ru_stime - before.usage.ru_stime)
    total_cpu_seconds = user_cpu_seconds + system_cpu_seconds
    rss_scale = 1 if sys.platform == "darwin" else 1024
    process_read_bytes = _counter_delta(before.process_io, after.process_io, "read_bytes")
    process_write_bytes = _counter_delta(before.process_io, after.process_io, "write_bytes")
    network_receive_bytes = _counter_delta(before.node_network, after.node_network, "receive_bytes")
    network_transmit_bytes = _counter_delta(before.node_network, after.node_network, "transmit_bytes")
    return {
        "sampling_boundary": "immediately_before_and_after_timed_materialize",
        "process": {
            "scope": "current_process_and_threads",
            "resource_source": "resource.getrusage(RUSAGE_SELF)",
            "user_cpu_seconds": user_cpu_seconds,
            "system_cpu_seconds": system_cpu_seconds,
            "total_cpu_seconds": total_cpu_seconds,
            "average_cpu_cores": total_cpu_seconds / materialize_seconds if materialize_seconds else 0.0,
            "peak_rss_bytes": int(after.usage.ru_maxrss * rss_scale),
            "peak_rss_scope": "process_lifetime_through_materialize_end",
            "peak_rss_increase_bytes": int(max(0, after.usage.ru_maxrss - before.usage.ru_maxrss) * rss_scale),
            "proc_self_io": {
                "source": "/proc/self/io",
                "available": process_read_bytes is not None and process_write_bytes is not None,
                "read_bytes": process_read_bytes,
                "write_bytes": process_write_bytes,
            },
        },
        "node_network": {
            "scope": "aggregate_non_loopback_interfaces",
            "source": "/proc/net/dev",
            "attribution": "node_level_inference_under_exclusive_node_allocation",
            "available": network_receive_bytes is not None and network_transmit_bytes is not None,
            "receive_bytes": network_receive_bytes,
            "transmit_bytes": network_transmit_bytes,
            "interfaces_before": (before.node_network["interface_count"] if before.node_network is not None else None),
            "interfaces_after": after.node_network["interface_count"] if after.node_network is not None else None,
        },
    }


def _validate_spool(
    spool: PayloadSpool,
    stable_ids: pa.Array,
    image_column: str,
    expected_rows: int,
) -> tuple[int, int, float]:
    started = time.perf_counter()
    validated_rows = 0
    null_payloads = 0
    seen_positions = np.zeros(expected_rows, dtype=np.bool_)
    for table in spool.iter_tables():
        table_stable_ids = table[STABLE_ROW_ID].combine_chunks()
        document_positions = table[DOCUMENT_POSITION].combine_chunks()
        if (
            table_stable_ids.type != pa.uint64()
            or document_positions.type != pa.uint64()
            or table_stable_ids.null_count
            or document_positions.null_count
        ):
            msg = "validated spool coordinates must be non-null uint64"
            raise TypeError(msg)
        positions_numpy = document_positions.to_numpy(zero_copy_only=False)
        if positions_numpy.size and int(positions_numpy.max()) >= expected_rows:
            msg = "validated spool document position is outside the coordinate input"
            raise RuntimeError(msg)
        unique_positions = np.unique(positions_numpy)
        if unique_positions.size != positions_numpy.size or seen_positions[unique_positions].any():
            msg = "validated spool contains duplicate document positions"
            raise RuntimeError(msg)
        expected_ids = pc.take(stable_ids, document_positions)
        if not table_stable_ids.equals(expected_ids):
            msg = "validated spool stable ID differs from the coordinate input at its document position"
            raise RuntimeError(msg)
        seen_positions[unique_positions] = True
        null_payloads += table[image_column].null_count
        validated_rows += table.num_rows
    if validated_rows != expected_rows or not seen_positions.all() or null_payloads:
        msg = "validated spool row or null-payload count is incorrect"
        raise RuntimeError(msg)
    return validated_rows, null_payloads, time.perf_counter() - started


def _validate_invocation(args: argparse.Namespace) -> None:
    validate_credential_free_uri_identity(args.image_lance_uri, "image Lance URI")
    if _COMMIT_PATTERN.fullmatch(args.code_commit) is None:
        msg = "code commit must be a full lowercase Git SHA"
        raise ValueError(msg)
    for path in (args.output, args.spool_root):
        if path.exists() or path.is_symlink():
            msg = f"refusing to overwrite {path}"
            raise FileExistsError(msg)
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        msg = "remote credentials must be provided through the environment"
        raise RuntimeError(msg)


def main() -> int:
    args = _parse_args()
    _validate_invocation(args)
    storage_options, storage_options_sha256 = _load_storage_options(args.storage_options_file)
    coordinate_sha256 = _file_sha256(args.coordinates)
    stable_ids = _load_stable_ids(args.coordinates, args.stable_id_column, args.expected_rows)
    plan = _coordinate_plan(stable_ids)

    session = lance.Session(metadata_cache_size_bytes=args.metadata_cache_mib * _MIB)
    dataset = lance.dataset(
        args.image_lance_uri,
        version=args.image_lance_version,
        storage_options=storage_options,
        session=session,
    )
    if not dataset.has_stable_row_ids:
        msg = "image dataset must have stable row IDs"
        raise ValueError(msg)
    manifest = _validate_stable_global_ordinal_manifest(dataset)
    if int(stable_ids[-1].as_py()) >= manifest.total_rows:
        msg = "stable IDs exceed the pinned image row count"
        raise ValueError(msg)
    image_field = dataset.schema.field(args.image_column)
    if not (pa.types.is_binary(image_field.type) or pa.types.is_large_binary(image_field.type)):
        msg = "image column must be binary or large_binary"
        raise TypeError(msg)

    spool_schema = pa.schema([*plan.schema, image_field])
    spool = PayloadSpool(
        args.spool_root,
        spool_schema,
        args.target_bytes,
        args.bucket_rows,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
    )
    dataset.io_stats_incremental()
    try:
        with ThreadPoolExecutor(max_workers=args.max_pending, thread_name_prefix="full-fragment-spool") as executor:
            before_telemetry = _capture_before_materialize()
            started = time.perf_counter()
            metrics = materialize_lance_payload_to_spool(
                dataset,
                plan,
                [args.image_column],
                spool,
                executor,
                fetch_batch_size=args.fetch_batch_size,
                max_pending=args.max_pending,
            )
            materialize_seconds = time.perf_counter() - started
            telemetry = _materialize_telemetry(
                before_telemetry,
                _capture_after_materialize(),
                materialize_seconds,
            )
        io_stats = dataset.io_stats_incremental()
        manifest_result = spool.finish()
        validated_rows, null_payloads, validation_seconds = _validate_spool(
            spool,
            stable_ids,
            args.image_column,
            args.expected_rows,
        )

        read_iops = _io_value(io_stats, "read_iops")
        read_bytes = _io_value(io_stats, "read_bytes")
        logical_payload_bytes = int(metrics["actual_payload_bytes"])
        result: dict[str, object] = {
            "schema_version": 2,
            "status": "completed",
            "code_commit": args.code_commit,
            "dataset": {
                "uri": args.image_lance_uri,
                "version": args.image_lance_version,
                "stable_row_ids": True,
                "rows": manifest.total_rows,
                "fragments": len(manifest.fragment_rows),
                "image_column": args.image_column,
            },
            "coordinates": {
                "path": str(args.coordinates),
                "file_sha256": coordinate_sha256,
                "stable_id_column": args.stable_id_column,
                "rows": len(stable_ids),
                "minimum": int(stable_ids[0].as_py()),
                "maximum": int(stable_ids[-1].as_py()),
                "ordering": "strictly_increasing_stable_global_ordinal",
                "document_positions": "synthetic_contiguous_materializer_canary_positions",
            },
            "configuration": {
                "fetch_batch_size": args.fetch_batch_size,
                "max_pending": args.max_pending,
                "target_bytes": args.target_bytes,
                "bucket_rows": args.bucket_rows,
                "metadata_cache_mib": args.metadata_cache_mib,
                "storage_option_keys": sorted(storage_options),
                "storage_options_sha256": storage_options_sha256,
            },
            "timing": {
                "materialize_seconds": materialize_seconds,
                "validation_seconds": validation_seconds,
                "images_per_second": len(stable_ids) / materialize_seconds,
            },
            "io": {
                "physical_reads": read_iops,
                "physical_bytes": read_bytes,
                "reads_per_image": read_iops / len(stable_ids),
                "average_read_bytes": read_bytes / read_iops if read_iops else 0.0,
                "read_amplification": read_bytes / logical_payload_bytes if logical_payload_bytes else 0.0,
                "physical_mib_per_second": read_bytes / _MIB / materialize_seconds,
                "logical_mib_per_second": logical_payload_bytes / _MIB / materialize_seconds,
            },
            "telemetry": telemetry,
            "materializer": metrics,
            "spool": {
                "manifest_sha256": manifest_result.sha256,
                "files": len(manifest_result.files),
                "rows": manifest_result.total_rows,
                "arrow_bytes": manifest_result.total_arrow_nbytes,
                "peak_active_bytes": manifest_result.peak_active_bytes,
                "peak_bounded_active_bytes": manifest_result.peak_bounded_active_bytes,
                "oversized_rows": len(manifest_result.oversized_rows),
                "validated_rows": validated_rows,
                "null_payloads": null_payloads,
                "retained": args.keep_spool,
            },
            "environment": {
                "python": sys.version.split()[0],
                "pylance": _package_version("pylance"),
                "pyarrow": _package_version("pyarrow"),
                "nemo_curator": _package_version("nemo-curator"),
                "lance_cpu_threads": os.environ.get("LANCE_CPU_THREADS"),
                "lance_io_threads": os.environ.get("LANCE_IO_THREADS"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "correctness": {
                "input_rows_match": len(stable_ids) == args.expected_rows,
                "strict_stable_id_order": True,
                "spool_manifest_and_file_hashes_validated": True,
                "stable_id_matches_input_at_document_position": True,
                "document_positions_globally_unique": True,
                "exact_contiguous_document_position_coverage": True,
                "zero_null_payloads": null_payloads == 0,
                "payload_identity_digest": False,
            },
        }
    except BaseException:
        spool.cleanup()
        raise

    if not args.keep_spool:
        spool.cleanup()
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
