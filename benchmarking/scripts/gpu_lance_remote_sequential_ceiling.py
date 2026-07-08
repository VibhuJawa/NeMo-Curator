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

"""Measure the sequential payload-read ceiling of a pinned remote Lance table.

The benchmark scans only the projected image column from deterministic,
stable-ordinal fragments. One fragment scanner runs per reader thread. Arrow
batches are reduced to counters and released immediately; payload bytes are
never accumulated in an output table.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_DATASET_VERSION = 4
DEFAULT_READER_CONCURRENCY = (1, 4, 8, 16)
DEFAULT_PROJECTIONS = ("image_only", "image_url", "full")
MIN_REPEATS = 2
MEASUREMENT_LABEL = "measured_remote_sequential_ceiling"
_SECRET_OPTION_PARTS = ("access_key", "secret", "token", "password", "credential")


class _ScannerLike(Protocol):
    def to_batches(self) -> Iterable[pa.RecordBatch]: ...


class _FragmentLike(Protocol):
    fragment_id: int

    def scanner(
        self,
        *,
        columns: list[str],
        batch_size: int,
        batch_readahead: int,
    ) -> _ScannerLike: ...

    def count_rows(self) -> int: ...

    def deletion_file(self) -> object | None: ...


class _IoStatsLike(Protocol):
    read_bytes: int
    read_iops: int


class _DatasetLike(Protocol):
    has_stable_row_ids: bool
    version: int
    schema: pa.Schema

    def get_fragments(self) -> list[_FragmentLike]: ...

    def io_stats_incremental(self) -> _IoStatsLike: ...


@dataclass(frozen=True)
class _FragmentTarget:
    ordinal: int
    fragment_id: int
    expected_rows: int
    fragment: _FragmentLike


@dataclass(frozen=True)
class _ScanSettings:
    image_column: str
    projection_columns: list[str]
    batch_rows: int
    batch_readahead: int


@dataclass(frozen=True)
class _MeasurementSettings:
    concurrency: int
    repeat: int
    schedule_index: int
    projection: str
    scan: _ScanSettings
    allow_null_payloads: bool


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be greater than zero"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _at_least_two(value: str) -> int:
    parsed = int(value)
    if parsed < MIN_REPEATS:
        msg = "value must be at least two"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _fragment_ids(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        msg = "fragment IDs must be comma-separated integers"
        raise argparse.ArgumentTypeError(msg) from exc
    if not parsed or any(item < 0 for item in parsed) or len(parsed) != len(set(parsed)):
        msg = "fragment IDs must be a nonempty set of unique nonnegative integers"
        raise argparse.ArgumentTypeError(msg)
    return sorted(parsed)


def _load_storage_options(path: Path) -> dict[str, str]:
    """Read non-secret storage tuning without including values in diagnostics."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = "failed to read storage options JSON"
        raise ValueError(msg) from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
    ):
        msg = "storage options JSON must be an object with string keys and values"
        raise TypeError(msg)
    secret_keys = sorted(key for key in parsed if any(part in key.casefold() for part in _SECRET_OPTION_PARTS))
    if secret_keys:
        msg = (
            f"storage options contain credential-like keys {secret_keys}; "
            "load credentials through the process environment instead"
        )
        raise ValueError(msg)
    return parsed


def _atomic_write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _evenly_spaced_ordinals(total: int, count: int) -> list[int]:
    if not 1 <= count <= total:
        msg = f"fragment count must be in [1, {total}], got {count}"
        raise ValueError(msg)
    if count == 1:
        return [0]
    ordinals = sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})
    if len(ordinals) != count:
        msg = "even fragment selection produced duplicate ordinals"
        raise AssertionError(msg)
    return ordinals


def _logical_binary_bytes(array: pa.Array) -> int:
    if not (pa.types.is_binary(array.type) or pa.types.is_large_binary(array.type)):
        msg = f"image projection must be binary or large_binary, got {array.type}"
        raise TypeError(msg)
    total = pc.sum(pc.binary_length(array)).as_py()
    return int(total or 0)


def _scan_fragment(
    fragment: _FragmentLike,
    settings: _ScanSettings,
    expected_rows: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    scanned_rows = 0
    null_payloads = 0
    logical_payload_bytes = 0
    projected_arrow_bytes = 0
    null_counts = dict.fromkeys(settings.projection_columns, 0)
    batches = 0
    scanner = fragment.scanner(
        columns=settings.projection_columns,
        batch_size=settings.batch_rows,
        batch_readahead=settings.batch_readahead,
    )
    for batch in scanner.to_batches():
        image = batch.column(batch.schema.get_field_index(settings.image_column))
        scanned_rows += batch.num_rows
        for column in settings.projection_columns:
            null_counts[column] += batch.column(batch.schema.get_field_index(column)).null_count
        null_payloads += image.null_count
        logical_payload_bytes += _logical_binary_bytes(image)
        projected_arrow_bytes += batch.nbytes
        batches += 1
        del image
        del batch
    elapsed_seconds = time.perf_counter() - started
    return {
        "fragment_id": int(fragment.fragment_id),
        "expected_rows": expected_rows,
        "scanned_rows": scanned_rows,
        "null_payloads": null_payloads,
        "logical_payload_bytes": logical_payload_bytes,
        "projected_arrow_bytes": projected_arrow_bytes,
        "null_counts": null_counts,
        "arrow_batches": batches,
        "elapsed_seconds": elapsed_seconds,
        "row_count_correct": scanned_rows == expected_rows,
    }


def _select_targets(
    dataset: _DatasetLike,
    requested_ids: list[int] | None,
    fragment_count: int,
) -> list[_FragmentTarget]:
    fragments = dataset.get_fragments()
    if not dataset.has_stable_row_ids:
        msg = "remote sequential ceiling requires stable Lance row IDs"
        raise ValueError(msg)
    for ordinal, fragment in enumerate(fragments):
        if int(fragment.fragment_id) != ordinal:
            msg = f"stable fragment ordering violated at ordinal {ordinal}: fragment ID {fragment.fragment_id}"
            raise ValueError(msg)
        if fragment.deletion_file() is not None:
            msg = f"fragment {fragment.fragment_id} has a deletion file"
            raise ValueError(msg)

    ordinals = requested_ids if requested_ids is not None else _evenly_spaced_ordinals(len(fragments), fragment_count)
    if ordinals[-1] >= len(fragments):
        msg = f"fragment ID {ordinals[-1]} is outside the pinned manifest"
        raise ValueError(msg)
    return [
        _FragmentTarget(
            ordinal=ordinal,
            fragment_id=int(fragments[ordinal].fragment_id),
            expected_rows=int(fragments[ordinal].count_rows()),
            fragment=fragments[ordinal],
        )
        for ordinal in ordinals
    ]


def _run_measurement(
    dataset: _DatasetLike,
    targets: list[_FragmentTarget],
    settings: _MeasurementSettings,
) -> dict[str, Any]:
    dataset.io_stats_incremental()
    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=settings.concurrency,
        thread_name_prefix="lance-sequential-reader",
    ) as executor:
        futures = [
            executor.submit(
                _scan_fragment,
                target.fragment,
                settings.scan,
                target.expected_rows,
            )
            for target in targets
        ]
        fragment_results = [future.result() for future in futures]
    elapsed_seconds = time.perf_counter() - started
    io_stats = dataset.io_stats_incremental()

    expected_rows = sum(item.expected_rows for item in targets)
    scanned_rows = sum(item["scanned_rows"] for item in fragment_results)
    null_payloads = sum(item["null_payloads"] for item in fragment_results)
    logical_payload_bytes = sum(item["logical_payload_bytes"] for item in fragment_results)
    projected_arrow_bytes = sum(item["projected_arrow_bytes"] for item in fragment_results)
    null_counts = {
        column: sum(item["null_counts"][column] for item in fragment_results)
        for column in settings.scan.projection_columns
    }
    read_bytes = int(io_stats.read_bytes)
    read_iops = int(io_stats.read_iops)
    correct = (
        scanned_rows == expected_rows
        and all(item["row_count_correct"] for item in fragment_results)
        and (settings.allow_null_payloads or null_payloads == 0)
        and all(count == 0 for column, count in null_counts.items() if column != settings.scan.image_column)
    )
    return {
        "label": MEASUREMENT_LABEL,
        "status": "measured" if correct else "incorrect",
        "storage_backend": "remote_s3_compatible",
        "repeat": settings.repeat,
        "schedule_index": settings.schedule_index,
        "projection": settings.projection,
        "projection_columns": settings.scan.projection_columns,
        "reader_concurrency": settings.concurrency,
        "reader_waves": math.ceil(len(targets) / settings.concurrency),
        "elapsed_seconds": elapsed_seconds,
        "fetch_seconds": elapsed_seconds,
        "expected_images": expected_rows,
        "images_scanned": scanned_rows,
        "logical_payload_bytes": logical_payload_bytes,
        "projected_arrow_bytes": projected_arrow_bytes,
        "lance_read_bytes": read_bytes,
        "lance_read_iops": read_iops,
        "average_physical_read_bytes": read_bytes / read_iops if read_iops else 0.0,
        "images_per_second": scanned_rows / elapsed_seconds if elapsed_seconds else 0.0,
        "logical_payload_mib_per_second": logical_payload_bytes / (1024**2 * elapsed_seconds)
        if elapsed_seconds
        else 0.0,
        "physical_read_mib_per_second": read_bytes / (1024**2 * elapsed_seconds) if elapsed_seconds else 0.0,
        "physical_reads_per_image": read_iops / scanned_rows if scanned_rows else 0.0,
        "physical_to_logical_byte_ratio": read_bytes / logical_payload_bytes if logical_payload_bytes else 0.0,
        "correctness": {
            "correct": correct,
            "expected_rows": expected_rows,
            "scanned_rows": scanned_rows,
            "null_payloads": null_payloads,
            "projected_column_null_counts": null_counts,
            "fragment_row_counts_correct": all(item["row_count_correct"] for item in fragment_results),
        },
        "fragments": fragment_results,
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _summarize(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_names = (
        "elapsed_seconds",
        "fetch_seconds",
        "images_per_second",
        "logical_payload_mib_per_second",
        "physical_read_mib_per_second",
        "lance_read_bytes",
        "lance_read_iops",
        "average_physical_read_bytes",
        "physical_reads_per_image",
        "physical_to_logical_byte_ratio",
    )
    for projection in sorted({str(item["projection"]) for item in measurements}):
        summary[projection] = {}
        projection_measurements = [item for item in measurements if item["projection"] == projection]
        for concurrency in sorted({int(item["reader_concurrency"]) for item in projection_measurements}):
            selected = [
                item
                for item in projection_measurements
                if item["reader_concurrency"] == concurrency and item["status"] == "measured"
            ]
            summary[projection][str(concurrency)] = {
                "label": MEASUREMENT_LABEL,
                "measured_repeats": len(selected),
                "metrics": {
                    name: _stats([float(item[name]) for item in selected]) if selected else None
                    for name in metric_names
                },
            }
    return summary


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("pylance", "pyarrow", "nemo-curator"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _rotated(values: list[int], repeat: int) -> list[int]:
    offset = repeat % len(values)
    return values[offset:] + values[:offset]


def _projection_axes(args: argparse.Namespace) -> dict[str, list[str]]:
    available = {
        "image_only": [args.image_column],
        "image_url": [args.image_column, args.url_column],
        "full": [args.url_column, args.image_column, args.md5_column, args.width_column, args.height_column],
    }
    selected = list(dict.fromkeys(args.projection or DEFAULT_PROJECTIONS))
    return {name: available[name] for name in selected}


def _raise_incorrect_measurement() -> None:
    msg = "remote sequential ceiling produced an incorrect row or null count"
    raise RuntimeError(msg)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dataset_uri.startswith("s3://"):
        msg = "remote sequential ceiling requires an s3:// Lance URI"
        raise ValueError(msg)
    storage_options = _load_storage_options(args.storage_options_file)

    import lance

    setup_started = time.perf_counter()
    session = lance.Session(metadata_cache_size_bytes=args.metadata_cache_size_mib * 1024**2)
    dataset = lance.dataset(
        args.dataset_uri,
        version=args.dataset_version,
        storage_options=storage_options,
        session=session,
    )
    if int(dataset.version) != args.dataset_version:
        msg = f"requested Lance version {args.dataset_version}, opened {dataset.version}"
        raise ValueError(msg)
    targets = _select_targets(dataset, args.fragment_ids, args.fragment_count)
    concurrency = list(dict.fromkeys(args.reader_concurrency or DEFAULT_READER_CONCURRENCY))
    projections = _projection_axes(args)
    missing_columns = sorted(
        {column for columns in projections.values() for column in columns} - set(dataset.schema.names)
    )
    if missing_columns:
        msg = f"projection columns are absent from the pinned manifest: {missing_columns}"
        raise ValueError(msg)
    if max(concurrency) > len(targets):
        msg = "selected fragment count must be at least the maximum reader concurrency"
        raise ValueError(msg)
    setup_seconds = time.perf_counter() - setup_started

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "label": MEASUREMENT_LABEL,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "dataset": {
            "uri": args.dataset_uri,
            "version": args.dataset_version,
            "image_column": args.image_column,
            "projection_axes": projections,
            "storage_options_supplied": True,
            "stable_row_ids": True,
            "stable_fragment_order_required": True,
            "deletions_allowed": False,
        },
        "selection": {
            "method": "explicit_fragment_ids" if args.fragment_ids is not None else "even_manifest_ordinals",
            "fragment_ids": [item.fragment_id for item in targets],
            "fragment_rows": [item.expected_rows for item in targets],
            "fragments": len(targets),
            "expected_rows": sum(item.expected_rows for item in targets),
        },
        "configuration": {
            "reader_concurrency": concurrency,
            "projection_axes": list(projections),
            "repeat_count": args.repeat_count,
            "batch_rows": args.batch_rows,
            "batch_readahead_per_reader": args.batch_readahead,
            "maximum_in_flight_arrow_batches": max(concurrency) * args.batch_readahead,
            "metadata_cache_size_mib": args.metadata_cache_size_mib,
            "allow_null_payloads": args.allow_null_payloads,
            "cache_policy": "one persistent Lance session; metadata and connections carry across measured sweeps",
        },
        "setup": {"label": "measured_setup", "elapsed_seconds": setup_seconds},
        "schedule": [],
        "measurements": [],
        "summary": {},
    }
    _atomic_write_json(args.output, report)

    try:
        for repeat in range(args.repeat_count):
            order = _rotated(concurrency, repeat)
            report["schedule"].append(
                {"repeat": repeat, "projection_axes": list(projections), "reader_concurrency": order}
            )
            for projection, projection_columns in projections.items():
                for schedule_index, readers in enumerate(order):
                    measurement_settings = _MeasurementSettings(
                        concurrency=readers,
                        repeat=repeat,
                        schedule_index=schedule_index,
                        projection=projection,
                        scan=_ScanSettings(
                            image_column=args.image_column,
                            projection_columns=projection_columns,
                            batch_rows=args.batch_rows,
                            batch_readahead=args.batch_readahead,
                        ),
                        allow_null_payloads=args.allow_null_payloads,
                    )
                    measurement = _run_measurement(
                        dataset,
                        targets,
                        measurement_settings,
                    )
                    report["measurements"].append(measurement)
                    report["summary"] = _summarize(report["measurements"])
                    _atomic_write_json(args.output, report)
                    if measurement["status"] != "measured":
                        _raise_incorrect_measurement()
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = {"type": type(exc).__name__}
        _atomic_write_json(args.output, report)
        raise

    report["status"] = "completed"
    report["summary"] = _summarize(report["measurements"])
    _atomic_write_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--dataset-version", type=_positive_int, default=DEFAULT_DATASET_VERSION)
    parser.add_argument(
        "--storage-options-file",
        type=Path,
        required=True,
        help="JSON object containing non-secret Lance S3 tuning; credentials come from the environment",
    )
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--md5-column", default="md5")
    parser.add_argument("--width-column", default="width")
    parser.add_argument("--height-column", default="height")
    parser.add_argument(
        "--projection",
        action="append",
        choices=DEFAULT_PROJECTIONS,
        default=[],
        help="Repeat to select projection axes; the default runs all three",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fragment-count", type=_positive_int, default=64)
    selection.add_argument("--fragment-ids", type=_fragment_ids)
    parser.add_argument(
        "--reader-concurrency",
        action="append",
        type=_positive_int,
        default=[],
        help="Repeat to override the default 1/4/8/16 sweep",
    )
    parser.add_argument("--repeat-count", type=_at_least_two, default=2)
    parser.add_argument("--batch-rows", type=_positive_int, default=256)
    parser.add_argument("--batch-readahead", type=_positive_int, default=2)
    parser.add_argument("--metadata-cache-size-mib", type=_positive_int, default=512)
    parser.add_argument("--allow-null-payloads", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_benchmark(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "label": report["label"],
                "output": str(args.output),
                "measurements": len(report["measurements"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
