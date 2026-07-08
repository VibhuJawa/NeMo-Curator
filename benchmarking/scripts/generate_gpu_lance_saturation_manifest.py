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

"""Build deterministic real-document manifests for GPU Lance saturation runs.

The published directory contains a combined ``manifest.parquet``, one balanced
Parquet shard per GPU actor, and ``manifest.json`` with the pinned source and
file digests. Publication is a single same-filesystem directory rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from typing import Any

DEFAULT_DOCUMENT_VERSION = 3
TASK_ROWS = 256
ACTORS_PER_NODE = 8
TASKS_PER_ACTOR = 64
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_OPTION_PARTS = ("access_key", "secret", "token", "password", "credential")


@dataclass(frozen=True)
class ManifestPreset:
    """Fixed weak-scaling geometry for one published manifest."""

    name: str
    nodes: int

    @property
    def actor_count(self) -> int:
        """Return the total persistent GPU actor count."""

        return self.nodes * ACTORS_PER_NODE

    @property
    def target_tasks(self) -> int:
        """Return the number of 256-row left tasks."""

        return self.actor_count * TASKS_PER_ACTOR

    @property
    def target_rows(self) -> int:
        """Return the total image-reference rows."""

        return self.target_tasks * TASK_ROWS


PRESETS = {
    "one-node": ManifestPreset("one-node", nodes=1),
    "two-node": ManifestPreset("two-node", nodes=2),
    "four-node": ManifestPreset("four-node", nodes=4),
    "eight-node": ManifestPreset("eight-node", nodes=8),
}


@dataclass(frozen=True)
class ManifestConfig:
    """Geometry and pinned source used by the streaming writer."""

    target_tasks: int
    actor_count: int
    document_uri: str
    document_version: int
    seed: int
    task_rows: int = TASK_ROWS

    def __post_init__(self) -> None:
        if self.target_tasks <= 0 or self.actor_count <= 0 or self.task_rows <= 0:
            msg = "target_tasks, actor_count, and task_rows must be positive"
            raise ValueError(msg)
        if self.target_tasks % self.actor_count:
            msg = "target_tasks must be divisible by actor_count"
            raise ValueError(msg)
        if not self.document_uri or self.document_version <= 0:
            msg = "document URI must be non-empty and version must be positive"
            raise ValueError(msg)

    @property
    def target_rows(self) -> int:
        """Return the exact output row count."""

        return self.target_tasks * self.task_rows

    @property
    def tasks_per_actor(self) -> int:
        """Return the exact balanced task count in every actor shard."""

        return self.target_tasks // self.actor_count


@dataclass(frozen=True)
class FragmentTask:
    """One left task containing image URLs from exactly one source fragment."""

    fragment_id: int
    source_refs: tuple[str, ...]


class _Fragment(Protocol):
    fragment_id: int


class _RecordBatchReader(Protocol):
    def to_batches(self) -> Iterable[pa.RecordBatch]: ...


class _LanceDataset(Protocol):
    version: int
    schema: pa.Schema

    def get_fragments(self) -> Iterable[_Fragment]: ...

    def scanner(self, **kwargs: object) -> _RecordBatchReader: ...


@dataclass(frozen=True)
class DocumentScanConfig:
    """Deterministic bounded scan parameters for the pinned document table."""

    source_ref_column: str
    modality_column: str
    seed: int
    target_tasks: int
    task_rows: int
    scan_batch_size: int


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


def _reject_secret_storage_options(options: Mapping[str, str]) -> None:
    secret_keys = sorted(key for key in options if any(part in key.casefold() for part in _SECRET_OPTION_PARTS))
    if secret_keys:
        msg = (
            f"storage options contain credential-like keys {secret_keys}; "
            "load credentials through the process environment instead"
        )
        raise ValueError(msg)


def _manifest_schema(config: ManifestConfig) -> pa.Schema:
    metadata = {
        b"gpu_lance_manifest_schema_version": b"1",
        b"document_uri": config.document_uri.encode(),
        b"document_version": str(config.document_version).encode(),
        b"seed": str(config.seed).encode(),
        b"task_rows": str(config.task_rows).encode(),
        b"actor_count": str(config.actor_count).encode(),
    }
    return pa.schema(
        [
            pa.field("source_ref", pa.string(), nullable=False),
            pa.field("left_task_id", pa.int32(), nullable=False),
            pa.field("source_fragment_id", pa.int32(), nullable=False),
            pa.field("source_position", pa.int32(), nullable=False),
            # Preserve the established benchmark spelling while making the
            # shorter source_position contract explicit.
            pa.field("source_position_in_task", pa.int32(), nullable=False),
        ],
        metadata=metadata,
    )


def _task_table(task: FragmentTask, task_id: int, config: ManifestConfig, schema: pa.Schema) -> pa.Table:
    if len(task.source_refs) != config.task_rows:
        msg = f"left task {task_id} has {len(task.source_refs)} rows; expected {config.task_rows}"
        raise ValueError(msg)
    if task.fragment_id < 0:
        msg = f"source fragment ID must be nonnegative, got {task.fragment_id}"
        raise ValueError(msg)
    if any(not isinstance(value, str) or not value for value in task.source_refs):
        msg = f"left task {task_id} contains a null, non-string, or empty source_ref"
        raise ValueError(msg)
    positions = pa.array(range(config.task_rows), type=pa.int32())
    return pa.Table.from_arrays(
        [
            pa.array(task.source_refs, type=pa.string()),
            pa.repeat(pa.scalar(task_id, type=pa.int32()), config.task_rows),
            pa.repeat(pa.scalar(task.fragment_id, type=pa.int32()), config.task_rows),
            positions,
            positions,
        ],
        schema=schema,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class _ManifestWriters:
    """Write the combined stream and balanced actor streams in one pass."""

    def __init__(self, staging: Path, config: ManifestConfig) -> None:
        self.staging = staging
        self.config = config
        self.schema = _manifest_schema(config)
        self.actor_dir = staging / "actors"
        self.actor_dir.mkdir()
        self.combined_path = staging / "manifest.parquet"
        self.combined = pq.ParquetWriter(self.combined_path, self.schema, compression="zstd")
        self.actors = {
            actor_id: pq.ParquetWriter(self.actor_path(actor_id), self.schema, compression="zstd")
            for actor_id in range(config.actor_count)
        }
        self.actor_tasks = [0] * config.actor_count
        self.actor_rows = [0] * config.actor_count
        self.closed = False

    def actor_path(self, actor_id: int) -> Path:
        return self.actor_dir / f"actor_{actor_id:03d}.parquet"

    def write(self, task: FragmentTask, task_id: int) -> None:
        table = _task_table(task, task_id, self.config, self.schema)
        actor_id = task_id % self.config.actor_count
        self.combined.write_table(table, row_group_size=self.config.task_rows)
        self.actors[actor_id].write_table(table, row_group_size=self.config.task_rows)
        self.actor_tasks[actor_id] += 1
        self.actor_rows[actor_id] += table.num_rows

    def close(self) -> None:
        if self.closed:
            return
        self.combined.close()
        for writer in self.actors.values():
            writer.close()
        self.closed = True


def write_manifest(
    tasks: Iterable[FragmentTask],
    output_dir: Path,
    config: ManifestConfig,
    *,
    storage_option_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Atomically publish exactly ``config.target_tasks`` from ``tasks``."""

    output_dir = output_dir.resolve()
    if output_dir.exists():
        msg = f"output directory already exists: {output_dir}"
        raise FileExistsError(msg)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    writers: _ManifestWriters | None = None
    fragment_counts: dict[int, int] = {}
    task_count = 0
    try:
        writers = _ManifestWriters(staging, config)
        iterator = iter(tasks)
        for task_id in range(config.target_tasks):
            try:
                task = next(iterator)
            except StopIteration as exc:
                msg = f"source produced {task_id} complete tasks; expected {config.target_tasks}"
                raise RuntimeError(msg) from exc
            writers.write(task, task_id)
            fragment_counts[task.fragment_id] = fragment_counts.get(task.fragment_id, 0) + 1
            task_count += 1

        writers.close()
        expected_tasks = [config.tasks_per_actor] * config.actor_count
        expected_rows = [config.tasks_per_actor * config.task_rows] * config.actor_count
        _validate_actor_balance(writers, expected_tasks, expected_rows)

        data_files = [writers.combined_path, *(writers.actor_path(i) for i in range(config.actor_count))]
        for path in data_files:
            _fsync_file(path)
        files = {
            str(path.relative_to(staging)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "rows": config.target_rows
                if path == writers.combined_path
                else config.tasks_per_actor * config.task_rows,
            }
            for path in data_files
        }
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "document": {
                "uri": config.document_uri,
                "version": config.document_version,
                "storage_option_keys": sorted(storage_option_keys),
            },
            "seed": config.seed,
            "task_rows": config.task_rows,
            "target_tasks": config.target_tasks,
            "target_rows": config.target_rows,
            "actor_count": config.actor_count,
            "tasks_per_actor": config.tasks_per_actor,
            "rows_per_actor": config.tasks_per_actor * config.task_rows,
            "actor_assignment": "left_task_id modulo actor_count",
            "source_fragments": {
                "count": len(fragment_counts),
                "tasks_by_fragment": {str(key): fragment_counts[key] for key in sorted(fragment_counts)},
            },
            "schema": str(writers.schema),
            "files": files,
        }
        metadata_path = staging / "manifest.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _fsync_file(metadata_path)
        _fsync_directory(writers.actor_dir)
        _fsync_directory(staging)
        os.replace(staging, output_dir)
        _fsync_directory(output_dir.parent)
    except Exception:
        if writers is not None:
            writers.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        return metadata
    finally:
        if task_count != config.target_tasks and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _validate_actor_balance(
    writers: _ManifestWriters,
    expected_tasks: list[int],
    expected_rows: list[int],
) -> None:
    if writers.actor_tasks != expected_tasks or writers.actor_rows != expected_rows:
        msg = "round-robin actor shards are not exactly balanced"
        raise RuntimeError(msg)


def _valid_urls(array: pa.Array | pa.ChunkedArray) -> list[str]:
    combined = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    if not (pa.types.is_string(combined.type) or pa.types.is_large_string(combined.type)):
        msg = f"source_ref must be string or large_string, got {combined.type}"
        raise TypeError(msg)
    valid = pc.and_(pc.is_valid(combined), pc.not_equal(combined, pa.scalar("", type=combined.type)))
    return combined.filter(valid).to_pylist()


def iter_document_tasks(
    dataset: _LanceDataset,
    config: DocumentScanConfig,
) -> Iterator[FragmentTask]:
    """Yield balanced-fragment tasks in deterministic seeded order."""

    fragments = list(dataset.get_fragments())
    if not fragments:
        msg = "pinned document dataset has no fragments"
        raise ValueError(msg)
    fragments.sort(key=lambda fragment: int(fragment.fragment_id))
    random.Random(config.seed).shuffle(fragments)  # noqa: S311 - deterministic benchmark input

    produced = 0
    for fragment_index, fragment in enumerate(fragments):
        remaining_tasks = config.target_tasks - produced
        if remaining_tasks <= 0:
            return
        remaining_fragments = len(fragments) - fragment_index
        quota = math.ceil(remaining_tasks / remaining_fragments)
        scanner = dataset.scanner(
            columns=[config.source_ref_column],
            filter=f"{config.modality_column} = 'image'",
            fragments=[fragment],
            scan_in_order=True,
            batch_size=config.scan_batch_size,
            batch_readahead=1,
            fragment_readahead=1,
        )
        pending: list[str] = []
        fragment_tasks = 0
        for batch in scanner.to_batches():
            column_index = batch.schema.get_field_index(config.source_ref_column)
            pending.extend(_valid_urls(batch.column(column_index)))
            while len(pending) >= config.task_rows and fragment_tasks < quota:
                refs = tuple(pending[: config.task_rows])
                del pending[: config.task_rows]
                yield FragmentTask(fragment_id=int(fragment.fragment_id), source_refs=refs)
                produced += 1
                fragment_tasks += 1
                if produced >= config.target_tasks:
                    return
            if fragment_tasks >= quota:
                break


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be greater than zero"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preset", required=True, choices=tuple(PRESETS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--document-uri", required=True)
    parser.add_argument("--document-version", type=_positive, default=DEFAULT_DOCUMENT_VERSION)
    parser.add_argument("--storage-options-json", default="{}", help="Inline JSON object or @path")
    parser.add_argument("--source-ref-column", default="source_ref")
    parser.add_argument("--modality-column", default="modality")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--scan-batch-size", type=_positive, default=65_536)
    return parser


def _open_document_dataset(args: argparse.Namespace, storage_options: Mapping[str, str]) -> _LanceDataset:
    try:
        import lance
    except ImportError as exc:
        msg = "manifest generation requires the lance Python package"
        raise ImportError(msg) from exc
    dataset = lance.dataset(
        args.document_uri,
        version=args.document_version,
        storage_options=dict(storage_options) or None,
    )
    if int(dataset.version) != args.document_version:
        msg = f"document dataset resolved version {dataset.version}; expected {args.document_version}"
        raise RuntimeError(msg)
    missing = sorted({args.source_ref_column, args.modality_column} - set(dataset.schema.names))
    if missing:
        msg = f"document dataset is missing columns: {missing}"
        raise ValueError(msg)
    return dataset


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (args.source_ref_column, args.modality_column):
        if not _SQL_IDENTIFIER.fullmatch(name):
            msg = f"column name must be a simple SQL identifier: {name!r}"
            raise ValueError(msg)
    preset = PRESETS[args.preset]
    config = ManifestConfig(
        target_tasks=preset.target_tasks,
        actor_count=preset.actor_count,
        document_uri=args.document_uri,
        document_version=args.document_version,
        seed=args.seed,
    )
    storage_options = _json_options(args.storage_options_json)
    _reject_secret_storage_options(storage_options)
    dataset = _open_document_dataset(args, storage_options)
    tasks = iter_document_tasks(
        dataset,
        DocumentScanConfig(
            source_ref_column=args.source_ref_column,
            modality_column=args.modality_column,
            seed=args.seed,
            target_tasks=config.target_tasks,
            task_rows=config.task_rows,
            scan_batch_size=args.scan_batch_size,
        ),
    )
    metadata = write_manifest(
        tasks,
        args.output_dir,
        config,
        storage_option_keys=tuple(storage_options),
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(args.output_dir.resolve()),
                "target_rows": metadata["target_rows"],
                "target_tasks": metadata["target_tasks"],
                "actor_count": metadata["actor_count"],
                "tasks_per_actor": metadata["tasks_per_actor"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
