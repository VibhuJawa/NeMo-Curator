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

"""Derive a nested weak-scaling manifest family from one validated scan.

The eight-node manifest is the sole sampled workload. Smaller manifests are
exact left-task prefixes of it, then re-sharded by ``left_task_id % actors``.
The complete one/two/four/eight-node family is published with one atomic,
same-filesystem directory rename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarking.scripts import generate_gpu_lance_saturation_manifest as generator  # noqa: E402 - direct CLI

_TASK_DIGEST_DOMAIN = b"gpu-lance-left-task-v1\0"
_SEQUENCE_DIGEST_DOMAIN = b"gpu-lance-left-task-sequence-v1\0"
_ACTOR_ASSIGNMENT = "left_task_id modulo actor_count"
_GEOMETRY_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class ScalingGeometry:
    """One member of a nested weak-scaling manifest family."""

    name: str
    target_tasks: int
    actor_count: int
    task_rows: int = generator.TASK_ROWS

    def __post_init__(self) -> None:
        if not _GEOMETRY_NAME.fullmatch(self.name):
            msg = "name must contain only lowercase alphanumerics separated by single hyphens"
            raise ValueError(msg)
        if self.target_tasks <= 0 or self.actor_count <= 0 or self.task_rows <= 0:
            msg = "target_tasks, actor_count, and task_rows must be positive"
            raise ValueError(msg)
        if self.target_tasks % self.actor_count:
            msg = "target_tasks must be divisible by actor_count"
            raise ValueError(msg)

    @property
    def tasks_per_actor(self) -> int:
        """Return the exact balanced task count for each actor."""

        return self.target_tasks // self.actor_count

    @property
    def target_rows(self) -> int:
        """Return the exact row count for this manifest."""

        return self.target_tasks * self.task_rows


DEFAULT_GEOMETRIES = tuple(
    ScalingGeometry(
        name=preset.name,
        target_tasks=preset.target_tasks,
        actor_count=preset.actor_count,
        task_rows=generator.TASK_ROWS,
    )
    for preset in generator.PRESETS.values()
)


@dataclass(frozen=True)
class ValidatedManifest:
    """Validated identities and logical task digests for one manifest."""

    metadata: dict[str, Any]
    config: generator.ManifestConfig
    manifest_json_sha256: str
    manifest_parquet_sha256: str
    task_digests: tuple[str, ...]

    @property
    def task_sequence_sha256(self) -> str:
        """Return a schema-independent digest of the ordered logical tasks."""

        return _task_sequence_sha256(self.task_digests)


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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"manifest metadata is unreadable: {path}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"manifest metadata must contain an object: {path}"
        raise TypeError(msg)
    return payload


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"manifest {key} must be a positive integer"
        raise ValueError(msg)
    return value


def _manifest_config(metadata: Mapping[str, Any], expected: ScalingGeometry) -> generator.ManifestConfig:
    schema_version = metadata.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        msg = "manifest schema_version must be 1"
        raise ValueError(msg)
    document = metadata.get("document")
    if not isinstance(document, Mapping):
        msg = "manifest document identity must be an object"
        raise TypeError(msg)
    uri = document.get("uri")
    version = document.get("version")
    option_keys = document.get("storage_option_keys")
    if not isinstance(uri, str) or not uri:
        msg = "manifest document URI must be a non-empty string"
        raise ValueError(msg)
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        msg = "manifest document version must be a positive integer"
        raise ValueError(msg)
    if (
        not isinstance(option_keys, list)
        or not all(isinstance(key, str) for key in option_keys)
        or option_keys != sorted(set(option_keys))
    ):
        msg = "manifest storage_option_keys must be a sorted list of unique strings"
        raise ValueError(msg)
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = "manifest seed must be an integer"
        raise TypeError(msg)
    task_rows = _required_positive_int(metadata, "task_rows")
    if task_rows != expected.task_rows:
        msg = f"manifest task_rows is {task_rows}; expected {expected.task_rows}"
        raise ValueError(msg)
    config = generator.ManifestConfig(
        target_tasks=expected.target_tasks,
        actor_count=expected.actor_count,
        document_uri=uri,
        document_version=version,
        seed=seed,
        task_rows=task_rows,
    )
    expected_values = {
        "target_tasks": config.target_tasks,
        "target_rows": config.target_rows,
        "actor_count": config.actor_count,
        "tasks_per_actor": config.tasks_per_actor,
        "rows_per_actor": config.tasks_per_actor * config.task_rows,
        "actor_assignment": _ACTOR_ASSIGNMENT,
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected_values.items()
        if metadata.get(key) != value
    }
    if mismatches:
        msg = f"manifest geometry is invalid: {json.dumps(mismatches, sort_keys=True)}"
        raise ValueError(msg)
    return config


def _validate_schema(path: Path, metadata: Mapping[str, Any], config: generator.ManifestConfig) -> pa.Schema:
    schema = pq.read_schema(path)
    if metadata.get("schema") != str(schema):
        msg = "manifest schema differs from its recorded schema"
        raise ValueError(msg)
    expected = generator._manifest_schema(config)
    if not schema.equals(expected, check_metadata=True):
        msg = "manifest Arrow schema or pinned source metadata is invalid"
        raise ValueError(msg)
    return schema


def _validate_file_inventory(
    manifest_dir: Path,
    metadata: Mapping[str, Any],
    config: generator.ManifestConfig,
) -> None:
    actor_dir = manifest_dir / "actors"
    actor_paths = sorted(actor_dir.glob("actor_*.parquet"))
    expected_actor_paths = [actor_dir / f"actor_{actor_id:03d}.parquet" for actor_id in range(config.actor_count)]
    if actor_paths != expected_actor_paths:
        msg = f"manifest actor inventory has {len(actor_paths)} shards; expected {config.actor_count} exact names"
        raise ValueError(msg)
    relative_paths = [Path("manifest.parquet"), *(Path("actors") / path.name for path in expected_actor_paths)]
    files = metadata.get("files")
    if not isinstance(files, Mapping) or set(files) != {str(path) for path in relative_paths}:
        msg = "manifest file inventory differs from the expected combined and actor files"
        raise ValueError(msg)
    for relative_path in relative_paths:
        path = manifest_dir / relative_path
        identity = files[str(relative_path)]
        if not isinstance(identity, Mapping):
            msg = f"manifest file identity must be an object: {relative_path}"
            raise TypeError(msg)
        expected_rows = (
            config.target_rows
            if relative_path == Path("manifest.parquet")
            else config.tasks_per_actor * config.task_rows
        )
        actual_bytes = path.stat().st_size
        actual_sha256 = _sha256(path)
        if identity.get("bytes") != actual_bytes or identity.get("sha256") != actual_sha256:
            msg = f"manifest file identity mismatch: {relative_path}"
            raise ValueError(msg)
        actual_rows = pq.read_metadata(path).num_rows
        expected_identity = {"bytes": actual_bytes, "rows": expected_rows, "sha256": actual_sha256}
        if dict(identity) != expected_identity or actual_rows != expected_rows:
            msg = f"manifest file identity mismatch: {relative_path}"
            raise ValueError(msg)


def _logical_task_digest(table: pa.Table) -> str:
    normalized = table.combine_chunks().replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, normalized.schema) as writer:
        writer.write_table(normalized)
    digest = hashlib.sha256()
    digest.update(_TASK_DIGEST_DOMAIN)
    digest.update(sink.getvalue().to_pybytes())
    return digest.hexdigest()


def _task_sequence_sha256(task_digests: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(_SEQUENCE_DIGEST_DOMAIN)
    for task_digest in task_digests:
        digest.update(bytes.fromhex(task_digest))
    return digest.hexdigest()


def _validate_task_table(table: pa.Table, task_id: int, task_rows: int) -> generator.FragmentTask:
    if table.num_rows != task_rows:
        msg = f"left task {task_id} has {table.num_rows} rows; expected {task_rows}"
        raise ValueError(msg)
    expected_positions = list(range(task_rows))
    task_ids = table["left_task_id"].combine_chunks().to_pylist()
    fragments = table["source_fragment_id"].combine_chunks().to_pylist()
    positions = table["source_position"].combine_chunks().to_pylist()
    positions_in_task = table["source_position_in_task"].combine_chunks().to_pylist()
    source_refs = table["source_ref"].combine_chunks().to_pylist()
    if task_ids != [task_id] * task_rows:
        msg = f"row group {task_id} does not contain its exact left_task_id"
        raise ValueError(msg)
    if not fragments or fragments[0] is None or fragments[0] < 0 or fragments != [fragments[0]] * task_rows:
        msg = f"left task {task_id} does not contain one nonnegative source fragment"
        raise ValueError(msg)
    if positions != expected_positions or positions_in_task != expected_positions:
        msg = f"left task {task_id} source positions are not 0..task_rows-1"
        raise ValueError(msg)
    if any(not isinstance(value, str) or not value for value in source_refs):
        msg = f"left task {task_id} contains a null, non-string, or empty source_ref"
        raise ValueError(msg)
    return generator.FragmentTask(fragment_id=fragments[0], source_refs=tuple(source_refs))


def _read_combined_tasks(path: Path, config: generator.ManifestConfig) -> Iterator[tuple[generator.FragmentTask, str]]:
    parquet = pq.ParquetFile(path)
    if parquet.num_row_groups != config.target_tasks:
        msg = (
            f"combined manifest has {parquet.num_row_groups} row groups; expected one per task ({config.target_tasks})"
        )
        raise ValueError(msg)
    for task_id in range(config.target_tasks):
        table = parquet.read_row_group(task_id)
        task = _validate_task_table(table, task_id, config.task_rows)
        yield task, _logical_task_digest(table)


def _validate_actor_shards(
    manifest_dir: Path,
    config: generator.ManifestConfig,
    combined_task_digests: Sequence[str],
) -> None:
    expected_schema = generator._manifest_schema(config)
    for actor_id in range(config.actor_count):
        parquet = pq.ParquetFile(manifest_dir / "actors" / f"actor_{actor_id:03d}.parquet")
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
            msg = f"actor {actor_id} Arrow schema or pinned source metadata is invalid"
            raise ValueError(msg)
        if parquet.num_row_groups != config.tasks_per_actor:
            msg = f"actor {actor_id} has {parquet.num_row_groups} row groups; expected {config.tasks_per_actor}"
            raise ValueError(msg)
        for actor_task_index in range(config.tasks_per_actor):
            expected_task_id = actor_id + actor_task_index * config.actor_count
            table = parquet.read_row_group(actor_task_index)
            _validate_task_table(table, expected_task_id, config.task_rows)
            if _logical_task_digest(table) != combined_task_digests[expected_task_id]:
                msg = f"actor {actor_id} task {expected_task_id} differs from the combined manifest"
                raise ValueError(msg)


def validate_manifest(manifest_dir: Path, expected: ScalingGeometry) -> ValidatedManifest:
    """Validate source identity, all file hashes, task order, and actor shards."""

    manifest_dir = manifest_dir.resolve()
    metadata_path = manifest_dir / "manifest.json"
    manifest_path = manifest_dir / "manifest.parquet"
    if not metadata_path.is_file() or not manifest_path.is_file():
        msg = f"manifest directory must contain manifest.json and manifest.parquet: {manifest_dir}"
        raise FileNotFoundError(msg)
    metadata = _read_json_object(metadata_path)
    config = _manifest_config(metadata, expected)
    _validate_file_inventory(manifest_dir, metadata, config)
    _validate_schema(manifest_path, metadata, config)
    fragment_counts: dict[int, int] = {}
    task_digests: list[str] = []
    for task, task_digest in _read_combined_tasks(manifest_path, config):
        fragment_counts[task.fragment_id] = fragment_counts.get(task.fragment_id, 0) + 1
        task_digests.append(task_digest)
    expected_source_fragments = {
        "count": len(fragment_counts),
        "tasks_by_fragment": {str(key): fragment_counts[key] for key in sorted(fragment_counts)},
    }
    if metadata.get("source_fragments") != expected_source_fragments:
        msg = "manifest source_fragments metadata differs from the combined task stream"
        raise ValueError(msg)
    _validate_actor_shards(manifest_dir, config, task_digests)
    return ValidatedManifest(
        metadata=metadata,
        config=config,
        manifest_json_sha256=_sha256(metadata_path),
        manifest_parquet_sha256=_sha256(manifest_path),
        task_digests=tuple(task_digests),
    )


def _iter_task_prefix(
    master_path: Path, config: generator.ManifestConfig, target_tasks: int
) -> Iterator[generator.FragmentTask]:
    parquet = pq.ParquetFile(master_path)
    for task_id in range(target_tasks):
        yield _validate_task_table(parquet.read_row_group(task_id), task_id, config.task_rows)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_geometries(geometries: Sequence[ScalingGeometry]) -> tuple[ScalingGeometry, ...]:
    values = tuple(geometries)
    if not values:
        msg = "at least one scaling geometry is required"
        raise ValueError(msg)
    if len({geometry.name for geometry in values}) != len(values):
        msg = "scaling geometry names must be unique"
        raise ValueError(msg)
    if any(current.target_tasks >= following.target_tasks for current, following in pairwise(values)):
        msg = "scaling geometries must have strictly increasing target_tasks"
        raise ValueError(msg)
    if any(current.actor_count >= following.actor_count for current, following in pairwise(values)):
        msg = "scaling geometries must have strictly increasing actor_count"
        raise ValueError(msg)
    tasks_per_actor = {geometry.tasks_per_actor for geometry in values}
    if len(tasks_per_actor) != 1:
        msg = "all nested weak-scaling geometries must preserve tasks_per_actor"
        raise ValueError(msg)
    if len({geometry.task_rows for geometry in values}) != 1:
        msg = "all nested weak-scaling geometries must preserve task_rows"
        raise ValueError(msg)
    return values


def _require_exact_prefix(
    geometry: ScalingGeometry,
    derived: ValidatedManifest,
    master_prefix_digests: tuple[str, ...],
    prefix_sha256: str,
) -> None:
    if derived.task_digests != master_prefix_digests or derived.task_sequence_sha256 != prefix_sha256:
        msg = f"derived {geometry.name} manifest is not the exact task prefix of the master"
        raise RuntimeError(msg)


def derive_nested_manifest_family(
    master_manifest_dir: Path,
    output_root: Path,
    geometries: Sequence[ScalingGeometry] = DEFAULT_GEOMETRIES,
) -> dict[str, Any]:
    """Atomically publish exact task prefixes derived from one master scan."""

    geometry_values = _validate_geometries(geometries)
    output_root = output_root.resolve()
    if output_root.exists():
        msg = f"output root already exists: {output_root}"
        raise FileExistsError(msg)
    master = validate_manifest(master_manifest_dir, geometry_values[-1])
    if "derivation" in master.metadata:
        msg = "master manifest must be a direct generator output, not another derived prefix"
        raise ValueError(msg)
    master_path = master_manifest_dir.resolve() / "manifest.parquet"
    document = master.metadata["document"]
    storage_option_keys = tuple(document["storage_option_keys"])
    master_field_schema_sha256 = hashlib.sha256(
        str(pq.read_schema(master_path).remove_metadata()).encode("utf-8")
    ).hexdigest()

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        family_manifests: dict[str, Any] = {}
        for geometry in geometry_values:
            config = generator.ManifestConfig(
                target_tasks=geometry.target_tasks,
                actor_count=geometry.actor_count,
                document_uri=master.config.document_uri,
                document_version=master.config.document_version,
                seed=master.config.seed,
                task_rows=master.config.task_rows,
            )
            child = staging / geometry.name
            metadata = generator.write_manifest(
                _iter_task_prefix(master_path, master.config, geometry.target_tasks),
                child,
                config,
                storage_option_keys=storage_option_keys,
            )
            master_prefix_digests = master.task_digests[: geometry.target_tasks]
            prefix_sha256 = _task_sequence_sha256(master_prefix_digests)
            metadata["derivation"] = {
                "kind": "exact_left_task_prefix_v1",
                "master": {
                    "manifest_json_sha256": master.manifest_json_sha256,
                    "manifest_parquet_sha256": master.manifest_parquet_sha256,
                    "target_tasks": master.config.target_tasks,
                    "target_rows": master.config.target_rows,
                    "actor_count": master.config.actor_count,
                    "schema": master.metadata["schema"],
                    "field_schema_sha256": master_field_schema_sha256,
                },
                "prefix": {
                    "task_id_start": 0,
                    "task_id_stop": geometry.target_tasks,
                    "task_sequence_sha256": prefix_sha256,
                    "verified_against_master": True,
                },
            }
            _write_json_atomic(child / "manifest.json", metadata)
            derived = validate_manifest(child, geometry)
            _require_exact_prefix(geometry, derived, master_prefix_digests, prefix_sha256)
            family_manifests[geometry.name] = {
                "directory": geometry.name,
                "target_tasks": config.target_tasks,
                "target_rows": config.target_rows,
                "actor_count": config.actor_count,
                "tasks_per_actor": config.tasks_per_actor,
                "manifest_json_sha256": derived.manifest_json_sha256,
                "manifest_parquet_sha256": derived.manifest_parquet_sha256,
                "task_sequence_sha256": derived.task_sequence_sha256,
            }

        family: dict[str, Any] = {
            "schema_version": 1,
            "kind": "gpu_lance_nested_weak_scaling_prefixes",
            "document": document,
            "seed": master.config.seed,
            "task_rows": master.config.task_rows,
            "actor_assignment": _ACTOR_ASSIGNMENT,
            "master": {
                "manifest_json_sha256": master.manifest_json_sha256,
                "manifest_parquet_sha256": master.manifest_parquet_sha256,
                "target_tasks": master.config.target_tasks,
                "target_rows": master.config.target_rows,
                "actor_count": master.config.actor_count,
                "schema": master.metadata["schema"],
                "field_schema_sha256": master_field_schema_sha256,
            },
            "manifests": family_manifests,
        }
        family_path = staging / "family.json"
        _write_json_atomic(family_path, family)
        _fsync_file(family_path)
        _fsync_directory(staging)
        os.replace(staging, output_root)
        _fsync_directory(output_root.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    else:
        return family


def build_parser() -> argparse.ArgumentParser:
    """Build the exact eight-node-master derivation CLI."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--master-manifest-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one master and publish its nested scaling family."""

    args = build_parser().parse_args(argv)
    family = derive_nested_manifest_family(args.master_manifest_dir, args.output_root)
    print(
        json.dumps(
            {
                "status": "completed",
                "output_root": str(args.output_root.resolve()),
                "master_manifest_parquet_sha256": family["master"]["manifest_parquet_sha256"],
                "manifests": family["manifests"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
