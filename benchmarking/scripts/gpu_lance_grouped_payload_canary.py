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

"""Run the fixed 64-block grouped Lance payload materializer canary.

This is a candidate-only grouped-materializer measurement. It is not an
overlay benchmark, an A/B comparison, or a GPU URL-lookup benchmark, and it
does not submit jobs or arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
    lance_coordinate_plan_schema,
)
from nemo_curator.utils.uri import validate_credential_free_uri_identity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


BLOCK_COUNT = 64
ROWS_PER_BLOCK = 4096
EXPECTED_ROWS = BLOCK_COUNT * ROWS_PER_BLOCK
SHARED_SPOOL_BUDGET_BYTES = 1024**3
COORDINATE_WINDOW_BYTES = 4 * 1024**3
SPOOL_BUCKET_ROWS = 131_072
SPOOL_SYNC_MODE = "fsync"
_QUERY_ORDINAL = "_benchmark_query_ordinal"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SECRET_OPTION_PARTS = ("access_key", "secret", "token", "password", "credential")
_MIB = 1024**2

CleanupPolicy = Literal["always", "on-success", "never"]


@dataclass(frozen=True)
class QueryContract:
    """Fully validated frozen query input and its fixed contiguous blocks."""

    path: Path
    table: pa.Table
    blocks: tuple[pa.Table, ...]
    file_sha256: str
    harness_logical_digest_sha256: str
    stable_id_sequence_digest_sha256: str
    expected_columns: tuple[str, ...]
    stable_id_column: str
    source_ref_column: str
    rows: int
    unique_stable_ids: int


@dataclass(frozen=True)
class StorageOptionsIdentity:
    """Secret-free persisted identity for runtime storage options."""

    keys: tuple[str, ...]
    sha256: str


class _SpoolManifest(Protocol):
    schema: pa.Schema
    total_rows: int
    total_arrow_nbytes: int
    peak_active_bytes: int
    peak_bounded_active_bytes: int
    files: Sequence[object]
    oversized_rows: Sequence[object]
    sha256: str


class _Spool(Protocol):
    schema: pa.Schema

    def finish(self) -> _SpoolManifest: ...

    def iter_tables(self) -> Sequence[pa.Table]: ...


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _sha256(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        msg = "value must be a lowercase SHA-256 digest"
        raise argparse.ArgumentTypeError(msg)
    return value


def _credential_free_uri(value: str) -> str:
    try:
        return validate_credential_free_uri_identity(value, "image Lance URI")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the single-process canary CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-parquet", required=True, type=Path)
    parser.add_argument("--query-file-sha256", required=True, type=_sha256)
    parser.add_argument("--harness-logical-digest-sha256", required=True, type=_sha256)
    parser.add_argument("--stable-id-sequence-digest-sha256", required=True, type=_sha256)
    parser.add_argument("--expected-rows", required=True, type=_positive)
    parser.add_argument("--expected-unique-stable-ids", required=True, type=_positive)
    parser.add_argument("--source-ref-column", default="source_ref")
    parser.add_argument("--stable-id-column", default=STABLE_ROW_ID)
    parser.add_argument("--image-lance-uri", required=True, type=_credential_free_uri)
    parser.add_argument("--image-lance-version", required=True, type=_positive)
    parser.add_argument("--image-fragment-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--storage-options-file", required=True, type=Path)
    parser.add_argument("--fetch-batch-size", type=_positive, default=1024)
    parser.add_argument("--max-pending", type=_positive, default=16)
    parser.add_argument("--io-threads", type=_positive, default=64)
    parser.add_argument("--metadata-cache-mib", type=_positive, default=512)
    parser.add_argument("--expected-payload-digest-sha256", required=True, type=_sha256)
    parser.add_argument("--expected-payload-bytes", required=True, type=_positive)
    parser.add_argument("--spool-root", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--cleanup-policy", choices=("always", "on-success", "never"), required=True)
    return parser


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * _MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow_stream_sha256(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue()).hexdigest()


def _harness_logical_digest(table: pa.Table, source_ref_column: str) -> tuple[str, tuple[str, ...]]:
    if _QUERY_ORDINAL in table.column_names:
        msg = f"query parquet contains reserved column {_QUERY_ORDINAL!r}"
        raise ValueError(msg)
    expected_columns = tuple(
        column
        for aliases in (("expected_md5", "md5"), ("expected_width", "width"), ("expected_height", "height"))
        if (column := next((name for name in aliases if name in table.column_names), None)) is not None
    )
    with_ordinal = table.append_column(_QUERY_ORDINAL, pa.array(range(table.num_rows), type=pa.int64()))
    selected = with_ordinal.select([source_ref_column, _QUERY_ORDINAL, *expected_columns])
    return _arrow_stream_sha256(selected), expected_columns


def _stable_id_sequence_digest(stable_ids: pa.Array | pa.ChunkedArray) -> str:
    """Hash ordered stable IDs as length-framed little-endian uint64 values."""

    chunks = stable_ids.chunks if isinstance(stable_ids, pa.ChunkedArray) else (stable_ids,)
    digest = hashlib.sha256()
    for chunk in chunks:
        values = chunk.to_numpy(zero_copy_only=False).astype("<u8", copy=False)
        framed = np.empty((len(values), 2), dtype="<u8")
        framed[:, 0] = 8
        framed[:, 1] = values
        digest.update(framed.tobytes(order="C"))
    return digest.hexdigest()


def _require_regular_file(path: Path, name: str) -> None:
    if not path.is_absolute():
        msg = f"{name} must be an absolute path"
        raise ValueError(msg)
    if path.is_symlink() or not path.is_file():
        msg = f"{name} must be an existing regular non-symlink file: {path}"
        raise ValueError(msg)


def _validate_new_absolute_path(path: Path, name: str) -> None:
    if not path.is_absolute():
        msg = f"{name} must be an absolute path"
        raise ValueError(msg)
    if path.exists() or path.is_symlink():
        msg = f"refusing to replace existing {name}: {path}"
        raise FileExistsError(msg)


def load_query_contract(  # noqa: C901, PLR0913
    path: Path,
    *,
    source_ref_column: str,
    stable_id_column: str,
    expected_file_sha256: str,
    expected_harness_logical_digest_sha256: str,
    expected_stable_id_sequence_digest_sha256: str,
    expected_rows: int,
    expected_unique_stable_ids: int,
    block_count: int = BLOCK_COUNT,
    rows_per_block: int = ROWS_PER_BLOCK,
) -> QueryContract:
    """Load and pin the complete query before any remote dataset setup."""

    _require_regular_file(path, "query parquet")
    required_rows = block_count * rows_per_block
    if expected_rows != required_rows:
        msg = f"expected_rows must pin the fixed geometry row count {required_rows}, got {expected_rows}"
        raise ValueError(msg)
    file_sha256 = _file_sha256(path)
    if file_sha256 != expected_file_sha256:
        msg = "query parquet file SHA-256 does not match the caller pin"
        raise ValueError(msg)
    table = pq.read_table(path)
    missing = [name for name in (source_ref_column, stable_id_column) if name not in table.column_names]
    if missing:
        msg = f"query parquet is missing required columns: {missing}"
        raise ValueError(msg)
    if table.num_rows != required_rows:
        msg = f"query parquet has {table.num_rows} rows; fixed canary geometry requires {required_rows}"
        raise ValueError(msg)
    source_refs = table[source_ref_column]
    if source_refs.null_count or not (
        pa.types.is_string(source_refs.type) or pa.types.is_large_string(source_refs.type)
    ):
        msg = "query source references must be non-null string values"
        raise TypeError(msg)
    stable_ids = table[stable_id_column]
    if stable_ids.type != pa.uint64() or stable_ids.null_count:
        msg = "query stable IDs must be non-null uint64 values"
        raise TypeError(msg)

    harness_digest, expected_columns = _harness_logical_digest(table, source_ref_column)
    if harness_digest != expected_harness_logical_digest_sha256:
        msg = "query harness logical digest does not match the caller pin"
        raise ValueError(msg)
    stable_digest = _stable_id_sequence_digest(stable_ids)
    if stable_digest != expected_stable_id_sequence_digest_sha256:
        msg = "query stable-ID sequence digest does not match the caller pin"
        raise ValueError(msg)
    unique_stable_ids = int(pc.count_distinct(stable_ids, mode="only_valid").as_py())
    if unique_stable_ids != expected_unique_stable_ids:
        msg = f"query contains {unique_stable_ids} unique stable IDs; caller pinned {expected_unique_stable_ids}"
        raise ValueError(msg)

    blocks = tuple(table.slice(index * rows_per_block, rows_per_block) for index in range(block_count))
    if len(blocks) != block_count or any(block.num_rows != rows_per_block for block in blocks):
        msg = "query parquet did not split into the exact contiguous block geometry"
        raise RuntimeError(msg)
    return QueryContract(
        path=path,
        table=table,
        blocks=blocks,
        file_sha256=file_sha256,
        harness_logical_digest_sha256=harness_digest,
        stable_id_sequence_digest_sha256=stable_digest,
        expected_columns=expected_columns,
        stable_id_column=stable_id_column,
        source_ref_column=source_ref_column,
        rows=table.num_rows,
        unique_stable_ids=unique_stable_ids,
    )


def build_coordinate_plans(contract: QueryContract) -> tuple[pa.Table, ...]:
    """Build one synthetic coordinate plan for each contiguous query block."""

    schema = lance_coordinate_plan_schema()
    plans: list[pa.Table] = []
    for block_index, block in enumerate(contract.blocks):
        start = block_index * block.num_rows
        positions = pa.array(range(start, start + block.num_rows), type=pa.uint64())
        stable_ids = block[contract.stable_id_column].combine_chunks()
        plans.append(pa.Table.from_arrays([positions, positions, stable_ids], schema=schema))
    return tuple(plans)


def _load_storage_options(path: Path) -> tuple[dict[str, str], StorageOptionsIdentity]:
    _require_regular_file(path, "storage options file")
    parsed = json.loads(path.read_bytes())
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
    ):
        msg = "storage options must be a JSON object with string keys and values"
        raise TypeError(msg)
    secret_keys = sorted(key for key in parsed if any(part in key.casefold() for part in _SECRET_OPTION_PARTS))
    if secret_keys:
        msg = f"storage option keys {secret_keys} look credential-bearing; use the process environment"
        raise ValueError(msg)
    canonical = _canonical_json_sha256(parsed)
    return parsed, StorageOptionsIdentity(keys=tuple(sorted(parsed)), sha256=canonical)


def _payload_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    msg = f"image payload must be bytes-like, got {type(value).__name__}"
    raise TypeError(msg)


def _historical_payload_prefix(ordinal: int, source_ref: str) -> bytes:
    return json.dumps([ordinal, source_ref, True], separators=(",", ":"), default=str).encode()


def historical_payload_oracle(source_refs: Sequence[str], payloads: Sequence[bytes]) -> tuple[str, int]:
    """Compute the historical image-only payload oracle for a tiny in-memory fixture."""

    if len(source_refs) != len(payloads):
        msg = "source_refs and payloads must have the same length"
        raise ValueError(msg)
    digest = hashlib.sha256()
    payload_bytes = 0
    for ordinal, (source_ref, payload) in enumerate(zip(source_refs, payloads, strict=True)):
        raw = _payload_bytes(payload)
        digest.update(_historical_payload_prefix(ordinal, source_ref))
        digest.update(hashlib.sha256(raw).digest())
        payload_bytes += len(raw)
    return digest.hexdigest(), payload_bytes


def validate_spools_and_payload_oracle(  # noqa: C901, PLR0912, PLR0913, PLR0915
    spools: Sequence[_Spool],
    contract: QueryContract,
    plans: Sequence[pa.Table],
    *,
    expected_schema: pa.Schema,
    image_column: str,
    expected_payload_digest_sha256: str,
    expected_payload_bytes: int,
) -> dict[str, object]:
    """Stream every block, validate exact coordinates, and reproduce the historical oracle."""

    if len(spools) != len(contract.blocks) or len(plans) != len(contract.blocks):
        msg = "spools, plans, and query blocks must have identical cardinality"
        raise ValueError(msg)
    digest = hashlib.sha256()
    total_payload_bytes = 0
    total_rows = 0
    block_results: list[dict[str, object]] = []
    for block_index, (spool, block, plan) in enumerate(zip(spools, contract.blocks, plans, strict=True)):
        if not spool.schema.equals(expected_schema, check_metadata=True):
            msg = f"spool {block_index} configured schema differs from the pinned image-only schema"
            raise TypeError(msg)
        manifest = spool.finish()
        if not manifest.schema.equals(expected_schema, check_metadata=True):
            msg = f"spool {block_index} manifest schema differs from the pinned image-only schema"
            raise TypeError(msg)
        if manifest.total_rows != block.num_rows:
            msg = f"spool {block_index} manifest row count is incorrect"
            raise RuntimeError(msg)

        block_start = block_index * block.num_rows
        block_stop = block_start + block.num_rows
        expected_stable_ids = plan[STABLE_ROW_ID].combine_chunks()
        payload_hashes: list[bytes | None] = [None] * block.num_rows
        seen = np.zeros(block.num_rows, dtype=np.bool_)
        block_payload_bytes = 0
        block_rows = 0
        table_count = 0
        for table in spool.iter_tables():
            table_count += 1
            if not table.schema.equals(expected_schema, check_metadata=True):
                msg = f"spool {block_index} yielded an unexpected table schema"
                raise TypeError(msg)
            for name in (DOCUMENT_ROWADDR, DOCUMENT_POSITION, STABLE_ROW_ID):
                values = table[name]
                if values.type != pa.uint64() or values.null_count:
                    msg = f"spool {block_index} coordinate {name!r} must be non-null uint64"
                    raise TypeError(msg)
            payload_column = table[image_column]
            if payload_column.null_count:
                msg = f"spool {block_index} contains null image payloads"
                raise RuntimeError(msg)

            positions = table[DOCUMENT_POSITION].combine_chunks().to_numpy(zero_copy_only=False)
            rowaddrs = table[DOCUMENT_ROWADDR].combine_chunks().to_numpy(zero_copy_only=False)
            if not np.array_equal(positions, rowaddrs):
                msg = f"spool {block_index} synthetic row addresses differ from document positions"
                raise RuntimeError(msg)
            if positions.size and (int(positions.min()) < block_start or int(positions.max()) >= block_stop):
                msg = f"spool {block_index} contains a position outside its contiguous query block"
                raise RuntimeError(msg)
            local_positions = positions.astype(np.int64, copy=False) - block_start
            if np.unique(local_positions).size != local_positions.size or seen[local_positions].any():
                msg = f"spool {block_index} contains duplicate document positions"
                raise RuntimeError(msg)
            expected_ids = expected_stable_ids.take(pa.array(local_positions, type=pa.int64()))
            if not table[STABLE_ROW_ID].combine_chunks().equals(expected_ids):
                msg = f"spool {block_index} stable IDs differ from the frozen query sequence"
                raise RuntimeError(msg)

            for row_index, local_position in enumerate(local_positions):
                payload = _payload_bytes(payload_column[row_index].as_py())
                payload_hashes[int(local_position)] = hashlib.sha256(payload).digest()
                block_payload_bytes += len(payload)
            seen[local_positions] = True
            block_rows += table.num_rows

        if block_rows != block.num_rows or not bool(seen.all()) or any(value is None for value in payload_hashes):
            msg = f"spool {block_index} does not cover its query block exactly once"
            raise RuntimeError(msg)
        source_refs = block[contract.source_ref_column].combine_chunks()
        for local_position, payload_sha256 in enumerate(payload_hashes):
            source_ref = source_refs[local_position].as_py()
            if not isinstance(source_ref, str) or payload_sha256 is None:  # pragma: no cover - validated above
                msg = "validated query/payload state became invalid"
                raise RuntimeError(msg)
            digest.update(_historical_payload_prefix(block_start + local_position, source_ref))
            digest.update(payload_sha256)
        total_rows += block_rows
        total_payload_bytes += block_payload_bytes
        block_results.append(
            {
                "block_index": block_index,
                "query_start": block_start,
                "query_stop": block_stop,
                "rows": block_rows,
                "payload_bytes": block_payload_bytes,
                "tables_streamed": table_count,
                "manifest_sha256": manifest.sha256,
                "arrow_bytes": manifest.total_arrow_nbytes,
                "files": len(manifest.files),
                "peak_active_bytes": manifest.peak_active_bytes,
                "peak_bounded_active_bytes": manifest.peak_bounded_active_bytes,
                "oversized_rows": len(manifest.oversized_rows),
            }
        )

    payload_digest = digest.hexdigest()
    if total_rows != contract.rows:
        msg = f"validated {total_rows} payload rows; expected {contract.rows}"
        raise RuntimeError(msg)
    if payload_digest != expected_payload_digest_sha256:
        msg = "streamed historical payload digest does not match the caller-pinned oracle"
        raise ValueError(msg)
    if total_payload_bytes != expected_payload_bytes:
        msg = f"streamed payload bytes are {total_payload_bytes}; caller pinned {expected_payload_bytes}"
        raise ValueError(msg)
    return {
        "payload_digest_sha256": payload_digest,
        "payload_bytes": total_payload_bytes,
        "rows": total_rows,
        "blocks": block_results,
        "whole_output_concatenated": False,
        "per_block_retained_payload_hash_bytes": max(block.num_rows for block in contract.blocks)
        * hashlib.sha256().digest_size,
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """Publish one immutable result through same-directory rename."""

    _validate_new_absolute_path(path, "result path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _git_output(path: Path, *arguments: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed git executable and arguments
            [git, "-C", str(path), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _path_code_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    identity: dict[str, object] = {"source_sha256": _file_sha256(resolved)}
    repository = _git_output(resolved.parent, "rev-parse", "--show-toplevel")
    if repository is None:
        identity["source_name"] = resolved.name
        return identity
    repository_path = Path(repository)
    commit = _git_output(repository_path, "rev-parse", "HEAD")
    status = _git_output(repository_path, "status", "--porcelain=v1", "--untracked-files=no")
    if commit is not None:
        identity["git_commit"] = commit
    identity["git_dirty"] = bool(status)
    try:
        identity["source_path"] = str(resolved.relative_to(repository_path))
    except ValueError:
        identity["source_name"] = resolved.name
    return identity


def _module_code_identity(module_name: str) -> dict[str, object] | None:
    try:
        specification = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    if specification is None or specification.origin is None:
        return None
    source = Path(specification.origin)
    return _path_code_identity(source) if source.is_file() else None


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("nemo-curator", "pylance", "lance-ray", "pyarrow", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _peak_rss_bytes() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def _cleanup_spools(spools: Sequence[object], spool_root: Path) -> None:
    for spool in spools:
        cleanup = getattr(spool, "cleanup", None)
        if callable(cleanup):
            cleanup()
    if spool_root.exists():
        spool_root.rmdir()


def _validate_invocation(args: argparse.Namespace) -> None:
    _require_regular_file(args.query_parquet, "query parquet")
    _require_regular_file(args.storage_options_file, "storage options file")
    _validate_new_absolute_path(args.spool_root, "spool root")
    _validate_new_absolute_path(args.result, "result path")
    if args.expected_rows != EXPECTED_ROWS:
        msg = f"--expected-rows must be {EXPECTED_ROWS} for the fixed 64 x 4096 canary"
        raise ValueError(msg)
    if not args.source_ref_column or not args.stable_id_column or not args.image_column:
        msg = "column names must be non-empty"
        raise ValueError(msg)
    try:
        args.result.relative_to(args.spool_root)
    except ValueError:
        pass
    else:
        msg = "result path must not be inside the ephemeral spool root"
        raise ValueError(msg)


def _result_document(  # noqa: PLR0913
    *,
    args: argparse.Namespace,
    contract: QueryContract,
    storage_identity: StorageOptionsIdentity,
    dataset_identity: Mapping[str, object],
    group_metrics: Mapping[str, object],
    plan_metrics: Sequence[Mapping[str, object]],
    validation: Mapping[str, object],
    timing: Mapping[str, float],
    rss: Mapping[str, int],
    retained: bool,
) -> dict[str, object]:
    configuration = {
        "block_count": BLOCK_COUNT,
        "rows_per_block": ROWS_PER_BLOCK,
        "fetch_batch_size": args.fetch_batch_size,
        "max_pending_fetch_batches": args.max_pending,
        "io_threads": args.io_threads,
        "metadata_cache_bytes": args.metadata_cache_mib * _MIB,
        "shared_spool_budget_bytes": SHARED_SPOOL_BUDGET_BYTES,
        "coordinate_window_bytes": COORDINATE_WINDOW_BYTES,
        "spool_bucket_rows": SPOOL_BUCKET_ROWS,
        "spool_sync_mode": SPOOL_SYNC_MODE,
        "cleanup_policy": args.cleanup_policy,
        "storage_option_keys": list(storage_identity.keys),
        "storage_options_identity_sha256": storage_identity.sha256,
    }
    physical_reads = int(group_metrics["lance_read_iops"])
    physical_bytes = int(group_metrics["lance_read_bytes"])
    logical_payload_bytes = int(group_metrics["actual_payload_bytes"])
    materialize_seconds = timing["materialize_seconds"]
    outer_seconds = timing["outer_seconds"]
    code_modules = {
        "benchmark": _path_code_identity(Path(__file__)),
        "grouped_materializer": _module_code_identity("nemo_curator.stages.interleaved.lance_payload_materialize"),
        "payload_spool": _module_code_identity("nemo_curator.stages.interleaved.lance_payload_spool"),
        "lance_ray": _module_code_identity("lance_ray"),
        "lance_ray_gpu": _module_code_identity("lance_ray.gpu"),
    }
    return {
        "schema_version": 1,
        "artifact_kind": "gpu_lance_grouped_payload_materializer_canary",
        "status": "completed",
        "evidence_class": "candidate_only_canary",
        "labels": {
            "measurement": "grouped materializer canary",
            "not_overlay_benchmark": True,
            "not_ab_comparison": True,
            "not_gpu_lookup_benchmark": True,
        },
        "input": {
            "query_parquet_path": str(contract.path),
            "query_file_sha256": contract.file_sha256,
            "harness_logical_digest_sha256": contract.harness_logical_digest_sha256,
            "stable_id_sequence_digest_sha256": contract.stable_id_sequence_digest_sha256,
            "stable_id_sequence_encoding": "length_framed_ordered_little_endian_uint64",
            "rows": contract.rows,
            "unique_stable_ids": contract.unique_stable_ids,
            "source_ref_column": contract.source_ref_column,
            "stable_id_column": contract.stable_id_column,
            "harness_expected_columns": list(contract.expected_columns),
            "block_count": len(contract.blocks),
            "rows_per_block": ROWS_PER_BLOCK,
        },
        "dataset": dict(dataset_identity),
        "configuration": {**configuration, "identity_sha256": _canonical_json_sha256(configuration)},
        "timing": dict(timing),
        "throughput": {
            "materialize_images_per_second": contract.rows / materialize_seconds,
            "outer_images_per_second": contract.rows / outer_seconds,
            "materialize_logical_mib_per_second": logical_payload_bytes / _MIB / materialize_seconds,
            "materialize_physical_mib_per_second": physical_bytes / _MIB / materialize_seconds,
        },
        "io": {
            "physical_reads": physical_reads,
            "physical_bytes": physical_bytes,
            "average_physical_read_bytes": physical_bytes / physical_reads if physical_reads else 0.0,
            "physical_reads_per_logical_payload": physical_reads / contract.rows,
            "physical_reads_per_unique_payload": physical_reads / contract.unique_stable_ids,
            "physical_to_logical_byte_ratio": (
                physical_bytes / logical_payload_bytes if logical_payload_bytes else 0.0
            ),
            "sparse_calls_avoided": int(group_metrics["sparse_calls_avoided"]),
        },
        "materializer": {
            "fetch_metrics": dict(group_metrics),
            "plan_metrics": list(plan_metrics),
        },
        "spool": {
            "retained": retained,
            "cleanup_policy": args.cleanup_policy,
            "shared_budget_bytes": SHARED_SPOOL_BUDGET_BYTES,
            "coordinate_window_bytes": COORDINATE_WINDOW_BYTES,
            "blocks": validation["blocks"],
        },
        "correctness": {
            "complete": True,
            "row_count": validation["rows"],
            "payload_bytes": validation["payload_bytes"],
            "payload_digest_sha256": validation["payload_digest_sha256"],
            "payload_digest_algorithm": "historical_image_only_query_identity_plus_payload_sha256_v1",
            "whole_output_concatenated": validation["whole_output_concatenated"],
            "query_file_pin_matches": True,
            "harness_logical_pin_matches": True,
            "stable_id_sequence_pin_matches": True,
            "row_and_unique_count_pins_match": True,
            "spool_schema_count_position_and_stable_ids_match": True,
            "payload_digest_and_bytes_match_oracle": True,
        },
        "process": {
            "peak_rss": dict(rss),
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "packages": _package_versions(),
            "code": code_modules,
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("LANCE_CPU_THREADS", "LANCE_IO_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS")
            },
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:  # noqa: PLR0915
    """Execute one local-process canary; no scheduler or array integration exists here."""

    _validate_invocation(args)
    input_started = time.perf_counter()
    storage_options, storage_identity = _load_storage_options(args.storage_options_file)
    contract = load_query_contract(
        args.query_parquet,
        source_ref_column=args.source_ref_column,
        stable_id_column=args.stable_id_column,
        expected_file_sha256=args.query_file_sha256,
        expected_harness_logical_digest_sha256=args.harness_logical_digest_sha256,
        expected_stable_id_sequence_digest_sha256=args.stable_id_sequence_digest_sha256,
        expected_rows=args.expected_rows,
        expected_unique_stable_ids=args.expected_unique_stable_ids,
    )
    plans = build_coordinate_plans(contract)
    input_validation_seconds = time.perf_counter() - input_started

    import lance
    from lance_ray import LanceStableIdPayloadConfig, LanceStableIdPayloadStreamer

    from nemo_curator.stages.interleaved.gpu_key_lookup import _stable_global_ordinal_manifest_sha256
    from nemo_curator.stages.interleaved.lance import _validate_stable_global_ordinal_manifest
    from nemo_curator.stages.interleaved.lance_payload_materialize import (
        materialize_lance_payload_group_to_spools,
    )
    from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool

    spools: list[PayloadSpool] = []
    streamer: LanceStableIdPayloadStreamer | None = None
    success = False
    outer_started = time.perf_counter()
    rss_before = _peak_rss_bytes()
    try:
        session = lance.Session(metadata_cache_size_bytes=args.metadata_cache_mib * _MIB)
        dataset = lance.dataset(
            args.image_lance_uri,
            version=args.image_lance_version,
            storage_options=storage_options or None,
            session=session,
        )
        if dataset.version != args.image_lance_version:
            msg = f"image dataset resolved version {dataset.version}; expected {args.image_lance_version}"
            raise RuntimeError(msg)
        if not dataset.has_stable_row_ids:
            msg = "image dataset must have stable row IDs"
            raise ValueError(msg)
        stable_manifest = _validate_stable_global_ordinal_manifest(dataset)
        fragment_manifest_sha256 = _stable_global_ordinal_manifest_sha256(
            args.image_lance_uri,
            args.image_lance_version,
            stable_manifest,
        )
        if fragment_manifest_sha256 != args.image_fragment_manifest_sha256:
            msg = "image fragment-order manifest does not match the caller pin"
            raise ValueError(msg)
        maximum_stable_id = int(pc.max(contract.table[contract.stable_id_column]).as_py())
        if maximum_stable_id >= stable_manifest.total_rows:
            msg = "query stable IDs exceed the pinned image dataset row range"
            raise ValueError(msg)
        image_field = dataset.schema.field(args.image_column)
        if not (pa.types.is_binary(image_field.type) or pa.types.is_large_binary(image_field.type)):
            msg = "image payload column must be binary or large_binary"
            raise TypeError(msg)

        spool_schema = pa.schema([*plans[0].schema, image_field])
        args.spool_root.mkdir(parents=True, exist_ok=False)
        for index in range(BLOCK_COUNT):
            spools.append(
                PayloadSpool(
                    args.spool_root / f"block-{index:02d}",
                    spool_schema,
                    target_bytes=SHARED_SPOOL_BUDGET_BYTES,
                    bucket_rows=SPOOL_BUCKET_ROWS,
                    stable_id_column=STABLE_ROW_ID,
                    document_position_column=DOCUMENT_POSITION,
                    sync_mode=SPOOL_SYNC_MODE,
                )
            )
        streamer = LanceStableIdPayloadStreamer(
            LanceStableIdPayloadConfig(
                dataset_uri=args.image_lance_uri,
                dataset_version=args.image_lance_version,
                expected_rows=stable_manifest.total_rows,
                columns={args.image_column: args.image_column},
                dataset_storage_options=storage_options,
                fetch_batch_size=args.fetch_batch_size,
                io_threads=args.io_threads,
                max_pending_fetch_batches=args.max_pending,
                metadata_cache_size_bytes=args.metadata_cache_mib * _MIB,
            ),
            dataset=dataset,
            stable_row_id_output_column=STABLE_ROW_ID,
        )
        setup_seconds = time.perf_counter() - outer_started
        dataset.io_stats_incremental()
        materialize_started = time.perf_counter()
        grouped = materialize_lance_payload_group_to_spools(
            streamer,
            plans,
            (args.image_column,),
            spools,
            shared_spool_budget_bytes=SHARED_SPOOL_BUDGET_BYTES,
            max_coordinate_workspace_bytes=COORDINATE_WINDOW_BYTES,
        )
        materialize_seconds = time.perf_counter() - materialize_started
        validation_started = time.perf_counter()
        validation = validate_spools_and_payload_oracle(
            spools,
            contract,
            plans,
            expected_schema=spool_schema,
            image_column=args.image_column,
            expected_payload_digest_sha256=args.expected_payload_digest_sha256,
            expected_payload_bytes=args.expected_payload_bytes,
        )
        validation_seconds = time.perf_counter() - validation_started
        outer_seconds = time.perf_counter() - outer_started
        rss_after = _peak_rss_bytes()
        timing = {
            "input_validation_seconds": input_validation_seconds,
            "setup_seconds": setup_seconds,
            "materialize_seconds": materialize_seconds,
            "validation_seconds": validation_seconds,
            "outer_seconds": outer_seconds,
            "outer_scope": "dataset_and_streamer_setup_through_streamed_payload_validation",
            "materialize_scope": "one_materialize_lance_payload_group_to_spools_call",
        }
        plan_metrics = [metrics.as_dict() for metrics in grouped.plan_metrics]
        dataset_identity = {
            "uri": args.image_lance_uri,
            "version": args.image_lance_version,
            "fragment_manifest_sha256": fragment_manifest_sha256,
            "rows": stable_manifest.total_rows,
            "fragments": len(stable_manifest.fragment_rows),
            "stable_row_ids": True,
            "projection": [args.image_column],
            "image_field": str(image_field),
        }
        success = True
    finally:
        try:
            if streamer is not None:
                streamer.close()
        finally:
            should_cleanup = args.cleanup_policy == "always" or (args.cleanup_policy == "on-success" and success)
            cleanup_started = time.perf_counter()
            if should_cleanup:
                _cleanup_spools(spools, args.spool_root)
            cleanup_seconds = time.perf_counter() - cleanup_started

    timing["cleanup_seconds"] = cleanup_seconds
    retained = args.cleanup_policy == "never"
    result = _result_document(
        args=args,
        contract=contract,
        storage_identity=storage_identity,
        dataset_identity=dataset_identity,
        group_metrics=grouped.fetch_metrics,
        plan_metrics=plan_metrics,
        validation=validation,
        timing=timing,
        rss={
            "before_outer_bytes": rss_before,
            "after_validation_bytes": rss_after,
            "increase_bytes": max(0, rss_after - rss_before),
            "scope": "process_lifetime_high_water_mark",
        },
        retained=retained,
    )
    _atomic_json(args.result, result)
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
