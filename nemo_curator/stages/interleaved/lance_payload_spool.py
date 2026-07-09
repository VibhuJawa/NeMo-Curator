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

"""Bounded, ephemeral Arrow IPC payload spooling for node-local storage.

This module is deliberately not a checkpoint format.  A spool is owned by one
runtime attempt, validated while it is read, and removed only by an explicit
``cleanup`` call.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from collections.abc import Iterator

_FORMAT_NAME: Final = "nemo_curator_lance_payload_spool"
_FORMAT_VERSION: Final = 1
_MANIFEST_NAME: Final = "manifest.json"
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_SHA256_HEX_LENGTH: Final = 64


@dataclass(frozen=True)
class PayloadSpoolFile:
    """One immutable Arrow IPC part in deterministic bucket order."""

    bucket: int
    part: int
    path: Path
    rows: int
    arrow_nbytes: int
    file_bytes: int
    sha256: str
    oversized: bool


@dataclass(frozen=True)
class OversizedPayloadRow:
    """A single row that was allowed to exceed the configured byte target."""

    stable_id: int
    document_position: int
    arrow_nbytes: int
    path: Path


@dataclass(frozen=True)
class PayloadSpoolManifest:
    """Immutable summary returned by :meth:`PayloadSpool.finish`."""

    root: Path
    path: Path
    schema: pa.Schema
    target_bytes: int
    bucket_rows: int
    stable_id_column: str
    document_position_column: str
    total_rows: int
    total_arrow_nbytes: int
    peak_active_bytes: int
    peak_bounded_active_bytes: int
    files: tuple[PayloadSpoolFile, ...]
    oversized_rows: tuple[OversizedPayloadRow, ...]
    sha256: str


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{name} must be a nonnegative integer"
        raise ValueError(msg)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_bytes(schema: pa.Schema) -> bytes:
    return schema.serialize().to_pybytes()


def _schema_from_text(value: object) -> pa.Schema:
    if not isinstance(value, str):
        msg = "payload spool schema must be base64 text"
        raise TypeError(msg)
    try:
        encoded = base64.b64decode(value, validate=True)
        return pa.ipc.read_schema(pa.BufferReader(encoded))
    except (ValueError, pa.ArrowException) as exc:
        msg = "payload spool schema is invalid"
        raise ValueError(msg) from exc


def _sync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists():
        msg = f"refusing to replace existing payload spool file: {path.name}"
        raise FileExistsError(msg)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_arrow_file(path: Path, table: pa.Table) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        msg = f"refusing to replace existing payload spool file: {path.name}"
        raise FileExistsError(msg)
    try:
        with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        _sync_file(temporary)
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _file_name(bucket: int, part: int) -> str:
    return f"bucket-{bucket:020d}-part-{part:08d}.arrow"


def _slice_table(table: pa.Table, offset: int, length: int | None = None) -> pa.Table:
    """Return one zero-copy table slice through an instrumentable boundary."""

    return table.slice(offset, length)


def _iter_bucket_runs(table: pa.Table, bucket_ids: pa.ChunkedArray) -> Iterator[tuple[int, pa.Table]]:
    """Yield zero-copy table slices for contiguous runs of equal bucket IDs."""

    encoded = pc.run_end_encode(bucket_ids, run_end_type=pa.int64())
    chunk_offset = 0
    for chunk in encoded.chunks:
        run_start = 0
        for run_end, raw_bucket in zip(chunk.run_ends, chunk.values, strict=True):
            run_stop = int(run_end.as_py())
            yield (
                int(raw_bucket.as_py()),
                _slice_table(
                    table,
                    chunk_offset + run_start,
                    run_stop - run_start,
                ),
            )
            run_start = run_stop
        chunk_offset += len(chunk)
    if chunk_offset != table.num_rows:
        msg = "payload spool bucket encoding did not conserve rows"
        raise RuntimeError(msg)


class PayloadSpool:
    """Accumulate payload rows under an actual-Arrow-byte memory target.

    ``schema`` must include non-null ``uint64`` stable-ID and document-position
    columns.  Document positions are assigned to deterministic coarse buckets
    with ``position // bucket_rows``.  Normal buffered tables never exceed
    ``target_bytes`` in aggregate.  A row larger than the target is flushed by
    itself and reported explicitly.
    """

    def __init__(  # noqa: PLR0913
        self,
        root: str | Path,
        schema: pa.Schema,
        target_bytes: int,
        bucket_rows: int,
        *,
        stable_id_column: str = "stable_id",
        document_position_column: str = "document_position",
    ) -> None:
        self.root = Path(root)
        if self.root.exists() or self.root.is_symlink():
            msg = f"payload spool root already exists: {self.root}"
            raise FileExistsError(msg)
        if not isinstance(schema, pa.Schema):
            msg = "schema must be a pyarrow.Schema"
            raise TypeError(msg)
        self.schema = schema
        self.target_bytes = _require_positive_integer("target_bytes", target_bytes)
        self.bucket_rows = _require_positive_integer("bucket_rows", bucket_rows)
        self.stable_id_column = stable_id_column
        self.document_position_column = document_position_column
        if (
            not isinstance(stable_id_column, str)
            or not isinstance(document_position_column, str)
            or not stable_id_column
            or not document_position_column
            or stable_id_column == document_position_column
        ):
            msg = "stable_id_column and document_position_column must be distinct non-empty names"
            raise ValueError(msg)
        self._validate_coordinate_schema()

        self.root.mkdir(parents=True, exist_ok=False)
        self._buffers: dict[int, list[pa.Table]] = defaultdict(list)
        self._next_part: dict[int, int] = defaultdict(int)
        self._files: list[PayloadSpoolFile] = []
        self._oversized_rows: list[OversizedPayloadRow] = []
        self._active_bytes = 0
        self._peak_active_bytes = 0
        self._peak_bounded_active_bytes = 0
        self._appended_rows = 0
        self._finished: PayloadSpoolManifest | None = None
        self._cleaned = False

    def _validate_coordinate_schema(self) -> None:
        for name in (self.stable_id_column, self.document_position_column):
            index = self.schema.get_field_index(name)
            if index < 0:
                msg = f"payload spool schema is missing coordinate column {name!r}"
                raise ValueError(msg)
            field = self.schema.field(index)
            if field.type != pa.uint64() or field.nullable:
                msg = f"payload spool coordinate column {name!r} must be non-nullable uint64"
                raise TypeError(msg)

    def _require_open(self) -> None:
        if self._cleaned:
            msg = "payload spool has been cleaned"
            raise RuntimeError(msg)
        if self._finished is not None:
            msg = "payload spool has already been finished"
            raise RuntimeError(msg)

    def append(self, table: pa.Table) -> None:
        """Append a full-schema Arrow table without converting payload values."""

        self._require_open()
        if not isinstance(table, pa.Table):
            msg = "payload spool append requires a pyarrow.Table"
            raise TypeError(msg)
        if not table.schema.equals(self.schema, check_metadata=True):
            msg = "payload spool append schema does not match the configured schema"
            raise TypeError(msg)
        stable_ids = table[self.stable_id_column]
        document_positions = table[self.document_position_column]
        if stable_ids.null_count or document_positions.null_count:
            msg = "payload spool coordinate columns must not contain nulls"
            raise ValueError(msg)
        if table.num_rows == 0:
            return

        bucket_ids = pc.divide_checked(document_positions, pa.scalar(self.bucket_rows, type=pa.uint64()))
        for bucket, run in _iter_bucket_runs(table, bucket_ids):
            self._add_materialized(bucket, run)
        self._appended_rows += table.num_rows

    def _add_materialized(self, bucket: int, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if table.nbytes > self.target_bytes and table.num_rows > 1:
            midpoint = table.num_rows // 2
            left = _slice_table(table, 0, midpoint)
            right = _slice_table(table, midpoint)
            self._add_materialized(bucket, left)
            self._add_materialized(bucket, right)
            return
        if table.nbytes > self.target_bytes:
            if self._active_bytes:
                self._flush_active()
            self._peak_active_bytes = max(self._peak_active_bytes, table.nbytes)
            file = self._write_table(bucket, table, oversized=True)
            self._oversized_rows.append(
                OversizedPayloadRow(
                    stable_id=int(table[self.stable_id_column][0].as_py()),
                    document_position=int(table[self.document_position_column][0].as_py()),
                    arrow_nbytes=table.nbytes,
                    path=file.path,
                )
            )
            return
        if self._active_bytes + table.nbytes > self.target_bytes:
            self._flush_active()
        self._buffers[bucket].append(table)
        self._active_bytes += table.nbytes
        self._peak_active_bytes = max(self._peak_active_bytes, self._active_bytes)
        self._peak_bounded_active_bytes = max(self._peak_bounded_active_bytes, self._active_bytes)
        if self._active_bytes == self.target_bytes:
            self._flush_active()

    def _write_table(self, bucket: int, table: pa.Table, *, oversized: bool) -> PayloadSpoolFile:
        part = self._next_part[bucket]
        self._next_part[bucket] += 1
        path = self.root / _file_name(bucket, part)
        _atomic_arrow_file(path, table)
        record = PayloadSpoolFile(
            bucket=bucket,
            part=part,
            path=path,
            rows=table.num_rows,
            arrow_nbytes=table.nbytes,
            file_bytes=path.stat().st_size,
            sha256=_file_sha256(path),
            oversized=oversized,
        )
        self._files.append(record)
        return record

    def _flush_active(self) -> None:
        for bucket in sorted(self._buffers):
            tables = self._buffers[bucket]
            if not tables:
                continue
            table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
            self._write_table(bucket, table, oversized=False)
        self._buffers.clear()
        self._active_bytes = 0

    def finish(self) -> PayloadSpoolManifest:
        """Atomically publish the manifest after flushing all retained rows."""

        if self._cleaned:
            msg = "payload spool has been cleaned"
            raise RuntimeError(msg)
        if self._finished is not None:
            return self._finished
        self._flush_active()
        files = tuple(sorted(self._files, key=lambda item: (item.bucket, item.part)))
        written_rows = sum(item.rows for item in files)
        if written_rows != self._appended_rows:
            msg = f"payload spool row conservation failed: appended={self._appended_rows}, written={written_rows}"
            raise RuntimeError(msg)
        total_arrow_nbytes = sum(item.arrow_nbytes for item in files)
        payload = {
            "format": _FORMAT_NAME,
            "version": _FORMAT_VERSION,
            "schema": base64.b64encode(_schema_bytes(self.schema)).decode("ascii"),
            "target_bytes": self.target_bytes,
            "bucket_rows": self.bucket_rows,
            "stable_id_column": self.stable_id_column,
            "document_position_column": self.document_position_column,
            "total_rows": written_rows,
            "total_arrow_nbytes": total_arrow_nbytes,
            "peak_active_bytes": self._peak_active_bytes,
            "peak_bounded_active_bytes": self._peak_bounded_active_bytes,
            "files": [
                {
                    "bucket": item.bucket,
                    "part": item.part,
                    "path": item.path.name,
                    "rows": item.rows,
                    "arrow_nbytes": item.arrow_nbytes,
                    "file_bytes": item.file_bytes,
                    "sha256": item.sha256,
                    "oversized": item.oversized,
                }
                for item in files
            ],
            "oversized_rows": [
                {
                    "stable_id": item.stable_id,
                    "document_position": item.document_position,
                    "arrow_nbytes": item.arrow_nbytes,
                    "path": item.path.name,
                }
                for item in sorted(self._oversized_rows, key=lambda item: item.path.name)
            ],
        }
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        manifest_path = self.root / _MANIFEST_NAME
        _atomic_bytes(manifest_path, content)
        self._finished = PayloadSpoolManifest(
            root=self.root,
            path=manifest_path,
            schema=self.schema,
            target_bytes=self.target_bytes,
            bucket_rows=self.bucket_rows,
            stable_id_column=self.stable_id_column,
            document_position_column=self.document_position_column,
            total_rows=written_rows,
            total_arrow_nbytes=total_arrow_nbytes,
            peak_active_bytes=self._peak_active_bytes,
            peak_bounded_active_bytes=self._peak_bounded_active_bytes,
            files=files,
            oversized_rows=tuple(sorted(self._oversized_rows, key=lambda item: item.path.name)),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        return self._finished

    def iter_tables(self) -> Iterator[pa.Table]:
        """Read validated IPC parts after :meth:`finish`."""

        if self._finished is None:
            msg = "finish must be called before reading the payload spool"
            raise RuntimeError(msg)
        return PayloadSpoolReader(self._finished).iter_tables()

    def read_all(self) -> pa.Table:
        """Read and validate all rows in deterministic bucket/file order."""

        if self._finished is None:
            msg = "finish must be called before reading the payload spool"
            raise RuntimeError(msg)
        return PayloadSpoolReader(self._finished).read_all()

    def cleanup(self) -> None:
        """Explicitly remove this attempt-local spool and all of its files."""

        if self._cleaned:
            return
        shutil.rmtree(self.root, ignore_errors=False)
        self._cleaned = True


class PayloadSpoolReader:
    """Validate and stream one finished ephemeral payload spool."""

    def __init__(self, manifest: PayloadSpoolManifest | str | Path) -> None:
        expected_sha256: str | None = None
        if isinstance(manifest, PayloadSpoolManifest):
            manifest_path = manifest.path
            expected_sha256 = manifest.sha256
        else:
            manifest_path = Path(manifest)
            if manifest_path.is_dir():
                manifest_path /= _MANIFEST_NAME
        self._manifest_path = manifest_path
        self.root = manifest_path.parent
        content = manifest_path.read_bytes()
        actual_manifest_sha256 = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and actual_manifest_sha256 != expected_sha256:
            msg = "payload spool manifest SHA-256 mismatch"
            raise ValueError(msg)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            msg = "payload spool manifest is not valid JSON"
            raise ValueError(msg) from exc
        self.manifest = self._parse_manifest(payload, actual_manifest_sha256)
        self._cleaned = False

    def _parse_manifest(  # noqa: C901, PLR0912, PLR0915
        self,
        payload: object,
        sha256: str,
    ) -> PayloadSpoolManifest:
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT_NAME:
            msg = "payload spool manifest format is invalid"
            raise ValueError(msg)
        if payload.get("version") != _FORMAT_VERSION:
            msg = "payload spool manifest version is unsupported"
            raise ValueError(msg)
        schema = _schema_from_text(payload.get("schema"))
        target_bytes = _require_positive_integer("manifest target_bytes", payload.get("target_bytes"))
        bucket_rows = _require_positive_integer("manifest bucket_rows", payload.get("bucket_rows"))
        stable_id_column = payload.get("stable_id_column")
        document_position_column = payload.get("document_position_column")
        if not isinstance(stable_id_column, str) or not isinstance(document_position_column, str):
            msg = "payload spool manifest coordinate columns are invalid"
            raise TypeError(msg)
        if not stable_id_column or not document_position_column or stable_id_column == document_position_column:
            msg = "payload spool manifest coordinate columns are invalid"
            raise ValueError(msg)
        for name in (stable_id_column, document_position_column):
            index = schema.get_field_index(name)
            if index < 0 or schema.field(index).type != pa.uint64() or schema.field(index).nullable:
                msg = f"payload spool manifest coordinate column {name!r} is invalid"
                raise ValueError(msg)

        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            msg = "payload spool manifest files must be a list"
            raise TypeError(msg)
        files: list[PayloadSpoolFile] = []
        for raw in raw_files:
            if not isinstance(raw, dict):
                msg = "payload spool file record is invalid"
                raise TypeError(msg)
            bucket = _require_nonnegative_integer("file bucket", raw.get("bucket"))
            part = _require_nonnegative_integer("file part", raw.get("part"))
            relative_path = raw.get("path")
            if not isinstance(relative_path, str) or Path(relative_path).name != relative_path:
                msg = "payload spool file path must be one relative file name"
                raise ValueError(msg)
            if relative_path != _file_name(bucket, part):
                msg = "payload spool file name does not match its bucket and part"
                raise ValueError(msg)
            digest = raw.get("sha256")
            if not isinstance(digest, str) or len(digest) != _SHA256_HEX_LENGTH:
                msg = "payload spool file SHA-256 is invalid"
                raise ValueError(msg)
            try:
                bytes.fromhex(digest)
            except ValueError as exc:
                msg = "payload spool file SHA-256 is invalid"
                raise ValueError(msg) from exc
            oversized = raw.get("oversized")
            if not isinstance(oversized, bool):
                msg = "payload spool file oversized flag is invalid"
                raise TypeError(msg)
            record = PayloadSpoolFile(
                bucket=bucket,
                part=part,
                path=self.root / relative_path,
                rows=_require_positive_integer("file rows", raw.get("rows")),
                arrow_nbytes=_require_positive_integer("file arrow_nbytes", raw.get("arrow_nbytes")),
                file_bytes=_require_positive_integer("file file_bytes", raw.get("file_bytes")),
                sha256=digest,
                oversized=oversized,
            )
            if (record.arrow_nbytes > target_bytes) != record.oversized:
                msg = "payload spool oversized file accounting is inconsistent"
                raise ValueError(msg)
            if record.oversized and record.rows != 1:
                msg = "an oversized payload spool file must contain exactly one row"
                raise ValueError(msg)
            files.append(record)

        sorted_files = sorted(files, key=lambda item: (item.bucket, item.part))
        if files != sorted_files:
            msg = "payload spool file records are not in deterministic bucket/file order"
            raise ValueError(msg)
        seen_paths: set[Path] = set()
        expected_parts: dict[int, int] = defaultdict(int)
        for record in files:
            if record.path in seen_paths:
                msg = "payload spool manifest contains a duplicate file"
                raise ValueError(msg)
            seen_paths.add(record.path)
            if record.part != expected_parts[record.bucket]:
                msg = "payload spool file parts are not contiguous within a bucket"
                raise ValueError(msg)
            expected_parts[record.bucket] += 1

        total_rows = _require_nonnegative_integer("manifest total_rows", payload.get("total_rows"))
        total_arrow_nbytes = _require_nonnegative_integer(
            "manifest total_arrow_nbytes", payload.get("total_arrow_nbytes")
        )
        if total_rows != sum(record.rows for record in files):
            msg = "payload spool manifest row conservation failed"
            raise ValueError(msg)
        if total_arrow_nbytes != sum(record.arrow_nbytes for record in files):
            msg = "payload spool manifest Arrow-byte conservation failed"
            raise ValueError(msg)
        peak_active_bytes = _require_nonnegative_integer(
            "manifest peak_active_bytes", payload.get("peak_active_bytes")
        )
        peak_bounded_active_bytes = _require_nonnegative_integer(
            "manifest peak_bounded_active_bytes", payload.get("peak_bounded_active_bytes")
        )
        if peak_bounded_active_bytes > target_bytes or peak_bounded_active_bytes > peak_active_bytes:
            msg = "payload spool active-byte accounting is inconsistent"
            raise ValueError(msg)

        raw_oversized = payload.get("oversized_rows")
        if not isinstance(raw_oversized, list):
            msg = "payload spool oversized_rows must be a list"
            raise TypeError(msg)
        oversized_rows: list[OversizedPayloadRow] = []
        oversized_files = {record.path for record in files if record.oversized}
        for raw in raw_oversized:
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                msg = "payload spool oversized row record is invalid"
                raise TypeError(msg)
            path = self.root / raw["path"]
            oversized_rows.append(
                OversizedPayloadRow(
                    stable_id=_require_nonnegative_integer("oversized stable_id", raw.get("stable_id")),
                    document_position=_require_nonnegative_integer(
                        "oversized document_position", raw.get("document_position")
                    ),
                    arrow_nbytes=_require_positive_integer("oversized arrow_nbytes", raw.get("arrow_nbytes")),
                    path=path,
                )
            )
        if {item.path for item in oversized_rows} != oversized_files or len(oversized_rows) != len(oversized_files):
            msg = "payload spool oversized row records do not match oversized files"
            raise ValueError(msg)
        files_by_path = {record.path: record for record in files}
        if any(item.arrow_nbytes != files_by_path[item.path].arrow_nbytes for item in oversized_rows):
            msg = "payload spool oversized row byte accounting is inconsistent"
            raise ValueError(msg)
        expected_peak = max(
            peak_bounded_active_bytes,
            max((item.arrow_nbytes for item in oversized_rows), default=0),
        )
        if peak_active_bytes != expected_peak:
            msg = "payload spool peak active-byte accounting is inconsistent"
            raise ValueError(msg)

        return PayloadSpoolManifest(
            root=self.root,
            path=self._manifest_path,
            schema=schema,
            target_bytes=target_bytes,
            bucket_rows=bucket_rows,
            stable_id_column=stable_id_column,
            document_position_column=document_position_column,
            total_rows=total_rows,
            total_arrow_nbytes=total_arrow_nbytes,
            peak_active_bytes=peak_active_bytes,
            peak_bounded_active_bytes=peak_bounded_active_bytes,
            files=tuple(files),
            oversized_rows=tuple(oversized_rows),
            sha256=sha256,
        )

    def iter_tables(self) -> Iterator[pa.Table]:
        """Yield fully validated tables in deterministic bucket/file order."""

        if self._cleaned:
            msg = "payload spool has been cleaned"
            raise RuntimeError(msg)
        rows_read = 0
        oversized_by_path = {item.path: item for item in self.manifest.oversized_rows}
        for record in self.manifest.files:
            if record.path.stat().st_size != record.file_bytes:
                msg = f"payload spool file size mismatch: {record.path.name}"
                raise ValueError(msg)
            if _file_sha256(record.path) != record.sha256:
                msg = f"payload spool file SHA-256 mismatch: {record.path.name}"
                raise ValueError(msg)
            with pa.memory_map(str(record.path), "r") as source:
                reader = pa.ipc.open_file(source)
                if not reader.schema.equals(self.manifest.schema, check_metadata=True):
                    msg = f"payload spool file schema mismatch: {record.path.name}"
                    raise TypeError(msg)
                table = reader.read_all()
                self._validate_table(record, table, oversized_by_path)
                rows_read += table.num_rows
                yield table
        if rows_read != self.manifest.total_rows:
            msg = "payload spool reader conservation failed"
            raise ValueError(msg)

    def _validate_table(
        self,
        record: PayloadSpoolFile,
        table: pa.Table,
        oversized_by_path: dict[Path, OversizedPayloadRow],
    ) -> None:
        if table.num_rows != record.rows:
            msg = f"payload spool file row-count mismatch: {record.path.name}"
            raise ValueError(msg)
        stable_ids = table[self.manifest.stable_id_column]
        document_positions = table[self.manifest.document_position_column]
        if stable_ids.null_count or document_positions.null_count:
            msg = f"payload spool file contains null coordinates: {record.path.name}"
            raise ValueError(msg)
        buckets = pc.divide_checked(
            document_positions,
            pa.scalar(self.manifest.bucket_rows, type=pa.uint64()),
        )
        in_bucket = pc.all(pc.equal(buckets, pa.scalar(record.bucket, type=pa.uint64()))).as_py()
        if not in_bucket:
            msg = f"payload spool file contains rows from another bucket: {record.path.name}"
            raise ValueError(msg)
        if record.oversized:
            expected = oversized_by_path[record.path]
            if (
                int(stable_ids[0].as_py()) != expected.stable_id
                or int(document_positions[0].as_py()) != expected.document_position
            ):
                msg = f"payload spool oversized row metadata mismatch: {record.path.name}"
                raise ValueError(msg)

    def read_all(self) -> pa.Table:
        """Materialize all validated parts in deterministic file order."""

        tables = list(self.iter_tables())
        if not tables:
            return pa.Table.from_batches([], schema=self.manifest.schema)
        return pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    def cleanup(self) -> None:
        """Explicitly remove the validated ephemeral spool directory."""

        if self._cleaned:
            return
        shutil.rmtree(self.root, ignore_errors=False)
        self._cleaned = True
