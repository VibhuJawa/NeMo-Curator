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

from __future__ import annotations

import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.lance import LANCE_FRAGID_COLUMN, LANCE_ROWADDR_COLUMN
from nemo_curator.tasks import DocumentBatch
from nemo_curator.tasks.tasks import Task

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_T = TypeVar("_T")
_RESERVED_LANCE_PREFIX = "__lance_"


def _retry_with_backoff(fn: Callable[[], _T], *, retries: int, label: str) -> _T:
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            logger.warning(f"{label}: attempt {attempt + 1}/{retries} failed, retrying in {wait}s: {exc}")
            time.sleep(wait)
    msg = "unreachable"
    raise RuntimeError(msg)  # pragma: no cover


def _schema_to_base64(schema: pa.Schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def _schema_from_base64(value: str) -> pa.Schema:
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value)))


def _commit_result_version(result: object) -> int:
    if hasattr(result, "version"):
        return int(result.version)  # type: ignore[attr-defined]
    return int(result)


def _dataset_kwargs(storage_options: dict[str, Any] | None = None, version: int | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    if version is not None:
        kwargs["version"] = version
    return kwargs


def _open_dataset_if_exists(path: str, storage_options: dict[str, Any] | None = None) -> object | None:
    import lance

    try:
        return lance.dataset(path, storage_options=storage_options)
    except (FileNotFoundError, ValueError):
        return None


def _checkpoint_fs(commit_path: str, storage_options: dict[str, Any] | None = None) -> tuple[object, str]:
    return url_to_fs(commit_path, **(storage_options or {}))


def _join(fs_path: str, *parts: str) -> str:
    return "/".join([fs_path.rstrip("/"), *(part.strip("/") for part in parts)])


def _write_json_record(
    commit_path: str,
    record: dict[str, Any],
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> str:
    fs, fs_path = _checkpoint_fs(commit_path, checkpoint_storage_options)
    records_dir = _join(fs_path, "records")
    fs.makedirs(records_dir, exist_ok=True)
    fragment_id = record.get("fragment_id", "na")
    filename = f"{record['kind']}-fragment-{fragment_id}-{uuid.uuid4().hex}.json"
    record_path = _join(records_dir, filename)
    with fs.open(record_path, "w") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return fs.unstrip_protocol(record_path)


def _read_json_records(
    commit_path: str,
    kind: str,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    fs, fs_path = _checkpoint_fs(commit_path, checkpoint_storage_options)
    record_paths = sorted(fs.glob(_join(fs_path, "records", "*.json")))
    records = []
    for record_path in record_paths:
        with fs.open(record_path) as stream:
            record = json.loads(stream.read())
        if record.get("kind") == kind:
            records.append(record)
    return records


def _marker_path(commit_path: str, name: str, checkpoint_storage_options: dict[str, Any] | None = None) -> tuple[Any, str]:
    fs, fs_path = _checkpoint_fs(commit_path, checkpoint_storage_options)
    return fs, _join(fs_path, name)


def _write_marker(
    commit_path: str,
    name: str,
    payload: dict[str, Any],
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> None:
    fs, path = _marker_path(commit_path, name, checkpoint_storage_options)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "w") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _read_committed_marker(
    commit_path: str,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int | None:
    fs, path = _marker_path(commit_path, "_COMMITTED", checkpoint_storage_options)
    if not fs.exists(path):
        return None
    with fs.open(path) as stream:
        payload = json.loads(stream.read())
    version = payload.get("version")
    return int(version) if version is not None else None


def _unique(values: Iterable[object], label: str) -> object:
    unique_values = set(values)
    if len(unique_values) != 1:
        msg = f"Expected one {label}; got {sorted(unique_values)}"
        raise ValueError(msg)
    return next(iter(unique_values))


def _json_fingerprint(value: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _drop_reserved_lance_columns(table: pa.Table) -> pa.Table:
    columns = [name for name in table.column_names if not name.startswith(_RESERVED_LANCE_PREFIX)]
    return table.select(columns)


@dataclass
class LanceCheckpointTask(Task[list[str]]):
    """Task carrying paths to durable Lance checkpoint records."""

    data: list[str] = field(default_factory=list)
    kind: str = ""

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


@dataclass
class LanceWriter(ProcessingStage[DocumentBatch, LanceCheckpointTask]):
    """Write ``DocumentBatch`` objects to uncommitted Lance fragments.

    Workers write fragments with ``lance_ray.fragment.write_fragment`` and then
    persist small JSON commit records under ``commit_path``.  Use
    :func:`commit_lance_checkpoint` from any driver process to commit the
    checkpointed fragments into the Lance dataset.
    """

    path: str
    commit_path: str
    schema: pa.Schema | None = None
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    fields: list[str] | None = None
    name: str = "lance_writer"
    mode: Literal["ignore", "overwrite", "append", "error"] = "ignore"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self.storage_options = (self.write_kwargs or {}).get("storage_options")
        self.checkpoint_storage_options = (self.write_kwargs or {}).get("checkpoint_storage_options")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _output_table(self, task: DocumentBatch) -> pa.Table:
        table = task.to_pyarrow()
        if self.fields is not None:
            table = table.select(self.fields)
        elif self.schema is None:
            table = _drop_reserved_lance_columns(table)
        if self.schema is not None:
            table = table.select(self.schema.names)
        return table

    def process(self, task: DocumentBatch) -> LanceCheckpointTask:
        from lance_ray.fragment import write_fragment

        write_kwargs = dict(self.write_kwargs)
        write_kwargs.pop("checkpoint_storage_options", None)
        write_kwargs.setdefault("max_rows_per_file", 500_000)
        table = self._output_table(task)
        results = write_fragment(
            [table],
            self.path,
            schema=self.schema,
            **write_kwargs,
        )
        if not results:
            msg = f"write_fragment returned no fragments for task {task.task_id}"
            raise RuntimeError(msg)

        record_paths = []
        for fragment, schema in results:
            record = {
                "schema_version": 1,
                "kind": "lance_write",
                "dataset_path": self.path,
                "mode": self.mode,
                "schema": _schema_to_base64(schema),
                "fragment_id": fragment.id,
                "fragment": fragment.to_json(),
                "row_count": fragment.num_rows,
                "task_id": task.task_id,
            }
            record_paths.append(_write_json_record(self.commit_path, record, self.checkpoint_storage_options))

        return LanceCheckpointTask(
            dataset_name=task.dataset_name,
            data=record_paths,
            kind="lance_write",
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class LanceAnnotationWriter(ProcessingStage[DocumentBatch, LanceCheckpointTask]):
    """Sparse-update annotation columns in an existing Lance dataset.

    The stage expects input Arrow tables to contain ``__lance_rowaddr`` and
    ``__lance_fragid`` columns produced by ``LanceReader``.  Workers update each
    owned fragment with ``fragment.update_columns`` and persist JSON commit
    records under ``commit_path``.  Use
    :func:`commit_lance_annotation_checkpoint` to commit the update.
    """

    path: str
    commit_path: str
    schema: pa.Schema
    create_columns: bool = False
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    fields: list[str] | None = None
    name: str = "lance_annotation_writer"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _prepared_version: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.storage_options = (self.write_kwargs or {}).get("storage_options")
        self.checkpoint_storage_options = (self.write_kwargs or {}).get("checkpoint_storage_options")
        if self.fields is None:
            self.fields = list(self.schema.names)
        missing = [field for field in self.fields if field not in self.schema.names]
        if missing:
            msg = f"fields must be present in schema; missing {missing}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def prepare(self) -> int:
        import lance

        dataset = lance.dataset(self.path, storage_options=self.storage_options)
        missing_fields = [field for field in self.schema if field.name not in dataset.schema.names]
        if missing_fields:
            if not self.create_columns:
                missing = [field.name for field in missing_fields]
                msg = f"Lance annotation columns do not exist: {missing}"
                raise ValueError(msg)
            dataset.add_columns(pa.schema(missing_fields))
            dataset = lance.dataset(self.path, storage_options=self.storage_options)
        self._validate_target_columns(dataset)
        self._prepared_version = dataset.version
        return dataset.version

    def _validate_target_columns(self, dataset: object) -> None:
        missing = [field for field in self.fields or [] if field not in dataset.schema.names]
        if missing:
            msg = (
                f"Lance annotation columns do not exist: {missing}. "
                "Call prepare() with create_columns=True before pipeline.run(), or create the columns manually."
            )
            raise ValueError(msg)

    def _update_version(self) -> int:
        import lance

        if self._prepared_version is not None:
            return self._prepared_version
        logger.warning(
            "LanceAnnotationWriter is running without prepare(). "
            "This is only safe when target columns already exist and the Lance dataset does not change during the run. "
            "To add missing columns or pin a version, call prepare() before pipeline.run()."
        )
        dataset = lance.dataset(self.path, storage_options=self.storage_options)
        self._validate_target_columns(dataset)
        return dataset.version

    def _update_table_for_fragment(self, table: pa.Table, fragment_id: int) -> pa.Table:
        fragids = table[LANCE_FRAGID_COLUMN].combine_chunks().to_pylist()
        mask = pa.array([int(value) == fragment_id for value in fragids], type=pa.bool_())
        fragment_table = table.filter(mask)
        columns = [LANCE_ROWADDR_COLUMN, *(self.fields or [])]
        update_table = fragment_table.select(columns).rename_columns(["_rowaddr", *(self.fields or [])])
        update_table = update_table.set_column(0, "_rowaddr", update_table["_rowaddr"].combine_chunks().cast(pa.uint64()))
        _validate_unique_rowaddrs(update_table, fragment_id)
        return update_table

    def process(self, task: DocumentBatch) -> LanceCheckpointTask:
        import lance

        table = task.to_pyarrow()
        _validate_annotation_input_table(table, self.fields or [])
        version = self._update_version()
        dataset = lance.dataset(self.path, **_dataset_kwargs(self.storage_options, version))

        record_paths = []
        fragment_ids = sorted({int(value) for value in table[LANCE_FRAGID_COLUMN].combine_chunks().to_pylist()})
        for fragment_id in fragment_ids:
            update_table = self._update_table_for_fragment(table, fragment_id)
            if update_table.num_rows == 0:
                continue
            fragment = dataset.get_fragment(fragment_id)
            updated_fragment, fields_modified = fragment.update_columns(
                update_table,
                left_on="_rowaddr",
                right_on="_rowaddr",
            )
            if not fields_modified:
                msg = f"update_columns returned no fields_modified for fragment {fragment_id}"
                raise RuntimeError(msg)
            record = {
                "schema_version": 1,
                "kind": "lance_annotation_update",
                "dataset_path": self.path,
                "dataset_version": version,
                "fragment_id": fragment_id,
                "fields_modified": sorted(set(fields_modified)),
                "columns": list(self.fields or []),
                "row_count": update_table.num_rows,
                "updated_fragment": updated_fragment.to_json(),
                "task_id": task.task_id,
            }
            record_paths.append(_write_json_record(self.commit_path, record, self.checkpoint_storage_options))

        return LanceCheckpointTask(
            dataset_name=task.dataset_name,
            data=record_paths,
            kind="lance_annotation_update",
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _validate_annotation_input_table(table: pa.Table, fields: list[str]) -> None:
    missing = [name for name in [LANCE_ROWADDR_COLUMN, LANCE_FRAGID_COLUMN, *fields] if name not in table.column_names]
    if missing:
        msg = f"Lance annotation update table is missing required columns: {missing}"
        raise ValueError(msg)


def _validate_unique_rowaddrs(table: pa.Table, fragment_id: int) -> None:
    rowaddrs = table["_rowaddr"].combine_chunks().to_pylist()
    seen = set()
    duplicates = []
    for rowaddr in rowaddrs:
        value = int(rowaddr)
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        sample = ", ".join(str(value) for value in duplicates[:5])
        msg = (
            f"Lance annotation update contains duplicate {LANCE_ROWADDR_COLUMN} values "
            f"for fragment {fragment_id}: {sample}"
        )
        raise ValueError(msg)


def commit_lance_checkpoint(  # noqa: PLR0913
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
    retries: int = 5,
    force: bool = False,
) -> int | None:
    """Commit checkpoint records produced by ``LanceWriter``."""

    import lance

    committed_version = _read_committed_marker(commit_path, checkpoint_storage_options)
    if committed_version is not None and not force:
        logger.warning(f"Lance checkpoint {commit_path} is already committed at version {committed_version}")
        return committed_version

    records = _read_json_records(commit_path, "lance_write", checkpoint_storage_options)
    if not records:
        msg = f"No lance_write checkpoint records found under {commit_path}"
        raise ValueError(msg)

    dataset_path = str(_unique((record["dataset_path"] for record in records), "dataset path"))
    if dataset_path != path:
        msg = f"Checkpoint records are for {dataset_path}, not {path}"
        raise ValueError(msg)
    mode = str(_unique((record["mode"] for record in records), "write mode"))
    schema = _schema_from_base64(str(_unique((record["schema"] for record in records), "schema")))
    fragments = [lance.FragmentMetadata.from_json(json.dumps(record["fragment"])) for record in records]

    def _commit() -> int | None:
        existing_dataset = _open_dataset_if_exists(path, storage_options)
        if mode in {"error", "create"} and existing_dataset is not None:
            msg = f"Lance dataset already exists at {path}"
            raise FileExistsError(msg)
        if mode == "ignore" and existing_dataset is not None:
            return existing_dataset.version
        if mode == "append":
            if existing_dataset is None:
                msg = f"Cannot append to missing Lance dataset at {path}"
                raise FileNotFoundError(msg)
            operation = lance.LanceOperation.Append(fragments)
            read_version = existing_dataset.version
        else:
            operation = lance.LanceOperation.Overwrite(schema, fragments)
            read_version = existing_dataset.version if existing_dataset is not None else None
        return _commit_result_version(
            lance.LanceDataset.commit(path, operation, read_version=read_version, storage_options=storage_options)
        )

    version = _retry_with_backoff(_commit, retries=retries, label="commit_lance_checkpoint")
    if version is not None:
        _write_marker(commit_path, "_COMMITTED", {"version": version}, checkpoint_storage_options)
    return version


def _annotation_records_by_fragment(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    fingerprints: dict[int, str] = {}
    selected: dict[int, dict[str, Any]] = {}
    for record in records:
        fragment_id = int(record["fragment_id"])
        fingerprint = _json_fingerprint(
            {
                "dataset_path": record["dataset_path"],
                "dataset_version": record["dataset_version"],
                "fields_modified": record["fields_modified"],
                "updated_fragment": record["updated_fragment"],
            }
        )
        if fragment_id in fingerprints and fingerprints[fragment_id] != fingerprint:
            msg = f"Conflicting Lance annotation checkpoint records for fragment {fragment_id}"
            raise ValueError(msg)
        fingerprints[fragment_id] = fingerprint
        selected[fragment_id] = record
    return selected


def commit_lance_annotation_checkpoint(  # noqa: PLR0913
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
    retries: int = 5,
    force: bool = False,
) -> int:
    """Commit checkpoint records produced by ``LanceAnnotationWriter``."""

    import lance

    committed_version = _read_committed_marker(commit_path, checkpoint_storage_options)
    if committed_version is not None and not force:
        logger.warning(f"Lance annotation checkpoint {commit_path} is already committed at version {committed_version}")
        return committed_version

    records = _read_json_records(commit_path, "lance_annotation_update", checkpoint_storage_options)
    if not records:
        msg = f"No lance_annotation_update checkpoint records found under {commit_path}"
        raise ValueError(msg)

    dataset_path = str(_unique((record["dataset_path"] for record in records), "dataset path"))
    if dataset_path != path:
        msg = f"Checkpoint records are for {dataset_path}, not {path}"
        raise ValueError(msg)
    read_version = int(_unique((record["dataset_version"] for record in records), "dataset version"))
    records_by_fragment = _annotation_records_by_fragment(records)
    updated_fragments = [
        lance.FragmentMetadata.from_json(json.dumps(record["updated_fragment"]))
        for record in records_by_fragment.values()
    ]
    fields_modified = sorted({field for record in records_by_fragment.values() for field in record["fields_modified"]})

    def _commit() -> int:
        operation = lance.LanceOperation.Update(updated_fragments=updated_fragments, fields_modified=fields_modified)
        return _commit_result_version(
            lance.LanceDataset.commit(path, operation, read_version=read_version, storage_options=storage_options)
        )

    version = _retry_with_backoff(_commit, retries=retries, label="commit_lance_annotation_checkpoint")
    _write_marker(commit_path, "_COMMITTED", {"version": version}, checkpoint_storage_options)
    return version


# Backwards-friendly alias for early prototypes.
LanceFragmentTask = LanceCheckpointTask
