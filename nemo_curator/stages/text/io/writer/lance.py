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

import json
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
from fsspec.core import url_to_fs
from loguru import logger
from ray._common.retry import call_with_retry

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.lance_utils import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    lance_dataset_kwargs,
    object_from_base64,
    object_to_base64,
    schema_from_json_value,
    schema_to_json_value,
)
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.file_utils import check_output_mode
from nemo_curator.utils.hash_utils import get_deterministic_hash

if TYPE_CHECKING:
    from collections.abc import Iterable

_RESERVED_LANCE_PREFIX = "__lance_"
_WRITE_FRAGMENT_KWARG_DENYLIST = {
    "checkpoint_storage_options",
    "source_storage_options",
    "base_store_params",
    "target_bases",
    "enable_stable_row_ids",
    "return_transaction",
}


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
    record_id: str = "",
) -> str:
    fs, fs_path = _checkpoint_fs(commit_path, checkpoint_storage_options)
    records_dir = _join(fs_path, "records")
    fs.makedirs(records_dir, exist_ok=True)
    filename = f"{record_id}.json"
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


def _marker_path(
    commit_path: str,
    name: str,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> tuple[Any, str]:
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
) -> dict[str, Any] | None:
    fs, path = _marker_path(commit_path, "_COMMITTED", checkpoint_storage_options)
    if not fs.exists(path):
        return None
    with fs.open(path) as stream:
        return json.loads(stream.read())


def _unique(values: Iterable[object], label: str) -> object:
    unique_values = set(values)
    if len(unique_values) != 1:
        msg = f"Expected one {label}; got {sorted(unique_values)}"
        raise ValueError(msg)
    return next(iter(unique_values))


def _load_checkpoint_records(
    commit_path: str,
    kind: str,
    checkpoint_storage_options: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], int | None]:
    marker = _read_committed_marker(commit_path, checkpoint_storage_options)
    if marker is not None:
        version = int(marker["version"])
        logger.warning(f"Lance checkpoint {commit_path} is already committed at version {version}")
        return [], version
    records = _read_json_records(commit_path, kind, checkpoint_storage_options)
    if not records:
        msg = f"No {kind} checkpoint records found under {commit_path}"
        raise ValueError(msg)
    return records, None


def _write_committed_marker(
    commit_path: str,
    version: int,
    checkpoint_storage_options: dict[str, Any] | None,
) -> None:
    _write_marker(commit_path, "_COMMITTED", {"version": version}, checkpoint_storage_options)


def _record_id(kind: str, *parts: object) -> str:
    values = [str(part) for part in parts if part not in {None, ""}]
    return f"{kind}-{get_deterministic_hash(values or [kind])}"


def _drop_reserved_lance_columns(table: pa.Table) -> pa.Table:
    columns = [name for name in table.column_names if not name.startswith(_RESERVED_LANCE_PREFIX)]
    return table.select(columns)


def _preflight_checkpoint_path(
    commit_path: str,
    checkpoint_storage_options: dict[str, Any] | None,
    checkpoint_mode: Literal["ignore", "overwrite", "error"],
) -> None:
    fs, fs_path = _checkpoint_fs(commit_path, checkpoint_storage_options)
    check_output_mode(checkpoint_mode, fs, fs_path)


def _metadata_lance_schema(task: DocumentBatch) -> pa.Schema | None:
    schema = (task._metadata.get("lance") or {}).get("schema")
    return schema_from_json_value(schema) if isinstance(schema, dict) else None


def _schema_for_table(schema: pa.Schema, table: pa.Table) -> pa.Schema:
    fields = [schema.field(name) if name in schema.names else table.schema.field(name) for name in table.column_names]
    return pa.schema(fields)


def _is_blob_descriptor_field(field: pa.Field) -> bool:
    return bool(field.metadata and field.metadata.get(b"lance-encoding:blob") == b"true")


def _materialize_lance_blobs(
    task: DocumentBatch,
    table: pa.Table,
    schema: pa.Schema,
    write_kwargs: dict[str, Any],
) -> pa.Table:
    blob_columns = [
        field.name
        for field in schema
        if field.name in table.column_names and _is_blob_descriptor_field(table.schema.field(field.name))
    ]
    if not blob_columns:
        return table
    if LANCE_ROWADDR_COLUMN not in table.column_names:
        msg = f"Cannot write Lance blob descriptor columns without {LANCE_ROWADDR_COLUMN}"
        raise ValueError(msg)

    import lance

    lance_metadata = task._metadata.get("lance") or {}
    source_path = lance_metadata.get("path")
    if not source_path:
        msg = "Cannot materialize Lance blob columns without source Lance path metadata"
        raise ValueError(msg)
    source_storage_options = write_kwargs.get("source_storage_options", write_kwargs.get("storage_options"))
    source_dataset = lance.dataset(
        source_path,
        **lance_dataset_kwargs(source_storage_options, lance_metadata.get("version")),
    )
    rowaddrs = [int(value) for value in table[LANCE_ROWADDR_COLUMN].combine_chunks().to_pylist()]
    for column in blob_columns:
        payloads = [
            payload for _, payload in source_dataset.read_blobs(column, addresses=rowaddrs, preserve_order=True)
        ]
        table = table.set_column(table.schema.get_field_index(column), column, lance.blob_array(payloads))
    return table


@dataclass
class LanceWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
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
    checkpoint_mode: Literal["ignore", "overwrite", "error"] = "ignore"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self.storage_options = (self.write_kwargs or {}).get("storage_options")
        self.checkpoint_storage_options = (self.write_kwargs or {}).get("checkpoint_storage_options")
        if self.mode not in {"ignore", "overwrite", "append", "error"}:
            msg = f"Invalid Lance write mode: {self.mode}"
            raise ValueError(msg)
        _preflight_checkpoint_path(self.commit_path, self.checkpoint_storage_options, self.checkpoint_mode)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _output_table_and_schema(self, task: DocumentBatch) -> tuple[pa.Table, pa.Schema | None]:
        table = task.to_pyarrow()
        schema = self.schema or _metadata_lance_schema(task)
        if schema is not None:
            table = _materialize_lance_blobs(task, table, schema, self.write_kwargs)
        if self.schema is not None:
            table = table.select(self.schema.names)
            return table, self.schema
        table = table.select(self.fields) if self.fields is not None else _drop_reserved_lance_columns(table)
        return table, _schema_for_table(schema, table) if schema is not None else None

    def process(self, task: DocumentBatch) -> FileGroupTask:
        from lance_ray.fragment import write_fragment

        write_kwargs = {
            key: value for key, value in self.write_kwargs.items() if key not in _WRITE_FRAGMENT_KWARG_DENYLIST
        }
        write_kwargs.setdefault("max_rows_per_file", 500_000)
        table, schema = self._output_table_and_schema(task)
        results = write_fragment(
            [table],
            self.path,
            schema=schema,
            **write_kwargs,
        )
        if not results:
            msg = f"write_fragment returned no fragments for task {task.task_id}"
            raise RuntimeError(msg)

        record_paths = []
        for index, (fragment, schema) in enumerate(results):
            record = {
                "kind": "lance_write",
                "dataset_path": self.path,
                "mode": self.mode,
                "schema": schema_to_json_value(schema),
                "fragment_id": fragment.id,
                "fragment": object_to_base64(fragment),
            }
            record_paths.append(
                _write_json_record(
                    self.commit_path,
                    record,
                    self.checkpoint_storage_options,
                    _record_id("lance_write", task.task_id, index),
                )
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=record_paths,
            reader_config={"format": "lance_checkpoint", "kind": "lance_write"},
            _metadata={**task._metadata, "format": "lance_checkpoint", "lance_checkpoint_kind": "lance_write"},
            _stage_perf=task._stage_perf,
        )


@dataclass
class LanceAnnotationWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
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
    checkpoint_mode: Literal["ignore", "overwrite", "error"] = "ignore"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    _prepared_version: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.storage_options = (self.write_kwargs or {}).get("storage_options")
        self.checkpoint_storage_options = (self.write_kwargs or {}).get("checkpoint_storage_options")
        _preflight_checkpoint_path(self.commit_path, self.checkpoint_storage_options, self.checkpoint_mode)
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
        update_table = update_table.set_column(
            0,
            "_rowaddr",
            update_table["_rowaddr"].combine_chunks().cast(pa.uint64()),
        )
        _validate_unique_rowaddrs(update_table, fragment_id)
        return update_table

    def process(self, task: DocumentBatch) -> FileGroupTask:
        import lance

        table = task.to_pyarrow()
        _validate_annotation_input_table(table, self.fields or [])
        version = self._update_version()
        dataset = lance.dataset(self.path, **lance_dataset_kwargs(self.storage_options, version))

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
                "kind": "lance_annotation_update",
                "dataset_path": self.path,
                "dataset_version": version,
                "fragment_id": fragment_id,
                "fields_modified": sorted(set(fields_modified)),
                "updated_fragment": object_to_base64(updated_fragment),
            }
            record_paths.append(
                _write_json_record(
                    self.commit_path,
                    record,
                    self.checkpoint_storage_options,
                    _record_id("lance_annotation_update", task.task_id, fragment_id),
                )
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=record_paths,
            reader_config={"format": "lance_checkpoint", "kind": "lance_annotation_update"},
            _metadata={
                **task._metadata,
                "format": "lance_checkpoint",
                "lance_checkpoint_kind": "lance_annotation_update",
            },
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


def commit_lance_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
    retries: int = 5,
) -> int:
    """Commit checkpoint records produced by ``LanceWriter``."""

    import lance
    from lance_ray import LanceFragmentCommitter

    records, committed_version = _load_checkpoint_records(commit_path, "lance_write", checkpoint_storage_options)
    if committed_version is not None:
        return committed_version

    dataset_path = str(_unique((record["dataset_path"] for record in records), "dataset path"))
    if dataset_path != path:
        msg = f"Checkpoint records are for {dataset_path}, not {path}"
        raise ValueError(msg)
    mode = str(_unique((record["mode"] for record in records), "write mode"))
    write_result = [
        [(object_from_base64(record["fragment"]), schema_from_json_value(record["schema"])) for record in records]
    ]
    schema = write_result[0][0][1]

    def _commit() -> int:
        existing_dataset = _open_dataset_if_exists(path, storage_options)
        if mode == "error" and existing_dataset is not None:
            msg = f"Lance dataset already exists at {path}"
            raise FileExistsError(msg)
        if mode == "ignore" and existing_dataset is not None:
            return existing_dataset.version
        if mode == "append" and existing_dataset is None:
            msg = f"Cannot append to missing Lance dataset at {path}"
            raise FileNotFoundError(msg)
        commit_mode = "append" if mode == "append" else "overwrite" if mode == "overwrite" else "create"
        committer = LanceFragmentCommitter(path, schema=schema, mode=commit_mode, storage_options=storage_options)
        if commit_mode == "append":
            committer.on_write_start(schema)
        committer.on_write_complete(
            [[(pickle.dumps(fragment), pickle.dumps(schema)) for fragment, schema in write_result[0]]]
        )
        return lance.dataset(path, storage_options=storage_options).version

    version = call_with_retry(_commit, description="commit Lance checkpoint", max_attempts=retries)
    _write_committed_marker(commit_path, version, checkpoint_storage_options)
    return version


def _annotation_records_by_fragment(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for record in records:
        fragment_id = int(record["fragment_id"])
        if fragment_id in selected:
            msg = (
                f"Conflicting Lance annotation checkpoint records for fragment {fragment_id}. "
                "Ensure each Lance fragment is updated by at most one writer task."
            )
            raise ValueError(msg)
        selected[fragment_id] = record
    return selected


def commit_lance_annotation_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
    retries: int = 5,
) -> int:
    """Commit checkpoint records produced by ``LanceAnnotationWriter``."""

    import lance

    records, committed_version = _load_checkpoint_records(
        commit_path, "lance_annotation_update", checkpoint_storage_options
    )
    if committed_version is not None:
        return committed_version

    dataset_path = str(_unique((record["dataset_path"] for record in records), "dataset path"))
    if dataset_path != path:
        msg = f"Checkpoint records are for {dataset_path}, not {path}"
        raise ValueError(msg)
    read_version = int(_unique((record["dataset_version"] for record in records), "dataset version"))
    records_by_fragment = _annotation_records_by_fragment(records)
    updated_fragments = [object_from_base64(record["updated_fragment"]) for record in records_by_fragment.values()]
    fields_modified = sorted({field for record in records_by_fragment.values() for field in record["fields_modified"]})
    operation = lance.LanceOperation.Update(updated_fragments=updated_fragments, fields_modified=fields_modified)
    version = lance.LanceDataset.commit(
        path,
        operation,
        read_version=read_version,
        storage_options=storage_options,
        max_retries=retries,
    ).version
    _write_committed_marker(
        commit_path,
        version,
        checkpoint_storage_options,
    )
    return version
