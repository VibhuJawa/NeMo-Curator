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

import pickle
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.lance_utils import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    lance_checkpoint_record_id,
    lance_dataset_kwargs,
    object_from_base64,
    object_to_base64,
    read_lance_checkpoint,
    schema_from_json_value,
    schema_to_json_value,
    write_lance_checkpoint_marker,
    write_lance_checkpoint_record,
)
from nemo_curator.tasks import DocumentBatch, FileGroupTask

_RESERVED_LANCE_PREFIX = "__lance_"


def _drop_reserved_lance_columns(table: pa.Table) -> pa.Table:
    columns = [name for name in table.column_names if not name.startswith(_RESERVED_LANCE_PREFIX)]
    return table.select(columns)


def _metadata_lance_schema(task: DocumentBatch) -> pa.Schema | None:
    schema = (task._metadata.get("lance") or {}).get("schema")
    return schema_from_json_value(schema) if isinstance(schema, dict) else None


def _schema_for_table(schema: pa.Schema, table: pa.Table) -> pa.Schema:
    fields = [schema.field(name) if name in schema.names else table.schema.field(name) for name in table.column_names]
    return pa.schema(fields)


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
    mode: Literal["create", "append", "overwrite"] = "create"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self.storage_options = (self.write_kwargs or {}).get("storage_options")
        self.checkpoint_storage_options = (self.write_kwargs or {}).get("checkpoint_storage_options")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _output_table_and_schema(self, task: DocumentBatch) -> tuple[pa.Table, pa.Schema | None]:
        table = task.to_pyarrow()
        schema = self.schema or _metadata_lance_schema(task)
        if self.schema is not None:
            table = table.select(self.schema.names)
            return table, self.schema
        table = table.select(self.fields) if self.fields is not None else _drop_reserved_lance_columns(table)
        return table, _schema_for_table(schema, table) if schema is not None else None

    def process(self, task: DocumentBatch) -> FileGroupTask:
        from lance_ray.fragment import write_fragment

        write_kwargs = dict(self.write_kwargs)
        write_kwargs.pop("checkpoint_storage_options", None)
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
                "fragment": object_to_base64(fragment),
            }
            record_paths.append(
                write_lance_checkpoint_record(
                    self.commit_path,
                    record,
                    lance_checkpoint_record_id("lance_write", task.task_id, index),
                    self.checkpoint_storage_options,
                )
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=record_paths,
            _metadata=task._metadata,
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
                write_lance_checkpoint_record(
                    self.commit_path,
                    record,
                    lance_checkpoint_record_id("lance_annotation_update", task.task_id, fragment_id),
                    self.checkpoint_storage_options,
                )
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=record_paths,
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


def commit_lance_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
    """Commit checkpoint records produced by ``LanceWriter``."""

    import lance
    from lance_ray import LanceFragmentCommitter

    records, committed_version = read_lance_checkpoint(commit_path, "lance_write", checkpoint_storage_options)
    if committed_version is not None:
        return committed_version

    dataset_paths = {record["dataset_path"] for record in records}
    if dataset_paths != {path}:
        msg = f"Checkpoint records are for {sorted(dataset_paths)}, not {path}"
        raise ValueError(msg)
    modes = {record["mode"] for record in records}
    if len(modes) != 1:
        msg = f"Expected one write mode; got {sorted(modes)}"
        raise ValueError(msg)
    mode = str(next(iter(modes)))
    write_result = [
        [(object_from_base64(record["fragment"]), schema_from_json_value(record["schema"])) for record in records]
    ]
    schema = write_result[0][0][1]

    committer = LanceFragmentCommitter(path, schema=schema, mode=mode, storage_options=storage_options)
    if mode == "append":
        committer.on_write_start(schema)
    committer.on_write_complete(
        [[(pickle.dumps(fragment), pickle.dumps(schema)) for fragment, schema in write_result[0]]]
    )
    version = lance.dataset(path, storage_options=storage_options).version
    write_lance_checkpoint_marker(commit_path, version, checkpoint_storage_options)
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
) -> int:
    """Commit checkpoint records produced by ``LanceAnnotationWriter``."""

    import lance

    records, committed_version = read_lance_checkpoint(
        commit_path, "lance_annotation_update", checkpoint_storage_options
    )
    if committed_version is not None:
        return committed_version

    dataset_paths = {record["dataset_path"] for record in records}
    if dataset_paths != {path}:
        msg = f"Checkpoint records are for {sorted(dataset_paths)}, not {path}"
        raise ValueError(msg)
    read_versions = {int(record["dataset_version"]) for record in records}
    if len(read_versions) != 1:
        msg = f"Expected one dataset version; got {sorted(read_versions)}"
        raise ValueError(msg)
    read_version = next(iter(read_versions))
    records_by_fragment = _annotation_records_by_fragment(records)
    updated_fragments = [object_from_base64(record["updated_fragment"]) for record in records_by_fragment.values()]
    fields_modified = sorted({field for record in records_by_fragment.values() for field in record["fields_modified"]})
    operation = lance.LanceOperation.Update(updated_fragments=updated_fragments, fields_modified=fields_modified)
    version = lance.LanceDataset.commit(
        path,
        operation,
        read_version=read_version,
        storage_options=storage_options,
    ).version
    write_lance_checkpoint_marker(
        commit_path,
        version,
        checkpoint_storage_options,
    )
    return version
