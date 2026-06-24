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
import pickle
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.io.lance_utils import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    lance_checkpoint_record_id,
    write_lance_checkpoint_record,
)
from nemo_curator.tasks import DocumentBatch, FileGroupTask

_RESERVED_LANCE_PREFIX = "__lance_"


def _drop_reserved_lance_columns(table: pa.Table) -> pa.Table:
    columns = [name for name in table.column_names if not name.startswith(_RESERVED_LANCE_PREFIX)]
    return table.select(columns)


def _metadata_lance_schema(task: DocumentBatch) -> pa.Schema | None:
    schema = (task._metadata.get("lance") or {}).get("schema")
    if not isinstance(schema, dict):
        return None
    from lance.schema import json_to_schema

    return json_to_schema(schema)


def _schema_for_table(schema: pa.Schema, table: pa.Table) -> pa.Schema:
    fields = [schema.field(name) if name in schema.names else table.schema.field(name) for name in table.column_names]
    return pa.schema(fields)


@dataclass
class LanceWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Write ``DocumentBatch`` tables to Lance fragments and checkpoint the commit."""
    path: str
    commit_path: str
    schema: pa.Schema | None = None
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    fields: list[str] | None = None
    name: str = "lance_writer"
    mode: Literal["create", "append", "overwrite"] = "create"

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
        from lance.schema import schema_to_json
        from lance_ray.fragment import write_fragment

        write_kwargs = dict(self.write_kwargs)
        checkpoint_storage_options = write_kwargs.pop("checkpoint_storage_options", None)
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
                "schema": schema_to_json(schema),
                "fragment": base64.b64encode(pickle.dumps(fragment)).decode("ascii"),
            }
            record_paths.append(
                write_lance_checkpoint_record(
                    self.commit_path,
                    record,
                    lance_checkpoint_record_id("lance_write", task.task_id, index),
                    checkpoint_storage_options,
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
    """Update existing Lance rows using metadata columns emitted by ``LanceReader``."""
    path: str
    commit_path: str
    schema: pa.Schema
    create_columns: bool = False
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    fields: list[str] | None = None
    name: str = "lance_annotation_writer"
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
        """Create or validate annotation columns and pin the Lance version for the run."""
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
            msg = f"Lance annotation columns do not exist: {missing}. Call prepare() before pipeline.run()."
            raise ValueError(msg)

    def _update_version(self) -> int:
        import lance

        if self._prepared_version is not None:
            return self._prepared_version
        logger.warning(
            "LanceAnnotationWriter is running without prepare(). "
            "This is only safe when target columns exist and the dataset does not change during the run."
        )
        dataset = lance.dataset(self.path, storage_options=self.storage_options)
        self._validate_target_columns(dataset)
        return dataset.version

    def _update_table_for_fragment(self, table: pa.Table, fragment_id: int) -> pa.Table:
        fragids = table[LANCE_FRAGID_COLUMN].combine_chunks().cast(pa.uint64())
        mask = pc.equal(fragids, pa.scalar(fragment_id, type=pa.uint64()))
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
        missing = [name for name in [LANCE_ROWADDR_COLUMN, LANCE_FRAGID_COLUMN, *(self.fields or [])] if name not in table.column_names]
        if missing:
            msg = f"Lance annotation update table is missing required columns: {missing}"
            raise ValueError(msg)
        version = self._update_version()
        options = {"storage_options": self.storage_options, "version": version}
        dataset = lance.dataset(self.path, **{k: v for k, v in options.items() if v is not None})

        record_paths = []
        fragment_ids = sorted(int(value) for value in pc.unique(table[LANCE_FRAGID_COLUMN].combine_chunks()).to_pylist())
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
                "updated_fragment": base64.b64encode(pickle.dumps(updated_fragment)).decode("ascii"),
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

def _validate_unique_rowaddrs(table: pa.Table, fragment_id: int) -> None:
    counts = pc.value_counts(table["_rowaddr"].combine_chunks())
    duplicate_values = counts.field("values").filter(pc.greater(counts.field("counts"), 1))
    if len(duplicate_values) > 0:
        sample = ", ".join(str(value) for value in duplicate_values.slice(0, 5).to_pylist())
        msg = (
            f"Lance annotation update contains duplicate {LANCE_ROWADDR_COLUMN} values "
            f"for fragment {fragment_id}: {sample}"
        )
        raise ValueError(msg)
