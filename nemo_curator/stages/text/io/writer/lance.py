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
from typing import Any, Literal

import lance
import pyarrow as pa
from lance.fragment import FragmentMetadata
from lance.schema import json_to_schema, schema_to_json
from lance_ray import LanceFragmentCommitter
from lance_ray.fragment import write_fragment

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.lance import (
    lance_checkpoint_record_id,
    read_lance_checkpoint,
    write_lance_committed_version,
    write_lance_fragment_record,
)

_RESERVED_LANCE_PREFIX = "__lance_"
_MAX_ERROR_TASK_IDS = 10


def _drop_reserved_lance_columns(table: pa.Table) -> pa.Table:
    columns = [name for name in table.column_names if not name.startswith(_RESERVED_LANCE_PREFIX)]
    return table.select(columns)


def _lance_schema_for_table(task: DocumentBatch, table: pa.Table) -> pa.Schema | None:
    """Preserve source Lance fields while retaining new fields from the table."""
    schema_json = (task._metadata.get("lance") or {}).get("schema")
    if schema_json is None:
        return None

    source_fields = {field.name: field for field in json_to_schema(schema_json)}
    return pa.schema([source_fields.get(field.name, field) for field in table.schema])


def _encode_blob_v2_columns(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Rebuild Lance Blob v2 arrays from materialized reader columns."""
    for schema_field in schema:
        column_index = table.schema.get_field_index(schema_field.name)
        if column_index < 0 or getattr(schema_field.type, "extension_name", None) != "lance.blob.v2":
            continue
        column = table.column(column_index).combine_chunks()
        if column.type != schema_field.type:
            table = table.set_column(column_index, schema_field, lance.blob_array(column.to_pylist()))
    return table


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

    def __post_init__(self) -> None:
        self.write_kwargs = dict(self.write_kwargs or {})

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _output_table_and_schema(self, task: DocumentBatch) -> tuple[pa.Table, pa.Schema | None]:
        table = task.to_pyarrow()
        schema = self.schema
        if self.schema is not None:
            table = table.select(self.schema.names)
        else:
            table = table.select(self.fields) if self.fields is not None else _drop_reserved_lance_columns(table)
            schema = _lance_schema_for_table(task, table)
        if schema is not None:
            table = _encode_blob_v2_columns(table, schema)
        return table, schema

    def process(self, task: DocumentBatch) -> FileGroupTask:
        """Write one batch as uncommitted fragments and persist their commit records."""
        write_kwargs = dict(self.write_kwargs)
        checkpoint_storage_options = write_kwargs.pop("checkpoint_storage_options", None)
        table, schema = self._output_table_and_schema(task)
        # Write physical data files; commit_lance_checkpoint publishes them as a dataset version.
        results = write_fragment(
            [table],
            self.path,
            schema=schema,
            **write_kwargs,
        )

        # Persist fragment metadata so the final commit can collect outputs from every task.
        record_paths = []
        for index, (fragment, schema) in enumerate(results):
            record = {
                "kind": "lance_write",
                "dataset_path": self.path,
                "mode": self.mode,
                "task_id": task.task_id,
                "fragment_index": index,
                "schema": schema_to_json(schema),
                "fragment": fragment.to_json(),
            }
            record_paths.append(
                write_lance_fragment_record(
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


def commit_lance_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
    """Publish all checkpointed fragments as one Lance dataset version."""
    records, committed_version = read_lance_checkpoint(commit_path, "lance_write", checkpoint_storage_options)
    if committed_version is not None:
        return committed_version

    # All records are committed together, so they must target one dataset and write mode.
    dataset_paths = {record["dataset_path"] for record in records}
    if dataset_paths != {path}:
        msg = f"Checkpoint records are for {sorted(dataset_paths)}, not {path}"
        raise ValueError(msg)
    modes = {str(record["mode"]) for record in records}
    if len(modes) != 1:
        msg = f"Expected one write mode; got {sorted(modes)}"
        raise ValueError(msg)
    mode = modes.pop()

    records.sort(key=lambda record: (str(record["task_id"]), record["fragment_index"]))
    fragments = [
        (FragmentMetadata.from_json(json.dumps(record["fragment"])), json_to_schema(record["schema"]))
        for record in records
    ]
    schema = fragments[0][1]

    try:
        committer = LanceFragmentCommitter(path, schema=schema, mode=mode, storage_options=storage_options)
        if mode == "append":
            # TODO: Detect an already-published append when retrying after committed-version persistence fails.
            committer.on_write_start(schema)
        fragment_payloads = [(pickle.dumps(fragment), pickle.dumps(schema)) for fragment, schema in fragments]
        committer.on_write_complete([fragment_payloads])
    except Exception as error:
        task_ids = sorted({str(record["task_id"]) for record in records})
        displayed_task_ids = task_ids[:_MAX_ERROR_TASK_IDS]
        remaining = (
            f" and {len(task_ids) - len(displayed_task_ids)} more" if len(task_ids) > _MAX_ERROR_TASK_IDS else ""
        )
        error.add_note(f"Lance commit includes fragments from Curator task IDs {displayed_task_ids}{remaining}")
        raise
    dataset = lance.dataset(path, storage_options=storage_options)
    version = dataset.version
    write_lance_committed_version(commit_path, version, checkpoint_storage_options)
    return version
