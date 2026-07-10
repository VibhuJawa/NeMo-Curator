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
import posixpath
from dataclasses import dataclass, field
from typing import Any, Literal

import lance
import pyarrow as pa
from fsspec.core import url_to_fs
from lance.fragment import FragmentMetadata
from lance.schema import json_to_schema, schema_to_json
from lance_ray import LanceFragmentCommitter
from lance_ray.fragment import write_fragment

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.hash_utils import get_deterministic_hash
from nemo_curator.utils.lance import encode_lance_blob_columns

_MAX_ERROR_TASK_IDS = 10
_COMMITTED_MARKER = "_COMMITTED"
_RECORDS_DIR = "records"


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
        if schema is not None:
            table = table.select(schema.names)
        else:
            columns = self.fields
            if columns is None:
                columns = [name for name in table.column_names if not name.startswith("__lance_")]
            table = table.select(columns)
            schema_json = (task._metadata.get("lance") or {}).get("schema")
            if schema_json is not None:
                source_fields = {field.name: field for field in json_to_schema(schema_json)}
                schema = pa.schema([source_fields.get(field.name, field) for field in table.schema])
        if schema is not None:
            table = encode_lance_blob_columns(table, schema)
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
        if results:
            checkpoint_fs, checkpoint_root = url_to_fs(self.commit_path, **(checkpoint_storage_options or {}))
            records_dir = posixpath.join(checkpoint_root.rstrip("/"), _RECORDS_DIR)
            checkpoint_fs.makedirs(records_dir, exist_ok=True)
            for index, (fragment, fragment_schema) in enumerate(results):
                record = {
                    "dataset_path": self.path,
                    "mode": self.mode,
                    "task_id": task.task_id,
                    "fragment_index": index,
                    "schema": schema_to_json(fragment_schema),
                    "fragment": fragment.to_json(),
                }
                record_id = get_deterministic_hash([task.task_id, str(index)])
                record_path = posixpath.join(records_dir, f"{record_id}.json")
                with checkpoint_fs.open(record_path, "w") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                record_paths.append(checkpoint_fs.unstrip_protocol(record_path))

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
    checkpoint_fs, checkpoint_root = url_to_fs(commit_path, **(checkpoint_storage_options or {}))
    marker_path = posixpath.join(checkpoint_root.rstrip("/"), _COMMITTED_MARKER)
    if checkpoint_fs.exists(marker_path):
        with checkpoint_fs.open(marker_path) as stream:
            return int(json.loads(stream.read())["version"])

    records = []
    records_glob = posixpath.join(checkpoint_root.rstrip("/"), _RECORDS_DIR, "*.json")
    for record_path in sorted(checkpoint_fs.glob(records_glob)):
        with checkpoint_fs.open(record_path) as stream:
            records.append(json.loads(stream.read()))
    if not records:
        msg = f"No Lance checkpoint records found under {commit_path}"
        raise ValueError(msg)

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
    checkpoint_fs.makedirs(posixpath.dirname(marker_path), exist_ok=True)
    with checkpoint_fs.open(marker_path, "w") as stream:
        stream.write(json.dumps({"version": version}, sort_keys=True, indent=2) + "\n")
    return version
