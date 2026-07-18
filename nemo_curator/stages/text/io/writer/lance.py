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
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.hash_utils import get_deterministic_hash
from nemo_curator.utils.lance import encode_lance_blob_columns

_MAX_ERROR_TASK_IDS = 10
_COMMITTED_MARKER = "_COMMITTED"
_RECORDS_DIR = "records"


def _committed_fragment_version(dataset: lance.LanceDataset, fragments: list[FragmentMetadata]) -> int | None:
    expected_paths = {data_file.path for fragment in fragments for data_file in fragment.files}
    for version_info in reversed(dataset.versions()):
        version = int(version_info["version"])
        transaction = dataset.read_transaction(version)
        transaction_fragments = getattr(getattr(transaction, "operation", None), "fragments", [])
        committed_paths = {data_file.path for fragment in transaction_fragments for data_file in fragment.files}
        if committed_paths == expected_paths:
            return version
    return None


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
        """Select output columns and restore source Lance field types when available."""
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
            record = {
                "mode": self.mode,
                "task_id": task.task_id,
                "schema": schema_to_json(results[0][1]),
                "fragments": [fragment.to_json() for fragment, _ in results],
            }
            record_path = posixpath.join(records_dir, f"{get_deterministic_hash([task.task_id])}.json")
            with checkpoint_fs.open(record_path, "w") as stream:
                json.dump(record, stream, sort_keys=True)
                stream.write("\n")
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
    """Publish task records from one ``LanceWriter`` as a Lance dataset version."""
    checkpoint_fs, checkpoint_root = url_to_fs(commit_path, **(checkpoint_storage_options or {}))
    checkpoint_root = checkpoint_root.rstrip("/")
    marker_path = posixpath.join(checkpoint_root, _COMMITTED_MARKER)
    if checkpoint_fs.exists(marker_path):
        with checkpoint_fs.open(marker_path) as stream:
            marker = json.load(stream)
        marker_dataset_path = marker.get("dataset_path")
        if marker_dataset_path != path:
            msg = f"Lance checkpoint {commit_path} is for dataset {marker_dataset_path!r}, not {path!r}"
            raise ValueError(msg)
        version = int(marker["version"])
        logger.warning(f"Lance checkpoint {commit_path} was already committed as version {version}; skipping commit")
        return version

    records = []
    records_glob = posixpath.join(checkpoint_root, _RECORDS_DIR, "*.json")
    for record_path in sorted(checkpoint_fs.glob(records_glob)):
        with checkpoint_fs.open(record_path) as stream:
            records.append(json.load(stream))
    if not records:
        msg = f"No Lance checkpoint records found under {commit_path}"
        raise ValueError(msg)

    records.sort(key=lambda record: str(record["task_id"]))
    mode = str(records[0]["mode"])
    schema = json_to_schema(records[0]["schema"])
    fragments = [
        FragmentMetadata.from_json(json.dumps(fragment)) for record in records for fragment in record["fragments"]
    ]

    version = None
    if mode == "append":
        version = _committed_fragment_version(lance.dataset(path, storage_options=storage_options), fragments)
    if version is None:
        try:
            committer = LanceFragmentCommitter(path, schema=schema, mode=mode, storage_options=storage_options)
            if mode == "append":
                committer.on_write_start(schema)
            committer.on_write_complete([[(pickle.dumps(fragment), pickle.dumps(schema)) for fragment in fragments]])
        except Exception as error:
            task_ids = sorted({str(record["task_id"]) for record in records})
            displayed_task_ids = task_ids[:_MAX_ERROR_TASK_IDS]
            remaining = (
                f" and {len(task_ids) - len(displayed_task_ids)} more" if len(task_ids) > _MAX_ERROR_TASK_IDS else ""
            )
            error.add_note(f"Lance commit includes fragments from Curator task IDs {displayed_task_ids}{remaining}")
            raise
        version = _committed_fragment_version(lance.dataset(path, storage_options=storage_options), fragments)
        if version is None:
            msg = f"Could not find the Lance transaction for checkpoint {commit_path}"
            raise RuntimeError(msg)
    else:
        logger.warning(
            f"Lance checkpoint {commit_path} fragments were already committed as version {version}; restoring marker"
        )
    checkpoint_fs.makedirs(posixpath.dirname(marker_path), exist_ok=True)
    with checkpoint_fs.open(marker_path, "w") as stream:
        json.dump({"dataset_path": path, "version": version}, stream, sort_keys=True)
        stream.write("\n")
    return version
