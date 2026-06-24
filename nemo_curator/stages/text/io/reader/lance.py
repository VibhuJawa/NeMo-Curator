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

from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import DocumentBatch, EmptyTask
from nemo_curator.tasks.tasks import Task
from nemo_curator.utils.hash_utils import get_deterministic_hash
from nemo_curator.utils.lance import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
)

from .base import BaseReader, ReaderOutput


def _read_dataset_kwargs(read_kwargs: dict[str, Any], version: int | None = None) -> dict[str, Any]:
    resolved_version = version if version is not None else read_kwargs.get("version")
    options = {"storage_options": read_kwargs.get("storage_options"), "version": resolved_version}
    return {**dict(read_kwargs.get("dataset_options") or {}), **{k: v for k, v in options.items() if v is not None}}


def _scanner_kwargs(read_kwargs: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
    scanner_kwargs = dict(read_kwargs.get("scanner_options") or {})
    for key, value in read_kwargs.items():
        if key in {"dataset_options", "scanner_options", "storage_options", "version"}:
            continue
        scanner_kwargs[key] = value
    if fields is not None:
        scanner_kwargs["columns"] = fields
    return scanner_kwargs


def _requested_blob_v2_columns(dataset: object, scanner_kwargs: dict[str, Any]) -> list[str]:
    requested_columns = scanner_kwargs.get("columns")
    if isinstance(requested_columns, dict | list):
        requested_columns = set(requested_columns)

    return [
        field.name
        for field in dataset.schema  # type: ignore[attr-defined]
        if getattr(field.type, "extension_name", None) == "lance.blob.v2"
        and (requested_columns is None or field.name in requested_columns)
    ]


def _restore_lance_blob_v2_columns(dataset: object, table: pa.Table, blob_columns: list[str]) -> pa.Table:
    import lance

    rowaddrs = [int(value) for value in table["_rowaddr"].combine_chunks().to_pylist()]
    for column in blob_columns:
        payloads = [
            payload
            for _, payload in dataset.read_blobs(column, addresses=rowaddrs, preserve_order=True)  # type: ignore[attr-defined]
        ]
        table = table.set_column(table.schema.get_field_index(column), column, lance.blob_array(payloads))
    return table


def _fragment_ids_from_row_addresses(rowaddr_column: pa.ChunkedArray) -> pa.Array:
    rowaddrs = rowaddr_column.combine_chunks().cast(pa.uint64())
    return pa.array([int(value) >> 32 for value in rowaddrs.to_pylist()], type=pa.uint64())


def _add_lance_metadata(table: pa.Table) -> pa.Table:
    if "_rowaddr" not in table.column_names:
        msg = "Lance scanner did not return _rowaddr; include_lance_metadata requires row addresses"
        raise ValueError(msg)

    table = table.rename_columns([LANCE_ROWADDR_COLUMN if name == "_rowaddr" else name for name in table.column_names])
    return table.append_column(LANCE_FRAGID_COLUMN, _fragment_ids_from_row_addresses(table[LANCE_ROWADDR_COLUMN]))


@dataclass
class LanceReadTask(Task[list[int]]):
    data: list[int] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return bool(self.data)

    def get_deterministic_id(self) -> str:
        lance_metadata = self._metadata.get("lance") or {}
        parts = [
            str(lance_metadata.get("path", self.dataset_name)),
            str(lance_metadata.get("version", "")),
            *(str(fragment_id) for fragment_id in self.data),
        ]
        return get_deterministic_hash(parts)


@dataclass
class LancePartitioningStage(ProcessingStage[EmptyTask, LanceReadTask]):
    path: str
    fragments_per_partition: int = 32
    fragment_ids: list[int] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "lance_partitioning"

    def __post_init__(self) -> None:
        if self.fragments_per_partition <= 0:
            msg = "fragments_per_partition must be greater than 0"
            raise ValueError(msg)

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, _: EmptyTask) -> list[LanceReadTask]:
        import lance

        dataset = lance.dataset(self.path, **_read_dataset_kwargs(self.read_kwargs))
        available_fragments = [fragment.fragment_id for fragment in dataset.get_fragments()]
        if self.fragment_ids is None:
            fragment_ids = available_fragments
        else:
            available = set(available_fragments)
            missing = sorted(set(self.fragment_ids) - available)
            if missing:
                msg = f"Lance dataset does not contain requested fragment ids: {missing[:10]}"
                raise ValueError(msg)
            fragment_ids = list(self.fragment_ids)

        tasks = []
        for start in range(0, len(fragment_ids), self.fragments_per_partition):
            owned_fragments = fragment_ids[start : start + self.fragments_per_partition]
            tasks.append(
                LanceReadTask(
                    dataset_name=self.path,
                    data=owned_fragments,
                    _metadata={
                        "source_files": [self.path],
                        "lance": {
                            "path": self.path,
                            "version": dataset.version,
                            "fragment_ids": owned_fragments,
                        },
                    },
                )
            )
        return tasks


@dataclass
class LanceReaderStage(BaseReader[LanceReadTask]):
    """Read Lance fragment groups into Arrow batches."""

    path: str = ""
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    include_lance_metadata: bool = True
    allow_empty: bool = True
    name: str = "lance_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.path:
            msg = "path is required"
            raise ValueError(msg)

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.fields or self.read_kwargs.get("columns") or [])
        if self.include_lance_metadata:
            output_fields.extend([LANCE_ROWADDR_COLUMN, LANCE_FRAGID_COLUMN])
        return ["data"], output_fields

    def _output_metadata(self, task: LanceReadTask, output: ReaderOutput) -> dict[str, Any]:
        return output.metadata if output.metadata is not None else task._metadata

    def read_task(
        self,
        task: LanceReadTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:
        import lance
        from lance.schema import schema_to_json

        read_kwargs = {} if read_kwargs is None else read_kwargs
        version = task._metadata["lance"]["version"]
        dataset = lance.dataset(self.path, **_read_dataset_kwargs(read_kwargs, version=version))
        fragments = [dataset.get_fragment(fragment_id) for fragment_id in task.data]
        scanner_kwargs = _scanner_kwargs(read_kwargs, fields)
        blob_columns = _requested_blob_v2_columns(dataset, scanner_kwargs)
        if self.include_lance_metadata or blob_columns:
            scanner_kwargs["with_row_address"] = True
        scanner_kwargs["fragments"] = fragments
        table = dataset.scanner(**scanner_kwargs).to_table()
        if blob_columns:
            table = _restore_lance_blob_v2_columns(dataset, table, blob_columns)
        if self.include_lance_metadata:
            table = _add_lance_metadata(table)
        elif blob_columns and "_rowaddr" in table.column_names:
            table = table.drop_columns(["_rowaddr"])

        metadata = dict(task._metadata)
        lance_metadata = dict(metadata.get("lance") or {})
        lance_metadata["schema"] = schema_to_json(dataset.schema)
        metadata["lance"] = lance_metadata
        return ReaderOutput(table, metadata)


@dataclass
class LanceReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Read a Lance dataset into Curator ``DocumentBatch`` objects by fragment."""
    path: str
    fragments_per_partition: int = 32
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    task_type: Literal["document"] = "document"
    name: str = "lance_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        if self.task_type != "document":
            msg = f"Converting DocumentBatch to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        return [
            LancePartitioningStage(
                path=self.path,
                fragments_per_partition=self.fragments_per_partition,
                fragment_ids=self.fragment_ids,
                read_kwargs=self.read_kwargs,
            ),
            LanceReaderStage(
                path=self.path,
                fields=self.fields,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]
