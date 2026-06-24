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
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

import pyarrow as pa

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.lance_utils import (
    LANCE_FRAGID_COLUMN,
    LANCE_ROWADDR_COLUMN,
    lance_dataset_kwargs,
    schema_to_json_value,
)
from nemo_curator.tasks import DocumentBatch, EmptyTask
from nemo_curator.tasks.tasks import Task
from nemo_curator.utils.hash_utils import get_deterministic_hash


def _infer_dataset_name(path: str) -> str:
    parsed = urlparse(path)
    posix_path = PurePosixPath(parsed.path.rstrip("/"))
    if posix_path.name:
        return posix_path.name
    if parsed.netloc:
        return parsed.netloc
    return "lance_dataset"


def _read_dataset_kwargs(read_kwargs: dict[str, Any], version: int | None = None) -> dict[str, Any]:
    kwargs = {}
    dataset_options = dict(read_kwargs.get("dataset_options") or {})
    kwargs.update(dataset_options)
    kwargs.update(lance_dataset_kwargs(read_kwargs.get("storage_options"), read_kwargs.get("version", version)))
    return kwargs


def _scanner_kwargs(read_kwargs: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
    scanner_kwargs = dict(read_kwargs.get("scanner_options") or {})
    for key, value in read_kwargs.items():
        if key in {"dataset_options", "scanner_options", "storage_options", "version"}:
            continue
        scanner_kwargs[key] = value
    if fields is not None:
        scanner_kwargs["columns"] = fields
    return scanner_kwargs


def _add_lance_metadata(table: pa.Table) -> pa.Table:
    if "_rowaddr" not in table.column_names:
        msg = "Lance scanner did not return _rowaddr; include_lance_metadata requires row addresses"
        raise ValueError(msg)

    rowaddrs = table["_rowaddr"].combine_chunks().cast(pa.uint64())
    fragids = pa.array([int(value) >> 32 for value in rowaddrs.to_pylist()], type=pa.uint64())
    table = table.rename_columns([LANCE_ROWADDR_COLUMN if name == "_rowaddr" else name for name in table.column_names])
    return table.append_column(LANCE_FRAGID_COLUMN, fragids)


def _lance_schema_for_table(dataset: object, table: pa.Table) -> pa.Schema:
    schema = dataset.schema  # type: ignore[attr-defined]
    return pa.schema([schema.field(name) for name in table.column_names if name in schema.names])


@dataclass
class LanceReadTask(Task[list[int]]):
    """Task carrying the Lance fragment ids owned by one reader partition."""

    data: list[int] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return bool(self.data)

    def get_deterministic_id(self) -> str:
        lance_metadata = self._metadata.get("lance", {})
        return get_deterministic_hash(
            [
                lance_metadata.get("path", ""),
                str(lance_metadata.get("version", "")),
                *[str(fragment_id) for fragment_id in sorted(self.data)],
            ]
        )


@dataclass
class LancePartitioningStage(ProcessingStage[EmptyTask, LanceReadTask]):
    """Partition a Lance dataset into fragment-owned Curator tasks."""

    path: str
    fragments_per_partition: int = 32
    fragment_ids: list[int] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_name: str | None = None
    name: str = "lance_partitioning"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def __post_init__(self) -> None:
        if self.fragments_per_partition <= 0:
            msg = "fragments_per_partition must be greater than 0"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

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

        dataset_name = self.dataset_name or _infer_dataset_name(self.path)
        tasks = []
        total = (len(fragment_ids) + self.fragments_per_partition - 1) // self.fragments_per_partition
        for index, start in enumerate(range(0, len(fragment_ids), self.fragments_per_partition)):
            owned_fragments = fragment_ids[start : start + self.fragments_per_partition]
            tasks.append(
                LanceReadTask(
                    dataset_name=dataset_name,
                    data=owned_fragments,
                    _metadata={
                        "source_files": [self.path],
                        "lance": {
                            "path": self.path,
                            "version": dataset.version,
                            "fragment_ids": owned_fragments,
                        },
                        "partition_index": index,
                        "total_partitions": total,
                    },
                )
            )
        return tasks


@dataclass
class LanceReaderStage(ProcessingStage[LanceReadTask, DocumentBatch]):
    """Read Lance fragment groups into ``DocumentBatch`` objects containing Arrow tables."""

    path: str
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    include_lance_metadata: bool = True
    name: str = "lance_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.fields or self.read_kwargs.get("columns") or [])
        if self.include_lance_metadata:
            output_fields.extend([LANCE_ROWADDR_COLUMN, LANCE_FRAGID_COLUMN])
        return ["data"], output_fields

    def process(self, task: LanceReadTask) -> DocumentBatch | None:
        import lance

        version = (task._metadata.get("lance") or {}).get("version")
        dataset = lance.dataset(self.path, **_read_dataset_kwargs(self.read_kwargs, version=version))
        fragments = [dataset.get_fragment(fragment_id) for fragment_id in task.data]
        scanner_kwargs = _scanner_kwargs(self.read_kwargs, self.fields)
        if self.include_lance_metadata:
            scanner_kwargs["with_row_address"] = True
        scanner_kwargs["fragments"] = fragments
        table = dataset.scanner(**scanner_kwargs).to_table()
        if table.num_rows == 0:
            return None
        lance_schema = _lance_schema_for_table(dataset, table)
        if self.include_lance_metadata:
            table = _add_lance_metadata(table)
        metadata = dict(task._metadata)
        lance_metadata = dict(metadata.get("lance") or {})
        lance_metadata["schema"] = schema_to_json_value(lance_schema)
        metadata["lance"] = lance_metadata
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=table,
            _metadata=metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class LanceReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Composite stage for reading Lance datasets.

    This mirrors the tabular reader shape used by ``ParquetReader`` while
    partitioning work by Lance fragments instead of files.  A Lance fragment is
    owned by at most one Curator task, and a Curator task may contain one or more
    fragments.
    """

    path: str
    fragments_per_partition: int = 32
    fields: list[str] | None = None
    read_kwargs: dict[str, Any] | None = None
    include_lance_metadata: bool = True
    fragment_ids: list[int] | None = None
    task_type: Literal["document"] = "document"
    dataset_name: str | None = None
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
                dataset_name=self.dataset_name,
            ),
            LanceReaderStage(
                path=self.path,
                fields=self.fields,
                read_kwargs=self.read_kwargs,
                include_lance_metadata=self.include_lance_metadata,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read Lance dataset from {self.path}"]
        parts.append(f"with {self.fragments_per_partition} fragments per partition")
        if self.fields:
            parts.append(f"reading columns: {self.fields}")
        return ", ".join(parts)
