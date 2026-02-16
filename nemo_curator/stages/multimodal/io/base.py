# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch
from nemo_curator.tasks.multimodal import METADATA_SCHEMA, MULTIMODAL_SCHEMA
from nemo_curator.utils.file_utils import (
    ensure_parent_directory,
    resolve_fs_and_path,
    resolve_sidecar_output_path,
    resolve_task_scoped_output_path,
)
from nemo_curator.utils.grouping import split_by_chunk_size
from nemo_curator.utils.multimodal_utils import sort_multimodal_table
from nemo_curator.utils.webdataset_utils import content_type_from_name

Row = dict[str, object]
_PAIR_ELEMENT_COUNT = 2
_SUPPORTED_WRITE_MODES: set[str] = {"overwrite", "error"}


@dataclass
class RowSource:
    """Generic source context for building multimodal rows."""

    source_shard: str
    content_path: str
    source_id: str | None = None


@dataclass
class BaseMultimodalReaderStage(ProcessingStage[FileGroupTask, MultimodalBatch], ABC):
    """Base reader contract for multimodal file formats.

    Custom reader implementations must define:
    - ``read_tables_and_metadata(source_path)``:
      returns ``(data_table, metadata_table)`` where:
      - ``data_table`` follows ``MULTIMODAL_SCHEMA``
      - ``metadata_table`` follows ``METADATA_SCHEMA`` (at minimum ``sample_id`` and ``metadata_json``)

    """

    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)

    # Public stage contract ----------------------------------------------------
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data", "metadata_index"], list(MULTIMODAL_SCHEMA.names)

    def process(self, task: FileGroupTask) -> MultimodalBatch | list[MultimodalBatch]:
        data_tables: list[pa.Table] = []
        metadata_tables: list[pa.Table] = []
        for source_path in task.data:
            shard_data_table, shard_metadata_table = self.read_tables_and_metadata(source_path)
            data_tables.append(shard_data_table)
            metadata_tables.append(shard_metadata_table)

        table = pa.concat_tables(data_tables) if data_tables else pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA)
        table = sort_multimodal_table(table)
        metadata_by_sample = self._metadata_map_from_tables(metadata_tables)
        table_splits = self.split_table(table)
        batches = [
            self._build_batch(
                task=task,
                table=batch_table,
                metadata_by_sample=metadata_by_sample,
                batch_index=batch_index,
                split_output=self.max_batch_bytes is not None,
            )
            for batch_index, batch_table in enumerate(table_splits)
        ]
        return batches[0] if self.max_batch_bytes is None else batches

    @abstractmethod
    def read_tables_and_metadata(self, source_path: str) -> tuple[pa.Table, pa.Table]:
        """Read one source and return normalized data + metadata tables."""

    # Optional override hooks --------------------------------------------------
    def split_table(self, table: pa.Table) -> list[pa.Table]:
        """Split a data table into output batch tables."""
        if self.max_batch_bytes is None:
            return [table]
        return self.split_table_by_sample_max_bytes(table, self.max_batch_bytes)

    def table_nbytes(self, table: pa.Table) -> int:
        """Estimate Arrow memory size for one table."""
        return int(table.nbytes)

    def split_table_by_sample_max_bytes(self, table: pa.Table, max_batch_bytes: int) -> list[pa.Table]:
        """Split table by bytes while keeping each sample id in one output batch."""
        if table.num_rows == 0:
            return [table]
        row_indices_by_sample: OrderedDict[str, list[int]] = OrderedDict()
        for idx, sample_id in enumerate(table["sample_id"].to_pylist()):
            sid = str(sample_id)
            row_indices_by_sample.setdefault(sid, [])
            row_indices_by_sample[sid].append(idx)
        tables_by_sample = [
            table.take(pa.array(indices, type=pa.int64()))
            for indices in row_indices_by_sample.values()
        ]
        grouped_batches = split_by_chunk_size(
            tables_by_sample,
            max_batch_bytes,
            custom_size_func=self.table_nbytes,
        )
        out: list[pa.Table] = []
        for batch_tables in grouped_batches:
            out.append(pa.concat_tables(batch_tables) if batch_tables else pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA))
        return out or [pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA)]

    def infer_sample_type(self, table: pa.Table, sample_id: str) -> str:
        """Infer sample type label for one sample id."""
        sample_rows = table.filter(pc.equal(table["sample_id"], sample_id))
        modalities = [str(v) for v in sample_rows["modality"].to_pylist()]
        if len(modalities) == 1:
            return "single"
        if len(modalities) == _PAIR_ELEMENT_COUNT and sorted(modalities) == ["image", "text"]:
            return "pair"
        return "interleaved"

    # Shared row builders ------------------------------------------------------
    def _text_row(self, sid: str, position: int, source_shard: str, content_type: str, text_content: str) -> Row:
        """Build one normalized text row."""
        return {
            "sample_id": sid,
            "position": position,
            "modality": "text",
            "content_type": content_type,
            "text_content": text_content,
            "binary_content": None,
            "source_id": sid,
            "source_shard": source_shard,
            "content_path": None,
            "content_key": None,
        }

    def _image_row(
        self,
        sid: str,
        position: int,
        source: RowSource,
        content_key: str | None,
        binary_content: bytes | None = None,
    ) -> Row:
        """Build one normalized image row."""
        return {
            "sample_id": sid,
            "position": position,
            "modality": "image",
            "content_type": content_type_from_name(content_key or ""),
            "text_content": None,
            "binary_content": binary_content,
            "source_id": source.source_id or sid,
            "source_shard": source.source_shard,
            "content_path": source.content_path,
            "content_key": content_key,
        }

    # Internal assembly helpers ------------------------------------------------
    def _task_metadata(self, task: FileGroupTask) -> dict[str, Any]:
        return {**task._metadata, "storage_options": dict(self.storage_options)}

    @staticmethod
    def _metadata_map_from_tables(metadata_tables: list[pa.Table]) -> dict[str, str]:
        metadata_by_sample: dict[str, str] = {}
        for metadata_table in metadata_tables:
            if metadata_table.num_rows == 0:
                continue
            for row in metadata_table.to_pylist():
                sample_id = str(row["sample_id"])
                metadata_json = row.get("metadata_json")
                if isinstance(metadata_json, str):
                    metadata_by_sample.setdefault(sample_id, metadata_json)
        return metadata_by_sample

    def _build_batch(
        self,
        task: FileGroupTask,
        table: pa.Table,
        metadata_by_sample: dict[str, str],
        batch_index: int,
        split_output: bool,
    ) -> MultimodalBatch:
        sample_ids = [str(v) for v in pc.unique(table["sample_id"]).to_pylist()] if table.num_rows else []
        metadata_rows = [
            {
                "sample_id": sid,
                "sample_type": self.infer_sample_type(table, sid),
                "metadata_json": metadata_by_sample.get(sid),
            }
            for sid in sample_ids
        ]
        metadata_table = (
            pa.Table.from_pylist(metadata_rows, schema=METADATA_SCHEMA)
            if metadata_rows
            else pa.Table.from_pylist([], schema=METADATA_SCHEMA)
        )
        task_id = task.task_id if not split_output else f"{task.task_id}.part_{batch_index:05d}"
        return MultimodalBatch(
            task_id=task_id,
            dataset_name=task.dataset_name,
            data=table,
            metadata_index=metadata_table,
            _metadata=self._task_metadata(task),
            _stage_perf=task._stage_perf,
        )


@dataclass
class BaseMultimodalWriterStage(ProcessingStage[MultimodalBatch, FileGroupTask], ABC):
    """Base writer contract for multimodal file formats.

    Custom writer implementations must define:
    - ``_write_data_artifact(task, output_path)``:
      format-specific data writing implementation.

    """

    output_path: str | None = None
    data_suffix: str = "parquet"
    metadata_format: Literal["parquet", "arrow"] = "parquet"
    mode: Literal["overwrite", "error"] = "overwrite"
    storage_options: dict[str, Any] = field(default_factory=dict)
    _base_output_path: str = field(init=False, repr=False)

    # Public stage contract ----------------------------------------------------
    def __post_init__(self) -> None:
        if self.output_path is None:
            msg = f"{self.__class__.__name__} requires output_path"
            raise ValueError(msg)
        self._base_output_path = self.output_path
        mode = self.mode.strip().lower()
        if mode not in _SUPPORTED_WRITE_MODES:
            msg = f"Unsupported mode='{self.mode}'. Expected one of: overwrite, error"
            raise ValueError(msg)
        self.mode = mode  # type: ignore[assignment]
        self._configure_writer()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: MultimodalBatch) -> FileGroupTask:
        data_output_path = resolve_task_scoped_output_path(
            base_output_path=self._base_output_path,
            task_id=task.task_id,
            default_suffix=self.data_suffix,
        )
        metadata_output_path = resolve_sidecar_output_path(
            primary_output_path=data_output_path,
            sidecar_tag="metadata",
            sidecar_suffix="parquet" if self.metadata_format == "parquet" else "arrow",
        )

        self._enforce_write_mode(data_output_path)
        self._enforce_write_mode(metadata_output_path)
        self._write_data_artifact(task, data_output_path)
        self._write_metadata_table(self._build_metadata_table(task), metadata_output_path)
        return self._as_file_group_task(task, [data_output_path, metadata_output_path])

    @abstractmethod
    def _write_data_artifact(self, task: MultimodalBatch, output_path: str) -> None:
        """Write primary data artifact."""

    def _configure_writer(self) -> None:
        """Hook for subclasses to validate and configure writer-specific settings."""

    # Shared writer helpers ----------------------------------------------------
    def _build_output_table(self, task: MultimodalBatch) -> pa.Table:
        """Build normalized multimodal data table for writer output."""
        table = sort_multimodal_table(task.data)
        if table.schema.equals(MULTIMODAL_SCHEMA):
            return table

        missing = [name for name in MULTIMODAL_SCHEMA.names if name not in table.column_names]
        if missing:
            msg = f"{self.__class__.__name__} requires columns: {missing}"
            raise ValueError(msg)
        return table.select(MULTIMODAL_SCHEMA.names).cast(MULTIMODAL_SCHEMA)

    def _write_tabular(self, table: pa.Table, output_path: str, format_name: Literal["parquet", "arrow"]) -> None:
        fs, fs_path = resolve_fs_and_path(output_path, self.storage_options)
        ensure_parent_directory(fs, fs_path)
        if format_name == "parquet":
            pq.write_table(table, fs_path, filesystem=fs)
            return
        with fs.open(fs_path, "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    def _write_tabular_data_artifact(
        self,
        task: MultimodalBatch,
        output_path: str,
        format_name: Literal["parquet", "arrow"],
    ) -> None:
        """Write normalized multimodal data table as parquet/arrow artifact."""
        self._write_tabular(self._build_output_table(task), output_path, format_name)

    def _write_metadata_table(self, table: pa.Table, output_path: str) -> None:
        self._write_tabular(table, output_path, self.metadata_format)

    def _enforce_write_mode(self, output_path: str) -> None:
        if self.mode != "error":
            return
        fs, fs_path = resolve_fs_and_path(output_path, self.storage_options)
        if fs.exists(fs_path):
            msg = f"Output path already exists: {output_path}"
            raise FileExistsError(msg)

    @staticmethod
    def _build_metadata_table(task: MultimodalBatch) -> pa.Table:
        if task.metadata_index is None:
            return pa.Table.from_pylist([], schema=METADATA_SCHEMA)
        if "sample_id" in task.metadata_index.column_names:
            return task.metadata_index.sort_by([("sample_id", "ascending")])
        return task.metadata_index

    @staticmethod
    def _as_file_group_task(task: MultimodalBatch, output_paths: list[str]) -> FileGroupTask:
        data_output_path, metadata_output_path = output_paths
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=output_paths,
            _metadata={
                **task._metadata,
                "data_output_path": data_output_path,
                "metadata_output_path": metadata_output_path,
            },
            _stage_perf=task._stage_perf,
        )
