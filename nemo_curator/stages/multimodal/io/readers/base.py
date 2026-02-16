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
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch
from nemo_curator.tasks.multimodal import METADATA_SCHEMA, MULTIMODAL_SCHEMA
from nemo_curator.utils.grouping import split_by_chunk_size
from nemo_curator.utils.multimodal_utils import sort_multimodal_table
from nemo_curator.utils.webdataset_utils import content_type_from_name

ReaderTask = FileGroupTask | tuple[FileGroupTask, FileGroupTask | None]
_PAIR_ELEMENT_COUNT = 2


@dataclass
class RowSource:
    """Source context used by shared multimodal row builders.

    Attributes:
        source_shard: Human-readable shard identifier (for example tar filename).
        content_path: Fully qualified source artifact path.
        source_id: Optional source-level identifier to preserve in normalized rows.
    """

    source_shard: str
    content_path: str
    source_id: str | None = None


@dataclass
class BaseMultimodalReaderStage(ProcessingStage[ReaderTask, MultimodalBatch], ABC):
    """Base stage contract for multimodal readers.

    Input contract:
        - ``FileGroupTask``: data-only sources.
        - ``tuple[FileGroupTask, FileGroupTask | None]``: explicit ``(data_task, metadata_task)``.

    Subclasses implement ``read_source_tables(data_path, metadata_path)`` and return:
        - data table normalized to ``MULTIMODAL_SCHEMA``
        - metadata table normalized to ``METADATA_SCHEMA`` (empty table if unavailable)
    """

    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shared reader options."""
        if self.max_batch_bytes is not None and self.max_batch_bytes <= 0:
            msg = f"max_batch_bytes must be > 0, got {self.max_batch_bytes}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Declare one input edge carrying data files (and optional paired metadata files)."""
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Declare multimodal row + metadata-index outputs."""
        return ["data", "metadata_index"], list(MULTIMODAL_SCHEMA.names)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Mark stage as fanout when byte-based splitting is enabled."""
        if self.max_batch_bytes is None:
            return {}
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: ReaderTask) -> MultimodalBatch | list[MultimodalBatch]:
        """Read all input sources and emit one batch or fanout batches."""
        data_task = task[0] if isinstance(task, tuple) else task
        data_tables: list[pa.Table] = []
        metadata_tables: list[pa.Table] = []
        for data_path, metadata_path in self._source_pairs(task):
            shard_data_table, shard_metadata_table = self.read_source_tables(data_path, metadata_path)
            data_tables.append(shard_data_table)
            metadata_tables.append(shard_metadata_table)
        return self._build_batches_from_tables(data_task, data_tables, metadata_tables)

    def _build_batches_from_tables(
        self,
        task: FileGroupTask,
        data_tables: list[pa.Table],
        metadata_tables: list[pa.Table],
    ) -> MultimodalBatch | list[MultimodalBatch]:
        table = pa.concat_tables(data_tables) if data_tables else self._empty_data_table()
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

    def _source_pairs(self, task: ReaderTask) -> list[tuple[str, str | None]]:
        """Align data and metadata sources for one ``process`` call."""
        if isinstance(task, tuple):
            data_task, metadata_task = task
            if metadata_task is None or len(metadata_task.data) == 0:
                return [(data_path, None) for data_path in data_task.data]
            if len(data_task.data) != len(metadata_task.data):
                msg = (
                    "Data and metadata file groups must have matching lengths: "
                    f"{len(data_task.data)} != {len(metadata_task.data)}"
                )
                raise ValueError(msg)
            return list(zip(data_task.data, metadata_task.data, strict=True))
        return [(data_path, self._metadata_path_for_data_path(data_path)) for data_path in task.data]

    def _metadata_path_for_data_path(self, _data_path: str) -> str | None:
        """Resolve metadata path for data-only input mode."""
        return None

    @abstractmethod
    def read_source_tables(self, data_path: str, metadata_path: str | None) -> tuple[pa.Table, pa.Table]:
        """Read one data source (+ optional metadata source) into normalized tables."""

    def split_table(self, table: pa.Table) -> list[pa.Table]:
        """Split one normalized data table into batch tables."""
        if self.max_batch_bytes is None:
            return [table]
        return self.split_table_by_sample_max_bytes(table, self.max_batch_bytes)

    def table_nbytes(self, table: pa.Table) -> int:
        """Estimate in-memory size of one table for split decisions."""
        return int(table.nbytes)

    def split_table_by_sample_max_bytes(self, table: pa.Table, max_batch_bytes: int) -> list[pa.Table]:
        """Split table by sample groups while preserving sample row locality."""
        if table.num_rows == 0:
            return [table]
        row_indices_by_sample: OrderedDict[str, list[int]] = OrderedDict()
        for idx, sample_id in enumerate(table["sample_id"].to_pylist()):
            sid = str(sample_id)
            row_indices_by_sample.setdefault(sid, [])
            row_indices_by_sample[sid].append(idx)
        tables_by_sample = [table.take(pa.array(indices, type=pa.int64())) for indices in row_indices_by_sample.values()]
        grouped_batches = split_by_chunk_size(
            tables_by_sample,
            max_batch_bytes,
            custom_size_func=self.table_nbytes,
        )
        out: list[pa.Table] = []
        for batch_tables in grouped_batches:
            out.append(pa.concat_tables(batch_tables) if batch_tables else self._empty_data_table())
        return out or [self._empty_data_table()]

    def infer_sample_type(self, table: pa.Table, sample_id: str) -> str:
        """Infer ``single``/``pair``/``interleaved`` from sample modalities."""
        sample_rows = table.filter(pc.equal(table["sample_id"], sample_id))
        modalities = [str(v) for v in sample_rows["modality"].to_pylist()]
        if len(modalities) == 1:
            return "single"
        if len(modalities) == _PAIR_ELEMENT_COUNT and sorted(modalities) == ["image", "text"]:
            return "pair"
        return "interleaved"

    def _text_row(
        self,
        sid: str,
        position: int,
        source_shard: str,
        content_type: str,
        text_content: str,
    ) -> dict[str, object]:
        """Build one normalized text row payload."""
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
    ) -> dict[str, object]:
        """Build one normalized image row payload."""
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

    @staticmethod
    def _rows_to_table(rows: list[dict[str, object]]) -> pa.Table:
        """Convert normalized row payloads into ``MULTIMODAL_SCHEMA`` table."""
        if not rows:
            return BaseMultimodalReaderStage._empty_data_table()
        return pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)

    @staticmethod
    def _empty_data_table() -> pa.Table:
        return pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA)

    @staticmethod
    def _empty_metadata_table() -> pa.Table:
        return pa.Table.from_pylist([], schema=METADATA_SCHEMA)

    def _task_metadata(self, task: FileGroupTask) -> dict[str, Any]:
        """Propagate task metadata and attach storage options used for reads."""
        return {**task._metadata, "storage_options": dict(self.storage_options)}

    @staticmethod
    def _metadata_map_from_tables(metadata_tables: list[pa.Table]) -> dict[str, str]:
        """Build first-wins sample->metadata_json map from metadata tables."""
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
        """Assemble one ``MultimodalBatch`` from normalized data and metadata."""
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
            else self._empty_metadata_table()
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
