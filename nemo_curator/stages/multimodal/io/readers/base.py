# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA
from nemo_curator.utils.file_utils import resolve_fs_and_path
from nemo_curator.utils.grouping import split_by_chunk_size
from nemo_curator.utils.multimodal_utils import build_metadata_row, sort_multimodal_table
from nemo_curator.utils.webdataset_utils import content_type_from_name


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
class BaseMultimodalReaderStage(ProcessingStage[FileGroupTask, MultimodalBatch], ABC):
    """Base stage contract for multimodal readers.

    Subclasses implement ``read_data(data_path)`` and return
    one data table normalized to ``MULTIMODAL_SCHEMA``.

    Main extension hook:
        - ``read_data``: required. This is the method dataset-specific
          readers implement.

    Common helper methods for subclass implementations:
        - ``_text_row`` / ``_image_row`` for normalized row construction
        - ``_rows_to_table`` for schema-safe table creation
        - ``_read_parquet_table`` for storage-option-aware parquet reads
        - ``_json_or_none`` for optional metadata JSON serialization

    Optional extension hooks:
        - ``split_table`` / ``table_nbytes`` to customize batch fanout behavior.
    """

    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate shared reader options."""
        if self.max_batch_bytes is not None and self.max_batch_bytes <= 0:
            msg = f"max_batch_bytes must be > 0, got {self.max_batch_bytes}"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Declare one input edge carrying data files."""
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Declare multimodal row outputs."""
        return ["data"], list(MULTIMODAL_SCHEMA.names)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Mark stage as fanout when byte-based splitting is enabled."""
        if self.max_batch_bytes is None:
            return {}
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: FileGroupTask) -> MultimodalBatch | list[MultimodalBatch]:
        """Run shared read orchestration around ``read_data``.

        This method is intentionally non-abstract so subclasses only implement
        source parsing logic while batching, sorting, and metadata alignment stay
        centralized in base code.
        """
        data_tables = [self.read_data(data_path) for data_path in task.data]
        return self._build_batches_from_tables(task, data_tables)

    def _build_batches_from_tables(
        self,
        task: FileGroupTask,
        data_tables: list[pa.Table],
    ) -> MultimodalBatch | list[MultimodalBatch]:
        table = self._concat_data_tables_or_empty(data_tables)
        table = sort_multimodal_table(table)
        table_splits = self.split_table(table)
        batches = [
            self._build_batch(
                task=task,
                table=batch_table,
                batch_index=batch_index,
                split_output=self.max_batch_bytes is not None,
            )
            for batch_index, batch_table in enumerate(table_splits)
        ]
        return batches[0] if self.max_batch_bytes is None else batches

    @abstractmethod
    def read_data(self, data_path: str) -> pa.Table:
        """Read one source path into one normalized data table."""

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
        row_indices_by_sample: dict[str, list[int]] = {}
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

    def _text_row(  # noqa: PLR0913
        self,
        sid: str,
        position: int,
        source_shard: str,
        content_type: str,
        text_content: str,
        element_metadata_json: str | None = None,
        source_id: str | None = None,
    ) -> dict[str, object]:
        """Build one normalized text row payload."""
        return {
            "sample_id": sid,
            "position": position,
            "modality": "text",
            "content_type": content_type,
            "text_content": text_content,
            "binary_content": None,
            "element_metadata_json": element_metadata_json,
            "source_id": source_id or sid,
            "source_shard": source_shard,
            "content_path": None,
            "content_key": None,
        }

    def _image_row(  # noqa: PLR0913
        self,
        sid: str,
        position: int,
        source: RowSource,
        content_key: str | None,
        binary_content: bytes | None = None,
        element_metadata_json: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, object]:
        """Build one normalized image row payload."""
        return {
            "sample_id": sid,
            "position": position,
            "modality": "image",
            "content_type": content_type if content_type is not None else content_type_from_name(content_key or ""),
            "text_content": None,
            "binary_content": binary_content,
            "element_metadata_json": element_metadata_json,
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
    def _concat_data_tables_or_empty(tables: list[pa.Table]) -> pa.Table:
        """Concatenate data tables, or return empty multimodal table."""
        if not tables:
            return BaseMultimodalReaderStage._empty_data_table()
        return pa.concat_tables(tables)

    @staticmethod
    def _empty_data_table() -> pa.Table:
        return pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA)

    def _read_parquet_table(self, source_path: str, columns: list[str] | None = None) -> pa.Table:
        """Read one parquet table from local/remote storage."""
        fs, fs_path = resolve_fs_and_path(source_path, self.storage_options)
        return pq.read_table(fs_path, filesystem=fs, columns=columns)

    @staticmethod
    def _json_or_none(value: object | None) -> str | None:
        """Serialize value to JSON string, or return None for missing payload."""
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=True)

    def _task_metadata(self, task: FileGroupTask) -> dict[str, Any]:
        """Propagate task metadata and attach storage options used for reads."""
        return {**task._metadata, "storage_options": dict(self.storage_options)}

    def _build_batch(
        self,
        task: FileGroupTask,
        table: pa.Table,
        batch_index: int,
        split_output: bool,
    ) -> MultimodalBatch:
        """Assemble one ``MultimodalBatch`` from normalized rows."""
        task_id = task.task_id if not split_output else f"{task.task_id}.part_{batch_index:05d}"
        return MultimodalBatch(
            task_id=task_id,
            dataset_name=task.dataset_name,
            data=table,
            _metadata=self._task_metadata(task),
            _stage_perf=task._stage_perf,
        )

    @staticmethod
    def _metadata_row(
        *,
        sid: str,
        metadata_json: str,
        sample_type: str | None = None,
        source_shard: str,
        source_id: str | None = None,
    ) -> dict[str, object]:
        """Build one normalized sample-level metadata row in the main table."""
        return build_metadata_row(
            sample_id=sid,
            metadata_json=metadata_json,
            sample_type=sample_type,
            source_shard=source_shard,
            source_id=source_id,
        )
