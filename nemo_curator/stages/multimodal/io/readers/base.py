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

Row = dict[str, object]
_PAIR_ELEMENT_COUNT = 2


@dataclass
class RowSource:
    """Generic source context for building multimodal rows."""

    source_shard: str
    content_path: str
    source_id: str | None = None


@dataclass
class BaseMultimodalReaderStage(ProcessingStage[FileGroupTask, MultimodalBatch], ABC):
    """Base reader contract for multimodal file formats."""

    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data", "metadata_index"], list(MULTIMODAL_SCHEMA.names)

    def ray_stage_spec(self) -> dict[str, Any]:
        if self.max_batch_bytes is None:
            return {}
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def process(self, task: FileGroupTask) -> MultimodalBatch | list[MultimodalBatch]:
        data_tables: list[pa.Table] = []
        metadata_tables: list[pa.Table] = []
        for source_path in task.data:
            shard_data_table, shard_metadata_table = self.read_tables_and_metadata(source_path)
            data_tables.append(shard_data_table)
            metadata_tables.append(shard_metadata_table)
        return self._build_batches_from_tables(task, data_tables, metadata_tables)

    def _build_batches_from_tables(
        self,
        task: FileGroupTask,
        data_tables: list[pa.Table],
        metadata_tables: list[pa.Table],
    ) -> MultimodalBatch | list[MultimodalBatch]:
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

    def split_table(self, table: pa.Table) -> list[pa.Table]:
        if self.max_batch_bytes is None:
            return [table]
        return self.split_table_by_sample_max_bytes(table, self.max_batch_bytes)

    def table_nbytes(self, table: pa.Table) -> int:
        return int(table.nbytes)

    def split_table_by_sample_max_bytes(self, table: pa.Table, max_batch_bytes: int) -> list[pa.Table]:
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
            out.append(pa.concat_tables(batch_tables) if batch_tables else pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA))
        return out or [pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA)]

    def infer_sample_type(self, table: pa.Table, sample_id: str) -> str:
        sample_rows = table.filter(pc.equal(table["sample_id"], sample_id))
        modalities = [str(v) for v in sample_rows["modality"].to_pylist()]
        if len(modalities) == 1:
            return "single"
        if len(modalities) == _PAIR_ELEMENT_COUNT and sorted(modalities) == ["image", "text"]:
            return "pair"
        return "interleaved"

    def _text_row(self, sid: str, position: int, source_shard: str, content_type: str, text_content: str) -> Row:
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
