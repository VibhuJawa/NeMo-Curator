# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal.io.base import BaseMultimodalReaderStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch, _EmptyTask
from nemo_curator.tasks.multimodal import METADATA_SCHEMA, MULTIMODAL_SCHEMA
from nemo_curator.utils.file_utils import resolve_fs_and_path

_DEFAULT_PARQUET_EXTENSIONS: Final[list[str]] = [".parquet"]


@dataclass
class ParquetMultimodalReaderStage(BaseMultimodalReaderStage):
    """Read normalized multimodal parquet artifacts into ``MultimodalBatch`` outputs."""

    metadata_paths_by_data_path: dict[str, str] = field(default_factory=dict)
    name: str = "parquet_multimodal_reader"

    def __post_init__(self) -> None:
        if self.max_batch_bytes is not None and self.max_batch_bytes <= 0:
            msg = f"max_batch_bytes must be > 0, got {self.max_batch_bytes}"
            raise ValueError(msg)
        for data_path, metadata_path in self.metadata_paths_by_data_path.items():
            if not data_path or not metadata_path:
                msg = "metadata_paths_by_data_path must contain non-empty data/metadata path pairs"
                raise ValueError(msg)

    def process(
        self,
        task: FileGroupTask | tuple[FileGroupTask, FileGroupTask],
    ) -> MultimodalBatch | list[MultimodalBatch]:
        if not isinstance(task, tuple):
            return super().process(task)

        data_task, metadata_task = task
        scoped_metadata_map = self._metadata_map_from_task_pair(data_task, metadata_task)
        original_map = self.metadata_paths_by_data_path
        self.metadata_paths_by_data_path = scoped_metadata_map
        try:
            return super().process(data_task)
        finally:
            self.metadata_paths_by_data_path = original_map

    @staticmethod
    def _metadata_map_from_task_pair(data_task: FileGroupTask, metadata_task: FileGroupTask) -> dict[str, str]:
        if len(data_task.data) != len(metadata_task.data):
            msg = (
                "Data and metadata file groups must have matching lengths: "
                f"{len(data_task.data)} != {len(metadata_task.data)}"
            )
            raise ValueError(msg)
        return dict(zip(data_task.data, metadata_task.data, strict=True))

    def read_tables_and_metadata(self, source_path: str) -> tuple[pa.Table, pa.Table]:
        data_table = self._normalize_data_table(self._read_parquet_table(source_path))
        metadata_table = self._read_metadata_for_source(source_path)
        return data_table, metadata_table

    def _read_parquet_table(self, source_path: str) -> pa.Table:
        fs, fs_path = resolve_fs_and_path(source_path, self.storage_options)
        return pq.read_table(fs_path, filesystem=fs)

    def _read_metadata_for_source(self, source_path: str) -> pa.Table:
        metadata_path = self.metadata_paths_by_data_path.get(source_path)
        if metadata_path is None:
            msg = f"No metadata parquet path configured for source '{source_path}'"
            raise ValueError(msg)
        fs, fs_path = resolve_fs_and_path(metadata_path, self.storage_options)
        if not fs.exists(fs_path):
            msg = f"Metadata parquet file does not exist for source '{source_path}': {metadata_path}"
            raise FileNotFoundError(msg)
        return self._normalize_metadata_table(pq.read_table(fs_path, filesystem=fs))

    def _normalize_data_table(self, table: pa.Table) -> pa.Table:
        missing = [name for name in MULTIMODAL_SCHEMA.names if name not in table.column_names]
        if missing:
            msg = f"ParquetMultimodalReaderStage requires columns: {missing}"
            raise ValueError(msg)
        return table.select(MULTIMODAL_SCHEMA.names).cast(MULTIMODAL_SCHEMA)

    def _normalize_metadata_table(self, table: pa.Table) -> pa.Table:
        if "sample_id" not in table.column_names:
            msg = "ParquetMultimodalReaderStage metadata sidecar must contain 'sample_id' column"
            raise ValueError(msg)
        source = table
        if "sample_type" not in source.column_names:
            source = source.append_column("sample_type", pa.nulls(source.num_rows, type=pa.string()))
        if "metadata_json" not in source.column_names:
            source = source.append_column("metadata_json", pa.nulls(source.num_rows, type=pa.string()))
        return source.select(METADATA_SCHEMA.names).cast(METADATA_SCHEMA)


@dataclass
class ParquetMultimodalReader(CompositeStage[_EmptyTask, MultimodalBatch]):
    """High-level parquet reader for multimodal row tables."""

    file_paths: str | list[str]
    metadata_file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: list(_DEFAULT_PARQUET_EXTENSIONS))
    limit: int | None = None
    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)
    _metadata_paths_by_data_path: dict[str, str] = field(init=False, repr=False, default_factory=dict)
    name: str = "parquet_multimodal_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self._metadata_paths_by_data_path = self._build_metadata_paths_by_data_path()

    def _build_metadata_paths_by_data_path(self) -> dict[str, str]:
        if isinstance(self.file_paths, str):
            if not self.file_paths.endswith(".parquet"):
                msg = (
                    "When file_paths is a string, it must point to a .parquet file. "
                    "Use an explicit list of parquet file paths when reading multiple files."
                )
                raise ValueError(msg)
            if not isinstance(self.metadata_file_paths, str):
                msg = "metadata_file_paths must be a string when file_paths is a single string path"
                raise TypeError(msg)
            return {self.file_paths: self.metadata_file_paths}

        if isinstance(self.metadata_file_paths, str):
            msg = "metadata_file_paths must be a list when file_paths is a list of paths"
            raise TypeError(msg)
        if len(self.file_paths) != len(self.metadata_file_paths):
            msg = (
                "metadata_file_paths length must match file_paths length: "
                f"{len(self.metadata_file_paths)} != {len(self.file_paths)}"
            )
            raise ValueError(msg)
        return dict(zip(self.file_paths, self.metadata_file_paths, strict=True))

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
                limit=self.limit,
            ),
            ParquetMultimodalReaderStage(
                metadata_paths_by_data_path=self._metadata_paths_by_data_path,
                max_batch_bytes=self.max_batch_bytes,
                storage_options=self.storage_options,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read multimodal parquet files from {self.file_paths}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        if self.limit is not None:
            parts.append(f"limited to {self.limit} partitions")
        return ", ".join(parts)
