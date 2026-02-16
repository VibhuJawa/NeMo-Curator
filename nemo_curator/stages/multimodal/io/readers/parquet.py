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
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal.io.readers.base import BaseMultimodalReaderStage
from nemo_curator.tasks import MultimodalBatch, _EmptyTask
from nemo_curator.tasks.multimodal import METADATA_SCHEMA, MULTIMODAL_SCHEMA
from nemo_curator.utils.file_utils import resolve_fs_and_path

_DEFAULT_PARQUET_EXTENSIONS: Final[list[str]] = [".parquet"]


@dataclass
class ParquetMultimodalReaderStage(BaseMultimodalReaderStage):
    """Read normalized multimodal parquet artifacts into ``MultimodalBatch`` outputs.

    Metadata is accepted only through explicit
    ``(data_task, metadata_task | None)`` input pairing.
    """

    name: str = "parquet_multimodal_reader"

    def read_source_tables(self, data_path: str, metadata_path: str | None) -> tuple[pa.Table, pa.Table]:
        data_table = self._normalize_data_table(self._read_parquet_table(data_path))
        if metadata_path is None:
            metadata_table = self._empty_metadata_table()
        else:
            metadata_table = self._read_metadata_table(metadata_path)
        return data_table, metadata_table

    def _read_parquet_table(self, source_path: str) -> pa.Table:
        fs, fs_path = resolve_fs_and_path(source_path, self.storage_options)
        return pq.read_table(fs_path, filesystem=fs)

    def _read_metadata_table(self, metadata_path: str) -> pa.Table:
        fs, fs_path = resolve_fs_and_path(metadata_path, self.storage_options)
        if not fs.exists(fs_path):
            logger.warning(
                "Skipping missing metadata parquet: {}",
                metadata_path,
            )
            return self._empty_metadata_table()
        return self._normalize_metadata_table(pq.read_table(fs_path, filesystem=fs))

    def _normalize_data_table(self, table: pa.Table) -> pa.Table:
        if table.schema.equals(MULTIMODAL_SCHEMA):
            return table
        missing = [name for name in MULTIMODAL_SCHEMA.names if name not in table.column_names]
        if missing:
            msg = f"ParquetMultimodalReaderStage requires columns: {missing}"
            raise ValueError(msg)
        return table.select(MULTIMODAL_SCHEMA.names).cast(MULTIMODAL_SCHEMA)

    def _normalize_metadata_table(self, table: pa.Table) -> pa.Table:
        if table.schema.equals(METADATA_SCHEMA):
            return table
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
    """Composite parquet reader for multimodal row tables."""

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: list(_DEFAULT_PARQUET_EXTENSIONS))
    limit: int | None = None
    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)
    name: str = "parquet_multimodal_reader"

    def __post_init__(self) -> None:
        super().__init__()
        if isinstance(self.file_paths, str) and not self.file_paths.endswith(".parquet"):
            msg = (
                "When file_paths is a string, it must point to a .parquet file. "
                "Use an explicit list of parquet file paths when reading multiple files."
            )
            raise ValueError(msg)

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
