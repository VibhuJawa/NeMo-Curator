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

    columns: list[str] | None = None
    metadata_columns: list[str] | None = None
    name: str = "parquet_multimodal_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.columns = self._validate_selected_columns(self.columns, option_name="columns")
        self.metadata_columns = self._validate_selected_columns(self.metadata_columns, option_name="metadata_columns")
        if self.columns is not None:
            missing_required = [name for name in MULTIMODAL_SCHEMA.names if name not in self.columns]
            if missing_required:
                msg = (
                    "ParquetMultimodalReaderStage columns must include all multimodal required columns. "
                    f"Missing: {missing_required}"
                )
                raise ValueError(msg)

    @staticmethod
    def _validate_selected_columns(columns: list[str] | None, option_name: str) -> list[str] | None:
        if columns is None:
            return None
        if len(columns) == 0:
            msg = f"{option_name} must be a non-empty list when provided"
            raise ValueError(msg)
        seen: set[str] = set()
        normalized: list[str] = []
        for column in columns:
            if not isinstance(column, str) or not column:
                msg = f"{option_name} entries must be non-empty strings"
                raise ValueError(msg)
            if column in seen:
                continue
            seen.add(column)
            normalized.append(column)
        return normalized

    def read_source_tables(self, data_path: str, metadata_path: str | None) -> tuple[pa.Table, pa.Table]:
        data_table = self._normalize_data_table(self._read_parquet_table(data_path, columns=self.columns))
        if metadata_path is None:
            metadata_table = self._empty_metadata_table()
        else:
            metadata_table = self._read_metadata_table(metadata_path)
        return data_table, metadata_table

    def _read_parquet_table(self, source_path: str, columns: list[str] | None = None) -> pa.Table:
        fs, fs_path = resolve_fs_and_path(source_path, self.storage_options)
        return pq.read_table(fs_path, filesystem=fs, columns=columns)

    def _read_metadata_table(self, metadata_path: str) -> pa.Table:
        fs, fs_path = resolve_fs_and_path(metadata_path, self.storage_options)
        if not fs.exists(fs_path):
            logger.warning(
                "Skipping missing metadata parquet: {}",
                metadata_path,
            )
            return self._empty_metadata_table()
        return self._normalize_metadata_table(pq.read_table(fs_path, filesystem=fs, columns=self.metadata_columns))

    def _normalize_data_table(self, table: pa.Table) -> pa.Table:
        missing = [name for name in MULTIMODAL_SCHEMA.names if name not in table.column_names]
        if missing:
            msg = f"ParquetMultimodalReaderStage requires columns: {missing}"
            raise ValueError(msg)
        return self._cast_required_fields(table, MULTIMODAL_SCHEMA)

    def _normalize_metadata_table(self, table: pa.Table) -> pa.Table:
        if "sample_id" not in table.column_names:
            msg = "ParquetMultimodalReaderStage metadata sidecar must contain 'sample_id' column"
            raise ValueError(msg)
        source = table
        if "sample_type" not in source.column_names:
            source = source.append_column("sample_type", pa.nulls(source.num_rows, type=pa.string()))
        if "metadata_json" not in source.column_names:
            source = source.append_column("metadata_json", pa.nulls(source.num_rows, type=pa.string()))
        return self._cast_required_fields(source, METADATA_SCHEMA)

    @staticmethod
    def _cast_required_fields(table: pa.Table, required_schema: pa.Schema) -> pa.Table:
        """Cast required fields in-place while preserving any extra columns."""
        out = table
        for required_field in required_schema:
            col_idx = out.schema.get_field_index(required_field.name)
            if col_idx < 0:
                continue
            col = out[required_field.name]
            if col.type.equals(required_field.type):
                continue
            out = out.set_column(col_idx, required_field.name, col.cast(required_field.type))
        return out


@dataclass
class ParquetMultimodalReader(CompositeStage[_EmptyTask, MultimodalBatch]):
    """Composite parquet reader for multimodal row tables."""

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: list(_DEFAULT_PARQUET_EXTENSIONS))
    limit: int | None = None
    columns: list[str] | None = None
    metadata_columns: list[str] | None = None
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
                columns=self.columns,
                metadata_columns=self.metadata_columns,
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
        if self.columns is not None:
            parts.append(f"columns={self.columns}")
        if self.metadata_columns is not None:
            parts.append(f"metadata_columns={self.metadata_columns}")
        return ", ".join(parts)
