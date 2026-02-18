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

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal.io.readers.base import BaseMultimodalReaderStage
from nemo_curator.tasks import MultimodalBatch, _EmptyTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA
from nemo_curator.utils.multimodal_utils import cast_required_fields

_DEFAULT_PARQUET_EXTENSIONS: Final[list[str]] = [".parquet"]


@dataclass
class ParquetMultimodalReaderStage(BaseMultimodalReaderStage):
    """Read normalized multimodal parquet artifacts into ``MultimodalBatch`` outputs.

    Base-class extension method implemented here:
    - ``read_data``: reads one parquet data file and normalizes it to the
      multimodal schema.
    """

    columns: list[str] | None = None
    name: str = "parquet_multimodal_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.columns = self._validate_column_selection(self.columns, field_name="data.columns")
        if self.columns is not None:
            missing_required = [name for name in MULTIMODAL_SCHEMA.names if name not in self.columns]
            if missing_required:
                msg = (
                    "ParquetMultimodalReaderStage columns must include all multimodal required columns. "
                    f"Missing: {missing_required}"
                )
                raise ValueError(msg)

    @staticmethod
    def _validate_column_selection(columns: list[str] | None, *, field_name: str) -> list[str] | None:
        """Validate optional column selection for data and metadata inputs."""
        if columns is None:
            return None
        if len(columns) == 0:
            msg = f"{field_name} must be a non-empty list when provided"
            raise ValueError(msg)
        seen: set[str] = set()
        normalized: list[str] = []
        for column in columns:
            if not isinstance(column, str) or not column:
                msg = f"{field_name} entries must be non-empty strings"
                raise ValueError(msg)
            if column not in seen:
                seen.add(column)
                normalized.append(column)
        return normalized

    def read_data(self, data_path: str) -> pa.Table:
        """Implement ``BaseMultimodalReaderStage.read_data`` for parquet."""
        return self._normalize_data_table(self._read_parquet_table(data_path, columns=self.columns))

    def _normalize_data_table(self, table: pa.Table) -> pa.Table:
        source = self._ensure_optional_string_column(table, "element_metadata_json")
        self._ensure_required_columns(source, MULTIMODAL_SCHEMA.names)
        return cast_required_fields(source, MULTIMODAL_SCHEMA)

    @staticmethod
    def _ensure_optional_string_column(table: pa.Table, column_name: str) -> pa.Table:
        if column_name in table.column_names:
            return table
        return table.append_column(column_name, pa.nulls(table.num_rows, type=pa.string()))

    @staticmethod
    def _ensure_required_columns(
        table: pa.Table,
        required_columns: list[str],
        *,
        context: str = "ParquetMultimodalReaderStage",
    ) -> None:
        missing = [name for name in required_columns if name not in table.column_names]
        if missing:
            msg = f"{context} requires columns: {missing}"
            raise ValueError(msg)


@dataclass
class ParquetMultimodalReader(CompositeStage[_EmptyTask, MultimodalBatch]):
    """Composite parquet reader wiring partitioning + ``ParquetMultimodalReaderStage``."""

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: list(_DEFAULT_PARQUET_EXTENSIONS))
    limit: int | None = None
    columns: list[str] | None = None
    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)
    name: str = "parquet_multimodal_reader"

    def __post_init__(self) -> None:
        super().__init__()

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
        return ", ".join(parts)
