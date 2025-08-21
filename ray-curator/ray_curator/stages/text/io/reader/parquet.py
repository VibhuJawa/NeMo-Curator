# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Parquet reader composite stage."""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ray_curator.stages.base import CompositeStage
from ray_curator.stages.file_partitioning import FilePartitioningStage
from ray_curator.tasks import DocumentBatch, _EmptyTask
from ray_curator.utils.file_utils import get_pyarrow_filesystem, pandas_select_columns, pyarrow_select_columns

from .dataframe import BaseReader


@dataclass
class ParquetReaderStage(BaseReader):
    """
    Stage that processes a group of Parquet files into a DocumentBatch.
    This stage accepts FileGroupTasks created by FilePartitioningStage
    and reads the actual file contents into DocumentBatches.

    Args:
        columns (list[str], optional): If specified, only read these columns. Defaults to None.
        reader (str, optional): Reader to use ("pyarrow" or "pandas"). Defaults to "pandas".
        read_kwargs (dict[str, Any], optional): Keyword arguments for the underlying reader. Defaults to {}.
    """

    _name: str = "parquet_reader"

    def _read_with_pandas(
        self,
        file_paths: list[str],
        storage_options: dict[str, Any],
        read_kwargs: dict[str, Any],
        columns: list[str] | None,
    ) -> pd.DataFrame | None:
        """Read Parquet files using Pandas."""

        dfs = []

        for file_path in file_paths:
            try:
                # Pandas read_parquet supports fsspec via storage_options
                if columns is None:
                    df = pd.read_parquet(file_path, storage_options=storage_options, **read_kwargs)
                else:
                    try:
                        df = pd.read_parquet(
                            file_path, columns=columns, storage_options=storage_options, **read_kwargs
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"Reading selected columns from {file_path} failed ({e}); falling back to full read"
                        )
                        df = pd.read_parquet(file_path, storage_options=storage_options, **read_kwargs)
                        df = pandas_select_columns(df, columns, file_path)
                        if df is None:
                            continue

                dfs.append(df)
                logger.debug(f"Read {len(df)} records from {file_path}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        if not dfs:
            return None

        # Concatenate all dataframes
        return pd.concat(dfs, ignore_index=True)

    def _read_with_pyarrow(
        self,
        file_paths: list[str],
        storage_options: dict[str, Any],
        read_kwargs: dict[str, Any],
        columns: list[str] | None,
    ) -> pa.Table | None:
        """Read Parquet files using PyArrow."""

        tables = []
        if self._generate_ids or self._assign_ids:
            msg = "Generating or assigning IDs is not supported for PyArrow reader"
            raise NotImplementedError(msg)

        filesystem = get_pyarrow_filesystem(file_paths, storage_options)

        for file_path in file_paths:
            try:
                if columns is None:
                    table = pq.read_table(file_path, filesystem=filesystem, **read_kwargs)
                else:
                    try:
                        table = pq.read_table(file_path, columns=columns, filesystem=filesystem, **read_kwargs)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"Reading selected columns from {file_path} failed ({e}); falling back to full read"
                        )
                        table_all = pq.read_table(file_path, filesystem=filesystem, **read_kwargs)
                        table = pyarrow_select_columns(table_all, columns, file_path)
                        if table is None:
                            continue

                tables.append(table)
                logger.debug(f"Read {len(table)} records from {file_path}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to read {file_path}: {e}")
                continue

        if not tables:
            return None
        # Concatenate all tables
        return pa.concat_tables(tables)


@dataclass
class ParquetReader(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite stage for reading Parquet files.

    This high-level stage decomposes into:
    1. FilePartitioningStage - partitions files into groups
    2. ParquetReaderStage - reads file groups into DocumentBatches
    """

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    columns: list[str] | None = None  # If specified, only read these columns
    reader: str = "pandas"  # "pandas" or "pyarrow"
    read_kwargs: dict[str, Any] | None = None
    task_type: Literal["document", "image", "video", "audio"] = "document"
    _generate_ids: bool = False
    _assign_ids: bool = False
    _name: str = "parquet_reader"

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        if self.read_kwargs is not None:
            self.storage_options = self.read_kwargs.get("storage_options", {})

    def decompose(self) -> list[ParquetReaderStage]:
        """Decompose into file partitioning and processing stages."""
        if self.task_type != "document":
            msg = f"Converting DocumentBatch to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)

        return [
            # First stage: partition files into groups
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=[
                    ".parquet"
                ],  # TODO: Expand to support other file extensions (e.g. .snappy, .gzip, etc.)
                storage_options=self.read_kwargs.get("storage_options", {}) if self.read_kwargs is not None else {},
            ),
            # Second stage: process file groups into document batches
            ParquetReaderStage(
                columns=self.columns,
                reader=self.reader,
                read_kwargs=self.read_kwargs or {},
                _generate_ids=self._generate_ids,
                _assign_ids=self._assign_ids,
            ),
        ]

    def get_description(self) -> str:
        """Get a description of this composite stage."""

        parts = [f"Read Parquet files from {self.file_paths}"]

        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")

        if self.columns:
            parts.append(f"reading columns: {self.columns}")

        return ", ".join(parts)
