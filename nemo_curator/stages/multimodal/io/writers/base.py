# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA
from nemo_curator.utils.file_utils import (
    ensure_parent_directory,
    resolve_fs_and_path,
    resolve_task_scoped_output_path,
)
from nemo_curator.utils.multimodal_utils import cast_required_fields, sort_multimodal_table

_SUPPORTED_WRITE_MODES: set[str] = {"overwrite", "error", "ignore"}


@dataclass
class BaseMultimodalWriterStage(ProcessingStage[MultimodalBatch, FileGroupTask], ABC):
    """Base stage contract for multimodal writers.

    Main extension hooks for subclasses:
    - ``write_data`` (required): serialize the primary data artifact.
    - ``configure`` (optional): validate options and derive format-specific
      settings such as suffixes.
    - ``prepare_task`` (optional): transform tasks before write
      (for example materialize/dematerialize or row filtering policies).

    ``process`` stays centralized so path resolution and write-mode checks
    are shared across all writer implementations.

    Write mode behavior:
    - ``overwrite``: always write outputs (replacing existing files).
    - ``error``: fail if any output file already exists.
    - ``ignore``: return existing output when file already exists.
    """

    output_path: str | None = None
    data_suffix: str = "parquet"
    mode: Literal["overwrite", "error", "ignore"] = "overwrite"
    storage_options: dict[str, Any] = field(default_factory=dict)
    _base_output_path: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate common writer options and initialize derived config."""
        if self.output_path is None:
            msg = f"{self.__class__.__name__} requires output_path"
            raise ValueError(msg)
        self._base_output_path = self.output_path
        mode = self.mode.strip().lower()
        if mode not in _SUPPORTED_WRITE_MODES:
            msg = f"Unsupported mode='{self.mode}'. Expected one of: overwrite, error, ignore"
            raise ValueError(msg)
        self.mode = mode  # type: ignore[assignment]
        self.configure()

    def inputs(self) -> tuple[list[str], list[str]]:
        """Declare one multimodal data input edge."""
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Declare one file-group output edge."""
        return ["data"], []

    def process(self, task: MultimodalBatch) -> FileGroupTask:
        """Run shared write orchestration and emit one output ``FileGroupTask``.

        Subclasses focus on format-specific behavior via hook methods while this
        method handles task-scoped output paths.
        """
        write_task = self.prepare_task(task)
        data_output_path = resolve_task_scoped_output_path(
            base_output_path=self._base_output_path,
            task_id=write_task.task_id,
            default_suffix=self.data_suffix,
        )

        if self.mode == "ignore":
            if self._path_exists(data_output_path):
                return self._as_file_group_task(write_task, [data_output_path])

        self._enforce_write_mode(data_output_path)
        self.write_data(write_task, data_output_path)
        return self._as_file_group_task(write_task, [data_output_path])

    def prepare_task(self, task: MultimodalBatch) -> MultimodalBatch:
        """Prepare batch before writing (materialize/dematerialize policy hook)."""
        return task

    @abstractmethod
    def write_data(self, task: MultimodalBatch, output_path: str) -> None:
        """Write primary data artifact for one batch."""

    def configure(self) -> None:
        """Configure writer-specific defaults after shared option validation."""

    def _build_output_table(self, task: MultimodalBatch) -> pa.Table:
        """Normalize and schema-cast batch rows to ``MULTIMODAL_SCHEMA``."""
        table = sort_multimodal_table(task.data)
        if table.schema.equals(MULTIMODAL_SCHEMA):
            return table

        missing = [name for name in MULTIMODAL_SCHEMA.names if name not in table.column_names]
        if missing:
            msg = f"{self.__class__.__name__} requires columns: {missing}"
            raise ValueError(msg)
        return cast_required_fields(table, MULTIMODAL_SCHEMA)

    def _write_tabular(self, table: pa.Table, output_path: str, format_name: Literal["parquet", "arrow"]) -> None:
        """Write Arrow table to parquet or arrow artifact path."""
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
        """Write normalized multimodal rows using a tabular artifact format."""
        self._write_tabular(self._build_output_table(task), output_path, format_name)

    def _enforce_write_mode(self, output_path: str) -> None:
        """Apply overwrite/error collision policy to target path."""
        if self.mode != "error":
            return
        if self._path_exists(output_path):
            msg = f"Output path already exists: {output_path}"
            raise FileExistsError(msg)

    def _path_exists(self, path: str) -> bool:
        fs, fs_path = resolve_fs_and_path(path, self.storage_options)
        return fs.exists(fs_path)

    @staticmethod
    def _filter_task_rows(task: MultimodalBatch, keep_mask: pa.Array) -> MultimodalBatch:
        """Filter task rows."""
        data = task.data.filter(keep_mask)
        return BaseMultimodalWriterStage._rebuild_task(task, data)

    @staticmethod
    def _rebuild_task(
        task: MultimodalBatch,
        data: pa.Table,
    ) -> MultimodalBatch:
        """Rebuild a task preserving execution metadata."""
        return task.__class__(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=data,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    @staticmethod
    def _as_file_group_task(task: MultimodalBatch, output_paths: list[str]) -> FileGroupTask:
        """Build output ``FileGroupTask`` with data artifact path."""
        data_output_path = output_paths[0]
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=output_paths,
            _metadata={
                **task._metadata,
                "data_output_path": data_output_path,
            },
            _stage_perf=task._stage_perf,
        )
