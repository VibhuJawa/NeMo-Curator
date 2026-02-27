# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.multimodal.utils import resolve_storage_options
from nemo_curator.stages.text.io.reader.parquet import read_parquet_files
from nemo_curator.tasks import FileGroupTask, MultiBatchTask

from .base import BaseMultimodalReader


@dataclass
class MultimodalParquetReaderStage(BaseMultimodalReader):
    """Read parquet files in MULTIMODAL_SCHEMA format into MultiBatchTask.

    Delegates actual parquet I/O to the shared ``read_parquet_files()`` utility,
    then reconciles schema and splits by sample_id to produce MultiBatchTask(s).
    """

    fields: list[str] | None = None
    max_batch_bytes: int | None = None
    name: str = "multimodal_parquet_reader"

    def process(self, task: FileGroupTask) -> MultiBatchTask | list[MultiBatchTask]:
        storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)
        effective_kwargs: dict[str, Any] = dict(self.read_kwargs) if self.read_kwargs else {}
        if storage_options:
            effective_kwargs["storage_options"] = storage_options

        df = read_parquet_files(task.data, read_kwargs=effective_kwargs, fields=self.fields)

        if df.empty:
            msg = f"No data read from parquet files in task {task.task_id}"
            raise ValueError(msg)

        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.cast(self.reconcile_schema(table.schema))
        splits = split_table_by_group_max_bytes(table, "sample_id", self.max_batch_bytes)

        metadata = dict(task._metadata)
        if storage_options:
            metadata["source_storage_options"] = storage_options

        batches: list[MultiBatchTask] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
            batches.append(
                MultiBatchTask(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]
