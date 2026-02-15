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

from __future__ import annotations

import tarfile
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from .tasks import Task

MULTIMODAL_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("content_type", pa.string()),
        pa.field("text_content", pa.string()),
        pa.field("binary_content", pa.large_binary()),
        pa.field("source_id", pa.string()),
        pa.field("source_shard", pa.string()),
        pa.field("content_path", pa.string()),
        pa.field("content_key", pa.string()),
    ]
)


@dataclass
class MultimodalBatch(Task[pa.Table]):
    """Task carrying normalized multimodal rows in an Arrow table.

    Each row is one modality item for a sample (`sample_id`, `position`,
    `modality`) with either inline content (`text_content`, `binary_content`) or
    lazy pointers (`content_path`, `content_key`) for later materialization.
    """
    data: pa.Table = field(default_factory=lambda: pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA))
    metadata_index: pa.Table | None = None

    @property
    def num_items(self) -> int:
        return self.data.num_rows

    def validate(self) -> bool:
        required = set(MULTIMODAL_SCHEMA.names)
        current = set(self.data.column_names)
        missing = required - current
        if missing:
            msg = f"MultimodalBatch missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        return True

    @property
    def is_lazy(self) -> bool:
        return bool(
            pc.any(
                pc.and_(
                    pc.equal(self.data["modality"], "image"),
                    pc.is_null(self.data["binary_content"]),
                )
            ).as_py()
        )

    def _clone(self, table: pa.Table) -> MultimodalBatch:
        return MultimodalBatch(
            task_id=self.task_id,
            dataset_name=self.dataset_name,
            data=table,
            metadata_index=self.metadata_index,
            _metadata=self._metadata,
            _stage_perf=self._stage_perf,
        )

    def materialize(self, modality: str = "image") -> MultimodalBatch:
        table = self.data
        is_target_modality = pc.equal(table["modality"], modality)
        is_unmaterialized = pc.is_null(table["binary_content"])
        has_content_path = pc.invert(pc.is_null(table["content_path"]))
        materialize_mask = pc.and_(pc.and_(is_target_modality, is_unmaterialized), has_content_path)
        if not bool(pc.any(materialize_mask).as_py()):
            return self

        binary_values = table["binary_content"].to_pylist()
        content_paths = pc.unique(pc.filter(table["content_path"], materialize_mask)).to_pylist()
    The :attr:`is_lazy` property can be used to detect whether any rows are still
        for path_value in content_paths:
            content_path = str(path_value)
            path_mask = pc.and_(materialize_mask, pc.equal(table["content_path"], content_path))
            path_indices = pc.indices_nonzero(path_mask).to_pylist()
            has_tar_members = any(table["content_key"][idx].as_py() is not None for idx in path_indices)

            if has_tar_members:
                with tarfile.open(content_path, "r") as tf:
                    for idx in path_indices:
                        content_key_value = table["content_key"][idx].as_py()
                        if content_key_value is None:
                            continue
                        content_key = str(content_key_value)
                        extracted = tf.extractfile(content_key)
                        if extracted is None:
                            msg = f"Missing tar member '{content_key}' in '{content_path}'"
                            raise FileNotFoundError(msg)
                        binary_values[idx] = extracted.read()
            else:
                with open(content_path, "rb") as f:
                    payload = f.read()
                for idx in path_indices:
                    binary_values[idx] = payload

        binary_idx = table.schema.get_field_index("binary_content")
        out = table.set_column(binary_idx, "binary_content", pa.array(binary_values, type=pa.large_binary()))
        return self._clone(out)

    def dematerialize(self, modality: str = "image") -> MultimodalBatch:
        table = self.data
        binary_idx = table.schema.get_field_index("binary_content")
        dematerialize_mask = pc.equal(table["modality"], modality)
        binary_column = pc.if_else(dematerialize_mask, pa.nulls(table.num_rows, type=pa.large_binary()), table["binary_content"])
        return self._clone(table.set_column(binary_idx, "binary_content", binary_column))

    def set_modality_annotation(self, name: str, modality: str, values: list[Any]) -> MultimodalBatch:
        table = self.data
        target_indices = pc.indices_nonzero(pc.equal(table["modality"], modality)).to_pylist()
        if len(values) != len(target_indices):
            msg = (
                f"Annotation values length mismatch for modality='{modality}': expected {len(target_indices)}, got {len(values)}"
            )
            raise ValueError(msg)

        num_rows = table.num_rows
        field_index = table.schema.get_field_index(name)
        if field_index >= 0:
            value_type = table.field(field_index).type
        elif values:
            value_type = pa.array(values).type
        else:
            value_type = pa.null()

        annotation_values: list[Any] = [None] * num_rows
        for idx, value in zip(target_indices, values, strict=True):
            annotation_values[idx] = value
        annotation_column = pa.array(annotation_values, type=value_type)

        out = (
            table.set_column(field_index, name, annotation_column)
            if field_index >= 0
            else table.append_column(name, annotation_column)
        )
        return self._clone(out)
