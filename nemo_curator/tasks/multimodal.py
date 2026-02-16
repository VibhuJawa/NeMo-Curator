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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.utils.file_utils import open_binary_reader, open_tar_path

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

METADATA_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("sample_type", pa.string()),
        pa.field("metadata_json", pa.string()),
    ]
)


@dataclass
class MultimodalBatch(Task[pa.Table]):
    """Task carrying normalized multimodal rows in an Arrow table.

    The ``data`` table follows ``MULTIMODAL_SCHEMA`` (sample/position/modality
    plus text or binary payload fields). For image rows, ``binary_content`` may
    be null in lazy mode and populated later via :meth:`materialize`.
    ``metadata_index`` stores per-sample metadata from the reader.
    """

    data: pa.Table = field(default_factory=lambda: pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA))
    metadata_index: pa.Table | None = None

    @property
    def num_items(self) -> int:
        """Return number of rows in the batch."""
        return self.data.num_rows

    def validate(self) -> bool:
        """Validate that ``data`` includes all required multimodal columns."""
        required = set(MULTIMODAL_SCHEMA.names)
        current = set(self.data.column_names)
        missing = required - current
        if missing:
            msg = f"MultimodalBatch missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        return True

    @property
    def is_lazy(self) -> bool:
        """Return whether any image rows still have null ``binary_content``."""
        return bool(
            pc.any(
                pc.and_(
                    pc.equal(self.data["modality"], "image"),
                    pc.is_null(self.data["binary_content"]),
                )
            ).as_py()
        )

    def get_modality_rows(self, modality: str) -> pa.Table:
        """Return rows whose ``modality`` equals ``modality``."""
        return self.data.filter(pc.equal(self.data["modality"], modality))

    def get_tar_paths(self, modality: str = "image") -> list[str]:
        """Return unique tar paths for rows of a modality that use ``content_key``."""
        mask = pc.and_(
            pc.equal(self.data["modality"], modality),
            pc.invert(pc.is_null(self.data["content_key"])),
        )
        filtered = pc.filter(self.data["content_path"], mask)
        return [str(v) for v in pc.unique(filtered).to_pylist() if v is not None]

    def get_file_paths(self, modality: str = "image") -> list[str]:
        """Return unique direct file paths for rows of a modality."""
        mask = pc.and_(
            pc.equal(self.data["modality"], modality),
            pc.is_null(self.data["content_key"]),
        )
        filtered = pc.filter(self.data["content_path"], mask)
        return [str(v) for v in pc.unique(filtered).to_pylist() if v is not None]

    def get_content_extensions(self, modality: str = "image") -> list[str]:
        """Return sorted unique file extensions observed for a modality."""
        paths = [*self.get_tar_paths(modality=modality), *self.get_file_paths(modality=modality)]
        exts: set[str] = set()
        for path in paths:
            if "." not in path:
                continue
            exts.add(path.rsplit(".", 1)[-1].lower())
        return sorted(exts)

    def _clone(self, table: pa.Table) -> MultimodalBatch:
        """Return a copy of this batch with replaced ``data`` table."""
        return MultimodalBatch(
            task_id=self.task_id,
            dataset_name=self.dataset_name,
            data=table,
            metadata_index=self.metadata_index,
            _metadata=self._metadata,
            _stage_perf=self._stage_perf,
        )

    def _resolved_storage_options(self, storage_options: dict[str, Any] | None) -> dict[str, Any]:
        """Resolve storage options from method arg or task metadata."""
        if storage_options is not None:
            return dict(storage_options)
        metadata_storage = self._metadata.get("storage_options")
        return dict(metadata_storage) if isinstance(metadata_storage, dict) else {}

    @staticmethod
    def _materialize_indices(table: pa.Table, modality: str) -> list[int]:
        """Return row indices requiring payload materialization for a modality."""
        is_target_modality = pc.equal(table["modality"], modality)
        is_unmaterialized = pc.is_null(table["binary_content"])
        has_content_path = pc.invert(pc.is_null(table["content_path"]))
        materialize_mask = pc.and_(pc.and_(is_target_modality, is_unmaterialized), has_content_path)
        return pc.indices_nonzero(materialize_mask).to_pylist()

    @staticmethod
    def _group_indices_by_content_path(table: pa.Table, indices: list[int]) -> dict[str, list[int]]:
        """Group row indices by ``content_path``."""
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            grouped[str(table["content_path"][idx].as_py())].append(idx)
        return grouped

    @staticmethod
    def _extract_tar_payloads(
        content_path: str,
        keys_needed: set[str],
        storage_options: dict[str, Any],
    ) -> dict[str, bytes]:
        """Extract requested tar members from one tar path."""
        extracted_payloads: dict[str, bytes] = {}
        with open_tar_path(content_path, storage_options) as tf:
            for member in tf:
                if member.name not in keys_needed:
                    continue
                payload = tf.extractfile(member)
                if payload is None:
                    continue
                extracted_payloads[member.name] = payload.read()
        return extracted_payloads

    def _materialize_from_tar(
        self,
        table: pa.Table,
        binary_values: list[Any],
        content_path: str,
        path_indices: list[int],
        storage_options: dict[str, Any],
    ) -> None:
        """Populate ``binary_values`` for tar-backed rows."""
        keys_needed = {
            str(table["content_key"][idx].as_py())
            for idx in path_indices
            if table["content_key"][idx].as_py() is not None
        }
        extracted_payloads = self._extract_tar_payloads(content_path, keys_needed, storage_options)

        for idx in path_indices:
            content_key_value = table["content_key"][idx].as_py()
            if content_key_value is None:
                continue
            content_key = str(content_key_value)
            if content_key not in extracted_payloads:
                msg = f"Missing tar member '{content_key}' in '{content_path}'"
                raise FileNotFoundError(msg)
            binary_values[idx] = extracted_payloads[content_key]

    @staticmethod
    def _materialize_from_file(
        binary_values: list[Any],
        content_path: str,
        path_indices: list[int],
        storage_options: dict[str, Any],
    ) -> None:
        """Populate ``binary_values`` for direct-file-backed rows."""
        with open_binary_reader(content_path, storage_options) as f:
            payload = f.read()
        for idx in path_indices:
            binary_values[idx] = payload

    def materialize(
        self,
        modality: str = "image",
        storage_options: dict[str, Any] | None = None,
    ) -> MultimodalBatch:
        """Load missing binary payloads for the requested modality."""
        table = self.data
        effective_storage_options = self._resolved_storage_options(storage_options)
        indices = self._materialize_indices(table, modality)
        if not indices:
            return self

        binary_values = table["binary_content"].to_pylist()
        path_to_indices = self._group_indices_by_content_path(table, indices)

        for content_path, path_indices in path_to_indices.items():
            has_tar_members = any(table["content_key"][idx].as_py() is not None for idx in path_indices)
            if has_tar_members:
                self._materialize_from_tar(
                    table=table,
                    binary_values=binary_values,
                    content_path=content_path,
                    path_indices=path_indices,
                    storage_options=effective_storage_options,
                )
            else:
                self._materialize_from_file(
                    binary_values=binary_values,
                    content_path=content_path,
                    path_indices=path_indices,
                    storage_options=effective_storage_options,
                )

        binary_idx = table.schema.get_field_index("binary_content")
        out = table.set_column(binary_idx, "binary_content", pa.array(binary_values, type=pa.large_binary()))
        return self._clone(out)

    def dematerialize(self, modality: str = "image") -> MultimodalBatch:
        """Clear binary payloads for the requested modality."""
        table = self.data
        binary_idx = table.schema.get_field_index("binary_content")
        dematerialize_mask = pc.equal(table["modality"], modality)
        binary_column = pc.if_else(dematerialize_mask, pa.nulls(table.num_rows, type=pa.large_binary()), table["binary_content"])
        return self._clone(table.set_column(binary_idx, "binary_content", binary_column))

    def set_modality_annotation(self, name: str, modality: str, values: list[Any]) -> MultimodalBatch:
        """Set or overwrite a per-row annotation column for one modality."""
        table = self.data
        target_indices = pc.indices_nonzero(pc.equal(table["modality"], modality)).to_pylist()
        if len(values) != len(target_indices):
            msg = f"Annotation values length mismatch for modality='{modality}': expected {len(target_indices)}, got {len(values)}"
            raise ValueError(msg)

        field_index = table.schema.get_field_index(name)
        if field_index >= 0:
            value_type = table.field(field_index).type
        elif values:
            value_type = pa.array(values).type
        else:
            value_type = pa.null()

        annotation_values: list[Any] = [None] * table.num_rows
        for idx, value in zip(target_indices, values, strict=True):
            annotation_values[idx] = value
        annotation_column = pa.array(annotation_values, type=value_type)

        out = (
            table.set_column(field_index, name, annotation_column)
            if field_index >= 0
            else table.append_column(name, annotation_column)
        )
        return self._clone(out)
