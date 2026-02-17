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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from nemo_curator.utils.multimodal_utils import (
    MaterializeContext,
    build_materialize_context,
    load_payloads_from_direct_path,
    load_payloads_from_tar_members,
    replace_binary_content,
    retry_materialize,
    validate_content_path_loading_mode,
    validate_materialize_options,
)

from .tasks import Task

MULTIMODAL_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("content_type", pa.string()),
        pa.field("text_content", pa.string()),
        pa.field("binary_content", pa.large_binary()),
        pa.field("element_metadata_json", pa.string()),
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

    def get_content_paths(
        self,
        modality: str = "image",
        source: Literal["all", "content_key", "direct"] = "all",
    ) -> list[str]:
        """Return unique ``content_path`` values for one modality.

        Args:
            modality: Row modality to filter on.
            source: Path selection mode:
                - ``all``: include all rows for modality
                - ``content_key``: only rows where ``content_key`` is set
                - ``direct``: only rows where ``content_key`` is null
        """
        if source not in {"all", "content_key", "direct"}:
            msg = f"Unsupported source='{source}'. Expected one of: all, content_key, direct"
            raise ValueError(msg)

        has_content_key = pc.invert(pc.is_null(self.data["content_key"]))
        modality_mask = pc.equal(self.data["modality"], modality)
        if source == "content_key":
            mask = pc.and_(modality_mask, has_content_key)
        elif source == "direct":
            mask = pc.and_(modality_mask, pc.invert(has_content_key))
        else:
            mask = modality_mask
        filtered = pc.filter(self.data["content_path"], mask)
        return [str(v) for v in pc.unique(filtered).to_pylist() if v is not None]

    def get_content_extensions(self, modality: str = "image") -> list[str]:
        """Return sorted unique file extensions observed for a modality."""
        paths = self.get_content_paths(modality=modality, source="all")
        return sorted(
            {
                suffix.lstrip(".").lower()
                for path in paths
                for suffix in [Path(path).suffix]
                if suffix
            }
        )

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

    def _build_materialize_context(
        self,
        *,
        table: pa.Table,
        modality: str,
        storage_options: dict[str, Any] | None,
    ) -> MaterializeContext:
        return build_materialize_context(
            table=table,
            modality=modality,
            storage_options=storage_options,
            metadata=self._metadata,
        )

    def materialize(
        self,
        modality: str = "image",
        storage_options: dict[str, Any] | None = None,
        max_retries: int = 0,
        retry_backoff_sec: float = 0.0,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> MultimodalBatch:
        """Load missing binary payloads for the requested modality.

        Args:
            modality: Target modality to materialize.
            storage_options: Optional storage options override.
            max_retries: Number of retries per ``content_path`` read on failure.
            retry_backoff_sec: Base backoff seconds between retries. Uses exponential
                backoff (``base * 2**attempt``).
            on_error: Failure policy.
                - ``raise``: raise the final read error
                - ``skip``: keep failed rows null and continue
        """
        table = self.data
        validate_materialize_options(max_retries=max_retries, retry_backoff_sec=retry_backoff_sec, on_error=on_error)
        context = self._build_materialize_context(
            table=table,
            modality=modality,
            storage_options=storage_options,
        )
        if not context.pending_rows_by_path:
            return self

        failed_paths: list[str] = []

        for content_path, row_indices in context.pending_rows_by_path.items():
            # One content_path must map to exactly one loading strategy.
            validate_content_path_loading_mode(
                content_path=content_path,
                row_indices=row_indices,
                content_keys=context.content_keys,
            )

            def materialize_path_once(
                content_path_value: str = content_path,
                row_indices_value: list[int] = row_indices,
            ) -> None:
                # Mixed storage is possible at dataset level, so choose loader per path-group.
                keyed_rows = {
                    idx: str(context.content_keys[idx])
                    for idx in row_indices_value
                    if context.content_keys[idx] is not None
                }
                if keyed_rows:
                    loaded_payloads = load_payloads_from_tar_members(
                        content_path=content_path_value,
                        keyed_rows=keyed_rows,
                        storage_options=context.storage_options,
                    )
                else:
                    loaded_payloads = load_payloads_from_direct_path(
                        content_path=content_path_value,
                        row_indices=row_indices_value,
                        storage_options=context.storage_options,
                    )
                for idx, payload in loaded_payloads.items():
                    context.binary_payloads[idx] = payload

            last_error = retry_materialize(
                materialize_path_once,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
            )
            if last_error:
                if on_error == "raise":
                    raise last_error
                failed_paths.append(content_path)
                logger.warning(
                    "Skipping materialize failure for path='{}' after {} attempts: {}",
                    content_path,
                    max_retries + 1,
                    last_error,
                )

        if failed_paths:
            logger.warning(
                "Materialize completed with {} failed paths (rows kept lazy).",
                len(set(failed_paths)),
            )

        return self._clone(replace_binary_content(table, context.binary_payloads))

    def dematerialize(self, modality: str = "image") -> MultimodalBatch:
        """Clear binary payloads for the requested modality."""
        table = self.data
        dematerialize_mask = pc.equal(table["modality"], modality)
        binary_values = pc.if_else(
            dematerialize_mask,
            pa.nulls(table.num_rows, type=pa.large_binary()),
            table["binary_content"],
        ).to_pylist()
        return self._clone(replace_binary_content(table, binary_values))

    def set_modality_annotation(self, name: str, modality: str, values: list[Any]) -> MultimodalBatch:
        """Set or overwrite a per-row annotation column for one modality."""
        table = self.data
        modality_indices = pc.indices_nonzero(pc.equal(table["modality"], modality)).to_pylist()
        if len(values) != len(modality_indices):
            msg = (
                f"Annotation values length mismatch for modality='{modality}': "
                f"expected {len(modality_indices)}, got {len(values)}"
            )
            raise ValueError(msg)

        field_index = table.schema.get_field_index(name)
        if field_index >= 0:
            value_type = table.field(field_index).type
        elif values:
            value_type = pa.array(values).type
        else:
            value_type = pa.null()

        annotation_values: list[Any] = [None] * table.num_rows
        for idx, value in zip(modality_indices, values, strict=True):
            annotation_values[idx] = value
        annotation_column = pa.array(annotation_values, type=value_type)

        out = (
            table.set_column(field_index, name, annotation_column)
            if field_index >= 0
            else table.append_column(name, annotation_column)
        )
        return self._clone(out)
