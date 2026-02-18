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

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from nemo_curator.utils.multimodal_utils import (
    METADATA_MODALITY,
    METADATA_POSITION,
    build_metadata_row,
    load_payloads_from_direct_path,
    load_payloads_from_tar_members,
    validate_content_path_loading_mode,
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

@dataclass
class _MaterializeContext:
    storage_options: dict[str, Any]
    pending_rows_by_path: dict[str, list[int]]
    binary_payloads: list[Any]
    content_keys: list[object | None]


@dataclass
class MultimodalBatch(Task[pa.Table]):
    """Task carrying normalized multimodal rows in an Arrow table.

    The ``data`` table follows ``MULTIMODAL_SCHEMA`` (sample/position/modality
    plus text or binary payload fields). For image rows, ``binary_content`` may
    be null in lazy mode and populated later via :meth:`materialize`.
    """

    data: pa.Table = field(default_factory=lambda: pa.Table.from_pylist([], schema=MULTIMODAL_SCHEMA))

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

    def _clone(self, table: pa.Table) -> MultimodalBatch:
        """Return a copy of this batch with replaced ``data`` table."""
        return MultimodalBatch(
            task_id=self.task_id,
            dataset_name=self.dataset_name,
            data=table,
            _metadata=self._metadata,
            _stage_perf=self._stage_perf,
        )

    @staticmethod
    def _normalize_json_payload(payload: str | dict[str, Any] | None) -> str | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=True)
        return payload

    @staticmethod
    def _build_single_row_table(
        schema: pa.Schema,
        row_values: dict[str, object | None],
    ) -> pa.Table:
        columns = [
            pa.array([row_values.get(field.name)], type=field.type)
            for field in schema
        ]
        return pa.Table.from_arrays(columns, schema=schema)

    @staticmethod
    def _validate_non_negative_position(position: int) -> None:
        if position < 0:
            msg = f"position must be >= 0 for content rows, got {position}"
            raise ValueError(msg)

    @staticmethod
    def _row_mask(
        table: pa.Table,
        *,
        sample_id: str,
        position: int,
        modality: str,
    ) -> pa.Array:
        return pc.and_(
            pc.and_(pc.equal(table["sample_id"], sample_id), pc.equal(table["position"], position)),
            pc.equal(table["modality"], modality),
        )

    @staticmethod
    def _shift_sample_positions_for_insert(
        table: pa.Table,
        *,
        sample_id: str,
        position: int,
    ) -> pa.Table:
        sample_ids = table["sample_id"].to_pylist()
        positions = table["position"].to_pylist()
        modalities = table["modality"].to_pylist()
        shifted_positions = [
            (int(pos) + 1 if str(sid) == sample_id and int(pos) >= position and str(modality) != METADATA_MODALITY else int(pos))
            for sid, pos, modality in zip(sample_ids, positions, modalities, strict=True)
        ]
        position_idx = table.schema.get_field_index("position")
        return table.set_column(
            position_idx,
            "position",
            pa.array(shifted_positions, type=pa.int32()),
        )

    def _content_row_values(  # noqa: PLR0913
        self,
        *,
        schema: pa.Schema,
        sample_id: str,
        position: int,
        modality: Literal["text", "image"],
        content_type: str | None,
        text_content: str | None,
        binary_content: bytes | None,
        element_metadata_json: str | dict[str, Any] | None,
        source_id: str | None,
        source_shard: str | None,
        content_path: str | None,
        content_key: str | None,
    ) -> dict[str, object | None]:
        row_values: dict[str, object | None] = {name: None for name in schema.names}
        row_values.update(
            {
                "sample_id": sample_id,
                "position": position,
                "modality": modality,
                "content_type": content_type,
                "text_content": text_content if modality == "text" else None,
                "binary_content": binary_content if modality == "image" else None,
                "element_metadata_json": self._normalize_json_payload(element_metadata_json),
                "source_id": source_id if source_id is not None else sample_id,
                "source_shard": source_shard,
                "content_path": content_path,
                "content_key": content_key,
            }
        )
        return row_values

    def _upsert_row_by_key(
        self,
        *,
        row_values: dict[str, object | None],
        sample_id: str,
        position: int,
        modality: str,
        table: pa.Table | None = None,
    ) -> MultimodalBatch:
        source = self.data if table is None else table
        existing_mask = self._row_mask(source, sample_id=sample_id, position=position, modality=modality)
        filtered = source.filter(pc.invert(existing_mask))
        updated = pa.concat_tables([filtered, self._build_single_row_table(filtered.schema, row_values)])
        return self._clone(updated)

    def _delete_row_by_key(
        self,
        *,
        sample_id: str,
        position: int,
        modality: str,
    ) -> MultimodalBatch:
        mask = self._row_mask(self.data, sample_id=sample_id, position=position, modality=modality)
        return self._clone(self.data.filter(pc.invert(mask)))

    def upsert_position_content(  # noqa: PLR0913
        self,
        *,
        sample_id: str,
        position: int,
        modality: Literal["text", "image"],
        content_type: str | None = None,
        text_content: str | None = None,
        binary_content: bytes | None = None,
        element_metadata_json: str | dict[str, Any] | None = None,
        source_id: str | None = None,
        source_shard: str | None = None,
        content_path: str | None = None,
        content_key: str | None = None,
    ) -> MultimodalBatch:
        """Add/replace one text or image row at ``(sample_id, position, modality)``."""
        self._validate_non_negative_position(position)

        row_values = self._content_row_values(
            schema=self.data.schema,
            sample_id=sample_id,
            position=position,
            modality=modality,
            content_type=content_type,
            text_content=text_content,
            binary_content=binary_content,
            element_metadata_json=element_metadata_json,
            source_id=source_id,
            source_shard=source_shard,
            content_path=content_path,
            content_key=content_key,
        )
        return self._upsert_row_by_key(
            row_values=row_values,
            sample_id=sample_id,
            position=position,
            modality=modality,
        )

    def insert_position_content(  # noqa: PLR0913
        self,
        *,
        sample_id: str,
        position: int,
        modality: Literal["text", "image"],
        content_type: str | None = None,
        text_content: str | None = None,
        binary_content: bytes | None = None,
        element_metadata_json: str | dict[str, Any] | None = None,
        source_id: str | None = None,
        source_shard: str | None = None,
        content_path: str | None = None,
        content_key: str | None = None,
    ) -> MultimodalBatch:
        """Insert one content row and shift sample positions at/after ``position`` by +1."""
        self._validate_non_negative_position(position)

        shifted = self._shift_sample_positions_for_insert(self.data, sample_id=sample_id, position=position)
        row_values = self._content_row_values(
            schema=shifted.schema,
            sample_id=sample_id,
            position=position,
            modality=modality,
            content_type=content_type,
            text_content=text_content,
            binary_content=binary_content,
            element_metadata_json=element_metadata_json,
            source_id=source_id,
            source_shard=source_shard,
            content_path=content_path,
            content_key=content_key,
        )
        return self._upsert_row_by_key(
            row_values=row_values,
            sample_id=sample_id,
            position=position,
            modality=modality,
            table=shifted,
        )

    def delete_position_content(
        self,
        *,
        sample_id: str,
        position: int,
        modality: Literal["text", "image"],
    ) -> MultimodalBatch:
        """Delete text/image rows at ``(sample_id, position, modality)``."""
        return self._delete_row_by_key(sample_id=sample_id, position=position, modality=modality)

    def upsert_sample_metadata(
        self,
        *,
        sample_id: str,
        metadata_json: str | dict[str, Any] | None,
        sample_type: str | None = None,
    ) -> MultimodalBatch:
        """Add/replace one sample metadata row using modality=metadata and position=-1."""
        row_values: dict[str, object | None] = {name: None for name in self.data.schema.names}
        row_values.update(
            build_metadata_row(
                sample_id=sample_id,
                metadata_json=self._normalize_json_payload(metadata_json) or "{}",
                sample_type=sample_type,
                source_shard=None,
                source_id=sample_id,
            )
        )
        return self._upsert_row_by_key(
            row_values=row_values,
            sample_id=sample_id,
            position=METADATA_POSITION,
            modality=METADATA_MODALITY,
        )

    def delete_sample_metadata(self, *, sample_id: str) -> MultimodalBatch:
        """Delete sample metadata row with modality=metadata and position=-1."""
        return self._delete_row_by_key(
            sample_id=sample_id,
            position=METADATA_POSITION,
            modality=METADATA_MODALITY,
        )

    @staticmethod
    def _validate_materialize_options(
        *,
        max_retries: int,
        retry_backoff_sec: float,
        on_error: Literal["raise", "skip"],
    ) -> None:
        if max_retries < 0:
            msg = f"max_retries must be >= 0, got {max_retries}"
            raise ValueError(msg)
        if retry_backoff_sec < 0:
            msg = f"retry_backoff_sec must be >= 0, got {retry_backoff_sec}"
            raise ValueError(msg)
        if on_error not in {"raise", "skip"}:
            msg = f"on_error must be one of: raise, skip; got {on_error}"
            raise ValueError(msg)

    @staticmethod
    def _replace_binary_content(table: pa.Table, binary_values: list[Any]) -> pa.Table:
        binary_idx = table.schema.get_field_index("binary_content")
        binary_column = pa.array(binary_values, type=pa.large_binary())
        return table.set_column(binary_idx, "binary_content", binary_column)

    @staticmethod
    def _materialize_with_retries(
        context: _MaterializeContext,
        content_path: str,
        row_indices: list[int],
        *,
        max_retries: int,
        retry_backoff_sec: float,
    ) -> Exception | None:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                MultimodalBatch._materialize_path(context, content_path, row_indices)
            except (OSError, RuntimeError, TimeoutError, ValueError) as err:  # noqa: PERF203
                last_error = err
                if attempt == max_retries:
                    break
                if retry_backoff_sec > 0:
                    time.sleep(retry_backoff_sec * (2**attempt))
            else:
                return None
        return last_error

    def _build_materialize_context(
        self,
        *,
        table: pa.Table,
        modality: str,
        storage_options: dict[str, Any] | None,
    ) -> _MaterializeContext:
        if storage_options is not None:
            resolved_storage = dict(storage_options)
        else:
            metadata_storage = self._metadata.get("storage_options")
            resolved_storage = dict(metadata_storage) if isinstance(metadata_storage, dict) else {}

        pending_rows_by_path: dict[str, list[int]] = defaultdict(list)
        row_modalities = table["modality"].to_pylist()
        binary_payloads = table["binary_content"].to_pylist()
        content_paths = table["content_path"].to_pylist()
        for idx, (row_modality, payload, content_path) in enumerate(
            zip(row_modalities, binary_payloads, content_paths, strict=True)
        ):
            # Materialize only rows that still need payload bytes and have a source path.
            if row_modality == modality and payload is None and content_path is not None:
                pending_rows_by_path[str(content_path)].append(idx)

        return _MaterializeContext(
            storage_options=resolved_storage,
            pending_rows_by_path=pending_rows_by_path,
            binary_payloads=binary_payloads,
            content_keys=table["content_key"].to_pylist(),
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
        self._validate_materialize_options(
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            on_error=on_error,
        )
        context = self._build_materialize_context(
            table=table,
            modality=modality,
            storage_options=storage_options,
        )
        if not context.pending_rows_by_path:
            return self

        failed_paths: list[str] = []

        for content_path, row_indices in context.pending_rows_by_path.items():
            last_error = self._materialize_with_retries(
                context,
                content_path,
                row_indices,
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

        return self._clone(self._replace_binary_content(table, context.binary_payloads))

    @staticmethod
    def _materialize_path(
        context: _MaterializeContext,
        content_path: str,
        row_indices: list[int],
    ) -> None:
        # One content_path must map to exactly one loading strategy.
        validate_content_path_loading_mode(
            content_path=content_path,
            row_indices=row_indices,
            content_keys=context.content_keys,
        )
        # Mixed storage is possible at dataset level, so choose loader per path-group.
        keyed_rows = {idx: str(context.content_keys[idx]) for idx in row_indices if context.content_keys[idx] is not None}
        if keyed_rows:
            loaded_payloads = load_payloads_from_tar_members(
                content_path=content_path,
                keyed_rows=keyed_rows,
                storage_options=context.storage_options,
            )
        else:
            loaded_payloads = load_payloads_from_direct_path(
                content_path=content_path,
                row_indices=row_indices,
                storage_options=context.storage_options,
            )
        for idx, payload in loaded_payloads.items():
            context.binary_payloads[idx] = payload

    def dematerialize(self, modality: str = "image") -> MultimodalBatch:
        """Clear binary payloads for the requested modality."""
        table = self.data
        dematerialize_mask = pc.equal(table["modality"], modality)
        binary_values = pc.if_else(
            dematerialize_mask,
            pa.nulls(table.num_rows, type=pa.large_binary()),
            table["binary_content"],
        ).to_pylist()
        return self._clone(self._replace_binary_content(table, binary_values))
