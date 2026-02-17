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

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tarfile import ReadError
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
from aiohttp import ClientError
from fsspec.exceptions import FSTimeoutError
from loguru import logger

from nemo_curator.utils.file_utils import open_binary_reader, open_tar_path

from .tasks import Task

if TYPE_CHECKING:
    from collections.abc import Callable

_RETRIABLE_MATERIALIZE_EXCEPTIONS: tuple[type[Exception], ...] = (
    OSError,
    ReadError,
    TimeoutError,
    ClientError,
    FSTimeoutError,
)

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

    def _get_unique_content_paths(self, modality: str, *, require_content_key: bool) -> list[str]:
        """Return unique content paths filtered by modality and key-presence mode."""
        has_content_key = pc.invert(pc.is_null(self.data["content_key"]))
        mask = pc.and_(
            pc.equal(self.data["modality"], modality),
            has_content_key if require_content_key else pc.invert(has_content_key),
        )
        filtered = pc.filter(self.data["content_path"], mask)
        return [str(v) for v in pc.unique(filtered).to_pylist() if v is not None]

    def get_tar_paths(self, modality: str = "image") -> list[str]:
        """Return unique tar paths for rows of a modality that use ``content_key``."""
        return self._get_unique_content_paths(modality, require_content_key=True)

    def get_file_paths(self, modality: str = "image") -> list[str]:
        """Return unique direct file paths for rows of a modality."""
        return self._get_unique_content_paths(modality, require_content_key=False)

    def get_content_extensions(self, modality: str = "image") -> list[str]:
        """Return sorted unique file extensions observed for a modality."""
        paths = [*self.get_tar_paths(modality), *self.get_file_paths(modality)]
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

    def _resolved_storage_options(self, storage_options: dict[str, Any] | None) -> dict[str, Any]:
        """Resolve storage options from method arg or task metadata."""
        if storage_options is not None:
            return dict(storage_options)
        metadata_storage = self._metadata.get("storage_options")
        return dict(metadata_storage) if isinstance(metadata_storage, dict) else {}

    @staticmethod
    def _materialize_plan(table: pa.Table, modality: str) -> dict[str, list[int]]:
        """Return content-path -> row indices for rows requiring materialization."""
        is_target_modality = pc.equal(table["modality"], modality)
        is_unmaterialized = pc.is_null(table["binary_content"])
        has_content_path = pc.invert(pc.is_null(table["content_path"]))
        materialize_mask = pc.and_(pc.and_(is_target_modality, is_unmaterialized), has_content_path)
        materialize_indices = pc.indices_nonzero(materialize_mask).to_pylist()
        if not materialize_indices:
            return {}
        content_paths = table["content_path"].to_pylist()
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx in materialize_indices:
            grouped[str(content_paths[idx])].append(idx)
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
                if member.name in keys_needed:
                    payload = tf.extractfile(member)
                    if payload is not None:
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
            if content_key_value is not None:
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

    @staticmethod
    def _validate_materialize_options(
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
    def _retry_materialize(
        materialize_once: Callable[[], None],
        max_retries: int,
        retry_backoff_sec: float,
    ) -> Exception | None:
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                materialize_once()
            except _RETRIABLE_MATERIALIZE_EXCEPTIONS as err:  # noqa: PERF203
                last_error = err
                if attempt == max_retries:
                    break
                if retry_backoff_sec > 0:
                    time.sleep(retry_backoff_sec * (2**attempt))
            else:
                return None
        return last_error

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
        storage = self._resolved_storage_options(storage_options)
        self._validate_materialize_options(max_retries=max_retries, retry_backoff_sec=retry_backoff_sec, on_error=on_error)
        path_to_indices = self._materialize_plan(table, modality)
        if not path_to_indices:
            return self

        binary_values = table["binary_content"].to_pylist()
        failed_paths: list[str] = []
        content_keys = table["content_key"].to_pylist()

        for content_path, path_indices in path_to_indices.items():
            has_tar_members = any(content_keys[idx] is not None for idx in path_indices)

            def materialize_once(
                content_path_value: str = content_path,
                path_indices_value: list[int] = path_indices,
                has_tar_members_value: bool = has_tar_members,
            ) -> None:
                if has_tar_members_value:
                    self._materialize_from_tar(
                        table=table,
                        binary_values=binary_values,
                        content_path=content_path_value,
                        path_indices=path_indices_value,
                        storage_options=storage,
                    )
                else:
                    self._materialize_from_file(
                        binary_values=binary_values,
                        content_path=content_path_value,
                        path_indices=path_indices_value,
                        storage_options=storage,
                    )
            last_error = self._retry_materialize(
                materialize_once,
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

        return self._clone(self._replace_binary_content(table, binary_values))

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
