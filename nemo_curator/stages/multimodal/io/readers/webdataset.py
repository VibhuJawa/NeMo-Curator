# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.multimodal.io.readers.base import BaseMultimodalReaderStage, Row, RowSource
from nemo_curator.tasks import MultimodalBatch, _EmptyTask
from nemo_curator.tasks.multimodal import METADATA_SCHEMA, MULTIMODAL_SCHEMA
from nemo_curator.utils.file_utils import open_tar_path
from nemo_curator.utils.webdataset_utils import (
    content_type_from_name,
    member_stem,
    modality_from_content_type,
    parse_sample_and_position,
)

_SUPPORTED_SAMPLE_FORMATS: Final[set[str]] = {"auto", "simple", "interleaved"}
_SUPPORTED_MODALITIES_TO_LOAD: Final[set[str]] = {"all", "image", "text"}
_SUPPORTED_ERROR_HANDLING: Final[set[str]] = {"raise", "skip", "log"}
_SUPPORTED_INTERLEAVED_MODALITIES: Final[set[str]] = {"text", "image"}
_TEXT_MEMBER_SUFFIXES: Final[set[str]] = {".txt", ".json"}
_DEFAULT_INTERLEAVED_FIELD_MAP: Final[dict[str, str]] = {
    "sample_id": "sample_id",
    "segments": "segments",
    "modality": "modality",
    "text": "text",
    "content_key": "content_key",
}
SampleFormat = Literal["auto", "simple", "interleaved"]
_DEFAULT_WEBDATASET_EXTENSIONS: Final[list[str]] = [".tar", ".tar.gz", ".tgz", ".tar.zst"]


@dataclass
class RowBuildState:
    """Per-shard mutable parse state.

    Attributes:
        sample_counters: Fallback position counters keyed by inferred sample id.
        seen_metadata_sample_ids: First-wins guard set for metadata rows.
        metadata_rows: Accumulated metadata rows for ``METADATA_SCHEMA``.
    """

    sample_counters: dict[str, int] = field(default_factory=dict)
    seen_metadata_sample_ids: set[str] = field(default_factory=set)
    metadata_rows: list[dict[str, object]] = field(default_factory=list)


def _record_metadata_row(state: RowBuildState, sample_id: str, metadata_json: str) -> None:
    """Record one metadata row using first-wins policy per sample id."""
    if sample_id in state.seen_metadata_sample_ids:
        return
    state.seen_metadata_sample_ids.add(sample_id)
    state.metadata_rows.append(
        {
            "sample_id": sample_id,
            "sample_type": None,
            "metadata_json": metadata_json,
        }
    )


def _required_segment_str(segment: Row, field: str) -> str:
    value = segment.get(field)
    if not isinstance(value, str) or not value:
        msg = f"Interleaved segment must include non-empty string '{field}'"
        raise ValueError(msg)
    return value


def _validate_interleaved_payload(decoded: object, field_map: dict[str, str]) -> tuple[str, list[Row]]:
    if not isinstance(decoded, dict):
        msg = "Interleaved JSON payload must decode to an object"
        raise TypeError(msg)

    sample_id_field = field_map["sample_id"]
    segments_field = field_map["segments"]
    sample_id = decoded.get(sample_id_field)
    if not isinstance(sample_id, str) or not sample_id:
        msg = f"Interleaved JSON payload must include non-empty string '{sample_id_field}'"
        raise ValueError(msg)

    segments = decoded.get(segments_field)
    if not isinstance(segments, list):
        msg = f"Interleaved JSON payload must include list field '{segments_field}'"
        raise TypeError(msg)

    typed_segments: list[Row] = []
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            msg = f"Interleaved segment at index={idx} for sample_id='{sample_id}' must be an object"
            raise TypeError(msg)
        typed_segments.append(segment)
    return sample_id, typed_segments


@dataclass
class WebDatasetReaderStage(BaseMultimodalReaderStage):
    """Parse WebDataset tar shards into normalized multimodal rows.

    Implements the base reader contract by reading from ``data_path`` and ignoring
    optional ``metadata_path`` (WebDataset metadata is produced from shard members).

    Execution order:
        ``read_source_tables``
        -> ``_should_read_member_payload``
        -> ``_rows_from_member``
        -> ``_rows_from_text_member`` / ``_rows_from_binary_member``
        -> ``_rows_from_interleaved_json`` (json path)
        -> ``_next_sample_and_position`` / ``_loads_modality``
        -> ``_handle_member_error`` (error path)
    """

    load_binary: bool = False
    sample_format: SampleFormat = "interleaved"
    error_handling: Literal["raise", "skip", "log"] = "log"
    modalities_to_load: Literal["all", "image", "text"] = "all"
    interleaved_field_map: dict[str, str] | None = None
    name: str = "webdataset_reader"

    @staticmethod
    def default_interleaved_field_map() -> dict[str, str]:
        """Return a copy of the default interleaved JSON field mapping."""
        return dict(_DEFAULT_INTERLEAVED_FIELD_MAP)

    def __post_init__(self) -> None:
        """Validate reader configuration."""
        if self.sample_format not in _SUPPORTED_SAMPLE_FORMATS:
            msg = f"Unsupported sample_format='{self.sample_format}'. Expected one of: auto, simple, interleaved"
            raise ValueError(msg)
        if self.modalities_to_load not in _SUPPORTED_MODALITIES_TO_LOAD:
            msg = f"Unsupported modalities_to_load='{self.modalities_to_load}'. Expected one of: all, image, text"
            raise ValueError(msg)
        if self.error_handling not in _SUPPORTED_ERROR_HANDLING:
            msg = f"Unsupported error_handling='{self.error_handling}'. Expected one of: raise, skip, log"
            raise ValueError(msg)
        if self.max_batch_bytes is not None and self.max_batch_bytes <= 0:
            msg = f"max_batch_bytes must be > 0, got {self.max_batch_bytes}"
            raise ValueError(msg)
        default_map = self.default_interleaved_field_map()
        unknown = sorted(set(self.interleaved_field_map or {}) - set(default_map))
        if unknown:
            msg = f"interleaved_field_map has unknown keys: {unknown}"
            raise ValueError(msg)
        resolved = default_map
        resolved.update(self.interleaved_field_map or {})
        for semantic, actual in resolved.items():
            if not isinstance(actual, str) or not actual:
                msg = f"interleaved_field_map['{semantic}'] must be a non-empty string"
                raise ValueError(msg)
        self.interleaved_field_map = resolved

    def read_source_tables(self, data_path: str, metadata_path: str | None) -> tuple[pa.Table, pa.Table]:
        """Read one tar shard into normalized data and metadata tables.

        ``metadata_path`` is unused for WebDataset inputs.
        """
        _ = metadata_path
        source = RowSource(source_shard=Path(data_path).name, content_path=data_path)
        rows: list[Row] = []
        state = RowBuildState()

        with open_tar_path(data_path, self.storage_options) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                member_name = member.name
                payload: bytes | None = None
                try:
                    if self._should_read_member_payload(member_name):
                        payload_obj = tf.extractfile(member)
                        payload = payload_obj.read() if payload_obj else b""
                except Exception as err:  # noqa: BLE001
                    self._handle_member_error(member_name, err)
                    continue

                try:
                    rows.extend(
                        self._rows_from_member(
                            state=state,
                            member_name=member_name,
                            payload=payload,
                            source=source,
                        )
                    )
                except Exception as err:  # noqa: BLE001
                    self._handle_member_error(member_name, err)
        return pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA), pa.Table.from_pylist(
            state.metadata_rows, schema=METADATA_SCHEMA
        )

    def _should_read_member_payload(self, member_name: str) -> bool:
        suffix = Path(member_name).suffix.lower()
        if suffix == ".json":
            return True
        if suffix == ".txt":
            return self._loads_modality("text")
        return self.load_binary

    def _handle_member_error(self, member_name: str, err: Exception) -> None:
        if self.error_handling == "raise":
            raise err
        if self.error_handling == "log":
            logger.warning(f"Skipping corrupt member '{member_name}': {err}")

    def _rows_from_member(
        self,
        state: RowBuildState,
        member_name: str,
        payload: bytes | None,
        source: RowSource,
    ) -> list[dict[str, object]]:
        suffix = Path(member_name).suffix.lower()
        if suffix in _TEXT_MEMBER_SUFFIXES:
            return self._rows_from_text_member(state, member_name, suffix, payload, source)
        return self._rows_from_binary_member(state, member_name, payload, source)

    def _rows_from_text_member(
        self,
        state: RowBuildState,
        member_name: str,
        suffix: str,
        payload: bytes | None,
        source: RowSource,
    ) -> list[dict[str, object]]:
        if suffix == ".json":
            if payload is None:
                msg = f"JSON member '{member_name}' missing payload bytes"
                raise ValueError(msg)
            try:
                parsed = self._rows_from_interleaved_json(payload, source, state)
            except (KeyError, TypeError, ValueError):
                if self.sample_format == "interleaved":
                    raise
                parsed = []
            if parsed:
                return parsed
        if not self._loads_modality("text"):
            return []
        if payload is None:
            msg = f"Text member '{member_name}' missing payload bytes"
            raise ValueError(msg)
        sid, position = self._next_sample_and_position(state.sample_counters, member_name, "text")
        text_content = payload.decode("utf-8") if payload else ""
        content_type = "application/json" if suffix == ".json" else "text/plain"
        if suffix == ".json":
            _record_metadata_row(state, sid, text_content or "{}")
        return [
            self._text_row(
                sid=sid,
                position=position,
                source_shard=source.source_shard,
                content_type=content_type,
                text_content=text_content,
            )
        ]

    def _rows_from_interleaved_json(
        self,
        payload: bytes,
        source: RowSource,
        state: RowBuildState,
    ) -> list[dict[str, object]]:
        decoded = json.loads(payload.decode("utf-8"))
        sample_id, segments = _validate_interleaved_payload(decoded, self.interleaved_field_map)
        _record_metadata_row(state, sample_id, json.dumps(decoded, ensure_ascii=True))

        rows: list[dict[str, object]] = []
        field_map = self.interleaved_field_map
        modality_field = field_map["modality"]
        text_field = field_map["text"]
        content_key_field = field_map["content_key"]
        for idx, segment in enumerate(segments):
            modality = _required_segment_str(segment, modality_field)
            if modality not in _SUPPORTED_INTERLEAVED_MODALITIES:
                msg = (
                    f"Unsupported interleaved modality='{modality}' for sample_id='{sample_id}' "
                    "in WebDatasetReaderStage (supported: text, image)"
                )
                raise ValueError(msg)
            if not self._loads_modality(modality):
                continue
            if modality == "text":
                rows.append(
                    self._text_row(
                        sid=sample_id,
                        position=idx,
                        source_shard=source.source_shard,
                        content_type="text/plain",
                        text_content=_required_segment_str(segment, text_field),
                    )
                )
                continue
            rows.append(
                self._image_row(
                    sid=sample_id,
                    position=idx,
                    source=source,
                    content_key=_required_segment_str(segment, content_key_field),
                )
            )
        return rows

    def _rows_from_binary_member(
        self,
        state: RowBuildState,
        member_name: str,
        payload: bytes | None,
        source: RowSource,
    ) -> list[dict[str, object]]:
        content_type = content_type_from_name(member_name)
        modality = modality_from_content_type(content_type)
        if modality == "unknown":
            msg = f"Unsupported content_type='{content_type}' for member '{member_name}' in WebDatasetReaderStage"
            raise ValueError(msg)
        if modality != "image":
            msg = (
                f"Unsupported binary modality='{modality}' for member '{member_name}' "
                "in WebDatasetReaderStage (supported: image)"
            )
            raise ValueError(msg)
        if not self._loads_modality(modality):
            return []
        sid, position = self._next_sample_and_position(state.sample_counters, member_name, modality)
        return [
            self._image_row(
                sid=sid,
                position=position,
                source=source,
                content_key=member_name,
                binary_content=payload if self.load_binary else None,
            )
        ]

    def _next_sample_and_position(
        self,
        sample_counters: dict[str, int],
        member_name: str,
        modality: str,
    ) -> tuple[str, int]:
        sid_guess = member_stem(member_name)
        fallback_position = sample_counters.get(sid_guess, 0)
        sid, position = parse_sample_and_position(member_name, modality, self.sample_format, fallback_position)
        sample_counters[sid] = max(sample_counters.get(sid, 0), position + 1)
        return sid, position

    def _loads_modality(self, modality: str) -> bool:
        return self.modalities_to_load in {"all", modality}


@dataclass
class WebDatasetReader(CompositeStage[_EmptyTask, MultimodalBatch]):
    """Composite WebDataset reader from shard paths to ``MultimodalBatch`` outputs."""

    file_paths: str | list[str]
    files_per_partition: int | None = None
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: list(_DEFAULT_WEBDATASET_EXTENSIONS))
    limit: int | None = None
    load_binary: bool = False
    sample_format: SampleFormat = "interleaved"
    error_handling: Literal["raise", "skip", "log"] = "log"
    modalities_to_load: Literal["all", "image", "text"] = "all"
    interleaved_field_map: dict[str, str] | None = None
    max_batch_bytes: int | None = None
    storage_options: dict[str, Any] = field(default_factory=dict)
    name: str = "webdataset_reader"

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
            WebDatasetReaderStage(
                load_binary=self.load_binary,
                sample_format=self.sample_format,
                error_handling=self.error_handling,
                modalities_to_load=self.modalities_to_load,
                interleaved_field_map=self.interleaved_field_map,
                max_batch_bytes=self.max_batch_bytes,
                storage_options=self.storage_options,
            ),
        ]

    def get_description(self) -> str:
        parts = [f"Read WebDataset shards from {self.file_paths}"]
        if self.files_per_partition:
            parts.append(f"with {self.files_per_partition} files per partition")
        elif self.blocksize:
            parts.append(f"with target blocksize {self.blocksize}")
        if self.limit is not None:
            parts.append(f"limited to {self.limit} partitions")
        parts.append(f"sample_format={self.sample_format}")
        parts.append(f"modalities_to_load={self.modalities_to_load}")
        return ", ".join(parts)
