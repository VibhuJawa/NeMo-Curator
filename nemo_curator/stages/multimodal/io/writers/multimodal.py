# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, Literal

import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from nemo_curator.stages.multimodal.io.writers.base import BaseMultimodalWriterStage
from nemo_curator.utils.file_utils import (
    open_binary_writer,
)
from nemo_curator.utils.multimodal_utils import METADATA_MODALITY
from nemo_curator.utils.webdataset_utils import (
    DEFAULT_BINARY_CONTENT_TYPE,
    webdataset_member_name,
)

if TYPE_CHECKING:
    from nemo_curator.tasks import MultimodalBatch

_SUPPORTED_OUTPUT_FORMATS: Final[set[str]] = {"parquet", "arrow", "webdataset"}
_DEFAULT_SUFFIX_BY_FORMAT: Final[dict[str, str]] = {"parquet": "parquet", "arrow": "arrow", "webdataset": "tar"}
_SUPPORTED_IMAGE_PAYLOAD_POLICIES: Final[set[str]] = {"preserve", "materialize", "dematerialize"}
_SUPPORTED_MATERIALIZE_FAILURE_POLICIES: Final[set[str]] = {"raise", "drop_image"}
OutputFormat = Literal["parquet", "arrow", "webdataset"]
ImagePayloadPolicy = Literal["preserve", "materialize", "dematerialize"]
MaterializeFailurePolicy = Literal["raise", "drop_image"]


@dataclass
class MultimodalWriterStage(BaseMultimodalWriterStage):
    """Write ``MultimodalBatch`` data and metadata artifacts.

    Data output formats:
    - ``parquet``
    - ``arrow``
    - ``webdataset`` (tar with deterministic member naming)

    Output paths are resolved per-task using the task id to avoid write collisions.

    Base-class extension methods implemented here:
    - ``configure``: validates writer policies and sets derived
      output contracts (data suffix and metadata format).
    - ``prepare_task``: applies image payload policy
      (materialize/dematerialize/preserve) before writing.
    - ``write_data``: writes parquet/arrow/webdataset data output.

    Lazy image payload handling:
    - ``task.materialize(modality="image")`` loads missing image bytes.
    - ``task.dematerialize(modality="image")`` clears loaded image bytes.
    - ``image_payload_policy`` controls image payload handling across formats.
    - ``webdataset`` output requires image bytes; ``dematerialize`` is unsupported.
      With ``preserve``, lazy image payloads are materialized automatically.

    Args:
        output_path: Base output location. This stage always resolves a per-task output
            path from this base.
        output_format: Output artifact format (``parquet``, ``arrow``, ``webdataset``).
        image_payload_policy: Controls image payload handling before write:
            - ``preserve``: keep payloads as-is
            - ``materialize``: load missing image payloads before write
            - ``dematerialize``: clear image payload bytes before write
        materialize_failure_policy: Behavior when URL/file materialization fails:
            - ``raise``: fail the write
            - ``drop_image``: for webdataset output, drop failed image rows and continue
        materialize_max_retries: Retry attempts per failing path during materialization.
        materialize_retry_backoff_sec: Base exponential backoff seconds between retries.
        mode: Output collision policy.
            - ``overwrite``: write result, replacing existing artifact
            - ``error``: fail if the target artifact already exists
        storage_options: fsspec options used for remote output writes.
    """

    output_format: OutputFormat = "parquet"
    image_payload_policy: ImagePayloadPolicy = "preserve"
    materialize_failure_policy: MaterializeFailurePolicy = "raise"
    materialize_max_retries: int = 1
    materialize_retry_backoff_sec: float = 0.25
    name: str = "multimodal_writer"

    def configure(self) -> None:
        """Validate writer format options and derive output contract settings."""
        normalized = self.output_format.strip().lower()
        if normalized not in _SUPPORTED_OUTPUT_FORMATS:
            msg = f"Unsupported output_format='{self.output_format}'. Expected one of: parquet, arrow, webdataset"
            raise ValueError(msg)
        self.output_format = normalized  # type: ignore[assignment]
        payload_policy = self.image_payload_policy.strip().lower()
        if payload_policy not in _SUPPORTED_IMAGE_PAYLOAD_POLICIES:
            msg = (
                "Unsupported image_payload_policy="
                f"'{self.image_payload_policy}'. Expected one of: preserve, materialize, dematerialize"
            )
            raise ValueError(msg)
        self.image_payload_policy = payload_policy  # type: ignore[assignment]
        failure_policy = self.materialize_failure_policy.strip().lower()
        if failure_policy not in _SUPPORTED_MATERIALIZE_FAILURE_POLICIES:
            msg = (
                "Unsupported materialize_failure_policy="
                f"'{self.materialize_failure_policy}'. Expected one of: raise, drop_image"
            )
            raise ValueError(msg)
        self.materialize_failure_policy = failure_policy  # type: ignore[assignment]
        if self.materialize_max_retries < 0:
            msg = f"materialize_max_retries must be >= 0, got {self.materialize_max_retries}"
            raise ValueError(msg)
        if self.materialize_retry_backoff_sec < 0:
            msg = f"materialize_retry_backoff_sec must be >= 0, got {self.materialize_retry_backoff_sec}"
            raise ValueError(msg)
        if self.output_format == "webdataset" and self.image_payload_policy == "dematerialize":
            msg = "image_payload_policy='dematerialize' is incompatible with webdataset output"
            raise ValueError(msg)
        self.data_suffix = _DEFAULT_SUFFIX_BY_FORMAT[self.output_format]

    def write_data(self, task: MultimodalBatch, output_path: str) -> None:
        if self.output_format == "webdataset":
            self._write_webdataset_tar(task, output_path)
            return
        self._write_tabular_data_artifact(task, output_path, self.output_format)

    def prepare_task(self, task: MultimodalBatch) -> MultimodalBatch:
        effective_policy = self._effective_image_payload_policy()
        if effective_policy == "materialize":
            if not task.is_lazy:
                return task
            on_error: Literal["raise", "skip"] = "raise" if self.materialize_failure_policy == "raise" else "skip"
            write_task = task.materialize(
                modality="image",
                storage_options=self.storage_options,
                max_retries=self.materialize_max_retries,
                retry_backoff_sec=self.materialize_retry_backoff_sec,
                on_error=on_error,
            )
            if self.output_format == "webdataset" and self.materialize_failure_policy == "drop_image":
                write_task = self._drop_image_rows_with_missing_payload(write_task)
            return write_task
        if effective_policy == "dematerialize":
            return task.dematerialize(modality="image")
        return task

    def _effective_image_payload_policy(self) -> ImagePayloadPolicy:
        """Resolve runtime image payload policy for this output format."""
        if self.output_format != "webdataset":
            return self.image_payload_policy
        if self.image_payload_policy == "preserve":
            return "materialize"
        return self.image_payload_policy

    @staticmethod
    def _drop_image_rows_with_missing_payload(task: MultimodalBatch) -> MultimodalBatch:
        missing_image_mask = pc.and_(pc.equal(task.data["modality"], "image"), pc.is_null(task.data["binary_content"]))
        if not bool(pc.any(missing_image_mask).as_py()):
            return task
        failed_image_count = task.data.filter(missing_image_mask).num_rows
        logger.warning("Dropping {} image rows with missing payloads", failed_image_count)
        return BaseMultimodalWriterStage._filter_task_rows(task, pc.invert(missing_image_mask))

    def _write_webdataset_tar(self, task: MultimodalBatch, output_path: str) -> None:
        with open_binary_writer(output_path, self.storage_options) as raw:
            self._write_webdataset_to_fileobj(task, raw)

    @staticmethod
    def _write_webdataset_to_fileobj(task: MultimodalBatch, fileobj: BinaryIO) -> None:
        table = task.data.take(
            pc.sort_indices(
                task.data,
                sort_keys=[("sample_id", "ascending"), ("position", "ascending"), ("modality", "ascending")],
            )
        )
        rows_by_sample = MultimodalWriterStage._rows_grouped_by_sample(table)

        with tarfile.open(fileobj=fileobj, mode="w|") as tf:
            for sid, sample_rows in rows_by_sample.items():
                json_member_name, json_payload, image_members = MultimodalWriterStage._build_sample_artifacts(
                    sample_id=sid,
                    sample_rows=sample_rows,
                )
                json_info = tarfile.TarInfo(name=json_member_name)
                json_info.size = len(json_payload)
                tf.addfile(json_info, BytesIO(json_payload))

                for image_member_name, image_payload in image_members:
                    info = tarfile.TarInfo(name=image_member_name)
                    info.size = len(image_payload)
                    tf.addfile(info, BytesIO(image_payload))

    @staticmethod
    def _rows_grouped_by_sample(table: pa.Table) -> dict[str, list[dict[str, object]]]:
        rows_by_sample: dict[str, list[dict[str, object]]] = {}
        for row in table.to_pylist():
            sid = str(row["sample_id"])
            rows_by_sample.setdefault(sid, []).append(row)
        return rows_by_sample

    @staticmethod
    def _build_sample_artifacts(
        *,
        sample_id: str,
        sample_rows: list[dict[str, object]],
    ) -> tuple[str, bytes, list[tuple[str, bytes]]]:
        json_root: dict[str, object] = {"sample_id": sample_id}
        texts: list[str | None] = []
        images: list[str | None] = []
        segments: list[dict[str, object]] = []
        image_members: list[tuple[str, bytes]] = []

        for row in sample_rows:
            modality = str(row["modality"])
            position = int(row["position"])
            if modality == METADATA_MODALITY:
                MultimodalWriterStage._merge_sample_metadata(json_root, row)
                continue
            if modality == "text":
                MultimodalWriterStage._append_text_entry(
                    row=row,
                    texts=texts,
                    images=images,
                    segments=segments,
                )
                continue
            if modality != "image":
                msg = f"Unsupported modality='{modality}' for sample_id='{sample_id}'"
                raise ValueError(msg)
            member_name, image_payload = MultimodalWriterStage._build_image_member(
                sample_id=sample_id,
                position=position,
                content_key=row["content_key"],
                content_type=row["content_type"],
                binary_content=row["binary_content"],
            )
            image_members.append((member_name, image_payload))
            texts.append(None)
            images.append(Path(member_name).stem)

        json_root["texts"] = texts
        json_root["images"] = images
        json_root["segments"] = segments
        json_member_name = webdataset_member_name(sample_id, 0, "json")
        json_payload = json.dumps(json_root, ensure_ascii=True).encode("utf-8")
        return json_member_name, json_payload, image_members

    @staticmethod
    def _merge_sample_metadata(json_root: dict[str, object], row: dict[str, object]) -> None:
        sample_metadata = MultimodalWriterStage._parse_json_or_raw(
            row["element_metadata_json"] if row["element_metadata_json"] is not None else row["text_content"]
        )
        if isinstance(sample_metadata, dict):
            for key, value in sample_metadata.items():
                if key in {"sample_id", "segments", "texts", "images"}:
                    continue
                json_root[key] = value
            return
        if sample_metadata is not None:
            json_root["metadata"] = sample_metadata

    @staticmethod
    def _append_text_entry(
        *,
        row: dict[str, object],
        texts: list[str | None],
        images: list[str | None],
        segments: list[dict[str, object]],
    ) -> None:
        text_value = str(row["text_content"] or "")
        texts.append(text_value)
        images.append(None)
        segment: dict[str, object] = {"modality": "text", "text": text_value}
        text_meta = MultimodalWriterStage._parse_json_or_raw(row["element_metadata_json"])
        if text_meta is not None:
            segment["element_metadata_json"] = text_meta
        segments.append(segment)

    @staticmethod
    def _build_image_member(
        *,
        sample_id: str,
        position: int,
        content_key: object | None,
        content_type: object | None,
        binary_content: object | None,
    ) -> tuple[str, bytes]:
        suffix, image_payload = MultimodalWriterStage._image_suffix_and_payload(
            sample_id=sample_id,
            position=position,
            content_key=content_key,
            content_type=content_type,
            binary_content=binary_content,
        )
        return webdataset_member_name(sample_id, position, suffix), image_payload

    @staticmethod
    def _parse_json_or_raw(value: object | None) -> object | None:
        if value is None:
            return None
        text = str(value)
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return text

    @staticmethod
    def _image_suffix_and_payload(
        *,
        sample_id: str,
        position: int,
        content_key: object | None,
        content_type: object | None,
        binary_content: object | None,
    ) -> tuple[str, bytes]:
        """Build suffix/payload for one image row."""
        if content_key is not None:
            suffix = Path(str(content_key)).suffix.lstrip(".") or "bin"
        else:
            ctype = str(content_type) if content_type is not None else DEFAULT_BINARY_CONTENT_TYPE
            suffix = ctype.partition("/")[2] or "bin"

        if binary_content is None:
            msg = f"Missing binary_content for sample_id={sample_id} position={position}"
            raise ValueError(msg)
        return suffix, bytes(binary_content)
