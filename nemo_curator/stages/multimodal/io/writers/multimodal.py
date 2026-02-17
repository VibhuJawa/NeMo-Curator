# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Final, Literal

import pyarrow.compute as pc
from loguru import logger

from nemo_curator.stages.multimodal.io.writers.base import BaseMultimodalWriterStage
from nemo_curator.utils.file_utils import (
    open_binary_writer,
)
from nemo_curator.utils.webdataset_utils import (
    DEFAULT_BINARY_CONTENT_TYPE,
    webdataset_member_name,
)

if TYPE_CHECKING:
    from nemo_curator.tasks import MultimodalBatch

_SUPPORTED_OUTPUT_FORMATS: Final[set[str]] = {"parquet", "arrow", "webdataset"}
_DEFAULT_SUFFIX_BY_FORMAT: Final[dict[str, str]] = {"parquet": "parquet", "arrow": "arrow", "webdataset": "tar"}
_METADATA_TABULAR_FORMAT_BY_DATA_FORMAT: Final[dict[str, Literal["parquet", "arrow"]]] = {
    "parquet": "parquet",
    "arrow": "arrow",
    "webdataset": "parquet",
}
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

    Metadata output:
    - For ``parquet``/``arrow`` data output, metadata uses the same tabular format.
    - For ``webdataset`` data output, metadata is written as parquet.

    Output paths are resolved per-task using the task id to avoid write collisions.

    Base-class extension methods implemented here:
    - ``_configure_writer``: validates writer policies and sets derived
      output contracts (data suffix and metadata format).
    - ``_prepare_task_for_write``: applies image payload policy
      (materialize/dematerialize/preserve) before writing.
    - ``_write_data_artifact``: writes parquet/arrow/webdataset data output.

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

    def _configure_writer(self) -> None:
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
        self.metadata_format = _METADATA_TABULAR_FORMAT_BY_DATA_FORMAT[self.output_format]

    def _write_data_artifact(self, task: MultimodalBatch, output_path: str) -> None:
        if self.output_format == "webdataset":
            self._write_webdataset_tar(task, output_path)
            return
        self._write_tabular_data_artifact(task, output_path, self.output_format)

    def _prepare_task_for_write(self, task: MultimodalBatch) -> MultimodalBatch:
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
        sample_ids = table["sample_id"].to_pylist()
        positions = table["position"].to_pylist()
        modalities = table["modality"].to_pylist()
        content_types = table["content_type"].to_pylist()
        text_contents = table["text_content"].to_pylist()
        binary_contents = table["binary_content"].to_pylist()
        content_keys = table["content_key"].to_pylist()

        first_text_index: dict[str, int] = {}
        text_row_count: dict[str, int] = {}
        merged_text_payload: dict[str, bytes] = {}
        for idx, (sample_id, modality) in enumerate(zip(sample_ids, modalities, strict=True)):
            sid = str(sample_id)
            if str(modality) != "text":
                continue
            if sid not in first_text_index:
                first_text_index[sid] = idx
                merged_text_payload[sid] = b""
                text_row_count[sid] = 0
            text_row_count[sid] += 1
            current = merged_text_payload[sid]
            text_bytes = str(text_contents[idx] or "").encode("utf-8")
            merged_text_payload[sid] = text_bytes if current == b"" else current + b"\n" + text_bytes

        with tarfile.open(fileobj=fileobj, mode="w|") as tf:
            for idx, (sample_id, modality) in enumerate(zip(sample_ids, modalities, strict=True)):
                sid = str(sample_id)
                if str(modality) == "text":
                    if idx != first_text_index[sid]:
                        continue
                    if text_row_count[sid] > 1:
                        logger.warning("Collapsing multiple text rows into one text member for sample_id='{}'", sid)
                    suffix, payload = MultimodalWriterStage._text_suffix_and_payload(
                        content_type=content_types[idx],
                        merged_text_payload=merged_text_payload[sid],
                    )
                else:
                    suffix, payload = MultimodalWriterStage._image_suffix_and_payload(
                        sample_id=sid,
                        position=int(positions[idx]),
                        content_key=content_keys[idx],
                        content_type=content_types[idx],
                        binary_content=binary_contents[idx],
                    )
                info = tarfile.TarInfo(name=webdataset_member_name(sid, int(positions[idx]), suffix))
                info.size = len(payload)
                tf.addfile(info, BytesIO(payload))

    @staticmethod
    def _text_suffix_and_payload(
        *,
        content_type: object | None,
        merged_text_payload: bytes,
    ) -> tuple[str, bytes]:
        """Build suffix/payload for collapsed text rows.

        Returns:
            tuple[str, bytes]:
            - suffix: Member extension without leading dot (for example ``txt``, ``json``, ``jpg``)
            - payload: UTF-8 encoded text bytes
        """
        ctype = str(content_type) if content_type is not None else "text/plain"
        suffix = "json" if ctype == "application/json" else "txt"
        return suffix, merged_text_payload

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
