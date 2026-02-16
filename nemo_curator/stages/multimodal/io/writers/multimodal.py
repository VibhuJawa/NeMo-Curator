# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import tarfile
from collections import defaultdict
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
OutputFormat = Literal["parquet", "arrow", "webdataset"]
ImagePayloadPolicy = Literal["preserve", "materialize", "dematerialize"]


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
        mode: Output collision policy.
            - ``overwrite``: write result, replacing existing artifact
            - ``error``: fail if the target artifact already exists
        storage_options: fsspec options used for remote output writes.
    """

    output_format: OutputFormat = "parquet"
    image_payload_policy: ImagePayloadPolicy = "preserve"
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
        effective_policy = self.image_payload_policy
        if self.output_format == "webdataset":
            if effective_policy == "dematerialize":
                msg = "image_payload_policy='dematerialize' is incompatible with webdataset output"
                raise ValueError(msg)
            if not self._has_image_rows(task):
                msg = "WebDataset output requires at least one image row in the batch"
                raise ValueError(msg)
            if effective_policy == "preserve":
                effective_policy = "materialize"
        if effective_policy == "materialize":
            return task.materialize(modality="image", storage_options=self.storage_options) if task.is_lazy else task
        if effective_policy == "dematerialize":
            return task.dematerialize(modality="image")
        return task

    def _write_webdataset_tar(self, task: MultimodalBatch, output_path: str) -> None:
        with open_binary_writer(output_path, self.storage_options) as raw:
            self._write_webdataset_to_fileobj(task, raw)

    @staticmethod
    def _has_image_rows(task: MultimodalBatch) -> bool:
        return bool(pc.any(pc.equal(task.data["modality"], "image")).as_py())

    @staticmethod
    def _write_webdataset_to_fileobj(task: MultimodalBatch, fileobj: BinaryIO) -> None:
        sorted_indices = pc.sort_indices(
            task.data,
            sort_keys=[("sample_id", "ascending"), ("position", "ascending"), ("modality", "ascending")],
        )
        sorted_rows = task.data.take(sorted_indices).to_pylist()

        text_positions_by_sample: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(sorted_rows):
            if str(row["modality"]) == "text":
                text_positions_by_sample[str(row["sample_id"])].append(idx)

        text_payload_by_sample: dict[str, tuple[str, bytes]] = {}
        first_text_pos_by_sample: dict[str, int] = {}
        for sample_id, text_positions in text_positions_by_sample.items():
            first_pos = text_positions[0]
            first_text_pos_by_sample[sample_id] = first_pos
            if len(text_positions) == 1:
                row = sorted_rows[first_pos]
                text_payload_by_sample[sample_id] = MultimodalWriterStage._row_suffix_and_payload(
                    row=row
                )
                continue
            logger.warning(
                "Collapsing {} text rows into one text member for sample_id='{}'",
                len(text_positions),
                sample_id,
            )
            merged_text = "\n".join(str(sorted_rows[pos]["text_content"] or "") for pos in text_positions)
            text_payload_by_sample[sample_id] = ("txt", merged_text.encode("utf-8"))

        with tarfile.open(fileobj=fileobj, mode="w|") as tf:
            for idx, row in enumerate(sorted_rows):
                sample_id = str(row["sample_id"])
                modality = str(row["modality"])
                if modality == "text":
                    if idx != first_text_pos_by_sample[sample_id]:
                        continue
                    suffix, payload = text_payload_by_sample[sample_id]
                else:
                    suffix, payload = MultimodalWriterStage._row_suffix_and_payload(
                        row=row
                    )
                position = int(row["position"])
                info = tarfile.TarInfo(name=webdataset_member_name(sample_id, position, suffix))
                info.size = len(payload)
                tf.addfile(info, BytesIO(payload))

    @staticmethod
    def _row_suffix_and_payload(
        *,
        row: dict[str, object],
    ) -> tuple[str, bytes]:
        """Convert one multimodal row into a WebDataset member suffix and payload bytes.

        Returns:
            tuple[str, bytes]:
            - suffix: Member extension without leading dot (for example ``txt``, ``json``, ``jpg``)
            - payload: UTF-8 encoded text bytes or raw binary bytes for the row
        """
        modality = str(row["modality"])
        if modality == "text":
            content_type = row["content_type"]
            text_content = row["text_content"]
            ctype = str(content_type) if content_type is not None else "text/plain"
            suffix = "json" if ctype == "application/json" else "txt"
            payload = str(text_content or "").encode("utf-8")
            return suffix, payload

        content_key = row["content_key"]
        if content_key is not None:
            suffix = Path(str(content_key)).suffix.lstrip(".") or "bin"
        else:
            content_type = row["content_type"]
            ctype = str(content_type) if content_type is not None else DEFAULT_BINARY_CONTENT_TYPE
            suffix = ctype.partition("/")[2] or "bin"

        binary_content = row["binary_content"]
        if binary_content is None:
            sample_id = str(row["sample_id"])
            position = int(row["position"])
            msg = f"Missing binary_content for sample_id={sample_id} position={position}"
            raise ValueError(msg)
        return suffix, bytes(binary_content)
