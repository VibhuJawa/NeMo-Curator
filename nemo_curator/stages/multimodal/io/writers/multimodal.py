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
OutputFormat = Literal["parquet", "arrow", "webdataset"]


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

    Args:
        output_path: Base output location. This stage always resolves a per-task output
            path from this base.
        output_format: Output artifact format (``parquet``, ``arrow``, ``webdataset``).
        materialize_on_write: For ``webdataset`` output, controls whether lazy image rows
            are materialized before writing tar members.
        mode: Output collision policy.
            - ``overwrite``: write result, replacing existing artifact
            - ``error``: fail if the target artifact already exists
        storage_options: fsspec options used for remote output writes.
    """

    output_format: OutputFormat = "parquet"
    materialize_on_write: bool = True
    name: str = "multimodal_writer"

    def _configure_writer(self) -> None:
        """Validate writer format options and derive output contract settings."""
        normalized = self.output_format.strip().lower()
        if normalized not in _SUPPORTED_OUTPUT_FORMATS:
            msg = f"Unsupported output_format='{self.output_format}'. Expected one of: parquet, arrow, webdataset"
            raise ValueError(msg)
        self.output_format = normalized  # type: ignore[assignment]
        self.data_suffix = _DEFAULT_SUFFIX_BY_FORMAT[self.output_format]
        self.metadata_format = _METADATA_TABULAR_FORMAT_BY_DATA_FORMAT[self.output_format]

    def _write_data_artifact(self, task: MultimodalBatch, output_path: str) -> None:
        if self.output_format == "webdataset":
            self._write_webdataset_tar(task, output_path)
            return
        self._write_tabular_data_artifact(task, output_path, self.output_format)

    def _write_webdataset_tar(self, task: MultimodalBatch, output_path: str) -> None:
        if task.is_lazy and not self.materialize_on_write:
            msg = "WebDataset writer received a lazy batch; set materialize_on_write=True or materialize upstream"
            raise ValueError(msg)
        write_task = task.materialize(modality="image") if task.is_lazy else task
        with open_binary_writer(output_path, self.storage_options) as raw:
            self._write_webdataset_to_fileobj(write_task, raw)

    @staticmethod
    def _write_webdataset_to_fileobj(task: MultimodalBatch, fileobj: BinaryIO) -> None:
        sort_indices = pc.sort_indices(
            task.data,
            sort_keys=[("sample_id", "ascending"), ("position", "ascending"), ("modality", "ascending")],
        ).to_pylist()
        text_indices_by_sample: dict[str, list[int]] = {}
        for idx in sort_indices:
            if str(task.data["modality"][idx].as_py()) != "text":
                continue
            sample_id = str(task.data["sample_id"][idx].as_py())
            text_indices_by_sample.setdefault(sample_id, []).append(idx)

        first_text_index_by_sample: dict[str, int] = {}
        text_payload_by_sample: dict[str, tuple[str, bytes]] = {}
        for sample_id, indices in text_indices_by_sample.items():
            first_idx = indices[0]
            first_text_index_by_sample[sample_id] = first_idx
            if len(indices) == 1:
                text_payload_by_sample[sample_id] = MultimodalWriterStage._row_suffix_and_payload(task, first_idx)
                continue
            logger.warning(
                "Collapsing {} text rows into one text member for sample_id='{}'",
                len(indices),
                sample_id,
            )
            merged_text = "\n".join(str(task.data["text_content"][idx].as_py() or "") for idx in indices)
            text_payload_by_sample[sample_id] = ("txt", merged_text.encode("utf-8"))

        with tarfile.open(fileobj=fileobj, mode="w|") as tf:
            for idx in sort_indices:
                sample_id = str(task.data["sample_id"][idx].as_py())
                modality = str(task.data["modality"][idx].as_py())
                if modality == "text":
                    if idx != first_text_index_by_sample[sample_id]:
                        continue
                    suffix, payload = text_payload_by_sample[sample_id]
                else:
                    suffix, payload = MultimodalWriterStage._row_suffix_and_payload(task, idx)
                position = int(task.data["position"][idx].as_py())
                info = tarfile.TarInfo(name=webdataset_member_name(sample_id, position, suffix))
                info.size = len(payload)
                tf.addfile(info, BytesIO(payload))

    @staticmethod
    def _row_suffix_and_payload(task: MultimodalBatch, idx: int) -> tuple[str, bytes]:
        """Convert one multimodal row into a WebDataset member suffix and payload bytes.

        Returns:
            tuple[str, bytes]:
            - suffix: Member extension without leading dot (for example ``txt``, ``json``, ``jpg``)
            - payload: UTF-8 encoded text bytes or raw binary bytes for the row
        """
        modality = str(task.data["modality"][idx].as_py())
        if modality == "text":
            ctype_scalar = task.data["content_type"][idx]
            ctype = str(ctype_scalar.as_py()) if ctype_scalar.is_valid else "text/plain"
            suffix = "json" if ctype == "application/json" else "txt"
            payload = str(task.data["text_content"][idx].as_py() or "").encode("utf-8")
            return suffix, payload

        key_scalar = task.data["content_key"][idx]
        if key_scalar.is_valid:
            suffix = Path(str(key_scalar.as_py())).suffix.lstrip(".") or "bin"
        else:
            ctype_scalar = task.data["content_type"][idx]
            ctype = str(ctype_scalar.as_py()) if ctype_scalar.is_valid else DEFAULT_BINARY_CONTENT_TYPE
            suffix = ctype.partition("/")[2] or "bin"

        binary_scalar = task.data["binary_content"][idx]
        if not binary_scalar.is_valid:
            sample_id = task.data["sample_id"][idx].as_py()
            position = task.data["position"][idx].as_py()
            msg = f"Missing binary_content for sample_id={sample_id} position={position}"
            raise ValueError(msg)
        return suffix, bytes(binary_scalar.as_py())
