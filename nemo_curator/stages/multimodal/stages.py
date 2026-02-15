# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import importlib
import re
import tarfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Final, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, MultimodalBatch
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

_SUPPORTED_OUTPUT_FORMATS: Final[set[str]] = {"parquet", "arrow", "lance", "webdataset"}
OutputFormat = Literal["parquet", "arrow", "lance", "webdataset"]
_DEFAULT_BINARY_CONTENT_TYPE: Final[str] = "application/octet-stream"
_IMAGE_SUFFIX_TO_CONTENT_TYPE: Final[dict[str, str]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
_IMAGE_SUFFIXES: Final[tuple[str, ...]] = tuple(_IMAGE_SUFFIX_TO_CONTENT_TYPE.keys())
_INDEXED_STEM_RE = re.compile(r"^(?P<sample>.+)\.(?P<position>\d+)$")


@contextmanager
def _open_tar(path: str) -> tarfile.TarFile:
    with tarfile.open(path, "r") as tf:
        yield tf


@dataclass
class WebDatasetReaderStage(ProcessingStage[FileGroupTask, MultimodalBatch]):
    """Read WebDataset-style tar shards into a multimodal Arrow table.

    This stage consumes a :class:`~nemo_curator.tasks.FileGroupTask` whose
    ``data`` field is an iterable of filesystem paths to WebDataset tar files.
    Each member file within the tar is interpreted as a sample for a specific
    modality (for example, image or text) and converted into rows that conform
    to :data:`~nemo_curator.tasks.multimodal.MULTIMODAL_SCHEMA`. The resulting
    rows are aggregated into a :class:`~nemo_curator.tasks.MultimodalBatch`.

    Parameters
    ----------
    load_binary
        If ``True``, load the raw bytes of non-text members (such as images)
        into the ``binary_content`` column of the output table. If ``False``,
        binary payloads are not materialized and only metadata columns are
        populated.
    name
        Logical name of the stage used in pipeline definitions and metrics.

    Notes
    -----
    The output rows are sorted by ``sample_id``, ``position``, and ``modality``
    before being assembled into an Arrow table.
    """
    load_binary: bool = False
    name: str = "webdataset_reader"

    @staticmethod
    def _infer_image_content_type(member_name: str) -> str:
        suffix = Path(member_name).suffix.lower()
        return _IMAGE_SUFFIX_TO_CONTENT_TYPE.get(suffix, _DEFAULT_BINARY_CONTENT_TYPE)

    @staticmethod
    def _parse_sample_and_position(member_name: str, modality: str) -> tuple[str, int]:
        stem = Path(member_name).stem
        match = _INDEXED_STEM_RE.match(stem)
        if match:
            return match.group("sample"), int(match.group("position"))
        return stem, 0 if modality == "text" else 1

    @staticmethod
    def _read_member_bytes(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes | None:
        payload = tf.extractfile(member)
        return payload.read() if payload else None

    def _row_for_member(
        self,
        tf: tarfile.TarFile,
        tar_path: str,
        source_shard: str,
        member: tarfile.TarInfo,
    ) -> dict[str, object] | None:
        name = member.name
        if name.endswith((".txt", ".json")):
            sid, position = self._parse_sample_and_position(name, "text")
            payload = self._read_member_bytes(tf, member)
            return {
                "sample_id": sid,
                "position": position,
                "modality": "text",
                "content_type": "application/json" if name.endswith(".json") else "text/plain",
                "text_content": payload.decode("utf-8") if payload else "",
                "binary_content": None,
                "source_id": sid,
                "source_shard": source_shard,
                "content_path": None,
                "content_key": None,
            }
        if name.endswith(_IMAGE_SUFFIXES):
            sid, position = self._parse_sample_and_position(name, "image")
            return {
                "sample_id": sid,
                "position": position,
                "modality": "image",
                "content_type": self._infer_image_content_type(name),
                "text_content": None,
                "binary_content": self._read_member_bytes(tf, member) if self.load_binary else None,
                "source_id": sid,
                "source_shard": source_shard,
                "content_path": tar_path,
                "content_key": name,
            }
        return None

    def _rows_from_tar(self, tar_path: str) -> list[dict[str, object]]:
        source_shard = Path(tar_path).name
        rows: list[dict[str, object]] = []
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                row = self._row_for_member(tf, tar_path, source_shard, member)
                if row is not None:
                    rows.append(row)
        return rows

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(MULTIMODAL_SCHEMA.names)

    def process(self, task: FileGroupTask) -> MultimodalBatch:
        rows: list[dict[str, object]] = []
        for tar_path in task.data:
            rows.extend(self._rows_from_tar(tar_path))
        rows.sort(key=lambda row: (str(row["sample_id"]), int(row["position"]), str(row["modality"])))
        table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
        return MultimodalBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class MultimodalWriterStage(ProcessingStage[MultimodalBatch, FileGroupTask]):
    """Write multimodal rows to parquet, arrow, lance, or webdataset tar output.

    For tabular formats (``parquet``, ``arrow``, ``lance``), the stage writes a
    normalized output table with sample id, position, modality, and text. For ``webdataset``,
    it writes tar members grouped by sample key with suffix-based sidecars/files.

    Args:
        output_path: Output artifact path.
        output_format: Output format. Supported values are ``parquet``,
            ``arrow``, ``lance``, and ``webdataset``.
        output_parquet: Legacy alias for ``output_path``.
        name: Stage identifier used by pipeline metadata/perf tracking.
    """

    output_path: str | None = None
    output_format: OutputFormat = "parquet"
    output_parquet: str | None = None
    _resolved_output_path: str = field(init=False, repr=False)
    name: str = "multimodal_writer"

    def __post_init__(self) -> None:
        if self.output_path and self.output_parquet and self.output_path != self.output_parquet:
            msg = "output_path and output_parquet refer to different locations; specify only one output target"
            raise ValueError(msg)

        resolved_output_path = self.output_path or self.output_parquet
        if resolved_output_path is None:
            msg = "MultimodalWriterStage requires output_path (or legacy output_parquet)"
            raise ValueError(msg)
        self.output_path = resolved_output_path
        self._resolved_output_path = resolved_output_path

        normalized = self.output_format.strip().lower()
        if normalized not in _SUPPORTED_OUTPUT_FORMATS:
            msg = f"Unsupported output_format='{self.output_format}'. Expected one of: parquet, arrow, lance, webdataset"
            raise ValueError(msg)
        self.output_format = normalized  # type: ignore[assignment]

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _build_output_table(self, task: MultimodalBatch) -> pa.Table:
        table = task.data.sort_by([("sample_id", "ascending"), ("position", "ascending")])
        num_rows = table.num_rows
        null_text = pa.nulls(num_rows, type=pa.string())
        text_mask = pc.equal(table["modality"], "text")

        return pa.table(
            {
                "sample_id": table["sample_id"],
                "position": table["position"],
                "modality": table["modality"],
                "text": pc.if_else(text_mask, table["text_content"], null_text),
            }
        )

    def _write_table(self, table: pa.Table) -> None:
        output_path = self._resolved_output_path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.output_format == "parquet":
            pq.write_table(table, output_path)
        elif self.output_format == "arrow":
            with pa.OSFile(output_path, "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        else:  # self.output_format == "lance"
            lance_module = importlib.import_module("lance")
            lance_module.write_dataset(table, output_path, mode="overwrite")

    @staticmethod
    def _resolve_image_payload(
        row: dict[str, object],
        stack: ExitStack,
        tar_handles: dict[str, tarfile.TarFile],
        file_payloads: dict[str, bytes],
    ) -> bytes:
        binary = row.get("binary_content")
        if binary is not None:
            return bytes(binary)

        content_path = row.get("content_path")
        if content_path is None:
            msg = f"Missing binary_content for image sample_id={row['sample_id']} position={row['position']}"
            raise ValueError(msg)
        content_path = str(content_path)

        content_key = row.get("content_key")
        if content_key:
            tf = tar_handles.get(content_path)
            if tf is None:
                tf = stack.enter_context(_open_tar(content_path))
                tar_handles[content_path] = tf
            extracted = tf.extractfile(str(content_key))
            if extracted is None:
                msg = f"Missing tar member '{content_key}' in '{content_path}'"
                raise FileNotFoundError(msg)
            return extracted.read()
        payload = file_payloads.get(content_path)
        if payload is None:
            with open(content_path, "rb") as f:
                payload = f.read()
            file_payloads[content_path] = payload
        return payload

    @staticmethod
    def _row_sort_key(row: dict[str, object]) -> tuple[str, int, int]:
        modality = str(row["modality"])
        modality_order = 0 if modality == "text" else 1 if modality == "image" else 2
        return str(row["sample_id"]), int(row["position"]), modality_order

    @staticmethod
    def _text_suffix_and_payload(row: dict[str, object]) -> tuple[str, bytes]:
        suffix = "json" if row.get("content_type") == "application/json" else "txt"
        payload = str(row.get("text_content") or "").encode("utf-8")
        return suffix, payload

    @staticmethod
    def _image_suffix(row: dict[str, object]) -> str:
        key = row.get("content_key")
        key_suffix = Path(str(key)).suffix if key else ""
        if key_suffix:
            return key_suffix.lstrip(".")

        ctype_suffix = str(row.get("content_type") or "").partition("/")[2]
        return ctype_suffix or "bin"

    def _suffix_and_payload_for_row(
        self,
        row: dict[str, object],
        stack: ExitStack,
        tar_handles: dict[str, tarfile.TarFile],
        file_payloads: dict[str, bytes],
    ) -> tuple[str, bytes] | None:
        modality = str(row["modality"])
        if modality == "text":
            return self._text_suffix_and_payload(row)
        if modality == "image":
            return self._image_suffix(row), self._resolve_image_payload(row, stack, tar_handles, file_payloads)
        return None

    @staticmethod
    def _assert_unique_suffix(seen_suffixes: set[tuple[str, str]], sample_id: str, suffix: str) -> None:
        pair = (sample_id, suffix)
        if pair in seen_suffixes:
            msg = f"Duplicate webdataset suffix '{suffix}' for sample_id='{sample_id}'"
            raise ValueError(msg)
        seen_suffixes.add(pair)

    @staticmethod
    def _write_member(tf: tarfile.TarFile, sample_id: str, suffix: str, payload: bytes) -> None:
        info = tarfile.TarInfo(name=f"{sample_id}.{suffix}")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

    def _as_file_group_task(self, task: MultimodalBatch) -> FileGroupTask:
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[self._resolved_output_path],
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def _write_webdataset_tar(self, task: MultimodalBatch) -> None:
        output_path = self._resolved_output_path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(task.data.to_pylist(), key=self._row_sort_key)
        seen_suffixes: set[tuple[str, str]] = set()
        with ExitStack() as stack, tarfile.open(output_path, "w") as tf:
            tar_handles: dict[str, tarfile.TarFile] = {}
            file_payloads: dict[str, bytes] = {}
            for row in rows:
                suffix_and_payload = self._suffix_and_payload_for_row(row, stack, tar_handles, file_payloads)
                if suffix_and_payload is None:
                    continue
                sample_id = str(row["sample_id"])
                suffix, payload = suffix_and_payload
                self._assert_unique_suffix(seen_suffixes, sample_id, suffix)
                self._write_member(tf, sample_id, suffix, payload)

    def process(self, task: MultimodalBatch) -> FileGroupTask:
        if self.output_format == "webdataset":
            self._write_webdataset_tar(task)
        else:
            out_table = self._build_output_table(task)
            self._write_table(out_table)
        return self._as_file_group_task(task)


@dataclass
class MetadataWriterStage(MultimodalWriterStage):
    """Write ``MultimodalBatch.metadata_index`` to a tabular output artifact.

    The stage reuses writer output backends from :class:`MultimodalWriterStage`
    and serializes the metadata table directly (sorted by ``sample_id`` when
    available).

    Args:
        output_path: Output artifact path.
        output_format: Output format (``parquet``, ``arrow``, or ``lance``).
        output_parquet: Legacy alias for ``output_path``.
        name: Stage identifier used by pipeline metadata/perf tracking.
    """

    name: str = "metadata_writer"

    def _build_output_table(self, task: MultimodalBatch) -> pa.Table:
        metadata_index = task.metadata_index
        if metadata_index is None:
            msg = "MetadataWriterStage requires `task.metadata_index`."
            raise ValueError(msg)
        if "sample_id" in metadata_index.column_names:
            return metadata_index.sort_by([("sample_id", "ascending")])
        return metadata_index
