# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import io
import json
import mimetypes
import re
import tarfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import fsspec
import pandas as pd
from loguru import logger

from .base import BaseMultimodalWriter

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.tasks import MultiBatchTask

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
}

_SANITIZE_RE = re.compile(r"[^\w\-.]")


def _sanitize_key(raw: str) -> str:
    """Produce a filesystem-safe WebDataset key from a raw sample_id."""
    return _SANITIZE_RE.sub("_", raw)[:200]


def _ext_from_content_type(content_type: object) -> str:
    if isinstance(content_type, str) and content_type in _MIME_TO_EXT:
        return _MIME_TO_EXT[content_type]
    if isinstance(content_type, str):
        ext = mimetypes.guess_extension(content_type, strict=False)
        if ext:
            return ext.lstrip(".")
    return "bin"


def _add_tar_member(tf: tarfile.TarFile, name: str, data: bytes, mtime: float) -> None:
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = mtime
    ti.mode = 0o0444
    ti.uname = "bigdata"
    ti.gname = "bigdata"
    tf.addfile(ti, io.BytesIO(data))


@dataclass
class _ColumnArrays:
    """Pre-extracted numpy arrays from the DataFrame for fast row-level access."""

    modality: np.ndarray
    position: np.ndarray
    text_content: np.ndarray
    binary_content: np.ndarray
    content_type: np.ndarray
    metadata_json: np.ndarray


def _build_index(sid_col: np.ndarray) -> list[tuple[str, list[int]]]:
    """Return (sample_id, row_indices) pairs in first-occurrence order."""
    sid_to_indices: dict[str, list[int]] = {}
    insertion_order: list[str] = []
    for i, raw in enumerate(sid_col):
        sid = str(raw)
        if sid not in sid_to_indices:
            sid_to_indices[sid] = []
            insertion_order.append(sid)
        sid_to_indices[sid].append(i)
    return [(sid, sid_to_indices[sid]) for sid in insertion_order]


def _extract_metadata_payload(meta_val: object) -> dict[str, Any]:
    if meta_val is None or pd.isna(meta_val):
        return {}
    try:
        parsed = json.loads(str(meta_val))
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    parsed.pop("_sample_source", None)
    return parsed


def _collect_images(
    image_entries: list[tuple[int, object, object]],
) -> tuple[list[str | None], list[tuple[str, bytes]]]:
    image_entries.sort(key=lambda x: x[0])
    images: list[str | None] = []
    binaries: list[tuple[str, bytes]] = []
    ext_counter: dict[str, int] = {}
    for _, binary, content_type in image_entries:
        has_binary = binary is not None and not pd.isna(binary) and isinstance(binary, (bytes, bytearray))
        if has_binary:
            ext = _ext_from_content_type(content_type)
            count = ext_counter.get(ext, 0)
            ext_counter[ext] = count + 1
            member_key = ext if count == 0 else f"{count}.{ext}"
            binaries.append((member_key, bytes(binary)))
            images.append(member_key)
        else:
            images.append(None)
    return images, binaries


def _write_sample(tf: tarfile.TarFile, key: str, indices: list[int], cols: _ColumnArrays, mtime: float) -> None:
    payload: dict[str, Any] = {}
    text_entries: list[tuple[int, object]] = []
    image_entries: list[tuple[int, object, object]] = []

    for idx in indices:
        mod = str(cols.modality[idx])
        if mod == "metadata":
            payload.update(_extract_metadata_payload(cols.metadata_json[idx]))
        elif mod == "text":
            text_entries.append((int(cols.position[idx]), cols.text_content[idx]))
        elif mod == "image":
            image_entries.append((int(cols.position[idx]), cols.binary_content[idx], cols.content_type[idx]))

    text_entries.sort(key=lambda x: x[0])
    payload["texts"] = [str(v) if v is not None and not pd.isna(v) else None for _, v in text_entries]
    images, binaries = _collect_images(image_entries)
    payload["images"] = images

    json_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    _add_tar_member(tf, f"{key}.json", json_bytes, mtime)
    for member_key, binary_data in binaries:
        _add_tar_member(tf, f"{key}.{member_key}", binary_data, mtime)


@dataclass
class MultimodalWebdatasetWriterStage(BaseMultimodalWriter):
    """Write multimodal rows to WebDataset tar shards."""

    file_extension: str = "tar"
    name: str = "multimodal_webdataset_writer"

    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        pass

    def write_data(self, task: MultiBatchTask, file_path: str) -> None:
        with self._time_metric("materialize_dataframe_total_s"):
            df = self._materialize_dataframe(task)

        with self._time_metric("webdataset_write_s"):
            self._write_tar(df, file_path)

    def _write_tar(self, df: pd.DataFrame, file_path: str) -> None:
        mtime = time.time()
        samples_written = 0

        cols = _ColumnArrays(
            modality=df["modality"].to_numpy(),
            position=df["position"].to_numpy(),
            text_content=df["text_content"].to_numpy(),
            binary_content=df["binary_content"].to_numpy(),
            content_type=df["content_type"].to_numpy(),
            metadata_json=df["metadata_json"].to_numpy(),
        )
        sample_index = _build_index(df["sample_id"].to_numpy())

        with (
            fsspec.open(file_path, mode="wb", **self.storage_options) as fobj,
            tarfile.open(fileobj=fobj, mode="w") as tf,
        ):
            for sid, indices in sample_index:
                _write_sample(tf, _sanitize_key(sid), indices, cols, mtime)
                samples_written += 1
                if samples_written % 10000 == 0:
                    logger.info(f"WebDataset writer: {samples_written}/{len(sample_index)} samples written")

        self._log_metric("samples_written", float(samples_written))
