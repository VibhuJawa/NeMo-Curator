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

from nemo_curator.tasks.multimodal import RESERVED_COLUMNS

from .base import BaseMultimodalWriter

if TYPE_CHECKING:
    from nemo_curator.tasks import MultiBatchTask

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
}

_SANITIZE_RE = re.compile(r"[^\w\-]")


def _sanitize_key(raw: str) -> str:
    """Produce a filesystem-safe WebDataset key free of dots.

    The WebDataset format uses dots as extension separators, so the sample key
    (the part before the first dot) must not contain any dots.
    """
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


def _build_index(sid_col: list | pd.Series) -> list[tuple[str, list[int]]]:
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


def _safe_json_value(val: object) -> object:
    """Convert a value to a JSON-safe type."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (bytes, bytearray)):
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


def _write_sample(
    tf: tarfile.TarFile,
    key: str,
    sample_df: pd.DataFrame,
    extra_columns: list[str],
    mtime: float,
) -> None:
    payload: dict[str, Any] = {}
    text_at_pos: dict[int, object] = {}
    image_at_pos: dict[int, tuple[object, object]] = {}
    text_extra_at_pos: dict[int, dict[str, Any]] = {}
    image_extra_at_pos: dict[int, dict[str, Any]] = {}
    metadata_extra: dict[str, Any] = {}

    for _, row in sample_df.iterrows():
        mod = str(row["modality"])
        pos = int(row["position"])
        row_extra = {c: _safe_json_value(row[c]) for c in extra_columns} if extra_columns else {}
        if mod == "metadata":
            payload.update(_extract_metadata_payload(row["metadata_json"]))
            metadata_extra = row_extra
        elif mod == "text":
            text_at_pos[pos] = row["text_content"]
            text_extra_at_pos[pos] = row_extra
        elif mod == "image":
            image_at_pos[pos] = (row["binary_content"], row["content_type"])
            image_extra_at_pos[pos] = row_extra

    all_positions = set(text_at_pos) | set(image_at_pos)
    n = max(all_positions) + 1 if all_positions else 0

    texts: list[str | None] = [None] * n
    images: list[str | None] = [None] * n
    binaries: list[tuple[str, bytes]] = []

    for pos in range(n):
        if pos in text_at_pos:
            v = text_at_pos[pos]
            texts[pos] = str(v) if v is not None and not pd.isna(v) else None
        if pos in image_at_pos:
            binary, content_type = image_at_pos[pos]
            has_binary = binary is not None and not pd.isna(binary) and isinstance(binary, (bytes, bytearray))
            if has_binary:
                ext = _ext_from_content_type(content_type)
                member_suffix = f"{pos}.{ext}"
                binaries.append((f"{key}.{member_suffix}", bytes(binary)))
                images[pos] = member_suffix
            else:
                images[pos] = None

    payload["texts"] = texts
    payload["images"] = images

    if extra_columns:
        text_extra_list: list[dict[str, Any] | None] = [
            text_extra_at_pos.get(pos) for pos in range(n)
        ]
        image_extra_list: list[dict[str, Any] | None] = [
            image_extra_at_pos.get(pos) for pos in range(n)
        ]
        payload["_row_extra"] = {
            "text": text_extra_list,
            "image": image_extra_list,
            "metadata": metadata_extra,
        }

    json_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    _add_tar_member(tf, f"{key}.json", json_bytes, mtime)
    for member_name, binary_data in binaries:
        _add_tar_member(tf, member_name, binary_data, mtime)


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

        extra_columns = [c for c in df.columns if c not in RESERVED_COLUMNS]
        sample_index = _build_index(df["sample_id"].tolist())

        with (
            fsspec.open(file_path, mode="wb", **self.storage_options) as fobj,
            tarfile.open(fileobj=fobj, mode="w") as tf,
        ):
            for sid, indices in sample_index:
                sample_df = df.iloc[indices]
                _write_sample(tf, _sanitize_key(sid), sample_df, extra_columns, mtime)
                samples_written += 1
                if samples_written % 10000 == 0:
                    logger.info(f"WebDataset writer: {samples_written}/{len(sample_index)} samples written")

        self._log_metric("samples_written", float(samples_written))
