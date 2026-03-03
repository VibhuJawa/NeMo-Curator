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
import tarfile
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator

import fsspec
import pandas as pd
from fsspec.core import url_to_fs
from loguru import logger
from PIL import Image as _Image

from nemo_curator.tasks import InterleavedBatch

from .validation_utils import resolve_storage_options

_TAR_EXTENSIONS = (".tar", ".tar.gz", ".tgz")


class _ClassifiedRows(NamedTuple):
    tar_extract: dict[str, list[tuple[int, str, int | None]]]
    range_read: dict[str, list[tuple[int, str, int, int, int | None]]]
    direct_read: dict[str, list[int]]
    missing: list[int]


def _get_frame_index(df: pd.DataFrame, idx: int) -> int | None:
    if "_src_frame_index" not in df.columns:
        return None
    val = df.loc[idx, "_src_frame_index"]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return int(val)


def _classify_rows(
    df: pd.DataFrame,
    image_mask: pd.Series,
) -> _ClassifiedRows:
    """Partition pending image rows into three I/O strategy groups.

    - tar_extract: has member name but no byte_offset (must open tar and extractfile)
    - range_read: has member + byte_offset + byte_size (can use fs.cat_ranges)
    - direct_read: no member (path is the file itself)
    - missing: path is None/NaN
    """
    tar_extract: dict[str, list[tuple[int, str, int | None]]] = {}
    range_read: dict[str, list[tuple[int, str, int, int, int | None]]] = {}
    direct_read: dict[str, list[int]] = {}
    missing: list[int] = []

    for idx in df[image_mask].index:
        path = df.loc[idx, "_src_path"]
        if path is None or (isinstance(path, float) and pd.isna(path)) or path == "":
            missing.append(idx)
            continue

        path_str = str(path)
        raw_member = df.loc[idx, "_src_member"]
        has_member = raw_member not in (None, "") and pd.notna(raw_member)

        if not has_member:
            direct_read.setdefault(path_str, []).append(idx)
            continue

        member_str = str(raw_member)
        frame_idx = _get_frame_index(df, idx)
        raw_offset = df.loc[idx, "_src_byte_offset"]
        raw_size = df.loc[idx, "_src_byte_size"]
        has_range = raw_offset is not None and raw_size is not None and pd.notna(raw_offset) and pd.notna(raw_size)

        if has_range and int(raw_size) > 0:
            range_read.setdefault(path_str, []).append((idx, member_str, int(raw_offset), int(raw_size), frame_idx))
        else:
            tar_extract.setdefault(path_str, []).append((idx, member_str, frame_idx))

    return _ClassifiedRows(tar_extract=tar_extract, range_read=range_read, direct_read=direct_read, missing=missing)


def _extract_tiff_frame(tiff_bytes: bytes, frame_index: int) -> bytes | None:
    """Extract a single frame from a multi-frame TIFF, returning it as a single-frame TIFF.

    Returns the raw bytes unchanged if the data is not a TIFF.
    """
    try:
        with _Image.open(io.BytesIO(tiff_bytes)) as img:
            if img.format != "TIFF":
                return tiff_bytes
            if frame_index >= getattr(img, "n_frames", 1):
                return None
            img.seek(frame_index)
            compression = img.info.get("compression", "tiff_deflate")
            buf = io.BytesIO()
            img.save(buf, format="TIFF", compression=compression)
            return buf.getvalue()
    except (OSError, SyntaxError, ValueError):
        return None


def _apply_frame_extraction(
    payload: bytes,
    frame_idx: int | None,
    member: str,
) -> tuple[bytes | None, str | None]:
    """Return ``(binary, error)`` after optional TIFF frame extraction."""
    if frame_idx is not None:
        extracted = _extract_tiff_frame(payload, frame_idx)
        if extracted is None:
            return None, f"failed to extract frame {frame_idx} from '{member}'"
        payload = extracted
    return payload, None


def _fill_tar_extract_rows(
    groups: dict[str, list[tuple[int, str, int | None]]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    """Open each tar once and extract all needed members sequentially."""
    for path, keyed_rows in groups.items():
        key_cache: dict[str, bytes | None] = {}
        try:
            with fsspec.open(path, mode="rb", **storage_options) as fobj, tarfile.open(fileobj=fobj, mode="r:*") as tf:
                for idx, member, frame_idx in keyed_rows:
                    if member not in key_cache:
                        try:
                            extracted = tf.extractfile(member)
                        except KeyError:
                            extracted = None
                        key_cache[member] = extracted.read() if extracted is not None else None

                    payload = key_cache[member]
                    if payload is None:
                        error_values[idx] = f"missing member '{member}'"
                        continue

                    binary, err = _apply_frame_extraction(payload, frame_idx, member)
                    binary_values[idx] = binary
                    error_values[idx] = err
        except (OSError, tarfile.TarError):
            for idx, _, _ in keyed_rows:
                error_values[idx] = "failed to read path"


def _fill_range_read_rows(
    groups: dict[str, list[tuple[int, str, int, int, int | None]]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    """Batch byte-range reads per path using fs.cat_ranges()."""
    for path, entries in groups.items():
        try:
            fs, fs_path = url_to_fs(path, **storage_options)
        except (ValueError, OSError):
            for idx, *_ in entries:
                error_values[idx] = "failed to resolve filesystem"
            continue

        paths = [fs_path] * len(entries)
        starts = [offset for _, _, offset, _, _ in entries]
        ends = [offset + size for _, _, offset, size, _ in entries]

        try:
            blobs = fs.cat_ranges(paths, starts, ends)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(f"cat_ranges failed for {path} ({len(entries)} ranges): {exc}")
            for idx, *_ in entries:
                error_values[idx] = "cat_ranges failed"
            continue

        for (idx, member, _offset, _size, frame_idx), blob in zip(entries, blobs, strict=True):
            if isinstance(blob, Exception):
                error_values[idx] = f"range read error for member '{member}'"
                continue
            if blob is None or len(blob) == 0:
                error_values[idx] = f"empty range read for member '{member}'"
                continue
            payload = bytes(blob) if not isinstance(blob, bytes) else blob
            binary, err = _apply_frame_extraction(payload, frame_idx, member)
            binary_values[idx] = binary
            error_values[idx] = err


def _fill_direct_read_rows(
    groups: dict[str, list[int]],
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    """Read each direct file once, share bytes across all rows referencing it."""
    for path, row_idxs in groups.items():
        payload = _read_direct_file(path, storage_options)
        for idx in row_idxs:
            if payload is not None:
                binary_values[idx] = payload
                error_values[idx] = None
            else:
                error_values[idx] = "failed to read path"


def _read_direct_file(path: str, storage_options: dict[str, object]) -> bytes | None:
    try:
        with fsspec.open(path, mode="rb", **storage_options) as fobj:
            return fobj.read()
    except (OSError, RuntimeError, ValueError):
        return None


def _fill_materialized_bytes(
    df: pd.DataFrame,
    image_mask: pd.Series,
    *,
    storage_options: dict[str, object],
    binary_values: list[object],
    error_values: list[str | None],
) -> None:
    classified = _classify_rows(df, image_mask)

    for idx in classified.missing:
        error_values[idx] = "missing path"

    _fill_tar_extract_rows(classified.tar_extract, storage_options, binary_values, error_values)
    _fill_range_read_rows(classified.range_read, storage_options, binary_values, error_values)
    _fill_direct_read_rows(classified.direct_read, storage_options, binary_values, error_values)


def _init_materialization_buffers(df: pd.DataFrame) -> tuple[list[object], list[str | None]]:
    error_values = (
        df["materialize_error"].astype("object").tolist() if "materialize_error" in df.columns else [None] * len(df)
    )
    binary_values = (
        df["binary_content"].astype("object").tolist() if "binary_content" in df.columns else [None] * len(df)
    )
    return binary_values, error_values


def _build_image_mask(
    df: pd.DataFrame,
    *,
    only_missing_binary: bool,
    image_content_types: tuple[str, ...] | None,
) -> pd.Series:
    image_mask = (
        (df["modality"] == "image") if "modality" in df.columns else pd.Series(False, index=df.index, dtype=bool)
    )
    if image_content_types is not None and "content_type" in df.columns:
        image_mask &= df["content_type"].isin(image_content_types)
    if only_missing_binary and "binary_content" in df.columns:
        image_mask &= df["binary_content"].isna()
    return image_mask


def iter_source_grouped_bytes(
    task: InterleavedBatch,
    row_indices: list[int],
    storage_options: dict[str, object] | None = None,
) -> Iterator[tuple[int, bytes | None]]:
    """Yield ``(row_index, image_bytes)`` grouped by source path.

    Opens one source (tar / file) at a time, yields all requested rows from
    that source, then releases the bytes before moving to the next source.
    This bounds peak memory to O(images_per_source) instead of O(all_images).
    """
    if not row_indices:
        return

    df = task.to_pandas()
    if storage_options is None:
        storage_options = resolve_storage_options(task=task)

    source_refs = [InterleavedBatch.parse_source_ref(df.loc[idx, "source_ref"]) for idx in row_indices]

    groups: dict[str, list[tuple[int, dict]]] = {}
    missing: list[int] = []
    for idx, ref in zip(row_indices, source_refs, strict=True):
        path = ref.get("path")
        if path is None or path == "":
            missing.append(idx)
        else:
            groups.setdefault(str(path), []).append((idx, ref))

    for idx in missing:
        yield idx, None

    for path, entries in groups.items():
        results: dict[int, bytes | None] = {}
        _fill_source_group(path, entries, storage_options, results)
        for idx, _ in entries:
            yield idx, results.get(idx)
        del results


def _fill_source_group(
    path: str,
    entries: list[tuple[int, dict]],
    storage_options: dict[str, object],
    results: dict[int, bytes | None],
) -> None:
    """Materialize bytes for all entries sharing a single source path."""
    range_entries = []
    tar_entries = []
    direct_entries = []

    for idx, ref in entries:
        member = ref.get("member")
        has_member = member is not None and member != ""
        if not has_member:
            direct_entries.append(idx)
            continue
        offset = ref.get("byte_offset")
        size = ref.get("byte_size")
        if offset is not None and size is not None and size > 0:
            range_entries.append((idx, str(member), int(offset), int(size)))
        else:
            tar_entries.append((idx, str(member)))

    if range_entries:
        _fill_source_group_range(path, range_entries, storage_options, results)
    if tar_entries:
        _fill_source_group_tar(path, tar_entries, storage_options, results)
    if direct_entries:
        payload = _read_direct_file(path, storage_options)
        for idx in direct_entries:
            results[idx] = payload


def _fill_source_group_range(
    path: str,
    entries: list[tuple[int, str, int, int]],
    storage_options: dict[str, object],
    results: dict[int, bytes | None],
) -> None:
    try:
        fs, fs_path = url_to_fs(path, **storage_options)
    except (ValueError, OSError):
        return
    paths = [fs_path] * len(entries)
    starts = [offset for _, _, offset, _ in entries]
    ends = [offset + size for _, _, offset, size in entries]
    try:
        blobs = fs.cat_ranges(paths, starts, ends)
    except OSError:
        return
    for (idx, _member, _offset, _size), blob in zip(entries, blobs, strict=True):
        if isinstance(blob, Exception) or blob is None or len(blob) == 0:
            results[idx] = None
        else:
            results[idx] = bytes(blob) if not isinstance(blob, bytes) else blob


def _fill_source_group_tar(
    path: str,
    entries: list[tuple[int, str]],
    storage_options: dict[str, object],
    results: dict[int, bytes | None],
) -> None:
    try:
        with fsspec.open(path, mode="rb", **storage_options) as fobj, tarfile.open(fileobj=fobj, mode="r:*") as tf:
            cache: dict[str, bytes | None] = {}
            for idx, member in entries:
                if member not in cache:
                    try:
                        extracted = tf.extractfile(member)
                    except KeyError:
                        extracted = None
                    cache[member] = extracted.read() if extracted is not None else None
                results[idx] = cache[member]
    except (OSError, tarfile.TarError):
        pass


def _task_with_dataframe(task: InterleavedBatch, df: pd.DataFrame) -> InterleavedBatch:
    return InterleavedBatch(
        task_id=task.task_id,
        dataset_name=task.dataset_name,
        data=df,
        _metadata=task._metadata,
        _stage_perf=task._stage_perf,
    )


def materialize_task_binary_content(
    task: InterleavedBatch,
    *,
    io_kwargs: dict[str, object] | None = None,
    only_missing_binary: bool = True,
    image_content_types: tuple[str, ...] | None = None,
) -> InterleavedBatch:
    """Return a task with image-row binary content materialized from source_ref.

    Dispatches to three I/O strategies based on source_ref contents:
    - range_read: byte_offset + byte_size present -> batched fs.cat_ranges()
    - tar_extract: member present, no byte range -> open tar + extractfile
    - direct_read: no member -> read file directly
    """
    df = task.to_pandas().reset_index(drop=True)
    if df.empty:
        return task
    parsed = [InterleavedBatch.parse_source_ref(v) for v in df["source_ref"].tolist()]
    parsed_df = pd.DataFrame.from_records(
        parsed, columns=["path", "member", "byte_offset", "byte_size", "frame_index"],
    )
    for col in parsed_df.columns:
        df[f"_src_{col}"] = parsed_df[col].to_numpy(copy=False)
    del parsed, parsed_df

    binary_values, error_values = _init_materialization_buffers(df)
    image_mask = _build_image_mask(
        df,
        only_missing_binary=only_missing_binary,
        image_content_types=image_content_types,
    )
    if not image_mask.any():
        for col in [c for c in df.columns if c.startswith("_src_")]:
            del df[col]
        return _task_with_dataframe(task, df)

    storage_options = resolve_storage_options(task=task, io_kwargs=io_kwargs)
    _fill_materialized_bytes(
        df,
        image_mask,
        storage_options=storage_options,
        binary_values=binary_values,
        error_values=error_values,
    )

    for col in [c for c in df.columns if c.startswith("_src_")]:
        del df[col]
    df["binary_content"] = pd.Series(binary_values, dtype="object")
    df["materialize_error"] = pd.Series(error_values, dtype="object")
    return _task_with_dataframe(task, df)
