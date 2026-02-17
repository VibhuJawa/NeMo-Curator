# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import time
from tarfile import ReadError
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
from aiohttp import ClientError
from fsspec.exceptions import FSTimeoutError

from nemo_curator.utils.file_utils import open_binary_reader, open_tar_path

if TYPE_CHECKING:
    from collections.abc import Callable


_RETRIABLE_MATERIALIZE_EXCEPTIONS: tuple[type[Exception], ...] = (
    OSError,
    ReadError,
    TimeoutError,
    ClientError,
    FSTimeoutError,
)


def validate_materialize_options(
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


def validate_content_path_loading_mode(*, content_path: str, row_indices: list[int], content_keys: list[object | None]) -> None:
    """Ensure one content_path group uses a single loading mode."""
    has_key_flags = [content_keys[idx] is not None for idx in row_indices]
    has_member_backed_rows = any(has_key_flags)
    has_direct_backed_rows = not all(has_key_flags)
    if has_member_backed_rows and has_direct_backed_rows:
        msg = (
            f"Invalid mixed loading modes for content_path='{content_path}'. "
            "Rows for the same content_path must be all content_key-backed or all direct-backed."
        )
        raise ValueError(msg)


def load_payloads_from_tar_members(
    *,
    content_path: str,
    keyed_rows: dict[int, str],
    storage_options: dict[str, Any],
) -> dict[int, bytes]:
    """Load payloads by ``content_key`` from a tar container path."""
    # ``content_path`` points to a container here, so rows map to tar member keys.
    required_keys = set(keyed_rows.values())
    extracted_by_key: dict[str, bytes] = {}
    with open_tar_path(content_path, storage_options) as tf:
        for member in tf:
            if member.name in required_keys:
                payload = tf.extractfile(member)
                if payload is not None:
                    extracted_by_key[member.name] = payload.read()
    missing_keys = sorted(required_keys - extracted_by_key.keys())
    if missing_keys:
        msg = f"Missing tar member '{missing_keys[0]}' in '{content_path}'"
        raise FileNotFoundError(msg)
    return {idx: extracted_by_key[key] for idx, key in keyed_rows.items()}


def load_payloads_from_direct_path(*, content_path: str, row_indices: list[int], storage_options: dict[str, Any]) -> dict[int, bytes]:
    """Load one direct-file payload and assign it to all row indices."""
    # Direct-path rows share one payload blob for all indices at this path.
    with open_binary_reader(content_path, storage_options) as f:
        payload = f.read()
    return dict.fromkeys(row_indices, payload)


def replace_binary_content(table: pa.Table, binary_values: list[Any]) -> pa.Table:
    binary_idx = table.schema.get_field_index("binary_content")
    binary_column = pa.array(binary_values, type=pa.large_binary())
    return table.set_column(binary_idx, "binary_content", binary_column)


def retry_materialize(
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


def sort_multimodal_table(table: pa.Table) -> pa.Table:
    """Sort rows by ``sample_id``, ``position``, and ``modality``."""
    if table.num_rows == 0:
        return table
    return table.sort_by([("sample_id", "ascending"), ("position", "ascending"), ("modality", "ascending")])
