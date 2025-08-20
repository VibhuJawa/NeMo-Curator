# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import warnings
from typing import TYPE_CHECKING

import fsspec
import pyarrow as pa
from fsspec.core import get_filesystem_class, split_protocol
from fsspec.utils import infer_storage_options
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

FILETYPE_TO_DEFAULT_EXTENSIONS = {
    "parquet": [".parquet"],
    "jsonl": [".jsonl", ".json"],
}


def get_fs(path: str, storage_options: dict[str, str] | None = None) -> fsspec.AbstractFileSystem:
    if not storage_options:
        storage_options = {}
    protocol, path = split_protocol(path)
    return get_filesystem_class(protocol)(**storage_options)


def is_not_empty(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> bool:
    if fs is None and storage_options is None:
        err_msg = "fs or storage_options must be provided"
        raise ValueError(err_msg)
    elif fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    return fs.exists(path) and fs.isdir(path) and fs.listdir(path)


def delete_dir(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> None:
    if fs is None and storage_options is None:
        err_msg = "fs or storage_options must be provided"
        raise ValueError(err_msg)
    elif fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    if fs.exists(path) and fs.isdir(path):
        fs.rm(path, recursive=True)


def filter_files_by_extension(
    files_list: list[str],
    keep_extensions: str | list[str],
) -> list[str]:
    filtered_files = []
    if isinstance(keep_extensions, str):
        keep_extensions = [keep_extensions]

    # Ensure that the extensions are prefixed with a dot
    file_extensions = tuple([s if s.startswith(".") else f".{s}" for s in keep_extensions])

    for file in files_list:
        if file.endswith(file_extensions):
            filtered_files.append(file)

    if len(files_list) != len(filtered_files):
        warnings.warn("Skipped at least one file due to unmatched file extension(s).", stacklevel=2)

    return filtered_files


def get_all_files_paths_under(
    input_dir: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
) -> list[str]:
    # TODO: update with a more robust fsspec method
    if fs is None:
        fs = get_fs(input_dir, storage_options)

    file_ls = fs.find(input_dir, maxdepth=None if recurse_subdirectories else 1)
    protocol, _ = split_protocol(input_dir)
    if protocol and protocol not in {"file", "local"}:
        file_ls = [fs.unstrip_protocol(f) for f in file_ls]

    file_ls.sort()
    if keep_extensions is not None:
        file_ls = filter_files_by_extension(file_ls, keep_extensions)
    return file_ls


def infer_protocol_from_paths(paths: Iterable[str]) -> str | None:
    """Infer a protocol from a list of paths, if any.

    Returns the first detected protocol scheme (e.g., "s3", "gcs", "gs", "abfs")
    or None for local paths.
    """
    for path in paths:
        opts = infer_storage_options(path)
        protocol = opts.get("protocol")
        if protocol and protocol not in {"file", "local"}:
            return protocol
    return None


def get_pyarrow_filesystem(paths: Iterable[str], storage_options: dict | None) -> pa.fs.FileSystem | None:
    """Return a PyArrow FileSystem backed by fsspec if a remote protocol is detected.

    If no remote protocol is found or fsspec is not available, returns None so callers can
    let Arrow or Pandas handle local files directly.
    """

    protocol = infer_protocol_from_paths(paths)
    if protocol is None:
        return None

    fs = fsspec.filesystem(protocol, **(storage_options or {}))
    return pa.PyFileSystem(pa.FSSpecHandler(fs))


def pandas_select_columns(df: pd.DataFrame, columns: list[str] | None, file_path: str) -> pd.DataFrame | None:
    """Project a Pandas DataFrame onto existing columns, logging warnings for missing ones.

    Returns the projected DataFrame. If no requested columns exist, returns None.
    """

    if columns is None:
        return df

    existing_columns = [col for col in columns if col in df.columns]
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        logger.warning(f"Columns {missing_columns} not found in {file_path}")

    if existing_columns:
        return df[existing_columns]

    logger.error(f"None of the requested columns found in {file_path}")
    return None


def pyarrow_select_columns(table: pa.Table, columns: list[str] | None, file_path: str) -> pa.Table | None:
    """Project a PyArrow Table onto existing columns, logging warnings for missing ones.

    Returns the projected Table. If no requested columns exist, returns None.
    """

    if columns is None:
        return table

    existing_columns = [col for col in columns if col in table.column_names]
    missing_columns = [col for col in columns if col not in table.column_names]

    if missing_columns:
        logger.warning(f"Columns {missing_columns} not found in {file_path}")

    if existing_columns:
        return table.select(existing_columns)

    logger.error(f"None of the requested columns found in {file_path}")
    return None
