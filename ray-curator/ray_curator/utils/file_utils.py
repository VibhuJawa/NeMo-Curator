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

import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

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


def _normalize_exts(keep_extensions: str | list[str] | None) -> set[str] | None:
    if keep_extensions is None:
        return None
    if isinstance(keep_extensions, str):
        keep_extensions = [keep_extensions]
    return {ext.lstrip(".").lower() for ext in keep_extensions}


def _matches_ext(path: str, extensions: set[str] | None) -> bool:
    if extensions is None:
        return True
    # works for remote paths too; os.path.splitext handles POSIX-style separators
    return os.path.splitext(path)[-1].lstrip(".").lower() in extensions


def _split_files_as_per_blocksize(
    sorted_file_sizes: list[tuple[str, int]], max_byte_per_chunk: int
) -> list[list[str]]:
    partitions = []
    current_partition = []
    current_size = 0

    for file, size in sorted_file_sizes:
        if current_size + size > max_byte_per_chunk:
            partitions.append(current_partition)
            current_partition = []
            current_size = 0
        current_partition.append(file)
        current_size += size
    if current_partition:
        partitions.append(current_partition)

    logger.info(
        f"Split {len(sorted_file_sizes)} files into {len(partitions)} partitions with max size {(max_byte_per_chunk / 1024 / 1024):.2f} MB."
    )
    return partitions


def get_all_files_paths_under(  # noqa: C901, PLR0913, PLR0912
    path: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict[str, str] | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
    return_sizes: bool = False,
) -> list[str] | list[tuple[str, int]]:
    """
    List all files under a given path or glob.  Optionally recurse into
    subdirectories, filter by extension, and return file sizes.

    Parameters
    ----------
    path : str
        A single path or a glob pattern (e.g. ``'s3://bucket/data/*.csv'``).
    recurse_subdirectories : bool
        If True, recursively list files under each matched directory.
    keep_extensions : str or sequence of str, optional
        Restrict results to files with these extensions (case-insensitive,
        leading dots optional).
    storage_options : dict, optional
        Extra options for the filesystem (e.g. credentials).
    fs : fsspec.AbstractFileSystem, optional
        Pre-instantiated filesystem.  If not provided, it will be inferred
        from the path.
    return_sizes : bool
        If True, return a list of ``(path, size)`` tuples; otherwise just paths.

    Returns
    -------
    list
        Sorted list of paths, optionally paired with their sizes.
    """
    # Get a filesystem instance
    if fs is None:
        fs, _ = fsspec.core.url_to_fs(path, **(storage_options or {}))

    exts = _normalize_exts(keep_extensions)

    # Expand globs/directories to a list of starting points
    # expand_path() handles glob patterns and returns both files and directories
    roots = fs.expand_path(path, recursive=False)

    out_paths: list[str] = []
    out_sizes: list[int] = []

    for root in roots:
        # Decide whether to descend into the root or treat it as a file
        is_dir = fs.isdir(root) if fs.exists(root) else False
        if is_dir:
            # Use find() for efficient recursive listing; detail=True fetches size info:contentReference[oaicite:2]{index=2}
            # maxdepth=1 restricts to top level when recursion is off
            detail = return_sizes
            files = fs.find(
                root,
                maxdepth=None if recurse_subdirectories else 1,
                withdirs=False,
                detail=detail,
            )
            if return_sizes:
                for p, info in files.items():
                    if _matches_ext(p, exts):
                        out_paths.append(fs.unstrip_protocol(p))
                        out_sizes.append(info.get("size", -1))
            else:
                for p in files:
                    if _matches_ext(p, exts):
                        out_paths.append(fs.unstrip_protocol(p))
        elif _matches_ext(root, exts) and fs.exists(root):
            # root is a single file (or does not exist)
            out_paths.append(fs.unstrip_protocol(root))
            if return_sizes:
                # Use fs.info() if detail was not available
                info = fs.info(root)
                out_sizes.append(info.get("size", -1))

    # Sort results for reproducibility
    sorted_indices = sorted(range(len(out_paths)), key=lambda i: out_paths[i])
    if return_sizes:
        return [(out_paths[i], out_sizes[i]) for i in sorted_indices]
    else:
        return [out_paths[i] for i in sorted_indices]


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


def check_output_mode(
    mode: Literal["overwrite", "append", "error", "ignore"],
    fs: fsspec.AbstractFileSystem,
    path: str,
    append_mode_implemented: bool = False,
) -> None:
    """
    Validate and act on the write mode for an output directory.

    Modes:
    - "overwrite": delete existing `output_dir` recursively if it exists.
    - "append": no-op here; raises if append is not implemented.
    - "error": raise FileExistsError if `output_dir` already exists.
    - "ignore": no-op.
    """
    normalized = mode.strip().lower()
    allowed = {"overwrite", "append", "error", "ignore"}
    if normalized not in allowed:
        msg = f"Invalid mode: {mode!r}. Allowed: {sorted(allowed)}"
        raise ValueError(msg)

    if normalized == "ignore":
        if not fs.exists(path):
            fs.makedirs(path)
        return

    if normalized == "append":
        if not append_mode_implemented:
            msg = "append mode is not implemented yet"
            raise NotImplementedError(msg)
        return

    if normalized == "overwrite":
        if fs.exists(path):
            msg = f"Removing output directory {path} for overwrite mode"
            logger.info(msg)
            delete_dir(path=path, fs=fs)
        else:
            msg = f"Overwrite mode: output directory {path} does not exist; nothing to remove"
            logger.info(msg)
        return

    if normalized == "error" and fs.exists(path):
        msg = f"Output directory {path} already exists"
        raise FileExistsError(msg)
    return
