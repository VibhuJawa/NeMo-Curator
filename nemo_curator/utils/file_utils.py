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
import posixpath
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

import fsspec
from fsspec.core import get_filesystem_class, split_protocol
from fsspec.utils import infer_storage_options
from loguru import logger

from nemo_curator.utils.client_utils import is_remote_url

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

FILETYPE_TO_DEFAULT_EXTENSIONS = {
    "parquet": [".parquet"],
    "jsonl": [".jsonl", ".json"],
    "megatron": [".bin", ".idx"],
}


def get_fs(path: str, storage_options: dict[str, str] | None = None) -> fsspec.AbstractFileSystem:
    if not storage_options:
        storage_options = {}
    protocol, path = split_protocol(path)
    return get_filesystem_class(protocol)(**storage_options)


def is_not_empty(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> bool:
    if fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    return fs.exists(path) and fs.isdir(path) and fs.listdir(path)


def delete_dir(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> None:
    if fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    if fs.exists(path) and fs.isdir(path):
        fs.rm(path, recursive=True)


def create_or_overwrite_dir(
    path: str, fs: fsspec.AbstractFileSystem | None = None, storage_options: dict[str, str] | None = None
) -> None:
    """
    Creates a directory if it does not exist and overwrites it if it does.
    Warning: This function will delete all files in the directory if it exists.
    """
    if fs is not None and storage_options is not None:
        err_msg = "fs and storage_options cannot be provided together"
        raise ValueError(err_msg)
    elif fs is None:
        fs = get_fs(path, storage_options)

    if is_not_empty(path, fs):
        logger.warning(f"Output directory {path} is not empty. Deleting it.")
        delete_dir(path, fs)

    fs.mkdirs(path, exist_ok=True)


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
        logger.warning("Skipped at least one file due to unmatched file extension(s).")

    return filtered_files


def _split_files_as_per_blocksize(
    sorted_file_sizes: list[tuple[str, int]], max_byte_per_chunk: int
) -> list[list[str]]:
    partitions = []
    current_partition = []
    current_size = 0

    for file, size in sorted_file_sizes:
        if current_size + size > max_byte_per_chunk:
            if current_partition:
                partitions.append(current_partition)
            current_partition = []
            current_size = 0
        current_partition.append(file)
        current_size += size
    if current_partition:
        partitions.append(current_partition)

    logger.debug(
        f"Split {len(sorted_file_sizes)} files into {len(partitions)} partitions with max size {(max_byte_per_chunk / 1024 / 1024):.2f} MB."
    )
    return partitions


def _gather_extention(path: str) -> str:
    """
    Gather the extension of a given path.
    Args:
        path: The path to get the extension from.
    Returns:
        The extension of the path.
    """
    name = posixpath.basename(path.rstrip("/"))
    return posixpath.splitext(name)[1][1:].casefold()


def _gather_file_records(  # noqa: PLR0913
    path: str,
    recurse_subdirectories: bool,
    keep_extensions: str | list[str] | None,
    storage_options: dict[str, str] | None,
    fs: fsspec.AbstractFileSystem | None,
    include_size: bool,
) -> list[tuple[str, int]]:
    """
    Gather file records from a given path.
    Args:
        path: The path to get the file paths from.
        recurse_subdirectories: Whether to recurse subdirectories.
        keep_extensions: The extensions to keep.
        storage_options: The storage options to use.
        fs: The filesystem to use.
        include_size: Whether to include the size of the files.
    Returns:
        A list of tuples (file_path, file_size).
    """
    fs = fs or fsspec.core.url_to_fs(path, **(storage_options or {}))[0]
    allowed_exts = (
        None
        if keep_extensions is None
        else {
            e.casefold().lstrip(".")
            for e in ([keep_extensions] if isinstance(keep_extensions, str) else keep_extensions)
        }
    )
    normalize = fs.unstrip_protocol if is_remote_url(path) else (lambda x: x)
    roots = fs.expand_path(path, recursive=False)
    records = []

    for root in roots:
        if fs.isdir(root):
            listing = fs.find(
                root,
                maxdepth=None if recurse_subdirectories else 1,
                withdirs=False,
                detail=include_size,
            )
            if include_size:
                entries = [(p, info.get("size")) for p, info in listing.items()]
            else:
                entries = [(p, None) for p in listing]

        elif fs.exists(root):
            entries = [(root, fs.info(root).get("size") if include_size else None)]
        else:
            entries = []

        for raw_path, raw_size in entries:
            if (allowed_exts is None) or (_gather_extention(raw_path) in allowed_exts):
                records.append((normalize(raw_path), -1 if include_size and raw_size is None else raw_size))

    return records


def get_all_file_paths_under(
    path: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict[str, str] | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
) -> list[str]:
    """
    Get all file paths under a given path.
    Args:
        path: The path to get the file paths from.
        recurse_subdirectories: Whether to recurse subdirectories.
        keep_extensions: The extensions to keep.
        storage_options: The storage options to use.
        fs: The filesystem to use.
    Returns:
        A list of file paths.
    """
    return sorted(
        [
            p
            for p, _ in _gather_file_records(
                path, recurse_subdirectories, keep_extensions, storage_options, fs, include_size=False
            )
        ]
    )


def get_all_file_paths_and_size_under(
    path: str,
    recurse_subdirectories: bool = False,
    keep_extensions: str | list[str] | None = None,
    storage_options: dict[str, str] | None = None,
    fs: fsspec.AbstractFileSystem | None = None,
) -> list[tuple[str, int]]:
    """
    Get all file paths and their sizes under a given path.
    Args:
        path: The path to get the file paths from.
        recurse_subdirectories: Whether to recurse subdirectories.
        keep_extensions: The extensions to keep.
        storage_options: The storage options to use.
        fs: The filesystem to use.
    Returns:
        A list of tuples (file_path, file_size).
    """
    # sort by size
    return sorted(
        [
            (p, int(s))
            for p, s in _gather_file_records(
                path, recurse_subdirectories, keep_extensions, storage_options, fs, include_size=True
            )
        ],
        key=lambda x: x[1],
    )


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

    if normalized == "append" and append_mode_implemented is False:
        msg = "append mode is not implemented yet"
        raise NotImplementedError(msg)

    if normalized == "error" and fs.exists(path):
        msg = f"Output directory {path} already exists"
        raise FileExistsError(msg)

    if normalized == "overwrite":
        if fs.exists(path):
            msg = f"Removing output directory {path} for overwrite mode"
            logger.info(msg)
            delete_dir(path=path, fs=fs)
        else:
            msg = f"Overwrite mode: output directory {path} does not exist; nothing to remove"
            logger.info(msg)

    # For ignore/append/overwrite mode (we delete the directory earlier in overwrite mode)
    # So we need to create the directory
    fs.makedirs(path, exist_ok=True)


def infer_dataset_name_from_path(path: str) -> str:
    """Infer a dataset name from a path, handling both local and cloud storage paths.
    Args:
        path: Local path or cloud storage URL (e.g. s3://, abfs://)
    Returns:
        Inferred dataset name from the path
    """
    # Split protocol and path for cloud storage
    protocol, pure_path = split_protocol(path)
    if protocol is None:
        # Local path handling
        first_file = Path(path)
        if first_file.parent.name and first_file.parent.name != ".":
            return first_file.parent.name.lower()
        return first_file.stem.lower()
    else:
        path_parts = pure_path.rstrip("/").split("/")
        if len(path_parts) <= 1:
            return path_parts[0]
        return path_parts[-1].lower()


def resolve_task_scoped_output_path(base_output_path: str, task_id: str, default_suffix: str) -> str:
    """Resolve a deterministic per-task output path from a base output location.

    Args:
        base_output_path: Output path or prefix configured by the writer.
        task_id: Task identifier used to disambiguate per-task outputs.
        default_suffix: File suffix (without leading dot) to use when base path
            is a directory/prefix instead of a concrete filename.
    """
    token = task_id.replace("/", "_")
    if not is_remote_url(base_output_path):
        base_path = Path(base_output_path)
        if base_output_path.endswith("/") or base_path.suffix == "":
            return str(base_path / f"{token}.{default_suffix}")
        return str(base_path.with_name(f"{base_path.stem}.{token}{base_path.suffix}"))

    protocol = fsspec.utils.get_protocol(base_output_path)
    inner_path = base_output_path.split("://", 1)[1]
    if inner_path.endswith("/") or "." not in posixpath.basename(inner_path):
        return f"{protocol}://{inner_path.rstrip('/')}/{token}.{default_suffix}"

    dirname = posixpath.dirname(inner_path)
    basename = posixpath.basename(inner_path)
    stem, suffix = basename.rsplit(".", 1)
    return f"{protocol}://{dirname}/{stem}.{token}.{suffix}"


def resolve_sidecar_output_path(primary_output_path: str, sidecar_tag: str, sidecar_suffix: str) -> str:
    """Resolve a deterministic sidecar output path from a primary output path.

    Examples:
        ``/tmp/out.task_0.parquet`` + (``metadata``, ``parquet``)
        -> ``/tmp/out.task_0.metadata.parquet``
    """
    if not is_remote_url(primary_output_path):
        out = Path(primary_output_path)
        if out.suffix:
            return str(out.with_name(f"{out.stem}.{sidecar_tag}.{sidecar_suffix}"))
        return f"{primary_output_path}.{sidecar_tag}.{sidecar_suffix}"

    protocol = fsspec.utils.get_protocol(primary_output_path)
    inner_path = primary_output_path.split("://", 1)[1]
    dirname = posixpath.dirname(inner_path)
    basename = posixpath.basename(inner_path)
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    return f"{protocol}://{dirname}/{stem}.{sidecar_tag}.{sidecar_suffix}"


def resolve_fs_and_path(
    output_path: str, storage_options: dict[str, Any] | None = None
) -> tuple[fsspec.AbstractFileSystem, str]:
    """Resolve ``output_path`` into an fsspec filesystem and filesystem-local path."""
    options = (storage_options or {}) if is_remote_url(output_path) else {}
    return fsspec.core.url_to_fs(output_path, **options)


def ensure_parent_directory(fs: fsspec.AbstractFileSystem, fs_path: str) -> None:
    """Create parent directory for ``fs_path`` if needed."""
    parent = posixpath.dirname(fs_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)


@contextmanager
def open_binary_writer(
    output_path: str, storage_options: dict[str, Any] | None = None
) -> BinaryIO:
    """Open a binary sink for ``output_path`` with parent creation."""
    fs, fs_path = resolve_fs_and_path(output_path, storage_options)
    ensure_parent_directory(fs, fs_path)
    with fs.open(fs_path, "wb") as sink:
        yield sink


@contextmanager
def open_binary_reader(
    input_path: str, storage_options: dict[str, Any] | None = None
) -> BinaryIO:
    """Open a binary source for ``input_path``."""
    fs, fs_path = resolve_fs_and_path(input_path, storage_options)
    with fs.open(fs_path, "rb") as source:
        yield source


@contextmanager
def open_tar_path(path: str, storage_options: dict[str, Any] | None = None) -> tarfile.TarFile:
    """Open a local or remote tar path and yield a tarfile handle."""
    with open_binary_reader(path, storage_options) as raw, tarfile.open(fileobj=raw, mode="r:*") as tf:
        yield tf


def check_disallowed_kwargs(
    kwargs: dict,
    disallowed_keys: list[str],
    raise_error: bool = True,
) -> None:
    """Check if any of the disallowed keys are in provided kwargs
    Used for read/write kwargs in stages.
    Args:
        kwargs: The dictionary to check
        disallowed_keys: The keys that are not allowed.
        raise_error: Whether to raise an error if any of the disallowed keys are in the kwargs.
    Raises:
        ValueError: If any of the disallowed keys are in the kwargs and raise_error is True.
        Warning: If any of the disallowed keys are in the kwargs and raise_error is False.
    Returns:
        None
    """
    found_keys = set(kwargs).intersection(disallowed_keys)
    if raise_error and found_keys:
        msg = f"Unsupported keys in kwargs: {', '.join(found_keys)}"
        raise ValueError(msg)
    elif found_keys:
        msg = f"Unsupported keys in kwargs: {', '.join(found_keys)}"
        logger.warning(msg)


def _is_safe_path(path: str, base_path: str) -> bool:
    """
    Check if a path is safe for extraction (no path traversal).

    Args:
        path: The path to check
        base_path: The base directory for extraction

    Returns:
        True if the path is safe, False otherwise
    """
    # Normalize paths to handle different path separators and resolve '..' components
    full_path = os.path.normpath(os.path.join(base_path, path))
    base_path = os.path.normpath(base_path)

    # Check if the resolved path is within the base directory
    return os.path.commonpath([full_path, base_path]) == base_path


def tar_safe_extract(tar: tarfile.TarFile, path: str) -> None:
    """
    Safely extract a tar file, preventing path traversal attacks.

    Args:
        tar: The TarFile object to extract
        path: The destination path for extraction

    Raises:
        ValueError: If any member has an unsafe path
    """
    for member in tar.getmembers():
        # Check for absolute paths
        if os.path.isabs(member.name):
            msg = f"Absolute path not allowed: {member.name}"
            raise ValueError(msg)

        # Check for path traversal attempts
        if not _is_safe_path(member.name, path):
            msg = f"Path traversal attempt detected: {member.name}"
            raise ValueError(msg)

        # Check for dangerous file types
        if member.isdev():
            msg = f"Device files not allowed: {member.name}"
            raise ValueError(msg)

        # For symlinks, check that the target is also safe
        if member.issym() or member.islnk():
            if os.path.isabs(member.linkname):
                msg = f"Absolute symlink target not allowed: {member.name} -> {member.linkname}"
                raise ValueError(msg)
            if not _is_safe_path(member.linkname, path):
                msg = f"Symlink target outside extraction directory: {member.name} -> {member.linkname}"
                raise ValueError(msg)

        # Extract the member
        tar.extract(member, path)
