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

"""Curator file group construction and order-locking helpers."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

FILEGROUP_SIGNATURE_FILENAME = "filegroup_signature.json"
FileGroupSignature = tuple[tuple[str, ...], ...]


def create_input_filegroups(input_paths: list[str]) -> list[Any]:
    """Create one Curator file group per source parquet shard."""
    from nemo_curator.tasks import FileGroupTask
    from nemo_curator.utils.file_utils import infer_dataset_name_from_path

    if not input_paths:
        msg = "No parquet file groups were created."
        raise RuntimeError(msg)

    logger.info(f"Phase 1: creating Curator file groups from {len(input_paths):,} input path(s)")
    dataset_name = infer_dataset_name_from_path(input_paths[0])
    tasks = [
        FileGroupTask(
            dataset_name=dataset_name,
            data=[path],
            _metadata={
                "partition_index": partition_index,
                "total_partitions": len(input_paths),
                "source_files": [path],
            },
            reader_config={},
        )
        for partition_index, path in enumerate(input_paths)
    ]
    logger.info(f"Phase 1 complete: {len(tasks):,} file group task(s)")
    return tasks


def filegroup_signature(input_tasks: list[Any]) -> FileGroupSignature:
    """Return the ordered file paths for each Curator file group task."""
    return tuple(tuple(task.data) for task in input_tasks)


def filegroup_signature_digest(signature: FileGroupSignature) -> str:
    """Return a stable digest for comparing file group order across runs."""
    hasher = hashlib.sha256()
    for filegroup in signature:
        hasher.update(b"\x1e")
        for path in filegroup:
            hasher.update(path.encode("utf-8"))
            hasher.update(b"\x00")
    return hasher.hexdigest()


def filegroup_signature_path(dedup_ids_dir: Path) -> Path:
    """Return the persisted filegroup signature path."""
    return dedup_ids_dir / FILEGROUP_SIGNATURE_FILENAME


def write_filegroup_signature(input_tasks: list[Any], dedup_ids_dir: Path) -> FileGroupSignature:
    """Persist the ordered source file groups used for exact deduplication."""
    signature = filegroup_signature(input_tasks)
    digest = filegroup_signature_digest(signature)
    path = filegroup_signature_path(dedup_ids_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sha256": digest,
        "filegroups": [list(filegroup) for filegroup in signature],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    logger.info(f"Locked file group order for {len(signature):,} task(s); sha256={digest}")
    logger.info(f"Wrote file group signature to {path}")
    return signature


def load_filegroup_signature(dedup_ids_dir: Path) -> FileGroupSignature:
    """Load the filegroup signature written by the exact-dedup phase."""
    path = filegroup_signature_path(dedup_ids_dir)
    if not path.exists():
        msg = f"Missing file group signature: {path}. Run identify_cc_index_url_duplicates.py first."
        raise FileNotFoundError(msg)

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        msg = f"Invalid file group signature payload: {path}"
        raise TypeError(msg)

    filegroups = payload.get("filegroups")
    if not isinstance(filegroups, list):
        msg = f"Invalid file group signature payload: {path}"
        raise TypeError(msg)
    if not all(isinstance(filegroup, list) and all(isinstance(item, str) for item in filegroup) for filegroup in filegroups):
        msg = f"Invalid file group entries in signature payload: {path}"
        raise TypeError(msg)

    signature = tuple(tuple(filegroup) for filegroup in filegroups)
    digest = filegroup_signature_digest(signature)
    if payload.get("sha256") != digest:
        msg = f"File group signature digest mismatch in {path}"
        raise ValueError(msg)
    logger.info(f"Loaded file group signature for {len(signature):,} task(s); sha256={digest}")
    return signature


def validate_filegroup_signature(input_tasks: list[Any], expected: FileGroupSignature, phase: str) -> None:
    """Fail if the file group membership or ordering changed between workflow phases."""
    actual = filegroup_signature(input_tasks)
    if actual != expected:
        msg = (
            f"Curator file group order changed before {phase}. "
            "Exact deduplication and duplicate removal must receive the same ordered file groups "
            "so the ID generator can map assigned IDs back to the source parquet files."
        )
        raise RuntimeError(msg)
    logger.info(f"Validated file group order for {phase}; sha256={filegroup_signature_digest(actual)}")
