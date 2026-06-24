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

import json
import posixpath
from typing import Any

from fsspec.core import url_to_fs

from nemo_curator.utils.hash_utils import get_deterministic_hash

LANCE_ROWADDR_COLUMN = "__lance_rowaddr"
LANCE_FRAGID_COLUMN = "__lance_fragid"
_COMMITTED_MARKER = "_COMMITTED"
_RECORDS_DIR = "records"


def lance_checkpoint_record_id(kind: str, *parts: object) -> str:
    values = [str(part) for part in parts if part not in {None, ""}]
    return f"{kind}-{get_deterministic_hash(values or [kind])}"


def _checkpoint_fs_path(commit_path: str, storage_options: dict[str, Any] | None = None) -> tuple[object, str]:
    return url_to_fs(commit_path, **(storage_options or {}))


def _checkpoint_path(fs_path: str, *parts: str) -> str:
    return posixpath.join(fs_path.rstrip("/"), *parts)


def write_lance_checkpoint_record(
    commit_path: str,
    record: dict[str, Any],
    record_id: str,
    storage_options: dict[str, Any] | None = None,
) -> str:
    fs, fs_path = _checkpoint_fs_path(commit_path, storage_options)
    records_dir = _checkpoint_path(fs_path, _RECORDS_DIR)
    fs.makedirs(records_dir, exist_ok=True)
    record_path = _checkpoint_path(records_dir, f"{record_id}.json")
    with fs.open(record_path, "w") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return fs.unstrip_protocol(record_path)


def read_lance_checkpoint(
    commit_path: str,
    kind: str,
    storage_options: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    fs, fs_path = _checkpoint_fs_path(commit_path, storage_options)
    marker_path = _checkpoint_path(fs_path, _COMMITTED_MARKER)
    if fs.exists(marker_path):
        with fs.open(marker_path) as stream:
            return [], int(json.loads(stream.read())["version"])

    records = []
    for record_path in sorted(fs.glob(_checkpoint_path(fs_path, _RECORDS_DIR, "*.json"))):
        with fs.open(record_path) as stream:
            record = json.loads(stream.read())
        if record.get("kind") == kind:
            records.append(record)
    if not records:
        msg = f"No {kind} checkpoint records found under {commit_path}"
        raise ValueError(msg)
    return records, None


def write_lance_checkpoint_marker(
    commit_path: str,
    version: int,
    storage_options: dict[str, Any] | None = None,
) -> None:
    fs, fs_path = _checkpoint_fs_path(commit_path, storage_options)
    marker_path = _checkpoint_path(fs_path, _COMMITTED_MARKER)
    fs.makedirs(posixpath.dirname(marker_path), exist_ok=True)
    with fs.open(marker_path, "w") as stream:
        stream.write(json.dumps({"version": version}, sort_keys=True, indent=2) + "\n")
