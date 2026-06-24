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

import pickle
from typing import Any

from nemo_curator.stages.text.io.lance_utils import (
    object_from_base64,
    read_lance_checkpoint,
    schema_from_json_value,
    write_lance_checkpoint_marker,
)


def commit_lance_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
    """Commit checkpoint records produced by ``LanceWriter``."""

    import lance
    from lance_ray import LanceFragmentCommitter

    records, committed_version = read_lance_checkpoint(commit_path, "lance_write", checkpoint_storage_options)
    if committed_version is not None:
        return committed_version

    dataset_paths = {record["dataset_path"] for record in records}
    if dataset_paths != {path}:
        msg = f"Checkpoint records are for {sorted(dataset_paths)}, not {path}"
        raise ValueError(msg)
    modes = {record["mode"] for record in records}
    if len(modes) != 1:
        msg = f"Expected one write mode; got {sorted(modes)}"
        raise ValueError(msg)
    mode = str(next(iter(modes)))
    fragments = [(object_from_base64(record["fragment"]), schema_from_json_value(record["schema"])) for record in records]
    schema = fragments[0][1]

    committer = LanceFragmentCommitter(path, schema=schema, mode=mode, storage_options=storage_options)
    if mode == "append":
        committer.on_write_start(schema)
    committer.on_write_complete([[(pickle.dumps(fragment), pickle.dumps(schema)) for fragment, schema in fragments]])
    version = lance.dataset(path, storage_options=storage_options).version
    write_lance_checkpoint_marker(commit_path, version, checkpoint_storage_options)
    return version


def _annotation_records_by_fragment(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for record in records:
        fragment_id = int(record["fragment_id"])
        if fragment_id in selected:
            msg = (
                f"Conflicting Lance annotation checkpoint records for fragment {fragment_id}. "
                "Ensure each Lance fragment is updated by at most one writer task."
            )
            raise ValueError(msg)
        selected[fragment_id] = record
    return selected


def commit_lance_annotation_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
    """Commit checkpoint records produced by ``LanceAnnotationWriter``."""

    import lance

    records, committed_version = read_lance_checkpoint(
        commit_path, "lance_annotation_update", checkpoint_storage_options
    )
    if committed_version is not None:
        return committed_version

    dataset_paths = {record["dataset_path"] for record in records}
    if dataset_paths != {path}:
        msg = f"Checkpoint records are for {sorted(dataset_paths)}, not {path}"
        raise ValueError(msg)
    read_versions = {int(record["dataset_version"]) for record in records}
    if len(read_versions) != 1:
        msg = f"Expected one dataset version; got {sorted(read_versions)}"
        raise ValueError(msg)
    read_version = next(iter(read_versions))
    records_by_fragment = _annotation_records_by_fragment(records)
    updated_fragments = [object_from_base64(record["updated_fragment"]) for record in records_by_fragment.values()]
    fields_modified = sorted({field for record in records_by_fragment.values() for field in record["fields_modified"]})
    operation = lance.LanceOperation.Update(updated_fragments=updated_fragments, fields_modified=fields_modified)
    version = lance.LanceDataset.commit(
        path,
        operation,
        read_version=read_version,
        storage_options=storage_options,
    ).version
    write_lance_checkpoint_marker(
        commit_path,
        version,
        checkpoint_storage_options,
    )
    return version
