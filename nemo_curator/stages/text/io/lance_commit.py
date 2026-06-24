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

import base64
import pickle
from typing import Any

from nemo_curator.stages.text.io.lance_utils import (
    read_lance_checkpoint,
    write_lance_checkpoint_marker,
)


def commit_lance_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
    import lance
    from lance.schema import json_to_schema
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
    fragments = [
        (pickle.loads(base64.b64decode(record["fragment"])), json_to_schema(record["schema"]))  # noqa: S301
        for record in records
    ]
    schema = fragments[0][1]

    committer = LanceFragmentCommitter(path, schema=schema, mode=mode, storage_options=storage_options)
    if mode == "append":
        committer.on_write_start(schema)
    committer.on_write_complete([[(pickle.dumps(fragment), pickle.dumps(schema)) for fragment, schema in fragments]])
    version = lance.dataset(path, storage_options=storage_options).version
    write_lance_checkpoint_marker(commit_path, version, checkpoint_storage_options)
    return version


def commit_lance_annotation_checkpoint(
    path: str,
    commit_path: str,
    *,
    storage_options: dict[str, Any] | None = None,
    checkpoint_storage_options: dict[str, Any] | None = None,
) -> int:
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
    records_by_fragment = {int(record["fragment_id"]): record for record in records}
    if len(records_by_fragment) != len(records):
        msg = "Ensure each Lance fragment is updated by at most one writer task."
        raise ValueError(msg)
    updated_fragments = [
        pickle.loads(base64.b64decode(record["updated_fragment"]))  # noqa: S301
        for record in records_by_fragment.values()
    ]
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
