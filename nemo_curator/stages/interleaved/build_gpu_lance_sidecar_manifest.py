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

"""Build the immutable contract required by GPU Lance sidecars."""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from nemo_curator.stages.interleaved.gpu_key_lookup import (
    _build_sidecar_contract_bytes,
    _stable_global_ordinal_manifest_sha256,
)
from nemo_curator.stages.interleaved.lance import _validate_stable_global_ordinal_manifest
from nemo_curator.utils.uri import validate_credential_free_uri_identity


def _credential_free_uri(value: str) -> str:
    try:
        return validate_credential_free_uri_identity(value, "dataset URI")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _read_options(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()
    ):
        msg = f"Storage-options file must contain a JSON object of string values: {path}"
        raise ValueError(msg)
    return payload


def _partition_files(args: argparse.Namespace) -> tuple[tuple[str, ...], ...]:
    if args.replicated_glob is not None:
        validate_credential_free_uri_identity(args.replicated_glob, "replicated sidecar glob")
        paths = tuple(sorted(glob.glob(args.replicated_glob)))
        if not paths:
            msg = f"replicated sidecar glob matched no files: {args.replicated_glob}"
            raise ValueError(msg)
        return (paths,)
    partitions: dict[int, list[str]] = {}
    for spec in args.partition_file:
        raw_partition, separator, path = spec.partition("=")
        if not separator or not path:
            msg = f"--partition-file must use PARTITION_ID=PATH, got {spec!r}"
            raise ValueError(msg)
        validate_credential_free_uri_identity(path, "sidecar partition file URI")
        partition_id = int(raw_partition)
        if partition_id < 0:
            msg = f"partition IDs must be nonnegative, got {partition_id}"
            raise ValueError(msg)
        partitions.setdefault(partition_id, []).append(path)
    expected = list(range(len(partitions)))
    if sorted(partitions) != expected:
        msg = f"partition IDs must be contiguous {expected}, got {sorted(partitions)}"
        raise ValueError(msg)
    return tuple(tuple(partitions[partition_id]) for partition_id in expected)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hash exact GPU sidecar Parquet files, verify MPF partition ownership, and bind them to one pinned, "
            "append-only Lance fragment manifest."
        )
    )
    parser.add_argument("--dataset-uri", required=True, type=_credential_free_uri)
    parser.add_argument("--dataset-version", required=True, type=int)
    parser.add_argument("--dataset-storage-options-file")
    parser.add_argument("--sidecar-storage-options-file")
    parser.add_argument("--key-column", default="url")
    parser.add_argument("--row-id-column", default="stable_row_id")
    parser.add_argument("--layout", choices=("replicated_sorted", "hash_partitioned"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replicated-glob")
    source.add_argument("--partition-file", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    partition_files = _partition_files(args)
    if args.layout == "replicated_sorted" and len(partition_files) != 1:
        msg = "replicated_sorted layout requires exactly one partition"
        raise ValueError(msg)

    import lance

    dataset_storage_options = _read_options(args.dataset_storage_options_file)
    sidecar_storage_options = _read_options(args.sidecar_storage_options_file)
    dataset = lance.dataset(
        args.dataset_uri,
        version=args.dataset_version,
        storage_options=dataset_storage_options or None,
    )
    if dataset.version != args.dataset_version:
        msg = f"Lance dataset resolved version {dataset.version}; expected {args.dataset_version}"
        raise RuntimeError(msg)
    if not dataset.has_stable_row_ids:
        msg = "GPU Lance sidecars require a dataset with stable row IDs"
        raise ValueError(msg)
    manifest = _validate_stable_global_ordinal_manifest(dataset)
    fragment_manifest_sha256 = _stable_global_ordinal_manifest_sha256(
        args.dataset_uri,
        args.dataset_version,
        manifest,
    )
    raw_contract, contract_sha256 = _build_sidecar_contract_bytes(
        dataset=dataset,
        dataset_uri=args.dataset_uri,
        dataset_version=args.dataset_version,
        fragment_manifest_sha256=fragment_manifest_sha256,
        total_rows=manifest.total_rows,
        key_column=args.key_column,
        row_id_column=args.row_id_column,
        layout=args.layout,
        partition_files=partition_files,
        storage_options=sidecar_storage_options,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(raw_contract)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, args.output)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
    contract_payload = json.loads(raw_contract)
    summary: dict[str, Any] = {
        "files": sum(len(paths) for paths in partition_files),
        "fragment_manifest_sha256": fragment_manifest_sha256,
        "key_stable_ordinal_sha256": contract_payload["key_stable_ordinal_sha256"],
        "layout": args.layout,
        "manifest_sha256": contract_sha256,
        "output": str(args.output),
        "partition_count": len(partition_files),
        "rows": manifest.total_rows,
    }
    if "partitioning" in contract_payload:
        summary["partitioning"] = contract_payload["partitioning"]
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
