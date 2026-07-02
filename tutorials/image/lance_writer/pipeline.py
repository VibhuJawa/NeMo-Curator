# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Build, ingest, and commit the tar-attempt-to-Lance image workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.writer import LanceWriter, commit_lance_checkpoint

from .manifest import METADATA_FILENAME, build_manifest
from .stages import (
    IMAGE_SCHEMA,
    FppPackMaterializationStage,
    FppPackPartitioningStage,
)


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        message = "storage options must be a JSON object"
        raise argparse.ArgumentTypeError(message)
    return parsed


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def build_pipeline(
    *,
    manifest_dir: str,
    dataset_uri: str,
    lance_commit_path: str,
    source_storage_options: dict[str, Any],
    lance_storage_options: dict[str, Any],
) -> Pipeline:
    """Compose a source, fused materializer, and checkpointed Lance sink."""
    writer = LanceWriter(
        path=dataset_uri,
        commit_path=lance_commit_path,
        schema=IMAGE_SCHEMA,
        mode="create",
        write_kwargs={
            "storage_options": lance_storage_options,
            "data_storage_version": "2.2",
            "max_rows_per_file": 500_000,
        },
    ).with_(resources=Resources(cpus=1), batch_size=1)
    writer.is_sink_stage = True
    return Pipeline(
        name="tar_attempts_to_lance_images",
        description="Stable FPP retry-attempt deduplication and large-binary Lance writing",
        stages=[
            FppPackPartitioningStage(
                manifest_dir=manifest_dir,
                dataset_name=dataset_uri,
            ),
            FppPackMaterializationStage(source_storage_options_json=_canonical_json(source_storage_options)),
            writer,
        ],
    )


def _ingest(args: argparse.Namespace) -> None:
    metadata = json.loads((Path(args.manifest_dir) / METADATA_FILENAME).read_text())
    pipeline = build_pipeline(
        manifest_dir=args.manifest_dir,
        dataset_uri=args.dataset_uri,
        lance_commit_path=args.lance_commit_path,
        source_storage_options=args.source_storage_options,
        lance_storage_options=args.lance_storage_options,
    )
    logger.info("Snapshot: {}\n{}", metadata["snapshot_id"], pipeline.describe())
    ray_kwargs = {"ray_temp_dir": args.ray_temp_dir} if args.ray_temp_dir else {}
    ray_client = RayClient(
        num_cpus=args.cpus,
        object_store_memory=args.object_store_memory,
        include_dashboard=False,
        **ray_kwargs,
    )
    try:
        ray_client.start()
        pipeline.run(
            executor=RayDataExecutor(),
            checkpoint_path=args.checkpoint_path,
        )
    finally:
        ray_client.stop()


def _commit(args: argparse.Namespace) -> None:
    version = commit_lance_checkpoint(
        args.dataset_uri,
        args.lance_commit_path,
        storage_options=args.lance_storage_options,
    )
    print(json.dumps({"dataset_uri": args.dataset_uri, "version": version}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a URL-addressable Lance image table from tar attempts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--inventory", required=True)
    manifest_parser.add_argument("--manifest-dir", required=True)
    manifest_parser.add_argument("--target-pack-bytes", type=int, default=1024**3)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--manifest-dir", required=True)
    ingest_parser.add_argument("--dataset-uri", required=True)
    ingest_parser.add_argument("--lance-commit-path", required=True)
    ingest_parser.add_argument("--checkpoint-path", required=True)
    ingest_parser.add_argument("--source-storage-options", type=_json_object, default={})
    ingest_parser.add_argument("--lance-storage-options", type=_json_object, default={})
    ingest_parser.add_argument("--cpus", type=int, default=8)
    ingest_parser.add_argument("--object-store-memory", type=int, default=None)
    ingest_parser.add_argument("--ray-temp-dir", default=None)

    commit_parser = subparsers.add_parser("commit")
    commit_parser.add_argument("--dataset-uri", required=True)
    commit_parser.add_argument("--lance-commit-path", required=True)
    commit_parser.add_argument("--lance-storage-options", type=_json_object, default={})
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build-manifest":
        result = build_manifest(
            args.inventory,
            args.manifest_dir,
            args.target_pack_bytes,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "ingest":
        _ingest(args)
    else:
        _commit(args)


if __name__ == "__main__":
    main()
