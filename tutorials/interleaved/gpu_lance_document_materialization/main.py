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

"""Materialize image payloads into pinned interleaved Lance documents."""

from __future__ import annotations

import argparse

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved import GpuLanceDocumentMaterializer


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "value must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        msg = "value must be nonnegative"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-uri", required=True)
    parser.add_argument("--document-version", required=True, type=_positive)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--image-version", required=True, type=_positive)
    parser.add_argument("--index-shard", action="append", required=True)
    parser.add_argument("--index-manifest-uri", required=True)
    parser.add_argument("--index-manifest-sha256", required=True)
    parser.add_argument("--coordinate-plan-output-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--node-local-spool-root")
    parser.add_argument(
        "--materialization-mode",
        choices=("payload_overlay", "document_patch"),
        default="payload_overlay",
    )
    parser.add_argument("--fragment-id", action="append", type=_nonnegative)
    parser.add_argument("--fetch-task-window", type=_positive, default=8)
    parser.add_argument("--fetch-batch-size", type=_positive, default=1024)
    parser.add_argument("--max-pending-takes", type=_positive, default=16)
    parser.add_argument("--payload-window-bytes", choices=("256MiB", "1GiB", "4GiB"), default="1GiB")
    parser.add_argument("--payload-actor-cpus", type=_positive, default=8)
    parser.add_argument("--payload-patch-workers", type=_positive)
    parser.add_argument("--num-cpus", type=_positive)
    parser.add_argument("--num-gpus", type=_positive)
    parser.add_argument("--slurm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    materializer = GpuLanceDocumentMaterializer(
        document_uri=args.document_uri,
        document_version=args.document_version,
        image_uri=args.image_uri,
        image_version=args.image_version,
        index_shards=args.index_shard,
        index_manifest_uri=args.index_manifest_uri,
        index_manifest_sha256=args.index_manifest_sha256,
        coordinate_plan_output_path=args.coordinate_plan_output_path,
        output_root=args.output_root,
        node_local_spool_root=args.node_local_spool_root,
        materialization_mode=args.materialization_mode,
        fragment_ids=args.fragment_id,
        fetch_task_window=args.fetch_task_window,
        fetch_batch_size=args.fetch_batch_size,
        max_pending_takes=args.max_pending_takes,
        payload_window_bytes=args.payload_window_bytes,
        payload_actor_cpus=args.payload_actor_cpus,
        payload_patch_workers=args.payload_patch_workers,
    )
    pipeline = Pipeline(
        name="gpu-lance-document-image-fetch",
        stages=[materializer],
    )
    client = SlurmRayClient() if args.slurm else RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    with client:
        pipeline.run(executor=RayActorPoolExecutor())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
