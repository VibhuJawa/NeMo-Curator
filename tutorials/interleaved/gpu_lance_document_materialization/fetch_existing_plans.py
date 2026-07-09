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

"""Fetch durable coordinate plans into checkpointed Arrow payload overlays."""

from __future__ import annotations

import argparse

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved import LanceCoordinatePayloadOverlayStage, LanceCoordinatePlanReader


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
    parser.add_argument("--plan-root", required=True)
    parser.add_argument("--document-uri", required=True)
    parser.add_argument("--document-version", required=True, type=_positive)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--image-version", required=True, type=_positive)
    parser.add_argument("--sidecar-manifest-sha256", required=True)
    parser.add_argument("--fragment-manifest-sha256", required=True)
    parser.add_argument("--missing-key-policy", choices=("error", "null"), default="error")
    parser.add_argument("--expected-fragment-id", action="append", required=True, type=_nonnegative)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--fetch-batch-size", type=_positive, default=1024)
    parser.add_argument("--max-pending-takes", type=_positive, default=16)
    parser.add_argument("--payload-window-bytes", choices=("256MiB", "1GiB", "4GiB"), default="1GiB")
    parser.add_argument("--payload-actor-cpus", type=_positive, default=8)
    parser.add_argument("--payload-overlay-workers", type=_positive)
    parser.add_argument("--num-cpus", type=_positive)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pipeline = Pipeline(
        name="gpu-lance-payload-overlay-fetch",
        stages=[
            LanceCoordinatePlanReader(
                plan_root=args.plan_root,
                document_uri=args.document_uri,
                document_version=args.document_version,
                image_uri=args.image_uri,
                image_version=args.image_version,
                sidecar_manifest_sha256=args.sidecar_manifest_sha256,
                fragment_manifest_sha256=args.fragment_manifest_sha256,
                missing_key_policy=args.missing_key_policy,
                expected_fragment_ids=args.expected_fragment_id,
            ),
            LanceCoordinatePayloadOverlayStage(
                image_uri=args.image_uri,
                image_version=args.image_version,
                output_root=args.output_root,
                payload_window_bytes=args.payload_window_bytes,
                fetch_batch_size=args.fetch_batch_size,
                max_pending=args.max_pending_takes,
                payload_actor_cpus=args.payload_actor_cpus,
                payload_overlay_workers=args.payload_overlay_workers,
            ),
        ],
    )
    with RayClient(num_cpus=args.num_cpus):
        pipeline.run(checkpoint_path=args.checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
