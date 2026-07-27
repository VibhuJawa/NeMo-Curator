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

"""Fetch image payloads from a Lance table for an interleaved corpus.

Pipeline::

    InterleavedParquetReader -> LanceColumnFetchStage -> InterleavedParquetWriterStage

Sparse fetch is limited by file-opens rather than bytes, and that cost is paid
per process, so running fewer and larger fetch actors is what makes it fast.
``--fetch-actors-per-node`` controls that; see ``README.md`` for the measurements.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved import (
    LanceColumnFetchStage,
    LanceDatasetConfig,
    LanceIndexCacheConfig,
)
from nemo_curator.stages.interleaved.io import InterleavedParquetReader, InterleavedParquetWriterStage
from nemo_curator.stages.resources import Resources

IN_FLIGHT_PER_NODE = 2048
METADATA_CACHE_BYTES = 4 * 1024**3


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        msg = "storage options must be a JSON object"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def build_fetch_stage(
    dataset: LanceDatasetConfig,
    *,
    actors_per_node: int,
    cpus_per_node: int,
    image_column: str = "image",
) -> LanceColumnFetchStage:
    """Return a fetch stage placing *actors_per_node* actors on each node.

    The CPU request is what controls placement. ``io_threads`` moves inversely so
    aggregate in-flight requests stay at ``IN_FLIGHT_PER_NODE`` however many actors
    there are -- consolidating actors *without* widening each one's request stream
    was measured to cancel the entire gain.
    """
    if actors_per_node < 1:
        msg = "actors_per_node must be at least 1"
        raise ValueError(msg)

    stage = LanceColumnFetchStage(
        dataset=dataset,
        # Must hold the fragment working set, or a long-lived actor silently
        # re-opens files and the saving disappears.
        index_cache=LanceIndexCacheConfig(metadata_cache_size_bytes=METADATA_CACHE_BYTES),
        columns={image_column: "binary_content"},
        # Not every reference resolves; mark them rather than failing the run.
        presence_column="image_fetched",
        io_threads=max(1, IN_FLIGHT_PER_NODE // actors_per_node),
    )
    return stage.with_(resources=Resources(cpus=float(cpus_per_node) / actors_per_node))


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--lance-version", type=int, required=True)
    parser.add_argument("--storage-options", type=_json_object, default={})
    parser.add_argument("--fetch-actors-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--mode", choices=["ignore", "overwrite", "error"], default="error")
    args = parser.parse_args()

    dataset = LanceDatasetConfig(
        uri=args.lance_uri,
        version=args.lance_version,
        key_column="url",
        index_name="url_btree",
        storage_options=args.storage_options,
    )

    pipeline = Pipeline(name="mint_sparse_image_fetch")
    pipeline.add_stage(InterleavedParquetReader(file_paths=args.input_path, files_per_partition=1))
    pipeline.add_stage(
        build_fetch_stage(
            dataset,
            actors_per_node=args.fetch_actors_per_node,
            cpus_per_node=args.cpus_per_node,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_path,
            materialize_on_write=False,
            mode=args.mode,
        )
    )

    print(pipeline.describe())
    ray_client = RayClient(num_cpus=args.cpus_per_node, include_dashboard=False)
    try:
        ray_client.start()
        pipeline.run(executor=RayDataExecutor())
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
