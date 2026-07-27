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

"""Fetch MINT-1T image payloads from a Lance table, with consolidated fetch actors.

Pipeline::

    InterleavedParquetReader
        -> LanceColumnFetchStage   (few, large actors -- see below)
        -> InterleavedParquetWriterStage

Why this recipe exists
----------------------
Sparse image fetch is dominated by *file-open* cost, not by bytes. Lance prefetches
the repetition index of every page in a column the first time a process opens a
fragment, so reading one image from a file costs almost as much as reading fifty.
Measured on the MINT-1T image table, only 0.822 of 3.520 GETs per image carried
image bytes; the rest was per-open overhead.

That overhead is paid **per process**. Within a process Lance's cache already
removes it entirely. So the fix is not a data rewrite and not a different sort
order -- both were tried and measured worse. The fix is to run *fewer, larger*
fetch actors so each fragment is opened once per node instead of once per worker.

Measured on the production table, 1,600,000 image occurrences on one node:

    16 actors x 128 io_threads   3.520 GETs/image   2,668 img/s
     1 actor  x 2048 io_threads  1.065 GETs/image   4,375 img/s   (1.64x)

``--fetch-actors-per-node`` encodes that choice. It sizes the stage's CPU request
so Ray places that many fetch actors per node, and scales ``io_threads`` inversely
so the *aggregate* in-flight request count stays fixed. Those two must move
together: consolidating actors without raising ``io_threads`` narrows the request
stream and gives the requests back, which was measured to cancel the entire gain.

Fewer actors is not unconditionally better. One actor per node cut requests 3.31x
but sustained only 0.50x the request rate, so the net was 1.64x rather than 3.31x.
The optimum is expected between 1 and 16 and is not yet measured; 2 is a
defensible default if you cannot measure your own workload.
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

# Aggregate in-flight requests per node, held constant as actor count varies.
# 2,048 is the measured operating point; the store served 16 nodes at this rate
# with zero errors, so it is not a store-side limit.
IN_FLIGHT_PER_NODE = 2048


def _json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        msg = "storage options must be a JSON object"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def build_fetch_stage(
    *,
    dataset: LanceDatasetConfig,
    actors_per_node: int,
    cpus_per_node: int,
    image_column: str,
    metadata_cache_bytes: int,
) -> LanceColumnFetchStage:
    """Return a fetch stage configured for *actors_per_node* actors on each node.

    The CPU request is what actually controls placement: asking for
    ``cpus_per_node / actors_per_node`` CPUs leaves room for exactly that many
    actors on a node. ``io_threads`` moves inversely so aggregate in-flight
    requests stay at :data:`IN_FLIGHT_PER_NODE` regardless of actor count --
    otherwise this recipe would confound two variables at once.
    """
    if actors_per_node < 1:
        msg = "actors_per_node must be at least 1"
        raise ValueError(msg)

    stage = LanceColumnFetchStage(
        dataset=dataset,
        # The metadata cache is what makes consolidation pay: it must hold the
        # fragment working set, or a long-lived actor silently re-opens files
        # and the saving disappears.
        index_cache=LanceIndexCacheConfig(metadata_cache_size_bytes=metadata_cache_bytes),
        columns={image_column: "binary_content"},
        # Not every document image reference resolves in the image table, and a
        # single unresolved key should not fail a corpus-scale run. Mark them
        # instead, so downstream stages can filter on a column rather than the
        # job dying on row one.
        presence_column="image_fetched",
        io_threads=max(1, IN_FLIGHT_PER_NODE // actors_per_node),
    )
    return stage.with_(resources=Resources(cpus=float(cpus_per_node) / actors_per_node))


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", required=True, help="Interleaved Parquet with image source_ref values")
    parser.add_argument("--output-path", required=True, help="Materialized interleaved Parquet output")
    parser.add_argument("--lance-uri", required=True, help="Pinned Lance image table URI")
    parser.add_argument("--lance-version", type=int, required=True)
    parser.add_argument("--key-column", default="url", help="Indexed key column in the Lance table")
    parser.add_argument("--index-name", default="url_btree")
    parser.add_argument("--image-column", default="image", help="Lance column holding image bytes")
    parser.add_argument("--storage-options", type=_json_object, default={})
    parser.add_argument(
        "--fetch-actors-per-node",
        type=int,
        default=1,
        help="Fetch actors per node. Fewer means fewer file-opens; 1 was measured fastest on one node",
    )
    parser.add_argument("--cpus-per-node", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--metadata-cache-gib", type=int, default=4)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--object-store-gib", type=int, default=32)
    parser.add_argument("--mode", choices=["ignore", "overwrite", "error"], default="error")
    args = parser.parse_args()

    dataset = LanceDatasetConfig(
        uri=args.lance_uri,
        version=args.lance_version,
        key_column=args.key_column,
        index_name=args.index_name,
        storage_options=args.storage_options,
    )

    pipeline = Pipeline(
        name="mint_sparse_image_fetch",
        description="Interleaved Parquet -> consolidated Lance image fetch -> Parquet",
    )
    pipeline.add_stage(InterleavedParquetReader(file_paths=args.input_path, files_per_partition=1))
    pipeline.add_stage(
        build_fetch_stage(
            dataset=dataset,
            actors_per_node=args.fetch_actors_per_node,
            cpus_per_node=args.cpus_per_node,
            image_column=args.image_column,
            metadata_cache_bytes=args.metadata_cache_gib * 1024**3,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_path,
            materialize_on_write=False,
            mode=args.mode,
            write_kwargs={"compression": "zstd"},
        )
    )

    print(pipeline.describe())
    ray_client = RayClient(
        num_cpus=args.cpus_per_node,
        object_store_memory=args.object_store_gib * 1024**3,
        include_dashboard=False,
    )
    try:
        ray_client.start()
        pipeline.run(executor=RayDataExecutor(), checkpoint_path=args.checkpoint_path)
    finally:
        ray_client.stop()

    # lance_gets_per_image and lance_images_per_file_open are emitted per batch by
    # the fetch stage; read them from the run's stage-performance output to confirm
    # consolidation actually took effect rather than assuming it did.
    print(
        f"\nfetch actors/node={args.fetch_actors_per_node} "
        f"io_threads={max(1, IN_FLIGHT_PER_NODE // args.fetch_actors_per_node)} "
        f"(aggregate in-flight per node: {IN_FLIGHT_PER_NODE})"
    )


if __name__ == "__main__":
    main()
