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

"""Materialize interleaved image payloads through a shared payload cache.

Pipeline::

    InterleavedParquetReader
        -> InterleavedParquetWriterStage(payload_cache_root=...)

Interleaved corpora reference the same image from many documents, so an
uncached materialization pass fetches the average MINT-1T HTML image 4.4 times.
``PayloadCache`` turns those repeats into reads from a shared filesystem.

Two limits are measured and documented in ``README.md``: entries are keyed by
``source_ref``, so the cache only helps when the same locator recurs, and 4.4x
is the serial limit -- concurrent workers miss on keys their peers have not
written yet, so a single pass realises a fraction of it without key-affinity
routing.
"""

from __future__ import annotations

import argparse
import os

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetReader, InterleavedParquetWriterStage


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-path", required=True, help="Normalized interleaved Parquet path")
    parser.add_argument("--cache-path", required=True, help="Payload cache directory on a shared filesystem")
    parser.add_argument("--output-path", required=True, help="Materialized interleaved Parquet output path")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--num-cpus", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--max-batch-mib", type=int, default=512)
    parser.add_argument("--object-store-gib", type=int, default=32)
    parser.add_argument("--mode", choices=["ignore", "overwrite", "error"], default="error")
    args = parser.parse_args()

    pipeline = Pipeline(
        name="interleaved_image_payload_cache",
        description="Interleaved Parquet -> cached image materialization on write -> Parquet",
    )
    pipeline.add_stage(
        InterleavedParquetReader(
            file_paths=args.input_path,
            files_per_partition=1,
            max_batch_bytes=args.max_batch_mib * 1024**2,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_path,
            payload_cache_root=args.cache_path,
            mode=args.mode,
            write_kwargs={"compression": "zstd"},
        )
    )

    print(pipeline.describe())
    ray_client = RayClient(
        num_cpus=args.num_cpus,
        object_store_memory=args.object_store_gib * 1024**3,
        include_dashboard=False,
    )
    try:
        ray_client.start()
        pipeline.run(executor=RayDataExecutor(), checkpoint_path=args.checkpoint_path)
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
