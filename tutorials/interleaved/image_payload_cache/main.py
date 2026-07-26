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
        -> CachedMaterializeStage
        -> InterleavedParquetWriterStage

Interleaved corpora reference the same image from many documents, so an
uncached materialization pass fetches the average MINT-1T HTML image 4.4 times.
``PayloadCache`` turns those repeats into reads from a shared filesystem.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import InterleavedParquetReader, InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.utils.materialization import materialize_task_binary_content
from nemo_curator.stages.interleaved.utils.payload_cache import PayloadCache
from nemo_curator.tasks import InterleavedBatch


@dataclass
class CachedMaterializeStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Fill ``binary_content`` on image rows, serving repeated images from *cache*."""

    cache: PayloadCache
    name: str = "cached_materialize"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return materialize_task_binary_content(task, cache=self.cache)


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
        description="Interleaved Parquet -> cached image materialization -> Parquet",
    )
    pipeline.add_stage(
        InterleavedParquetReader(
            file_paths=args.input_path,
            files_per_partition=1,
            max_batch_bytes=args.max_batch_mib * 1024**2,
        )
    )
    pipeline.add_stage(CachedMaterializeStage(cache=PayloadCache(Path(args.cache_path))))
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
