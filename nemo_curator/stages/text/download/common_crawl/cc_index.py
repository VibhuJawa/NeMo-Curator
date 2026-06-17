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

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

import boto3
import botocore.config
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask

# Columns to read from the CC index Parquet shards.
INDEX_COLS = [
    "url",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
    "content_digest",
    "url_host_name",
]

_SNAPSHOT_RE = re.compile(r"crawl=(CC-MAIN-[^/]+)")


@dataclass
class CCIndexShardListStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Fan-out stage: 1 EmptyTask -> list[FileGroupTask] (one per Parquet shard).

    Lists all Parquet shards for a given Common Crawl snapshot from the
    cc-index S3 bucket and emits one FileGroupTask per shard so that
    downstream stages can read shards in parallel.
    """

    snapshot: str
    cc_key: str
    cc_secret: str
    s3_endpoint: str = "https://pdx.s8k.io"
    cc_index_bucket: str = "cc-index"
    cc_index_prefix: str = "table/cc-main/warc/"
    max_shards: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))
    name: str = "cc_index_shard_list"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ([], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def process(self, task: EmptyTask) -> list[FileGroupTask]:
        """List S3 keys and return one FileGroupTask per shard.

        Args:
            task: Empty input task carrying only the dataset name.

        Returns:
            One FileGroupTask per Parquet shard, sorted by S3 key.
        """
        prefix = f"{self.cc_index_prefix}crawl={self.snapshot}/subset=warc/"

        s3 = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.cc_key,
            aws_secret_access_key=self.cc_secret,
            config=botocore.config.Config(signature_version="s3v4"),
        )

        paginator = s3.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=self.cc_index_bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]

        keys = sorted(keys)
        if self.max_shards is not None:
            keys = keys[: self.max_shards]

        logger.info(f"CCIndexShardListStage: found {len(keys)} shards for snapshot={self.snapshot}")

        return [FileGroupTask(dataset_name=task.dataset_name, data=[key]) for key in keys]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}


@dataclass
class CCIndexParquetReaderStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Fan-out stage: 1 FileGroupTask (shard key) -> list[DocumentBatch] (5K-row chunks).

    Downloads a single CC index Parquet shard from S3, streams it in
    chunk_size-row batches, filters out short WARC records, annotates
    each batch with the crawl snapshot id, and emits one DocumentBatch
    per chunk.
    """

    cc_key: str
    cc_secret: str
    s3_endpoint: str = "https://pdx.s8k.io"
    cc_index_bucket: str = "cc-index"
    chunk_size: int = 1_000  # smaller blocks → smoother pipeline, higher CPU utilization
    min_warc_length: int = 5_000
    max_batches: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))
    name: str = "cc_index_parquet_reader"
    # Cached per-actor state populated in setup() — avoids reconstructing the
    # S3FileSystem (TLS + Arrow thread-pool allocation) on every process() call.
    _s3fs: Any = field(init=False, repr=False, default=None)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        parsed = urllib.parse.urlparse(self.s3_endpoint)
        self._s3fs = pafs.S3FileSystem(
            access_key=self.cc_key,
            secret_key=self.cc_secret,
            endpoint_override=parsed.netloc,
            scheme=parsed.scheme,
            force_virtual_addressing=False,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [*INDEX_COLS, "cc_snapshot_id"])

    def process(self, task: FileGroupTask) -> list[DocumentBatch]:
        """Download one Parquet shard and yield filtered DocumentBatch chunks.

        Args:
            task: FileGroupTask whose data[0] is the S3 key of the shard.

        Returns:
            One DocumentBatch per chunk_size-row slice after length filtering.
        """
        key = task.data[0]

        # Extract snapshot id from the S3 key path.
        match = _SNAPSHOT_RE.search(key)
        if match:
            snapshot_id = match.group(1)
        else:
            logger.warning(
                f"CCIndexParquetReaderStage: could not extract snapshot id from key={key!r}; "
                "setting cc_snapshot_id to empty string"
            )
            snapshot_id = ""

        # Stream the Parquet shard row-group by row-group — first row group in ~2s,
        # no full 898 MB download. Uses self._s3fs cached in setup().
        if self._s3fs is None:
            # Fallback for direct (non-actor) calls — construct on demand.
            parsed = urllib.parse.urlparse(self.s3_endpoint)
            self._s3fs = pafs.S3FileSystem(
                access_key=self.cc_key,
                secret_key=self.cc_secret,
                endpoint_override=parsed.netloc,
                scheme=parsed.scheme,
                force_virtual_addressing=False,
            )

        batches: list[DocumentBatch] = []
        with self._s3fs.open_input_file(f"{self.cc_index_bucket}/{key}") as fh:
            pf = pq.ParquetFile(fh)
            for arrow_batch in pf.iter_batches(batch_size=self.chunk_size, columns=INDEX_COLS):
                df = arrow_batch.to_pandas()

                # Filter out WARC records that are too short.
                df = df[df["warc_record_length"] >= self.min_warc_length]

                if df.empty:
                    continue

                df = df.reset_index(drop=True)
                df["cc_snapshot_id"] = snapshot_id
                # CC index stores warc_filename as "<bucket>/<key>" (e.g. "crawl-data/CC-MAIN-...").
                # Strip the bucket prefix here at the source so all downstream consumers
                # receive a clean S3 key without needing to know about this quirk.
                if "warc_filename" in df.columns:
                    df["warc_filename"] = df["warc_filename"].str.removeprefix("crawl-data/")

                batches.append(DocumentBatch(dataset_name=task.dataset_name, data=df))

                if self.max_batches is not None and len(batches) >= self.max_batches:
                    break

        logger.info(
            f"CCIndexParquetReaderStage: key={key!r} -> {len(batches)} batches "
            f"(chunk_size={self.chunk_size}, min_warc_length={self.min_warc_length}, max_batches={self.max_batches})"
        )

        return batches

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}
