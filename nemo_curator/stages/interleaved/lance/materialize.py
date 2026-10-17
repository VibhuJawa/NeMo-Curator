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

"""Materialize interleaved image payloads from a Lance table by stable row id."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import InterleavedBatch

if TYPE_CHECKING:
    import lance

    from nemo_curator.backends.base import WorkerMetadata

_IMAGE_MODALITY = "image"


@dataclass
class InterleavedLanceMaterializeStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Fill ``binary_content`` on image rows from a separate Lance image table.

    Interleaved documents and image payloads are stored apart: the document table has
    one row per text or image position, and image bytes live in a much larger Lance
    table. Each image row carries the **stable row id** of its payload, and this stage
    turns those ids into bytes.

    This is a sparse gather -- the ids are scattered across the whole image table --
    and its cost is dominated by *opening fragments*, not by reading bytes. Lance
    prefetches per-page metadata the first time a process touches a fragment. Measured
    on a 50 TB image table, only 0.822 of 3.520 requests per image carried image data;
    the rest was that fixed per-open cost, and files a worker read a *single* image
    from still averaged 9.73 metadata requests -- the same as files it read fifteen
    images from.

    The cost is paid **per process**; inside one process Lance's own cache removes it
    entirely. So the dataset opened in :meth:`setup` and held for the worker's lifetime
    is the point of this stage -- it lets one worker amortise a fragment open across
    every batch it handles. Measured on one node with aggregate in-flight requests
    held fixed at 2,048:

    =========================  ==================  ============  ==============
    Fetch actors per node      Requests per image  Throughput    Amplification
    =========================  ==================  ============  ==============
    16                         3.520               2,668 img/s   1.416
    1                          1.065               4,375 img/s   0.997
    =========================  ==================  ============  ==============

    **Prefer few, large actors.** Give this stage a large ``Resources(cpus=...)`` so
    Ray places few per node; every extra actor re-pays every fragment open. Raise
    ``io_threads`` as you lower the actor count, so consolidating processes does not
    also narrow the request stream -- doing only the former was measured to give the
    entire gain back. Fewer is not unconditionally better: one actor per node cut
    requests 3.31x but sustained only half the request rate, netting 1.64x.

    Watch ``lance_requests_per_image`` in the emitted metrics. It trends towards 1
    when a worker is reusing its opens and sits near 3.5 when it is not.

    Args:
        uri: Lance image table URI. Must have stable row ids.
        version: Pinned version. Pinning keeps a long-lived handle valid.
        row_id_column: Interleaved column holding each image's stable row id.
        binary_column: Lance column holding the image bytes.
        storage_options: Object-store options for Lance.
        io_threads: Concurrent take calls for this worker. Raise as actors per node fall.
        take_batch_size: Row ids per take call.
        metadata_cache_bytes: Shared session cache. Must hold the fragment working set,
            or the handle is kept while its metadata is evicted and the saving is lost.
    """

    uri: str
    version: int
    row_id_column: str
    binary_column: str = "image"
    storage_options: dict[str, str] = field(default_factory=dict)
    io_threads: int = 256
    take_batch_size: int = 2048
    metadata_cache_bytes: int = 4 * 1024**3
    name: str = "interleaved_lance_materialize"

    _dataset: lance.LanceDataset | None = field(default=None, init=False, repr=False, compare=False)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version <= 0:
            msg = "version must be positive; pin a version so a long-lived handle stays valid"
            raise ValueError(msg)
        if not self.row_id_column:
            msg = "row_id_column must not be empty"
            raise ValueError(msg)
        for name, value in {"io_threads": self.io_threads, "take_batch_size": self.take_batch_size}.items():
            if value <= 0:
                msg = f"{name} must be greater than 0"
                raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.row_id_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["binary_content"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Open the image table once and hold it for the life of this worker."""
        import lance

        dataset = lance.dataset(
            self.uri,
            version=self.version,
            storage_options=self.storage_options or None,
            session=lance.Session(metadata_cache_size_bytes=self.metadata_cache_bytes),
        )
        # Without stable row ids, Lance interprets these values fragment-locally and a
        # global id either errors or addresses the wrong row. Fail here rather than
        # corrupt payloads.
        if not dataset.has_stable_row_ids:
            msg = f"{self.uri} does not have stable row ids, so global row ids cannot be resolved"
            raise ValueError(msg)
        if self.binary_column not in dataset.schema.names:
            msg = f"Lance column {self.binary_column!r} is missing from {self.uri}"
            raise ValueError(msg)
        if not callable(getattr(dataset, "_take_rows", None)):
            msg = "this pylance build does not expose LanceDataset._take_rows"
            raise TypeError(msg)

        self._dataset = dataset
        self._executor = ThreadPoolExecutor(max_workers=self.io_threads, thread_name_prefix="lance-materialize")

    def teardown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._dataset = None

    def _pending_row_ids(self, table: pa.Table) -> list[int]:
        """Distinct row ids of image rows still missing a payload, in fetch order."""
        pending = pc.equal(table["modality"], pa.scalar(_IMAGE_MODALITY))
        if "binary_content" in table.column_names:
            pending = pc.and_(pending, pc.is_null(table["binary_content"]))
        row_ids = table[self.row_id_column].filter(pending).drop_null().to_pylist()
        # Sorting keeps each take inside as few fragments as possible, so the opens
        # this worker has already paid for are the ones it reuses.
        return sorted({int(row_id) for row_id in row_ids})

    def _take(self, row_ids: list[int]) -> dict[int, bytes]:
        """Fetch payloads for *row_ids*.

        ``_take_rows`` returns rows in the order requested, which is what lets the
        result be zipped back onto the ids. It does **not** return the id, and it
        silently returns *fewer* rows when an id is absent -- which would shift every
        subsequent payload onto the wrong document. So a short result is never mapped
        positionally; that chunk is resolved one id at a time to find the gaps.
        """
        fetched = self._dataset._take_rows(row_ids, columns=[self.binary_column])
        payloads = fetched[self.binary_column].to_pylist()
        if len(payloads) == len(row_ids):
            return {row_id: payload for row_id, payload in zip(row_ids, payloads, strict=True) if payload is not None}

        resolved: dict[int, bytes] = {}
        for row_id in row_ids:
            single = self._dataset._take_rows([row_id], columns=[self.binary_column])
            if single.num_rows == 1:
                payload = single[self.binary_column][0].as_py()
                if payload is not None:
                    resolved[row_id] = payload
        return resolved

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        assert self._dataset is not None, "setup() must run before process()"  # noqa: S101
        table = task.to_pyarrow()
        if self.row_id_column not in table.column_names or "modality" not in table.column_names:
            return task

        row_ids = self._pending_row_ids(table)
        if not row_ids:
            return task

        started = time.perf_counter()
        before = self._dataset.io_stats_incremental()
        chunks = [row_ids[i : i + self.take_batch_size] for i in range(0, len(row_ids), self.take_batch_size)]
        payloads: dict[int, bytes] = {}
        for chunk_payloads in self._executor.map(self._take, chunks):
            payloads.update(chunk_payloads)
        after = self._dataset.io_stats_incremental()

        existing = (
            table["binary_content"].to_pylist() if "binary_content" in table.column_names else [None] * table.num_rows
        )
        filled = [
            payloads.get(int(row_id)) if value is None and row_id is not None else value
            for row_id, value in zip(table[self.row_id_column].to_pylist(), existing, strict=True)
        ]
        column = pa.array(filled, type=pa.large_binary())
        table = (
            table.set_column(table.column_names.index("binary_content"), "binary_content", column)
            if "binary_content" in table.column_names
            else table.append_column("binary_content", column)
        )

        requests = int(after.read_iops) - int(before.read_iops)
        self._log_metrics(
            {
                "lance_row_ids_requested": float(len(row_ids)),
                "lance_payloads_returned": float(len(payloads)),
                "lance_row_ids_missing": float(len(row_ids) - len(payloads)),
                "lance_requests": float(requests),
                "lance_requests_per_image": requests / len(row_ids),
                "lance_read_bytes": float(int(after.read_bytes) - int(before.read_bytes)),
                "lance_materialize_seconds": time.perf_counter() - started,
            }
        )
        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
