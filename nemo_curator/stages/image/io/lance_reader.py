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

"""High-throughput, coordinate-based reads from Lance image datasets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pyarrow as pa

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.base import BaseReader, ReaderOutput
from nemo_curator.tasks import DocumentBatch, EmptyTask
from nemo_curator.tasks.tasks import Task
from nemo_curator.utils.hash_utils import get_deterministic_hash
from nemo_curator.utils.lance import LANCE_FRAGID_COLUMN

LANCE_ROW_OFFSET_COLUMN = "__lance_row_offset"
DEFAULT_INDEX_CACHE_SIZE_BYTES = 4 * 1024**3


@dataclass(frozen=True, slots=True)
class LanceImageSlice:
    """A version-local, contiguous range of logical rows in one fragment."""

    fragment_id: int
    row_offset: int
    row_count: int

    def __post_init__(self) -> None:
        if self.fragment_id < 0:
            msg = "fragment_id must be non-negative"
            raise ValueError(msg)
        if self.row_offset < 0:
            msg = "row_offset must be non-negative"
            raise ValueError(msg)
        if self.row_count <= 0:
            msg = "row_count must be greater than 0"
            raise ValueError(msg)


@dataclass
class LanceImageReadTask(Task[list[LanceImageSlice]]):
    """Task containing contiguous fragment slices to read concurrently."""

    data: list[LanceImageSlice] = field(default_factory=list)

    @property
    def num_items(self) -> int:
        return sum(image_slice.row_count for image_slice in self.data)

    def validate(self) -> bool:
        return bool(self.data)

    def get_deterministic_id(self) -> str:
        lance_metadata = self._metadata.get("lance") or {}
        parts = [
            str(lance_metadata.get("path", self.dataset_name)),
            str(lance_metadata.get("version", "")),
            *(
                f"{image_slice.fragment_id}:{image_slice.row_offset}:{image_slice.row_count}"
                for image_slice in self.data
            ),
        ]
        return get_deterministic_hash(parts)

    @classmethod
    def from_slices(
        cls,
        path: str,
        version: int,
        image_slices: list[LanceImageSlice],
    ) -> LanceImageReadTask:
        """Build a pinned task from a compact coordinate work manifest."""
        return cls(
            dataset_name=path,
            data=list(image_slices),
            _metadata={
                "source_files": [path],
                "lance": {
                    "path": path,
                    "version": version,
                },
            },
        )


@dataclass
class LanceImageSlicePartitioningStage(ProcessingStage[EmptyTask, LanceImageReadTask]):
    """Partition a pinned Lance dataset into locality-preserving image reads.

    Slices are interleaved across fragments before they are packed into tasks.
    This gives every reader task independent object-store reads instead of
    serially walking one large fragment.
    """

    path: str
    rows_per_slice: int = 100
    slices_per_partition: int = 40
    fragment_ids: list[int] | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "lance_image_slice_partitioning"

    def __post_init__(self) -> None:
        if not self.path:
            msg = "path is required"
            raise ValueError(msg)
        if self.rows_per_slice <= 0:
            msg = "rows_per_slice must be greater than 0"
            raise ValueError(msg)
        if self.slices_per_partition <= 0:
            msg = "slices_per_partition must be greater than 0"
            raise ValueError(msg)
        self.read_kwargs = dict(self.read_kwargs or {})

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def _dataset_kwargs(self) -> dict[str, Any]:
        read_kwargs = dict(self.read_kwargs)
        dataset_kwargs = dict(read_kwargs.pop("dataset_options", {}) or {})
        version = dataset_kwargs.pop("version", None)
        version = read_kwargs.pop("version", version)
        if version is not None:
            dataset_kwargs["version"] = version
        storage_options = read_kwargs.pop("storage_options", None)
        if storage_options is not None:
            dataset_kwargs["storage_options"] = storage_options
        return dataset_kwargs

    def _selected_fragments(self, dataset: object) -> list[object]:
        fragments_by_id = {fragment.fragment_id: fragment for fragment in dataset.get_fragments()}  # type: ignore[attr-defined]
        if self.fragment_ids is None:
            selected_ids = sorted(fragments_by_id)
        else:
            selected_ids = sorted(set(self.fragment_ids))
            missing = sorted(set(selected_ids) - set(fragments_by_id))
            if missing:
                msg = f"Lance dataset does not contain requested fragment ids: {missing[:10]}"
                raise ValueError(msg)
        return [fragments_by_id[fragment_id] for fragment_id in selected_ids]

    def process(self, _: EmptyTask) -> list[LanceImageReadTask]:
        import lance

        dataset = lance.dataset(self.path, **self._dataset_kwargs())
        fragments = self._selected_fragments(dataset)
        fragment_rows = {
            fragment.fragment_id: int(fragment.physical_rows - fragment.num_deletions) for fragment in fragments
        }
        max_rows = max(fragment_rows.values(), default=0)

        tasks: list[LanceImageReadTask] = []
        pending: list[LanceImageSlice] = []
        for row_offset in range(0, max_rows, self.rows_per_slice):
            for fragment in fragments:
                row_count = min(self.rows_per_slice, fragment_rows[fragment.fragment_id] - row_offset)
                if row_count <= 0:
                    continue
                pending.append(LanceImageSlice(fragment.fragment_id, row_offset, row_count))
                if len(pending) == self.slices_per_partition:
                    tasks.append(self._make_task(dataset.version, pending))
                    pending = []
        if pending:
            tasks.append(self._make_task(dataset.version, pending))
        return tasks

    def _make_task(self, version: int, image_slices: list[LanceImageSlice]) -> LanceImageReadTask:
        return LanceImageReadTask.from_slices(self.path, version, image_slices)


@dataclass
class LanceImageSliceReaderStage(BaseReader[LanceImageReadTask]):
    """Read encoded image slices through Lance's public fragment scanner API.

    The stage is an actor so its Lance ``Session`` and dataset metadata cache
    survive across tasks. Threaded scans preserve task order because
    ``ThreadPoolExecutor.map`` returns results in input order.
    """

    path: str = ""
    fields: list[str] | None = field(default_factory=lambda: ["image"])
    reader_threads: int = 32
    lance_cpu_threads: int = 16
    lance_io_threads: int = 64
    index_cache_size_bytes: int = DEFAULT_INDEX_CACHE_SIZE_BYTES
    include_coordinates: bool = True
    resources: Resources = field(default_factory=lambda: Resources(cpus=32.0))
    allow_empty: bool = False
    name: str = "lance_image_slice_reader"
    _dataset: Any = field(default=None, init=False, repr=False, compare=False)
    _session: Any = field(default=None, init=False, repr=False, compare=False)
    _opened_version: int | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.path:
            msg = "path is required"
            raise ValueError(msg)
        if self.fields is not None and not self.fields:
            msg = "fields must contain at least one column"
            raise ValueError(msg)
        if self.reader_threads <= 0:
            msg = "reader_threads must be greater than 0"
            raise ValueError(msg)
        if self.lance_cpu_threads <= 0:
            msg = "lance_cpu_threads must be greater than 0"
            raise ValueError(msg)
        if self.lance_io_threads <= 0:
            msg = "lance_io_threads must be greater than 0"
            raise ValueError(msg)
        if self.index_cache_size_bytes < 0:
            msg = "index_cache_size_bytes must be non-negative"
            raise ValueError(msg)
        self.read_kwargs = dict(self.read_kwargs or {})
        self.runtime_env = {
            "env_vars": {
                "LANCE_CPU_THREADS": str(self.lance_cpu_threads),
                "LANCE_IO_THREADS": str(self.lance_io_threads),
            }
        }

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.fields or [])
        if self.include_coordinates:
            output_fields.extend([LANCE_FRAGID_COLUMN, LANCE_ROW_OFFSET_COLUMN])
        return ["data"], output_fields

    def teardown(self) -> None:
        self._dataset = None
        self._session = None
        self._opened_version = None

    def _task_version(self, task: LanceImageReadTask) -> int:
        lance_metadata = task._metadata.get("lance") or {}
        task_path = lance_metadata.get("path")
        if task_path != self.path:
            msg = f"Lance image read path mismatch: task path={task_path}, reader path={self.path}"
            raise ValueError(msg)
        version = lance_metadata.get("version")
        if version is None:
            msg = f"Lance image read task {task.task_id} is missing a pinned Lance version"
            raise ValueError(msg)
        return int(version)

    def _split_read_kwargs(
        self, read_kwargs: dict[str, Any], version: int, fields: list[str] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset_kwargs = dict(read_kwargs.pop("dataset_options", {}) or {})
        requested_version = dataset_kwargs.pop("version", None)
        requested_version = read_kwargs.pop("version", requested_version)
        if requested_version is not None and int(requested_version) != version:
            msg = f"Lance read version mismatch: task version={version}, requested version={requested_version}"
            raise ValueError(msg)

        storage_options = read_kwargs.pop("storage_options", None)
        if storage_options is not None:
            dataset_kwargs["storage_options"] = storage_options
        if "session" in dataset_kwargs:
            msg = "dataset_options.session is managed by LanceImageSliceReaderStage"
            raise ValueError(msg)
        dataset_kwargs["version"] = version

        scanner_kwargs = dict(read_kwargs.pop("scanner_options", {}) or {})
        scanner_kwargs.update(read_kwargs)
        reserved = sorted({"fragments", "offset", "limit", "filter"} & set(scanner_kwargs))
        if reserved:
            msg = f"scanner options are managed by Lance image slices: {reserved}"
            raise ValueError(msg)
        if fields is not None:
            scanner_kwargs["columns"] = fields
        return dataset_kwargs, scanner_kwargs

    def _open_dataset(self, version: int, dataset_kwargs: dict[str, Any]) -> object:
        import lance

        if self._dataset is None or self._opened_version != version:
            self._session = lance.Session(index_cache_size_bytes=self.index_cache_size_bytes)
            self._dataset = lance.dataset(self.path, session=self._session, **dataset_kwargs)
            self._opened_version = version
        return self._dataset

    def _scan_slice(
        self,
        fragment: object,
        image_slice: LanceImageSlice,
        scanner_kwargs: dict[str, Any],
    ) -> pa.Table:
        import pyarrow as pa

        table = fragment.scanner(  # type: ignore[attr-defined]
            offset=image_slice.row_offset,
            limit=image_slice.row_count,
            **scanner_kwargs,
        ).to_table()
        if self.include_coordinates:
            table = table.append_column(
                LANCE_FRAGID_COLUMN,
                pa.array([image_slice.fragment_id] * table.num_rows, type=pa.uint64()),
            )
            table = table.append_column(
                LANCE_ROW_OFFSET_COLUMN,
                pa.array(range(image_slice.row_offset, image_slice.row_offset + table.num_rows), type=pa.uint64()),
            )
        return table

    def read_task(
        self,
        task: LanceImageReadTask,
        read_kwargs: dict[str, Any] | None,
        fields: list[str] | None,
    ) -> ReaderOutput:
        import pyarrow as pa
        from lance.schema import schema_to_json

        version = self._task_version(task)
        dataset_kwargs, scanner_kwargs = self._split_read_kwargs(dict(read_kwargs or {}), version, fields)
        dataset = self._open_dataset(version, dataset_kwargs)
        fragments = {
            fragment_id: dataset.get_fragment(fragment_id) for fragment_id in {s.fragment_id for s in task.data}
        }  # type: ignore[attr-defined]

        def scan(image_slice: LanceImageSlice) -> pa.Table:
            return self._scan_slice(fragments[image_slice.fragment_id], image_slice, scanner_kwargs)

        if self.reader_threads == 1 or len(task.data) == 1:
            tables = [scan(image_slice) for image_slice in task.data]
        else:
            with ThreadPoolExecutor(max_workers=min(self.reader_threads, len(task.data))) as executor:
                tables = list(executor.map(scan, task.data))

        table = pa.concat_tables(tables)
        metadata = dict(task._metadata)
        lance_metadata = dict(metadata.get("lance") or {})
        lance_metadata["schema"] = schema_to_json(dataset.schema)  # type: ignore[attr-defined]
        metadata["lance"] = lance_metadata
        return ReaderOutput(table, metadata)

    def _output_metadata(self, task: LanceImageReadTask, output: ReaderOutput) -> dict[str, Any]:
        return output.metadata if output.metadata is not None else task._metadata


@dataclass
class LanceImageReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Read encoded images from Lance with fragment-local parallel scans.

    Defaults are based on a 32-CPU object-store benchmark: 100 rows per
    fragment slice, 40 slices per task, and 32 concurrent scanner calls.
    """

    path: str
    rows_per_slice: int = 100
    slices_per_partition: int = 40
    fields: list[str] | None = field(default_factory=lambda: ["image"])
    read_kwargs: dict[str, Any] | None = None
    fragment_ids: list[int] | None = None
    reader_threads: int = 32
    lance_cpu_threads: int = 16
    lance_io_threads: int = 64
    index_cache_size_bytes: int = DEFAULT_INDEX_CACHE_SIZE_BYTES
    include_coordinates: bool = True
    task_type: Literal["document"] = "document"
    name: str = "lance_image_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = {} if self.read_kwargs is None else dict(self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        if self.task_type != "document":
            msg = f"Converting encoded Lance images to {self.task_type} is not supported yet."
            raise NotImplementedError(msg)
        return [
            LanceImageSlicePartitioningStage(
                path=self.path,
                rows_per_slice=self.rows_per_slice,
                slices_per_partition=self.slices_per_partition,
                fragment_ids=self.fragment_ids,
                read_kwargs=self.read_kwargs,
            ),
            LanceImageSliceReaderStage(
                path=self.path,
                fields=self.fields,
                read_kwargs=self.read_kwargs,
                reader_threads=self.reader_threads,
                lance_cpu_threads=self.lance_cpu_threads,
                lance_io_threads=self.lance_io_threads,
                index_cache_size_bytes=self.index_cache_size_bytes,
                include_coordinates=self.include_coordinates,
            ),
        ]
