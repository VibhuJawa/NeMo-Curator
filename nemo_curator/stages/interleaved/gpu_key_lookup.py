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

"""Persistent GPU exact-key membership for row-wise interleaved batches."""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class _GpuMatchResult:
    matched: np.ndarray
    transfer_seconds: float
    probe_seconds: float
    gather_seconds: float


class _GpuExactKeyMatcher:
    """Own persistent RAPIDS filtered joins for immutable reference segments."""

    def __init__(  # noqa: PLR0915
        self,
        reference_files: Sequence[str],
        reference_key_column: str,
        storage_options: dict[str, str],
        expected_reference_rows: int | None,
        load_factor: float,
    ) -> None:
        try:
            import cudf
            import cupy as cp
            import pylibcudf as plc
            from pylibcudf.join import FilteredJoin
            from pylibcudf.types import NullEquality
        except ImportError as exc:  # pragma: no cover - exercised only without the optional GPU dependency
            msg = "GpuExactKeyLookupStage requires cudf-cu12>=26.6,<26.7"
            raise ImportError(msg) from exc

        self._cp = cp
        self._plc = plc
        self._frames: list[Any] = []
        self._build_tables: list[Any] = []
        self._joins: list[Any] = []
        reference_type: pa.DataType | None = None

        free_before, total_memory = cp.cuda.runtime.memGetInfo()
        load_started = time.perf_counter()
        build_seconds = 0.0
        reference_rows = 0
        for path in reference_files:
            frame = cudf.read_parquet(
                path,
                columns=[reference_key_column],
                storage_options=storage_options or None,
            )
            if len(frame) == 0:
                msg = f"Reference key segment is empty: {path}"
                raise ValueError(msg)
            frame_type = frame[reference_key_column].head(1).to_arrow().type
            if reference_type is None:
                reference_type = frame_type
            elif frame_type != reference_type:
                msg = f"Reference key column has type {frame_type} in {path}; expected {reference_type}"
                raise TypeError(msg)
            if frame[reference_key_column].null_count:
                msg = f"Reference key column {reference_key_column!r} contains nulls in {path}"
                raise ValueError(msg)
            build_table = frame[[reference_key_column]].to_pylibcudf()[0]
            build_started = time.perf_counter()
            join = FilteredJoin(build_table, NullEquality.UNEQUAL, load_factor)
            cp.cuda.runtime.deviceSynchronize()
            build_seconds += time.perf_counter() - build_started
            reference_rows += len(frame)
            # libcudf's filtered_join stores a view of the build table. Retain
            # both owners for the complete lifetime of the join object.
            self._frames.append(frame)
            self._build_tables.append(build_table)
            self._joins.append(join)

        if expected_reference_rows is not None and reference_rows != expected_reference_rows:
            msg = f"Reference sidecars contain {reference_rows} rows; expected {expected_reference_rows}"
            raise ValueError(msg)
        free_after, _ = cp.cuda.runtime.memGetInfo()
        self.reference_rows = reference_rows
        self.load_seconds = time.perf_counter() - load_started - build_seconds
        self.build_seconds = build_seconds
        self.gpu_bytes = free_before - free_after
        self.gpu_total_bytes = total_memory
        if reference_type is None:  # pragma: no cover - reference_files is validated as non-empty
            msg = "GPU reference loading did not discover a key type"
            raise RuntimeError(msg)
        self.reference_type = reference_type

    def match(self, keys: pa.Array) -> _GpuMatchResult:
        if not len(keys):
            return _GpuMatchResult(np.zeros(0, dtype=np.bool_), 0.0, 0.0, 0.0)

        transfer_started = time.perf_counter()
        probe = self._plc.Table([self._plc.Column.from_arrow(keys)])
        self._cp.cuda.runtime.deviceSynchronize()
        transfer_seconds = time.perf_counter() - transfer_started

        probe_started = time.perf_counter()
        gather_maps = [join.semi_join(probe) for join in self._joins]
        self._cp.cuda.runtime.deviceSynchronize()
        probe_seconds = time.perf_counter() - probe_started

        gather_started = time.perf_counter()
        matched = np.zeros(len(keys), dtype=np.bool_)
        for gather_map in gather_maps:
            indices = gather_map.to_arrow().to_numpy(zero_copy_only=False)
            matched[indices] = True
        gather_seconds = time.perf_counter() - gather_started
        return _GpuMatchResult(matched, transfer_seconds, probe_seconds, gather_seconds)

    def close(self) -> None:
        self._joins.clear()
        self._build_tables.clear()
        self._frames.clear()


@dataclass
class GpuExactKeyLookupStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Mark exact key membership using immutable Parquet reference segments on one GPU.

    Each actor loads the reference segments once and builds one persistent
    ``pylibcudf.join.FilteredJoin`` per segment. Multiple input tasks can be
    coalesced with ``ProcessingStage.with_(batch_size=...)`` and are probed as
    one Arrow array while their output boundaries and row order are preserved.
    """

    reference_files: list[str]
    reference_key_column: str
    input_key_column: str = "source_ref"
    presence_column: str = "image_present"
    storage_options: dict[str, str] = field(default_factory=dict)
    expected_reference_rows: int | None = None
    load_factor: float = 0.5
    name: str = "gpu_exact_key_lookup"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))
    _matcher: _GpuExactKeyMatcher | None = field(default=None, init=False, repr=False)
    _setup_metrics_pending: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reference_files = list(self.reference_files)
        self.storage_options = dict(self.storage_options or {})
        if not self.reference_files:
            msg = "reference_files must not be empty"
            raise ValueError(msg)
        if len(set(self.reference_files)) != len(self.reference_files):
            msg = "reference_files must not contain duplicates"
            raise ValueError(msg)
        if not self.reference_key_column or not self.input_key_column or not self.presence_column:
            msg = "reference_key_column, input_key_column, and presence_column must not be empty"
            raise ValueError(msg)
        if self.expected_reference_rows is not None and self.expected_reference_rows <= 0:
            msg = "expected_reference_rows must be greater than zero"
            raise ValueError(msg)
        if not 0.0 < self.load_factor <= 1.0:
            msg = "load_factor must be in the interval (0, 1]"
            raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_key_column]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.presence_column]

    def setup(self, _worker_metadata: object | None = None) -> None:
        self._matcher = _GpuExactKeyMatcher(
            self.reference_files,
            self.reference_key_column,
            self.storage_options,
            self.expected_reference_rows,
            self.load_factor,
        )
        self._setup_metrics_pending = True

    def teardown(self) -> None:
        if self._matcher is not None:
            self._matcher.close()
            self._matcher = None

    def _ensure_matcher(self) -> _GpuExactKeyMatcher:
        if self._matcher is None:
            self.setup()
        if self._matcher is None:  # pragma: no cover - setup returns a matcher or raises
            msg = "GPU exact-key matcher setup did not initialize the worker"
            raise RuntimeError(msg)
        return self._matcher

    def _validate_table(self, table: pa.Table, reference_type: pa.DataType) -> None:
        if self.input_key_column not in table.column_names:
            msg = f"Input key column {self.input_key_column!r} does not exist"
            raise ValueError(msg)
        if self.presence_column in table.column_names:
            msg = f"Presence column {self.presence_column!r} already exists"
            raise ValueError(msg)
        input_type = table.schema.field(self.input_key_column).type
        both_string = (pa.types.is_string(input_type) or pa.types.is_large_string(input_type)) and (
            pa.types.is_string(reference_type) or pa.types.is_large_string(reference_type)
        )
        if input_type != reference_type and not both_string:
            msg = f"Input key column has type {input_type}; reference key column has type {reference_type}"
            raise TypeError(msg)

    @staticmethod
    def _eligible_mask(keys: pa.Array) -> pa.BooleanArray:
        eligible = pc.is_valid(keys)
        if pa.types.is_string(keys.type) or pa.types.is_large_string(keys.type):
            eligible = pc.and_kleene(eligible, pc.not_equal(keys, ""))
        return pc.fill_null(eligible, False)

    def _process_tasks(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        if len(tasks) == 0:
            return []
        matcher = self._ensure_matcher()

        tables = [task.to_pyarrow() for task in tasks]
        for table in tables:
            self._validate_table(table, matcher.reference_type)
        key_arrays = [table[self.input_key_column].combine_chunks() for table in tables]
        combined_keys = pa.concat_arrays(key_arrays)
        eligible = self._eligible_mask(combined_keys)
        eligible_indices = pc.indices_nonzero(eligible)
        eligible_keys = pc.take(combined_keys, eligible_indices)

        match_result = matcher.match(eligible_keys)
        presence_values = np.zeros(len(combined_keys), dtype=np.bool_)
        presence_valid = eligible.to_numpy(zero_copy_only=False)
        presence_values[eligible_indices.to_numpy(zero_copy_only=False)] = match_result.matched
        presence = pa.array(presence_values, mask=~presence_valid, type=pa.bool_())

        outputs: list[InterleavedBatch] = []
        offset = 0
        for task, table in zip(tasks, tables, strict=True):
            task_presence = presence.slice(offset, table.num_rows)
            result = table.append_column(self.presence_column, task_presence)
            outputs.append(
                InterleavedBatch(
                    dataset_name=task.dataset_name,
                    data=result,
                    _metadata=task._metadata,
                    _stage_perf=task._stage_perf,
                )
            )
            offset += table.num_rows

        found = int(match_result.matched.sum())
        metrics = {
            "input_tasks": float(len(tasks)),
            "input_rows": float(len(combined_keys)),
            "eligible_keys": float(len(eligible_keys)),
            "found_keys": float(found),
            "missing_keys": float(len(eligible_keys) - found),
            "gpu_key_transfer_seconds": match_result.transfer_seconds,
            "gpu_key_probe_seconds": match_result.probe_seconds,
            "gpu_result_gather_seconds": match_result.gather_seconds,
            "peak_rss_bytes": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        }
        if self._setup_metrics_pending:
            metrics.update(
                {
                    "reference_rows": float(matcher.reference_rows),
                    "gpu_reference_load_seconds": matcher.load_seconds,
                    "gpu_hash_build_seconds": matcher.build_seconds,
                    "gpu_reference_bytes": float(matcher.gpu_bytes),
                    "gpu_total_bytes": float(matcher.gpu_total_bytes),
                }
            )
            self._setup_metrics_pending = False
        self._log_metrics(metrics)
        return outputs

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        return self._process_tasks([task])[0]

    def process_batch(self, tasks: list[InterleavedBatch]) -> list[InterleavedBatch]:
        """Probe one coalesced key array and preserve task boundaries."""
        return self._process_tasks(tasks)
