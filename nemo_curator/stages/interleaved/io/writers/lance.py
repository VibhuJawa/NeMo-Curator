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

import contextlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger
from pyarrow import ipc

from nemo_curator.stages.interleaved.io.readers.base import BaseInterleavedReader
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

from .base import BaseInterleavedWriter

if TYPE_CHECKING:
    import pandas as pd


def _serialize_schema(schema: pa.Schema) -> str:
    """Serialize a pyarrow schema to a base64-safe string via IPC."""
    sink = pa.BufferOutputStream()
    ipc.new_stream(sink, schema).close()
    return sink.getvalue().to_pybytes().hex()


def _deserialize_schema(encoded: str) -> pa.Schema:
    """Recover a pyarrow schema serialized by ``_serialize_schema``."""
    buf = pa.py_buffer(bytes.fromhex(encoded))
    reader = ipc.open_stream(buf)
    return reader.schema


def _align_table_to_schema(table: pa.Table, target: pa.Schema) -> pa.Table:
    """Pad *table* with null columns and reorder to match *target* exactly.

    Columns in *table* that are absent from *target* are dropped.
    Columns in *target* that are absent from *table* are added as null arrays.
    """
    existing = set(table.schema.names)
    arrays: list[pa.Array] = []
    for field in target:
        if field.name in existing:
            arrays.append(table.column(field.name).cast(field.type, safe=False))
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.table(arrays, schema=target)


@dataclass
class InterleavedLanceFragmentWriterStage(BaseInterleavedWriter):
    """Write interleaved rows as Lance fragments to a shared dataset path.

    Each task writes fragment data files via ``lance.fragment.write_fragments()``.
    After the pipeline finishes, call :func:`commit_lance_fragments` to assemble
    all fragments into a single LanceDB dataset.

    When *lance_schema* is provided, every fragment is padded/reordered to match
    it exactly (missing columns become null arrays).  This prevents the lance
    parallel scanner crash that occurs when fragments have heterogeneous schemas.
    """

    file_extension: str = "lance"
    name: str = "interleaved_lance_fragment_writer"
    lance_schema: pa.Schema | None = None

    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        pass

    def write_data(self, task: InterleavedBatch, file_path: str) -> None:  # noqa: ARG002
        import lance.fragment

        with self._time_metric("materialize_dataframe_total_s"):
            df = self._materialize_dataframe(task)
        df = self._enforce_schema(df)

        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.cast(BaseInterleavedReader.reconcile_schema(table.schema))

        if self.lance_schema is not None:
            table = _align_table_to_schema(table, self.lance_schema)

        with self._time_metric("lance_write_s"):
            fragments = lance.fragment.write_fragments(table, self.path, schema=table.schema)

        task._metadata["lance_fragments"] = [json.dumps(f.to_json()) for f in fragments]
        task._metadata["lance_schema"] = _serialize_schema(table.schema)

    def process(self, task: InterleavedBatch) -> FileGroupTask:
        self.write_data(task, self.path)
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[self.path],
            _metadata={**task._metadata, "format": self.file_extension},
            _stage_perf=task._stage_perf,
        )


def commit_lance_fragments(
    path: str,
    output_tasks: list[FileGroupTask],
    mode: str = "overwrite",
) -> None:
    """Commit lance fragments written by InterleavedLanceFragmentWriterStage into a single dataset.

    Args:
        path: Lance dataset URI (same path passed to the writer stage).
        output_tasks: Output tasks from ``pipeline.run()``.
        mode: ``"overwrite"`` creates a new dataset; ``"append"`` adds to existing.
    """
    import lance

    all_fragment_jsons: list[str] = []
    schemas: list[pa.Schema] = []
    for task in output_tasks:
        all_fragment_jsons.extend(task._metadata.get("lance_fragments", []))
        if "lance_schema" in task._metadata:
            schemas.append(_deserialize_schema(task._metadata["lance_schema"]))

    if not all_fragment_jsons:
        logger.warning("No lance fragments found in output tasks; nothing to commit")
        return

    if not schemas:
        msg = "No lance_schema found in any output task metadata"
        raise ValueError(msg)

    schema = pa.unify_schemas(schemas, promote_options="permissive")

    fragments = [lance.FragmentMetadata.from_json(fj) for fj in all_fragment_jsons]

    existing = None
    with contextlib.suppress(FileNotFoundError, ValueError, OSError):
        existing = lance.dataset(path)

    if mode == "overwrite" or existing is None:
        op = lance.LanceOperation.Overwrite(schema, fragments)
        read_version = 0
    else:
        op = lance.LanceOperation.Append(fragments)
        read_version = existing.version

    lance.LanceDataset.commit(path, op, read_version=read_version)
    ds = lance.dataset(path)
    logger.info("Committed {} fragments -> {} rows at {}", len(fragments), ds.count_rows(), path)
