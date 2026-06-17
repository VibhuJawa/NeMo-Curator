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

"""Lance writer stages for NeMo Curator pipelines.

``LanceFragmentWriterStage``
    Distributed writer built on ``lance_ray.LanceFragmentWriter``.  Each
    parallel actor writes lance fragments directly to object storage with
    no manifest coordination.  After ``pipeline.run()`` returns the fragment
    metadata rows, call ``lance_commit_fragments()`` on the driver to atomically
    commit everything in a single ``LanceDataset.commit()``.

    Example::

        stages = [...processing_stages..., LanceFragmentWriterStage(uri=..., schema=...)]
        tasks  = pipeline.run(RayDataExecutor(), initial_tasks=[EmptyTask(...)])
        lance_commit_fragments(tasks, uri=..., storage_options=...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.writer.utils import retry_with_backoff, s3_storage_options_from_env
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata
    from nemo_curator.tasks import Task


def _add_blob_encoding_metadata(schema: pa.Schema) -> pa.Schema:
    """Return a copy of *schema* with ``lance-encoding:blob`` on every large_binary field."""
    if not any(pa.types.is_large_binary(fld.type) for fld in schema):
        return schema
    new_fields: list[pa.Field] = []
    for fld in schema:
        if pa.types.is_large_binary(fld.type):
            existing_meta: dict = dict(fld.metadata or {})
            blob_key = b"lance-encoding:blob"
            if blob_key not in existing_meta:
                existing_meta[blob_key] = b"true"
            new_fields.append(fld.with_metadata(existing_meta))
        else:
            new_fields.append(fld)
    return pa.schema(new_fields)


@dataclass
class LanceFragmentWriterStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Distributed lance fragment writer using ``lance_ray.LanceFragmentWriter``.

    Each parallel actor writes lance fragment files directly to object storage —
    no manifest coordination between actors.  The output ``DocumentBatch`` rows
    carry pickled ``(fragment_metadata, schema)`` pairs.  Pass the tasks returned
    by ``pipeline.run()`` to ``lance_commit_fragments()`` to atomically commit.

    Requires ``lance-ray`` (``pip install lance-ray``).
    """

    uri: str
    schema: pa.Schema | None = None
    storage_options: dict[str, Any] | None = field(default=None)
    name: str = "lance_fragment_writer"
    max_rows_per_file: int = 500_000
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))

    _writer: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.schema is not None:
            self.schema = _add_blob_encoding_metadata(self.schema)
        if self.storage_options is None:
            self.storage_options = s3_storage_options_from_env()

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        from lance_ray.fragment import LanceFragmentWriter

        self._writer = LanceFragmentWriter(
            uri=self.uri,
            schema=self.schema,
            storage_options=self.storage_options or {},
            max_rows_per_file=self.max_rows_per_file,
        )
        logger.info(f"LanceFragmentWriterStage: ready for {self.uri}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["fragment", "schema"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        meta_table: pa.Table = self._writer(task.to_pyarrow())
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=meta_table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def lance_commit_fragments(
    tasks: list[Task],
    uri: str,
    storage_options: dict[str, Any] | None = None,
    retries: int = 5,
) -> None:
    """Atomically commit lance fragments produced by ``LanceFragmentWriterStage``.

    Appends to an existing dataset or creates one if the URI does not yet exist.
    Safe for concurrent Slurm array jobs: on a manifest version conflict the
    commit is retried up to *retries* times with exponential back-off.

    Args:
        tasks: Output of ``pipeline.run()`` when ``LanceFragmentWriterStage``
            is the last pipeline stage.
        uri: Lance dataset URI (must match the URI used in the stage).
        storage_options: S3/object-store credentials; auto-read from env if None.
        retries: Maximum commit attempts on version conflict (default 5).
    """
    import pickle

    import lance

    if storage_options is None:
        storage_options = s3_storage_options_from_env()

    fragments: list[Any] = []
    schema: pa.Schema | None = None
    for task in tasks:
        if "fragment" not in task.get_columns():
            continue
        df = task.to_pandas()
        if schema is None and len(df) > 0:
            schema = pickle.loads(df["schema"].iloc[0])  # noqa: S301
        for frag_bytes in df["fragment"]:
            fragments.append(pickle.loads(frag_bytes))  # noqa: S301

    if not fragments or schema is None:
        logger.warning("lance_commit_fragments: no fragments to commit — skipping")
        return

    def _commit() -> None:
        try:
            ds = lance.dataset(uri, storage_options=storage_options)
            read_version: int | None = ds.version
            op = lance.LanceOperation.Append(fragments)
        except FileNotFoundError:
            read_version = None
            op = lance.LanceOperation.Overwrite(schema, fragments)
        lance.LanceDataset.commit(uri, op, read_version=read_version, storage_options=storage_options)

    retry_with_backoff(_commit, retries=retries, label="lance_commit_fragments")
    logger.info(f"lance_commit_fragments: committed {len(fragments)} fragments to {uri}")
