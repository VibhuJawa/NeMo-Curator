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

import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    import pandas as pd

    from nemo_curator.backends.base import WorkerMetadata

# Write-time fragment size: 200_000 rows x ~50KB avg = ~10 GB per fragment.
# Larger fragments -> fewer S3 objects -> faster sequential scans.
_LANCEDB_BATCH_SIZE = 200_000


def _add_blob_encoding_metadata(schema: pa.Schema) -> pa.Schema:
    """Return a copy of *schema* where every large_binary field carries the
    ``lance-encoding:blob`` metadata hint required by LanceDB 0.33 to activate
    the efficient blob-storage code path.

    Fields that already carry the metadata are left unchanged.
    """
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
class LanceDBWriter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Appends each DocumentBatch to a LanceDB table (local or S3-compatible object store).

    Supports any S3-compatible endpoint (AWS, PBSS/SwiftStack, MinIO) via storage_options.
    Uses ``exist_ok=True`` on ``create_table`` so the call is idempotent (creates OR opens),
    then always calls ``table.add(..., mode='append')`` — safe for concurrent Ray workers
    writing to the same table.

    LanceDB 0.33 specifics applied:
    - ``lance-encoding:blob`` metadata on large_binary schema fields.
    - ``virtual_hosted_style_request='false'`` and companion perf options injected
      automatically when an S3-compatible (non-AWS) endpoint is configured.
    - ``batch_size=20_000`` passed to internal batching helpers.
    - ``post_load_optimize()`` method for compaction + index creation after a full load.
    """

    uri: str
    table_name: str = "documents"
    schema: pa.Schema | None = None
    storage_options: dict[str, Any] | None = field(default=None)
    name: str = "lancedb_writer"
    # One tbl.add() per block — set to match the upstream chunk_size so each
    # process() call receives exactly one DocumentBatch and writes one fragment.
    # Must be > 1 (batch_size=1 means 1 ROW per call in Ray Data, not 1 block).
    batch_size: int = 5_000
    # 4 CPUs: PyArrow serialisation of blob columns is CPU-bound; more cores
    # reduce the serialise-then-upload latency per fragment.
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0))
    # Cached per-actor state — populated in setup(), used in process(), cleared in teardown()
    _tbl: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Inject blob-encoding metadata for all large_binary fields in the schema.
        if self.schema is not None:
            self.schema = _add_blob_encoding_metadata(self.schema)

        if self.storage_options is None:
            opts: dict = {}

            # S3-compatible endpoint override (e.g. PBSS/SwiftStack, MinIO).
            # AWS_ENDPOINT_URL_S3 being set signals a non-AWS S3-compatible store,
            # so we add path-style and performance options that are specific to PBSS.
            endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")

            if endpoint:
                opts["endpoint"] = endpoint

            if key_id := os.environ.get("AWS_ACCESS_KEY_ID"):
                opts["aws_access_key_id"] = key_id
            if secret := os.environ.get("AWS_SECRET_ACCESS_KEY"):
                opts["aws_secret_access_key"] = secret
            opts["aws_region"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

            # PBSS / path-style S3-compatible extras (not appropriate for real AWS).
            if endpoint:
                opts["virtual_hosted_style_request"] = "false"
                opts["new_table_data_storage_version"] = "stable"
                opts["new_table_enable_v2_manifest_paths"] = "true"
                opts["io_threads"] = "128"

            self.storage_options = opts  # opts always contains at least aws_region

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        """Called once per actor worker — creates and caches the LanceDB connection.

        Follows the NeMo Curator actor-stage pattern (same as ClientPartitioningStage):
        expensive initialisation happens here, process() just uses self._tbl.
        """
        import lancedb  # lazy import — lancedb is an optional dependency

        db = lancedb.connect(self.uri, storage_options=self.storage_options or None)
        self._tbl = db.create_table(
            self.table_name,
            schema=self.schema,
            mode="create",
            exist_ok=True,
        )
        logger.info(f"LanceDBWriter.setup: connected to {self.uri}/{self.table_name}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _to_pyarrow_table(self, df: pd.DataFrame) -> pa.Table:
        """Convert a pandas DataFrame to a PyArrow table respecting self.schema.

        Builds each column explicitly so that large_binary / binary columns are
        constructed via pa.array(..., type=fld.type) rather than inferred by
        from_pandas — which can misidentify bytes objects as pa.binary() when the
        schema declares pa.large_binary(), or fail entirely on mixed-type object cols.
        """
        if self.schema is None:
            return pa.Table.from_pandas(df)

        arrays: dict[str, pa.Array] = {}
        for fld in self.schema:
            if fld.name in df.columns:
                col = df[fld.name]
            else:
                arrays[fld.name] = pa.nulls(len(df), type=fld.type)
                continue

            if pa.types.is_large_binary(fld.type) or pa.types.is_binary(fld.type):
                # pa.array() with an explicit binary type handles bytes/str/None safely;
                # from_pandas would infer the wrong offset width from object-dtype series.
                vals = [
                    v.encode("utf-8") if isinstance(v, str) else (v if isinstance(v, (bytes, bytearray)) else b"")
                    for v in col
                ]
                arrays[fld.name] = pa.array(vals, type=fld.type)
            elif pa.types.is_large_string(fld.type) or pa.types.is_string(fld.type):
                # large_string uses 64-bit offsets — needed for extracted text that can
                # exceed 2 GB per column across rows. Cast explicitly to avoid offset overflow.
                vals = [str(v) if v is not None else "" for v in col]
                arrays[fld.name] = pa.array(vals, type=fld.type)
            else:
                arrays[fld.name] = pa.array(col.tolist(), type=fld.type)

        return pa.table(arrays, schema=self.schema)

    def teardown(self) -> None:
        """Release the LanceDB table reference so pending writes are flushed."""
        self._tbl = None

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self._tbl is None:
            msg = (
                "LanceDBWriter: setup() was not called before process(). "
                "Ensure this stage is run through RayDataExecutor (actor mode)."
            )
            raise RuntimeError(msg)

        df = task.to_pandas()
        pa_table = self._to_pyarrow_table(df)
        self._tbl.add(pa_table, mode="append")

        logger.debug(f"LanceDBWriter: wrote {len(df)} rows → {self.uri}/{self.table_name}")
        return task

    def post_load_optimize(self) -> None:
        """Run compaction and clean up old versions after a full data load.

        Call this once after all workers have finished writing to the table
        (e.g. from a dedicated post-processing step or a Slurm job step).
        Compaction merges many small fragment files into larger ones, which
        dramatically improves subsequent scan performance.

        Recommended index creation follows compaction so that the index covers
        the compacted fragments rather than fragmented pre-compaction files.
        """
        import lancedb  # lazy import

        db = lancedb.connect(self.uri, storage_options=self.storage_options or None)
        tbl = db.open_table(self.table_name)

        logger.info(f"LanceDBWriter.post_load_optimize: compacting {self.uri}/{self.table_name}")
        tbl.compact_files(
            max_bytes_per_file=8_589_934_592,  # 8 GiB per fragment (~160K rows at 50KB/row)
            target_rows_per_fragment=4_000_000,  # 4 M rows (bytes limit wins for blob data)
            materialize_deletions=True,
            defer_index_remap=True,
            num_threads=16,
            batch_size=_LANCEDB_BATCH_SIZE,
        )

        logger.info("LanceDBWriter.post_load_optimize: cleaning up old versions (>1 h)")
        tbl.cleanup_old_versions(older_than=timedelta(hours=1))

        logger.info("LanceDBWriter.post_load_optimize: done")
