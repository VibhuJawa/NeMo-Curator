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

"""Public document-to-image Lance materialization graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.interleaved.gpu_lance_shuffle import (
    FetchWindowBytes,
    GpuLanceShuffleFetchStage,
)
from nemo_curator.stages.interleaved.lance_payload_patch_stage import (
    LanceCoordinatePayloadPatchStage,
    PayloadWindowBytes,
)
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage
from nemo_curator.tasks import EmptyTask, FileGroupTask

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpoolSyncMode


@dataclass
class GpuLanceDocumentMaterializer(CompositeStage[EmptyTask, FileGroupTask]):
    """Resolve image URLs on GPUs and publish patched document fragments.

    The graph keeps payload bytes out of both RAPIDS-MPF shuffles. It emits one
    durable coordinate plan per deletion-free document fragment, then fetches
    image-only payloads directly into an attempt-local Arrow spool before
    reconstructing the original document order.

    Run this composite with ``RayActorPoolExecutor``. The collective coordinate
    phase is intentionally non-resumable; use ``LanceCoordinatePlanReader``
    followed by ``LanceCoordinatePayloadPatchStage`` for a checkpointed phase
    that adopts already-published coordinate plans.
    """

    document_uri: str
    document_version: int
    image_uri: str
    image_version: int
    index_shards: Sequence[str | Sequence[str]] | Mapping[int, str | Sequence[str]]
    index_manifest_uri: str
    index_manifest_sha256: str
    coordinate_plan_output_path: str
    output_root: str
    node_local_spool_root: str
    fragment_ids: Sequence[int] | None = None
    image_columns: Mapping[str, str] | None = None
    document_url_column: str = "source_ref"
    document_filter: str | None = "modality = 'image'"
    document_projection: Sequence[str] | None = None
    index_url_column: str = "url"
    index_stable_row_id_column: str = "stable_row_id"
    document_storage_options: Mapping[str, str] | None = field(default=None, repr=False)
    image_storage_options: Mapping[str, str] | None = field(default=None, repr=False)
    index_storage_options: Mapping[str, str] | None = field(default=None, repr=False)
    existing_column_policy: Literal["error", "fill_null", "overwrite"] = "fill_null"
    missing_key_policy: Literal["error", "null"] = "error"
    scan_batch_size: int = 65_536
    fetch_task_window: int = 8
    fetch_window_bytes: FetchWindowBytes | int = "1GiB"
    payload_window_bytes: PayloadWindowBytes | int = "1GiB"
    bucket_rows: int = 131_072
    payload_spool_sync_mode: PayloadSpoolSyncMode = "attempt_local"
    estimated_payload_bytes_per_row: int = 128 * 1024
    fetch_batch_size: int = 1024
    max_pending_takes: int = 16
    payload_actor_cpus: int = 8
    payload_patch_workers: int | None = None
    rmm_pool_size: int | Literal["auto"] | None = "auto"
    spill_memory_limit: int | Literal["auto"] | None = "auto"
    enable_statistics: bool = False
    name: str = "gpu_lance_document_materializer"
    _partitioner: LancePartitioningStage = field(init=False, repr=False)
    _coordinate_resolver: GpuLanceShuffleFetchStage = field(init=False, repr=False)
    _payload_patcher: LanceCoordinatePayloadPatchStage = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        document_storage_options = dict(self.document_storage_options or {})
        image_storage_options = dict(self.image_storage_options or {})
        index_storage_options = dict(self.index_storage_options or {})
        image_columns = {"image": "binary_content"} if self.image_columns is None else dict(self.image_columns)
        fragment_ids = None if self.fragment_ids is None else list(self.fragment_ids)
        document_projection = None if self.document_projection is None else tuple(self.document_projection)

        self.document_storage_options = document_storage_options
        self.image_storage_options = image_storage_options
        self.index_storage_options = index_storage_options
        self.image_columns = image_columns
        self.fragment_ids = fragment_ids
        self.document_projection = document_projection

        self._partitioner = LancePartitioningStage(
            path=self.document_uri,
            fragments_per_partition=1,
            fragment_ids=fragment_ids,
            read_kwargs={
                "version": self.document_version,
                "storage_options": document_storage_options,
            },
        )
        self._coordinate_resolver = GpuLanceShuffleFetchStage(
            image_uri=self.image_uri,
            image_version=self.image_version,
            index_shards=self.index_shards,
            index_manifest_uri=self.index_manifest_uri,
            index_manifest_sha256=self.index_manifest_sha256,
            image_columns=image_columns,
            document_uri=self.document_uri,
            document_version=self.document_version,
            document_url_column=self.document_url_column,
            document_filter=self.document_filter,
            document_projection=document_projection,
            index_url_column=self.index_url_column,
            index_stable_row_id_column=self.index_stable_row_id_column,
            document_storage_options=document_storage_options,
            image_storage_options=image_storage_options,
            index_storage_options=index_storage_options,
            existing_column_policy=self.existing_column_policy,
            missing_key_policy=self.missing_key_policy,
            scan_batch_size=self.scan_batch_size,
            fetch_task_window=self.fetch_task_window,
            fetch_window_bytes=self.fetch_window_bytes,
            estimated_payload_bytes_per_row=self.estimated_payload_bytes_per_row,
            fetch_batch_size=self.fetch_batch_size,
            max_pending_takes=self.max_pending_takes,
            coordinate_plan_output_path=self.coordinate_plan_output_path,
            rmm_pool_size=self.rmm_pool_size,
            spill_memory_limit=self.spill_memory_limit,
            enable_statistics=self.enable_statistics,
        )
        self._payload_patcher = LanceCoordinatePayloadPatchStage(
            image_uri=self.image_uri,
            image_version=self.image_version,
            output_root=self.output_root,
            node_local_spool_root=self.node_local_spool_root,
            image_storage_options=image_storage_options,
            image_columns=image_columns,
            payload_window_bytes=self.payload_window_bytes,
            bucket_rows=self.bucket_rows,
            payload_spool_sync_mode=self.payload_spool_sync_mode,
            estimated_payload_bytes_per_row=self.estimated_payload_bytes_per_row,
            fetch_batch_size=self.fetch_batch_size,
            max_pending=self.max_pending_takes,
            payload_actor_cpus=self.payload_actor_cpus,
            payload_patch_workers=self.payload_patch_workers,
            document_projection=document_projection,
            document_storage_options=document_storage_options,
            existing_column_policy=self.existing_column_policy,
        )

    def decompose(self) -> list[ProcessingStage[Any, Any]]:
        """Return partition, coordinate-resolution, and payload-patch stages."""
        return [self._partitioner, self._coordinate_resolver, self._payload_patcher]
