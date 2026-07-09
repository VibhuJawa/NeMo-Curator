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

from typing import TYPE_CHECKING
from unittest import mock

import pytest

from nemo_curator.backends.ray_actor_pool.utils import calculate_optimal_actors_for_stage
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved import (
    GpuLanceDocumentMaterializer as PublicGpuLanceDocumentMaterializer,
)
from nemo_curator.stages.interleaved import (
    LanceCoordinatePayloadOverlayStage as PublicLanceCoordinatePayloadOverlayStage,
)
from nemo_curator.stages.interleaved import LanceCoordinatePlanReader as PublicLanceCoordinatePlanReader
from nemo_curator.stages.interleaved import LancePartitioningStage as PublicLancePartitioningStage
from nemo_curator.stages.interleaved import LancePayloadOverlayReader as PublicLancePayloadOverlayReader
from nemo_curator.stages.interleaved.gpu_lance_document import GpuLanceDocumentMaterializer
from nemo_curator.stages.interleaved.gpu_lance_shuffle import GpuLanceShuffleFetchStage
from nemo_curator.stages.interleaved.lance_coordinate_plan_reader import LanceCoordinatePlanReader
from nemo_curator.stages.interleaved.lance_payload_overlay_reader import LancePayloadOverlayReader
from nemo_curator.stages.interleaved.lance_payload_overlay_stage import LanceCoordinatePayloadOverlayStage
from nemo_curator.stages.interleaved.lance_payload_patch_stage import LanceCoordinatePayloadPatchStage
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage

if TYPE_CHECKING:
    from pathlib import Path


def _materializer(tmp_path: Path, **kwargs: object) -> GpuLanceDocumentMaterializer:
    values: dict[str, object] = {
        "document_uri": "s3://documents/dataset",
        "document_version": 3,
        "image_uri": "s3://images/dataset",
        "image_version": 4,
        "index_shards": [str(tmp_path / "sidecar-0.parquet"), str(tmp_path / "sidecar-1.parquet")],
        "index_manifest_uri": str(tmp_path / "sidecar.manifest.json"),
        "index_manifest_sha256": "a" * 64,
        "coordinate_plan_output_path": str(tmp_path / "plans"),
        "output_root": str(tmp_path / "patches"),
        "node_local_spool_root": str(tmp_path / "spool"),
        "materialization_mode": "document_patch",
    }
    values.update(kwargs)
    return GpuLanceDocumentMaterializer(**values)  # type: ignore[arg-type]


def test_materializer_decomposes_to_one_fragment_coordinate_and_patch_graph(tmp_path: Path) -> None:
    stage = _materializer(
        tmp_path,
        fragment_ids=[9, 2, 9],
        document_storage_options={"region": "us-west-2"},
        image_storage_options={"endpoint": "https://object-store"},
        index_storage_options={"anonymous": "false"},
        image_columns={"image": "binary_content"},
        fetch_task_window=64,
        fetch_batch_size=4096,
        max_pending_takes=16,
        payload_window_bytes="4GiB",
        payload_actor_cpus=4,
        payload_patch_workers=3,
    )

    partitioner, resolver, patcher = stage.decompose()
    assert isinstance(partitioner, LancePartitioningStage)
    assert isinstance(resolver, GpuLanceShuffleFetchStage)
    assert isinstance(patcher, LanceCoordinatePayloadPatchStage)

    assert partitioner.path == "s3://documents/dataset"
    assert partitioner.fragments_per_partition == 1
    assert partitioner.fragment_ids == [9, 2, 9]
    assert partitioner.read_kwargs == {
        "version": 3,
        "storage_options": {"region": "us-west-2"},
    }

    assert resolver.document_uri == partitioner.path
    assert resolver.document_version == 3
    assert resolver.image_uri == patcher.image_uri == "s3://images/dataset"
    assert resolver.image_version == patcher.image_version == 4
    assert resolver.coordinate_plan_output_path == str(tmp_path / "plans")
    assert resolver.fetch_task_window == 64
    assert resolver.fetch_batch_size == patcher.fetch_batch_size == 4096
    assert resolver.max_pending_takes == patcher.max_pending == 16
    assert resolver.image_columns == patcher.image_columns == {"image": "binary_content"}
    assert resolver.document_storage_options == patcher.document_storage_options == {"region": "us-west-2"}
    assert resolver.image_storage_options == patcher.image_storage_options == {"endpoint": "https://object-store"}
    assert resolver.index_storage_options == {"anonymous": "false"}
    assert patcher.payload_window_bytes == 4 * 1024**3
    assert patcher.payload_actor_cpus == 4
    assert patcher.resources.cpus == 4.0
    assert patcher.num_workers() == 3
    assert patcher.estimated_payload_actor_reservation_bytes == 12 * 1024**3


def test_default_payload_actor_geometry_admits_eight_actors_on_64_cpu_node(tmp_path: Path) -> None:
    _, _, patcher = _materializer(tmp_path).decompose()

    with mock.patch(
        "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
        return_value=(64.0, 0.0),
    ):
        actors = calculate_optimal_actors_for_stage(patcher, num_tasks=64)

    assert patcher.payload_actor_cpus == 8
    assert patcher.resources.cpus == 8.0
    assert patcher.num_workers() is None
    assert actors == 8


def test_materializer_supports_durable_payload_overlay_without_node_local_scratch(tmp_path: Path) -> None:
    stage = _materializer(
        tmp_path,
        materialization_mode="payload_overlay",
        node_local_spool_root=None,
    )

    _, _, outputter = stage.decompose()

    assert isinstance(outputter, LanceCoordinatePayloadOverlayStage)
    assert outputter.payload_spool_sync_mode == "fsync"
    assert outputter.output_root == tmp_path / "patches"
    assert outputter.node_local_spool_root == outputter.output_root
    assert outputter.batch_size == 64
    assert outputter.coordinate_window_bytes == 4 * 1024**3
    assert outputter.payload_actor_cpus == 64
    assert outputter.resources.cpus == 64.0
    with mock.patch(
        "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
        return_value=(64.0, 0.0),
    ):
        assert calculate_optimal_actors_for_stage(outputter, num_tasks=128) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payload_actor_cpus", 0),
        ("payload_actor_cpus", True),
        ("payload_patch_workers", 0),
        ("payload_patch_workers", True),
    ],
)
def test_materializer_rejects_invalid_payload_actor_geometry(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        _materializer(tmp_path, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("payload_overlay_actor_cpus", 0, "payload_actor_cpus"),
        ("payload_overlay_actor_cpus", True, "payload_actor_cpus"),
        ("payload_overlay_workers", 0, "payload_patch_workers"),
        ("payload_overlay_workers", True, "payload_patch_workers"),
    ],
)
def test_materializer_rejects_invalid_overlay_actor_geometry(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"{message} must be a positive integer"):
        _materializer(
            tmp_path,
            materialization_mode="payload_overlay",
            node_local_spool_root=None,
            **{field: value},
        )


def test_materializer_and_partitioner_are_public_exports() -> None:
    assert PublicGpuLanceDocumentMaterializer is GpuLanceDocumentMaterializer
    assert PublicLanceCoordinatePlanReader is LanceCoordinatePlanReader
    assert PublicLanceCoordinatePayloadOverlayStage is LanceCoordinatePayloadOverlayStage
    assert PublicLancePartitioningStage is LancePartitioningStage
    assert PublicLancePayloadOverlayReader is LancePayloadOverlayReader


def test_materializer_builds_as_public_pipeline_source_and_sink(tmp_path: Path) -> None:
    composite = _materializer(tmp_path)
    pipeline = Pipeline(name="gpu-lance-document", stages=[composite])

    pipeline.build()

    assert [stage.name for stage in pipeline.stages] == [
        "lance_partitioning",
        "gpu_lance_shuffle_fetch",
        "lance_coordinate_payload_patch",
    ]
    assert pipeline.stages[0].is_source_stage is True
    assert pipeline.stages[-1].is_sink_stage is True
    assert pipeline.decomposition_info == {
        "gpu_lance_document_materializer": [
            "lance_partitioning",
            "gpu_lance_shuffle_fetch",
            "lance_coordinate_payload_patch",
        ]
    }


def test_materializer_repr_hides_all_storage_options(tmp_path: Path) -> None:
    stage = _materializer(
        tmp_path,
        document_storage_options={"secret": "document-sentinel"},
        image_storage_options={"secret": "image-sentinel"},
        index_storage_options={"secret": "index-sentinel"},
    )

    rendered = repr(stage)

    assert "document-sentinel" not in rendered
    assert "image-sentinel" not in rendered
    assert "index-sentinel" not in rendered
    assert "s3://documents/dataset" in rendered
    assert "s3://images/dataset" in rendered


def test_materializer_rejects_unknown_mode_or_patch_without_scratch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported materialization_mode"):
        _materializer(tmp_path, materialization_mode="unknown")
    with pytest.raises(ValueError, match="node_local_spool_root is required"):
        _materializer(tmp_path, materialization_mode="document_patch", node_local_spool_root=None)


@pytest.mark.parametrize(
    "field",
    ["coordinate_plan_output_path", "output_root", "node_local_spool_root"],
)
def test_materializer_rejects_relative_artifact_paths(tmp_path: Path, field: str) -> None:
    with pytest.raises(ValueError, match="absolute"):
        _materializer(tmp_path, **{field: "relative/path"})
