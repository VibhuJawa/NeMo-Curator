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

import weakref
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.backends import base as base_module
from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.stages.interleaved import lance_payload_overlay_stage
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    STABLE_ROW_ID,
    CoordinatePlanIdentity,
    LanceCoordinatePlanTask,
    lance_coordinate_plan_schema,
    publish_coordinate_plan,
)
from nemo_curator.stages.interleaved.lance_payload_overlay import validate_lance_payload_overlay
from nemo_curator.stages.interleaved.lance_payload_overlay_stage import LanceCoordinatePayloadOverlayStage
from tests.stages.interleaved import test_lance_payload_patch_stage as patch_stage_tests


class _RecordingPayloadStreamer(patch_stage_tests._FakePayloadStreamer):
    def __init__(self, image: patch_stage_tests._FakeImageDataset) -> None:
        super().__init__(image)
        self.invocations: list[list[int]] = []

    def iter_stable_row_ids(self, values: pa.Array):  # noqa: ANN202
        self.invocations.append([int(value) for value in values.to_pylist()])
        yield from super().iter_stable_row_ids(values)


def _coordinate_task(
    root: Path,
    *,
    positions: list[int],
    stable_ids: list[int | None],
) -> LanceCoordinatePlanTask:
    table = pa.Table.from_arrays(
        [
            pa.array(
                [(patch_stage_tests._FRAGMENT_ID << 32) | position for position in positions],
                type=pa.uint64(),
            ),
            pa.array(positions, type=pa.uint64()),
            pa.array(stable_ids, type=pa.uint64()),
        ],
        schema=lance_coordinate_plan_schema(allow_missing=any(value is None for value in stable_ids)),
    )
    task = publish_coordinate_plan(
        root,
        table,
        CoordinatePlanIdentity(
            document_uri=patch_stage_tests._DOCUMENT_URI,
            document_version=1,
            image_uri=patch_stage_tests._IMAGE_URI,
            image_version=4,
            fragment_id=patch_stage_tests._FRAGMENT_ID,
            sidecar_manifest_sha256="a" * 64,
            fragment_manifest_sha256=patch_stage_tests._IMAGE_FRAGMENT_DIGEST,
        ),
        allow_missing=any(value is None for value in stable_ids),
    )
    task._stage_perf = {"coordinate": float(len(positions))}
    return task


def _stage(
    tmp_path: Path,
) -> tuple[LanceCoordinatePayloadOverlayStage, patch_stage_tests._FakeImageDataset, _RecordingPayloadStreamer]:
    stage, image = patch_stage_tests._overlay_stage(tmp_path)
    reader = _RecordingPayloadStreamer(image)
    stage._payload_streamer = reader
    return stage, image, reader


def _read_output(paths: list[str]) -> pa.Table:
    return patch_stage_tests._read_arrow_parts(paths).sort_by([(DOCUMENT_POSITION, "ascending")])


def test_grouped_overlay_fetches_global_union_once_and_returns_positional_outputs(tmp_path: Path) -> None:
    first = _coordinate_task(
        tmp_path / "coordinates-first",
        positions=[0, 3, 5],
        stable_ids=[1, 2, 1],
    )
    second = _coordinate_task(
        tmp_path / "coordinates-second",
        positions=[1, 4, 6],
        stable_ids=[2, 3, None],
    )
    stage, image, reader = _stage(tmp_path)
    try:
        outputs = stage.process_batch([second, first])
    finally:
        stage.teardown()

    assert reader.invocations == [[1, 2, 3]]
    assert image.requests == [(3,), (1, 2)]
    assert len(outputs) == 2
    second_output = _read_output(outputs[0].data)
    first_output = _read_output(outputs[1].data)
    assert second_output[DOCUMENT_POSITION].to_pylist() == [1, 4]
    assert second_output[STABLE_ROW_ID].to_pylist() == [2, 3]
    assert second_output["image"].to_pylist() == [b"two", b"three"]
    assert first_output[DOCUMENT_POSITION].to_pylist() == [0, 3, 5]
    assert first_output[STABLE_ROW_ID].to_pylist() == [1, 2, 1]
    assert first_output["image"].to_pylist() == [b"one", b"two", b"one"]

    artifacts = [validate_lance_payload_overlay(Path(output.manifest_path).parent) for output in outputs]
    assert artifacts[0].fetch_group == artifacts[1].fetch_group
    fetch_group = artifacts[0].fetch_group
    assert fetch_group is not None
    assert len(fetch_group["member_source_identity_sha256"]) == 2
    metrics = fetch_group["metrics"]
    assert metrics["logical_rows"] == 5
    assert metrics["unique_rows"] == 3
    assert metrics["sum_plan_unique_rows"] == 4
    assert metrics["cross_plan_unique_ids_coalesced"] == 1
    assert metrics["average_physical_read_bytes"] > 0
    assert metrics["physical_reads_per_unique_payload"] > 0
    assert metrics["physical_reads_per_logical_payload"] > 0
    assert metrics["read_amplification"] > 0
    assert outputs[0]._metadata["lance_payload_overlay"]["logical_rows"] == 2
    assert outputs[1]._metadata["lance_payload_overlay"]["logical_rows"] == 3
    for output in outputs:
        overlay_metadata = output._metadata["lance_payload_overlay"]
        assert overlay_metadata["fetch_group"]["fetch_group_sha256"] == fetch_group["fetch_group_sha256"]
        assert "lance_read_iops" not in overlay_metadata["producer_metrics"]
        assert "payload_unique_images_per_second" not in overlay_metadata["producer_metrics"]
        assert overlay_metadata["fetch_group"]["metrics"]["payload_unique_images_per_second"] > 0


def test_grouped_overlay_partial_publish_retry_adopts_finished_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _coordinate_task(tmp_path / "coordinates-first", positions=[0], stable_ids=[1])
    second = _coordinate_task(tmp_path / "coordinates-second", positions=[1], stable_ids=[2])
    stage, _image, reader = _stage(tmp_path)
    real_task_factory = lance_payload_overlay_stage.lance_payload_overlay_task

    def crash_after_first_publish(*_args: object, **_kwargs: object) -> None:
        msg = "synthetic grouped post-publication crash"
        raise RuntimeError(msg)

    monkeypatch.setattr(lance_payload_overlay_stage, "lance_payload_overlay_task", crash_after_first_publish)
    try:
        with pytest.raises(RuntimeError, match="grouped post-publication crash"):
            stage.process_batch([first, second])
        assert reader.invocations == [[1, 2]]
        assert len([path for path in stage.output_root.iterdir() if path.is_dir()]) == 1
        assert not [path for path in stage.output_root.iterdir() if path.name.endswith(".tmp")]

        monkeypatch.setattr(lance_payload_overlay_stage, "lance_payload_overlay_task", real_task_factory)
        outputs = stage.process_batch([first, second])
        invocations_after_retry = list(reader.invocations)
        adopted = [output._metadata["lance_payload_overlay"]["adopted"] for output in outputs]
        outputs_again = stage.process_batch([first, second])
    finally:
        stage.teardown()

    assert len(invocations_after_retry) == 2
    assert len(invocations_after_retry[1]) == 1
    assert invocations_after_retry[0] == [1, 2]
    assert invocations_after_retry[1] in ([1], [2])
    assert sorted(adopted) == [False, True]
    assert reader.invocations == invocations_after_retry
    assert all(output._metadata["lance_payload_overlay"]["adopted"] is True for output in outputs_again)
    assert len([path for path in stage.output_root.iterdir() if path.is_dir()]) == 2


def test_coordinate_window_partitions_pending_members_and_rejects_oversized_singleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _coordinate_task(tmp_path / "coordinates-first", positions=[0], stable_ids=[1])
    second = _coordinate_task(tmp_path / "coordinates-second", positions=[1], stable_ids=[2])
    stage, _image, reader = _stage(tmp_path)
    stage.coordinate_window_bytes = 250
    monkeypatch.setattr(
        lance_payload_overlay_stage,
        "estimate_grouped_coordinate_workspace_bytes",
        lambda plans: 200 * len(plans),
    )
    try:
        outputs = stage.process_batch([first, second])
    finally:
        stage.teardown()

    assert len(outputs) == 2
    assert len(reader.invocations) == 2
    assert sorted(reader.invocations) == [[1], [2]]
    groups = [validate_lance_payload_overlay(Path(output.manifest_path).parent).fetch_group for output in outputs]
    assert all(group is not None and len(group["member_source_identity_sha256"]) == 1 for group in groups)
    for output, group in zip(outputs, groups, strict=True):
        producer_metrics = output._metadata["lance_payload_overlay"]["producer_metrics"]
        assert group is not None
        for name in (
            "lance_read_iops",
            "lance_read_bytes",
            "payload_materialize_seconds",
            "payload_unique_images_per_second",
            "payload_logical_images_per_second",
            "average_physical_read_bytes",
            "physical_reads_per_unique_payload",
            "physical_reads_per_logical_payload",
            "read_amplification",
        ):
            assert name not in producer_metrics
            assert name in group["metrics"]

    oversized_root = tmp_path / "oversized"
    oversized_root.mkdir()
    oversized_stage, _image, oversized_reader = _stage(oversized_root)
    oversized_stage.coordinate_window_bytes = 250
    monkeypatch.setattr(
        lance_payload_overlay_stage,
        "estimate_grouped_coordinate_workspace_bytes",
        lambda _plans: 251,
    )
    third = _coordinate_task(tmp_path / "coordinates-third", positions=[2], stable_ids=[3])
    try:
        with pytest.raises(ValueError, match="coordinate plan workspace requires 251 bytes"):
            oversized_stage.process_batch([third])
    finally:
        oversized_stage.teardown()
    assert oversized_reader.invocations == []
    assert not [path for path in oversized_stage.output_root.iterdir() if path.is_dir()]


def test_grouped_overlay_defaults_to_one_64_plan_actor_and_validates_batch_bounds(tmp_path: Path) -> None:
    stage = LanceCoordinatePayloadOverlayStage(
        image_uri=patch_stage_tests._IMAGE_URI,
        image_version=4,
        output_root=str(tmp_path / "overlays"),
    )

    assert stage.batch_size == 64
    assert stage.payload_actor_cpus == 64
    assert stage.resources.cpus == 64.0
    assert stage.coordinate_window_bytes == 4 * 1024**3
    expected_retained_payload_bytes = (2 * 16 + 1) * 1024 * 128 * 1024
    assert stage.estimated_retained_payload_bytes == expected_retained_payload_bytes
    assert stage.estimated_payload_actor_reservation_bytes == (expected_retained_payload_bytes + 1024**3 + 4 * 1024**3)
    with pytest.raises(ValueError, match="at least one"):
        stage.process_batch([])
    with pytest.raises(ValueError, match="at most 64"):
        stage.process_batch([object()] * 65)  # type: ignore[list-item]

    explicit = LanceCoordinatePayloadOverlayStage(
        image_uri=patch_stage_tests._IMAGE_URI,
        image_version=4,
        output_root=str(tmp_path / "explicit"),
        coordinate_window_bytes="256MiB",
        payload_actor_cpus=8,
        payload_overlay_workers=2,
    )
    assert explicit.coordinate_window_bytes == 256 * 1024**2
    assert explicit.resources.cpus == 8.0
    assert explicit.num_workers() == 2
    with pytest.raises(ValueError, match="coordinate_window_bytes must be"):
        LanceCoordinatePayloadOverlayStage(
            image_uri=patch_stage_tests._IMAGE_URI,
            image_version=4,
            output_root=str(tmp_path / "invalid"),
            coordinate_window_bytes=512,
        )


def test_grouped_overlay_releases_prevalidated_plans_before_loading_active_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _coordinate_task(tmp_path / "coordinates-first", positions=[0], stable_ids=[1])
    second = _coordinate_task(tmp_path / "coordinates-second", positions=[1], stable_ids=[2])
    stage, _image, _reader = _stage(tmp_path)
    real_load = lance_payload_overlay_stage.load_coordinate_plan
    plan_references: list[weakref.ReferenceType[pa.Table]] = []
    live_plans_before_load: list[int] = []

    def recording_load(*args: object, **kwargs: object) -> tuple[pa.Table, dict[str, object]]:
        live_plans_before_load.append(sum(reference() is not None for reference in plan_references))
        table, manifest = real_load(*args, **kwargs)
        plan_references.append(weakref.ref(table))
        return table, manifest

    monkeypatch.setattr(lance_payload_overlay_stage, "load_coordinate_plan", recording_load)
    try:
        outputs = stage.process_batch([first, second])
    finally:
        stage.teardown()

    assert len(outputs) == 2
    assert live_plans_before_load == [0, 0, 0, 1]
    assert all(reference() is None for reference in plan_references)


def test_grouped_overlay_revalidates_entire_active_group_before_creating_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _coordinate_task(tmp_path / "coordinates-first", positions=[0], stable_ids=[1])
    second = _coordinate_task(tmp_path / "coordinates-second", positions=[1], stable_ids=[2])
    stage, _image, reader = _stage(tmp_path)
    real_reload = stage._reload_member_plan
    reload_count = 0

    def fail_second_reload(*args: object, **kwargs: object) -> pa.Table:
        nonlocal reload_count
        reload_count += 1
        if reload_count == 2:
            msg = "synthetic identity drift"
            raise ValueError(msg)
        return real_reload(*args, **kwargs)

    monkeypatch.setattr(stage, "_reload_member_plan", fail_second_reload)
    try:
        with pytest.raises(ValueError, match="synthetic identity drift"):
            stage.process_batch([first, second])
    finally:
        stage.teardown()

    assert reload_count == 2
    assert reader.invocations == []
    assert not [path for path in stage.output_root.iterdir() if path.is_dir()]


def test_base_stage_adapter_preserves_grouped_n_to_n_ancestry_and_checkpoint_deltas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _coordinate_task(tmp_path / "coordinates-first", positions=[0], stable_ids=[1])
    second = _coordinate_task(tmp_path / "coordinates-second", positions=[1], stable_ids=[2])
    inputs = [second, first]
    for index, task in enumerate(inputs):
        task.task_id = f"coordinate-parent-{index}"
        task._source_id = f"source-{index}"
        task._stage_perf = []
    captured_deltas: list[tuple[str, str, int]] = []
    monkeypatch.setattr(base_module, "is_resumability_actor_active", lambda: True)
    monkeypatch.setattr(base_module, "flush_resumability_deltas", captured_deltas.extend)
    monkeypatch.setattr(base_module, "resolve_slurm_array_config", lambda **_kwargs: None)
    stage, _image, _reader = _stage(tmp_path)
    try:
        outputs = BaseStageAdapter(stage).process_batch(inputs)
    finally:
        stage.teardown()

    assert len(outputs) == len(inputs) == 2
    assert [output.task_id for output in outputs] == [f"{task.task_id}_0" for task in inputs]
    assert [output._source_id for output in outputs] == [task._source_id for task in inputs]
    assert captured_deltas == [(outputs[index].task_id, inputs[index]._source_id, 0) for index in range(len(inputs))]
