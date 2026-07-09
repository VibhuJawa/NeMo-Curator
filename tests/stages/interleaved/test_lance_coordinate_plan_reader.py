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

import json
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    CoordinatePlanIdentity,
    lance_coordinate_plan_schema,
    load_coordinate_plan,
    publish_coordinate_plan,
)
from nemo_curator.stages.interleaved.lance_coordinate_plan_reader import LanceCoordinatePlanReader
from nemo_curator.tasks import EmptyTask, FileGroupTask

_DOCUMENT_URI = "s3://document-bucket/documents.lance"
_IMAGE_URI = "s3://image-bucket/images.lance"
_SIDECAR_DIGEST = "a" * 64
_FRAGMENT_DIGEST = "b" * 64


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:  # type: ignore[override]
    """These filesystem-only source tests do not need a Ray cluster."""


def _identity(
    fragment_id: int,
    *,
    sidecar_manifest_sha256: str = _SIDECAR_DIGEST,
) -> CoordinatePlanIdentity:
    return CoordinatePlanIdentity(
        document_uri=_DOCUMENT_URI,
        document_version=1,
        image_uri=_IMAGE_URI,
        image_version=4,
        fragment_id=fragment_id,
        sidecar_manifest_sha256=sidecar_manifest_sha256,
        fragment_manifest_sha256=_FRAGMENT_DIGEST,
    )


def _table(fragment_id: int) -> pa.Table:
    positions = [0, 2, 5]
    return pa.Table.from_arrays(
        [
            pa.array([(fragment_id << 32) | position for position in positions], type=pa.uint64()),
            pa.array(positions, type=pa.uint64()),
            pa.array([fragment_id * 10 + offset for offset in range(len(positions))], type=pa.uint64()),
        ],
        schema=lance_coordinate_plan_schema(),
    )


def _publish(root: Path, fragment_id: int, **identity_kwargs: str):  # noqa: ANN202
    return publish_coordinate_plan(root, _table(fragment_id), _identity(fragment_id, **identity_kwargs))


def _pinned_reader(root: Path, **overrides: object) -> LanceCoordinatePlanReader:
    settings: dict[str, object] = {
        "plan_root": str(root),
        "document_uri": _DOCUMENT_URI,
        "document_version": 1,
        "image_uri": _IMAGE_URI,
        "image_version": 4,
        "sidecar_manifest_sha256": _SIDECAR_DIGEST,
        "fragment_manifest_sha256": _FRAGMENT_DIGEST,
        "missing_key_policy": "error",
    }
    settings.update(overrides)
    return LanceCoordinatePlanReader(**settings)  # type: ignore[arg-type]


def test_reader_is_checkpointable_fanout_and_emits_fragment_order_with_content_ids(tmp_path: Path) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    _publish(root, 9)
    _publish(root, 2)

    reader = _pinned_reader(root)
    tasks = reader.process(EmptyTask())

    assert reader.is_source_stage is True
    assert reader.is_resumable is True
    assert reader.num_workers() == 1
    assert reader.ray_stage_spec()[RayStageSpecKeys.IS_FANOUT_STAGE] is True
    assert [task._metadata["lance_coordinate_plan"]["fragment_id"] for task in tasks] == [2, 9]
    assert all(task.validate() for task in tasks)
    assert all(Path(task.data).is_absolute() and Path(task.manifest_path).is_absolute() for task in tasks)
    assert [task.get_deterministic_id() for task in tasks] == [task.source_identity_sha256 for task in tasks]
    assert all(len(task.get_deterministic_id()) == 64 for task in tasks)
    assert [load_coordinate_plan(task)[0].num_rows for task in tasks] == [3, 3]

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    _publish(relocated, 2)
    _publish(relocated, 9)
    relocated_tasks = _pinned_reader(relocated).process(EmptyTask())
    assert [task.get_deterministic_id() for task in relocated_tasks] == [task.get_deterministic_id() for task in tasks]


def test_reader_allows_an_empty_complete_root(tmp_path: Path) -> None:
    root = tmp_path / "plans"
    root.mkdir()

    assert LanceCoordinatePlanReader(plan_root=str(root)).process(EmptyTask()) == []


@pytest.mark.parametrize("missing", ["manifest", "parquet"])
def test_reader_rejects_partial_pairs(tmp_path: Path, missing: str) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    task = _publish(root, 3)
    Path(task.manifest_path if missing == "manifest" else task.data).unlink()

    with pytest.raises(ValueError, match="partial artifact pairs"):
        LanceCoordinatePlanReader(plan_root=str(root)).process(EmptyTask())


@pytest.mark.parametrize("stray_kind", ["file", "directory", "symlink"])
def test_reader_rejects_stray_entries(tmp_path: Path, stray_kind: str) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    _publish(root, 3)
    if stray_kind == "file":
        (root / "_SUCCESS").write_text("done", encoding="utf-8")
    elif stray_kind == "directory":
        (root / "staging").mkdir()
    else:
        (root / "link").symlink_to(next(root.glob("*.parquet")))

    with pytest.raises(ValueError, match=r"stray|non-regular"):
        LanceCoordinatePlanReader(plan_root=str(root)).process(EmptyTask())


def test_reader_rejects_duplicate_document_fragments(tmp_path: Path) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    _publish(root, 7)
    _publish(root, 7, sidecar_manifest_sha256="c" * 64)

    with pytest.raises(ValueError, match="duplicate document fragment 7"):
        LanceCoordinatePlanReader(plan_root=str(root)).process(EmptyTask())


@pytest.mark.parametrize(
    ("pin", "wrong_value"),
    [
        ("document_uri", "s3://other/documents.lance"),
        ("document_version", 2),
        ("image_uri", "s3://other/images.lance"),
        ("image_version", 5),
        ("sidecar_manifest_sha256", "c" * 64),
        ("fragment_manifest_sha256", "d" * 64),
        ("missing_key_policy", "null"),
    ],
)
def test_reader_rejects_identity_pin_mismatch(tmp_path: Path, pin: str, wrong_value: object) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    _publish(root, 4)

    with pytest.raises(ValueError, match=pin):
        _pinned_reader(root, **{pin: wrong_value}).process(EmptyTask())


def test_reader_runs_full_existing_artifact_validation(tmp_path: Path) -> None:
    root = tmp_path / "plans"
    root.mkdir()
    task = _publish(root, 4)
    manifest_path = Path(task.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coordinates"]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not reconcile"):
        LanceCoordinatePlanReader(plan_root=str(root)).process(EmptyTask())


def test_reader_rejects_relative_missing_symlink_roots_and_wrong_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        LanceCoordinatePlanReader(plan_root="relative/plans")

    missing = LanceCoordinatePlanReader(plan_root=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="existing regular directory"):
        missing.process(EmptyTask())

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="not a symlink"):
        LanceCoordinatePlanReader(plan_root=str(linked)).process(EmptyTask())

    with pytest.raises(TypeError, match="Expected EmptyTask"):
        LanceCoordinatePlanReader(plan_root=str(real)).process(  # type: ignore[arg-type]
            FileGroupTask(dataset_name="wrong", data=["file"])
        )
