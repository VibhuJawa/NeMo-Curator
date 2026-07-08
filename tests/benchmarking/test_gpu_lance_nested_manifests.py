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

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from benchmarking.scripts import derive_gpu_lance_scaling_manifests as derivation
from benchmarking.scripts import generate_gpu_lance_saturation_manifest as generator

TEST_GEOMETRIES = (
    derivation.ScalingGeometry("one-node", target_tasks=2, actor_count=1, task_rows=3),
    derivation.ScalingGeometry("two-node", target_tasks=4, actor_count=2, task_rows=3),
    derivation.ScalingGeometry("four-node", target_tasks=6, actor_count=3, task_rows=3),
    derivation.ScalingGeometry("eight-node", target_tasks=8, actor_count=4, task_rows=3),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tasks(count: int, rows: int) -> list[generator.FragmentTask]:
    return [
        generator.FragmentTask(
            fragment_id=100 + task_id // 2,
            source_refs=tuple(f"https://example.test/{task_id}/{row}.jpg" for row in range(rows)),
        )
        for task_id in range(count)
    ]


def _write_master(tmp_path: Path) -> Path:
    master = tmp_path / "master"
    generator.write_manifest(
        _tasks(8, 3),
        master,
        generator.ManifestConfig(
            target_tasks=8,
            actor_count=4,
            task_rows=3,
            document_uri="s3://bucket/documents",
            document_version=7,
            seed=19,
        ),
        storage_option_keys=("endpoint",),
    )
    return master


def test_default_geometry_is_exact_nested_one_two_four_eight_node_weak_scaling() -> None:
    assert [(value.name, value.target_tasks, value.actor_count) for value in derivation.DEFAULT_GEOMETRIES] == [
        ("one-node", 512, 8),
        ("two-node", 1_024, 16),
        ("four-node", 2_048, 32),
        ("eight-node", 4_096, 64),
    ]
    assert {value.tasks_per_actor for value in derivation.DEFAULT_GEOMETRIES} == {64}
    assert [value.target_rows for value in derivation.DEFAULT_GEOMETRIES] == [
        131_072,
        262_144,
        524_288,
        1_048_576,
    ]


def test_nested_family_is_exact_task_prefix_with_recomputed_identities(tmp_path: Path) -> None:
    master = _write_master(tmp_path)
    output = tmp_path / "nested"

    returned = derivation.derive_nested_manifest_family(master, output, TEST_GEOMETRIES)

    assert output.is_dir()
    assert not list(tmp_path.glob(".nested.tmp-*"))
    family = json.loads((output / "family.json").read_text(encoding="utf-8"))
    assert returned == family
    assert family["document"] == {
        "storage_option_keys": ["endpoint"],
        "uri": "s3://bucket/documents",
        "version": 7,
    }
    assert family["seed"] == 19
    master_table = pq.read_table(master / "manifest.parquet")
    validated_master = derivation.validate_manifest(master, TEST_GEOMETRIES[-1])

    for geometry in TEST_GEOMETRIES:
        child = output / geometry.name
        child_table = pq.read_table(child / "manifest.parquet")
        child_metadata = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
        validated_child = derivation.validate_manifest(child, geometry)
        expected_rows = geometry.target_tasks * 3

        assert child_table.to_pydict() == master_table.slice(0, expected_rows).to_pydict()
        assert validated_child.task_digests == validated_master.task_digests[: geometry.target_tasks]
        assert child_metadata["document"] == family["document"]
        assert child_metadata["seed"] == family["seed"]
        assert child_metadata["target_tasks"] == geometry.target_tasks
        assert child_metadata["target_rows"] == expected_rows
        assert child_metadata["actor_count"] == geometry.actor_count
        assert child_metadata["tasks_per_actor"] == geometry.tasks_per_actor
        assert child_metadata["derivation"]["prefix"] == {
            "task_id_start": 0,
            "task_id_stop": geometry.target_tasks,
            "task_sequence_sha256": validated_child.task_sequence_sha256,
            "verified_against_master": True,
        }
        assert child_metadata["derivation"]["master"]["manifest_parquet_sha256"] == _sha256(
            master / "manifest.parquet"
        )
        assert family["manifests"][geometry.name]["manifest_json_sha256"] == _sha256(child / "manifest.json")
        assert family["manifests"][geometry.name]["manifest_parquet_sha256"] == _sha256(child / "manifest.parquet")

        for actor_id in range(geometry.actor_count):
            actor = pq.read_table(child / "actors" / f"actor_{actor_id:03d}.parquet")
            task_ids = sorted(set(actor["left_task_id"].to_pylist()))
            assert task_ids == list(range(actor_id, geometry.target_tasks, geometry.actor_count))


def test_source_file_tamper_fails_before_any_family_is_published(tmp_path: Path) -> None:
    master = _write_master(tmp_path)
    with (master / "actors" / "actor_000.parquet").open("ab") as stream:
        stream.write(b"tampered")
    output = tmp_path / "nested"

    with pytest.raises(ValueError, match="file identity mismatch"):
        derivation.derive_nested_manifest_family(master, output, TEST_GEOMETRIES)

    assert not output.exists()
    assert not list(tmp_path.glob(".nested.tmp-*"))


def test_actor_assignment_tamper_fails_even_with_recomputed_file_hash(tmp_path: Path) -> None:
    master = _write_master(tmp_path)
    actor_zero = master / "actors" / "actor_000.parquet"
    actor_one = master / "actors" / "actor_001.parquet"
    pq.write_table(pq.read_table(actor_one), actor_zero, row_group_size=3)
    metadata_path = master / "manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["actors/actor_000.parquet"] = {
        "bytes": actor_zero.stat().st_size,
        "rows": pq.read_metadata(actor_zero).num_rows,
        "sha256": _sha256(actor_zero),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = tmp_path / "nested"

    with pytest.raises(ValueError, match="exact left_task_id"):
        derivation.derive_nested_manifest_family(master, output, TEST_GEOMETRIES)

    assert not output.exists()
    assert not list(tmp_path.glob(".nested.tmp-*"))


def test_mid_derivation_failure_removes_hidden_partial_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = _write_master(tmp_path)
    output = tmp_path / "nested"
    real_write = derivation.generator.write_manifest
    call_count = 0

    def write_then_fail_second(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        metadata = real_write(*args, **kwargs)
        if call_count == 2:
            msg = "injected derivation failure"
            raise RuntimeError(msg)
        return metadata

    monkeypatch.setattr(derivation.generator, "write_manifest", write_then_fail_second)

    with pytest.raises(RuntimeError, match="injected derivation failure"):
        derivation.derive_nested_manifest_family(master, output, TEST_GEOMETRIES)

    assert call_count == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".nested.tmp-*"))
