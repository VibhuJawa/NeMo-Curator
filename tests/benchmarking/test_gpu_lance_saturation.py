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

import argparse
import io
import json
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from benchmarking.scripts import generate_gpu_lance_saturation_manifest as generator
from benchmarking.scripts import gpu_lance_column_fetch_benchmark as benchmark
from benchmarking.scripts import gpu_lance_saturation_runner as runner
from benchmarking.scripts import gpu_lance_saturation_telemetry as telemetry


def _tasks(count: int, rows: int) -> list[generator.FragmentTask]:
    return [
        generator.FragmentTask(
            fragment_id=100 + task_id // 2,
            source_refs=tuple(f"https://example.test/{task_id}/{row}.jpg" for row in range(rows)),
        )
        for task_id in range(count)
    ]


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)


def _mock_slurm_environment(
    tmp_path: Path,
    *,
    state: str = "RUNNING",
    requeue: str = "0",
    oversubscribe: str = "NO",
) -> dict[str, str]:
    tools = tmp_path / "tools"
    _write_executable(
        tools / "scontrol",
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' 'JobId=123 JobState={state} Requeue={requeue} OverSubscribe={oversubscribe}'\n",
    )
    return {
        "PATH": f"{tools}:/usr/bin:/bin",
        "PYTHON_BIN": sys.executable,
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_NODELIST": "node-a",
        "SLURM_NNODES": "1",
        "SLURM_RESTART_COUNT": "0",
    }


def _explicit_storage_environment() -> dict[str, str]:
    return {
        "IMAGE_LANCE_URI": "s3://bucket/images",
        "IMAGE_LANCE_VERSION": "4",
        "STORAGE_OPTIONS_JSON": '{"endpoint":"https://object.test"}',
        "REFERENCE_GLOB": "/indexes/part-?.parquet",
        "REFERENCE_STORAGE_OPTIONS_JSON": "{}",
        "REFERENCE_MANIFEST_URI": "/indexes/sidecar-v2.json",
        "REFERENCE_MANIFEST_SHA256": "a" * 64,
        "EXPECTED_REFERENCE_ROWS": "1000",
    }


def test_preset_geometry_is_per_actor_weak_scaling() -> None:
    one = generator.PRESETS["one-node"]
    eight = generator.PRESETS["eight-node"]

    assert (one.actor_count, one.target_tasks, one.target_rows) == (8, 512, 131_072)
    assert (eight.actor_count, eight.target_tasks, eight.target_rows) == (64, 4_096, 1_048_576)


def test_saturation_manifest_rejects_credential_bearing_document_identity() -> None:
    dummy_uri = "s3://dummy-user:dummy-pass@bucket/documents?dummy-token=value#dummy-fragment"

    with pytest.raises(ValueError, match="userinfo") as raised:
        generator.ManifestConfig(
            target_tasks=4,
            actor_count=2,
            task_rows=3,
            document_uri=dummy_uri,
            document_version=7,
            seed=19,
        )

    assert "dummy-pass" not in str(raised.value)


def test_manifest_publish_is_atomic_balanced_and_deterministic(tmp_path: Path) -> None:
    config = generator.ManifestConfig(
        target_tasks=4,
        actor_count=2,
        task_rows=3,
        document_uri="s3://bucket/documents",
        document_version=7,
        seed=19,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_metadata = generator.write_manifest(_tasks(4, 3), first, config, storage_option_keys=("endpoint",))
    second_metadata = generator.write_manifest(_tasks(4, 3), second, config, storage_option_keys=("endpoint",))

    assert first_metadata["target_rows"] == 12
    assert first_metadata["tasks_per_actor"] == 2
    assert (
        first_metadata["files"]["manifest.parquet"]["sha256"] == second_metadata["files"]["manifest.parquet"]["sha256"]
    )
    combined = pq.read_table(first / "manifest.parquet")
    assert combined.column_names == [
        "source_ref",
        "left_task_id",
        "source_fragment_id",
        "source_position",
        "source_position_in_task",
    ]
    assert combined.num_rows == 12
    for task_id in range(4):
        task = combined.filter(pc.equal(combined["left_task_id"], task_id))
        assert task.num_rows == 3
        assert task["source_position"].to_pylist() == [0, 1, 2]
        assert task["source_position_in_task"].to_pylist() == [0, 1, 2]
        assert len(set(task["source_fragment_id"].to_pylist())) == 1

    actor_zero = pq.read_table(first / "actors/actor_000.parquet")
    actor_one = pq.read_table(first / "actors/actor_001.parquet")
    assert sorted(set(actor_zero["left_task_id"].to_pylist())) == [0, 2]
    assert sorted(set(actor_one["left_task_id"].to_pylist())) == [1, 3]

    geometry = SimpleNamespace(
        task_rows=3,
        target_tasks=4,
        target_rows=12,
        actor_count=2,
        tasks_per_actor=2,
    )
    runner.load_manifest_metadata(first, geometry)
    with (first / "actors/actor_000.parquet").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="file identity mismatch"):
        runner.load_manifest_metadata(first, geometry)

    second_metadata_path = second / "manifest.json"
    second_metadata = json.loads(second_metadata_path.read_text(encoding="utf-8"))
    second_metadata["document"]["version"] = 8
    second_metadata_path.write_text(json.dumps(second_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="schema source identity mismatch"):
        runner.load_manifest_metadata(second, geometry)


def test_manifest_failure_does_not_publish_partial_directory(tmp_path: Path) -> None:
    config = generator.ManifestConfig(
        target_tasks=4,
        actor_count=2,
        task_rows=3,
        document_uri="s3://bucket/documents",
        document_version=7,
        seed=19,
    )
    output = tmp_path / "broken"

    with pytest.raises(RuntimeError, match="source produced 3 complete tasks"):
        generator.write_manifest(_tasks(3, 3), output, config)

    assert not output.exists()
    assert not list(tmp_path.glob(".broken.tmp-*"))


@pytest.mark.parametrize(
    ("preset_name", "nodes", "expected"),
    [
        ("two-node", 2, (16, 1_024, 262_144)),
        ("four-node", 4, (32, 2_048, 524_288)),
    ],
)
def test_intermediate_node_presets_preserve_per_actor_weak_scaling(
    preset_name: str,
    nodes: int,
    expected: tuple[int, int, int],
) -> None:
    preset = generator.PRESETS[preset_name]

    assert preset.name == preset_name
    assert preset.nodes == nodes
    assert (preset.actor_count, preset.target_tasks, preset.target_rows) == expected


@pytest.mark.parametrize(
    ("nodes", "waves", "expected"),
    [
        (1, 1, (8, 512, 131_072, 64, 16_384, 8, "locality_sensitivity")),
        (1, 2, (8, 512, 131_072, 32, 8_192, 16, "locality_sensitivity")),
        (1, 8, (8, 512, 131_072, 8, 2_048, 64, "primary_saturation")),
        (2, 8, (16, 1_024, 262_144, 8, 2_048, 128, "primary_saturation")),
        (4, 8, (32, 2_048, 524_288, 8, 2_048, 256, "primary_saturation")),
        (8, 4, (64, 4_096, 1_048_576, 16, 4_096, 256, "primary_saturation")),
    ],
)
def test_runner_geometry(nodes: int, waves: int, expected: tuple[object, ...]) -> None:
    geometry = runner.SaturationGeometry(nodes=nodes, waves=waves)

    assert (
        geometry.actor_count,
        geometry.target_tasks,
        geometry.target_rows,
        geometry.coalesce_tasks,
        geometry.actor_batch_rows,
        geometry.expected_actor_calls,
        geometry.evidence_class,
    ) == expected


def test_runner_parser_accepts_only_supported_wave_counts(tmp_path: Path) -> None:
    assert runner.SUPPORTED_WAVES == (1, 2, 4, 8)
    arguments = [
        "--manifest-dir",
        str(tmp_path / "manifest"),
        "--output-dir",
        str(tmp_path / "output"),
        "--nodes",
        "1",
        "--image-lance-uri",
        "s3://bucket/images",
        "--reference-manifest-uri",
        "/indexes/manifest.json",
        "--reference-manifest-sha256",
        "a" * 64,
        "--reference-glob",
        "/indexes/*.parquet",
        "--expected-reference-rows",
        "1",
    ]

    for waves in runner.SUPPORTED_WAVES:
        parsed = runner.build_parser().parse_args([*arguments, "--waves", str(waves)])
        assert parsed.waves == waves
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([*arguments, "--waves", "3"])

    credentialed = list(arguments)
    credentialed[credentialed.index("s3://bucket/images")] = (
        "s3://dummy-user:dummy-pass@bucket/images?dummy-token=value#dummy-fragment"
    )
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(credentialed)


def test_runner_identity_matches_benchmark_manifest_and_uri_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.table(
            {
                "source_ref": ["https://example.test/0.jpg", "https://example.test/1.jpg"],
                "expected_md5": ["a" * 32, "b" * 32],
            }
        ),
        manifest_path,
    )
    manifest = benchmark._load_manifest(
        SimpleNamespace(
            query_manifest=manifest_path,
            max_queries=None,
            expected_md5_column=None,
            expected_width_column=None,
            expected_height_column=None,
        )
    )

    assert runner._query_manifest_digest(manifest_path) == manifest.digest
    credentialed_uri = "s3://user:password@bucket.example/path?token=secret#fragment"
    assert runner._redact_uri_for_identity(credentialed_uri) == benchmark._redact_uri_for_report(credentialed_uri)


def test_scaling_launcher_requires_and_forwards_sidecar_manifest(tmp_path: Path) -> None:
    scripts = Path(__file__).resolve().parents[2] / "benchmarking/scripts"
    rank_script = (scripts / "run_gpu_lance_scaling_rank.sh").read_text(encoding="utf-8")
    assert '--reference-manifest-uri "${REFERENCE_MANIFEST_URI}"' in rank_script
    assert '--reference-manifest-sha256 "${REFERENCE_MANIFEST_SHA256}"' in rank_script
    manifest = tmp_path / "manifests/shards_1/rank_00.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    environment = _mock_slurm_environment(tmp_path)
    environment.update(_explicit_storage_environment())
    environment.pop("REFERENCE_MANIFEST_URI")
    environment.pop("REFERENCE_MANIFEST_SHA256")
    environment.update(
        {
            "SCALE_NODES": "1",
            "RUN_ID": "preflight",
            "GPUS_PER_TASK": "1",
            "BENCHMARK_ARM": "gpu_lance_column_fetch_stage",
            "MANIFEST_ROOT": str(tmp_path / "manifests"),
            "OUTPUT_ROOT": str(output_root),
            "LOG_ROOT": str(log_root),
        }
    )

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(scripts / "run_gpu_lance_scaling_job.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert "REFERENCE_MANIFEST_URI is required" in completed.stderr
    assert not output_root.exists()
    assert not log_root.exists()


@pytest.mark.parametrize(
    "script_name",
    [
        "run_gpu_lance_saturation_job.sh",
        "run_gpu_lance_scaling_job.sh",
        "run_gpu_lance_scaling_rank.sh",
    ],
)
@pytest.mark.parametrize(
    ("guard_environment", "message"),
    [
        ({"SLURM_ARRAY_JOB_ID": "1234", "SLURM_ARRAY_TASK_ID": "0"}, "must not run as Slurm array elements"),
        ({"SLURM_RESTART_COUNT": "1"}, "do not resume requeued jobs"),
    ],
)
def test_scaling_launchers_reject_unsafe_scheduler_context_before_creating_output(
    tmp_path: Path,
    script_name: str,
    guard_environment: dict[str, str],
    message: str,
) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts" / script_name
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "SCALE_NODES": "1",
            "RUN_ID": "array-preflight",
            "LOG_ROOT": str(log_root),
            "OUTPUT_ROOT": str(output_root),
            **guard_environment,
        },
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert not output_root.exists()
    assert not log_root.exists()


@pytest.mark.parametrize(
    "script_name",
    [
        "run_gpu_lance_saturation_job.sh",
        "run_gpu_lance_scaling_job.sh",
        "run_gpu_lance_scaling_rank.sh",
    ],
)
@pytest.mark.parametrize(
    ("slurm_fields", "message"),
    [
        ({"state": "PENDING"}, "must be RUNNING"),
        ({"requeue": "1"}, "submitted with --no-requeue"),
        ({"oversubscribe": "OK"}, "OverSubscribe=NO"),
    ],
)
def test_launchers_reject_unsafe_live_allocation_before_creating_output(
    tmp_path: Path,
    script_name: str,
    slurm_fields: dict[str, str],
    message: str,
) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts" / script_name
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    environment = _mock_slurm_environment(tmp_path, **slurm_fields)
    environment.update(
        {
            "OUTPUT_ROOT": str(output_root),
            "LOG_ROOT": str(log_root),
        }
    )

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert not output_root.exists()
    assert not log_root.exists()


@pytest.mark.parametrize(
    "script_name",
    [
        "run_gpu_lance_saturation_job.sh",
        "run_gpu_lance_scaling_job.sh",
        "run_gpu_lance_scaling_rank.sh",
    ],
)
def test_launchers_require_slurm_before_creating_output(tmp_path: Path, script_name: str) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts" / script_name
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "OUTPUT_ROOT": str(output_root),
            "LOG_ROOT": str(log_root),
        },
    )

    assert completed.returncode == 2
    assert "SLURM_JOB_ID is required" in completed.stderr
    assert not output_root.exists()
    assert not log_root.exists()


def test_scaling_gpu_arm_requires_gpu_before_creating_output(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_scaling_job.sh"
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    environment = _mock_slurm_environment(tmp_path)
    environment.update(
        {
            "OUTPUT_ROOT": str(output_root),
            "LOG_ROOT": str(log_root),
            "BENCHMARK_ARM": "gpu_lance_column_fetch_stage",
        }
    )

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "GPUS_PER_TASK must be positive" in completed.stderr
    assert not output_root.exists()
    assert not log_root.exists()


def test_scaling_launcher_forwards_explicit_configuration(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_scaling_job.sh"
    manifest = tmp_path / "manifests/shards_1/rank_00.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    capture = tmp_path / "srun-args.txt"
    environment_capture = tmp_path / "srun-environment.txt"
    environment = _mock_slurm_environment(tmp_path)
    _write_executable(
        tmp_path / "tools/srun",
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$@\" > {capture!s}\n"
        f'printf \'%s\\n\' "${{IMAGE_LANCE_URI}}" "${{RAY_ADDRESS-unset}}" > {environment_capture!s}\n',
    )
    environment.update(_explicit_storage_environment())
    environment.update(
        {
            "MANIFEST_ROOT": str(tmp_path / "manifests"),
            "OUTPUT_ROOT": str(output_root),
            "LOG_ROOT": str(log_root),
            "RUN_ID": "explicit-config",
            "SCALE_NODES": "1",
            "GPUS_PER_TASK": "1",
            "BENCHMARK_ARM": "gpu_lance_column_fetch_stage",
            "RAY_ADDRESS": "stale.example:6379",
        }
    )

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "--gpus-per-task=1" in arguments
    assert arguments[-2:] == ["bash", str(script.with_name("run_gpu_lance_scaling_rank.sh"))]
    assert environment_capture.read_text(encoding="utf-8").splitlines() == ["s3://bucket/images", "unset"]
    assert (log_root / "explicit-config").is_dir()
    assert not output_root.exists()


def test_saturation_launcher_forwards_time_guard_without_creating_output(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").touch()
    (manifest_dir / "manifest.parquet").touch()
    capture = tmp_path / "srun-args.txt"
    output_root = tmp_path / "output"
    environment = _mock_slurm_environment(tmp_path)
    _write_executable(
        tmp_path / "tools/srun",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {capture!s}\n",
    )
    end_epoch = int(time.time()) + 7_200
    environment.update(_explicit_storage_environment())
    environment.update(
        {
            "MANIFEST_DIR": str(manifest_dir),
            "OUTPUT_ROOT": str(output_root),
            "RUN_ID": "time-guard",
            "MINIMUM_REMAINING_SLURM_SECONDS": "3600",
            "ALLOCATION_END_EPOCH": str(end_epoch),
        }
    )

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "--minimum-remaining-slurm-seconds" in arguments
    assert "3600" in arguments
    assert "--allocation-end-epoch" in arguments
    assert str(end_epoch) in arguments
    assert "--image-lance-uri" in arguments
    assert "s3://bucket/images" in arguments
    assert not output_root.exists()


def test_saturation_rejects_secret_storage_options_without_echoing_values(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").touch()
    (manifest_dir / "manifest.parquet").touch()
    output_root = tmp_path / "output"
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": sys.executable,
        "DRY_RUN": "1",
        "NODES": "1",
        "MANIFEST_DIR": str(manifest_dir),
        "OUTPUT_ROOT": str(output_root),
        **_explicit_storage_environment(),
        "STORAGE_OPTIONS_JSON": '{"secret_access_key":"do-not-echo"}',
    }

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "credential-like keys" in completed.stderr
    assert "do-not-echo" not in completed.stderr
    assert not output_root.exists()


@pytest.mark.parametrize("field", ["IMAGE_LANCE_URI", "REFERENCE_MANIFEST_URI", "REFERENCE_GLOB"])
def test_saturation_launcher_rejects_credential_bearing_uri_before_exec(tmp_path: Path, field: str) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").touch()
    (manifest_dir / "manifest.parquet").touch()
    output_root = tmp_path / "output"
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": sys.executable,
        "DRY_RUN": "1",
        "NODES": "1",
        "MANIFEST_DIR": str(manifest_dir),
        "OUTPUT_ROOT": str(output_root),
        **_explicit_storage_environment(),
        field: "s3://dummy-user:dummy-pass@bucket/path?dummy-token=value#dummy-fragment",
    }

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "credential-free URI validation" in completed.stderr
    assert "dummy-pass" not in completed.stderr
    assert "dummy-token" not in completed.stderr
    assert not output_root.exists()


@pytest.mark.parametrize("waves", runner.SUPPORTED_WAVES)
def test_saturation_dry_run_remains_portable_without_slurm(tmp_path: Path, waves: int) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").touch()
    (manifest_dir / "manifest.parquet").touch()
    capture = tmp_path / "python-args.txt"
    fake_python = tmp_path / "tools/python"
    _write_executable(
        fake_python,
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == \'-c\' ]]; then exec {sys.executable!s} "$@"; fi\n'
        f"printf '%s\\n' \"$@\" > {capture!s}\n",
    )
    environment = {
        "PATH": f"{fake_python.parent}:/usr/bin:/bin",
        "PYTHON_BIN": str(fake_python),
        "DRY_RUN": "1",
        "NODES": "1",
        "WAVES": str(waves),
        "MANIFEST_DIR": str(manifest_dir),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        **_explicit_storage_environment(),
    }

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "--dry-run" in arguments
    assert "--minimum-remaining-slurm-seconds" not in arguments
    waves_index = arguments.index("--waves")
    assert arguments[waves_index + 1] == str(waves)
    assert not (tmp_path / "output").exists()


def test_saturation_launcher_rejects_unsupported_wave_count() -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": sys.executable,
        "DRY_RUN": "1",
        "NODES": "1",
        "WAVES": "3",
    }

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "WAVES must be 1, 2, 4, or 8" in completed.stderr


def test_saturation_launcher_rejects_unsupported_node_count() -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHON_BIN": sys.executable,
        "DRY_RUN": "1",
        "NODES": "3",
    }

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "NODES must be 1, 2, 4, or 8" in completed.stderr


@pytest.mark.parametrize("nodes", [2, 4, 8])
def test_saturation_launcher_dry_run_supports_multinode_presets(tmp_path: Path, nodes: int) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").touch()
    (manifest_dir / "manifest.parquet").touch()
    capture = tmp_path / "python-args.txt"
    fake_python = tmp_path / "tools/python"
    _write_executable(
        fake_python,
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == \'-c\' ]]; then exec {sys.executable!s} "$@"; fi\n'
        f"printf '%s\\n' \"$@\" > {capture!s}\n",
    )
    environment = {
        "PATH": f"{fake_python.parent}:/usr/bin:/bin",
        "PYTHON_BIN": str(fake_python),
        "DRY_RUN": "1",
        "NODES": str(nodes),
        "MANIFEST_DIR": str(manifest_dir),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        **_explicit_storage_environment(),
    }

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    nodes_index = arguments.index("--nodes")
    assert arguments[nodes_index + 1] == str(nodes)
    assert "--dry-run" in arguments
    assert "--minimum-remaining-slurm-seconds" not in arguments
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("python_bin_kind", ["absolute", "relative", "bare"])
def test_saturation_launcher_exposes_path_valued_python_tools(
    tmp_path: Path,
    python_bin_kind: str,
) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_saturation_job.sh"
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").touch()
    (manifest_dir / "manifest.parquet").touch()
    capture = tmp_path / "tool-path.txt"
    tools = tmp_path / "venv/bin"
    fake_python = tools / "python"
    fake_ray = tools / "ray"
    _write_executable(
        fake_python,
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == \'-c\' ]]; then exec {sys.executable!s} "$@"; fi\n'
        f'printf \'%s\\n\' "$(command -v ray)" "$PATH" > {capture!s}\n',
    )
    _write_executable(fake_ray, "#!/usr/bin/env bash\nexit 0\n")

    base_path = "/usr/bin:/bin"
    if python_bin_kind == "absolute":
        python_bin = str(fake_python)
        initial_path = base_path
    elif python_bin_kind == "relative":
        python_bin = str(fake_python.relative_to(tmp_path))
        initial_path = base_path
    else:
        python_bin = "python"
        initial_path = f"{tools}:{base_path}"
    environment = {
        "PATH": initial_path,
        "PYTHON_BIN": python_bin,
        "DRY_RUN": "1",
        "NODES": "1",
        "MANIFEST_DIR": str(manifest_dir),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        **_explicit_storage_environment(),
    }

    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    resolved_ray, observed_path = capture.read_text(encoding="utf-8").splitlines()
    assert resolved_ray == str(fake_ray)
    expected_path = initial_path if python_bin_kind == "bare" else f"{tools}:{initial_path}"
    assert observed_path == expected_path
    assert not (tmp_path / "output").exists()


def test_scaling_rank_refuses_to_overwrite_existing_result(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "benchmarking/scripts/run_gpu_lance_scaling_rank.sh"
    manifest = tmp_path / "manifests/shards_1/rank_00.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    output = tmp_path / "output/cpu_lance_column_fetch_stage/1_nodes_1_ranks/existing/rank_00.json"
    output.parent.mkdir(parents=True)
    output.write_text("preserve me\n", encoding="utf-8")

    environment = _mock_slurm_environment(tmp_path)
    environment.update(_explicit_storage_environment())
    environment.update(
        {
            "MANIFEST_ROOT": str(tmp_path / "manifests"),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "RUN_ID": "existing",
            "SCALE_NODES": "1",
            "SCALE_RANKS": "1",
            "SCALE_RANK": "0",
            "SLURM_NTASKS": "1",
            "SLURM_PROCID": "0",
            "BENCHMARK_ARM": "cpu_lance_column_fetch_stage",
        }
    )
    completed = subprocess.run(  # noqa: S603 - fixed executable and repository-owned script
        ["/usr/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "refusing to overwrite" in completed.stderr
    assert output.read_text(encoding="utf-8") == "preserve me\n"


def test_launchers_contain_no_personal_paths_or_credential_loading() -> None:
    scripts = Path(__file__).resolve().parents[2] / "benchmarking/scripts"
    forbidden = (
        "/home/",
        "/lustre/",
        "pdx.s8k.io",
        "mm-nemo-curator",
        ".config/datamover",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "WORKSPACE_ROOT",
        "lance-ray-pr",
    )
    for name in (
        "run_gpu_lance_saturation_job.sh",
        "run_gpu_lance_scaling_job.sh",
        "run_gpu_lance_scaling_rank.sh",
    ):
        source = (scripts / name).read_text(encoding="utf-8")
        assert not any(value in source for value in forbidden)


def _runner_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest_dir=tmp_path / "manifest",
        output_dir=tmp_path / "output",
        image_lance_uri="s3://bucket/images",
        image_lance_version=4,
        storage_options_json='{"endpoint":"https://object.test","secret_access_key":"do-not-leak"}',
        reference_storage_options_json='{"access_key_id":"also-secret"}',
        reference_manifest_uri="/indexes/sidecar-manifest.json",
        reference_manifest_sha256="a" * 64,
        expected_reference_rows=10,
        payload_projection="image_only",
        fetch_batch_size=1024,
        max_lookup_bytes_mib=256,
        max_pending_fetch_batches=4,
        io_threads_per_actor=4,
        actor_warmup_rows=128,
        warmup_count=1,
        repeat_count=3,
        arm="lance_ray_gpu_actor",
        reference_glob=["/indexes/*.parquet"],
        copy_reference_to_node_local=False,
        reference_node_local_root="/local/index",
        telemetry_interval_seconds=5.0,
        storage_axis="remote_s3",
        filesystem_path=[tmp_path],
        minimum_remaining_slurm_seconds=None,
        allocation_end_epoch=None,
        allocation_time_guard=None,
    )


def test_sanitized_command_never_contains_json_values(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.image_lance_uri = "s3://dummy-user:dummy-pass@bucket.example/images?dummy-token=value#dummy-fragment"
    args.reference_manifest_uri = (
        "https://dummy-user:dummy-pass@index.example/manifest.json?dummy-token=value#dummy-fragment"
    )
    args.reference_glob = ["s3://dummy-user:dummy-pass@index.example/*.parquet?dummy-token=value#dummy-fragment"]
    command = runner.build_benchmark_command(
        args,
        runner.SaturationGeometry(nodes=1, waves=8),
        ray_address="10.0.0.1:6379",
        report_path=tmp_path / "report.json",
    )

    rendered = runner.sanitized_command(command)

    assert "do-not-leak" not in rendered
    assert "also-secret" not in rendered
    assert "dummy-pass" not in rendered
    assert "dummy-token" not in rendered
    assert "s3://bucket.example/images" in rendered
    assert "https://index.example/manifest.json" in rendered
    assert "keys=endpoint,secret_access_key" in rendered
    assert "keys=access_key_id" in rendered
    assert "--ray-gpu-actors 8" in rendered
    assert "--coalesce-tasks 8" in rendered
    assert "--fetch-batch-size 1024" in rendered
    assert "--reference-manifest-sha256 " + "a" * 64 in rendered
    assert "--md5-column ''" in rendered
    assert "--validate-payload-keys" not in rendered
    assert "--row-id-layout" not in rendered
    assert "--evidence-class primary_saturation" in rendered


@pytest.mark.parametrize(
    ("waves", "expected_evidence_class"),
    [(1, "locality_sensitivity"), (8, "primary_saturation")],
)
def test_run_identity_records_evidence_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    waves: int,
    expected_evidence_class: str,
) -> None:
    args = _runner_args(tmp_path)
    geometry = runner.SaturationGeometry(nodes=1, waves=waves)
    monkeypatch.setattr(runner, "_query_manifest_digest", lambda _path: "c" * 64)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=1))

    assert runner._run_head(args, geometry, "10.0.0.1:6379") == 1

    identity = json.loads((args.output_dir / "run_identity.json").read_text(encoding="utf-8"))
    assert identity["schema_version"] == 2
    assert identity["evidence_class"] == expected_evidence_class


def test_storage_options_reject_command_line_credentials() -> None:
    for validator in (generator._reject_secret_storage_options, runner._reject_secret_storage_options):
        with pytest.raises(ValueError, match="process environment"):
            validator({"endpoint": "https://object.test", "secret_access_key": "do-not-leak"})


def test_remaining_slurm_time_guard_rejects_late_launch_and_accepts_portable_inputs() -> None:
    with pytest.raises(ValueError, match="minimum-remaining-slurm-seconds"):
        runner.validate_remaining_slurm_time(
            minimum_remaining_seconds=None,
            allocation_end_epoch=2_000.0,
            now_epoch=1_000.0,
        )
    with pytest.raises(RuntimeError, match="has 1500s remaining"):
        runner.validate_remaining_slurm_time(
            minimum_remaining_seconds=1_800,
            allocation_end_epoch=2_500.0,
            now_epoch=1_000.0,
        )

    guard = runner.validate_remaining_slurm_time(
        minimum_remaining_seconds=1_800,
        allocation_end_epoch=3_000.0,
        now_epoch=1_000.0,
    )
    assert guard["remaining_seconds"] == 2_000


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (None, "cluster_setup"),
        ({"arms": {}}, "benchmark_setup"),
        ({"arms": {"arm": {"status": "pending", "cold_setup": None}}}, "benchmark_setup"),
        (
            {"arms": {"arm": {"status": "ready", "cold_setup": {}, "warmups": [], "repeats": []}}},
            "warmup_0",
        ),
        (
            {"arms": {"arm": {"status": "ready", "cold_setup": {}, "warmups": [{}], "repeats": []}}},
            "steady_repeat_0",
        ),
        (
            {
                "arms": {
                    "arm": {
                        "status": "completed",
                        "cold_setup": {},
                        "warmups": [{}],
                        "repeats": [{}, {}, {}],
                    }
                }
            },
            "complete",
        ),
    ],
)
def test_telemetry_phase_classification(report: dict[str, object] | None, expected: str) -> None:
    assert telemetry.derive_phase(report, "arm", warmups=1, repeats=3) == expected


def _telemetry_sample(node_id: int, hostname: str, phase: str, monotonic: float) -> dict[str, object]:
    return {
        "schema_version": 2,
        "record_type": "sample",
        "timestamp_epoch": 1_700_000_000.0 + monotonic,
        "monotonic_seconds": monotonic,
        "node_id": node_id,
        "hostname": hostname,
        "phase": phase,
        "cpu": {"logical_cpus": 64, "busy_percent": 50.0},
        "gpus": [
            {"index": gpu, "utilization.gpu": 75.0, "memory.used": 40_000, "memory.total": 80_000} for gpu in range(8)
        ],
        "network": {
            "eth0": {
                "receive_bytes": 100 + int(monotonic) * 10,
                "transmit_bytes": 20 + int(monotonic),
            }
        },
        "block_devices": {
            "nvme0n1": {
                "reads_completed": 10 + int(monotonic),
                "sectors_read": 100 + int(monotonic) * 8,
            }
        },
        "filesystems": {"/data": {"total_bytes": 1_000_000, "free_bytes": 500_000}},
        "errors": [],
    }


def _write_telemetry(
    path: Path,
    *,
    node_id: int,
    hostname: str,
    status: str = "complete",
    samples: list[dict[str, object]] | None = None,
) -> None:
    samples = (
        samples
        if samples is not None
        else [
            _telemetry_sample(node_id, hostname, "steady_repeat_0", 0.0),
            _telemetry_sample(node_id, hostname, "steady_repeat_0", 1.0),
            _telemetry_sample(node_id, hostname, "steady_repeat_1", 2.0),
            _telemetry_sample(node_id, hostname, "steady_repeat_1", 3.0),
            _telemetry_sample(node_id, hostname, "complete", 4.0),
        ]
    )
    phases = Counter(str(sample["phase"]) for sample in samples)
    last_monotonic = float(samples[-1]["monotonic_seconds"]) if samples else None
    summary = {
        "schema_version": 2,
        "record_type": "summary",
        "status": status,
        "node_id": node_id,
        "hostname": hostname,
        "sample_count": len(samples),
        "phase_counts": dict(sorted(phases.items())),
        "started_timestamp_epoch": 1_700_000_000.0,
        "finished_timestamp_epoch": 1_700_000_006.0,
        "duration_seconds": 6.0,
        "last_sample_monotonic_seconds": last_monotonic,
        "interval_seconds": 5.0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in [*samples, summary]), encoding="utf-8")


def _completed_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "arms": {"lance_ray_gpu_actor": {"status": "completed", "repeats": [{"status": "completed"}]}},
            }
        ),
        encoding="utf-8",
    )


def _artifact_spec(
    *, node_id: int = 0, hostname: str = "node-a", repeat_count: int = 0
) -> runner.TelemetryValidationSpec:
    return runner.TelemetryValidationSpec(
        node_id=node_id,
        hostname=hostname,
        gpu_count=8,
        interval_seconds=5.0,
        required_steady_repeat_count=repeat_count,
        storage_axis="remote_s3",
    )


def test_telemetry_cluster_accepts_one_complete_stream_per_node(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    report_path = tmp_path / "benchmark.json"
    _completed_report(report_path)
    _write_telemetry(telemetry_dir / "node_0000.jsonl", node_id=0, hostname="node-a")
    _write_telemetry(telemetry_dir / "node_0001.jsonl", node_id=1, hostname="node-b")

    result = runner.validate_telemetry_cluster(
        telemetry_dir,
        runner.TelemetryClusterSpec(
            nodes={0: "node-a", 1: "node-b"},
            report_path=report_path,
            arm="lance_ray_gpu_actor",
            gpu_count=8,
            interval_seconds=5.0,
            wait_seconds=0,
            storage_axis="remote_s3",
            repeat_count=2,
        ),
    )

    assert result["status"] == "passed"
    assert result["required_steady_state_coverage"] is True
    assert result["required_steady_repeat_count"] == 2
    assert {node["sample_count"] for node in result["nodes"].values()} == {5}


def test_telemetry_complete_only_never_satisfies_steady_coverage(tmp_path: Path) -> None:
    path = tmp_path / "node_0000.jsonl"
    samples = [
        _telemetry_sample(0, "node-a", "complete", 0.0),
        _telemetry_sample(0, "node-a", "complete", 1.0),
    ]
    _write_telemetry(path, node_id=0, hostname="node-a", samples=samples)

    result = runner.validate_telemetry_artifact(path, _artifact_spec(repeat_count=1))

    assert result["status"] == "failed"
    assert result["steady_state_observed"] is False
    assert result["network_receive_bytes_delta"] == 0
    assert any("undersampled phases" in failure for failure in result["failures"])


def test_telemetry_deltas_use_only_samples_within_steady_repeats(tmp_path: Path) -> None:
    path = tmp_path / "node_0000.jsonl"
    samples = [
        _telemetry_sample(0, "node-a", "cluster_setup", 0.0),
        _telemetry_sample(0, "node-a", "steady_repeat_0", 1.0),
        _telemetry_sample(0, "node-a", "steady_repeat_0", 2.0),
        _telemetry_sample(0, "node-a", "complete", 3.0),
    ]
    samples[0]["network"]["eth0"]["receive_bytes"] = 0
    samples[1]["network"]["eth0"]["receive_bytes"] = 10_000
    samples[2]["network"]["eth0"]["receive_bytes"] = 10_010
    samples[3]["network"]["eth0"]["receive_bytes"] = 50_000
    _write_telemetry(path, node_id=0, hostname="node-a", samples=samples)

    result = runner.validate_telemetry_artifact(path, _artifact_spec(repeat_count=1))

    assert result["status"] == "passed"
    assert result["steady_delta_sample_count"] == 2
    assert result["network_receive_bytes_delta"] == 10
    assert result["network_receive_bytes_delta_by_phase"] == {"steady_repeat_0": 10}
    assert result["block_read_sectors_delta"] == 8


def test_telemetry_requires_non_loopback_storage_activity_in_every_repeat(tmp_path: Path) -> None:
    path = tmp_path / "node_0000.jsonl"
    samples = [
        _telemetry_sample(0, "node-a", "steady_repeat_0", 0.0),
        _telemetry_sample(0, "node-a", "steady_repeat_0", 1.0),
        _telemetry_sample(0, "node-a", "steady_repeat_1", 2.0),
        _telemetry_sample(0, "node-a", "steady_repeat_1", 3.0),
    ]
    samples[1]["network"]["eth0"] = dict(samples[0]["network"]["eth0"])
    samples[0]["network"]["lo"] = {"receive_bytes": 0, "transmit_bytes": 0}
    samples[1]["network"]["lo"] = {"receive_bytes": 1_000_000, "transmit_bytes": 1_000_000}
    _write_telemetry(path, node_id=0, hostname="node-a", samples=samples)

    result = runner.validate_telemetry_artifact(path, _artifact_spec(repeat_count=2))

    assert result["status"] == "failed"
    assert result["network_receive_bytes_delta_by_phase"]["steady_repeat_0"] == 0
    assert result["network_receive_bytes_delta_by_phase"]["steady_repeat_1"] == 10
    assert any("non-loopback receive-byte delta in steady_repeat_0" in failure for failure in result["failures"])


def test_telemetry_rejects_missing_or_static_storage_counters(tmp_path: Path) -> None:
    path = tmp_path / "node_0000.jsonl"
    samples = [
        _telemetry_sample(0, "node-a", "steady_repeat_0", 0.0),
        _telemetry_sample(0, "node-a", "complete", 5.0),
    ]
    samples[0]["block_devices"] = {}
    _write_telemetry(path, node_id=0, hostname="node-a", samples=samples)
    missing = runner.validate_telemetry_artifact(path, _artifact_spec(repeat_count=1))
    assert missing["status"] == "failed"
    assert missing["plausible_sample_count"] == 1

    samples[0]["block_devices"] = samples[1]["block_devices"]
    samples[1]["phase"] = "steady_repeat_0"
    samples[1]["network"] = samples[0]["network"]
    _write_telemetry(path, node_id=0, hostname="node-a", samples=samples)
    static = runner.validate_telemetry_artifact(path, _artifact_spec(repeat_count=1))
    assert static["status"] == "failed"
    assert any("no positive non-loopback receive-byte delta" in failure for failure in static["failures"])


def test_stop_telemetry_rejects_nonzero_collector_process(tmp_path: Path) -> None:
    class FailedProcess:
        returncode = 17

        def poll(self) -> int:
            return self.returncode

    handle = runner.TelemetryHandle(
        process=FailedProcess(),  # type: ignore[arg-type]
        log_stream=io.StringIO(),
        output=tmp_path / "node_0000.jsonl",
        node_id=0,
        hostname="node-a",
    )

    result = runner._stop_telemetry(handle)

    assert result["status"] == "failed"
    assert result["returncode"] == 17
    assert result["failures"] == ["telemetry collector exited with code 17"]


def test_telemetry_cluster_rejects_missing_node_artifact(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    _write_telemetry(telemetry_dir / "node_0000.jsonl", node_id=0, hostname="node-a")

    result = runner.validate_telemetry_cluster(
        telemetry_dir,
        runner.TelemetryClusterSpec(
            nodes={0: "node-a", 1: "node-b"},
            report_path=tmp_path / "missing-report.json",
            arm="lance_ray_gpu_actor",
            gpu_count=8,
            interval_seconds=5.0,
            wait_seconds=0,
            storage_axis="remote_s3",
            repeat_count=2,
        ),
    )

    assert result["status"] == "failed"
    assert result["missing"] == ["node_0001.jsonl"]
    assert any("missing telemetry artifact" in failure for failure in result["failures"])


def test_telemetry_artifact_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "node_0000.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    result = runner.validate_telemetry_artifact(path, _artifact_spec())

    assert result["status"] == "failed"
    assert any("invalid telemetry JSONL" in failure for failure in result["failures"])


def test_telemetry_artifact_rejects_incomplete_summary(tmp_path: Path) -> None:
    path = tmp_path / "node_0000.jsonl"
    _write_telemetry(path, node_id=0, hostname="node-a", status="incomplete")

    result = runner.validate_telemetry_artifact(path, _artifact_spec(repeat_count=2))

    assert result["status"] == "failed"
    assert "telemetry summary status is 'incomplete'; expected 'complete'" in result["failures"]


def test_telemetry_artifact_rejects_zero_samples(tmp_path: Path) -> None:
    path = tmp_path / "node_0000.jsonl"
    _write_telemetry(path, node_id=0, hostname="node-a", status="incomplete", samples=[])

    result = runner.validate_telemetry_artifact(path, _artifact_spec())

    assert result["status"] == "failed"
    assert "telemetry stream contains zero samples" in result["failures"]


def test_collector_zero_sample_request_is_incomplete_and_nonzero(tmp_path: Path) -> None:
    output = tmp_path / "telemetry.jsonl"
    args = argparse.Namespace(
        output=output,
        benchmark_report=None,
        arm="lance_ray_gpu_actor",
        warmup_count=1,
        repeat_count=3,
        interval_seconds=5.0,
        node_id=0,
        expected_hostname=socket.gethostname().split(".", maxsplit=1)[0],
        filesystem_path=[],
        sample_count=0,
    )

    assert telemetry.run(args) == 1
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"
    assert summary["sample_count"] == 0


def test_collector_publishes_complete_terminal_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hostname = socket.gethostname().split(".", maxsplit=1)[0]
    output = tmp_path / "telemetry.jsonl"
    sample = _telemetry_sample(0, hostname, "complete", 0.0)
    monkeypatch.setattr(telemetry, "_sample", lambda _config, _previous: (sample, {}))
    args = argparse.Namespace(
        output=output,
        benchmark_report=None,
        arm="lance_ray_gpu_actor",
        warmup_count=1,
        repeat_count=3,
        interval_seconds=5.0,
        node_id=0,
        expected_hostname=hostname,
        filesystem_path=[],
        sample_count=1,
    )

    assert telemetry.run(args) == 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["record_type"] for record in records] == ["sample", "summary"]
    assert records[-1]["status"] == "complete"
    assert records[-1]["sample_count"] == 1


def test_finalize_telemetry_raises_on_invalid_stream(tmp_path: Path) -> None:
    class PassedProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

    output = tmp_path / "telemetry" / "node_0001.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("{bad-json}\n", encoding="utf-8")
    handle = runner.TelemetryHandle(
        process=PassedProcess(),  # type: ignore[arg-type]
        log_stream=io.StringIO(),
        output=output,
        node_id=1,
        hostname="node-b",
    )
    context = runner.TelemetryRunContext(
        node_id=1,
        hostname="node-b",
        nodes={0: "node-a", 1: "node-b"},
        report_path=tmp_path / "benchmark.json",
        output_dir=tmp_path,
        arm="lance_ray_gpu_actor",
        interval_seconds=5.0,
        storage_axis="remote_s3",
        geometry=runner.SaturationGeometry(nodes=2, waves=8),
        warmup_count=1,
        repeat_count=3,
    )

    with pytest.raises(RuntimeError, match="telemetry validation failed"):
        runner._finalize_telemetry(handle, context)

    validation = json.loads((tmp_path / "telemetry/node_0001.validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "failed"


def _valid_saturation_repeat(geometry: runner.SaturationGeometry) -> dict[str, Any]:
    payload_calls = 128
    payload_bytes = 1024**2
    process_seconds = 10.0
    read_iops = 1_000
    read_bytes = 1_000_000
    fetched_bytes = 500_000
    return {
        "status": "completed",
        "wall_seconds": 12.0,
        "warm_process_seconds": process_seconds,
        "cold_setup_seconds": 0.0,
        "internal_warmup_seconds": 0.0,
        "images_per_second": geometry.target_rows / process_seconds,
        "payload_mib_per_second": payload_bytes / (1024**2 * process_seconds),
        "payload_bytes": payload_bytes,
        "correctness": {
            "correct": True,
            "output_digest_sha256": "a" * 64,
            "payload_bytes": payload_bytes,
        },
        "fetch_calls": geometry.expected_actor_calls,
        "lance_read_iops": read_iops,
        "lance_read_bytes": read_bytes,
        "backend_metrics": {
            "process_seconds": process_seconds,
            "ray_gpu_actors_used": geometry.actor_count,
            "ray_input_blocks": geometry.target_tasks,
            "lance_fetched_bytes": fetched_bytes,
            "average_physical_read_bytes": read_bytes / read_iops,
            "read_amplification": read_bytes / fetched_bytes,
            "payload_take_calls": payload_calls,
            "payload_take_rows": geometry.target_rows,
            "rows_per_payload_take": geometry.target_rows / payload_calls,
            "sparse_calls_avoided": geometry.target_rows - payload_calls,
            "take_rows_calls": payload_calls,
            "take_scan_calls": 0,
            "strategy_sparse_fragments": 512,
            "strategy_range_fragments": 0,
            "strategy_sequential_fragments": 0,
            "planned_scan_rows": 0,
            "range_overread_rows": 0,
            "found_unique_keys": geometry.target_rows,
            "duplicate_queries_coalesced": 0,
            "fragment_take_calls": 0,
        },
    }


def _valid_saturation_report(
    geometry: runner.SaturationGeometry,
    *,
    repeat_count: int = 2,
    warmup_count: int = 1,
) -> dict[str, Any]:
    repeat = _valid_saturation_repeat(geometry)
    warmup = {
        "status": "completed",
        "wall_seconds": 11.0,
        "correctness": {"correct": True, "output_digest_sha256": "a" * 64},
    }
    sidecar_uri = "/indexes/manifest.json"
    sidecar_sha256 = "b" * 64
    return {
        "status": "completed",
        "evidence_class": geometry.evidence_class,
        "environment": {
            "python": "3.12.0",
            "platform": "linux-test",
            "packages": {
                "nemo-curator": "1.3.0+test",
                "lance-ray": "0.5.0",
                "pyarrow": "22.0.0",
                "pylance": "9.0.0b11",
                "ray": "2.55.1",
            },
        },
        "manifest": {"digest_sha256": "c" * 64},
        "dataset": {"uri": "s3://bucket/dataset", "version": 4, "source_columns": {"image": "image"}},
        "configuration": {
            "repeat_count": repeat_count,
            "warmup_count": warmup_count,
            "payload_read_mode": "sparse",
            "io_threads": 4,
            "max_lookup_bytes": 256 * 1024**2,
            "max_pending_fetch_batches": 16,
            "take_scan_batch_readahead": 16,
            "copy_index_to_node_local": False,
            "index_mirror": None,
            "validate_payload_keys": False,
            "reference_manifest_uri": sidecar_uri,
            "reference_manifest_sha256": sidecar_sha256,
            "ray_actor_pool_size": geometry.actor_count,
            "ray_actor_input_blocks": geometry.target_tasks,
            "ray_actor_input_block_rows": geometry.task_rows,
            "ray_actor_coalesce_tasks": geometry.coalesce_tasks,
            "ray_actor_target_batch_rows": geometry.actor_batch_rows,
            "rows_per_coalesced_fetch": geometry.actor_batch_rows,
        },
        "arms": {
            "lance_ray_gpu_actor": {
                "status": "completed",
                "cold_setup": {
                    "wall_seconds": 5.0,
                    "backend_metrics": {
                        "persistent_actor_pool": True,
                        "persistent_actor_count": geometry.actor_count,
                    },
                },
                "warmups": [json.loads(json.dumps(warmup)) for _ in range(warmup_count)],
                "repeats": [json.loads(json.dumps(repeat)) for _ in range(repeat_count)],
            }
        },
    }


def _valid_run_identity(geometry: runner.SaturationGeometry, *, repeat_count: int = 2) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "evidence_class": geometry.evidence_class,
        "geometry": {
            "nodes": geometry.nodes,
            "actors_per_node": geometry.actors_per_node,
            "actor_count": geometry.actor_count,
            "tasks_per_actor": geometry.tasks_per_actor,
            "task_rows": geometry.task_rows,
            "target_rows": geometry.target_rows,
            "waves": geometry.waves,
            "coalesce_tasks": geometry.coalesce_tasks,
            "actor_batch_rows": geometry.actor_batch_rows,
            "payload_projection": "image_only",
        },
        "dataset": {"uri": "s3://bucket/dataset", "version": 4},
        "manifest": {"digest_sha256": "c" * 64},
        "reference_manifest_uri": "/indexes/manifest.json",
        "reference_manifest_sha256": "b" * 64,
        "slurm_job_id": None,
        "benchmark_policy": {
            "arm": "lance_ray_gpu_actor",
            "repeat_count": repeat_count,
            "warmup_count": 1,
            "payload_read_mode": "sparse",
            "io_threads_per_actor": 4,
            "max_pending_fetch_batches": 16,
            "validate_payload_keys": False,
            "copy_reference_to_node_local": False,
        },
    }


def _validate_report(tmp_path: Path, report: dict[str, Any], *, repeat_count: int = 2) -> dict[str, Any]:
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return runner.validate_benchmark_report(
        report_path,
        "lance_ray_gpu_actor",
        runner.SaturationGeometry(nodes=1, waves=8),
        repeat_count,
        1,
    )


def test_report_validation_requires_all_actors_and_waves(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry, repeat_count=3)

    passed = _validate_report(tmp_path, report, repeat_count=3)
    assert passed["status"] == "passed"
    assert passed["observed"]["ray_actor_stage_windows"] == [64, 64, 64]
    assert passed["observed"]["private_take_calls"] == [128, 128, 128]
    assert passed["observed"]["fragment_strategy_calls"] == [0, 0, 0]
    assert passed["observed"]["physical_io_tracker_reads"] == [1_000, 1_000, 1_000]

    report["arms"]["lance_ray_gpu_actor"]["repeats"][0]["backend_metrics"]["ray_gpu_actors_used"] = 7
    failed = _validate_report(tmp_path, report, repeat_count=3)
    assert failed["status"] == "failed"
    assert any("used 7 actors; expected 8" in failure for failure in failed["failures"])


def test_report_validation_rejects_nonadditive_io_and_sparse_metrics(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    metrics = report["arms"]["lance_ray_gpu_actor"]["repeats"][0]["backend_metrics"]
    metrics.update(
        average_physical_read_bytes=999,
        read_amplification=999,
        rows_per_payload_take=1,
        sparse_calls_avoided=1,
    )

    result = _validate_report(tmp_path, report)

    assert result["status"] == "failed"
    assert any("average_physical_read_bytes" in failure for failure in result["failures"])
    assert any("read_amplification" in failure for failure in result["failures"])
    assert any("sparse_calls_avoided" in failure for failure in result["failures"])


@pytest.mark.parametrize(
    ("field", "value", "failure_text"),
    [
        ("wall_seconds", 9.0, "exceeds wall_seconds"),
        ("cold_setup_seconds", 1.0, "expected 0 for steady-state timing"),
        ("internal_warmup_seconds", 1.0, "expected 0 for steady-state timing"),
        ("images_per_second", 1.0, "images_per_second"),
        ("payload_mib_per_second", 1.0, "payload_mib_per_second"),
    ],
)
def test_report_validation_rejects_invalid_steady_timing(
    tmp_path: Path, field: str, value: float, failure_text: str
) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    report["arms"]["lance_ray_gpu_actor"]["repeats"][0][field] = value

    result = _validate_report(tmp_path, report)

    assert result["status"] == "failed"
    assert any(failure_text in failure for failure in result["failures"])


def test_report_validation_rejects_backend_process_timing_drift(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    report["arms"]["lance_ray_gpu_actor"]["repeats"][0]["backend_metrics"]["process_seconds"] = 9.0

    result = _validate_report(tmp_path, report)

    assert result["status"] == "failed"
    assert any("warm_process_seconds" in failure for failure in result["failures"])


def test_report_validation_requires_warmup_digest_and_persistent_pool(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    report["configuration"]["warmup_count"] = 0
    report["arms"]["lance_ray_gpu_actor"]["warmups"][0]["status"] = "failed"
    report["arms"]["lance_ray_gpu_actor"]["warmups"][0]["correctness"]["output_digest_sha256"] = "d" * 64
    setup = report["arms"]["lance_ray_gpu_actor"]["cold_setup"]["backend_metrics"]
    setup["persistent_actor_pool"] = False
    setup["persistent_actor_count"] = 7

    result = _validate_report(tmp_path, report)

    assert result["status"] == "failed"
    assert any("configuration warmup_count" in failure for failure in result["failures"])
    assert any("warmup 0 did not pass correctness" in failure for failure in result["failures"])
    assert any("warmup and repeat correctness digests" in failure for failure in result["failures"])
    assert any("persistent actor pool" in failure for failure in result["failures"])
    assert any("persistent_actor_count" in failure for failure in result["failures"])

    missing_warmup = _valid_saturation_report(geometry)
    missing_warmup["arms"]["lance_ray_gpu_actor"]["warmups"] = []
    missing_result = _validate_report(tmp_path, missing_warmup)
    assert any("warmup count is 0; expected 1" in failure for failure in missing_result["failures"])

    invalid_digest = _valid_saturation_report(geometry)
    whitespace_digest = "a" * 62 + "  "
    invalid_digest["arms"]["lance_ray_gpu_actor"]["warmups"][0]["correctness"]["output_digest_sha256"] = (
        whitespace_digest
    )
    for repeat in invalid_digest["arms"]["lance_ray_gpu_actor"]["repeats"]:
        repeat["correctness"]["output_digest_sha256"] = whitespace_digest
    digest_result = _validate_report(tmp_path, invalid_digest)
    assert any("warmup 0 correctness digest is missing or invalid" in failure for failure in digest_result["failures"])
    assert any(
        "repeat correctness digests are missing or unstable" in failure for failure in digest_result["failures"]
    )

    truthy_correctness = _valid_saturation_report(geometry)
    truthy_correctness["arms"]["lance_ray_gpu_actor"]["warmups"][0]["correctness"]["correct"] = 1
    truthy_correctness["arms"]["lance_ray_gpu_actor"]["repeats"][0]["correctness"]["correct"] = "false"
    correctness_result = _validate_report(tmp_path, truthy_correctness)
    assert any("warmup 0 did not pass correctness" in failure for failure in correctness_result["failures"])
    assert any("repeat 0 did not pass correctness" in failure for failure in correctness_result["failures"])


def test_report_validation_preserves_supported_ray_data_actor_contract(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    arm_result = report["arms"].pop("lance_ray_gpu_actor")
    arm_result["cold_setup"]["backend_metrics"] = {}
    for repeat in arm_result["repeats"]:
        repeat["cold_setup_seconds"] = 2.0
        repeat["internal_warmup_seconds"] = 1.0
    report["arms"]["ray_data_persistent_gpu_actor"] = arm_result
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = runner.validate_benchmark_report(
        report_path,
        "ray_data_persistent_gpu_actor",
        geometry,
        2,
        1,
    )

    assert result["status"] == "passed"


def _write_terminal_inputs(tmp_path: Path, report: dict[str, Any], identity: dict[str, Any]) -> None:
    (tmp_path / "benchmark.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (tmp_path / "telemetry_validation.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")


def test_terminal_eligibility_requires_benchmark_and_telemetry_pass(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    identity = _valid_run_identity(geometry)
    _write_terminal_inputs(tmp_path, report, identity)

    eligible = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )
    assert eligible["status"] == "eligible"
    assert eligible["schema_version"] == 2
    assert eligible["evidence_class"] == "primary_saturation"
    assert eligible["policy"]["evidence_class"] == "primary_saturation"
    assert eligible["benchmark_validation"]["evidence_class"] == "primary_saturation"

    legacy_identity = json.loads(json.dumps(identity))
    legacy_identity["schema_version"] = 1
    legacy_identity.pop("evidence_class")
    _write_terminal_inputs(tmp_path, report, legacy_identity)
    legacy_ineligible = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )
    assert legacy_ineligible["status"] == "ineligible"
    assert any("schema_version is 1; expected 2" in failure for failure in legacy_ineligible["failures"])

    report["status"] = "running"
    report["arms"]["lance_ray_gpu_actor"]["status"] = "ready"
    report["arms"]["lance_ray_gpu_actor"]["repeats"] = report["arms"]["lance_ray_gpu_actor"]["repeats"][:1]
    _write_terminal_inputs(tmp_path, report, identity)
    ineligible = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )
    assert ineligible["status"] == "ineligible"
    assert ineligible["telemetry_validation_status"] == "passed"
    assert any("report status is 'running'" in failure for failure in ineligible["failures"])


def test_locality_sensitivity_eligibility_is_classified_and_identity_bound(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=2)
    report = _valid_saturation_report(geometry)
    identity = _valid_run_identity(geometry)
    _write_terminal_inputs(tmp_path, report, identity)

    eligible = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )

    assert eligible["status"] == "eligible"
    assert eligible["evidence_class"] == "locality_sensitivity"
    assert eligible["policy"]["evidence_class"] == "locality_sensitivity"
    assert eligible["policy"]["primary_saturation_waves"] == [4, 8]
    assert eligible["policy"]["locality_sensitivity_waves"] == [1, 2]
    assert eligible["benchmark_validation"]["evidence_class"] == "locality_sensitivity"
    assert eligible["identity_validation"]["status"] == "passed"

    identity["evidence_class"] = "primary_saturation"
    _write_terminal_inputs(tmp_path, report, identity)
    mismatched = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )

    assert mismatched["status"] == "ineligible"
    assert any("run identity evidence_class" in failure for failure in mismatched["failures"])

    identity["schema_version"] = 1
    identity["evidence_class"] = "locality_sensitivity"
    _write_terminal_inputs(tmp_path, report, identity)
    invalid_legacy = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )

    assert invalid_legacy["status"] == "ineligible"
    assert any("schema_version is 1; expected 2" in failure for failure in invalid_legacy["failures"])


@pytest.mark.parametrize(
    ("identity_section", "field", "value", "failure_text"),
    [
        ("dataset", "uri", "s3://other/dataset", "dataset.uri disagrees"),
        ("dataset", "version", 5, "dataset.version disagrees"),
        ("manifest", "digest_sha256", "d" * 64, "manifest digest disagrees"),
        ("benchmark_policy", "warmup_count", 0, "benchmark_policy.warmup_count disagrees"),
    ],
)
def test_terminal_identity_rejects_dataset_or_manifest_drift(
    tmp_path: Path,
    identity_section: str,
    field: str,
    value: object,
    failure_text: str,
) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    identity = _valid_run_identity(geometry)
    identity[identity_section][field] = value
    _write_terminal_inputs(tmp_path, report, identity)

    result = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )

    assert result["status"] == "ineligible"
    assert any(failure_text in failure for failure in result["failures"])


def test_terminal_identity_rejects_matching_invalid_dataset_and_manifest_values(tmp_path: Path) -> None:
    geometry = runner.SaturationGeometry(nodes=1, waves=8)
    report = _valid_saturation_report(geometry)
    identity = _valid_run_identity(geometry)
    report["dataset"].update(uri=None, version=None)
    identity["dataset"].update(uri=None, version=None)
    report["manifest"]["digest_sha256"] = "z" * 64
    identity["manifest"]["digest_sha256"] = "z" * 64
    _write_terminal_inputs(tmp_path, report, identity)

    result = runner.build_terminal_eligibility(
        tmp_path,
        arm="lance_ray_gpu_actor",
        geometry=geometry,
        repeat_count=2,
        warmup_count=1,
    )

    assert result["status"] == "ineligible"
    assert any("benchmark dataset URI is missing or invalid" in failure for failure in result["failures"])
    assert any("run identity dataset.version is missing or invalid" in failure for failure in result["failures"])
    assert any("manifest.digest_sha256 is missing or invalid" in failure for failure in result["failures"])
