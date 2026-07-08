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
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from benchmarking.scripts import gpu_lance_column_fetch_benchmark as benchmark


def _parse(*extra: str) -> argparse.Namespace:
    return benchmark.build_parser().parse_args(
        [
            "--query-manifest",
            "queries.parquet",
            "--image-lance-uri",
            "s3://bucket/images",
            "--image-lance-version",
            "4",
            "--output",
            "report.json",
            *extra,
        ]
    )


def test_private_take_defaults_reach_benchmark_settings() -> None:
    args = _parse()

    settings = benchmark._make_settings(args, [])

    assert args.query_manifest == Path("queries.parquet")
    assert settings.fetch_batch_size == 1024
    assert settings.max_pending_fetch_batches == 16
    assert settings.payload_read_mode == "sparse"
    assert settings.validate_payload_keys is False


def test_storage_options_reject_credentials_and_report_uris_are_redacted() -> None:
    with pytest.raises(ValueError, match="process environment"):
        benchmark._json_options('{"secret_access_key":"do-not-load"}')

    assert (
        benchmark._redact_uri_for_report("https://user:password@example.test:8443/data?token=secret#fragment")
        == "https://example.test:8443/data"
    )
    error = benchmark._error_record(RuntimeError("failed https://user:password@example.test/data?token=secret"))
    assert error == {"type": "RuntimeError", "message": "failed https://example.test/data"}


def test_lance_ray_gpu_actor_tuning_reaches_settings() -> None:
    args = _parse(
        "--arm",
        "lance_ray_gpu_actor",
        "--max-lookup-bytes-mib",
        "4096",
        "--max-pending-fetch-batches",
        "8",
    )

    settings = benchmark._make_settings(args, ["sidecar.parquet"])

    assert args.arm == ["lance_ray_gpu_actor"]
    assert settings.max_lookup_bytes == 4 * 1024**3
    assert settings.max_pending_fetch_batches == 8


def test_lance_ray_gpu_fetcher_is_selectable() -> None:
    args = _parse(
        "--arm",
        "lance_ray_gpu_fetcher",
        "--payload-read-mode",
        "adaptive_unmeasured",
        "--validate-payload-keys",
    )
    settings = benchmark._make_settings(args, ["sidecar.parquet"])

    assert args.arm == ["lance_ray_gpu_fetcher"]
    assert settings.payload_read_mode == "adaptive_unmeasured"
    assert settings.validate_payload_keys is True


def test_index_mirror_requires_and_parses_pinned_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be configured together"):
        benchmark._make_settings(_parse("--index-mirror", "/mirror/images"), [])

    contract_path = tmp_path / "mirror-contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "format": "nemo-curator-lance-index-mirror-v1",
                "remote_uri": "s3://bucket/images",
                "remote_version": 4,
                "remote_fragment_manifest_sha256": "a" * 64,
                "mirror_uri": "/mirror/images",
                "mirror_version": 4,
                "key_column": "url",
                "key_stable_ordinal_sha256": "b" * 64,
                "index_name": "url_btree",
                "index_artifacts_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    settings = benchmark._make_settings(
        _parse(
            "--index-mirror",
            "/mirror/images",
            "--index-mirror-contract-json",
            f"@{contract_path}",
        ),
        [],
    )

    assert settings.index_mirror_contract is not None
    assert settings.index_mirror_contract["mirror_uri"] == "/mirror/images"
    assert "format" not in settings.index_mirror_contract


def test_lance_ray_actor_run_reuses_balanced_persistent_pool() -> None:
    calls: list[tuple[int, int]] = []

    class RemoteMethod:
        def __init__(self, actor_id: int) -> None:
            self.actor_id = actor_id

        def remote(self, table: pa.Table) -> pa.Table:
            calls.append((self.actor_id, table.num_rows))
            return table

    class Actor:
        def __init__(self, actor_id: int) -> None:
            self.process = RemoteMethod(actor_id)

    arm = object.__new__(benchmark.LanceRayGpuActorArm)
    arm.settings = SimpleNamespace(coalesce_tasks=2)
    arm._persistent_actors = [Actor(0), Actor(1)]
    arm._actor_blocks = [
        [pa.table({"row": [row]}) for row in range(4)],
        [pa.table({"row": [row]}) for row in range(4, 8)],
    ]
    arm.ray = SimpleNamespace(get=lambda values: values)
    captured: dict[str, object] = {}

    def finish(outputs: list[pa.Table], count: int, *, actors_used: int) -> benchmark.ArmRun:
        captured.update(outputs=outputs, count=count, actors_used=actors_used)
        return benchmark.ArmRun(pa.concat_tables(outputs), {})

    arm._finish_actor_run = finish

    result = arm.run()

    assert result.table.num_rows == 8
    assert calls == [(0, 2), (1, 2), (0, 2), (1, 2)]
    assert len(captured["outputs"]) == 4
    assert captured["count"] == 8
    assert captured["actors_used"] == 2


def test_actor_metrics_use_sum_max_and_ratio_reducers() -> None:
    def batch(ordinal: int, **metrics: float) -> pa.Table:
        table = pa.table({benchmark._ORDINAL: [ordinal], "source_ref": [f"key-{ordinal}"]})
        for name, value in metrics.items():
            table = table.append_column(f"{benchmark._ACTOR_PREFIX}{name}", pa.array([value]))
        return table

    arm = object.__new__(benchmark.RayDataActorArm)
    arm.input_table = pa.table({benchmark._ORDINAL: [0, 1], "source_ref": ["key-0", "key-1"]})
    arm.settings = SimpleNamespace(ray_gpu_actors=2)
    result = arm._finish_actor_run(
        [
            batch(
                1,
                process_seconds=7,
                process_started_epoch=20,
                process_ended_epoch=27,
                setup_seconds=3,
                warmup_seconds=1,
                lance_read_bytes=1_000,
                lance_read_iops=10,
                lance_fetched_bytes=400,
                payload_take_calls=1,
                payload_take_rows=10,
                max_pending_payload_reads=4,
                average_physical_read_bytes=100,
                read_amplification=2.5,
            ),
            batch(
                0,
                process_seconds=8,
                process_started_epoch=21,
                process_ended_epoch=29,
                setup_seconds=5,
                warmup_seconds=2,
                lance_read_bytes=2_000,
                lance_read_iops=20,
                lance_fetched_bytes=600,
                payload_take_calls=2,
                payload_take_rows=20,
                max_pending_payload_reads=8,
                average_physical_read_bytes=100,
                read_amplification=10 / 3,
            ),
        ],
        input_block_count=2,
        actors_used=2,
    )

    assert result.table[benchmark._ORDINAL].to_pylist() == [0, 1]
    assert result.metrics["lance_read_bytes"] == 3_000
    assert result.metrics["lance_read_iops"] == 30
    assert result.metrics["max_pending_payload_reads"] == 8
    assert result.metrics["rows_per_payload_take"] == 10
    assert result.metrics["average_physical_read_bytes"] == 100
    assert result.metrics["read_amplification"] == 3
    assert result.metrics["process_seconds"] == 9


def test_decompression_bomb_is_a_validation_skip_not_corruption(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_safety_limit(*_args: object, **_kwargs: object) -> Never:
        message = "image exceeds Pillow safety limit"
        raise Image.DecompressionBombError(message)

    monkeypatch.setattr(Image, "open", raise_safety_limit)
    manifest_table = pa.table(
        {
            "source_ref": ["https://example.test/large.jpg"],
            benchmark._ORDINAL: pa.array([0], type=pa.int64()),
        }
    )
    manifest = benchmark.QueryManifest(manifest_table, {}, "unused")
    output = manifest_table.append_column(benchmark._PRESENT, pa.array([True])).append_column(
        benchmark._FETCHED["image"],
        pa.array([b"valid-payload"], type=pa.large_binary()),
    )

    result = benchmark._validate_output(output, manifest)

    assert result["correct"] is True
    assert result["dimensions"]["decoded_mismatch_count"] == 0
    assert result["dimensions"]["decode_safety_skipped_count"] == 1
    assert result["dimensions"]["decode_safety_skipped_rows"] == [0]


class _FakeArm(benchmark.BenchmarkArm):
    def __init__(  # noqa: PLR0913 - explicit failure knobs keep the regression cases readable
        self,
        name: str,
        manifest: benchmark.QueryManifest,
        *,
        setup_error: BaseException | None = None,
        run_error: BaseException | None = None,
        close_error: BaseException | None = None,
        incorrect: bool = False,
    ) -> None:
        self.name = name
        self.manifest = manifest
        self.setup_metrics = {}
        self._setup_error = setup_error
        self._run_error = run_error
        self._close_error = close_error
        self._incorrect = incorrect

    def setup(self) -> None:
        if self._setup_error is not None:
            raise self._setup_error

    def run(self) -> benchmark.ArmRun:
        if self._run_error is not None:
            raise self._run_error
        sink = io.BytesIO()
        Image.new("RGB", (1, 1)).save(sink, format="PNG")
        payload = None if self._incorrect else sink.getvalue()
        output = self.manifest.table.append_column(
            benchmark._FETCHED["image"],
            pa.array([payload], type=pa.large_binary()),
        ).append_column(benchmark._PRESENT, pa.array([True]))
        return benchmark.ArmRun(output, {})

    def close(self) -> None:
        if self._close_error is not None:
            raise self._close_error


def _benchmark_args(tmp_path: Path, *arms: str) -> argparse.Namespace:
    manifest_path = tmp_path / "queries.parquet"
    pq.write_table(pa.table({"source_ref": ["https://example.test/image.png"]}), manifest_path)
    arm_args = [item for arm in arms for item in ("--arm", arm)]
    return _parse(
        "--query-manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
        "--warmup-count",
        "0",
        "--repeat-count",
        "1",
        *arm_args,
    )


def _install_fake_arms(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[str, benchmark.QueryManifest], benchmark.BenchmarkArm],
) -> None:
    def construct(
        selected: list[str], manifest: benchmark.QueryManifest, _settings: benchmark.BenchmarkSettings
    ) -> dict[str, benchmark.BenchmarkArm]:
        return {name: factory(name, manifest) for name in selected}

    monkeypatch.setattr(benchmark, "_construct_arms", construct)


def test_all_skipped_arms_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _benchmark_args(tmp_path, "gpu_lance_column_fetch_stage")

    def factory(name: str, manifest: benchmark.QueryManifest) -> _FakeArm:
        return _FakeArm(name, manifest, setup_error=benchmark.ArmUnavailableError("GPU unavailable"))

    _install_fake_arms(monkeypatch, factory)

    report = benchmark.run_benchmark(args)

    assert report["arms"]["gpu_lance_column_fetch_stage"]["status"] == "skipped"
    assert report["status"] == "failed"


def test_setup_failure_fails_closed_and_main_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arm_name = "gpu_lance_column_fetch_stage"
    args = _benchmark_args(tmp_path, arm_name)

    def factory(name: str, manifest: benchmark.QueryManifest) -> _FakeArm:
        return _FakeArm(name, manifest, setup_error=TypeError("constructor contract changed"))

    _install_fake_arms(monkeypatch, factory)
    monkeypatch.setattr(benchmark, "build_parser", lambda: _ParsedArgsParser(args))

    assert benchmark.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["arms"][arm_name]["status"] == "setup_failed"
    assert report["arms"][arm_name]["error"]["type"] == "TypeError"
    assert report["status"] == "failed"
    assert json.loads(args.output.read_text())["status"] == "failed"


def test_mixed_skipped_and_completed_arms_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skipped = "gpu_lance_column_fetch_stage"
    completed = "cpu_lance_column_fetch_stage"
    args = _benchmark_args(tmp_path, skipped, completed)

    def factory(name: str, manifest: benchmark.QueryManifest) -> _FakeArm:
        error = benchmark.ArmUnavailableError("GPU unavailable") if name == skipped else None
        return _FakeArm(name, manifest, setup_error=error)

    _install_fake_arms(monkeypatch, factory)

    report = benchmark.run_benchmark(args)

    assert report["arms"][skipped]["status"] == "skipped"
    assert report["arms"][completed]["status"] == "completed"
    assert report["status"] == "failed"


def test_teardown_error_fails_completed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arm_name = "cpu_lance_column_fetch_stage"
    args = _benchmark_args(tmp_path, arm_name)

    def factory(name: str, manifest: benchmark.QueryManifest) -> _FakeArm:
        return _FakeArm(name, manifest, close_error=RuntimeError("teardown failed"))

    _install_fake_arms(monkeypatch, factory)

    report = benchmark.run_benchmark(args)

    assert report["arms"][arm_name]["status"] == "completed"
    assert report["teardown_errors"][arm_name]["type"] == "RuntimeError"
    assert report["status"] == "failed"
    assert json.loads(args.output.read_text())["status"] == "failed"


def test_application_failure_makes_main_exit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arm_name = "cpu_lance_column_fetch_stage"
    args = _benchmark_args(tmp_path, arm_name)

    def factory(name: str, manifest: benchmark.QueryManifest) -> _FakeArm:
        return _FakeArm(name, manifest, run_error=RuntimeError("application failed"))

    _install_fake_arms(monkeypatch, factory)
    monkeypatch.setattr(benchmark, "build_parser", lambda: _ParsedArgsParser(args))

    assert benchmark.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["arms"][arm_name]["status"] == "run_failed"
    assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("failure_mode", "expected_arm_status"),
    [
        ("unavailable", "skipped"),
        ("warmup_error", "warmup_failed"),
        ("warmup_incorrect", "warmup_incorrect"),
        ("repeat_incorrect", "incorrect"),
        ("teardown_error", "completed"),
    ],
)
def test_every_benchmark_failure_class_makes_main_exit_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_mode: str,
    expected_arm_status: str,
) -> None:
    arm_name = "cpu_lance_column_fetch_stage"
    args = _benchmark_args(tmp_path, arm_name)
    if failure_mode.startswith("warmup_"):
        args.warmup_count = 1

    def factory(name: str, manifest: benchmark.QueryManifest) -> _FakeArm:
        return _FakeArm(
            name,
            manifest,
            setup_error=(
                benchmark.ArmUnavailableError("dependency unavailable") if failure_mode == "unavailable" else None
            ),
            run_error=RuntimeError("warmup failed") if failure_mode == "warmup_error" else None,
            close_error=RuntimeError("teardown failed") if failure_mode == "teardown_error" else None,
            incorrect=failure_mode in {"warmup_incorrect", "repeat_incorrect"},
        )

    _install_fake_arms(monkeypatch, factory)
    monkeypatch.setattr(benchmark, "build_parser", lambda: _ParsedArgsParser(args))

    assert benchmark.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["arms"][arm_name]["status"] == expected_arm_status
    assert report["status"] == "failed"


class _ParsedArgsParser:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

    def parse_args(self) -> argparse.Namespace:
        return self._args
