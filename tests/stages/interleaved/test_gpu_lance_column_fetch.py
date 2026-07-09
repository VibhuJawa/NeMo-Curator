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

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved import (
    GpuLanceColumnFetchStage,
    GpuLanceIndexCacheConfig,
    LanceDatasetConfig,
    LanceIndexCacheConfig,
    gpu_lance,
)
from nemo_curator.stages.interleaved.gpu_key_lookup import (
    _STABLE_ID_COVERAGE_DTYPE,
    _build_sidecar_contract_bytes,
    _GpuMapResult,
    _stable_global_ordinal_manifest_sha256,
)
from nemo_curator.stages.interleaved.lance import _validate_stable_global_ordinal_manifest
from nemo_curator.tasks import InterleavedBatch

lance = pytest.importorskip("lance")


class _FakeArrowMapper:
    """CPU stand-in that preserves the GPU mapper's Arrow contract."""

    instances: ClassVar[list["_FakeArrowMapper"]] = []
    forced_reference_type: ClassVar[pa.DataType | None] = None

    def __init__(  # noqa: PLR0913
        self,
        reference_files: Sequence[str],
        reference_key_column: str,
        reference_row_id_column: str,
        storage_options: dict[str, str],
        expected_reference_rows: int,
        load_factor: float,
    ) -> None:
        self.reference_files = list(reference_files)
        self.storage_options = dict(storage_options)
        self.load_factor = load_factor
        tables = [
            pq.read_table(path, columns=[reference_key_column, reference_row_id_column]) for path in reference_files
        ]
        table = pa.concat_tables(tables)
        self.reference_type = self.forced_reference_type or table.schema.field(reference_key_column).type
        self.reference_rows = table.num_rows
        if self.reference_rows != expected_reference_rows:
            msg = f"Reference key segments contain {self.reference_rows} rows; expected {expected_reference_rows}"
            raise ValueError(msg)
        self.load_seconds = 0.125
        self.build_seconds = 0.25
        self.gpu_bytes = 4_096
        self.gpu_total_bytes = 16_384
        self.calls: list[list[object]] = []
        self.closed = False
        keys = table[reference_key_column].combine_chunks().to_pylist()
        row_ids = table[reference_row_id_column].combine_chunks().to_pylist()
        self._row_ids_by_key = {key: int(row_id) for key, row_id in zip(keys, row_ids, strict=True)}
        self.instances.append(self)

    def map(self, keys: pa.Array) -> _GpuMapResult:
        values = keys.to_pylist()
        self.calls.append(values)
        matched = np.array([key in self._row_ids_by_key for key in values], dtype=np.bool_)
        row_ids = np.array([self._row_ids_by_key.get(key, 0) for key in values], dtype=np.uint64)
        return _GpuMapResult(
            matched=matched,
            row_ids=row_ids,
            transfer_seconds=0.01,
            probe_seconds=0.02,
            search_seconds=0.03,
            gather_seconds=0.04,
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_mapper(monkeypatch: pytest.MonkeyPatch) -> type[_FakeArrowMapper]:
    _FakeArrowMapper.instances.clear()
    _FakeArrowMapper.forced_reference_type = None
    monkeypatch.setattr(gpu_lance, "_GpuExactKeyMapper", _FakeArrowMapper)
    return _FakeArrowMapper


def _reference_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("url", pa.string(), nullable=False),
            pa.field("image", pa.large_binary(), nullable=False),
            pa.field("width", pa.int32(), nullable=False),
        ]
    )


def _write_reference(dataset_path: Path, sidecar_path: Path) -> int:
    table = pa.Table.from_pylist(
        [
            {"url": "url-c", "image": b"image-c", "width": 300},
            {"url": "url-a", "image": b"image-a", "width": 100},
            {"url": "url-b", "image": b"image-b", "width": 200},
        ],
        schema=_reference_schema(),
    )
    lance.write_dataset(
        table,
        str(dataset_path),
        mode="create",
        data_storage_version="2.2",
        enable_stable_row_ids=True,
    )
    dataset = lance.dataset(str(dataset_path))
    rows = dataset.scanner(columns=["url"], with_row_id=True).to_table()
    sidecar = pa.table(
        {
            "url": rows["url"].combine_chunks(),
            "stable_row_id": pa.array(rows["_rowid"].combine_chunks().to_pylist(), type=pa.uint64()),
        }
    ).sort_by([("url", "ascending")])
    pq.write_table(sidecar, sidecar_path)
    return dataset.version


def _dataset_config(path: Path, version: int) -> LanceDatasetConfig:
    return LanceDatasetConfig(
        uri=str(path),
        version=version,
        key_column="url",
        index_name="unused_by_gpu_fetch",
    )


def _write_contract(dataset: LanceDatasetConfig, sidecar_path: Path) -> tuple[Path, str]:
    lance_dataset = lance.dataset(dataset.uri, version=dataset.version)
    manifest = _validate_stable_global_ordinal_manifest(lance_dataset)
    fragment_digest = _stable_global_ordinal_manifest_sha256(dataset.uri, dataset.version, manifest)
    raw_contract, contract_digest = _build_sidecar_contract_bytes(
        dataset=lance_dataset,
        dataset_uri=dataset.uri,
        dataset_version=dataset.version,
        fragment_manifest_sha256=fragment_digest,
        total_rows=manifest.total_rows,
        key_column="url",
        row_id_column="stable_row_id",
        layout="replicated_sorted",
        partition_files=((str(sidecar_path),),),
        storage_options={},
    )
    contract_path = sidecar_path.with_suffix(".manifest.json")
    contract_path.write_bytes(raw_contract)
    return contract_path, contract_digest


def _index_cache() -> LanceIndexCacheConfig:
    return LanceIndexCacheConfig(
        prewarm=False,
        index_cache_size_bytes=1024**2,
        metadata_cache_size_bytes=1024**2,
    )


def _batch(keys: list[str | None]) -> InterleavedBatch:
    count = len(keys)
    return InterleavedBatch(
        dataset_name="interleaved",
        data=pa.table(
            {
                "sample_id": [f"sample-{index}" for index in range(count)],
                "position": pa.array(range(count), type=pa.int32()),
                "modality": ["image"] * count,
                "source_ref": pa.array(keys, type=pa.string()),
            }
        ),
    )


def test_byte_windows_bound_encoded_bytes_and_reject_oversized_values() -> None:
    values = pa.array(["aa", "bbb", "cccc", "c"], type=pa.string())

    windows = list(gpu_lance._byte_windows(values, max_bytes=9))

    assert pa.concat_arrays(windows).equals(values)
    assert all(window.nbytes <= 9 for window in windows)

    with pytest.raises(MemoryError, match="exceeds max_lookup_bytes"):
        list(gpu_lance._byte_windows(pa.array(["value-larger-than-cap"]), max_bytes=9))


def test_gpu_lance_column_fetch_projects_payloads_batches_tasks_and_reports_metrics(  # noqa: PLR0915
    tmp_path: Path,
    fake_mapper: type[_FakeArrowMapper],
) -> None:
    dataset_path = tmp_path / "reference.lance"
    sidecar_path = tmp_path / "reference-index.parquet"
    dataset = _dataset_config(dataset_path, _write_reference(dataset_path, sidecar_path))
    contract_path, contract_digest = _write_contract(dataset, sidecar_path)
    stage = GpuLanceColumnFetchStage(
        dataset=dataset,
        index_cache=_index_cache(),
        reference_files=[str(sidecar_path)],
        reference_manifest_uri=str(contract_path),
        reference_manifest_sha256=contract_digest,
        expected_reference_rows=3,
        columns={"image": "binary_content", "width": "reference_width"},
        presence_column="image_present",
        fetch_batch_size=1,
        max_lookup_bytes=20,
    )

    outputs = stage.process_batch(
        [
            _batch(["url-c", "missing", "url-c", None, ""]),
            _batch(["url-a", "url-b", "url-a"]),
        ]
    )
    first_metrics = stage._consume_custom_metrics()
    second = stage.process(_batch(["url-b"]))
    second_metrics = stage._consume_custom_metrics()
    mapper = fake_mapper.instances[0]
    stage.teardown()

    assert mapper.calls == [["url-c", "missing"], ["url-a", "url-b"], ["url-b"]]
    assert mapper.closed
    assert outputs[0].to_pyarrow()["source_ref"].to_pylist() == ["url-c", "missing", "url-c", None, ""]
    assert outputs[0].to_pyarrow()["binary_content"].to_pylist() == [
        b"image-c",
        None,
        b"image-c",
        None,
        None,
    ]
    assert outputs[0].to_pyarrow()["reference_width"].to_pylist() == [300, None, 300, None, None]
    assert outputs[0].to_pyarrow()["image_present"].to_pylist() == [True, False, True, None, None]
    assert outputs[1].to_pyarrow()["binary_content"].to_pylist() == [b"image-a", b"image-b", b"image-a"]
    assert outputs[1].to_pyarrow()["reference_width"].to_pylist() == [100, 200, 100]
    assert outputs[1].to_pyarrow()["image_present"].to_pylist() == [True, True, True]
    assert second.to_pyarrow()["binary_content"].to_pylist() == [b"image-b"]

    assert first_metrics["input_tasks"] == 2.0
    assert first_metrics["requested_unique_keys"] == 4.0
    assert first_metrics["found_unique_keys"] == 3.0
    assert first_metrics["missing_unique_keys"] == 1.0
    assert first_metrics["gpu_eligible_keys"] == 4.0
    assert first_metrics["gpu_mapped_keys"] == 3.0
    assert first_metrics["gpu_lookup_windows"] == 2.0
    assert first_metrics["gpu_key_transfer_seconds"] == pytest.approx(0.02)
    assert first_metrics["gpu_key_probe_seconds"] == pytest.approx(0.04)
    assert first_metrics["gpu_row_id_search_seconds"] == pytest.approx(0.06)
    assert first_metrics["gpu_row_id_gather_seconds"] == pytest.approx(0.08)
    assert first_metrics["gpu_reference_rows"] == 3.0
    assert first_metrics["gpu_reference_load_seconds"] == 0.125
    assert first_metrics["gpu_hash_build_seconds"] == 0.25
    assert first_metrics["gpu_reference_bytes"] == 4_096.0
    assert first_metrics["gpu_total_bytes"] == 16_384.0
    assert first_metrics["private_take_calls"] == 3.0
    assert first_metrics["private_take_rows"] == 3.0
    assert first_metrics["rows_per_private_take"] == 1.0
    assert first_metrics["logical_payload_requests"] == 5.0
    assert first_metrics["unique_payloads"] == 3.0
    assert first_metrics["logical_duplicate_requests"] == 2.0
    assert first_metrics["duplicate_fanout"] == pytest.approx(5 / 3)
    assert first_metrics["sparse_calls_avoided"] == 0.0
    assert first_metrics["average_physical_read_bytes"] >= 0.0
    assert first_metrics["physical_reads_per_payload"] >= 0.0
    assert first_metrics["read_amplification"] >= 0.0
    assert second_metrics["gpu_mapped_keys"] == 1.0
    assert second_metrics["gpu_lookup_windows"] == 1.0
    assert second_metrics["private_take_calls"] == 1.0
    assert "gpu_reference_rows" not in second_metrics
    assert "gpu_reference_load_seconds" not in second_metrics


def test_gpu_lance_column_fetch_validates_gpu_index_configuration() -> None:
    assert _STABLE_ID_COVERAGE_DTYPE == "uint32"
    dataset = LanceDatasetConfig(uri="reference.lance", version=1, key_column="url", index_name="unused")
    stage_kwargs = {
        "dataset": dataset,
        "index_cache": _index_cache(),
        "columns": {"image": "binary_content"},
        "presence_column": "image_present",
        "reference_manifest_uri": "sidecar-manifest.json",
        "reference_manifest_sha256": "0" * 64,
        "expected_reference_rows": 3,
    }

    with pytest.raises(ValueError, match="reference_files must not be empty"):
        GpuLanceColumnFetchStage(reference_files=[], **stage_kwargs)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        GpuLanceColumnFetchStage(reference_files=["same.parquet", "same.parquet"], **stage_kwargs)
    with pytest.raises(ValueError, match="unique basenames"):
        GpuLanceColumnFetchStage(
            reference_files=["first/index.parquet", "second/index.parquet"],
            **stage_kwargs,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        GpuLanceColumnFetchStage(
            reference_files=["index.parquet"],
            reference_key_column="",
            **stage_kwargs,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        GpuLanceColumnFetchStage(
            reference_files=["index.parquet"],
            reference_row_id_column="",
            **stage_kwargs,
        )
    with pytest.raises(ValueError, match="greater than zero"):
        GpuLanceColumnFetchStage(reference_files=["index.parquet"], **{**stage_kwargs, "expected_reference_rows": 0})
    with pytest.raises(ValueError, match="reference_manifest_uri"):
        GpuLanceColumnFetchStage(
            reference_files=["index.parquet"],
            **{**stage_kwargs, "reference_manifest_uri": ""},
        )
    with pytest.raises(ValueError, match="interval"):
        GpuLanceColumnFetchStage(reference_files=["index.parquet"], load_factor=0.0, **stage_kwargs)
    with pytest.raises(ValueError, match="interval"):
        GpuLanceColumnFetchStage(reference_files=["index.parquet"], load_factor=1.01, **stage_kwargs)
    with pytest.raises(ValueError, match="max_lookup_bytes must be greater than zero"):
        GpuLanceColumnFetchStage(reference_files=["index.parquet"], max_lookup_bytes=0, **stage_kwargs)
    with pytest.raises(ValueError, match="node_local_root must not be empty"):
        GpuLanceIndexCacheConfig(copy_to_node_local=True, node_local_root="")


def test_gpu_lance_config_rejects_credential_bearing_ready_marker_identities() -> None:
    dummy_uri = "s3://dummy-user:dummy-pass@bucket/path?dummy-token=value#dummy-fragment"
    with pytest.raises(ValueError, match="userinfo") as dataset_error:
        LanceDatasetConfig(uri=dummy_uri, version=1, key_column="url", index_name="unused")
    assert "dummy-pass" not in str(dataset_error.value)

    dataset = LanceDatasetConfig(uri="reference.lance", version=1, key_column="url", index_name="unused")
    base = {
        "dataset": dataset,
        "index_cache": _index_cache(),
        "columns": {"image": "binary_content"},
        "presence_column": "image_present",
        "reference_manifest_sha256": "0" * 64,
        "expected_reference_rows": 3,
    }
    with pytest.raises(ValueError, match="userinfo") as manifest_error:
        GpuLanceColumnFetchStage(
            reference_files=["index.parquet"],
            reference_manifest_uri=dummy_uri,
            **base,
        )
    assert "dummy-pass" not in str(manifest_error.value)

    with pytest.raises(ValueError, match="userinfo") as file_error:
        GpuLanceColumnFetchStage(
            reference_files=[dummy_uri],
            reference_manifest_uri="sidecar-manifest.json",
            **base,
        )
    assert "dummy-pass" not in str(file_error.value)


def test_gpu_lance_column_fetch_rejects_reference_key_type_and_closes_mapper(
    tmp_path: Path,
    fake_mapper: type[_FakeArrowMapper],
) -> None:
    dataset_path = tmp_path / "reference.lance"
    sidecar_path = tmp_path / "reference-index.parquet"
    dataset = _dataset_config(dataset_path, _write_reference(dataset_path, sidecar_path))
    contract_path, contract_digest = _write_contract(dataset, sidecar_path)
    fake_mapper.forced_reference_type = pa.int64()
    stage = GpuLanceColumnFetchStage(
        dataset=dataset,
        index_cache=_index_cache(),
        reference_files=[str(sidecar_path)],
        reference_manifest_uri=str(contract_path),
        reference_manifest_sha256=contract_digest,
        expected_reference_rows=3,
        columns={"image": "binary_content"},
        presence_column="image_present",
    )

    with pytest.raises(TypeError, match="Lance key column has type string; GPU reference key column has type int64"):
        stage.setup()

    assert fake_mapper.instances[0].closed
    assert stage._fetcher is None


def test_gpu_lance_column_fetch_rejects_permuted_sidecar_before_mapper_setup(
    tmp_path: Path,
    fake_mapper: type[_FakeArrowMapper],
) -> None:
    dataset_path = tmp_path / "reference.lance"
    sidecar_path = tmp_path / "reference-index.parquet"
    dataset = _dataset_config(dataset_path, _write_reference(dataset_path, sidecar_path))
    contract_path, contract_digest = _write_contract(dataset, sidecar_path)
    sidecar = pq.read_table(sidecar_path)
    permuted_ids = pa.array(list(reversed(sidecar["stable_row_id"].to_pylist())), type=pa.uint64())
    pq.write_table(sidecar.set_column(1, "stable_row_id", permuted_ids), sidecar_path)
    stage = GpuLanceColumnFetchStage(
        dataset=dataset,
        index_cache=_index_cache(),
        reference_files=[str(sidecar_path)],
        reference_manifest_uri=str(contract_path),
        reference_manifest_sha256=contract_digest,
        expected_reference_rows=3,
        columns={"image": "binary_content"},
        presence_column="image_present",
    )

    with pytest.raises(ValueError, match="Sidecar file identity mismatch"):
        stage.setup()

    assert fake_mapper.instances == []
    assert stage._fetcher is None


def test_gpu_lance_column_fetch_stages_index_once_per_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_mapper: type[_FakeArrowMapper],
) -> None:
    dataset_path = tmp_path / "reference.lance"
    sidecar_path = tmp_path / "source" / "reference-index.parquet"
    sidecar_path.parent.mkdir()
    dataset = _dataset_config(dataset_path, _write_reference(dataset_path, sidecar_path))
    contract_path, contract_digest = _write_contract(dataset, sidecar_path)
    gpu_index_cache = GpuLanceIndexCacheConfig(
        copy_to_node_local=True,
        node_local_root=str(tmp_path / "node-local"),
    )
    stage = GpuLanceColumnFetchStage(
        dataset=dataset,
        index_cache=_index_cache(),
        reference_files=[str(sidecar_path)],
        reference_manifest_uri=str(contract_path),
        reference_manifest_sha256=contract_digest,
        expected_reference_rows=3,
        gpu_index_cache=gpu_index_cache,
        columns={"image": "binary_content"},
        presence_column="image_present",
    )

    with pytest.raises(RuntimeError, match="Node-local GPU Lance index is not ready"):
        stage.setup()

    stage.setup_on_node()
    resolved = stage._resolved_reference_files()
    target = gpu_index_cache.node_local_path(
        dataset,
        [str(sidecar_path)],
        "url",
        "stable_row_id",
        contract_digest,
    )
    assert resolved == [str(target / sidecar_path.name)]
    assert Path(resolved[0]).read_bytes() == sidecar_path.read_bytes()

    real_fsspec_open = gpu_lance.fsspec.open

    def fail_if_source_is_reopened(*args: object, **kwargs: object) -> object:
        if args[0] == str(sidecar_path):
            msg = "ready node-local index should not reopen source files"
            raise AssertionError(msg)
        return real_fsspec_open(*args, **kwargs)

    monkeypatch.setattr(gpu_lance.fsspec, "open", fail_if_source_is_reopened)
    stage.setup_on_node()
    stage.setup()
    mapper = fake_mapper.instances[0]
    output = stage.process(_batch(["url-a", "missing"])).to_pyarrow()
    stage.teardown()

    assert mapper.reference_files == resolved
    assert mapper.storage_options == {}
    assert mapper.closed
    assert output["binary_content"].to_pylist() == [b"image-a", None]
    assert output["image_present"].to_pylist() == [True, False]
