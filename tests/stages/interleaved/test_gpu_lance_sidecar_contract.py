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

import hashlib
import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved import gpu_key_lookup
from nemo_curator.stages.interleaved.lance import _key_stable_ordinal_sha256

if TYPE_CHECKING:
    from pathlib import Path


_PARTITIONING = gpu_key_lookup._MpfHashPartitioningContract(
    algorithm="cudf::hash_id::HASH_MURMUR3",
    implementation="rapidsmpf.integrations.cudf.partition.partition_and_pack",
    libcudf_version="26.6.0",
    rapidsmpf_version="26.6.0",
    seed=0,
)


class _FakeColumn:
    def __init__(self, values: list[str | None]) -> None:
        self._values = values

    def isnull(self) -> np.ndarray:
        return np.asarray([value is None for value in self._values])

    def duplicated(self) -> np.ndarray:
        seen: set[str | None] = set()
        result = []
        for value in self._values:
            result.append(value in seen)
            seen.add(value)
        return np.asarray(result)


class _FakeFrame:
    def __init__(self, keys: list[str | None], hashes: list[int]) -> None:
        self._keys = keys
        self._hashes = np.asarray(hashes, dtype=np.uint32)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str | list[str]) -> _FakeColumn | _FakeFrame:
        return self if isinstance(key, list) else _FakeColumn(self._keys)

    def hash_values(self, *, method: str, seed: int) -> np.ndarray:
        assert (method, seed) == ("murmur3", 0)
        return self._hashes


class _FakeIdentityDataset:
    def __init__(self, keys: list[str], key_type: pa.DataType | None = None) -> None:
        self._keys = keys
        self._key_type = pa.string() if key_type is None else key_type
        self.schema = pa.schema([pa.field("url", self._key_type)])

    def scanner(self, **kwargs: object) -> SimpleNamespace:
        batch_size = int(kwargs.get("batch_size", len(self._keys)))
        batches = [
            pa.record_batch(
                {
                    "url": pa.array(self._keys[offset : offset + batch_size], type=self._key_type),
                    "_rowid": pa.array(
                        range(offset, min(offset + batch_size, len(self._keys))),
                        type=pa.uint64(),
                    ),
                }
            )
            for offset in range(0, len(self._keys), batch_size)
        ]
        return SimpleNamespace(to_batches=lambda: batches)


@pytest.mark.parametrize("credential_location", ["dataset", "sidecar"])
def test_sidecar_contract_rejects_credential_bearing_uri_identities(credential_location: str) -> None:
    dummy_uri = "s3://dummy-user:dummy-pass@bucket/path?dummy-token=value#dummy-fragment"
    dataset_uri = dummy_uri if credential_location == "dataset" else "s3://bucket/images.lance"
    sidecar_uri = dummy_uri if credential_location == "sidecar" else "sidecar.parquet"

    with pytest.raises(ValueError, match="userinfo") as raised:
        gpu_key_lookup._build_sidecar_contract_bytes(
            dataset=_FakeIdentityDataset(["alpha"]),
            dataset_uri=dataset_uri,
            dataset_version=4,
            fragment_manifest_sha256="1" * 64,
            total_rows=1,
            key_column="url",
            row_id_column="stable_row_id",
            layout="replicated_sorted",
            partition_files=((sidecar_uri,),),
            storage_options={},
        )

    assert "dummy-pass" not in str(raised.value)
    assert "dummy-token" not in str(raised.value)


def _install_fake_cudf(
    monkeypatch: pytest.MonkeyPatch,
    frames: dict[str, _FakeFrame],
) -> None:
    def read_parquet(paths: list[str], **_: object) -> _FakeFrame:
        assert len(paths) == 1
        return frames[paths[0]]

    monkeypatch.setitem(sys.modules, "cudf", SimpleNamespace(read_parquet=read_parquet))
    monkeypatch.setattr(gpu_key_lookup, "_runtime_mpf_hash_partitioning_contract", lambda: _PARTITIONING)


def _write_partition(path: Path, keys: list[str], stable_ids: list[int]) -> None:
    pq.write_table(
        pa.table(
            {
                "url": pa.array(keys, type=pa.string()),
                "stable_row_id": pa.array(stable_ids, type=pa.uint64()),
            }
        ),
        path,
    )


@pytest.mark.parametrize("key_type", [pa.string(), pa.large_string()], ids=("string", "large_string"))
def test_sorted_sidecar_identity_digest_preserves_pinned_key_schema(key_type: pa.DataType) -> None:
    stable_keys = ["alpha", "beta", "gamma", "delta"]
    identity = pa.table(
        {
            "url": pa.array(["gamma", "alpha", "delta", "beta"], type=key_type),
            "stable_row_id": pa.array([2, 0, 3, 1], type=pa.uint64()),
        }
    )
    dataset = _FakeIdentityDataset(stable_keys, key_type)

    actual = gpu_key_lookup._sorted_sidecar_identity_sha256(
        identity,
        key_column="url",
        row_id_column="stable_row_id",
        key_field=dataset.schema.field("url"),
        total_rows=len(stable_keys),
    )

    assert actual == _key_stable_ordinal_sha256(dataset, "url", total_rows=len(stable_keys))


def test_sorted_sidecar_identity_digest_normalizes_chunked_take_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable_keys = ["alpha", "beta", "gamma", "delta"]
    chunked_keys = pa.chunked_array(
        [pa.array(["gamma", "alpha"]), pa.array(["delta", "beta"])],
        type=pa.string(),
    )
    identity = pa.table(
        {
            "url": chunked_keys,
            "stable_row_id": pa.chunked_array(
                [pa.array([2, 0], type=pa.uint64()), pa.array([3, 1], type=pa.uint64())]
            ),
        }
    )
    dataset = _FakeIdentityDataset(stable_keys)
    take_sources: list[pa.Array | pa.ChunkedArray] = []
    original_take = gpu_key_lookup.pc.take

    def recording_take(
        values: pa.Array | pa.ChunkedArray,
        indices: pa.Array,
    ) -> pa.Array | pa.ChunkedArray:
        take_sources.append(values)
        return original_take(values, indices)

    monkeypatch.setattr(gpu_key_lookup.pc, "take", recording_take)

    take_source = gpu_key_lookup._offset_safe_key_take_source(chunked_keys, pa.string())
    actual = gpu_key_lookup._sorted_sidecar_identity_sha256(
        identity,
        key_column="url",
        row_id_column="stable_row_id",
        key_field=dataset.schema.field("url"),
        total_rows=len(stable_keys),
    )

    assert isinstance(take_source, pa.LargeStringArray)
    assert take_source.to_pylist() == chunked_keys.to_pylist()
    assert all(isinstance(source, pa.Array) for source in take_sources)
    assert [source.type for source in take_sources] == [pa.uint64(), pa.large_string()]
    assert actual == _key_stable_ordinal_sha256(dataset, "url", total_rows=len(stable_keys))


def test_hash_sidecar_contract_pins_exact_mpf_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "partition-0.parquet"
    second = tmp_path / "partition-1.parquet"
    _write_partition(first, ["alpha"], [0])
    _write_partition(second, ["beta"], [1])
    _install_fake_cudf(
        monkeypatch,
        {
            str(first): _FakeFrame(["alpha"], [0]),
            str(second): _FakeFrame(["beta"], [1]),
        },
    )

    raw, digest = gpu_key_lookup._build_sidecar_contract_bytes(
        dataset=_FakeIdentityDataset(["alpha", "beta"]),
        dataset_uri="s3://bucket/images.lance",
        dataset_version=4,
        fragment_manifest_sha256="1" * 64,
        total_rows=2,
        key_column="url",
        row_id_column="stable_row_id",
        layout="hash_partitioned",
        partition_files=((str(first),), (str(second),)),
        storage_options={},
    )

    payload = json.loads(raw)
    assert payload["partition_count"] == 2
    assert payload["partitioning"] == _PARTITIONING.to_payload()
    assert len(payload["key_stable_ordinal_sha256"]) == 64
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(raw)
    contract = gpu_key_lookup._load_and_validate_sidecar_contract(
        manifest_uri=str(manifest),
        manifest_sha256=digest,
        dataset_uri="s3://bucket/images.lance",
        dataset_version=4,
        fragment_manifest_sha256="1" * 64,
        total_rows=2,
        key_column="url",
        row_id_column="stable_row_id",
        layout="hash_partitioned",
        partition_files=((str(first),), (str(second),)),
        storage_options={},
    )
    assert contract.partitioning == _PARTITIONING
    assert contract.key_stable_ordinal_sha256 == payload["key_stable_ordinal_sha256"]


def test_hash_sidecar_contract_rejects_runtime_version_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "partition-0.parquet"
    _write_partition(first, ["alpha"], [0])
    _install_fake_cudf(monkeypatch, {str(first): _FakeFrame(["alpha"], [0])})
    raw, _ = gpu_key_lookup._build_sidecar_contract_bytes(
        dataset=_FakeIdentityDataset(["alpha"]),
        dataset_uri="images",
        dataset_version=4,
        fragment_manifest_sha256="2" * 64,
        total_rows=1,
        key_column="url",
        row_id_column="stable_row_id",
        layout="hash_partitioned",
        partition_files=((str(first),),),
        storage_options={},
    )
    payload = json.loads(raw)
    payload["partitioning"]["rapidsmpf_version"] = "26.6.1"
    mutated = gpu_key_lookup._canonical_json_bytes(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(mutated)

    with pytest.raises(ValueError, match="partitioning"):
        gpu_key_lookup._load_and_validate_sidecar_contract(
            manifest_uri=str(manifest),
            manifest_sha256=hashlib.sha256(mutated).hexdigest(),
            dataset_uri="images",
            dataset_version=4,
            fragment_manifest_sha256="2" * 64,
            total_rows=1,
            key_column="url",
            row_id_column="stable_row_id",
            layout="hash_partitioned",
            partition_files=((str(first),),),
            storage_options={},
        )


def test_sidecar_builder_rejects_permuted_key_to_ordinal_identity(tmp_path: Path) -> None:
    sidecar = tmp_path / "permuted.parquet"
    _write_partition(sidecar, ["alpha", "beta"], [1, 0])

    with pytest.raises(ValueError, match="key-to-stable-ordinal identity"):
        gpu_key_lookup._build_sidecar_contract_bytes(
            dataset=_FakeIdentityDataset(["alpha", "beta"]),
            dataset_uri="images",
            dataset_version=4,
            fragment_manifest_sha256="3" * 64,
            total_rows=2,
            key_column="url",
            row_id_column="stable_row_id",
            layout="replicated_sorted",
            partition_files=((str(sidecar),),),
            storage_options={},
        )


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        (
            {
                "part-0": _FakeFrame(["alpha"], [1]),
                "part-1": _FakeFrame(["beta"], [1]),
            },
            "owned by another RAPIDS-MPF partition",
        ),
        (
            {
                "part-0": _FakeFrame(["duplicate", "duplicate"], [0, 0]),
                "part-1": _FakeFrame(["beta"], [1]),
            },
            "duplicate keys",
        ),
        (
            {
                "part-0": _FakeFrame(["cross-partition"], [0]),
                "part-1": _FakeFrame(["cross-partition"], [0]),
            },
            "owned by another RAPIDS-MPF partition",
        ),
    ],
    ids=("misplaced", "duplicate-in-owner", "duplicate-across-owners"),
)
def test_hash_sidecar_builder_rejects_invalid_partition_ownership(
    frames: dict[str, _FakeFrame],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_cudf(monkeypatch, frames)

    with pytest.raises(ValueError, match=match):
        gpu_key_lookup._validate_hash_partitioned_sidecars(
            partition_files=(("part-0",), ("part-1",)),
            key_column="url",
            storage_options={},
        )


@pytest.mark.gpu
def test_partition_ids_match_actual_rapidsmpf_for_unicode_keys() -> None:
    import cudf
    import rmm.mr
    from rapidsmpf.buffer.resource import BufferResource
    from rapidsmpf.integrations.cudf.partition import partition_and_pack, unpack_and_concat
    from rapidsmpf.utils.cudf import cudf_to_pylibcudf_table, pylibcudf_to_cudf_dataframe
    from rmm.pylibrmm.stream import DEFAULT_STREAM

    keys = [
        "",
        "ascii",
        "caf\u00e9",
        "\u6771\u4eac",
        "\ud55c\uad6d\uc5b4",
        "emoji-\U0001f680",
        "combining-e\u0301",
        "nul-\x00-byte",
    ]
    frame = cudf.DataFrame({"url": keys, "row": np.arange(len(keys), dtype=np.int32)})
    br = BufferResource(rmm.mr.CudaMemoryResource())
    for partition_count in (1, 2, 3, 7, 16):
        expected = gpu_key_lookup._mpf_partition_ids(frame, "url", partition_count).to_arrow().to_pylist()
        packed = partition_and_pack(
            cudf_to_pylibcudf_table(frame),
            columns_to_hash=(0,),
            num_partitions=partition_count,
            br=br,
            stream=DEFAULT_STREAM,
        )
        actual = [-1] * len(keys)
        for partition_id, partition in packed.items():
            unpacked = pylibcudf_to_cudf_dataframe(
                unpack_and_concat((partition,), br=br, stream=DEFAULT_STREAM),
                column_names=["url", "row"],
            )
            for row in unpacked["row"].to_arrow().to_pylist():
                actual[row] = partition_id
        assert actual == expected
