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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarking.scripts import gpu_lance_grouped_payload_canary as canary
from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class _FakeManifest:
    schema: pa.Schema
    total_rows: int
    total_arrow_nbytes: int
    peak_active_bytes: int
    peak_bounded_active_bytes: int
    files: tuple[str, ...]
    oversized_rows: tuple[object, ...]
    sha256: str


class _FakeSpool:
    def __init__(self, schema: pa.Schema, tables: list[pa.Table], *, sha256: str) -> None:
        self.schema = schema
        self._tables = tables
        self._manifest = _FakeManifest(
            schema=schema,
            total_rows=sum(table.num_rows for table in tables),
            total_arrow_nbytes=sum(table.nbytes for table in tables),
            peak_active_bytes=max(table.nbytes for table in tables),
            peak_bounded_active_bytes=max(table.nbytes for table in tables),
            files=tuple(f"part-{index}" for index in range(len(tables))),
            oversized_rows=(),
            sha256=sha256,
        )

    def finish(self) -> _FakeManifest:
        return self._manifest

    def iter_tables(self) -> Iterator[pa.Table]:
        return iter(self._tables)


def _query_file(tmp_path: Path) -> tuple[Path, pa.Table]:
    table = pa.table(
        {
            "source_ref": pa.array(["a", "b", "c", "d"]),
            STABLE_ROW_ID: pa.array([3, 1, 3, 2], type=pa.uint64()),
            "expected_width": pa.array([10, 20, 30, 40], type=pa.int32()),
        }
    )
    path = tmp_path / "query.parquet"
    pq.write_table(table, path)
    return path, table


def _contract(tmp_path: Path) -> canary.QueryContract:
    path, table = _query_file(tmp_path)
    harness_digest, _ = canary._harness_logical_digest(table, "source_ref")
    stable_digest = canary._stable_id_sequence_digest(table[STABLE_ROW_ID])
    return canary.load_query_contract(
        path,
        source_ref_column="source_ref",
        stable_id_column=STABLE_ROW_ID,
        expected_file_sha256=canary._file_sha256(path),
        expected_harness_logical_digest_sha256=harness_digest,
        expected_stable_id_sequence_digest_sha256=stable_digest,
        expected_rows=4,
        expected_unique_stable_ids=3,
        block_count=2,
        rows_per_block=2,
    )


def _spool_table(
    schema: pa.Schema,
    positions: list[int],
    stable_ids: list[int],
    payloads: list[bytes],
) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(positions, type=pa.uint64()),
            pa.array(positions, type=pa.uint64()),
            pa.array(stable_ids, type=pa.uint64()),
            pa.array(payloads, type=pa.large_binary()),
        ],
        schema=schema,
    )


def test_query_contract_pins_file_logical_sequence_counts_and_contiguous_blocks(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    plans = canary.build_coordinate_plans(contract)

    assert contract.rows == 4
    assert contract.unique_stable_ids == 3
    assert contract.expected_columns == ("expected_width",)
    assert [block["source_ref"].to_pylist() for block in contract.blocks] == [["a", "b"], ["c", "d"]]
    assert [plan[DOCUMENT_POSITION].to_pylist() for plan in plans] == [[0, 1], [2, 3]]
    assert [plan[DOCUMENT_ROWADDR].to_pylist() for plan in plans] == [[0, 1], [2, 3]]
    assert [plan[STABLE_ROW_ID].to_pylist() for plan in plans] == [[3, 1], [3, 2]]

    path = contract.path
    with pytest.raises(ValueError, match="sequence digest"):
        canary.load_query_contract(
            path,
            source_ref_column="source_ref",
            stable_id_column=STABLE_ROW_ID,
            expected_file_sha256=contract.file_sha256,
            expected_harness_logical_digest_sha256=contract.harness_logical_digest_sha256,
            expected_stable_id_sequence_digest_sha256="0" * 64,
            expected_rows=4,
            expected_unique_stable_ids=3,
            block_count=2,
            rows_per_block=2,
        )


def test_stable_id_sequence_digest_has_a_fixed_length_framed_encoding() -> None:
    stable_ids = pa.array([1, 2], type=pa.uint64())

    assert canary._stable_id_sequence_digest(stable_ids) == (
        "cb684e4e5eb52004fedca2d2218a58cf835e3a6abd2e932f8d8d2aa9f7a8048c"
    )


def test_streamed_spool_validation_restores_manifest_order_without_payload_concatenation(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    plans = canary.build_coordinate_plans(contract)
    schema = pa.schema([*plans[0].schema, pa.field("image", pa.large_binary(), nullable=True)])
    payloads = [b"payload-a", b"payload-bb", b"payload-ccc", b"payload-dddd"]
    expected_digest, expected_bytes = canary.historical_payload_oracle(["a", "b", "c", "d"], payloads)
    spools = [
        _FakeSpool(
            schema,
            [
                _spool_table(schema, [1], [1], [payloads[1]]),
                _spool_table(schema, [0], [3], [payloads[0]]),
            ],
            sha256="1" * 64,
        ),
        _FakeSpool(
            schema,
            [_spool_table(schema, [3, 2], [2, 3], [payloads[3], payloads[2]])],
            sha256="2" * 64,
        ),
    ]

    result = canary.validate_spools_and_payload_oracle(
        spools,
        contract,
        plans,
        expected_schema=schema,
        image_column="image",
        expected_payload_digest_sha256=expected_digest,
        expected_payload_bytes=expected_bytes,
    )

    assert result["rows"] == 4
    assert result["payload_bytes"] == expected_bytes
    assert result["payload_digest_sha256"] == expected_digest
    assert result["whole_output_concatenated"] is False
    assert [block["tables_streamed"] for block in result["blocks"]] == [2, 1]

    corrupt = _FakeSpool(
        schema,
        [_spool_table(schema, [0, 1], [3, 999], payloads[:2])],
        sha256="3" * 64,
    )
    with pytest.raises(RuntimeError, match="stable IDs differ"):
        canary.validate_spools_and_payload_oracle(
            [corrupt, spools[1]],
            contract,
            plans,
            expected_schema=schema,
            image_column="image",
            expected_payload_digest_sha256=expected_digest,
            expected_payload_bytes=expected_bytes,
        )


def test_storage_options_persist_only_keys_and_identity_hash(tmp_path: Path) -> None:
    path = tmp_path / "storage.json"
    path.write_text(json.dumps({"region": "test-region", "endpoint": "https://object.test"}))

    options, identity = canary._load_storage_options(path)

    assert options == {"region": "test-region", "endpoint": "https://object.test"}
    assert identity.keys == ("endpoint", "region")
    assert len(identity.sha256) == 64
    assert "test-region" not in json.dumps({"keys": identity.keys, "sha256": identity.sha256})

    path.write_text(json.dumps({"secret_access_key": "must-not-load"}))
    with pytest.raises(ValueError, match="process environment"):
        canary._load_storage_options(path)


def test_atomic_result_publication_refuses_relative_or_existing_paths(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    canary._atomic_json(result, {"status": "completed", "rows": 4})

    assert json.loads(result.read_text()) == {"rows": 4, "status": "completed"}
    assert not list(tmp_path.glob(".result.json.*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to replace"):
        canary._atomic_json(result, {"status": "replacement"})
    with pytest.raises(ValueError, match="absolute"):
        canary._atomic_json(Path("relative-result.json"), {"status": "invalid"})


def test_peak_rss_cap_is_inclusive() -> None:
    canary._validate_peak_rss(10, 10)

    with pytest.raises(MemoryError, match="11 exceeded the caller cap 10"):
        canary._validate_peak_rss(11, 10)


def test_parser_rejects_credential_bearing_image_uri(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        canary.build_parser().parse_args(
            [
                "--query-parquet",
                "/workspace/query.parquet",
                "--query-file-sha256",
                "0" * 64,
                "--harness-logical-digest-sha256",
                "0" * 64,
                "--stable-id-sequence-digest-sha256",
                "0" * 64,
                "--expected-rows",
                str(canary.EXPECTED_ROWS),
                "--expected-unique-stable-ids",
                "1",
                "--image-lance-uri",
                "s3://user:password@example.invalid/data?token=value",
                "--image-lance-version",
                "4",
                "--image-fragment-manifest-sha256",
                "0" * 64,
                "--storage-options-file",
                "/workspace/storage.json",
                "--max-peak-rss-bytes",
                str(400 * 1024**3),
                "--expected-payload-digest-sha256",
                "0" * 64,
                "--expected-payload-bytes",
                "1",
                "--spool-root",
                "/workspace/spool",
                "--result",
                "/workspace/result.json",
                "--cleanup-policy",
                "always",
            ]
        )
    assert "password" not in capsys.readouterr().err
