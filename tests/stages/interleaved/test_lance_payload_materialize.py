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
from typing import TYPE_CHECKING

import lance
import pyarrow as pa
import pytest
from lance_ray import LanceStableIdPayloadConfig, LanceStableIdPayloadStreamer

from nemo_curator.stages.interleaved.lance_coordinate_plan import (
    DOCUMENT_POSITION,
    DOCUMENT_ROWADDR,
    STABLE_ROW_ID,
    lance_coordinate_plan_schema,
)
from nemo_curator.stages.interleaved.lance_payload_materialize import materialize_lance_payload_to_spool
from nemo_curator.stages.interleaved.lance_payload_spool import PayloadSpool

if TYPE_CHECKING:
    from pathlib import Path


_PAYLOAD_SCHEMA = pa.schema(
    [
        pa.field("image", pa.large_binary(), nullable=False, metadata={b"content-type": b"image/jpeg"}),
        pa.field("width", pa.int32(), nullable=False, metadata={b"units": b"pixels"}),
    ]
)
_SPOOL_SCHEMA = pa.schema(
    [
        pa.field(DOCUMENT_ROWADDR, pa.uint64(), nullable=False),
        pa.field(DOCUMENT_POSITION, pa.uint64(), nullable=False),
        pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False),
        *_PAYLOAD_SCHEMA,
    ],
    metadata={b"purpose": b"stable-id-materialize-test"},
)


def _coordinate_plan(stable_ids: list[int | None]) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(range(100, 100 + len(stable_ids)), type=pa.uint64()),
            pa.array(range(len(stable_ids)), type=pa.uint64()),
            pa.array(stable_ids, type=pa.uint64()),
        ],
        schema=lance_coordinate_plan_schema(allow_missing=True),
    )


def _payload_spool(tmp_path: Path, name: str = "payload") -> PayloadSpool:
    return PayloadSpool(
        tmp_path / name,
        _SPOOL_SCHEMA,
        target_bytes=4096,
        bucket_rows=100,
        stable_id_column=STABLE_ROW_ID,
        document_position_column=DOCUMENT_POSITION,
    )


def _real_payload_streamer(tmp_path: Path, *, batch_size: int = 2) -> LanceStableIdPayloadStreamer:
    names = ("zero", "one", "two", "three", "four", "five", "six")
    table = pa.Table.from_arrays(
        [
            pa.array([name.encode() for name in names], type=pa.large_binary()),
            pa.array([index * 10 for index in range(len(names))], type=pa.int32()),
        ],
        schema=_PAYLOAD_SCHEMA,
    )
    dataset = lance.write_dataset(
        table,
        str(tmp_path / "images.lance"),
        enable_stable_row_ids=True,
    )
    return LanceStableIdPayloadStreamer(
        LanceStableIdPayloadConfig(
            dataset_uri=dataset.uri,
            dataset_version=dataset.version,
            expected_rows=table.num_rows,
            columns={"image": "image", "width": "width"},
            fetch_batch_size=batch_size,
            io_threads=2,
            max_pending_fetch_batches=2,
        ),
        dataset=dataset,
    )


def _final_reader_metrics(  # noqa: PLR0913
    *,
    rows: int,
    batches: int,
    payload_bytes: int,
    output_rows: int | None = None,
    reordered_batches: int = 0,
    pending_limit: int = 2,
) -> dict[str, int | float | bool]:
    output_rows = rows if output_rows is None else output_rows
    peak_in_flight = min(pending_limit, batches)
    peak_ready = min(pending_limit, batches)
    peak_producer = min(2 * pending_limit, batches)
    peak_total = min(2 * pending_limit + 1, batches)
    return {
        "stream_complete": True,
        "input_stable_rows": rows,
        "stream_output_rows": output_rows,
        "payload_take_rows": rows,
        "payload_batches_planned": batches,
        "payload_batches_emitted": batches,
        "payload_read_calls": batches,
        "payload_bytes": payload_bytes,
        "max_pending_payload_reads": peak_in_flight,
        "max_retained_payload_batches": peak_total,
        "peak_in_flight_payload_reads": peak_in_flight,
        "peak_running_payload_reads": peak_in_flight,
        "peak_ready_payload_batches": peak_ready,
        "peak_producer_retained_payload_batches": peak_producer,
        "peak_total_retained_payload_batches": peak_total,
        "retained_payload_batch_upper_bound": 2 * pending_limit + 1,
        "consumer_held_payload_batch_limit": 1,
        "completion_order_output": True,
        "completion_order_reordered_batches": reordered_batches,
        "batch_stable_ids_sorted": True,
        "exact_operation_coverage": True,
        "sparse_calls_avoided": max(0, rows - batches),
        "payload_read_call_sum_seconds": float(batches),
        "payload_read_envelope_seconds": float(batches),
        "lance_read_iops": batches,
        "lance_read_bytes": payload_bytes,
    }


def _payload_batch(stable_ids: list[int], *, schema: pa.Schema | None = None) -> pa.Table:
    payload_schema = _PAYLOAD_SCHEMA if schema is None else schema
    arrays: list[pa.Array] = [pa.array(stable_ids, type=pa.uint64())]
    for field in payload_schema:
        if field.name == "image":
            arrays.append(pa.array([f"image-{stable_id}".encode() for stable_id in stable_ids], type=field.type))
        elif field.name == "width":
            arrays.append(pa.array([stable_id * 10 for stable_id in stable_ids], type=field.type))
        else:
            arrays.append(pa.nulls(len(stable_ids), type=field.type))
    return pa.Table.from_arrays(
        arrays,
        schema=pa.schema([pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False), *payload_schema]),
    )


class _ScriptedPayloadStreamer:
    def __init__(self, batches: list[pa.Table], *, reordered_batches: int = 0) -> None:
        self._batches = batches
        self._reordered_batches = reordered_batches
        self.last_metrics: dict[str, int | float | bool] = {}
        self.iterator_closed = False

    def iter_stable_row_ids(self, values: pa.Array):  # noqa: ANN202
        requested_rows = len(values)
        output_rows = 0
        payload_bytes = 0
        try:
            for batch in self._batches:
                output_rows += batch.num_rows
                payload_bytes += sum(batch[name].nbytes for name in batch.column_names if name != STABLE_ROW_ID)
                yield batch
        finally:
            self.iterator_closed = True
        self.last_metrics = _final_reader_metrics(
            rows=requested_rows,
            output_rows=output_rows,
            batches=len(self._batches),
            payload_bytes=payload_bytes,
            reordered_batches=self._reordered_batches,
        )

    def close(self) -> None:
        pass


def test_materialize_uses_lance_ray_reader_and_preserves_exact_fanout(tmp_path: Path) -> None:
    coordinate_plan = _coordinate_plan([5, 1, None, 4, 1, 6, 2, 3])
    reader = _real_payload_streamer(tmp_path)
    spool = _payload_spool(tmp_path)
    try:
        metrics = materialize_lance_payload_to_spool(
            reader,
            coordinate_plan,
            ("image", "width"),
            spool,
        )
        output = spool.read_all().sort_by([(DOCUMENT_POSITION, "ascending")])

        assert output.schema.equals(_SPOOL_SCHEMA, check_metadata=True)
        assert output.schema.field("image").metadata == {b"content-type": b"image/jpeg"}
        assert output.schema.field("width").metadata == {b"units": b"pixels"}
        assert output[DOCUMENT_ROWADDR].to_pylist() == [100, 101, 103, 104, 105, 106, 107]
        assert output[DOCUMENT_POSITION].to_pylist() == [0, 1, 3, 4, 5, 6, 7]
        assert output[STABLE_ROW_ID].to_pylist() == [5, 1, 4, 1, 6, 2, 3]
        payloads = output["image"].to_pylist()
        assert payloads == [b"five", b"one", b"four", b"one", b"six", b"two", b"three"]
        assert output["width"].to_pylist() == [50, 10, 40, 10, 60, 20, 30]
        assert (
            hashlib.sha256(b"".join(payloads)).hexdigest() == hashlib.sha256(b"fiveonefouronesixtwothree").hexdigest()
        )

        assert metrics["logical_rows"] == 7
        assert metrics["unique_rows"] == 6
        assert metrics["null_rows_skipped"] == 1
        assert metrics["duplicate_fanout"] == pytest.approx(7 / 6)
        assert metrics["input_stable_rows"] == metrics["take_rows"] == metrics["unique_rows"]
        assert metrics["stream_output_rows"] == metrics["unique_rows"]
        assert metrics["payload_batches_emitted"] == metrics["take_calls"] == 3
        assert metrics["payload_batches_planned"] == metrics["payload_batches_emitted"]
        assert metrics["payload_read_calls"] == metrics["take_calls"]
        assert metrics["payload_bytes"] == metrics["actual_payload_bytes"]
        assert metrics["completion_order_output"] is True
        assert metrics["batch_stable_ids_sorted"] is True
        assert metrics["exact_operation_coverage"] is True
        assert metrics["max_pending_payload_reads"] == metrics["peak_pending"] <= 2
        assert metrics["max_retained_payload_batches"] == metrics["peak_retained_batches"] <= 5
        assert metrics["peak_in_flight_payload_reads"] <= 2
        assert metrics["peak_ready_payload_batches"] <= 2
        assert metrics["peak_producer_retained_payload_batches"] <= 4
        assert metrics["peak_total_retained_payload_batches"] <= 5
        assert metrics["retained_payload_batch_upper_bound"] == 5
        assert metrics["private_take_call_seconds_sum"] == metrics["payload_read_call_sum_seconds"]
        assert metrics["private_take_execution_envelope_seconds"] == metrics["payload_read_envelope_seconds"]
        assert reader.last_metrics["stream_complete"] is True
        assert reader.last_metrics["payload_bytes"] == metrics["actual_payload_bytes"]
    finally:
        reader.close()
        spool.cleanup()


def test_materialize_maps_deliberately_out_of_order_results_by_stable_id(tmp_path: Path) -> None:
    coordinate_plan = _coordinate_plan([5, 1, None, 4, 1, 6, 2, 3])
    reader = _ScriptedPayloadStreamer(
        [_payload_batch([4, 5, 6]), _payload_batch([1, 2, 3])],
        reordered_batches=2,
    )
    spool = _payload_spool(tmp_path)
    try:
        metrics = materialize_lance_payload_to_spool(
            reader,
            coordinate_plan,
            ("image", "width"),
            spool,
        )
        output = spool.read_all().sort_by([(DOCUMENT_POSITION, "ascending")])

        assert output[DOCUMENT_ROWADDR].to_pylist() == [100, 101, 103, 104, 105, 106, 107]
        assert output[STABLE_ROW_ID].to_pylist() == [5, 1, 4, 1, 6, 2, 3]
        payloads = output["image"].to_pylist()
        assert payloads == [b"image-5", b"image-1", b"image-4", b"image-1", b"image-6", b"image-2", b"image-3"]
        assert output["width"].to_pylist() == [50, 10, 40, 10, 60, 20, 30]
        assert (
            hashlib.sha256(b"".join(payloads)).hexdigest()
            == hashlib.sha256(b"image-5image-1image-4image-1image-6image-2image-3").hexdigest()
        )
        assert metrics["completion_order_reordered_batches"] == 2
        assert metrics["unique_rows"] == 6
        assert metrics["logical_rows"] == 7
        assert metrics["scatter_input_rows"] == 7
        assert reader.iterator_closed is True
    finally:
        spool.cleanup()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate", "overlapping or duplicate"),
        ("overlap", "overlapping or duplicate"),
        ("missing", "first missing stable row ID is 2"),
        ("unknown", "unknown stable row ID 9"),
        ("noncontiguous", "not one contiguous interval"),
    ],
)
def test_materialize_rejects_invalid_completion_order_coverage(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    scripted_ids = {
        "duplicate": [[3, 4], [3, 4], [1, 2]],
        "overlap": [[2, 3], [3, 4], [1]],
        "missing": [[3, 4], [1]],
        "unknown": [[9], [1, 2, 3, 4]],
        "noncontiguous": [[1, 3], [2, 4]],
    }[case]
    reader = _ScriptedPayloadStreamer([_payload_batch(ids) for ids in scripted_ids])
    spool = _payload_spool(tmp_path, case)

    with pytest.raises(RuntimeError, match=message):
        materialize_lance_payload_to_spool(
            reader,
            _coordinate_plan([1, 2, 3, 4]),
            ("image", "width"),
            spool,
        )

    assert reader.iterator_closed is True
    with pytest.raises(RuntimeError, match="finish must be called"):
        spool.iter_tables()
    spool.cleanup()
    assert not spool.root.exists()


@pytest.mark.parametrize("schema_error", ["metadata", "nullability"])
def test_materialize_rejects_completion_batch_schema_errors(tmp_path: Path, schema_error: str) -> None:
    if schema_error == "metadata":
        image_field = pa.field("image", pa.large_binary(), nullable=False, metadata={b"content-type": b"image/png"})
    else:
        image_field = pa.field("image", pa.large_binary(), nullable=True, metadata={b"content-type": b"image/jpeg"})
    invalid_schema = pa.schema([image_field, _PAYLOAD_SCHEMA.field("width")])
    reader = _ScriptedPayloadStreamer([_payload_batch([1, 2], schema=invalid_schema)])
    spool = _payload_spool(tmp_path, schema_error)

    with pytest.raises(TypeError, match="payload schema"):
        materialize_lance_payload_to_spool(
            reader,
            _coordinate_plan([1, 2]),
            ("image", "width"),
            spool,
        )

    assert reader.iterator_closed is True
    with pytest.raises(RuntimeError, match="finish must be called"):
        spool.iter_tables()
    spool.cleanup()


def test_materialize_closes_partial_reader_and_leaves_spool_unpublished(tmp_path: Path) -> None:
    class _FailingReader:
        def __init__(self) -> None:
            self.last_metrics: dict[str, int | float | bool] = {}
            self.iterator_closed = False

        def iter_stable_row_ids(self, values: pa.Array):  # noqa: ANN202
            assert values.to_pylist() == [1, 2, 3, 4]
            try:
                yield pa.Table.from_arrays(
                    [
                        pa.array([3, 4], type=pa.uint64()),
                        pa.array([b"three", b"four"], type=pa.large_binary()),
                        pa.array([30, 40], type=pa.int32()),
                    ],
                    schema=pa.schema([pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False), *_PAYLOAD_SCHEMA]),
                )
                msg = "injected stable-ID reader failure"
                raise RuntimeError(msg)
            finally:
                self.iterator_closed = True

        def close(self) -> None:
            pass

    reader = _FailingReader()
    spool = _payload_spool(tmp_path)

    with pytest.raises(RuntimeError, match="injected stable-ID reader failure"):
        materialize_lance_payload_to_spool(
            reader,
            _coordinate_plan([1, 2, 3, 4]),
            ("image", "width"),
            spool,
        )

    assert reader.iterator_closed is True
    assert reader.last_metrics == {}
    with pytest.raises(RuntimeError, match="finish must be called"):
        spool.iter_tables()
    spool.cleanup()
    assert not spool.root.exists()


def test_materialize_bounds_duplicate_fanout_before_spool_append(tmp_path: Path) -> None:
    class _OneRowReader:
        def __init__(self) -> None:
            self.last_metrics: dict[str, int | float | bool] = {}

        def iter_stable_row_ids(self, values: pa.Array):  # noqa: ANN202
            assert values.to_pylist() == [1]
            output = pa.Table.from_arrays(
                [
                    pa.array([1], type=pa.uint64()),
                    pa.array([b"x" * 64], type=pa.large_binary()),
                    pa.array([10], type=pa.int32()),
                ],
                schema=pa.schema([pa.field(STABLE_ROW_ID, pa.uint64(), nullable=False), *_PAYLOAD_SCHEMA]),
            )
            yield output
            payload_bytes = output["image"].nbytes + output["width"].nbytes
            self.last_metrics = _final_reader_metrics(rows=1, batches=1, payload_bytes=payload_bytes)

        def close(self) -> None:
            pass

    class _RecordingSpool(PayloadSpool):
        def __init__(self, root: Path) -> None:
            super().__init__(
                root,
                _SPOOL_SCHEMA,
                target_bytes=128,
                bucket_rows=100,
                stable_id_column=STABLE_ROW_ID,
                document_position_column=DOCUMENT_POSITION,
            )
            self.append_sizes: list[int] = []

        def append(self, table: pa.Table) -> None:
            self.append_sizes.append(table.nbytes)
            super().append(table)

    spool = _RecordingSpool(tmp_path / "fanout")
    metrics = materialize_lance_payload_to_spool(
        _OneRowReader(),
        _coordinate_plan([1] * 20),
        ("image", "width"),
        spool,
    )

    assert len(spool.append_sizes) > 1
    assert all(size <= spool.target_bytes for size in spool.append_sizes)
    assert metrics["logical_rows"] == 20
    assert metrics["unique_rows"] == 1
    assert metrics["scatter_input_rows"] == 20
    spool.cleanup()
