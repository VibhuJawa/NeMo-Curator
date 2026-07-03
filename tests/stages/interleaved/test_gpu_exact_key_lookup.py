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

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved import GpuExactKeyLookupStage
from nemo_curator.tasks import InterleavedBatch


def _batch(keys: list[str | None], **extra: pa.Array) -> InterleavedBatch:
    count = len(keys)
    columns: dict[str, pa.Array] = {
        "sample_id": pa.array([f"sample-{index}" for index in range(count)]),
        "position": pa.array(list(range(count)), type=pa.int32()),
        "modality": pa.array(["image"] * count),
        "source_ref": pa.array(keys, type=pa.string()),
    }
    columns.update(extra)
    return InterleavedBatch(dataset_name="interleaved", data=pa.table(columns))


class _FakeMatcher:
    # cuDF materializes Parquet UTF-8 columns as Arrow large_string even when
    # the corresponding input InterleavedBatch uses Arrow string.
    reference_type = pa.large_string()

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def match(self, keys: pa.Array) -> SimpleNamespace:
        values = keys.to_pylist()
        self.calls.append(values)
        return SimpleNamespace(
            matched=np.array([key in {"a", "c"} for key in values], dtype=np.bool_),
            transfer_seconds=0.0,
            probe_seconds=0.0,
            gather_seconds=0.0,
        )

    def close(self) -> None:
        return


def test_gpu_exact_key_lookup_batches_tasks_and_preserves_boundaries() -> None:
    stage = GpuExactKeyLookupStage(reference_files=["reference.parquet"], reference_key_column="url")
    matcher = _FakeMatcher()
    stage._matcher = matcher

    tasks = [
        _batch(["a", None, ""]),
        _batch(["b", "a"]),
    ]
    outputs = stage.process_batch(np.array(tasks, dtype=object))

    assert matcher.calls == [["a", "b", "a"]]
    assert outputs[0].to_pyarrow()["image_present"].to_pylist() == [True, None, None]
    assert outputs[1].to_pyarrow()["image_present"].to_pylist() == [False, True]
    assert outputs[0].to_pyarrow()["source_ref"].to_pylist() == ["a", None, ""]


def test_gpu_exact_key_lookup_validates_configuration_and_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GpuExactKeyLookupStage(reference_files=[], reference_key_column="url")
    with pytest.raises(ValueError, match="must not contain duplicates"):
        GpuExactKeyLookupStage(reference_files=["same", "same"], reference_key_column="url")
    with pytest.raises(ValueError, match="interval"):
        GpuExactKeyLookupStage(reference_files=["reference"], reference_key_column="url", load_factor=0.0)

    stage = GpuExactKeyLookupStage(reference_files=["reference.parquet"], reference_key_column="url")
    stage._matcher = _FakeMatcher()
    with pytest.raises(ValueError, match="already exists"):
        stage.process(_batch(["a"], image_present=pa.array([True], type=pa.bool_())))

    integer_batch = InterleavedBatch(
        dataset_name="interleaved",
        data=pa.table(
            {
                "sample_id": ["sample"],
                "position": pa.array([0], type=pa.int32()),
                "modality": ["image"],
                "source_ref": pa.array([1], type=pa.int64()),
            }
        ),
    )
    with pytest.raises(TypeError, match="Input key column has type int64"):
        stage.process(integer_batch)
