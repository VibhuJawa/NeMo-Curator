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

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from nemo_curator.tasks import DocumentBatch


@lru_cache
def _tutorial() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "tutorials"
        / "text"
        / "llm-as-a-judge"
        / "classify_cc_for_pretraining.py"
    )
    spec = importlib.util.spec_from_file_location("classify_cc_for_pretraining_test", path)
    if spec is None or spec.loader is None:
        msg = f"Could not load tutorial module from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_parquet(path: Path, rows: int) -> None:
    pd.DataFrame(
        {"url": [f"u-{index}" for index in range(rows)], "text": [f"t-{index}" for index in range(rows)]}
    ).to_parquet(path)


def test_proportional_sample_allocation_is_exact_and_row_proportional(tmp_path: Path) -> None:
    tutorial = _tutorial()
    files = [tmp_path / "small.parquet", tmp_path / "medium.parquet", tmp_path / "large.parquet"]
    for path, rows in zip(files, (10, 30, 60), strict=True):
        _write_parquet(path, rows)

    plan = tutorial.allocate_proportional_sample([str(path) for path in files], 20, 7)

    assert plan.files == [str(path) for path in files]
    assert plan.sample_rows_by_source == {str(files[0]): 2, str(files[1]): 6, str(files[2]): 12}
    assert plan.source_rows_by_file == {str(files[0]): 10, str(files[1]): 30, str(files[2]): 60}


def test_sampling_stage_records_reproducible_inverse_probability_weights(tmp_path: Path) -> None:
    tutorial = _tutorial()
    source = str(tmp_path / "source.parquet")
    data = pd.DataFrame({"url": [f"u-{index}" for index in range(10)], "text": [f"t-{index}" for index in range(10)]})
    stage = tutorial.DeterministicSampleStage(
        rows_per_partition=0,
        seed=11,
        sample_rows_by_source={source: 2},
        source_rows_by_file={source: 10},
    )
    batch = DocumentBatch(dataset_name="cc", data=data, _metadata={"source_files": [source]})

    first = stage.process(batch).to_pandas()
    second = stage.process(batch).to_pandas()

    assert first["url"].tolist() == second["url"].tolist()
    assert len(first) == 2
    assert first["_eval_source_file"].unique().tolist() == [source]
    assert first["_eval_inclusion_probability"].unique().tolist() == [0.2]
    assert first["_eval_sample_weight"].unique().tolist() == [5.0]
    assert first["_eval_sample_weight"].sum() == 10.0


def test_sampling_stage_detects_changed_source_row_count(tmp_path: Path) -> None:
    tutorial = _tutorial()
    source = str(tmp_path / "source.parquet")
    stage = tutorial.DeterministicSampleStage(
        rows_per_partition=0,
        sample_rows_by_source={source: 2},
        source_rows_by_file={source: 9},
    )
    batch = DocumentBatch(
        dataset_name="cc",
        data=pd.DataFrame({"url": ["u"] * 10, "text": ["t"] * 10}),
        _metadata={"source_files": [source]},
    )

    with pytest.raises(ValueError, match="row count changed"):
        stage.process(batch)


def test_proportional_sample_rejects_oversized_target(tmp_path: Path) -> None:
    tutorial = _tutorial()
    source = tmp_path / "source.parquet"
    _write_parquet(source, 3)

    with pytest.raises(ValueError, match="exceeds"):
        tutorial.allocate_proportional_sample([str(source)], 4, 0)
