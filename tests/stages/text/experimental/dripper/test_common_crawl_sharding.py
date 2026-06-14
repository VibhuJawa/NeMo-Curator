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

"""Tests for Dripper Common Crawl tutorial page sharding."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def common_crawl_main() -> ModuleType:
    if sys.platform != "linux":
        pytest.skip("Common Crawl tutorial only supports Linux")
    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "tutorials/text/dripper-common-crawl/main.py"
    spec = importlib.util.spec_from_file_location("dripper_common_crawl_main_for_tests", module_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Common Crawl tutorial dependencies unavailable: {exc.name}")
    return module


def test_url_host_key_uses_normalized_hostname_not_registrable_domain(common_crawl_main: ModuleType) -> None:
    assert common_crawl_main._url_host_key("https://www.Example.Co.UK:443/path") == "www.example.co.uk"
    assert common_crawl_main._url_host_key("https://blog.example.co.uk/path") == "blog.example.co.uk"
    assert common_crawl_main._url_host_key("example.com/no-scheme") == "example.com"
    assert common_crawl_main._url_host_key(None) == ""
    assert common_crawl_main._host_key_or_row_fallback(None, 7) == "~missing-host-000000000007"


def test_layout_cluster_threshold_default_is_strict_for_common_crawl(
    common_crawl_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = common_crawl_main.parse_args()

    assert args.layout_cluster_threshold == 0.99
    assert args.layout_page_signature_mode == "none"


def test_domain_clustered_shards_group_normalized_hosts(common_crawl_main: ModuleType) -> None:
    tasks = common_crawl_main.build_page_tasks(
        [
            {"url": "https://b.example/1", "html": "b1"},
            {"url": "https://a.example/1", "html": "a1"},
            {"url": "https://b.example/2", "html": "b2"},
            {"url": "https://www.a.example/2", "html": "a2"},
            {"url": None, "html": "missing1"},
            {"url": "", "html": "missing2"},
        ],
        shard_size=2,
        shard_strategy="domain_clustered",
        task_id="task",
        dataset_name="dataset",
    )

    rows = _rows(tasks)

    assert [len(task.to_pandas()) for task in tasks] == [1, 2, 2, 1]
    assert [row["_dripper_row_index"] for row in rows] == [1, 0, 2, 3, 4, 5]
    assert all("_dripper_host_key" not in task.to_pandas().columns for task in tasks)
    assert all("_dripper_html_bytes" not in task.to_pandas().columns for task in tasks)


def test_domain_then_html_bytes_packs_host_chunks_without_exceeding_shard_size(
    common_crawl_main: ModuleType,
) -> None:
    tasks = common_crawl_main.build_page_tasks(
        [
            {"url": "https://a.example/1", "html": b"a" * 100},
            {"url": "https://a.example/2", "html": b"a" * 100},
            {"url": "https://a.example/3", "html": b"a" * 100},
            {"url": "https://b.example/1", "html": b"b"},
            {"url": "https://b.example/2", "html": b"b"},
            {"url": "https://c.example/1", "html": b"c"},
        ],
        shard_size=3,
        shard_strategy="domain_then_html_bytes",
        task_id="task",
        dataset_name="dataset",
    )

    shard_row_indexes = _row_indexes_by_task(tasks)
    flat_row_indexes = [row_index for shard in shard_row_indexes for row_index in shard]

    assert len(tasks) == 2
    assert all(len(shard) <= 3 for shard in shard_row_indexes)
    assert sorted(flat_row_indexes) == [0, 1, 2, 3, 4, 5]
    assert [0, 1, 2] in shard_row_indexes
    assert [3, 4, 5] in shard_row_indexes


def test_domain_complete_shards_never_split_large_hosts(common_crawl_main: ModuleType) -> None:
    tasks = common_crawl_main.build_page_tasks(
        [
            {"url": "https://a.example/1", "html": "a1"},
            {"url": "https://a.example/2", "html": "a2"},
            {"url": "https://a.example/3", "html": "a3"},
            {"url": "https://b.example/1", "html": "b1"},
            {"url": "https://c.example/1", "html": "c1"},
        ],
        shard_size=2,
        shard_strategy="domain_complete",
        task_id="task",
        dataset_name="dataset",
    )

    shard_row_indexes = _row_indexes_by_task(tasks)

    assert [0, 1, 2] in shard_row_indexes
    assert [3, 4] in shard_row_indexes
    assert sorted(row for shard in shard_row_indexes for row in shard) == [0, 1, 2, 3, 4]


def test_layout_complete_shards_never_split_precomputed_layouts(common_crawl_main: ModuleType) -> None:
    tasks = common_crawl_main.build_page_tasks(
        [
            {"url": "https://a.example/1", "html": "a1", "dripper_layout_id": "a.example_0"},
            {"url": "https://b.example/1", "html": "b1", "dripper_layout_id": "b.example_0"},
            {"url": "https://a.example/2", "html": "a2", "dripper_layout_id": "a.example_0"},
            {"url": "https://c.example/1", "html": "c1", "dripper_layout_id": "-1"},
            {"url": "https://a.example/3", "html": "a3", "dripper_layout_id": "a.example_0"},
            {"url": "https://d.example/1", "html": "d1", "dripper_layout_id": ""},
        ],
        shard_size=2,
        shard_strategy="layout_complete",
        task_id="task",
        dataset_name="dataset",
    )

    shard_row_indexes = _row_indexes_by_task(tasks)

    assert [0, 2, 4] in shard_row_indexes
    assert sorted(row for shard in shard_row_indexes for row in shard) == [0, 1, 2, 3, 4, 5]
    assert all("_dripper_layout_key" not in task.to_pandas().columns for task in tasks)


def test_layout_complete_defaults_to_dripper_layout_id(common_crawl_main: ModuleType) -> None:
    tasks = common_crawl_main.build_page_tasks(
        [
            {"url": "https://a.example/1", "html": "a1", "dripper_layout_id": "a.example_0"},
            {"url": "https://a.example/2", "html": "a2", "dripper_layout_id": "a.example_0"},
        ],
        shard_size=1,
        shard_strategy="layout_complete",
        task_id="task",
        dataset_name="dataset",
    )

    assert _row_indexes_by_task(tasks) == [[0, 1]]


def test_domain_html_hash_keeps_same_host_exact_html_duplicates_adjacent(
    common_crawl_main: ModuleType,
) -> None:
    tasks = common_crawl_main.build_page_tasks(
        [
            {"url": "https://a.example/first", "html": "<html>same</html>"},
            {"url": "https://a.example/second", "html": "<html>middle-a</html>"},
            {"url": "https://a.example/third", "html": "<html>middle-b</html>"},
            {"url": "https://a.example/fourth", "html": "<html>same</html>"},
            {"url": "https://b.example/first", "html": "<html>same</html>"},
        ],
        shard_size=2,
        shard_strategy="domain_html_hash",
        task_id="task",
        dataset_name="dataset",
    )

    shard_row_indexes = _row_indexes_by_task(tasks)

    assert [0, 3] in shard_row_indexes
    assert sorted(row for shard in shard_row_indexes for row in shard) == [0, 1, 2, 3, 4]
    assert all("_dripper_html_hash" not in task.to_pandas().columns for task in tasks)
    assert all("_dripper_host_key" not in task.to_pandas().columns for task in tasks)


def test_read_manifest_dataframe_stops_after_max_rows(
    common_crawl_main: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[str] = []

    def fake_read_manifest_file(path: str) -> pd.DataFrame:
        reads.append(path)
        return pd.DataFrame({"url": [f"{path}-0", f"{path}-1", f"{path}-2"]})

    monkeypatch.setattr(common_crawl_main, "read_manifest_file", fake_read_manifest_file)

    out = common_crawl_main.read_manifest_dataframe(["a.parquet", "b.parquet", "c.parquet"], max_rows=5)

    assert reads == ["a.parquet", "b.parquet"]
    assert out["url"].tolist() == ["a.parquet-0", "a.parquet-1", "a.parquet-2", "b.parquet-0", "b.parquet-1"]


def _rows(tasks: list[Any]) -> list[dict[str, Any]]:
    return [row for task in tasks for row in task.to_pandas().to_dict("records")]


def _row_indexes_by_task(tasks: list[Any]) -> list[list[int]]:
    return [[int(r["_dripper_row_index"]) for r in task.to_pandas().to_dict("records")] for task in tasks]
