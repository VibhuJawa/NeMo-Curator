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

from pathlib import Path

import lance
import pyarrow as pa
import pytest

from nemo_curator.utils.lance import clear_lance_dataset_cache, open_lance_dataset


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_lance_dataset_cache()
    yield
    clear_lance_dataset_cache()


def _write(path: Path, values: list[int]) -> None:
    lance.write_dataset(pa.table({"n": values}), str(path), mode="overwrite")


def test_pinned_version_returns_the_same_handle(tmp_path: Path) -> None:
    """The whole point: opening the same pinned dataset twice costs one open."""
    _write(tmp_path / "d", [1, 2, 3])
    first = open_lance_dataset(str(tmp_path / "d"), version=1)
    assert open_lance_dataset(str(tmp_path / "d"), version=1) is first


def test_unpinned_version_is_never_cached(tmp_path: Path) -> None:
    """Without a version Lance resolves to latest, so a cached handle could go stale."""
    path = tmp_path / "d"
    _write(path, [1, 2, 3])
    first = open_lance_dataset(str(path))
    _write(path, [1, 2, 3, 4])
    second = open_lance_dataset(str(path))

    assert second is not first
    assert second.to_table().num_rows == 4


def test_different_versions_are_separate_entries(tmp_path: Path) -> None:
    path = tmp_path / "d"
    _write(path, [1, 2, 3])
    _write(path, [1, 2, 3, 4])

    v1 = open_lance_dataset(str(path), version=1)
    v2 = open_lance_dataset(str(path), version=2)

    assert v1 is not v2
    assert v1.to_table().num_rows == 3
    assert v2.to_table().num_rows == 4


def test_different_paths_are_separate_entries(tmp_path: Path) -> None:
    _write(tmp_path / "a", [1])
    _write(tmp_path / "b", [2])
    assert open_lance_dataset(str(tmp_path / "a"), version=1) is not open_lance_dataset(str(tmp_path / "b"), version=1)


def test_differing_storage_options_are_separate_entries(tmp_path: Path) -> None:
    """Storage options change identity, so they must not collide in the cache."""
    _write(tmp_path / "d", [1, 2, 3])
    plain = open_lance_dataset(str(tmp_path / "d"), version=1)
    with_options = open_lance_dataset(str(tmp_path / "d"), version=1, storage_options={"anonymous": "true"})
    assert plain is not with_options


def test_clear_releases_handles(tmp_path: Path) -> None:
    _write(tmp_path / "d", [1, 2, 3])
    first = open_lance_dataset(str(tmp_path / "d"), version=1)
    clear_lance_dataset_cache()
    assert open_lance_dataset(str(tmp_path / "d"), version=1) is not first


def test_cached_handle_still_reads_correctly(tmp_path: Path) -> None:
    _write(tmp_path / "d", [1, 2, 3])
    for _ in range(3):
        assert open_lance_dataset(str(tmp_path / "d"), version=1).to_table()["n"].to_pylist() == [1, 2, 3]
