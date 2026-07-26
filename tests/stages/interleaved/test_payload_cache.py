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
from unittest.mock import patch

from nemo_curator.stages.interleaved.utils.materialization import materialize_task_binary_content
from nemo_curator.stages.interleaved.utils.payload_cache import PayloadCache

from .conftest import make_image_row, make_image_task


def test_get_returns_none_on_miss(tmp_path: Path) -> None:
    assert PayloadCache(tmp_path).get("absent") is None


def test_round_trip_and_overwrite(tmp_path: Path) -> None:
    cache = PayloadCache(tmp_path)
    cache.put("k", b"first")
    assert cache.get("k") == b"first"
    cache.put("k", b"second")
    assert cache.get("k") == b"second"


def test_put_leaves_no_temporary_files(tmp_path: Path) -> None:
    cache = PayloadCache(tmp_path)
    cache.put("k", b"payload")
    assert not list(tmp_path.rglob("*.tmp"))


def test_faults_are_not_fatal(tmp_path: Path) -> None:
    """A broken cache degrades to a miss rather than failing the pipeline."""
    cache = PayloadCache(tmp_path)
    with patch("pathlib.Path.read_bytes", side_effect=OSError("read-only")):
        assert cache.get("k") is None
    with patch("pathlib.Path.write_bytes", side_effect=OSError("disk full")):
        cache.put("k", b"payload")
    assert cache.get("k") is None


def test_materialize_populates_then_serves_cache(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"jpeg-bytes")
    cache = PayloadCache(tmp_path / "cache")
    task = make_image_task([make_image_row(path=str(image))])

    filled = materialize_task_binary_content(task, cache=cache)
    assert filled.data["binary_content"].iloc[0] == b"jpeg-bytes"

    # The payload now comes from the cache, so the source is never touched.
    image.unlink()
    served = materialize_task_binary_content(task, cache=cache)
    assert served.data["binary_content"].iloc[0] == b"jpeg-bytes"


def test_materialize_without_cache_is_unchanged(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"jpeg-bytes")
    task = make_image_task([make_image_row(path=str(image))])

    assert materialize_task_binary_content(task).data["binary_content"].iloc[0] == b"jpeg-bytes"
    assert not (tmp_path / "cache").exists()
