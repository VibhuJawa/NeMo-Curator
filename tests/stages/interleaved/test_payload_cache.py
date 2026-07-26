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

import pickle
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage
from nemo_curator.stages.interleaved.utils.materialization import materialize_task_binary_content
from nemo_curator.stages.interleaved.utils.payload_cache import PayloadCache, build_payload_cache

from .conftest import make_image_row, make_image_task, write_tar


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


def test_repeated_key_in_one_batch_is_stored_once(tmp_path: Path) -> None:
    """A source_ref appearing N times in one batch costs one cache write, not N."""
    image = tmp_path / "img.jpg"
    image.write_bytes(b"jpeg-bytes")
    row = make_image_row(path=str(image))
    task = make_image_task([{**row, "position": i} for i in range(3)])
    cache = PayloadCache(tmp_path / "cache")

    with patch.object(PayloadCache, "put", autospec=True, side_effect=PayloadCache.put) as put:
        materialize_task_binary_content(task, cache=cache)

    assert put.call_count == 1
    assert _cache_entry_count(tmp_path / "cache") == 1


def test_identical_payloads_under_different_locators_are_separate_entries(tmp_path: Path) -> None:
    """Documents the key semantics: the key is the locator, not the image identity.

    Two tar members holding byte-identical images are two ``source_ref`` values,
    so they occupy two cache entries and neither ever hits on the other. The
    reference-multiplicity argument for this cache therefore only holds for a
    source layout that stores each unique image exactly once.
    """
    tar_path = write_tar(tmp_path / "shard.tar", {"a.jpg": b"same-bytes", "b.jpg": b"same-bytes"})
    task = make_image_task(
        [
            make_image_row(path=tar_path, member="a.jpg"),
            {**make_image_row(path=tar_path, member="b.jpg"), "position": 1},
        ]
    )
    cache = PayloadCache(tmp_path / "cache")

    filled = materialize_task_binary_content(task, cache=cache)

    assert filled.data["binary_content"].tolist() == [b"same-bytes", b"same-bytes"]
    assert _cache_entry_count(tmp_path / "cache") == 2


# ---------------------------------------------------------------------------
# Stage-level wiring: payload_cache_root -> setup() -> materialization
# ---------------------------------------------------------------------------


def _cache_entry_count(root: Path) -> int:
    return len([p for p in root.rglob("*") if p.is_file()]) if root.exists() else 0


def _jpeg_bytes(width: int = 200, height: int = 100) -> bytes:
    """A JPEG whose aspect ratio (2.0) passes InterleavedAspectRatioFilterStage defaults."""
    buf = BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="JPEG")
    return buf.getvalue()


def test_build_payload_cache_returns_none_without_root(tmp_path: Path) -> None:
    assert build_payload_cache(None) is None
    assert build_payload_cache(str(tmp_path)) == PayloadCache(tmp_path)


def _filter_stage_with_cache(tmp_path: Path) -> InterleavedAspectRatioFilterStage:
    return InterleavedAspectRatioFilterStage(payload_cache_root=str(tmp_path / "cache"))


def _writer_stage_with_cache(tmp_path: Path) -> InterleavedParquetWriterStage:
    return InterleavedParquetWriterStage(
        path=str(tmp_path / "out"),
        mode="overwrite",
        payload_cache_root=str(tmp_path / "cache"),
    )


@pytest.mark.parametrize(
    "make_stage",
    [
        pytest.param(_filter_stage_with_cache, id="filter"),
        pytest.param(_writer_stage_with_cache, id="writer"),
    ],
)
def test_stage_builds_cache_in_setup_so_it_is_never_pickled(make_stage: object, tmp_path: Path) -> None:
    """The stage carries the root string across the wire; the handle is worker-local."""
    stage = make_stage(tmp_path)
    assert stage._payload_cache is None

    assert pickle.loads(pickle.dumps(stage))._payload_cache is None  # noqa: S301 - our own object

    stage.setup()
    assert stage._payload_cache == PayloadCache(tmp_path / "cache")


def test_filter_stage_serves_second_task_from_cache(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(_jpeg_bytes())
    task = make_image_task([make_image_row(path=str(image))])
    stage = InterleavedAspectRatioFilterStage(payload_cache_root=str(tmp_path / "cache"))
    stage.setup()

    assert len(stage.process(task).to_pandas()) == 1
    assert _cache_entry_count(tmp_path / "cache") == 1

    # The payload now comes from the cache, so the source is never touched.
    image.unlink()
    assert len(stage.process(task).to_pandas()) == 1


def test_filter_stage_without_cache_root_reads_the_source_every_time(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(_jpeg_bytes())
    task = make_image_task([make_image_row(path=str(image))])
    stage = InterleavedAspectRatioFilterStage()
    stage.setup()

    assert stage._payload_cache is None
    assert len(stage.process(task).to_pandas()) == 1

    # Nothing was cached, so an unreadable image is now dropped.
    image.unlink()
    assert stage.process(task).to_pandas().empty


def test_writer_stage_serves_second_task_from_cache(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"jpeg-bytes")
    task = make_image_task([make_image_row(path=str(image))], metadata={"source_files": [str(image)]})
    stage = InterleavedParquetWriterStage(
        path=str(tmp_path / "out"),
        mode="overwrite",
        payload_cache_root=str(tmp_path / "cache"),
    )
    stage.setup()

    first = stage.process(task)
    assert pq.read_table(first.data[0])["binary_content"].to_pylist() == [b"jpeg-bytes"]
    assert _cache_entry_count(tmp_path / "cache") == 1

    image.unlink()
    second = stage.process(task)
    assert pq.read_table(second.data[0])["binary_content"].to_pylist() == [b"jpeg-bytes"]


def test_writer_stage_without_cache_root_writes_no_cache(tmp_path: Path) -> None:
    image = tmp_path / "img.jpg"
    image.write_bytes(b"jpeg-bytes")
    task = make_image_task([make_image_row(path=str(image))], metadata={"source_files": [str(image)]})
    stage = InterleavedParquetWriterStage(path=str(tmp_path / "out"), mode="overwrite")
    stage.setup()

    assert stage._payload_cache is None
    result = stage.process(task)
    assert pq.read_table(result.data[0])["binary_content"].to_pylist() == [b"jpeg-bytes"]
    assert not (tmp_path / "cache").exists()
