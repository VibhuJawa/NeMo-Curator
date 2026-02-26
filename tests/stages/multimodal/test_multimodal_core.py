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

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.multimodal.io.reader import WebdatasetReader
from nemo_curator.stages.multimodal.stages import (
    BaseMultimodalFilterStage,
    MultimodalJpegAspectRatioFilterStage,
)
from nemo_curator.stages.multimodal.utils import load_bytes_from_content_reference
from nemo_curator.stages.multimodal.utils.materialization import materialize_task_binary_content
from nemo_curator.tasks import MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA


@pytest.fixture
def single_row_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": json.dumps(
                    {
                        "path": "/dataset/shard-00000.tar",
                        "member": "s1.json",
                        "byte_offset": 10,
                        "byte_size": 20,
                    }
                ),
                "metadata_json": None,
                "materialize_error": None,
            }
        ],
        schema=MULTIMODAL_SCHEMA,
    )


@pytest.fixture
def single_row_task(single_row_table: pa.Table) -> MultiBatchTask:
    return MultiBatchTask(task_id="t1", dataset_name="d1", data=single_row_table)


def test_with_parsed_source_ref_columns(single_row_task: MultiBatchTask) -> None:
    df = single_row_task.with_parsed_source_ref_columns()
    assert df.loc[0, "_src_path"] == "/dataset/shard-00000.tar"
    assert df.loc[0, "_src_member"] == "s1.json"
    assert df.loc[0, "_src_byte_offset"] == 10
    assert df.loc[0, "_src_byte_size"] == 20


def test_parse_source_ref_soft_migration() -> None:
    old_format = json.dumps({"content_path": "/old/path.tar", "content_key": "old.json"})
    parsed = MultiBatchTask.parse_source_ref(old_format)
    assert parsed["path"] == "/old/path.tar"
    assert parsed["member"] == "old.json"
    assert parsed["byte_offset"] is None
    assert parsed["byte_size"] is None


def test_parse_source_ref_empty_values() -> None:
    assert MultiBatchTask.parse_source_ref(None)["path"] is None
    assert MultiBatchTask.parse_source_ref("")["path"] is None


def test_load_bytes_from_content_reference_direct_and_keyed(tmp_path: Path) -> None:
    direct_path = tmp_path / "direct.bin"
    direct_payload = b"direct-bytes"
    direct_path.write_bytes(direct_payload)
    tar_path = tmp_path / "blob.tar"
    tar_payload = b"tar-bytes"
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="x.bin")
        info.size = len(tar_payload)
        tf.addfile(info, BytesIO(tar_payload))

    cache: dict[tuple[str, str], bytes | None] = {}
    assert load_bytes_from_content_reference(str(direct_path), None, {}, cache) == direct_payload
    assert load_bytes_from_content_reference(str(tar_path), "x.bin", {}, cache) == tar_payload


def test_load_bytes_from_content_reference_caches() -> None:
    cache: dict[tuple[str, str], bytes | None] = {("/fake", ""): b"cached"}
    assert load_bytes_from_content_reference("/fake", None, {}, cache) == b"cached"


def test_jpeg_filter_handles_non_default_dataframe_index() -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "ok",
                "binary_content": None,
                "source_ref": None,
                "metadata_json": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": b"not-a-valid-jpeg",
                "source_ref": None,
                "metadata_json": None,
                "materialize_error": None,
            },
        ]
    )
    df.index = pd.Index([10, 42])
    task = MultiBatchTask(task_id="non_default_index", dataset_name="d1", data=df)
    stage = MultimodalJpegAspectRatioFilterStage(drop_invalid_rows=False)
    out = stage.process(task).to_pandas()
    assert len(out) == 1
    assert out.iloc[0]["modality"] == "text"


# --- split_table_by_group_max_bytes tests ---


def test_split_table_none_max_bytes() -> None:
    table = pa.table({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = split_table_by_group_max_bytes(table, "g", None)
    assert len(result) == 1
    assert result[0].num_rows == 3


def test_split_table_empty_table() -> None:
    table = pa.table({"g": pa.array([], type=pa.string()), "v": pa.array([], type=pa.int64())})
    result = split_table_by_group_max_bytes(table, "g", 100)
    assert len(result) == 1
    assert result[0].num_rows == 0


def test_split_table_invalid_max_bytes() -> None:
    table = pa.table({"g": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="max_batch_bytes must be > 0"):
        split_table_by_group_max_bytes(table, "g", 0)


def test_split_table_missing_column() -> None:
    table = pa.table({"g": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="not found in table"):
        split_table_by_group_max_bytes(table, "missing", 100)


def test_split_table_single_large_group() -> None:
    table = pa.table({"g": ["a"] * 100, "v": list(range(100))})
    result = split_table_by_group_max_bytes(table, "g", 1)
    assert len(result) == 1
    assert result[0].num_rows == 100


def test_split_table_multiple_groups_split() -> None:
    table = pa.table({"g": ["a", "a", "b", "b", "c", "c"], "v": [1, 2, 3, 4, 5, 6]})
    small_limit = table.slice(0, 2).nbytes + 1
    result = split_table_by_group_max_bytes(table, "g", small_limit)
    assert len(result) >= 2
    total_rows = sum(t.num_rows for t in result)
    assert total_rows == 6


def test_split_table_preserves_group_integrity() -> None:
    table = pa.table({"g": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
    result = split_table_by_group_max_bytes(table, "g", 1)
    for chunk in result:
        groups = chunk["g"].to_pylist()
        assert len(set(groups)) == 1 or all(g == groups[0] for g in groups)


# --- basic_row_validity_mask tests ---


def test_basic_row_validity_mask_filters_bad_modality() -> None:
    df = pd.DataFrame(
        {
            "modality": ["text", "image", "video", "metadata"],
            "position": [0, 1, 2, -1],
        }
    )
    mask = BaseMultimodalFilterStage._basic_row_validity_mask(df)
    assert mask.tolist() == [True, True, False, True]


def test_basic_row_validity_mask_enforces_position_rules() -> None:
    df = pd.DataFrame(
        {
            "modality": ["metadata", "metadata", "text", "text"],
            "position": [-1, 0, 0, -1],
        }
    )
    mask = BaseMultimodalFilterStage._basic_row_validity_mask(df)
    assert mask.tolist() == [True, False, True, False]


# --- CompositeStage decomposition test ---


def test_webdataset_reader_composite_decompose(tmp_path: Path) -> None:
    reader = WebdatasetReader(
        file_paths=str(tmp_path),
        source_id_field="pdf_name",
    )
    stages = reader.decompose()
    assert len(stages) == 2
    assert stages[0].name == "file_partitioning"
    assert stages[1].name == "webdataset_reader"


# --- materialize_task_binary_content tests ---


def test_materialize_empty_task() -> None:
    task = MultiBatchTask(task_id="empty", dataset_name="d", data=pa.table({"sample_id": pa.array([], type=pa.string()), "position": pa.array([], type=pa.int32()), "modality": pa.array([], type=pa.string()), "content_type": pa.array([], type=pa.string()), "text_content": pa.array([], type=pa.string()), "binary_content": pa.array([], type=pa.large_binary()), "source_ref": pa.array([], type=pa.string()), "metadata_json": pa.array([], type=pa.string()), "materialize_error": pa.array([], type=pa.string())}))
    result = materialize_task_binary_content(task)
    assert result.num_items == 0


def test_materialize_no_image_rows() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "metadata_json": None,
                "materialize_error": None,
            }
        ],
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultiBatchTask(task_id="no_img", dataset_name="d", data=table)
    result = materialize_task_binary_content(task)
    assert result.num_items == 1


def test_materialize_fills_binary_from_direct_path(tmp_path: Path) -> None:
    image_bytes = b"test-image-content"
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(image_bytes)

    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": None,
                "source_ref": MultiBatchTask.build_source_ref(path=str(img_path), member=None),
                "metadata_json": None,
                "materialize_error": None,
            }
        ],
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultiBatchTask(task_id="mat", dataset_name="d", data=table)
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert df.loc[0, "binary_content"] == image_bytes
