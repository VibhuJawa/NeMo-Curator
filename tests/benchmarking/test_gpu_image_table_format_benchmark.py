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

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarking" / "scripts"))

from gpu_image_table_format_benchmark import (
    StorageConfig,
    _arrow_large_binary_as_large_string,
    _validate_parquet_image_encoding,
    _write_parquet,
    choose_representative_fragment,
    choose_representative_fragments,
    fraction_label,
    parse_sample_fractions,
    sample_row_offsets,
    selected_parquet_row_groups,
    trial_arm_order,
    uri_join,
)


def test_large_binary_string_view_reuses_jpeg_buffers_without_decoding() -> None:
    table = pa.table({"image": pa.array([b"\xff\xd8\xff\xe0", b"\x00\xff"], type=pa.large_binary())})

    viewed = _arrow_large_binary_as_large_string(table)
    source = table["image"].chunk(0)
    string_view = viewed["image"].chunk(0)
    restored = pa.Array.from_buffers(
        pa.large_binary(),
        len(string_view),
        string_view.buffers(),
        null_count=string_view.null_count,
    )

    assert string_view.type == pa.large_string()
    assert string_view.buffers()[2].address == source.buffers()[2].address
    assert restored.to_pylist() == [b"\xff\xd8\xff\xe0", b"\x00\xff"]


def test_prepared_parquet_images_are_uncompressed_plain_without_dictionary(tmp_path: Path) -> None:
    path = tmp_path / "images.parquet"
    table = pa.table(
        {
            "id": [1, 2],
            "image": pa.array([b"\xff\xd8\xff\xe0", b"\x00\xff"], type=pa.large_binary()),
        }
    )

    _write_parquet(table, str(path), row_group_size=1, storage=StorageConfig())

    assert _validate_parquet_image_encoding(str(path), StorageConfig()) == {
        "compressions": ["UNCOMPRESSED"],
        "encodings": ["PLAIN", "RLE"],
    }


def test_sample_row_offsets_is_stable_ten_percent_without_replacement() -> None:
    first = sample_row_offsets(10_003, 0.10, "seed", 17)
    second = sample_row_offsets(10_003, 0.10, "seed", 17)

    assert first == second
    assert len(first) == 1_001
    assert first == sorted(set(first))
    assert first != sample_row_offsets(10_003, 0.10, "other-seed", 17)


def test_sample_fraction_matrix_is_sorted_and_has_stable_labels() -> None:
    fractions = parse_sample_fractions("1.0, .1, .8, .2, .4")

    assert fractions == (0.10, 0.20, 0.40, 0.80, 1.00)
    assert [fraction_label(fraction) for fraction in fractions] == [
        "010pct",
        "020pct",
        "040pct",
        "080pct",
        "100pct",
    ]


@pytest.mark.parametrize("value", ["", "0,.1", ".1,.1", "1.1", "wat"])
def test_sample_fraction_matrix_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="sample fraction"):
        parse_sample_fractions(value)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_sample_row_offsets_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        sample_row_offsets(100, fraction, "seed", 1)


def test_choose_representative_fragment_uses_median_rows_and_stable_tie_break() -> None:
    assert choose_representative_fragment([(9, 90), (3, 50), (7, 70), (4, 60)]) == (4, 60)


def test_choose_representative_fragments_builds_stable_ten_fragment_cohort() -> None:
    fragments = [(index, 1_000 + (index % 3)) for index in range(12)]

    selected = choose_representative_fragments(fragments, 10)

    assert len(selected) == 10
    assert len({fragment_id for fragment_id, _ in selected}) == 10
    assert selected == choose_representative_fragments(fragments, 10)


def test_selected_parquet_row_groups_reports_read_amplification_basis() -> None:
    groups, touched_rows = selected_parquet_row_groups([0, 4, 1_024, 2_049], 2_050, 1_024)

    assert groups == [0, 1, 2]
    assert touched_rows == 2_050


def test_selected_parquet_row_groups_rejects_out_of_range_offsets() -> None:
    with pytest.raises(ValueError, match="within"):
        selected_parquet_row_groups([100], 100, 10)


def test_trial_arm_order_alternates() -> None:
    assert trial_arm_order(0) == ("parquet", "lance")
    assert trial_arm_order(1) == ("lance", "parquet")
    assert trial_arm_order(2) == ("parquet", "lance")


def test_uri_join_preserves_local_and_s3_roots() -> None:
    assert uri_join("/local/work", "prepared", "x") == "/local/work/prepared/x"
    assert uri_join("s3://bucket/root", "/prepared/", "x") == "s3://bucket/root/prepared/x"


def test_single_gpu_config_is_pinned_to_requested_dataset() -> None:
    config = (Path(__file__).parents[2] / "benchmarking" / "gpu-image-table-format-single-gpu.yaml").read_text()

    assert "num_gpus: 1" in config
    assert "--sample-fractions=0.10,0.20,0.40,0.80,1.00" in config
    assert "persistent GPU actor" in config
    assert "--blur-threshold=0.10" in config
    assert "--source-version=4" in config
    assert "47f4e65f452f20ffca8b205a/stable_row_ids/dataset" in config


def test_ten_x_config_uses_ten_fragments_and_warm_actor_queue() -> None:
    config = (Path(__file__).parents[2] / "benchmarking" / "gpu-image-table-format-10x-single-gpu.yaml").read_text()

    assert "--fragment-count=10" in config
    assert "async Lance prefetch" in config
    assert "pinned double-buffered D2H" in config
