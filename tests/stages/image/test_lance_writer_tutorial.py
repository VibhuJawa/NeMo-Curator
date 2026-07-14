# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from tutorials.image.lance_writer.manifest import PhysicalTar, build_manifest, load_manifest
from tutorials.image.lance_writer.pipeline import build_pipeline
from tutorials.image.lance_writer.stages import (
    IMAGE_SCHEMA,
    candidate_is_better,
    iter_tar_candidates,
)


def _inventory(path: Path, *, etag: str = "etag-a") -> None:
    rows = [
        {
            "source_shard": shard,
            "tar_id": tar_id,
            "attempt": attempt,
            "tar_uri": f"s3://source/shard_{shard:05d}/attempt_{attempt}/{tar_id}.tar",
            "tar_size": size,
            "tar_etag": etag if tar_id == "00000" else f"etag-{tar_id}-{attempt}",
            "last_modified": "2026-01-01T00:00:00+00:00",
        }
        for shard, tar_id, attempt, size in [
            (1, "00000", 1, 700),
            (1, "00000", 2, 700),
            (1, "00001", 1, 700),
            (1, "00002", 1, 700),
            (2, "00000", 1, 400),
        ]
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_manifest_is_stable_and_packs_never_cross_source_shards(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.parquet"
    manifest_dir = tmp_path / "manifest"
    _inventory(inventory_path)

    first = build_manifest(str(inventory_path), str(manifest_dir), target_pack_bytes=1_000)
    second = build_manifest(str(inventory_path), str(manifest_dir), target_pack_bytes=1_000)
    metadata, packs = load_manifest(str(manifest_dir))

    assert first == second == metadata
    assert metadata["physical_tar_count"] == 5
    assert metadata["fpp_count"] == 4
    assert all({fpp.source_shard for fpp in pack.fpps} == {pack.source_shard} for pack in packs)
    retried = next(fpp for pack in packs for fpp in pack.fpps if fpp.fpp_id == "shard_00001/00000")
    assert [item.attempt for item in retried.attempts] == [1, 2]


def test_winner_policy_prefers_resolution_then_newest_attempt() -> None:
    base = {"width": 100, "height": 100, "source_attempt": 1, "source_id": "b"}
    assert candidate_is_better({**base, "width": 101}, base)
    assert candidate_is_better({**base, "source_attempt": 2}, base)
    assert candidate_is_better({**base, "source_id": "a"}, base)
    assert not candidate_is_better({**base, "source_attempt": 0}, base)


def test_materializer_and_writer_are_fusible_one_cpu_stages() -> None:
    pipeline = build_pipeline(
        manifest_dir="/manifest",
        dataset_uri="s3://output/images",
        lance_commit_path="/checkpoints/lance",
        source_storage_options={},
        lance_storage_options={},
    )

    materializer, writer = pipeline.stages[1:]
    assert materializer.resources == writer.resources
    assert materializer.resources.cpus == 1
    assert pa.types.is_large_binary(IMAGE_SCHEMA.field("image").type)


def test_tar_reader_pairs_json_with_encoded_image_and_preserves_format() -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (12, 8), color="red").save(image_buffer, format="JPEG")
    metadata = json.dumps(
        {
            "status": "success",
            "url": "https://example.test/image",
            "width": 12,
            "height": 8,
        }
    ).encode()
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for name, payload in (("sample.json", metadata), ("sample.jpg", image_buffer.getvalue())):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    tar_buffer.seek(0)
    physical_tar = PhysicalTar(
        source_shard=1,
        tar_id="00000",
        attempt=1,
        tar_uri="s3://source/shard_00001/attempt_1/00000.tar",
        tar_size=len(tar_buffer.getvalue()),
        tar_etag="etag",
        last_modified="2026-01-01T00:00:00+00:00",
    )

    rows = list(iter_tar_candidates(tar_buffer, physical_tar))

    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.test/image"
    assert rows[0]["image_format"] == "JPEG"
    assert rows[0]["mime_type"] == "image/jpeg"
    assert (rows[0]["width"], rows[0]["height"]) == (12, 8)
