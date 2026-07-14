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

"""Custom source and materialization stages for the Lance image workflow."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO

import fsspec
import pyarrow as pa
from loguru import logger
from PIL import Image

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, EmptyTask, Task

from .manifest import FppPack, PhysicalTar, load_manifest

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fsspec.spec import AbstractFileSystem

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".mpo", ".png", ".webp"})

# The stage verifies encoded images without decoding their raster into memory.
Image.MAX_IMAGE_PIXELS = None

IMAGE_SCHEMA = pa.schema(
    [
        pa.field("url", pa.string(), nullable=False),
        pa.field("image", pa.large_binary(), nullable=False),
        pa.field("image_format", pa.string(), nullable=False),
        pa.field("mime_type", pa.string(), nullable=False),
        pa.field("image_size_bytes", pa.int64(), nullable=False),
        pa.field("md5", pa.string(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=True),
        pa.field("source_sha256_matches", pa.bool_(), nullable=True),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        pa.field("source_shard", pa.int32(), nullable=False),
        pa.field("source_attempt", pa.int16(), nullable=False),
        pa.field("source_tar_id", pa.string(), nullable=False),
        pa.field("source_tar_uri", pa.string(), nullable=False),
        pa.field("source_image_member", pa.string(), nullable=False),
        pa.field("source_json_member", pa.string(), nullable=False),
        pa.field("metadata_json", pa.large_string(), nullable=False),
    ]
)


@dataclass
class FppPackTask(Task[FppPack]):
    """One deterministic, within-shard pack and its complete retry history."""

    @property
    def num_items(self) -> int:
        return sum(len(fpp.attempts) for fpp in self.data.fpps)

    def validate(self) -> bool:
        if not self.data.fpps:
            message = f"FPP pack {self.data.pack_id} is empty"
            raise ValueError(message)
        for fpp in self.data.fpps:
            if fpp.source_shard != self.data.source_shard:
                message = f"FPP pack {self.data.pack_id} crosses source shards"
                raise ValueError(message)
            attempts = [item.attempt for item in fpp.attempts]
            if attempts != sorted(attempts) or len(attempts) != len(set(attempts)):
                message = f"FPP {fpp.fpp_id} attempts are not sorted and unique"
                raise ValueError(message)
        return True

    def get_deterministic_id(self) -> str:
        return self.data.deterministic_id


@dataclass
class FppPackPartitioningStage(ProcessingStage[EmptyTask, FppPackTask]):
    """Emit the frozen FPP packs; Curator applies logical Slurm-array sharding."""

    manifest_dir: str
    dataset_name: str
    name: str = "fpp_pack_partitioning"
    is_source_stage: bool = True
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def num_workers(self) -> int:
        return 1

    def process(self, _: EmptyTask) -> list[FppPackTask]:
        metadata, packs = load_manifest(self.manifest_dir)
        logger.info(
            "Loaded snapshot {} with {:,} FPP packs",
            metadata["snapshot_id"],
            len(packs),
        )
        return [FppPackTask(dataset_name=self.dataset_name, data=pack) for pack in packs]


@lru_cache(maxsize=8)
def _s3_filesystem(storage_options_json: str) -> AbstractFileSystem:
    return fsspec.filesystem("s3", **json.loads(storage_options_json))


def _member_key(member_name: str) -> tuple[str, str] | None:
    suffix = PurePosixPath(member_name).suffix.lower()
    if suffix == ".json":
        return member_name[: -len(suffix)], "json"
    if suffix in IMAGE_SUFFIXES:
        return member_name[: -len(suffix)], "image"
    return None


def _image_properties(image_bytes: bytes, source_id: str) -> tuple[str, str, int, int]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            mime_type = image.get_format_mimetype()
            width, height = image.size
            if not image_format or not mime_type:
                _raise_missing_image_type()
            image.verify()
    except Exception as error:
        message = f"Invalid encoded image {source_id}: {error}"
        raise ValueError(message) from error
    return image_format, mime_type, int(width), int(height)


def _raise_missing_image_type() -> None:
    message = "Pillow did not identify the format and MIME type"
    raise ValueError(message)


def _candidate(
    physical_tar: PhysicalTar,
    image_member: str,
    json_member: str,
    image_bytes: bytes,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    url = metadata.get("url")
    if not isinstance(url, str) or not url:
        message = f"Successful record has no URL: {physical_tar.tar_uri}#{json_member}"
        raise ValueError(message)
    source_id = f"{physical_tar.tar_uri}#{image_member}"
    image_format, mime_type, width, height = _image_properties(image_bytes, source_id)
    if metadata.get("width") is not None and metadata.get("height") is not None:
        expected_size = int(metadata["width"]), int(metadata["height"])
        if (width, height) != expected_size:
            message = f"Image/JSON resolution mismatch in {source_id}: {(width, height)} != {expected_size}"
            raise ValueError(message)
    return {
        "url": url,
        "image": image_bytes,
        "image_format": image_format,
        "mime_type": mime_type,
        "width": width,
        "height": height,
        "source_sha256": metadata.get("sha256"),
        "source_id": source_id,
        "source_shard": physical_tar.source_shard,
        "source_attempt": physical_tar.attempt,
        "source_tar_id": physical_tar.tar_id,
        "source_tar_uri": physical_tar.tar_uri,
        "source_image_member": image_member,
        "source_json_member": json_member,
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
    }


def iter_tar_candidates(fileobj: BinaryIO, physical_tar: PhysicalTar) -> Iterator[dict[str, Any]]:
    """Pair image/JSON members from one streaming tar without temporary files."""
    pending_images: dict[str, tuple[str, bytes]] = {}
    pending_json: dict[str, tuple[str, dict[str, Any]]] = {}
    seen_members: set[str] = set()
    with tarfile.open(fileobj=fileobj, mode="r|*") as archive:
        for member in archive:
            member_type = _member_key(member.name)
            if not member.isfile() or member_type is None:
                continue
            if member.name in seen_members:
                message = f"Duplicate tar member {physical_tar.tar_uri}#{member.name}"
                raise ValueError(message)
            seen_members.add(member.name)
            stream = archive.extractfile(member)
            if stream is None:
                message = f"Unable to extract {physical_tar.tar_uri}#{member.name}"
                raise ValueError(message)
            key, kind = member_type
            if kind == "image":
                pending_images[key] = member.name, stream.read()
            else:
                try:
                    metadata = json.loads(stream.read())
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(metadata, dict) or metadata.get("status") not in (None, "success"):
                    continue
                pending_json[key] = member.name, metadata

            if key in pending_images and key in pending_json:
                image_member, image_bytes = pending_images.pop(key)
                json_member, metadata = pending_json.pop(key)
                try:
                    yield _candidate(
                        physical_tar,
                        image_member,
                        json_member,
                        image_bytes,
                        metadata,
                    )
                except (KeyError, TypeError, ValueError):
                    continue


def candidate_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    """Prefer larger resolution, then newer attempt, then stable source ID."""
    candidate_area = int(candidate["width"]) * int(candidate["height"])
    incumbent_area = int(incumbent["width"]) * int(incumbent["height"])
    if candidate_area != incumbent_area:
        return candidate_area > incumbent_area
    if int(candidate["source_attempt"]) != int(incumbent["source_attempt"]):
        return int(candidate["source_attempt"]) > int(incumbent["source_attempt"])
    return str(candidate["source_id"]) < str(incumbent["source_id"])


def _finalize(candidate: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in candidate.items() if key != "source_id"}
    image_bytes = row["image"]
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    source_sha256 = row["source_sha256"]
    row.update(
        {
            "image_size_bytes": len(image_bytes),
            "md5": hashlib.md5(image_bytes, usedforsecurity=False).hexdigest(),
            "sha256": sha256,
            "source_sha256_matches": (None if source_sha256 is None else sha256 == source_sha256),
        }
    )
    return row


@dataclass
class FppPackMaterializationStage(ProcessingStage[FppPackTask, DocumentBatch]):
    """Read all attempts, choose one exact-URL winner, and emit Arrow large binary."""

    source_storage_options_json: str = "{}"
    name: str = "fpp_pack_materialization"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1))

    def process(self, task: FppPackTask) -> DocumentBatch:
        started = time.perf_counter()
        filesystem = _s3_filesystem(self.source_storage_options_json)
        winners: dict[str, dict[str, Any]] = {}
        candidate_count = 0
        for fpp in task.data.fpps:
            for physical_tar in fpp.attempts:
                tar_bytes = filesystem.cat_ranges(
                    [physical_tar.tar_uri],
                    [0],
                    [physical_tar.tar_size],
                    on_error="raise",
                )[0]
                if len(tar_bytes) != physical_tar.tar_size:
                    message = f"Short read for {physical_tar.tar_uri}: {len(tar_bytes)} != {physical_tar.tar_size}"
                    raise ValueError(message)
                for candidate in iter_tar_candidates(io.BytesIO(tar_bytes), physical_tar):
                    candidate_count += 1
                    url = str(candidate["url"])
                    incumbent = winners.get(url)
                    if incumbent is None or candidate_is_better(candidate, incumbent):
                        winners[url] = candidate
                del tar_bytes

        rows = [_finalize(winners[url]) for url in sorted(winners)]
        table = pa.Table.from_pylist(rows, schema=IMAGE_SCHEMA)
        metrics = {
            "snapshot_id": task.data.snapshot_id,
            "pack_id": task.data.pack_id,
            "source_shard": task.data.source_shard,
            "fpp_count": len(task.data.fpps),
            "physical_attempts": sum(len(fpp.attempts) for fpp in task.data.fpps),
            "candidate_rows": candidate_count,
            "winner_rows": len(rows),
            "image_payload_bytes": sum(row["image_size_bytes"] for row in rows),
            "arrow_bytes": table.nbytes,
            "elapsed_seconds": time.perf_counter() - started,
        }
        logger.info("Materialized {}: {}", task.data.pack_id, metrics)
        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=table,
            _metadata={"fpp_pack": metrics},
        )
