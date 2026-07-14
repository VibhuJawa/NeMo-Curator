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

"""Create and read the immutable input manifest used by the Lance tutorial."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_SCHEMA_VERSION = 1
PACKS_FILENAME = "fpp_packs.parquet"
METADATA_FILENAME = "manifest.json"


@dataclass(frozen=True)
class PhysicalTar:
    """One physical attempt for a stable ``(source_shard, tar_id)`` unit."""

    source_shard: int
    tar_id: str
    attempt: int
    tar_uri: str
    tar_size: int
    tar_etag: str
    last_modified: str

    @property
    def descriptor(self) -> str:
        return "\0".join((self.tar_uri, str(self.tar_size), self.tar_etag, self.last_modified))


@dataclass(frozen=True)
class FppPartition:
    """All attempts for one stable ``(source_shard, tar_id)`` partition."""

    snapshot_id: str
    fpp_id: str
    source_shard: int
    tar_id: str
    attempts: tuple[PhysicalTar, ...]
    total_tar_bytes: int
    max_tar_bytes: int

    @property
    def deterministic_id(self) -> str:
        digest = hashlib.sha256(f"{self.snapshot_id}\0{self.fpp_id}\n".encode())
        for item in self.attempts:
            digest.update(f"{item.attempt}\0{item.descriptor}\n".encode())
        return digest.hexdigest()[:32]


@dataclass(frozen=True)
class FppPack:
    """A deterministic within-shard group targeting one Lance fragment."""

    snapshot_id: str
    pack_id: str
    source_shard: int
    fpps: tuple[FppPartition, ...]
    estimated_output_bytes: int
    total_tar_bytes: int

    @property
    def deterministic_id(self) -> str:
        digest = hashlib.sha256(f"{self.snapshot_id}\0{self.pack_id}\n".encode())
        for fpp in self.fpps:
            digest.update(f"{fpp.deterministic_id}\n".encode())
        return digest.hexdigest()[:32]


PHYSICAL_TAR_SCHEMA = pa.struct(
    [
        pa.field("source_shard", pa.int32(), nullable=False),
        pa.field("tar_id", pa.string(), nullable=False),
        pa.field("attempt", pa.int16(), nullable=False),
        pa.field("tar_uri", pa.string(), nullable=False),
        pa.field("tar_size", pa.int64(), nullable=False),
        pa.field("tar_etag", pa.string(), nullable=False),
        pa.field("last_modified", pa.string(), nullable=False),
    ]
)
FPP_SCHEMA = pa.struct(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("fpp_id", pa.string(), nullable=False),
        pa.field("source_shard", pa.int32(), nullable=False),
        pa.field("tar_id", pa.string(), nullable=False),
        pa.field("attempts", pa.list_(PHYSICAL_TAR_SCHEMA), nullable=False),
        pa.field("total_tar_bytes", pa.int64(), nullable=False),
        pa.field("max_tar_bytes", pa.int64(), nullable=False),
    ]
)
PACK_SCHEMA = pa.schema(
    [
        pa.field("snapshot_id", pa.string(), nullable=False),
        pa.field("pack_id", pa.string(), nullable=False),
        pa.field("source_shard", pa.int32(), nullable=False),
        pa.field("fpps", pa.list_(FPP_SCHEMA), nullable=False),
        pa.field("estimated_output_bytes", pa.int64(), nullable=False),
        pa.field("total_tar_bytes", pa.int64(), nullable=False),
    ]
)

INVENTORY_COLUMNS = {
    "source_shard",
    "tar_id",
    "attempt",
    "tar_uri",
    "tar_size",
    "tar_etag",
    "last_modified",
}


def _snapshot_id(rows: list[PhysicalTar], target_pack_bytes: int) -> str:
    digest = hashlib.sha256(f"schema={MANIFEST_SCHEMA_VERSION}\0target_pack_bytes={target_pack_bytes}\n".encode())
    for row in sorted(rows, key=lambda item: item.tar_uri):
        digest.update(row.descriptor.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:24]


def _validated_inventory(inventory_path: str) -> list[PhysicalTar]:
    table = pq.read_table(inventory_path)
    missing = INVENTORY_COLUMNS - set(table.column_names)
    if missing:
        message = f"Inventory is missing required columns: {sorted(missing)}"
        raise ValueError(message)
    rows = [PhysicalTar(**row) for row in table.select(sorted(INVENTORY_COLUMNS)).to_pylist()]
    if not rows:
        message = "Inventory contains no tar objects"
        raise ValueError(message)

    identities: set[tuple[int, str, int]] = set()
    uris: set[str] = set()
    for row in rows:
        identity = (row.source_shard, row.tar_id, row.attempt)
        if identity in identities:
            message = f"Duplicate physical tar identity: {identity}"
            raise ValueError(message)
        if row.tar_uri in uris:
            message = f"Duplicate physical tar URI: {row.tar_uri}"
            raise ValueError(message)
        if row.tar_size <= 0:
            message = f"Tar has non-positive size: {row.tar_uri}"
            raise ValueError(message)
        identities.add(identity)
        uris.add(row.tar_uri)
    return sorted(rows, key=lambda item: (item.source_shard, item.tar_id, item.attempt))


def build_fpp_partitions(physical_tars: list[PhysicalTar], snapshot_id: str) -> list[FppPartition]:
    grouped: dict[tuple[int, str], list[PhysicalTar]] = defaultdict(list)
    for physical_tar in physical_tars:
        grouped[(physical_tar.source_shard, physical_tar.tar_id)].append(physical_tar)

    partitions = []
    for (source_shard, tar_id), attempts in sorted(grouped.items()):
        ordered = tuple(sorted(attempts, key=lambda item: item.attempt))
        partitions.append(
            FppPartition(
                snapshot_id=snapshot_id,
                fpp_id=f"shard_{source_shard:05d}/{tar_id}",
                source_shard=source_shard,
                tar_id=tar_id,
                attempts=ordered,
                total_tar_bytes=sum(item.tar_size for item in ordered),
                max_tar_bytes=max(item.tar_size for item in ordered),
            )
        )
    return partitions


def build_fpp_packs(partitions: list[FppPartition], target_pack_bytes: int) -> list[FppPack]:
    """Balance indivisible FPPs within each source shard toward a byte target."""
    if target_pack_bytes <= 0:
        message = "target_pack_bytes must be positive"
        raise ValueError(message)
    by_shard: dict[int, list[FppPartition]] = defaultdict(list)
    for partition in partitions:
        by_shard[partition.source_shard].append(partition)

    packs = []
    for source_shard, shard_fpps in sorted(by_shard.items()):
        estimated_total = sum(fpp.max_tar_bytes for fpp in shard_fpps)
        pack_count = max(1, int(estimated_total / target_pack_bytes + 0.5))
        pack_count = min(pack_count, len(shard_fpps))
        bins: list[list[FppPartition]] = [[] for _ in range(pack_count)]
        bin_bytes = [0] * pack_count
        for fpp in sorted(shard_fpps, key=lambda item: (-item.max_tar_bytes, item.tar_id)):
            bin_index = min(range(pack_count), key=lambda index: (bin_bytes[index], index))
            bins[bin_index].append(fpp)
            bin_bytes[bin_index] += fpp.max_tar_bytes

        ordered_bins = sorted(
            (tuple(sorted(items, key=lambda item: item.tar_id)) for items in bins if items),
            key=lambda items: items[0].tar_id,
        )
        for index, items in enumerate(ordered_bins):
            packs.append(
                FppPack(
                    snapshot_id=items[0].snapshot_id,
                    pack_id=f"shard_{source_shard:05d}/pack_{index:03d}",
                    source_shard=source_shard,
                    fpps=items,
                    estimated_output_bytes=sum(item.max_tar_bytes for item in items),
                    total_tar_bytes=sum(item.total_tar_bytes for item in items),
                )
            )
    return packs


def _fpp_dict(fpp: FppPartition) -> dict[str, Any]:
    return {
        **asdict(fpp),
        "attempts": [asdict(attempt) for attempt in fpp.attempts],
    }


def _pack_dict(pack: FppPack) -> dict[str, Any]:
    return {
        **asdict(pack),
        "fpps": [_fpp_dict(fpp) for fpp in pack.fpps],
    }


def build_manifest(inventory_path: str, manifest_dir: str, target_pack_bytes: int) -> dict[str, Any]:
    """Build the manifest once; refuse to overwrite a different snapshot."""
    physical_tars = _validated_inventory(inventory_path)
    snapshot_id = _snapshot_id(physical_tars, target_pack_bytes)
    partitions = build_fpp_partitions(physical_tars, snapshot_id)
    packs = build_fpp_packs(partitions, target_pack_bytes)
    metadata = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "target_pack_bytes": target_pack_bytes,
        "physical_tar_count": len(physical_tars),
        "fpp_count": len(partitions),
        "fpp_pack_count": len(packs),
        "source_shard_count": len({item.source_shard for item in physical_tars}),
        "physical_tar_bytes": sum(item.tar_size for item in physical_tars),
    }

    root = Path(manifest_dir)
    metadata_path = root / METADATA_FILENAME
    packs_path = root / PACKS_FILENAME
    if metadata_path.exists() or packs_path.exists():
        if metadata_path.exists() and packs_path.exists():
            existing = json.loads(metadata_path.read_text())
            if existing == metadata:
                return existing
        message = f"Refusing to overwrite a different manifest in {root}"
        raise ValueError(message)

    root.mkdir(parents=True, exist_ok=True)
    temporary_packs = packs_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist([_pack_dict(pack) for pack in packs], schema=PACK_SCHEMA), temporary_packs)
    temporary_packs.replace(packs_path)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary_metadata.replace(metadata_path)
    return metadata


def _physical_tar(row: dict[str, Any]) -> PhysicalTar:
    return PhysicalTar(**row)


def _fpp(row: dict[str, Any]) -> FppPartition:
    return FppPartition(
        snapshot_id=str(row["snapshot_id"]),
        fpp_id=str(row["fpp_id"]),
        source_shard=int(row["source_shard"]),
        tar_id=str(row["tar_id"]),
        attempts=tuple(_physical_tar(item) for item in row["attempts"]),
        total_tar_bytes=int(row["total_tar_bytes"]),
        max_tar_bytes=int(row["max_tar_bytes"]),
    )


def load_manifest(manifest_dir: str) -> tuple[dict[str, Any], list[FppPack]]:
    root = Path(manifest_dir)
    metadata = json.loads((root / METADATA_FILENAME).read_text())
    rows = pq.read_table(root / PACKS_FILENAME).to_pylist()
    packs = [
        FppPack(
            snapshot_id=str(row["snapshot_id"]),
            pack_id=str(row["pack_id"]),
            source_shard=int(row["source_shard"]),
            fpps=tuple(_fpp(item) for item in row["fpps"]),
            estimated_output_bytes=int(row["estimated_output_bytes"]),
            total_tar_bytes=int(row["total_tar_bytes"]),
        )
        for row in rows
    ]
    if {pack.snapshot_id for pack in packs} != {metadata["snapshot_id"]}:
        message = "Manifest metadata and FPP packs have different snapshot IDs"
        raise ValueError(message)
    return metadata, packs
