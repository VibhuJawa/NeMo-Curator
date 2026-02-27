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

import json
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path


def generate_jpeg_bytes(width: int = 100, height: int = 80, seed: int = 0) -> bytes:
    img = Image.new("RGB", (width, height), color=(seed * 37 % 256, seed * 71 % 256, seed * 113 % 256))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def generate_png_bytes(width: int = 100, height: int = 80, seed: int = 0) -> bytes:
    img = Image.new("RGB", (width, height), color=(seed * 41 % 256, seed * 67 % 256, seed * 97 % 256))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_mint1t_tar(
    tmp_path: Path,
    samples: list[dict[str, Any]] | None = None,
    tar_name: str = "shard-00000.tar",
) -> str:
    if samples is None:
        samples = [
            {
                "sample_id": "sample_a",
                "json_payload": {
                    "pdf_name": "doc_a.pdf",
                    "texts": ["Hello world", "Second paragraph"],
                    "images": ["sample_a.jpg"],
                    "score": 0.95,
                },
                "image_bytes": generate_jpeg_bytes(seed=1),
                "image_ext": ".jpg",
            },
            {
                "sample_id": "sample_b",
                "json_payload": {
                    "pdf_name": "doc_b.pdf",
                    "texts": ["Another doc"],
                    "images": ["sample_b.jpg", None],
                    "score": 0.72,
                },
                "image_bytes": generate_jpeg_bytes(seed=2),
                "image_ext": ".jpg",
            },
        ]

    tmp_path.mkdir(parents=True, exist_ok=True)
    tar_path = tmp_path / tar_name
    with tarfile.open(tar_path, "w") as tf:
        for s in samples:
            sid = s["sample_id"]
            payload_bytes = json.dumps(s["json_payload"]).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{sid}.json")
            json_info.size = len(payload_bytes)
            tf.addfile(json_info, BytesIO(payload_bytes))

            img_bytes = s["image_bytes"]
            img_info = tarfile.TarInfo(name=f"{sid}{s['image_ext']}")
            img_info.size = len(img_bytes)
            tf.addfile(img_info, BytesIO(img_bytes))

    return str(tar_path)


def build_multimodal_parquet(
    tmp_path: Path,
    num_samples: int = 3,
    materialized: bool = True,
    file_name: str = "test_multimodal.parquet",
    image_dir: Path | None = None,
) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i in range(num_samples):
        sid = f"sample_{i:03d}"
        img_bytes = generate_jpeg_bytes(seed=i) if materialized else None

        source_ref_path = None
        if not materialized and image_dir is not None:
            img_file = image_dir / f"{sid}.jpg"
            img_file.write_bytes(generate_jpeg_bytes(seed=i))
            source_ref_path = str(img_file)

        source_ref = json.dumps({
            "path": source_ref_path,
            "member": None,
            "byte_offset": None,
            "byte_size": None,
        }) if source_ref_path else None

        metadata_json = json.dumps({
            "pdf_name": f"doc_{i}.pdf",
            "texts": [f"Text from sample {i}", f"More text {i}"],
            "images": [f"{sid}.jpg"],
            "score": 0.5 + i * 0.1,
        })

        rows.append({
            "sample_id": sid, "position": -1, "modality": "metadata",
            "content_type": "application/json", "text_content": None,
            "binary_content": None, "source_ref": None,
            "metadata_json": metadata_json, "materialize_error": None,
        })
        rows.append({
            "sample_id": sid, "position": 0, "modality": "text",
            "content_type": "text/plain", "text_content": f"Text from sample {i}",
            "binary_content": None, "source_ref": None,
            "metadata_json": None, "materialize_error": None,
        })
        rows.append({
            "sample_id": sid, "position": 1, "modality": "text",
            "content_type": "text/plain", "text_content": f"More text {i}",
            "binary_content": None, "source_ref": None,
            "metadata_json": None, "materialize_error": None,
        })
        rows.append({
            "sample_id": sid, "position": 0, "modality": "image",
            "content_type": "image/jpeg", "text_content": None,
            "binary_content": img_bytes, "source_ref": source_ref,
            "metadata_json": None, "materialize_error": None,
        })

    table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
    parquet_path = tmp_path / file_name
    pq.write_table(table, str(parquet_path))
    return str(parquet_path)


def build_bad_source_ref_parquet(
    tmp_path: Path,
    num_samples: int = 2,
    file_name: str = "bad_refs.parquet",
) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i in range(num_samples):
        sid = f"bad_sample_{i:03d}"
        bad_ref = json.dumps({
            "path": f"/nonexistent/path/image_{i}.jpg",
            "member": None,
            "byte_offset": None,
            "byte_size": None,
        })
        rows.append({
            "sample_id": sid, "position": -1, "modality": "metadata",
            "content_type": "application/json", "text_content": None,
            "binary_content": None, "source_ref": None,
            "metadata_json": json.dumps({"texts": [f"text {i}"], "images": [None]}),
            "materialize_error": None,
        })
        rows.append({
            "sample_id": sid, "position": 0, "modality": "text",
            "content_type": "text/plain", "text_content": f"text {i}",
            "binary_content": None, "source_ref": None,
            "metadata_json": None, "materialize_error": None,
        })
        rows.append({
            "sample_id": sid, "position": 0, "modality": "image",
            "content_type": "image/jpeg", "text_content": None,
            "binary_content": None, "source_ref": bad_ref,
            "metadata_json": None, "materialize_error": None,
        })

    table = pa.Table.from_pylist(rows, schema=MULTIMODAL_SCHEMA)
    parquet_path = tmp_path / file_name
    pq.write_table(table, str(parquet_path))
    return str(parquet_path)
