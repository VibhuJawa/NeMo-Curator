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
import mimetypes
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.multimodal.utils import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_JSON_EXTENSIONS,
    require_source_id_field,
    resolve_storage_options,
    validate_and_project_source_fields,
)
from nemo_curator.tasks import FileGroupTask, MultiBatchTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA

from .base import BaseMultimodalReader


@dataclass
class _ReadContext:
    tar_path: str
    member_names: set[str]
    member_info: dict[str, tarfile.TarInfo]
    storage_options: dict[str, object]
    byte_cache: dict[str, bytes | None]


@dataclass
class WebdatasetReaderStage(BaseMultimodalReader):
    """Read MINT1T-style WebDataset shards into a row-wise multimodal task."""

    materialize_on_read: bool = False
    max_batch_bytes: int | None = None
    json_extensions: tuple[str, ...] = DEFAULT_JSON_EXTENSIONS
    image_extensions: tuple[str, ...] = field(default_factory=lambda: DEFAULT_IMAGE_EXTENSIONS)
    source_id_field: str = ""
    sample_id_field: str | None = None
    texts_field: str = "texts"
    images_field: str = "images"
    image_member_field: str | None = None
    fields: tuple[str, ...] | None = None
    name: str = "webdataset_reader"

    def __post_init__(self) -> None:
        self.source_id_field = require_source_id_field(self.source_id_field)

    def _rows_from_sample(
        self,
        sample_id: str,
        sample: dict[str, Any],
        source: dict[str, str],
        member_names: set[str],
        member_info: dict[str, tarfile.TarInfo] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        images = sample.get(self.images_field)
        image_member_name = self._resolve_default_image_member_name(sample_id, sample, images, member_names)
        tar_path = source["tar_path"]
        json_member_name = source["json_member_name"]
        passthrough_row = self._build_passthrough_row(sample)

        def build_source_ref(content_key: str | None) -> str:
            byte_offset = None
            byte_size = None
            if content_key and member_info and content_key in member_info:
                info = member_info[content_key]
                byte_offset = info.offset_data
                byte_size = info.size
            return MultiBatchTask.build_source_ref(
                path=tar_path,
                member=content_key,
                byte_offset=byte_offset,
                byte_size=byte_size,
            )

        def append_row(row: dict[str, Any]) -> None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "position": row["position"],
                    "modality": row["modality"],
                    "content_type": row.get("content_type"),
                    "text_content": row.get("text_content"),
                    "binary_content": row.get("binary_content"),
                    "source_ref": row.get("source_ref"),
                    "metadata_json": row.get("metadata_json"),
                    "materialize_error": None,
                    **passthrough_row,
                }
            )

        append_row(
            {
                "position": -1,
                "modality": "metadata",
                "content_type": "application/json",
                "source_ref": build_source_ref(json_member_name),
                "metadata_json": json.dumps(
                    {
                        **sample,
                        "_sample_source": {
                            "source_shard": Path(tar_path).name,
                            "tar_path": tar_path,
                            "json_member_name": json_member_name,
                        },
                    },
                    ensure_ascii=True,
                ),
            }
        )

        texts = sample.get(self.texts_field)
        if isinstance(texts, list):
            for idx, text_value in enumerate(texts):
                append_row(
                    {
                        "position": idx,
                        "modality": "text",
                        "content_type": "text/plain",
                        "text_content": text_value if isinstance(text_value, str) else None,
                        "source_ref": build_source_ref(json_member_name),
                    }
                )

        if isinstance(images, list):
            for idx, image_token in enumerate(images):
                content_key = self._resolve_image_content_key(image_token, image_member_name, member_names)
                content_type, _ = mimetypes.guess_type(content_key or image_member_name or "")
                append_row(
                    {
                        "position": idx,
                        "modality": "image",
                        "content_type": content_type or ("application/octet-stream" if image_member_name else None),
                        "source_ref": build_source_ref(content_key),
                    }
                )

        return rows

    def _empty_output_schema(self) -> pa.Schema:
        schema = MULTIMODAL_SCHEMA
        if not self.fields:
            return schema
        existing = set(schema.names)
        passthrough_fields = [pa.field(name, pa.null()) for name in self.fields if name not in existing]
        return pa.schema([*schema, *passthrough_fields]) if passthrough_fields else schema

    def _build_passthrough_row(self, sample: dict[str, Any]) -> dict[str, Any]:
        excluded = {
            self.source_id_field,
            *([self.sample_id_field] if self.sample_id_field else []),
            self.texts_field,
            self.images_field,
            *([self.image_member_field] if self.image_member_field else []),
            "sample_id",
            "position",
            "modality",
            "content_type",
            "text_content",
            "binary_content",
            "source_ref",
            "metadata_json",
            "materialize_error",
        }
        return validate_and_project_source_fields(sample=sample, fields=self.fields, excluded_fields=excluded)

    def _resolve_default_image_member_name(
        self,
        sample_id: str,
        sample: dict[str, Any],
        images: list[object] | None,
        member_names: set[str],
    ) -> str | None:
        if self.image_member_field:
            image_member_name = sample.get(self.image_member_field)
            if isinstance(image_member_name, str) and image_member_name in member_names:
                return image_member_name
        if isinstance(images, list):
            for image_token in images:
                if isinstance(image_token, str) and image_token in member_names:
                    return image_token
        return next(
            (f"{sample_id}{ext}" for ext in self.image_extensions if f"{sample_id}{ext}" in member_names), None
        )

    @staticmethod
    def _resolve_image_content_key(
        image_token: object,
        default_image_member_name: str | None,
        member_names: set[str],
    ) -> str | None:
        if image_token is None:
            return None
        if isinstance(image_token, str) and image_token in member_names:
            return image_token
        return default_image_member_name

    @staticmethod
    def _extract_tar_member(tf: tarfile.TarFile, member_name: str, cache: dict[str, bytes | None]) -> bytes | None:
        if member_name in cache:
            return cache[member_name]
        try:
            extracted = tf.extractfile(member_name)
        except KeyError:
            extracted = None
        payload = extracted.read() if extracted is not None else None
        cache[member_name] = payload
        return payload

    def _rows_from_member(
        self,
        tf: tarfile.TarFile,
        member: tarfile.TarInfo,
        context: _ReadContext,
    ) -> list[dict[str, Any]]:
        extracted = tf.extractfile(member)
        if extracted is None:
            return []
        payload = json.load(extracted)
        sample_id = (
            str(payload.get(self.sample_id_field))
            if self.sample_id_field and payload.get(self.sample_id_field) is not None
            else Path(member.name).stem
        )
        source = {
            "source_shard": Path(context.tar_path).name,
            "tar_path": context.tar_path,
            "json_member_name": member.name,
        }
        sample_rows = self._rows_from_sample(
            sample_id=sample_id,
            sample=payload,
            source=source,
            member_names=context.member_names,
            member_info=context.member_info,
        )
        if self.materialize_on_read:
            for row in sample_rows:
                if row["modality"] != "image" or row["position"] < 0:
                    continue
                parsed_ref = MultiBatchTask.parse_source_ref(row["source_ref"])
                content_key = parsed_ref.get("member")
                if content_key:
                    row["binary_content"] = self._extract_tar_member(tf, content_key, context.byte_cache)
        return sample_rows

    def process(self, task: FileGroupTask) -> MultiBatchTask | list[MultiBatchTask]:
        rows: list[dict[str, Any]] = []
        storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

        for tar_path in task.data:
            with (
                fsspec.open(tar_path, mode="rb", **storage_options) as fobj,
                tarfile.open(fileobj=fobj, mode="r:*") as tf,
            ):
                members = [m for m in tf.getmembers() if m.isfile()]
                member_names = {m.name for m in members}
                context = _ReadContext(
                    tar_path=tar_path,
                    member_names=member_names,
                    member_info={m.name: m for m in members},
                    storage_options=storage_options,
                    byte_cache={},
                )
                for member in members:
                    if not member.name.endswith(self.json_extensions):
                        continue
                    rows.extend(
                        self._rows_from_member(
                            tf=tf,
                            member=member,
                            context=context,
                        )
                    )

        table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=self._empty_output_schema())
        splits = split_table_by_group_max_bytes(table, "sample_id", self.max_batch_bytes)
        batches: list[MultiBatchTask] = []
        for idx, split in enumerate(splits):
            task_id = f"{task.task_id}_processed" if len(splits) == 1 else f"{task.task_id}_processed_{idx:05d}"
            metadata = dict(task._metadata)
            if storage_options:
                metadata["source_storage_options"] = storage_options
            batches.append(
                MultiBatchTask(
                    task_id=task_id,
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]
