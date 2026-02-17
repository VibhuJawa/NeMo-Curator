# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from nemo_curator.utils.file_utils import open_binary_reader, open_tar_path

if TYPE_CHECKING:
    import pyarrow as pa


def cast_required_fields(table: pa.Table, required_schema: pa.Schema) -> pa.Table:
    """Cast required fields in-place while preserving any extra columns."""
    out = table
    for required_field in required_schema:
        col_idx = out.schema.get_field_index(required_field.name)
        if col_idx >= 0:
            col = out[required_field.name]
            if not col.type.equals(required_field.type):
                out = out.set_column(col_idx, required_field.name, col.cast(required_field.type))
    return out


def validate_content_path_loading_mode(*, content_path: str, row_indices: list[int], content_keys: list[object | None]) -> None:
    """Ensure one content_path group uses a single loading mode."""
    for idx in row_indices:
        key = content_keys[idx]
        if key is not None and (not isinstance(key, str) or not key):
            msg = (
                f"Invalid content_key for content_path='{content_path}' at row index {idx}. "
                "content_key must be a non-empty string when provided."
            )
            raise ValueError(msg)

    has_key_flags = [content_keys[idx] is not None for idx in row_indices]
    has_member_backed_rows = any(has_key_flags)
    has_direct_backed_rows = not all(has_key_flags)
    if has_member_backed_rows and has_direct_backed_rows:
        msg = (
            f"Invalid mixed loading modes for content_path='{content_path}'. "
            "Rows for the same content_path must be all content_key-backed or all direct-backed."
        )
        raise ValueError(msg)


def load_payloads_from_tar_members(
    *,
    content_path: str,
    keyed_rows: dict[int, str],
    storage_options: dict[str, Any],
) -> dict[int, bytes]:
    """Load payloads by ``content_key`` from a tar container path."""
    # ``content_path`` points to a container here, so rows map to tar member keys.
    required_keys = set(keyed_rows.values())
    extracted_by_key: dict[str, bytes] = {}
    with open_tar_path(content_path, storage_options) as tf:
        for member in tf:
            if member.name in required_keys:
                payload = tf.extractfile(member)
                if payload is not None:
                    extracted_by_key[member.name] = payload.read()
    missing_keys = sorted(required_keys - extracted_by_key.keys())
    if missing_keys:
        msg = f"Missing tar member '{missing_keys[0]}' in '{content_path}'"
        raise FileNotFoundError(msg)
    return {idx: extracted_by_key[key] for idx, key in keyed_rows.items()}


def load_payloads_from_direct_path(*, content_path: str, row_indices: list[int], storage_options: dict[str, Any]) -> dict[int, bytes]:
    """Load one direct-file payload and assign it to all row indices."""
    # Direct-path rows share one payload blob for all indices at this path.
    with open_binary_reader(content_path, storage_options) as f:
        payload = f.read()
    return dict.fromkeys(row_indices, payload)


def sort_multimodal_table(table: pa.Table) -> pa.Table:
    """Sort rows by ``sample_id``, ``position``, and ``modality``."""
    if table.num_rows == 0:
        return table
    return table.sort_by([("sample_id", "ascending"), ("position", "ascending"), ("modality", "ascending")])


def metadata_map_from_tables(metadata_tables: list[pa.Table]) -> dict[str, str]:
    """Build first-wins sample->metadata_json map from metadata tables."""
    metadata_by_sample: dict[str, str] = {}
    for metadata_table in metadata_tables:
        if metadata_table.num_rows == 0 or "sample_id" not in metadata_table.column_names:
            continue
        sample_ids = metadata_table["sample_id"].to_pylist()
        if "metadata_json" in metadata_table.column_names:
            metadata_json_values = metadata_table["metadata_json"].to_pylist()
        else:
            metadata_json_values = [None] * len(sample_ids)
        for sample_id, metadata_json in zip(sample_ids, metadata_json_values, strict=True):
            if isinstance(metadata_json, str):
                metadata_by_sample.setdefault(str(sample_id), metadata_json)
    return metadata_by_sample


def sample_type_from_summary(num_rows: int, image_count: int, text_count: int) -> str:
    """Infer sample type from modality counts."""
    if num_rows == 1:
        return "single"
    if text_count == num_rows:
        return "multi_text"
    if image_count == num_rows:
        return "multi_image"
    return "interleaved"


def metadata_rows_for_table(
    table: pa.Table,
    metadata_by_sample: dict[str, str],
) -> list[dict[str, object]]:
    """Build metadata rows with single-pass sample type inference."""
    if table.num_rows == 0:
        return []

    sample_stats: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
    for sample_id, modality in zip(table["sample_id"].to_pylist(), table["modality"].to_pylist(), strict=True):
        sid = str(sample_id)
        modality_name = str(modality)
        count, image_count, text_count = sample_stats.get(sid, (0, 0, 0))
        sample_stats[sid] = (
            count + 1,
            image_count + int(modality_name == "image"),
            text_count + int(modality_name == "text"),
        )

    return [
        {
            "sample_id": sid,
            "sample_type": sample_type_from_summary(
                num_rows=num_rows,
                image_count=image_count,
                text_count=text_count,
            ),
            "metadata_json": metadata_by_sample.get(sid),
        }
        for sid, (num_rows, image_count, text_count) in sample_stats.items()
    ]
