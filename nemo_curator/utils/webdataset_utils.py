# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

DEFAULT_BINARY_CONTENT_TYPE = "application/octet-stream"
IMAGE_SUFFIX_TO_CONTENT_TYPE: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".avif": "image/avif",
}
_INDEXED_STEM_RE = re.compile(r"^(?P<sample>.+)\.(?P<position>\d+)$")


def member_stem(member_name: str) -> str:
    """Return member stem while preserving directory components."""
    suffix = Path(member_name).suffix
    if not suffix:
        return member_name
    return member_name[: -len(suffix)]


def parse_sample_and_position(
    member_name: str,
    modality: str,
    sample_format: str,
    fallback_position: int,
) -> tuple[str, int]:
    """Derive ``(sample_id, position)`` from a WebDataset member name."""
    stem = member_stem(member_name)
    match = _INDEXED_STEM_RE.match(stem)
    if match:
        return match.group("sample"), int(match.group("position"))
    if sample_format == "simple":
        return stem, 0 if modality == "image" else 1
    if sample_format == "interleaved":
        return stem, fallback_position
    return stem, 0 if modality == "text" else 1


def content_type_from_name(name: str, default: str = DEFAULT_BINARY_CONTENT_TYPE) -> str:
    """Infer MIME type from a file name, with explicit image suffix overrides."""
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_SUFFIX_TO_CONTENT_TYPE:
        return IMAGE_SUFFIX_TO_CONTENT_TYPE[suffix]
    guessed, _ = mimetypes.guess_type(name)
    return guessed or default


def modality_from_content_type(content_type: str) -> str:
    """Map a MIME content type to a top-level modality label."""
    top_level = content_type.partition("/")[0]
    if top_level in {"image", "text", "audio", "video"}:
        return top_level
    return "unknown"


def webdataset_member_name(sample_id: str, position: int, suffix: str) -> str:
    """Build a deterministic WebDataset member name for one sample position."""
    return f"{sample_id}.{position:06d}.{suffix}" if suffix else f"{sample_id}.{position:06d}"
