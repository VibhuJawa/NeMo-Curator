from __future__ import annotations

import json
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.multimodal import WebDatasetReaderStage
from nemo_curator.tasks import FileGroupTask

if TYPE_CHECKING:
    from pathlib import Path


def _write_members(tar_path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))


@pytest.mark.parametrize(
    ("name", "members", "reader_hints", "expected_rows"),
    [
        (
            "default_interleaved",
            {
                "sample.json": json.dumps(
                    {
                        "sample_id": "s1",
                        "segments": [
                            {"modality": "text", "text": "caption"},
                            {"modality": "image", "content_key": "s1.000.jpg"},
                        ],
                    },
                    ensure_ascii=True,
                ).encode("utf-8")
            },
            {"sample_format": "interleaved"},
            [("s1", 0, "text", "caption", None), ("s1", 1, "image", None, "s1.000.jpg")],
        ),
        (
            "custom_interleaved_mapping",
            {
                "sample.json": json.dumps(
                    {
                        "sid": "s2",
                        "chunks": [
                            {"kind": "text", "body": "hello"},
                            {"kind": "image", "path": "s2.000.png"},
                        ],
                    },
                    ensure_ascii=True,
                ).encode("utf-8")
            },
            {
                "sample_format": "interleaved",
                "interleaved_field_map": {
                    "sample_id": "sid",
                    "segments": "chunks",
                    "modality": "kind",
                    "text": "body",
                    "content_key": "path",
                },
            },
            [("s2", 0, "text", "hello", None), ("s2", 1, "image", None, "s2.000.png")],
        ),
        (
            "text_only_hint",
            {
                "sample.json": json.dumps(
                    {
                        "sample_id": "s3",
                        "segments": [
                            {"modality": "text", "text": "only text"},
                            {"modality": "image", "content_key": "s3.000.jpg"},
                        ],
                    },
                    ensure_ascii=True,
                ).encode("utf-8")
            },
            {"sample_format": "interleaved", "modalities_to_load": "text"},
            [("s3", 0, "text", "only text", None)],
        ),
    ],
)
def test_reader_surface_area_accepts_data_plus_hints(
    tmp_path: Path,
    name: str,
    members: dict[str, bytes],
    reader_hints: dict[str, object],
    expected_rows: list[tuple[str, int, str, str | None, str | None]],
) -> None:
    tar_path = tmp_path / f"{name}.tar"
    _write_members(tar_path, members)

    task = FileGroupTask(task_id=f"task-{name}", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, **reader_hints).process(task)
    actual_rows = sorted(
        [
            (
                str(row["sample_id"]),
                int(row["position"]),
                str(row["modality"]),
                str(row["text_content"]) if row["text_content"] is not None else None,
                str(row["content_key"]) if row["content_key"] is not None else None,
            )
            for row in out.data.to_pylist()
        ],
        key=lambda v: (v[0], v[1], v[2]),
    )
    assert actual_rows == sorted(expected_rows)


def test_reader_surface_area_rejects_invalid_hints() -> None:
    with pytest.raises(ValueError, match="interleaved_field_map has unknown keys"):
        WebDatasetReaderStage(
            sample_format="interleaved",
            interleaved_field_map={"unknown_key": "bad"},
        )
