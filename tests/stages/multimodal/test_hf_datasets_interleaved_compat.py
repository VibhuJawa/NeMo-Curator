from __future__ import annotations

import json
import os
import tarfile
from io import BytesIO
from itertools import islice
from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.multimodal.io.readers.webdataset import WebDatasetReaderStage
from nemo_curator.tasks import FileGroupTask

if TYPE_CHECKING:
    from pathlib import Path

_LIVE_HF_DATASETS_DEFAULT = [
    "HuggingFaceH4/llava-instruct-mix-vsft",
    "HuggingFaceM4/the_cauldron:ai2d",
    "mlfoundations/MINT-1T-HTML",
    "OpenGVLab/OmniCorpus-CC:CC-MAIN-2013-20",
    "liuhaotian/LLaVA-Instruct-150K",
    "MMMU/MMMU:Accounting",
    "moca-embed/docmatix",
    "lmms-lab/TextCaps",
]


def _write_members(tar_path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows), encoding="utf-8")


def _expected_rows_from_hf_row(
    row: dict[str, object],
    field_map: dict[str, str],
    *,
    modalities_to_load: str = "all",
) -> list[tuple[str, int, str, str | None, str | None]]:
    sample_id = str(row[field_map["sample_id"]])
    segments = row[field_map["segments"]]
    assert isinstance(segments, list)
    expected: list[tuple[str, int, str, str | None, str | None]] = []
    for pos, segment_obj in enumerate(segments):
        assert isinstance(segment_obj, dict)
        modality = str(segment_obj[field_map["modality"]])
        if modalities_to_load == "text" and modality != "text":
            continue
        if modalities_to_load == "image" and modality == "text":
            continue
        if modality == "text":
            expected.append((sample_id, pos, "text", str(segment_obj[field_map["text"]]), None))
            continue
        expected.append((sample_id, pos, modality, None, str(segment_obj[field_map["content_key"]])))
    return expected


def _actual_rows(
    tar_path: Path,
    *,
    field_map: dict[str, str] | None = None,
    modalities_to_load: str = "all",
) -> list[tuple[str, int, str, str | None, str | None]]:
    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])
    stage = WebDatasetReaderStage(
        load_binary=False,
        sample_format="interleaved",
        modalities_to_load=modalities_to_load,  # type: ignore[arg-type]
        interleaved_field_map=field_map or WebDatasetReaderStage.default_interleaved_field_map(),
    )
    out = stage.process(task)
    return sorted(
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


def test_hf_datasets_default_interleaved_compat(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    record = {
        "sample_id": "doc1",
        "segments": [
            {"modality": "text", "text": "alpha"},
            {"modality": "image", "content_key": "doc1.000.jpg"},
            {"modality": "text", "text": "omega"},
        ],
    }
    tar_path = tmp_path / "default.tar"
    _write_members(tar_path, {"record.json": json.dumps(record, ensure_ascii=True).encode("utf-8")})
    jsonl_path = tmp_path / "default.jsonl"
    _write_jsonl(jsonl_path, [record])

    hf_row = datasets.load_dataset("json", data_files=str(jsonl_path), split="train")[0]
    expected = sorted(
        _expected_rows_from_hf_row(
            dict(hf_row),
            {
                "sample_id": "sample_id",
                "segments": "segments",
                "modality": "modality",
                "text": "text",
                "content_key": "content_key",
            },
        )
    )
    assert _actual_rows(tar_path) == expected


def test_hf_datasets_custom_map_interleaved_compat(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    record = {
        "sid": "doc2",
        "chunks": [
            {"kind": "text", "body": "hello"},
            {"kind": "image", "path": "doc2.000.png"},
        ],
    }
    field_map = {
        "sample_id": "sid",
        "segments": "chunks",
        "modality": "kind",
        "text": "body",
        "content_key": "path",
    }
    tar_path = tmp_path / "custom.tar"
    _write_members(tar_path, {"mapped.json": json.dumps(record, ensure_ascii=True).encode("utf-8")})
    jsonl_path = tmp_path / "custom.jsonl"
    _write_jsonl(jsonl_path, [record])

    hf_row = datasets.load_dataset("json", data_files=str(jsonl_path), split="train")[0]
    expected = sorted(_expected_rows_from_hf_row(dict(hf_row), field_map))
    assert _actual_rows(tar_path, field_map=field_map) == expected


def test_hf_datasets_text_only_filter_matches_expected(tmp_path: Path) -> None:
    datasets = pytest.importorskip("datasets")
    record = {
        "sample_id": "doc3",
        "segments": [
            {"modality": "image", "content_key": "doc3.000.jpg"},
            {"modality": "text", "text": "caption"},
        ],
    }
    tar_path = tmp_path / "text-only.tar"
    _write_members(tar_path, {"record.json": json.dumps(record, ensure_ascii=True).encode("utf-8")})
    jsonl_path = tmp_path / "text-only.jsonl"
    _write_jsonl(jsonl_path, [record])

    hf_row = datasets.load_dataset("json", data_files=str(jsonl_path), split="train")[0]
    expected = sorted(
        _expected_rows_from_hf_row(
            dict(hf_row),
            {
                "sample_id": "sample_id",
                "segments": "segments",
                "modality": "modality",
                "text": "text",
                "content_key": "content_key",
            },
            modalities_to_load="text",
        )
    )
    assert _actual_rows(tar_path, modalities_to_load="text") == expected


def _normalize_dataset_spec(dataset_spec: str) -> str:
    aliases = {
        "HuggingFaceM4/the_cauldron": "HuggingFaceM4/the_cauldron:ai2d",
        "MINT-1T": "mlfoundations/MINT-1T-HTML",
        "OpenGVLab/OmniCorpus": "OpenGVLab/OmniCorpus-CC:CC-MAIN-2013-20",
        "OpenGVLab/OmniCorpus-CC": "OpenGVLab/OmniCorpus-CC:CC-MAIN-2013-20",
    }
    return aliases.get(dataset_spec, dataset_spec)


def _parse_dataset_spec(dataset_spec: str) -> tuple[str, str | None]:
    normalized = _normalize_dataset_spec(dataset_spec)
    if ":" not in normalized:
        return normalized, None
    dataset_name, config_name = normalized.rsplit(":", 1)
    return dataset_name, config_name


def _preferred_split(dataset_name: str, config_name: str | None) -> str:
    datasets = pytest.importorskip("datasets")
    split_names = datasets.get_dataset_split_names(dataset_name, config_name)
    if split_names is None:
        msg = f"Could not resolve splits for dataset '{dataset_name}'"
        raise AssertionError(msg)
    preferred = ["train", "validation", "dev", "test"]
    for split in preferred:
        if split in split_names:
            return split
    return split_names[0]


def _first_live_row(dataset_spec: str) -> dict[str, object]:
    datasets = pytest.importorskip("datasets")
    dataset_name, config_name = _parse_dataset_spec(dataset_spec)
    kwargs: dict[str, object] = {"split": _preferred_split(dataset_name, config_name), "streaming": True}
    if config_name is not None:
        kwargs["name"] = config_name
    ds = datasets.load_dataset(dataset_name, **kwargs)
    return dict(next(iter(ds)))


def _iter_live_rows(dataset_spec: str, limit: int) -> list[dict[str, object]]:
    datasets = pytest.importorskip("datasets")
    dataset_name, config_name = _parse_dataset_spec(dataset_spec)
    kwargs: dict[str, object] = {"split": _preferred_split(dataset_name, config_name), "streaming": True}
    if config_name is not None:
        kwargs["name"] = config_name
    ds = datasets.load_dataset(dataset_name, **kwargs)
    return [dict(row) for row in islice(ds, limit)]


def _extract_message_texts(row: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for message in row.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            texts.append(content)
    return texts


def _extract_text_list_texts(row: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for text_obj in row.get("texts", []):
        if isinstance(text_obj, str) and text_obj:
            texts.append(text_obj)
            continue
        if isinstance(text_obj, dict):
            value = text_obj.get("text")
            if isinstance(value, str) and value:
                texts.append(value)
    return texts


def _extract_conversation_texts(row: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for conv in row.get("conversations", []):
        if not isinstance(conv, dict):
            continue
        value = conv.get("value")
        if isinstance(value, str) and value:
            texts.append(value)
    return texts


def _extract_scalar_texts(row: dict[str, object]) -> list[str]:
    texts: list[str] = []
    for key in ("caption", "text", "instruction", "response", "question", "hint", "subject"):
        value = row.get(key)
        if isinstance(value, str) and value:
            texts.append(value)
        elif isinstance(value, (list, dict)) and value:
            texts.append(json.dumps(value, ensure_ascii=True))
    return texts


def _extract_text_segments(row: dict[str, object]) -> list[str]:
    return [
        *_extract_message_texts(row),
        *_extract_text_list_texts(row),
        *_extract_conversation_texts(row),
        *_extract_scalar_texts(row),
    ]


def _image_count(row: dict[str, object]) -> int:
    count = 0
    images = row.get("images")
    if isinstance(images, list):
        count += len(images)
    image = row.get("image")
    if image is not None:
        count += 1
    for key in ("image_path", "img_path", "image_url"):
        if isinstance(row.get(key), str) and row.get(key):
            count += 1
    for key, value in row.items():
        if key.startswith("image_") and value is not None:
            count += 1
    return count


def _row_to_interleaved_payload(row: dict[str, object], fallback_id: str) -> dict[str, object]:
    sample_id = str(row.get("id") or row.get("sample_id") or row.get("uuid") or fallback_id)
    segments: list[dict[str, object]] = []
    for text in _extract_text_segments(row):
        segments.append({"modality": "text", "text": text})
    for image_idx in range(_image_count(row)):
        segments.append({"modality": "image", "content_key": f"{sample_id}.{image_idx:03d}.jpg"})
    if not segments:
        msg = "Live row did not expose recognizable text/image fields"
        raise ValueError(msg)
    return {"sample_id": sample_id, "segments": segments}


@pytest.mark.skipif(os.environ.get("HF_DATASETS_LIVE_TESTS") != "1", reason="Set HF_DATASETS_LIVE_TESTS=1 to run live HF tests")
def test_live_hf_llava_messages_images_projection(tmp_path: Path) -> None:
    row = _first_live_row("HuggingFaceH4/llava-instruct-mix-vsft")
    if "messages" not in row or "images" not in row:
        msg = "Dataset row does not expose expected keys: messages/images"
        raise AssertionError(msg)

    sample_id = str(row.get("id") or "live-llava")
    segments: list[dict[str, object]] = []
    for message in row.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            segments.append({"modality": "text", "text": content})
    for image_idx, _ in enumerate(row.get("images", [])):
        segments.append({"modality": "image", "content_key": f"{sample_id}.{image_idx:03d}.jpg"})
    if not segments:
        msg = "No usable text/image segments extracted from live row"
        raise AssertionError(msg)

    payload = {"sample_id": sample_id, "segments": segments}
    tar_path = tmp_path / "live-llava.tar"
    _write_members(tar_path, {"sample.json": json.dumps(payload, ensure_ascii=True).encode("utf-8")})

    actual = _actual_rows(tar_path)
    expected = sorted(_expected_rows_from_hf_row(payload, {"sample_id": "sample_id", "segments": "segments", "modality": "modality", "text": "text", "content_key": "content_key"}))
    assert actual == expected


@pytest.mark.skipif(os.environ.get("HF_DATASETS_LIVE_TESTS") != "1", reason="Set HF_DATASETS_LIVE_TESTS=1 to run live HF tests")
def test_live_hf_cauldron_texts_images_projection(tmp_path: Path) -> None:
    row = _first_live_row("HuggingFaceM4/the_cauldron:ai2d")
    if "texts" not in row or "images" not in row:
        msg = "Dataset row does not expose expected keys: texts/images"
        raise AssertionError(msg)

    sample_id = str(row.get("sample_id") or row.get("id") or "live-cauldron")
    segments: list[dict[str, object]] = []
    for text_obj in row.get("texts", []):
        if isinstance(text_obj, str) and text_obj:
            segments.append({"modality": "text", "text": text_obj})
        elif isinstance(text_obj, dict):
            value = text_obj.get("text")
            if isinstance(value, str) and value:
                segments.append({"modality": "text", "text": value})
    for image_idx, _ in enumerate(row.get("images", [])):
        segments.append({"modality": "image", "content_key": f"{sample_id}.{image_idx:03d}.jpg"})
    if not segments:
        msg = "No usable text/image segments extracted from live row"
        raise AssertionError(msg)

    payload = {"sample_id": sample_id, "segments": segments}
    tar_path = tmp_path / "live-cauldron.tar"
    _write_members(tar_path, {"sample.json": json.dumps(payload, ensure_ascii=True).encode("utf-8")})

    actual = _actual_rows(tar_path)
    expected = sorted(_expected_rows_from_hf_row(payload, {"sample_id": "sample_id", "segments": "segments", "modality": "modality", "text": "text", "content_key": "content_key"}))
    assert actual == expected


@pytest.mark.skipif(os.environ.get("HF_DATASETS_LIVE_TESTS") != "1", reason="Set HF_DATASETS_LIVE_TESTS=1 to run live HF tests")
@pytest.mark.parametrize(
    "dataset_name",
    os.environ.get("HF_LIVE_DATASET_IDS", ",".join(_LIVE_HF_DATASETS_DEFAULT)).split(","),
)
def test_live_hf_dataset_generic_projection(tmp_path: Path, dataset_name: str) -> None:
    dataset_name = dataset_name.strip()
    if not dataset_name:
        msg = "HF_LIVE_DATASET_IDS contains an empty dataset name"
        raise AssertionError(msg)
    max_rows = int(os.environ.get("HF_LIVE_MAX_ROWS", "100"))
    rows = _iter_live_rows(dataset_name, max_rows)
    if not rows:
        msg = f"Dataset '{dataset_name}' returned zero rows"
        raise AssertionError(msg)

    dataset_token = dataset_name.replace("/", "_").replace(":", "_")
    tar_path = tmp_path / f"{dataset_token}.tar"
    members: dict[str, bytes] = {}
    expected: list[tuple[str, int, str, str | None, str | None]] = []
    for idx, row in enumerate(rows):
        try:
            payload = _row_to_interleaved_payload(row, fallback_id=f"live-{dataset_token}-{idx:06d}")
        except ValueError as exc:
            msg = f"Dataset '{dataset_name}' row {idx} not recognized as text/image row: {exc}"
            raise AssertionError(msg) from exc
        members[f"sample_{idx:06d}.json"] = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        expected.extend(
            _expected_rows_from_hf_row(
                payload,
                {
                    "sample_id": "sample_id",
                    "segments": "segments",
                    "modality": "modality",
                    "text": "text",
                    "content_key": "content_key",
                },
            )
        )
    _write_members(tar_path, members)
    actual = _actual_rows(tar_path)
    expected = sorted(expected)
    assert actual == expected
