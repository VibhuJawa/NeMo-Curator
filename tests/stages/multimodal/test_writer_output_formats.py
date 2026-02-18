from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.multimodal import MultimodalWriterStage
from nemo_curator.stages.multimodal.io.readers.parquet import ParquetMultimodalReaderStage
from nemo_curator.stages.multimodal.io.readers.webdataset import WebDatasetReaderStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.multimodal import MULTIMODAL_SCHEMA, MultimodalBatch


def _sample_task(task_id: str = "t0") -> MultimodalBatch:
    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc"],
            "position": [0, 1, 2],
            "modality": ["text", "image", "text"],
            "content_type": ["text/plain", "image/jpeg", "text/plain"],
            "text_content": ["alpha", None, "omega"],
            "binary_content": [None, b"img", None],
            "element_metadata_json": [None, None, None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard", "shard", "shard"],
            "content_path": [None, None, None],
            "content_key": [None, "doc.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    return MultimodalBatch(task_id=task_id, dataset_name="ds", data=table)


def _read_output_rows(out: Path, output_format: str) -> list[dict[str, object]]:
    if output_format == "parquet":
        return pq.read_table(out).to_pylist()
    with pa.memory_map(str(out), "r") as source:
        return pa.ipc.open_file(source).read_all().to_pylist()


def _webdataset_task(task_id: str) -> MultimodalBatch:
    return MultimodalBatch(
        task_id=task_id,
        dataset_name="ds",
        data=pa.table(
            {
                "sample_id": ["doc", "doc"],
                "position": [0, 1],
                "modality": ["text", "image"],
                "content_type": ["text/plain", "image/jpeg"],
                "text_content": ["caption", None],
                "binary_content": [None, b"jpg-bytes"],
                "element_metadata_json": [None, None],
                "source_id": ["src", "src"],
                "source_shard": ["shard", "shard"],
                "content_path": [None, None],
                "content_key": [None, "doc.jpg"],
            },
            schema=MULTIMODAL_SCHEMA,
        ),
    )


def _lazy_image_task(task_id: str, image_path: Path, *, with_binary: bool) -> MultimodalBatch:
    table = pa.table(
        {
            "sample_id": ["doc", "doc"],
            "position": [0, 1],
            "modality": ["text", "image"],
            "content_type": ["text/plain", "image/jpeg"],
            "text_content": ["caption", None],
            "binary_content": [None, b"already-loaded" if with_binary else None],
            "element_metadata_json": [None, None],
            "source_id": ["src", "src"],
            "source_shard": ["shard", "shard"],
            "content_path": [None, str(image_path)],
            "content_key": [None, None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    return MultimodalBatch(task_id=task_id, dataset_name="ds", data=table)


def _lazy_mixed_image_task(tmp_path: Path, task_id: str) -> MultimodalBatch:
    good = tmp_path / "good.jpg"
    good.write_bytes(b"good-bytes")
    bad = tmp_path / "missing.jpg"
    table = pa.table(
        {
            "sample_id": ["good", "good", "bad", "bad"],
            "position": [0, 1, 0, 1],
            "modality": ["text", "image", "text", "image"],
            "content_type": ["text/plain", "image/jpeg", "text/plain", "image/jpeg"],
            "text_content": ["good-caption", None, "bad-caption", None],
            "binary_content": [None, None, None, None],
            "element_metadata_json": [None, None, None, None],
            "source_id": ["src", "src", "src", "src"],
            "source_shard": ["shard", "shard", "shard", "shard"],
            "content_path": [None, str(good), None, str(bad)],
            "content_key": [None, None, None, None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    return MultimodalBatch(task_id=task_id, dataset_name="ds", data=table)


def _read_tar_members(out: Path) -> tuple[list[str], dict[str, bytes]]:
    with tarfile.open(out, "r") as tf:
        names = [m.name for m in tf.getmembers()]
        return names, {m.name: tf.extractfile(m).read() for m in tf.getmembers()}


def _aggregate_text_by_sample(table: pa.Table) -> dict[str, str]:
    rows = sorted(table.to_pylist(), key=lambda row: (str(row["sample_id"]), int(row["position"])))
    by_sample: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["modality"] != "text":
            continue
        by_sample[str(row["sample_id"])].append(str(row["text_content"] or ""))
    return {sample_id: "\n".join(parts) for sample_id, parts in by_sample.items()}


def _image_payloads_by_sample(table: pa.Table) -> dict[str, list[bytes]]:
    rows = sorted(table.to_pylist(), key=lambda row: (str(row["sample_id"]), int(row["position"])))
    by_sample: dict[str, list[bytes]] = defaultdict(list)
    for row in rows:
        if row["modality"] != "image":
            continue
        payload = row["binary_content"]
        assert payload is not None
        by_sample[str(row["sample_id"])].append(bytes(payload))
    return dict(by_sample)


@pytest.mark.parametrize(
    ("writer_kwargs", "output_format"),
    [
        ({"output_path": "out.parquet"}, "parquet"),
        ({"output_path": "out.arrow", "output_format": "arrow"}, "arrow"),
    ],
)
def test_writer_roundtrip_formats(
    tmp_path: Path,
    writer_kwargs: dict[str, str],
    output_format: str,
) -> None:
    kwargs = {
        key: str(tmp_path / value) if key == "output_path" else value
        for key, value in writer_kwargs.items()
    }
    stage = MultimodalWriterStage(**kwargs)
    result = stage.process(_sample_task(task_id="t0"))

    assert len(result.data) == 1
    output_file = Path(result.data[0])
    assert output_file.exists()
    assert result._metadata["data_output_path"] == str(output_file)

    rows = sorted(_read_output_rows(output_file, output_format), key=lambda r: int(r["position"]))
    assert set(rows[0]) == set(MULTIMODAL_SCHEMA.names)
    assert {r["modality"] for r in rows} == {"image", "text"}
    assert [r["text_content"] for r in rows] == ["alpha", None, "omega"]
    assert [r["content_key"] for r in rows] == [None, "doc.jpg", None]

@pytest.mark.parametrize(("name", "output_format"), [("out.parquet", "parquet"), ("out.arrow", "arrow")])
def test_tabular_writer_preserves_extra_columns(tmp_path: Path, name: str, output_format: str) -> None:
    table = pa.table(
        {
            "sample_id": ["doc", "doc"],
            "position": [0, 1],
            "modality": ["text", "image"],
            "content_type": ["text/plain", "image/jpeg"],
            "text_content": ["caption", None],
            "binary_content": [None, b"img"],
            "element_metadata_json": [None, None],
            "source_id": ["src", "src"],
            "source_shard": ["shard", "shard"],
            "content_path": [None, None],
            "content_key": [None, "doc.jpg"],
            "quality_score": [0.9, 0.1],
        },
        schema=pa.schema(
            [*MULTIMODAL_SCHEMA, pa.field("quality_score", pa.float64())],
        ),
    )
    task = MultimodalBatch(task_id="extra-cols", dataset_name="ds", data=table)
    out = MultimodalWriterStage(output_path=str(tmp_path / name), output_format=output_format).process(task)
    rows = _read_output_rows(Path(out.data[0]), output_format)
    assert "quality_score" in rows[0]
    assert [float(row["quality_score"]) for row in rows] == [0.9, 0.1]


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"output_path": "out.any", "output_format": "csv"}, "Unsupported output_format"),
        ({}, "requires output_path"),
    ],
)
def test_writer_validation_errors(tmp_path: Path, kwargs: dict[str, str], error_match: str) -> None:
    resolved = {
        key: str(tmp_path / value) if key == "output_path" else value
        for key, value in kwargs.items()
    }
    with pytest.raises(ValueError, match=error_match):
        MultimodalWriterStage(**resolved)


def test_writer_rejects_invalid_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        MultimodalWriterStage(
            output_path=str(tmp_path / "out.parquet"),
            output_format="parquet",
            mode="append",  # type: ignore[arg-type]
        )


def test_writer_ignore_mode_skips_existing_outputs(tmp_path: Path) -> None:
    output_base = str(tmp_path / "out.parquet")
    task = _sample_task(task_id="ignore-0")
    first = MultimodalWriterStage(output_path=output_base, output_format="parquet", mode="overwrite").process(task)
    second = MultimodalWriterStage(output_path=output_base, output_format="parquet", mode="ignore").process(task)
    assert second.data == first.data
    assert Path(second.data[0]).exists()


def test_writer_outputs_are_isolated_per_task(tmp_path: Path) -> None:
    stage = MultimodalWriterStage(output_path=str(tmp_path / "out.parquet"), output_format="parquet")
    t0 = stage.process(_sample_task(task_id="task-0"))
    t1 = stage.process(_sample_task(task_id="task-1"))
    assert t0.data[0] != t1.data[0]
    assert Path(t0.data[0]).exists()
    assert Path(t1.data[0]).exists()


def test_multimodal_batch_get_content_paths_source_modes() -> None:
    batch = _sample_task(task_id="paths")
    assert batch.get_content_paths(modality="image", source="all") == []
    assert batch.get_content_paths(modality="image", source="content_key") == []
    assert batch.get_content_paths(modality="image", source="direct") == []

    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc"],
            "position": [0, 1, 2],
            "modality": ["text", "image", "image"],
            "content_type": ["text/plain", "image/jpeg", "image/jpeg"],
            "text_content": ["caption", None, None],
            "binary_content": [None, None, None],
            "element_metadata_json": [None, None, None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard", "shard", "shard"],
            "content_path": [None, "s3://bucket/shard.tar", "file:///data/image.jpg"],
            "content_key": [None, "doc.000001.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    batch = MultimodalBatch(task_id="paths2", dataset_name="ds", data=table)
    assert batch.get_content_paths(modality="image", source="all") == ["s3://bucket/shard.tar", "file:///data/image.jpg"]
    assert batch.get_content_paths(modality="image", source="content_key") == ["s3://bucket/shard.tar"]
    assert batch.get_content_paths(modality="image", source="direct") == ["file:///data/image.jpg"]
    with pytest.raises(ValueError, match="Unsupported source"):
        batch.get_content_paths(modality="image", source="bad")  # type: ignore[arg-type]


def test_multimodal_batch_materialize_rejects_mixed_loading_modes_per_content_path() -> None:
    table = pa.table(
        {
            "sample_id": ["doc", "doc"],
            "position": [0, 1],
            "modality": ["image", "image"],
            "content_type": ["image/jpeg", "image/jpeg"],
            "text_content": [None, None],
            "binary_content": [None, None],
            "element_metadata_json": [None, None],
            "source_id": ["src", "src"],
            "source_shard": ["shard", "shard"],
            "content_path": ["s3://bucket/shared.tar", "s3://bucket/shared.tar"],
            "content_key": ["doc.000000.jpg", None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    batch = MultimodalBatch(task_id="mixed-load-modes", dataset_name="ds", data=table)
    with pytest.raises(ValueError, match="Invalid mixed loading modes"):
        batch.materialize(modality="image")


def test_multimodal_batch_materialize_rejects_empty_content_key() -> None:
    table = pa.table(
        {
            "sample_id": ["doc"],
            "position": [0],
            "modality": ["image"],
            "content_type": ["image/jpeg"],
            "text_content": [None],
            "binary_content": [None],
            "element_metadata_json": [None],
            "source_id": ["src"],
            "source_shard": ["shard"],
            "content_path": ["s3://bucket/shared.tar"],
            "content_key": [""],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    batch = MultimodalBatch(task_id="empty-key", dataset_name="ds", data=table)
    with pytest.raises(ValueError, match="non-empty string"):
        batch.materialize(modality="image")


def test_webdataset_writer_writes_tar_members(tmp_path: Path) -> None:
    out = tmp_path / "out.tar"
    stage = MultimodalWriterStage(output_path=str(out), output_format="webdataset")
    result = stage.process(_webdataset_task(task_id="t2"))
    assert len(result.data) == 1
    output_file = Path(result.data[0])
    names, members = _read_tar_members(output_file)
    assert names == ["doc.000000.json", "doc.000001.jpg"]
    payload = json.loads(members["doc.000000.json"].decode("utf-8"))
    assert payload["sample_id"] == "doc"
    assert payload["texts"] == ["caption", None]
    assert payload["images"] == [None, "doc.000001"]
    assert members["doc.000001.jpg"] == b"jpg-bytes"


@pytest.mark.parametrize(
    "case",
    [
        ("out.parquet", "parquet", "materialize", False, b"image-bytes"),
        ("out.arrow", "arrow", "materialize", False, b"image-bytes"),
        ("out.parquet", "parquet", "dematerialize", True, None),
        ("out.arrow", "arrow", "dematerialize", True, None),
        ("out.parquet", "parquet", "preserve", True, b"already-loaded"),
        ("out.arrow", "arrow", "preserve", True, b"already-loaded"),
    ],
)
def test_tabular_writer_image_payload_policies(
    tmp_path: Path,
    case: tuple[str, str, str, bool, bytes | None],
) -> None:
    name, output_format, policy, with_binary, expected = case
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"image-bytes")
    task = _lazy_image_task(f"lazy-{policy}", image_path, with_binary=with_binary)

    out = MultimodalWriterStage(
        output_path=str(tmp_path / name),
        output_format=output_format,
        image_payload_policy=policy,  # type: ignore[arg-type]
    ).process(task)

    rows = _read_output_rows(Path(out.data[0]), output_format)
    image_rows = [row for row in rows if row["modality"] == "image"]
    assert len(image_rows) == 1
    payload = image_rows[0]["binary_content"]
    if expected is None:
        assert payload is None
    else:
        assert payload is not None
        assert bytes(payload) == expected


def test_webdataset_writer_rejects_dematerialize_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatible with webdataset output"):
        MultimodalWriterStage(
            output_path=str(tmp_path / "out.tar"),
            output_format="webdataset",
            image_payload_policy="dematerialize",
        )


def test_webdataset_writer_preserve_policy_materializes_lazy_batch(tmp_path: Path) -> None:
    image_path = tmp_path / "img.jpg"
    payload = b"image-bytes"
    image_path.write_bytes(payload)
    task = _lazy_image_task("lazy-preserve", image_path, with_binary=False)
    out = MultimodalWriterStage(
        output_path=str(tmp_path / "out.tar"),
        output_format="webdataset",
        image_payload_policy="preserve",
    ).process(task)
    names, members = _read_tar_members(Path(out.data[0]))
    image_members = [name for name in names if name.startswith("doc.000001.")]
    assert len(image_members) == 1
    assert members[image_members[0]] == payload


def test_materialize_skip_keeps_failed_rows_lazy(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    task = _lazy_image_task("skip-fail", missing, with_binary=False)
    out = task.materialize(modality="image", on_error="skip", max_retries=1, retry_backoff_sec=0.0)
    assert out.is_lazy


def test_webdataset_writer_materialize_failure_raises_by_default(tmp_path: Path) -> None:
    task = _lazy_mixed_image_task(tmp_path, task_id="raise-fail")
    with pytest.raises(FileNotFoundError):
        MultimodalWriterStage(
            output_path=str(tmp_path / "out.tar"),
            output_format="webdataset",
            image_payload_policy="preserve",
            materialize_max_retries=0,
        ).process(task)


def test_webdataset_writer_drop_image_rows_on_materialize_failure(tmp_path: Path) -> None:
    task = _lazy_mixed_image_task(tmp_path, task_id="drop-fail")
    out = MultimodalWriterStage(
        output_path=str(tmp_path / "out.tar"),
        output_format="webdataset",
        image_payload_policy="preserve",
        materialize_failure_policy="drop_image",
        materialize_max_retries=0,
    ).process(task)

    names, members = _read_tar_members(Path(out.data[0]))
    assert names == ["bad.000000.json", "good.000000.json", "good.000001.jpeg"]
    bad_payload = json.loads(members["bad.000000.json"].decode("utf-8"))
    good_payload = json.loads(members["good.000000.json"].decode("utf-8"))
    assert bad_payload["texts"] == ["bad-caption"]
    assert good_payload["texts"] == ["good-caption", None]
    assert members["good.000001.jpeg"] == b"good-bytes"


def test_webdataset_writer_drop_image_failure_drops_orphan_metadata_rows(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"
    table = pa.table(
        {
            "sample_id": ["badimg"],
            "position": [0],
            "modality": ["image"],
            "content_type": ["image/jpeg"],
            "text_content": [None],
            "binary_content": [None],
            "element_metadata_json": [None],
            "source_id": ["src"],
            "source_shard": ["shard"],
            "content_path": [str(missing)],
            "content_key": [None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultimodalBatch(task_id="drop-orphan-meta", dataset_name="ds", data=table)

    out = MultimodalWriterStage(
        output_path=str(tmp_path / "out.tar"),
        output_format="webdataset",
        image_payload_policy="preserve",
        materialize_failure_policy="drop_image",
        materialize_max_retries=0,
    ).process(task)

    names, _ = _read_tar_members(Path(out.data[0]))
    assert names == []


def test_webdataset_writer_multiple_text_rows_collapses_into_single_json_member(tmp_path: Path) -> None:
    out = tmp_path / "single_text_no_data_loss.tar"
    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc"],
            "position": [0, 1, 2],
            "modality": ["text", "text", "image"],
            "content_type": ["text/plain", "text/plain", "image/jpeg"],
            "text_content": ["alpha", "omega", None],
            "binary_content": [None, None, b"img"],
            "element_metadata_json": [None, None, None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard", "shard", "shard"],
            "content_path": [None, None, None],
            "content_key": [None, None, "doc.jpg"],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultimodalBatch(task_id="t7", dataset_name="ds", data=table)
    result = MultimodalWriterStage(output_path=str(out), output_format="webdataset").process(task)
    names, members = _read_tar_members(Path(result.data[0]))

    text_like_members = [name for name in names if name.endswith((".txt", ".json"))]
    assert text_like_members == ["doc.000000.json"]
    assert names == ["doc.000000.json", "doc.000002.jpg"]
    payload = json.loads(members["doc.000000.json"].decode("utf-8"))
    assert payload["sample_id"] == "doc"
    assert [segment["text"] for segment in payload["segments"]] == ["alpha", "omega"]


def test_webdataset_writer_collapsed_text_preserves_element_metadata_json(tmp_path: Path) -> None:
    out = tmp_path / "collapsed_text_metadata.tar"
    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc"],
            "position": [0, 1, 2],
            "modality": ["text", "text", "image"],
            "content_type": ["text/plain", "text/plain", "image/jpeg"],
            "text_content": ["alpha", "omega", None],
            "binary_content": [None, None, b"img"],
            "element_metadata_json": ['{"quality": 0.9}', '{"lang": "en"}', None],
            "source_id": ["src", "src", "src"],
            "source_shard": ["shard", "shard", "shard"],
            "content_path": [None, None, None],
            "content_key": [None, None, "doc.jpg"],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultimodalBatch(task_id="t-meta", dataset_name="ds", data=table)
    result = MultimodalWriterStage(output_path=str(out), output_format="webdataset").process(task)
    names, members = _read_tar_members(Path(result.data[0]))

    assert names == ["doc.000000.json", "doc.000002.jpg"]
    payload = json.loads(members["doc.000000.json"].decode("utf-8"))
    assert payload["sample_id"] == "doc"
    assert [segment["text"] for segment in payload["segments"]] == ["alpha", "omega"]
    assert payload["segments"][0]["element_metadata_json"]["quality"] == 0.9
    assert payload["segments"][1]["element_metadata_json"]["lang"] == "en"

    roundtrip = WebDatasetReaderStage(load_binary=False, sample_format="auto").process(
        FileGroupTask(task_id="rt-meta", dataset_name="ds", data=[result.data[0]])
    )
    rows = sorted(
        [row for row in roundtrip.data.to_pylist() if row["modality"] == "text"],
        key=lambda row: int(row["position"]),
    )
    assert [row["text_content"] for row in rows] == ["alpha", "omega"]
    assert json.loads(str(rows[0]["element_metadata_json"]))["element_metadata_json"]["quality"] == 0.9
    assert json.loads(str(rows[1]["element_metadata_json"]))["element_metadata_json"]["lang"] == "en"


def test_webdataset_writer_collapsed_text_writes_full_segment_metadata_payload(tmp_path: Path) -> None:
    out = tmp_path / "collapsed_text_full_metadata.tar"
    table = pa.table(
        {
            "sample_id": ["doc", "doc", "doc", "doc"],
            "position": [0, 1, 2, 3],
            "modality": ["text", "text", "text", "image"],
            "content_type": ["text/plain", "text/plain", "text/plain", "image/jpeg"],
            "text_content": ["alpha", "beta", "gamma", None],
            "binary_content": [None, None, None, b"img"],
            "element_metadata_json": [
                '{"quality": 0.91, "token_count": 1}',
                '{"quality": 0.77, "lang": "en", "attrs": {"source": "ocr"}}',
                '{"quality": 0.55, "tags": ["x", "y"]}',
                None,
            ],
            "source_id": ["src", "src", "src", "src"],
            "source_shard": ["shard", "shard", "shard", "shard"],
            "content_path": [None, None, None, None],
            "content_key": [None, None, None, "doc.jpg"],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultimodalBatch(task_id="t-full-meta", dataset_name="ds", data=table)
    result = MultimodalWriterStage(output_path=str(out), output_format="webdataset").process(task)
    names, members = _read_tar_members(Path(result.data[0]))

    assert names == ["doc.000000.json", "doc.000003.jpg"]
    payload = json.loads(members["doc.000000.json"].decode("utf-8"))
    assert payload["sample_id"] == "doc"
    assert [segment["text"] for segment in payload["segments"]] == ["alpha", "beta", "gamma"]
    assert payload["segments"][0]["element_metadata_json"] == {"quality": 0.91, "token_count": 1}
    assert payload["segments"][1]["element_metadata_json"] == {
        "quality": 0.77,
        "lang": "en",
        "attrs": {"source": "ocr"},
    }
    assert payload["segments"][2]["element_metadata_json"] == {"quality": 0.55, "tags": ["x", "y"]}

    roundtrip = WebDatasetReaderStage(load_binary=False, sample_format="auto").process(
        FileGroupTask(task_id="rt-full-meta", dataset_name="ds", data=[result.data[0]])
    )
    rows = sorted(
        [row for row in roundtrip.data.to_pylist() if row["modality"] == "text"],
        key=lambda row: int(row["position"]),
    )
    assert [row["text_content"] for row in rows] == ["alpha", "beta", "gamma"]
    assert json.loads(str(rows[0]["element_metadata_json"]))["element_metadata_json"] == {"quality": 0.91, "token_count": 1}
    assert json.loads(str(rows[1]["element_metadata_json"]))["element_metadata_json"] == {
        "quality": 0.77,
        "lang": "en",
        "attrs": {"source": "ocr"},
    }
    assert json.loads(str(rows[2]["element_metadata_json"]))["element_metadata_json"] == {"quality": 0.55, "tags": ["x", "y"]}


def test_webdataset_writer_allows_text_only_batch(tmp_path: Path) -> None:
    out = tmp_path / "text-only.tar"
    table = pa.table(
        {
            "sample_id": ["doc"],
            "position": [0],
            "modality": ["text"],
            "content_type": ["text/plain"],
            "text_content": ["caption"],
            "binary_content": [None],
            "element_metadata_json": [None],
            "source_id": ["src"],
            "source_shard": ["shard"],
            "content_path": [None],
            "content_key": [None],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    task = MultimodalBatch(task_id="text-only", dataset_name="ds", data=table)
    result = MultimodalWriterStage(output_path=str(out), output_format="webdataset").process(task)
    names, members = _read_tar_members(Path(result.data[0]))
    assert names == ["doc.000000.json"]
    payload = json.loads(members["doc.000000.json"].decode("utf-8"))
    assert payload["texts"] == ["caption"]
    assert payload["images"] == [None]


def test_webdataset_reader_writer_reader_roundtrip_preserves_semantic_payloads(tmp_path: Path) -> None:
    in_tar = tmp_path / "in_realistic.tar"
    out_tar = tmp_path / "out_realistic.tar"
    with tarfile.open(in_tar, "w") as tf:
        for name, payload in {
            "docA.000000.txt": b"alpha",
            "docA.000001.jpg": b"jpg-a",
            "docA.000002.txt": b"omega",
            "docB.000000.json": b'{"caption":"json-caption"}',
            "docB.000001.png": b"png-b",
            "nested/docC.000000.txt": b"nested-caption",
            "nested/docC.000001.jpeg": b"jpeg-c",
        }.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))

    original_task = FileGroupTask(task_id="rt0", dataset_name="ds", data=[str(in_tar)])
    original_batch = WebDatasetReaderStage(load_binary=True, sample_format="auto").process(original_task)

    written = MultimodalWriterStage(output_path=str(out_tar), output_format="webdataset").process(original_batch)
    roundtrip_task = FileGroupTask(task_id="rt1", dataset_name="ds", data=[written.data[0]])
    roundtrip_batch = WebDatasetReaderStage(load_binary=True, sample_format="auto").process(roundtrip_task)

    assert _aggregate_text_by_sample(roundtrip_batch.data) == _aggregate_text_by_sample(original_batch.data)
    assert _image_payloads_by_sample(roundtrip_batch.data) == _image_payloads_by_sample(original_batch.data)


def test_webdataset_interleaved_rows_store_element_metadata_json(tmp_path: Path) -> None:
    tar_path = tmp_path / "interleaved_meta.tar"
    payload = {
        "sample_id": "docX",
        "source": "synthetic",
        "segments": [
            {"modality": "text", "text": "hello", "quality": 0.9},
            {"modality": "image", "content_key": "docX.000001.jpg", "width": 1024, "height": 768},
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    with tarfile.open(tar_path, "w") as tf:
        info = tarfile.TarInfo(name="docX.json")
        info.size = len(encoded)
        tf.addfile(info, BytesIO(encoded))

    task = FileGroupTask(task_id="wds-meta", dataset_name="ds", data=[str(tar_path)])
    out = WebDatasetReaderStage(load_binary=False, sample_format="interleaved", error_handling="raise").process(task)
    rows = out.data.sort_by([("position", "ascending")]).to_pylist()
    assert len(rows) == 3
    assert rows[0]["modality"] == "metadata"
    assert rows[0]["position"] == -1
    assert rows[1]["modality"] == "text"
    assert rows[2]["modality"] == "image"
    assert json.loads(str(rows[1]["element_metadata_json"]))["quality"] == 0.9
    assert json.loads(str(rows[2]["element_metadata_json"]))["width"] == 1024
    metadata_payload = json.loads(str(rows[0]["text_content"]))
    assert metadata_payload["sample_id"] == "docX"
    assert metadata_payload["source"] == "synthetic"
    assert "segments" not in metadata_payload


def test_webdataset_reader_does_not_swallow_unexpected_runtime_error(tmp_path: Path) -> None:
    tar_path = tmp_path / "bad.tar"
    with tarfile.open(tar_path, "w") as tf:
        payload = b"hello"
        info = tarfile.TarInfo(name="doc.000000.txt")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

    class _ExplodingReader(WebDatasetReaderStage):
        def _rows_from_member(  # type: ignore[override]
            self,
            _state: object,
            _member_name: str,
            _payload: bytes | None,
            _source: object,
        ) -> list[dict[str, object]]:
            msg = "unexpected bug"
            raise RuntimeError(msg)

    task = FileGroupTask(task_id="boom", dataset_name="ds", data=[str(tar_path)])
    with pytest.raises(RuntimeError, match="unexpected bug"):
        _ExplodingReader(error_handling="log").process(task)


def test_webdataset_reader_does_not_swallow_unexpected_value_error(tmp_path: Path) -> None:
    tar_path = tmp_path / "bad_value_error.tar"
    with tarfile.open(tar_path, "w") as tf:
        payload = b"hello"
        info = tarfile.TarInfo(name="doc.000000.txt")
        info.size = len(payload)
        tf.addfile(info, BytesIO(payload))

    class _ExplodingReader(WebDatasetReaderStage):
        def _rows_from_member(  # type: ignore[override]
            self,
            _state: object,
            _member_name: str,
            _payload: bytes | None,
            _source: object,
        ) -> list[dict[str, object]]:
            msg = "unexpected value bug"
            raise ValueError(msg)

    task = FileGroupTask(task_id="boom-value", dataset_name="ds", data=[str(tar_path)])
    with pytest.raises(ValueError, match="unexpected value bug"):
        _ExplodingReader(error_handling="log").process(task)


def test_parquet_reader_writer_reader_roundtrip_preserves_rows(tmp_path: Path) -> None:
    in_data = tmp_path / "in.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["docA", "docA", "docB"],
                "position": [0, 1, 0],
                "modality": ["text", "image", "text"],
                "content_type": ["text/plain", "image/jpeg", "application/json"],
                "text_content": ["caption-a", None, '{"caption":"b"}'],
                "binary_content": [None, b"img-a", None],
                "element_metadata_json": [None, None, None],
                "source_id": ["src", "src", "src"],
                "source_shard": ["shard-0", "shard-0", "shard-1"],
                "content_path": [None, "s3://bucket/shard-0.tar", None],
                "content_key": [None, "docA.000001.jpg", None],
            },
            schema=MULTIMODAL_SCHEMA,
        ),
        in_data,
    )

    data_task = FileGroupTask(task_id="in_data", dataset_name="ds", data=[str(in_data)])
    original = ParquetMultimodalReaderStage().process(data_task)

    written = MultimodalWriterStage(output_path=str(tmp_path / "out.parquet"), output_format="parquet").process(original)
    out_data_task = FileGroupTask(task_id="out_data", dataset_name="ds", data=[written.data[0]])
    roundtrip = ParquetMultimodalReaderStage().process(out_data_task)

    assert roundtrip.data.to_pylist() == original.data.to_pylist()
