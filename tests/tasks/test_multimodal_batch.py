from __future__ import annotations

import pyarrow as pa

from nemo_curator.tasks.multimodal import (
    METADATA_MODALITY,
    METADATA_POSITION,
    MULTIMODAL_SCHEMA,
    MultimodalBatch,
)


def _batch() -> MultimodalBatch:
    table = pa.table(
        {
            "sample_id": ["docA", "docA"],
            "position": [0, 1],
            "modality": ["text", "image"],
            "content_type": ["text/plain", "image/jpeg"],
            "text_content": ["hello", None],
            "binary_content": [None, b"img-1"],
            "element_metadata_json": [None, None],
            "source_id": ["src", "src"],
            "source_shard": ["shard-0", "shard-0"],
            "content_path": [None, None],
            "content_key": [None, "docA.000001.jpg"],
        },
        schema=MULTIMODAL_SCHEMA,
    )
    return MultimodalBatch(task_id="t0", dataset_name="ds", data=table)


def test_upsert_position_content_replaces_existing_text_row() -> None:
    batch = _batch()

    out = batch.upsert_position_content(
        sample_id="docA",
        position=0,
        modality="text",
        content_type="text/plain",
        text_content="updated",
    )

    rows = [row for row in out.data.to_pylist() if row["sample_id"] == "docA" and row["position"] == 0]
    assert len(rows) == 1
    assert rows[0]["modality"] == "text"
    assert rows[0]["text_content"] == "updated"


def test_delete_position_content_removes_only_targeted_row() -> None:
    batch = _batch()

    out = batch.delete_position_content(sample_id="docA", position=1, modality="image")

    rows = out.data.to_pylist()
    assert len(rows) == 1
    assert rows[0]["modality"] == "text"
    assert rows[0]["position"] == 0


def test_insert_position_content_shifts_existing_rows_for_sample() -> None:
    batch = _batch()

    out = batch.insert_position_content(
        sample_id="docA",
        position=1,
        modality="text",
        content_type="text/plain",
        text_content="between",
    )

    rows = sorted(out.data.to_pylist(), key=lambda row: int(row["position"]))
    assert [(row["position"], row["modality"], row["text_content"]) for row in rows] == [
        (0, "text", "hello"),
        (1, "text", "between"),
        (2, "image", None),
    ]


def test_upsert_sample_metadata_creates_metadata_row() -> None:
    batch = _batch()

    out = batch.upsert_sample_metadata(
        sample_id="docA",
        metadata_json={"license": "cc-by"},
        sample_type="interleaved",
    )

    metadata_rows = [
        row
        for row in out.data.to_pylist()
        if row["sample_id"] == "docA" and row["modality"] == METADATA_MODALITY and row["position"] == METADATA_POSITION
    ]
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["element_metadata_json"] == '{"license": "cc-by"}'
    assert metadata_rows[0]["text_content"] == '{"license": "cc-by"}'


def test_insert_position_content_does_not_shift_metadata_rows() -> None:
    batch = _batch().upsert_sample_metadata(sample_id="docA", metadata_json='{"a":1}')

    out = batch.insert_position_content(
        sample_id="docA",
        position=0,
        modality="text",
        content_type="text/plain",
        text_content="new0",
    )

    rows = out.data.to_pylist()
    metadata_rows = [
        row
        for row in rows
        if row["sample_id"] == "docA" and row["modality"] == METADATA_MODALITY
    ]
    assert len(metadata_rows) == 1
    assert metadata_rows[0]["position"] == METADATA_POSITION


def test_delete_sample_metadata_removes_metadata_row() -> None:
    batch = _batch().upsert_sample_metadata(sample_id="docA", metadata_json='{"a":1}')

    out = batch.delete_sample_metadata(sample_id="docA")

    rows = out.data.to_pylist()
    assert all(row["modality"] != METADATA_MODALITY for row in rows)
