# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from nemo_curator.stages.text.html_extraction.mineru_atlas import (
    ATLAS_CACHED_TEXT_FIELD,
    ATLAS_ROUTE_FIELD,
    FrozenHostAtlasFinalizeStage,
    FrozenHostAtlasRouteStage,
)
from nemo_curator.stages.text.html_extraction.mineru_utils import STATUS_FIELD
from nemo_curator.tasks import DocumentBatch


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_route_restores_certified_page_after_normal_pipeline(tmp_path) -> None:  # noqa: ANN001
    raw_reuse = b"<html>cached</html>"
    raw_direct = b"<html>direct</html>"
    manifest = tmp_path / "atlas.parquet"
    pd.DataFrame(
        {
            "_atlas_page_id": ["p0", "p1"],
            "route": ["reuse", "direct"],
            "cached_text": ["cached markdown", ""],
            "raw_html_sha256": [_digest(raw_reuse), _digest(raw_direct)],
        }
    ).to_parquet(manifest, index=False)

    batch = DocumentBatch(
        dataset_name="atlas",
        data=pd.DataFrame(
            {
                "_atlas_page_id": ["p0", "p1"],
                "url": ["https://example.test/0", "https://example.test/1"],
                "content": [raw_reuse, raw_direct],
            }
        ),
    )
    routed = FrozenHostAtlasRouteStage(
        manifest,
        page_id_field="_atlas_page_id",
    ).process(batch)
    routed_frame = routed.to_pandas()
    assert routed_frame[ATLAS_ROUTE_FIELD].tolist() == ["reuse", "direct"]
    assert routed_frame["content"].tolist() == [raw_reuse, raw_direct]
    assert routed_frame[STATUS_FIELD].tolist() == ["layout_reused", "layout_pending"]

    # Simulate the normal extractor: direct rows receive their model output and
    # certified rows keep the cached text while their raw HTML was hidden.
    extracted = routed_frame.copy()
    extracted.loc[1, "text"] = "direct markdown"
    extracted.loc[1, STATUS_FIELD] = "ok"
    extracted_batch = DocumentBatch(dataset_name="atlas", data=extracted)
    finalized = FrozenHostAtlasFinalizeStage().process(extracted_batch).to_pandas()
    assert finalized["text"].tolist() == ["cached markdown", "direct markdown"]
    assert finalized["content"].tolist() == [raw_reuse, raw_direct]
    assert finalized[STATUS_FIELD].tolist() == ["layout_reused", "ok"]
    assert ATLAS_ROUTE_FIELD not in finalized
    assert ATLAS_CACHED_TEXT_FIELD not in finalized


def test_route_rejects_stale_html(tmp_path) -> None:  # noqa: ANN001
    manifest = tmp_path / "atlas.parquet"
    pd.DataFrame(
        {
            "_atlas_page_id": ["p0"],
            "route": ["reuse"],
            "cached_text": ["cached"],
            "raw_html_sha256": [_digest(b"old")],
        }
    ).to_parquet(manifest, index=False)
    batch = DocumentBatch(
        dataset_name="atlas",
        data=pd.DataFrame(
            {"_atlas_page_id": ["p0"], "url": ["https://example.test/0"], "content": [b"new"]}
        ),
    )
    with pytest.raises(ValueError, match="raw HTML is stale"):
        FrozenHostAtlasRouteStage(manifest, page_id_field="_atlas_page_id").process(batch)


def test_route_non_strictly_falls_back_to_direct(tmp_path) -> None:  # noqa: ANN001
    manifest = tmp_path / "atlas.parquet"
    pd.DataFrame(
        {
            "_atlas_page_id": ["p0"],
            "route": ["reuse"],
            "cached_text": ["cached"],
            "raw_html_sha256": [_digest(b"old")],
        }
    ).to_parquet(manifest, index=False)
    batch = DocumentBatch(
        dataset_name="atlas",
        data=pd.DataFrame(
            {"_atlas_page_id": ["missing"], "url": ["https://example.test/0"], "content": [b"new"]}
        ),
    )
    routed = FrozenHostAtlasRouteStage(
        manifest,
        page_id_field="_atlas_page_id",
        strict=False,
    ).process(batch).to_pandas()
    assert routed[ATLAS_ROUTE_FIELD].tolist() == ["direct"]
    assert routed["content"].tolist() == [b"new"]
