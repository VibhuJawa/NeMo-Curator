# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Apply a setup-built host layout atlas during a measured extraction pass.

The atlas is deliberately a narrow execution primitive: setup produces an
immutable per-page route and cached text, while this stage verifies the raw
HTML identity and either restores the cached result or leaves the page for the
normal MinerU extractor.  It does not infer labels or make quality decisions
at runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.html_extraction.mineru_utils import STATUS_FIELD
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


ATLAS_ROUTE_FIELD = "_mineru_host_atlas_route"
ATLAS_CACHED_TEXT_FIELD = "_mineru_host_atlas_cached_text"


def _page_keys(
    frame: pd.DataFrame,
    *,
    url_field: str,
    page_id_field: str | None,
) -> list[str | tuple[str, int]]:
    if page_id_field:
        return ["" if pd.isna(value) else str(value) for value in frame[page_id_field]]
    occurrences = frame.groupby(url_field, sort=False, dropna=False).cumcount().tolist()
    return [
        (
            "" if pd.isna(frame[url_field].iloc[pos]) else str(frame[url_field].iloc[pos]),
            int(occurrence),
        )
        for pos, occurrence in enumerate(occurrences)
    ]


def _sha256(value: object) -> str:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8", "surrogatepass")
    elif value is None or pd.isna(value):
        payload = b""
    else:
        payload = bytes(value)
    return hashlib.sha256(payload).hexdigest()


class FrozenHostAtlasRouteStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Verify and install cached atlas actions before MinerU processing.

    Reuse rows receive their cached text and a ``layout_reused`` status.  The
    ordinary MinerU stages recognize that status and leave those rows alone,
    so the original HTML never has to be copied into another column.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        html_field: str = "content",
        text_field: str = "text",
        url_field: str = "url",
        *,
        page_id_field: str | None = None,
        strict: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.html_field = html_field
        self.text_field = text_field
        self.url_field = url_field
        self.page_id_field = page_id_field
        self.strict = strict
        self.resources = Resources(cpus=0.25)
        self.name = "mineru_html_frozen_host_atlas_route"
        self._actions: dict[str | tuple[str, int], tuple[str, str, str]] | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        fields = [self.url_field, self.html_field]
        if self.page_id_field:
            fields.append(self.page_id_field)
        return ["data"], fields

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            ATLAS_ROUTE_FIELD,
            ATLAS_CACHED_TEXT_FIELD,
            STATUS_FIELD,
        ]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        key_fields = [self.page_id_field] if self.page_id_field else ["url", "url_occurrence"]
        columns = [*key_fields, "route", "cached_text", "raw_html_sha256"]
        manifest = pd.read_parquet(self.manifest_path, columns=columns)
        missing = set(columns) - set(manifest.columns)
        if missing:
            raise ValueError(f"host atlas manifest is missing fields: {sorted(missing)}")
        invalid = set(manifest["route"].dropna().astype(str)) - {"reuse", "direct"}
        if invalid:
            raise ValueError(f"host atlas manifest has invalid routes: {sorted(invalid)}")

        keys: list[str | tuple[str, int]] = (
            manifest[self.page_id_field].astype(str).tolist()
            if self.page_id_field
            else list(zip(manifest["url"].astype(str), manifest["url_occurrence"].astype(int), strict=True))
        )
        actions: dict[str | tuple[str, int], tuple[str, str, str]] = {}
        for key, route, text, raw_hash in zip(
            keys,
            manifest["route"],
            manifest["cached_text"],
            manifest["raw_html_sha256"],
            strict=True,
        ):
            action = (str(route), "" if pd.isna(text) else str(text), str(raw_hash))
            if key in actions and actions[key] != action:
                raise ValueError(f"host atlas manifest has conflicting duplicate key: {key}")
            if action[0] == "reuse" and not action[1]:
                raise ValueError(f"host atlas reuse action has no cached text: {key}")
            actions[key] = action
        self._actions = actions

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        if self._actions is None:
            self.setup()
        assert self._actions is not None

        frame = batch.to_pandas().copy()
        keys = _page_keys(frame, url_field=self.url_field, page_id_field=self.page_id_field)
        routes: list[str] = []
        cached_texts = [""] * len(frame)
        statuses = frame[STATUS_FIELD].tolist() if STATUS_FIELD in frame else ["layout_pending"] * len(frame)
        missing = mismatches = reused = 0
        for pos, key in enumerate(keys):
            action = self._actions.get(key)
            if action is None:
                missing += 1
                if self.strict:
                    raise ValueError(f"host atlas action missing for key={key!r}")
                action = ("direct", "", "")
            route, cached_text, expected_hash = action
            actual_hash = _sha256(frame[self.html_field].iloc[pos])
            if route == "reuse" and actual_hash != expected_hash:
                mismatches += 1
                if self.strict:
                    raise ValueError(
                        f"host atlas raw HTML is stale for key={key!r}; "
                        f"expected={expected_hash}, actual={actual_hash}"
                    )
                route, cached_text = "direct", ""
            routes.append(route)
            if route == "reuse":
                reused += 1
                cached_texts[pos] = cached_text
                statuses[pos] = "layout_reused"

        frame[self.text_field] = cached_texts
        frame[ATLAS_ROUTE_FIELD] = routes
        frame[ATLAS_CACHED_TEXT_FIELD] = cached_texts
        frame[STATUS_FIELD] = statuses
        self._log_metrics(
            {
                "atlas_rows": float(len(frame)),
                "atlas_reuse_rows": float(reused),
                "atlas_direct_rows": float(len(frame) - reused),
                "atlas_missing_actions": float(missing),
                "atlas_html_hash_mismatches": float(mismatches),
            }
        )
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=frame,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class FrozenHostAtlasFinalizeStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Restore certified rows and remove atlas implementation columns."""

    def __init__(self, text_field: str = "text") -> None:
        self.text_field = text_field
        self.resources = Resources(cpus=0.1)
        self.name = "mineru_html_frozen_host_atlas_finalize"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [ATLAS_ROUTE_FIELD, ATLAS_CACHED_TEXT_FIELD]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [STATUS_FIELD, self.text_field]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        frame = batch.to_pandas().copy()
        reused = frame[ATLAS_ROUTE_FIELD].eq("reuse")
        frame.loc[reused, self.text_field] = frame.loc[reused, ATLAS_CACHED_TEXT_FIELD]
        frame.loc[reused, STATUS_FIELD] = "layout_reused"
        frame = frame.drop(
            columns=[ATLAS_ROUTE_FIELD, ATLAS_CACHED_TEXT_FIELD],
            errors="ignore",
        )
        self._log_metrics({"atlas_final_reused_rows": float(reused.sum())})
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=frame,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
