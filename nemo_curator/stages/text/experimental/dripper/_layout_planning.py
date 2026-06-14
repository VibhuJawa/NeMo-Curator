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

"""Layout-group planning helpers for DripperHTMLLayoutTemplateStage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd  # noqa: TC002 — used at runtime (df.iterrows, df.iloc, etc.)
from loguru import logger

from nemo_curator.stages.text.experimental.dripper._url_helpers import (
    _coerce_item_count,
    _layout_dom_path_fingerprint,
    _layout_feature_fingerprint,
    _layout_page_signature_key,
    _layout_page_signature_key_with_low_card_queries,
    _low_card_query_value_keys,
    _url_host_key,
    _validation_query_values,
)
from nemo_curator.stages.text.experimental.dripper.stage import (
    _DRIPPER_NEEDS_LLM_COL,
    _coerce_html,
    _is_missing,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nemo_curator.stages.text.experimental.dripper.layout_template import (
        DripperLayoutAdvancedConfig,
    )
    from nemo_curator.stages.text.experimental.dripper.stage import (
        _LLMWebKitBindings,
    )

# Column name duplicated here to avoid a circular import with layout_template.py.
_DRIPPER_ITEM_COUNT_COL = "dripper_item_count"
_MAX_EXEMPLARS_PER_LAYOUT = 3


@dataclass(frozen=True)
class _LayoutGroupPlan:
    indexes: list[int]
    host_key: str = ""
    source: str = "dom"
    fallback_groups: tuple[list[int], ...] = ()


@dataclass(frozen=True)
class _LayoutPlanningConfig:
    html_col: str
    url_col: str | None
    host_col: str | None
    layout_id_col: str | None
    layout_cluster_threshold: float
    min_cluster_size: int
    adv: DripperLayoutAdvancedConfig
    web_bindings: _LLMWebKitBindings | None


def _build_layout_group_plans(cfg: _LayoutPlanningConfig, df: pd.DataFrame) -> list[_LayoutGroupPlan]:
    if len(df) < cfg.min_cluster_size:
        return []
    precomputed_plans = _build_precomputed_layout_group_plans(cfg, df)
    if precomputed_plans is not None:
        return precomputed_plans

    samples_by_host = _build_host_samples(cfg, df)
    return _build_plans_from_host_samples(cfg, df, samples_by_host)


def _build_host_samples(cfg: _LayoutPlanningConfig, df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in df.iterrows():
        if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
            continue
        html_text = _coerce_html(row.get(cfg.html_col, ""))
        if not html_text.strip():
            continue
        try:
            feature = cfg.web_bindings.get_feature(html_text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Dripper layout feature extraction failed for row {}: {}", idx, exc)
            continue
        if feature is None:
            continue
        samples_by_host[_row_host_key(cfg, row)].append({"track_id": str(idx), "html": html_text, "feature": feature})
    return samples_by_host


def _build_plans_from_host_samples(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    samples_by_host: dict[str, list[dict[str, Any]]],
) -> list[_LayoutGroupPlan]:
    plans: list[_LayoutGroupPlan] = []
    adv = cfg.adv
    for host_key, samples in samples_by_host.items():
        if len(samples) < cfg.min_cluster_size:
            continue
        host_indexes = sorted(int(sample["track_id"]) for sample in samples)
        fallback_groups = _build_layout_groups_for_host_samples(cfg, df, host_key, samples)
        n = len(samples)
        try_single = (
            adv.host_single_cluster_min_pages > 0
            and n >= adv.host_single_cluster_min_pages
            and not (adv.host_single_cluster_max_pages > 0 and n > adv.host_single_cluster_max_pages)
        )
        if try_single:
            plans.append(
                _LayoutGroupPlan(
                    indexes=host_indexes,
                    host_key=host_key,
                    source="host_single_cluster",
                    fallback_groups=tuple(fallback_groups),
                )
            )
            continue
        for indexes in fallback_groups:
            plans.append(
                _LayoutGroupPlan(
                    indexes=indexes,
                    host_key=host_key,
                    source="dom",
                    fallback_groups=tuple(_build_failed_layout_fallback_groups(cfg, df, indexes)),
                )
            )
    return plans


def _build_precomputed_layout_group_plans(
    cfg: _LayoutPlanningConfig, df: pd.DataFrame
) -> list[_LayoutGroupPlan] | None:
    if not cfg.layout_id_col or cfg.layout_id_col not in df.columns:
        return None

    by_layout: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        if not bool(row.get(_DRIPPER_NEEDS_LLM_COL, False)):
            continue
        html_text = _coerce_html(row.get(cfg.html_col, ""))
        if not html_text.strip():
            continue
        layout_key = _row_layout_id_key(cfg, row)
        if not layout_key:
            continue
        by_layout[(_row_host_key(cfg, row), layout_key)].append(int(idx))

    plans: list[_LayoutGroupPlan] = []
    for (host_key, layout_key), indexes in sorted(by_layout.items(), key=lambda item: (min(item[1]), item[0])):
        sorted_indexes = sorted(indexes)
        if len(sorted_indexes) < cfg.min_cluster_size:
            continue
        plan_groups = _split_large_precomputed_layout_group(cfg, df, host_key, layout_key, sorted_indexes)
        for plan_indexes in plan_groups:
            if len(plan_indexes) < cfg.min_cluster_size:
                continue
            plans.append(
                _LayoutGroupPlan(
                    indexes=plan_indexes,
                    host_key=host_key,
                    source=f"precomputed_layout:{layout_key}",
                    fallback_groups=tuple(_build_failed_layout_fallback_groups(cfg, df, plan_indexes)),
                )
            )
    return plans


def _split_large_precomputed_layout_group(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    host_key: str,
    _layout_key: str,
    indexes: list[int],
) -> list[list[int]]:
    adv = cfg.adv
    if not adv.max_exact_host_pages or len(indexes) <= adv.max_exact_host_pages:
        return [indexes]
    if adv.large_host_mode == "standalone":
        return []

    samples: list[dict[str, Any]] = []
    for idx in indexes:
        html_text = _coerce_html(df.iloc[idx].get(cfg.html_col, ""))
        if not html_text.strip():
            continue
        sample: dict[str, Any] = {"track_id": str(idx), "html": html_text}
        if adv.large_host_mode == "feature_hash":
            try:
                feature = cfg.web_bindings.get_feature(html_text) if cfg.web_bindings else None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper precomputed layout feature extraction failed for row {}: {}", idx, exc)
                continue
            if feature is None:
                continue
            sample["feature"] = feature
        samples.append(sample)
    fingerprint_fn = (
        (lambda sample: _layout_feature_fingerprint(sample.get("feature")))
        if adv.large_host_mode == "feature_hash"
        else (lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or "")))
    )
    return _build_fingerprint_groups(cfg, df, host_key, samples, fingerprint_fn=fingerprint_fn)


def _row_host_key(cfg: _LayoutPlanningConfig, row: pd.Series) -> str:
    if cfg.host_col and cfg.host_col in row:
        host_key = _url_host_key(row.get(cfg.host_col))
        if host_key:
            return host_key
    return _url_host_key(row.get(cfg.url_col) if cfg.url_col else None)


def _row_layout_id_key(cfg: _LayoutPlanningConfig, row: pd.Series) -> str:
    if not cfg.layout_id_col:
        return ""
    value = row.get(cfg.layout_id_col)
    text = "" if _is_missing(value) else str(value).strip()
    if not text or text in {"-1", "-2"} or text.endswith(("_-1", "_-2")):
        return ""
    return text


def _build_layout_groups_for_host_samples(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    host_key: str,
    samples: list[dict[str, Any]],
) -> list[list[int]]:
    if len(samples) < cfg.min_cluster_size:
        return []

    # Large-host fast path: skip clustering, use fingerprint bucketing instead.
    adv = cfg.adv
    if adv.max_exact_host_pages and len(samples) > adv.max_exact_host_pages:
        if adv.large_host_mode == "feature_hash":
            fingerprint_fn = lambda sample: _layout_feature_fingerprint(sample.get("feature"))  # noqa: E731
        elif adv.large_host_mode == "dom_path_hash":
            fingerprint_fn = lambda sample: _layout_dom_path_fingerprint(str(sample.get("html") or ""))  # noqa: E731
        else:
            return []
        return _build_fingerprint_groups(cfg, df, host_key, samples, fingerprint_fn=fingerprint_fn)

    try:
        clustered_samples, _layout_ids = cfg.web_bindings.cluster_html_struct(
            samples,
            threshold=cfg.layout_cluster_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Dripper layout clustering failed for host {}: {}", host_key, exc)
        return []

    if not clustered_samples:
        return []
    return _build_clustered_host_groups(cfg, df, host_key, clustered_samples)


def _build_clustered_host_groups(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    _host_key: str,
    clustered_samples: list[dict[str, Any]],
) -> list[list[int]]:
    max_layer_n = int(
        next((s.get("max_layer_n") for s in clustered_samples if int(s.get("layout_id", -1)) >= 0), None) or 5
    )
    exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in clustered_samples:
        layout_id = int(sample.get("layout_id", -1))
        if layout_id < 0:
            continue
        if len(exemplars_by_layout[layout_id]) < _MAX_EXEMPLARS_PER_LAYOUT:
            exemplars_by_layout[layout_id].append(sample)

    by_layout: dict[tuple[int, str], list[int]] = defaultdict(list)
    for sample in clustered_samples:
        layout_id = _assign_layout_by_exemplar_similarity(cfg, sample.get("feature"), exemplars_by_layout, max_layer_n)
        if layout_id < 0:
            continue
        row_idx = int(sample["track_id"])
        _row = df.iloc[row_idx]
        signature_key = _layout_page_signature_key(
            _row.get(cfg.url_col) if cfg.url_col else None,
            _row.get(_DRIPPER_ITEM_COUNT_COL),
            cfg.adv.page_signature_mode,
        )
        by_layout[(layout_id, signature_key)].append(row_idx)
    groups: list[list[int]] = []
    for (_layout_id, _signature_key), indexes in sorted(by_layout.items()):
        if len(indexes) >= cfg.min_cluster_size:
            groups.append(sorted(indexes))
    return groups


def _build_failed_layout_fallback_groups(
    cfg: _LayoutPlanningConfig, df: pd.DataFrame, indexes: list[int]
) -> list[list[int]]:
    mode = cfg.adv.failed_layout_fallback_signature_mode
    if mode == "none" or len(indexes) < cfg.min_cluster_size:
        return []

    children = _split_fallback_groups_by_signature(cfg, df, [indexes], mode)
    parent_set = set(indexes)
    return [child for child in children if set(child) != parent_set]


def _assign_layout_by_exemplar_similarity(
    cfg: _LayoutPlanningConfig,
    feature: object,
    exemplars_by_layout: dict[int, list[dict[str, Any]]],
    max_layer_n: int,
) -> int:
    for layout_id, exemplars in sorted(exemplars_by_layout.items()):
        for exemplar in exemplars:
            try:
                score = cfg.web_bindings.similarity(feature, exemplar.get("feature"), max_layer_n)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Dripper layout similarity failed for layout {}: {}", layout_id, exc)
                continue
            if score is not None and score >= cfg.layout_cluster_threshold:
                return layout_id
    return -2


def _build_fingerprint_groups(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    _host_key: str,
    samples: list[dict[str, Any]],
    *,
    fingerprint_fn: Callable[[dict[str, Any]], str],
) -> list[list[int]]:
    by_fingerprint: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        by_fingerprint[fingerprint_fn(sample)].append(int(sample["track_id"]))

    groups: list[list[int]] = []
    for _fingerprint, indexes in sorted(by_fingerprint.items(), key=lambda item: (min(item[1]), item[0])):
        by_signature: dict[str, list[int]] = defaultdict(list)
        for row_idx in indexes:
            _row = df.iloc[row_idx]
            signature_key = _layout_page_signature_key(
                _row.get(cfg.url_col) if cfg.url_col else None,
                _row.get(_DRIPPER_ITEM_COUNT_COL),
                cfg.adv.page_signature_mode,
            )
            by_signature[signature_key].append(row_idx)
        for _signature_key, signature_indexes in sorted(by_signature.items()):
            if len(signature_indexes) < cfg.min_cluster_size:
                continue
            groups.append(sorted(signature_indexes))
    return groups


def _split_fallback_groups_by_signature(
    cfg: _LayoutPlanningConfig,
    df: pd.DataFrame,
    groups: list[list[int]],
    mode: str,
) -> list[list[int]]:
    split_groups: list[list[int]] = []
    for group in groups:
        low_card_query_keys: set[str] = set()
        if "url_low_card_query_shape" in mode and cfg.url_col:
            low_card_query_keys = _low_card_query_value_keys([df.iloc[row_idx].get(cfg.url_col) for row_idx in group])
        by_signature: dict[str, list[int]] = defaultdict(list)
        use_low_card = "url_low_card_query_shape" in mode
        for row_idx in group:
            row = df.iloc[row_idx]
            url = row.get(cfg.url_col) if cfg.url_col else None
            if use_low_card:
                signature_key = _layout_page_signature_key_with_low_card_queries(
                    url, row.get(_DRIPPER_ITEM_COUNT_COL), mode, low_card_query_keys
                )
            else:
                signature_key = _layout_page_signature_key(url, row.get(_DRIPPER_ITEM_COUNT_COL), mode)
            by_signature[signature_key].append(row_idx)
        for _signature, indexes in sorted(by_signature.items(), key=lambda item: (min(item[1]), item[0])):
            if len(indexes) >= cfg.min_cluster_size:
                split_groups.append(sorted(indexes))
    return split_groups


_QUERY_POSITIONS_THRESHOLD = 8
_QUERY_POSITIONS_HIGH = 4
_QUERY_POSITIONS_LOW = 3

_ColSpec = tuple[str | None, str]


@dataclass
class _SelectorState:
    selected: list[int]
    selected_set: set[int]
    count: int
    url_col: str | None
    item_count_col: str

    def add(self, idx: int) -> None:
        if len(self.selected) >= self.count or idx in self.selected_set:
            return
        self.selected.append(idx)
        self.selected_set.add(idx)

    def is_full(self) -> bool:
        return len(self.selected) >= self.count


def _select_by_signature(
    df: pd.DataFrame,
    indexes: list[int],
    *,
    signature_mode: str,
    state: _SelectorState,
) -> bool:
    url_col = state.url_col
    item_count_col = state.item_count_col
    low_card_query_keys: set[str] = set()
    if "url_low_card_query_shape" in signature_mode and url_col:
        low_card_query_keys = _low_card_query_value_keys([df.iloc[idx].get(url_col) for idx in indexes])
    by_signature: dict[str, list[int]] = defaultdict(list)
    for idx in indexes:
        row = df.iloc[idx]
        signature_key = _layout_page_signature_key_with_low_card_queries(
            row.get(url_col) if url_col else None,
            row.get(item_count_col) if item_count_col in row else None,
            signature_mode,
            low_card_query_keys,
        )
        by_signature[signature_key].append(idx)
    signature_groups = sorted(
        by_signature.values(),
        key=lambda group: (-len(group), _validation_sample_key(df.iloc[group[0]], group[0], url_col, item_count_col)),
    )
    for group in signature_groups:
        for idx in _select_validation_indexes(df, sorted(group), 1, (url_col, item_count_col), signature_mode="none"):
            state.add(idx)
            break
        if state.is_full():
            return True
    return False


def _select_by_url(
    df: pd.DataFrame,
    indexes: list[int],
    *,
    state: _SelectorState,
) -> None:
    url_col = state.url_col
    count = state.count
    query_value_rows: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx in indexes:
        url_text = str(df.iloc[idx].get(url_col) or "")
        for key, value in _validation_query_values(url_text):
            query_value_rows[key].append((value, idx))
    for key in sorted(query_value_rows):
        entries = sorted(query_value_rows[key])
        query_positions = _QUERY_POSITIONS_HIGH if count >= _QUERY_POSITIONS_THRESHOLD else _QUERY_POSITIONS_LOW
        for position in _spread_positions(len(entries), min(count, query_positions)):
            state.add(entries[position][1])
        if state.is_full():
            return

    url_sorted = sorted(indexes, key=lambda idx: (str(df.iloc[idx].get(url_col) or ""), idx))
    for position in _spread_positions(len(url_sorted), count):
        state.add(url_sorted[position])
        if state.is_full():
            return


def _select_validation_indexes(
    df: pd.DataFrame,
    indexes: list[int],
    count: int,
    cols: _ColSpec,
    *,
    signature_mode: str = "none",
) -> list[int]:
    url_col, item_count_col = cols
    if count <= 0 or not indexes:
        return []
    if count >= len(indexes):
        return list(indexes)
    if count == 1:
        return [indexes[-1]]

    state = _SelectorState(
        selected=[], selected_set=set(), count=count, url_col=url_col, item_count_col=item_count_col
    )

    if (
        signature_mode
        and signature_mode != "none"
        and _select_by_signature(df, indexes, signature_mode=signature_mode, state=state)
    ):
        return sorted(state.selected)

    state.add(indexes[0])
    state.add(indexes[-1])

    item_sorted = sorted(indexes, key=lambda idx: (_coerce_item_count(df.iloc[idx].get(item_count_col)), idx))
    state.add(item_sorted[0])
    state.add(item_sorted[-1])

    if url_col:
        _select_by_url(df, indexes, state=state)
        if state.is_full():
            return sorted(state.selected)

    remaining = [idx for idx in indexes if idx not in state.selected_set]
    remaining.sort(key=lambda idx: _validation_sample_key(df.iloc[idx], idx, url_col, item_count_col))
    for idx in remaining:
        state.add(idx)
        if state.is_full():
            break
    return sorted(state.selected)


def _spread_positions(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(slot * (length - 1) / (count - 1)) for slot in range(count)})


def _validation_sample_key(
    row: pd.Series,
    row_index: int,
    url_col: str | None,
    item_count_col: str,
) -> tuple[int, int]:
    import hashlib

    url_text = str(row.get(url_col) or "") if url_col else ""
    item_count = str(row.get(item_count_col) or "")
    payload = f"{url_text}\0{item_count}\0{row_index}".encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False), row_index
