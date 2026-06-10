from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import pandas as pd

from llm_web_kit.html_layout.html_layout_cosin import cluster_html_struct, get_feature, similarity
from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser
from llm_web_kit.main_html_parser.parser.tag_mapping import MapItemToHtmlTagsParser
from llm_web_kit.main_html_parser.typical_html.typical_html import select_representative_html
from mineru_html.base import (
    MinerUHTMLCase,
    MinerUHTMLGenerateOutput,
    MinerUHTMLInput,
    MinerUHTMLOutput,
    MinerUHTMLProcessData,
)
from mineru_html.process import convert2content, parse_result, simplify_single_input
from mineru_html.process.map_to_main import extract_main_html


ITEM_ID_RE = re.compile(r"""_item_id\s*=\s*["']?([^"'\s>]+)""")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
LAYOUT_TAGS_TO_IGNORE = {"script", "style", "meta", "link", "br", "noscript"}
LAYOUT_TAGS_IGNORE_ATTR = {"a", "i", "b", "li", "tr", "td", "img", "p", "body"}
LAYOUT_RE_MD5 = re.compile(r"^[0-9a-f]{32}$")
LAYOUT_RE_SHA1 = re.compile(r"^[0-9a-f]{40}$")
LAYOUT_RE_UUID = re.compile(r"^[a-z0-9]{8}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{12}$")
LAYOUT_RE_TIMESTAMP = re.compile(r"^\d{10,13}$")
LAYOUT_RE_NUM = re.compile(r"\d+")
LAYOUT_EXACT_QUERY_VALUE_KEYS = {"id"}
PROPAGATION_VARIANT_MODES = ("synthetic_mapped", "direct_mapped", "direct_raw")


@dataclass(frozen=True)
class PropagationVariant:
    response: str
    html: str
    content: str
    error: str = ""
    sim: float | None = None
    selected_ratio: float | None = None


@dataclass(frozen=True)
class RepresentativeStats:
    selected_ratio: float | None = None


def load_df(path: Path) -> pd.DataFrame:
    parquet_path = path / "dripper_results.parquet"
    jsonl_path = path / "dripper_results.jsonl"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, orient="records", lines=True)
    raise FileNotFoundError(f"No Dripper output rows under {path}")


def digest(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def compact(value: Any, limit: int = 220) -> str:
    return " ".join(str(value or "").split())[:limit]


def token_f1(candidate: Any, reference: Any) -> float:
    candidate_tokens = Counter(TOKEN_RE.findall(str(candidate or "").lower()))
    reference_tokens = Counter(TOKEN_RE.findall(str(reference or "").lower()))
    if not candidate_tokens and not reference_tokens:
        return 1.0
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = sum((candidate_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def select_validation_indexes(
    indexes: list[int],
    count: int,
    df: pd.DataFrame | None = None,
    signature_mode: str = "none",
) -> list[int]:
    if count <= 0 or not indexes:
        return []
    if count >= len(indexes):
        return list(indexes)
    if count == 1:
        return [indexes[-1]]
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(idx: int) -> None:
        if len(selected) >= count or idx in selected_set:
            return
        selected.append(idx)
        selected_set.add(idx)

    if df is not None and signature_mode and signature_mode != "none":
        low_card_query_keys: set[str] = set()
        if "url_low_card_query_shape" in signature_mode:
            low_card_query_keys = low_card_query_value_keys(
                [df.loc[idx, "url"] if "url" in df.columns else None for idx in indexes]
            )
        by_signature: dict[str, list[int]] = defaultdict(list)
        for idx in indexes:
            by_signature[page_signature_key(df, idx, signature_mode, low_card_query_keys)].append(idx)
        signature_groups = sorted(by_signature.values(), key=lambda group: (-len(group), min(group)))
        for group in signature_groups:
            for idx in select_validation_indexes(sorted(group), 1):
                add(idx)
                break
            if len(selected) >= count:
                return sorted(selected)

    positions = sorted({round(position * (len(indexes) - 1) / (count - 1)) for position in range(count)})
    for position in positions:
        add(indexes[position])
        if len(selected) >= count:
            return sorted(selected)
    for idx in indexes:
        add(idx)
        if len(selected) >= count:
            break
    return sorted(selected)


def coerce_html(value: Any) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ""
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def url_host_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def url_shape_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")

    path = parsed.path or ""
    raw_segments = [segment for segment in path.split("/") if segment]
    query_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_path_segment(segment) for segment in raw_segments]
    return f"path={'/'.join(normalized_segments)}|q={query_keys}"


def url_low_card_query_shape_key(value: Any, low_card_query_keys: set[str]) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")

    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [_normalize_path_segment(segment) for segment in raw_segments]

    include_all_query_values = bool(parsed.query) and not low_card_query_keys
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.strip().lower()
        if not lowered_key:
            continue
        if include_all_query_values or lowered_key in low_card_query_keys or lowered_key in LAYOUT_EXACT_QUERY_VALUE_KEYS:
            query_parts.append(f"{lowered_key}={query_value.strip().lower()}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _normalize_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        stem, suffix = segment.rsplit(".", 1)
        segment = stem
        suffix = f".{suffix}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


SEMANTIC_QUERY_VALUE_KEYS = {"hl", "lang", "language", "locale"}


def url_semantic_shape_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.hostname and "://" not in text:
        parsed = urlparse(f"//{text}")

    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    normalized_segments = [_normalize_semantic_path_segment(segment) for segment in raw_segments]
    query_parts = []
    for key, query_value in sorted(parse_qsl(parsed.query, keep_blank_values=True)):
        lowered_key = key.lower()
        if lowered_key in SEMANTIC_QUERY_VALUE_KEYS:
            query_parts.append(f"{lowered_key}={_normalize_semantic_query_value(query_value)}")
        else:
            query_parts.append(lowered_key)
    return f"path={'/'.join(normalized_segments)}|q={','.join(query_parts)}"


def _normalize_semantic_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        stem, extension = segment.rsplit(".", 1)
        segment = stem
        suffix = f".{extension}"
    if (
        segment.isdigit()
        or LAYOUT_RE_MD5.fullmatch(segment)
        or LAYOUT_RE_SHA1.fullmatch(segment)
        or LAYOUT_RE_UUID.fullmatch(segment)
        or LAYOUT_RE_TIMESTAMP.fullmatch(segment)
    ):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def _normalize_semantic_query_value(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if (
        text.isdigit()
        or LAYOUT_RE_MD5.fullmatch(text)
        or LAYOUT_RE_SHA1.fullmatch(text)
        or LAYOUT_RE_UUID.fullmatch(text)
        or LAYOUT_RE_TIMESTAMP.fullmatch(text)
    ):
        return "#num"
    return text


def low_card_query_value_keys(url_values: list[Any], max_distinct: int = 16) -> set[str]:
    values_by_key: dict[str, set[str]] = defaultdict(set)
    for value in url_values:
        text = "" if value is None else str(value)
        if not text:
            continue
        parsed = urlparse(text)
        if not parsed.hostname and "://" not in text:
            parsed = urlparse(f"//{text}")
        for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
            lowered_key = key.strip().lower()
            if lowered_key:
                values_by_key[lowered_key].add(query_value.strip().lower())
    return {key for key, values in values_by_key.items() if 1 < len(values) <= max_distinct}


def item_count_bucket(value: Any) -> str:
    try:
        count = int(float(value))
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return "0"
    if count <= 8:
        return str(count)
    if count <= 16:
        return "9-16"
    if count <= 32:
        return "17-32"
    if count <= 64:
        return "33-64"
    if count <= 128:
        return "65-128"
    return "129+"


def page_signature_key(
    df: pd.DataFrame,
    idx: int,
    mode: str,
    low_card_query_keys: set[str] | None = None,
) -> str:
    if not mode or mode == "none":
        return ""
    parts: list[str] = []
    if "url_low_card_query_shape" in mode:
        parts.append(
            "url="
            + url_low_card_query_shape_key(
                df.loc[idx, "url"] if "url" in df.columns else None,
                low_card_query_keys or set(),
            )
        )
    elif "url_semantic_shape" in mode:
        parts.append(f"url={url_semantic_shape_key(df.loc[idx, 'url'] if 'url' in df.columns else None)}")
    elif "url_shape" in mode:
        parts.append(f"url={url_shape_key(df.loc[idx, 'url'] if 'url' in df.columns else None)}")
    if "item_count_exact" in mode:
        parts.append(f"items={_coerce_item_count(df, idx)}")
    elif "item_count_bucket" in mode:
        parts.append(f"items={item_count_bucket(_coerce_item_count(df, idx))}")
    return "|".join(parts)


def split_indexes_by_page_signature(
    df: pd.DataFrame,
    indexes: list[int],
    mode: str,
    min_cluster_size: int,
) -> list[list[int]]:
    if not mode or mode == "none" or len(indexes) < min_cluster_size:
        return []
    low_card_query_keys: set[str] = set()
    if "url_low_card_query_shape" in mode:
        low_card_query_keys = low_card_query_value_keys(
            [df.loc[idx, "url"] if "url" in df.columns else None for idx in indexes]
        )
    by_signature: dict[str, list[int]] = defaultdict(list)
    for idx in indexes:
        by_signature[page_signature_key(df, idx, mode, low_card_query_keys)].append(idx)
    groups = [
        sorted(signature_indexes)
        for _signature, signature_indexes in sorted(by_signature.items(), key=lambda item: (min(item[1]), item[0]))
        if len(signature_indexes) >= min_cluster_size
    ]
    parent_set = set(indexes)
    return [group for group in groups if set(group) != parent_set]


def layout_feature_fingerprint(feature: Any) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): normalize(inner) for key, inner in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [normalize(inner) for inner in value]
        if isinstance(value, set):
            return sorted(normalize(inner) for inner in value)
        return value

    try:
        return json.dumps(normalize(feature), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return repr(feature)


def layout_dom_path_fingerprint(html_text: str) -> str:
    from lxml.html import HTMLParser, fromstring

    try:
        parser = HTMLParser(collect_ids=False, encoding="utf-8", remove_comments=True, remove_pis=True)
        root = fromstring(html_text.encode("utf-8", errors="ignore"), parser=parser)
        body_nodes = root.xpath("//body")
        root = body_nodes[0] if body_nodes else root
    except Exception:  # noqa: BLE001
        return ""

    def normalize_dynamic_attribute(value: str) -> str:
        lowered = value.strip().lower()
        if LAYOUT_RE_MD5.fullmatch(lowered):
            return "[MD5]"
        if LAYOUT_RE_SHA1.fullmatch(lowered):
            return "[SHA1]"
        if LAYOUT_RE_UUID.fullmatch(lowered):
            return "[UUID]"
        if LAYOUT_RE_TIMESTAMP.fullmatch(lowered):
            return "[TIMESTAMP]"
        return LAYOUT_RE_NUM.sub("", lowered)

    def normalize_attr_tokens(value: str | None) -> str:
        if not value:
            return ""
        tokens = value.split()
        if len(tokens) > 1:
            normalized = [token.lower() for token in tokens if not LAYOUT_RE_NUM.search(token)]
        else:
            normalized = [normalize_dynamic_attribute(tokens[0])] if tokens else []
        return " ".join(token for token in normalized if token)

    def walk(element: Any) -> Any:
        raw_tag = getattr(element, "tag", None)
        if not isinstance(raw_tag, str):
            return None
        tag = raw_tag.lower()
        if tag in LAYOUT_TAGS_TO_IGNORE:
            return None
        attrs: list[tuple[str, str]] = []
        if tag not in LAYOUT_TAGS_IGNORE_ATTR:
            class_attr = normalize_attr_tokens(element.get("class"))
            id_attr = normalize_attr_tokens(element.get("id"))
            if class_attr:
                attrs.append(("class", class_attr))
            if id_attr:
                attrs.append(("id", id_attr))
        children = [child for child in (walk(child) for child in element) if child is not None]
        return [tag, attrs, children]

    return json.dumps(walk(root), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _coerce_item_count(df: pd.DataFrame, idx: int) -> int:
    if "dripper_item_count" not in df.columns:
        return 0
    try:
        return int(float(df.loc[idx, "dripper_item_count"]))
    except (TypeError, ValueError):
        return 0


def item_ids_in_html(html: str) -> list[str]:
    seen: set[str] = set()
    item_ids: list[str] = []
    for item_id in ITEM_ID_RE.findall(html):
        if item_id in seen:
            continue
        seen.add(item_id)
        item_ids.append(item_id)
    return item_ids


def item_id_response(all_item_ids: list[str], main_item_ids: set[str]) -> str:
    labels = {item_id: ("main" if item_id in main_item_ids else "other") for item_id in all_item_ids}
    if all(item_id.isdigit() for item_id in all_item_ids):
        return "".join(f"{item_id}{label}" for item_id, label in labels.items())
    return json.dumps(labels, ensure_ascii=False, separators=(",", ":"))


def labels_to_webkit_response(labels: Any) -> dict[str, int]:
    if not isinstance(labels, dict):
        return {}
    return {
        f"item_id {item_id}": 1 if str(label).strip().lower() in {"main", "1", "true"} else 0
        for item_id, label in labels.items()
    }


def build_case(
    raw_html: str,
    *,
    simplified_html: str = "",
    mapped_html: str = "",
    response: str = "",
) -> MinerUHTMLCase:
    case = MinerUHTMLCase(MinerUHTMLInput(raw_html=raw_html))
    if simplified_html or mapped_html:
        case.process_data = MinerUHTMLProcessData(simpled_html=simplified_html, map_html=mapped_html)
    if response:
        case.generate_output = MinerUHTMLGenerateOutput(response=response)
    return case


def simplify(raw_html: str) -> tuple[str, str]:
    case = simplify_single_input(build_case(raw_html))
    if case.process_data is None:
        return "", ""
    return case.process_data.simpled_html, case.process_data.map_html


def postprocess_response(raw_html: str, mapped_html: str, response: str) -> PropagationVariant:
    response_case = build_case(raw_html, mapped_html=mapped_html, response=response)
    response_case = parse_result(response_case)
    main_html = extract_main_html(mapped_html, response_case.parse_result.item_label)
    output_case = build_case(raw_html)
    output_case.output_data = MinerUHTMLOutput(main_html=main_html)
    output_case = convert2content(output_case, output_format="mm_md")
    return PropagationVariant(
        response=response,
        html=output_case.output_data.main_html,
        content=output_case.output_data.main_content or "",
    )


def convert_direct(raw_html: str, main_html: str) -> PropagationVariant:
    case = build_case(raw_html)
    case.output_data = MinerUHTMLOutput(main_html=main_html)
    case = convert2content(case, output_format="mm_md")
    return PropagationVariant(response="", html=case.output_data.main_html, content=case.output_data.main_content or "")


def build_mapping(rep_raw_html: str, rep_mapped_html: str, rep_response: str) -> dict[str, Any]:
    rep_case = build_case(rep_raw_html, mapped_html=rep_mapped_html, response=rep_response)
    rep_case = parse_result(rep_case)
    return MapItemToHtmlTagsParser({}).parse(
        {
            "typical_raw_tag_html": rep_mapped_html,
            "typical_raw_html": rep_raw_html,
            "llm_response": labels_to_webkit_response(rep_case.parse_result.item_label),
        }
    )


def representative_stats(rep_mapped_html: str, rep_response: str) -> RepresentativeStats:
    try:
        rep_case = build_case("", mapped_html=rep_mapped_html, response=rep_response)
        rep_case = parse_result(rep_case)
        labels = getattr(rep_case.parse_result, "item_label", {})
        all_item_ids = item_ids_in_html(rep_mapped_html)
        main_item_ids = {
            str(item_id)
            for item_id, label in labels.items()
            if str(label).strip().lower() in {"main", "1", "true"}
        }
        selected_ratio = len(main_item_ids) / len(all_item_ids) if all_item_ids else None
    except Exception:
        selected_ratio = None
    return RepresentativeStats(selected_ratio=selected_ratio)


def propagate(
    mapping_data: dict[str, Any],
    target_raw_html: str,
    target_mapped_html: str,
    *,
    more_noise_enable: bool,
    dynamic_classid_similarity_threshold: float,
    variant_modes: tuple[str, ...] = PROPAGATION_VARIANT_MODES,
    variant_timing_s: Counter[str] | None = None,
) -> dict[str, PropagationVariant]:
    variants: dict[str, PropagationVariant] = {}
    html_sources = {
        "synthetic_mapped": target_mapped_html,
        "direct_mapped": target_mapped_html,
        "direct_raw": target_raw_html,
    }
    for mode in variant_modes:
        html_source = html_sources[mode]
        started = time.perf_counter()
        try:
            task_data = dict(mapping_data)
            task_data.update(
                {
                    "html_source": html_source,
                    "dynamic_id_enable": True,
                    "dynamic_classid_enable": True,
                    "more_noise_enable": more_noise_enable,
                    "dynamic_classid_similarity_threshold": dynamic_classid_similarity_threshold,
                }
            )
            parts = LayoutBatchParser({}).parse(task_data)
            main_html = str(parts.get("main_html_body") or "")
            sim_value = parts.get("main_html_sim")
            sim = float(sim_value) if isinstance(sim_value, (int, float)) else None
            if mode == "synthetic_mapped":
                all_item_ids = item_ids_in_html(target_mapped_html)
                main_item_ids = set(item_ids_in_html(main_html))
                response = item_id_response(all_item_ids, main_item_ids)
                variant = postprocess_response(target_raw_html, target_mapped_html, response)
                selected_ratio = len(main_item_ids) / len(all_item_ids) if all_item_ids else None
                variants[mode] = PropagationVariant(
                    response=variant.response,
                    html=variant.html,
                    content=variant.content,
                    error=variant.error,
                    sim=sim,
                    selected_ratio=selected_ratio,
                )
            else:
                variant = convert_direct(target_raw_html, main_html)
                variants[mode] = PropagationVariant(
                    response=variant.response,
                    html=variant.html,
                    content=variant.content,
                    error=variant.error,
                    sim=sim,
                )
        except Exception as exc:  # noqa: BLE001
            variants[mode] = PropagationVariant(response="", html="", content="", error=str(exc))
        finally:
            if variant_timing_s is not None:
                variant_timing_s[mode] += time.perf_counter() - started
    return variants


def parse_variant_modes(raw_value: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in raw_value.split(",") if value.strip())
    if not values:
        return PROPAGATION_VARIANT_MODES
    invalid = sorted(set(values) - set(PROPAGATION_VARIANT_MODES))
    if invalid:
        raise SystemExit(
            "LAYOUT_DIAG_VARIANT_MODES contains unsupported value(s): "
            f"{','.join(invalid)}; expected one or more of {','.join(PROPAGATION_VARIANT_MODES)}"
        )
    return values


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def build_domain_clustered_shards(df: pd.DataFrame, shard_size: int) -> list[list[int]]:
    host_values = df["url"].tolist() if "url" in df.columns else [""] * len(df)
    work = pd.DataFrame(
        {
            "row_index": list(range(len(df))),
            "host_key": [url_host_key(value) for value in host_values],
        }
    )
    ordered = work.sort_values(["host_key", "row_index"], kind="stable")
    shards: list[list[int]] = []
    current_shard: list[int] = []
    for _host_key, host_df in ordered.groupby("host_key", sort=False):
        host_indexes = host_df["row_index"].astype(int).tolist()
        for start in range(0, len(host_indexes), shard_size):
            host_chunk = host_indexes[start : start + shard_size]
            if current_shard and len(current_shard) + len(host_chunk) > shard_size:
                shards.append(current_shard)
                current_shard = []
            current_shard.extend(host_chunk)
            if len(current_shard) >= shard_size:
                shards.append(current_shard)
                current_shard = []
    if current_shard:
        shards.append(current_shard)
    return shards


def build_layout_groups_for_shard(
    df: pd.DataFrame,
    shard_indexes: list[int],
    *,
    threshold: float,
    min_cluster_size: int,
    page_signature_mode: str,
    max_exact_host_pages: int,
    large_host_mode: str,
) -> list[list[int]]:
    samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx in shard_indexes:
        if not str(df.loc[idx, "dripper_response"] or "").strip():
            continue
        html_text = coerce_html(df.loc[idx, "html"])
        if not html_text.strip():
            continue
        try:
            feature = get_feature(html_text)
        except Exception:
            continue
        if feature is None:
            continue
        samples_by_host[url_host_key(df.loc[idx, "url"] if "url" in df.columns else None)].append(
            {"track_id": str(idx), "html": html_text, "feature": feature}
        )

    groups: list[list[int]] = []
    for _host_key, samples in samples_by_host.items():
        if len(samples) < min_cluster_size:
            continue
        if max_exact_host_pages > 0 and len(samples) > max_exact_host_pages:
            if large_host_mode not in {"feature_hash", "dom_path_hash"}:
                continue
            by_fingerprint: dict[str, list[int]] = defaultdict(list)
            for sample in samples:
                if large_host_mode == "dom_path_hash":
                    fingerprint = layout_dom_path_fingerprint(coerce_html(sample.get("html")))
                else:
                    fingerprint = layout_feature_fingerprint(sample.get("feature"))
                by_fingerprint[fingerprint].append(int(sample["track_id"]))
            for indexes in by_fingerprint.values():
                by_signature: dict[str, list[int]] = defaultdict(list)
                for row_idx in indexes:
                    by_signature[page_signature_key(df, row_idx, page_signature_mode)].append(row_idx)
                groups.extend(sorted(signature_indexes) for signature_indexes in by_signature.values() if len(signature_indexes) >= min_cluster_size)
            continue
        try:
            clustered_samples, _layout_ids = cluster_html_struct(samples, threshold=threshold)
        except Exception:
            continue
        max_layer_n = int(clustered_samples[0].get("max_layer_n") or 5)
        exemplars_by_layout: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = int(sample.get("layout_id", -1))
            if layout_id < 0:
                continue
            if len(exemplars_by_layout[layout_id]) < 3:
                exemplars_by_layout[layout_id].append(sample)

        by_layout: dict[tuple[int, str], list[int]] = defaultdict(list)
        for sample in clustered_samples:
            layout_id = assign_layout_by_exemplar_similarity(
                sample.get("feature"),
                exemplars_by_layout,
                max_layer_n,
                threshold,
            )
            if layout_id < 0:
                continue
            row_idx = int(sample["track_id"])
            by_layout[(layout_id, page_signature_key(df, row_idx, page_signature_mode))].append(row_idx)
        groups.extend(sorted(indexes) for indexes in by_layout.values() if len(indexes) >= min_cluster_size)
    return groups


def assign_layout_by_exemplar_similarity(
    feature: Any,
    exemplars_by_layout: dict[int, list[dict[str, Any]]],
    max_layer_n: int,
    threshold: float,
) -> int:
    for layout_id, exemplars in exemplars_by_layout.items():
        for exemplar in exemplars:
            try:
                score = similarity(feature, exemplar.get("feature"), max_layer_n)
            except Exception:
                continue
            if score is not None and score >= threshold:
                return layout_id
    return -2


def select_representative_index(df: pd.DataFrame, indexes: list[int]) -> int:
    candidates = [{"track_id": str(idx), "html": coerce_html(df.loc[idx, "html"])} for idx in indexes]
    try:
        representative = select_representative_html(candidates)
    except Exception:
        representative = None
    if representative is None:
        return indexes[0]
    try:
        selected = int(representative["track_id"])
    except (KeyError, TypeError, ValueError):
        return indexes[0]
    return selected if selected in indexes else indexes[0]


def main() -> None:
    base_dir = Path(os.environ["BASE_OUTPUT_DIR"])
    candidate_dir = Path(os.environ["CANDIDATE_OUTPUT_DIR"])
    max_rows = int(os.environ.get("MAX_ROWS", "300"))
    example_rows = int(os.environ.get("EXAMPLE_ROWS", "5"))
    shard_size = int(os.environ.get("SHARD_SIZE", "64"))
    threshold = float(os.environ.get("LAYOUT_CLUSTER_THRESHOLD", "0.95"))
    min_cluster_size = int(os.environ.get("LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE", "2"))
    max_exact_host_pages = int(os.environ.get("LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES", "0"))
    large_host_mode = os.environ.get("LAYOUT_TEMPLATE_LARGE_HOST_MODE", "standalone").strip().lower()
    max_selected_item_ratio_value = float(os.environ.get("LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO", "0.50"))
    max_selected_item_ratio = max_selected_item_ratio_value if max_selected_item_ratio_value > 0 else None
    max_rep_selected_item_ratio_value = float(os.environ.get("LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO", "0"))
    max_rep_selected_item_ratio = (
        max_rep_selected_item_ratio_value if max_rep_selected_item_ratio_value > 0 else None
    )
    more_noise_enable = truthy(os.environ.get("LAYOUT_TEMPLATE_MORE_NOISE_ENABLE", "1"))
    dynamic_classid_similarity_threshold = float(os.environ.get("DYNAMIC_CLASSID_SIMILARITY_THRESHOLD", "0.85"))
    min_consensus_f1_value = float(os.environ.get("LAYOUT_TEMPLATE_MIN_CONSENSUS_F1", "0"))
    min_consensus_f1 = min_consensus_f1_value if min_consensus_f1_value > 0 else None
    validation_rows = int(os.environ.get("LAYOUT_TEMPLATE_VALIDATION_ROWS", "0"))
    validation_min_f1 = float(os.environ.get("LAYOUT_TEMPLATE_VALIDATION_MIN_F1", "0.98"))
    validation_signature_mode = os.environ.get("LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE", "none").strip().lower()
    large_cluster_validation_rows = int(os.environ.get("LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS", "0"))
    large_cluster_min_size = int(os.environ.get("LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE", "0"))
    min_content_length_ratio_value = float(os.environ.get("LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO", "0"))
    min_content_length_ratio = min_content_length_ratio_value if min_content_length_ratio_value > 0 else None
    max_content_length_ratio_value = float(os.environ.get("LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO", "0"))
    max_content_length_ratio = max_content_length_ratio_value if max_content_length_ratio_value > 0 else None
    page_signature_mode = os.environ.get("LAYOUT_PAGE_SIGNATURE_MODE", "none").strip().lower()
    failed_layout_fallback_signature_mode = os.environ.get(
        "LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE",
        "none",
    ).strip().lower()
    propagation_target = os.environ.get("LAYOUT_TEMPLATE_PROPAGATION_TARGET", "raw_html").strip().lower()
    validation_mode = "synthetic_mapped" if propagation_target == "mapped_item_ids" else "direct_raw"
    variant_modes = parse_variant_modes(os.environ.get("LAYOUT_DIAG_VARIANT_MODES", ""))
    target_hosts = {
        host.strip().lower()
        for host in os.environ.get("LAYOUT_TARGET_HOSTS", "").split(",")
        if host.strip()
    }
    force_host_single_cluster = truthy(os.environ.get("LAYOUT_FORCE_HOST_SINGLE_CLUSTER", "0"))

    base_df = load_df(base_dir).reset_index(drop=True)
    candidate_df = load_df(candidate_dir).reset_index(drop=True)
    if len(base_df) != len(candidate_df):
        raise SystemExit(f"row count mismatch: base={len(base_df)} candidate={len(candidate_df)}")

    missing_base = sorted({"html", "dripper_response", "dripper_html", "dripper_content"} - set(base_df.columns))
    if missing_base:
        raise SystemExit(f"baseline missing columns: {missing_base}")

    if target_hosts:
        host_indexes: dict[str, list[int]] = defaultdict(list)
        for idx, row in base_df.iterrows():
            host_key = url_host_key(row.get("url") if "url" in base_df.columns else None)
            if host_key in target_hosts:
                host_indexes[host_key].append(int(idx))
        missing_hosts = sorted(target_hosts - set(host_indexes))
        if missing_hosts:
            raise SystemExit(f"target host(s) not found in output rows: {missing_hosts}")
        shards = [indexes for _host, indexes in sorted(host_indexes.items())]
    else:
        shards = build_domain_clustered_shards(base_df, shard_size)

    print("LAYOUT_PROPAGATION_DIAG_BEGIN")
    print(f"base_dir={base_dir}")
    print(f"candidate_dir={candidate_dir}")
    print(f"rows={len(base_df)}")
    print(f"rebuilt_shards={len(shards)}")
    print(f"shard_size={shard_size}")
    print(f"layout_cluster_threshold={threshold}")
    print(f"layout_template_min_cluster_size={min_cluster_size}")
    print(f"layout_template_max_exact_host_pages={max_exact_host_pages}")
    print(f"layout_template_large_host_mode={large_host_mode}")
    print(f"layout_template_max_selected_item_ratio={max_selected_item_ratio_value}")
    print(f"layout_template_max_rep_selected_item_ratio={max_rep_selected_item_ratio_value}")
    print(f"layout_template_more_noise_enable={int(more_noise_enable)}")
    print(f"dynamic_classid_similarity_threshold={dynamic_classid_similarity_threshold}")
    print(f"layout_template_min_consensus_f1={min_consensus_f1_value}")
    print(f"layout_template_validation_rows={validation_rows}")
    print(f"layout_template_validation_min_f1={validation_min_f1}")
    print(f"layout_template_validation_signature_mode={validation_signature_mode}")
    print(f"layout_template_large_cluster_validation_rows={large_cluster_validation_rows}")
    print(f"layout_template_large_cluster_min_size={large_cluster_min_size}")
    print(f"layout_template_min_content_length_ratio={min_content_length_ratio_value}")
    print(f"layout_template_max_content_length_ratio={max_content_length_ratio_value}")
    print(f"layout_template_propagation_target={propagation_target}")
    print(f"layout_template_validation_mode={validation_mode}")
    print(f"layout_diag_variant_modes={','.join(variant_modes)}")
    print(f"layout_page_signature_mode={page_signature_mode}")
    print(f"layout_template_failed_layout_fallback_signature_mode={failed_layout_fallback_signature_mode}")
    print(f"layout_target_hosts={','.join(sorted(target_hosts))}")
    print(f"layout_force_host_single_cluster={int(force_host_single_cluster)}")

    simplified_cache: dict[int, tuple[str, str]] = {}
    mapping_cache: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    f1_sums: Counter[str] = Counter()
    f1_counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    variant_timing_s: Counter[str] = Counter()
    cluster_trace_rows: list[dict[str, Any]] = []
    propagation_trace_rows: list[dict[str, Any]] = []
    examples: list[str] = []
    failed_cluster_examples: list[str] = []
    passed_cluster_examples: list[str] = []

    def get_simplified(idx: int) -> tuple[str, str]:
        if idx not in simplified_cache:
            simplified_cache[idx] = simplify(coerce_html(base_df.loc[idx, "html"]))
        return simplified_cache[idx]

    def content_length_ratio(
        variant: PropagationVariant | None,
        mapping: dict[str, Any],
    ) -> float | None:
        if variant is None or variant.error:
            return None
        rep_len = mapping.get("_diagnostic_rep_content_len")
        if not isinstance(rep_len, (int, float)) or rep_len <= 0:
            return None
        return len(str(variant.content or "")) / rep_len

    def content_length_ratio_reject(
        variant: PropagationVariant | None,
        mapping: dict[str, Any],
    ) -> tuple[bool, float | None, str]:
        ratio = content_length_ratio(variant, mapping)
        if ratio is None:
            return False, ratio, ""
        if min_content_length_ratio is not None and ratio < min_content_length_ratio:
            return True, ratio, f"content_length_ratio={ratio:.3f}<min={min_content_length_ratio:.3f}"
        if max_content_length_ratio is not None and ratio > max_content_length_ratio:
            return True, ratio, f"content_length_ratio={ratio:.3f}>max={max_content_length_ratio:.3f}"
        return False, ratio, ""

    def parent_layout_validation_fails(cluster_id: str, indexes: list[int]) -> bool:
        rep_idx = select_representative_index(base_df, indexes)
        sibling_indexes = [idx for idx in indexes if idx != rep_idx]
        if not sibling_indexes:
            return False

        effective_validation_rows = validation_rows
        if (
            large_cluster_validation_rows > 0
            and large_cluster_min_size > 0
            and len(indexes) >= large_cluster_min_size
        ):
            effective_validation_rows = max(effective_validation_rows, large_cluster_validation_rows)
        validation_indexes = select_validation_indexes(
            sibling_indexes,
            effective_validation_rows,
            base_df,
            validation_signature_mode,
        )
        if not validation_indexes:
            return False

        counts["failed_layout_parent_representative_llm"] += 1
        counts["failed_layout_parent_validation_llm"] += len(validation_indexes)
        try:
            _, rep_mapped_html = get_simplified(rep_idx)
            rep_stats = representative_stats(
                rep_mapped_html,
                str(base_df.loc[rep_idx, "dripper_response"] or ""),
            )
            mapping = build_mapping(
                coerce_html(base_df.loc[rep_idx, "html"]),
                rep_mapped_html,
                str(base_df.loc[rep_idx, "dripper_response"] or ""),
            )
            mapping["_diagnostic_rep_selected_ratio"] = rep_stats.selected_ratio
            mapping["_diagnostic_rep_content_len"] = len(str(base_df.loc[rep_idx, "dripper_content"] or ""))
            mapping_cache[cluster_id] = mapping
        except Exception as exc:  # noqa: BLE001
            counts["failed_layout_parent_setup_error"] += 1
            errors[f"failed_layout_parent: {str(exc)[:140]}"] += 1
            return True

        for idx in validation_indexes:
            try:
                _, target_mapped_html = get_simplified(idx)
                variants = propagate(
                    mapping,
                    coerce_html(base_df.loc[idx, "html"]),
                    target_mapped_html,
                    more_noise_enable=more_noise_enable,
                    dynamic_classid_similarity_threshold=dynamic_classid_similarity_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                counts["failed_layout_parent_setup_error"] += 1
                errors[f"failed_layout_parent: {str(exc)[:140]}"] += 1
                return True

            validation_variant = variants.get(validation_mode)
            validation_f1 = (
                token_f1(validation_variant.content, str(base_df.loc[idx, "dripper_content"] or ""))
                if validation_variant is not None and not validation_variant.error
                else None
            )
            if validation_f1 is None or validation_f1 < validation_min_f1:
                counts["failed_layout_parent_failed_validation_samples"] += 1
                return True
            ratio_reject, _ratio, _ratio_reason = content_length_ratio_reject(validation_variant, mapping)
            if ratio_reject:
                counts["failed_layout_parent_failed_length_ratio_samples"] += 1
                return True
        return False

    processed_rows = 0
    processed_groups = 0
    representative_rows = 0
    for shard_index, shard_indexes in enumerate(shards):
        if max_rows > 0 and processed_rows >= max_rows:
            break
        if target_hosts and force_host_single_cluster:
            raw_groups = [sorted(shard_indexes)] if len(shard_indexes) >= min_cluster_size else []
        else:
            raw_groups = build_layout_groups_for_shard(
                base_df,
                shard_indexes,
                threshold=threshold,
                min_cluster_size=min_cluster_size,
                page_signature_mode=page_signature_mode,
                max_exact_host_pages=max_exact_host_pages,
                large_host_mode=large_host_mode,
            )

        groups: list[tuple[str, list[int]]] = []
        for raw_group_index, indexes in enumerate(raw_groups):
            parent_cluster_id = f"shard-{shard_index:06d}/layout-{raw_group_index:06d}"
            child_groups = split_indexes_by_page_signature(
                base_df,
                indexes,
                failed_layout_fallback_signature_mode,
                min_cluster_size,
            )
            if child_groups and parent_layout_validation_fails(parent_cluster_id, indexes):
                counts["failed_layout_parent_groups"] += 1
                counts["failed_layout_child_groups"] += len(child_groups)
                grouped_child_indexes = {idx for child_group in child_groups for idx in child_group}
                counts["failed_layout_child_group_rows"] += len(grouped_child_indexes)
                counts["failed_layout_uncovered_parent_rows"] += len(set(indexes) - grouped_child_indexes)
                cluster_trace_rows.append(
                    {
                        "cluster_id": parent_cluster_id,
                        "shard_index": shard_index,
                        "group_index": raw_group_index,
                        "rows": len(indexes),
                        "representative_row": select_representative_index(base_df, indexes),
                        "representative_url": base_df.loc[indexes[0], "url"] if "url" in base_df.columns else "",
                        "hosts": json.dumps(
                            dict(
                                Counter(
                                    url_host_key(base_df.loc[idx, "url"] if "url" in base_df.columns else None)
                                    for idx in indexes
                                )
                            ),
                            sort_keys=True,
                        ),
                        "status": "failed_parent_split",
                    }
                )
                for child_index, child_indexes in enumerate(child_groups):
                    groups.append((f"{parent_cluster_id}/child-{child_index:06d}", child_indexes))
                continue
            groups.append((parent_cluster_id, indexes))

        for group_index, (cluster_id, indexes) in enumerate(groups):
            if max_rows > 0 and processed_rows >= max_rows:
                break
            processed_groups += 1
            rep_idx = select_representative_index(base_df, indexes)
            representative_rows += 1
            group_rows = len(indexes)
            cluster_hosts = Counter(
                url_host_key(base_df.loc[idx, "url"] if "url" in base_df.columns else None)
                for idx in indexes
            )
            cluster_trace_rows.append(
                {
                    "cluster_id": cluster_id,
                    "shard_index": shard_index,
                    "group_index": group_index,
                    "rows": group_rows,
                    "representative_row": rep_idx,
                    "representative_url": base_df.loc[rep_idx, "url"] if "url" in base_df.columns else "",
                    "hosts": json.dumps(dict(cluster_hosts), sort_keys=True),
                    "status": "active",
                }
            )
            for size_threshold in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                if group_rows >= size_threshold:
                    counts[f"layout_group_size_ge_{size_threshold}"] += 1
            sibling_indexes = [idx for idx in indexes if idx != rep_idx]
            if not sibling_indexes:
                continue
            try:
                _, rep_mapped_html = get_simplified(rep_idx)
                mapping = mapping_cache.get(cluster_id)
                if mapping is None:
                    rep_stats = representative_stats(
                        rep_mapped_html,
                        str(base_df.loc[rep_idx, "dripper_response"] or ""),
                    )
                    mapping = build_mapping(
                        coerce_html(base_df.loc[rep_idx, "html"]),
                        rep_mapped_html,
                        str(base_df.loc[rep_idx, "dripper_response"] or ""),
                    )
                    mapping["_diagnostic_rep_selected_ratio"] = rep_stats.selected_ratio
                    mapping["_diagnostic_rep_content_len"] = len(str(base_df.loc[rep_idx, "dripper_content"] or ""))
                    mapping_cache[cluster_id] = mapping
            except Exception as exc:  # noqa: BLE001
                counts["setup_error"] += len(sibling_indexes)
                errors[str(exc)[:160]] += 1
                continue

            effective_validation_rows = validation_rows
            if (
                large_cluster_validation_rows > 0
                and large_cluster_min_size > 0
                and group_rows >= large_cluster_min_size
            ):
                effective_validation_rows = max(effective_validation_rows, large_cluster_validation_rows)
            validation_indexes = select_validation_indexes(
                sibling_indexes,
                effective_validation_rows,
                base_df,
                validation_signature_mode,
            )
            validation_index_set = set(validation_indexes)
            diagnostic_indexes = validation_indexes + [idx for idx in sibling_indexes if idx not in validation_index_set]
            group_validation_failed = False
            group_validation_failure_counted = False
            validation_records: list[str] = []
            for idx in diagnostic_indexes:
                if max_rows > 0 and processed_rows >= max_rows:
                    break
                processed_rows += 1
                if processed_rows == 1 or processed_rows % 100 == 0:
                    print(
                        "PROGRESS "
                        f"processed_rows={processed_rows} "
                        f"shard_index={shard_index} "
                        f"group_index={group_index} "
                        f"group_rows={len(indexes)}",
                        flush=True,
                    )
                try:
                    _, target_mapped_html = get_simplified(idx)
                    variants = propagate(
                        mapping,
                        coerce_html(base_df.loc[idx, "html"]),
                        target_mapped_html,
                        more_noise_enable=more_noise_enable,
                        dynamic_classid_similarity_threshold=dynamic_classid_similarity_threshold,
                        variant_modes=variant_modes,
                        variant_timing_s=variant_timing_s,
                    )
                except Exception as exc:  # noqa: BLE001
                    counts["setup_error"] += 1
                    errors[str(exc)[:160]] += 1
                    continue

                base_content_hash = digest(base_df.loc[idx, "dripper_content"])
                base_html_hash = digest(base_df.loc[idx, "dripper_html"])
                base_content = str(base_df.loc[idx, "dripper_content"] or "")
                candidate_content_hash = digest(candidate_df.loc[idx, "dripper_content"])
                synthetic_variant = variants.get("synthetic_mapped")
                direct_raw_variant = variants.get("direct_raw")
                synthetic_direct_raw_f1: float | None = None
                rep_selected_ratio = mapping.get("_diagnostic_rep_selected_ratio")
                if not isinstance(rep_selected_ratio, (int, float)):
                    rep_selected_ratio = None
                if (
                    synthetic_variant is not None
                    and direct_raw_variant is not None
                    and not synthetic_variant.error
                    and not direct_raw_variant.error
                ):
                    synthetic_direct_raw_f1 = token_f1(synthetic_variant.content, direct_raw_variant.content)
                synthetic_f1 = (
                    token_f1(synthetic_variant.content, base_content)
                    if synthetic_variant is not None and not synthetic_variant.error
                    else None
                )
                direct_raw_f1 = (
                    token_f1(direct_raw_variant.content, base_content)
                    if direct_raw_variant is not None and not direct_raw_variant.error
                    else None
                )
                validation_variant = variants.get(validation_mode)
                validation_length_reject, validation_length_ratio, validation_length_reason = (
                    content_length_ratio_reject(validation_variant, mapping)
                )
                propagation_trace_rows.append(
                    {
                        "row_index": idx,
                        "cluster_id": cluster_id,
                        "representative_row": rep_idx,
                        "url": base_df.loc[idx, "url"] if "url" in base_df.columns else "",
                        "base_content_hash": base_content_hash,
                        "base_html_hash": base_html_hash,
                        "candidate_content_hash": candidate_content_hash,
                        "candidate_content_match": candidate_content_hash == base_content_hash,
                        "synthetic_mapped_f1": synthetic_f1,
                        "synthetic_mapped_content_match": (
                            synthetic_variant is not None
                            and digest(synthetic_variant.content) == base_content_hash
                        ),
                        "synthetic_mapped_error": synthetic_variant.error if synthetic_variant is not None else "",
                        "synthetic_mapped_sim": synthetic_variant.sim if synthetic_variant is not None else None,
                        "synthetic_mapped_selected_ratio": (
                            synthetic_variant.selected_ratio if synthetic_variant is not None else None
                        ),
                        "direct_raw_f1": direct_raw_f1,
                        "direct_raw_content_match": (
                            direct_raw_variant is not None
                            and digest(direct_raw_variant.content) == base_content_hash
                        ),
                        "direct_raw_error": direct_raw_variant.error if direct_raw_variant is not None else "",
                        "direct_raw_sim": direct_raw_variant.sim if direct_raw_variant is not None else None,
                        "direct_raw_content_length_ratio": content_length_ratio(direct_raw_variant, mapping),
                        "synthetic_direct_raw_f1": synthetic_direct_raw_f1,
                        "rep_selected_ratio": rep_selected_ratio,
                        "validation_sample": idx in validation_index_set,
                        "validation_content_length_ratio": validation_length_ratio,
                        "validation_content_length_reject": validation_length_reject,
                    }
                )
                validation_f1 = (
                    token_f1(validation_variant.content, base_content)
                    if validation_variant is not None and not validation_variant.error
                    else None
                )
                validation_sample = False
                if validation_rows > 0 and validation_variant is not None:
                    validation_sample = idx in validation_index_set
                    if validation_sample:
                        counts[f"{validation_mode}_validation_llm"] += 1
                        validation_records.append(
                            "idx="
                            f"{idx}"
                            f":f1={validation_f1 if validation_f1 is not None else -1:.3f}"
                            f":length_ratio={validation_length_ratio if validation_length_ratio is not None else -1:.3f}"
                            f":selected_ratio={getattr(validation_variant, 'selected_ratio', None)}"
                            f":error={compact(validation_variant.error, 80)!r}"
                            f":url={compact(base_df.loc[idx, 'url'] if 'url' in base_df.columns else '', 120)!r}"
                        )
                        if validation_f1 is None or validation_f1 < validation_min_f1 or validation_length_reject:
                            group_validation_failed = True
                            if not group_validation_failure_counted:
                                counts[f"{validation_mode}_validation_failed_clusters"] += 1
                                group_validation_failure_counted = True
                            if validation_length_reject:
                                counts[f"{validation_mode}_validation_length_ratio_reject"] += 1
                for mode, variant in variants.items():
                    if mode == "synthetic_mapped" and synthetic_direct_raw_f1 is not None:
                        for consensus_threshold in (0.80, 0.90, 0.95, 0.98):
                            if synthetic_direct_raw_f1 >= consensus_threshold:
                                suffix = str(consensus_threshold).replace(".", "_")
                                counts[f"{mode}_direct_raw_consensus_ge_{suffix}"] += 1
                                if token_f1(variant.content, base_content) >= 0.95:
                                    counts[f"{mode}_direct_raw_consensus_ge_{suffix}_f1_ge_0.95"] += 1
                    if mode == "synthetic_mapped" and rep_selected_ratio is not None:
                        for rep_ratio_threshold in (0.25, 0.35, 0.50, 0.65):
                            if rep_selected_ratio <= rep_ratio_threshold:
                                suffix = str(rep_ratio_threshold).replace(".", "_")
                                counts[f"{mode}_rep_selected_ratio_le_{suffix}"] += 1
                                if token_f1(variant.content, base_content) >= 0.95:
                                    counts[f"{mode}_rep_selected_ratio_le_{suffix}_f1_ge_0.95"] += 1

                    if (
                        mode == "synthetic_mapped"
                        and max_selected_item_ratio is not None
                        and (
                            variant.error
                            or variant.selected_ratio is None
                            or variant.selected_ratio > max_selected_item_ratio
                            or (
                                max_rep_selected_item_ratio is not None
                                and (
                                    rep_selected_ratio is None
                                    or rep_selected_ratio > max_rep_selected_item_ratio
                                )
                            )
                            or (
                                min_consensus_f1 is not None
                                and (
                                    synthetic_direct_raw_f1 is None
                                    or synthetic_direct_raw_f1 < min_consensus_f1
                                )
                            )
                        )
                    ):
                        counts[f"{mode}_cap_fallback_llm"] += 1
                        counts[f"{mode}_cap_effective_content_match"] += 1
                        counts[f"{mode}_cap_effective_html_match"] += 1
                        counts[f"{mode}_cap_effective_f1_ge_0.95"] += 1
                        counts[f"{mode}_cap_effective_f1_ge_0.90"] += 1
                        counts[f"{mode}_cap_effective_f1_ge_0.80"] += 1
                    elif mode == "synthetic_mapped" and max_selected_item_ratio is not None:
                        cap_f1 = token_f1(variant.content, base_content)
                        counts[f"{mode}_cap_saved"] += 1
                        if cap_f1 >= 0.95:
                            counts[f"{mode}_cap_effective_f1_ge_0.95"] += 1
                        if cap_f1 >= 0.90:
                            counts[f"{mode}_cap_effective_f1_ge_0.90"] += 1
                        if cap_f1 >= 0.80:
                            counts[f"{mode}_cap_effective_f1_ge_0.80"] += 1
                        if digest(variant.content) == base_content_hash:
                            counts[f"{mode}_cap_effective_content_match"] += 1
                        if digest(variant.html) == base_html_hash:
                            counts[f"{mode}_cap_effective_html_match"] += 1

                    if mode == validation_mode and validation_rows > 0:
                        if validation_length_reject:
                            counts[f"{mode}_content_length_ratio_reject"] += 1
                        selected_ratio_reject = (
                            mode == "synthetic_mapped"
                            and max_selected_item_ratio is not None
                            and (
                                variant.selected_ratio is None
                                or variant.selected_ratio > max_selected_item_ratio
                            )
                        )
                        rep_selected_ratio_reject = (
                            mode == "synthetic_mapped"
                            and max_rep_selected_item_ratio is not None
                            and (
                                rep_selected_ratio is None
                                or rep_selected_ratio > max_rep_selected_item_ratio
                            )
                        )
                        validation_reject = (
                            validation_sample
                            or group_validation_failed
                            or variant.error
                            or (mode == validation_mode and validation_length_reject)
                            or selected_ratio_reject
                            or rep_selected_ratio_reject
                            or (
                                min_consensus_f1 is not None
                                and (
                                    synthetic_direct_raw_f1 is None
                                    or synthetic_direct_raw_f1 < min_consensus_f1
                                )
                            )
                        )
                        if validation_reject:
                            counts[f"{mode}_validated_fallback_llm"] += 1
                            counts[f"{mode}_validated_effective_content_match"] += 1
                            counts[f"{mode}_validated_effective_html_match"] += 1
                            counts[f"{mode}_validated_effective_f1_ge_0.95"] += 1
                            counts[f"{mode}_validated_effective_f1_ge_0.90"] += 1
                            counts[f"{mode}_validated_effective_f1_ge_0.80"] += 1
                        else:
                            counts[f"{mode}_validated_saved"] += 1
                            validated_f1 = token_f1(variant.content, base_content)
                            if validated_f1 >= 0.95:
                                counts[f"{mode}_validated_effective_f1_ge_0.95"] += 1
                            if validated_f1 >= 0.90:
                                counts[f"{mode}_validated_effective_f1_ge_0.90"] += 1
                            if validated_f1 >= 0.80:
                                counts[f"{mode}_validated_effective_f1_ge_0.80"] += 1
                            if digest(variant.content) == base_content_hash:
                                counts[f"{mode}_validated_effective_content_match"] += 1
                            if digest(variant.html) == base_html_hash:
                                counts[f"{mode}_validated_effective_html_match"] += 1

                    if variant.error:
                        counts[f"{mode}_error"] += 1
                        errors[f"{mode}: {variant.error[:140]}"] += 1
                        continue
                    f1 = token_f1(variant.content, base_content)
                    f1_sums[mode] += f1
                    f1_counts[mode] += 1
                    if variant.sim is not None:
                        for sim_threshold in (0.80, 0.85, 0.90, 0.95):
                            if variant.sim >= sim_threshold:
                                suffix = str(sim_threshold).replace(".", "_")
                                counts[f"{mode}_sim_ge_{suffix}"] += 1
                                if f1 >= 0.95:
                                    counts[f"{mode}_sim_ge_{suffix}_f1_ge_0.95"] += 1
                    if variant.selected_ratio is not None:
                        for ratio_threshold in (0.50, 0.65, 0.80):
                            if variant.selected_ratio <= ratio_threshold:
                                suffix = str(ratio_threshold).replace(".", "_")
                                counts[f"{mode}_selected_ratio_le_{suffix}"] += 1
                                if f1 >= 0.95:
                                    counts[f"{mode}_selected_ratio_le_{suffix}_f1_ge_0.95"] += 1
                    if f1 >= 0.95:
                        counts[f"{mode}_f1_ge_0.95"] += 1
                    if f1 >= 0.90:
                        counts[f"{mode}_f1_ge_0.90"] += 1
                    if f1 >= 0.80:
                        counts[f"{mode}_f1_ge_0.80"] += 1
                    if digest(variant.content) == base_content_hash:
                        counts[f"{mode}_content_match"] += 1
                    if digest(variant.html) == base_html_hash:
                        counts[f"{mode}_html_match"] += 1
                    if digest(variant.content) == candidate_content_hash:
                        counts[f"{mode}_candidate_content_match"] += 1
                counts["rows"] += 1

                if len(examples) < example_rows:
                    mode_bits = []
                    for mode, variant in variants.items():
                        mode_bits.append(
                            f"{mode}:content_match={digest(variant.content) == base_content_hash}"
                            f":html_match={digest(variant.html) == base_html_hash}"
                            f":f1={token_f1(variant.content, base_content):.3f}"
                            f":sim={variant.sim}"
                            f":selected_ratio={variant.selected_ratio}"
                            f":rep_selected_ratio={rep_selected_ratio if mode == 'synthetic_mapped' else None}"
                            f":synthetic_direct_raw_f1={synthetic_direct_raw_f1 if mode == 'synthetic_mapped' else None}"
                            f":content_len={len(variant.content)}"
                            f":error={compact(variant.error, 80)!r}"
                        )
                    examples.append(
                        "EXAMPLE "
                        f"idx={idx} cluster={cluster_id} rep_idx={rep_idx} "
                        f"url={str(base_df.loc[idx, 'url'])[:180]!r} "
                        f"base_content_len={len(str(base_df.loc[idx, 'dripper_content'] or ''))} "
                        f"candidate_content_len={len(str(candidate_df.loc[idx, 'dripper_content'] or ''))} "
                        f"base={compact(base_df.loc[idx, 'dripper_content'])!r} "
                        f"candidate={compact(candidate_df.loc[idx, 'dripper_content'])!r} "
                        f"variants={' | '.join(mode_bits)}"
                    )

            if validation_records:
                cluster_summary = (
                    f"cluster={cluster_id} rows={group_rows} rep_idx={rep_idx} "
                    f"rep_url={compact(base_df.loc[rep_idx, 'url'] if 'url' in base_df.columns else '', 160)!r} "
                    f"rep_selected_ratio={mapping_cache.get(cluster_id, {}).get('_diagnostic_rep_selected_ratio')} "
                    f"validation={' ; '.join(validation_records)}"
                )
                if group_validation_failed and len(failed_cluster_examples) < example_rows:
                    failed_cluster_examples.append(f"FAILED_CLUSTER {cluster_summary}")
                elif not group_validation_failed and len(passed_cluster_examples) < example_rows:
                    passed_cluster_examples.append(f"PASSED_CLUSTER {cluster_summary}")

    print(f"rebuilt_layout_groups={processed_groups}")
    print(f"representative_rows={representative_rows}")
    print(f"diagnosed_rows={processed_rows}")

    print("COUNTS_BEGIN")
    for key in sorted(counts):
        print(f"{key}={counts[key]}")
    print("COUNTS_END")
    if counts["rows"]:
        print("VARIANT_TIMING_BEGIN")
        for mode in variant_modes:
            elapsed_s = float(variant_timing_s.get(mode, 0.0))
            print(
                f"{mode}_elapsed_s={elapsed_s:.6f} "
                f"{mode}_mean_elapsed_s={elapsed_s / counts['rows']:.6f} "
                f"{mode}_rows={counts['rows']}"
            )
        print("VARIANT_TIMING_END")
        print("F1_MEAN_BEGIN")
        for mode in sorted(f1_sums):
            denom = f1_counts[mode] or counts["rows"]
            print(f"{mode}_mean_f1={f1_sums[mode] / denom:.6f}")
        print("F1_MEAN_END")
    if errors:
        print("ERRORS_BEGIN")
        for error, count in errors.most_common(10):
            print(f"count={count} error={error!r}")
        print("ERRORS_END")
    if failed_cluster_examples:
        print("FAILED_CLUSTERS_BEGIN")
        for example in failed_cluster_examples:
            print(example)
        print("FAILED_CLUSTERS_END")
    if passed_cluster_examples:
        print("PASSED_CLUSTERS_BEGIN")
        for example in passed_cluster_examples:
            print(example)
        print("PASSED_CLUSTERS_END")
    if examples:
        print("EXAMPLES_BEGIN")
        for example in examples:
            print(example)
        print("EXAMPLES_END")
    output_dir_value = os.environ.get("DIAG_OUTPUT_DIR") or os.environ.get("RUN_DIR") or ""
    if output_dir_value:
        output_dir = Path(output_dir_value)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_rows": int(len(base_df)),
            "candidate_rows": int(len(candidate_df)),
            "max_rows": int(max_rows),
            "diagnosed_rows": int(processed_rows),
            "rebuilt_shards": int(len(shards)),
            "rebuilt_layout_groups": int(processed_groups),
            "representative_rows": int(representative_rows),
            "layout_cluster_threshold": float(threshold),
            "layout_page_signature_mode": page_signature_mode,
            "layout_template_validation_rows": int(validation_rows),
            "layout_template_validation_min_f1": float(validation_min_f1),
            "layout_template_validation_signature_mode": validation_signature_mode,
            "layout_template_min_content_length_ratio": float(min_content_length_ratio_value),
            "layout_template_max_content_length_ratio": float(max_content_length_ratio_value),
            "layout_template_failed_layout_fallback_signature_mode": failed_layout_fallback_signature_mode,
            "layout_template_propagation_target": propagation_target,
            "layout_diag_variant_modes": list(variant_modes),
            "layout_target_hosts": sorted(target_hosts),
            "layout_force_host_single_cluster": bool(force_host_single_cluster),
            "counts": {str(key): int(value) for key, value in sorted(counts.items())},
            "variant_timing_s": {str(key): float(value) for key, value in sorted(variant_timing_s.items())},
        }
        (output_dir / "layout_diag_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"METADATA_JSON={output_dir / 'layout_diag_metadata.json'}")
        if cluster_trace_rows:
            pd.DataFrame(cluster_trace_rows).to_csv(output_dir / "layout_diag_clusters.csv", index=False)
            print(f"CLUSTER_TRACE_CSV={output_dir / 'layout_diag_clusters.csv'}")
        if propagation_trace_rows:
            pd.DataFrame(propagation_trace_rows).to_csv(output_dir / "layout_diag_propagation.csv", index=False)
            print(f"PROPAGATION_TRACE_CSV={output_dir / 'layout_diag_propagation.csv'}")
    print("LAYOUT_PROPAGATION_DIAG_END")


if __name__ == "__main__":
    main()
