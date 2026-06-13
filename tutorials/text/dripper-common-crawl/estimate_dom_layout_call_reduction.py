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

"""Estimate global Dripper call reduction from llm-webkit DOM layouts.

This is CPU-only and intentionally read-only.  It consumes a Dripper output
directory or a parquet/jsonl file containing at least ``url`` and ``html``.  If
Dripper response/token columns are present, they are used to estimate how many
LLM calls and tokens would remain after snapshot-wide host-bounded DOM-layout
representative selection.

Unlike ``estimate_layout_call_reduction.py``, this runs the actual
ccprocessor/llm-webkit structural feature extraction and DBSCAN layout
clustering.  That makes it useful for checking the AICC paper's core thesis:
infer one representative per host/layout cluster, then propagate templates on
CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import pandas as pd
from llm_web_kit.html_layout.html_layout_cosin import cluster_html_struct, get_feature
from llm_web_kit.main_html_parser.typical_html.typical_html import select_representative_html

SIGNATURE_MODES = {
    "none",
    "url_shape",
    "item_count_bucket",
    "item_count_exact",
    "url_shape_item_count_bucket",
    "url_shape_item_count_exact",
}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Dripper DOM-layout representative-call reduction")
    parser.add_argument("--input", required=True, help="Dripper output dir, parquet/jsonl file, directory, or glob")
    parser.add_argument("--output", required=True, help="Output JSON metrics path")
    parser.add_argument("--html-col", default="html")
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--host-col", default="url_host_name")
    parser.add_argument("--response-col", default="dripper_response")
    parser.add_argument("--token-col", default="dripper_total_tokens")
    parser.add_argument("--item-count-col", default="dripper_item_count")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--thresholds", default="0.95,0.97,0.99")
    parser.add_argument(
        "--signature-modes",
        default="none,url_shape",
        help=f"Comma-separated values from {sorted(SIGNATURE_MODES)}",
    )
    parser.add_argument(
        "--max-exact-host-pages",
        type=int,
        default=2048,
        help=("Skip exact O(n^2) DBSCAN for hosts above this candidate-page count. Use 0 to disable the cap."),
    )
    parser.add_argument(
        "--large-host-mode",
        choices=["standalone", "feature_hash"],
        default="standalone",
        help=(
            "How to handle hosts above --max-exact-host-pages. standalone counts their rows as LLM calls. "
            "feature_hash groups exact normalized DOM structural feature fingerprints as conservative layouts."
        ),
    )
    parser.add_argument("--top-hosts", type=int, default=20)
    parser.add_argument("--top-groups", type=int, default=20)
    parser.add_argument(
        "--log-hosts-min-pages",
        type=int,
        default=1024,
        help="Print per-host clustering progress for hosts with at least this many candidate pages. Use 0 to disable.",
    )
    args = parser.parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows must be non-negative")
    if args.min_cluster_size <= 1:
        raise ValueError("--min-cluster-size must be greater than 1")
    if args.max_exact_host_pages < 0:
        raise ValueError("--max-exact-host-pages must be non-negative")
    if args.top_hosts < 0 or args.top_groups < 0 or args.log_hosts_min_pages < 0:
        raise ValueError("--top-hosts, --top-groups, and --log-hosts-min-pages must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    thresholds = parse_float_list(args.thresholds)
    signature_modes = parse_signature_modes(args.signature_modes)
    input_files = resolve_input_files(args.input)
    df = read_input_dataframe(input_files)
    if args.max_rows:
        df = df.head(args.max_rows)
    df = df.reset_index(drop=True)
    if args.html_col not in df.columns:
        raise ValueError(f"Input is missing HTML column: {args.html_col!r}")

    rows = len(df)
    if rows == 0:
        raise RuntimeError(f"Input has no rows: {args.input}")

    print(
        "DOM_LAYOUT_ESTIMATE_LOAD "
        f"rows={rows} files={len(input_files)} thresholds={thresholds} signature_modes={signature_modes}",
        flush=True,
    )

    features = build_feature_index(df, args)
    metrics_by_threshold: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        threshold_key = f"{threshold:.4g}"
        metrics_by_threshold[threshold_key] = {}
        print(f"DOM_LAYOUT_CLUSTER_THRESHOLD_BEGIN threshold={threshold_key}", flush=True)
        clustered = cluster_by_host(features, threshold=threshold, args=args)
        for signature_mode in signature_modes:
            estimate = estimate_calls_for_signature(df, features, clustered, signature_mode=signature_mode, args=args)
            metrics_by_threshold[threshold_key][signature_mode] = estimate
            print(
                "DOM_LAYOUT_ESTIMATE_RESULT "
                f"threshold={threshold_key} signature={signature_mode} "
                f"estimated_calls={estimate['estimated_llm_calls']} "
                f"call_ratio={estimate['llm_call_ratio']:.6f} "
                f"reduction={estimate['llm_call_reduction_factor']:.3f} "
                f"token_reduction={estimate['token_reduction_factor']:.3f} "
                f"groups={estimate['layout_groups']} propagated_pages={estimate['propagated_pages']}",
                flush=True,
            )
        print(f"DOM_LAYOUT_CLUSTER_THRESHOLD_END threshold={threshold_key}", flush=True)

    metrics = {
        "input": args.input,
        "files": [str(path) for path in input_files],
        "rows": rows,
        "html_col": args.html_col,
        "url_col": args.url_col,
        "host_col": args.host_col,
        "response_col": args.response_col,
        "token_col": args.token_col,
        "item_count_col": args.item_count_col,
        "max_rows": args.max_rows,
        "min_cluster_size": args.min_cluster_size,
        "max_exact_host_pages": args.max_exact_host_pages,
        "large_host_mode": args.large_host_mode,
        "feature_metrics": features.summary,
        "threshold_metrics": metrics_by_threshold,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print("DOM_LAYOUT_CALL_REDUCTION_ESTIMATE_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("DOM_LAYOUT_CALL_REDUCTION_ESTIMATE_END")
    print(f"OUTPUT={output_path}")
    return 0


class FeatureIndex:
    def __init__(
        self,
        *,
        samples_by_host: dict[str, list[dict[str, Any]]],
        needs_llm_rows: set[int],
        feature_rows: set[int],
        no_feature_rows: set[int],
        no_llm_rows: set[int],
        row_hosts: dict[int, str],
        row_tokens: dict[int, int],
        summary: dict[str, Any],
    ) -> None:
        self.samples_by_host = samples_by_host
        self.needs_llm_rows = needs_llm_rows
        self.feature_rows = feature_rows
        self.no_feature_rows = no_feature_rows
        self.no_llm_rows = no_llm_rows
        self.row_hosts = row_hosts
        self.row_tokens = row_tokens
        self.summary = summary


def build_feature_index(df: pd.DataFrame, args: argparse.Namespace) -> FeatureIndex:
    samples_by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    needs_llm_rows: set[int] = set()
    feature_rows: set[int] = set()
    no_feature_rows: set[int] = set()
    no_llm_rows: set[int] = set()
    row_hosts: dict[int, str] = {}
    row_tokens: dict[int, int] = {}
    feature_errors: Counter[str] = Counter()

    for idx, row in df.iterrows():
        row_hosts[idx] = row_host(row, args)
        row_tokens[idx] = coerce_int(row.get(args.token_col)) if args.token_col in df.columns else 0
        if not row_needs_llm(row, args):
            no_llm_rows.add(idx)
            continue
        needs_llm_rows.add(idx)
        html = coerce_html(row.get(args.html_col))
        if not html.strip():
            no_feature_rows.add(idx)
            continue
        try:
            feature = get_feature(html)
        except Exception as exc:
            feature_errors[str(exc)[:160]] += 1
            no_feature_rows.add(idx)
            continue
        if feature is None:
            no_feature_rows.add(idx)
            continue
        feature_rows.add(idx)
        samples_by_host[row_hosts[idx]].append({"track_id": str(idx), "html": html, "feature": feature})

    host_sizes = Counter({host: len(samples) for host, samples in samples_by_host.items()})
    summary = {
        "rows": len(df),
        "needs_llm_rows": len(needs_llm_rows),
        "no_llm_rows": len(no_llm_rows),
        "feature_rows": len(feature_rows),
        "no_feature_rows": len(no_feature_rows),
        "hosts_with_features": len(samples_by_host),
        "host_feature_page_quantiles": histogram_quantiles(Counter(host_sizes.values())),
        "feature_error_count": sum(feature_errors.values()),
        "feature_errors": dict(feature_errors.most_common(20)),
        "baseline_total_tokens": int(sum(row_tokens[idx] for idx in needs_llm_rows)),
    }
    print(
        "DOM_LAYOUT_FEATURES "
        f"needs_llm={summary['needs_llm_rows']} feature_rows={summary['feature_rows']} "
        f"hosts={summary['hosts_with_features']} no_feature={summary['no_feature_rows']} "
        f"errors={summary['feature_error_count']}",
        flush=True,
    )
    return FeatureIndex(
        samples_by_host=dict(samples_by_host),
        needs_llm_rows=needs_llm_rows,
        feature_rows=feature_rows,
        no_feature_rows=no_feature_rows,
        no_llm_rows=no_llm_rows,
        row_hosts=row_hosts,
        row_tokens=row_tokens,
        summary=summary,
    )


def cluster_by_host(features: FeatureIndex, *, threshold: float, args: argparse.Namespace) -> dict[str, Any]:
    layout_by_row: dict[int, int] = {}
    skipped_rows: set[int] = set()
    skipped_hosts: dict[str, int] = {}
    feature_hash_hosts: dict[str, int] = {}
    cluster_errors: Counter[str] = Counter()
    layout_key_counter = 0

    for host, samples in features.samples_by_host.items():
        log_host = bool(args.log_hosts_min_pages and len(samples) >= args.log_hosts_min_pages)
        if log_host:
            print(
                f"DOM_LAYOUT_CLUSTER_HOST_BEGIN threshold={threshold:.4g} host={host} rows={len(samples)}",
                flush=True,
            )
        if len(samples) < args.min_cluster_size:
            for sample in samples:
                layout_by_row[int(sample["track_id"])] = -1
            if log_host:
                print(
                    "DOM_LAYOUT_CLUSTER_HOST_END "
                    f"threshold={threshold:.4g} host={host} rows={len(samples)} mode=too_small layouts=0",
                    flush=True,
                )
            continue
        if args.max_exact_host_pages and len(samples) > args.max_exact_host_pages:
            if args.large_host_mode == "feature_hash":
                feature_hash_hosts[host] = len(samples)
                by_fingerprint: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for sample in samples:
                    by_fingerprint[feature_fingerprint(sample["feature"])].append(sample)
                for fingerprint_samples in by_fingerprint.values():
                    if len(fingerprint_samples) < args.min_cluster_size:
                        for sample in fingerprint_samples:
                            layout_by_row[int(sample["track_id"])] = -1
                        continue
                    layout_id = layout_key_counter
                    layout_key_counter += 1
                    for sample in fingerprint_samples:
                        layout_by_row[int(sample["track_id"])] = layout_id
            else:
                skipped_hosts[host] = len(samples)
                skipped_rows.update(int(sample["track_id"]) for sample in samples)
            if log_host:
                print(
                    "DOM_LAYOUT_CLUSTER_HOST_END "
                    f"threshold={threshold:.4g} host={host} rows={len(samples)} mode=large_host "
                    f"layouts={layout_key_counter}",
                    flush=True,
                )
            continue
        try:
            clustered_samples, _layout_ids = cluster_html_struct(samples, threshold=threshold)
        except Exception as exc:
            cluster_errors[str(exc)[:160]] += 1
            skipped_hosts[host] = len(samples)
            skipped_rows.update(int(sample["track_id"]) for sample in samples)
            if log_host:
                print(
                    "DOM_LAYOUT_CLUSTER_HOST_END "
                    f"threshold={threshold:.4g} host={host} rows={len(samples)} mode=error",
                    flush=True,
                )
            continue

        host_layout_ids: dict[int, int] = {}
        for sample in clustered_samples:
            row_idx = int(sample["track_id"])
            local_layout_id = int(sample.get("layout_id", -1))
            if local_layout_id < 0:
                layout_by_row[row_idx] = -1
                continue
            if local_layout_id not in host_layout_ids:
                host_layout_ids[local_layout_id] = layout_key_counter
                layout_key_counter += 1
            layout_by_row[row_idx] = host_layout_ids[local_layout_id]
        if log_host:
            clustered_rows = sum(1 for sample in clustered_samples if int(sample.get("layout_id", -1)) >= 0)
            print(
                "DOM_LAYOUT_CLUSTER_HOST_END "
                f"threshold={threshold:.4g} host={host} rows={len(samples)} "
                f"layouts={len(host_layout_ids)} clustered_rows={clustered_rows}",
                flush=True,
            )

    return {
        "layout_by_row": layout_by_row,
        "skipped_rows": skipped_rows,
        "skipped_hosts": skipped_hosts,
        "feature_hash_hosts": feature_hash_hosts,
        "cluster_errors": dict(cluster_errors.most_common(20)),
    }


def estimate_calls_for_signature(
    df: pd.DataFrame,
    features: FeatureIndex,
    clustered: dict[str, Any],
    *,
    signature_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    layout_by_row: dict[int, int] = clustered["layout_by_row"]
    skipped_rows: set[int] = clustered["skipped_rows"]

    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    standalone_rows: set[int] = set(features.no_feature_rows)
    standalone_rows.update(skipped_rows)

    for row_idx in features.feature_rows:
        if row_idx in skipped_rows:
            continue
        layout_id = layout_by_row.get(row_idx, -1)
        if layout_id < 0:
            standalone_rows.add(row_idx)
            continue
        signature = layout_page_signature_key(df.iloc[row_idx], args, signature_mode)
        grouped[(layout_id, signature)].append(row_idx)

    layout_groups: list[list[int]] = []
    for indexes in grouped.values():
        if len(indexes) >= args.min_cluster_size:
            layout_groups.append(sorted(indexes))
        else:
            standalone_rows.update(indexes)

    representative_rows: set[int] = set()
    group_size_hist: Counter[int] = Counter()
    group_host_counter: Counter[str] = Counter()
    top_groups: list[dict[str, Any]] = []
    for indexes in layout_groups:
        representative = select_representative_index(df, indexes, args)
        representative_rows.add(representative)
        group_size = len(indexes)
        group_size_hist[group_size] += 1
        host = features.row_hosts.get(indexes[0], "")
        group_host_counter[host] += 1
        if args.top_groups and len(top_groups) < args.top_groups:
            top_groups.append(
                {
                    "host": host,
                    "rows": group_size,
                    "representative_row": int(representative),
                    "representative_url": str(df.iloc[representative].get(args.url_col, ""))[:300]
                    if args.url_col in df.columns
                    else "",
                }
            )

    estimated_llm_calls = len(standalone_rows) + len(layout_groups)
    baseline_llm_calls = len(features.needs_llm_rows)
    propagated_pages = sum(len(indexes) - 1 for indexes in layout_groups)
    baseline_total_tokens = int(features.summary.get("baseline_total_tokens", 0))
    estimated_total_tokens = int(
        sum(features.row_tokens.get(row_idx, 0) for row_idx in standalone_rows)
        + sum(features.row_tokens.get(row_idx, 0) for row_idx in representative_rows)
    )

    group_pages = sum(size * count for size, count in group_size_hist.items())
    host_sizes = Counter()
    for row_idx in features.needs_llm_rows:
        host_sizes[features.row_hosts.get(row_idx, "")] += 1

    return {
        "baseline_llm_calls": baseline_llm_calls,
        "estimated_llm_calls": estimated_llm_calls,
        "saved_llm_calls": baseline_llm_calls - estimated_llm_calls,
        "llm_call_ratio": safe_ratio(estimated_llm_calls, baseline_llm_calls),
        "all_page_call_ratio": safe_ratio(estimated_llm_calls, len(df)),
        "llm_call_reduction_factor": safe_ratio(baseline_llm_calls, estimated_llm_calls),
        "baseline_total_tokens": baseline_total_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "saved_total_tokens": baseline_total_tokens - estimated_total_tokens,
        "token_ratio": safe_ratio(estimated_total_tokens, baseline_total_tokens),
        "token_reduction_factor": safe_ratio(baseline_total_tokens, estimated_total_tokens),
        "layout_groups": len(layout_groups),
        "layout_group_pages": group_pages,
        "layout_group_page_ratio": safe_ratio(group_pages, baseline_llm_calls),
        "propagated_pages": propagated_pages,
        "propagated_page_ratio": safe_ratio(propagated_pages, baseline_llm_calls),
        "standalone_llm_rows": len(standalone_rows),
        "representative_rows": len(representative_rows),
        "no_llm_rows": len(features.no_llm_rows),
        "no_feature_rows": len(features.no_feature_rows),
        "skipped_exact_host_rows": len(clustered["skipped_rows"]),
        "skipped_exact_hosts": len(clustered["skipped_hosts"]),
        "feature_hash_hosts": len(clustered["feature_hash_hosts"]),
        "feature_hash_host_rows": int(sum(clustered["feature_hash_hosts"].values())),
        "cluster_errors": clustered["cluster_errors"],
        "layout_group_size_quantiles": histogram_quantiles(group_size_hist),
        "layout_group_size_buckets": size_buckets(group_size_hist),
        "top_hosts_by_need_llm_pages": [
            {"host": host, "pages": count, "layout_groups": group_host_counter.get(host, 0)}
            for host, count in host_sizes.most_common(args.top_hosts)
        ],
        "top_layout_groups_sample": top_groups,
        "skipped_hosts_sample": [
            {"host": host, "pages": count}
            for host, count in sorted(clustered["skipped_hosts"].items(), key=lambda item: (-item[1], item[0]))[
                : args.top_hosts
            ]
        ],
        "feature_hash_hosts_sample": [
            {"host": host, "pages": count}
            for host, count in sorted(clustered["feature_hash_hosts"].items(), key=lambda item: (-item[1], item[0]))[
                : args.top_hosts
            ]
        ],
    }


def select_representative_index(df: pd.DataFrame, indexes: list[int], args: argparse.Namespace) -> int:
    candidates = [{"track_id": str(idx), "html": coerce_html(df.iloc[idx].get(args.html_col))} for idx in indexes]
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


def row_needs_llm(row: pd.Series, args: argparse.Namespace) -> bool:
    if args.response_col not in row.index:
        return True
    return bool(str(row.get(args.response_col) or "").strip())


def row_host(row: pd.Series, args: argparse.Namespace) -> str:
    if args.host_col in row.index:
        host = normalize_host(row.get(args.host_col))
        if host:
            return host
    if args.url_col in row.index:
        return url_host_key(row.get(args.url_col))
    return ""


def layout_page_signature_key(row: pd.Series, args: argparse.Namespace, mode: str) -> str:
    if mode == "none":
        return ""
    parts: list[str] = []
    if "url_shape" in mode:
        url_value = row.get(args.url_col) if args.url_col in row.index else None
        parts.append(f"url={url_shape_key(url_value)}")
    if "item_count_exact" in mode:
        parts.append(f"items={coerce_int(row.get(args.item_count_col))}")
    elif "item_count_bucket" in mode:
        parts.append(f"items={item_count_bucket(coerce_int(row.get(args.item_count_col)))}")
    return "|".join(parts)


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


def coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def item_count_bucket(count: int) -> str:
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


def url_host_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        if not parsed.hostname and "://" not in text:
            parsed = urlparse(f"//{text}")
    except ValueError:
        return ""
    return normalize_host(parsed.hostname or "")


def normalize_host(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower().rstrip(".")
    if not text:
        return ""
    try:
        return text.encode("idna").decode("ascii")
    except UnicodeError:
        return text


def url_shape_key(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        if not parsed.hostname and "://" not in text:
            parsed = urlparse(f"//{text}")
    except ValueError:
        return ""
    raw_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    query_keys = ",".join(sorted({key for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}))
    if parsed.query:
        normalized_segments = [segment.lower() for segment in raw_segments]
    else:
        normalized_segments = [normalize_url_path_segment(segment) for segment in raw_segments]
    return f"path={'/'.join(normalized_segments)}|q={query_keys}"


def normalize_url_path_segment(segment: str) -> str:
    segment = segment.lower()
    suffix = ""
    if "." in segment:
        segment, extension = segment.rsplit(".", 1)
        suffix = f".{extension}"
    if re.search(r"\d", segment):
        return f"#num{suffix}"
    return f"{segment}{suffix}"


def feature_fingerprint(feature: Any) -> str:
    if not isinstance(feature, dict):
        return ""

    def normalize_part(part: str) -> dict[str, list[tuple[str, int]]]:
        raw_layers = feature.get(part, {})
        if not isinstance(raw_layers, dict):
            return {}
        normalized: dict[str, list[tuple[str, int]]] = {}
        for layer, values in raw_layers.items():
            if not isinstance(values, list):
                continue
            counts = Counter(str(value) for value in values)
            normalized[str(layer)] = sorted(counts.items())
        return normalized

    payload = {
        "tags": normalize_part("tags"),
        "attrs": normalize_part("attrs"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def resolve_input_files(input_value: str) -> list[Path]:
    path = Path(input_value)
    if path.is_dir():
        preferred = [path / "dripper_results.parquet", path / "dripper_results.jsonl"]
        for candidate in preferred:
            if candidate.exists():
                return [candidate]
        files: list[Path] = []
        for extension in ("*.parquet", "*.jsonl", "*.json", "*.csv"):
            files.extend(sorted(path.glob(extension)))
        return [candidate for candidate in files if not candidate.name.startswith("_")]
    if any(char in input_value for char in "*?["):
        return [Path(candidate) for candidate in sorted(glob(input_value))]
    return [path]


def read_input_dataframe(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No input files matched")
    frames = [read_input_file(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def read_input_file(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".jsonl"):
        return pd.read_json(path, orient="records", lines=True)
    if suffixes.endswith(".json"):
        return pd.read_json(path)
    if suffixes.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file extension: {path}")


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one threshold")
    for threshold in values:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"Invalid threshold: {threshold}")
    return values


def parse_signature_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise ValueError("Expected at least one signature mode")
    unknown = sorted(set(modes).difference(SIGNATURE_MODES))
    if unknown:
        raise ValueError(f"Unknown signature mode(s): {unknown}")
    return modes


def histogram_quantiles(hist: Counter[int]) -> dict[str, float | int]:
    total = sum(hist.values())
    if total == 0:
        return {"count": 0}
    targets = {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}
    out: dict[str, float | int] = {
        "count": int(total),
        "mean": sum(size * count for size, count in hist.items()) / total,
        "max": int(max(hist)),
    }
    seen = 0
    pending = sorted(targets.items(), key=lambda item: item[1])
    pending_index = 0
    for size, count in sorted(hist.items()):
        seen += count
        while pending_index < len(pending) and seen >= math.ceil(total * pending[pending_index][1]):
            out[pending[pending_index][0]] = int(size)
            pending_index += 1
    return out


def size_buckets(hist: Counter[int]) -> dict[str, dict[str, int]]:
    buckets = {
        "1": (1, 1),
        "2-3": (2, 3),
        "4-7": (4, 7),
        "8-15": (8, 15),
        "16-31": (16, 31),
        "32-63": (32, 63),
        "64-127": (64, 127),
        "128-255": (128, 255),
        "256+": (256, None),
    }
    out = {name: {"groups": 0, "pages": 0} for name in buckets}
    for size, count in hist.items():
        for name, (start, end) in buckets.items():
            if size >= start and (end is None or size <= end):
                out[name]["groups"] += int(count)
                out[name]["pages"] += int(size * count)
                break
    return out


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
