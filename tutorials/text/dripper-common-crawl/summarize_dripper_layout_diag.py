#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _cluster_hosts(row: dict[str, str]) -> str:
    try:
        hosts = json.loads(row.get("hosts") or "{}")
    except json.JSONDecodeError:
        hosts = {}
    if not hosts:
        return ""
    return ",".join(f"{host}:{count}" for host, count in sorted(hosts.items()))


def _url_host(url: str) -> str:
    if "://" in url:
        url = url.split("://", 1)[1]
    return url.split("/", 1)[0].lower()


def _guard_summary(
    name: str,
    rows: list[dict[str, str]],
    baseline_pages: int,
    quality_key: str,
    predicate: Any,
) -> str:
    saved_f1s: list[float] = []
    saved = 0
    content_matches = 0
    for row in rows:
        if not predicate(row):
            continue
        f1 = _float(row.get(quality_key))
        if f1 is None:
            continue
        saved += 1
        saved_f1s.append(f1)
        if _bool(row.get("direct_raw_content_match")):
            content_matches += 1
    estimated_calls = baseline_pages - saved
    reduction = saved / baseline_pages if baseline_pages else 0.0
    mean_f1 = statistics.fmean(saved_f1s) if saved_f1s else 0.0
    f1_ge_080 = sum(value >= 0.80 for value in saved_f1s)
    f1_ge_090 = sum(value >= 0.90 for value in saved_f1s)
    f1_ge_095 = sum(value >= 0.95 for value in saved_f1s)
    f1_ge_098 = sum(value >= 0.98 for value in saved_f1s)
    return (
        "GUARD "
        f"name={name} "
        f"saved={saved} "
        f"estimated_calls={estimated_calls} "
        f"call_reduction={reduction:.6f} "
        f"mean_direct_raw_f1={mean_f1:.6f} "
        f"direct_raw_f1_lt_0_80={saved - f1_ge_080} "
        f"direct_raw_f1_lt_0_90={saved - f1_ge_090} "
        f"direct_raw_f1_lt_0_95={saved - f1_ge_095} "
        f"direct_raw_f1_lt_0_98={saved - f1_ge_098} "
        f"content_matches={content_matches}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("diag_dir", type=Path)
    parser.add_argument("--validation-mode", default="direct_raw")
    parser.add_argument("--validation-min-f1", type=float, default=0.98)
    parser.add_argument("--input-rows", type=int, default=None)
    parser.add_argument("--assume-uncapped", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    clusters_path = args.diag_dir / "layout_diag_clusters.csv"
    propagation_path = args.diag_dir / "layout_diag_propagation.csv"
    if not clusters_path.exists() or not propagation_path.exists():
        raise SystemExit(f"missing diagnostic CSVs under {args.diag_dir}")

    clusters = _read_csv(clusters_path)
    rows = _read_csv(propagation_path)
    metadata = _read_metadata(args.diag_dir / "layout_diag_metadata.json")
    mode = args.validation_mode
    f1_key = f"{mode}_f1"
    error_key = f"{mode}_error"
    match_key = f"{mode}_content_match"

    cluster_by_id = {row["cluster_id"]: row for row in clusters}
    rows_by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_cluster[row["cluster_id"]].append(row)

    active_cluster_statuses = {"", "active"}
    active_clusters = sum(1 for row in clusters if row.get("status", "active") in active_cluster_statuses)

    failed_clusters: set[str] = set()
    validation_counts = Counter()
    for cluster_id, cluster_rows in rows_by_cluster.items():
        validation_rows = [row for row in cluster_rows if _bool(row.get("validation_sample"))]
        for row in validation_rows:
            validation_counts["samples"] += 1
            f1 = _float(row.get(f1_key))
            if row.get(error_key) or f1 is None or f1 < args.validation_min_f1 or _bool(row.get("validation_content_length_reject")):
                failed_clusters.add(cluster_id)
                validation_counts["failed_samples"] += 1
        if validation_rows and cluster_id not in failed_clusters:
            validation_counts["passed_clusters"] += 1
        elif validation_rows:
            validation_counts["failed_clusters"] += 1

    saved_rows = 0
    fallback_rows = 0
    content_matches = 0
    f1_values: list[float] = []
    saved_f1_values: list[float] = []
    f1_ge = Counter()
    host_counts = Counter()
    passed_clusters_with_low_f1 = 0
    passed_clusters_bad_saved_rows = 0
    for cluster_id, cluster_rows in rows_by_cluster.items():
        if cluster_id in failed_clusters:
            continue
        non_validation_f1s = [
            _float(row.get(f1_key))
            for row in cluster_rows
            if (
                not _bool(row.get("validation_sample"))
                and not row.get(error_key)
                and not _bool(row.get("validation_content_length_reject"))
            )
        ]
        non_validation_f1s = [value for value in non_validation_f1s if value is not None]
        if not non_validation_f1s:
            continue
        min_f1 = min(non_validation_f1s)
        if min_f1 < args.validation_min_f1:
            passed_clusters_with_low_f1 += 1
            passed_clusters_bad_saved_rows += sum(value < args.validation_min_f1 for value in non_validation_f1s)
    for row in rows:
        cluster_id = row["cluster_id"]
        if (
            _bool(row.get("validation_sample"))
            or cluster_id in failed_clusters
            or row.get(error_key)
            or _bool(row.get("validation_content_length_reject"))
        ):
            fallback_rows += 1
            continue
        saved_rows += 1
        f1 = _float(row.get(f1_key))
        if f1 is not None:
            saved_f1_values.append(f1)
            for threshold in (0.80, 0.90, 0.95, 0.98):
                if f1 >= threshold:
                    f1_ge[f"saved_f1_ge_{threshold:.2f}"] += 1
        if _bool(row.get(match_key)):
            content_matches += 1
        host_counts[_url_host(row.get("url") or "")] += 1

    for row in rows:
        f1 = _float(row.get(f1_key))
        if f1 is not None:
            f1_values.append(f1)

    print("SUMMARY_BEGIN")
    print(f"diag_dir={args.diag_dir}")
    print(f"validation_mode={mode}")
    print(f"validation_min_f1={args.validation_min_f1}")
    print(f"clusters={len(clusters)}")
    print(f"active_representative_rows={active_clusters}")
    print(f"propagation_rows={len(rows)}")
    baseline_pages = len(rows) + active_clusters
    estimated_llm_calls = baseline_pages - saved_rows
    print(f"estimated_baseline_llm_calls={baseline_pages}")
    print(f"estimated_layout_llm_calls_without_parent_probe_overhead={estimated_llm_calls}")
    print(
        f"estimated_call_reduction_without_parent_probe_overhead={saved_rows / baseline_pages:.6f}"
        if baseline_pages
        else "estimated_call_reduction_without_parent_probe_overhead=0"
    )
    input_rows = args.input_rows or metadata.get("input_rows")
    max_rows = metadata.get("max_rows")
    diagnosed_rows = metadata.get("diagnosed_rows")
    uncapped = args.assume_uncapped or (
        isinstance(max_rows, int)
        and isinstance(diagnosed_rows, int)
        and (max_rows <= 0 or diagnosed_rows < max_rows)
    )
    if input_rows and uncapped:
        full_standalone_rows = max(0, int(input_rows) - baseline_pages)
        full_estimated_llm_calls = estimated_llm_calls + full_standalone_rows
        print(f"full_input_rows={int(input_rows)}")
        print(f"full_input_standalone_rows={full_standalone_rows}")
        print(f"full_input_estimated_layout_llm_calls={full_estimated_llm_calls}")
        print(
            f"full_input_estimated_call_reduction={saved_rows / int(input_rows):.6f}"
            if input_rows
            else "full_input_estimated_call_reduction=0"
        )
    elif input_rows:
        print(f"full_input_rows={int(input_rows)}")
        print("full_input_metrics_available=0")
        if max_rows is not None:
            print(f"full_input_metrics_unavailable_reason=max_rows_cap_reached:{max_rows}")
    print(f"validation_samples={validation_counts['samples']}")
    print(f"validation_failed_samples={validation_counts['failed_samples']}")
    print(f"validation_passed_clusters={validation_counts['passed_clusters']}")
    print(f"validation_failed_clusters={validation_counts['failed_clusters']}")
    print(f"validated_saved_rows={saved_rows}")
    print(f"validated_fallback_rows={fallback_rows}")
    print(f"validated_saved_fraction={saved_rows / len(rows):.6f}" if rows else "validated_saved_fraction=0")
    print(f"validated_saved_content_matches={content_matches}")
    print(f"validated_saved_rows_f1_lt_threshold={sum(value < args.validation_min_f1 for value in saved_f1_values)}")
    print(f"passed_validation_clusters_with_saved_min_f1_lt_threshold={passed_clusters_with_low_f1}")
    print(f"passed_validation_bad_saved_rows_below_threshold={passed_clusters_bad_saved_rows}")
    print(
        f"validated_saved_content_match_fraction={content_matches / saved_rows:.6f}"
        if saved_rows
        else "validated_saved_content_match_fraction=0"
    )
    if f1_values:
        print(f"all_rows_mean_{mode}_f1={statistics.fmean(f1_values):.6f}")
    if saved_f1_values:
        print(f"saved_rows_mean_{mode}_f1={statistics.fmean(saved_f1_values):.6f}")
    for key in sorted(f1_ge):
        print(f"{key}={f1_ge[key]}")
    print("CPU_GUARDRAILS_BEGIN")
    print(
        _guard_summary(
            "direct_raw_no_error",
            rows,
            baseline_pages,
            f1_key,
            lambda row: not row.get("direct_raw_error"),
        )
    )
    for threshold in (0.80, 0.90, 0.95, 0.98):
        print(
            _guard_summary(
                f"synthetic_direct_raw_consensus_ge_{threshold:.2f}",
                rows,
                baseline_pages,
                f1_key,
                lambda row, threshold=threshold: (
                    not row.get("direct_raw_error")
                    and not row.get("synthetic_mapped_error")
                    and (_float(row.get("synthetic_direct_raw_f1")) or 0.0) >= threshold
                ),
            )
        )
    for threshold in (0.50, 0.65, 0.80):
        print(
            _guard_summary(
                f"synthetic_selected_ratio_le_{threshold:.2f}",
                rows,
                baseline_pages,
                f1_key,
                lambda row, threshold=threshold: (
                    not row.get("direct_raw_error")
                    and (_float(row.get("synthetic_mapped_selected_ratio")) or 2.0) <= threshold
                ),
            )
        )
    for threshold in (0.35, 0.50, 0.65):
        print(
            _guard_summary(
                f"representative_selected_ratio_le_{threshold:.2f}",
                rows,
                baseline_pages,
                f1_key,
                lambda row, threshold=threshold: (
                    not row.get("direct_raw_error")
                    and (_float(row.get("rep_selected_ratio")) or 2.0) <= threshold
                ),
            )
        )
    print("CPU_GUARDRAILS_END")
    print("HOST_SAVED_ROWS_BEGIN")
    for host, count in host_counts.most_common(args.top):
        print(f"{host}={count}")
    print("HOST_SAVED_ROWS_END")
    print("SUMMARY_END")

    scored_clusters: list[tuple[float, int, str, dict[str, Any]]] = []
    for cluster_id, cluster_rows in rows_by_cluster.items():
        f1s = [_float(row.get(f1_key)) for row in cluster_rows]
        f1s = [value for value in f1s if value is not None]
        mean_f1 = statistics.fmean(f1s) if f1s else -1.0
        min_f1 = min(f1s) if f1s else -1.0
        validation_f1s = [
            _float(row.get(f1_key))
            for row in cluster_rows
            if _bool(row.get("validation_sample"))
        ]
        validation_f1s = [value for value in validation_f1s if value is not None]
        cluster_row = cluster_by_id.get(cluster_id, {})
        scored_clusters.append(
            (
                min_f1,
                -len(cluster_rows),
                cluster_id,
                {
                    "cluster_id": cluster_id,
                    "status": "failed_validation" if cluster_id in failed_clusters else "passed_validation",
                    "rows": len(cluster_rows),
                    "declared_rows": cluster_row.get("rows", ""),
                    "mean_f1": mean_f1,
                    "min_f1": min_f1,
                    "validation_min_f1": min(validation_f1s) if validation_f1s else None,
                    "representative_row": cluster_row.get("representative_row", ""),
                    "representative_url": cluster_row.get("representative_url", ""),
                    "hosts": _cluster_hosts(cluster_row),
                    "worst_url": min(
                        cluster_rows,
                        key=lambda row: _float(row.get(f1_key)) if _float(row.get(f1_key)) is not None else -1.0,
                    ).get("url", ""),
                },
            )
        )

    print("WORST_CLUSTERS_BEGIN")
    for _min_f1, _neg_rows, _cluster_id, row in sorted(scored_clusters)[: args.top]:
        print(json.dumps(row, sort_keys=True))
    print("WORST_CLUSTERS_END")


if __name__ == "__main__":
    main()
