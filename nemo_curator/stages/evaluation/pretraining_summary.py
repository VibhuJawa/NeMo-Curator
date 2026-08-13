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

"""Streaming aggregation for phase-2 pretraining judge outputs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_SINGLE_LABEL_SUFFIXES = (
    "taxonomy_version",
    "topic_family",
    "primary_topic",
    "content_form",
    "phase2_bucket",
    "language_code",
    "language_script",
    "language_bcp47",
    "multilingual",
    "multilingual_mode",
    "language_register",
    "locale_region",
    "quality_tier",
    "knowledge_depth",
    "reasoning_density",
    "temporal_profile",
    "phase2_action",
    "partial_document",
    "judge_view_strategy",
)
_MULTI_LABEL_SUFFIXES = (
    "secondary_topics",
    "training_value_tags",
    "other_language_codes",
    "quality_flags",
    "risk_flags",
)
_CONFIDENCE_SUFFIXES = ("topic_confidence", "language_confidence", "action_confidence")
_VIEW_NUMERIC_SUFFIXES = ("language_share", "original_chars", "judged_chars")
_QUALITY_BANDS = (
    ("1_to_lt_2", 1.0, 2.0, False),
    ("2_to_lt_3", 2.0, 3.0, False),
    ("3_to_lt_4", 3.0, 4.0, False),
    ("4_to_5", 4.0, 5.0, True),
)


@dataclass(slots=True)
class NumericSummary:
    """Mergeable numeric moments for streaming summaries."""

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce")
        numeric = numeric[numeric.map(math.isfinite)]
        if numeric.empty:
            return
        self.count += int(numeric.count())
        self.total += float(numeric.sum())
        batch_min = float(numeric.min())
        batch_max = float(numeric.max())
        self.minimum = batch_min if self.minimum is None else min(self.minimum, batch_min)
        self.maximum = batch_max if self.maximum is None else max(self.maximum, batch_max)

    def update_repeated(self, value: float, count: int) -> None:
        """Merge one already-validated value repeated ``count`` times."""
        self.count += count
        self.total += value * count
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": round(self.total / self.count, 4) if self.count else None,
            "min": self.minimum,
            "max": self.maximum,
        }


@dataclass(slots=True)
class PretrainingJudgeSummary:
    """Accumulate a bounded-memory summary of judged document batches."""

    output_prefix: str = "pretrain_judge"
    weight_column: str | None = None
    total_rows: int = 0
    successful_rows: int = 0
    failed_rows: int = 0
    total_weight: float = 0.0
    successful_weight: float = 0.0
    failed_weight: float = 0.0
    distributions: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    weighted_distributions: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    invalid_structured_values: Counter[str] = field(default_factory=Counter)
    numeric: dict[str, NumericSummary] = field(default_factory=lambda: defaultdict(NumericSummary))
    quality_bands: Counter[str] = field(default_factory=Counter)
    weighted_quality_bands: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    weighted_quality_total: float = 0.0
    weighted_quality_weight: float = 0.0
    topic_quality_total: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    topic_quality_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    topic_quality_weighted_total: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    topic_weight_total: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def __post_init__(self) -> None:
        if not self.output_prefix.strip():
            msg = "output_prefix must not be empty"
            raise ValueError(msg)
        self.output_prefix = self.output_prefix.strip()

    def required_columns(self) -> list[str]:
        """Columns required from the classified parquet output."""
        columns = [
            *(self._column(suffix) for suffix in _SINGLE_LABEL_SUFFIXES),
            *(self._column(suffix) for suffix in _MULTI_LABEL_SUFFIXES),
            *(self._column(suffix) for suffix in _CONFIDENCE_SUFFIXES),
            *(self._column(suffix) for suffix in _VIEW_NUMERIC_SUFFIXES),
            self._column("quality_score"),
            self._column("quality_scores"),
            self._column("error"),
        ]
        if self.weight_column:
            columns.append(self.weight_column)
        return columns

    def update(self, frame: pd.DataFrame) -> None:
        """Merge one pandas batch into the summary."""
        missing = set(self.required_columns()) - set(frame.columns)
        if missing:
            msg = f"judge output is missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        self.total_rows += len(frame)
        weights = self._weights(frame)
        self.total_weight += float(weights.sum())
        errors = frame[self._column("error")]
        success_mask = errors.isna() | errors.astype("string").fillna("").str.strip().eq("")
        successful = frame.loc[success_mask]
        failed = frame.loc[~success_mask]
        self.successful_rows += len(successful)
        self.failed_rows += len(failed)
        successful_weights = weights.loc[success_mask]
        failed_weights = weights.loc[~success_mask]
        self.successful_weight += float(successful_weights.sum())
        self.failed_weight += float(failed_weights.sum())
        self._update_errors(failed[self._column("error")], failed_weights)

        for suffix in _SINGLE_LABEL_SUFFIXES:
            self._update_single(suffix, successful[self._column(suffix)], successful_weights)
        for suffix in _MULTI_LABEL_SUFFIXES:
            self._update_json_list(suffix, successful[self._column(suffix)], successful_weights)
        for suffix in _CONFIDENCE_SUFFIXES:
            self.numeric[suffix].update(successful[self._column(suffix)])
        for suffix in _VIEW_NUMERIC_SUFFIXES:
            self.numeric[suffix].update(successful[self._column(suffix)])
        self.numeric["quality_score"].update(successful[self._column("quality_score")])
        self._update_weighted_quality(successful[self._column("quality_score")], successful_weights)
        self._update_quality_bands(successful[self._column("quality_score")], successful_weights)
        self._update_quality_dimensions(successful[self._column("quality_scores")])
        self._update_topic_quality(successful, successful_weights)

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable report."""
        return {
            "schema_version": "pretrain_judge_summary_v1",
            "output_prefix": self.output_prefix,
            "rows": {
                "total": self.total_rows,
                "successful": self.successful_rows,
                "failed": self.failed_rows,
                "success_rate": round(self.successful_rows / self.total_rows, 6) if self.total_rows else None,
                "estimated_total": round(self.total_weight, 4) if self.weight_column else None,
                "estimated_successful": round(self.successful_weight, 4) if self.weight_column else None,
                "estimated_failed": round(self.failed_weight, 4) if self.weight_column else None,
            },
            "distributions": {
                name: self._distribution(counter, self.weighted_distributions[name])
                for name, counter in sorted(self.distributions.items())
            },
            "numeric": {name: summary.as_dict() for name, summary in sorted(self.numeric.items())},
            "weighted_quality": {
                "estimated_weight": round(self.weighted_quality_weight, 4),
                "weighted_mean": round(self.weighted_quality_total / self.weighted_quality_weight, 4)
                if self.weighted_quality_weight
                else None,
            }
            if self.weight_column
            else None,
            "quality_bands": self._distribution(self.quality_bands, self.weighted_quality_bands),
            "mean_quality_by_primary_topic": [
                {
                    "label": topic,
                    "count": self.topic_quality_count[topic],
                    "mean": round(self.topic_quality_total[topic] / self.topic_quality_count[topic], 4),
                    "estimated_weight": round(self.topic_weight_total[topic], 4) if self.weight_column else None,
                    "weighted_mean": round(
                        self.topic_quality_weighted_total[topic] / self.topic_weight_total[topic], 4
                    )
                    if self.weight_column and self.topic_weight_total[topic]
                    else None,
                }
                for topic in sorted(self.topic_quality_count)
            ],
            "invalid_structured_values": dict(sorted(self.invalid_structured_values.items())),
        }

    def _column(self, suffix: str) -> str:
        return f"{self.output_prefix}_{suffix}"

    def _weights(self, frame: pd.DataFrame) -> pd.Series:
        if not self.weight_column:
            return pd.Series(1.0, index=frame.index)
        weights = pd.to_numeric(frame[self.weight_column], errors="coerce")
        if weights.isna().any() or not weights.map(math.isfinite).all() or weights.le(0).any():
            msg = f"{self.weight_column} must contain finite positive weights"
            raise ValueError(msg)
        return weights.astype(float)

    def _update_single(self, suffix: str, values: pd.Series, weights: pd.Series) -> None:
        grouped = _group_values(values, weights)
        self.distributions[suffix].update({str(label): int(row["count"]) for label, row in grouped.iterrows()})
        for label, row in grouped.iterrows():
            self.weighted_distributions[suffix][str(label)] += float(row["weight"])

    def _update_json_list(self, suffix: str, values: pd.Series, weights: pd.Series) -> None:
        grouped = _group_values(values, weights)
        for serialized, row in grouped.iterrows():
            count = int(row["count"])
            try:
                parsed = _parse_json_string_list(str(serialized))
            except (json.JSONDecodeError, TypeError):
                self.invalid_structured_values[suffix] += count
                continue
            self.distributions[suffix].update(dict.fromkeys(parsed, count))
            for item in parsed:
                self.weighted_distributions[suffix][item] += float(row["weight"])

    def _update_errors(self, values: pd.Series, weights: pd.Series) -> None:
        kinds = values.dropna().astype(str).str.split(":", n=1).str[0]
        grouped = _group_values(kinds, weights)
        self.distributions["errors"].update({str(label): int(row["count"]) for label, row in grouped.iterrows()})
        for label, row in grouped.iterrows():
            self.weighted_distributions["errors"][str(label)] += float(row["weight"])

    def _update_weighted_quality(self, values: pd.Series, weights: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.map(math.isfinite)
        self.weighted_quality_total += float((numeric[valid] * weights[valid]).sum())
        self.weighted_quality_weight += float(weights[valid].sum())

    def _update_quality_bands(self, values: pd.Series, weights: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce")
        for label, lower, upper, include_upper in _QUALITY_BANDS:
            mask = numeric.ge(lower) & (numeric.le(upper) if include_upper else numeric.lt(upper))
            self.quality_bands[label] += int(mask.sum())
            self.weighted_quality_bands[label] += float(weights[mask].sum())

    def _update_quality_dimensions(self, values: pd.Series) -> None:
        serialized_counts = values.dropna().astype(str).value_counts()
        for serialized, count in serialized_counts.items():
            try:
                parsed = _parse_quality_score_map(serialized)
                for name, value in parsed.items():
                    self.numeric[f"quality_dimension.{name}"].update_repeated(value, int(count))
            except (json.JSONDecodeError, TypeError, ValueError):
                self.invalid_structured_values["quality_scores"] += int(count)

    def _update_topic_quality(self, frame: pd.DataFrame, weights: pd.Series) -> None:
        topic = self._column("primary_topic")
        quality = self._column("quality_score")
        grouped = frame.assign(
            **{
                quality: pd.to_numeric(frame[quality], errors="coerce"),
                "_summary_weight": weights,
            }
        ).dropna(subset=[topic, quality])
        if grouped.empty:
            return
        grouped["_weighted_quality"] = grouped[quality] * grouped["_summary_weight"]
        aggregates = grouped.groupby(topic).agg(
            count=(quality, "count"),
            total=(quality, "sum"),
            weight=("_summary_weight", "sum"),
            weighted_total=("_weighted_quality", "sum"),
        )
        for label, row in aggregates.iterrows():
            self.topic_quality_count[str(label)] += int(row["count"])
            self.topic_quality_total[str(label)] += float(row["total"])
            self.topic_weight_total[str(label)] += float(row["weight"])
            self.topic_quality_weighted_total[str(label)] += float(row["weighted_total"])

    def _distribution(
        self, counter: Counter[str], weighted: dict[str, float]
    ) -> list[dict[str, float | int | str | None]]:
        return [
            {
                "label": label,
                "count": count,
                "share_of_successful_rows": round(count / self.successful_rows, 6) if self.successful_rows else None,
                "estimated_count": round(weighted[label], 4) if self.weight_column else None,
                "estimated_share_of_successful_rows": round(weighted[label] / self.successful_weight, 6)
                if self.weight_column and self.successful_weight
                else None,
            }
            for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        ]


def _group_values(values: pd.Series, weights: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"value": values, "weight": weights}).dropna(subset=["value"])
    if frame.empty:
        return pd.DataFrame(columns=["count", "weight"])
    frame["value"] = frame["value"].astype(str)
    return frame.groupby("value", sort=False).agg(count=("value", "size"), weight=("weight", "sum"))


def _parse_json_string_list(serialized: str) -> list[str]:
    value = json.loads(serialized)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        msg = "expected a JSON list of strings"
        raise TypeError(msg)
    return value


def _parse_quality_score_map(serialized: str) -> dict[str, float]:
    value = json.loads(serialized)
    if not isinstance(value, dict):
        msg = "expected a JSON object of numeric quality scores"
        raise TypeError(msg)
    parsed = {}
    for name, score in value.items():
        if not isinstance(name, str) or isinstance(score, bool) or not isinstance(score, (int, float)):
            msg = "quality score keys must be strings and values must be numeric"
            raise TypeError(msg)
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            msg = "quality score values must be finite"
            raise ValueError(msg)
        parsed[name] = numeric_score
    return parsed
