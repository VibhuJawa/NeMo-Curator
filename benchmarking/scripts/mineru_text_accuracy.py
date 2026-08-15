# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Score extracted text against a URL-aligned standalone MinerU reference."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

import pyarrow.dataset as ds

if TYPE_CHECKING:
    from pathlib import Path

F1_THRESHOLD = 0.95


def token_overlap_f1(output: str, reference: str) -> float:
    """Return bag-of-words F1 for one output/reference pair."""
    left = Counter(re.findall(r"\w+", str(output).lower()))
    right = Counter(re.findall(r"\w+", str(reference).lower()))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    return 2 * overlap / (sum(left.values()) + sum(right.values()))


def evaluate_text_accuracy(
    output_path: Path,
    reference_path: Path,
    url_field: str,
    text_field: str,
) -> dict[str, int | float]:
    """Join duplicate-safe URL occurrences and summarize text overlap."""
    output = ds.dataset(output_path, format="parquet").to_table(columns=[url_field, text_field]).to_pandas()
    reference = (
        ds.dataset(reference_path, format="parquet")
        .to_table(columns=[url_field, text_field], filter=ds.field(url_field).isin(output[url_field].tolist()))
        .to_pandas()
    )
    occurrence = "_mineru_accuracy_occurrence"
    output[occurrence] = output.groupby(url_field).cumcount()
    reference[occurrence] = reference.groupby(url_field).cumcount()
    merged = output.merge(
        reference,
        on=[url_field, occurrence],
        suffixes=("_output", "_reference"),
        validate="one_to_one",
    )
    scores = [
        token_overlap_f1(left, right)
        for left, right in zip(
            merged[f"{text_field}_output"],
            merged[f"{text_field}_reference"],
            strict=True,
        )
    ]
    return {
        "scored_rows": len(output),
        "reference_rows_matched": len(merged),
        "unmatched_output_rows": len(output) - len(merged),
        "mean_token_overlap_f1": sum(scores) / len(scores) if scores else 0.0,
        "fraction_f1_ge_095": (sum(score >= F1_THRESHOLD for score in scores) / len(scores) if scores else 0.0),
    }
