# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Validate and analyze the NMCUR-357 jusText/MinerU boilerplate cohort."""

# ruff: noqa: EM101, EM102

from __future__ import annotations

import argparse
import filecmp
import hashlib
import html
import json
import os
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import zstandard as zstd

NON_SPACED_LANGUAGES = ("CHINESE", "CHINESET", "JAPANESE", "KOREAN", "THAI")
LANGUAGES = (*NON_SPACED_LANGUAGES, "ENGLISH")
EXPECTED_LANGUAGE_COUNTS = {
    "CHINESE": 408_221,
    "CHINESET": 129_780,
    "JAPANESE": 508_462,
    "KOREAN": 77_511,
    "THAI": 47_348,
    "ENGLISH": 100_000,
}
EXPECTED_TOTAL = 1_271_322
MIN_STATUS_OK_RATE = 0.95
MAX_CONVERT_ERROR_RATE = 0.02
REQUIRED_INPUT_FIELDS = {
    "_html_zstd",
    "_html_uncompressed_bytes",
    "url",
    "justext_extracted_text",
    "html_cld2_lang",
    "source_file",
    "source_row_index",
    "document_id",
}
REQUIRED_OUTPUT_FIELDS = {
    "document_id",
    "url",
    "justext_extracted_text",
    "html_cld2_lang",
    "mineru_main_text",
    "mineru_other_text",
    "mineru_labels",
    "_mineru_status",
}
METRIC_PAIRS = {
    "justext_boilerplate_share": ("justext_other_occurrences", "justext_shingle_occurrences"),
    "justext_main_share": ("justext_main_occurrences", "justext_shingle_occurrences"),
    "justext_ambiguous_share": ("justext_ambiguous_occurrences", "justext_shingle_occurrences"),
    "justext_unmatched_share": ("justext_unmatched_occurrences", "justext_shingle_occurrences"),
    "mineru_other_retention_by_justext": (
        "mineru_other_retained_occurrences",
        "mineru_other_shingle_occurrences",
    ),
    "excess_justext_length_explained_by_other": (
        "excess_justext_explained_occurrences",
        "excess_justext_occurrences",
    ),
}
PER_DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("html_cld2_lang", pa.string(), nullable=False),
        pa.field("_mineru_status", pa.string(), nullable=False),
        *[
            pa.field(name, pa.int64())
            for name in (
                "justext_normalized_codepoints",
                "mineru_main_normalized_codepoints",
                "mineru_other_normalized_codepoints",
                "justext_shingle_occurrences",
                "mineru_main_shingle_occurrences",
                "mineru_other_shingle_occurrences",
                "justext_main_occurrences",
                "justext_other_occurrences",
                "justext_ambiguous_occurrences",
                "justext_unmatched_occurrences",
                "mineru_other_retained_occurrences",
                "excess_justext_occurrences",
                "excess_justext_explained_occurrences",
            )
        ],
        *[pa.field(name, pa.float64()) for name in METRIC_PAIRS],
    ]
)


def normalize_text(value: object) -> str:
    """Entity-decode, NFKC-normalize, case-fold, and remove Unicode whitespace."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    normalized = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    return "".join(character for character in normalized if not character.isspace())


def shingle_counts(text: str, width: int = 8) -> Counter[str]:
    """Count overlapping Unicode shingles, treating short nonempty text as one unit."""
    if not text:
        return Counter()
    if len(text) < width:
        return Counter({text: 1})
    return Counter(text[index : index + width] for index in range(len(text) - width + 1))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def document_metrics(row: dict[str, object], shingle_width: int = 8) -> dict[str, object]:
    """Attribute every jusText shingle occurrence to main/other/both/neither."""
    base = {
        "document_id": str(row["document_id"]),
        "url": "" if row.get("url") is None else str(row.get("url")),
        "html_cld2_lang": str(row["html_cld2_lang"]),
        "_mineru_status": str(row["_mineru_status"]),
    }
    if base["_mineru_status"] != "ok":
        return base | dict.fromkeys(
            (
                "justext_normalized_codepoints",
                "mineru_main_normalized_codepoints",
                "mineru_other_normalized_codepoints",
                "justext_shingle_occurrences",
                "mineru_main_shingle_occurrences",
                "mineru_other_shingle_occurrences",
                "justext_main_occurrences",
                "justext_other_occurrences",
                "justext_ambiguous_occurrences",
                "justext_unmatched_occurrences",
                "mineru_other_retained_occurrences",
                "excess_justext_occurrences",
                "excess_justext_explained_occurrences",
                *METRIC_PAIRS,
            )
        )

    justext = normalize_text(row.get("justext_extracted_text"))
    main = normalize_text(row.get("mineru_main_text"))
    other = normalize_text(row.get("mineru_other_text"))
    justext_units = shingle_counts(justext, shingle_width)
    main_units = shingle_counts(main, shingle_width)
    other_units = shingle_counts(other, shingle_width)
    main_keys = set(main_units)
    other_keys = set(other_units)
    justext_keys = set(justext_units)

    attributed = Counter({"main": 0, "other": 0, "ambiguous": 0, "unmatched": 0})
    for unit, occurrences in justext_units.items():
        in_main = unit in main_keys
        in_other = unit in other_keys
        label = "ambiguous" if in_main and in_other else "main" if in_main else "other" if in_other else "unmatched"
        attributed[label] += occurrences

    justext_total = sum(justext_units.values())
    main_total = sum(main_units.values())
    other_total = sum(other_units.values())
    other_retained = sum(occurrences for unit, occurrences in other_units.items() if unit in justext_keys)
    excess = max(justext_total - main_total, 0)
    excess_explained = min(attributed["other"], excess)
    result: dict[str, object] = base | {
        "justext_normalized_codepoints": len(justext),
        "mineru_main_normalized_codepoints": len(main),
        "mineru_other_normalized_codepoints": len(other),
        "justext_shingle_occurrences": justext_total,
        "mineru_main_shingle_occurrences": main_total,
        "mineru_other_shingle_occurrences": other_total,
        "justext_main_occurrences": attributed["main"],
        "justext_other_occurrences": attributed["other"],
        "justext_ambiguous_occurrences": attributed["ambiguous"],
        "justext_unmatched_occurrences": attributed["unmatched"],
        "mineru_other_retained_occurrences": other_retained,
        "excess_justext_occurrences": excess,
        "excess_justext_explained_occurrences": excess_explained,
    }
    for metric, (numerator, denominator) in METRIC_PAIRS.items():
        result[metric] = _ratio(int(result[numerator]), int(result[denominator]))
    return result


def _dataset(path: Path) -> pads.Dataset:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No Parquet files found at {path}")
    return pads.dataset([str(file) for file in files], format="parquet")


def write_per_document_metrics(output_path: Path, destination: Path, batch_size: int = 1024) -> int:
    """Stream MinerU output into an atomically published per-document Parquet table."""
    dataset = _dataset(output_path)
    missing = REQUIRED_OUTPUT_FIELDS - set(dataset.schema.names)
    if missing:
        raise ValueError(f"MinerU output is missing fields: {sorted(missing)}")
    columns = sorted(REQUIRED_OUTPUT_FIELDS - {"mineru_labels"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
            records = [document_metrics(row) for row in batch.to_pylist()]
            table = pa.Table.from_pylist(records, schema=PER_DOCUMENT_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            rows_written += len(records)
        if writer is None:
            raise ValueError("MinerU output contained no rows")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    return rows_written


def validated_per_document_rows(path: Path) -> int:
    """Validate a durable per-document artifact before reusing it on retry."""
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(PER_DOCUMENT_SCHEMA, check_metadata=False):
        raise ValueError(f"Per-document metric schema mismatch at {path}")
    if parquet.metadata.num_rows <= 0:
        raise ValueError(f"Per-document metric artifact is empty at {path}")
    return parquet.metadata.num_rows


def poisson_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 4096,
) -> dict[str, dict[str, float | None]]:
    """Compute document-cluster Poisson bootstrap intervals in bounded memory."""
    if frame.empty or replicates <= 0:
        return {metric: {"low": None, "high": None} for metric in METRIC_PAIRS}
    ordered_columns = [column for pair in METRIC_PAIRS.values() for column in pair]
    values = frame[ordered_columns].fillna(0).to_numpy(dtype=np.float64)
    totals = np.zeros((replicates, len(ordered_columns)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, len(values), chunk_size):
        chunk = values[start : start + chunk_size]
        weights = rng.poisson(1.0, size=(replicates, len(chunk))).astype(np.float64)
        totals += weights @ chunk
    intervals: dict[str, dict[str, float | None]] = {}
    for index, metric in enumerate(METRIC_PAIRS):
        numerator = totals[:, 2 * index]
        denominator = totals[:, 2 * index + 1]
        ratios = np.divide(numerator, denominator, out=np.full(replicates, np.nan), where=denominator > 0)
        finite = ratios[np.isfinite(ratios)]
        intervals[metric] = (
            {"low": float(np.quantile(finite, 0.025)), "high": float(np.quantile(finite, 0.975))}
            if len(finite)
            else {"low": None, "high": None}
        )
    return intervals


def summarize_group(frame: pd.DataFrame, *, bootstrap_replicates: int, seed: int) -> dict[str, object]:
    primary = frame[frame["_mineru_status"] == "ok"]
    source_justext_available = int(primary["justext_shingle_occurrences"].sum()) > 0
    status_counts = {str(key): int(value) for key, value in frame["_mineru_status"].value_counts().items()}
    result: dict[str, object] = {
        "documents": len(frame),
        "primary_documents": len(primary),
        "status_counts": status_counts,
        "status_ok_rate": len(primary) / len(frame) if len(frame) else None,
        "convert_error_rate": status_counts.get("convert_error", 0) / len(frame) if len(frame) else None,
        "metrics": {},
    }
    intervals = poisson_bootstrap_intervals(
        primary,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    for metric, (numerator, denominator) in METRIC_PAIRS.items():
        denominator_total = int(primary[denominator].sum())
        micro = _ratio(int(primary[numerator].sum()), denominator_total) if source_justext_available else None
        values = primary[metric].dropna() if source_justext_available else pd.Series(dtype=np.float64)
        result["metrics"][metric] = {
            "micro": micro,
            "bootstrap_95": intervals[metric] if source_justext_available else {"low": None, "high": None},
            "document_p10": float(values.quantile(0.10)) if len(values) else None,
            "document_median": float(values.quantile(0.50)) if len(values) else None,
            "document_p90": float(values.quantile(0.90)) if len(values) else None,
            "numerator": int(primary[numerator].sum()),
            "denominator": denominator_total,
        }
    return result


def population_weighted_language_metric(
    groups: dict[str, object],
    metric: str,
) -> tuple[float | None, float | None]:
    """Weight language micros over strata with source jusText availability."""
    total_primary = sum(groups[language]["primary_documents"] for language in NON_SPACED_LANGUAGES)
    available = [
        language
        for language in NON_SPACED_LANGUAGES
        if groups[language]["metrics"]["justext_boilerplate_share"]["denominator"] > 0
        and groups[language]["metrics"][metric]["micro"] is not None
    ]
    available_primary = sum(groups[language]["primary_documents"] for language in available)
    if not available_primary:
        return None, 0.0 if total_primary else None
    weighted = sum(
        groups[language]["metrics"][metric]["micro"] * groups[language]["primary_documents"]
        for language in available
    )
    return weighted / available_primary, available_primary / total_primary if total_primary else None


def summarize_metrics(per_document_path: Path, *, bootstrap_replicates: int, seed: int) -> dict[str, object]:
    frame = pd.read_parquet(per_document_path)
    groups: dict[str, object] = {}
    for language in LANGUAGES:
        language_seed = seed + int.from_bytes(hashlib.sha256(language.encode()).digest()[:4], "big")
        groups[language] = summarize_group(
            frame[frame["html_cld2_lang"] == language],
            bootstrap_replicates=bootstrap_replicates,
            seed=language_seed,
        )
    groups["NON_SPACED"] = summarize_group(
        frame[frame["html_cld2_lang"].isin(NON_SPACED_LANGUAGES)],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    for metric in METRIC_PAIRS:
        weighted, coverage = population_weighted_language_metric(groups, metric)
        groups["NON_SPACED"]["metrics"][metric]["population_weighted_language_micro"] = weighted
        groups["NON_SPACED"]["metrics"][metric]["population_weighted_language_coverage"] = coverage
    groups["ALL"] = summarize_group(frame, bootstrap_replicates=bootstrap_replicates, seed=seed + 1)
    return {
        "method": {
            "normalization": "HTML entity decode, Unicode NFKC, casefold, remove Unicode whitespace",
            "shingle_width_codepoints": 8,
            "bootstrap": "document-cluster Poisson(1) bootstrap",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "primary_population": '_mineru_status == "ok"',
        },
        "groups": groups,
    }


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_summary_csv(summary: dict[str, object], path: Path) -> None:
    rows = []
    for group, group_summary in summary["groups"].items():
        for metric, values in group_summary["metrics"].items():
            rows.append(
                {
                    "group": group,
                    "documents": group_summary["documents"],
                    "primary_documents": group_summary["primary_documents"],
                    "status_ok_rate": group_summary["status_ok_rate"],
                    "convert_error_rate": group_summary["convert_error_rate"],
                    "metric": metric,
                    "micro": values["micro"],
                    "population_weighted_language_micro": values.get("population_weighted_language_micro"),
                    "population_weighted_language_coverage": values.get("population_weighted_language_coverage"),
                    "bootstrap_95_low": values["bootstrap_95"]["low"],
                    "bootstrap_95_high": values["bootstrap_95"]["high"],
                    "document_p10": values["document_p10"],
                    "document_median": values["document_median"],
                    "document_p90": values["document_p90"],
                    "numerator": values["numerator"],
                    "denominator": values["denominator"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _review_selection(
    frame: pd.DataFrame,
    per_decile: int = 5,
) -> dict[str, tuple[str, int | None, float | None]]:
    selected: dict[str, tuple[str, int | None, float | None]] = {}
    primary = frame[frame["_mineru_status"] == "ok"]
    for language in LANGUAGES:
        language_primary = primary[primary["html_cld2_lang"] == language]
        language_frame = language_primary[language_primary["justext_boilerplate_share"].notna()].sort_values(
            ["justext_boilerplate_share", "document_id"], kind="stable"
        )
        if language_frame.empty:
            ranked = sorted(
                language_primary.itertuples(index=False),
                key=lambda row: hashlib.sha256(str(row.document_id).encode()).digest(),
            )[: per_decile * 10]
            for row in ranked:
                selected[str(row.document_id)] = (language, None, None)
            continue
        for decile, positions in enumerate(np.array_split(np.arange(len(language_frame)), 10)):
            candidates = language_frame.iloc[positions]
            ranked = sorted(
                candidates.itertuples(index=False),
                key=lambda row: hashlib.sha256(str(row.document_id).encode()).digest(),
            )[:per_decile]
            for row in ranked:
                selected[str(row.document_id)] = (language, decile, float(row.justext_boilerplate_share))
    return selected


def write_review_examples(output_path: Path, per_document_path: Path, destination: Path) -> int:
    frame = pd.read_parquet(
        per_document_path,
        columns=["document_id", "html_cld2_lang", "_mineru_status", "justext_boilerplate_share"],
    )
    selected = _review_selection(frame)
    examples = []
    columns = [
        "document_id",
        "url",
        "justext_extracted_text",
        "mineru_main_text",
        "mineru_other_text",
    ]
    for batch in _dataset(output_path).to_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            document_id = str(row["document_id"])
            if document_id not in selected:
                continue
            language, decile, ratio = selected[document_id]
            examples.append(
                {
                    "language": language,
                    "ratio_decile": decile,
                    "document_id": document_id,
                    "url": row.get("url"),
                    "justext_boilerplate_share": ratio,
                    "justext_excerpt": str(row.get("justext_extracted_text") or "")[:500],
                    "mineru_main_excerpt": str(row.get("mineru_main_text") or "")[:500],
                    "mineru_other_excerpt": str(row.get("mineru_other_text") or "")[:500],
                }
            )
    examples.sort(key=lambda row: (row["language"], row["ratio_decile"], row["document_id"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)
    return len(examples)


def checkpoint_completed_sources(checkpoint_path: Path) -> set[str]:
    """Read the union of durable completed-source IDs through stable local copies."""
    completed: set[str] = set()
    metadata = checkpoint_path / ".nemo_curator_metadata"
    with tempfile.TemporaryDirectory(prefix="nmcur357-checkpoint-") as temporary_dir:
        for index, source in enumerate(sorted(metadata.rglob("*.mdb"))):
            local = Path(temporary_dir, f"{index}.mdb")
            verify = Path(temporary_dir, f"{index}.verify.mdb")
            shutil.copyfile(source, local)
            shutil.copyfile(source, verify)
            if not filecmp.cmp(local, verify, shallow=False):
                raise ValueError(f"Checkpoint changed during validation: {source}")
            environment = lmdb.open(str(local), subdir=False, readonly=True, lock=False, max_dbs=1)
            try:
                database = environment.open_db(b"completed_sources")
                with environment.begin() as transaction, transaction.cursor(db=database) as cursor:
                    completed.update(key.decode() for key, _ in cursor)
            except lmdb.Error:
                pass
            finally:
                environment.close()
    return completed


def _scan_ids_and_counts(
    dataset: pads.Dataset, *, include_status: bool
) -> tuple[set[str], Counter[str], Counter[str]]:
    columns = ["document_id", "html_cld2_lang"] + (["_mineru_status"] if include_status else [])
    identifiers: set[str] = set()
    language_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    rows = 0
    for batch in dataset.to_batches(columns=columns, batch_size=65_536):
        for row in batch.to_pylist():
            rows += 1
            identifier = str(row["document_id"])
            if identifier in identifiers:
                raise ValueError(f"Duplicate document_id: {identifier}")
            identifiers.add(identifier)
            language_counts[str(row["html_cld2_lang"])] += 1
            if include_status:
                status_counts[str(row["_mineru_status"])] += 1
    if rows != len(identifiers):
        raise AssertionError("internal document-ID accounting mismatch")
    return identifiers, language_counts, status_counts


def validate_zstandard_sample(input_dataset: pads.Dataset, sample_size: int = 128) -> int:
    validated = 0
    decompressor = zstd.ZstdDecompressor()
    columns = ["document_id", "_html_zstd", "_html_uncompressed_bytes"]
    for batch in input_dataset.to_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            if int.from_bytes(hashlib.sha256(str(row["document_id"]).encode()).digest()[:4], "big") % 1009:
                continue
            expected = int(row["_html_uncompressed_bytes"])
            raw = decompressor.decompress(row["_html_zstd"], max_output_size=expected)
            if len(raw) != expected:
                raise ValueError(f"Zstandard size mismatch for {row['document_id']}: {len(raw)} != {expected}")
            validated += 1
            if validated == sample_size:
                return validated
    if validated < sample_size:
        raise ValueError(f"Only found {validated} deterministic Zstandard samples; expected {sample_size}")
    return validated


def validate_run(  # noqa: C901
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    input_dataset = _dataset(input_path)
    output_dataset = _dataset(output_path)
    input_missing = REQUIRED_INPUT_FIELDS - set(input_dataset.schema.names)
    output_missing = REQUIRED_OUTPUT_FIELDS - set(output_dataset.schema.names)
    if input_missing or output_missing:
        raise ValueError(
            f"Schema mismatch: input_missing={sorted(input_missing)}, output_missing={sorted(output_missing)}"
        )

    input_ids, input_counts, _ = _scan_ids_and_counts(input_dataset, include_status=False)
    output_ids, output_counts, statuses = _scan_ids_and_counts(output_dataset, include_status=True)
    if dict(input_counts) != EXPECTED_LANGUAGE_COUNTS:
        raise ValueError(f"Unexpected staged language counts: {dict(input_counts)}")
    if sum(input_counts.values()) != EXPECTED_TOTAL:
        raise ValueError(f"Unexpected staged total: {sum(input_counts.values())} != {EXPECTED_TOTAL}")
    if output_counts != input_counts:
        raise ValueError(f"Input/output language counts differ: {dict(input_counts)} != {dict(output_counts)}")
    if output_ids != input_ids:
        raise ValueError(
            f"Input/output document IDs differ: missing={len(input_ids - output_ids)}, extra={len(output_ids - input_ids)}"
        )

    total = len(output_ids)
    ok_rate = statuses.get("ok", 0) / total
    convert_error_rate = statuses.get("convert_error", 0) / total
    if ok_rate < MIN_STATUS_OK_RATE:
        raise ValueError(f"_mineru_status ok rate below threshold: {ok_rate:.6f}")
    if convert_error_rate > MAX_CONVERT_ERROR_RATE:
        raise ValueError(f"convert_error rate above threshold: {convert_error_rate:.6f}")
    temporary_files = sorted(str(path) for path in output_path.rglob("*.tmp"))
    if temporary_files:
        raise ValueError(f"Unpublished temporary output files remain: {temporary_files[:5]}")
    failed_markers = sorted(str(path) for path in checkpoint_path.rglob("failed_tasks.json"))
    if failed_markers:
        raise ValueError(f"Failed-task markers remain: {failed_markers[:5]}")
    input_files = len(list(input_path.glob("*.parquet")))
    completed_sources = checkpoint_completed_sources(checkpoint_path)
    if len(completed_sources) != input_files:
        raise ValueError(
            f"Checkpoint completion mismatch: {len(completed_sources)} completed != {input_files} sources"
        )
    sampled_zstd = validate_zstandard_sample(input_dataset)
    return {
        "documents": total,
        "language_counts": dict(input_counts),
        "status_counts": dict(statuses),
        "status_ok_rate": ok_rate,
        "convert_error_rate": convert_error_rate,
        "input_parquet_files": input_files,
        "completed_checkpoint_sources": len(completed_sources),
        "zstandard_cells_validated": sampled_zstd,
        "document_id_sets_identical": True,
        "all_parquet_footers_readable": True,
        "failed_task_markers": 0,
        "pending_checkpoint_sources": 0,
    }


def compute_gpt_neo_tokens(output_path: Path, tokenizer_name: str) -> dict[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    fields = ("justext_extracted_text", "mineru_main_text", "mineru_other_text")
    totals = dict.fromkeys(fields, 0)
    for batch in _dataset(output_path).to_batches(columns=list(fields), batch_size=512):
        rows = batch.to_pylist()
        for field in fields:
            texts = [str(row.get(field) or "") for row in rows]
            encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
            totals[field] += sum(map(len, encoded))
    return {f"{field}_gpt_neo_tokens": total for field, total in totals.items()}


def render_report(summary: dict[str, object], validation: dict[str, object], metadata: dict[str, object]) -> str:
    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.2%}"

    lines = [
        "# NMCUR-357: jusText CJK Boilerplate Study",
        "",
        "## Run provenance",
        "",
        f"- Curator commit/version: {metadata.get('curator_version', 'RECORD BEFORE FINAL RUN')}",
        f"- Allocation/job IDs: {metadata.get('slurm_job_ids', 'RECORD BEFORE FINAL RUN')}",
        f"- MinerU command: `{metadata.get('command', 'RECORD BEFORE FINAL RUN')}`",
        f"- Analysis command: `{metadata.get('analysis_command', 'RECORD BEFORE FINAL RUN')}`",
        f"- Configuration: `{json.dumps(metadata.get('configuration', {}), sort_keys=True)}`",
        f"- Performance trials: `{json.dumps(metadata.get('performance_trials', []), sort_keys=True)}`",
        f"- Performance trial spread: `{json.dumps(metadata.get('performance_trial_spread', {}), sort_keys=True)}`",
        "",
        "## Validation",
        "",
        f"- Documents: {validation['documents']:,}",
        f"- `_mineru_status == ok`: {validation['status_ok_rate']:.4%}",
        f"- Conversion errors: {validation['convert_error_rate']:.4%}",
        f"- Deterministic Zstandard cells checked: {validation['zstandard_cells_validated']}",
        f"- Completed/pending checkpoint sources: {validation['completed_checkpoint_sources']}/0",
        "- Input/output IDs: unique and identical",
        "- All Parquet footers: readable",
        "",
        "### Status by group",
        "",
        "| Group | Documents | Primary `ok` | `ok` rate | Conversion-error rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in (*LANGUAGES, "NON_SPACED", "ALL"):
        group_summary = summary["groups"][group]
        lines.append(
            f"| {group} | {group_summary['documents']:,} | {group_summary['primary_documents']:,} | "
            f"{group_summary['status_ok_rate']:.2%} | {group_summary['convert_error_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Primary results",
            "",
            "Only `_mineru_status == ok` documents contribute to the ratios below. Non-spaced headline values are "
            "population-weighted language micros over strata with available source jusText.",
            "",
            "| Group | jusText boilerplate | jusText main | Ambiguous | Unmatched | MinerU-other retained | Excess explained |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in (*LANGUAGES, "NON_SPACED", "ALL"):
        metrics = summary["groups"][group]["metrics"]
        values = [
            metrics[name].get("population_weighted_language_micro", metrics[name]["micro"]) for name in METRIC_PAIRS
        ]
        lines.append(f"| {group} | " + " | ".join(percent(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "### Detailed occurrence and document distributions",
            "",
            "The micro value is occurrence-weighted. Document columns summarize per-document ratios; the "
            "1,000-replicate document-cluster Poisson bootstrap intervals use seed 357 and apply to the occurrence micro.",
            "",
            "| Group | Metric | Micro | Bootstrap 95% | p10 | Median | p90 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in (*LANGUAGES, "NON_SPACED", "ALL"):
        for metric in METRIC_PAIRS:
            values = summary["groups"][group]["metrics"][metric]
            interval = values["bootstrap_95"]
            rendered_interval = (
                "n/a"
                if interval["low"] is None
                else f"{interval['low']:.2%}-{interval['high']:.2%}"
            )
            lines.append(
                f"| {group} | {metric} | {percent(values['micro'])} | {rendered_interval} | "
                f"{percent(values['document_p10'])} | {percent(values['document_median'])} | "
                f"{percent(values['document_p90'])} |"
            )
    lines.extend(
        [
            "",
            "## Secondary dashboard comparison",
            "",
            (
                f"GPT-Neo token totals: `{json.dumps(summary['gpt_neo_tokens'], sort_keys=True)}`"
                if "gpt_neo_tokens" in summary
                else "GPT-Neo token totals were not computed because the exact pinned tokenizer ID remains unresolved."
            ),
            "",
            "## Manual review",
            "",
            "The non-spaced population-weighted language metrics renormalize across strata with available source "
            "jusText. Coverage among primary non-spaced documents is "
            f"{summary['groups']['NON_SPACED']['metrics']['justext_boilerplate_share'].get('population_weighted_language_coverage', 0):.2%}.",
            "",
            "Fifty deterministic examples per language are written to `review_examples.jsonl` for human review. "
            "Languages with available jusText ratios contribute five examples per ratio decile. If a language has "
            "no source jusText text, its examples are deterministic availability-review samples with null ratio and "
            "decile fields.",
            "",
            "## Measurements, estimates, and uncertainty",
            "",
            "The cohort counts, status rates, overlap ratios, bootstrap intervals, and trial rates are measurements. "
            "Any full-run duration projected from the canary is an estimate recorded in run metadata.",
            "",
            "Remaining uncertainties:",
            "",
            "- Traditional Chinese source jusText is entirely null, so its overlap metrics are unavailable.",
            "- Review whether ambiguous shingles should be apportioned instead of reported separately.",
            "- MinerU labels are a model-based reference, not human boilerplate ground truth.",
            "",
            "## Artifacts",
            "",
            "- `per_document_metrics.parquet`",
            "- `summary.json` and `summary.csv`",
            "- `review_examples.jsonl`",
            "- final `_SUCCESS.json` validation receipt",
            "",
            f"Anything still running or unresolved: {metadata.get('still_running_or_unresolved', 'RECORD BEFORE FINAL RUN')}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--analysis-path", type=Path, required=True)
    parser.add_argument("--success-path", type=Path, required=True)
    parser.add_argument("--run-metadata-json", type=Path)
    parser.add_argument("--gpt-neo-tokenizer")
    parser.add_argument(
        "--reuse-per-document-metrics",
        action="store_true",
        help="Reuse an existing durable per-document Parquet after validating its schema and footer.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=357)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.success_path.unlink(missing_ok=True)
    args.analysis_path.mkdir(parents=True, exist_ok=True)
    per_document = args.analysis_path / "per_document_metrics.parquet"
    rows = (
        validated_per_document_rows(per_document)
        if args.reuse_per_document_metrics
        else write_per_document_metrics(args.output_path, per_document)
    )
    summary = summarize_metrics(
        per_document,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    if args.gpt_neo_tokenizer:
        summary["gpt_neo_tokens"] = compute_gpt_neo_tokens(args.output_path, args.gpt_neo_tokenizer)
    examples = write_review_examples(args.output_path, per_document, args.analysis_path / "review_examples.jsonl")
    validation = validate_run(args.input_path, args.output_path, args.checkpoint_path)
    if rows != validation["documents"]:
        raise ValueError(f"Per-document metric row count differs: {rows} != {validation['documents']}")
    if examples != 50 * len(LANGUAGES):
        raise ValueError(f"Expected 300 review examples, wrote {examples}")
    summary["validation"] = validation
    _atomic_json(summary, args.analysis_path / "summary.json")
    write_summary_csv(summary, args.analysis_path / "summary.csv")
    metadata = json.loads(args.run_metadata_json.read_text()) if args.run_metadata_json else {}
    report = render_report(summary, validation, metadata)
    report_path = args.analysis_path / "NMCUR-357-report.md"
    temporary_report = report_path.with_name(f".{report_path.name}.tmp")
    temporary_report.write_text(report)
    os.replace(temporary_report, report_path)
    _atomic_json(
        {
            "status": "complete",
            "validation": validation,
            "artifacts": sorted(path.name for path in args.analysis_path.iterdir()),
        },
        args.success_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
