# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Materialize and validate the NMCUR-357 language cohort.

The source dataset already contains independently Zstandard-compressed HTML,
jusText output, and CLD2 labels.  This utility projects only the columns needed
for the boilerplate study, keeps every Curator non-spaced-language row, and
selects an exact deterministic English control by immutable source position.

Selection is deliberately two-pass.  The first pass reads only the language
column and finds the globally smallest SHA-256 document hashes.  The second
pass reads and writes one source Parquet at a time, so a source file is the
stable and independently recoverable work unit.
"""

# ruff: noqa: EM101, EM102

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LANGUAGE_FIELD = "html_cld2_lang"
HTML_FIELD = "_html_zstd"
HTML_BYTES_FIELD = "_html_uncompressed_bytes"
JUSTEXT_FIELD = "justext_extracted_text"
URL_FIELD = "url"
SOURCE_FILE_FIELD = "source_file"
SOURCE_ROW_FIELD = "source_row_index"
DOCUMENT_ID_FIELD = "document_id"

NON_SPACED_LANGUAGES = ("CHINESE", "CHINESET", "JAPANESE", "KOREAN", "THAI")
ENGLISH = "ENGLISH"
OUTPUT_FIELDS = (
    HTML_FIELD,
    HTML_BYTES_FIELD,
    URL_FIELD,
    JUSTEXT_FIELD,
    LANGUAGE_FIELD,
    SOURCE_FILE_FIELD,
    SOURCE_ROW_FIELD,
    DOCUMENT_ID_FIELD,
)
SOURCE_FIELDS = (HTML_FIELD, HTML_BYTES_FIELD, URL_FIELD, JUSTEXT_FIELD, LANGUAGE_FIELD)


def document_id(source_file: str, source_row_index: int) -> str:
    """Return the immutable identity used for sampling and downstream joins."""
    return f"{source_file}:{source_row_index}"


def document_hash(doc_id: str) -> int:
    """Return a stable unsigned SHA-256 value for deterministic top-k sampling."""
    return int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest(), "big")


class SmallestHashSample:
    """Streaming exact sample containing the ``size`` smallest document hashes."""

    def __init__(self, size: int):
        if size < 0:
            raise ValueError("sample size must be non-negative")
        self.size = size
        self._heap: list[tuple[int, str]] = []

    def add(self, doc_id: str) -> None:
        if self.size == 0:
            return
        entry = (-document_hash(doc_id), doc_id)
        if len(self._heap) < self.size:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def values(self) -> set[str]:
        return {doc_id for _, doc_id in self._heap}


@dataclass(frozen=True)
class Selection:
    english_ids: frozenset[str]
    canary_ids: frozenset[str]
    source_language_counts: dict[str, int]
    source_rows: int


def discover_source_files(input_root: Path) -> list[Path]:
    files = sorted(input_root.glob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet files found directly under {input_root}")
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise ValueError("Source filenames must be unique")
    return files


def plan_selection(
    source_files: Iterable[Path],
    *,
    english_size: int,
    canary_per_language: int,
) -> Selection:
    """Scan language columns and choose exact English and canary document IDs."""
    english = SmallestHashSample(english_size)
    canary = {language: SmallestHashSample(canary_per_language) for language in (*NON_SPACED_LANGUAGES, ENGLISH)}
    counts: Counter[str] = Counter()
    source_rows = 0

    for path in source_files:
        language_values = pq.read_table(path, columns=[LANGUAGE_FIELD]).column(0).to_pylist()
        source_rows += len(language_values)
        for row_index, language in enumerate(language_values):
            if language is not None:
                counts[language] += 1
            if language not in canary:
                continue
            doc_id = document_id(path.name, row_index)
            canary[language].add(doc_id)
            if language == ENGLISH:
                english.add(doc_id)

    if len(english.values()) != english_size:
        raise ValueError(f"Requested {english_size} English rows, found only {counts[ENGLISH]}")
    undersized = {language: counts[language] for language in canary if counts[language] < canary_per_language}
    if undersized:
        raise ValueError(f"Canary request exceeds available language rows: {undersized}")

    canary_ids = frozenset().union(*(sample.values() for sample in canary.values()))
    return Selection(
        english_ids=frozenset(english.values()),
        canary_ids=canary_ids,
        source_language_counts=dict(sorted(counts.items())),
        source_rows=source_rows,
    )


def _validate_existing_output(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        parquet = pq.ParquetFile(path)
    except Exception:
        return False
    return parquet.metadata.num_rows == expected_rows and parquet.schema_arrow.names == list(OUTPUT_FIELDS)


def _atomic_write_parquet(table: pa.Table, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _project_rows(table: pa.Table, source_file: str, admitted_ids: set[str] | frozenset[str]) -> pa.Table:
    languages = table[LANGUAGE_FIELD].to_pylist()
    row_indices: list[int] = []
    doc_ids: list[str] = []
    for row_index, language in enumerate(languages):
        doc_id = document_id(source_file, row_index)
        if language in NON_SPACED_LANGUAGES or (language == ENGLISH and doc_id in admitted_ids):
            row_indices.append(row_index)
            doc_ids.append(doc_id)

    selected = table.take(pa.array(row_indices, type=pa.int64()))
    selected = selected.append_column(SOURCE_FILE_FIELD, pa.array([source_file] * len(row_indices), type=pa.string()))
    selected = selected.append_column(SOURCE_ROW_FIELD, pa.array(row_indices, type=pa.int64()))
    selected = selected.append_column(DOCUMENT_ID_FIELD, pa.array(doc_ids, type=pa.string()))
    return selected.select(OUTPUT_FIELDS)


def materialize_selection(
    source_files: Iterable[Path],
    *,
    output_root: Path,
    canary_root: Path,
    selection: Selection,
    resume: bool,
) -> tuple[dict, dict]:
    """Write full and canary projections and return their measured summaries."""
    summaries = {
        "full": {"rows": 0, "uncompressed_html_bytes": 0, "language_counts": Counter(), "files": 0},
        "canary": {"rows": 0, "uncompressed_html_bytes": 0, "language_counts": Counter(), "files": 0},
    }
    english_by_file: dict[str, set[str]] = defaultdict(set)
    canary_by_file: dict[str, set[str]] = defaultdict(set)
    for doc_id in selection.english_ids:
        english_by_file[doc_id.rsplit(":", 1)[0]].add(doc_id)
    for doc_id in selection.canary_ids:
        canary_by_file[doc_id.rsplit(":", 1)[0]].add(doc_id)

    for source_path in source_files:
        source = pq.read_table(source_path, columns=list(SOURCE_FIELDS))
        full = _project_rows(source, source_path.name, english_by_file[source_path.name])
        canary_mask = pc.is_in(full[DOCUMENT_ID_FIELD], value_set=pa.array(sorted(canary_by_file[source_path.name])))
        canary = full.filter(canary_mask)

        for name, table, root in (("full", full, output_root), ("canary", canary, canary_root)):
            output_path = root / source_path.name
            if len(table) and not (resume and _validate_existing_output(output_path, len(table))):
                _atomic_write_parquet(table, output_path)
            if not len(table):
                continue
            summaries[name]["files"] += 1
            summaries[name]["rows"] += len(table)
            summaries[name]["uncompressed_html_bytes"] += int(pc.sum(table[HTML_BYTES_FIELD]).as_py() or 0)
            summaries[name]["language_counts"].update(table[LANGUAGE_FIELD].to_pylist())

    for name, root in (("full", output_root), ("canary", canary_root)):
        summaries[name]["bytes"] = sum(path.stat().st_size for path in root.glob("*.parquet"))
        summaries[name]["language_counts"] = dict(sorted(summaries[name]["language_counts"].items()))
    return summaries["full"], summaries["canary"]


def _atomic_write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_summary(summary: dict, *, expected_language_counts: dict[str, int]) -> None:
    actual = summary["language_counts"]
    if actual != expected_language_counts:
        raise ValueError(f"Language counts differ: actual={actual}, expected={expected_language_counts}")
    if summary["rows"] != sum(expected_language_counts.values()):
        raise ValueError("Summary row count does not equal the language-count total")


def run(args: argparse.Namespace) -> dict:
    source_files = discover_source_files(args.input_root)
    selection = plan_selection(
        source_files,
        english_size=args.english_size,
        canary_per_language=args.canary_per_language,
    )
    full, canary = materialize_selection(
        source_files,
        output_root=args.output_root,
        canary_root=args.canary_root,
        selection=selection,
        resume=args.resume,
    )
    expected_full = {language: selection.source_language_counts[language] for language in NON_SPACED_LANGUAGES}
    expected_full[ENGLISH] = args.english_size
    expected_full = dict(sorted(expected_full.items()))
    expected_canary = dict.fromkeys((*NON_SPACED_LANGUAGES, ENGLISH), args.canary_per_language)
    expected_canary = dict(sorted(expected_canary.items()))
    validate_summary(full, expected_language_counts=expected_full)
    validate_summary(canary, expected_language_counts=expected_canary)

    manifest = {
        "schema_version": 1,
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "canary_root": str(args.canary_root.resolve()),
        "source_files": len(source_files),
        "source_rows": selection.source_rows,
        "non_spaced_languages": list(NON_SPACED_LANGUAGES),
        "english_sample_size": args.english_size,
        "canary_per_language": args.canary_per_language,
        "sampling": "smallest_sha256(source_file:source_row_index)",
        "fields": list(OUTPUT_FIELDS),
        "full": full,
        "canary": canary,
    }
    _atomic_write_json(manifest, args.manifest_path)
    success = {
        "manifest": str(args.manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest_path.read_bytes()).hexdigest(),
        "rows": full["rows"],
    }
    _atomic_write_json(success, args.output_root / "_SUCCESS.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canary-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--english-size", type=int, default=100_000)
    parser.add_argument("--canary-per-language", type=int, default=5_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
