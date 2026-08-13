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

"""Build a deterministic stratified MinerU-HTML/jusText judge cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_INPUT = Path(
    "/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/crawl_data/"
    "crawl_extraction_experiments/justext_vs_dripper_10m/output"
)
_MINERU_FIELD = "text"
_JUSTEXT_FIELD = "justext_extracted_text"
_REQUIRED_FIELDS = ("url", _MINERU_FIELD, _JUSTEXT_FIELD)
_SIMILAR_LENGTH_THRESHOLD = 0.15
_LARGE_LENGTH_RATIO = 2.0
_SHORT_TEXT_CHARS = 200
_STRATUM_DESCRIPTIONS = {
    "both_empty": "Both extracted texts are empty after trimming whitespace.",
    "mineru_html_only": "Only MinerU-HTML produced non-empty text.",
    "justext_only": "Only jusText produced non-empty text.",
    "both_short": "Both are non-empty and the longer extraction has fewer than 200 characters.",
    "both_similar_length": "Both are substantive and their character counts differ by at most 15%.",
    "mineru_html_much_longer": "MinerU-HTML has at least twice as many characters as jusText.",
    "justext_much_longer": "jusText has at least twice as many characters as MinerU-HTML.",
    "both_moderate_difference": "Both are substantive with an intermediate length disagreement.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sampling-mode",
        choices=("stratified", "population"),
        default="stratified",
        help="Balanced diagnostic strata or an equal-probability population sample",
    )
    parser.add_argument("--rows-per-stratum", type=int, default=25)
    parser.add_argument("--target-rows", type=int, default=5000, help="Rows for --sampling-mode population")
    parser.add_argument("--seed", type=int, default=17, help="Stable population-sample seed")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-files", type=int, default=0, help="Sorted files to scan; 0 means all")
    return parser.parse_args()


def resolve_inputs(input_path: Path, num_files: int) -> list[Path]:
    if num_files < 0:
        msg = "--num-files must be non-negative"
        raise ValueError(msg)
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.parquet"))
    if not files:
        msg = f"No parquet inputs found under {input_path}"
        raise FileNotFoundError(msg)
    return files if num_files == 0 else files[:num_files]


def classify_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic length diagnostics and mutually exclusive strata."""
    mineru = frame[_MINERU_FIELD].fillna("").astype(str).str.strip()
    justext = frame[_JUSTEXT_FIELD].fillna("").astype(str).str.strip()
    mineru_chars = mineru.str.len()
    justext_chars = justext.str.len()
    maximum = pd.concat([mineru_chars, justext_chars], axis=1).max(axis=1)
    difference = (mineru_chars - justext_chars).abs()
    relative_difference = difference.div(maximum.where(maximum.gt(0), 1))

    category = pd.Series("both_moderate_difference", index=frame.index, dtype="string")
    mineru_empty = mineru_chars.eq(0)
    justext_empty = justext_chars.eq(0)
    both_present = ~mineru_empty & ~justext_empty
    category.loc[mineru_empty & justext_empty] = "both_empty"
    category.loc[~mineru_empty & justext_empty] = "mineru_html_only"
    category.loc[mineru_empty & ~justext_empty] = "justext_only"
    category.loc[both_present & maximum.lt(_SHORT_TEXT_CHARS)] = "both_short"
    category.loc[both_present & maximum.ge(_SHORT_TEXT_CHARS) & relative_difference.le(_SIMILAR_LENGTH_THRESHOLD)] = (
        "both_similar_length"
    )
    category.loc[
        both_present & maximum.ge(_SHORT_TEXT_CHARS) & mineru_chars.ge(justext_chars.mul(_LARGE_LENGTH_RATIO))
    ] = "mineru_html_much_longer"
    category.loc[
        both_present & maximum.ge(_SHORT_TEXT_CHARS) & justext_chars.ge(mineru_chars.mul(_LARGE_LENGTH_RATIO))
    ] = "justext_much_longer"

    result = frame.copy()
    result["mineru_html_chars"] = mineru_chars
    result["justext_chars"] = justext_chars
    result["char_count_difference"] = difference
    result["relative_char_count_difference"] = relative_difference
    result["parser_comparison_stratum"] = category
    return result


def build_cohort(
    files: list[Path],
    rows_per_stratum: int,
    batch_size: int,
    *,
    progress_every_files: int = 0,
) -> tuple[pd.DataFrame, Counter[str]]:
    """Scan parquet in bounded memory and retain the lowest stable hashes per stratum."""
    if rows_per_stratum <= 0 or batch_size <= 0:
        msg = "--rows-per-stratum and --batch-size must be positive"
        raise ValueError(msg)
    retained: dict[str, pd.DataFrame] = {}
    populations: Counter[str] = Counter()
    processed_rows = 0
    for file_index, source_path in enumerate(files, start=1):
        parquet = pq.ParquetFile(source_path)
        _validate_parquet_fields(parquet, source_path)
        source_row = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(_REQUIRED_FIELDS)):
            mineru_chars = _arrow_text_lengths(batch.column(_MINERU_FIELD))
            justext_chars = _arrow_text_lengths(batch.column(_JUSTEXT_FIELD))
            strata, difference, relative_difference = _length_strata(mineru_chars, justext_chars)
            row_numbers = np.arange(source_row, source_row + batch.num_rows, dtype=np.uint64)
            priorities = _stable_priorities(str(source_path), row_numbers)
            source_row += batch.num_rows
            processed_rows += batch.num_rows
            for stratum in np.unique(strata):
                indices = np.flatnonzero(strata == stratum)
                populations[str(stratum)] += len(indices)
                if len(indices) > rows_per_stratum:
                    local = np.argpartition(priorities[indices], rows_per_stratum - 1)[:rows_per_stratum]
                    indices = indices[local]
                selected = batch.take(pa.array(indices)).to_pandas()
                selected["_eval_source_file"] = str(source_path)
                selected["_eval_source_row"] = row_numbers[indices]
                selected["mineru_html_chars"] = mineru_chars[indices]
                selected["justext_chars"] = justext_chars[indices]
                selected["char_count_difference"] = difference[indices]
                selected["relative_char_count_difference"] = relative_difference[indices]
                selected["parser_comparison_stratum"] = str(stratum)
                selected["_eval_stable_priority"] = priorities[indices]
                previous = retained.get(str(stratum))
                combined = selected if previous is None else pd.concat([previous, selected], ignore_index=True)
                retained[str(stratum)] = combined.nsmallest(rows_per_stratum, "_eval_stable_priority")
        if progress_every_files > 0 and (file_index % progress_every_files == 0 or file_index == len(files)):
            print(f"Scanned {file_index}/{len(files)} files and {processed_rows} rows", flush=True)
    if not retained:
        return pd.DataFrame(), populations
    cohort = pd.concat(retained.values(), ignore_index=True)
    cohort = cohort.sort_values(["parser_comparison_stratum", "_eval_stable_priority"], kind="stable").reset_index(
        drop=True
    )
    return cohort, populations


def build_population_sample(
    files: list[Path],
    target_rows: int,
    batch_size: int,
    *,
    seed: int = 17,
    progress_every_files: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Select an exact equal-probability bottom-k sample and attach survey weights."""
    if target_rows <= 0 or batch_size <= 0:
        msg = "--target-rows and --batch-size must be positive"
        raise ValueError(msg)
    candidates, total_rows = _select_population_candidates(files, target_rows, seed)
    if target_rows > total_rows:
        msg = f"--target-rows={target_rows} exceeds the {total_rows}-row population"
        raise ValueError(msg)
    sample = _materialize_population_candidates(files, candidates, batch_size, progress_every_files)
    if len(sample) != target_rows:
        msg = f"Expected {target_rows} sampled rows but materialized {len(sample)}"
        raise RuntimeError(msg)
    probability = target_rows / total_rows
    sample["_eval_inclusion_probability"] = probability
    sample["_eval_sample_weight"] = 1.0 / probability
    return sample, total_rows


def _select_population_candidates(files: list[Path], target_rows: int, seed: int) -> tuple[pd.DataFrame, int]:
    candidates: pd.DataFrame | None = None
    total_rows = 0
    priority_chunk_rows = max(target_rows, 1_000_000)
    for source_path in files:
        parquet = pq.ParquetFile(source_path)
        _validate_parquet_fields(parquet, source_path)
        file_rows = parquet.metadata.num_rows
        total_rows += file_rows
        for start in range(0, file_rows, priority_chunk_rows):
            row_numbers = np.arange(start, min(start + priority_chunk_rows, file_rows), dtype=np.uint64)
            priorities = _stable_priorities(str(source_path), row_numbers, seed=seed)
            if len(row_numbers) > target_rows:
                indices = np.argpartition(priorities, target_rows - 1)[:target_rows]
                row_numbers = row_numbers[indices]
                priorities = priorities[indices]
            local = pd.DataFrame(
                {
                    "_eval_source_file": str(source_path),
                    "_eval_source_row": row_numbers,
                    "_eval_stable_priority": priorities,
                }
            )
            candidates = (
                local
                if candidates is None
                else pd.concat([candidates, local], ignore_index=True).nsmallest(target_rows, "_eval_stable_priority")
            )
    if candidates is None:
        return pd.DataFrame(columns=["_eval_source_file", "_eval_source_row", "_eval_stable_priority"]), 0
    return candidates, total_rows


def _materialize_population_candidates(
    files: list[Path],
    candidates: pd.DataFrame,
    batch_size: int,
    progress_every_files: int,
) -> pd.DataFrame:
    selected_frames = []
    grouped = {
        Path(path): group.sort_values("_eval_source_row") for path, group in candidates.groupby("_eval_source_file")
    }
    for file_index, source_path in enumerate(files, start=1):
        selected = grouped.get(source_path)
        if selected is None:
            continue
        parquet = pq.ParquetFile(source_path)
        target_source_rows = selected["_eval_source_row"].to_numpy(dtype=np.int64)
        priorities_by_row = dict(
            zip(target_source_rows, selected["_eval_stable_priority"].to_numpy(dtype=np.uint64), strict=True)
        )
        source_offset = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(_REQUIRED_FIELDS)):
            positions = target_source_rows[
                (target_source_rows >= source_offset) & (target_source_rows < source_offset + batch.num_rows)
            ]
            if len(positions):
                local_positions = positions - source_offset
                frame = classify_strata(batch.take(pa.array(local_positions)).to_pandas())
                frame["_eval_source_file"] = str(source_path)
                frame["_eval_source_row"] = positions
                frame["_eval_stable_priority"] = [priorities_by_row[int(row)] for row in positions]
                selected_frames.append(frame)
            source_offset += batch.num_rows
        if progress_every_files > 0 and (file_index % progress_every_files == 0 or file_index == len(files)):
            print(f"Materialized population sample through {file_index}/{len(files)} files", flush=True)
    return pd.concat(selected_frames, ignore_index=True).sort_values("_eval_stable_priority").reset_index(drop=True)


def _validate_parquet_fields(parquet: pq.ParquetFile, source_path: Path) -> None:
    missing = set(_REQUIRED_FIELDS) - set(parquet.schema_arrow.names)
    if missing:
        msg = f"{source_path} is missing required fields: {sorted(missing)}"
        raise ValueError(msg)


def _arrow_text_lengths(column: pa.Array) -> np.ndarray:
    text = pc.fill_null(column, "")
    return np.asarray(pc.utf8_length(pc.utf8_trim_whitespace(text)), dtype=np.int64)


def _length_strata(mineru_chars: np.ndarray, justext_chars: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = np.maximum(mineru_chars, justext_chars)
    difference = np.abs(mineru_chars - justext_chars)
    relative_difference = np.divide(
        difference,
        np.maximum(maximum, 1),
        dtype=np.float64,
    )
    mineru_empty = mineru_chars == 0
    justext_empty = justext_chars == 0
    both_present = ~mineru_empty & ~justext_empty
    strata = np.full(len(mineru_chars), "both_moderate_difference", dtype=object)
    strata[mineru_empty & justext_empty] = "both_empty"
    strata[~mineru_empty & justext_empty] = "mineru_html_only"
    strata[mineru_empty & ~justext_empty] = "justext_only"
    strata[both_present & (maximum < _SHORT_TEXT_CHARS)] = "both_short"
    strata[both_present & (maximum >= _SHORT_TEXT_CHARS) & (relative_difference <= _SIMILAR_LENGTH_THRESHOLD)] = (
        "both_similar_length"
    )
    strata[both_present & (maximum >= _SHORT_TEXT_CHARS) & (mineru_chars >= justext_chars * _LARGE_LENGTH_RATIO)] = (
        "mineru_html_much_longer"
    )
    strata[both_present & (maximum >= _SHORT_TEXT_CHARS) & (justext_chars >= mineru_chars * _LARGE_LENGTH_RATIO)] = (
        "justext_much_longer"
    )
    return strata, difference, relative_difference


def _stable_priorities(source_path: str, row_numbers: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Return deterministic SplitMix64 priorities without hashing text payloads."""
    identity = f"{seed}\0{source_path}".encode()
    file_key = int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little")
    with np.errstate(over="ignore"):
        values = row_numbers.astype(np.uint64, copy=True) ^ np.uint64(file_key)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def write_cohort(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def write_manifest(manifest: dict, output_path: Path) -> Path:
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    temporary_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_path.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
    os.replace(temporary_path, manifest_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    files = resolve_inputs(args.input, args.num_files)
    if args.sampling_mode == "population":
        cohort, total_rows = build_population_sample(
            files,
            args.target_rows,
            args.batch_size,
            seed=args.seed,
            progress_every_files=25,
        )
        populations = None
    else:
        cohort, populations = build_cohort(
            files,
            args.rows_per_stratum,
            args.batch_size,
            progress_every_files=25,
        )
        total_rows = sum(populations.values())
    if cohort.empty:
        msg = "No rows were available for the parser comparison cohort"
        raise RuntimeError(msg)
    write_cohort(cohort, args.output)
    selected_counts = cohort["parser_comparison_stratum"].value_counts().sort_index().to_dict()
    manifest = {
        "schema_version": f"html_parser_{args.sampling_mode}_cohort_v1",
        "source": {
            "input": str(args.input),
            "files": [str(path) for path in files],
            "file_count": len(files),
            "row_count": total_rows,
            "columns": list(_REQUIRED_FIELDS),
        },
        "selection": {
            "mode": args.sampling_mode,
            "rows_per_stratum": args.rows_per_stratum if args.sampling_mode == "stratified" else None,
            "target_rows": args.target_rows if args.sampling_mode == "population" else None,
            "seed": args.seed if args.sampling_mode == "population" else 0,
            "inclusion_probability": args.target_rows / total_rows if args.sampling_mode == "population" else None,
            "sample_weight": total_rows / args.target_rows if args.sampling_mode == "population" else None,
            "stable_priority": "splitmix64(blake2b(seed + source_file) xor source_row)",
            "short_text_chars": _SHORT_TEXT_CHARS,
            "similar_length_threshold": _SIMILAR_LENGTH_THRESHOLD,
            "large_length_ratio": _LARGE_LENGTH_RATIO,
        },
        "strata": {
            label: {
                "description": _STRATUM_DESCRIPTIONS[label],
                "population": int(populations[label]) if populations is not None else None,
                "selected": int(selected_counts.get(label, 0)),
            }
            for label in sorted(selected_counts)
        },
        "selected_rows": len(cohort),
        "selected_source_files": int(cohort["_eval_source_file"].nunique()),
        "output": str(args.output),
    }
    manifest_path = write_manifest(manifest, args.output)
    print(f"Wrote {len(cohort)} rows from {len(files)} files to {args.output}")
    print(f"Wrote manifest to {manifest_path}")
    for label, selected in selected_counts.items():
        population = populations[label] if populations is not None else "population estimate deferred"
        print(f"{label}: selected={selected}, population={population}")


if __name__ == "__main__":
    main()
