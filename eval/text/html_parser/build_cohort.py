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
"""Build balanced-diagnostic or population-proportional HTML parser cohorts."""
# ruff: noqa: EM101, EM102

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
import pyarrow.parquet as pq

FIELDS = ("url", "text", "justext_extracted_text")


def _files(path: Path, limit: int = 0) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    return files if not limit else files[:limit]


def _priority(path: Path, rows: np.ndarray, seed: int) -> np.ndarray:
    key = int.from_bytes(hashlib.blake2b(f"{seed}\0{path}".encode(), digest_size=8).digest(), "little")
    with np.errstate(over="ignore"):
        value = rows.astype(np.uint64) ^ np.uint64(key)
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ value >> np.uint64(30)) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ value >> np.uint64(27)) * np.uint64(0x94D049BB133111EB)
        return value ^ value >> np.uint64(31)


def _strata(frame: pd.DataFrame) -> pd.DataFrame:
    left = frame["text"].fillna("").astype(str).str.strip().str.len()
    right = frame["justext_extracted_text"].fillna("").astype(str).str.strip().str.len()
    maximum, difference = pd.concat([left, right], axis=1).max(axis=1), (left - right).abs()
    relative = difference / maximum.where(maximum.gt(0), 1)
    label = pd.Series("both_moderate_difference", index=frame.index)
    both, substantive = left.gt(0) & right.gt(0), maximum.ge(200)
    label[left.eq(0) & right.eq(0)] = "both_empty"
    label[left.gt(0) & right.eq(0)] = "mineru_html_only"
    label[left.eq(0) & right.gt(0)] = "justext_only"
    label[both & ~substantive] = "both_short"
    label[both & substantive & relative.le(0.15)] = "both_similar_length"
    label[both & substantive & left.ge(right * 2)] = "mineru_html_much_longer"
    label[both & substantive & right.ge(left * 2)] = "justext_much_longer"
    return frame.assign(
        mineru_html_chars=left,
        justext_chars=right,
        char_count_difference=difference,
        relative_char_count_difference=relative,
        parser_comparison_stratum=label,
    )


def _validate(parquet: pq.ParquetFile, path: Path) -> None:
    missing = set(FIELDS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")


def stratified(files: list[Path], rows_per_stratum: int, batch_size: int) -> tuple[pd.DataFrame, Counter[str]]:
    kept, populations = {}, Counter()
    for path in files:
        parquet, offset = pq.ParquetFile(path), 0
        _validate(parquet, path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(FIELDS)):
            frame = _strata(batch.to_pandas())
            frame["_eval_source_file"], frame["_eval_source_row"] = str(path), range(offset, offset + len(frame))
            frame["_eval_stable_priority"] = _priority(path, np.arange(offset, offset + len(frame)), 0)
            offset += len(frame)
            for label, candidates in frame.groupby("parser_comparison_stratum"):
                populations[label] += len(candidates)
                prior = kept.get(label)
                kept[label] = (candidates if prior is None else pd.concat([prior, candidates])).nsmallest(
                    rows_per_stratum, "_eval_stable_priority"
                )
    return pd.concat(kept.values()).sort_values(["parser_comparison_stratum", "_eval_stable_priority"]), populations


def population(files: list[Path], target: int, batch_size: int, seed: int) -> tuple[pd.DataFrame, int]:
    candidates, total = None, 0
    for path in files:
        parquet = pq.ParquetFile(path)
        _validate(parquet, path)
        rows = np.arange(parquet.metadata.num_rows, dtype=np.uint64)
        priorities = _priority(path, rows, seed)
        take = np.argpartition(priorities, min(target, len(rows)) - 1)[:target]
        local = pd.DataFrame(
            {"_eval_source_file": str(path), "_eval_source_row": rows[take], "_eval_stable_priority": priorities[take]}
        )
        candidates = (
            local if candidates is None else pd.concat([candidates, local]).nsmallest(target, "_eval_stable_priority")
        )
        total += len(rows)
    if target > total:
        raise ValueError(f"target {target} exceeds population {total}")
    output = []
    for path, selected in candidates.groupby("_eval_source_file"):
        parquet, wanted, offset = pq.ParquetFile(path), selected.set_index("_eval_source_row"), 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(FIELDS)):
            rows = wanted.index[(wanted.index >= offset) & (wanted.index < offset + batch.num_rows)]
            if len(rows):
                frame = _strata(batch.take(pa.array(rows - offset)).to_pandas())
                frame["_eval_source_file"], frame["_eval_source_row"] = path, rows
                frame["_eval_stable_priority"] = wanted.loc[rows, "_eval_stable_priority"].to_numpy()
                output.append(frame)
            offset += batch.num_rows
    sample = pd.concat(output).sort_values("_eval_stable_priority")
    sample["_eval_inclusion_probability"], sample["_eval_sample_weight"] = target / total, total / target
    return sample, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("stratified", "population"), default="stratified")
    parser.add_argument("--rows", type=int, default=25, help="rows per stratum, or population target")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-files", type=int, default=0)
    args = parser.parse_args()
    files = _files(args.input, args.num_files)
    if args.rows <= 0 or args.batch_size <= 0:
        raise ValueError("rows and batch size must be positive")
    if args.mode == "stratified":
        frame, populations = stratified(files, args.rows, args.batch_size)
        total, weights = sum(populations.values()), None
    else:
        frame, total = population(files, args.rows, args.batch_size, args.seed)
        populations, weights = None, {"probability": args.rows / total, "weight": total / args.rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, args.output)
    manifest = {
        "schema_version": "html_parser_cohort_v1",
        "mode": args.mode,
        "population": total,
        "selected": len(frame),
        "seed": args.seed,
        "weights": weights,
        "strata": frame["parser_comparison_stratum"].value_counts().sort_index().to_dict(),
        "stratum_populations": dict(sorted(populations.items())) if populations else None,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
