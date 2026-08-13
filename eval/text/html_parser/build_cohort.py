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
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

FIELDS = ("url", "text", "justext_extracted_text")


def _fragments(path: Path, limit: int = 0) -> list[ds.Fragment]:
    """Discover and validate Parquet inputs through PyArrow Dataset."""
    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
    missing = set(FIELDS).difference(dataset.schema.names)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    fragments = sorted(dataset.get_fragments(), key=lambda item: item.path)
    if not fragments:
        raise FileNotFoundError(f"no parquet files under {path}")
    return fragments if not limit else fragments[:limit]


def _strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign mutually exclusive extraction-behavior strata from text lengths."""
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


def stratified(
    fragments: list[ds.Fragment], rows_per_stratum: int, batch_size: int, seed: int = 0
) -> tuple[pd.DataFrame, Counter[str]]:
    """Return a bounded, deterministic random sample from every observed stratum."""
    kept, populations = {}, Counter()
    rng = np.random.default_rng(seed)
    for fragment in fragments:
        offset = 0
        for batch in fragment.to_batches(batch_size=batch_size, columns=list(FIELDS)):
            frame = _strata(batch.to_pandas())
            frame["_eval_source_file"] = fragment.path
            frame["_eval_source_row"] = range(offset, offset + len(frame))
            frame["_eval_stable_priority"] = rng.random(len(frame))
            offset += len(frame)
            for label, candidates in frame.groupby("parser_comparison_stratum"):
                populations[label] += len(candidates)
                prior = kept.get(label)
                kept[label] = (candidates if prior is None else pd.concat([prior, candidates])).nsmallest(
                    rows_per_stratum, "_eval_stable_priority"
                )
    return pd.concat(kept.values()).sort_values(["parser_comparison_stratum", "_eval_stable_priority"]), populations


def population(fragments: list[ds.Fragment], target: int, batch_size: int, seed: int) -> tuple[pd.DataFrame, int]:
    """Draw an exact equal-probability sample with NumPy and gather it with PyArrow."""
    sizes = np.array([fragment.count_rows() for fragment in fragments], dtype=np.int64)
    total = int(sizes.sum())
    if target > total:
        raise ValueError(f"target {target} exceeds population {total}")
    chosen = np.random.default_rng(seed).choice(total, target, replace=False)
    sorter = np.argsort(chosen)
    ordered, starts = chosen[sorter], np.concatenate(([0], np.cumsum(sizes)))
    output = []
    for fragment, start, end in zip(fragments, starts[:-1], starts[1:], strict=True):
        mask = (ordered >= start) & (ordered < end)
        selected = ordered[mask]
        if not len(selected):
            continue
        local = selected - start
        frame = _strata(fragment.take(local, columns=list(FIELDS), batch_size=batch_size).to_pandas())
        frame["_eval_source_file"], frame["_eval_source_row"] = fragment.path, local
        frame["_eval_stable_priority"] = sorter[mask]
        output.append(frame)
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
    fragments = _fragments(args.input, args.num_files)
    if args.rows <= 0 or args.batch_size <= 0:
        raise ValueError("rows and batch size must be positive")
    if args.mode == "stratified":
        frame, populations = stratified(fragments, args.rows, args.batch_size, args.seed)
        total, weights = sum(populations.values()), None
    else:
        frame, total = population(fragments, args.rows, args.batch_size, args.seed)
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
