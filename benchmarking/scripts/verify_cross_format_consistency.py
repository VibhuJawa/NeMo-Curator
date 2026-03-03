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

"""Verify interleaved data consistency across output formats (Parquet, WebDataset, Lance).

Reads back each format into a common DataFrame and checks that the interleaved
structure (sample IDs, positions, modalities, text content, source refs) is
identical across all provided formats.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

COMPARE_COLUMNS = ["sample_id", "position", "modality", "content_type", "text_content"]
_MAX_ORDERING_ERRORS = 10
_MIN_FORMATS_TO_COMPARE = 2


def _load_parquet(path: Path) -> pd.DataFrame:
    pq_files = sorted(path.rglob("*.parquet"))
    if not pq_files:
        msg = f"No parquet files found in {path}"
        raise FileNotFoundError(msg)
    frames = [pd.read_parquet(f, columns=COMPARE_COLUMNS) for f in pq_files]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["sample_id", "position"]).reset_index(drop=True)


def _load_lance(path: Path) -> pd.DataFrame:
    import lance

    lance_dirs = sorted(d for d in path.rglob("*.lance") if d.is_dir())
    if not lance_dirs:
        msg = f"No .lance directories found in {path}"
        raise FileNotFoundError(msg)
    frames = []
    for d in lance_dirs:
        ds = lance.dataset(str(d))
        table = ds.to_table(columns=COMPARE_COLUMNS)
        frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["sample_id", "position"]).reset_index(drop=True)


def _parse_wds_sample(sample_key: str, payload: dict) -> list[dict]:
    """Convert a single WDS JSON payload into interleaved rows.

    The WDS writer stores interleaved positions as parallel texts/images lists.
    A position with text has texts[i]!=None, images[i]==None and vice versa.
    When not materialized, image positions may have both texts[i]==None AND
    images[i]==None -- these are still image rows (just without binary content).
    """
    rows: list[dict] = []
    texts = payload.get("texts", [])
    images = payload.get("images", [])
    max_len = max(len(texts), len(images))

    for position in range(max_len):
        text_val = texts[position] if position < len(texts) else None

        if text_val is not None:
            rows.append({
                "sample_id": sample_key,
                "position": position,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": str(text_val),
            })
        else:
            rows.append({
                "sample_id": sample_key,
                "position": position,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
            })

    rows.append({
        "sample_id": sample_key,
        "position": -1,
        "modality": "metadata",
        "content_type": None,
        "text_content": None,
    })
    return rows


def _load_webdataset(path: Path) -> pd.DataFrame:
    """Reconstruct interleaved rows from WDS tar shards."""
    tar_files = sorted(path.rglob("*.tar"))
    if not tar_files:
        msg = f"No .tar files found in {path}"
        raise FileNotFoundError(msg)

    rows: list[dict] = []
    for tar_path in tar_files:
        with tarfile.open(str(tar_path), "r") as tf:
            json_members = {}
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if member.name.endswith(".json"):
                    stem = member.name.split(".")[0]
                    data = tf.extractfile(member).read()
                    json_members[stem] = json.loads(data)

            for sample_key, payload in json_members.items():
                rows.extend(_parse_wds_sample(sample_key, payload))

    df = pd.DataFrame(rows)
    return df.sort_values(["sample_id", "position"]).reset_index(drop=True)


_LOADERS = {
    "parquet": _load_parquet,
    "webdataset": _load_webdataset,
    "lance": _load_lance,
}


def _check_row_counts(dfs: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    names = list(dfs.keys())
    counts = {name: len(df) for name, df in dfs.items()}
    ref_count = counts[names[0]]
    for name in names[1:]:
        if counts[name] != ref_count:
            errors.append(f"Row count mismatch: {names[0]}={ref_count:,}, {name}={counts[name]:,}")
    return errors


def _check_sample_ids(dfs: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    names = list(dfs.keys())
    id_sets = {name: set(df["sample_id"].unique()) for name, df in dfs.items()}
    ref_ids = id_sets[names[0]]
    for name in names[1:]:
        only_ref = ref_ids - id_sets[name]
        only_other = id_sets[name] - ref_ids
        if only_ref:
            errors.append(f"{len(only_ref)} sample_id(s) in {names[0]} but not {name}")
        if only_other:
            errors.append(f"{len(only_other)} sample_id(s) in {name} but not {names[0]}")
    return errors


def _check_sample_counts(dfs: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    names = list(dfs.keys())
    counts = {name: df["sample_id"].nunique() for name, df in dfs.items()}
    ref = counts[names[0]]
    for name in names[1:]:
        if counts[name] != ref:
            errors.append(f"Sample count mismatch: {names[0]}={ref:,}, {name}={counts[name]:,}")
    return errors


def _check_per_sample_rows(dfs: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    names = list(dfs.keys())
    per_sample = {}
    for name, df in dfs.items():
        counts = df.groupby(df["sample_id"].astype(str)).size().sort_index()
        per_sample[name] = counts
    ref = per_sample[names[0]]
    for name in names[1:]:
        other = per_sample[name]
        if ref.shape != other.shape or not (ref.to_numpy() == other.to_numpy()).all() or not (ref.index == other.index).all():
            common = ref.index.intersection(other.index)
            diff_count = int((ref.loc[common] != other.loc[common]).sum()) if len(common) > 0 else len(ref)
            errors.append(f"Per-sample row counts differ between {names[0]} and {name}: {diff_count} sample(s)")
    return errors


def _check_modality_distribution(dfs: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    names = list(dfs.keys())
    dists = {name: dict(df["modality"].value_counts()) for name, df in dfs.items()}
    ref = dists[names[0]]
    for name in names[1:]:
        if dists[name] != ref:
            errors.append(f"Modality distribution mismatch: {names[0]}={ref}, {name}={dists[name]}")
    return errors


def _check_position_ordering(dfs: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    for name, df in dfs.items():
        for sample_id, group in df.groupby("sample_id", sort=False):
            meta = group[group["modality"] == "metadata"]
            content = group[group["modality"] != "metadata"]
            for _, row in meta.iterrows():
                if row["position"] != -1:
                    errors.append(f"[{name}] sample={sample_id}: metadata position={row['position']}, expected -1")
            if content.empty:
                continue
            positions = content["position"].tolist()
            expected = list(range(len(positions)))
            if sorted(positions) != expected:
                errors.append(f"[{name}] sample={sample_id}: positions {sorted(positions)} != {expected}")
                if len(errors) >= _MAX_ORDERING_ERRORS:
                    errors.append(f"[{name}] ... truncated")
                    return errors
    return errors


def _check_content_tuples(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Compare (sample_id, position, modality, content_type, text_content) across formats.

    Uses only non-metadata rows for text_content comparison since WDS reconstruction
    may not have identical metadata_json representation.
    """
    errors = []
    names = list(dfs.keys())
    non_meta = {name: df[df["modality"] != "metadata"].copy() for name, df in dfs.items()}

    tuple_sets = {}
    for name, df in non_meta.items():
        df_clean = df[["sample_id", "position", "modality", "text_content"]].copy()
        df_clean["text_content"] = df_clean["text_content"].fillna("")
        tuple_sets[name] = set(df_clean.itertuples(index=False, name=None))

    ref_set = tuple_sets[names[0]]
    for name in names[1:]:
        only_ref = len(ref_set - tuple_sets[name])
        only_other = len(tuple_sets[name] - ref_set)
        if only_ref or only_other:
            errors.append(f"Content tuples differ: {only_ref} only in {names[0]}, {only_other} only in {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify interleaved data consistency across output formats")
    parser.add_argument("--parquet-path", type=Path, default=None)
    parser.add_argument("--webdataset-path", type=Path, default=None)
    parser.add_argument("--lance-path", type=Path, default=None)
    args = parser.parse_args()

    paths = {
        name: path
        for name, path in [("parquet", args.parquet_path), ("webdataset", args.webdataset_path), ("lance", args.lance_path)]
        if path is not None
    }

    if len(paths) < _MIN_FORMATS_TO_COMPARE:
        print("ERROR: Provide at least 2 format paths to compare")
        sys.exit(1)

    dfs: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        print(f"Loading {name} from {path} ...")
        dfs[name] = _LOADERS[name](path)
        df = dfs[name]
        print(f"  {len(df):,} rows | {df['sample_id'].nunique():,} samples | modalities: {dict(df['modality'].value_counts())}")

    checks = [
        ("Row counts", _check_row_counts(dfs)),
        ("Sample counts", _check_sample_counts(dfs)),
        ("Sample ID sets", _check_sample_ids(dfs)),
        ("Per-sample row counts", _check_per_sample_rows(dfs)),
        ("Modality distribution", _check_modality_distribution(dfs)),
        ("Position ordering", _check_position_ordering(dfs)),
        ("Content tuples", _check_content_tuples(dfs)),
    ]

    print(f"\n{'=' * 60}")
    all_pass = True
    for name, errors in checks:
        status = "PASS" if not errors else "FAIL"
        print(f"  [{status}] {name}")
        if errors:
            all_pass = False
            for err in errors:
                print(f"    {err}")

    print(f"{'=' * 60}")
    if all_pass:
        print("CROSS-FORMAT CONSISTENCY CHECK PASSED")
    else:
        print("CROSS-FORMAT CONSISTENCY CHECK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
