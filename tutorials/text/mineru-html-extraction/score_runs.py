#!/usr/bin/env python3
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
"""
Score extraction runs against a gold run with WebMainBench's ROUGE-N F1.

The metric is theirs, reimplemented to match `TextRougeNgramMetric` in
`webmainbench/metrics/text_metrics.py` line for line: `jieba.lcut` on both sides,
`rouge_score.rouge_scorer._create_ngrams` at n=5, `_score_ngrams`, and the **F1** is the
score. Precision and recall are kept too, because which of the two a model is losing is
the whole diagnosis and the F1 alone hides it.

    python score_runs.py --gold out-5k-opus --candidate qwen=out-5k-qwen \
        --input sample-5k.parquet --out scores/

**Why the baselines are not optional.** A ROUGE score is only evidence about extraction
quality if it moves when quality moves, and that has to be shown rather than assumed —
particularly here, where "gold" is another model's output rather than a human's. So
every run is scored alongside controls whose quality is known by construction:

    identical      gold against itself                   must be 1.0, or the harness is broken
    empty          nothing at all                        0.0
    keep-all       every word of the page, unfiltered    the no-filtering floor
    gold-plus-junk gold with the unfiltered remainder    kept the boilerplate too
    gold-half      gold with the back half dropped       over-filtered

If `keep-all` does not score clearly below a real extractor, the metric is not measuring
filtering on this corpus and no ranking built on it means anything. That is a result
worth having before the ranking, not after.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


# Excluded documents are named in summary.json, not just counted, so the exclusion can
# be audited rather than taken on trust. Capped because a gold run that failed on
# everything would otherwise write its entire index into the summary.
def rouge_n(target: str, prediction: str, n: int = 5) -> dict[str, float]:
    """WebMainBench's `calc_rouge_n_score`, reproduced.

    Kept deliberately close to the original — including the both-empty case scoring 1.0
    — so a number here is comparable with a number from their toolkit rather than
    merely similar to one.
    """
    import jieba
    from rouge_score.rouge_scorer import _create_ngrams, _score_ngrams

    if len(target.strip()) == 0 and len(prediction.strip()) == 0:
        return {"prec": 1.0, "rec": 1.0, "f1": 1.0}

    target_ngrams = _create_ngrams(list(jieba.lcut(target)), n)
    prediction_ngrams = _create_ngrams(list(jieba.lcut(prediction)), n)
    score = _score_ngrams(target_ngrams, prediction_ngrams)
    return {"prec": score.precision, "rec": score.recall, "f1": score.fmeasure}


def read_run(path: Path, text_field: str, id_field: str) -> pd.DataFrame:
    """A run's extracted text, keyed by document."""
    files = sorted(path.glob("**/*.parquet")) if path.is_dir() else [path]
    if not files:
        msg = f"no parquet under {path}"
        raise SystemExit(msg)
    frames = [pq.read_table(f, use_threads=False).to_pandas() for f in files]
    frame = pd.concat(frames, ignore_index=True)
    if id_field not in frame.columns:
        msg = f"{path} has no {id_field!r} column; have {sorted(frame.columns)[:12]}"
        raise SystemExit(msg)
    return frame[[c for c in (id_field, text_field, "_mineru_status") if c in frame.columns]]


def plain_text(html: str) -> str:
    """Every word on the page, tags removed and nothing judged. The no-filtering floor."""
    return _WS.sub(" ", _TAGS.sub(" ", html or "")).strip()


def build_controls(gold: pd.Series, unfiltered: pd.Series) -> dict[str, pd.Series]:
    """Runs whose quality is known by construction, so the metric can be read against them."""

    # The words the unfiltered page has and gold does not — i.e. exactly the boilerplate
    # a perfect extractor dropped. Appending it simulates an extractor that kept it.
    def junk(row_gold: str, row_all: str) -> str:
        kept = set(row_gold.split())
        return row_gold + " " + " ".join(w for w in row_all.split() if w not in kept)

    def half(text: str) -> str:
        words = text.split()
        return " ".join(words[: len(words) // 2])

    return {
        "identical": gold.copy(),
        "empty": pd.Series([""] * len(gold), index=gold.index),
        "keep-all": unfiltered,
        "gold-plus-junk": pd.Series([junk(g, a) for g, a in zip(gold, unfiltered, strict=True)], index=gold.index),
        "gold-half": gold.map(half),
    }


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="Run directory treated as ground truth")
    ap.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="A run to score against gold; repeatable",
    )
    ap.add_argument("--input", help="The sample parquet, for the keep-all control")
    ap.add_argument("--out", required=True, help="Directory for the scores")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--id-field", default="arxiv_id")
    ap.add_argument("--html-field", default="html")
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--no-controls", action="store_true", help="Skip the calibration baselines")
    ap.add_argument(
        "--common-only",
        action="store_true",
        help="Score only documents every run produced, instead of filling absences with empty",
    )
    args = ap.parse_args()

    gold = read_run(Path(args.gold), args.text_field, args.id_field).set_index(args.id_field)
    gold_text = gold[args.text_field].fillna("")
    print(f"  gold: {len(gold_text)} documents from {args.gold}")

    runs: dict[str, pd.Series] = {}
    raw: dict[str, pd.Series] = {}
    for entry in args.candidate:
        if "=" not in entry:
            msg = f"--candidate wants NAME=DIR, got {entry!r}"
            raise SystemExit(msg)
        name, directory = entry.split("=", 1)
        frame = read_run(Path(directory), args.text_field, args.id_field).set_index(args.id_field)
        raw[name] = frame[args.text_field]

    # A document a run never saw is not a document that run scored zero on. Reindexing a
    # 500-document run onto a 5,000-document gold silently fills 4,500 empties and reports
    # the result as the run's quality, which is how a subset comparison turns into a
    # fabricated collapse. Restricting to the documents every run actually produced is the
    # only honest way to put two runs of different size beside each other -- and the count
    # that was dropped is printed, because a comparison over an unstated subset is not a
    # comparison.
    if args.common_only and raw:
        shared = gold_text.index
        for series in raw.values():
            shared = shared.intersection(series.index)
        dropped = len(gold_text) - len(shared)
        if dropped:
            print(f"  restricting to {len(shared)} documents scored by every run ({dropped} dropped)")
        gold_text = gold_text.reindex(shared)

    for name, series in raw.items():
        missing = len(gold_text.index.difference(series.index))
        if missing:
            print(f"  {name}: {missing} of {len(gold_text)} documents absent, scored as empty")
        runs[name] = series.reindex(gold_text.index).fillna("")

    if not args.no_controls:
        if not args.input:
            msg = "--input is needed for the keep-all control; pass --no-controls to skip them"
            raise SystemExit(msg)
        # Name the parquet explicitly rather than handing pyarrow the directory: a
        # sample directory also holds the selection.json that records how it was drawn,
        # and pyarrow reads every entry it is given and fails on the first non-parquet.
        source_path = Path(args.input)
        source_files = sorted(source_path.glob("**/*.parquet")) if source_path.is_dir() else [source_path]
        if not source_files:
            msg = f"no parquet under {source_path}"
            raise SystemExit(msg)
        source = pd.concat(
            [
                pq.read_table(f, columns=[args.id_field, args.html_field], use_threads=False).to_pandas()
                for f in source_files
            ],
            ignore_index=True,
        )
        unfiltered = (
            source.set_index(args.id_field)[args.html_field].map(plain_text).reindex(gold_text.index).fillna("")
        )
        runs = {**build_controls(gold_text, unfiltered), **runs}

    rows: list[dict[str, object]] = []
    for name, text in runs.items():
        started = time.perf_counter()
        for doc_id, prediction in text.items():
            scored = rouge_n(str(gold_text[doc_id]), str(prediction), n=args.ngram)
            rows.append({"run": name, args.id_field: doc_id, **scored, "chars": len(str(prediction))})
        print(f"  {name:<16} scored {len(text)} documents in {time.perf_counter() - started:.1f}s")

    per_doc = pd.DataFrame(rows)
    # Mean over documents, not over n-grams: every document counts once, so one enormous
    # paper cannot decide the corpus score.
    summary = (
        per_doc.groupby("run")[["prec", "rec", "f1", "chars"]].mean().sort_values("f1", ascending=False).reset_index()
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_doc.to_parquet(out / "per_document.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "gold": str(args.gold),
                "ngram": args.ngram,
                "documents": len(gold_text),
                "metric": "WebMainBench rouge_n (jieba + rouge_score), F1",
                "runs": summary.to_dict(orient="records"),
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )

    print()
    print(f"  {'run':<16} {'F1':>7} {'prec':>7} {'rec':>7} {'chars':>9}")
    for row in summary.itertuples():
        print(f"  {row.run:<16} {row.f1:>7.4f} {row.prec:>7.4f} {row.rec:>7.4f} {row.chars:>9,.0f}")
    print(f"\n  wrote {out}/summary.csv, summary.json, per_document.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
