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
Assemble one report.json from everything a set of extraction runs produced.

Every number the report viewer shows is computed here and read there. That split is
DESIGN.md §6a: a figure recomputed in the browser drifts from the figure that produced
the conclusion, and a reader cannot tell which of the two is lying. The viewer's job is
to draw what this writes.

What it reads, all optional so a partial report is still a report:

  --scores DIR        `score_runs.py --out` -- summary.json and per_document.parquet
  --run NAME=DIR      an extraction run, for status / coverage / window counts
  --bench FILE        stdout of bench.sbatch, for the scheduler A/B wall clocks (repeatable)
  --vllm-log FILE     a server log, for the measured prefill/decode throughput series

Anything absent is recorded as absent rather than omitted, so the viewer can say "not
measured" instead of silently dropping a section.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

# Per-document rows the report carries in full, for the failure gallery. Enough to see
# the shape of a failure, few enough that the JSON stays loadable in one request.
WORST_N = 30
EXCERPT_CHARS = 1500

# Thresholds the report counts documents against. 0.8 is roughly where a document stops
# looking like the same document; 0.5 is where it stops being recognisable at all.
POOR = 0.8
BAD = 0.5

# A throughput sample below this is the server idling between batches, not serving, and
# averaging it in would understate what the engine does when it has work.
BUSY_PREFILL = 1000

# Two runs over the same documents is what a paired comparison needs.
PAIR = 2


def read_scores(path: Path) -> dict[str, Any]:
    """`score_runs.py` output: the summary table and the per-document detail."""
    import pandas as pd

    summary_file = path / "summary.json"
    per_doc_file = path / "per_document.parquet"
    out: dict[str, Any] = {"dir": str(path)}

    if summary_file.is_file():
        out["summary"] = json.loads(summary_file.read_text())
    if per_doc_file.is_file():
        frame = pd.read_parquet(per_doc_file)
        out["perDocument"] = frame
    return out


def histogram(values: list[float], bins: int = 40, lo: float = 0.0, hi: float = 1.0) -> list[dict[str, float]]:
    """Counts per equal-width bin. The distribution is the finding, so it ships whole."""
    if not values:
        return []
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, max(0, int((v - lo) / width)))
        counts[idx] += 1
    return [{"x0": lo + i * width, "x1": lo + (i + 1) * width, "n": counts[i]} for i in range(bins)]


def describe(values: list[float]) -> dict[str, float]:
    """The five numbers worth quoting about a distribution, plus the tail."""
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "p01": q(0.01),
        "p10": q(0.10),
        "median": q(0.50),
        "p90": q(0.90),
        "min": ordered[0],
        "max": ordered[-1],
        "below50": sum(1 for v in ordered if v < BAD),
        "below80": sum(1 for v in ordered if v < POOR),
    }


# Two-tailed 95% t critical values by degrees of freedom. Beyond 30 the normal value is
# within 4% and the table stops. Using t rather than a flat 1.96 matters only at small n --
# and at small n it is the difference between an interval that admits it knows little and
# one that pretends otherwise.
T95 = {
    1: 12.71,
    2: 4.30,
    3: 3.18,
    4: 2.78,
    5: 2.57,
    6: 2.45,
    7: 2.36,
    8: 2.31,
    9: 2.26,
    10: 2.23,
    12: 2.18,
    15: 2.13,
    20: 2.09,
    25: 2.06,
    30: 2.04,
}


def t_critical(n: int) -> float:
    df = n - 1
    if df <= 0:
        return float("nan")
    if df > 30:  # noqa: PLR2004
        return 1.96
    return T95.get(df) or T95[min(T95, key=lambda k: abs(k - df))]


def interval(values: list[float]) -> dict[str, float]:
    """A mean with the 95% interval around it, over documents.

    Each document contributes one value, so this says how well the sample pins the mean of
    the population it was drawn from. Note what it does NOT say: the sampling fraction is
    irrelevant here (500 of 3.04 million carries a finite-population correction of
    0.99992), so precision is bought with n and nothing else.
    """
    n = len(values)
    if n == 0:
        return {}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / (n**0.5) if n > 1 else 0.0
    half = t_critical(n) * se if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ciLow": mean - half,
        "ciHigh": mean + half,
    }


def run_facts(name: str, path: Path) -> dict[str, Any]:
    """Status, coverage and window counts, keyed by document."""
    import pyarrow.parquet as pq

    wanted = [
        "url",
        "text",
        "_mineru_status",
        "_mineru_coverage",
        "_mineru_n_items",
        "_mineru_chunk_ids",
    ]
    rows: dict[str, dict[str, Any]] = {}
    status: dict[str, int] = {}
    windows: list[int] = []
    coverage: list[float] = []

    files = sorted(path.glob("**/*.parquet"))
    for f in files:
        available = [c for c in wanted if c in pq.ParquetFile(f).schema_arrow.names]
        for row in pq.read_table(f, columns=available, use_threads=False).to_pylist():
            n_windows = len(json.loads(row.get("_mineru_chunk_ids") or "[]"))
            windows.append(n_windows)
            cov = row.get("_mineru_coverage") or 0.0
            coverage.append(cov)
            st = row.get("_mineru_status") or "unknown"
            status[st] = status.get(st, 0) + 1
            rows[row["url"]] = {
                "status": st,
                "coverage": cov,
                "items": row.get("_mineru_n_items"),
                "windows": n_windows,
                "chars": len(row.get("text") or ""),
                "text": (row.get("text") or "")[:EXCERPT_CHARS],
            }

    return {
        "name": name,
        "dir": str(path),
        "documents": len(rows),
        "status": status,
        "coverageMean": statistics.fmean(coverage) if coverage else None,
        "windowsMean": statistics.fmean(windows) if windows else None,
        "windowsMax": max(windows) if windows else None,
        "_rows": rows,
    }


def parse_bench(path: Path) -> list[dict[str, Any]]:
    """Wall clock per scheduler config, from the A/B job's stdout."""
    text = path.read_text(errors="replace")
    out = []
    header = re.compile(r"=== (\S+): budget=(\d+) seqs=(\S+) concurrency=(\d+) ===")
    result = re.compile(r"^\s+(\S+): exit (\d+), wall (\d+)s", re.MULTILINE)
    configs = {m.group(1): m.groups() for m in header.finditer(text)}
    for m in result.finditer(text):
        name, exit_code, wall = m.group(1), int(m.group(2)), int(m.group(3))
        cfg = configs.get(name)
        out.append(
            {
                "name": name,
                "batchedTokens": int(cfg[1]) if cfg else None,
                "maxNumSeqs": cfg[2] if cfg else None,
                "concurrency": int(cfg[3]) if cfg else None,
                "wallSeconds": wall,
                "ok": exit_code == 0,
            }
        )
    return out


def parse_vllm_log(path: Path) -> dict[str, Any]:
    """The throughput series vLLM logs, which is the evidence for the decode arithmetic."""
    line = re.compile(
        r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
        r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
    )
    samples = []
    with path.open(errors="replace") as fh:
        for row in fh:
            m = line.search(row)
            if m:
                samples.append(
                    {
                        "prefill": float(m.group(1)),
                        "decode": float(m.group(2)),
                        "running": int(m.group(3)),
                        "waiting": int(m.group(4)),
                        "kv": float(m.group(5)),
                    }
                )
    if not samples:
        return {"file": path.name, "samples": []}

    busy = [s for s in samples if s["prefill"] > BUSY_PREFILL]
    return {
        "file": path.name,
        "samples": samples,
        "prefillMedian": statistics.median(s["prefill"] for s in busy) if busy else None,
        "decodeMedian": statistics.median(s["decode"] for s in busy) if busy else None,
        "runningMedian": statistics.median(s["running"] for s in busy) if busy else None,
        "kvMax": max(s["kv"] for s in samples),
    }


def main() -> int:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", action="append", default=[], help="score_runs.py output dir")
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR of an extraction run")
    ap.add_argument("--bench", action="append", default=[], help="stdout of a scheduler A/B job")
    ap.add_argument("--vllm-log", action="append", default=[], help="a vllm server log")
    ap.add_argument("--gold-run", help="NAME=DIR of the gold run, for failure excerpts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report: dict[str, Any] = {"generated": generated, "sections": {}}

    # --- runs -------------------------------------------------------------------
    runs = []
    facts: dict[str, dict[str, Any]] = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        f = run_facts(name, Path(path))
        facts[name] = f.pop("_rows")
        runs.append(f)
    report["sections"]["runs"] = runs

    gold_rows: dict[str, dict[str, Any]] = {}
    if args.gold_run:
        gname, _, gpath = args.gold_run.partition("=")
        g = run_facts(gname, Path(gpath))
        gold_rows = g.pop("_rows")
        report["sections"]["gold"] = g

    # --- scores -----------------------------------------------------------------
    scored = []
    worst: list[dict[str, Any]] = []
    for spec in args.scores:
        s = read_scores(Path(spec))
        frame = s.pop("perDocument", None)
        summary = s.get("summary") or {}
        entry: dict[str, Any] = {"dir": s["dir"], "summary": summary or None}

        # Lifted out of the summary blob and onto the entry, because the scored/excluded
        # split is a caveat on every mean below it rather than a detail of the run that
        # produced them: a report that quotes a mean without saying what it was taken
        # over is the thing this exclusion was added to stop. Defaulted from the older
        # keys so a score directory written before the exclusion still reads.
        entry["documentsScored"] = summary.get("documentsScored", summary.get("documents"))
        entry["documentsExcluded"] = summary.get("documentsExcluded", 0)
        entry["excluded"] = summary.get("excluded", {})

        if frame is not None and len(frame):
            # score_runs writes long format: one row per (run, document).
            runs_in = sorted(frame["run"].unique().tolist())
            entry["runs"] = {}
            by_run = {r: frame[frame["run"] == r] for r in runs_in}
            for run, rows in by_run.items():
                values = [float(v) for v in rows["f1"].tolist()]
                entry["runs"][run] = {
                    "describe": describe(values),
                    "histogram": histogram(values),
                    # Precision and recall carry their own intervals because which of the
                    # two a run is losing is the whole diagnosis, and a mean without an
                    # interval invites a reader to rank two runs that are not separable.
                    "metrics": {
                        "f1": interval(values),
                        "precision": interval([float(v) for v in rows["prec"].tolist()]),
                        "recall": interval([float(v) for v in rows["rec"].tolist()]),
                    },
                }

            control = {"identical", "empty", "keep-all", "gold-plus-junk", "gold-half"}
            real = [r for r in runs_in if r not in control]

            # The failure gallery: the worst documents of the first real run, with
            # whatever that run recorded about why -- status, coverage, window count.
            # A low score with status ok and coverage 1.0 is a different problem from a
            # low score with half the elements unlabelled, and the fix differs too.
            if real:
                primary = real[0]
                entry["primary"] = primary
                for _, row in by_run[primary].sort_values("f1").head(WORST_N).iterrows():
                    url = row["url"]
                    fact = facts.get(primary, {}).get(url, {})
                    worst.append(
                        {
                            "url": url,
                            "run": primary,
                            "f1": float(row["f1"]),
                            "precision": float(row["prec"]),
                            "recall": float(row["rec"]),
                            "status": fact.get("status"),
                            "coverage": fact.get("coverage"),
                            "items": fact.get("items"),
                            "windows": fact.get("windows"),
                            "candidateChars": fact.get("chars"),
                            "goldChars": gold_rows.get(url, {}).get("chars"),
                            "candidateExcerpt": fact.get("text"),
                            "goldExcerpt": gold_rows.get(url, {}).get("text"),
                        }
                    )

            # Paired comparison: same documents, two runs, per-document difference. This
            # is what turns a 0.003 gap between two means into a claim or a non-claim.
            if len(real) >= PAIR:
                a, b = real[0], real[1]
                left = by_run[a].set_index("url")["f1"]
                right = by_run[b].set_index("url")["f1"]
                shared = [u for u in left.index if u in right.index]
                pairs = [{"url": u, "a": float(left[u]), "b": float(right[u])} for u in shared]
                deltas = [p["b"] - p["a"] for p in pairs]
                entry["paired"] = {
                    "a": a,
                    "b": b,
                    "points": pairs,
                    "deltaMean": statistics.fmean(deltas) if deltas else None,
                    "deltaMedian": statistics.median(deltas) if deltas else None,
                    "bWins": sum(1 for d in deltas if d > 0),
                    "aWins": sum(1 for d in deltas if d < 0),
                    "ties": sum(1 for d in deltas if d == 0),
                    # The standard error of the mean difference, so a reader can see
                    # whether the gap clears its own noise. No scipy needed for that.
                    "deltaStdErr": (statistics.stdev(deltas) / (len(deltas) ** 0.5) if len(deltas) > 1 else None),
                }
        scored.append(entry)

    report["sections"]["scores"] = scored
    report["sections"]["worst"] = worst

    # --- performance ------------------------------------------------------------
    perf: dict[str, Any] = {}
    if args.bench:
        # One row per config, across every A/B stdout given. Appended rather than
        # replaced: a rerun of the sweep writes its own file, and both are evidence.
        perf["bench"] = [row for f in args.bench for row in parse_bench(Path(f))]
    if args.vllm_log:
        perf["servers"] = [parse_vllm_log(Path(p)) for p in args.vllm_log]
    report["sections"]["performance"] = perf

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    for name, section in report["sections"].items():
        size = len(section) if isinstance(section, (list, dict)) else 1
        print(f"  {name}: {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
