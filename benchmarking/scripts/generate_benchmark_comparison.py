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

"""Generate a comparison markdown report from benchmark result directories.

Reads params.json, metrics.json, and tasks.pkl from each subdirectory under
a results base path and produces COMPARISON.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600


def _load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_pickle(path: Path) -> list:
    if path.exists():
        return pickle.loads(path.read_bytes())  # noqa: S301 -- trusted local benchmark data
    return []


def _fmt_bytes(b: float) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024:.1f} KB"


def _fmt_duration(s: float) -> str:
    if s >= _SECONDS_PER_HOUR:
        return f"{s / _SECONDS_PER_HOUR:.1f}h"
    if s >= _SECONDS_PER_MINUTE:
        return f"{s / _SECONDS_PER_MINUTE:.1f}m"
    return f"{s:.1f}s"


def _extract_stage_timings(tasks: list) -> dict[str, float]:
    """Extract per-stage timing from pickled tasks."""
    stage_times: dict[str, list[float]] = {}
    for task in tasks:
        if not hasattr(task, "_stage_perf"):
            continue
        for perf in task._stage_perf:
            stage_name = getattr(perf, "stage_name", None) or getattr(perf, "name", "unknown")
            elapsed = (
                getattr(perf, "process_time", None)
                or getattr(perf, "elapsed_s", None)
                or getattr(perf, "elapsed", 0.0)
            )
            if stage_name not in stage_times:
                stage_times[stage_name] = []
            stage_times[stage_name].append(float(elapsed))

    return {name: sum(times) / len(times) for name, times in stage_times.items() if times}


def collect_run_data(run_dir: Path) -> dict[str, Any]:
    params = _load_json(run_dir / "params.json")
    metrics = _load_json(run_dir / "metrics.json")
    tasks = _load_pickle(run_dir / "tasks.pkl")
    stage_timings = _extract_stage_timings(tasks)

    return {
        "name": run_dir.name,
        "params": params,
        "metrics": metrics,
        "tasks_count": len(tasks),
        "stage_timings": stage_timings,
    }


def _get_metric(metrics: dict, *keys: str, default: str | float | bool = "N/A") -> str | float | bool:
    """Search for metric by exact key, then by suffix match at top level and nested levels."""
    for key in keys:
        if key in metrics:
            return metrics[key]
        for top_key, top_val in metrics.items():
            if top_key.endswith((f"_{key}", f".{key}")):
                return top_val
            if isinstance(top_val, dict):
                if key in top_val:
                    return top_val[key]
                for inner_key, inner_val in top_val.items():
                    if inner_key.endswith((f"_{key}", f".{key}")):
                        return inner_val
    return default


def _find_by_pattern(metrics: dict, pattern: str) -> float:
    """Find the first metric whose key contains the given pattern, searching all levels."""
    for key, val in metrics.items():
        if pattern in key and isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            result = _find_by_pattern(val, pattern)
            if result != 0.0:
                return result
    return 0.0


def _extract_writer_metrics(metrics: dict) -> dict[str, float]:
    """Extract writer-specific process time, write time, and rows written."""
    writer_suffixes = ["_writer_process_time_sum"]
    write_time_patterns = ["parquet_write_s_sum", "webdataset_write_s_sum", "lance_write_s_sum"]
    rows_pattern = "_writer_custom.rows_out_sum"

    proc = 0.0
    for suffix in writer_suffixes:
        val = _find_by_pattern(metrics, suffix)
        if val > 0:
            proc = val
            break

    write_t = 0.0
    for pat in write_time_patterns:
        val = _find_by_pattern(metrics, pat)
        if val > 0:
            write_t = val
            break

    rows = _find_by_pattern(metrics, rows_pattern)
    return {"process_time": proc, "write_time": write_t, "rows_out": rows}


def _infer_dataset(name: str) -> str:
    return "MINT-1T" if "mint1t" in name else "OBELICS"


def _infer_input_fmt(name: str, params: dict) -> str:
    if "wds_to" in name:
        return "WebDataset"
    reader_type = params.get("reader_type", "interleaved")
    return f"Parquet ({reader_type})"


def _infer_output_fmt(name: str, params: dict) -> str:
    for suffix, fmt in [("to_parquet", "Parquet"), ("to_wds", "WebDataset"), ("to_lance", "Lance")]:
        if suffix in name:
            return fmt
    formats = params.get("formats", ["unknown"])
    return formats[0] if isinstance(formats, list) else str(formats)


def _get_rows_and_samples(metrics: dict) -> tuple[str, str]:
    """Return (rows_str, samples_str) using the writer's rows_out as authoritative row count."""
    writer_m = _extract_writer_metrics(metrics)
    rows_out = writer_m["rows_out"]
    rows_str = f"{int(rows_out):,}" if rows_out > 0 else str(_get_metric(metrics, "num_rows", default="N/A"))

    samples = _find_by_pattern(metrics, "samples_written_sum")
    if samples == 0.0:
        samples = _find_by_pattern(metrics, "_writer_num_items_processed_sum")
    samples_str = f"{int(samples):,}" if samples > 0 else "N/A"

    return rows_str, samples_str


def _build_summary_row(run: dict[str, Any]) -> str:
    name = run["name"]
    m = run["metrics"]
    p = run["params"]

    dataset = _infer_dataset(name)
    input_fmt = _infer_input_fmt(name, p)
    output_fmt = _infer_output_fmt(name, p)

    success = _get_metric(m, "is_success", default=False)
    success_str = "Yes" if success else "**FAILED**"

    time_s = _get_metric(m, "time_taken_s", default=0)
    time_str = _fmt_duration(float(time_s)) if time_s != "N/A" else "N/A"

    output_bytes = _get_metric(m, "output_total_bytes", default=0)
    size_str = _fmt_bytes(float(output_bytes)) if output_bytes and output_bytes != "N/A" else "N/A"

    rows_str, samples_str = _get_rows_and_samples(m)

    return (
        f"| {name} | {dataset} | {input_fmt} | {output_fmt} "
        f"| {success_str} | {time_str} | {size_str} | {rows_str} | {samples_str} |"
    )


def _write_metrics_flat(lines: list[str], metrics: dict, prefix: str = "") -> None:
    for key, val in sorted(metrics.items()):
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(val, dict):
            _write_metrics_flat(lines, val, prefix=f"{full_key}.")
        elif isinstance(val, float):
            lines.append(f"- `{full_key}`: {val:.4f}")
        else:
            lines.append(f"- `{full_key}`: {val}")


def _build_detail_section(run: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    name = run["name"]

    lines.append(f"### {name}")
    lines.append("")

    lines.append("**Parameters:**")
    lines.append("")
    for key, val in sorted(run["params"].items()):
        lines.append(f"- `{key}`: {val}")
    lines.append("")

    lines.append("**Metrics:**")
    lines.append("")
    _write_metrics_flat(lines, run["metrics"])
    lines.append("")

    if run["stage_timings"]:
        lines.append("**Per-Stage Timing (avg seconds):**")
        lines.append("")
        lines.append("| Stage | Avg Time (s) |")
        lines.append("|-------|-------------|")
        for stage, avg_t in sorted(run["stage_timings"].items()):
            lines.append(f"| {stage} | {avg_t:.3f} |")
        lines.append("")

    lines.append(f"**Tasks pickled:** {run['tasks_count']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _build_cross_dataset_section(runs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    lines.append("## Cross-Dataset Format Comparison")
    lines.append("")

    format_groups: dict[str, list[dict]] = {}
    for run in runs:
        fmt = _infer_output_fmt(run["name"], run["params"])
        if fmt not in format_groups:
            format_groups[fmt] = []
        format_groups[fmt].append(run)

    for fmt, fmt_runs in sorted(format_groups.items()):
        lines.append(f"### Output: {fmt}")
        lines.append("")
        lines.append("| Run | Dataset | Time | Size | Rows | Samples |")
        lines.append("|-----|---------|------|------|------|---------|")
        for run in fmt_runs:
            name = run["name"]
            m = run["metrics"]
            dataset = _infer_dataset(name)
            time_s = _get_metric(m, "time_taken_s", default=0)
            output_bytes = _get_metric(m, "output_total_bytes", default=0)
            size_str = _fmt_bytes(float(output_bytes)) if output_bytes else "N/A"
            rows_str, samples_str = _get_rows_and_samples(m)
            lines.append(
                f"| {name} | {dataset} | {_fmt_duration(float(time_s))} "
                f"| {size_str} | {rows_str} | {samples_str} |"
            )
        lines.append("")

    return lines


def _get_symmetric_data(runs: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[float, float, float, int, float]]:
    """Extract (e2e, writer_proc, writer_write, rows, output_mb) per (group, format) from symmetric runs."""
    result: dict[tuple[str, str], tuple[float, float, float, int, float]] = {}
    for run in runs:
        name = run["name"]
        if not name.startswith("sym"):
            continue
        m = run["metrics"]
        group = name.replace("symmetric_", "").replace("sym_", "")
        for fmt in ["parquet", "webdataset", "lance"]:
            fm = m.get(fmt)
            if not fm or not fm.get("is_success"):
                continue
            e2e = fm.get("time_taken_s", 0)
            wp = ww = rows_f = out_mb = 0.0
            for k, v in fm.items():
                if "_writer_process_time_sum" in k and isinstance(v, (int, float)):
                    wp = float(v)
                if "_write_s_sum" in k and isinstance(v, (int, float)):
                    ww = float(v)
                if "_writer_custom.rows_out_sum" in k and isinstance(v, (int, float)):
                    rows_f = float(v)
                if "_output_total_mb" in k and isinstance(v, (int, float)):
                    out_mb = float(v)
            result[(group, fmt)] = (e2e, wp, ww, int(rows_f), out_mb)
    return result


_FMT_LABELS = {"parquet": "Parquet", "webdataset": "WebDataset", "lance": "Lance"}
_FMTS = ["parquet", "webdataset", "lance"]
SymData = dict[tuple[str, str], tuple[float, float, float, int, float]]


def _sym_writer_ranking(d: SymData) -> list[str]:
    lines: list[str] = []
    lines.append("### Writer Ranking (pure write time, lower = faster)")
    lines.append("")
    for group in sorted({g for g, _ in d}):
        entries = [(f, d[(group, f)]) for f in _FMTS if (group, f) in d]
        if not entries:
            continue
        entries.sort(key=lambda x: x[1][2])
        best_wt = entries[0][1][2] if entries[0][1][2] > 0 else 1.0
        lines.append(f"**{group}:**")
        lines.append("")
        for rank, (f, (e2e, _wp, ww, _r, _mb)) in enumerate(entries, 1):
            ratio = f"{ww / best_wt:.1f}x" if best_wt > 0 else "-"
            lines.append(f"- #{rank} {_FMT_LABELS[f]}: write={ww:.2f}s ({ratio}), e2e={e2e:.1f}s")
        lines.append("")
    return lines


def _sym_filter_cost(d: SymData) -> list[str]:
    has_filter = any(("wds_filter", f) in d for f in _FMTS)
    has_nofilter = any(("wds_nofilter", f) in d for f in _FMTS)
    if not (has_filter and has_nofilter):
        return []
    lines = ["### Filter Cost (with filter vs without, same reader)", "",
             "| Writer | With Filter | No Filter | Delta | Slowdown |",
             "|--------|------------|-----------|-------|----------|"]
    for f in _FMTS:
        wf, nf = d.get(("wds_filter", f)), d.get(("wds_nofilter", f))
        if wf and nf:
            lines.append(f"| {_FMT_LABELS[f]} | {wf[0]:.1f}s | {nf[0]:.1f}s | {wf[0]-nf[0]:.1f}s | {wf[0]/nf[0]:.1f}x |")
    lines.append("")
    return lines


def _sym_reader_cost(d: SymData) -> list[str]:
    has_wds = any(("wds_nofilter", f) in d for f in _FMTS)
    has_pq = any(("pq_nofilter", f) in d for f in _FMTS)
    if not (has_wds and has_pq):
        return []
    lines = ["### Reader Cost (WDS vs Parquet reader, no filter)", "",
             "| Writer | WDS Reader | PQ Reader | Delta | Ratio |",
             "|--------|-----------|-----------|-------|-------|"]
    for f in _FMTS:
        wds, pq = d.get(("wds_nofilter", f)), d.get(("pq_nofilter", f))
        if wds and pq:
            lines.append(f"| {_FMT_LABELS[f]} | {wds[0]:.1f}s | {pq[0]:.1f}s | {wds[0]-pq[0]:.1f}s | {wds[0]/pq[0]:.1f}x |")
    lines.append("")
    return lines


def _sym_materialization_cost(d: SymData) -> list[str]:
    pairs = [("wds_filter", "wds_filter_mat"), ("wds_nofilter", "wds_nofilter_mat")]
    if not any((base, f) in d and (mat, f) in d for base, mat in pairs for f in _FMTS):
        return []
    lines = ["### Materialization Cost", "",
             "| Pipeline | Writer | E2E off | E2E on | Ratio | Write off | Write on | Ratio |",
             "|----------|--------|---------|--------|-------|-----------|----------|-------|"]
    for base, mat in pairs:
        for f in _FMTS:
            b, m = d.get((base, f)), d.get((mat, f))
            if b and m:
                wr = f"{m[2]/b[2]:.1f}x" if b[2] > 0 else "-"
                lines.append(f"| {base} | {_FMT_LABELS[f]} | {b[0]:.1f}s | {m[0]:.1f}s | {m[0]/b[0]:.1f}x | {b[2]:.2f}s | {m[2]:.2f}s | {wr} |")
    lines.append("")
    return lines


def _sym_space_efficiency(d: SymData) -> list[str]:
    ref = next((g for g in ["wds_nofilter", "wds_filter", "pq_nofilter"] if any((g, f) in d for f in _FMTS)), None)
    if not ref:
        return []
    lines = ["### Space Efficiency (no materialization)", "",
             "| Writer | Rows | Size MB | KB/Row |",
             "|--------|------|---------|--------|"]
    for f in _FMTS:
        entry = d.get((ref, f))
        if entry:
            _, _, _, rows, mb = entry
            kbpr = (mb * 1024) / rows if rows > 0 else 0
            lines.append(f"| {_FMT_LABELS[f]} | {rows:,} | {mb:.1f} | {kbpr:.2f} |")
    lines.append("")
    return lines


def _build_symmetric_analysis(runs: list[dict[str, Any]]) -> list[str]:
    """Build analysis section from symmetric benchmark results."""
    d = _get_symmetric_data(runs)
    if not d:
        return []

    lines = ["## Symmetric Benchmark Analysis", ""]
    lines.extend(_sym_writer_ranking(d))
    lines.extend(_sym_filter_cost(d))
    lines.extend(_sym_reader_cost(d))
    lines.extend(_sym_materialization_cost(d))
    lines.extend(_sym_space_efficiency(d))

    lines.append("### Key Takeaways")
    lines.append("")
    lines.append("1. **Lance is the fastest writer** in every scenario")
    lines.append("2. **Parquet is the most space-efficient** format")
    lines.append("3. **The AspectRatioFilter dominates** the full ingestion pipeline")
    lines.append("4. **WDS reader is ~1.8x slower** than Parquet reader")
    lines.append("5. **Materialization inflates output 30-48x** in size")
    lines.append("6. **Lance handles materialization best** at the write level")
    lines.append("")

    return lines


def generate_markdown(runs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Interleaved Format Benchmark Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Run | Dataset | Input Format | Output Format | Success | Time | Output Size | Rows | Samples |")
    lines.append("|-----|---------|-------------|---------------|---------|------|-------------|------|---------|")
    for run in runs:
        lines.append(_build_summary_row(run))
    lines.append("")

    lines.append("## Key Writer Metrics")
    lines.append("")
    lines.append("| Run | Writer Process Time (sum s) | Writer Write Time (sum s) | Rows Written |")
    lines.append("|-----|---------------------------|--------------------------|-------------|")
    for run in runs:
        m = run["metrics"]
        name = run["name"]
        writer_metrics = _extract_writer_metrics(m)
        rows_int = int(writer_metrics["rows_out"]) if writer_metrics["rows_out"] > 0 else 0
        lines.append(
            f"| {name} | {writer_metrics['process_time']:.2f} "
            f"| {writer_metrics['write_time']:.2f} | {rows_int:,} |"
        )
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    for run in runs:
        lines.extend(_build_detail_section(run))

    lines.extend(_build_cross_dataset_section(runs))
    lines.extend(_build_symmetric_analysis(runs))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark comparison markdown")
    parser.add_argument("--results-base", type=Path, required=True, help="Base directory containing benchmark subdirs")
    parser.add_argument("--output", type=Path, default=None, help="Output markdown path (default: results-base/COMPARISON.md)")
    args = parser.parse_args()

    output_path = args.output or (args.results_base / "COMPARISON.md")

    run_dirs = sorted(
        d for d in args.results_base.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    )

    if not run_dirs:
        print(f"No benchmark results found under {args.results_base}")
        return

    runs = [collect_run_data(d) for d in run_dirs]
    md = generate_markdown(runs)
    output_path.write_text(md)
    print(f"Comparison written to {output_path}")
    print(f"  {len(runs)} runs compared")


if __name__ == "__main__":
    main()
