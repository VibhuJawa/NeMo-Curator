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
pipeline_metrics.py — Shared throughput tracking for all 3-stage pipeline stages.

Each stage imports this module and calls:
  tracker = StageMetrics("stage1a", shard_index=0, n_workers=64, n_gpus=0)
  tracker.start()
  ... do work ...
  tracker.checkpoint(pages_done=1000)   # periodic progress log
  tracker.finish(total_pages=44117)
  tracker.save(output_dir)              # writes metrics_stage1a_shard_0000.json

Stage 4 (metrics aggregator) calls:
  summary = aggregate_pipeline_metrics(output_base_dir)
  print_dashboard(summary)
"""
from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageMetrics:
    stage_name: str          # e.g. "stage1a", "stage1b", "stage2", "stage3"
    shard_index: int
    num_shards: int = 1
    n_workers: int = 0       # CPU workers (for CPU stages)
    n_gpus: int = 0          # GPU count (for GPU stages)
    node_hostname: str = field(default_factory=socket.gethostname)

    # Filled by start/finish
    start_time: float = 0.0
    end_time: float = 0.0
    total_pages: int = 0
    errors: int = 0

    # Stage-specific extras (set by caller)
    extra: dict = field(default_factory=dict)

    def start(self) -> "StageMetrics":
        self.start_time = time.perf_counter()
        print(f"[{self.stage_name}] START shard={self.shard_index}/{self.num_shards} "
              f"node={self.node_hostname} workers={self.n_workers} gpus={self.n_gpus}",
              flush=True)
        return self

    def checkpoint(self, pages_done: int, label: str = "") -> None:
        if self.start_time == 0:
            return
        elapsed = time.perf_counter() - self.start_time
        rate = pages_done / max(elapsed, 1e-6)
        per_worker = rate / max(self.n_workers or self.n_gpus or 1, 1)
        tag = f" [{label}]" if label else ""
        print(f"[{self.stage_name}{tag}] "
              f"{pages_done:>8,} pages  "
              f"{rate:>8.1f} pages/s/node  "
              f"{per_worker:>7.2f} pages/s/{'gpu' if self.n_gpus else 'worker'}  "
              f"{elapsed:>6.1f}s elapsed",
              flush=True)

    def finish(self, total_pages: int, errors: int = 0) -> "StageMetrics":
        self.end_time = time.perf_counter()
        self.total_pages = total_pages
        self.errors = errors
        elapsed = self.elapsed_s
        rate = total_pages / max(elapsed, 1e-6)
        per_worker = rate / max(self.n_workers or self.n_gpus or 1, 1)
        print(f"[{self.stage_name}] DONE  "
              f"pages={total_pages:,}  "
              f"elapsed={elapsed:.1f}s  "
              f"throughput={rate:.1f} pages/s/node  "
              f"per_{'gpu' if self.n_gpus else 'worker'}={per_worker:.2f} pages/s  "
              f"errors={errors}",
              flush=True)
        return self

    @property
    def elapsed_s(self) -> float:
        t_end = self.end_time if self.end_time else time.perf_counter()
        return max(t_end - self.start_time, 1e-6)

    @property
    def pages_per_s_per_node(self) -> float:
        return self.total_pages / self.elapsed_s

    @property
    def pages_per_s_per_worker(self) -> float:
        denom = self.n_workers or self.n_gpus or 1
        return self.pages_per_s_per_node / denom

    def to_dict(self) -> dict:
        return {
            "stage":                  self.stage_name,
            "shard_index":            self.shard_index,
            "num_shards":             self.num_shards,
            "node_hostname":          self.node_hostname,
            "n_workers":              self.n_workers,
            "n_gpus":                 self.n_gpus,
            "total_pages":            self.total_pages,
            "errors":                 self.errors,
            "elapsed_s":              round(self.elapsed_s, 3),
            "pages_per_s_per_node":   round(self.pages_per_s_per_node, 2),
            "pages_per_s_per_worker": round(self.pages_per_s_per_worker, 4),
            **self.extra,
        }

    def save(self, output_dir: str) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"metrics_{self.stage_name}_shard_{self.shard_index:04d}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: aggregate all stage metrics into a dashboard
# ─────────────────────────────────────────────────────────────────────────────

def load_all_metrics(output_base: str) -> list[dict]:
    """Load all metrics_*.json files from all stage output dirs."""
    base = Path(output_base)
    all_metrics = []
    for json_file in sorted(base.rglob("metrics_stage*.json")):
        try:
            all_metrics.append(json.loads(json_file.read_text()))
        except Exception:
            pass
    return all_metrics


def aggregate_pipeline_metrics(output_base: str) -> dict:
    """Aggregate per-shard metrics into per-stage totals."""
    records = load_all_metrics(output_base)

    by_stage: dict[str, list[dict]] = {}
    for r in records:
        by_stage.setdefault(r["stage"], []).append(r)

    summary = {}
    for stage, shards in by_stage.items():
        total_pages = sum(s["total_pages"] for s in shards)
        total_elapsed = max(s["elapsed_s"] for s in shards)  # wall clock = max (parallel)
        n_shards = len(shards)
        n_workers = shards[0].get("n_workers", 0)
        n_gpus    = shards[0].get("n_gpus", 0)
        errors    = sum(s.get("errors", 0) for s in shards)

        # Wall-clock throughput: total pages / max elapsed (parallel runs)
        wall_rate = total_pages / max(total_elapsed, 1e-6)
        per_unit  = wall_rate / max(n_workers or n_gpus or 1, 1)

        summary[stage] = {
            "stage":                  stage,
            "n_shards":               n_shards,
            "total_pages":            total_pages,
            "wall_elapsed_s":         round(total_elapsed, 1),
            "pages_per_s_per_node":   round(wall_rate, 1),
            "pages_per_s_per_worker": round(per_unit, 3),
            "n_workers_per_node":     n_workers,
            "n_gpus_per_node":        n_gpus,
            "errors":                 errors,
            "extra": {k: v for s in shards for k, v in s.items()
                      if k not in {"stage","shard_index","num_shards","node_hostname",
                                   "n_workers","n_gpus","total_pages","errors",
                                   "elapsed_s","pages_per_s_per_node","pages_per_s_per_worker"}},
        }
    return summary


def print_dashboard(summary: dict, output_base: str = "") -> None:
    """Print a clear per-stage throughput dashboard."""
    STAGES_ORDER = ["stage1a", "stage1b", "stage1c", "stage2", "stage2b", "stage3"]

    print()
    print("=" * 78)
    print("  PIPELINE THROUGHPUT DASHBOARD")
    if output_base:
        print(f"  Output: {output_base}")
    print("=" * 78)
    print(f"  {'Stage':<12} {'Pages':>10} {'Wall(s)':>8} {'pages/s/node':>14} "
          f"{'pages/s/worker':>16} {'Workers':>8} {'GPUs':>5} {'Errors':>7}")
    print("  " + "-" * 76)

    total_pages_all = 0
    for stage in STAGES_ORDER:
        if stage not in summary:
            continue
        s = summary[stage]
        total_pages_all = max(total_pages_all, s["total_pages"])
        worker_label = f"{s['n_workers_per_node']}×CPU" if s["n_workers_per_node"] else ""
        gpu_label    = f"{s['n_gpus_per_node']}×GPU"     if s["n_gpus_per_node"]    else ""
        print(f"  {stage:<12} "
              f"{s['total_pages']:>10,} "
              f"{s['wall_elapsed_s']:>8.1f} "
              f"{s['pages_per_s_per_node']:>14.1f} "
              f"{s['pages_per_s_per_worker']:>16.3f} "
              f"{worker_label:>8} "
              f"{gpu_label:>5} "
              f"{s['errors']:>7}")

    print("  " + "-" * 76)

    # End-to-end
    all_elapsed = sum(summary.get(s, {}).get("wall_elapsed_s", 0) for s in STAGES_ORDER)
    if total_pages_all > 0 and all_elapsed > 0:
        e2e_rate = total_pages_all / all_elapsed
        # Projected for full CC-MAIN (2.4B pages) at this throughput with N nodes
        n_shards  = max(summary.get(s, {}).get("n_shards", 1) for s in STAGES_ORDER)
        print(f"\n  End-to-end wall time (sequential):  {all_elapsed:.0f}s")
        print(f"  Effective throughput (1 node):       {e2e_rate:.1f} pages/s/node")

        FULL_CC = 2_385_603_949
        for n_nodes in [1, 10, 80]:
            t_full = FULL_CC / (e2e_rate * n_nodes)
            print(f"  Full CC-MAIN @ {n_nodes:>2} nodes:           "
                  f"{t_full/3600:>6.1f}h  ({t_full/86400:.1f} days)")

    # Call reduction
    if "stage1b" in summary:
        s1b = summary["stage1b"]
        n_reps = s1b["extra"].get("representative_pages", 0)
        n_sing = s1b["extra"].get("singleton_pages", 0)
        gpu_pg = n_reps + n_sing
        call_red = 1.0 - gpu_pg / max(s1b["total_pages"], 1)
        print(f"\n  LLM call reduction (Stage 1b):       {call_red*100:.1f}%")
        print(f"    Representatives:  {n_reps:>8,}  ({n_reps/max(s1b['total_pages'],1)*100:.1f}%)")
        print(f"    Singletons:       {n_sing:>8,}  ({n_sing/max(s1b['total_pages'],1)*100:.1f}%)")
        print(f"    Pages skip LLM:   {s1b['total_pages']-gpu_pg:>8,}  "
              f"({(1-call_red)*100:.1f}%)")

    # Stage 2 setup vs inference breakdown
    if "stage2" in summary:
        s2 = summary["stage2"]
        ex = s2.get("extra", {})
        setup_s = ex.get("setup_time_s", 0)
        infer_s = ex.get("inference_time_s", s2.get("wall_elapsed_s", 0))
        pure_rate = ex.get("pure_inference_pages_per_s", s2["pages_per_s_per_node"])
        wall_rate = ex.get("wall_pages_per_s_incl_startup", s2["pages_per_s_per_node"])
        print(f"\n  Stage 2 timing breakdown:")
        print(f"    Setup (Ray + model load):  {setup_s:>8.1f}s")
        print(f"    Inference only:            {infer_s:>8.1f}s")
        print(f"    Pure inference throughput: {pure_rate:>8.1f} pages/s/node")
        print(f"    Wall throughput (w/ setup):{wall_rate:>8.1f} pages/s/node")

    # Stage 3 propagation method breakdown
    if "stage3" in summary:
        s3 = summary["stage3"]
        ex = s3.get("extra", {})
        total = max(s3["total_pages"], 1)
        n_xpath  = ex.get("xpath_pages", 0)
        n_lbp    = ex.get("layout_batch_parser_pages", 0)
        n_rep    = ex.get("representative_pages", 0)
        n_sing   = ex.get("singleton_pages", 0)
        n_succ   = ex.get("success_pages", n_xpath + n_lbp + n_rep + n_sing)
        n_fall   = s3["total_pages"] - n_succ
        print(f"\n  Propagation method breakdown (Stage 3):")
        for method, n in [("xpath",               n_xpath),
                           ("layout_batch_parser", n_lbp),
                           ("representative",      n_rep),
                           ("singleton",           n_sing),
                           ("fallback",            n_fall)]:
            print(f"    {method:<22} {n:>8,}  ({n/total*100:.1f}%)")

    print("=" * 78)
