#!/usr/bin/env bash
# =============================================================================
# report_pipeline_metrics.sh
#
# Fetch and display pipeline metrics from a completed or in-progress run.
#
# Usage:
#   bash report_pipeline_metrics.sh OUTPUT_BASE [nebius-host]
#
# Example:
#   bash report_pipeline_metrics.sh \
#     /lustre/fsw/portfolios/llmservice/users/vjawa/cc_scale_run_20260611_120000 \
#     vjawa@nb-hel-cs-001-vscode-01.nvidia.com
#
# Metrics reported:
#   - LLM calls: representative + singletons + fallbacks vs total pages
#   - Call reduction fraction
#   - GPU time used
#   - Estimated H100-hours for full CC-MAIN-2025-26 snapshot
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_nebius_ssh.sh"

OUTPUT_BASE="${1:?Usage: $0 OUTPUT_BASE [host]}"
HOST="${2:-${HOST:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}}"

resolved_host="$(nebius_resolve_ssh_host "$HOST")"

CLUSTER_ASSIGNMENTS_DIR="${OUTPUT_BASE}/cluster_assignments"
GPU_RESULTS_DIR="${OUTPUT_BASE}/gpu_results"
PROPAGATION_RESULTS_DIR="${OUTPUT_BASE}/propagation_results"
MERGED_RESULTS_DIR="${OUTPUT_BASE}/merged_results"
LOGS_DIR="${OUTPUT_BASE}/logs"

# ── Helper: count parquet rows for a role ─────────────────────────────────────
CACHED_VENV="${CACHED_VENV:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_codex_20260611_221330/.venv}"
PYTHON="${CACHED_VENV}/bin/python3"

sep() { printf '=%.0s' {1..72}; printf '\n'; }
hdr() { printf '\n  [%s]\n' "$*"; }

sep
printf '  MinerU Pipeline Metrics Report\n'
printf '  Output base: %s\n' "${OUTPUT_BASE}"
sep

# ── 1. Check if final metrics JSON exists ─────────────────────────────────────
hdr "Final pipeline metrics (pipeline_metrics.json)"
METRICS_JSON="${MERGED_RESULTS_DIR}/pipeline_metrics.json"
if nebius_ssh_command "$resolved_host" "test -f '${METRICS_JSON}' 2>/dev/null"; then
    nebius_ssh_command "$resolved_host" "cat '${METRICS_JSON}'" | \
        python3 -c "
import json, sys
m = json.load(sys.stdin)
ps = m.get('pipeline_summary', {})
lc = m.get('llm_calls', {})
gt = m.get('gpu_timing', {})
cc = m.get('cc_scale_projection', {})

print()
print('  Pipeline Summary')
print(f\"    Total pages:            {ps.get('total_pages_processed', 0):>14,}\")
print(f\"    Representatives (LLM):  {ps.get('representative_pages', 0):>14,}  ({100*ps.get('representative_pages',0)/max(ps.get('total_pages_processed',1),1):.1f}%)\")
print(f\"    Singletons (LLM):       {ps.get('singleton_pages', 0):>14,}  ({100*ps.get('singleton_pages',0)/max(ps.get('total_pages_processed',1),1):.1f}%)\")
print(f\"    Siblings processed:     {ps.get('sibling_pages', 0):>14,}\")
print(f\"    Propagation success:    {ps.get('propagation_success', 0):>14,}  ({100*ps.get('propagation_success_rate',0):.1f}%)\")
print(f\"    Propagation failures:   {ps.get('propagation_failures', 0):>14,}\")
print()
print('  LLM Call Reduction')
print(f\"    Total LLM calls:        {lc.get('total_llm_calls', 0):>14,}\")
print(f\"    Templated (no LLM):     {lc.get('templated_pages', 0):>14,}\")
print(f\"    Call reduction:         {lc.get('call_reduction_fraction',0):>13.1%}\")
print()
print('  GPU Timing (Stage 2)')
print(f\"    GPU inference time:     {gt.get('total_gpu_inference_s',0)/3600:>13.2f}h\")
print(f\"    GPU pages processed:    {gt.get('total_gpu_pages',0):>14,}\")
print(f\"    Avg throughput:         {gt.get('avg_throughput_pages_s',0):>13.1f} pages/s\")
print()
print('  CC-MAIN-2025-26 Projection (2.4B pages)')
print(f\"    Projected LLM calls:    {cc.get('projected_llm_calls',0):>14,.0f}  ({100*cc.get('projected_llm_calls',0)/cc.get('cc_total_pages',2.4e9):.2f}% of pages)\")
print(f\"    Projected H100-hours:   {cc.get('projected_h100_hours',0):>14,.0f}\")
print(f\"    Baseline H100-hours:    {cc.get('baseline_h100_hours_run_b',0):>14,.0f}  (Run B: every page → LLM)\")
print(f\"    H100-hour reduction:    {cc.get('h100_hour_reduction_vs_baseline',0)*100:>13.1f}%\")
print(f\"    Wall time (64 GPUs):    {cc.get('projected_wall_hours_64gpu',0):>13.1f}h  (budget=48h)\")
"
else
    printf '  (pipeline_metrics.json not yet available — Stage 4 may not have run)\n'
fi

# ── 2. In-progress counters from shard files ──────────────────────────────────
hdr "Shard completion (from metrics JSON files)"

nebius_ssh_command "$resolved_host" "${PYTHON} - '${CLUSTER_ASSIGNMENTS_DIR}' '${GPU_RESULTS_DIR}' '${PROPAGATION_RESULTS_DIR}'" << 'PYEOF'
import json, glob, sys
from pathlib import Path

def count_metrics(directory, label):
    d = Path(directory)
    if not d.exists():
        print(f"  {label}: directory not found ({directory})")
        return
    files = sorted(d.glob("metrics_shard_*.json"))
    n = len(files)
    if n == 0:
        print(f"  {label}: 0 shards complete")
        return
    total_pages = sum(json.loads(p.read_text()).get("total_pages", 0) for p in files)
    elapsed = [json.loads(p.read_text()).get("elapsed_s", 0) for p in files]
    print(f"  {label}: {n} shards complete, {total_pages:,} pages, avg {sum(elapsed)/max(len(elapsed),1):.0f}s/shard")

cluster_dir = sys.argv[1]
gpu_dir     = sys.argv[2]
prop_dir    = sys.argv[3]

count_metrics(cluster_dir,  "Stage 1 (cluster)")
count_metrics(gpu_dir,      "Stage 2 (GPU inference)")
count_metrics(prop_dir,     "Stage 3 (propagation)")
PYEOF

# ── 3. Slurm job status ───────────────────────────────────────────────────────
hdr "Slurm job status (all jobs, user=vjawa)"
nebius_ssh_command "$resolved_host" \
    "squeue -u vjawa --format='%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null | head -40 || true"

# ── 4. Recent Stage 2 GPU log tail ───────────────────────────────────────────
hdr "Recent Stage 2 GPU log (last 20 lines of task 0)"
GPU_LOG="${LOGS_DIR}/s2_gpu_0000.out"
if nebius_ssh_command "$resolved_host" "test -f '${GPU_LOG}' 2>/dev/null"; then
    nebius_ssh_command "$resolved_host" "tail -20 '${GPU_LOG}'"
else
    printf '  (s2_gpu_0000.out not yet available)\n'
fi

# ── 5. Quick H100-hour estimates at different thresholds ─────────────────────
hdr "H100-hour estimates at different clustering thresholds"
python3 - << 'PYEOF'
# Measured baseline: Run B (every page → LLM, 44.7 pages/s, 8 H100s)
# Measured: 44K pages, 19% reduction at threshold=0.95 (Run A naive)
# Target:   60-70% reduction at threshold=0.95 (Run A v2, no validation)

CC_TOTAL    = 2.4e9
BASELINE_TP = 44.7   # pages/s, 8 GPUs → Run B
BASELINE_H100_HOURS = (CC_TOTAL / BASELINE_TP) * 8 / 3600

# MinerU standalone per GPU at TP=1: ~6 pages/s
GPU_TP = 6.0  # pages/s per H100

configs = [
    ("threshold=0.80 (aggressive)", 0.825),   # 82.5% call reduction
    ("threshold=0.90 (balanced)",   0.775),   # 77.5% call reduction
    ("threshold=0.95 (production)", 0.650),   # 65.0% call reduction (our target)
    ("threshold=0.95 Run A naive",  0.212),   # 21.2% (measured Run A)
    ("threshold=0.95 Run B baseline",0.000),  # 0% (no clustering)
]

print(f"  Baseline H100-hours (Run B, 8 GPUs):  {BASELINE_H100_HOURS:>10,.0f}")
print()
print(f"  {'Configuration':<40}  {'Reduction':>10}  {'H100-hours':>11}  {'vs baseline':>11}  {'Wall 64GPU':>10}")
print(f"  {'-'*40}  {'-'*10}  {'-'*11}  {'-'*11}  {'-'*10}")
for name, reduction in configs:
    llm_fraction = 1.0 - reduction
    llm_calls    = CC_TOTAL * llm_fraction
    h100_hours   = (llm_calls / GPU_TP) / 3600
    wall_64gpu_h = llm_calls / (GPU_TP * 64) / 3600
    savings_pct  = (1.0 - h100_hours / BASELINE_H100_HOURS) * 100
    print(f"  {name:<40}  {reduction*100:>9.1f}%  {h100_hours:>11,.0f}  {savings_pct:>10.1f}%  {wall_64gpu_h:>9.1f}h")
PYEOF

sep
printf '  Report complete.\n'
sep
