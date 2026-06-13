#!/usr/bin/env bash
# =============================================================================
# run_mineru_pipeline.sh — 3-stage MinerU-HTML extraction pipeline orchestrator
#
# Usage:
#   bash run_mineru_pipeline.sh <INPUT> <OUTPUT> <MODE>
#
#   INPUT  — path to the input manifest parquet (url + html columns)
#   OUTPUT — base output directory (shared filesystem path)
#   MODE   — smoke  -> 1 shard  (fast validation)
#             fleet -> 80 shards (full production run)
#
# Job chain — streaming (aftercorr) dependencies: array task K of stage N+1
# starts as soon as array task K of stage N succeeds, not after all N tasks finish.
# This eliminates idle GPU time between stage transitions (~28% wall-clock savings
# at fleet scale). JOB4 keeps afterok because it needs all shards to aggregate.
#
#   JOB1a (Stage 1a): CPU array  — DOM feature extraction (get_feature)
#   JOB1b (Stage 1b): GPU array  — cuML DBSCAN clustering + representative selection
#   JOB_GPU (combined): GPU array — Stage 1c+2+2b in one job (no intermediate parquet)
#   JOB3  (Stage 3):  CPU array  — two-tier LayoutBatchParser propagation to siblings
#   JOB4  (Stage 4):  1 CPU job  — merge metrics, print call-reduction report
#
# stage3b_fallback_llm.py (re-infer propagation failures with the LLM) is run
# manually after the chain when you want baseline-parity F1; see the README.
#
# Configure the environment via these variables before running:
#   VENV_CPU   path to a venv with cuml/cupy + llm_web_kit + mineru_html (CPU + Stage 1b)
#   VENV_GPU   path to a venv with vllm (Stage 2 GPU inference)
#   HF_CACHE   HuggingFace cache directory ($HF_HOME)
#   MODEL      MinerU-HTML model id
#   SLURM_ACCOUNT, CPU_PARTITION, GPU_PARTITION  Slurm scheduling knobs
#   ENV_SETUP  optional path to a script sourced at the top of every job
#
# Smoke test command:
#   bash run_mineru_pipeline.sh /path/to/manifest.parquet /path/to/output smoke
# =============================================================================

set -eu

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
INPUT="${1:?Usage: $0 <INPUT_PARQUET> <OUTPUT_DIR> <MODE: smoke|fleet>}"
OUTPUT="${2:?Usage: $0 <INPUT_PARQUET> <OUTPUT_DIR> <MODE: smoke|fleet>}"
MODE="${3:?Usage: $0 <INPUT_PARQUET> <OUTPUT_DIR> <MODE: smoke|fleet>}"

case "${MODE}" in
    smoke) N_SHARDS=1  ;;
    fleet) N_SHARDS=80 ;;
    *)
        echo "ERROR: MODE must be 'smoke' or 'fleet', got: '${MODE}'" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# venvs: CPU stages + Stage 1b use a cuML/cupy + llm_web_kit/mineru_html venv;
# Stage 2 uses a vllm venv. Override these to point at your environments.
VENV_CPU="${VENV_CPU:?set VENV_CPU to a venv with cuml/cupy + llm_web_kit + mineru_html}"
VENV_GPU="${VENV_GPU:?set VENV_GPU to a venv with vllm}"
PYTHON_CPU="${VENV_CPU}/bin/python3"
PYTHON_GPU="${VENV_GPU}/bin/python3"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
MODEL="${MODEL:-opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact}"
ACCOUNT="${SLURM_ACCOUNT:?set SLURM_ACCOUNT}"
CPU_PARTITION="${CPU_PARTITION:-cpu}"
GPU_PARTITION="${GPU_PARTITION:-batch}"
# Optional environment setup sourced at the top of every Slurm job.
ENV_SETUP="${ENV_SETUP:-}"

# ---------------------------------------------------------------------------
# Derived output dirs
# ---------------------------------------------------------------------------
STAGE1A_OUT="${OUTPUT}/stage1a"   # CPU feature extraction
STAGE1_OUT="${OUTPUT}/stage1b"    # GPU DBSCAN cluster assignments
STAGE1C_OUT="${OUTPUT}/stage1c"   # CPU: simplify + build_prompt (NEW)
STAGE2_OUT="${OUTPUT}/stage2"     # GPU: vLLM inference only (NEW lean version)
STAGE2B_OUT="${OUTPUT}/stage2b"   # CPU: map_parser_cls + convert2content (NEW)
STAGE3_OUT="${OUTPUT}/stage3"     # CPU: XPath propagation
LOGS_DIR="${OUTPUT}/logs"
SBATCH_DIR="${OUTPUT}/sbatch_scripts"

mkdir -p "${STAGE1A_OUT}" "${STAGE1_OUT}" "${STAGE1C_OUT}" "${STAGE2_OUT}" "${STAGE2B_OUT}" "${STAGE3_OUT}" "${LOGS_DIR}" "${SBATCH_DIR}"

LAST_IDX=$(( N_SHARDS - 1 ))

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
log() { printf '[pipeline] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# JOB1a — Stage 1a: CPU-only DOM feature extraction
# ---------------------------------------------------------------------------
log "Submitting JOB1a (Stage 1a CPU feature extraction, ${N_SHARDS} shards)..."

STAGE1A_OUT="${OUTPUT}/stage1a"
mkdir -p "${STAGE1A_OUT}"

S1A_SCRIPT="${SBATCH_DIR}/stage1a.sh"
cat > "${S1A_SCRIPT}" << SCRIPT_EOF
#!/usr/bin/env bash
#SBATCH --job-name=s1a-feat-${MODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH --time=01:00:00
#SBATCH --array=0-${LAST_IDX}
#SBATCH --output=${LOGS_DIR}/s1a_%04a.out
#SBATCH --error=${LOGS_DIR}/s1a_%04a.err

set -eu
[ -n "${ENV_SETUP}" ] && source "${ENV_SETUP}" 2>/dev/null || true
export PYTHONPATH='${SCRIPT_DIR}:\${PYTHONPATH:-}'

echo "=== Stage 1a (CPU feature extraction) task \${SLURM_ARRAY_TASK_ID}/${LAST_IDX} on \$(hostname) ==="
'${PYTHON_CPU}' '${SCRIPT_DIR}/stage1a_feature_extraction.py' \
    --input       '${INPUT}' \
    --output      '${STAGE1A_OUT}' \
    --shard-index \${SLURM_ARRAY_TASK_ID} \
    --num-shards  ${N_SHARDS} \
    --workers     \${SLURM_CPUS_PER_TASK:-62}
echo "=== Stage 1a task \${SLURM_ARRAY_TASK_ID} DONE ==="
SCRIPT_EOF

JOB1A=$(sbatch --parsable "${S1A_SCRIPT}")
log "JOB1a submitted: ${JOB1A}  (CPU-only: get_feature() × 64 workers)"

# ---------------------------------------------------------------------------
# JOB1b — Stage 1b: GPU-only DBSCAN clustering on pre-computed features
# ---------------------------------------------------------------------------
log "Submitting JOB1b (Stage 1b GPU DBSCAN, ${N_SHARDS} shards, depends on ${JOB1A})..."

S1B_SCRIPT="${SBATCH_DIR}/stage1b.sh"
cat > "${S1B_SCRIPT}" << SCRIPT_EOF
#!/usr/bin/env bash
#SBATCH --job-name=s1b-dbscan-${MODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-${LAST_IDX}
#SBATCH --dependency=aftercorr:${JOB1A}
#SBATCH --output=${LOGS_DIR}/s1b_%04a.out
#SBATCH --error=${LOGS_DIR}/s1b_%04a.err

set -eu
[ -n "${ENV_SETUP}" ] && source "${ENV_SETUP}" 2>/dev/null || true
export PYTHONPATH='${SCRIPT_DIR}:\${PYTHONPATH:-}'

# Expose cuML/cupy nvidia libs for GPU DBSCAN
SITE_PKGS='${VENV_CPU}/lib/python3.12/site-packages'
for pkg_dir in "\${SITE_PKGS}/nvidia"/*/lib; do
    [ -d "\${pkg_dir}" ] && export LD_LIBRARY_PATH="\${pkg_dir}:\${LD_LIBRARY_PATH:-}"
done

echo "=== Stage 1b (GPU DBSCAN, \$(nvidia-smi -L | wc -l) GPUs) task \${SLURM_ARRAY_TASK_ID}/${LAST_IDX} on \$(hostname) ==="
nvidia-smi -L
'${PYTHON_CPU}' '${SCRIPT_DIR}/stage1b_gpu_dbscan.py' \
    --input       '${STAGE1A_OUT}' \
    --output      '${STAGE1_OUT}' \
    --shard-index \${SLURM_ARRAY_TASK_ID} \
    --num-shards  ${N_SHARDS}
echo "=== Stage 1b task \${SLURM_ARRAY_TASK_ID} DONE ==="
SCRIPT_EOF

JOB1=$(sbatch --parsable "${S1B_SCRIPT}")
log "JOB1b submitted: ${JOB1}  (GPU-only: cuML DBSCAN × 8 GPUs, depends on ${JOB1A})"

# ---------------------------------------------------------------------------
# JOB_GPU — Stage 1c + 2 + 2b: combined GPU pipeline (no intermediate parquet)
#
# Eliminates 2 parquet round-trips and 2 Slurm queue waits vs the old 3-job design.
# stage_gpu_pipeline.py runs simplify+prompt → vLLM offline → parse+template in one
# GPU job. See STREAMING_ARCHITECTURE.md for the design rationale.
# ---------------------------------------------------------------------------
log "Submitting JOB_GPU (Stage 1c+2+2b combined GPU pipeline, ${N_SHARDS} shards, depends on ${JOB1})..."

S_GPU_SCRIPT="${SBATCH_DIR}/stage_gpu.sh"
cat > "${S_GPU_SCRIPT}" << SCRIPT_EOF
#!/usr/bin/env bash
#SBATCH --job-name=s-gpu-${MODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --array=0-${LAST_IDX}
#SBATCH --dependency=aftercorr:${JOB1}
#SBATCH --output=${LOGS_DIR}/s_gpu_%04a.out
#SBATCH --error=${LOGS_DIR}/s_gpu_%04a.err

set -eu
[ -n "${ENV_SETUP}" ] && source "${ENV_SETUP}" 2>/dev/null || true
export HF_HOME='${HF_CACHE}'
export TRANSFORMERS_CACHE='${HF_CACHE}'
export PYTHONPATH='${SCRIPT_DIR}:\${PYTHONPATH:-}'

echo "=== GPU Pipeline (1c+2+2b combined) task \${SLURM_ARRAY_TASK_ID}/${LAST_IDX} on \$(hostname) ==="
nvidia-smi -L
'${PYTHON_GPU}' '${SCRIPT_DIR}/stage_gpu_pipeline.py' \
    --input          '${STAGE1_OUT}' \
    --output         '${STAGE2B_OUT}' \
    --shard-index    \${SLURM_ARRAY_TASK_ID} \
    --num-shards     ${N_SHARDS} \
    --kv-cache-dtype fp8 \
    --model          '${MODEL}' \
    --hf-cache       '${HF_CACHE}'
echo "=== GPU Pipeline task \${SLURM_ARRAY_TASK_ID} DONE ==="
SCRIPT_EOF

JOB2B=$(sbatch --parsable "${S_GPU_SCRIPT}")
# JOB2B variable kept for compatibility with JOB3 dependency below
log "JOB_GPU submitted: ${JOB2B}  (GPU: 1c+2+2b combined, no intermediate parquet, kv-fp8)"
JOB1C=${JOB2B}; JOB2=${JOB2B}  # aliases for the old stage variable names

# ---------------------------------------------------------------------------
# JOB3 — Stage 3: CPU propagation array (depends on JOB2)
# ---------------------------------------------------------------------------
log "Submitting JOB3 (Stage 3 CPU propagation, ${N_SHARDS} shards, depends on ${JOB2B})..."

S3_SCRIPT="${SBATCH_DIR}/stage3.sh"
cat > "${S3_SCRIPT}" << SCRIPT_EOF
#!/usr/bin/env bash
#SBATCH --job-name=s3-prop-${MODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --array=0-${LAST_IDX}
#SBATCH --dependency=aftercorr:${JOB2B}
#SBATCH --output=${LOGS_DIR}/s3_%04a.out
#SBATCH --error=${LOGS_DIR}/s3_%04a.err

set -eu
[ -n "${ENV_SETUP}" ] && source "${ENV_SETUP}" 2>/dev/null || true
export PYTHONPATH='${SCRIPT_DIR}:\${PYTHONPATH:-}'

# Expose cuML libs for any optional GPU fallback in stage3
SITE_PKGS='${VENV_CPU}/lib/python3.12/site-packages'
for pkg_dir in "\${SITE_PKGS}/nvidia"/*/lib "\${SITE_PKGS}/cuml"/*/lib; do
    [ -d "\${pkg_dir}" ] && export LD_LIBRARY_PATH="\${pkg_dir}:\${LD_LIBRARY_PATH:-}"
done

echo "=== Stage 3 task \${SLURM_ARRAY_TASK_ID}/${LAST_IDX} on \$(hostname) ==="

'${PYTHON_CPU}' '${SCRIPT_DIR}/stage3_cpu_propagation.py' \
    --cluster-manifest  '${STAGE1_OUT}' \
    --inference-results '${STAGE2B_OUT}' \
    --output-dir        '${STAGE3_OUT}' \
    --shard-index       \${SLURM_ARRAY_TASK_ID} \
    --num-shards        ${N_SHARDS} \
    --num-workers       \${SLURM_CPUS_PER_TASK:-64}
echo "=== Stage 3 task \${SLURM_ARRAY_TASK_ID} DONE ==="
SCRIPT_EOF

JOB3=$(sbatch --parsable "${S3_SCRIPT}")
log "JOB3 submitted: ${JOB3}"

# ---------------------------------------------------------------------------
# JOB4 — Merge + metrics (1 job, depends on JOB3)
# ---------------------------------------------------------------------------
log "Submitting JOB4 (merge + metrics, depends on ${JOB3})..."

S4_SCRIPT="${SBATCH_DIR}/stage4_metrics.sh"
cat > "${S4_SCRIPT}" << SCRIPT_EOF
#!/usr/bin/env bash
#SBATCH --job-name=s4-metrics-${MODE}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:${JOB3}
#SBATCH --output=${LOGS_DIR}/s4_metrics_%j.out
#SBATCH --error=${LOGS_DIR}/s4_metrics_%j.err

set -eu
[ -n "${ENV_SETUP}" ] && source "${ENV_SETUP}" 2>/dev/null || true
export PYTHONPATH='${SCRIPT_DIR}:\${PYTHONPATH:-}'

echo '=== Stage 4 merge + metrics ==='

# Use pipeline_metrics.py dashboard for unified throughput reporting
'${PYTHON_CPU}' - << 'PYEOF'
import sys, json, pathlib
sys.path.insert(0, '${SCRIPT_DIR}')
from pipeline_metrics import print_dashboard

OUTPUT = pathlib.Path('${OUTPUT}')

# Collect metrics from all stages.
# pipeline_metrics.py writes metrics_stageXX_shard_NNNN.json in each stage output dir.
STAGE_DIRS = [(name, OUTPUT / name) for name in
              ('stage1a', 'stage1b', 'stage1c', 'stage2', 'stage2b', 'stage3')]

all_metrics = []
for _, d in STAGE_DIRS:
    for f in sorted(d.glob('metrics_stage*.json')) if d.exists() else []:
        try:
            all_metrics.append(json.loads(f.read_text()))
        except Exception:
            pass

# Fall back to old-style metrics if pipeline_metrics not yet wired in all stages
def load_old_metrics(d, stage_name):
    ms = []
    if not d.exists():
        return ms
    for f in sorted(d.glob('metrics_shard_*.json')):
        try:
            m = json.loads(f.read_text())
            m['stage'] = stage_name
            if 'n_workers' not in m:
                m['n_workers'] = 64
            if 'n_gpus' not in m:
                m['n_gpus'] = 8 if 'gpu' in stage_name else 0
            ms.append(m)
        except Exception:
            pass
    return ms

for stage_name, d in STAGE_DIRS:
    if not any(m['stage'] == stage_name for m in all_metrics):
        all_metrics.extend(load_old_metrics(d, stage_name))

# Write unified metrics file
(OUTPUT / 'all_stage_metrics.json').write_text(json.dumps(all_metrics, indent=2))

# Aggregate per-shard metrics into per-stage summaries (same shape as
# pipeline_metrics.aggregate_pipeline_metrics, but over our in-memory list).
by_stage = {}
for m in all_metrics:
    by_stage.setdefault(m['stage'], []).append(m)

summary = {}
for stage, shards in by_stage.items():
    total_pages = sum(s.get('total_pages', 0) for s in shards)
    wall_elapsed = max(s.get('elapsed_s', 0) for s in shards)
    n_workers = shards[0].get('n_workers', 0)
    n_gpus    = shards[0].get('n_gpus', 0)
    errors    = sum(s.get('errors', 0) for s in shards)
    wall_rate = total_pages / max(wall_elapsed, 1e-6)
    per_unit  = wall_rate / max(n_workers or n_gpus or 1, 1)
    extra = {k: v for s in shards for k, v in s.items()
             if k not in {'stage','shard_index','num_shards','node_hostname',
                          'n_workers','n_gpus','total_pages','errors',
                          'elapsed_s','pages_per_s_per_node','pages_per_s_per_worker'}}
    summary[stage] = {
        'stage': stage, 'n_shards': len(shards),
        'total_pages': total_pages, 'wall_elapsed_s': round(wall_elapsed, 1),
        'pages_per_s_per_node': round(wall_rate, 1),
        'pages_per_s_per_worker': round(per_unit, 4),
        'n_workers_per_node': n_workers, 'n_gpus_per_node': n_gpus,
        'errors': errors, 'extra': extra,
    }

print_dashboard(summary, output_base=str(OUTPUT))

# Save pipeline summary
out_path = OUTPUT / 'pipeline_summary.json'
out_path.write_text(json.dumps(summary, indent=2))
print(f'\n  Full summary: {out_path}')

# Propagation method value_counts from Stage 3 output parquet
import glob as _pglob
s3_parquets = sorted(_pglob.glob(str(OUTPUT / 'stage3' / 'shard_*.parquet')))
if s3_parquets:
    try:
        import pandas as _pd
        # read only propagation_method column, tolerating missing
        frames = []
        for f in s3_parquets:
            try:
                df_s = _pd.read_parquet(f, columns=['propagation_method'])
                frames.append(df_s)
            except Exception:
                pass
        if frames:
            combined = _pd.concat(frames, ignore_index=True)
            vc = combined['propagation_method'].value_counts()
            total_s3 = len(combined)
            print(f'\n  Stage 3 propagation_method value_counts ({total_s3:,} total rows):')
            for method, count in vc.items():
                print(f'    {str(method):<25} {count:>10,}  ({count/total_s3*100:.2f}%)')
        else:
            print('\n  Stage 3 parquets found but no propagation_method column readable.')
    except Exception as _e:
        print(f'\n  WARNING: could not read Stage 3 propagation_method column: {_e}')
else:
    print('\n  No Stage 3 shard parquets found for propagation_method breakdown.')
PYEOF

echo '=== Stage 4 DONE ==='
SCRIPT_EOF

JOB4=$(sbatch --parsable "${S4_SCRIPT}")
log "JOB4 submitted: ${JOB4}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
printf '\n'
printf '=%.0s' {1..68}
printf '\n'
printf '  Pipeline submitted (%s mode, %d shards)\n' "${MODE}" "${N_SHARDS}"
printf '=%.0s' {1..68}
printf '\n'
printf '  INPUT:      %s\n' "${INPUT}"
printf '  OUTPUT:     %s\n' "${OUTPUT}"
printf '  Stage 1a:   JOB %-12s  (CPU,   64 CPUs — get_feature())\n'              "${JOB1A}"
printf '  Stage 1b:   JOB %-12s  (GPU,   8xH100 — cuML DBSCAN)\n'              "${JOB1}"
printf '  Stage 1c:   JOB %-12s  (CPU,   64 CPUs — simplify+build_prompt)\n'   "${JOB1C}"
printf '  Stage 2:    JOB %-12s  (GPU,   8xH100 — vLLM inference ONLY)\n'      "${JOB2}"
printf '  Stage 2b:   JOB %-12s  (CPU,   64 CPUs — map_parser_cls+content)\n'  "${JOB2B}"
printf '  Stage 3:    JOB %-12s  (CPU,   64 CPUs — XPath propagation)\n'       "${JOB3}"
printf '  Stage 4:    JOB %-12s  (CPU,   metrics dashboard)\n'                 "${JOB4}"
printf '\n'
printf '  Monitor:  squeue -u "$USER" --format="%%.10i %%.20j %%.8T %%.10M %%R"\n'
printf '  Stage 2 log: %s/s2_0000.out\n' "${LOGS_DIR}"
printf '  Final metrics: %s/pipeline_summary.json\n' "${OUTPUT}"
printf '=%.0s' {1..68}
printf '\n'
