#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_nebius_ssh.sh
source "${script_dir}/lib_nebius_ssh.sh"

usage() {
  cat >&2 <<'USAGE'
Usage: submit_nebius_dripper_layout_diag.sh [OPTIONS] HOST REMOTE_ENV_DIR BASE_OUTPUT_DIR CANDIDATE_OUTPUT_DIR [RUN_DIR]

Common options:
  --max-rows N
  --example-rows N
  --layout-cluster-threshold X
  --layout-page-signature-mode MODE
  --layout-target-hosts HOST1,HOST2
  --layout-template-propagation-target raw_html|mapped_item_ids
  --layout-template-validation-min-f1 X
  --layout-template-validation-rows N
  --layout-template-validation-signature-mode MODE
  --layout-template-large-cluster-validation-rows N
  --layout-template-large-cluster-min-size N
  --layout-template-min-content-length-ratio X
  --layout-template-max-content-length-ratio X
  --layout-template-failed-layout-fallback-signature-mode MODE
  --layout-template-more-noise-enable 0|1
USAGE
}

account="${SLURM_ACCOUNT:-nemotron_n4_pre}"
partition="${SLURM_PARTITION:-cpu_short}"
cpus_per_task="${CPUS_PER_TASK:-16}"
time_limit="${TIME_LIMIT:-01:00:00}"
max_rows="${DRIPPER_LAYOUT_DIAG_MAX_ROWS:-300}"
example_rows="${DRIPPER_LAYOUT_DIAG_EXAMPLES:-5}"
shard_size="${SHARD_SIZE:-64}"
layout_cluster_threshold="${LAYOUT_CLUSTER_THRESHOLD:-0.99}"
layout_template_min_cluster_size="${LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE:-2}"
layout_template_max_exact_host_pages="${LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES:-0}"
layout_template_large_host_mode="${LAYOUT_TEMPLATE_LARGE_HOST_MODE:-standalone}"
layout_template_max_selected_item_ratio="${LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO:-0.50}"
layout_template_max_rep_selected_item_ratio="${LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO:-0}"
layout_template_more_noise_enable="${LAYOUT_TEMPLATE_MORE_NOISE_ENABLE:-0}"
dynamic_classid_similarity_threshold="${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD:-0.85}"
layout_template_min_consensus_f1="${LAYOUT_TEMPLATE_MIN_CONSENSUS_F1:-0}"
layout_template_validation_rows="${LAYOUT_TEMPLATE_VALIDATION_ROWS:-2}"
layout_template_validation_min_f1="${LAYOUT_TEMPLATE_VALIDATION_MIN_F1:-0.98}"
layout_template_validation_signature_mode="${LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE:-none}"
layout_template_large_cluster_validation_rows="${LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS:-0}"
layout_template_large_cluster_min_size="${LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE:-0}"
layout_template_min_content_length_ratio="${LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO:-0}"
layout_template_max_content_length_ratio="${LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO:-0}"
layout_template_failed_layout_fallback_signature_mode="${LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE:-none}"
layout_template_propagation_target="${LAYOUT_TEMPLATE_PROPAGATION_TARGET:-raw_html}"
layout_diag_variant_modes="${LAYOUT_DIAG_VARIANT_MODES:-}"
layout_page_signature_mode="${LAYOUT_PAGE_SIGNATURE_MODE:-url_shape}"
layout_target_hosts="${LAYOUT_TARGET_HOSTS:-}"
layout_force_host_single_cluster="${LAYOUT_FORCE_HOST_SINGLE_CLUSTER:-0}"
layout_precomputed_manifest="${LAYOUT_PRECOMPUTED_MANIFEST:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      account="$2"
      shift 2
      ;;
    --account=*)
      account="${1#*=}"
      shift
      ;;
    --partition)
      partition="$2"
      shift 2
      ;;
    --partition=*)
      partition="${1#*=}"
      shift
      ;;
    --cpus-per-task)
      cpus_per_task="$2"
      shift 2
      ;;
    --cpus-per-task=*)
      cpus_per_task="${1#*=}"
      shift
      ;;
    --time-limit)
      time_limit="$2"
      shift 2
      ;;
    --time-limit=*)
      time_limit="${1#*=}"
      shift
      ;;
    --max-rows)
      max_rows="$2"
      shift 2
      ;;
    --max-rows=*)
      max_rows="${1#*=}"
      shift
      ;;
    --example-rows)
      example_rows="$2"
      shift 2
      ;;
    --example-rows=*)
      example_rows="${1#*=}"
      shift
      ;;
    --shard-size)
      shard_size="$2"
      shift 2
      ;;
    --shard-size=*)
      shard_size="${1#*=}"
      shift
      ;;
    --layout-cluster-threshold)
      layout_cluster_threshold="$2"
      shift 2
      ;;
    --layout-cluster-threshold=*)
      layout_cluster_threshold="${1#*=}"
      shift
      ;;
    --layout-template-min-cluster-size)
      layout_template_min_cluster_size="$2"
      shift 2
      ;;
    --layout-template-min-cluster-size=*)
      layout_template_min_cluster_size="${1#*=}"
      shift
      ;;
    --layout-template-max-exact-host-pages)
      layout_template_max_exact_host_pages="$2"
      shift 2
      ;;
    --layout-template-max-exact-host-pages=*)
      layout_template_max_exact_host_pages="${1#*=}"
      shift
      ;;
    --layout-template-large-host-mode)
      layout_template_large_host_mode="$2"
      shift 2
      ;;
    --layout-template-large-host-mode=*)
      layout_template_large_host_mode="${1#*=}"
      shift
      ;;
    --layout-template-max-selected-item-ratio)
      layout_template_max_selected_item_ratio="$2"
      shift 2
      ;;
    --layout-template-max-selected-item-ratio=*)
      layout_template_max_selected_item_ratio="${1#*=}"
      shift
      ;;
    --layout-template-max-rep-selected-item-ratio)
      layout_template_max_rep_selected_item_ratio="$2"
      shift 2
      ;;
    --layout-template-max-rep-selected-item-ratio=*)
      layout_template_max_rep_selected_item_ratio="${1#*=}"
      shift
      ;;
    --layout-template-more-noise-enable)
      layout_template_more_noise_enable="$2"
      shift 2
      ;;
    --layout-template-more-noise-enable=*)
      layout_template_more_noise_enable="${1#*=}"
      shift
      ;;
    --dynamic-classid-similarity-threshold)
      dynamic_classid_similarity_threshold="$2"
      shift 2
      ;;
    --dynamic-classid-similarity-threshold=*)
      dynamic_classid_similarity_threshold="${1#*=}"
      shift
      ;;
    --layout-template-min-consensus-f1)
      layout_template_min_consensus_f1="$2"
      shift 2
      ;;
    --layout-template-min-consensus-f1=*)
      layout_template_min_consensus_f1="${1#*=}"
      shift
      ;;
    --layout-template-validation-rows)
      layout_template_validation_rows="$2"
      shift 2
      ;;
    --layout-template-validation-rows=*)
      layout_template_validation_rows="${1#*=}"
      shift
      ;;
    --layout-template-validation-min-f1)
      layout_template_validation_min_f1="$2"
      shift 2
      ;;
    --layout-template-validation-min-f1=*)
      layout_template_validation_min_f1="${1#*=}"
      shift
      ;;
    --layout-template-validation-signature-mode)
      layout_template_validation_signature_mode="$2"
      shift 2
      ;;
    --layout-template-validation-signature-mode=*)
      layout_template_validation_signature_mode="${1#*=}"
      shift
      ;;
    --layout-template-large-cluster-validation-rows)
      layout_template_large_cluster_validation_rows="$2"
      shift 2
      ;;
    --layout-template-large-cluster-validation-rows=*)
      layout_template_large_cluster_validation_rows="${1#*=}"
      shift
      ;;
    --layout-template-large-cluster-min-size)
      layout_template_large_cluster_min_size="$2"
      shift 2
      ;;
    --layout-template-large-cluster-min-size=*)
      layout_template_large_cluster_min_size="${1#*=}"
      shift
      ;;
    --layout-template-min-content-length-ratio)
      layout_template_min_content_length_ratio="$2"
      shift 2
      ;;
    --layout-template-min-content-length-ratio=*)
      layout_template_min_content_length_ratio="${1#*=}"
      shift
      ;;
    --layout-template-max-content-length-ratio)
      layout_template_max_content_length_ratio="$2"
      shift 2
      ;;
    --layout-template-max-content-length-ratio=*)
      layout_template_max_content_length_ratio="${1#*=}"
      shift
      ;;
    --layout-template-failed-layout-fallback-signature-mode)
      layout_template_failed_layout_fallback_signature_mode="$2"
      shift 2
      ;;
    --layout-template-failed-layout-fallback-signature-mode=*)
      layout_template_failed_layout_fallback_signature_mode="${1#*=}"
      shift
      ;;
    --layout-template-propagation-target)
      layout_template_propagation_target="$2"
      shift 2
      ;;
    --layout-template-propagation-target=*)
      layout_template_propagation_target="${1#*=}"
      shift
      ;;
    --layout-page-signature-mode)
      layout_page_signature_mode="$2"
      shift 2
      ;;
    --layout-page-signature-mode=*)
      layout_page_signature_mode="${1#*=}"
      shift
      ;;
    --layout-target-hosts)
      layout_target_hosts="$2"
      shift 2
      ;;
    --layout-target-hosts=*)
      layout_target_hosts="${1#*=}"
      shift
      ;;
    --layout-force-host-single-cluster)
      layout_force_host_single_cluster="$2"
      shift 2
      ;;
    --layout-force-host-single-cluster=*)
      layout_force_host_single_cluster="${1#*=}"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR=unknown_option option=$1" >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage
  exit 2
fi

host="$1"
remote_env_dir="$2"
base_output_dir="$3"
candidate_output_dir="$4"
run_dir="${5:-/lustre/fsw/portfolios/llmservice/users/vjawa/nemo_curator_dripper_layout_diag_$(date -u +%Y%m%d_%H%M%S)}"

diag_py="${script_dir}/remote_dripper_layout_diag.py"
if [[ ! -f "$diag_py" ]]; then
  echo "ERROR=missing_diag_py path=$diag_py" >&2
  exit 2
fi

resolved_host="$(nebius_resolve_ssh_host "$host")"
rsync_host="$(nebius_resolve_rsync_host "$resolved_host")"
rsync_ssh="$(nebius_ssh_command_string "$rsync_host" "${NEBIUS_SSH_CONNECT_TIMEOUT:-30}")"

echo "SUBMIT_LAYOUT_DIAG_BEGIN"
echo "HOST=$host"
echo "RESOLVED_HOST=$resolved_host"
echo "REMOTE_ENV_DIR=$remote_env_dir"
echo "BASE_OUTPUT_DIR=$base_output_dir"
echo "CANDIDATE_OUTPUT_DIR=$candidate_output_dir"
echo "RUN_DIR=$run_dir"
echo "ACCOUNT=$account"
echo "PARTITION=$partition"
echo "CPUS_PER_TASK=$cpus_per_task"
echo "TIME_LIMIT=$time_limit"
echo "MAX_ROWS=$max_rows"
echo "EXAMPLE_ROWS=$example_rows"
echo "SHARD_SIZE=$shard_size"
echo "LAYOUT_CLUSTER_THRESHOLD=$layout_cluster_threshold"
echo "LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE=$layout_template_min_cluster_size"
echo "LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES=$layout_template_max_exact_host_pages"
echo "LAYOUT_TEMPLATE_LARGE_HOST_MODE=$layout_template_large_host_mode"
echo "LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO=$layout_template_max_selected_item_ratio"
echo "LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO=$layout_template_max_rep_selected_item_ratio"
echo "LAYOUT_TEMPLATE_MORE_NOISE_ENABLE=$layout_template_more_noise_enable"
echo "DYNAMIC_CLASSID_SIMILARITY_THRESHOLD=$dynamic_classid_similarity_threshold"
echo "LAYOUT_TEMPLATE_MIN_CONSENSUS_F1=$layout_template_min_consensus_f1"
echo "LAYOUT_TEMPLATE_VALIDATION_ROWS=$layout_template_validation_rows"
echo "LAYOUT_TEMPLATE_VALIDATION_MIN_F1=$layout_template_validation_min_f1"
echo "LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE=$layout_template_validation_signature_mode"
echo "LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS=$layout_template_large_cluster_validation_rows"
echo "LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE=$layout_template_large_cluster_min_size"
echo "LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO=$layout_template_min_content_length_ratio"
echo "LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO=$layout_template_max_content_length_ratio"
echo "LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE=$layout_template_failed_layout_fallback_signature_mode"
echo "LAYOUT_TEMPLATE_PROPAGATION_TARGET=$layout_template_propagation_target"
echo "LAYOUT_DIAG_VARIANT_MODES=$layout_diag_variant_modes"
echo "LAYOUT_PAGE_SIGNATURE_MODE=$layout_page_signature_mode"
echo "LAYOUT_TARGET_HOSTS=$layout_target_hosts"
echo "LAYOUT_FORCE_HOST_SINGLE_CLUSTER=$layout_force_host_single_cluster"

nebius_ssh_command "$resolved_host" "mkdir -p '$(printf "%q" "$run_dir")/logs'"
rsync -a -e "$rsync_ssh" "$diag_py" "$rsync_host:$run_dir/remote_dripper_layout_diag.py"

job_script="$run_dir/logs/dripper-layout-diag-$(date -u +%Y%m%dT%H%M%SZ).sh"
log_out="$run_dir/logs/dripper-layout-diag-%j.out"
log_err="$run_dir/logs/dripper-layout-diag-%j.err"

{
  printf 'export JOB_SCRIPT=%q\n' "$job_script"
  printf 'export ACCOUNT=%q\n' "$account"
  printf 'export PARTITION=%q\n' "$partition"
  printf 'export CPUS_PER_TASK=%q\n' "$cpus_per_task"
  printf 'export TIME_LIMIT=%q\n' "$time_limit"
  printf 'export LOG_OUT=%q\n' "$log_out"
  printf 'export LOG_ERR=%q\n' "$log_err"
  printf 'export RUN_DIR=%q\n' "$run_dir"
  printf 'export REMOTE_ENV_DIR=%q\n' "$remote_env_dir"
  printf 'export BASE_OUTPUT_DIR=%q\n' "$base_output_dir"
  printf 'export CANDIDATE_OUTPUT_DIR=%q\n' "$candidate_output_dir"
  printf 'export MAX_ROWS=%q\n' "$max_rows"
  printf 'export EXAMPLE_ROWS=%q\n' "$example_rows"
  printf 'export SHARD_SIZE=%q\n' "$shard_size"
  printf 'export LAYOUT_CLUSTER_THRESHOLD=%q\n' "$layout_cluster_threshold"
  printf 'export LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE=%q\n' "$layout_template_min_cluster_size"
  printf 'export LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES=%q\n' "$layout_template_max_exact_host_pages"
  printf 'export LAYOUT_TEMPLATE_LARGE_HOST_MODE=%q\n' "$layout_template_large_host_mode"
  printf 'export LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO=%q\n' "$layout_template_max_selected_item_ratio"
  printf 'export LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO=%q\n' "$layout_template_max_rep_selected_item_ratio"
  printf 'export LAYOUT_TEMPLATE_MORE_NOISE_ENABLE=%q\n' "$layout_template_more_noise_enable"
  printf 'export DYNAMIC_CLASSID_SIMILARITY_THRESHOLD=%q\n' "$dynamic_classid_similarity_threshold"
  printf 'export LAYOUT_TEMPLATE_MIN_CONSENSUS_F1=%q\n' "$layout_template_min_consensus_f1"
  printf 'export LAYOUT_TEMPLATE_VALIDATION_ROWS=%q\n' "$layout_template_validation_rows"
  printf 'export LAYOUT_TEMPLATE_VALIDATION_MIN_F1=%q\n' "$layout_template_validation_min_f1"
  printf 'export LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE=%q\n' "$layout_template_validation_signature_mode"
  printf 'export LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS=%q\n' "$layout_template_large_cluster_validation_rows"
  printf 'export LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE=%q\n' "$layout_template_large_cluster_min_size"
  printf 'export LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO=%q\n' "$layout_template_min_content_length_ratio"
  printf 'export LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO=%q\n' "$layout_template_max_content_length_ratio"
  printf 'export LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE=%q\n' "$layout_template_failed_layout_fallback_signature_mode"
  printf 'export LAYOUT_TEMPLATE_PROPAGATION_TARGET=%q\n' "$layout_template_propagation_target"
  printf 'export LAYOUT_DIAG_VARIANT_MODES=%q\n' "$layout_diag_variant_modes"
  printf 'export LAYOUT_PAGE_SIGNATURE_MODE=%q\n' "$layout_page_signature_mode"
  printf 'export LAYOUT_TARGET_HOSTS=%q\n' "$layout_target_hosts"
  printf 'export LAYOUT_FORCE_HOST_SINGLE_CLUSTER=%q\n' "$layout_force_host_single_cluster"
  printf 'export LAYOUT_PRECOMPUTED_MANIFEST=%q\n' "$layout_precomputed_manifest"
  cat <<'REMOTE'
set -euo pipefail

cat >"$JOB_SCRIPT" <<'JOB'
#!/usr/bin/env bash
#SBATCH --job-name=dripper-layout-diag
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=__PARTITION__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__CPUS_PER_TASK__
#SBATCH --time=__TIME_LIMIT__
#SBATCH --output=__LOG_OUT__
#SBATCH --error=__LOG_ERR__

set -euo pipefail

set +u
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
set -u

export BASE_OUTPUT_DIR="__BASE_OUTPUT_DIR__"
export CANDIDATE_OUTPUT_DIR="__CANDIDATE_OUTPUT_DIR__"
export MAX_ROWS="__MAX_ROWS__"
export EXAMPLE_ROWS="__EXAMPLE_ROWS__"
export SHARD_SIZE="__SHARD_SIZE__"
export LAYOUT_CLUSTER_THRESHOLD="__LAYOUT_CLUSTER_THRESHOLD__"
export LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE="__LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE__"
export LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES="__LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES__"
export LAYOUT_TEMPLATE_LARGE_HOST_MODE="__LAYOUT_TEMPLATE_LARGE_HOST_MODE__"
export LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO="__LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO__"
export LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO="__LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO__"
export LAYOUT_TEMPLATE_MORE_NOISE_ENABLE="__LAYOUT_TEMPLATE_MORE_NOISE_ENABLE__"
export DYNAMIC_CLASSID_SIMILARITY_THRESHOLD="__DYNAMIC_CLASSID_SIMILARITY_THRESHOLD__"
export LAYOUT_TEMPLATE_MIN_CONSENSUS_F1="__LAYOUT_TEMPLATE_MIN_CONSENSUS_F1__"
export LAYOUT_TEMPLATE_VALIDATION_ROWS="__LAYOUT_TEMPLATE_VALIDATION_ROWS__"
export LAYOUT_TEMPLATE_VALIDATION_MIN_F1="__LAYOUT_TEMPLATE_VALIDATION_MIN_F1__"
export LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE="__LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE__"
export LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS="__LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS__"
export LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE="__LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE__"
export LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO="__LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO__"
export LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO="__LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO__"
export LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE="__LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE__"
export LAYOUT_TEMPLATE_PROPAGATION_TARGET="__LAYOUT_TEMPLATE_PROPAGATION_TARGET__"
export LAYOUT_DIAG_VARIANT_MODES="__LAYOUT_DIAG_VARIANT_MODES__"
export LAYOUT_PAGE_SIGNATURE_MODE="__LAYOUT_PAGE_SIGNATURE_MODE__"
export LAYOUT_TARGET_HOSTS="__LAYOUT_TARGET_HOSTS__"
export LAYOUT_FORCE_HOST_SINGLE_CLUSTER="__LAYOUT_FORCE_HOST_SINGLE_CLUSTER__"
export LAYOUT_PRECOMPUTED_MANIFEST="__LAYOUT_PRECOMPUTED_MANIFEST__"
export RUN_DIR="__RUN_DIR__"
export DIAG_OUTPUT_DIR="__RUN_DIR__"

cd "__REMOTE_ENV_DIR__"
export UV_PROJECT_ENVIRONMENT="__REMOTE_ENV_DIR__/.venv"
uv run --no-sync python -u "__RUN_DIR__/remote_dripper_layout_diag.py"
JOB

python - "$JOB_SCRIPT" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
replacements = {
    "__ACCOUNT__": os.environ["ACCOUNT"],
    "__PARTITION__": os.environ["PARTITION"],
    "__CPUS_PER_TASK__": os.environ["CPUS_PER_TASK"],
    "__TIME_LIMIT__": os.environ["TIME_LIMIT"],
    "__LOG_OUT__": os.environ["LOG_OUT"],
    "__LOG_ERR__": os.environ["LOG_ERR"],
    "__REMOTE_ENV_DIR__": os.environ["REMOTE_ENV_DIR"],
    "__BASE_OUTPUT_DIR__": os.environ["BASE_OUTPUT_DIR"],
    "__CANDIDATE_OUTPUT_DIR__": os.environ["CANDIDATE_OUTPUT_DIR"],
    "__MAX_ROWS__": os.environ["MAX_ROWS"],
    "__EXAMPLE_ROWS__": os.environ["EXAMPLE_ROWS"],
    "__SHARD_SIZE__": os.environ["SHARD_SIZE"],
    "__LAYOUT_CLUSTER_THRESHOLD__": os.environ["LAYOUT_CLUSTER_THRESHOLD"],
    "__LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE__": os.environ["LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE"],
    "__LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES__": os.environ["LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES"],
    "__LAYOUT_TEMPLATE_LARGE_HOST_MODE__": os.environ["LAYOUT_TEMPLATE_LARGE_HOST_MODE"],
    "__LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO__": os.environ["LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO"],
    "__LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO__": os.environ["LAYOUT_TEMPLATE_MAX_REP_SELECTED_ITEM_RATIO"],
    "__LAYOUT_TEMPLATE_MORE_NOISE_ENABLE__": os.environ["LAYOUT_TEMPLATE_MORE_NOISE_ENABLE"],
    "__DYNAMIC_CLASSID_SIMILARITY_THRESHOLD__": os.environ["DYNAMIC_CLASSID_SIMILARITY_THRESHOLD"],
    "__LAYOUT_TEMPLATE_MIN_CONSENSUS_F1__": os.environ["LAYOUT_TEMPLATE_MIN_CONSENSUS_F1"],
    "__LAYOUT_TEMPLATE_VALIDATION_ROWS__": os.environ["LAYOUT_TEMPLATE_VALIDATION_ROWS"],
    "__LAYOUT_TEMPLATE_VALIDATION_MIN_F1__": os.environ["LAYOUT_TEMPLATE_VALIDATION_MIN_F1"],
    "__LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE__": os.environ["LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE"],
    "__LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS__": os.environ["LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS"],
    "__LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE__": os.environ["LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE"],
    "__LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO__": os.environ["LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO"],
    "__LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO__": os.environ["LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO"],
    "__LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE__": os.environ["LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE"],
    "__LAYOUT_TEMPLATE_PROPAGATION_TARGET__": os.environ["LAYOUT_TEMPLATE_PROPAGATION_TARGET"],
    "__LAYOUT_DIAG_VARIANT_MODES__": os.environ["LAYOUT_DIAG_VARIANT_MODES"],
    "__LAYOUT_PAGE_SIGNATURE_MODE__": os.environ["LAYOUT_PAGE_SIGNATURE_MODE"],
    "__LAYOUT_TARGET_HOSTS__": os.environ["LAYOUT_TARGET_HOSTS"],
    "__LAYOUT_FORCE_HOST_SINGLE_CLUSTER__": os.environ["LAYOUT_FORCE_HOST_SINGLE_CLUSTER"],
    "__LAYOUT_PRECOMPUTED_MANIFEST__": os.environ.get("LAYOUT_PRECOMPUTED_MANIFEST", ""),
    "__RUN_DIR__": os.environ["RUN_DIR"],
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text)
PY
chmod +x "$JOB_SCRIPT"
job_id="$(sbatch --parsable "$JOB_SCRIPT")"
echo "JOB_ID=$job_id"
echo "JOB_SCRIPT=$JOB_SCRIPT"
echo "LOG_OUT=${LOG_OUT//%j/$job_id}"
echo "LOG_ERR=${LOG_ERR//%j/$job_id}"
echo "SQUEUE_BEGIN"
squeue -j "$job_id" -h -o "%i|%T|%P|%j|%D|%M|%R|%E" || true
echo "SQUEUE_END"
REMOTE
} | nebius_ssh_stdin "$resolved_host" "bash -s"

echo "SUBMIT_LAYOUT_DIAG_END"
