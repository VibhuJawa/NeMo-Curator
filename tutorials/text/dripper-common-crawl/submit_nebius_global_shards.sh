#!/usr/bin/env bash
# Submit Dripper global shard run: Stage 1 CPU array → Stage 2 GPU array.
#
# Each array task maps to one host_bucket shard (0–10000).  Stage 2 depends on
# Stage 1 via --dependency=aftercorr so each shard starts GPU inference only
# after its CPU precompute completes.
#
# Usage:
#   bash tutorials/text/dripper-common-crawl/submit_nebius_global_shards.sh HOST
#   bash tutorials/text/dripper-common-crawl/submit_nebius_global_shards.sh HOST 0-9
#   bash tutorials/text/dripper-common-crawl/submit_nebius_global_shards.sh HOST 0-10000
#
# Env vars (all optional):
#   GLOBAL_OUTPUT_ROOT   default: /lustre/.../dripper_global
#   SHARD_DIR            default: .../nemo_curator_dripper_sorted_host_buckets_20260611
#   S1_MAX_WARCS         0 = unlimited (default)
#   S2_GPU_MEM_UTIL      vLLM GPU memory utilization (default 0.9)
#   DRY_RUN              set to 1 to print sbatch commands without submitting

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_dir="$(cd "${script_dir}/../../../scripts" && pwd)"

HOST="${1:-vjawa@nb-hel-cs-001-vscode-01.nvidia.com}"
ARRAY_RANGE="${2:-0-10000}"
DRY_RUN="${DRY_RUN:-0}"

REMOTE_USER="${REMOTE_USER:-vjawa}"
USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${REMOTE_USER}"
SHARED_CODE="${USER_CACHE_ROOT}/nemo_curator_shared"
GLOBAL_OUTPUT_ROOT="${GLOBAL_OUTPUT_ROOT:-${USER_CACHE_ROOT}/dripper_global}"
SHARD_DIR="${SHARD_DIR:-${USER_CACHE_ROOT}/nemo_curator_dripper_sorted_host_buckets_20260611}"

S1_MAX_WARCS="${S1_MAX_WARCS:-0}"
S2_GPU_MEM_UTIL="${S2_GPU_MEM_UTIL:-0.9}"

echo "==================================================="
echo "  Dripper global shard submission"
echo "  Host    : ${HOST}"
echo "  Shards  : ${ARRAY_RANGE}"
echo "  S1 out  : ${GLOBAL_OUTPUT_ROOT}/stage1/"
echo "  S2 out  : ${GLOBAL_OUTPUT_ROOT}/stage2/"
echo "  Dry run : ${DRY_RUN}"
echo "==================================================="

# Resolve SSH control socket from the existing nebius_ssh helper
NEBIUS_SSH_CONTROL_DIR="${NEBIUS_SSH_CONTROL_DIR:-/tmp/.nebius_ctl}"
ssh_ctl="-o ControlMaster=auto -o ControlPath=${NEBIUS_SSH_CONTROL_DIR}/%C.sock -o ConnectTimeout=30"

run_remote() {
    # shellcheck disable=SC2086
    ssh ${ssh_ctl} "${HOST}" "$@"
}

emit_vars() {
    printf 'export GLOBAL_OUTPUT_ROOT=%q\n' "${GLOBAL_OUTPUT_ROOT}"
    printf 'export SHARD_DIR=%q\n' "${SHARD_DIR}"
    printf 'export SHARED_CODE=%q\n' "${SHARED_CODE}"
    printf 'export REMOTE_USER=%q\n' "${REMOTE_USER}"
    printf 'export ARRAY_RANGE=%q\n' "${ARRAY_RANGE}"
    printf 'export S1_MAX_WARCS=%q\n' "${S1_MAX_WARCS}"
    printf 'export S2_GPU_MEM_UTIL=%q\n' "${S2_GPU_MEM_UTIL}"
    printf 'export DRY_RUN=%q\n' "${DRY_RUN}"
}

{
    emit_vars
    cat <<'REMOTE'
set -euo pipefail

mkdir -p "${SHARED_CODE}/logs"

S1_SCRIPT="${SHARED_CODE}/tutorials/text/dripper-common-crawl/submit_nebius_dripper_stage1_global.sh"
S2_SCRIPT="${SHARED_CODE}/tutorials/text/dripper-common-crawl/submit_nebius_dripper_stage2_global.sh"

if [ ! -f "${S1_SCRIPT}" ]; then
    echo "ERROR: Stage 1 script not found: ${S1_SCRIPT}" >&2
    exit 1
fi
if [ ! -f "${S2_SCRIPT}" ]; then
    echo "ERROR: Stage 2 script not found: ${S2_SCRIPT}" >&2
    exit 1
fi

s1_cmd=(
    sbatch
    --parsable
    "--array=${ARRAY_RANGE}"
    --export="ALL,GLOBAL_OUTPUT_ROOT=${GLOBAL_OUTPUT_ROOT},SHARD_DIR=${SHARD_DIR},REMOTE_USER=${REMOTE_USER},MAX_WARCS=${S1_MAX_WARCS}"
    --chdir="${SHARED_CODE}/logs"
    "${S1_SCRIPT}"
)

echo "STAGE1_CMD: ${s1_cmd[*]}"

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN: skipping Stage 1 submission"
    echo "STAGE1_JOB_ID=DRY_RUN"
    echo "STAGE2_CMD: sbatch --array=${ARRAY_RANGE} --dependency=aftercorr:DRY_RUN ${S2_SCRIPT}"
else
    S1_JOB_ID="$("${s1_cmd[@]}")"
    echo "STAGE1_JOB_ID=${S1_JOB_ID}"

    S2_JOB_ID="$(sbatch \
        --parsable \
        "--array=${ARRAY_RANGE}" \
        "--dependency=aftercorr:${S1_JOB_ID}" \
        --export="ALL,GLOBAL_OUTPUT_ROOT=${GLOBAL_OUTPUT_ROOT},REMOTE_USER=${REMOTE_USER},GPU_MEMORY_UTILIZATION=${S2_GPU_MEM_UTIL}" \
        --chdir="${SHARED_CODE}/logs" \
        "${S2_SCRIPT}")"
    echo "STAGE2_JOB_ID=${S2_JOB_ID}"

    echo "SQUEUE_BEGIN"
    squeue -u "${REMOTE_USER}" -h -o "%i|%T|%P|%j|%D|%M|%R" 2>/dev/null || true
    echo "SQUEUE_END"

    echo ""
    echo "Submitted:"
    echo "  Stage 1 (CPU): array job ${S1_JOB_ID}  [${ARRAY_RANGE}]"
    echo "  Stage 2 (GPU): array job ${S2_JOB_ID}  [depends on aftercorr:${S1_JOB_ID}]"
    echo ""
    echo "Monitor:  squeue -u ${REMOTE_USER}"
    echo "S1 logs:  ${SHARED_CODE}/logs/dripper_s1_${S1_JOB_ID}_*.log"
    echo "S2 logs:  ${SHARED_CODE}/logs/dripper_s2_${S2_JOB_ID}_*.log"
    echo "Outputs:  ${GLOBAL_OUTPUT_ROOT}/stage{1,2}/shard_NNNN/"
fi
REMOTE
} | run_remote "bash -s"

echo "==================================================="
echo "  Submission complete"
echo "==================================================="
