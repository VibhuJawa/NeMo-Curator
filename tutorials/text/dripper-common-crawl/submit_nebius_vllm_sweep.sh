#!/bin/bash
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

#SBATCH --job-name=curator-dripper-vllm-sweep
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/dripper_vllm_sweep_%j.log
#SBATCH --error=logs/dripper_vllm_sweep_%j.log

set -euo pipefail

if [ -n "${CURATOR_DIR:-}" ]; then
    CURATOR_DIR="$(cd "${CURATOR_DIR}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    CURATOR_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    CURATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${USER}"
OUTPUT_DIR="${OUTPUT_DIR:-${USER_CACHE_ROOT}/dripper_cc_main_2025_26_vllm_sweep/${SLURM_JOB_ID}}"

MAX_PAGES="${MAX_PAGES:-320}"
MAX_WARCS="${MAX_WARCS:-4}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
REPLICAS="${REPLICAS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TOP_P="${TOP_P:-1.0}"
H100_COUNT="${H100_COUNT:-8}"
MODEL_IDENTIFIER="${MODEL_IDENTIFIER:-opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact}"
PREFETCH_MODEL="${PREFETCH_MODEL:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DTYPE="${DTYPE:-}"
QUANTIZATION="${QUANTIZATION:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
CALCULATE_KV_SCALES="${CALCULATE_KV_SCALES:-}"
GENERATION_CONFIG="${GENERATION_CONFIG:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
SAFETENSORS_LOAD_STRATEGY="${SAFETENSORS_LOAD_STRATEGY:-}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-}"
ENABLE_DBO="${ENABLE_DBO:-}"
DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-}"
DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-}"
MAX_NUM_PARTIAL_PREFILLS="${MAX_NUM_PARTIAL_PREFILLS:-}"
MAX_LONG_PARTIAL_PREFILLS="${MAX_LONG_PARTIAL_PREFILLS:-}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}"
SERVER_PORT="${SERVER_PORT:-}"
SERVER_VERBOSE="${SERVER_VERBOSE:-0}"
PROMPT_VERSION="${PROMPT_VERSION:-short_compact}"
DYNAMIC_MAX_TOKENS="${DYNAMIC_MAX_TOKENS:-0}"
DYNAMIC_MAX_TOKEN_PADDING="${DYNAMIC_MAX_TOKEN_PADDING:-16}"
DYNAMIC_MAX_TOKENS_PER_ITEM="${DYNAMIC_MAX_TOKENS_PER_ITEM:-6}"
DYNAMIC_MIN_MAX_TOKENS="${DYNAMIC_MIN_MAX_TOKENS:-32}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-ray_serve}"
DYNAMO_MODE="${DYNAMO_MODE:-aggregated}"
DYNAMO_PREFILL_REPLICAS="${DYNAMO_PREFILL_REPLICAS:-1}"
DYNAMO_DECODE_REPLICAS="${DYNAMO_DECODE_REPLICAS:-1}"
DYNAMO_ROUTER_MODE="${DYNAMO_ROUTER_MODE:-auto}"
DYNAMO_ROUTER_KV_EVENTS="${DYNAMO_ROUTER_KV_EVENTS:-0}"
DYNAMO_ETCD_ENDPOINT="${DYNAMO_ETCD_ENDPOINT:-}"
DYNAMO_NATS_URL="${DYNAMO_NATS_URL:-}"
DYNAMO_INFRA_BIN_DIR="${DYNAMO_INFRA_BIN_DIR:-${USER_CACHE_ROOT}/dynamo_infra/bin}"
DYNAMO_USE_DRIVER_ENV="${DYNAMO_USE_DRIVER_ENV:-1}"
DYNAMO_DRIVER_ENV_INSTALL_EXTRAS="${DYNAMO_DRIVER_ENV_INSTALL_EXTRAS:-1}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-16,32,64,128}"
GPU_MEMORY_UTILIZATION_VALUES="${GPU_MEMORY_UTILIZATION_VALUES:-0.9}"
PREFIX_CACHING_VALUES="${PREFIX_CACHING_VALUES:-true}"
CHUNKED_PREFILL_VALUES="${CHUNKED_PREFILL_VALUES:-true}"
MAX_NUM_SEQS_VALUES="${MAX_NUM_SEQS_VALUES:-64,128}"
MAX_NUM_BATCHED_TOKENS_VALUES="${MAX_NUM_BATCHED_TOKENS_VALUES:-16384,32768}"
MAX_SWEEP_CASES="${MAX_SWEEP_CASES:-0}"
NUM_WARMUPS="${NUM_WARMUPS:-concurrency}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-1800}"
RAY_CLEANUP_ON_START="${RAY_CLEANUP_ON_START:-0}"
USE_SRUN="${USE_SRUN:-1}"

set +u
source "${HOME}/.bashrc"
set -u

if [ -f "${USER_CACHE_ROOT}/cache_env.sh" ]; then
    set -a
    set +u
    # shellcheck disable=SC1090
    source "${USER_CACHE_ROOT}/cache_env.sh"
    set -u
    set +a
fi

export AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-https://pdx.s8k.io}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
if [ -n "${PBSS_ACCESS_KEY_ID:-}" ]; then
    export AWS_ACCESS_KEY_ID="${PBSS_ACCESS_KEY_ID}"
fi
if [ -n "${PBSS_SECRET_ACCESS_KEY:-}" ]; then
    export AWS_SECRET_ACCESS_KEY="${PBSS_SECRET_ACCESS_KEY}"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${USER_CACHE_ROOT}/uv_cache}"
export UV_PROJECT_ENVIRONMENT="${CURATOR_DIR}/.venv"
export HF_HOME="${HF_HOME:-${USER_CACHE_ROOT}/hf_cache}"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
export RAY_PORT_BROADCAST_DIR="${RAY_PORT_BROADCAST_DIR:-${USER_CACHE_ROOT}/ray_ports}"
export TMPDIR="/tmp"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,::1"
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,::1"
if [ "${INFERENCE_BACKEND}" = "dynamo" ]; then
    export PATH="${DYNAMO_INFRA_BIN_DIR}:${PATH}"
    export NEMO_CURATOR_DYNAMO_USE_DRIVER_ENV="${DYNAMO_USE_DRIVER_ENV}"
fi

mkdir -p "${CURATOR_DIR}/logs" "${OUTPUT_DIR}" "${RAY_PORT_BROADCAST_DIR}"

echo "=================================================="
echo "  NeMo Curator Dripper vLLM sweep"
echo "=================================================="
echo "  Host         : $(hostname)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Nodes        : ${SLURM_JOB_NODELIST}"
echo "  Curator      : ${CURATOR_DIR}"
echo "  Output       : ${OUTPUT_DIR}"
echo "  Max pages    : ${MAX_PAGES}"
echo "  Num prompts  : ${NUM_PROMPTS}"
echo "  Replicas     : ${REPLICAS}"
echo "  Backend      : ${INFERENCE_BACKEND}/${DYNAMO_MODE}"
echo "  Concurrency  : ${CONCURRENCY_VALUES}"
echo "  max seqs     : ${MAX_NUM_SEQS_VALUES}"
echo "  batch tokens : ${MAX_NUM_BATCHED_TOKENS_VALUES}"
echo "  Runtime      : dtype=${DTYPE:-default} quant=${QUANTIZATION:-none} kv=${KV_CACHE_DTYPE:-default} gen=${GENERATION_CONFIG:-auto} perf=${PERFORMANCE_MODE:-default} exec=${DISTRIBUTED_EXECUTOR_BACKEND:-default} attn=${ATTENTION_BACKEND:-default} async=${ASYNC_SCHEDULING:-default} dbo=${ENABLE_DBO:-default} verbose=${SERVER_VERBOSE}"
echo "  Dynamic max tokens: ${DYNAMIC_MAX_TOKENS}"
echo "  Ray cleanup on start: ${RAY_CLEANUP_ON_START}"
if [ "${INFERENCE_BACKEND}" = "dynamo" ]; then
    echo "  Dynamo bin   : ${DYNAMO_INFRA_BIN_DIR}"
    echo "  Dynamo env   : driver_env=${DYNAMO_USE_DRIVER_ENV}"
fi
echo "=================================================="

cd "${CURATOR_DIR}"
python --version || true
uv --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

env_lock="${UV_PROJECT_ENVIRONMENT}.lock"
(
    flock 9
    uv sync --inexact --extra inference_server --extra text_cpu
    if ! uv run --no-sync python -c "import mineru_html" >/dev/null 2>&1; then
        uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" "mineru_html>=1.1.2"
    fi

    if [ "${INFERENCE_BACKEND}" = "dynamo" ] && [ "${DYNAMO_USE_DRIVER_ENV}" = "1" ] && [ "${DYNAMO_DRIVER_ENV_INSTALL_EXTRAS}" = "1" ]; then
        dynamo_override_file="${OUTPUT_DIR}/dynamo_driver_env_overrides.txt"
        uv run --no-sync python - <<'PY' > "${dynamo_override_file}"
import ray

print(f"ray=={ray.__version__}")
PY
        echo "Installing ai-dynamo[vllm] into driver env with override ${dynamo_override_file}"
        uv pip install \
            --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
            --override "${dynamo_override_file}" \
            "ai-dynamo[vllm]==1.1.0"
    fi
) 9>"${env_lock}"

if [ "${PREFETCH_MODEL}" = "1" ]; then
    MODEL_IDENTIFIER="${MODEL_IDENTIFIER}" uv run --no-sync python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_IDENTIFIER"]
path = snapshot_download(model_id)
print(f"PREFETCHED_MODEL={model_id}")
print(f"PREFETCHED_PATH={path}")
PY
fi

extra_args=()
if [ "${ENFORCE_EAGER}" = "1" ]; then
    extra_args+=(--enforce-eager)
fi
if [ "${MAX_SWEEP_CASES}" != "0" ]; then
    extra_args+=(--max-sweep-cases "${MAX_SWEEP_CASES}")
fi
if [ -n "${DTYPE}" ]; then
    extra_args+=(--dtype "${DTYPE}")
fi
if [ -n "${QUANTIZATION}" ]; then
    extra_args+=(--quantization "${QUANTIZATION}")
fi
if [ -n "${KV_CACHE_DTYPE}" ]; then
    extra_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [ -n "${CALCULATE_KV_SCALES}" ]; then
    if [ "${CALCULATE_KV_SCALES}" = "1" ]; then
        extra_args+=(--calculate-kv-scales)
    else
        extra_args+=(--no-calculate-kv-scales)
    fi
fi
if [ -n "${GENERATION_CONFIG}" ]; then
    extra_args+=(--generation-config "${GENERATION_CONFIG}")
fi
if [ -n "${LOAD_FORMAT}" ]; then
    extra_args+=(--load-format "${LOAD_FORMAT}")
fi
if [ -n "${SAFETENSORS_LOAD_STRATEGY}" ]; then
    extra_args+=(--safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}")
fi
if [ -n "${PERFORMANCE_MODE}" ]; then
    extra_args+=(--performance-mode "${PERFORMANCE_MODE}")
fi
if [ -n "${DISTRIBUTED_EXECUTOR_BACKEND}" ]; then
    extra_args+=(--distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}")
fi
if [ -n "${ATTENTION_BACKEND}" ]; then
    extra_args+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [ -n "${ASYNC_SCHEDULING}" ]; then
    if [ "${ASYNC_SCHEDULING}" = "1" ]; then
        extra_args+=(--async-scheduling)
    else
        extra_args+=(--no-async-scheduling)
    fi
fi
if [ -n "${ENABLE_DBO}" ]; then
    if [ "${ENABLE_DBO}" = "1" ]; then
        extra_args+=(--enable-dbo)
    else
        extra_args+=(--no-enable-dbo)
    fi
fi
if [ -n "${DBO_DECODE_TOKEN_THRESHOLD}" ]; then
    extra_args+=(--dbo-decode-token-threshold "${DBO_DECODE_TOKEN_THRESHOLD}")
fi
if [ -n "${DBO_PREFILL_TOKEN_THRESHOLD}" ]; then
    extra_args+=(--dbo-prefill-token-threshold "${DBO_PREFILL_TOKEN_THRESHOLD}")
fi
if [ -n "${MAX_NUM_PARTIAL_PREFILLS}" ]; then
    extra_args+=(--max-num-partial-prefills "${MAX_NUM_PARTIAL_PREFILLS}")
fi
if [ -n "${MAX_LONG_PARTIAL_PREFILLS}" ]; then
    extra_args+=(--max-long-partial-prefills "${MAX_LONG_PARTIAL_PREFILLS}")
fi
if [ -n "${LONG_PREFILL_TOKEN_THRESHOLD}" ]; then
    extra_args+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
fi
if [ "${SERVER_VERBOSE}" = "1" ]; then
    extra_args+=(--server-verbose)
fi
if [ "${DYNAMIC_MAX_TOKENS}" = "1" ]; then
    extra_args+=(--dynamic-max-tokens)
else
    extra_args+=(--no-dynamic-max-tokens)
fi
extra_args+=(--dynamic-max-token-padding "${DYNAMIC_MAX_TOKEN_PADDING}")
extra_args+=(--dynamic-max-tokens-per-item "${DYNAMIC_MAX_TOKENS_PER_ITEM}")
extra_args+=(--dynamic-min-max-tokens "${DYNAMIC_MIN_MAX_TOKENS}")
if [ "${RAY_CLEANUP_ON_START}" = "1" ]; then
    extra_args+=(--ray-cleanup-on-start)
else
    extra_args+=(--no-ray-cleanup-on-start)
fi
extra_args+=(--inference-backend "${INFERENCE_BACKEND}")
if [ "${INFERENCE_BACKEND}" = "dynamo" ]; then
    extra_args+=(--dynamo-mode "${DYNAMO_MODE}")
    extra_args+=(--dynamo-prefill-replicas "${DYNAMO_PREFILL_REPLICAS}")
    extra_args+=(--dynamo-decode-replicas "${DYNAMO_DECODE_REPLICAS}")
    extra_args+=(--dynamo-router-mode "${DYNAMO_ROUTER_MODE}")
    if [ "${DYNAMO_ROUTER_KV_EVENTS}" = "1" ]; then
        extra_args+=(--dynamo-router-kv-events)
    else
        extra_args+=(--no-dynamo-router-kv-events)
    fi
    if [ -n "${DYNAMO_ETCD_ENDPOINT}" ]; then
        extra_args+=(--dynamo-etcd-endpoint "${DYNAMO_ETCD_ENDPOINT}")
    fi
    if [ -n "${DYNAMO_NATS_URL}" ]; then
        extra_args+=(--dynamo-nats-url "${DYNAMO_NATS_URL}")
    fi
fi

RAY_PORT="${RAY_PORT:-$((20000 + SLURM_JOB_ID % 10000))}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$((30000 + SLURM_JOB_ID % 10000))}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-$((40000 + SLURM_JOB_ID % 10000))}"
RAY_METRICS_PORT="${RAY_METRICS_PORT:-$((50000 + SLURM_JOB_ID % 10000))}"
SERVER_PORT="${SERVER_PORT:-$((60000 + SLURM_JOB_ID % 5000))}"
RAY_WORKER_PORT_BASE="${RAY_WORKER_PORT_BASE:-$((10000 + (SLURM_JOB_ID % 90) * 100))}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-${RAY_WORKER_PORT_BASE}}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-$((RAY_WORKER_PORT_BASE + 99))}"
RAY_CPUS="${RAY_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"
RAY_GPUS="${RAY_GPUS:-${H100_COUNT}}"

main_cmd=(
uv run --no-sync python tutorials/text/dripper-common-crawl/vllm_sweep.py \
    --model-identifier "${MODEL_IDENTIFIER}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-pages "${MAX_PAGES}" \
    --max-warcs "${MAX_WARCS}" \
    --num-prompts "${NUM_PROMPTS}" \
    --replicas "${REPLICAS}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-tokens "${MAX_TOKENS}" \
    --top-p "${TOP_P}" \
    --prompt-version "${PROMPT_VERSION}" \
    --server-port "${SERVER_PORT}" \
    --h100-count "${H100_COUNT}" \
    --concurrency-values "${CONCURRENCY_VALUES}" \
    --gpu-memory-utilization-values "${GPU_MEMORY_UTILIZATION_VALUES}" \
    --prefix-caching-values "${PREFIX_CACHING_VALUES}" \
    --chunked-prefill-values "${CHUNKED_PREFILL_VALUES}" \
    --max-num-seqs-values "${MAX_NUM_SEQS_VALUES}" \
    --max-num-batched-tokens-values "${MAX_NUM_BATCHED_TOKENS_VALUES}" \
    --num-warmups "${NUM_WARMUPS}" \
    --bench-timeout-s "${BENCH_TIMEOUT_S}" \
    --ray-temp-dir "${RAY_TMPDIR}" \
    --ray-port "${RAY_PORT}" \
    --ray-dashboard-port "${RAY_DASHBOARD_PORT}" \
    --ray-client-server-port "${RAY_CLIENT_SERVER_PORT}" \
    --ray-metrics-port "${RAY_METRICS_PORT}" \
    --ray-min-worker-port "${RAY_MIN_WORKER_PORT}" \
    --ray-max-worker-port "${RAY_MAX_WORKER_PORT}" \
    --ray-num-cpus "${RAY_CPUS}" \
    --ray-num-gpus "${RAY_GPUS}" \
    "${extra_args[@]}"
)

if [ "${USE_SRUN}" = "1" ]; then
    srun --ntasks-per-node=1 "${main_cmd[@]}"
else
    "${main_cmd[@]}"
fi

echo "=================================================="
echo "  DONE"
echo "  Summary: ${OUTPUT_DIR}/sweep_summary.csv"
echo "  Plot   : ${OUTPUT_DIR}/concurrency_vs_req_s.png"
echo "=================================================="
