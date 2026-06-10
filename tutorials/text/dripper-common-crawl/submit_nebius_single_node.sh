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

#SBATCH --job-name=curator-dripper-cc25
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/dripper_cc2025_26_%j.log
#SBATCH --error=logs/dripper_cc2025_26_%j.log

set -euo pipefail

if [ -n "${CURATOR_DIR:-}" ]; then
    CURATOR_DIR="$(cd "${CURATOR_DIR}" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    CURATOR_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    CURATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
USER_CACHE_ROOT="/lustre/fsw/portfolios/llmservice/users/${USER}"
OUTPUT_DIR="${OUTPUT_DIR:-${USER_CACHE_ROOT}/dripper_cc_main_2025_26_smoke/${SLURM_JOB_ID}}"

MAX_PAGES="${MAX_PAGES:-128}"
MAX_WARCS="${MAX_WARCS:-4}"
INPUT_MANIFEST_PATH="${INPUT_MANIFEST_PATH:-}"
MANIFEST_WARC_BUCKET="${MANIFEST_WARC_BUCKET:-crawl-data}"
MANIFEST_FETCH_WORKERS="${MANIFEST_FETCH_WORKERS:-64}"
REPLICAS="${REPLICAS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-64}"
DEPLOYMENT_MAX_ONGOING_REQUESTS="${DEPLOYMENT_MAX_ONGOING_REQUESTS:-}"
INGRESS_REPLICAS="${INGRESS_REPLICAS:-}"
INGRESS_MAX_ONGOING_REQUESTS="${INGRESS_MAX_ONGOING_REQUESTS:-}"
INGRESS_TARGET_ONGOING_REQUESTS="${INGRESS_TARGET_ONGOING_REQUESTS:-}"
EXECUTOR_BACKEND="${EXECUTOR_BACKEND:-ray_data}"
PIPELINE_SHARD_SIZE="${PIPELINE_SHARD_SIZE:-64}"
PIPELINE_SHARD_STRATEGY="${PIPELINE_SHARD_STRATEGY:-sequential}"
PIPELINE_PREPROCESS_WORKERS="${PIPELINE_PREPROCESS_WORKERS:-}"
PIPELINE_INFERENCE_WORKERS="${PIPELINE_INFERENCE_WORKERS:-}"
PIPELINE_POSTPROCESS_WORKERS="${PIPELINE_POSTPROCESS_WORKERS:-}"
PIPELINE_LAYOUT_WORKERS="${PIPELINE_LAYOUT_WORKERS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TOP_P="${TOP_P:-1.0}"
H100_COUNT="${H100_COUNT:-8}"
if [ -z "${PIPELINE_PREPROCESS_WORKERS}" ]; then
    if [ "${H100_COUNT}" -ge 8 ]; then
        PIPELINE_PREPROCESS_WORKERS=16
    else
        PIPELINE_PREPROCESS_WORKERS=4
    fi
fi
if [ -z "${PIPELINE_INFERENCE_WORKERS}" ]; then
    if [ "${H100_COUNT}" -ge 8 ]; then
        PIPELINE_INFERENCE_WORKERS=16
    else
        PIPELINE_INFERENCE_WORKERS=4
    fi
fi
if [ -z "${PIPELINE_POSTPROCESS_WORKERS}" ]; then
    if [ "${H100_COUNT}" -ge 8 ]; then
        PIPELINE_POSTPROCESS_WORKERS=16
    else
        PIPELINE_POSTPROCESS_WORKERS=4
    fi
fi
if [ -z "${PIPELINE_LAYOUT_WORKERS}" ]; then
    PIPELINE_LAYOUT_WORKERS="${PIPELINE_INFERENCE_WORKERS}"
fi
MODEL_IDENTIFIER="${MODEL_IDENTIFIER:-opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact}"
PREFETCH_MODEL="${PREFETCH_MODEL:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
WARMUP_PAGES="${WARMUP_PAGES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
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
OUTPUT_FORMAT="${OUTPUT_FORMAT:-mm_md}"
FALLBACK="${FALLBACK:-trafilatura}"
DYNAMIC_MAX_TOKENS="${DYNAMIC_MAX_TOKENS:-0}"
DYNAMIC_MAX_TOKEN_PADDING="${DYNAMIC_MAX_TOKEN_PADDING:-16}"
DYNAMIC_MAX_TOKENS_PER_ITEM="${DYNAMIC_MAX_TOKENS_PER_ITEM:-6}"
DYNAMIC_MIN_MAX_TOKENS="${DYNAMIC_MIN_MAX_TOKENS:-32}"
STRUCTURED_OUTPUT_MODE="${STRUCTURED_OUTPUT_MODE:-none}"
LAYOUT_TEMPLATE_MODE="${LAYOUT_TEMPLATE_MODE:-0}"
LAYOUT_TEMPLATE_LAYOUT_ID_COL="${LAYOUT_TEMPLATE_LAYOUT_ID_COL:-}"
LAYOUT_TEMPLATE_PRECOMPUTE_LAYOUT_IDS="${LAYOUT_TEMPLATE_PRECOMPUTE_LAYOUT_IDS:-0}"
LAYOUT_BASELINE_OUTPUT_DIR="${LAYOUT_BASELINE_OUTPUT_DIR:-}"
LAYOUT_CLUSTER_THRESHOLD="${LAYOUT_CLUSTER_THRESHOLD:-0.95}"
LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE="${LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE:-2}"
LAYOUT_TEMPLATE_FALLBACK_LLM="${LAYOUT_TEMPLATE_FALLBACK_LLM:-1}"
LAYOUT_TEMPLATE_REQUIRE_SUCCESS="${LAYOUT_TEMPLATE_REQUIRE_SUCCESS:-1}"
LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO="${LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO:-0.50}"
LAYOUT_TEMPLATE_MORE_NOISE_ENABLE="${LAYOUT_TEMPLATE_MORE_NOISE_ENABLE:-0}"
LAYOUT_TEMPLATE_VALIDATION_ROWS="${LAYOUT_TEMPLATE_VALIDATION_ROWS:-2}"
LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1="${LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1:-0.98}"
LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE="${LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE:-none}"
LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS="${LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS:-0}"
LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE="${LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE:-0}"
LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO="${LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO:-}"
LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO="${LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO:-}"
LAYOUT_TEMPLATE_REPRESENTATIVE_CANDIDATES="${LAYOUT_TEMPLATE_REPRESENTATIVE_CANDIDATES:-1}"
LAYOUT_TEMPLATE_PROPAGATION_TARGET="${LAYOUT_TEMPLATE_PROPAGATION_TARGET:-raw_html}"
LAYOUT_TEMPLATE_MIN_MAIN_HTML_SIM="${LAYOUT_TEMPLATE_MIN_MAIN_HTML_SIM:-}"
LAYOUT_TEMPLATE_DEFER_FALLBACK_LLM="${LAYOUT_TEMPLATE_DEFER_FALLBACK_LLM:-0}"
LAYOUT_PAGE_SIGNATURE_MODE="${LAYOUT_PAGE_SIGNATURE_MODE:-none}"
LAYOUT_TEMPLATE_FAILED_HOST_FALLBACK_SIGNATURE_MODE="${LAYOUT_TEMPLATE_FAILED_HOST_FALLBACK_SIGNATURE_MODE:-none}"
LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE="${LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE:-none}"
LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MIN_PAGES="${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MIN_PAGES:-0}"
LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MAX_PAGES="${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MAX_PAGES:-0}"
LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES="${LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES:-0}"
LAYOUT_TEMPLATE_LARGE_HOST_MODE="${LAYOUT_TEMPLATE_LARGE_HOST_MODE:-standalone}"
LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY="${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY:-32}"
DYNAMIC_CLASSID_SIMILARITY_THRESHOLD="${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD:-0.85}"
LLM_WEB_KIT_PACKAGE="${LLM_WEB_KIT_PACKAGE:-git+https://github.com/ccprocessor/llm-webkit.git@dev}"
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
RAY_CLEANUP_ON_START="${RAY_CLEANUP_ON_START:-0}"
USE_SRUN="${USE_SRUN:-1}"
COPY_RAY_LOGS_ON_EXIT="${COPY_RAY_LOGS_ON_EXIT:-1}"

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

copy_ray_logs() {
    if [ "${COPY_RAY_LOGS_ON_EXIT}" != "1" ]; then
        return
    fi
    if [ -d "${RAY_TMPDIR}/session_latest/logs" ]; then
        mkdir -p "${OUTPUT_DIR}/ray_logs"
        cp -a "${RAY_TMPDIR}/session_latest/logs/." "${OUTPUT_DIR}/ray_logs/" 2>/dev/null || true
    fi
}
trap copy_ray_logs EXIT

echo "=================================================="
echo "  NeMo Curator Dripper CC-MAIN-2025-26 smoke"
echo "=================================================="
echo "  Host      : $(hostname)"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Nodes     : ${SLURM_JOB_NODELIST}"
echo "  Curator   : ${CURATOR_DIR}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Max pages : ${MAX_PAGES}"
echo "  Manifest  : ${INPUT_MANIFEST_PATH:-none} bucket=${MANIFEST_WARC_BUCKET} fetch_workers=${MANIFEST_FETCH_WORKERS}"
echo "  Replicas  : ${REPLICAS}"
echo "  Warmup    : ${WARMUP_PAGES}"
echo "  Backend   : ${INFERENCE_BACKEND}/${DYNAMO_MODE}"
echo "  Executor  : ${EXECUTOR_BACKEND} shard=${PIPELINE_SHARD_SIZE} strategy=${PIPELINE_SHARD_STRATEGY} workers=${PIPELINE_PREPROCESS_WORKERS:-auto}/${PIPELINE_LAYOUT_WORKERS:-auto}/${PIPELINE_INFERENCE_WORKERS:-auto}/${PIPELINE_POSTPROCESS_WORKERS:-auto}"
echo "  Output    : structured=${STRUCTURED_OUTPUT_MODE}"
echo "  Layout    : template=${LAYOUT_TEMPLATE_MODE} layout_id_col=${LAYOUT_TEMPLATE_LAYOUT_ID_COL:-none} precompute_layout_ids=${LAYOUT_TEMPLATE_PRECOMPUTE_LAYOUT_IDS} baseline=${LAYOUT_BASELINE_OUTPUT_DIR:-none} threshold=${LAYOUT_CLUSTER_THRESHOLD} signature=${LAYOUT_PAGE_SIGNATURE_MODE} failed_host_signature=${LAYOUT_TEMPLATE_FAILED_HOST_FALLBACK_SIGNATURE_MODE} failed_layout_signature=${LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE} min_cluster=${LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE} fallback_llm=${LAYOUT_TEMPLATE_FALLBACK_LLM} defer_fallback_llm=${LAYOUT_TEMPLATE_DEFER_FALLBACK_LLM} require_success=${LAYOUT_TEMPLATE_REQUIRE_SUCCESS} max_selected_ratio=${LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO} min_main_html_sim=${LAYOUT_TEMPLATE_MIN_MAIN_HTML_SIM:-default} content_len_ratio=${LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO:-default}:${LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO:-default} more_noise=${LAYOUT_TEMPLATE_MORE_NOISE_ENABLE} validation_rows=${LAYOUT_TEMPLATE_VALIDATION_ROWS} validation_min_f1=${LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1} validation_signature=${LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE} large_validation_rows=${LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS} large_min_size=${LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE} representative_candidates=${LAYOUT_TEMPLATE_REPRESENTATIVE_CANDIDATES} propagation_target=${LAYOUT_TEMPLATE_PROPAGATION_TARGET} host_single_min=${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MIN_PAGES} host_single_max=${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MAX_PAGES} max_exact_host_pages=${LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES} large_host_mode=${LAYOUT_TEMPLATE_LARGE_HOST_MODE} propagation_concurrency=${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY}"
echo "  Runtime   : dtype=${DTYPE:-default} quant=${QUANTIZATION:-none} kv=${KV_CACHE_DTYPE:-default} gen=${GENERATION_CONFIG:-auto} perf=${PERFORMANCE_MODE:-default} exec=${DISTRIBUTED_EXECUTOR_BACKEND:-default} attn=${ATTENTION_BACKEND:-default} async=${ASYNC_SCHEDULING:-default} dbo=${ENABLE_DBO:-default} verbose=${SERVER_VERBOSE}"
echo "  Ingress   : replicas=${INGRESS_REPLICAS:-default} max_ongoing=${INGRESS_MAX_ONGOING_REQUESTS:-default} target_ongoing=${INGRESS_TARGET_ONGOING_REQUESTS:-default}"
echo "  Ray cleanup on start: ${RAY_CLEANUP_ON_START}"
if [ "${INFERENCE_BACKEND}" = "dynamo" ]; then
    echo "  Dynamo bin: ${DYNAMO_INFRA_BIN_DIR}"
    echo "  Dynamo env: driver_env=${DYNAMO_USE_DRIVER_ENV}"
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
    if [ "${LAYOUT_TEMPLATE_MODE}" = "1" ] && ! uv run --no-sync python -c "import llm_web_kit" >/dev/null 2>&1; then
        uv pip install \
            --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
            "selectolax==0.3.33" \
            "scikit-learn>=1.6.1"
        uv pip install \
            --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
            --no-deps \
            "${LLM_WEB_KIT_PACKAGE}"
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
if [ "${ENABLE_PREFIX_CACHING}" = "1" ]; then
    extra_args+=(--enable-prefix-caching)
else
    extra_args+=(--no-enable-prefix-caching)
fi
if [ -n "${ENABLE_CHUNKED_PREFILL}" ]; then
    if [ "${ENABLE_CHUNKED_PREFILL}" = "1" ]; then
        extra_args+=(--enable-chunked-prefill)
    else
        extra_args+=(--no-enable-chunked-prefill)
    fi
fi
if [ -n "${MAX_NUM_SEQS}" ]; then
    extra_args+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi
if [ -n "${MAX_NUM_BATCHED_TOKENS}" ]; then
    extra_args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [ -n "${DEPLOYMENT_MAX_ONGOING_REQUESTS}" ]; then
    extra_args+=(--deployment-max-ongoing-requests "${DEPLOYMENT_MAX_ONGOING_REQUESTS}")
fi
if [ -n "${INGRESS_REPLICAS}" ]; then
    extra_args+=(--ingress-replicas "${INGRESS_REPLICAS}")
fi
if [ -n "${INGRESS_MAX_ONGOING_REQUESTS}" ]; then
    extra_args+=(--ingress-max-ongoing-requests "${INGRESS_MAX_ONGOING_REQUESTS}")
fi
if [ -n "${INGRESS_TARGET_ONGOING_REQUESTS}" ]; then
    extra_args+=(--ingress-target-ongoing-requests "${INGRESS_TARGET_ONGOING_REQUESTS}")
fi
if [ -n "${INPUT_MANIFEST_PATH}" ]; then
    extra_args+=(--input-manifest-path "${INPUT_MANIFEST_PATH}")
fi
extra_args+=(--manifest-warc-bucket "${MANIFEST_WARC_BUCKET}")
extra_args+=(--manifest-fetch-workers "${MANIFEST_FETCH_WORKERS}")
extra_args+=(--executor-backend "${EXECUTOR_BACKEND}")
extra_args+=(--pipeline-shard-size "${PIPELINE_SHARD_SIZE}")
extra_args+=(--pipeline-shard-strategy "${PIPELINE_SHARD_STRATEGY}")
if [ -n "${PIPELINE_PREPROCESS_WORKERS}" ]; then
    extra_args+=(--pipeline-preprocess-workers "${PIPELINE_PREPROCESS_WORKERS}")
fi
if [ -n "${PIPELINE_INFERENCE_WORKERS}" ]; then
    extra_args+=(--pipeline-inference-workers "${PIPELINE_INFERENCE_WORKERS}")
fi
if [ -n "${PIPELINE_LAYOUT_WORKERS}" ]; then
    extra_args+=(--pipeline-layout-workers "${PIPELINE_LAYOUT_WORKERS}")
fi
if [ -n "${PIPELINE_POSTPROCESS_WORKERS}" ]; then
    extra_args+=(--pipeline-postprocess-workers "${PIPELINE_POSTPROCESS_WORKERS}")
fi
if [ "${DISABLE_THINKING}" = "1" ]; then
    extra_args+=(--disable-thinking)
else
    extra_args+=(--no-disable-thinking)
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
if [ "${RAY_CLEANUP_ON_START}" = "1" ]; then
    extra_args+=(--ray-cleanup-on-start)
else
    extra_args+=(--no-ray-cleanup-on-start)
fi
if [ "${LAYOUT_TEMPLATE_MODE}" = "1" ]; then
    extra_args+=(--layout-template-mode)
else
    extra_args+=(--no-layout-template-mode)
fi
if [ "${LAYOUT_TEMPLATE_FALLBACK_LLM}" = "1" ]; then
    extra_args+=(--layout-template-fallback-llm)
else
    extra_args+=(--no-layout-template-fallback-llm)
fi
if [ "${LAYOUT_TEMPLATE_REQUIRE_SUCCESS}" = "1" ]; then
    extra_args+=(--layout-template-require-success)
else
    extra_args+=(--no-layout-template-require-success)
fi
if [ "${LAYOUT_TEMPLATE_MORE_NOISE_ENABLE}" = "1" ]; then
    extra_args+=(--layout-template-more-noise-enable)
else
    extra_args+=(--no-layout-template-more-noise-enable)
fi
if [ "${LAYOUT_TEMPLATE_DEFER_FALLBACK_LLM}" = "1" ]; then
    extra_args+=(--layout-template-defer-fallback-llm)
else
    extra_args+=(--no-layout-template-defer-fallback-llm)
fi
extra_args+=(--dynamic-max-token-padding "${DYNAMIC_MAX_TOKEN_PADDING}")
extra_args+=(--dynamic-max-tokens-per-item "${DYNAMIC_MAX_TOKENS_PER_ITEM}")
extra_args+=(--dynamic-min-max-tokens "${DYNAMIC_MIN_MAX_TOKENS}")
extra_args+=(--structured-output-mode "${STRUCTURED_OUTPUT_MODE}")
if [ -n "${LAYOUT_TEMPLATE_LAYOUT_ID_COL}" ]; then
    extra_args+=(--layout-template-layout-id-col "${LAYOUT_TEMPLATE_LAYOUT_ID_COL}")
fi
if [ "${LAYOUT_TEMPLATE_PRECOMPUTE_LAYOUT_IDS}" = "1" ]; then
    extra_args+=(--layout-template-precompute-layout-ids)
else
    extra_args+=(--no-layout-template-precompute-layout-ids)
fi
if [ -n "${LAYOUT_BASELINE_OUTPUT_DIR}" ]; then
    extra_args+=(--layout-baseline-output-dir "${LAYOUT_BASELINE_OUTPUT_DIR}")
fi
extra_args+=(--layout-cluster-threshold "${LAYOUT_CLUSTER_THRESHOLD}")
extra_args+=(--layout-template-min-cluster-size "${LAYOUT_TEMPLATE_MIN_CLUSTER_SIZE}")
extra_args+=(--layout-template-max-selected-item-ratio "${LAYOUT_TEMPLATE_MAX_SELECTED_ITEM_RATIO}")
extra_args+=(--layout-template-validation-rows "${LAYOUT_TEMPLATE_VALIDATION_ROWS}")
extra_args+=(--layout-template-validation-min-content-f1 "${LAYOUT_TEMPLATE_VALIDATION_MIN_CONTENT_F1}")
extra_args+=(--layout-template-validation-signature-mode "${LAYOUT_TEMPLATE_VALIDATION_SIGNATURE_MODE}")
extra_args+=(--layout-template-large-cluster-validation-rows "${LAYOUT_TEMPLATE_LARGE_CLUSTER_VALIDATION_ROWS}")
extra_args+=(--layout-template-large-cluster-min-size "${LAYOUT_TEMPLATE_LARGE_CLUSTER_MIN_SIZE}")
extra_args+=(--layout-template-representative-candidates "${LAYOUT_TEMPLATE_REPRESENTATIVE_CANDIDATES}")
extra_args+=(--layout-template-propagation-target "${LAYOUT_TEMPLATE_PROPAGATION_TARGET}")
if [ -n "${LAYOUT_TEMPLATE_MIN_MAIN_HTML_SIM}" ]; then
    extra_args+=(--layout-template-min-main-html-sim "${LAYOUT_TEMPLATE_MIN_MAIN_HTML_SIM}")
fi
if [ -n "${LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO}" ]; then
    extra_args+=(--layout-template-min-content-length-ratio "${LAYOUT_TEMPLATE_MIN_CONTENT_LENGTH_RATIO}")
fi
if [ -n "${LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO}" ]; then
    extra_args+=(--layout-template-max-content-length-ratio "${LAYOUT_TEMPLATE_MAX_CONTENT_LENGTH_RATIO}")
fi
extra_args+=(--layout-page-signature-mode "${LAYOUT_PAGE_SIGNATURE_MODE}")
extra_args+=(--layout-template-failed-host-fallback-signature-mode "${LAYOUT_TEMPLATE_FAILED_HOST_FALLBACK_SIGNATURE_MODE}")
extra_args+=(--layout-template-failed-layout-fallback-signature-mode "${LAYOUT_TEMPLATE_FAILED_LAYOUT_FALLBACK_SIGNATURE_MODE}")
extra_args+=(--layout-template-host-single-cluster-min-pages "${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MIN_PAGES}")
extra_args+=(--layout-template-host-single-cluster-max-pages "${LAYOUT_TEMPLATE_HOST_SINGLE_CLUSTER_MAX_PAGES}")
extra_args+=(--layout-template-max-exact-host-pages "${LAYOUT_TEMPLATE_MAX_EXACT_HOST_PAGES}")
extra_args+=(--layout-template-large-host-mode "${LAYOUT_TEMPLATE_LARGE_HOST_MODE}")
extra_args+=(--layout-template-propagation-concurrency "${LAYOUT_TEMPLATE_PROPAGATION_CONCURRENCY}")
extra_args+=(--dynamic-classid-similarity-threshold "${DYNAMIC_CLASSID_SIMILARITY_THRESHOLD}")
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
RAY_WORKER_PORT_BASE="${RAY_WORKER_PORT_BASE:-10000}"
RAY_WORKER_PORT_SPAN="${RAY_WORKER_PORT_SPAN:-2000}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-${RAY_WORKER_PORT_BASE}}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-$((RAY_WORKER_PORT_BASE + RAY_WORKER_PORT_SPAN - 1))}"
RAY_CPUS="${RAY_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"
RAY_GPUS="${RAY_GPUS:-${H100_COUNT}}"

main_cmd=(
uv run --no-sync python tutorials/text/dripper-common-crawl/main.py \
    --model-identifier "${MODEL_IDENTIFIER}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-pages "${MAX_PAGES}" \
    --max-warcs "${MAX_WARCS}" \
    --replicas "${REPLICAS}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-tokens "${MAX_TOKENS}" \
    --top-p "${TOP_P}" \
    --prompt-version "${PROMPT_VERSION}" \
    --output-format "${OUTPUT_FORMAT}" \
    --fallback "${FALLBACK}" \
    --server-port "${SERVER_PORT}" \
    --warmup-pages "${WARMUP_PAGES}" \
    --h100-count "${H100_COUNT}" \
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
echo "  Metrics: ${OUTPUT_DIR}/metrics.json"
echo "=================================================="
