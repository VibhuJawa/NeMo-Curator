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

"""Bounded Dripper/MinerU-HTML run over CC-MAIN-2025-26 WARC data."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from glob import glob
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import ProxyHandler, build_opener

import pandas as pd
from loguru import logger
from warcio.archiveiterator import ArchiveIterator

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.core.serve import (
    DynamoRoleConfig,
    DynamoRouterConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    RayServeModelConfig,
    RayServeServerConfig,
)
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.experimental.dripper import (
    DripperHTMLExtractionStage,
    DripperHTMLExtractionPipelineStage,
    DripperHTMLLayoutClusteringStage,
)
from nemo_curator.stages.text.experimental.dripper.propagation_stage import (
    DripperHTMLLayoutPropagationStage,
)
from nemo_curator.tasks import DocumentBatch

DEFAULT_MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
DEFAULT_WARC_PATHS = "s3://crawl-data/CC-MAIN-2025-26/warc.paths.gz"
DEFAULT_SNAPSHOT_PAGES = 2_385_603_949
PIPELINE_SHARD_STRATEGIES = (
    "sequential",
    "balanced_html_bytes",
    "domain_clustered",
    "domain_complete",
    "domain_html_hash",
    "domain_then_html_bytes",
    "layout_complete",
)
_DRIPPER_HOST_KEY_COL = "_dripper_host_key"
_DRIPPER_LAYOUT_KEY_COL = "_dripper_layout_key"
_DRIPPER_HTML_BYTES_COL = "_dripper_html_bytes"
_DRIPPER_HTML_HASH_COL = "_dripper_html_hash"
DEFAULT_LAYOUT_ID_COL = "dripper_layout_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dripper over a bounded CC-MAIN-2025-26 sample")
    parser.add_argument(
        "--input-manifest-path",
        default=None,
        help=(
            "Optional parquet/jsonl/csv manifest. If it contains html or binary_content, those bytes are used "
            "directly. Otherwise warc_filename, warc_record_offset, and warc_record_length are range-fetched."
        ),
    )
    parser.add_argument("--warc-paths-uri", default=DEFAULT_WARC_PATHS)
    parser.add_argument("--output-dir", default="outputs/dripper_cc_main_2025_26_smoke")
    parser.add_argument("--max-pages", type=int, default=64, help="Maximum HTML pages to process; 0 exhausts selected WARCs")
    parser.add_argument("--max-warcs", type=int, default=4)
    parser.add_argument("--html-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-html-bytes", type=int, default=1)
    parser.add_argument("--manifest-warc-bucket", default=os.environ.get("DRIPPER_MANIFEST_WARC_BUCKET", "crawl-data"))
    parser.add_argument("--manifest-fetch-workers", type=int, default=64)
    parser.add_argument("--s3-endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL"))
    parser.add_argument("--s3-region", default=os.environ.get("AWS_REGION", "us-east-1"))
    parser.add_argument("--model-identifier", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="dripper")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float", "float16", "float32", "half"], default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "bfloat16", "float16", "fp8", "fp8_ds_mla", "fp8_e4m3", "fp8_e5m2", "fp8_inc"],
        default=None,
    )
    parser.add_argument("--calculate-kv-scales", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--generation-config", default=None)
    parser.add_argument("--load-format", default=None)
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=["lazy", "eager", "prefetch", "torchao"],
        default=None,
    )
    parser.add_argument("--performance-mode", choices=["balanced", "interactivity", "throughput"], default=None)
    parser.add_argument("--distributed-executor-backend", choices=["ray", "mp", "uni", "external_launcher"], default=None)
    parser.add_argument("--attention-backend", choices=["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "XFORMERS"], default=None)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-dbo", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dbo-decode-token-threshold", type=int, default=None)
    parser.add_argument("--dbo-prefill-token-threshold", type=int, default=None)
    parser.add_argument("--max-num-partial-prefills", type=int, default=None)
    parser.add_argument("--max-long-partial-prefills", type=int, default=None)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=None)
    parser.add_argument("--max-concurrent-requests", type=int, default=16)
    parser.add_argument("--deployment-max-ongoing-requests", type=int, default=None)
    parser.add_argument("--ingress-replicas", type=int, default=None)
    parser.add_argument("--ingress-max-ongoing-requests", type=int, default=None)
    parser.add_argument("--ingress-target-ongoing-requests", type=int, default=None)
    parser.add_argument("--executor-backend", choices=["direct", "ray_data"], default="ray_data")
    parser.add_argument("--pipeline-shard-size", type=int, default=64)
    parser.add_argument(
        "--pipeline-shard-strategy",
        choices=PIPELINE_SHARD_STRATEGIES,
        default="sequential",
        help=(
            "How to split pages into Ray Data tasks; balanced_html_bytes reduces long-tail shard imbalance, "
            "domain_clustered groups full hostnames but can split large hosts, domain_complete never splits "
            "a host across tasks, domain_html_hash keeps exact-HTML duplicates adjacent within each host, "
            "domain_then_html_bytes keeps host runs while byte-balancing shards, and layout_complete never "
            "splits precomputed layout IDs."
        ),
    )
    parser.add_argument("--pipeline-preprocess-workers", type=int, default=None)
    parser.add_argument("--pipeline-inference-workers", type=int, default=None)
    parser.add_argument("--pipeline-postprocess-workers", type=int, default=None)
    parser.add_argument(
        "--pipeline-layout-workers",
        type=int,
        default=None,
        help="Worker count for the CPU layout-template stage; defaults to pipeline inference workers.",
    )
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--health-check-timeout-s", type=int, default=1800)
    parser.add_argument("--client-ready-timeout-s", type=int, default=120)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-verbose", action="store_true")
    parser.add_argument("--prompt-version", default="short_compact")
    parser.add_argument("--output-format", default="mm_md")
    parser.add_argument("--fallback", choices=["trafilatura", "bypass", "empty"], default="trafilatura")
    parser.add_argument("--dynamic-max-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic-max-token-padding", type=int, default=16)
    parser.add_argument("--dynamic-max-tokens-per-item", type=int, default=6)
    parser.add_argument("--dynamic-min-max-tokens", type=int, default=32)
    parser.add_argument(
        "--structured-output-mode",
        choices=["none", "structured_outputs", "guided_regex"],
        default="none",
        help=(
            "Optional vLLM structured-output mode for compact Dripper responses. "
            "structured_outputs uses extra_body.structured_outputs.regex; guided_regex uses the older guided_regex key."
        ),
    )
    parser.add_argument(
        "--layout-template-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Infer one representative per host/layout cluster and propagate its template on CPU.",
    )
    parser.add_argument(
        "--layout-template-layout-id-col",
        default=None,
        help=(
            "Optional precomputed layout ID column. When set, layout-template mode groups by this column instead "
            "of rebuilding DOM clusters inside each Ray task. Use with --pipeline-shard-strategy layout_complete."
        ),
    )
    parser.add_argument(
        "--layout-template-precompute-layout-ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run a CPU-only Ray pre-pass that computes host-bounded llm-webkit DOM layout IDs before starting "
            "the inference server. Use with --layout-template-layout-id-col and preferably "
            "--pipeline-shard-strategy layout_complete."
        ),
    )
    parser.add_argument(
        "--layout-baseline-output-dir",
        default=None,
        help=(
            "Optional pure-Dripper output directory containing dripper_results.parquet/jsonl. "
            "When set, layout-template metrics include exact-prompt-dedup overlap and incremental "
            "non-exact propagated savings against that baseline."
        ),
    )
    parser.add_argument(
        "--precompute-layout-manifest-only",
        action="store_true",
        help=(
            "Load the requested input pages, precompute host-bounded Dripper layout IDs, write "
            "layout_precompute_manifest.parquet under --output-dir, and exit before starting an inference server."
        ),
    )
    parser.add_argument(
        "--layout-cluster-threshold",
        type=float,
        default=0.95,
        help="llm-webkit DOM structural similarity threshold for host-bounded layout clustering.",
    )
    parser.add_argument(
        "--layout-page-signature-mode",
        choices=[
            "none",
            "url_shape",
            "url_low_card_query_shape",
            "url_semantic_shape",
            "item_count_bucket",
            "item_count_exact",
            "url_shape_item_count_bucket",
            "url_shape_item_count_exact",
            "url_low_card_query_shape_item_count_bucket",
            "url_low_card_query_shape_item_count_exact",
            "url_semantic_shape_item_count_bucket",
            "url_semantic_shape_item_count_exact",
        ],
        default="none",
        help="Optional cheap split applied inside each host/layout cluster before representative selection.",
    )
    parser.add_argument(
        "--layout-template-failed-host-fallback-signature-mode",
        choices=[
            "none",
            "url_shape",
            "url_low_card_query_shape",
            "url_semantic_shape",
            "item_count_bucket",
            "item_count_exact",
            "url_shape_item_count_bucket",
            "url_shape_item_count_exact",
            "url_low_card_query_shape_item_count_bucket",
            "url_low_card_query_shape_item_count_exact",
            "url_semantic_shape_item_count_bucket",
            "url_semantic_shape_item_count_exact",
        ],
        default="none",
        help="Optional cheap split applied to DOM fallback groups only after a host-single template attempt fails.",
    )
    parser.add_argument(
        "--layout-template-failed-layout-fallback-signature-mode",
        choices=[
            "none",
            "url_shape",
            "url_low_card_query_shape",
            "url_semantic_shape",
            "item_count_bucket",
            "item_count_exact",
            "url_shape_item_count_bucket",
            "url_shape_item_count_exact",
            "url_low_card_query_shape_item_count_bucket",
            "url_low_card_query_shape_item_count_exact",
            "url_semantic_shape_item_count_bucket",
            "url_semantic_shape_item_count_exact",
        ],
        default="none",
        help=(
            "Optional cheap child split retried only after a normal layout/precomputed layout template "
            "proposal fails validation."
        ),
    )
    parser.add_argument("--layout-template-min-cluster-size", type=int, default=2)
    parser.add_argument("--layout-template-fallback-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layout-template-require-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--layout-template-max-selected-item-ratio",
        type=float,
        default=0.50,
        help=(
            "Fail closed to LLM when layout propagation selects more than this fraction of target _item_id nodes. "
            "Use 0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--layout-template-more-noise-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow llm-webkit layout propagation to keep unmatched natural-language noise nodes under main parents.",
    )
    parser.add_argument(
        "--layout-template-validation-rows",
        type=int,
        default=2,
        help=(
            "Run full LLM extraction on this many non-representative rows per layout cluster before propagating "
            "the template to the rest of the cluster."
        ),
    )
    parser.add_argument(
        "--layout-template-validation-min-content-f1",
        type=float,
        default=0.98,
        help="Minimum token-F1 between propagated and validation LLM content required to trust a layout cluster.",
    )
    parser.add_argument(
        "--layout-template-validation-signature-mode",
        choices=[
            "none",
            "url_shape",
            "url_low_card_query_shape",
            "url_semantic_shape",
            "item_count_bucket",
            "item_count_exact",
            "url_shape_item_count_bucket",
            "url_shape_item_count_exact",
            "url_low_card_query_shape_item_count_bucket",
            "url_low_card_query_shape_item_count_exact",
            "url_semantic_shape_item_count_bucket",
            "url_semantic_shape_item_count_exact",
        ],
        default="none",
        help=(
            "Optional cheap signature used only for choosing validation rows inside a layout cluster. "
            "This does not split the cluster; it spends the validation budget across diverse URL/item-count buckets."
        ),
    )
    parser.add_argument(
        "--layout-template-large-cluster-validation-rows",
        type=int,
        default=0,
        help=(
            "If positive, use at least this many validation rows for layout clusters whose size is at least "
            "--layout-template-large-cluster-min-size."
        ),
    )
    parser.add_argument(
        "--layout-template-large-cluster-min-size",
        type=int,
        default=0,
        help="Minimum layout-cluster size that triggers --layout-template-large-cluster-validation-rows.",
    )
    parser.add_argument(
        "--layout-template-representative-candidates",
        type=int,
        default=1,
        help=(
            "Maximum representative candidates to try per layout cluster before falling back to per-page LLM. "
            "The llm-webkit selected representative is tried first."
        ),
    )
    parser.add_argument(
        "--layout-template-propagation-target",
        choices=["raw_html", "mapped_item_ids"],
        default="raw_html",
        help=(
            "HTML source passed to llm-webkit LayoutBatchParser for sibling propagation. "
            "raw_html matches upstream llm-webkit; mapped_item_ids keeps the older MinerU item-id remapping path."
        ),
    )
    parser.add_argument(
        "--layout-template-min-main-html-sim",
        type=float,
        default=None,
        help=(
            "Optional stricter minimum llm-webkit main_html_sim for accepting propagated layout output when "
            "the parser reports that similarity. Unset keeps llm-webkit's built-in success threshold."
        ),
    )
    parser.add_argument(
        "--layout-template-min-content-length-ratio",
        type=float,
        default=None,
        help=(
            "Optional fail-closed guard: reject propagated content when its character length is below this "
            "fraction of the representative content length."
        ),
    )
    parser.add_argument(
        "--layout-template-max-content-length-ratio",
        type=float,
        default=None,
        help=(
            "Optional fail-closed guard: reject propagated content when its character length exceeds this "
            "multiple of the representative content length."
        ),
    )
    parser.add_argument(
        "--layout-template-defer-fallback-llm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep layout-template fallback and standalone rows in the normal inference/postprocess stages instead "
            "of issuing those LLM calls inside the CPU layout-template stage."
        ),
    )
    parser.add_argument(
        "--layout-template-defer-propagation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip LayoutBatchParser propagation inside the GPU stage. Sibling rows are marked "
            "dripper_layout_pending_propagation=True and the mapping JSON is stored so a separate "
            "DripperHTMLLayoutPropagationStage can run propagation on cheap CPU nodes afterwards. "
            "Removes ~23,000s of CPU work from the H100 critical path."
        ),
    )
    parser.add_argument(
        "--layout-template-host-single-cluster-min-pages",
        type=int,
        default=0,
        help=(
            "If positive, first try one representative/template for a host with at least this many pages. "
            "Failed host attempts fall back to normal DOM-layout groups."
        ),
    )
    parser.add_argument(
        "--layout-template-host-single-cluster-max-pages",
        type=int,
        default=0,
        help=(
            "Optional upper bound for --layout-template-host-single-cluster-min-pages. "
            "Use 0 for no upper bound."
        ),
    )
    parser.add_argument(
        "--layout-template-max-exact-host-pages",
        type=int,
        default=0,
        help=(
            "If positive, skip exact O(n^2) DOM DBSCAN for hosts above this many LLM-needed pages. "
            "Use with --layout-template-large-host-mode feature_hash or dom_path_hash to still reuse conservative layouts."
        ),
    )
    parser.add_argument(
        "--layout-template-large-host-mode",
        choices=["standalone", "feature_hash", "dom_path_hash"],
        default="standalone",
        help=(
            "How layout-template mode handles hosts above --layout-template-max-exact-host-pages. "
            "standalone leaves them as per-page LLM calls; feature_hash groups exact normalized DOM bag features; "
            "dom_path_hash groups a stricter normalized DOM tree fingerprint."
        ),
    )
    parser.add_argument(
        "--layout-template-propagation-concurrency",
        type=int,
        default=32,
        help="Maximum CPU worker-thread fanout for llm-webkit layout propagation inside one stage actor.",
    )
    parser.add_argument("--dynamic-classid-similarity-threshold", type=float, default=0.85)
    parser.add_argument("--warmup-pages", type=int, default=0)
    parser.add_argument("--h100-count", type=int, default=1)
    parser.add_argument("--snapshot-pages", type=int, default=DEFAULT_SNAPSHOT_PAGES)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-backend", choices=["ray_serve", "dynamo"], default="ray_serve")
    parser.add_argument("--dynamo-mode", choices=["aggregated", "disagg"], default="aggregated")
    parser.add_argument("--dynamo-prefill-replicas", type=int, default=1)
    parser.add_argument("--dynamo-decode-replicas", type=int, default=1)
    parser.add_argument(
        "--dynamo-router-mode",
        choices=[
            "auto",
            "round-robin",
            "round_robin",
            "random",
            "power-of-two",
            "kv",
            "direct",
            "least-loaded",
            "device-aware-weighted",
        ],
        default="auto",
    )
    parser.add_argument("--dynamo-router-kv-events", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamo-etcd-endpoint", default=None)
    parser.add_argument("--dynamo-nats-url", default=None)
    parser.add_argument("--ray-temp-dir", default=os.environ.get("RAY_TMPDIR", "/tmp/ray_dripper"))
    parser.add_argument("--ray-port", type=int, default=None)
    parser.add_argument("--ray-dashboard-port", type=int, default=None)
    parser.add_argument("--ray-client-server-port", type=int, default=None)
    parser.add_argument("--ray-metrics-port", type=int, default=None)
    parser.add_argument("--ray-min-worker-port", type=int, default=None)
    parser.add_argument("--ray-max-worker-port", type=int, default=None)
    parser.add_argument("--ray-dashboard-host", default=os.environ.get("RAY_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--ray-num-cpus", type=int, default=None)
    parser.add_argument("--ray-num-gpus", type=int, default=None)
    parser.add_argument("--ray-object-store-memory-gb", type=float, default=None)
    parser.add_argument("--ray-worker-connect-timeout-s", type=int, default=600)
    parser.add_argument("--ray-cleanup-on-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ray-include-dashboard-metrics", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    job_started = time.perf_counter()
    args = parse_args()
    if args.max_pages < 0:
        raise ValueError("--max-pages must be non-negative; use 0 to exhaust selected WARCs")
    if args.replicas <= 0:
        raise ValueError("--replicas must be positive")
    if args.dynamo_prefill_replicas <= 0:
        raise ValueError("--dynamo-prefill-replicas must be positive")
    if args.dynamo_decode_replicas <= 0:
        raise ValueError("--dynamo-decode-replicas must be positive")
    if args.warmup_pages < 0:
        raise ValueError("--warmup-pages must be non-negative")
    if args.min_html_bytes < 0:
        raise ValueError("--min-html-bytes must be non-negative")
    if args.manifest_fetch_workers <= 0:
        raise ValueError("--manifest-fetch-workers must be positive")
    if args.deployment_max_ongoing_requests is not None and args.deployment_max_ongoing_requests <= 0:
        raise ValueError("--deployment-max-ongoing-requests must be positive")
    if args.ingress_replicas is not None and args.ingress_replicas <= 0:
        raise ValueError("--ingress-replicas must be positive")
    if args.ingress_max_ongoing_requests is not None and args.ingress_max_ongoing_requests <= 0:
        raise ValueError("--ingress-max-ongoing-requests must be positive")
    if args.ingress_target_ongoing_requests is not None and args.ingress_target_ongoing_requests <= 0:
        raise ValueError("--ingress-target-ongoing-requests must be positive")
    if args.pipeline_shard_size <= 0:
        raise ValueError("--pipeline-shard-size must be positive")
    if args.precompute_layout_manifest_only:
        args.layout_template_precompute_layout_ids = True
    if args.layout_template_precompute_layout_ids and not args.layout_template_layout_id_col:
        args.layout_template_layout_id_col = DEFAULT_LAYOUT_ID_COL
    if args.pipeline_shard_strategy == "layout_complete" and not args.layout_template_layout_id_col:
        args.layout_template_layout_id_col = DEFAULT_LAYOUT_ID_COL
    for worker_arg in (
        "pipeline_preprocess_workers",
        "pipeline_inference_workers",
        "pipeline_postprocess_workers",
        "pipeline_layout_workers",
    ):
        value = getattr(args, worker_arg)
        if value is not None and value <= 0:
            raise ValueError(f"--{worker_arg.replace('_', '-')} must be positive when set")
    if args.dynamic_max_token_padding < 0:
        raise ValueError("--dynamic-max-token-padding must be non-negative")
    if args.dynamic_max_tokens_per_item <= 0:
        raise ValueError("--dynamic-max-tokens-per-item must be positive")
    if args.dynamic_min_max_tokens <= 0:
        raise ValueError("--dynamic-min-max-tokens must be positive")
    if not 0.0 < args.layout_cluster_threshold <= 1.0:
        raise ValueError("--layout-cluster-threshold must be in (0, 1]")
    if args.layout_template_min_cluster_size <= 1:
        raise ValueError("--layout-template-min-cluster-size must be greater than 1")
    if args.layout_template_max_selected_item_ratio < 0 or args.layout_template_max_selected_item_ratio > 1.0:
        raise ValueError("--layout-template-max-selected-item-ratio must be in [0, 1]")
    if args.layout_template_validation_rows < 0:
        raise ValueError("--layout-template-validation-rows must be non-negative")
    if args.layout_template_large_cluster_validation_rows < 0:
        raise ValueError("--layout-template-large-cluster-validation-rows must be non-negative")
    if args.layout_template_large_cluster_min_size < 0:
        raise ValueError("--layout-template-large-cluster-min-size must be non-negative")
    if args.layout_template_representative_candidates <= 0:
        raise ValueError("--layout-template-representative-candidates must be positive")
    if args.layout_template_min_main_html_sim is not None and not 0.0 <= args.layout_template_min_main_html_sim <= 1.0:
        raise ValueError("--layout-template-min-main-html-sim must be in [0, 1] when set")
    if args.layout_template_min_content_length_ratio is not None and args.layout_template_min_content_length_ratio < 0:
        raise ValueError("--layout-template-min-content-length-ratio must be non-negative when set")
    if args.layout_template_max_content_length_ratio is not None and args.layout_template_max_content_length_ratio < 0:
        raise ValueError("--layout-template-max-content-length-ratio must be non-negative when set")
    if (
        args.layout_template_min_content_length_ratio is not None
        and args.layout_template_max_content_length_ratio is not None
        and args.layout_template_min_content_length_ratio > args.layout_template_max_content_length_ratio
    ):
        raise ValueError("--layout-template-min-content-length-ratio must be <= --layout-template-max-content-length-ratio")
    if not 0.0 <= args.layout_template_validation_min_content_f1 <= 1.0:
        raise ValueError("--layout-template-validation-min-content-f1 must be in [0, 1]")
    if args.layout_template_host_single_cluster_min_pages < 0:
        raise ValueError("--layout-template-host-single-cluster-min-pages must be non-negative")
    if args.layout_template_host_single_cluster_max_pages < 0:
        raise ValueError("--layout-template-host-single-cluster-max-pages must be non-negative")
    if (
        args.layout_template_host_single_cluster_max_pages > 0
        and args.layout_template_host_single_cluster_min_pages > args.layout_template_host_single_cluster_max_pages
    ):
        raise ValueError(
            "--layout-template-host-single-cluster-min-pages must be <= "
            "--layout-template-host-single-cluster-max-pages when max is set"
        )
    if args.layout_template_max_exact_host_pages < 0:
        raise ValueError("--layout-template-max-exact-host-pages must be non-negative")
    if args.layout_template_propagation_concurrency <= 0:
        raise ValueError("--layout-template-propagation-concurrency must be positive")
    if args.dynamic_classid_similarity_threshold <= 0:
        raise ValueError("--dynamic-classid-similarity-threshold must be positive")
    layout_template_max_selected_item_ratio = (
        None if args.layout_template_max_selected_item_ratio == 0 else args.layout_template_max_selected_item_ratio
    )

    ray_client = build_ray_client(args)
    ray_client.start()
    # On Slurm worker nodes, SlurmRayClient.start() never returns; only the
    # head process continues into WARC loading, serving, and extraction.
    ray_start_s = time.perf_counter() - job_started
    server: InferenceServer | None = None

    try:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        _log_environment(args)
        page_load_started = time.perf_counter()
        pages, warc_paths, load_stats = load_input_pages(args)
        page_load_s = time.perf_counter() - page_load_started
        if not pages:
            raise RuntimeError("No HTML pages were loaded from the requested Common Crawl sample")
        logger.info("Loaded {} HTML page(s) from {} WARC path(s)", len(pages), len(warc_paths))

        layout_precompute_s = 0.0
        if args.layout_template_precompute_layout_ids:
            precompute_started = time.perf_counter()
            pages = precompute_layout_ids(
                args,
                pages,
                task_id="cc-main-2025-26-dripper-layout-precompute",
                dataset_name="CC-MAIN-2025-26",
            )
            layout_precompute_s = time.perf_counter() - precompute_started

        if args.precompute_layout_manifest_only:
            result_df = pd.DataFrame(pages)
            timings = {
                "ray_start_s": ray_start_s,
                "page_load_s": page_load_s,
                "layout_precompute_s": layout_precompute_s,
                "python_end_to_end_s": time.perf_counter() - job_started,
            }
            metrics = build_layout_precompute_metrics(args, result_df, timings, warc_paths, load_stats)
            write_layout_precompute_outputs(output_dir, result_df, metrics)
            logger.info("LAYOUT_PRECOMPUTE_METRICS {}", json.dumps(metrics, sort_keys=True))
            return 0

        server = build_inference_server(args)
        server_start_started = time.perf_counter()
        server.start()
        server_start_s = time.perf_counter() - server_start_started
        client_endpoint = normalize_loopback_endpoint(server.endpoint)
        client_ready_started = time.perf_counter()
        wait_for_openai_models(client_endpoint, args.client_ready_timeout_s)
        client_ready_s = time.perf_counter() - client_ready_started
        stage_setup_s = 0.0
        if args.executor_backend == "direct":
            client = build_openai_client(args, client_endpoint)
            stage = build_dripper_stage(args, client)
            stage_setup_started = time.perf_counter()
            stage.setup()
            stage_setup_s = time.perf_counter() - stage_setup_started
            warmup_elapsed_s, warmup_pages = run_warmup(stage, pages, args)
            result, elapsed_s = run_dripper_batch(
                stage,
                pages,
                task_id="cc-main-2025-26-dripper-smoke",
                dataset_name="CC-MAIN-2025-26",
            )
        else:
            warmup_elapsed_s, warmup_pages = run_warmup_direct(client_endpoint, pages, args)
            result, elapsed_s = run_dripper_pipeline(
                args,
                client_endpoint,
                pages,
                task_id="cc-main-2025-26-dripper-smoke",
                dataset_name="CC-MAIN-2025-26",
            )

        result_df = result.to_pandas()
        timings = {
            "ray_start_s": ray_start_s,
            "page_load_s": page_load_s,
            "server_start_s": server_start_s,
            "client_ready_s": client_ready_s,
            "stage_setup_s": stage_setup_s,
            "warmup_elapsed_s": warmup_elapsed_s,
            "layout_precompute_s": layout_precompute_s,
            "stage_elapsed_s": elapsed_s,
            "python_end_to_end_s": time.perf_counter() - job_started,
        }
        metrics = build_metrics(args, result_df, timings, warc_paths, client_endpoint, warmup_pages, load_stats)
        write_outputs(output_dir, result_df, metrics)
        logger.info("METRICS {}", json.dumps(metrics, sort_keys=True))
    finally:
        try:
            if server is not None:
                server.stop()
        finally:
            ray_client.stop()
    return 0


def normalize_loopback_endpoint(endpoint: str) -> str:
    """Prefer 127.0.0.1 for local OpenAI clients so proxy env vars cannot intercept localhost."""
    parsed = urlparse(endpoint)
    if parsed.hostname != "localhost":
        return endpoint

    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"127.0.0.1{port}"
    return urlunparse(parsed._replace(netloc=netloc))


def build_ray_client(args: argparse.Namespace) -> RayClient:
    kwargs: dict[str, Any] = {
        "ray_temp_dir": args.ray_temp_dir,
        "include_dashboard": args.ray_include_dashboard_metrics,
        "ray_dashboard_host": args.ray_dashboard_host,
    }
    optional_ints = {
        "ray_port": args.ray_port,
        "ray_dashboard_port": args.ray_dashboard_port,
        "ray_client_server_port": args.ray_client_server_port,
        "ray_metrics_port": args.ray_metrics_port,
        "ray_min_worker_port": args.ray_min_worker_port,
        "ray_max_worker_port": args.ray_max_worker_port,
        "num_cpus": args.ray_num_cpus,
        "num_gpus": args.ray_num_gpus,
    }
    kwargs.update({name: value for name, value in optional_ints.items() if value is not None})
    if args.ray_object_store_memory_gb is not None:
        kwargs["object_store_memory"] = int(args.ray_object_store_memory_gb * (1024**3))

    if os.environ.get("SLURM_JOB_ID"):
        kwargs["worker_connect_timeout_s"] = args.ray_worker_connect_timeout_s
        kwargs["cleanup_on_start"] = args.ray_cleanup_on_start
        logger.info("Using SlurmRayClient for Ray lifecycle")
        return SlurmRayClient(**kwargs)

    logger.info("Using RayClient for Ray lifecycle")
    return RayClient(**kwargs)


def build_openai_client(
    args: argparse.Namespace,
    client_endpoint: str,
    *,
    ray_serializable: bool = False,
) -> AsyncOpenAIClient:
    kwargs: dict[str, Any] = {
        "base_url": client_endpoint,
        "api_key": "not-needed",
        "timeout": args.request_timeout_s,
    }
    if not ray_serializable:
        import httpx

        kwargs["http_client"] = httpx.AsyncClient(trust_env=False)

    return AsyncOpenAIClient(
        max_concurrent_requests=args.max_concurrent_requests,
        **kwargs,
    )


def build_dripper_stage(
    args: argparse.Namespace,
    client: AsyncOpenAIClient,
    *,
    health_check: bool = True,
) -> DripperHTMLExtractionStage:
    return DripperHTMLExtractionStage(
        client=client,
        model_name=args.served_model_name,
        html_col="html",
        url_col="url",
        prompt_version=args.prompt_version,
        output_format=args.output_format,
        fallback=args.fallback,
        generation_config=build_generation_config(args),
        dynamic_max_tokens=args.dynamic_max_tokens,
        dynamic_max_token_padding=args.dynamic_max_token_padding,
        dynamic_max_tokens_per_item=args.dynamic_max_tokens_per_item,
        dynamic_min_max_tokens=args.dynamic_min_max_tokens,
        structured_output_mode=args.structured_output_mode,
        max_concurrent_requests=args.max_concurrent_requests,
        health_check=health_check,
    )


def build_dripper_pipeline(args: argparse.Namespace, client_endpoint: str) -> Pipeline:
    generation_config = build_generation_config(args)
    layout_template_max_selected_item_ratio = (
        None if args.layout_template_max_selected_item_ratio == 0 else args.layout_template_max_selected_item_ratio
    )
    pipeline = Pipeline(
        name="dripper_common_crawl",
        description="Dripper HTML extraction split into preprocess, inference, and postprocess stages.",
    )
    pipeline.add_stage(
        DripperHTMLExtractionPipelineStage(
            client=build_openai_client(args, client_endpoint, ray_serializable=True),
            model_name=args.served_model_name,
            html_col="html",
            url_col="url",
            host_col="url_host_name",
            layout_id_col=args.layout_template_layout_id_col,
            prompt_version=args.prompt_version,
            output_format=args.output_format,
            fallback=args.fallback,
            generation_config=generation_config,
            dynamic_max_tokens=args.dynamic_max_tokens,
            dynamic_max_token_padding=args.dynamic_max_token_padding,
            dynamic_max_tokens_per_item=args.dynamic_max_tokens_per_item,
            dynamic_min_max_tokens=args.dynamic_min_max_tokens,
            structured_output_mode=args.structured_output_mode,
            max_concurrent_requests=args.max_concurrent_requests,
            health_check=False,
            keep_intermediate=False,
            preprocess_worker_count=args.pipeline_preprocess_workers,
            inference_worker_count=args.pipeline_inference_workers,
            postprocess_worker_count=args.pipeline_postprocess_workers,
            layout_worker_count=args.pipeline_layout_workers,
            layout_template_mode=args.layout_template_mode,
            layout_cluster_threshold=args.layout_cluster_threshold,
            layout_template_min_cluster_size=args.layout_template_min_cluster_size,
            layout_template_fallback_llm=args.layout_template_fallback_llm,
            layout_template_require_success=args.layout_template_require_success,
            layout_template_max_selected_item_ratio=layout_template_max_selected_item_ratio,
            layout_template_more_noise_enable=args.layout_template_more_noise_enable,
            layout_template_validation_rows=args.layout_template_validation_rows,
            layout_template_validation_min_content_f1=args.layout_template_validation_min_content_f1,
            layout_template_validation_signature_mode=args.layout_template_validation_signature_mode,
            layout_template_large_cluster_validation_rows=args.layout_template_large_cluster_validation_rows,
            layout_template_large_cluster_min_size=args.layout_template_large_cluster_min_size,
            layout_template_representative_candidates=args.layout_template_representative_candidates,
            layout_template_propagation_target=args.layout_template_propagation_target,
            layout_template_min_main_html_sim=args.layout_template_min_main_html_sim,
            layout_template_min_content_length_ratio=args.layout_template_min_content_length_ratio,
            layout_template_max_content_length_ratio=args.layout_template_max_content_length_ratio,
            layout_template_defer_fallback_llm=args.layout_template_defer_fallback_llm,
            layout_template_defer_propagation=args.layout_template_defer_propagation,
            layout_page_signature_mode=args.layout_page_signature_mode,
            layout_template_failed_host_fallback_signature_mode=(
                args.layout_template_failed_host_fallback_signature_mode
            ),
            layout_template_failed_layout_fallback_signature_mode=(
                args.layout_template_failed_layout_fallback_signature_mode
            ),
            layout_template_host_single_cluster_min_pages=args.layout_template_host_single_cluster_min_pages,
            layout_template_host_single_cluster_max_pages=args.layout_template_host_single_cluster_max_pages,
            layout_template_max_exact_host_pages=args.layout_template_max_exact_host_pages,
            layout_template_large_host_mode=args.layout_template_large_host_mode,
            layout_template_propagation_concurrency=args.layout_template_propagation_concurrency,
            dynamic_classid_similarity_threshold=args.dynamic_classid_similarity_threshold,
        )
    )
    if args.layout_template_mode and args.layout_template_defer_propagation:
        pipeline.add_stage(
            DripperHTMLLayoutPropagationStage(
                html_col="html",
                url_col="url",
                dynamic_classid_similarity_threshold=args.dynamic_classid_similarity_threshold,
                more_noise_enable=args.layout_template_more_noise_enable,
                layout_template_validation_min_content_f1=args.layout_template_validation_min_content_f1,
                layout_template_min_content_length_ratio=args.layout_template_min_content_length_ratio,
                layout_template_max_content_length_ratio=args.layout_template_max_content_length_ratio,
                propagation_target=args.layout_template_propagation_target,
            )
        )
    return pipeline


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    extra_kwargs: dict[str, Any] = {}
    if args.disable_thinking:
        extra_kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": False,
                "thinking": False,
            }
        }

    return GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=args.top_p,
        extra_kwargs=extra_kwargs or None,
    )


def run_warmup(
    stage: DripperHTMLExtractionStage,
    pages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[float, int]:
    warmup_pages = min(args.warmup_pages, len(pages))
    if warmup_pages <= 0:
        return 0.0, 0

    _, elapsed_s = run_dripper_batch(
        stage,
        pages[:warmup_pages],
        task_id="cc-main-2025-26-dripper-warmup",
        dataset_name="CC-MAIN-2025-26-warmup",
    )
    logger.info("Warmup processed {} page(s) in {:.3f}s", warmup_pages, elapsed_s)
    return elapsed_s, warmup_pages


def run_warmup_direct(
    client_endpoint: str,
    pages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[float, int]:
    warmup_pages = min(args.warmup_pages, len(pages))
    if warmup_pages <= 0:
        return 0.0, 0

    client = build_openai_client(args, client_endpoint)
    stage = build_dripper_stage(args, client, health_check=False)
    stage.setup()
    _, elapsed_s = run_dripper_batch(
        stage,
        pages[:warmup_pages],
        task_id="cc-main-2025-26-dripper-warmup",
        dataset_name="CC-MAIN-2025-26-warmup",
    )
    logger.info("Warmup processed {} page(s) in {:.3f}s", warmup_pages, elapsed_s)
    return elapsed_s, warmup_pages


def run_dripper_batch(
    stage: DripperHTMLExtractionStage,
    pages: list[dict[str, Any]],
    *,
    task_id: str,
    dataset_name: str,
) -> tuple[DocumentBatch, float]:
    batch = DocumentBatch(
        task_id=task_id,
        dataset_name=dataset_name,
        data=pd.DataFrame(pages),
    )
    started = time.perf_counter()
    result = stage.process(batch)
    return result, time.perf_counter() - started


def precompute_layout_ids(
    args: argparse.Namespace,
    pages: list[dict[str, Any]],
    *,
    task_id: str,
    dataset_name: str,
) -> list[dict[str, Any]]:
    layout_id_col = args.layout_template_layout_id_col or DEFAULT_LAYOUT_ID_COL
    if args.pipeline_shard_strategy != "layout_complete":
        logger.warning(
            "--layout-template-precompute-layout-ids is enabled but shard strategy is {}; "
            "layout IDs will still skip DBSCAN rebuilds, but layout_complete sharding is needed to keep "
            "large layout groups together.",
            args.pipeline_shard_strategy,
        )

    tasks = build_page_tasks(
        pages,
        shard_size=args.pipeline_shard_size,
        shard_strategy="domain_complete",
        task_id=task_id,
        dataset_name=dataset_name,
    )
    pipeline = Pipeline(
        name="dripper_layout_precompute",
        description="Precompute host-bounded llm-webkit DOM layout IDs before Dripper inference.",
    )
    pipeline.add_stage(
        DripperHTMLLayoutClusteringStage(
            html_col="html",
            url_col="url",
            host_col="url_host_name",
            item_count_col="dripper_item_count",
            layout_id_col=layout_id_col,
            layout_cluster_threshold=args.layout_cluster_threshold,
            layout_template_min_cluster_size=args.layout_template_min_cluster_size,
            layout_page_signature_mode=args.layout_page_signature_mode,
            layout_template_max_exact_host_pages=args.layout_template_max_exact_host_pages,
            layout_template_large_host_mode=args.layout_template_large_host_mode,
            worker_count=args.pipeline_layout_workers,
        )
    )
    logger.info(
        "Precomputing Dripper layout IDs with {} domain-complete shard(s), shard_size={}, layout_col={}",
        len(tasks),
        args.pipeline_shard_size,
        layout_id_col,
    )
    output_tasks = pipeline.run(executor=RayDataExecutor(), initial_tasks=tasks) or []
    if not output_tasks:
        raise RuntimeError("Dripper layout precompute produced no output tasks")

    result_df = pd.concat([task.to_pandas() for task in output_tasks], ignore_index=True)
    if "_dripper_row_index" in result_df.columns:
        result_df = result_df.sort_values("_dripper_row_index", kind="stable").drop(columns=["_dripper_row_index"])
    result_df = result_df.reset_index(drop=True)
    assigned = int((result_df[layout_id_col].astype(str) != "").sum()) if layout_id_col in result_df else 0
    logger.info(
        "Precomputed Dripper layout IDs for {}/{} page(s) across {} layout ID(s)",
        assigned,
        len(result_df),
        int(result_df[layout_id_col].nunique()) if layout_id_col in result_df else 0,
    )
    return result_df.to_dict(orient="records")


def run_dripper_pipeline(
    args: argparse.Namespace,
    client_endpoint: str,
    pages: list[dict[str, Any]],
    *,
    task_id: str,
    dataset_name: str,
) -> tuple[DocumentBatch, float]:
    tasks = build_page_tasks(
        pages,
        shard_size=args.pipeline_shard_size,
        shard_strategy=args.pipeline_shard_strategy,
        layout_id_col=args.layout_template_layout_id_col,
        task_id=task_id,
        dataset_name=dataset_name,
    )
    pipeline = build_dripper_pipeline(args, client_endpoint)
    logger.info(
        "Running Dripper pipeline with {} shard(s), shard_size={}, workers pre/layout/infer/post={}/{}/{}/{}",
        len(tasks),
        args.pipeline_shard_size,
        args.pipeline_preprocess_workers or "auto",
        args.pipeline_layout_workers or args.pipeline_inference_workers or "auto",
        args.pipeline_inference_workers or "auto",
        args.pipeline_postprocess_workers or "auto",
    )
    started = time.perf_counter()
    output_tasks = pipeline.run(executor=RayDataExecutor(), initial_tasks=tasks) or []
    elapsed_s = time.perf_counter() - started
    if not output_tasks:
        raise RuntimeError("Dripper pipeline produced no output tasks")

    result_df = pd.concat([task.to_pandas() for task in output_tasks], ignore_index=True)
    if "_dripper_row_index" in result_df.columns:
        result_df = result_df.sort_values("_dripper_row_index", kind="stable").drop(columns=["_dripper_row_index"])
    result_df = result_df.reset_index(drop=True)
    return (
        DocumentBatch(
            task_id=task_id,
            dataset_name=dataset_name,
            data=result_df,
        ),
        elapsed_s,
    )


def build_page_tasks(
    pages: list[dict[str, Any]],
    *,
    shard_size: int,
    shard_strategy: str,
    layout_id_col: str | None = None,
    task_id: str,
    dataset_name: str,
) -> list[DocumentBatch]:
    df = pd.DataFrame(pages).copy()
    df["_dripper_row_index"] = range(len(df))
    if shard_strategy == "balanced_html_bytes":
        return build_balanced_page_tasks(df, shard_size=shard_size, task_id=task_id, dataset_name=dataset_name)
    if shard_strategy == "domain_clustered":
        return build_domain_clustered_page_tasks(df, shard_size=shard_size, task_id=task_id, dataset_name=dataset_name)
    if shard_strategy == "domain_complete":
        return build_domain_complete_page_tasks(df, shard_size=shard_size, task_id=task_id, dataset_name=dataset_name)
    if shard_strategy == "domain_html_hash":
        return build_domain_html_hash_page_tasks(df, shard_size=shard_size, task_id=task_id, dataset_name=dataset_name)
    if shard_strategy == "domain_then_html_bytes":
        return build_domain_then_html_byte_tasks(df, shard_size=shard_size, task_id=task_id, dataset_name=dataset_name)
    if shard_strategy == "layout_complete":
        return build_layout_complete_page_tasks(
            df,
            shard_size=shard_size,
            layout_id_col=layout_id_col or DEFAULT_LAYOUT_ID_COL,
            task_id=task_id,
            dataset_name=dataset_name,
        )
    if shard_strategy != "sequential":
        raise ValueError(f"Unsupported pipeline shard strategy: {shard_strategy}")

    tasks = []
    for shard_index, start in enumerate(range(0, len(df), shard_size)):
        shard = df.iloc[start : start + shard_size].reset_index(drop=True)
        tasks.append(
            DocumentBatch(
                task_id=f"{task_id}-shard-{shard_index:06d}",
                dataset_name=dataset_name,
                data=shard,
            )
        )
    return tasks


def build_domain_clustered_page_tasks(
    df: pd.DataFrame,
    *,
    shard_size: int,
    task_id: str,
    dataset_name: str,
) -> list[DocumentBatch]:
    work = _with_host_keys(df)
    shards: list[list[int]] = []
    current_shard: list[int] = []
    ordered = work.sort_values([_DRIPPER_HOST_KEY_COL, "_dripper_row_index"], kind="stable")
    for _host_key, host_df in ordered.groupby(_DRIPPER_HOST_KEY_COL, sort=False):
        host_indexes = host_df.index.tolist()
        for start in range(0, len(host_indexes), shard_size):
            host_chunk = host_indexes[start : start + shard_size]
            if current_shard and len(current_shard) + len(host_chunk) > shard_size:
                shards.append(current_shard)
                current_shard = []
            current_shard.extend(host_chunk)
            if len(current_shard) >= shard_size:
                shards.append(current_shard)
                current_shard = []
    if current_shard:
        shards.append(current_shard)

    tasks = _tasks_from_shards(
        work,
        shards,
        task_id=task_id,
        dataset_name=dataset_name,
        sort_columns=[_DRIPPER_HOST_KEY_COL, "_dripper_row_index"],
    )
    _log_domain_shards(work, tasks, shard_size=shard_size, strategy="domain_clustered")
    return tasks


def build_domain_complete_page_tasks(
    df: pd.DataFrame,
    *,
    shard_size: int,
    task_id: str,
    dataset_name: str,
) -> list[DocumentBatch]:
    work = _with_host_keys(df)
    ordered = work.sort_values([_DRIPPER_HOST_KEY_COL, "_dripper_row_index"], kind="stable")
    shards: list[list[int]] = []
    current_shard: list[int] = []

    for _host_key, host_df in ordered.groupby(_DRIPPER_HOST_KEY_COL, sort=False):
        host_indexes = host_df.index.tolist()
        if not host_indexes:
            continue
        if current_shard and len(current_shard) + len(host_indexes) > shard_size:
            shards.append(current_shard)
            current_shard = []
        if len(host_indexes) >= shard_size:
            shards.append(host_indexes)
            continue
        current_shard.extend(host_indexes)
    if current_shard:
        shards.append(current_shard)

    tasks = _tasks_from_shards(
        work,
        shards,
        task_id=task_id,
        dataset_name=dataset_name,
        sort_columns=[_DRIPPER_HOST_KEY_COL, "_dripper_row_index"],
    )
    _log_domain_shards(work, tasks, shard_size=shard_size, strategy="domain_complete")
    return tasks


def build_layout_complete_page_tasks(
    df: pd.DataFrame,
    *,
    shard_size: int,
    layout_id_col: str,
    task_id: str,
    dataset_name: str,
) -> list[DocumentBatch]:
    work = _with_layout_keys(df, layout_id_col)
    ordered = work.sort_values([_DRIPPER_LAYOUT_KEY_COL, "_dripper_row_index"], kind="stable")
    shards: list[list[int]] = []
    current_shard: list[int] = []

    for _layout_key, layout_df in ordered.groupby(_DRIPPER_LAYOUT_KEY_COL, sort=False):
        layout_indexes = layout_df.index.tolist()
        if not layout_indexes:
            continue
        if current_shard and len(current_shard) + len(layout_indexes) > shard_size:
            shards.append(current_shard)
            current_shard = []
        if len(layout_indexes) >= shard_size:
            shards.append(layout_indexes)
            continue
        current_shard.extend(layout_indexes)
    if current_shard:
        shards.append(current_shard)

    tasks = _tasks_from_shards(
        work,
        shards,
        task_id=task_id,
        dataset_name=dataset_name,
        sort_columns=[_DRIPPER_LAYOUT_KEY_COL, "_dripper_row_index"],
    )
    _log_layout_shards(work, tasks, shard_size=shard_size, layout_id_col=layout_id_col)
    return tasks


def build_domain_html_hash_page_tasks(
    df: pd.DataFrame,
    *,
    shard_size: int,
    task_id: str,
    dataset_name: str,
) -> list[DocumentBatch]:
    work = _with_host_keys(df)
    work[_DRIPPER_HTML_HASH_COL] = work["html"].map(_html_hash_key)
    shards: list[list[int]] = []
    current_shard: list[int] = []
    ordered = work.sort_values([_DRIPPER_HOST_KEY_COL, _DRIPPER_HTML_HASH_COL, "_dripper_row_index"], kind="stable")
    for _host_key, host_df in ordered.groupby(_DRIPPER_HOST_KEY_COL, sort=False):
        host_indexes = host_df.index.tolist()
        for start in range(0, len(host_indexes), shard_size):
            host_chunk = host_indexes[start : start + shard_size]
            if current_shard and len(current_shard) + len(host_chunk) > shard_size:
                shards.append(current_shard)
                current_shard = []
            current_shard.extend(host_chunk)
            if len(current_shard) >= shard_size:
                shards.append(current_shard)
                current_shard = []
    if current_shard:
        shards.append(current_shard)

    tasks = _tasks_from_shards(
        work,
        shards,
        task_id=task_id,
        dataset_name=dataset_name,
        sort_columns=[_DRIPPER_HOST_KEY_COL, _DRIPPER_HTML_HASH_COL, "_dripper_row_index"],
    )
    _log_domain_shards(work, tasks, shard_size=shard_size, strategy="domain_html_hash")
    return tasks


def build_domain_then_html_byte_tasks(
    df: pd.DataFrame,
    *,
    shard_size: int,
    task_id: str,
    dataset_name: str,
) -> list[DocumentBatch]:
    work = _with_host_keys(df)
    work[_DRIPPER_HTML_BYTES_COL] = work["html"].map(_byte_len).astype("int64")

    host_chunks: list[tuple[str, list[int], int, int]] = []
    ordered = work.sort_values([_DRIPPER_HOST_KEY_COL, "_dripper_row_index"], kind="stable")
    for host_key, host_df in ordered.groupby(_DRIPPER_HOST_KEY_COL, sort=False):
        row_indexes = host_df.index.tolist()
        for start in range(0, len(row_indexes), shard_size):
            chunk_indexes = row_indexes[start : start + shard_size]
            chunk_bytes = int(work.loc[chunk_indexes, _DRIPPER_HTML_BYTES_COL].sum())
            first_row = int(work.loc[chunk_indexes, "_dripper_row_index"].min())
            host_chunks.append((str(host_key), chunk_indexes, chunk_bytes, first_row))

    shard_count = max(1, (len(work) + shard_size - 1) // shard_size)
    shards: list[list[int]] = [[] for _ in range(shard_count)]
    shard_weights = [0 for _ in range(shard_count)]
    shard_rows = [0 for _ in range(shard_count)]

    for _host_key, row_indexes, chunk_bytes, _first_row in sorted(
        host_chunks,
        key=lambda chunk: (-chunk[2], chunk[0], chunk[3]),
    ):
        candidates = [idx for idx in range(len(shards)) if shard_rows[idx] + len(row_indexes) <= shard_size]
        if not candidates:
            shards.append([])
            shard_weights.append(0)
            shard_rows.append(0)
            candidates = [len(shards) - 1]

        shard_index = min(candidates, key=lambda idx: (shard_weights[idx], shard_rows[idx], idx))
        shards[shard_index].extend(row_indexes)
        shard_weights[shard_index] += chunk_bytes
        shard_rows[shard_index] += len(row_indexes)

    tasks = _tasks_from_shards(
        work,
        shards,
        task_id=task_id,
        dataset_name=dataset_name,
        sort_columns=[_DRIPPER_HOST_KEY_COL, "_dripper_row_index"],
    )
    _log_domain_shards(work, tasks, shard_size=shard_size, strategy="domain_then_html_bytes")
    return tasks


def build_balanced_page_tasks(
    df: pd.DataFrame,
    *,
    shard_size: int,
    task_id: str,
    dataset_name: str,
) -> list[DocumentBatch]:
    shard_count = max(1, (len(df) + shard_size - 1) // shard_size)
    shards: list[list[int]] = [[] for _ in range(shard_count)]
    shard_weights = [0 for _ in range(shard_count)]
    weights = df["html"].map(_byte_len).astype("int64")

    for row_index in weights.sort_values(ascending=False).index:
        shard_index = min(
            (idx for idx in range(shard_count) if len(shards[idx]) < shard_size),
            key=lambda idx: (shard_weights[idx], len(shards[idx]), idx),
        )
        shards[shard_index].append(row_index)
        shard_weights[shard_index] += int(weights.at[row_index])

    non_empty_weights = pd.Series([weight for weight, shard in zip(shard_weights, shards, strict=True) if shard])
    if len(non_empty_weights):
        logger.info(
            "Built {} balanced shard(s) by input HTML bytes: shard_size={}, p50_bytes={}, p95_bytes={}, max_bytes={}",
            len(non_empty_weights),
            shard_size,
            int(non_empty_weights.quantile(0.5)),
            int(non_empty_weights.quantile(0.95)),
            int(non_empty_weights.max()),
        )

    tasks = []
    for shard_index, row_indexes in enumerate(shards):
        if not row_indexes:
            continue
        shard = df.loc[row_indexes].sort_values("_dripper_row_index", kind="stable").reset_index(drop=True)
        tasks.append(
            DocumentBatch(
                task_id=f"{task_id}-shard-{shard_index:06d}",
                dataset_name=dataset_name,
                data=shard,
            )
        )
    return tasks


def _with_host_keys(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    url_values = work["url"].tolist() if "url" in work.columns else [None] * len(work)
    work[_DRIPPER_HOST_KEY_COL] = [
        _host_key_or_row_fallback(url_value, row_index)
        for url_value, row_index in zip(url_values, work["_dripper_row_index"].tolist(), strict=True)
    ]
    return work


def _with_layout_keys(df: pd.DataFrame, layout_id_col: str) -> pd.DataFrame:
    if layout_id_col not in df.columns:
        raise ValueError(
            f"--pipeline-shard-strategy layout_complete requires layout ID column {layout_id_col!r}"
        )
    work = df.copy()
    url_values = work["url"].tolist() if "url" in work.columns else [None] * len(work)
    work[_DRIPPER_LAYOUT_KEY_COL] = [
        _layout_key_or_row_fallback(layout_id, row_index, url_value)
        for layout_id, row_index, url_value in zip(
            work[layout_id_col].tolist(),
            work["_dripper_row_index"].tolist(),
            url_values,
            strict=True,
        )
    ]
    return work


def _html_hash_key(value: Any) -> str:
    if _is_missing_scalar(value):
        data = b""
    elif isinstance(value, bytes | bytearray | memoryview):
        data = bytes(value)
    else:
        data = str(value).encode("utf-8", errors="replace")
    return hashlib.sha256(data).hexdigest()


def _host_key_or_row_fallback(url_value: Any, row_index: Any) -> str:
    host_key = _url_host_key(url_value)
    if host_key:
        return host_key
    try:
        row_id = int(row_index)
    except (TypeError, ValueError):
        row_id = 0
    return f"~missing-host-{row_id:012d}"


def _layout_key_or_row_fallback(layout_id: Any, row_index: Any, url_value: Any = None) -> str:
    if not _is_missing_scalar(layout_id):
        key = str(layout_id).strip()
        if key and key not in {"-1", "-2"} and not key.endswith("_-1") and not key.endswith("_-2"):
            return key
    # Unassigned pages: group by host so they share shards instead of becoming
    # singleton shards (one per row), which serializes scheduling.
    host = _url_host_key(url_value) if url_value is not None else ""
    if host:
        return f"~unassigned-host-{host}"
    try:
        row_id = int(row_index)
    except (TypeError, ValueError):
        row_id = 0
    return f"~unassigned-layout-{row_id:012d}"


def _url_host_key(url_value: Any) -> str:
    """Return llm-webkit-compatible full lowercase hostname for URL locality grouping."""
    if _is_missing_scalar(url_value):
        return ""

    url_text = str(url_value).strip()
    if not url_text:
        return ""

    host = _parsed_hostname(url_text)
    if not host and "://" not in url_text:
        host = _parsed_hostname(f"//{url_text}")
    host = host.rstrip(".").lower()
    if not host:
        return ""

    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass

    return host


def _parsed_hostname(url_text: str) -> str:
    try:
        return urlparse(url_text).hostname or ""
    except ValueError:
        return ""


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _tasks_from_shards(
    df: pd.DataFrame,
    shards: list[list[int]],
    *,
    task_id: str,
    dataset_name: str,
    sort_columns: list[str],
) -> list[DocumentBatch]:
    tasks = []
    for shard_index, row_indexes in enumerate(shards):
        if not row_indexes:
            continue
        shard = df.loc[row_indexes].sort_values(sort_columns, kind="stable")
        shard = shard.drop(
            columns=[
                _DRIPPER_HOST_KEY_COL,
                _DRIPPER_LAYOUT_KEY_COL,
                _DRIPPER_HTML_BYTES_COL,
                _DRIPPER_HTML_HASH_COL,
            ],
            errors="ignore",
        )
        tasks.append(
            DocumentBatch(
                task_id=f"{task_id}-shard-{shard_index:06d}",
                dataset_name=dataset_name,
                data=shard.reset_index(drop=True),
            )
        )
    return tasks


def _log_domain_shards(
    work: pd.DataFrame,
    tasks: list[DocumentBatch],
    *,
    shard_size: int,
    strategy: str,
) -> None:
    host_sizes = work.groupby(_DRIPPER_HOST_KEY_COL, sort=False).size()
    shard_bytes = pd.Series(
        [task.to_pandas()["html"].map(_byte_len).sum() for task in tasks],
        dtype="int64",
    )
    html_hashes = work[_DRIPPER_HTML_HASH_COL] if _DRIPPER_HTML_HASH_COL in work else work["html"].map(_html_hash_key)
    exact_html_duplicate_pages = max(0, len(html_hashes) - int(html_hashes.nunique()))
    if len(host_sizes) and len(shard_bytes):
        logger.info(
            "Built {} {} shard(s): shard_size={}, host_keys={}, p95_host_pages={}, "
            "max_host_pages={}, exact_html_duplicate_pages={}, p50_shard_bytes={}, "
            "p95_shard_bytes={}, max_shard_bytes={}",
            len(tasks),
            strategy,
            shard_size,
            len(host_sizes),
            int(host_sizes.quantile(0.95)),
            int(host_sizes.max()),
            exact_html_duplicate_pages,
            int(shard_bytes.quantile(0.5)),
            int(shard_bytes.quantile(0.95)),
            int(shard_bytes.max()),
        )


def _log_layout_shards(
    work: pd.DataFrame,
    tasks: list[DocumentBatch],
    *,
    shard_size: int,
    layout_id_col: str,
) -> None:
    layout_sizes = work.groupby(_DRIPPER_LAYOUT_KEY_COL, sort=False).size()
    assigned_layouts = layout_sizes[~layout_sizes.index.astype(str).str.startswith("~unassigned-layout-")]
    shard_bytes = pd.Series(
        [task.to_pandas()["html"].map(_byte_len).sum() for task in tasks],
        dtype="int64",
    )
    if len(layout_sizes) and len(shard_bytes):
        logger.info(
            "Built {} layout_complete shard(s): shard_size={}, layout_col={}, layout_keys={}, "
            "assigned_layout_keys={}, p95_layout_pages={}, max_layout_pages={}, "
            "p50_shard_bytes={}, p95_shard_bytes={}, max_shard_bytes={}",
            len(tasks),
            shard_size,
            layout_id_col,
            len(layout_sizes),
            len(assigned_layouts),
            int(layout_sizes.quantile(0.95)),
            int(layout_sizes.max()),
            int(shard_bytes.quantile(0.5)),
            int(shard_bytes.quantile(0.95)),
            int(shard_bytes.max()),
        )


def _log_environment(args: argparse.Namespace) -> None:
    logger.info("HOST={}", socket.gethostname())
    logger.info("SLURM_JOB_ID={}", os.environ.get("SLURM_JOB_ID", ""))
    logger.info("SLURM_JOB_NODELIST={}", os.environ.get("SLURM_JOB_NODELIST", ""))
    logger.info("COMMAND={}", " ".join(shlex.quote(part) for part in sys.argv))
    logger.info("PYTHON={}", sys.version.replace("\n", " "))
    logger.info("CUDA_VISIBLE_DEVICES={}", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    logger.info("RAY_ADDRESS={}", os.environ.get("RAY_ADDRESS", ""))
    logger.info("RAY_TMPDIR={}", args.ray_temp_dir)
    logger.info("MODEL={}", args.model_identifier)
    logger.info("INPUT_MANIFEST_PATH={}", args.input_manifest_path or "")
    logger.info("WARC_PATHS_URI={}", args.warc_paths_uri)
    logger.info("GPU_SUMMARY={}", _run_command(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"]))


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)  # noqa: S603
    except FileNotFoundError:
        return f"{command[0]} not found"
    except Exception as exc:  # noqa: BLE001
        return f"failed to run {command[0]}: {exc}"
    output = result.stdout.strip() or result.stderr.strip()
    return output.replace("\n", " | ")


def wait_for_openai_models(base_url: str, timeout_s: int) -> None:
    """Wait until the local OpenAI-compatible endpoint is reachable without proxies."""
    models_url = f"{base_url.rstrip('/')}/models"
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with opener.open(models_url, timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    logger.info("OpenAI client endpoint ready at {}", models_url)
                    return
        except (OSError, URLError) as exc:
            last_error = str(exc)
        time.sleep(1)

    raise TimeoutError(f"OpenAI client endpoint did not become reachable at {models_url}: {last_error}")


def build_inference_server(args: argparse.Namespace) -> InferenceServer:
    deployment_config = {
        "autoscaling_config": {
            "min_replicas": args.replicas,
            "max_replicas": args.replicas,
        }
    }
    if args.deployment_max_ongoing_requests is not None:
        deployment_config["max_ongoing_requests"] = args.deployment_max_ongoing_requests
    engine_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
    }
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    engine_kwargs["enable_prefix_caching"] = args.enable_prefix_caching
    if args.enable_chunked_prefill is not None:
        engine_kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    add_optional_engine_kwargs(args, engine_kwargs)

    logger.info("{} engine kwargs: {}", args.inference_backend, engine_kwargs)
    model_config, backend_config = build_model_server_config(args, deployment_config, engine_kwargs)

    server_kwargs: dict[str, Any] = {
        "models": [model_config],
        "port": args.server_port,
        "health_check_timeout_s": args.health_check_timeout_s,
        "verbose": args.server_verbose,
    }
    if backend_config is not None:
        server_kwargs["backend"] = backend_config
    return InferenceServer(**server_kwargs)


def add_optional_engine_kwargs(args: argparse.Namespace, engine_kwargs: dict[str, Any]) -> None:
    """Pass optional vLLM runtime knobs through without changing defaults."""
    for name in (
        "dtype",
        "quantization",
        "kv_cache_dtype",
        "calculate_kv_scales",
        "generation_config",
        "load_format",
        "safetensors_load_strategy",
        "performance_mode",
        "distributed_executor_backend",
        "attention_backend",
        "async_scheduling",
        "enable_dbo",
        "dbo_decode_token_threshold",
        "dbo_prefill_token_threshold",
        "max_num_partial_prefills",
        "max_long_partial_prefills",
        "long_prefill_token_threshold",
    ):
        value = getattr(args, name, None)
        if value is not None and value != "":
            engine_kwargs[name] = value


def build_model_server_config(
    args: argparse.Namespace,
    deployment_config: dict[str, Any],
    engine_kwargs: dict[str, Any],
) -> tuple[RayServeModelConfig | DynamoVLLMModelConfig, RayServeServerConfig | DynamoServerConfig | None]:
    if args.inference_backend == "ray_serve":
        ingress_deployment_config: dict[str, Any] = {}
        ingress_autoscaling_config: dict[str, Any] = {}
        if args.ingress_replicas is not None:
            ingress_autoscaling_config["min_replicas"] = args.ingress_replicas
            ingress_autoscaling_config["max_replicas"] = args.ingress_replicas
        if args.ingress_target_ongoing_requests is not None:
            ingress_autoscaling_config["target_ongoing_requests"] = args.ingress_target_ongoing_requests
        if ingress_autoscaling_config:
            ingress_deployment_config["autoscaling_config"] = ingress_autoscaling_config
        if args.ingress_max_ongoing_requests is not None:
            ingress_deployment_config["max_ongoing_requests"] = args.ingress_max_ongoing_requests
        return (
            RayServeModelConfig(
                model_identifier=args.model_identifier,
                model_name=args.served_model_name,
                deployment_config=deployment_config,
                engine_kwargs=engine_kwargs,
            ),
            RayServeServerConfig(ingress_deployment_config=ingress_deployment_config),
        )

    router_mode = None if args.dynamo_router_mode == "auto" else args.dynamo_router_mode
    backend = DynamoServerConfig(
        etcd_endpoint=args.dynamo_etcd_endpoint,
        nats_url=args.dynamo_nats_url,
        router=DynamoRouterConfig(mode=router_mode, kv_events=args.dynamo_router_kv_events),
    )
    if args.dynamo_mode == "disagg":
        model = DynamoVLLMModelConfig(
            model_identifier=args.model_identifier,
            model_name=args.served_model_name,
            mode="disagg",
            engine_kwargs=engine_kwargs,
            prefill=DynamoRoleConfig(num_replicas=args.dynamo_prefill_replicas),
            decode=DynamoRoleConfig(num_replicas=args.dynamo_decode_replicas),
        )
    else:
        model = DynamoVLLMModelConfig(
            model_identifier=args.model_identifier,
            model_name=args.served_model_name,
            num_replicas=args.replicas,
            mode="aggregated",
            engine_kwargs=engine_kwargs,
        )
    return model, backend


def load_input_pages(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    if args.input_manifest_path:
        return load_manifest_pages(args)
    return load_common_crawl_pages(args)


def load_manifest_pages(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    manifest_files = resolve_manifest_files(args.input_manifest_path)
    logger.info("Reading input manifest from {} file(s): {}", len(manifest_files), manifest_files[:8])
    manifest_df = read_manifest_dataframe(manifest_files, max_rows=args.max_pages)
    if manifest_df.empty:
        raise RuntimeError(f"Input manifest has no rows: {args.input_manifest_path}")

    stats = {
        "input_manifest_files": len(manifest_files),
        "input_manifest_rows": int(len(manifest_df)),
        "manifest_html_rows_loaded": 0,
        "manifest_warc_rows_requested": 0,
        "manifest_warc_rows_loaded": 0,
        "manifest_rows_skipped_min_bytes": 0,
        "manifest_rows_skipped_non_html": 0,
        "manifest_warc_fetch_failed": 0,
        "stopped_by_max_pages": int(args.max_pages > 0 and len(manifest_df) >= args.max_pages),
    }
    pages: list[dict[str, Any]]
    if "html" in manifest_df.columns or "binary_content" in manifest_df.columns:
        pages = pages_from_manifest_html(manifest_df, args=args, stats=stats)
    else:
        required = {"warc_filename", "warc_record_offset", "warc_record_length"}
        missing = sorted(required.difference(manifest_df.columns))
        if missing:
            raise ValueError(
                "Input manifest must contain html/binary_content or CC WARC byte-range columns; "
                f"missing {missing}"
            )
        pages = fetch_manifest_warc_pages(manifest_df, args=args, stats=stats)

    if args.max_pages > 0:
        pages = pages[: args.max_pages]
    return pages, manifest_files, stats


def resolve_manifest_files(manifest_path: str) -> list[str]:
    paths: list[str] = []
    if any(char in manifest_path for char in "*?["):
        paths = sorted(glob(manifest_path))
    else:
        path = Path(manifest_path)
        if path.is_dir():
            for extension in ("*.parquet", "*.jsonl", "*.json", "*.csv"):
                paths.extend(str(candidate) for candidate in sorted(path.glob(extension)))
        else:
            paths = [manifest_path]
    if not paths:
        raise FileNotFoundError(f"No input manifest files matched {manifest_path!r}")
    return paths


def read_manifest_dataframe(manifest_files: list[str], *, max_rows: int = 0) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rows_remaining = max_rows
    for path in manifest_files:
        if max_rows > 0 and rows_remaining <= 0:
            break
        frame = read_manifest_file(path)
        if max_rows > 0:
            frame = frame.head(rows_remaining)
            rows_remaining -= len(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def read_manifest_file(path: str) -> pd.DataFrame:
    suffixes = "".join(Path(path).suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".jsonl"):
        return pd.read_json(path, orient="records", lines=True)
    if suffixes.endswith(".json"):
        return pd.read_json(path)
    if suffixes.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input manifest file extension: {path}")


def pages_from_manifest_html(
    manifest_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    html_col = "html" if "html" in manifest_df.columns else "binary_content"
    pages: list[dict[str, Any]] = []
    for row in manifest_df.to_dict("records"):
        html = row.get(html_col)
        if _byte_len(html) < args.min_html_bytes:
            stats["manifest_rows_skipped_min_bytes"] += 1
            continue
        content_type = str(row.get("content_type") or row.get("content_mime_type") or row.get("content_mime_detected") or "")
        if args.html_only and content_type and "html" not in content_type.lower():
            stats["manifest_rows_skipped_non_html"] += 1
            continue
        pages.append(
            {
                **row,
                "url": row.get("url"),
                "warc_id": str(row.get("warc_id") or ""),
                "content_type": content_type,
                "html": html,
            }
        )
    stats["manifest_html_rows_loaded"] = len(pages)
    logger.info("Loaded {} page(s) directly from manifest HTML column {}", len(pages), html_col)
    return pages


def fetch_manifest_warc_pages(
    manifest_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    client = make_s3_client(args)
    rows = manifest_df.to_dict("records")
    stats["manifest_warc_rows_requested"] = len(rows)
    pages: list[dict[str, Any] | None] = [None] * len(rows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.manifest_fetch_workers) as executor:
        futures = {
            executor.submit(fetch_manifest_warc_page, client, args.manifest_warc_bucket, row, args): index
            for index, row in enumerate(rows)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                pages[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                stats["manifest_warc_fetch_failed"] += 1
                logger.warning("Manifest WARC fetch failed for row {}: {}", index, exc)

    loaded = [page for page in pages if page is not None]
    stats["manifest_warc_rows_loaded"] = len(loaded)
    logger.info(
        "Fetched {} / {} manifest WARC record(s) with {} worker(s)",
        len(loaded),
        len(rows),
        args.manifest_fetch_workers,
    )
    return loaded


def fetch_manifest_warc_page(
    client: Any,
    default_bucket: str,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    filename = str(row["warc_filename"])
    offset = int(row["warc_record_offset"])
    length = int(row["warc_record_length"])
    bucket, key = parse_manifest_warc_location(default_bucket, filename)
    end_byte = offset + length - 1
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{end_byte}")
    raw_bytes = response["Body"].read()
    try:
        decompressed = gzip.decompress(raw_bytes)
    except gzip.BadGzipFile:
        decompressed = raw_bytes

    for record in ArchiveIterator(io.BytesIO(decompressed), arc2warc=True):
        if record.rec_type != "response":
            continue
        content_type = ""
        if record.http_headers is not None:
            content_type = record.http_headers.get_header("Content-Type") or ""
        if args.html_only and "html" not in content_type.lower():
            return None
        html = record.content_stream().read()
        if len(html) < args.min_html_bytes:
            return None
        warc_id = record.rec_headers.get_header("WARC-Record-ID") or ""
        return {
            **row,
            "url": row.get("url") or record.rec_headers.get_header("WARC-Target-URI"),
            "warc_id": warc_id.strip("<>"),
            "warc_filename": key,
            "content_type": content_type,
            "html": html,
        }
    return None


def parse_manifest_warc_location(default_bucket: str, filename: str) -> tuple[str, str]:
    parsed = urlparse(filename)
    if parsed.scheme == "s3" and parsed.netloc:
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
    elif parsed.scheme in ("http", "https") and parsed.netloc:
        bucket = default_bucket
        key = parsed.path.lstrip("/")
    else:
        bucket = default_bucket
        key = filename.lstrip("/")
    key = normalize_warc_key(bucket, key)
    return bucket, key


def load_common_crawl_pages(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    client = make_s3_client(args)
    warc_bucket, warc_paths_key = parse_s3_uri(args.warc_paths_uri)
    warc_paths = read_warc_paths(client, warc_bucket, warc_paths_key, args.max_warcs)

    pages: list[dict[str, Any]] = []
    used_warc_paths: list[str] = []
    stats = {
        "response_records_seen": 0,
        "html_records_seen": 0,
        "html_records_skipped_min_bytes": 0,
        "warc_paths_considered": 0,
        "warc_paths_exhausted": 0,
        "stopped_by_max_pages": 0,
    }
    for warc_path in warc_paths:
        used_warc_paths.append(warc_path)
        stats["warc_paths_considered"] += 1
        warc_key = normalize_warc_key(warc_bucket, warc_path)
        for record in iter_warc_html_records(
            client,
            warc_bucket,
            warc_key,
            html_only=args.html_only,
            min_html_bytes=args.min_html_bytes,
            stats=stats,
        ):
            pages.append(record)
            if args.max_pages > 0 and len(pages) >= args.max_pages:
                stats["stopped_by_max_pages"] = 1
                return pages, used_warc_paths, stats
        stats["warc_paths_exhausted"] += 1
    return pages, used_warc_paths, stats


def make_s3_client(args: argparse.Namespace) -> Any:
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError("boto3 is required to stream Common Crawl WARC data from S3/PBSS") from exc

    if _is_pbss_endpoint(args.s3_endpoint_url) and os.environ.get("PBSS_ACCESS_KEY_ID"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["PBSS_ACCESS_KEY_ID"]
    if _is_pbss_endpoint(args.s3_endpoint_url) and os.environ.get("PBSS_SECRET_ACCESS_KEY"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["PBSS_SECRET_ACCESS_KEY"]

    max_pool_connections = max(10, int(getattr(args, "manifest_fetch_workers", 10) or 10))
    return boto3.client(
        "s3",
        endpoint_url=args.s3_endpoint_url,
        region_name=args.s3_region,
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            read_timeout=120,
            max_pool_connections=max_pool_connections,
        ),
    )


def _is_pbss_endpoint(endpoint_url: str | None) -> bool:
    return bool(endpoint_url and "pdx.s8k.io" in endpoint_url)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Expected an s3://bucket/key URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def normalize_warc_key(bucket: str, key: str) -> str:
    """Normalize public Common Crawl paths for the PBSS ``crawl-data`` bucket."""
    if bucket == "crawl-data" and key.startswith("crawl-data/"):
        return key.removeprefix("crawl-data/")
    return key


def read_warc_paths(client: Any, bucket: str, key: str, limit: int) -> list[str]:
    logger.info("Reading WARC paths from s3://{}/{}", bucket, key)
    response = client.get_object(Bucket=bucket, Key=key)
    with gzip.GzipFile(fileobj=response["Body"]) as gz:
        paths = []
        for raw_line in gz:
            line = raw_line.decode("utf-8").strip()
            if line:
                paths.append(line)
            if len(paths) >= limit:
                break
    return paths


def iter_warc_html_records(
    client: Any,
    bucket: str,
    key: str,
    *,
    html_only: bool,
    min_html_bytes: int,
    stats: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    logger.info("Streaming WARC s3://{}/{}", bucket, key)
    response = client.get_object(Bucket=bucket, Key=key)
    for record in ArchiveIterator(response["Body"], arc2warc=True):
        if record.rec_type != "response":
            continue
        if stats is not None:
            stats["response_records_seen"] += 1
        content_type = ""
        if record.http_headers is not None:
            content_type = record.http_headers.get_header("Content-Type") or ""
        if html_only and "html" not in content_type.lower():
            continue
        if stats is not None:
            stats["html_records_seen"] += 1
        warc_id = record.rec_headers.get_header("WARC-Record-ID") or ""
        html = record.content_stream().read()
        if len(html) < min_html_bytes:
            if stats is not None:
                stats["html_records_skipped_min_bytes"] += 1
            continue
        yield {
            "url": record.rec_headers.get_header("WARC-Target-URI"),
            "warc_id": warc_id.strip("<>"),
            "warc_filename": key,
            "content_type": content_type,
            "html": html,
        }


def build_metrics(
    args: argparse.Namespace,
    result_df: pd.DataFrame,
    timings: dict[str, float],
    warc_paths: list[str],
    server_endpoint: str,
    warmup_pages: int,
    load_stats: dict[str, int],
) -> dict[str, Any]:
    pages = len(result_df)
    elapsed_s = timings["stage_elapsed_s"]
    pages_per_second = pages / elapsed_s if elapsed_s > 0 else 0.0
    h100_hours_per_page = (args.h100_count * elapsed_s / 3600) / pages if pages else 0.0
    python_end_to_end_s = timings["python_end_to_end_s"]
    python_end_to_end_h100_hours_per_page = (
        (args.h100_count * python_end_to_end_s / 3600) / pages if pages else 0.0
    )
    errors = result_df["dripper_error"].astype(str) if "dripper_error" in result_df else pd.Series([], dtype=str)
    error_pages = int((errors != "").sum()) if len(errors) else 0
    warnings = (
        result_df["dripper_warning"].astype(str) if "dripper_warning" in result_df else pd.Series([], dtype=str)
    )
    warning_pages = int((warnings != "").sum()) if len(warnings) else 0
    output_content_nonempty = (
        result_df["dripper_content"].astype(str).str.len() > 0
        if "dripper_content" in result_df
        else pd.Series([], dtype=bool)
    )
    output_html_nonempty = (
        result_df["dripper_html"].astype(str).str.len() > 0
        if "dripper_html" in result_df
        else pd.Series([], dtype=bool)
    )
    inference_times = (
        pd.to_numeric(result_df["dripper_inference_time_s"], errors="coerce")
        if "dripper_inference_time_s" in result_df
        else pd.Series([], dtype="float64")
    )
    inference_times = inference_times.dropna()
    preprocess_times = (
        pd.to_numeric(result_df["dripper_preprocess_time_s"], errors="coerce")
        if "dripper_preprocess_time_s" in result_df
        else pd.Series([], dtype="float64")
    ).dropna()
    postprocess_times = (
        pd.to_numeric(result_df["dripper_postprocess_time_s"], errors="coerce")
        if "dripper_postprocess_time_s" in result_df
        else pd.Series([], dtype="float64")
    ).dropna()
    total_times = (
        pd.to_numeric(result_df["dripper_time_s"], errors="coerce")
        if "dripper_time_s" in result_df
        else pd.Series([], dtype="float64")
    ).dropna()
    item_counts = (
        pd.to_numeric(result_df["dripper_item_count"], errors="coerce")
        if "dripper_item_count" in result_df
        else pd.Series([], dtype="float64")
    ).dropna()
    prompt_chars = (
        pd.to_numeric(result_df["dripper_prompt_chars"], errors="coerce")
        if "dripper_prompt_chars" in result_df
        else pd.Series([], dtype="float64")
    ).dropna()
    request_max_tokens = (
        pd.to_numeric(result_df["dripper_request_max_tokens"], errors="coerce")
        if "dripper_request_max_tokens" in result_df
        else pd.Series([], dtype="float64")
    ).dropna()
    llm_candidate_pages = int((request_max_tokens > 0).sum()) if len(request_max_tokens) else 0
    raw_responses = (
        result_df["dripper_response"].astype(str) if "dripper_response" in result_df else pd.Series([], dtype=str)
    )
    prompt_tokens = (
        pd.to_numeric(result_df["dripper_prompt_tokens"], errors="coerce").fillna(0)
        if "dripper_prompt_tokens" in result_df
        else pd.Series([], dtype="float64")
    )
    completion_tokens = (
        pd.to_numeric(result_df["dripper_completion_tokens"], errors="coerce").fillna(0)
        if "dripper_completion_tokens" in result_df
        else pd.Series([], dtype="float64")
    )
    total_tokens = (
        pd.to_numeric(result_df["dripper_total_tokens"], errors="coerce").fillna(0)
        if "dripper_total_tokens" in result_df
        else pd.Series([], dtype="float64")
    )
    token_bearing_response = (
        (prompt_tokens > 0) | (completion_tokens > 0) if len(prompt_tokens) else pd.Series([], dtype=bool)
    )
    layout_representative = _bool_series(result_df, "dripper_layout_representative")
    layout_propagated = _bool_series(result_df, "dripper_layout_propagated")
    layout_propagation_success = _bool_series(result_df, "dripper_layout_propagation_success")
    layout_fallback_llm = _bool_series(result_df, "dripper_layout_fallback_llm")
    layout_standalone_llm = _bool_series(result_df, "dripper_layout_standalone_llm")
    layout_llm_request_pages = 0
    layout_template_saved_call_pages = 0
    layout_template_call_reduction_fraction = 0.0
    layout_category_timing = build_layout_category_timing_metrics(result_df)
    layout_cluster_timing = build_layout_cluster_timing_metrics(result_df)
    layout_baseline_comparison = build_layout_baseline_comparison_metrics(
        args.layout_baseline_output_dir,
        result_df,
    )
    if args.layout_template_mode and len(raw_responses):
        layout_llm_request = layout_representative | layout_fallback_llm | layout_standalone_llm
        response_request_pages = int(layout_llm_request.sum())
        layout_llm_request_pages = response_request_pages
        llm_request_pages = (
            int((token_bearing_response & layout_llm_request).sum()) if len(token_bearing_response) else response_request_pages
        )
        llm_response_pages = int((raw_responses[layout_llm_request] != "").sum())
        llm_empty_response_pages = max(0, response_request_pages - llm_response_pages)
        layout_template_saved_pages = int(layout_propagation_success.sum())
        layout_template_saved_call_pages = max(0, llm_candidate_pages - layout_llm_request_pages)
        layout_template_call_reduction_fraction = (
            layout_template_saved_call_pages / llm_candidate_pages if llm_candidate_pages else 0.0
        )
    else:
        llm_response_pages = int((raw_responses != "").sum()) if len(raw_responses) else llm_candidate_pages
        llm_request_pages = int(token_bearing_response.sum()) if len(token_bearing_response) and token_bearing_response.any() else llm_response_pages
        llm_empty_response_pages = max(0, llm_candidate_pages - llm_response_pages)
        layout_template_saved_pages = 0
    llm_saved_by_exact_prompt_dedup_pages = max(0, llm_response_pages - llm_request_pages)
    input_html_bytes = (
        result_df["html"].map(_byte_len) if "html" in result_df else pd.Series([], dtype="float64")
    )
    input_html_bytes = pd.to_numeric(input_html_bytes, errors="coerce").dropna()
    return {
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "model_identifier": args.model_identifier,
        "served_model_name": args.served_model_name,
        "server_endpoint": server_endpoint,
        "server_port": args.server_port,
        "input_manifest_path": args.input_manifest_path,
        "input_source": "manifest" if args.input_manifest_path else "warc_paths",
        "manifest_warc_bucket": args.manifest_warc_bucket,
        "manifest_fetch_workers": args.manifest_fetch_workers,
        "warc_paths_uri": args.warc_paths_uri,
        "warc_paths_sampled": warc_paths,
        "input_load_stats": load_stats,
        "max_pages": args.max_pages,
        "max_warcs": args.max_warcs,
        "html_only": args.html_only,
        "min_html_bytes": args.min_html_bytes,
        "sample_pages": pages,
        "output_nonempty_pages": int(output_content_nonempty.sum()),
        "output_content_nonempty_pages": int(output_content_nonempty.sum()),
        "output_html_nonempty_pages": int(output_html_nonempty.sum()),
        "error_pages": error_pages,
        "warning_pages": warning_pages,
        "llm_candidate_pages": llm_candidate_pages,
        "llm_request_pages": llm_request_pages,
        "llm_response_pages": llm_response_pages,
        "llm_empty_response_pages": llm_empty_response_pages,
        "llm_saved_by_exact_prompt_dedup_pages": llm_saved_by_exact_prompt_dedup_pages,
        "llm_saved_by_layout_template_pages": layout_template_saved_pages,
        "layout_template_llm_request_pages": layout_llm_request_pages,
        "layout_template_saved_call_pages": layout_template_saved_call_pages,
        "layout_template_call_reduction_fraction": layout_template_call_reduction_fraction,
        "fallback_only_pages": max(0, pages - llm_candidate_pages),
        "warmup_pages": warmup_pages,
        "elapsed_s": elapsed_s,
        "timings_s": timings,
        "pages_per_second": pages_per_second,
        "h100_count": args.h100_count,
        "h100_hours_per_page": h100_hours_per_page,
        "python_end_to_end_h100_hours_per_page": python_end_to_end_h100_hours_per_page,
        "snapshot_pages": args.snapshot_pages,
        "estimated_h100_hours_full_snapshot": h100_hours_per_page * args.snapshot_pages,
        "estimated_h100_hours_full_snapshot_python_end_to_end": python_end_to_end_h100_hours_per_page
        * args.snapshot_pages,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "replicas": args.replicas,
        "tensor_parallel_size": args.tensor_parallel_size,
        "inference_backend": args.inference_backend,
        "dynamo_mode": args.dynamo_mode,
        "dynamo_prefill_replicas": args.dynamo_prefill_replicas,
        "dynamo_decode_replicas": args.dynamo_decode_replicas,
        "dynamo_router_mode": args.dynamo_router_mode,
        "dynamo_router_kv_events": args.dynamo_router_kv_events,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_concurrent_requests": args.max_concurrent_requests,
        "deployment_max_ongoing_requests": args.deployment_max_ongoing_requests,
        "ingress_replicas": args.ingress_replicas,
        "ingress_max_ongoing_requests": args.ingress_max_ongoing_requests,
        "ingress_target_ongoing_requests": args.ingress_target_ongoing_requests,
        "executor_backend": args.executor_backend,
        "pipeline_shard_size": args.pipeline_shard_size,
        "pipeline_shard_strategy": args.pipeline_shard_strategy,
        "layout_template_layout_id_col": args.layout_template_layout_id_col,
        "layout_template_precompute_layout_ids": args.layout_template_precompute_layout_ids,
        "layout_baseline_output_dir": args.layout_baseline_output_dir or "",
        "layout_template_category_timing_s": layout_category_timing,
        "layout_template_top_cluster_timing_s": layout_cluster_timing,
        **layout_baseline_comparison,
        "pipeline_preprocess_workers": args.pipeline_preprocess_workers,
        "pipeline_inference_workers": args.pipeline_inference_workers,
        "pipeline_postprocess_workers": args.pipeline_postprocess_workers,
        "pipeline_layout_workers": args.pipeline_layout_workers,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "calculate_kv_scales": args.calculate_kv_scales,
        "generation_config": args.generation_config,
        "load_format": args.load_format,
        "safetensors_load_strategy": args.safetensors_load_strategy,
        "performance_mode": args.performance_mode,
        "distributed_executor_backend": args.distributed_executor_backend,
        "attention_backend": args.attention_backend,
        "async_scheduling": args.async_scheduling,
        "enable_dbo": args.enable_dbo,
        "dbo_decode_token_threshold": args.dbo_decode_token_threshold,
        "dbo_prefill_token_threshold": args.dbo_prefill_token_threshold,
        "max_num_partial_prefills": args.max_num_partial_prefills,
        "max_long_partial_prefills": args.max_long_partial_prefills,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "server_verbose": args.server_verbose,
        "disable_thinking": args.disable_thinking,
        "prompt_version": args.prompt_version,
        "output_format": args.output_format,
        "fallback": args.fallback,
        "dynamic_max_tokens": args.dynamic_max_tokens,
        "dynamic_max_token_padding": args.dynamic_max_token_padding,
        "dynamic_max_tokens_per_item": args.dynamic_max_tokens_per_item,
        "dynamic_min_max_tokens": args.dynamic_min_max_tokens,
        "structured_output_mode": args.structured_output_mode,
        "layout_template_mode": args.layout_template_mode,
        "layout_cluster_threshold": args.layout_cluster_threshold,
        "layout_template_min_cluster_size": args.layout_template_min_cluster_size,
        "layout_template_fallback_llm": args.layout_template_fallback_llm,
        "layout_template_require_success": args.layout_template_require_success,
        "layout_template_max_selected_item_ratio": args.layout_template_max_selected_item_ratio,
        "layout_template_more_noise_enable": args.layout_template_more_noise_enable,
        "layout_template_validation_rows": args.layout_template_validation_rows,
        "layout_template_validation_min_content_f1": args.layout_template_validation_min_content_f1,
        "layout_template_validation_signature_mode": args.layout_template_validation_signature_mode,
        "layout_template_large_cluster_validation_rows": args.layout_template_large_cluster_validation_rows,
        "layout_template_large_cluster_min_size": args.layout_template_large_cluster_min_size,
        "layout_template_representative_candidates": args.layout_template_representative_candidates,
        "layout_template_propagation_target": args.layout_template_propagation_target,
        "layout_template_min_main_html_sim": args.layout_template_min_main_html_sim,
        "layout_template_min_content_length_ratio": args.layout_template_min_content_length_ratio,
        "layout_template_max_content_length_ratio": args.layout_template_max_content_length_ratio,
        "layout_template_defer_fallback_llm": args.layout_template_defer_fallback_llm,
        "layout_template_defer_propagation": args.layout_template_defer_propagation,
        "layout_page_signature_mode": args.layout_page_signature_mode,
        "layout_template_failed_host_fallback_signature_mode": args.layout_template_failed_host_fallback_signature_mode,
        "layout_template_failed_layout_fallback_signature_mode": (
            args.layout_template_failed_layout_fallback_signature_mode
        ),
        "layout_template_host_single_cluster_min_pages": args.layout_template_host_single_cluster_min_pages,
        "layout_template_host_single_cluster_max_pages": args.layout_template_host_single_cluster_max_pages,
        "layout_template_propagation_concurrency": args.layout_template_propagation_concurrency,
        "dynamic_classid_similarity_threshold": args.dynamic_classid_similarity_threshold,
        "layout_template_representative_pages": int(layout_representative.sum()),
        "layout_template_propagated_pages": int(layout_propagated.sum()),
        "layout_template_propagation_success_pages": int(layout_propagation_success.sum()),
        "layout_template_fallback_llm_pages": int(layout_fallback_llm.sum()),
        "layout_template_standalone_llm_pages": int(layout_standalone_llm.sum()),
        "mean_dripper_preprocess_time_s": float(preprocess_times.mean()) if len(preprocess_times) else 0.0,
        "p50_dripper_preprocess_time_s": float(preprocess_times.quantile(0.5)) if len(preprocess_times) else 0.0,
        "p95_dripper_preprocess_time_s": float(preprocess_times.quantile(0.95)) if len(preprocess_times) else 0.0,
        "mean_dripper_inference_time_s": float(inference_times.mean()) if len(inference_times) else 0.0,
        "p50_dripper_inference_time_s": float(inference_times.quantile(0.5)) if len(inference_times) else 0.0,
        "p95_dripper_inference_time_s": float(inference_times.quantile(0.95)) if len(inference_times) else 0.0,
        "mean_dripper_postprocess_time_s": float(postprocess_times.mean()) if len(postprocess_times) else 0.0,
        "p50_dripper_postprocess_time_s": float(postprocess_times.quantile(0.5)) if len(postprocess_times) else 0.0,
        "p95_dripper_postprocess_time_s": float(postprocess_times.quantile(0.95)) if len(postprocess_times) else 0.0,
        "mean_dripper_total_time_s": float(total_times.mean()) if len(total_times) else 0.0,
        "p50_dripper_total_time_s": float(total_times.quantile(0.5)) if len(total_times) else 0.0,
        "p95_dripper_total_time_s": float(total_times.quantile(0.95)) if len(total_times) else 0.0,
        "mean_dripper_item_count": float(item_counts.mean()) if len(item_counts) else 0.0,
        "p50_dripper_item_count": float(item_counts.quantile(0.5)) if len(item_counts) else 0.0,
        "p95_dripper_item_count": float(item_counts.quantile(0.95)) if len(item_counts) else 0.0,
        "mean_dripper_prompt_chars": float(prompt_chars.mean()) if len(prompt_chars) else 0.0,
        "p50_dripper_prompt_chars": float(prompt_chars.quantile(0.5)) if len(prompt_chars) else 0.0,
        "p95_dripper_prompt_chars": float(prompt_chars.quantile(0.95)) if len(prompt_chars) else 0.0,
        "mean_dripper_request_max_tokens": float(request_max_tokens.mean()) if len(request_max_tokens) else 0.0,
        "p50_dripper_request_max_tokens": float(request_max_tokens.quantile(0.5)) if len(request_max_tokens) else 0.0,
        "p95_dripper_request_max_tokens": float(request_max_tokens.quantile(0.95)) if len(request_max_tokens) else 0.0,
        "total_dripper_prompt_tokens": int(prompt_tokens.sum()) if len(prompt_tokens) else 0,
        "mean_dripper_prompt_tokens": float(prompt_tokens.mean()) if len(prompt_tokens) else 0.0,
        "p50_dripper_prompt_tokens": float(prompt_tokens.quantile(0.5)) if len(prompt_tokens) else 0.0,
        "p95_dripper_prompt_tokens": float(prompt_tokens.quantile(0.95)) if len(prompt_tokens) else 0.0,
        "total_dripper_completion_tokens": int(completion_tokens.sum()) if len(completion_tokens) else 0,
        "mean_dripper_completion_tokens": float(completion_tokens.mean()) if len(completion_tokens) else 0.0,
        "p50_dripper_completion_tokens": float(completion_tokens.quantile(0.5)) if len(completion_tokens) else 0.0,
        "p95_dripper_completion_tokens": float(completion_tokens.quantile(0.95)) if len(completion_tokens) else 0.0,
        "total_dripper_tokens": int(total_tokens.sum()) if len(total_tokens) else 0,
        "mean_dripper_total_tokens": float(total_tokens.mean()) if len(total_tokens) else 0.0,
        "p50_dripper_total_tokens": float(total_tokens.quantile(0.5)) if len(total_tokens) else 0.0,
        "p95_dripper_total_tokens": float(total_tokens.quantile(0.95)) if len(total_tokens) else 0.0,
        "dripper_prompt_tokens_per_second": float(prompt_tokens.sum() / elapsed_s)
        if len(prompt_tokens) and elapsed_s > 0
        else 0.0,
        "dripper_completion_tokens_per_second": float(completion_tokens.sum() / elapsed_s)
        if len(completion_tokens) and elapsed_s > 0
        else 0.0,
        "dripper_total_tokens_per_second": float(total_tokens.sum() / elapsed_s)
        if len(total_tokens) and elapsed_s > 0
        else 0.0,
        "total_input_html_bytes": int(input_html_bytes.sum()) if len(input_html_bytes) else 0,
        "mean_input_html_bytes": float(input_html_bytes.mean()) if len(input_html_bytes) else 0.0,
        "p50_input_html_bytes": float(input_html_bytes.quantile(0.5)) if len(input_html_bytes) else 0.0,
        "p95_input_html_bytes": float(input_html_bytes.quantile(0.95)) if len(input_html_bytes) else 0.0,
        "p99_input_html_bytes": float(input_html_bytes.quantile(0.99)) if len(input_html_bytes) else 0.0,
        "max_input_html_bytes": int(input_html_bytes.max()) if len(input_html_bytes) else 0,
    }


_LAYOUT_BASELINE_KEY_COLUMNS = ("warc_filename", "warc_id", "url")


def build_layout_category_timing_metrics(result_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if result_df.empty or "dripper_postprocess_time_s" not in result_df:
        return {}

    category_rows: dict[str, list[int]] = defaultdict(list)
    for idx, row in result_df.iterrows():
        category_rows[_layout_row_category(row)].append(idx)

    timing_columns = {
        "preprocess": "dripper_preprocess_time_s",
        "inference": "dripper_inference_time_s",
        "postprocess": "dripper_postprocess_time_s",
        "total": "dripper_time_s",
    }
    metrics: dict[str, dict[str, float]] = {}
    for category, indexes in sorted(category_rows.items()):
        category_metrics: dict[str, float] = {"rows": float(len(indexes))}
        category_df = result_df.loc[indexes]
        for label, column in timing_columns.items():
            if column not in category_df:
                continue
            series = pd.to_numeric(category_df[column], errors="coerce").dropna()
            if series.empty:
                continue
            category_metrics[f"{label}_sum"] = float(series.sum())
            category_metrics[f"{label}_mean"] = float(series.mean())
            category_metrics[f"{label}_p50"] = float(series.quantile(0.5))
            category_metrics[f"{label}_p95"] = float(series.quantile(0.95))
        metrics[category] = category_metrics
    return metrics


def build_layout_cluster_timing_metrics(result_df: pd.DataFrame, *, top: int = 20) -> list[dict[str, Any]]:
    if result_df.empty or "dripper_layout_cluster" not in result_df:
        return []

    rows: list[dict[str, Any]] = []
    cluster_indexes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in result_df.iterrows():
        cluster_value = row.get("dripper_layout_cluster")
        cluster_text = "" if _is_missing_scalar(cluster_value) else str(cluster_value)
        if not cluster_text:
            continue
        cluster_indexes[(cluster_text, _layout_host_key(row))].append(idx)

    for (cluster_text, host_key), indexes in cluster_indexes.items():
        cluster_df = result_df.loc[indexes]
        postprocess = (
            pd.to_numeric(cluster_df["dripper_postprocess_time_s"], errors="coerce").dropna()
            if "dripper_postprocess_time_s" in cluster_df
            else pd.Series([], dtype="float64")
        )
        total = (
            pd.to_numeric(cluster_df["dripper_time_s"], errors="coerce").dropna()
            if "dripper_time_s" in cluster_df
            else pd.Series([], dtype="float64")
        )
        rows.append(
            {
                "cluster_id": cluster_text,
                "host": host_key,
                "rows": int(len(cluster_df)),
                "representative_rows": int(_bool_series(cluster_df, "dripper_layout_representative").sum()),
                "propagated_rows": int(_bool_series(cluster_df, "dripper_layout_propagated").sum()),
                "propagation_success_rows": int(_bool_series(cluster_df, "dripper_layout_propagation_success").sum()),
                "fallback_llm_rows": int(_bool_series(cluster_df, "dripper_layout_fallback_llm").sum()),
                "standalone_llm_rows": int(_bool_series(cluster_df, "dripper_layout_standalone_llm").sum()),
                "postprocess_sum": float(postprocess.sum()) if len(postprocess) else 0.0,
                "postprocess_mean": float(postprocess.mean()) if len(postprocess) else 0.0,
                "total_sum": float(total.sum()) if len(total) else 0.0,
                "total_mean": float(total.mean()) if len(total) else 0.0,
            }
        )
    rows.sort(key=lambda row: (row["postprocess_sum"], row["propagated_rows"], row["rows"]), reverse=True)
    return rows[:top]


def build_layout_baseline_comparison_metrics(
    baseline_output_dir: str | None,
    result_df: pd.DataFrame,
) -> dict[str, Any]:
    if not baseline_output_dir:
        return {}
    metrics: dict[str, Any] = {
        "layout_baseline_comparison_available": 0,
        "layout_baseline_comparison_error": "",
    }
    try:
        baseline_df = read_dripper_output_dataframe(Path(baseline_output_dir))
        baseline_rows = {
            _layout_baseline_key(row): row
            for _, row in baseline_df.iterrows()
            if _layout_baseline_key(row)
        }
        if not baseline_rows:
            metrics["layout_baseline_comparison_error"] = "baseline output has no usable row keys"
            return metrics

        propagated = _bool_series(result_df, "dripper_layout_propagated")
        propagated_success = _bool_series(result_df, "dripper_layout_propagation_success")
        propagated_rows = result_df[propagated & propagated_success]
        matched = 0
        missing = 0
        content_mismatch = 0
        baseline_zero_token = 0
        baseline_zero_inference = 0
        baseline_likely_exact_dedup = 0
        baseline_prompt_tokens = 0
        baseline_completion_tokens = 0
        baseline_total_tokens = 0
        for _, row in propagated_rows.iterrows():
            key = _layout_baseline_key(row)
            baseline_row = baseline_rows.get(key)
            if baseline_row is None:
                missing += 1
                continue
            matched += 1
            if _stable_digest(baseline_row.get("dripper_content")) != _stable_digest(row.get("dripper_content")):
                content_mismatch += 1
            total_tokens = _coerce_int(baseline_row.get("dripper_total_tokens"))
            prompt_tokens = _coerce_int(baseline_row.get("dripper_prompt_tokens"))
            completion_tokens = _coerce_int(baseline_row.get("dripper_completion_tokens"))
            inference_time = _coerce_float(baseline_row.get("dripper_inference_time_s"))
            zero_token = total_tokens == 0
            zero_inference = inference_time == 0.0
            baseline_zero_token += int(zero_token)
            baseline_zero_inference += int(zero_inference)
            baseline_likely_exact_dedup += int(zero_token or zero_inference)
            baseline_prompt_tokens += prompt_tokens
            baseline_completion_tokens += completion_tokens
            baseline_total_tokens += total_tokens

        metrics.update(
            {
                "layout_baseline_comparison_available": 1,
                "layout_baseline_rows": int(len(baseline_df)),
                "layout_propagated_baseline_matched_pages": matched,
                "layout_propagated_baseline_missing_pages": missing,
                "layout_propagated_baseline_content_mismatch_pages": content_mismatch,
                "layout_propagated_baseline_zero_token_pages": baseline_zero_token,
                "layout_propagated_baseline_zero_inference_pages": baseline_zero_inference,
                "layout_propagated_baseline_likely_exact_dedup_pages": baseline_likely_exact_dedup,
                "layout_propagated_baseline_non_exact_pages": max(0, matched - baseline_likely_exact_dedup),
                "layout_propagated_baseline_prompt_tokens": baseline_prompt_tokens,
                "layout_propagated_baseline_completion_tokens": baseline_completion_tokens,
                "layout_propagated_baseline_total_tokens": baseline_total_tokens,
            }
        )
    except Exception as exc:  # noqa: BLE001
        metrics["layout_baseline_comparison_error"] = str(exc)
    return metrics


def read_dripper_output_dataframe(output_dir: Path) -> pd.DataFrame:
    parquet_path = output_dir / "dripper_results.parquet"
    jsonl_path = output_dir / "dripper_results.jsonl"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, orient="records", lines=True)
    raise FileNotFoundError(f"No Dripper output rows under {output_dir}")


def _layout_row_category(row: pd.Series) -> str:
    if _truthy_scalar(row.get("dripper_layout_representative")):
        return "layout_representative"
    if _truthy_scalar(row.get("dripper_layout_propagation_success")):
        return "layout_propagated_success"
    if _truthy_scalar(row.get("dripper_layout_propagated")):
        return "layout_propagated_failed"
    if _truthy_scalar(row.get("dripper_layout_fallback_llm")):
        return "layout_fallback_llm"
    if _truthy_scalar(row.get("dripper_layout_standalone_llm")):
        return "layout_standalone_llm"
    if _coerce_int(row.get("dripper_request_max_tokens")) <= 0:
        return "fallback_only"
    return "llm_standard"


def _layout_baseline_key(row: pd.Series) -> str:
    values = []
    for column in _LAYOUT_BASELINE_KEY_COLUMNS:
        if column not in row:
            return ""
        value = row.get(column)
        values.append("" if _is_missing_scalar(value) else str(value))
    return "\0".join(values)


def _layout_host_key(row: pd.Series) -> str:
    for column in ("url_host_name", "host", "domain"):
        if column in row and not _is_missing_scalar(row.get(column)):
            text = str(row.get(column)).strip().lower()
            if text:
                return text
    if "url" not in row or _is_missing_scalar(row.get("url")):
        return ""
    try:
        return (urlparse(str(row.get("url"))).hostname or "").lower()
    except ValueError:
        return ""


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def _truthy_scalar(value: Any) -> bool:
    if _is_missing_scalar(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _coerce_int(value: Any) -> int:
    if _is_missing_scalar(value):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    if _is_missing_scalar(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_layout_precompute_metrics(
    args: argparse.Namespace,
    result_df: pd.DataFrame,
    timings: dict[str, float],
    warc_paths: list[str],
    load_stats: dict[str, int],
) -> dict[str, Any]:
    layout_id_col = args.layout_template_layout_id_col or DEFAULT_LAYOUT_ID_COL
    layout_ids = result_df[layout_id_col].astype(str) if layout_id_col in result_df else pd.Series([], dtype=str)
    assigned = int((layout_ids != "").sum()) if len(layout_ids) else 0
    html_bytes = result_df["html"].map(_byte_len) if "html" in result_df else pd.Series([], dtype="float64")
    html_bytes = pd.to_numeric(html_bytes, errors="coerce").dropna()
    return {
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "input_manifest_path": args.input_manifest_path,
        "input_source": "manifest" if args.input_manifest_path else "warc_paths",
        "manifest_warc_bucket": args.manifest_warc_bucket,
        "manifest_fetch_workers": args.manifest_fetch_workers,
        "warc_paths_uri": args.warc_paths_uri,
        "warc_paths_sampled": warc_paths,
        "input_load_stats": load_stats,
        "max_pages": args.max_pages,
        "max_warcs": args.max_warcs,
        "sample_pages": int(len(result_df)),
        "layout_id_col": layout_id_col,
        "layout_cluster_threshold": args.layout_cluster_threshold,
        "layout_template_min_cluster_size": args.layout_template_min_cluster_size,
        "layout_page_signature_mode": args.layout_page_signature_mode,
        "layout_template_max_exact_host_pages": args.layout_template_max_exact_host_pages,
        "layout_template_large_host_mode": args.layout_template_large_host_mode,
        "pipeline_shard_size": args.pipeline_shard_size,
        "pipeline_layout_workers": args.pipeline_layout_workers,
        "layout_precompute_assigned_pages": assigned,
        "layout_precompute_unassigned_pages": max(0, int(len(result_df)) - assigned),
        "layout_precompute_layout_ids": int(layout_ids[layout_ids != ""].nunique()) if len(layout_ids) else 0,
        "layout_precompute_assignment_fraction": assigned / len(result_df) if len(result_df) else 0.0,
        "timings_s": timings,
        "total_input_html_bytes": int(html_bytes.sum()) if len(html_bytes) else 0,
        "mean_input_html_bytes": float(html_bytes.mean()) if len(html_bytes) else 0.0,
        "p50_input_html_bytes": float(html_bytes.quantile(0.5)) if len(html_bytes) else 0.0,
        "p95_input_html_bytes": float(html_bytes.quantile(0.95)) if len(html_bytes) else 0.0,
        "p99_input_html_bytes": float(html_bytes.quantile(0.99)) if len(html_bytes) else 0.0,
        "max_input_html_bytes": int(html_bytes.max()) if len(html_bytes) else 0,
    }


def _byte_len(value: Any) -> int:
    if isinstance(value, bytes | bytearray):
        return len(value)
    if value is None:
        return 0
    return len(str(value).encode("utf-8"))


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series([False] * len(df), index=df.index)
    return df[column].fillna(False).astype(bool)


def write_outputs(output_dir: Path, result_df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    parquet_path = output_dir / "dripper_results.parquet"
    try:
        result_df.to_parquet(parquet_path, index=False)
        rows_path = parquet_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write parquet output: {}. Falling back to JSONL.", exc)
        rows_path = output_dir / "dripper_results.jsonl"
        result_df.to_json(rows_path, orient="records", lines=True)

    logger.info("Wrote rows to {}", rows_path)
    logger.info("Wrote metrics to {}", metrics_path)


def write_layout_precompute_outputs(output_dir: Path, result_df: pd.DataFrame, metrics: dict[str, Any]) -> None:
    metrics_path = output_dir / "layout_precompute_metrics.json"
    manifest_path = output_dir / "layout_precompute_manifest.parquet"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    result_df.to_parquet(manifest_path, index=False)
    logger.info("Wrote layout precompute manifest to {}", manifest_path)
    logger.info("Wrote layout precompute metrics to {}", metrics_path)


if __name__ == "__main__":
    raise SystemExit(main())
