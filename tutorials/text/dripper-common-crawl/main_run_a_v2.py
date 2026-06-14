#!/usr/bin/env python3
"""
main_run_a_v2.py — Dripper Run A v2: looser validation + looser propagation.

This script is a self-contained experiment driver. All parameters are defined
as constants here so the experiment is fully reproducible without env vars.

WHAT CHANGED FROM RUN A (job 335166) AND WHY
─────────────────────────────────────────────
Run A achieved only 21% LLM call reduction vs theoretical 79%. Root causes:

  Problem 1: Cluster validation too strict (VALIDATION_ROWS=2, F1>=0.95)
    → ~14,000 cluster pages fell to standalone LLM because 2 test pages
      didn't reach F1>=0.95 at apply time.
    → But full-run analysis shows only 2 bad clusters (33 pages) had mean
      F1 < 0.80 across the entire dataset. Validation was over-conservative.
    FIX: VALIDATION_ROWS = 0  (disable cluster validation entirely)
         LARGE_CLUSTER_VALIDATION_ROWS = 0

  Problem 2: Propagation similarity threshold too strict (0.85)
    → 13,469 pages were in accepted clusters but propagation failed
      (e.g. catalogue.eglisejura.com: 641/776 = 82% fallback rate)
    FIX: DYNAMIC_CLASSID_SIMILARITY_THRESHOLD = 0.70

STATS RECORDED IN OUTPUT PARQUET (per-row flags):
  dripper_layout_propagated          bool — templated, no LLM call
  dripper_layout_representative      bool — cluster representative, 1 LLM call
  dripper_layout_fallback_llm        bool — in cluster, propagation failed → LLM
  dripper_layout_standalone_llm      bool — no cluster → standalone LLM
  dripper_layout_cluster             str  — cluster ID
  dripper_layout_propagation_success bool — propagation succeeded (subset of propagated)
  dripper_time_s                     float — total time
  dripper_inference_time_s           float — GPU inference time (0 for templated)
  dripper_postprocess_time_s         float — propagation time (0 for LLM pages)

STATS RECORDED IN metrics.json:
  layout_template_call_reduction_fraction
  layout_template_propagated_pages
  layout_template_fallback_llm_pages
  layout_template_standalone_llm_pages
  layout_template_representative_pages
  layout_template_category_timing_s.{category}.{rows,inference_sum,postprocess_sum}

EXPECTED vs RUN A:
  Templated pages:     ~60-70%  (was 19.1%)
  LLM call reduction:  ~60-70%  (was 21.2%)
  Mean F1 quality:     ~0.985   (was 0.9891) — slight drop from no validation
"""

import os
import sys
from pathlib import Path

# ── Experiment parameters ─────────────────────────────────────────────────────

INPUT_MANIFEST = os.environ.get(
    "INPUT_MANIFEST",
    "/lustre/fsw/portfolios/llmservice/users/vjawa"
    "/nemo_curator_dripper_layout_clustering_20260611_194849"
    "/output_00/layout_precompute_manifest.parquet",
)

# OUTPUT_DIR is set by the SBATCH script via env var so job ID appears in path.
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/lustre/fsw/portfolios/llmservice/users/vjawa/dripper_cc_main_2025_26_smoke/run_a_v2_local",
)

# ── Inference parameters (same as Run A) ─────────────────────────────────────
REPLICAS = 8  # 1 node x 8 H100s
TENSOR_PARALLEL_SIZE = 1  # model fits on 1 GPU
MAX_MODEL_LEN = 32768
MAX_TOKENS = 2048
GPU_MEMORY_UTILIZATION = 0.9
MAX_CONCURRENT_REQUESTS = 128  # more concurrent requests to keep 16 GPUs fed
MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"

# ── Pipeline parameters (same as Run A) ──────────────────────────────────────
PIPELINE_SHARD_SIZE = 64
PIPELINE_SHARD_STRATEGY = "layout_complete"  # keeps same-layout pages together
PIPELINE_WORKERS = 16

# ── Layout clustering (same as Run A) ────────────────────────────────────────
LAYOUT_TEMPLATE_MODE = True
LAYOUT_ID_COL = "dripper_layout_id"  # use precomputed global manifest IDs
LAYOUT_CLUSTER_THRESHOLD = 0.95
LAYOUT_MIN_CLUSTER_SIZE = 2

# ── KEY CHANGES vs Run A ─────────────────────────────────────────────────────
VALIDATION_ROWS = 0  # was 2  → DISABLED
LARGE_CLUSTER_VALIDATION_ROWS = 0  # was 8  → DISABLED
DYNAMIC_CLASSID_SIMILARITY_THRESHOLD = 0.78  # bisect: 0.70 too loose (F1=0.891), 0.85 too strict (19% reduction)

# ── Propagation parameters (same as Run A) ───────────────────────────────────
PROPAGATION_TARGET = "raw_html"
PROPAGATION_CONCURRENCY = 64
REPRESENTATIVE_CANDIDATES = 1
MAX_SELECTED_ITEM_RATIO = 0.5
VALIDATION_MIN_F1 = 0.95
VALIDATION_SIGNATURE_MODE = "url_low_card_query_shape_item_count_exact"
FAILED_LAYOUT_FALLBACK_SIGNATURE = "url_low_card_query_shape_item_count_exact"
FAILED_HOST_FALLBACK_SIGNATURE = "none"
MIN_CONTENT_LENGTH_RATIO = 0.25
MAX_CONTENT_LENGTH_RATIO = 4.0
LAYOUT_PAGE_SIGNATURE_MODE = "none"
LARGE_CLUSTER_MIN_SIZE = 32


def build_argv() -> list[str]:
    """Build the sys.argv list that main.parse_args() will consume."""
    return [
        "main_run_a_v2.py",
        "--input-manifest-path",
        INPUT_MANIFEST,
        "--output-dir",
        OUTPUT_DIR,
        "--max-pages",
        "0",  # process all pages
        # Inference
        "--model-identifier",
        MODEL,
        "--replicas",
        str(REPLICAS),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-tokens",
        str(MAX_TOKENS),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-concurrent-requests",
        str(MAX_CONCURRENT_REQUESTS),
        "--enable-prefix-caching",
        "--disable-thinking",
        "--output-format",
        "mm_md",
        "--prompt-version",
        "short_compact",
        "--fallback",
        "trafilatura",
        "--dynamic-max-tokens",
        "--dynamic-max-token-padding",
        "16",
        "--dynamic-max-tokens-per-item",
        "6",
        "--dynamic-min-max-tokens",
        "32",
        "--structured-output-mode",
        "none",
        # Pipeline
        "--executor-backend",
        "ray_data",
        "--inference-backend",
        "ray_serve",
        "--pipeline-shard-size",
        str(PIPELINE_SHARD_SIZE),
        "--pipeline-shard-strategy",
        PIPELINE_SHARD_STRATEGY,
        "--pipeline-preprocess-workers",
        str(PIPELINE_WORKERS),
        "--pipeline-inference-workers",
        str(PIPELINE_WORKERS),
        "--pipeline-postprocess-workers",
        str(PIPELINE_WORKERS),
        "--pipeline-layout-workers",
        str(PIPELINE_WORKERS),
        # Dynamo router (same as Run A)
        "--dynamo-mode",
        "aggregated",
        "--dynamo-prefill-replicas",
        "1",
        "--dynamo-decode-replicas",
        "1",
        "--dynamo-router-mode",
        "auto",
        # --dynamo-router-kv-events defaults to False, so just omit it
        # Layout template
        "--layout-template-mode",
        "--layout-template-layout-id-col",
        LAYOUT_ID_COL,
        "--layout-cluster-threshold",
        str(LAYOUT_CLUSTER_THRESHOLD),
        "--layout-template-min-cluster-size",
        str(LAYOUT_MIN_CLUSTER_SIZE),
        # KEY CHANGES
        "--layout-template-validation-rows",
        str(VALIDATION_ROWS),
        "--layout-template-large-cluster-validation-rows",
        str(LARGE_CLUSTER_VALIDATION_ROWS),
        "--dynamic-classid-similarity-threshold",
        str(DYNAMIC_CLASSID_SIMILARITY_THRESHOLD),
        # Propagation
        "--layout-template-propagation-target",
        PROPAGATION_TARGET,
        "--layout-template-propagation-concurrency",
        str(PROPAGATION_CONCURRENCY),
        "--layout-template-representative-candidates",
        str(REPRESENTATIVE_CANDIDATES),
        "--layout-template-max-selected-item-ratio",
        str(MAX_SELECTED_ITEM_RATIO),
        "--layout-template-validation-min-content-f1",
        str(VALIDATION_MIN_F1),
        "--layout-template-validation-signature-mode",
        VALIDATION_SIGNATURE_MODE,
        "--layout-template-large-cluster-min-size",
        str(LARGE_CLUSTER_MIN_SIZE),
        "--layout-template-failed-layout-fallback-signature-mode",
        FAILED_LAYOUT_FALLBACK_SIGNATURE,
        "--layout-template-failed-host-fallback-signature-mode",
        FAILED_HOST_FALLBACK_SIGNATURE,
        "--layout-template-min-content-length-ratio",
        str(MIN_CONTENT_LENGTH_RATIO),
        "--layout-template-max-content-length-ratio",
        str(MAX_CONTENT_LENGTH_RATIO),
        "--layout-page-signature-mode",
        LAYOUT_PAGE_SIGNATURE_MODE,
        "--layout-template-fallback-llm",
        "--layout-template-defer-fallback-llm",
        # require_success=False: accept propagation even on partial match,
        # fall back to trafilatura (not LLM) for true failures.
        # This eliminates ~30% of LLM calls that were fallback-to-LLM.
        "--no-layout-template-require-success",
        "--layout-template-more-noise-enable",
    ]


def main() -> int:
    print("=" * 65)
    print("  Dripper Run A v2")
    print("=" * 65)
    print(f"  Input:   {INPUT_MANIFEST}")
    print(f"  Output:  {OUTPUT_DIR}")
    print()
    print("  KEY CHANGES vs Run A (335166):")
    print(f"    validation_rows:             {VALIDATION_ROWS}    (was 2)")
    print(f"    large_cluster_validation:    {LARGE_CLUSTER_VALIDATION_ROWS}    (was 8)")
    print(f"    classid_similarity_thresh:   {DYNAMIC_CLASSID_SIMILARITY_THRESHOLD}  (was 0.85)")
    print("    defer_propagation:           False (was True in job 335798 — broke clustering)")
    print()
    print("  SAME AS RUN A:")
    print(f"    layout_id_col:  {LAYOUT_ID_COL}")
    print(f"    shard_strategy: {PIPELINE_SHARD_STRATEGY}")
    print(f"    replicas:       {REPLICAS}  (8× H100)")
    print("=" * 65)
    print()

    # Inject args and call main.main()
    sys.argv = build_argv()
    sys.path.insert(0, str(Path(__file__).parent))
    import main as dripper_main

    return dripper_main.main()


if __name__ == "__main__":
    sys.exit(main())
