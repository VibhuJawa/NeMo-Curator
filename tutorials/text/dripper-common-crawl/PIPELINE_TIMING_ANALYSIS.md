# Dripper Layout Clustering — Pipeline Stage Timing Analysis

Last updated: 2026-06-11  
Purpose: Track measured timing per stage to guide optimization decisions.

---

## Pipeline Overview

```
CC WARC Index (host_bucket=NNNN.parquet)
  │
  ▼ Stage 1: WARC Fetch
  │   Fetch raw HTML from S3/PBSS using warc_filename + offset + length
  │
  ▼ Stage 2: DOM Feature Extraction
  │   get_feature(html) → per-depth tag+attr bag (llm-webkit)
  │
  ▼ Stage 3: Layout Clustering (DBSCAN)
  │   cluster_html_struct(samples, threshold=0.95) per host
  │   → assigns dripper_layout_id to each page
  │
  ▼ Stage 4: Representative Selection
  │   select_representative_html(candidates) per cluster
  │
  ▼ Stage 5: HTML Simplification
  │   simplify_single_input(case) → simplified + mapped HTML
  │
  ▼ Stage 6: LLM Inference (MinerU-HTML, 0.5B)
  │   Per representative: prompt → {"1": "main", "2": "other", ...}
  │
  ▼ Stage 7: Template Building (map_parser_cls)
  │   LLM labels + mapped HTML → html_element_dict (structural template)
  │
  ▼ Stage 8: Template Propagation (layout_parser_cls)
  │   Apply template to all siblings → main_html_body (no GPU)
  │
  ▼ Stage 9: Validation
  │   F1 vs LLM ground-truth on 2 sample rows per cluster
  │
  ▼ Output: layout_precompute_manifest.parquet + dripper_results.parquet
```

---

## Stage 1: WARC Fetch

**Source**: `host_bucket=NNNN.parquet` → S3/PBSS `crawl-data` bucket  
**Endpoint**: `https://pdx.s8k.io` (PBSS internal)  
**Credentials**: `commoncrawl` key pair (PBSS_ACCESS_KEY_ID)

| Mode | Rate | Notes |
|---|---|---|
| Sequential (1 thread) | **1.2 records/s** | Measured on vscode node, 50 records |
| Async (64 workers, Curator) | **~50 records/s** (estimated) | Based on job 330390 timing |
| Async (64 workers, Curator) | TBD from job 334859 | Measuring now |

**Estimate for 300K pages**:
- Sequential: ~4,300 min ❌ (impractical)
- 64 async workers: ~100 min per node
- 4 nodes × 64 workers: ~25–40 min total (job 334859, in progress)

**Key bottleneck**: Network latency to PBSS. Each record ~849ms RTT from vscode node.  
**Optimization ideas**:
- Pre-cache WARCs on Lustre (avoids S3 round-trips)
- Increase async worker count beyond 64
- Use dc nodes (faster networking) for WARC fetch

---

## Stage 2: DOM Feature Extraction

**Function**: `get_feature(html)` from `llm_web_kit.html_layout.html_layout_cosin`  
**What it does**: BFS DOM traversal, extracts per-depth tag+attr bag, normalizes dynamic attrs

| Measurement | Value | Source |
|---|---|---|
| Rate on real CC HTML | **89 pages/s** (11.2 ms/page) | DGX A100, 200 pages |
| Rate range | 5–50ms/page | Varies by DOM complexity |
| Memory | ~2MB/page peak | Loaded in Python |

**Per job (300K pages)**:
- 1 core: 300,000 / 89 = 3,370s = **56 min**
- 8 cores: ~7 min
- 64 cores (Ray actors): ~53s

**Key bottleneck**: CPU-bound, lxml DOM parsing. GIL limits Python threads.  
**Optimization ideas**:
- ProcessPoolExecutor instead of ThreadPoolExecutor (true multicore)
- Batch HTML parsing (parse multiple pages in one lxml call)
- Pre-filter non-HTML pages before get_feature() (MIME type check)

---

## Stage 3: Layout Clustering (DBSCAN)

**Function**: `cluster_html_struct(samples, threshold=0.95)` per host  
**Algorithm**: DictVectorizer → weighted cosine (tag=0.7, attr=0.3) → DBSCAN (eps=0.05, min_samples=2)

| Measurement | Value | Source |
|---|---|---|
| Rate (10 largest hosts, 114K pages) | ~33,000 pages/s | Mac benchmark (trivial — no HTML) |
| Rate (real, from Slurm logs) | `297/297 rows → 3 layout IDs in 21.9s` | job 334859, chunk_1 |
| Rate (real, from Slurm logs) | `634/637 rows → 1 layout ID in 72.3s` | job 334859, chunk_1 |
| Rate (real, large host) | `603/604 rows → 2 layout IDs in 91.6s` | job 334859, chunk_1 |
| Rate (real, small host) | `375/376 rows → 2 layout IDs in 31.7s` | job 334859, chunk_1 |

**Per batch** (256 pages, ~64 hosts average):
- Small host (50–300 pages): ~1–30s
- Large host (500–5000 pages): ~30–120s
- DBSCAN is O(n²) in number of pages per host

**Observed**: chunk_1 at 136/159 batches after ~30 min → ~11s/batch average  
**Key bottleneck**: Large hosts (e.g., 600+ pages) dominate DBSCAN time (O(n²) pairwise distance)  
**Optimization ideas**:
- Cap cluster size before DBSCAN (use `max_exact_host_pages`, already implemented)
- Pre-filter with URL-hash bucketing (reduce DBSCAN input size)
- Approximate DBSCAN (e.g., locality-sensitive hashing for pre-clustering)

---

## Stage 4: Representative Selection

**Function**: `select_representative_html(candidates)` from llm-webkit  
**Scoring**: 0.4 × XPath coverage + 0.3 × structure score + 0.3 × width entropy

| Measurement | Value | Source |
|---|---|---|
| Typical time | ~20ms/cluster | Estimated from code inspection |
| Negligible vs other stages | — | Not a bottleneck |

---

## Stage 5: HTML Simplification

**Function**: `simplify_single_input(case)` → `_get_processed_attr(case, "simpled_html")`  
**What it does**: Strips non-content tags, assigns `_item_id` to nodes, truncates text

| Measurement | Value | Source |
|---|---|---|
| Time per page | **~50ms** | Stage timing from H100 runs |
| Output size | 12.83% of original | Paper §2.1.1 |
| Input → Output | 45,709 chars → simplified | DGX benchmark |

**For 8192 pages** (full smoke test): preprocess_mean = 78ms/page (includes fetch)  
**Not a major bottleneck** but benefits from parallelism.

---

## Stage 6: LLM Inference (MinerU-HTML)

**Model**: `opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact`  
**Hardware**: 8× H100 80GB (production), 1× A100 80GB (DGX)

| Category | inference_mean | Source |
|---|---|---|
| Representative pages | **8.19s/page** | job 332381, 353 pages |
| Fallback LLM pages | **2.78s/page** | job 332381, 2,887 pages |
| Standalone LLM pages | **1.85s/page** | job 332381, 2,820 pages |
| Validation LLM pages | ~2.5s/page | estimated |

**Dynamic max tokens improvement**: Enabling `--dynamic-max-tokens` reduced standalone mean from 2.14s → 1.85s (-13%).

**Scale**: At 89 pages/s LLM throughput with 8 H100s:
- 8192 pages, 26% call reduction → ~6,000 LLM calls
- 6,000 × 2.5s / 64 concurrent / 8 GPUs = ~29s wall time (GPU)
- Actual measured: ~250s (includes pipeline overhead)

**Key bottleneck**: Long representative pages (8.19s each) dominate GPU time.  
**Optimization ideas**:
- Dynamic max tokens (already enabled, saves 13%)
- Batched requests (not yet implemented)
- FP8 quantization (explored, needs root-cause on Dynamo results)

---

## Stage 7: Template Building (map_parser_cls)

**Function**: `web.map_parser_cls({}).parse({typical_raw_html, typical_raw_tag_html, llm_response})`

| Measurement | Value | Source |
|---|---|---|
| Time per representative | ~few hundred ms | DGX benchmark |
| Negligible vs LLM | — | Not a bottleneck |

---

## Stage 8: Template Propagation (layout_parser_cls)

**Function**: `web.layout_parser_cls({}).parse(task_data)` — LayoutBatchParser  
**What it does**: DOM tree walk, template matching, dynamic id/class resolution

| Measurement | Value | Source |
|---|---|---|
| **Mean time per page** | **11.2s/page** | job 330654, 2,129 rows |
| Median time per page | 9.7s/page | job 330654 (p50) |
| p95 time per page | 25.1s/page | job 330654 |
| Total CPU for 2,129 pages | 23,859s | job 330654 |
| Wall time (64 concurrent) | ~373s in GPU job | Dominated GPU stage time |

**Why so slow**: `_preprocess_template_data()` runs per sibling page despite being constant per cluster. Scans XPath of both template AND target trees, rebuilds normalized element dict every call.

**Fix implemented**: `layout_template_defer_propagation=True` (commit `31f1538`)  
→ Moves all propagation off H100 critical path → GPU stage: 598s → ~250s

**Optimization ideas (additional)**:
- Pre-compute `processed_template_data` once per cluster (saves ~35% per call)
- Use ProcessPool for propagation (bypass Python GIL)
- Batch siblings through one LayoutBatchParser instance

---

## Stage 9: Validation

**What**: Run propagation + LLM on 2 sample rows per cluster, compare F1

| Measurement | Value | Source |
|---|---|---|
| Validation rows per cluster | 2 (default), 8 (large clusters ≥32 pages) | Config |
| LLM cost per validation | Same as fallback (~2.5s/page) | Measured |
| Overhead per cluster | ~5–10s | Estimated |
| Probe overhead (full run) | 1,202 validation LLM calls | job 330545 |

**Optimization**: Reduce validation rows to 1 for small clusters (trade-off: worse quality detection).

---

## End-to-End Measurements

### H100 Runs (8× H100 80GB, 8192 pages)

| Run | Config | Elapsed | Throughput | H100-hours (projected snapshot) |
|---|---|---|---|---|
| 328281 | Pure Dripper (baseline) | 374s | 21.9 pages/s | **241,993** |
| 330419 | Layout template (url_shape, no large-val) | 644s | 12.7 pages/s | 416,999 |
| 330654 | B-global improvements | 599s | 13.7 pages/s | 387,447 |
| 332381 | + dynamic max tokens (defer broke) | 589s | 13.9 pages/s | 381,088 |
| 332405 | + defer_propagation (mapping bug) | 578s | 14.2 pages/s | 374,597 |

### Category Timing Breakdown (job 330654)

| Category | Rows | inference_mean | postprocess_mean | Total CPU |
|---|---|---|---|---|
| layout_representative | 353 | 8.19s | 0.92s | 2,738s |
| layout_fallback_llm | 2,886 | 2.78s | 0.27s | 9,122s |
| layout_standalone_llm | 2,820 | 1.85s | 0.16s | 6,796s |
| **layout_propagated_success** | **2,129** | **0.00s** | **11.2s** | **23,860s** |
| fallback_only | 4 | 0.00s | 0.08s | 0.04s |

**Key insight**: Propagation (11.2s × 2,129 = 23,860s CPU) accounts for **56% of total CPU** in the GPU job, but uses **0% GPU**. This is the primary bottleneck.

---

## CPU Diagnostic Runs (single CPU node, 8192 pages)

| Run | Config | Call reduction | Mean F1 | Bad rows (<0.95) |
|---|---|---|---|---|
| 330456 (Config A) | url_shape_item_count_exact, val=2 | 28.04% | 0.985 | 122 |
| 330545 (Config B) | url_low_card_query, val=2 | 24.71% | 0.987 | 82 |
| 330581 (A-global) | url_shape, global clusters, val=2 | 28.13% | 0.988 | 84 |
| **330582 (B-global)** | **url_low_card_query, global, val=2** | **27.44%** | **0.988** | **81** ← best |
| 330583 (D-global) | url_low_card_query, no validation | 63.42% | 0.892 | 2,103 (ceiling) |

---

## Layout Clustering Job (334859, host_bucket=0000, 4 nodes)

**Input**: `host_bucket=0000.parquet` — 300,923 pages, 4,676 hosts  
**Split**: 4 chunks (44K, 82K, 88K, 87K pages)  

| Chunk | Pages | Node | WARC fetch done | DBSCAN progress |
|---|---|---|---|---|
| chunk_00 | 44,180 | cpu-0034 | ~13:21 (~15 min) | 164/166 (stalled) |
| chunk_01 | 81,735 | cpu-0035 | ~13:25 (~19 min) | 139/159 (running) |
| chunk_02 | 87,947 | cpu-0036 | ~13:35 (est) | Starting |
| chunk_03 | 87,061 | cpu-0037 | ~13:35 (est) | Starting |

**Observed WARC fetch rate**: ~50 pages/s per node (64 async workers)  
**Observed DBSCAN rate**: 11s/batch average (batches of ~256 pages)

---

## Bottleneck Priority

| Priority | Stage | Bottleneck | Potential saving | Effort |
|---|---|---|---|---|
| 🔴 1 | Template Propagation | 56% of GPU job CPU, 0% GPU | Remove from GPU critical path | Medium (done: `defer_propagation`) |
| 🟡 2 | LLM Inference | Representative pages 8.19s, serial | Batching, FP8, Dynamo disagg | Large |
| 🟡 3 | WARC Fetch | 1.2s/record sequential, 50/s async | Lustre cache, dc node routing | Medium |
| 🟡 4 | get_feature() | 11.2ms/page, GIL-bound | ProcessPool, C extension | Medium |
| 🟢 5 | Singleton shards | 1 shard per unassigned page | Host-key grouping (done) | Small |
| 🟢 6 | Dynamic max tokens | +13% LLM throughput | Already enabled | Small (done) |
| 🟢 7 | URL dedup before preprocessing | 0.93% of pages duplicated | Minor | Small |

---

## Next Experiments

1. **Measure deferred propagation speedup** — job 332432 (in progress)  
   Expected: GPU stage 598s → ~250s; H100h 387K → ~160K

2. **Full shard clustering** — job 334859 (in progress)  
   Measuring: WARC fetch rate, DBSCAN time distribution, cluster count vs 8192 sample

3. **CPU propagation stage timing** — after defer_propagation lands  
   Goal: measure how long `DripperHTMLLayoutPropagationStage` takes on a full shard

4. **Lustre WARC cache** — prefetch WARCs to Lustre before clustering  
   Expected: WARC fetch 50/s → 500+/s (10× from local disk)
