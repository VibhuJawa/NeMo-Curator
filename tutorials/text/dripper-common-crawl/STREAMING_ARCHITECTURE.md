# Streaming Architecture for the CC-Scale MinerU-HTML Layout-Clustering Pipeline

**Target**: Redesign the 7-job Slurm parquet-handoff pipeline into a streaming,
NeMo Curator-native architecture that eliminates redundant I/O, reduces wall-clock
time, and lowers operational complexity.

All file paths are relative to the repo root
`nemo_curator_dc_v2/`.

---

## 1. Which stages can collapse into a single streaming pipeline — and which cannot

### Can collapse (no global barrier)

| Stages | Reason |
|--------|--------|
| JOB1a (feature extraction) + JOB1b ONLY IF running per-shard independently | Feature extraction is an embarrassingly parallel row map; DBSCAN is also per-row within a host-bucket. However, see the caveat in Section 4. |
| JOB1c (preprocess) + JOB2 (vLLM inference) + JOB2b (postprocess) | All three operate on the same ~9 % representative/singleton rows and are pure transforms with no cross-row dependency. The intermediate parquets (~260 MB 1c output, ~250 MB 2 output at tutorial scale, GBs at CC scale) exist only because these are separate Slurm jobs today. A single streaming pipeline can chain them with zero on-disk handoff. |
| JOB3 (propagation) streams behind JOB2b | Once a cluster's representative result is written by JOB2b, that cluster's siblings can start propagating immediately. Today JOB3 waits for ALL of JOB2b to finish. |

### Cannot collapse (require a global gather or broadcast join)

| Boundary | Reason |
|----------|--------|
| JOB1a → JOB1b | Stage 1b DBSCAN requires ALL pages for a given host-bucket to be present before clustering. This is a global reduce across the shard (and potentially across shards for large hosts). You cannot pipeline a DBSCAN that has only seen part of the input — the cluster labels would be wrong. This is a hard barrier. |
| JOB1b → JOB1c/JOB2 | Stage 1b produces the cluster manifest (which pages are representatives vs. siblings). JOB1c/JOB2 must know `cluster_role` before deciding which rows to send to GPU. Until the manifest is complete, neither filtering nor routing is possible. Another hard barrier. |
| JOB2b → JOB3 (broadcast join) | Stage 3 joins the cluster manifest (from JOB1b, columns: url, cluster_id, cluster_role, html for all 100 % of pages) with the GPU results (from JOB2b, columns: mapping_json, dripper_content, dripper_html, one row per representative/singleton). This is not a per-row map — it is a hash-join on `cluster_id`. The join can start as soon as a cluster's representative result lands, but it requires the manifest to be available in memory. |

**Summary**: The pipeline has exactly two hard barriers that require separate Slurm
jobs or Ray Data shuffles:

```
[JOB1a+1b: GLOBAL DBSCAN barrier]
         ↓  cluster manifest (parquet)
[JOB1c+2+2b: single streaming GPU job — the minimal refactor]
         ↓  mapping_json results (parquet)
[JOB3: streaming broadcast-join propagation — can start cluster-by-cluster]
         ↓
[JOB4: metrics]
```

---

## 2. How DripperHTMLExtractionPipelineStage solves "some rows skip inference"

`DripperHTMLExtractionPipelineStage` (
`nemo_curator/stages/text/experimental/dripper/stage.py`, line 3500)
is a `CompositeStage` that `decompose()`s into a sequence of `ProcessingStage`
instances. It does NOT use IS_FANOUT_STAGE or IS_ACTOR_STAGE flags (those are not
defined in this codebase's `ProcessingStage` base — the base only has
`is_source_stage` / `is_sink_stage`). Instead, it solves the "skip" problem through
three mechanisms:

**Mechanism 1 — Column-based routing flags.**
`DripperHTMLPreprocessStage` writes two internal columns into every row's DataFrame
that cross the stage boundary inside the batch:

```python
_DRIPPER_NEEDS_LLM_COL = "_dripper_needs_llm"   # bool: does this row need LLM?
_DRIPPER_EMPTY_INPUT_COL = "_dripper_empty_input" # bool: is input empty?
_DRIPPER_LAYOUT_FINALIZED_COL = "_dripper_layout_finalized"
```

`DripperHTMLInferenceStage` reads `_dripper_needs_llm` per row and skips inference
for rows where it is False, writing empty results. The DataFrame for the entire batch
passes through all three sub-stages; rows that do not need inference receive empty
`_dripper_prompt` and a `False` flag, and the inference stage fast-paths them.

**Mechanism 2 — Intra-batch async deduplication.**
Within a single `DocumentBatch`, the inference stage caches in-flight async tasks
keyed by `(prompt, max_tokens)`. If two rows have identical prompts (a common pattern
when multiple pages on the same host have the same template), only one LLM request is
made and both rows receive the same response.

**Mechanism 3 — `layout_template_defer_propagation` flag.**
When `layout_template_defer_propagation=True` is set on
`DripperHTMLLayoutTemplateStage`, the stage marks sibling rows with
`layout_pending_propagation=True` and `layout_finalized=False` instead of running
`LayoutBatchParser` inline. The expensive CPU propagation is then performed by a
separate downstream stage (`DripperHTMLLayoutPropagationStage`,
`nemo_curator/stages/text/experimental/dripper/propagation_stage.py`), which only
processes rows with `layout_pending_propagation=True`.

**Can we use the same pattern for the tutorial pipeline?**
Yes. The same column-flag pattern directly applies:

- `cluster_role` (already present in Stage 1b output) serves as the routing flag.
  Rows with `cluster_role == "representative"` or `"singleton"` have
  `_needs_llm = True`; rows with `cluster_role == "sibling"` have
  `_needs_llm = False`.
- A merged preprocessing+inference+postprocessing stage can filter on
  `_needs_llm` at the DataFrame level, process only the ~9 % of rows that need
  it, and write results back into the same DataFrame before passing to Stage 3.

---

## 3. Proposed new architecture: Curator primitive mapping

### Job topology

```
SLURM JOB A — "clustering" — CPU+GPU, array of shards
  [Stage1aFeatureStage]   ProcessingStage, CPU map (ProcessPoolExecutor inside process())
       ↓  in-memory DataFrame, no disk write
  [Stage1bDBSCANStage]    ProcessingStage with IS_ACTOR_STAGE semantics,
                           GPU node, cuML DBSCAN per host-bucket
       ↓  cluster manifest parquet (HARD BARRIER — global gather complete)

SLURM JOB B — "gpu-pipeline" — GPU node, 8 GPUs
  [Stage1cPreprocessStage] ProcessingStage, CPU map inside GPU job
       ↓  in-memory DataFrame
  [Stage2InferenceStage]   IS_ACTOR_STAGE, GPU, vLLM offline batched
       ↓  in-memory DataFrame
  [Stage2bPostprocessStage] ProcessingStage, CPU map
       ↓  mapping_json + dripper_content results parquet

SLURM JOB C — "propagation" — CPU-only, array of shards
  [Stage3PropagationStage] IS_ACTOR_STAGE (holds cluster manifest in memory),
                            broadcast-join + LayoutBatchParser per sibling
       ↓  dripper_content + propagation_method output parquet

SLURM JOB D — metrics aggregation (unchanged)
```

### Curator primitive for each original stage

| Original stage | New Curator primitive | Key notes |
|---------------|----------------------|-----------|
| JOB1a feature extraction | `ProcessingStage[DocumentBatch, DocumentBatch]` — standard CPU map; override `process_batch()` to call `get_feature()` via `ProcessPoolExecutor` | Merges into JOB A |
| JOB1b GPU DBSCAN | `ProcessingStage` with `resources = Resources(gpus=1)` and `setup()` loading cuML; `process_batch()` calls `cluster_html_struct_gpu()` per host-bucket group | HARD BARRIER: must see all pages for a host-bucket; stays as separate job or Ray Data groupby |
| JOB1c CPU preprocess | `ProcessingStage[DocumentBatch, DocumentBatch]` — CPU map; filters to reps/singletons; calls `simplify_single_input` + `build_prompt`; merges into JOB B |
| JOB2 vLLM inference | `ProcessingStage` with `resources = Resources(gpus=8)` and `setup()` spawning vLLM workers; this is the critical GPU stage | Stays on GPU node; merges into JOB B |
| JOB2b CPU postprocess | `ProcessingStage[DocumentBatch, DocumentBatch]` — CPU map; calls `parse_result`, `extract_main_html_single`, `MapItemToHtmlTagsParser`; merges into JOB B |
| JOB3 propagation | `ProcessingStage` with stateful `setup()` loading the cluster manifest into a dict; `process_batch()` does the hash-join + LayoutBatchParser per sibling | JOB C; see Section 7 for full sketch |
| JOB4 metrics | Thin Python script or Curator sink stage | Unchanged |

### Which stages collapse

**JOB A replaces JOB1a + JOB1b** — still separate from the GPU job because
the manifest must be complete before GPU inference can start.

**JOB B replaces JOB1c + JOB2 + JOB2b** — this is the **minimal refactor** and the
highest-value change (see Section 6).

**JOB C replaces JOB3** — now a single Curator `ProcessingStage` that holds the
cluster manifest in memory via `setup()`, enabling per-cluster streaming without
waiting for all of JOB B.

---

## 4. The clustering barrier: recommendation

Three options were considered:

### Option (a) — Keep Stage 1b as a separate Slurm job with a parquet barrier (RECOMMENDED)

**Reasoning**: The DBSCAN barrier is fundamental, not operational. Clustering requires
seeing ALL pages for every host-bucket simultaneously to compute the N×N cosine
similarity matrix (cuBLAS matmul). For a host with 3,000 pages this is a 3000×3000
float32 matrix = 36 MB on GPU — manageable. But the host-bucket boundaries are only
known after all input shards are read. A parquet handoff after JOB1a/1b is the only
correct solution that does not require a distributed shuffle.

At CC scale (2.4B pages), the feature extraction + DBSCAN job runs as a Slurm array.
Each array task owns a shard; hosts that span multiple shards are handled by the
manifest-building scripts (`build_host_clustered_manifest_from_shards.py` already
exists in the tutorial directory). The parquet handoff is ~GB per shard — modest
compared to the HTML itself.

### Option (b) — Ray Data groupby/repartition in one job

Ray Data can do a shuffle-groupby on `url_host_name`, which would let Stage 1a and
Stage 1b run in one job. However:

- A full shuffle of all pages by host name at CC scale is a very large distributed
  sort. Ray Data's shuffle is bounded by object store memory and generates significant
  network I/O between nodes.
- The existing tutorial pipeline already shards the input by host before Stage 1a
  (see `build_host_clustered_manifest.py`). If sharding is done correctly, each shard
  owns complete host-buckets and no cross-shard shuffle is needed.
- The added operational complexity of a Ray cluster for Stage 1 is not justified when
  the existing Slurm array approach already handles the sharding correctly.

**Do not use Ray Data groupby for Stage 1b.**

### Option (c) — Use existing DripperHTMLLayoutClusteringStage

`DripperHTMLLayoutClusteringStage` (in `stage.py`) is a CPU-only Curator stage that
runs GPU DBSCAN or sklearn fallback and produces `layout_id` column assignments. It
is designed for in-process use (all pages for a host-bucket passed as a single
`DocumentBatch`). It does NOT address the cross-shard gather problem — it assumes the
batch already contains all pages for each host being clustered.

**Use `DripperHTMLLayoutClusteringStage` inside JOB A**, but keep the parquet
barrier between JOB A and JOB B. The stage solves the GPU/CPU dispatch and
representative-selection logic; the Slurm manifest-building step handles cross-shard
host merging.

---

## 5. Streaming throughput gains: Stage 3 is the bottleneck

### Current wall-clock breakdown (tutorial: 3,869 input pages, 9 GPU pages ~350 reps/singletons)

At CC scale the proportions hold but numbers scale up by ~620,000x:

| Stage | Throughput | Notes |
|-------|-----------|-------|
| Stage 1a feature | ~300 pages/s/core × 64 cores | Fast |
| Stage 1b DBSCAN | ~2,000 pages/s per GPU | Fast |
| Stage 1c preprocess | ~350 pages/s/core × 64 cores | Fast |
| Stage 2 inference | ~163 pages/s/node (tutorial claim) | 9 % of pages |
| Stage 2b postprocess | ~500 pages/s/core × 64 cores | Fast |
| Stage 3 propagation | ~77 pages/s/node | 91 % of pages — BOTTLENECK |

Stage 3 is ~2.1× slower than Stage 2 at the page level, but processes 10.1× more
pages (91 % vs. 9 %). The effective wall-clock ratio is:

```
Stage 2 effective wall-clock weight:  0.09 pages × (1/163 s/page) = 0.00055 nodes·s/page
Stage 3 effective wall-clock weight:  0.91 pages × (1/77  s/page)  = 0.0118  nodes·s/page
Ratio: Stage 3 is 21× more expensive in node·seconds than Stage 2.
```

### How streaming helps

Today Stage 3 does not start until Stage 2 (and 2b) are 100 % complete. The last
cluster's representative is processed at time T_end_2b. Stage 3 then starts from
scratch.

With streaming, Stage 3 can begin processing a cluster's siblings as soon as that
cluster's representative `mapping_json` is written by Stage 2b, which happens while
Stage 2 is still running for other clusters.

**Estimated wall-clock improvement** (back-of-envelope, CC scale):

Let N = total clusters, throughput_2b = fast (CPU, negligible), throughput_3 = 77
pages/s/node per sibling, cluster_size = 11.1 (91/9 ratio).

- **Without streaming**: Wall clock = T(Stage 2) + T(Stage 3 full).
  For 2.4B pages: T(Stage 3) = (2.4B × 0.91) / (77 × 80 nodes) ≈ 3.55 hours.
  T(Stage 2) = (2.4B × 0.09) / (163 × 8 GPU nodes) ≈ 0.17 hours.
  Sequential total ≈ **3.72 hours** (Stage 3 dominates).

- **With streaming**: Stage 3 starts processing cluster C's siblings as soon as
  cluster C's representative completes Stage 2b. Because Stage 3 is the bottleneck,
  Stage 2 completes (for the last cluster) at time 0.17h, while Stage 3 has already
  been running for 0.17h worth of clusters. The remaining Stage 3 work is:
  (3.55h - 0.17h) = 3.38h. Total ≈ 0.17h + 3.38h = **3.55 hours**.

  **Wall-clock savings ≈ 0.17 hours (about 10 minutes at CC scale on 8 GPU + 80 CPU
  nodes running in parallel)**. The gain is bounded by T(Stage 2) because Stage 3 is
  the bottleneck and cannot start until Stage 2 starts producing results.

The more meaningful gain from streaming is **eliminating Stage 2b's parquet write and
Stage 3's parquet read** at CC scale. At 2.4B × 9 % = 216M rows of representative
results, the Stage 2b parquet is ~10–15 GB (snappy). Reading that in Stage 3 takes
~60–90 s at NVMe speeds across 80 nodes. Eliminating this read saves one full I/O
pass per node.

**Conclusion**: The bigger win from streaming JOB1c+JOB2+JOB2b is not primarily
overlap — it is eliminating two parquet round-trips (~520 MB at tutorial scale, ~15
GB at CC scale) and the associated queueing delays between Slurm jobs.

---

## 6. Minimal refactor path: Combine JOB1c + JOB2 + JOB2b into one GPU Slurm job

This is the highest-value, lowest-risk change. It requires zero changes to Stage 1b
or Stage 3. It eliminates two parquet handoffs and three Slurm job submissions.

### What to build

Create a new script `stage_gpu_pipeline.py` that runs as a single Slurm GPU job:

```
INPUT:   stage1b cluster manifest parquet (all rows: reps, singletons, siblings)
DOES:
  1. Filter to reps + singletons in memory (~9 % of rows)
  2. Run simplify_single_input + build_prompt (CPU, ProcessPoolExecutor, 64 workers)
  3. Load vLLM engine (once, stays resident)
  4. Run LLM.generate() over all prompts (GPU, offline batched)
  5. Run parse_result + MapItemToHtmlTagsParser + convert2content (CPU, ProcessPoolExecutor)
OUTPUT:  mapping_json + dripper_content parquet (one per shard)
         (same schema as current Stage 2b output — Stage 3 unchanged)
```

This is architecturally equivalent to
`DripperHTMLExtractionPipelineStage.decompose()` with
`layout_template_mode=True` and `layout_template_defer_propagation=True`, minus the
clustering step (which stays in JOB A).

### I/O savings

At tutorial scale (3,869 pages):
- Stage 1c output parquet: ~260 MB (eliminated)
- Stage 2 output parquet: ~250 MB (eliminated)
- Total: **~510 MB per shard avoided at tutorial scale**

At CC scale (2.4B pages, 80 shards, 9 % reps/singletons = 216M rows):
- Stage 1c output: ~12 GB total (eliminated)
- Stage 2 output: ~11 GB total (eliminated)
- Total: **~23 GB of intermediate I/O eliminated**

### Slurm job impact

Before: 3 Slurm jobs (JOB1c → JOB2 → JOB2b) + queue delays between each.
After: 1 Slurm GPU job. Queue delay between JOB1c and JOB2 was the largest
wall-clock tax at CC scale (GPU queues are often 10–60 minutes).

### Implementation sketch

```python
# stage_gpu_pipeline.py — replaces JOB1c + JOB2 + JOB2b
# Slurm: --partition=gpu_batch --gres=gpu:8 --cpus-per-task=64 --mem=235G

def run(args):
    # 1. Load Stage 1b manifest, filter to reps + singletons
    df = pq.read_table(args.manifest).to_pandas()
    llm_rows = df[df["cluster_role"].isin(["representative", "singleton"])].copy()

    # 2. CPU preprocess (Stage 1c logic)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_mineru) as pool:
        llm_rows = _preprocess_parallel(pool, llm_rows)

    # 3. GPU inference (Stage 2 logic — vLLM offline batched, already works)
    llm_rows = _run_vllm_inference(llm_rows, args)

    # 4. CPU postprocess (Stage 2b logic — map_parser + convert2content)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_bindings) as pool:
        llm_rows = _postprocess_parallel(pool, llm_rows)

    # 5. Write output (Stage 3 reads this — schema unchanged)
    llm_rows.to_parquet(args.output, index=False, compression="snappy")
```

The inner functions `_preprocess_parallel`, `_run_vllm_inference`, and
`_postprocess_parallel` are direct copies of the per-stage logic from the existing
scripts. No algorithmic changes are required.

---

## 7. Stage3PropagationStage: concrete ProcessingStage sketch

This sketch illustrates how to implement Stage 3 as a Curator `ProcessingStage` with
proper `setup()`, `process_batch()`, the actor pattern for holding state, and the
broadcast-join from the cluster manifest.

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


@dataclass(kw_only=True)
class Stage3PropagationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """CPU propagation stage: broadcast-join cluster manifest + LBP propagation.

    This stage is STATEFUL — it loads two large tables into memory during setup():
      1. The cluster manifest (url -> cluster_id, cluster_role, html for ALL pages)
      2. The GPU results (cluster_id -> mapping_json, dripper_content for reps only)

    Both tables are held in memory for the lifetime of the actor. Each call to
    process_batch() receives a DocumentBatch of sibling rows and performs
    the LayoutBatchParser propagation JOIN without any disk reads.

    The stage must NOT be used with stateless per-row executors. It requires
    the actor pool pattern (RayActorPoolStageAdapter) so that setup() is called
    once per actor and the in-memory state persists across batches.

    resources: CPU-only (no GPU). Set cpus to match the ProcessPoolExecutor
    worker count you want for the inner parallelism (64 per node typical).
    """

    name: str = "Stage3PropagationStage"
    resources: Resources = field(
        default_factory=lambda: Resources(cpus=64.0)  # 64 CPU workers per actor
    )
    batch_size: int = 10_000  # rows per DocumentBatch call

    # Config — must be set before setup() is called
    manifest_path: str = ""           # path to Stage 1b cluster manifest parquet
    gpu_results_path: str = ""        # path to Stage 2b mapping_json results parquet
    dynamic_classid_similarity_threshold: float = 0.85
    more_noise_enable: bool = True
    min_content_length_ratio: float = 0.25
    max_content_length_ratio: float = 4.0

    # Internal state — populated by setup(), NOT part of __init__
    # These are per-actor state (held in the Ray actor's heap):
    _manifest_by_url: dict[str, dict[str, Any]] = field(
        init=False, repr=False, default_factory=dict
    )
    _mapping_by_cluster: dict[str, dict[str, Any]] = field(
        init=False, repr=False, default_factory=dict
    )
    _web_bindings: Any = field(init=False, repr=False, default=None)
    _mineru_bindings: Any = field(init=False, repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["url", "cluster_id", "cluster_role", "html"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [
            "dripper_content",
            "dripper_html",
            "dripper_error",
            "propagation_method",
            "propagation_success",
        ]

    def setup(self, worker_metadata=None) -> None:
        """Called once per actor. Loads the cluster manifest and GPU results
        into memory. This is the broadcast-join setup step.

        At CC scale: manifest ~ a few GB per shard (url, cluster_id, cluster_role
        only — HTML is dropped after Stage 1b for siblings). GPU results are
        ~hundreds of MB per shard (mapping_json is the large column).
        """
        if self._initialized:
            return

        # Load llm_web_kit / mineru bindings once per worker process
        from nemo_curator.stages.text.experimental.dripper.stage import (
            _load_llm_web_kit_bindings,
            _load_mineru_html_bindings,
        )
        self._web_bindings = _load_llm_web_kit_bindings()
        self._mineru_bindings = _load_mineru_html_bindings()

        # --- Broadcast join table 1: cluster manifest ---
        # Loaded into a dict keyed by url for O(1) lookup per sibling row.
        # Columns needed: cluster_id, cluster_role, html (for siblings only).
        # At CC scale: filter to sibling rows before loading to save memory.
        manifest = pq.read_table(
            self.manifest_path,
            columns=["url", "cluster_id", "cluster_role", "html"],
        ).to_pandas()
        self._manifest_by_url = {
            row["url"]: {
                "cluster_id": row["cluster_id"],
                "cluster_role": row["cluster_role"],
                "html": row.get("html", ""),
            }
            for _, row in manifest.iterrows()
        }

        # --- Broadcast join table 2: GPU results (mapping_json per cluster) ---
        # One row per representative (cluster_role == "representative").
        # cluster_id -> mapping_json (deserialized dict).
        gpu_results = pq.read_table(
            self.gpu_results_path,
            columns=["cluster_id", "mapping_json", "dripper_content", "dripper_html"],
        ).to_pandas()
        gpu_results = gpu_results[gpu_results["cluster_id"].notna()]
        for _, row in gpu_results.iterrows():
            cid = str(row["cluster_id"])
            mapping_json = row.get("mapping_json", "")
            if mapping_json:
                try:
                    self._mapping_by_cluster[cid] = json.loads(mapping_json)
                except Exception:
                    pass

        self._initialized = True

    def process_batch(self, tasks: list[DocumentBatch]) -> list[DocumentBatch]:
        """Process a batch of DocumentBatch objects.

        Each DocumentBatch contains rows for one shard partition. The stage
        does the hash-join (lookup in _mapping_by_cluster) and runs
        LayoutBatchParser propagation for sibling rows.

        Returns one output DocumentBatch per input batch (1-to-1 transform).
        """
        results = []
        for batch in tasks:
            df = batch.to_pandas().copy()
            df = self._propagate_dataframe(df)
            results.append(
                DocumentBatch(
                    task_id=batch.task_id,
                    dataset_name=batch.dataset_name,
                    data=df,
                    _metadata=batch._metadata,
                    _stage_perf=batch._stage_perf,
                )
            )
        return results

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Single-task fallback (used if process_batch is not called by executor)."""
        return self.process_batch([task])[0]

    def _propagate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core logic: join and propagate one DataFrame partition.

        Per-row routing:
          - cluster_role == "representative": copy GPU result directly
          - cluster_role == "singleton": copy GPU result directly
          - cluster_role == "sibling": run LayoutBatchParser against
            the representative's mapping_json from _mapping_by_cluster

        This method runs in the actor's main thread. For large batches,
        delegate to a ProcessPoolExecutor for parallelism across sibling rows.
        """
        # Initialize output columns
        for col in ["dripper_content", "dripper_html", "dripper_error",
                    "propagation_method", "propagation_success"]:
            if col not in df.columns:
                df[col] = ""
        df["propagation_success"] = False

        for idx, row in df.iterrows():
            role = str(row.get("cluster_role", ""))
            if role in ("representative", "singleton"):
                # GPU result already in the row — just label the method
                df.at[idx, "propagation_method"] = role
                df.at[idx, "propagation_success"] = not bool(row.get("dripper_error", ""))
            elif role == "sibling":
                cluster_id = str(row.get("cluster_id") or "")
                mapping_data = self._mapping_by_cluster.get(cluster_id)
                html = str(row.get("html") or "")

                if not mapping_data or not html.strip():
                    df.at[idx, "dripper_error"] = (
                        "no_mapping_data" if not mapping_data else "empty_html"
                    )
                    df.at[idx, "propagation_method"] = "fallback"
                    continue

                # Run LayoutBatchParser — the expensive CPU step
                main_html, error = self._run_lbp(html, mapping_data)
                if not error and main_html:
                    content, conv_err = self._convert_content(main_html, row.get("url", ""))
                    df.at[idx, "dripper_html"] = main_html
                    df.at[idx, "dripper_content"] = content
                    df.at[idx, "dripper_error"] = conv_err
                    df.at[idx, "propagation_method"] = "layout_batch_parser"
                    df.at[idx, "propagation_success"] = not bool(error or conv_err)
                else:
                    df.at[idx, "dripper_error"] = error
                    df.at[idx, "propagation_method"] = "fallback"

        return df

    def _run_lbp(
        self,
        html: str,
        mapping_data: dict[str, Any],
        dynamic: bool = True,
    ) -> tuple[str, str]:
        """Run LayoutBatchParser. Returns (main_html, error)."""
        if self._web_bindings is None:
            return "", "llm_web_kit_not_available"
        try:
            task_data = dict(mapping_data)
            task_data.update({
                "html_source": html,
                "dynamic_id_enable": dynamic,
                "dynamic_classid_enable": dynamic,
                "more_noise_enable": self.more_noise_enable,
                "dynamic_classid_similarity_threshold": (
                    self.dynamic_classid_similarity_threshold
                ),
            })
            parts = self._web_bindings.layout_parser_cls({}).parse(task_data)
            if parts.get("main_html_success") is False:
                return "", "main_html_success_false"
            return str(parts.get("main_html_body") or ""), ""
        except Exception as exc:
            return "", f"lbp_error={exc!s:.200}"

    def _convert_content(self, main_html: str, url: str) -> tuple[str, str]:
        """Convert main_html -> text content. Returns (content, error)."""
        if self._mineru_bindings is None:
            return "", "mineru_not_available"
        try:
            M = self._mineru_bindings
            case = M.case_cls(M.input_cls(raw_html="", url=url))
            case.output_data = M.output_cls(main_html=main_html)
            result = M.convert2content(case, output_format="mm_md")
            od = getattr(result, "output_data", None)
            return str(getattr(od, "main_content", "") or ""), ""
        except Exception as exc:
            return "", f"content_error={exc!s:.150}"

    def teardown(self) -> None:
        """Release in-memory broadcast tables when the actor is destroyed."""
        self._manifest_by_url.clear()
        self._mapping_by_cluster.clear()
        self._web_bindings = None
        self._mineru_bindings = None
        self._initialized = False
```

### How the actor pattern applies

The `RayActorPoolStageAdapter`
(`nemo_curator/backends/ray_actor_pool/adapter.py`) wraps
`Stage3PropagationStage` as a Ray actor. When the actor is created,
`RayActorPoolStageAdapter.__init__()` calls `stage.setup(worker_metadata)` once.
The `_manifest_by_url` and `_mapping_by_cluster` dicts are then resident in the
actor's heap for the lifetime of the Ray actor — no per-batch disk reads.

The Pipeline executor routes `DocumentBatch` objects to available actors. Because the
cluster manifest and GPU results are loaded once in `setup()`, each `process_batch()`
call does only:

1. A dict lookup on `cluster_id` — O(1) per row.
2. `LayoutBatchParser.parse()` — the expensive CPU work, same as today.

This is functionally equivalent to the current Stage 3, but expressed as a Curator
primitive that can be composed into a `Pipeline` with other stages and run under any
executor.

### Handling the broadcast join correctly

The `mapping_data` dict (the propagation template) is read from
`_mapping_by_cluster[cluster_id]`. This dict is populated in `setup()` by reading the
Stage 2b output parquet that was written by the GPU pipeline job (JOB B). At the
point Stage 3 starts, JOB B is complete — this is still a hard sequencing constraint.

If you want Stage 3 to start before JOB B completes (true streaming), you need a
shared key-value store (Redis, Ray object store with a RefManager actor, or a
distributed dict) that JOB B writes to as each cluster's representative finishes.
Stage 3 workers poll for the key. This is technically feasible but operationally
complex. The parquet barrier is simpler and the gain is small (Section 5 quantifies
it as ~10 minutes at CC scale).

---

## Summary table: upstream Curator components that already solve each subproblem

| Subproblem | Upstream component that solves it |
|-----------|----------------------------------|
| CPU map per row with ProcessPoolExecutor | `ProcessingStage.process_batch()` override |
| GPU stage with cuML DBSCAN | `DripperHTMLLayoutClusteringStage` (stage.py) — directly reusable for JOB A |
| Routing some rows to LLM, others skip | Column-flag pattern in `DripperHTMLPreprocessStage` (`_dripper_needs_llm`) |
| Deferred CPU propagation after GPU inference | `DripperHTMLLayoutPropagationStage` (propagation_stage.py) — directly reusable for JOB C |
| Composing preprocess + inference + postprocess into one streaming job | `DripperHTMLExtractionPipelineStage.decompose()` — the exact pattern for JOB B |
| Actor lifecycle management (setup once, process many batches) | `RayActorPoolStageAdapter` (adapter.py) |
| LLM inference with deduplication within batch | `DripperHTMLInferenceStage` with `_infer_row_cached()` |
| CompositeStage decomposition | `CompositeStage.decompose()` + `Pipeline._decompose_stages()` (pipeline.py) |

---

## Appendix: Slurm job count reduction

| Phase | Before | After |
|-------|--------|-------|
| Feature + clustering | 2 jobs (1a, 1b) | 1 job (A) |
| Preprocess + inference + postprocess | 3 jobs (1c, 2, 2b) | 1 job (B) — **highest-value change** |
| Propagation | 1 job (3) | 1 job (C) |
| Fallback LLM | 2 jobs (3b build + 3b merge) | Optional — kept separate |
| Metrics | 1 job (4) | 1 job (D) |
| **Total** | **7–9 jobs** | **3–4 jobs** |

Eliminating 3–4 Slurm job submissions at CC scale also eliminates 3–4 × average
queue wait times. On a shared cluster with 10–60 minute GPU queue waits, this alone
can save 30–120 minutes of wall-clock time per pipeline run.
