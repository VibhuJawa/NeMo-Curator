# Dripper Common Crawl Smoke

This tutorial runs Dripper/MinerU-HTML through NeMo Curator's inference server
path on a bounded Common Crawl sample. It is intended for single-node H100
smoke runs. For the multi-node, snapshot-scale pipeline -- where the per-page
MinerU postprocess moves *off* the GPU and DOM-layout propagation lets most
pages skip the LLM -- see
[Scaling: the broadcast-template pipeline](#scaling-the-broadcast-template-pipeline)
below.

The Python runner:

1. Streams WARC records from `CC-MAIN-2025-26`.
2. Starts Ray through Curator's `SlurmRayClient` on SLURM, or `RayClient`
   outside SLURM.
3. Starts a Curator `InferenceServer` with the Dripper model.
4. Points `AsyncOpenAIClient` at the server endpoint.
5. Optionally runs warmup pages, then runs `DripperHTMLExtractionStage`.
6. Writes extracted rows plus steady-state and end-to-end H100-hour metrics.

On Nebius, submit:

```bash
sbatch tutorials/text/dripper-common-crawl/submit_nebius_single_node.sh
```

Useful overrides:

```bash
MAX_PAGES=1024 REPLICAS=8 MAX_CONCURRENT_REQUESTS=64 WARMUP_PAGES=8 \
  sbatch tutorials/text/dripper-common-crawl/submit_nebius_single_node.sh
```

Throughput knobs that should not change Dripper extraction semantics:

- `ENABLE_PREFIX_CACHING=1` is the default and reuses identical prompt prefixes
  in vLLM.
- `DISABLE_THINKING=1` is the default and passes
  `chat_template_kwargs={"enable_thinking": false, "thinking": false}` through
  the OpenAI-compatible vLLM request. Dripper expects JSON/compact labels, so
  disabling thinking avoids `<think>...` text that MinerU-HTML cannot parse.
- `MAX_CONCURRENT_REQUESTS`, `MAX_NUM_SEQS`, and `MAX_NUM_BATCHED_TOKENS` tune
  request batching.
- `GPU_MEMORY_UTILIZATION` defaults to `0.9` in the Nebius wrapper to increase
  KV-cache capacity.
- `WARMUP_PAGES` excludes cold first-request overhead from the steady-state
  `h100_hours_per_page` metric while still reporting end-to-end timing.

Use `ENFORCE_EAGER=1` for short debug runs where startup time matters more than
steady-state throughput. Leave it unset for cost estimation runs.

The submit script expects PBSS/Common Crawl credentials to be available from
the environment or from the user's remote cache environment file. It does not
print secret values.

## Scaling: the broadcast-template pipeline

The smoke path above runs everything -- vLLM inference *and* per-page MinerU
extraction -- on one GPU node. At snapshot scale that is the wrong shape:
profiling shows roughly **78% of the wall is CPU per-page MinerU postprocess**
(`convert2content`, the recognizer chain) running *on the GPU node while the GPU
sits idle*. The scaling pipeline splits the work into five phases (**A -> E**) so
the GPU does only inference and the heavy CPU postprocess fans out across CPU
nodes.

### Layout-template propagation (why most pages skip the LLM)

Within a host, most pages share a DOM **layout** (e.g. every article page of a
news site). Dripper clusters pages by layout, sends one **representative** per
cluster to the LLM, derives a reusable `mapping_data` **template**, and
**propagates** that template to the cluster's members instead of calling the LLM
again. Only representatives, validation rows, and unclustered/standalone pages
hit vLLM; propagated members are cheap.

### The five phases

| Phase | Stage | Runs on | What it does |
|-------|-------|---------|--------------|
| **A** | preprocess | CPU | WARC -> parse -> preprocess; precompute the llm-webkit DOM feature |
| **B** | cluster | GPU | cuML DOM-layout clustering from Phase A's features |
| **C** | plan | CPU | pick one representative + the pending-propagation members per cluster; repartition into balanced shards |
| **D** | infer + emit templates | GPU | vLLM on representatives; finalize validates each cluster's template and, on passing the content-F1 gate, writes it to the side-table column `_dripper_layout_template_json` |
| **E** | broadcast-propagate + postprocess | CPU | `DripperHTMLBroadcastPropagateStage` loads the side-table into a global `{cluster_id -> mapping_data}` map, replays each cluster's template to its members **off the GPU**, then postprocesses |

Phase **E** is the one that fans out across the CPU fleet. Because the template
travels through the Phase **D** side-table (not through block-local co-location),
Phase **E** shards can be **row-balanced** rather than cluster-intact -- so a
dominant layout (one host with tens of thousands of pages) no longer lands on a
single shard/actor as a straggler. The split is **F1-neutral**: Phase **E**
replays the identical `_propagate_layout_template` from the identical validated
`mapping_data`, so propagated content is byte-for-byte the same as the
single-node path (verified: 36,917 propagated pages, exact match).

### Running the phases

Phases **A**, **C**, **E** run on CPU; **B**, **D** run on GPU. The CPU phases
use `submit_nebius_dripper_cpu.sh` with phase flags (they default to the
`cpu`/`cpu_dataprocessing` partition, away from the batch-partition GPU-idle
watchdog):

```bash
# Phase A -- preprocess (CPU)
PREPROCESS_ONLY=1 \
  bash submit_nebius_dripper_cpu.sh <user@host> <warc_manifest> <out_A>

# Phase B -- cluster (GPU)
CLUSTER_FROM_PREPROCESSED=1 GPUS_PER_NODE=1 PARTITION=batch CLUSTER_GPUS=1.0 \
  bash submit_nebius_dripper_cpu.sh <user@host> <out_A> <out_B>

# Phase C -- plan (CPU)
PLAN_ONLY=1 \
  bash submit_nebius_dripper_cpu.sh <user@host> <out_B> <out_C>

# Phase D -- vLLM infer + emit the template side-table (GPU)
#   pipeline.py --input-parquet <out_C> --output-dir <out_D> ...

# Phase E -- broadcast-propagate + postprocess (CPU); reads Phase C (plan)
# AND the Phase D side-table
BROADCAST_PROPAGATE_ONLY=1 TEMPLATE_TABLE_PATH=<out_D> \
  bash submit_nebius_dripper_cpu.sh <user@host> <out_C> <out_E>
```

For the CPU fan-out at scale, submit Phase **E** as a **1-node-per-shard array**
on `cpu_dataprocessing` (one self-contained Ray cluster per job). In-job
multi-node Ray is fragile here (worker-connect timeouts); the array pattern is
robust and the partition's `normal` QOS allows many concurrent 1-node jobs.

### Tuning knobs

- **`layout_template_validation_aggregation`** (`min` default | `mean` |
  `median`) -- how a cluster's template is accepted across its validation rows in
  Phase **D**. `min` is strict all-must-pass; `mean`/`median` relax it and raise
  the propagation rate (measured 44% -> 53% under `mean` at median content-F1
  ~0.96) at a small quality cost. `min` reproduces the original behavior exactly.
- **`propagation_content_source`** (`converted` default | `layout_text`) --
  `layout_text` returns the layout parser's text directly and **skips the
  `convert2content` recognizer chain** for propagated pages (faster Phase **E**),
  at the cost of some table/code structure; prefer gating it by content type.
- **`layout_template_validation_min_content_f1`** -- the template-accept
  threshold (lower => more propagation, lower fidelity).

### Where the cost goes

The binding constraint is the **per-page `convert2content` cost** (the MinerU
recognizer chain), and it runs on **~every page** -- not just the non-propagated
ones. Under the default `propagation_content_source="converted"`, a propagated
page's template output is still pushed through `convert2content`
(`layout_template.py::_convert_main_html`), and the non-propagated pages run it in
the postprocess (`postprocess.py`, after the `_dripper_layout_finalized` early-return
for already-propagated rows). So **propagation removes the vLLM call but NOT
`convert2content`** -- which is why moving postprocess to CPU and raising propagation
help but don't break the wall.

The biggest remaining lever is **`propagation_content_source="layout_text"`**: the
~49% propagated pages then take the layout parser's text directly and skip
`convert2content` entirely, roughly halving Phase E (~80 -> ~160 pg/s/node; snapshot
~12.5 -> ~10 days). The tradeoff is some table/code structure loss (measured median
content-F1 ~0.96 but ~21% of pages <0.9 vs `converted`), so it is a gated flag, not the
default. Pruning recognizers that do not fire on a text-dominant corpus is a further
lever.

### Measured per-node throughput

Benchmarked on a real Common Crawl mega-host slice (100k pages from host buckets 0-1 --
`tengrinews.kz` 31.6k, `tgcom24` 20.8k, ...), one node per phase, with validated
propagation **49.4%** (the `mean`@0.80 gate; mega-hosts form big clusters but
intra-cluster layout diversity still fails ~half at the gate):

| Phase | Device | pages/s/node | Notes |
|---|---|---|---|
| A preprocess | CPU (64-core) | ~116 | streaming; `_consolidate_by_host` adds a serial tail |
| B cluster | GPU | ~318 / GPU | cuML; ~8x per 8-GPU node |
| C plan | CPU | ~280 | gated by input block count: needs B `OUTPUT_SHARDS` >= #cores (8 shards starved it to 78) |
| D infer + emit | 8-GPU node | vLLM-fast (not binding) | postprocess must move to E, not run here |
| E broadcast + postprocess | CPU (64-core) | ~80 | `convert2content` runs on ~all pages (propagated too, under default `converted`) -- the per-page wall |

Extrapolated to **80 CPU + 40 GPU nodes**: 1% of a snapshot (~27M pages) finishes in
~4 h, but a **full snapshot (~2.7B pages) is ~12.5 days** -- ~12x over a 24h budget, bound
by the CPU phases **E (postprocess) and A (preprocess)**, not the GPU (C dropped from the
top spot once its input was parallelized). Closing the gap is a per-page-cost problem
(`layout_text` for E, recognizer pruning, a faster preprocess), not a node-count one.

Two parallelism fixes alone -- the Phase E zero-copy side-table and the plan-stage input
sharding -- cut the modeled snapshot wall **27.5 -> 16 -> 12.5 days** with no algorithm
change, just by removing per-stage parallelism starvation.

**Phase E fan-out (the scaling lever) needs two things, both handled in the code now:**
(1) the per-cluster template side-table is loaded once on the driver and shared *zero-copy*
across actors (a `ray.put` Arrow table), not loaded per-actor -- a per-actor load of the
~GB table caps the actor pool at ~16 and stalls it; (2) do **not** set `WORKER_COUNT` for
Phase E -- it runs two concurrent CPU actor stages (broadcast + postprocess) and the
pipeline default splits the cores between them, whereas a manual override forces both to the
full pool and the downstream postprocess starves the upstream broadcast.
