# Stage 3 Deeper Speedup Plan (Track H4)

Goal: push Stage 3 CPU propagation past the current ~77 pages/s/node, F1-safe
(no approximation that changes extracted content vs the dynamic-LBP baseline).

This plan **revises the earlier `STAGE3_PERF_AUDIT.md` cost estimates with
direct microbenchmarks** taken on the cluster (login node, CPU venv) against the
real `LayoutBatchParser` vendor code. The headline correction: the audit's #2
(per-cluster template reuse, "1.3-2x") is **not supported by measurement** — it
is ~1.06x. The genuine remaining levers are (c) convert2content reuse and (b)
load balancing; (a) reuse is worth doing only as cheap insurance on the
fallback path; (d) the L1 HTML load is a memory/startup fix, not throughput.

---

## 0. Current state and where the time goes

Stage 3 today (per the project memory) runs at ~77 pages/s/node via a two-tier
LBP: ~79% of siblings take a **static-only** LBP path (dynamic id/classid
matching disabled), ~21% fall back to **dynamic** LBP. F1 is held at the
dynamic-LBP baseline by per-cluster validation (`_cluster_static_trustworthy`).

The remaining cost is dominated by the static-LBP path (it runs on ~79% of
siblings) plus the convert2content call that runs on **every** sibling.

### Measured per-page costs (cluster microbench)

Synthetic but realistic: 800-node sibling page, 60-entry × 8-layer template,
`dripper_cached_venv` CPU venv, single process:

| Operation | Measured |
|---|---|
| `LayoutBatchParser.parse()` **static** (dynamic disabled) | **~12.7 ms/page** |
| `_preprocess_template_data` (inside that parse) | **~1.23 ms (9.7% of parse)** |
| ↳ page-side `tree.xpath('//*[@id]')` (NOT reusable) | ~0.21 ms |
| ↳ template-side + `processed_template_data` build (reusable) | ~0.6–0.8 ms |
| `parse_tuple_key` over 480 keys (only if template is a *string*) | ~0.1 ms — **already avoided** (Stage 2b pickles the dict, so the `isinstance(...,dict)` branch is taken; no per-page json work) |
| `convert2content(mm_md)` | ~20–80 ms (audit; could not re-time — login node hit `std::bad_alloc` under contention) |

Two facts dominate the plan:

1. **`_preprocess_template_data` is only ~9.7% of a static parse, and only
   ~60–70% of that is reusable per cluster.** So eliminating the redundant
   per-sibling template setup (audit #2 / W2) saves **~0.7 ms of ~12.7 ms ≈
   5–6% → ~1.06x on the static-LBP path.** The audit's 1.3–2x was an
   over-estimate (it assumed the *whole* preprocess was reusable and a larger
   share of parse).

2. **convert2content runs on 100% of siblings and is 20–80 ms — i.e. it is
   comparable to or LARGER than a 12.7 ms static parse.** Once the static path
   is the common case, convert2content is plausibly **the single largest
   per-sibling cost.** This is the real lever (audit item (c)/#4), which the
   audit under-weighted.

---

## (a) Vendor subclass: `_preprocess_template_data` once per cluster — REVISED DOWN

**Expected: ~1.06x on the LBP path (NOT 1.3–2x). Effort S. F1 risk: none (bit-identical, with the correctness constraint below).**

The prototype `stage3_reuse_proto.py` (`ReusableLayoutBatchParser`) splits
`parse()` into `prepare_template()` (once/cluster) + `parse_page()` (per
sibling). It correctly reuses the template-side work and the normalized
`html_element_dict`.

### The load-bearing correctness constraint (why naive caching is unsafe)

`_preprocess_template_data` builds `self.ids` from **both** the template doc
**and the sibling tree** (any id appearing >3× in *that page* is marked
invalid → `False`). It then builds `self.processed_template_data` by calling
`normalize_key(...)`, which **reads `self.ids`**. Therefore
`processed_template_data` is, in general, **page-dependent**: a sibling that
repeats some id >3× can flip how a template key normalizes (id-bearing key →
class/id key). Caching `processed_template_data` blindly across siblings would
change `find_blocks_drop`'s matching on those pages → **change output → break
F1 parity.**

The prototype handles this exactly: it caches the template-only processed dict,
and per page rebuilds **only if** the page introduces a volatile id (count>3)
that collides with a template key (rare). Otherwise it reuses the cache. Output
is bit-identical to the vendor `parse()`. A `verify_equivalence()` harness is
included to assert body-for-body equality on a sibling sample before rollout.

**Verdict:** worth landing as a small, F1-safe win, but it does **not** move the
needle alone. Land it folded into the existing static-first tier; the marginal
~6% compounds with (c).

---

## (b) Page-level load-balancing refinements — KEEP, modest headroom

**Expected: protects wall-clock against the dynamic-LBP tail; ~1.0–1.3x on already-balanced shards, more on pathological ones. Effort S (already 80% done). F1 risk: none.**

`stage3_cpu_propagation.py` already implements the core of audit #3:
`PAGES_PER_TASK = 300` splits giant clusters into page-level tasks that share a
`mapping_data`/`red_selectors` reference (lines 1069–1123). Remaining refinements:

1. **Chunk by page count, not task count.** `cluster_chunk_size=500` still
   chunks *tasks*; a chunk of 500 tasks ranges 500–150k pages. Replace with a
   target pages-per-chunk (e.g. 30k) so progress/memory and the executor's
   in-flight set are bounded. Pure scheduling; no output change.
2. **`PAGES_PER_TASK` re-tune.** 300 is fine for static LBP (~12.7 ms → ~3.8 s
   per task) but a 300-page task that lands entirely on the **dynamic** fallback
   (~0.3–3 s/page) is a 90–900 s straggler. Drop `PAGES_PER_TASK` to ~128 for
   un-validated (dynamic-bound) clusters so the tail parallelizes; keep 300+ for
   static-validated clusters (cheap pages, less per-task overhead). This needs
   `use_static` to be known at task-build time — hoist the per-cluster
   validation out of `_process_cluster_task` into task construction (it's
   currently decided inside the worker, so the splitter can't see it). Doing the
   K-sample validation once on the driver also removes the redundant
   re-validation that happens in every page-level task of the same cluster
   today (`_cluster_static_trustworthy` is memoized **per worker**, so a cluster
   split across W workers is validated W times).

   That last point is a real, currently-paid cost: the validation runs
   `2*K` LBP parses (static+dynamic) + `2*K` convert2content per worker per
   cluster (K=3 → up to 6 parses + 6 converts). For a cluster split across 20
   workers that's up to 120 parses + 120 converts of pure overhead. Hoisting
   validation to the driver (compute once, ship a `use_static` bool per task)
   removes ~ (W-1)/W of it. On heavily-split clusters this is a **bigger real
   win than (a)**.

**Verdict:** finish (b): driver-side validation + pages-based chunking +
role-aware `PAGES_PER_TASK`. F1-safe. Net ~1.1–1.3x on realistic shards, more
where big clusters are split (removes the duplicated validation tax).

---

## (c) convert2content reuse / skip mm_md when only text is needed — BIGGEST LEVER

**Expected: up to ~2x on the static path if convert can be halved; ~1.3–1.6x realistically. Effort S–M. F1 risk: none for object reuse; LOW–MEDIUM if changing output_format.**

convert2content (20–80 ms) runs on **every** sibling and, once the parse is the
fast static ~12.7 ms, convert is the dominant per-page term. Levers:

1. **Reuse a single MinerU case/bindings object per worker** (prototype `R2`,
   `ReusableConverter`). Removes per-page import/lookup and object churn. Output
   identical. Small but free. (Effort S, risk none.)
2. **Avoid the second lxml parse.** `_layout_batch_parser_propagate` returns
   `main_html_body` (a serialized HTML string); `_convert_main_html_to_content`
   then **re-parses** it with lxml inside MinerU. The body is produced from an
   already-parsed lxml tree (`element_to_html(body)` in `htmll_to_content2`).
   A vendor-aware path could hand MinerU the **lxml element** (or have the
   reusable parser emit the text directly) and skip one full parse+serialize+
   reparse round-trip. This is the single largest mechanical waste on the fast
   path. Requires confirming MinerU's `convert2content` can accept a pre-parsed
   tree or that the parser's own `get_text_with_newlines` output matches MinerU
   `mm_md` for the propagated fragment (it likely does NOT match byte-for-byte —
   MinerU adds markdown structure — so **gate on F1**, this is the MEDIUM-risk
   part). If MinerU markdown fidelity is required for F1, keep mm_md but still
   eliminate the redundant re-parse by passing the element.
3. **Text-only fast path for content-only consumers.** If a downstream consumer
   only needs `dripper_content` (text), `convert2content(output_format='txt')`
   or the parser's own text extraction is much cheaper than `mm_md` markdown
   rendering. **Only if** the F1 metric is computed on text (it is — token-F1);
   markdown structure tokens could change F1 slightly. **Gate on compare_f1.**

**Verdict:** (c.1) reuse is free; land it. (c.2) eliminating the re-parse is the
highest-value mechanical fix on the fast path and is F1-safe if MinerU keeps the
same content. (c.3) is the largest potential win but must be F1-gated. Combined,
(c) is where the real 1.3–2x lives — not (a).

---

## (d) `_load_cluster_manifest_shard` full-HTML-load — MEMORY/STARTUP, not throughput

**Expected: 0 throughput change at 44k rows; required for large shards (avoid OOM / cut startup). Effort S. F1 risk: none.**

`_load_cluster_manifest_shard` (lines 804–846) reads `["url","html"]` for the
**whole** shard then nulls non-siblings — it materializes every page's HTML
(GBs) even though only siblings need it, contradicting its own docstring. At the
planned per-node shard sizes this inflates peak RSS and delays first-page work,
and will OOM 220 GB nodes if shards grow. Fix: read HTML only for sibling URLs
via `pq.iter_batches(columns=['url','html'])` + an in-loop filter against the
sibling-URL set, or a row-group predicate. Pure I/O; output unchanged. Do it for
robustness at scale, not for steady-state pages/s.

---

## Combined throughput arithmetic

Per-sibling time on the **static** path today (dominant ~79% of siblings):

    parse_static (~12.7 ms) + convert_mm_md (~20–80, take 50 ms) ≈ 62.7 ms
      => ~16 sibling-pages/s/worker static-only.

The reported ~77 pages/s/node (64 workers) reflects the mix of fast static,
near-free reps/singletons (copies), and the dynamic tail; treat 62.7 ms as the
static-path unit and optimize that.

| Change | static-path ms/page | static-path pages/s/worker | note |
|---|---|---|---|
| Today (static parse + mm_md convert) | 12.7 + 50 = 62.7 | 16.0 | baseline |
| + (a) template reuse | 12.0 + 50 = 62.0 | 16.1 | ~1.01x (whole-page) — negligible vs convert |
| + (c.1) converter reuse | 12.0 + ~45 = 57.0 | 17.5 | object churn removed |
| + (c.2) skip redundant re-parse | 12.0 + ~30 = 42.0 | 23.8 | **1.49x vs baseline** |
| + (c.3) txt instead of mm_md (IF F1-safe) | 12.0 + ~12 = 24.0 | 41.7 | **2.6x vs baseline** (gate on compare_f1) |
| + (b) hoisted validation on split clusters | — | — | removes (W−1)/W duplicate validation cost; protects wall-clock on the dynamic tail |

So the realistic, F1-safe target is **(a)+(b)+(c.1)+(c.2) ≈ 1.5x → ~115
pages/s/node**, and **if (c.3) passes the F1 gate, ~2.5x → ~190 pages/s/node**.
(a) alone is ~1.01–1.06x and is NOT a path to 2–3x; the audit's framing of #2 as
the second-biggest lever is wrong — **convert2content is.**

### Does this hit the project target?

The hard project target is GPU 2-day (Stage 2), not Stage 3 — Stage 3 at 77
pages/s/node already comfortably exceeds the GPU's 27 pages/s/node, so Stage 3
is **not** the pipeline bottleneck. The value of H4 is (i) shrinking the CPU
node count (40 CPU nodes) needed to keep up with the GPU stage and the fallback
LLM path, and (ii) headroom if `PAGES_PER_TASK`/validation overhead bites at
scale. At 1.5–2.5x, Stage 3 needs roughly half the CPU nodes, freeing budget —
but it does **not** by itself move overall F1 (>0.90 target) or the GPU 2-day
target.

---

## Recommendation (priority order)

1. **(c.2) Eliminate the redundant lxml re-parse between LBP body and
   convert2content** — biggest F1-safe mechanical win (~1.5x). Then **(c.1)**
   reuse the converter object (free).
2. **(b) Hoist per-cluster static-validation to the driver** (compute once, ship
   `use_static` per task) + **pages-based chunking** + role-aware
   `PAGES_PER_TASK`. Removes the duplicated validation tax on split clusters and
   tames the dynamic-LBP tail. F1-safe.
3. **(c.3) Evaluate `txt` vs `mm_md` convert on a compare_f1 sample.** If
   token-F1 ≥ 0.99 vs the mm_md baseline, switch the fast path to txt for ~2.6x.
   Gate strictly.
4. **(a) Fold `ReusableLayoutBatchParser` into the static tier** as cheap
   insurance (~1.06x), using the prototype's id-collision-safe reuse. Verify
   with `verify_equivalence()` first.
5. **(d) Stream sibling HTML in `_load_cluster_manifest_shard`** for memory/
   startup robustness at large shard sizes.

Prototype: `stage3_reuse_proto.py` (R1 reusable parser with the F1-safe
id-collision rebuild rule + R2 reusable converter + an equivalence harness).

## F1-safety summary

- (a) reuse: **bit-identical** given the id-collision rebuild rule — verify with
  `verify_equivalence()`.
- (b) load-balance / driver validation: **no output change** (the validation
  decision and the parse are unchanged; only *where* they run).
- (c.1) converter reuse: identical output.
- (c.2) skip re-parse: identical content **iff** MinerU consumes the same tree;
  gate on compare_f1 if any serialization difference.
- (c.3) txt vs mm_md: **changes content format** — MUST pass compare_f1 ≥ 0.99
  before enabling. Do not ship blind.
- (d) HTML streaming: no output change.
