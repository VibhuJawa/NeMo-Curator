# F1 Improvement Plan — CC-scale MinerU-HTML Clustering + Propagation Pipeline

**Goal:** raise full-pipeline token-multiset F1 (vs standalone Dripper job 335168) from **0.81 → >0.90**, with the least added GPU-LLM cost.

**Scope:** analysis + design only. No stage scripts are edited here. This document quantifies the levers, gives the F1 arithmetic, and specifies the concrete design for the recommended change.

---

## 1. Current state (measured, 44,117-page smoke)

| Role | Pages | Share | F1 |
|---|---|---|---|
| representative | 1,429 | 3.2% | 0.97 |
| singleton | 2,411 | 5.5% | 0.95 |
| sibling | 40,084 | 90.9% | 0.80 |
| **overall** | **44,117** | | **0.81** |

Recomputed overall from the role rows = **0.8102** ✓ (matches the reported 0.81).

### Sibling decomposition (the whole problem lives here)

- **~11.7% of siblings are "fallback" pages** → **~4,690 pages** where Stage 3's two-tier LayoutBatchParser (LBP) propagation failed (`main_html_success=False`, both static and dynamic) → `propagation_method="fallback"`, **empty content → F1 == 0**.
- **Non-fallback siblings (~35,394) already average ~0.91.**
- Check: `(4,690·0 + 35,394·0.91) / 40,084 = 0.804` ✓ ≈ the measured sibling 0.80.

So the **F1==0 fallbacks are the dominant drag.** They alone hold the sibling tier (and therefore the whole corpus, since siblings are 91% of pages) ~0.10 below where it could be.

A second, smaller drag sits *inside* the non-fallback group: **~7.4% of siblings (~2,966 pages) propagated content but still score F1==0** (see Lever 3). The implied average of the non-fallback-**nonzero** siblings is ~**0.993** — i.e. when propagation lands on the right region the token match is essentially exact.

---

## 2. How the standalone baseline avoids this (root cause)

The standalone Dripper stage (`nemo_curator/.../dripper/stage.py`) runs the LLM on **every** page conceptually, but for layout clusters it propagates a template and **routes any propagation failure back to the LLM**. The relevant flags from the baseline command:

- `--layout-template-fallback-llm` (`layout_template_fallback_llm=True`): when propagation errors, re-infer that page with the LLM instead of emitting empty/garbage. See `stage.py:2890-2903` — on `propagated.error` it appends an `_infer_and_postprocess_row(...)` task and awaits it.
- `--layout-template-require-success` (`layout_template_require_success=True`): treat `main_html_success=False` (and `typical_main_html_success=false`) as a hard propagation failure (`stage.py:3011, 3089`) → triggers the fallback-LLM path above. This is exactly the condition our Stage 3 marks as `"fallback"` (`stage3_cpu_propagation.py:470, 607-611`).
- `layout_template_validation_rows` / `layout_template_validation_min_content_f1=0.98` (`stage.py:2759-2829`): for each cluster, run BOTH propagation and LLM on a few sibling "validation" rows and require `token_f1(propagated, llm) ≥ 0.98`. If a cluster fails validation, **all** its remaining siblings are sent to the LLM rather than propagated → bad templates never emit garbage.
- `layout_template_max_selected_item_ratio=0.50` (`stage.py:3111-3117`): reject a template that selected too large a fraction of the page (a "grab everything" template) → propagation failure → fallback LLM.
- `--layout-cluster-threshold 0.95`, `--layout-template-min-cluster-size 2`: tighter clusters → siblings more structurally identical to the representative → propagation succeeds more often.
- `layout_template_defer_fallback_llm` (`stage.py:2722-2729, 3397-3421`, output cols `stage.py:1984-1994`): instead of calling the LLM inline, **emit a deferred row** carrying `simp_html`, `map_html`, the built `prompt`, and `needs_llm=True`, so a *separate downstream pass* runs the LLM in bulk. **This is the multi-stage equivalent of our CC pipeline and is the blueprint for the fix below.**

**Our CC pipeline implements the propagation half but drops the fallback-to-LLM half:** Stage 3 marks failures as `"fallback"` and writes empty content. That single missing routing step is the 0.81-vs-baseline gap.

---

## 3. The levers, quantified

All overall-F1 figures use the fixed role mix (rep 1,429@0.97, singleton 2,411@0.95, sibling 40,084) and only move the sibling number.

### Lever 1 — Route fallback siblings to the LLM (highest value)

Send the ~4,690 fallback siblings through the LLM (the baseline's quality, ~0.96) instead of leaving them empty.

- New sibling F1 = `(4,690·0.96 + 35,394·0.91) / 40,084 = 0.916`.
- **New overall F1 = 0.916** (from 0.81). **Clears 0.90.**

**Extra GPU-LLM cost:** today the LLM runs on reps+singletons = 3,840 pages = **8.7%** of the corpus. Adding 4,690 fallback siblings → **8,530 pages = 19.3%** of the corpus. That is **+10.6 percentage points** of corpus, i.e. the LLM-call count goes up **~2.22×**. This is the price of reaching the baseline's quality on the hard pages — but still ~5× fewer LLM calls than the all-pages baseline.

### Lever 2 — Reduce the fallback rate itself (cheaper, but insufficient alone)

Make propagation succeed on more siblings so fewer fall back at all. Mechanisms (all baseline-supported, would need porting into Stage 1b/2b/3 config — *not done here*):

1. **Tighter clustering** — lower DBSCAN threshold below 0.95 in `stage1b_gpu_dbscan.py` so siblings are more structurally identical to the rep → LBP static/dynamic matching succeeds more often.
2. **Template validation** — port `layout_template_validation_rows` + `min_content_f1=0.98` into Stage 2b/3 so bad templates are *rejected* (and those clusters routed to LLM) rather than silently propagating, and so good templates are trusted with confidence.
3. **`max_selected_item_ratio` gate** — reject "grab-everything" templates.
4. **Multiple representatives per cluster** — pick 2–3 reps and propagate the best-matching template per sibling.

Effect on overall F1 if the fallback rate drops but the still-failing pages stay at F1==0 (i.e. Lever 2 *without* Lever 1):

| Fallback rate | sibling F1 | overall F1 |
|---|---|---|
| 11.7% (today) | 0.804 | 0.813 |
| 8.0% | 0.837 | 0.844 |
| 6.0% | 0.855 | 0.861 |
| 4.0% | 0.874 | 0.877 |

**Lever 2 alone cannot reach 0.90** — even halving the fallback rate to ~6% only gets to ~0.86, because the residual failures still score 0. Its real value is **lowering the volume that Lever 1 must send to the LLM** (cost reduction), not reaching the target by itself.

### Lever 1 + Lever 2 combined (the cost-optimal path)

Reduce fallbacks to ~6% via Lever 2, then route the *remaining* ~2,405 fallbacks to the LLM (Lever 1):

- sibling F1 = `(2,405·0.96 + 37,679·0.91)/40,084 = 0.913`
- **overall F1 = 0.913**
- LLM pages = 3,840 + 2,405 = **6,245 = 14.2%** of corpus (vs 19.3% for Lever 1 alone).

Same >0.90 result, **~half the added LLM cost** of Lever 1 alone. (Recovered pages propagate at ~0.91, almost identical to LLM 0.96, so quality barely changes while cost drops materially.)

### Lever 3 — The ~7.4% non-fallback F1==0 pages (~2,966 pages)

These propagated *something* but token-F1 with the baseline is 0. Likely causes:

- **Baseline is itself empty** (the standalone fell back to trafilatura / produced nothing, or the page is genuinely contentless). When the reference is empty, *any* non-empty output scores 0 and *empty* scores 1.0 — so for these pages F1==0 is an artifact, not a defect, and is **unavoidable / not worth chasing**. A meaningful slice of the 7.4% is expected to be this.
- **Wrong region extracted** — the red-key XPath selectors or LBP matched a sibling-specific block (nav/sidebar/related-posts) that the representative's template didn't intend. Fixable by the validation gate (Lever 2.2) and by the `max_selected_item_ratio` gate.
- **Encoding / charset** — `_coerce_html` decodes bytes as UTF-8 with `errors="replace"`; pages in other encodings yield mojibake tokens that share nothing with the baseline. Small slice; fixable by honoring the WARC/HTTP charset.

**Recommended handling:** *measure first, do not engineer blind.* A short offline diagnostic (no stage edits) over the smoke output should bucket these 2,966 pages into `baseline_empty` (accept, exclude from the F1 denominator as unavoidable) vs `wrong_region` / `encoding` (fixable). Modeling: if ~half are baseline-empty and the other half are lifted from 0 → ~0.9 by the validation gate, the non-fallback average rises 0.91 → ~0.948, adding roughly **+0.01–0.02** overall. This is a *secondary* gain layered on top of Lever 1, not a path to 0.90 on its own.

### Lever 4 — Representative / singleton headroom (near-ceiling, do not pursue)

Reps score 0.97 and singletons 0.95 even though they run the *same* model and prompt as the baseline. The residual ~3% is **model nondeterminism** between our run and job 335168 (sampling, batching, vLLM vs the baseline client, kernel/version differences). This is structural; closing it would require bit-exact decoding parity and yields at most `1,429·0.03 + 2,411·0.05 ≈ 163` token-F1·pages ≈ **+0.004 overall**. **Not worth engineering effort.** Treat ~0.97 as the practical ceiling for any LLM-produced page; this is also why Lever 1 fallbacks are modeled at 0.96, not 1.0.

---

## 4. F1 arithmetic summary — which combination clears 0.90

| Scenario | sibling F1 | **overall F1** | extra LLM (corpus %) | LLM ×cost |
|---|---|---|---|---|
| Baseline (today) | 0.804 | **0.810** | — | 1.00× |
| Lever 2 only → 6% fallback | 0.855 | 0.861 | 0 | 1.00× |
| Lever 2 only → 4% fallback | 0.874 | 0.877 | 0 | 1.00× |
| **Lever 1 only (route all 11.7%)** | 0.916 | **0.916** | +10.6 pts | 2.22× |
| **Lever 1+2 (→6% then route)** | 0.913 | **0.913** | +5.5 pts | 1.63× |
| Lever 1+2+3 | ~0.92 | **~0.92–0.93** | +5.5 pts | 1.63× |

Only scenarios that include **Lever 1 (fallback→LLM)** clear 0.90. Lever 2 is a cost optimizer, not a standalone solution.

---

## 5. Prioritized action list

| # | Lever | Overall F1 after | Effort | Extra GPU-LLM cost |
|---|---|---|---|---|
| 1 | **Fallback siblings → LLM (Stage 3.5)** | **0.916** | **M** | +10.6 pts corpus (2.22×) |
| 2 | Reduce fallback rate (tighter clustering + template validation + ratio gate) | 0.86 alone; enables #1 at half cost | M–L | 0 (saves cost on #1) |
| 3 | Diagnose & fix non-fallback F1==0 (wrong-region / encoding; exclude baseline-empty) | +0.01–0.02 on top | S (diagnose) / M (fix) | ~0 |
| 4 | Rep/singleton determinism | +~0.004 | L | ~0 (not recommended) |

---

## 6. Recommended plan (least added GPU cost to exceed 0.90)

**Do Lever 1, and combine it with the cheap half of Lever 2 (template validation) to keep the LLM volume down.** Concretely:

1. **Lever 2 (validation gate) first**, because it's free at inference time and shrinks the Lever-1 bill: port the baseline's `layout_template_validation_rows` + `validation_min_content_f1=0.98` + `max_selected_item_ratio=0.50` checks into Stage 2b/3 so (a) trustworthy templates propagate confidently and (b) clusters whose template is unreliable are *flagged for LLM* rather than emitting garbage. This is expected to pull the fallback rate from ~11.7% toward ~6%.
2. **Lever 1 (the Stage 3.5 re-inference pass)** to take every page Stage 3 marks `propagation_method="fallback"` (plus the validation-rejected clusters from step 1) through the LLM.

**Projected overall F1: ~0.91 (0.913 modeled), at ~14% LLM corpus coverage (≈1.6× the current LLM cost), vs ~19% / 2.2× for Lever 1 alone.** Both clear the 0.90 target; the combined plan does it at roughly half the added GPU spend.

---

## 7. Design for the #1 path: the **Stage 3.5 fallback re-inference** loop

This mirrors the baseline's `layout_template_defer_fallback_llm` mechanism (`stage.py:2722-2729, 3397-3421`) — propagation failures are *deferred* and re-inferred in a bulk LLM pass — adapted to the CC multi-stage layout.

### 7.1 Which stage emits the fallback set

**Stage 3** already labels every failed sibling with `propagation_method="fallback"` and writes empty `dripper_content` (`stage3_cpu_propagation.py:607-626`). No new emission logic is required — these rows are the fallback set, identified by:

```
propagation_method == "fallback"  AND  cluster_role == "sibling"
```

Stage 3 (or a thin selector) writes these rows' **urls + cluster_id** to a `fallback_manifest/shard_NNNN.parquet`. The HTML is *not* re-stored — it is re-read from the WARC via the `warc_filename / warc_record_offset / warc_record_length` columns that already flow through Stage 1b → the cluster manifest (`stage1b_gpu_dbscan.py:31-36`, read in Stage 3's manifest loader).

### 7.2 How the fallbacks are re-inferred (the second LLM pass)

The fallback set re-enters the **existing Stage 1c → Stage 2 → Stage 2b chain**, run as a small "Stage 3.5" sub-job over only the fallback manifest:

1. **Prompt build (reuse Stage 1c):** for each fallback url, fetch HTML from the WARC, run the same simplification → `simp_html`, `map_html`, and **`prompt`** that Stage 1c produces for representatives. Crucially, each fallback page is now treated as its **own representative** (a standalone page), not a sibling — so it gets a full per-page prompt. (The baseline's deferred row already carries `simp_html`/`map_html`/`prompt`; here we rebuild them, which is simpler than threading them through Stage 3.)
2. **vLLM inference (reuse Stage 2):** run `stage2_gpu_inference.py` unchanged on the fallback prompts. It emits `llm_response`. Because the fallback set is ~6–11% of siblings, this is a *small* GPU job (one or a few GPU nodes), not a re-run of the corpus.
3. **Postprocess (reuse Stage 2b):** run `stage2b_cpu_postprocess.py` with `cluster_role="singleton"` for these rows so it takes the `parse_result → extract_main_html_single → convert2content` path (`stage2b_cpu_postprocess.py:78-111`) and produces `dripper_content` / `dripper_html` — identical to how singletons/reps get their final text today. No template/mapping is needed for these (they are one-offs).

This reuses three existing, tested stages with **zero changes to their algorithms** — only orchestration (a new submit script that points the existing stages at the fallback manifest) and a `cluster_role` override to "singleton".

### 7.3 How results merge back

A final **merge step** (parallel to / extending `merge_stage2_results.py`) overlays the Stage 3.5 LLM results onto the Stage 3 output, keyed by `url`:

- For each url in the fallback set, replace `dripper_content` / `dripper_html` / `dripper_error` from Stage 3 (empty) with the Stage 3.5 LLM result, and set `propagation_method = "fallback_llm"` and `propagation_success = True`.
- All non-fallback rows pass through Stage 3 output unchanged.
- This is a left-join overwrite on `url`; it is idempotent and checkpoint-friendly (same write-to-tmp-then-rename pattern Stage 3 already uses).

```
Stage 1b (cluster)
   → Stage 2/2b (LLM on reps+singletons, build templates)
       → Stage 3 (propagate to siblings)
            ├─ success rows ─────────────────────────────┐
            └─ propagation_method=="fallback" siblings    │
                  → fallback_manifest (url, cluster_id,    │
                    warc locator)                          │
                  → Stage 3.5:  [Stage1c prompt build]     │
                                [Stage2 vLLM infer]         │
                                [Stage2b postprocess]       │
                                  (role forced "singleton") │
                  → fallback_llm results ──────────────────┤
                                                            ▼
                                                   Stage 4 merge
                                          (overlay fallback_llm on url)
                                                  → final output  (F1 ≈ 0.91)
```

### 7.4 Cost & scale notes

- Re-inference volume = fallback count. With the validation gate (step 1 of §6) this is ~2,405 pages on the smoke (5.5% of corpus); at CC scale it scales with the same fraction of siblings. The LLM pass therefore stays a small fraction of the original Stage 2 GPU job.
- Per Nebius parallelism preference: the Stage 3.5 prompt-build (CPU, WARC fetch + simplification) should be parallelized across 4+ nodes / 64+ CPUs; the vLLM pass sizes to the fallback volume (typically 1–few GPU nodes).
- Because re-inferred fallbacks are treated as standalone pages, they inherit the rep/singleton ceiling (~0.96), which is exactly what the F1 model assumes.

---

## 8. Bottom line

- **The 0.81→0.90 gap is almost entirely the ~11.7% fallback siblings scoring F1==0** because our CC pipeline implements template propagation but not the baseline's fallback-to-LLM routing.
- **Recommended:** add a **Stage 3.5 fallback re-inference loop** (Lever 1) that reuses the existing Stage 1c/2/2b stages over only the `propagation_method=="fallback"` siblings, and **first** add the baseline's **template-validation + ratio gates** (cheap half of Lever 2) to shrink the fallback volume.
- **Projected overall F1 ≈ 0.91**, at ~14% LLM corpus coverage (~1.6× current LLM cost) — clearing the >0.90 target at roughly half the added GPU spend of routing every fallback. Levers 3 and 4 are secondary (≤+0.02 and ~+0.004) and not required to hit the goal.
