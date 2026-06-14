# CPU Stages Performance Optimization Plan — CC-scale MinerU-HTML Pipeline

Scope: the CPU stages of the 3-stage Dripper / MinerU-HTML pipeline that run on
the 40 CPU nodes (`cpu_short`, 64 workers/node via `ProcessPoolExecutor`):

- `stage1a_feature_extraction.py` — `get_feature()` on **all** pages.
- `stage1c_cpu_preprocess.py` — `simplify_single_input` + `build_prompt` on reps+singletons (~9%).
- `stage2b_cpu_postprocess.py` — `parse_result` → `extract_main_html_single` → `convert2content` + `map_parser_cls` on reps+singletons (~9%).
- `stage3_cpu_propagation.py` — LayoutBatchParser propagation on siblings (~91%). **Already separately optimized (~77 pages/s/node); not re-optimized here, see `STAGE3_PERF_AUDIT.md`.**

Target: ≥50% of CC-MAIN (≈1.2B of 2.4B pages) in ~1–2 days on 40 CPU + 16 GPU nodes.
This document is **analysis + design only** — no stage scripts are edited (stage2/stage3 are under concurrent edit).

---

## 1. Effective whole-corpus throughput (the key reframing)

Each CPU stage processes a different **subset** of the corpus. To find the true
per-corpus-page CPU bottleneck, convert each stage's *raw* rate (pages/s/node
measured on the subset it actually touches) into an **effective whole-corpus
rate** = `raw_rate / subset_fraction`. The effective rate is "if this stage were
the only thing gating the corpus, how many corpus-pages/s/node would it sustain."

| Stage | Op | Subset of corpus | Raw pages/s/node (64w) | Effective corpus pages/s/node |
|---|---|---|---:|---:|
| stage1a | `get_feature` (DOM parse + layout feature) | 100% | 595 | **595** |
| stage1c | `simplify_single_input` + `build_prompt` | ~9% | 73 | 73 / 0.09 ≈ **810** |
| stage2b | `parse_result`+`extract_main_html_single`+`convert2content`+`map_parser_cls` | ~9% | 95 | 95 / 0.09 ≈ **1055** |
| stage3 | LayoutBatchParser propagation | ~91% | 77 | 77 / 0.91 ≈ **85** |

**True CPU bottleneck per corpus-page is stage3 (~85 eff).** After stage3,
**the next CPU bottleneck is stage1a (~595 eff)** — it is the only other CPU stage
that touches 100% of pages, and its effective rate is ~1.4× faster than stage1c
and ~1.8× faster than stage2b on a whole-corpus basis. stage1c and stage2b are
**not** corpus bottlenecks in the baseline 9%-LLM regime.

### End-to-end CPU throughput (stages are sequential SLURM jobs)

The pipeline runs the CPU stages **sequentially** (1a → [1b GPU] → 1c → [2 GPU] → 2b → 3),
so the combined CPU wall-time per corpus-page is the **sum of reciprocals** of the
effective rates (each stage's wall time adds up):

```
1/T_cpu = 1/595 (1a) + 1/810 (1c) + 1/1055 (2b) + 1/85 (3)
        = 0.001681 + 0.001235 + 0.000948 + 0.011765
        = 0.015629  s·node/page
T_cpu  ≈ 64 effective corpus pages/s/node  (CPU-only, sequential)
```

stage3 alone consumes **0.01176 / 0.01563 = 75%** of the CPU wall budget.
stage1a is the second-largest at **11%**; 1c+2b together are **14%**.

**40-node projection (CPU-only, baseline 9% LLM):**
`64 × 40 = 2,560 corpus pages/s` → `2,560 × 86,400 = 221M pages/day`.
1.2B pages (50% of CC) ⇒ **≈5.4 days CPU-only** — over the 1–2 day target.
The plan below closes that gap.

> Note: GPU stages (1b DBSCAN, 2 vLLM on 16 GPU nodes) run on different nodes and
> overlap is possible at the fleet level, but within one segment the SLURM chain is
> sequential, so CPU and GPU wall times currently add. The CPU budget is the binding
> constraint addressed here.

---

## 2. Redundant DOM parsing across stages (the cross-cutting waste)

The same raw HTML string is parsed into a DOM **independently and repeatedly**.
`mineru_html` caches a parsed/simplified DOM on the `case` object *within* a single
stage's worker call, but **nothing is cached across stages or across processes**.
Per corpus-page, counting full HTML→DOM parses:

| Stage (subset) | Full HTML DOM parses per page it touches |
|---|---|
| stage1a (100%) | 1 (`get_feature`) |
| stage1c (9%) | 1 (`simplify_single_input`; `build_prompt` reuses `case.process_data`) |
| stage2b (9%) | 3–4 (`extract_main_html_single` re-parses; `convert2content` re-parses the extracted fragment; `map_parser_cls.parse` parses `typical_raw_html` **and** `typical_raw_tag_html`) |
| stage3 (91%) | 2 (LayoutBatchParser parses sibling HTML; `convert2content` re-parses extracted fragment) — plus per-call template re-normalization (see W2 in STAGE3_PERF_AUDIT) |

A corpus-page that is a representative is parsed ~1 (1a) + 1 (1c) + 3–4 (2b) ≈ **5–6 times**.
A sibling is parsed 1 (1a) + 2 (3) = **3 times**. Parsing is 5–30 ms (median) up to
150 ms (large pages) per parse — a large fraction of every CPU stage's cost.

**Reality check on cross-stage DOM reuse:** parsed lxml/selectolax trees are **not**
picklable/serializable cheaply, and stages run as separate SLURM jobs in separate
processes (and partly separate venvs), so passing a live DOM between stages is **not
feasible**. The actionable levers are: (a) reduce parses *within* a stage, (b) reduce
the HTML bytes parsed (truncate/clean before parse), and (c) avoid re-parsing the same
fragment twice in 2b/3.

---

## 3. Per-stage optimization plan

Effort key: **S** ≤1 day, **M** a few days, **L** ≥1 week / cross-team.
F1 risk = risk of changing extraction quality (Dripper main-content F1).

### stage1a — `get_feature`, 595/s, 100% of pages (2nd CPU bottleneck)

`_extract_one` submits **one `ProcessPoolExecutor` future per page** (line 101),
pickling the full HTML string into the worker and the full HTML string back out
(`html` is echoed into the output row, lines 56/97). At ~595 pages/s/node the
per-task scheduling + double-pickle of 50–500 KB HTML is a measurable fraction of cost.

| # | Lever | Expected speedup | Effort | F1 risk |
|---|---|---|---|---|
| 1a-1 | **Batch tasks**: submit chunks of N≈256 records per future (map over a list inside the worker) instead of one-future-per-page. Cuts future scheduling + result-marshalling overhead by ~256×. | 1.1–1.3× | S | none |
| 1a-2 | **Stop echoing `html` back through the pickle boundary.** `get_feature` only needs `html` as input; the output row re-emits the full HTML (worker→parent pickle of every page). Have the worker return only `(idx, dom_feature)` and re-attach `html` in the parent from the already-loaded `shard_df` (zero-copy). Halves the bytes crossing the IPC boundary. | 1.1–1.25× | S | none |
| 1a-3 | **Truncate oversized HTML before `get_feature`.** Layout features saturate well below full page size; cap at e.g. 512 KB–1 MB. Bounds the parse tail (the 50–150 ms pages). | 1.05–1.15× (tail) | S | low — verify clustering F1 on capped pages |
| 1a-4 | **Right-size workers.** 64 workers on a 64-CPU node leaves no core for the parent's pickle/concat loop and parquet I/O; the parent thread that drains `as_completed` becomes a serialization bottleneck at high rate. Test 56–60 workers + larger result batches (pairs with 1a-1). | 1.0–1.1× | S | none |
| 1a-5 | **Persist `html` once, not per stage.** Currently 1a, 1c, 2b, 3 each re-read `html` from parquet. If the manifest stored `html` compressed once and stages keyed by `warc_*` offsets, repeated full-HTML materialization shrinks — but this is a manifest redesign. | I/O only | L | none |

Realistic stage1a: **1.3–1.6×** → ~770–950 eff pages/s/node from S-effort levers (1a-1+1a-2+1a-4).

### stage1c — `simplify_single_input` + `build_prompt`, 73/s raw, ~9% (NOT a baseline bottleneck)

`simplify_single_input` is one full DOM parse + tree simplification; `build_prompt`
reuses the cached `case.process_data` (0 extra parses). Same per-future overhead
pattern as 1a (one future per record, `html` echoed into the output, lines 84/159).

| # | Lever | Expected speedup | Effort | F1 risk |
|---|---|---|---|---|
| 1c-1 | **Batch tasks** (chunk records per future), same as 1a-1. | 1.1–1.3× | S | none |
| 1c-2 | **Don't echo full `html` through worker pickle** if 2b can re-read it from the stage1b/1a parquet by url/offset. Currently `html` is carried 1c→2→2b purely so 2b can re-parse it. Carrying `simp_html`+`map_html` (already produced) is necessary; the *raw* `html` round-trip is the expensive part. | 1.1–1.2× + downstream I/O | M | none |
| 1c-3 | **Reuse simplification in 2b.** `simplify_single_input` in 1c already produced `simp_html`/`map_html`; 2b re-derives DOM state from raw `html` again. Passing enough state to skip 2b's re-parse is the cross-stage win (see 2b-1). | see 2b | M | low |

stage1c is fast enough on the corpus (810 eff) that S-effort batching is sufficient; do not over-invest unless the LLM fraction rises (Section 4).

### stage2b — postprocess, 95/s raw, ~9% (NOT a baseline bottleneck, but most parses/page)

This stage does the **most redundant parsing**: `extract_main_html_single` parses,
`convert2content` parses the extracted fragment, and for representatives
`map_parser_cls({}).parse(...)` parses **both** `typical_raw_html` and
`typical_raw_tag_html`. The `pickle+base64` of the template (`mapping_json`, line 125)
is also non-trivial CPU + output size.

| # | Lever | Expected speedup | Effort | F1 risk |
|---|---|---|---|---|
| 2b-1 | **Build the `case` from `simp_html`/`map_html` already computed in 1c instead of re-parsing raw `html`.** 1c ran `simplify_single_input`; 2b reconstructs `process_data` from `simp_html`/`map_html` (it already does, line 85) but `extract_main_html_single`/`convert2content` still re-parse. Audit whether the raw-HTML parse in `extract_main_html_single` can be fed the cached simplified DOM. | 1.2–1.4× | M | medium — must match standalone path exactly; validate F1 |
| 2b-2 | **Batch tasks per future**, same as 1a-1/1c-1. | 1.1–1.3× | S | none |
| 2b-3 | **Don't echo raw `html` out**; 2b's output (`mapping_json`, `dripper_content`, `dripper_html`) doesn't need raw html re-emitted. Reduces output pickle + parquet size. | 1.05–1.15× + I/O | S | none |
| 2b-4 | **Cheaper template serialization.** `pickle.dumps`+`b64encode` per representative is CPU and ~1.3× size inflation; representatives are 9% of pages but mapping_json is large. Consider raw pickle bytes in a binary parquet column (skip base64) — stage3 reads it. | 1.0–1.1× + big I/O | S | none — format-only, keep pickle |
| 2b-5 | **Truncate oversized HTML** before parse (same as 1a-3). | tail | S | low |

Realistic stage2b: **1.3–1.6×** combining 2b-1 (M) + 2b-2/2b-3 (S).

### stage3 — already optimized (~77/s, 91%, the bottleneck)

Out of scope per instructions; see `STAGE3_PERF_AUDIT.md`. Noted here only because it
dominates the CPU budget (75%). The single highest-leverage CPU win for the whole
pipeline remains stage3 (W1 dead XPath fast-path, W2 per-sibling template
re-normalization, W3 cluster-level load imbalance, L1 full-table HTML load). Even a
2× on stage3 (85→170 eff) does more for end-to-end than maxing out 1a/1c/2b combined.

---

## 4. Scenario: LLM fraction rises to ~20% (fallback-to-LLM)

If the fallback-to-LLM effort raises the share of pages sent through the LLM path
from ~9% to ~20%, then **stage1c and stage2b loads roughly double** (subset 0.09 → 0.20)
and the sibling share for stage3 drops from 0.91 to 0.80.

Recompute effective rates (raw per-page cost unchanged):

| Stage | Subset | Raw /s | Effective /s (20% regime) |
|---|---:|---:|---:|
| stage1a | 100% | 595 | 595 |
| stage1c | 20% | 73 | 73 / 0.20 = **365** |
| stage2b | 20% | 95 | 95 / 0.20 = **475** |
| stage3 | 80% | 77 | 77 / 0.80 = **96** |

```
1/T_cpu = 1/595 + 1/365 + 1/475 + 1/96
        = 0.001681 + 0.002740 + 0.002105 + 0.010417 = 0.016942
T_cpu  ≈ 59 eff corpus pages/s/node   (vs 64 in the 9% regime)
```

Stage3 is still the bottleneck (61% of budget), but **stage1c+stage2b jump from 14%
to 29% of the CPU budget** and stage1c (365 eff) becomes the clear #2. In this regime
the stage1c/2b optimizations (especially the M-effort DOM-reuse levers 1c-3/2b-1)
move from "nice to have" to "required." The S-effort batching levers should be done
regardless.

---

## 5. End-to-end math vs the 50%/day target

Target: 1.2B pages in ≤2 days on 40 nodes ⇒ need ≥ **1.2e9 / (2×86,400) / 40 = 174 corpus pages/s/node** CPU effective. (For 1 day: ≥347.)

| Regime | Eff pages/s/node | 40-node pages/day | 1.2B pages wall |
|---|---:|---:|---:|
| Baseline today (9% LLM) | 64 | 221M | **5.4 days** |
| + S-effort batching on 1a/1c/2b (no stage3 change) | ~66 | 228M | 5.3 days |
| + stage3 2× (the real lever) | ~118 | 408M | **2.9 days** |
| + stage3 2× AND 1a 1.5×, 2b 1.4× | ~128 | 442M | **2.7 days** |
| + stage3 3× AND 1a/1c/2b S+M levers | ~165 | 570M | **2.1 days** |

**Conclusion:** The CPU pipeline is **stage3-bound**. No amount of 1a/1c/2b
optimization alone reaches the 2-day target — the sum-of-reciprocals is dominated by
stage3 (75% of budget). Hitting ≤2 days requires **stage3 ≥2.5–3×** *plus* the
S-effort batching/IPC fixes on the other stages to keep them from becoming the new
bottleneck once stage3 speeds up. Once stage3 reaches ~3×, stage1a (the 100%-of-pages
stage) becomes the next ceiling, so its S-effort levers (1a-1, 1a-2, 1a-4) should land
in the same pass.

A reach for ≤1 day (≥347 eff/node) is not achievable on 40 CPU nodes with this
architecture; it would require either ~80 CPU nodes or moving stage3's hot
LayoutBatchParser kernel off the per-sibling Python path.

---

## 6. Prioritized action list (CPU stages, excluding stage3 internals)

1. **(S, all stages)** Batch `ProcessPoolExecutor` tasks: N≈256 records/future instead of one-per-page. Removes per-page scheduling + a large share of IPC. Applies to 1a/1c/2b identically. ~1.1–1.3× each, zero F1 risk.
2. **(S, 1a & 2b)** Stop echoing raw `html` through the worker→parent pickle; re-attach from the parent-side DataFrame. ~1.1–1.25× plus smaller output parquet.
3. **(S, all)** Right-size workers to ~56–60 and verify the parent drain loop isn't serializing; truncate oversized HTML before parse to bound the tail.
4. **(M, 2b)** Feed `extract_main_html_single`/`convert2content` the already-simplified DOM/HTML from 1c rather than re-parsing raw `html` — the single biggest *redundant-parse* removal (3–4 parses → 1–2). Must be F1-validated against the standalone path.
5. **(S, 2b)** Store `mapping_json` as binary pickle (drop base64) in a binary parquet column; stage3 reads bytes directly.
6. **(Required if LLM→20%)** Land levers 1c-3/2b-1 (DOM reuse) — 1c/2b become 29% of the CPU budget in that regime.
7. **(L / separate effort, highest leverage)** stage3 — see `STAGE3_PERF_AUDIT.md`. This is where the 2-day target is actually won or lost.

---

## Summary

- **Effective whole-corpus CPU rates:** stage1a 595, stage1c ~810, stage2b ~1055, stage3 ~85 pages/s/node.
- **True CPU bottleneck = stage3 (~85 eff, 75% of the CPU wall budget). Next bottleneck after stage3 = stage1a (595 eff, the only other 100%-of-pages stage).** stage1c/2b are not corpus bottlenecks at 9% LLM.
- **Baseline end-to-end CPU ≈ 64 eff pages/s/node** (sum of reciprocals) → ~221M pages/day on 40 nodes → ~5.4 days for 1.2B pages. **Does not meet the 1–2 day target on CPU alone.**
- **Top CPU optimizations:** (1) batch ProcessPool tasks across 1a/1c/2b; (2) stop round-tripping raw `html` through the IPC/pickle boundary in 1a/2b; (3) in 2b, reuse 1c's simplified DOM instead of re-parsing raw HTML 3–4×; (4) binary (non-base64) `mapping_json`; (5) right-size workers + truncate oversized HTML. These give ~1.3–1.6× on each of 1a/2b but only nudge end-to-end (+~3%) because stage3 dominates.
- **The 2-day target is stage3-bound:** it requires stage3 ≈2.5–3× *and* the S-effort fixes above so stage1a doesn't become the new ceiling. Projected end-to-end with stage3 3× + 1a/2b S/M levers: **~165 eff pages/s/node → ~2.1 days for 1.2B pages on 40 nodes.**
- **If LLM fraction → 20%:** end-to-end drops to ~59 eff/node; stage1c (365 eff) becomes the clear #2 bottleneck and the M-effort DOM-reuse levers in 1c/2b become required.
