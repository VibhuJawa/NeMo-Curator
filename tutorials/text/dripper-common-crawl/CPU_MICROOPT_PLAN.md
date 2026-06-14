# CPU Stages Micro-Optimization Plan (Track H5)

Implement-ready, diff-level designs for **stage1a / stage1c / stage2b** of the
MinerU-HTML CPU pipeline. Scope = the four S/M-effort levers requested:

- (a) **Batch ProcessPoolExecutor tasks** (~256 records/future) — cut per-page IPC + scheduling.
- (b) **Stop echoing the raw `html` column** through the worker→parent pickle in 1a/2b.
- (c) **Reuse 1c's simplified DOM in 2b** instead of re-parsing raw HTML 3-4×.
- (d) **Binary `mapping_json`** (drop base64) + **right-size workers**.

This doc references measurements from `CPU_STAGES_PERF_PLAN.md` (baseline raw rates:
1a 595/s, 1c 73/s, 2b 95/s; stage3 77/s is the corpus bottleneck and out of scope).
**No production stage scripts are edited here** — all changes are given as before/after
diffs to be applied by the owner of those files.

---

## Cross-cutting: the IPC/scheduling cost model

`ProcessPoolExecutor` with one `submit()` per page incurs, per page:
- pickle the input `dict` (incl. full `html`, 50-500 KB) parent→worker,
- pickle the output `dict` (re-echoing full `html` in 1a/1c) worker→parent,
- a future object + `as_completed` dispatch + a Python-level result append in the
  single parent drain thread.

At 595 pages/s/node (1a) the parent drain thread is doing ~595 unpickles/s of
50-500 KB payloads = **30-300 MB/s of pure deserialization on one core**, plus dict
construction. That single-threaded parent loop is the realistic ceiling, not the
workers. Batching + not echoing `html` attack exactly this.

---

## stage1a — `get_feature`, 595/s raw, 100% of pages (the #2 CPU bottleneck after stage3)

### Lever 1a-1 + 1a-2 + 1a-4 combined (batch + drop html echo + right-size)

The single most impactful rewrite: process **chunks** in the worker, return only
`(idx, dom_feature)`, and re-attach `html` parent-side from the already-loaded
`shard_df` (zero-copy slice — `html` never crosses IPC twice).

**BEFORE** (`stage1a_feature_extraction.py`, `_extract_one` + the submit loop):

```python
def _extract_one(rec: dict) -> dict:
    global _WEB
    html = rec.get("html", "")
    ...
    return {
        "url": rec.get("url",""), "url_host_name": rec.get("url_host_name",""),
        "html": html,                                   # <-- echoed back
        "dom_feature": json.dumps(feat) if feat else "",
        "warc_filename": rec.get("warc_filename"), ...
    }
...
records = shard_df.to_dict("records")
with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
    futures = {pool.submit(_extract_one, r): i for i, r in enumerate(records)}
    for fut in as_completed(futures):
        results.append(fut.result())
out_df = pd.DataFrame(results)
```

**AFTER** (worker takes `(base_idx, list_of_html)`, returns `(base_idx, list_of_feat_json)`):

```python
def _extract_chunk(payload):
    """payload = (base_idx, [html_str, ...]); returns (base_idx, [feat_json, ...])."""
    global _WEB
    base_idx, htmls = payload
    feats = []
    for html in htmls:
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        feat = None
        if _WEB and html and html.strip():
            try:
                feat = _WEB.get_feature(html)
            except Exception:
                feat = None
        feats.append(json.dumps(feat) if feat else "")
    return base_idx, feats

CHUNK = 256
htmls = shard_df["html"].tolist()
chunks = [(i, htmls[i:i+CHUNK]) for i in range(0, len(htmls), CHUNK)]
feat_col = [None] * len(htmls)
with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
    done = 0
    for base_idx, feats in pool.map(_extract_chunk, chunks, chunksize=1):
        feat_col[base_idx:base_idx+len(feats)] = feats
        done += len(feats)
        if done // 5000 != (done-len(feats)) // 5000:
            tracker.checkpoint(done)

# Re-attach html + passthrough cols parent-side from shard_df (no extra IPC):
out_df = shard_df[["url","url_host_name","html","warc_filename",
                   "warc_record_offset","warc_record_length"]].copy()
out_df["dom_feature"] = feat_col
out_df = out_df[OUTPUT_COLS]
```

Key wins, quantified for a node at the current 595/s:
- **html no longer echoed worker→parent**: removes ~50-500 KB/page from the return
  pickle. The output pickle shrinks from `~html + feat_json` to just `feat_json`
  (~1-5 KB). Parent drain bytes drop ~10-100×. Worth **1.10-1.25×** (1a-2).
- **256/future**: per-future overhead (future alloc, `as_completed` bookkeeping,
  result append) amortized 256×. The parent now does ~2.3 result-merges/s instead of
  595. Worth **1.10-1.30×** (1a-1).
- `html` still ships parent→worker once (unavoidable — it is the input), but only
  once and inside a list (cheaper framing than 595 individual pickles).

> Note: `feat_col[base:base+n] = feats` requires order-preserving assignment, which
> `pool.map` guarantees (results returned in submission order). The explicit
> `base_idx` makes it robust even if you switch back to `submit`/`as_completed`.

### Lever 1a-4 (right-size workers)

Change the default from `cpu_count()-2` to leave 2-4 cores for the now-heavier parent
merge + parquet write:

```python
p.add_argument("--workers", type=int,
               default=max(1, (os.cpu_count() or 4) - 4))   # was -2
```

On a 64-CPU node: 60 workers. With the parent thread no longer the bottleneck (it now
merges chunks, not pages), this prevents oversubscription stalls. Worth **1.0-1.1×**.

### Lever 1a-3 / 1a-5 (truncate / persist-once)

Optional, low-risk tail trim — cap `html` at 1 MB before `get_feature` to bound the
50-150 ms parse tail. Insert in `_extract_chunk`: `if len(html) > 1_000_000: html =
html[:1_000_000]`. F1-low-risk but **must validate clustering F1** on capped pages.
Persist-once (1a-5) is a manifest redesign (L) — out of scope here.

**stage1a expected:** 1.10-1.25 (1a-2) × 1.10-1.30 (1a-1) × 1.0-1.1 (1a-4) ≈
**1.3-1.6×** → 595 → **~770-950 eff pages/s/node**. Effort **S**, F1 risk **none**
(1a-1/1a-2/1a-4) / **low** (1a-3, gated on validation).

---

## stage1c — `simplify_single_input` + `build_prompt`, 73/s raw, ~9% (not a baseline bottleneck; #2 if LLM→20%)

### Lever 1c-1 (batch tasks) — same pattern as 1a-1

`_preprocess_one` returns a dict that re-echoes `html` (line 85) plus the produced
`simp_html`/`map_html`/`prompt`. The `simp_html`/`map_html`/`prompt` are *required*
downstream; only the raw `html` round-trip out is removable, but unlike 1a the raw
`html` must be carried forward to 2b (2b currently re-parses it). So for 1c the lever
is **batching only**, plus optionally adding the state needed for 2b reuse (see 2b-1).

**BEFORE / AFTER** (mirror of 1a):

```python
def _preprocess_chunk(payload):
    base_idx, recs = payload
    return base_idx, [_preprocess_one(r) for r in recs]   # _preprocess_one unchanged

CHUNK = 256
records = df.to_dict("records")
chunks = [(i, records[i:i+CHUNK]) for i in range(0, len(records), CHUNK)]
results = [None] * len(records)
with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
    done = 0
    for base_idx, recs_out in pool.map(_preprocess_chunk, chunks, chunksize=1):
        results[base_idx:base_idx+len(recs_out)] = recs_out
        done += len(recs_out)
        if done // 500 != (done-len(recs_out)) // 500:
            tracker.checkpoint(pages_done=done)
result_df = pd.DataFrame(results)
```

Worth **1.10-1.30×** from per-future amortization. At 73/s raw the absolute parent
overhead is lower than 1a, but at LLM→20% the subset doubles and the per-future cost
matters more — do it regardless.

### Lever 1c-3 (produce reuse state for 2b)

`simplify_single_input` already produces `simp_html` + `map_html`, which 1c emits.
**No additional parse is needed in 1c** to enable 2b reuse — the simplified HTML is
already on the wire. The reuse work lives in 2b (lever 2b-1). The only 1c change to
support it: ensure `simp_html`/`map_html` are emitted **even on the singleton path**
(they are today), so 2b can always skip the raw re-parse. No diff required beyond
confirming this in validation.

`--workers` right-size: same `-4` change as 1a.

**stage1c expected:** **~1.1-1.3×** → 73 → **~80-95 raw** (≈890-1055 eff at 9%;
≈400-475 eff at 20%). Effort **S**, F1 risk **none**.

---

## stage2b — postprocess, 95/s raw, ~9%, **most redundant parsing** (3-4 parses/page)

This is the highest-value micro-opt target because each representative is parsed
3-4× (`extract_main_html_single` parses raw, `convert2content` re-parses the
extracted fragment, `map_parser_cls.parse` parses **both** `typical_raw_html` and
`typical_raw_tag_html`).

### Lever 2b-2 (batch tasks) — S, none

Identical wrapper to 1c-1: `_postprocess_chunk(payload)` calls `_postprocess_one` over
a 256-record list; use `pool.map(..., chunksize=1)` and order-preserving assignment.
Worth **1.10-1.30×**.

### Lever 2b-3 (don't echo raw html out) — S, none

2b's output columns are `mapping_json`, `dripper_content`, `dripper_html`,
`dripper_error`, `inference_time_s` plus passthrough ids — it does **not** re-emit raw
`html`, so the *output* side is already clean. The waste is on the **input** side:
the Stage 2 parquet still carries raw `html` (echoed 1c→2→2b) only so 2b can re-parse
it. The fix is structural (2b-1): once 2b reuses the simplified DOM, the raw `html`
column can be **dropped from the Stage 2 output entirely**, shrinking the 1c→2→2b
parquet by the dominant column. Quantify: raw `html` is ~50-500 KB/page vs
`simp_html`+`map_html` ~5-50 KB combined → **~5-10× smaller intermediate parquet** and
proportionally less parent-side `to_dict("records")` + worker-input pickle. Worth
**1.05-1.15×** CPU + large I/O win.

### Lever 2b-1 (reuse simplified DOM; eliminate raw-html re-parse) — **M, medium F1 risk**

Today (line 83): `case = M.case_cls(M.input_cls(raw_html=raw_html, url=url))` then line
85 attaches `process_data` from `simp_html`/`map_html`. But `extract_main_html_single`
and `convert2content` still re-derive structure from `raw_html`, and `map_parser_cls`
parses raw twice more.

**Two sub-levers:**

1. **Avoid the `map_parser_cls` double-parse of raw.** Line 117-121 passes
   `typical_raw_html=raw_html` **and** `typical_raw_tag_html=map_html or simp_html`.
   `map_parser_cls({}).parse` parses both. The `typical_raw_tag_html` (the tag-mapped
   simplified HTML) is already the structure-bearing artifact; the `typical_raw_html`
   raw parse is needed only for exact text spans. **Action:** confirm with the
   standalone Dripper layout-template stage whether `typical_raw_html` can be fed the
   *already-cleaned* simplified HTML when `simp_html` preserves text (it usually does
   for representatives). If yes, drop one full raw parse here. **F1 risk medium — must
   diff `mapping_json` byte-for-byte against the standalone path on a validation
   shard.** If templates differ, keep raw and skip this sub-lever.

2. **Truncate oversized raw before the `extract_main_html_single` parse** (2b-5): cap
   at 1 MB like 1a-3 — bounds the parse tail. Low risk.

The honest assessment: the `case` object already short-circuits re-simplification via
the attached `process_data`, so the *simplify* parse is not repeated in 2b. The
remaining raw parses (`extract_main_html_single`, `convert2content` fragment parse,
`map_parser` raw parse) are tied to the standalone extraction contract. Removing them
requires matching that contract exactly. **Realistic, F1-safe** subset of 2b-1:
sub-lever (1) only if validated → removes 1 of the 3-4 parses → **1.15-1.30×**. Full
3-4→1-2 reduction is only achievable with deeper standalone-path refactoring (out of
S/M scope, flagged as medium risk).

### Lever 2b-4 (binary mapping_json, drop base64) — S, none

**BEFORE** (line 125):

```python
out["mapping_json"] = base64.b64encode(pickle.dumps(template)).decode("ascii")
```

**AFTER** — emit raw pickle bytes into a **binary parquet column**:

```python
out["mapping_json"] = pickle.dumps(template)   # bytes, not str
```

and ensure the column stays `bytes` (pandas keeps `object` dtype; pyarrow writes it as
`binary`). Stage 3 then reads bytes directly: `pickle.loads(row["mapping_json"])`
instead of `pickle.loads(base64.b64decode(row["mapping_json"]))`.

Quantified: base64 inflates payload **1.333×** and adds an encode (2b) + decode
(stage3) pass over the whole template blob. Templates are large (the dominant per-rep
output). Removing base64: **~25% smaller `mapping_json` column** + drops the encode CPU
in 2b and the decode CPU in stage3. CPU win **1.0-1.1×** in 2b, but the **I/O + stage3
read win is the real prize** (stage3 is the corpus bottleneck — see note below).

> **Cross-stage note:** 2b-4 also benefits **stage3** (the actual bottleneck): stage3
> reads `mapping_json` for the 9-20% of pages that are templates and base64-decodes
> them per sibling group. Dropping base64 removes that decode from the hot
> propagation path. Coordinate the format change with the stage3 owner — both ends
> must flip together (this is a one-line change on each side).

`--workers` right-size: same `-4`.

**stage2b expected:** 1.10-1.30 (2b-2) × 1.05-1.15 (2b-3 I/O) × 1.15-1.30 (2b-1
sub-lever 1, *if validated*) ≈ **1.3-1.6×** → 95 → **~125-150 raw** (≈1390-1670 eff at
9%; ≈625-750 eff at 20%). Without the M-effort 2b-1 (S-only): **1.15-1.45×** →
~110-140 raw. Effort **S** (2b-2/3/4) + **M** (2b-1). F1 risk **none** (2b-2/3/4) /
**medium** (2b-1, gated on byte-diff validation).

---

## End-to-end CPU throughput after these micro-opts (40 nodes)

Using the sum-of-reciprocals model from `CPU_STAGES_PERF_PLAN.md §1`. stage3 stays at
77/s raw (85 eff, out of scope) — it dominates, so the micro-opts move the needle only
a few percent end-to-end, exactly as the perf plan predicts. Apply realistic mid-range
multipliers: 1a ×1.45 (595→863 eff), 1c ×1.20 (810→972 eff), 2b ×1.45 (1055→1530 eff).

### Baseline 9%-LLM regime

```
1/T = 1/863 (1a) + 1/972 (1c) + 1/1530 (2b) + 1/85 (3)
    = 0.001159 + 0.001029 + 0.000654 + 0.011765 = 0.014607
T   ≈ 68.5 eff corpus pages/s/node   (was 64 → +7%)
```

- 40 nodes: 68.5 × 40 = **2,740 pages/s → 237M pages/day** (was 221M).
- 1.2B pages (50% of CC): **≈5.1 days CPU-only** (was 5.4). **Still over the 2-day
  target** — because stage3 is 80% of the post-opt budget. The micro-opts' value is to
  **stop 1a/2b becoming the new ceiling once stage3 is sped up**, not to hit the target
  alone (consistent with `CPU_STAGES_PERF_PLAN.md §5`).

### With stage3 at 3× (the real lever, owned elsewhere) + these micro-opts

```
1/T = 1/863 + 1/972 + 1/1530 + 1/255   (stage3 85→255 eff)
    = 0.001159 + 0.001029 + 0.000654 + 0.003922 = 0.006764
T   ≈ 148 eff corpus pages/s/node
```

- 40 nodes: 148 × 40 = **5,920 pages/s → 511M pages/day**.
- 1.2B pages: **≈2.3 days**. Add 1a-3/2b-5 tail-trims and worker right-sizing margin
  → **~2.1 days**, matching the perf plan's reach case. **The micro-opts contribute
  ~10-12 eff pages/s/node here vs ~4.5 in the baseline — they matter *more* once stage3
  is fixed**, because 1a (the 100%-of-pages stage) is then the binding non-stage3 term.

### LLM→20% regime (1c/2b subset doubles, stage3 subset 0.91→0.80)

Raw per-page costs unchanged; recompute effective at 20% with the micro-opt raw rates
(1a 863 eff stays — 100% of pages; 1c raw 88→/0.20=440 eff; 2b raw 138→/0.20=690 eff;
stage3 77 raw /0.80 = 96 eff):

```
1/T = 1/863 + 1/440 + 1/690 + 1/96
    = 0.001159 + 0.002273 + 0.001449 + 0.010417 = 0.015298
T   ≈ 65 eff corpus pages/s/node   (vs 59 without micro-opts → +10%)
```

The micro-opts help **more** in the 20% regime (+10% vs +7%) because 1c+2b grow to
~29% of the CPU budget. **The M-effort DOM-reuse lever 2b-1 becomes worth landing
here** — without it 2b is 690 eff; with the full 3-4→1-2 parse reduction (~2×) 2b would
reach ~1380 eff, lifting end-to-end to ~67/node. The S-effort batching (1a-1/1c-1/2b-2)
and binary mapping_json (2b-4) should land regardless of regime.

---

## Summary table

| Lever | Stage | Effort | F1 risk | Per-stage speedup | Status / gate |
|---|---|---|---|---|---|
| 1a-1 batch 256/future | 1a | S | none | 1.10-1.30× | apply |
| 1a-2 drop html echo (re-attach parent-side) | 1a | S | none | 1.10-1.25× | apply |
| 1a-4 workers cpu-4 | 1a | S | none | 1.0-1.1× | apply |
| 1a-3 truncate >1MB | 1a | S | low | tail | validate clustering F1 |
| 1c-1 batch 256/future | 1c | S | none | 1.10-1.30× | apply |
| 1c-3 emit reuse state (no extra parse) | 1c | S | none | enables 2b-1 | confirm singleton path |
| 2b-2 batch 256/future | 2b | S | none | 1.10-1.30× | apply |
| 2b-3 drop raw html from 1c→2→2b parquet | 2b | S | none | 1.05-1.15× + I/O | apply with 2b-1 |
| 2b-4 binary mapping_json (drop base64) | 2b | S | none | 1.0-1.1× + I/O + stage3 read | coordinate stage3 flip |
| 2b-1 reuse simplified DOM (1 raw parse removed) | 2b | M | medium | 1.15-1.30× | byte-diff vs standalone |
| 2b-5 truncate >1MB before parse | 2b | S | low | tail | validate F1 |

**Net:** 1a **1.3-1.6×**, 1c **1.1-1.3×**, 2b **1.3-1.6×**. End-to-end CPU
**64→~68.5 eff/node (+7%)** at 9% LLM, **~148 eff/node** once stage3 hits 3×
(≈2.1-2.3 days for 1.2B on 40 nodes), and **+10%** in the 20%-LLM regime where 2b-1
becomes worth its M cost. The micro-opts do **not** independently reach the 2-day
target — consistent with the parent plan, the target is stage3-bound — but they keep
stage1a/2b from becoming the new ceiling and deliver a cross-stage win to stage3 via
binary `mapping_json`.
