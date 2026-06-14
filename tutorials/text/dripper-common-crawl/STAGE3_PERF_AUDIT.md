# Stage 3 Performance Audit — CC-scale MinerU-HTML Template Propagation

Scope: `stage3_cpu_propagation.py` (the per-cluster CPU propagation kernel),
with reference to the standalone `dripper/stage.py` `_propagate_layout_template`,
the producer `stage2b_cpu_postprocess.py`, and the installed
`llm_web_kit` package (`LayoutBatchParser`, `MapItemToHtmlTagsParser`,
`html_layout_cosin`) inspected on the Nebius cluster.

Observed today: ~20-60 pages/s/node on one 64-worker node for a 44,117-page
shard (≈40k siblings, ≈3.8k clusters); 12-35 min wall. **100% of siblings take
the slow LayoutBatchParser (LBP) path** because the XPath fast-path is dead code
(AUDIT H1 — confirmed: no upstream stage emits `xpath_rules`).

---

## 1. Where the time goes (reasoned profile)

### What LBP actually does (confirmed from source)

`LayoutBatchParser.parse(task_data)` is a **pure-CPU, single-page** lxml +
selectolax operation. There is **no GPU and no network**. The "Batch" in the
name refers to batch *template matching* strategy, not multi-page batching — it
accepts exactly one `HTML_SOURCE`. Per call it does:

1. `HTMLParser(html_source)` (selectolax) then `html_to_element` (lxml parse) — full DOM parse of the sibling page.
2. `_preprocess_template_data(element_dict, template_doc, tree)` — **re-normalizes the entire template dict and re-parses the template doc on EVERY page** (rebuilds `self.processed_template_data`, `self.ids`).
3. `find_blocks_drop(...)` — recursive DOM walk pruning non-"red" subtrees.
4. When a sibling node's `(tag,class,id)` key does **not** exactly match the template (the common case — class/id hashes, post-ids, session ids drift page-to-page), it falls into the **dynamic-id / dynamic-classid** branches, which call `get_feature()` + `similarity()` (sklearn `DictVectorizer` + `cosine_similarity`) **per candidate node per layer**. This is the dominant cost and explains the 100x spread (hundreds of ms → 12 s): pages whose layout matches the template exactly are fast; pages that force many dynamic similarity computations are slow.
5. Final page-level `get_feature` + `similarity` for the `main_html_success` gate.

Then `_convert_main_html_to_content` runs MinerU `convert2content` (another lxml
parse of the extracted fragment + markdown serialization).

### Per-page cost breakdown (estimated, sibling path)

| Step | Typical | Worst (dynamic-heavy) | Notes |
|---|---|---|---|
| selectolax + lxml parse of sibling HTML | 5-30 ms | 50-150 ms | scales with page size (50-500 KB) |
| `_preprocess_template_data` (redundant per page) | 2-10 ms | 10-40 ms | **rebuilt every call — should be once/cluster** |
| `find_blocks_drop` static matching | 10-50 ms | 100-300 ms | DOM-size bound |
| dynamic-id/classid `get_feature`+`similarity` | 0 ms | **1-11 s** | sklearn cosine per node; the real tail |
| final similarity gate | 5-20 ms | 50-100 ms | one get_feature+similarity |
| `convert2content` (MinerU) | 20-80 ms | 100-300 ms | second lxml parse + md render |
| **Total** | **~50-250 ms** | **~2-12 s** | matches observed 20-60 pages/s/node |

So at 64 workers, 20-60 pages/s/node implies ~0.3-3 s mean per page — i.e. a
**heavy tail of dynamic-matching pages dominates wall time**, not the median page.

### Three structural waste sources (independent of the tail)

- **W1 — XPath fast-path is dead (AUDIT H1).** `_parse_xpath_rules(gpu_row["xpath_rules"])` is always `None`; the `if xpath_rules:` branch (lines 369-396) never executes. 100% of siblings hit LBP.
- **W2 — Redundant per-sibling template work.** `_layout_batch_parser_propagate` calls `LayoutBatchParser({}).parse(task_data)` with `task_data = dict(mapping_data)` **once per sibling**. Inside, `_preprocess_template_data` re-normalizes the cluster's template dict on every one of the cluster's siblings. For a 5,000-sibling cluster that is 5,000 redundant template re-normalizations + 5,000 template-doc re-parses. The template is identical for the whole cluster.
- **W3 — Load imbalance.** Tasks are per-cluster (`_process_cluster_task` does one whole cluster). A 5,000-sibling cluster runs serially on one worker while 63 workers idle. The log "chunk 6 jumps 9k→23k pages" is exactly this: one chunk contained a few giant clusters. `cluster_chunk_size=500` chunks *tasks* (clusters), not pages, so a chunk's page count is unbounded.

### I/O cost (AUDIT L1, confirmed)

`_load_cluster_manifest_shard` (line 636) does `pq.read_table(path, columns=["url","html"])` — reads **every** row's HTML into memory, then nulls non-siblings. The comment claims it avoids the full-table load; it does not. For a 44k-row shard this is tolerable, but it adds a full-shard HTML materialization (~GBs) up front and a `drop_duplicates` + `set_index().map()` pass. At the planned per-node shard sizes this is a fixed startup tax, not the steady-state bottleneck, but it inflates peak RSS and delays first-page processing. Not the throughput limiter at 44k rows; would matter if shards grow.

---

## 2. Prioritized optimizations

Effort: S (<1 day), M (1-3 days), L (>3 days). Speedups are per-node throughput multipliers vs the current ~20-60 pages/s baseline.

### #1 — XPath / CSS-selector fast-path derived once per cluster  ⭐ highest value
**Speedup: ~10-50x on the pages it covers (LBP ~0.3-3 s → lxml ~10-50 ms). Effort: M. Risk: MEDIUM (correctness — see §4).**

The template already contains everything needed to build deterministic
selectors. `MapItemToHtmlTagsParser` produces `html_element_dict` as
`{layer_no: {(tag, class, id, sha256, layer_no, idx): (label, (parent_tag,parent_class,parent_id))}}`
where `label ∈ {red, green}`; `red` = main content. The cluster's "keep set" is
the set of `(tag, class, id)` keys labeled `red`. Because `LayoutBatchParser`'s
static path keeps a node iff its normalized `(tag, class, id)` key is in a red
layer entry with a matching parent key, the **static** decision is fully
expressible as lxml/CSS selectors:

Rule-derivation (once per cluster, from `mapping_data`):
```
red_keys = []
for layer, nodes in html_element_dict.items():
    for (tag, cls, idd, *_), (label, parent_key) in nodes.items():
        if label == 'red':
            red_keys.append((tag, cls, idd))
# normalize the same way LayoutBatchParser.normalize_key does:
#   - body/html -> (tag,None,None)
#   - if id present and not blacklisted -> (tag, None, replace_post_number(id))
#   - else -> (tag, replace_post_number(class), replace_post_number(id))
# emit a CSS/xpath selector per red key, e.g.
#   tag[id='...']  or  tag.classfirsttoken  (first class token, post-number stripped)
```
Then per sibling: `doc.cssselect(sel)` / `doc.xpath(expr)` for each red selector,
union the matched subtrees, serialize. lxml `cssselect` compiles the selector
once and matches in a single tree pass.

This is precisely what the existing (dead) `_xpath_propagate` kernel was meant to
consume. The fix is to **populate `xpath_rules`** — either:
- (a) **In Stage 2b**: after building `template`, derive the red-key selector list and write it as a new `xpath_rules` column (pickle/JSON). Stage 3 already reads it. Minimal Stage 3 change; clean separation. (Recommended.)
- (b) **In Stage 3 task-build**: derive selectors from `mapping_data["html_element_dict"]` once per cluster (in `_process_cluster_task`, before the sibling loop) and pass to `_process_sibling_row`. No Stage 2b rerun needed; good for the currently-running data.

**Expected coverage:** the static selector path reproduces LBP exactly when no
dynamic matching was needed — i.e. for siblings whose class/id are stable across
the cluster. That is the *majority* of siblings (same CMS/template → same classes).
Pages that LBP only resolved via dynamic similarity will produce 0 matches and must
**fall back to LBP** (keep it as the fallback, as the design intended). So the
realistic split flips from today's "100% LBP" to "~70-90% fast XPath + ~10-30% LBP".

**Verification gate (mandatory):** before trusting selectors, run a sample where
both XPath and LBP are computed and require near-identical extracted content
(token-level F1 ≥ 0.99) on representatives + a sibling sample. Ship only if the
ratio check (fixed per M1, see §4) and the F1 spot-check pass.

### #2 — Per-cluster template compilation reuse (eliminate W2)
**Speedup: ~1.3-2x on the LBP-fallback pages. Effort: S. Risk: LOW (no F1 change).**

Instantiate and pre-process the parser **once per cluster**, reuse across siblings.
The redundant work is `_preprocess_template_data` (template normalization +
template-doc parse) which is currently rerun per sibling inside
`LayoutBatchParser.parse`. Two ways:

- Cheap, no-vendor-change: in `_process_cluster_task`, pre-`json.loads`/normalize the
  `html_element_dict` once (build the `int`-keyed, tuple-keyed dict the parser
  expects) and pass that as `mapping_data` so the `isinstance(template_data_str, dict)`
  branch is taken (skips the `json.loads` + `parse_tuple_key` loop per page). Stage 2b
  already pickles the dict losslessly (Bug #4), so the dict branch is already hit — but
  `_preprocess_template_data` still reruns. The pure-python win here is modest.
- Bigger win (vendor-aware): add a thin subclass that exposes a `prepare(template)`
  (runs `_preprocess_template_data` once, caches `self.processed_template_data`,
  `self.ids`, parsed `template_doc`) and a `parse_page(html_source)` that reuses them.
  Reset only the per-page `normalize_key_cache`. This removes the per-sibling template
  re-normalization and template-doc re-parse entirely.

Note: the **dynamic similarity** cost (the real tail) is per *page* and is **not**
removed by reuse — only the static template setup is amortized. So #2 alone is a
1.3-2x, not a game-changer; its value is multiplicative with #1 (it speeds the
remaining fallback pages).

### #3 — Page-level / size-balanced work distribution (fix W3)
**Speedup: ~2-4x effective node utilization on imbalanced shards. Effort: M. Risk: LOW.**

Stop submitting one future per cluster. Instead:
- Compute selectors / prepared template **once per cluster** (cheap, on the main
  process or a first map pass), then **fan siblings out at page granularity** into
  fixed-size work units (e.g. 256 siblings each) carrying a *reference* to the
  cluster's compiled template. A 5,000-sibling cluster becomes ~20 units spread
  across workers instead of one 5,000-page serial task.
- Chunk by **page count**, not cluster count: replace `cluster_chunk_size` (tasks)
  with a target pages-per-chunk so progress and memory are bounded and the "9k→23k
  jump" disappears.
- To avoid re-pickling the (large) template per page-unit, key units by `cluster_id`
  and ship the compiled template once via a per-worker LRU cache (worker memoizes
  `cluster_id -> compiled_template`), or pass the template once per chunk.

This converts straggler clusters into parallel work and is what makes the tail
distribution stop dominating wall time.

### #4 — Other / smaller
- **MinerU `convert2content` is per-sibling and cannot be GPU-batched** (it's lxml + md render, ~20-80 ms). It's small relative to LBP today but becomes a meaningful share once #1 lands (XPath 10-50 ms + convert 20-80 ms → convert is ~half the fast-path cost). Mitigations: skip the `mm_md` formatting if only text is needed; reuse a single MinerU case object per worker; or, for the XPath path, consider a lighter text extraction when full markdown fidelity isn't required (risk: changes content format — keep MinerU for parity unless F1 confirms equivalence). **Effort S, do after #1.**
- **L1 HTML load:** switch `_load_cluster_manifest_shard` to read HTML only for sibling URLs via a row-group/predicate filter (or batched `iter_batches` keeping only sibling urls). Reduces peak RSS and startup latency. **Effort S, Risk LOW.** Not a throughput fix at 44k rows but de-risks larger shards.
- **M1 ratio check (correctness, not perf):** the XPath path compares `len(main_html)` (HTML) to `representative_content_len` (text) — dimensionally wrong, will spuriously reject valid siblings. Must be fixed *as part of* #1 or the fast-path will silently drop good pages. Compare text-to-text: convert the sibling first, compare `len(content)` to `representative_content_len` (matches the standalone `_propagated_content_length_ratio_error`).

---

## 3. Target-throughput math

Goal: **50% of CC-MAIN (2.4B pages) in 1 day on 80 CPU nodes.**

- Pages to process in 24 h: 0.5 × 2.4e9 = **1.2e9 pages**.
- Seconds/day: 86,400. With ~85% efficiency (I/O, startup, stragglers) ≈ 73,000 effective s.
- Required aggregate rate: 1.2e9 / 73,000 ≈ **16,440 pages/s**, across 80 nodes
  → **~205 pages/s/node** (≈ **3.2 pages/s/worker** at 64 workers).

Note: not every page is a sibling. Representatives + singletons are **copies**
(near-free, thousands/s). If, say, ~85% of pages are siblings needing extraction,
the sibling-processing rate must be ~205/0.85 ≈ **240 sibling-pages/s/node**.

| Scenario | per-node pages/s | Meets 205/node? |
|---|---|---|
| Today (100% LBP, imbalanced) | 20-60 | ❌ (3.5-10x short) |
| +#3 balance only (LBP still) | 60-120 | ❌ |
| +#2 reuse + #3 balance | 90-180 | ❌ borderline |
| **+#1 XPath fast-path (80% fast @ ~40 ms incl. convert, 20% LBP @ ~1.5 s) + #2 + #3** | **see below** | ✅ |

Fast-path mix calculation (per worker), with 80% XPath @ 40 ms, 20% LBP @ 1500 ms mean:
- mean page time = 0.8×0.040 + 0.2×1.5 = 0.032 + 0.30 = **0.332 s/page → 3.0 pages/s/worker → ~193/node**. Just under target.
- Push LBP share to 10% (better selectors / accept lower-confidence static matches with the ratio+sim gate) @ 1.5 s: 0.9×0.040 + 0.1×1.5 = 0.036+0.15 = 0.186 s → **5.4 pages/s/worker → ~344/node**. ✅ comfortably over.
- Even at a pessimistic 30% LBP @ 1.5 s: 0.7×0.04 + 0.3×1.5 = 0.478 s → 2.1/worker → ~134/node. ❌ — so **driving LBP fallback share down is the lever**, and #3 (so the LBP tail runs in parallel, not serially behind a straggler cluster) is what protects the wall-clock when the tail is non-trivial.

**Conclusion:** #1 is *necessary* to hit ~205/node; #2 and #3 provide the margin
and protect against the LBP tail. The combination **#1 + #2 + #3 reaches the
target** provided the XPath fast-path covers ≥80-90% of siblings (verify
empirically). #2 or #3 alone do **not** get there.

---

## 4. Correctness / F1 risk callouts

The baseline to preserve is the **standalone Dripper** `_propagate_layout_template`,
which runs LBP per sibling with the same `task_data`. Stage 3's LBP path is a
faithful reimplementation (AUDIT confirms the `main_html_body` key is correct).

- **#1 XPath fast-path is the only optimization that changes extraction output.** It approximates LBP's *static* matching but omits LBP's dynamic-id/classid similarity matching and the `more_noise_enable` heuristic (which relabels `p/ul/br/b` natural-language nodes as `red`). On pages where LBP relied on those, pure selectors will under- or over-select. **Mandatory mitigations:**
  - Keep LBP as the fallback (already designed): if selectors return 0 elements OR the (fixed, text-vs-text) ratio gate fails, fall back to LBP. This bounds the worst case to "no worse than today" for those pages.
  - Add the same `main_html_success` similarity gate the standalone uses: after XPath extraction, optionally run `get_feature`/`similarity(template_main_html, extracted)` and fall back to LBP if below `SIMILARITY_THRESHOLD`. (Costs one similarity call ~5-20 ms; cheap insurance for F1.)
  - **Gate the rollout on an F1 spot-check** (`compare_f1.py`) of XPath vs LBP output on a representative sample; require token-F1 ≥ 0.99 before enabling broadly.
- **M1 ratio bug must be fixed with #1.** As written the XPath ratio compares HTML length to text length and will reject valid siblings (`xpath_content_ratio_oob`). Convert sibling → text first, then compare text length to `representative_content_len` (as the standalone does). Without this fix the fast-path's F1 will look artificially bad.
- **#2 (template reuse) and #3 (load balancing) do not change output** — pure performance, LOW risk, provided the per-page `normalize_key_cache` is reset between pages (it is keyed by node tuple and would otherwise leak across pages within a reused parser instance).
- **#4 convert2content shortcuts** (skipping `mm_md`) *can* change content format — keep MinerU `convert2content` for parity unless F1 confirms a lighter path is equivalent.

---

## Top 3 recommendations (summary)

1. **XPath/CSS fast-path from the template's red-key set (`html_element_dict`), with LBP fallback + similarity/ratio gate.** ~10-50x on covered pages, flips siblings from 100% LBP to ~80-90% fast. Effort M, risk MEDIUM (F1 — gate on `compare_f1`). *This is the one that makes the target reachable.*
2. **Compile the cluster template once and reuse across all its siblings** (eliminate per-sibling `_preprocess_template_data` / template re-parse). ~1.3-2x on fallback pages. Effort S, risk LOW.
3. **Page-level, size-balanced work distribution** (split giant clusters across workers; chunk by page count not cluster count). ~2-4x effective utilization on imbalanced shards; removes the straggler "9k→23k" tail. Effort M, risk LOW.

Target math: need **~205 pages/s/node** (16.4k/s aggregate over 80 nodes, 85%
eff.). #1+#2+#3 reach ~190-344/node depending on the LBP fallback share; #2/#3
alone (≤180/node) do not. Driving the LBP fallback fraction below ~20% is the
deciding lever.

A reviewable prototype of the #1+#2 kernel is in `stage3_fast_prototype.py`.
