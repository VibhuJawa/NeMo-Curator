# Pipeline Correctness Audit — MinerU-HTML 7-stage CC-scale extraction

Scope: `stage1a_feature_extraction.py`, `stage1b_gpu_dbscan.py`,
`stage1c_cpu_preprocess.py`, `stage2_gpu_inference.py`,
`stage2b_cpu_postprocess.py`, `stage3_cpu_propagation.py`,
`run_mineru_pipeline.sh` (Stage 4 embedded), `pipeline_metrics.py`,
`compare_f1.py`.

This audit is read-only. No stage scripts were modified. The four previously
fixed bugs (#1 stage3→stage2b wiring, #2 standalone extraction path, #3 chat
template, #4 pickle+base64 template serialization) were re-verified as fixed and
are locked in by `test_pipeline_correctness.py`.

Severity counts: **3 high, 7 medium, 6 low**.

---

## HIGH

### H1 — XPath fast-path in Stage 3 is dead code; ALL siblings hit the slow LayoutBatchParser path
- **Where:** `stage3_cpu_propagation.py:179-228, 368-396, 893`; producers `stage2_gpu_inference.py:25-33`, `stage2b_cpu_postprocess.py:58-68`.
- **Problem:** Stage 3 builds `xpath_rules` from `gpu_row.get("xpath_rules")` and uses it as the primary (~50 ms/page) propagation path. **No upstream stage ever produces an `xpath_rules` column.** Stage 2 `OUTPUT_COLS` and Stage 2b output both omit it (only `mapping_json` is produced). Therefore `_parse_xpath_rules` always returns `None`, the XPath branch never runs, and every sibling falls through to `_layout_batch_parser_propagate` (the ~12 s/page LayoutBatchParser path). The module docstring/perf targets (lines 44-48: "XPath path ~50ms/page … LayoutBatchParser fallback expected <10% of siblings") are therefore inverted in practice — 100% of siblings take the slow path. At CC scale this is the difference between a ~3-4 h run and an effectively infeasible one.
- **Fix:** Either (a) have Stage 2b additionally emit a serialized `xpath_rules` list (derive XPaths from the map_parser template / webkit_response and write them as a column Stage 3 reads), or (b) if XPath propagation is intentionally deferred, delete the dead XPath kernel + ratio logic and update the docstring/perf claims so the design matches reality. Do not ship with the perf section claiming an XPath path that cannot execute.

### H2 — Stage 1b/1c run as 80 independent shards but Stage 3 re-shards the SAME manifest by file slice, risking cross-shard cluster splits
- **Where:** `stage3_cpu_propagation.py:783-787` (`file_start = total_files*idx//num_shards`), vs `stage1b_gpu_dbscan.py:142-278` (one cluster-assignment shard per array task).
- **Problem:** Clustering (Stage 1b) is performed **per shard** — a host's pages are only grouped within the rows that landed in that Stage 1a/1b shard. Stage 3 then re-partitions `cluster_assignments/shard_*.parquet` by *file index* (`manifest_files[file_start:file_end]`). With `num_shards == number of manifest files` (the fleet=80 case) each task gets exactly one file, so a cluster stays whole. But the slicing is generic (`total_files * idx // num_shards`): if the number of manifest files ever differs from `num_shards` (e.g. resubmission with a different `--num-shards`, or merged/re-split manifests), a single host's representative and its siblings can land in **different** Stage 3 tasks. The representative's `gpu_row` would then be absent in the sibling's task → siblings silently degrade to `missing`/`fallback`. There is no assertion that `len(manifest_files) == num_shards`.
- **Fix:** Add a guard at load time: if `len(manifest_files) != num_shards`, either fail loudly or group strictly by `cluster_id` across all files (load all manifests, partition by hash(cluster_id) % num_shards) so clusters are never split. At minimum, log `len(manifest_files)` vs `num_shards` and warn on mismatch.

### H3 — `set -eu` with `afterok` chaining: a single failed array *task* can silently drop pages from all downstream stages
- **Where:** `run_mineru_pipeline.sh:29, 141, 185, 223, 267, 305, 350` (every `--dependency=afterok:${JOB}`).
- **Problem:** Each stage depends on `afterok` of the *whole* array job. If one array task (e.g. shard 37 of Stage 2) fails, Slurm marks that array element failed; depending on cluster config `afterok` may still launch downstream stages for the succeeded elements, and the downstream stages will simply find no input for shard 37 and write an empty/partial shard (Stage 3 `process_shard` even writes an empty shard on missing input, lines 789-793). At CC scale this is a **silent data-loss** path: pages from the failed shard never get extracted, and the final merge has no completeness check (Stage 4 does not verify that all `N_SHARDS` outputs exist with expected row counts). There is no per-shard row-count reconciliation anywhere.
- **Fix:** Add a completeness gate before Stage 4 (or inside it): assert every stage produced exactly `N_SHARDS` shard parquets and that Stage 3 total rows == Stage 1b total rows (modulo dedup). Fail the pipeline loudly otherwise. Consider `afternotok`/`--kill-on-invalid-dep` semantics so a failed array element blocks the chain instead of producing silent gaps.

---

## MEDIUM

### M1 — Content-length ratio check compares HTML length to text-content length (apples to oranges)
- **Where:** `stage3_cpu_propagation.py:373-381` with `representative_content_len` set at `:898-900`.
- **Problem:** `representative_content_len = len(rep_content)` where `rep_content = gpu_row["dripper_content"]` (extracted **text**). The sibling ratio uses `quick_len = len(main_html)` (raw **HTML** fragment). HTML is typically 3-10× longer than its extracted text, so the ratio is systematically inflated; valid siblings will frequently exceed `max_content_length_ratio=4.0` and be rejected (`xpath_content_ratio_oob`), or invalid ones pass. The comparison is dimensionally inconsistent.
- **Fix:** Compare like-with-like: either store the representative's `dripper_html` length and compare to sibling `main_html` length, or convert the sibling to content first and compare `len(content)` to `representative_content_len`.

### M2 — Stage 2 `dripper_error` for failed/empty prompts can be lost in OUTPUT_COLS spread
- **Where:** `stage2_gpu_inference.py:118-124`.
- **Problem:** The empty/ERROR-prompt branch returns `{**{k: row.get(k,"") for k in OUTPUT_COLS}, "llm_response":"", "dripper_error":..., "inference_time_s":0.0}`. `OUTPUT_COLS` includes `llm_response` and `dripper_error`, so `row.get("llm_response","")` etc. are pulled from the *input* row (which has no such keys → "") and then overwritten — harmless but fragile. More importantly the input row's `simp_html/map_html/html` are preserved here (good), but this dict shape differs from the success/except branches, making the three return shapes easy to drift out of sync.
- **Fix:** Build all three return dicts from one shared helper so columns can't diverge.

### M3 — Stage 2b drops the `prompt` column but Stage 2 also drops `simp_html`/`map_html` correctness depends on passthrough that isn't asserted
- **Where:** `stage1c…OUTPUT_COLS` → `stage2_gpu_inference.py:25-33` → `stage2b_cpu_postprocess.py:51-56`.
- **Problem:** Stage 2b's template build (`:117-121`) needs `typical_raw_tag_html = map_html or simp_html` and `typical_raw_html = raw_html (html)`. These are passed through Stage 2 untouched, but Stage 2's output write (`:169-172`) does `pd.DataFrame(results)` then only back-fills missing `OUTPUT_COLS`; if vLLM rows ever omit `simp_html`/`map_html` (they shouldn't, but the except branch at `:142-148` re-supplies them while the empty-prompt branch at `:118-124` supplies them via the spread) the template build silently produces an empty/degraded template with no error surfaced beyond `map_parser:...`. There is no validation that representatives carry non-empty `map_html`/`html` into 2b.
- **Fix:** In Stage 2b, when `role=="representative"` and `map_html`/`html` are empty, set an explicit `dripper_error="missing_map_html_for_template"` instead of letting map_parser fail opaquely.

### M4 — `_build_gpu_lookup` keeps only the FIRST row per cluster_id; representative ambiguity is silent
- **Where:** `stage3_cpu_propagation.py:681-690`.
- **Problem:** `if cid is not None and str(cid) not in lookup: lookup[str(cid)] = row`. If Stage 2b ever emits more than one row for a cluster_id (e.g. duplicate representative rows from a re-run or a sibling accidentally carrying the cluster_id), the first-seen row wins arbitrarily — no warning. Combined with H2 this can pick the wrong template.
- **Fix:** Prefer the row with `cluster_role=="representative"` and `mapping_json` non-empty; warn if multiple representatives share a cluster_id.

### M5 — Stage 3 representative/singleton rows pull `dripper_error` from `gpu_row.get("error")`, but the column is only renamed conditionally
- **Where:** `stage3_cpu_propagation.py:466-469, 489-494` (`gpu_row.get("error","")`) vs `_load_inference_results:675-676`.
- **Problem:** Stage 2b emits `dripper_error` (not `error`). `_load_inference_results` renames `dripper_error`→`error` **only if `error` not already a column**. That holds for current Stage 2b output, so it works. But it's a brittle coupling: if a future Stage 2b adds both `error` and `dripper_error`, the rename is skipped and `gpu_row.get("error")` reads the wrong column. The `propagation_success` flag (`:327, 343`) derives from this, so a mis-read silently flips success/fallback accounting.
- **Fix:** Normalise to a single canonical error column with an explicit precedence and assert exactly one of `{error, dripper_error}` is present.

### M6 — Stage 4 dashboard reads `metrics_stage*.json` but Stage 3 writes `metrics_shard_NNNN.json` (no `stage` field) — Stage 3 silently missing from dashboard unless the legacy loader catches it
- **Where:** `run_mineru_pipeline.sh:382-410`; `stage3_cpu_propagation.py:1021-1022` writes `metrics_shard_{idx}.json` (not `metrics_stage3_...`), and that dict has no `"stage"` key.
- **Problem:** Stages 1a/1b/1c/2/2b use `StageMetrics.save()` → `metrics_stage{name}_shard_NNNN.json` with a `stage` field. Stage 3 writes its own `metrics_shard_NNNN.json` with **no `stage` key**. The primary glob (`d.glob('metrics_stage*.json')`, line 382) misses it. The legacy fallback (`load_old_metrics`, lines 389-404) globs `metrics_shard_*.json` and injects `stage=stage_name` — so Stage 3 is only rescued by the fallback, and only because `aggregate` keys on the injected name. `pipeline_metrics.aggregate_pipeline_metrics` (used elsewhere, line 128) would silently drop Stage 3 because it `rglob("metrics_stage*.json")` and accesses `r["stage"]`.
- **Fix:** Make Stage 3 write via `StageMetrics.save()` (consistent filename + `stage` field), or at minimum add `"stage": "stage3"` to its metrics dict and rename the file to `metrics_stage3_shard_NNNN.json`.

### M7 — `asyncio.get_event_loop().run_until_complete` in a loop is deprecated and can break on Python ≥3.12
- **Where:** `stage2_gpu_inference.py:156`.
- **Problem:** `asyncio.get_event_loop()` with no running loop is deprecated and, on newer Python, raises `DeprecationWarning`/`RuntimeError` when no current loop exists in the main thread. Repeatedly calling `run_until_complete` per batch on the implicitly-fetched loop is fragile under the vLLM/Ray runtime which may install its own loop policy.
- **Fix:** Create one loop explicitly (`loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)`) before the batch loop, or use `asyncio.run(...)` once over an outer coroutine that iterates batches.

---

## LOW

### L1 — `_load_cluster_manifest_shard` loads `html` for the WHOLE table even though it only keeps siblings
- **Where:** `stage3_cpu_propagation.py:636`.
- **Problem:** The comment (lines 629-635) claims it avoids the full-table html load, but `pq.read_table(path, columns=["url","html"])` reads every row's html into memory before masking non-siblings to `None`. At "30M+ rows × 50-500 KB" this is exactly the OOM the comment says it avoids.
- **Fix:** Use a parquet row-group filter / predicate pushdown on `cluster_role=="sibling"`, or read html in batches and keep only sibling urls.

### L2 — Stage 1b silently treats `feat is None` rows two different ways
- **Where:** `stage1b_gpu_dbscan.py:194-225`.
- **Problem:** Rows with unparseable `dom_feature` are skipped in the clustering loop (`continue`, line 200) AND separately re-added as singletons only when `feat_json` is falsy (line 216). A row with a **non-empty but invalid** JSON `dom_feature` is skipped from clustering (line 199) but NOT re-added as a singleton (line 216 checks `if not feat_json`), so it is **dropped entirely** from the output.
- **Fix:** Make the singleton fallback condition match the clustering skip condition (treat parse failure as a singleton too).

### L3 — Stage 1b `min_cluster_size` default 2 but cluster_size written before dedup
- **Where:** `stage1b_gpu_dbscan.py:131` (`"cluster_size": len(members)`).
- **Problem:** `cluster_size` is the member count from clustering; if Stage 3 later dedups URLs (`drop_duplicates`, line 639) the recorded size can disagree with the actual propagated count. Purely a metric inconsistency.
- **Fix:** Recompute or annotate as pre-dedup size.

### L4 — `compare_f1.load_url_content` last-writer-wins on duplicate URLs
- **Where:** `compare_f1.py:48-51`.
- **Problem:** `out[str(u)] = (...)` overwrites silently on duplicate urls (which Stage 3 explicitly says can occur). The F1 comparison then uses an arbitrary row.
- **Fix:** De-dup deterministically (e.g. prefer non-empty content) and count collisions.

### L5 — Stage 2 `request_id` uses `id(row)` which is not unique across GC cycles
- **Where:** `stage2_gpu_inference.py:127` (`rid = f"...{id(row)}"`).
- **Problem:** `id()` is only unique among *live* objects; within one batch the rows are alive so it's fine, but the pattern is a latent collision risk if reused. Low impact given per-batch scope.
- **Fix:** Use a monotonic counter or `uuid4()`.

### L6 — Dead/contradictory artifacts in Stage 4 inline Python
- **Where:** `run_mineru_pipeline.sh:462-466`.
- **Problem:** The `dfs = [... if 'propagation_method' in ... or True]` list comprehension is dead (the `or True` makes the condition always true and `dfs` is never used; the real read happens in the `frames` loop below). Confusing but harmless.
- **Fix:** Delete the dead `dfs` comprehension.

---

## Verified-correct (no action)

- **Bug #1** Stage 3 `--inference-results '${STAGE2B_OUT}'` — confirmed (`run_mineru_pipeline.sh:323`).
- **Bug #2** Stage 2b content via `parse_result → extract_main_html_single → convert2content`; no `main_html_body` key, no `_sanitize` — confirmed (`stage2b_cpu_postprocess.py:89-111`).
- **Bug #3** Stage 2 `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` before `engine.generate` — confirmed (`stage2_gpu_inference.py:67-89`).
- **Bug #4** Stage 2b serializes template via `base64.b64encode(pickle.dumps(template))`; Stage 3 `_parse_mapping_json` decodes pickle+base64 with dict/bytes/JSON/None fallbacks and preserves tuple keys — confirmed (`stage2b:125`, `stage3:564-600`).
- Stage 3 `_layout_batch_parser_propagate` reads `parts.get("main_html_body")` — this is the **LayoutBatchParser.parse()** output key (distinct from the map_parser template key that was bug #2), so it is correct here.
- Singleton lookup: Stage 1b writes `cluster_id=""` for singletons; Stage 3 `_build_singleton_gpu_lookup` treats `""` as null — consistent.
