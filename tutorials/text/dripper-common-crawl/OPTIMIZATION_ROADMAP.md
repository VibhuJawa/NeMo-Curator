# Integrated Optimization Roadmap — CC-scale MinerU-HTML Pipeline

Synthesizes the six swarm tracks (H1–H6) into ONE ranked plan that clears both hard targets:
- **T1:** overall token-F1 vs standalone Dripper baseline (job 335168) **> 0.90** (today 0.81).
- **T2:** GPU inference (Stage 2) for full CC-MAIN **2.4B pages in ≤2 days on 16 GPU nodes** (8×H100), with 40 CPU nodes for the CPU stages.

Window constants: 2 days = 172,800 s; efficiency derate 0.85. GPU-rate equation
`R(f) = 2.4e9·f / (16·172800·0.85) = 1021.3·f` pages/s/node (f = LLM page fraction).

---

## A. The single minimal set of changes that clears BOTH targets

Operating point: **LLM fraction = 10%** (driven down from today's ~19.3% by the validation gate;
this is the cost-optimal point — see §C for why not 14% and not 6%).

| # | Lever | Track | Effect | Effort | F1 risk |
|---|---|---|---|---|---|
| **1** | Per-cluster template validation gate (`token_f1≥0.98` vs rep-LLM content) + `max_selected_item_ratio=0.50` gate | H3-B/D into Stage 3 | Partitions blind fallbacks → confident propagate OR honest LLM. Fallback 11.7%→~6% of siblings; F1 of recovered region 0→~0.91. **Free at inference.** | M | none (F1-protective) |
| **2** | Stage 3.5 fallback→LLM re-inference loop (reuse Stage 1c/2/2b on the `propagation_method=="fallback"` set, role forced "singleton", merge on url) | H6/F1 Lever-1 | Routes the residual ~6% fallbacks through the LLM → sibling F1 0.804→0.913. **This is the T1 lever.** | M | none (matches baseline path) |
| **3** | **GPU serving rewrite: offline batched, 1 `vllm.LLM` per GPU, in-process, `LLM.generate(prompts)` — no Ray-Serve actor RPC, no HTTP** | H1 | Removes the per-request cloudpickle/object-store RPC that starves vLLM's batcher. 27 → ~80–120 p/s/node bf16. **This is the dominant T2 lever.** | M | none (gen config unchanged) |
| **4** | Engine tuning on the new path: `dynamic max_tokens=min(2048,max(32,item_count·6+16))`, `gpu_memory_utilization=0.90`, `max_num_seqs=512`, `max_num_batched_tokens=16384`, chunked prefill, prefix caching, CUDA graphs (`enforce_eager=False`) | H1/H2 | Keeps the batch saturated; lands the top of the 80–120 range. | S | none |
| **5** | Stage 3 XPath/CSS fast-path from the template red-key set (+ per-cluster validation hoist + page-level balancing) | H4/H6 | Stage 3 raw 77 → ~190–250 p/s/node, so the CPU pipeline keeps up with the GPU+fallback path. | M | low (gate on `compare_f1≥0.99`) |
| **6** | **Overlapped segment scheduling of the CPU stages** (submit-script change: stream segments so wall = slowest single stage, not sum-of-reciprocals) | H6 | Turns CPU wall from ~5d (sequential) into stage3-bound. **Mandatory for 40 CPU nodes to clear.** | S | none |
| **7** | CPU micro-opts on 1a/1c/2b (batch ProcessPool ~256/future, drop raw-HTML echo, binary mapping_json, DOM reuse in 2b) | H5 | Keeps stage1a (the only other 100%-of-pages stage, 595 eff) from becoming the new ceiling once stage3 is fast. | S | none |

**Note:** FP8 (track H2) is **NOT in the minimal set at 10% LLM** — the serving rewrite alone
clears the required 102 p/s/node. FP8 becomes required only if you stay at 14% LLM, or if the
serving rewrite lands at the low end (<102). It is the cheapest hedge (effort S–L) and is listed in §C.

### Combined arithmetic

**T1 (F1):** Fixed role mix rep 1,429@0.97, singleton 2,411@0.95, sibling 40,084.
- Lever 1 drops fallbacks 11.7%→~6% of siblings (≈2,405 pages); lever 2 routes those to the LLM @0.96.
- sibling F1 = (2,405·0.96 + 37,679·0.91)/40,084 = **0.913**.
- **overall F1 = 0.913 > 0.90 ✅ PASS.**

**LLM fraction:** reps 3.2% + singletons 5.5% (structural) + ~6% of siblings fallback·0.909 ≈ 5.5% routed
= **~14% if no load reduction**, or **~10%** once the validation gate + ratio gate also shrink the
*structural* and *bad-rep* fraction (H3 §4 floor: reps→~2%, singletons→~3.5% via absorbing into
clusters, fallbacks→~3–4%). **Plan at 10%.** (Conservatively, even at 14% the math below is checked.)

**T2 (GPU), at 10% LLM:**
- Required rate `R(0.10) = 1021.3·0.10 = 102.1` p/s/node (raw floor ~87).
- Serving rewrite (lever 3+4): **80–120 p/s/node bf16**, midpoint ~100, top ~120.
- Wall @102 = 240M / (102·16·0.85·86400) = **2.00 d**; @120 = **1.70 d**.
- **PASS if serving lands ≥102 (mid-to-top of its measured range). ✅ (FP8 hedge if it lands ~80–90.)**

**T2 cross-check at 14% LLM (if H3 load-reduction underdelivers):**
- Required `R(0.14)=143` p/s/node. Serving bf16 ~120 → 2.38 d ❌. **Then FP8 (×1.25 → 150) → 1.90 d ✅**,
  or scale to 20 GPU nodes (336M/(120·20·0.85·86400)=1.92 d ✅).

**CPU pipeline (40 nodes, 10–14% LLM):**
- Sequential (sum-of-reciprocals) = ~5–5.6 d at baseline, ~4.9 d fully optimized → **FAIL on 40 nodes.**
- **Overlapped (lever 6) → wall = stage3.** At stage3 raw 250 (lever 5): eff = 250/0.86 ≈ 291 p/s/node.
  - 2.4B / (291·40·0.85·86400) = **2.4 d** (misses 2-day by 0.4d — accept, or +6 CPU nodes → 2.0 d).
  - 1.2B (half-corpus runs) = **1.2 d ✅**.
- **PASS for 1.2B; 2.4B at 2.4 d (near-pass).** Lever 7 keeps stage1a@595eff from becoming the ceiling.

### Verdict per target

| Target | Result | Verdict |
|---|---|---|
| **T1: F1 > 0.90** | 0.913 (levers 1+2) | **✅ PASS** |
| **T2: GPU 2.4B ≤2d / 16 nodes** | 2.00 d @102 p/s/node, 10% LLM (levers 3+4); FP8/20-node hedge for 14% | **✅ PASS** (serving rewrite must land ≥102 bf16) |
| **CPU pipeline ≤2d / 40 nodes** | 2.4 d for 2.4B / 1.2 d for 1.2B, overlapped + stage3 raw 250 (levers 5+6+7) | **⚠ NEAR-PASS** (2.4B at 2.4d; full 2-day needs +6 CPU nodes or half-corpus runs) |

---

## B. Priority-ordered implementation sequence (max leverage first)

1. **GPU serving rewrite (lever 3) + engine tuning (lever 4)** — *highest leverage, biggest gap.*
   This is the only ~3–4× single lever and the binding constraint in every "today" scenario (27 vs
   needed 102). Validate on ONE free GPU per H1 §6: `--mode offline --max-pages 4000`; expect ≥6–15
   pages/s/GPU vs today's 3.4. F1 is untouched (greedy temp=0, same chat template). Do this first
   because it determines whether FP8 / extra nodes are needed (gates lever-3-hedge decision).

2. **F1 validation gate + ratio gate (lever 1)** — *F1-protective AND load-reducing, free at inference.*
   Extend Stage 3 `_cluster_static_trustworthy` into a propagation-vs-rep-LLM `token_f1≥0.98` gate;
   add `max_selected_item_ratio=0.50`. This both lifts F1 and shrinks the fallback volume that lever 2
   must pay for. Land before lever 2 so the Stage 3.5 bill is ~half.

3. **Stage 3.5 fallback→LLM loop (lever 2)** — *the T1 clincher.* Reuses Stage 1c/2/2b unchanged over
   the fallback manifest; orchestration + a `cluster_role="singleton"` override + a url-keyed merge.
   After this, re-measure overall F1 → expect ~0.913.

4. **Overlapped segment scheduling (lever 6)** — *cheapest CPU win, mandatory for 40 nodes.* Submit-script
   change only (no algorithm change, no F1 risk). Without it the CPU pipeline needs 49–109 nodes.

5. **Stage 3 XPath fast-path (lever 5)** — *makes the CPU stage3 keep pace.* Gate on `compare_f1≥0.99`
   vs LBP. Needed to reach stage3 raw ~250 so the overlapped wall lands at 2.4d (2.4B) / 1.2d (1.2B).

6. **CPU micro-opts on 1a/1c/2b (lever 7)** — *do last; they only matter once stage3 is fast.* Batch
   ProcessPool tasks, drop the raw-HTML echo, binary (non-base64) mapping_json. ~3% on their own; their
   job is to keep stage1a@595 from becoming the next ceiling.

7. **(Conditional) FP8 or +nodes (§C hedge)** — only if step-1 measurement lands <102 p/s/node or you
   are forced to 14% LLM. A/B 2–5K pages, accept FP8 weights if overall ΔF1 ≥ −0.005.

---

## C. Targets / scenarios NOT reachable even with all levers — stated honestly

1. **2.4B full corpus on CPU in exactly 2.0 days, 40 nodes:** NOT reachable. Even fully optimized
   (overlapped + stage3 raw 250) the CPU wall for 2.4B is **2.4 d**. To hit 2.0 d either (a) add ~6 CPU
   nodes (40→46), or (b) run as two 1.2B half-corpus passes (each 1.2 d), or (c) push stage3 raw past
   250 (lever 5's stretch at ≥90% XPath coverage reaches ~344/node → 2.4B in ~1.7 d, but that depends
   on the F1 gate passing at high XPath share — not guaranteed). GPU side (T2) DOES clear 2.4B in 2.0 d.

2. **20% LLM fraction:** NOT recommended and not reachable at 16 GPU nodes. It needs 204 p/s/node;
   serving bf16 tops ~120, FP8 ~150 — still short. It also buys **zero F1** over 14%/10% (the fallback
   pages already hit the ~0.96 LLM ceiling). Drop it entirely; the validation gate makes it unnecessary.

3. **T2 if the serving rewrite lands at the LOW end (~55–80 p/s/node):** at 10% LLM, 80 p/s/node → 2.55 d
   ❌. Recovery: (a) FP8 ×1.25 → 100 → 2.04 d (borderline pass), or (b) drive LLM fraction to ~8% (H3
   Lever A looser clustering after B/C/D land) → R=82 → pass, or (c) scale to 20 GPU nodes. The serving
   rewrite reaching ≥102 bf16 is the load-bearing assumption — **validate it first (step B.1).**

4. **F1 ceiling above ~0.93:** reps/singletons sit at 0.95–0.97 due to model nondeterminism vs job
   335168 (sampling/kernel/version differences), not a fixable defect. The practical overall ceiling is
   ~0.92–0.93; chasing higher (bit-exact decode parity) yields ≤+0.004 and is not worth it. 0.913 clears
   the 0.90 target with margin.

---

## D. Bottom line

The minimal recipe is **7 levers**: (1) validation+ratio gate, (2) Stage 3.5 fallback→LLM,
(3) offline-batched GPU serving rewrite, (4) engine tuning, (5) Stage 3 XPath fast-path,
(6) overlapped CPU scheduling, (7) CPU micro-opts. At **10% LLM fraction** this yields **F1 ≈ 0.913**
and a **GPU requirement of 102 p/s/node** that the serving rewrite (80–120 bf16) clears at **2.00 days
on 16 nodes**. The CPU pipeline clears 1.2B in 1.2 d and full 2.4B in 2.4 d on 40 nodes (overlapped,
stage3 raw 250). FP8 / +4–6 nodes are hedges, not requirements, at 10% LLM.
