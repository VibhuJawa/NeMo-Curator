# End-to-End Throughput & Cost Model — CC-scale MinerU-HTML Pipeline (Track H6)

Definitive throughput/cost model for the 3-stage clustering+propagation pipeline.
Fleet: **40 CPU nodes** (64 workers/node) + **16 GPU nodes** (8×H100 = 128 GPUs).
Two hard targets: **(T1)** overall token-F1 > 0.90 (currently 0.81); **(T2)** GPU
inference (Stage 2) for full CC-MAIN (**2.4B pages**) in **≤2 days** on 16 GPU nodes.

All numbers below are reproducible arithmetic from the measured per-stage rates in
`STAGE2_GPU_PERF_PLAN.md`, `CPU_STAGES_PERF_PLAN.md`, `STAGE3_PERF_AUDIT.md`,
`F1_IMPROVEMENT_PLAN.md`. Window constants: 2 days = **172,800 s**; 1 day = 86,400 s;
efficiency derate **85%** (startup, stragglers, I/O, shard skew).

Measured raw rates used throughout (pages/s/node on the subset each stage touches):
stage1a **595** (100%), stage1c **73**, stage2b **95**, stage3 **77**; stage2 GPU **27**.

---

## 0. TL;DR verdict table

| Scenario (LLM frac) | GPU target rate | **GPU pass @2d?** | **CPU pass @2d (40 nodes)?** | Binding constraint |
|---|---|---|---|---|
| 8.8% | 90 p/s/node | only @≥120 (FAIL @27/62) | **NO** (needs ~109 nodes) | both; CPU=stage3, GPU=serving |
| 14% (recommended F1) | 143 p/s/node | only @143 (FAIL @27/62/120) | **NO** (needs ~67 nodes) | both; CPU=stage3 |
| 20% | 204 p/s/node | **NO at any modeled rate** | **NO** (needs ~134 nodes) | GPU (needs FP8 or +nodes) |

**Headline:** Neither target is met by today's rates. **T2 (GPU)** is reachable for
8.8% and 14% *only after the serving fix lands* (≥120 and 143 p/s/node respectively);
20% needs FP8 on top. **The CPU pipeline is the silent killer**: as sequential SLURM
jobs (sum-of-reciprocals) it needs **~67–109 CPU nodes** for 2 days — 40 is not enough
**unless stages are run as overlapped/streaming work**, in which case stage3 alone at
~250 raw clears 1.2B in 1.2d / 2.4B in 2.4d on 40 nodes. **The single most important
finding: how the CPU stages are scheduled (sequential vs overlapped) matters more than
any micro-opt.**

The **minimal lever set that passes BOTH targets** is in §5.

---

## 1. GPU Stage 2 — wall time for full CC-MAIN (2.4B pages), 16 nodes

LLM runs only on the routed fraction (reps+singletons+fallbacks). Wall time =
`(2.4e9 × frac) / (rate × 16 × 0.85) / 86400` days.

| LLM frac | LLM pages | @27 (today) | @62 (standalone-class) | @120 | @143 | Target rate (85% eff) |
|---|---|---|---|---|---|---|
| **8.8%** | 211 M | 6.66 d ❌ | 2.90 d ❌ | **1.50 d ✅** | 1.26 d ✅ | **90** p/s/node |
| **14%** | 336 M | 10.59 d ❌ | 4.61 d ❌ | 2.38 d ❌ | **2.00 d ✅** | **143** p/s/node |
| **20%** | 480 M | 15.13 d ❌ | 6.59 d ❌ | 3.40 d ❌ | 2.86 d ❌ | **204** p/s/node |

**Which rate clears 2 days:**
- 8.8% → need **≥90 p/s/node** (raw floor 76). 120 and 143 both clear; 62 does **not**.
- 14% → need **≥143 p/s/node** (raw floor 122). Only 143 clears.
- 20% → need **≥204 p/s/node** (raw floor 174). **No modeled rate clears** — requires FP8 (§5).

So **62 p/s/node (matching the standalone) is NOT enough for any scenario.** The serving
fix must reach 120+ (8.8%) or 143+ (14%). Per `STAGE2_GPU_PERF_PLAN.md`, levers 1
(dynamic max_tokens + item_count) + 3 (continuous-batching dispatch) + 4–5
(max_num_seqs/CUDA-graphs/gpu_mem 0.90) project **55–120 p/s/node** in bf16; FP8 (lever 6)
adds 1.2–1.3× → **~150–156**, which clears the 14% target.

---

## 2. End-to-end CPU pipeline — 40 nodes

CPU stages run as **sequential SLURM jobs** (1a → [1b GPU] → 1c → [2 GPU] → 2b → 3), so
per-corpus-page CPU wall = **sum of reciprocals** of each stage's *effective whole-corpus*
rate (`eff = raw / subset_fraction`). `T_cpu = 1 / Σ(1/eff_s)`.

### Baseline rates, three LLM fractions

| LLM frac | eff 1a | eff 1c | eff 2b | eff 3 | **T_cpu (eff/node)** | budget shares (1a/1c/2b/3) | 40-node agg | **2.4B wall** | **1.2B wall** |
|---|---|---|---|---|---|---|---|---|---|
| 8.8% | 595 | 830 | 1080 | 84 | **64** | 11/8/6/**76%** | 2,555/s | **10.9 d** | **5.4 d** |
| 14% | 595 | 521 | 679 | 90 | **62** | 10/12/9/**69%** | 2,463/s | **11.3 d** | **5.6 d** |
| 20% | 595 | 365 | 475 | 96 | **59** | 10/16/12/**61%** | 2,365/s | **11.7 d** | **5.9 d** |

Required CPU eff/node for 2 days: **347** (2.4B) / **174** (1.2B). Baseline is 59–64 →
**5.4–11.7 days. Sequential CPU does NOT meet 2 days at any LLM fraction.**

### With CPU optimizations (from CPU plan + stage3 audit)

stage3 is **75% of the CPU budget**; it is the only lever that moves the needle.
Stage3 audit projects raw **150–250** p/s/node on the sibling subset with XPath
fast-path (#1) + template reuse (#2) + page-level balancing (#3). Pairing with S/M
opts on 1a/1c/2b (batch ProcessPool tasks, drop raw-HTML echo, DOM reuse):

| Scenario (14% LLM) | stage3 raw | 1a/1c/2b raw | **T_cpu** | 2.4B | 1.2B |
|---|---|---|---|---|---|
| mid-opt | 150 | 850/88/130 | **104** | 6.7 d | **3.3 d** |
| high-opt | 250 | 900/95/140 | **142** | 4.9 d | **2.4 d** |

Even fully optimized sequential CPU = **142 eff/node → 2.4 d for 1.2B, 4.9 d for 2.4B
on 40 nodes. Still misses 2-day for 2.4B; misses 1.2B by 0.4 d.**

### CPU nodes actually required (sequential, 2-day window)

| T_cpu | 1.2B → nodes | 2.4B → nodes |
|---|---|---|
| 64 (baseline) | **109** | 217 |
| 104 (mid) | 67 | 134 |
| 142 (high) | 49 | 98 |

**40 nodes is short by 1.2–5× for the sequential CPU model.** This is the dominant,
under-appreciated risk — the GPU debate is moot if CPU takes 5 days.

### The decisive reframe — overlapped/streaming execution

The sum-of-reciprocals assumes each stage drains the *whole corpus* before the next
starts. If instead the pipeline streams in segments (stage N+1 starts on segment K while
stage N works segment K+1), CPU wall is governed by the **single slowest stage**
(max reciprocal = stage3), not the sum. Then on 40 nodes:

| stage3 raw | eff (86% siblings) | 1.2B wall | 2.4B wall |
|---|---|---|---|
| 150 | 174 | **2.0 d** | 4.0 d |
| **250** | **291** | **1.2 d** | **2.4 d** |

**Overlapped + stage3 raw 250 → 1.2B in 1.2 d and 2.4B in 2.4 d on 40 nodes.**
This is the only way 40 CPU nodes clears (or nearly clears) 2 days. **Recommendation: run
the CPU stages as an overlapped segment pipeline, not as four full-corpus barriers.**

---

## 3. Binding constraint per scenario

| Scenario | CPU (40n) | GPU (16n) | **Binding** |
|---|---|---|---|
| 8.8%, today | 5.4 d (seq) / stage3 | 6.66 d @27 | **GPU** (serving), CPU close 2nd |
| 8.8%, serving fixed @120 | 5.4 d seq / 2.0–4.0 d overlap | 1.50 d ✅ | **CPU** (stage3 / scheduling) |
| 14%, today | 5.6 d / stage3 | 10.59 d @27 | **GPU** |
| 14%, serving @143 + CPU opt overlap | 1.2–2.4 d | 2.00 d ✅ | balanced (stage3 ≈ GPU) |
| 20%, full stack | 5.9 d / stage3 | 2.86 d @143 | **GPU** (needs FP8) |

In every "today" column the **GPU serving architecture is the binding constraint**
(27 vs 62 standalone = the 2.3× serving/batching gap). Once serving is fixed, the
**CPU pipeline — specifically stage3 and whether stages overlap — becomes binding.**
stage1a (the only other 100%-of-pages stage, 595 eff) is the next ceiling after stage3.
stage1c/2b only matter at 20% LLM (they jump to ~29% of the CPU budget).

---

## 4. Other agents' levers (inputs to the minimal set)

| Lever | Owner track | Effect | Cost/risk |
|---|---|---|---|
| Serving fix (dynamic max_tokens + continuous batching + concurrency/CUDA-graph) | Stage2 GPU | 27 → 55–120 p/s/node | M, no F1 risk |
| FP8 weights + fp8 KV | Stage2 GPU | ×1.2–1.3 on top → ~150–156 | L, low-med F1 (verify parity) |
| Reduced LLM fraction (validation gate, Lever 2) | F1 | 19.3% → 14% routed | M, no F1 loss |
| Stage3 reuse/XPath fast-path (#1+#2+#3) | Stage3 | 77 → 150–250 raw | M, med F1 (gate on compare_f1≥0.99) |
| CPU micro-opts (batch ProcessPool, drop html echo, DOM reuse) | CPU | 1a ×1.3–1.6, 2b ×1.4 | S–M, no/low F1 |
| Overlapped segment scheduling | orchestration | sum → max reciprocal | S (submit-script), no F1 |

F1 lever choice fixes the LLM fraction that *both* the GPU and CPU models consume:
**14%** (Lever 1+2 in `F1_IMPROVEMENT_PLAN.md`) gives F1 ≈ 0.913 > 0.90 at half the GPU
cost of routing all fallbacks (19.3%). 8.8% does **not** clear F1 (it omits the fallback
routing → stays ~0.81). So **T1 forces LLM frac ≥ ~14%**, which in turn sets the GPU bar
at **143 p/s/node** and makes 20% unnecessary.

---

## 5. Minimal lever set that passes BOTH targets — with arithmetic

**Operating point: LLM fraction = 14%** (the F1-minimal choice that clears T1).

### T1 (F1 > 0.90) — minimal set
- **F1 Lever 2** (template validation + max_selected_item_ratio gate): fallback rate 11.7% → ~6%, free at inference.
- **F1 Lever 1** (Stage 3.5 fallback→LLM re-inference): routes the residual ~6% fallbacks + reps + singletons = **14% corpus** through the LLM.
- Result: sibling F1 0.913, **overall F1 ≈ 0.913 > 0.90 ✅** (computed in `F1_IMPROVEMENT_PLAN.md`).
Effort: M. F1 risk: none (matches standalone path). **This sets LLM frac = 14% for the throughput models below.**

### T2 (GPU ≤2 d @14% on 16 nodes) — minimal set
Need **143 p/s/node** (raw floor 122). Today 27.
- **Serving fix** (dynamic max_tokens + item_count column + continuous-batching dispatch + max_num_seqs=256 + gpu_mem 0.90 + CUDA graphs): projected **55–120 p/s/node** bf16. Midpoint ~90; optimistic 120.
- **FP8 weights + fp8 KV** (×1.25): 90→**112** (miss) … 120→**150 ✅**.
- Arithmetic: 336M / (143 × 16 × 0.85 × 86400) = **2.00 d ✅** exactly at 143; at 150 = 1.90 d.
**Verdict:** serving fix alone is *borderline* (must land at the top of its range, ~120);
**serving fix + FP8 is required to comfortably clear 143.** Effort: M (serving) + L (FP8).
Hedge if FP8 F1 fails parity: **18–20 GPU nodes** instead of 16 (336M /(120×18×0.85×86400)=2.13 d → 20 nodes = 1.92 d ✅).

### CPU pipeline ≤2 d — minimal set (the binding piece nobody else owns)
40 nodes, 14% LLM. Sequential is 5.6 d (baseline) → 4.9 d (fully optimized). **Sequential
cannot clear 2 d on 40 nodes for 2.4B.** Two routes:

1. **Overlapped segment scheduling + stage3 raw ≥250** (XPath fast-path #1+#2+#3): wall
   governed by stage3 → eff 291 → **2.4 B in 2.4 d, 1.2B in 1.2 d ✅ on 40 nodes.**
   (2.4B misses 2-day by 0.4 d — acceptable, or do 1.2B/half-corpus runs which pass.)
2. **If staying sequential:** need stage3 raw 250 **and** add CPU nodes to ~50 (1.2B) /
   ~98 (2.4B), which exceeds the 40 available → not viable. **Overlap is mandatory.**

CPU micro-opts (batch ProcessPool, drop raw-html echo) are **required** so stage1a (595)
and 1c/2b don't become the new ceiling once stage3 is fast — but they only buy ~3% on
their own; their job is to stay out of the way.

### Minimal combined recipe (PASS BOTH)

| # | Lever | Track | Why required |
|---|---|---|---|
| 1 | F1 validation gate + Stage 3.5 fallback→LLM | F1 | T1 (0.913>0.90); fixes LLM frac=14% |
| 2 | GPU serving fix (dyn max_tokens + continuous batch + concurrency/CUDA-graph) | Stage2 | 27→~120; necessary, not sufficient for 143 |
| 3 | GPU FP8 (verify F1 parity) **or** scale to 18–20 GPU nodes | Stage2 | closes 120→143+ for T2 @14% |
| 4 | Stage3 XPath fast-path #1+#2+#3 (raw→250) | Stage3 | makes CPU stage3 fast enough |
| 5 | Overlapped segment scheduling of CPU stages | orchestration | turns sum-of-reciprocals into max → 40 nodes clears |
| 6 | CPU micro-opts on 1a/1c/2b (S-effort) | CPU | keep stage1a from becoming the new ceiling |

**Net result with the recipe (14% LLM):**
- **F1 ≈ 0.913 ✅ (T1)**
- **GPU: 2.00 d @143 (serving+FP8) on 16 nodes — clears 2.4B ✅ (T2)** (hedge: 20 nodes if FP8 fails parity)
- **CPU: 2.4B in 2.4 d (overlapped, stage3 raw 250) on 40 nodes** — clears 1.2B in 1.2 d; for full 2.4B in exactly 2 d add ~6 CPU nodes or accept 2.4 d.

20% LLM is **not recommended**: it raises the GPU bar to 204 (unreachable at 16 nodes even
with FP8) and buys no F1 over 14%. Stay at 14%.

---

## 6. Sensitivity / risk notes
- **GPU serving fix landing low (~55–70):** T2 fails at 14% even with FP8 → must drop to
  8.8% LLM (but then T1 fails) or scale to 28–32 GPU nodes. The serving fix is the
  highest-leverage single item; it must reach ≥120 bf16.
- **Stage3 XPath F1 gate fails (<0.99 vs LBP):** stage3 stays ~77–150, CPU 2.4B → 3.3–4 d
  even overlapped → add CPU nodes or run half-corpus.
- **Sequential-only scheduling (no overlap):** CPU needs 49–109 nodes; 40 is insufficient
  at every LLM fraction. Overlap is the cheapest single CPU win (submit-script change, no
  F1 risk) and is **mandatory** for the 40-node constraint.
- **FP8 F1 parity:** lever 3's FP8 path carries low-med F1 risk; the 18–20-node fallback
  removes that risk for ~25% more GPU allocation.
