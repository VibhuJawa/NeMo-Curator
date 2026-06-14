# Reduce LLM Load Plan — Track H3

**Goal:** hit the GPU 2-day target by *shrinking the LLM page fraction*, not just speeding inference.
The LLM serving speedup (Track H2) and this track are multiplicative: lowering the LLM fraction
relaxes the required pages/s/node by the same ratio. This doc quantifies the LLM-fraction levers in
`stage1b_gpu_dbscan.py` and `stage3_cpu_propagation.py` (vs the standalone
`nemo_curator/.../dripper/stage.py`), gives the floor, and the resulting throughput relaxation.

Analysis/design only. No production stage scripts are edited.

---

## 1. The throughput equation — what 1% of LLM fraction is worth

Required per-node inference rate to finish the full CC-MAIN LLM pass in 2 days on 16 GPU nodes:

```
R(f) = (2.4e9 * f) / (16 nodes * 172800 s * 0.85 eff)
     = (2.4e9 * f) / 2.350e9
     = 1021.3 * f      pages/s/node     (f = LLM page fraction, 0..1)
```

So **each 1 percentage point of LLM fraction costs ~10.2 pages/s/node** of required throughput.

| LLM fraction f | pages routed to LLM | required pages/s/node | vs current 27 |
|---|---|---|---|
| 20.0% (pre-validation, today's worst case) | 480M | 204.3 | 7.6x gap |
| 14.0% (post-validation, current plan)      | 336M | 143.0 | 5.3x gap |
| 10.0%                                      | 240M | 102.1 | 3.8x gap |
| 8.8%  (reps+singletons only, NO fallback)  | 211M | 89.9  | 3.3x gap |
| 6.0%                                       | 144M | 61.3  | 2.3x gap |
| 4.0%                                       | 96M  | 40.9  | 1.5x gap |

**Reading:** the current plan (14% LLM) needs 143 pages/s/node — a 5.3x serving speedup.
If H3 drives the LLM fraction to **6%**, the requirement drops to **61 pages/s/node** — which is
already roughly the standalone baseline's measured ~62 pages/s. In other words, **at 6% LLM fraction
the 2-day target is reachable with the serving architecture that already exists** (the standalone
LLMServer), with no exotic inference speedup required. That is the strategic prize of this track.

---

## 2. Decomposing today's LLM fraction (44,117-page smoke)

| Role | Pages | Share | Sent to LLM? |
|---|---|---|---|
| representative | 1,429 | 3.2% | yes (template source) |
| singleton      | 2,411 | 5.5% | yes (one-off) |
| sibling        | 40,084 | 90.9% | only on fallback |
| **reps+singletons (unavoidable LLM floor today)** | **3,840** | **8.7%** | yes |
| sibling fallbacks (~11.7% of siblings) | ~4,690 | ~10.6% | yes (Stage 3.5) |
| **total LLM with full fallback routing** | **~8,530** | **~19.3%** | |

So today's LLM fraction is **8.7% structural + 10.6% fallback = ~19.3% pre-validation**, which the
current plan shrinks to **~14%** by reducing fallbacks to ~6% of siblings. H3's job is to push both
terms down further. Note the structural 8.7% and the fallback 10.6% have **different levers**:

- The **8.7% structural** floor is set by *cluster count* (one rep per cluster) + singleton count.
  Lowered by **bigger/more clusters** (Lever A) and **fewer singletons**.
- The **10.6% fallback** is set by *propagation failure rate*. Lowered by **validation gating +
  multi-rep + ratio gate** (Levers B, C, D) so more siblings propagate instead of falling back.

Mean cluster size today = (1,429 reps + ~37,673 clustered siblings) / 1,429 reps ≈ **29 pages/cluster**
(the 90.9% siblings are not all clustered; some are the fallback set). The 1,429 reps over 41,513
clustered pages gives the structural rep cost: **reps = clustered_pages / mean_cluster_size**.

---

## 3. The levers, quantified

### Lever A — Clustering threshold (structural fraction)

`stage1b_gpu_dbscan.py:303` `--threshold 0.95` (DBSCAN cosine on DOM features). This is a two-edged knob:

- **Looser threshold (e.g. 0.92):** merges more pages into each cluster → **fewer clusters → fewer
  reps → lower structural %**, and fewer singletons (pages that currently fail the min-cluster-size=2
  test get absorbed). BUT siblings are now less structurally identical to the rep → **higher
  propagation-failure rate → bigger fallback set**. Net LLM fraction can go *either way*.
- **Tighter threshold (e.g. 0.97):** purer clusters → propagation succeeds more (smaller fallback) but
  **more, smaller clusters → more reps + more singletons → higher structural %**.

Arithmetic for the structural term as a function of mean cluster size `m` (clustered pages ≈ 41,513):
`reps = 41,513 / m`. Today m≈29 → 1,429 reps (3.2%). If looser clustering raises m to 50 →
**830 reps (1.9%)**, saving ~1.3 pts. To 100 → **415 reps (0.9%)**, saving ~2.3 pts. The structural
saving from looser clustering is **bounded (~2 pts max)** because reps are already only 3.2%.

The singleton term (5.5%) is the larger structural prize: a looser threshold that pulls even half the
singletons into clusters saves ~2.7 pts directly. **But** this only helps net LLM fraction if those
newly-absorbed pages then *propagate* (don't just become fallbacks). Whether they do depends entirely
on Lever B/C/D quality gating. **Lever A is not a standalone win — its value is conditional on the
propagation quality machinery being in place.**

**Recommendation:** keep threshold at **0.95** (the baseline-validated value), and *measure a small
sweep 0.92/0.95/0.97* offline against propagation success before changing it. Do not loosen
clustering until Levers B/C/D are landed, or the fallback set will grow faster than the structural
saving. **F1 risk: medium if loosened without quality gates (more wrong-region propagation); none at
0.95.**

### Lever B — Per-cluster template validation gate (the cheap, high-value lever)

The standalone (`stage.py:2759-2829`) runs BOTH propagation and the LLM on a few sibling
"validation rows" per cluster, and requires `token_f1(propagated, llm) >= 0.98`
(`layout_template_validation_min_content_f1`). **A cluster that passes validation is trusted: ALL its
remaining siblings propagate with zero LLM cost and high confidence.** A cluster that fails is routed
to the LLM wholesale — protecting F1.

Our Stage 3 already has the *machinery* for this — `_cluster_static_trustworthy`
(`stage3_cpu_propagation.py:368-401`) runs static-vs-dynamic LBP on K=3 sample siblings — but it only
decides the fast-path (static vs dynamic), **not** whether the template is good enough to trust vs
route to LLM. There is no propagation-vs-LLM validation. Porting the standalone gate means:

- For each cluster, on K validation siblings compute `token_f1(propagated, rep_llm_content)`. If
  `>= 0.98`, mark the cluster `template_trusted=True`; **all siblings propagate, none fall back.**
- If `< 0.98`, mark the cluster untrusted → its siblings go to the Stage 3.5 LLM pass.

**Effect on LLM fraction:** the validation gate does not by itself reduce LLM calls — it *correctly
partitions* siblings into "safe to propagate" vs "must LLM". Its value is that it lets you safely use
**looser clustering (Lever A)** and **trust large clusters** without growing the F1==0 fallback set.
It converts blind fallbacks (F1==0) into either confident propagation (F1≈0.91) or honest LLM
(F1≈0.96). Combined with the current Stage 3.5 routing, it is what pulls the fallback term from 11.7%
→ ~6% (per `F1_IMPROVEMENT_PLAN.md` §6) — i.e. it removes ~5 pts of *fallback* LLM load while keeping
F1 ≥ 0.90.

**F1 risk: none** — it is strictly F1-protective (it is exactly the baseline's mechanism). Effort: M
(K extra propagation+LLM calls per cluster on validation rows; the LLM calls are the rep result we
already have for K=cluster's rep, so the marginal LLM cost is ~0 if validated against the existing rep
content rather than fresh inference).

### Lever C — Multiple representatives per cluster (reduces fallback, small structural cost)

The standalone tries up to `layout_template_representative_candidates` reps
(`stage.py:2939-2955, 2681-2697`): it infers candidate reps in order and **uses the first one whose
mapping/template succeeds**. A cluster only fails (→ all siblings to LLM) if *every* candidate rep
fails to produce a valid template. Our Stage 1b picks exactly **one** rep
(`stage1b_gpu_dbscan.py:114-120`); if that rep's template is unusable, the whole cluster's siblings
fall back.

**Effect:** suppose a single rep yields a usable template with probability `p` per cluster. With `c`
candidate reps the cluster-level template-failure probability drops from `(1-p)` to `(1-p)^c`. If
~11.7% of clusters currently produce templates that fail on their siblings and that is dominated by a
*bad rep choice* (rather than a genuinely heterogeneous cluster), then going from 1→2 reps could cut
the *rep-driven* portion of fallbacks roughly in half. Concretely, if half of the 10.6% fallback load
is "bad rep, good cluster," 2 reps removes ~2.5 pts of fallback; 3 reps ~3.5 pts.

**Structural cost:** extra reps are extra LLM calls. With `c` candidates tried but only failures
re-tried, the *expected* extra rep inferences ≈ `(c-1) * (fraction of clusters needing a 2nd rep)`.
If 1,429 clusters and ~12% need a 2nd rep: +~170 LLM pages = **+0.4 pts**. Net: spend ~0.4 pts of
structural LLM to remove ~2.5 pts of fallback LLM → **net ~-2 pts LLM fraction.** Good trade.

**F1 risk: low** (more clusters get a working template; the gate in Lever B still protects against a
bad-but-passing template). Effort: M — Stage 1b would emit 2-3 candidate rep urls per cluster; Stage 2
infers them; Stage 3 picks the first whose template validates. This is a real cross-stage change.

### Lever D — `max_selected_item_ratio` gate (reject grab-everything templates)

Standalone `stage.py:3111-3117` rejects a template that selected > 50% of the page
(`layout_template_max_selected_item_ratio=0.50`) — a degenerate "grab everything" template that would
emit garbage. Our pipeline has `representative_content_len` plumbed (`stage3:647`) but does not gate
on it. Adding this catches a slice of the **non-fallback F1==0** pages (Lever 3 in F1 plan, ~7.4% of
siblings) that propagate *something wrong*. **Effect on LLM fraction:** small (routes a few % of
templates to LLM) but **F1-protective**; effort S; **F1 risk: none.**

---

## 4. Realistic LLM-fraction floor

| Term | Today | With H3 levers | Floor mechanism |
|---|---|---|---|
| Reps (structural) | 3.2% | ~2.0% | Lever A looser threshold raises mean cluster size (bounded) |
| Singletons | 5.5% | ~3.5% | Lever A absorbs ~⅓ of singletons into clusters (only safe with Lever B) |
| + multi-rep extra | 0% | +0.4% | Lever C 2nd-rep inferences |
| Sibling fallbacks | 10.6% | ~3-4% | Lever B validation + Lever C multi-rep + Lever D ratio gate |
| **Total LLM fraction** | **~19.3%** | **~9-10%** | |

**Realistic floor: ~9-10% LLM fraction** (vs ~14% in the current plan, ~19% pre-validation). Pushing
below ~9% is hard because reps+singletons are an irreducible structural floor (~5.5-6%) — every
distinct layout *must* be seen by the LLM once, and the long tail of one-off pages (singletons) is
genuine. The fallback term has a soft floor of ~3% (genuinely heterogeneous clusters + baseline-empty
pages that can never validate).

**Aggressive-but-credible target: 10% LLM fraction.**

---

## 5. Resulting throughput-target relaxation

| Plan | LLM fraction | required pages/s/node | serving speedup needed vs 27 |
|---|---|---|---|
| Current plan (F1 doc §6) | 14% | 143 | 5.3x |
| **H3 levers B+C+D (validation+multi-rep+ratio)** | **~10%** | **102** | **3.8x** |
| H3 + looser clustering (A, if it pays off) | ~9% | 92 | 3.4x |
| Stretch (everything lands) | 6% | 61 | 2.3x = standalone baseline rate |

**Bottom line:** H3 alone takes the requirement from **143 → ~102 pages/s/node** (a 1.4x relaxation
of the H2 serving target) at **zero F1 cost** (Levers B and D are strictly F1-protective; Lever C is
low-risk and net-reduces LLM load). If looser clustering (A) also pays off after the offline sweep,
the requirement drops toward ~90. The combined H2 (serving) + H3 (load reduction) attack is
multiplicative: H2 getting to ~62 pages/s (matching the standalone) at H3's 10% fraction would
**already meet the 2-day target with ~40% headroom** (62 vs 102 needed... not quite — see note).

> Note on whether this alone hits 2-day: at 10% fraction we need 102 pages/s/node and currently have
> 27, so H3 alone does **not** reach the target — it relaxes it from 5.3x to 3.8x. The target is hit
> only by **H3 (this track) × H2 (serving)** together: e.g. H2 reaching ~62 pages/s (standalone parity)
> combined with H3 at **6% fraction (61 needed)** clears it. The cheapest credible joint path is
> H3→~6-10% AND H2→~62-102 pages/s. H3's contribution is to make H2's job 1.4-2.3x easier and to
> remove the F1==0 fallback drag at the same time.

---

## 6. F1 impact summary

| Lever | LLM-fraction effect | F1 effect | F1 risk | Effort |
|---|---|---|---|---|
| A — looser clustering | -1 to -3 pts structural (conditional) | +0 if gated; -drag if not | medium | S (sweep) |
| B — validation gate | partitions fallback correctly; -~5 pts via §6 path | **+0.10** (kills F1==0 fallbacks) | none | M |
| C — multi-rep | net -~2 pts | +0.01-0.02 (more clusters get good template) | low | M |
| D — ratio gate | small | +0.01-0.02 (kills wrong-region F1==0) | none | S |

Levers B+D are pure wins (F1 up, no risk). Lever C is a good trade (net LLM down, F1 up slightly).
Lever A is the only one with downside and must be measured before adoption.

---

## 7. Prioritized recommendation

1. **Lever B (validation gate)** — port `layout_template_validation_rows` /
   `validation_min_content_f1=0.98` semantics into Stage 3's per-cluster decision (extend
   `_cluster_static_trustworthy` to a propagation-vs-rep-LLM-content F1 gate). Strictly F1-protective,
   converts blind fallbacks into confident propagation or honest LLM. Biggest F1 lever, ~0 marginal
   LLM (validates against the rep content already computed).
2. **Lever D (ratio gate)** — cheap, F1-protective, catches wrong-region propagation.
3. **Lever C (multi-rep)** — Stage 1b emits 2-3 candidate reps; Stage 3 uses first that validates.
   Net-reduces LLM fraction ~2 pts.
4. **Lever A (threshold sweep)** — offline-measure 0.92/0.95/0.97 against propagation success ONLY
   after B/C/D land; adopt looser only if net LLM fraction drops.

Expected outcome: **LLM fraction ~14% → ~10%**, required throughput **143 → ~102 pages/s/node**,
overall F1 ≥ 0.91 (the current-plan F1, preserved/improved). This relaxes the H2 serving target by
~1.4x at no F1 cost, and is the cheapest lever to make the joint 2-day target reachable.
