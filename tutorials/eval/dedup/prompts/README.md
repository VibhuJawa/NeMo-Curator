# Fuzzy-pair judge prompts: side by side

Four prompts now judge the same question: *can the kept document replace the removed one without losing useful training
text?* They disagree on several page types, and that changes the false-positive (FP) rate by 3-4x. This folder holds them
side by side, so we can agree on one labelling policy and then one prompt. Nothing here is wired into the pipeline.

| Prompt | Author | Where | Input | Output |
|---|---|---|---|---|
| Base | Sarah | `../judge_config/` (this branch) | span packet from step 4 | ledger + `a_can_replace_b` / `b_can_replace_a` |
| C0 (`c0-boundary-v2`) | Hongsheng | `c0/`, copied from `fhschina/Curator` `codex/c0-retention-pr2484` @ `3aeaf1f` (main judge only; its critics live there) | span packet | same ledger as Base |
| PV1 | Praateek | `pv1/`, verbatim from his semantic-dedup judge prompt v1 (`system.md` sha256 `5af36608c93d…`) | full texts | `DUPLICATE` / `SUBSET` / `DISTINCT` / `UNSURE` + `similarity` |
| PV2 | Vibhu | `pv2/`, PV1 + the policy decisions below (`diff pv1/system.md pv2/system.md`) | full texts | same as PV1 |

For PV1 and PV2, `DUPLICATE` means a safe drop and `DISTINCT` means an FP. `SUBSET` has no direction, so the longer document is
taken to contain the shorter.

## Where they disagree

| Page pair | Base | C0 | PV1 | PV2 | Decide |
|---|---|---|---|---|---|
| Same template, only slot values differ (name, SKU, price, specs, date) | FP ("different SKU/title/profile/date") | FP, except a few named slots (city in a service page, currency on an empty cart, generic licence nouns) | duplicate | duplicate | **1** |
| Same template, but each side has its own 1-4 sentence description (product blurb, a lawyer's bio) | FP | FP | unclear; judges split | FP | **2** |
| Both sides only boilerplate (comment form, cookie notice), different sites | duplicate if the same complete message, else FP | same as Base, with detailed cookie-purpose rules | not stated | duplicate | **3** |
| Same announcement re-run with new dates, venue or counts | FP (date/count change) | FP unless the date is incidental metadata | not stated; the "reported change" rule reads both ways | duplicate unless a change is reported | **4** |
| Faithful translation | duplicate (yes/yes) | FP (keep both language versions) | FP ("different languages are never duplicates") | FP | **5** |
| One side contains the other plus more | directional containment | directional containment | `SUBSET`, no direction | `SUBSET`, no direction | **6** |

PV2's answers to 1-4 are Vibhu's proposal, taken from labelling the pairs below. They are not agreed yet.

## Evidence: 50 low-Jaccard disagreements

At 300M CC-MAIN-2025-26 records, the PV1 judges (Qwen3.8-27B, DeepSeek-V4.1-Flash) agree on ~95% of pairs. Almost all the
disagreement sits below Jaccard 0.80, which is about half of fuzzy dedup's removals. `eval/low_jaccard_disagreements_50.csv`
has 50 of those pairs (URLs, a blind label, a note, and every judge's verdict). The labels are Claude's under the PV2 policy;
there are no human labels yet.

| Judge | Resolved | Agrees with label | FP over-called | FP missed |
|---|---|---|---|---|
| C0 Qwen | 45 | 23 | 22 | 0 |
| C0 DeepSeek | 44 | 24 | 20 | 0 |
| PV1 Qwen | 50 | 18 | 11 | 21 |
| PV1 DeepSeek | 50 | 32 | 17 | 1 |
| PV2 DeepSeek | 43 | 34 | 9 | 0 |

The sample is chosen from disagreements, so these are not population accuracies. C0 calls almost every template pair an FP,
which is right under C0's policy. The table mostly shows how far each prompt is from the PV2 policy, so it is only a
judge-quality signal once 1-4 are settled.

On the whole 300M sample, the share of fuzzy-dedup removals judged FP, jusText / resiliparse / Harrier extraction:
C0 75-78% / 64-69% / 70-75%, PV1 29-39% / 19-27% / 16-24%, and PV2 36% / 23% / 23% (DeepSeek; PV2 below Jaccard 0.80, PV1 above).
Most of the gap between C0 and PV1 is decision 1.

## Questions for Sarah, Hongsheng, Praateek and Vibhu

1. Are pure slot-swap template pages an FP or a safe drop? This is the largest single effect on the FP rate.
2. Does a unique 1-4 sentence description of the page's own item make it an FP?
3. Are two boilerplate-only pages (after extraction) a safe drop even when the source sites differ?
4. Is a re-run of the same announcement with new facts a safe drop?
5. Translations: keep both (C0, PV1) or drop one (Base)?
6. Do we need a direction for containment, or is `SUBSET` enough?
7. Input format: span packet (Base, C0) or full text (PV1, PV2)? The packet needs step 4; full text needs a long context.

Once 1-7 are settled, we pick one prompt, label ~100 random pairs by hand (not chosen from disagreements), and report its
agreement with those labels.
