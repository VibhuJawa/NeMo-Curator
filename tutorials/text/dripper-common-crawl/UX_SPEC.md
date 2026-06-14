# Dripper × MinerU-HTML — Mission-Control Dashboard UX Spec

Operator-first. One person watches a multi-day optimization run on a single screen and
occasionally types instructions back. The dashboard must answer two questions in 3 seconds:
**Are we hitting the two targets?** and **What is running right now?** Everything else is support.

Single self-contained `dashboard.html` (inline CSS + vanilla JS, offline, no build, no CDN).
Polls `GET /api/status` and `GET /api/prompts`; posts `POST /api/prompt`.

---

## 0. Visual system (foundation for "polished, not amateur")

- **Theme:** dark mission-control. Background `#0d1117` (near-black blue), surface `#161b22`,
  elevated surface `#1c2430`, hairline borders `#2a3340` (1px). Avoid pure black/white.
- **Type:** system UI stack for prose/labels (`-apple-system, "Segoe UI", Roboto, sans-serif`);
  monospace (`ui-monospace, "SF Mono", Menlo, monospace`) for all numbers/metrics so digits
  align and don't reflow as values change. Tabular numerals (`font-variant-numeric: tabular-nums`).
- **Scale:** 8px spacing grid. Page max-width ~1280px, centered, 24px gutters.
- **Accent palette (semantic, used consistently everywhere):**
  - Pass / healthy: `#3fb950` (green)
  - Close / warming: `#d29922` (amber)
  - Bottleneck / behind / error: `#f85149` (red)
  - Info / neutral progress: `#58a6ff` (blue)
  - Muted text: `#8b949e`; primary text `#e6edf3`.
- **Depth:** subtle 1px borders + a single soft shadow on cards (`0 1px 3px rgba(0,0,0,.4)`).
  No heavy drop shadows, no gradients except one restrained header bar.
- **Corners:** 10px on cards, 6px on chips/inputs. Consistent everywhere.
- **Motion:** 180–250ms ease-out for value/state transitions; nothing bounces; respect
  `prefers-reduced-motion` (disable number roll + pulse, keep instant updates).

---

## 1. Information hierarchy (top → bottom) and why

The page is a vertical priority stack. Reading order = importance order.

1. **Header / status bar (always visible, sticky).** Product name, global health verdict,
   freshness indicator. Anchors trust: the operator must always know the page is live.
2. **TIER 1 — The two targets (hero zone).** The entire reason this run exists. Two large
   side-by-side "scorecards": **Token-F1 → 0.90** and **GPU throughput → ~143 pages/s/node**.
   These are the biggest, brightest elements on the page. Everything below is *how we get there*.
3. **TIER 2 — Live operations.** What is happening right now:
   - **Pipeline stages** (the 7-stage chain, with the bottleneck visually called out).
   - **Slurm job queue** (live jobs, state, runtime, node).
   These are co-equal secondary; stages explain the throughput target, jobs explain "is work
   actually running."
4. **TIER 3 — Context & control.**
   - **Swarm deliverable docs** (10 chips — coverage of the planning effort).
   - **Operator prompt composer + history** (send instructions, see the log).
   Tertiary because they're reference/async, not the live pulse — but the prompt box is the
   operator's only *action*, so it gets a distinct, inviting treatment (not buried as an afterthought).

**Why this order:** an operator glancing for 3s lands on the verdict bar + two scorecards (am I
winning?). If something looks off, the eye travels down to stages/jobs (why?). Docs and prompt
history are intentionally last — consulted deliberately, not monitored.

Layout: TIER 1 full-width hero (2-up). TIER 2 a responsive 2-column row (stages left/wider,
jobs right). TIER 3 a 2-column row (docs left, prompt composer right) — or stacked when narrow.

---

## 2. The 3-second at-a-glance summary (header verdict bar)

A sticky top bar conveys the whole run in one line, computed client-side:

- **Left:** title `Dripper × MinerU-HTML` + small subtitle `Common Crawl parse optimization`.
- **Center — GLOBAL VERDICT pill.** One of:
  - `ON TARGET` (green) — both targets met.
  - `F1 READY · THROUGHPUT BEHIND` (amber→red split) — the realistic current state; name
    *which* target is the blocker so the operator instantly knows the story.
  - `WARMING UP` (amber) — neither met but progressing.
  - `STALLED` / `ERROR` (red) — see §3 error/stale rules.
  The pill text is explicit ("throughput behind"), never a bare color.
- **Right — freshness cluster:** a small live dot + `updated 3s ago` (relative, ticks every
  second) and a subtle spinning indicator only during an in-flight fetch (see §4).

Directly under the verdict, a one-line **mini-readout** of the two headline numbers so they're
visible even before scrolling: `F1 0.8905 → 0.90  ·  GPU 27.2 → 143 pages/s/node`. Each number
colored by its own pass/close/behind state.

This means: in 3 seconds the operator reads the pill ("throughput behind"), sees `F1 0.89 / GPU 27`,
and knows: F1 essentially there, throughput is the fight, page is live.

---

## 3. Per-component spec (data, states, rendering)

Universal states every data component must implement: **loading** (first paint, before any
successful fetch), **empty** (fetch ok but no data), **error** (`status.error` non-empty or fetch
failed), **stale** (last good `ts` older than threshold), **success**.

- **Skeletons, not spinners,** for first load: gray shimmer blocks matching final layout so the
  page doesn't jump. Spinner is reserved for the tiny header refresh indicator.
- **Stale rule:** if `now - ts > 15s` → mark *stale*: dim the affected cards to 70% opacity, add
  an amber `STALE · last good 42s ago` ribbon on the header, keep showing last known values
  (never blank good data just because one poll was late). At `> 60s` escalate header pill to red
  `CONNECTION LOST` but still hold last values.
- **Error rule:** `status.error` non-empty → header pill red with the error text truncated +
  hover/expand for full text; data cards keep last values dimmed. Never throw away the screen.

### 3.1 TIER 1 — Target scorecards (two cards)

**Card A — Token-F1.** Data: `final_f1` header line + `f1_roles[]`; static target 0.90;
journey milestones (static domain facts).
- Hero number: parse the F1 mean from payload (`0.8905`), shown huge (48–56px, mono).
  State: `>=0.90` green "MET"; `0.88–0.899` amber "0.0095 to go"; `<0.88` red.
- **Progress arc/bar** from 0.80→0.90 (the meaningful operating band, not 0→1, so movement is
  visible). Marker for current value; ghost ticks for journey milestones
  (0.025 → 0.51 → 0.81 → 0.89 → 0.90) shown as a tiny sparkline/stepline labeled
  "F1 journey" so the operator sees momentum.
- **Per-role breakdown:** render `f1_roles[]` as a small 3-row table — role · pages · mean F1 ·
  ≥0.80 · F1==0 — using the columns already in the payload. Color each role's F1 cell by band.
  Empty state (no roles yet): "Per-role F1 pending re-inference."
- Empty `final_f1`: card shows "F1 not yet computed" with the target + journey still visible.

**Card B — GPU throughput.** Data: `s2rate_raw` (`inference_only=26.4 pages/s`) as the truth
source for current inference rate; `fb2` for re-inference progress; `s3_rate` as supporting;
static target 143 pages/s/node and the "16 nodes → CC-MAIN in 2 days" framing.
- Hero number: current pages/s/node parsed from `s2rate_raw` (`27.2`/`26.4`), mono, big.
  Always red/amber until ≥143 — this is the known bottleneck; the card should *feel* like the
  open problem (subtle red left-border accent).
- **Gap visualization:** horizontal bar 0→143 with current fill; explicit `5.3× to target`
  multiplier label (computed) — multipliers communicate "how far" better than raw deltas here.
- **Re-inference progress:** parse `fb2` (`4592/4592 pages 27.2 pages/s`) → a determinate
  progress bar `4592/4592 (100%)`; when complete show a green check + "re-inference complete".
- **Projected-time readout (derived, high value):** "At 27 p/s: CC-MAIN ≈ N days on 16 nodes →
  target 2 days." Recompute from live rate so the operator sees the prize shrink as throughput climbs.

### 3.2 TIER 2 — Pipeline stages

Data: `queue` (live), `s2rate_raw`, `s3_rate` for live overrides; otherwise the static stage
table (1a 595 done; 1b 150 done; 1c 88 done; 2 vLLM 27 BOTTLENECK; 2b 95 done; 3 77 done, 4.8× from 16).
- Render as a **horizontal pipeline rail**: 7 nodes (1a→1b→1c→2→2b→3) connected by chevrons,
  left→right = data flow. Each node = a compact tile: stage id, short name, `pages/s`, status dot.
- Status colors: done = green, bottleneck = red (Stage 2 gets a pulsing red ring + a
  `BOTTLENECK` tag so the eye is dragged to it). Stage 3 shows an "improved 4.8× from 16" badge
  to credit progress.
- Overlay live rates when available: Stage 2 rate from `s2rate_raw`, Stage 3 from `s3_rate`,
  so the rail reflects reality, not just defaults.
- Narrow screens: rail wraps to a vertical list (chevrons rotate to down-arrows).
- Empty/error: keep static stage definitions visible (they're known facts) but gray the live
  rate field and tag it "rate unavailable".

### 3.3 TIER 2 — Slurm job queue

Data: `queue[] = {id, name, state, time, node}`.
- A clean table: STATE badge · NAME · JOB ID (mono) · RUNTIME (mono, right-aligned) · NODE (mono).
- State badges: `RUNNING` green, `PENDING` amber, `COMPLETING`/`COMPLETED` blue, `FAILED`/`CANCELLED`
  red. Sort RUNNING first, then PENDING, then others.
- Header shows count: `2 jobs · 2 running`.
- Empty state: friendly, not alarming — "No jobs in queue" with a small idle icon (an empty
  queue mid-run may be intentional between submissions).
- Runtime updates are the classic "jarring" risk — animate per §4 (no row flash; just the digit).

### 3.4 TIER 3 — Swarm deliverable docs

Data: `docs{name: bool}` (10 known names).
- Render as a wrap of 10 chips, each: status glyph + filename. `true` → green check chip
  (solid-ish), `false` → muted outline chip with a hollow circle.
- Header: completion counter `Deliverables 10/10` with a thin progress bar. When all true,
  the whole group gets a subtle green tint + "swarm complete".
- These are presence indicators only (no link target promised by the API) — render filename as
  plain mono text; if a doc flips false→true, briefly highlight that chip (§4).

### 3.5 TIER 3 — Operator prompt composer + history (see §5).

---

## 4. Live-refresh UX (freshness without jank)

- **Poll cadence:** `/api/status` every 5s, `/api/prompts` every 10s (or after a successful POST).
  Use a single `setInterval` per endpoint; guard against overlap (skip a tick if the previous
  fetch is still in flight).
- **Freshness display:** header shows a relative `updated Ns ago` that increments every second
  off the last good `ts` (separate 1s ticker from the 5s poll) so it feels alive between polls.
  A small filled dot pulses green once per successful fetch.
- **In-flight indicator:** a tiny 14px ring spinner appears next to the freshness text only while
  a fetch is outstanding; it must be subtle (low-contrast, no layout shift). No full-page loading
  overlay after first paint.
- **No jarring re-renders — diff, don't replace:**
  - Never rebuild whole sections via `innerHTML` on each poll. On first render, build the DOM;
    on subsequent polls, **update only changed text nodes / attributes**. Keep stable element
    keys (job id, stage id, doc name) so rows/tiles persist and only their fields update.
  - **Animate numeric deltas:** when a metric changes, roll the number from old→new over ~250ms
    (simple requestAnimationFrame tween on the parsed float) and flash the text color toward the
    direction of change (greenish if improving toward target, reddish if regressing) for ~600ms,
    then settle to its band color. Tabular-nums prevents width jitter during the roll.
  - **State changes** (job RUNNING→COMPLETED, doc false→true, stage rate update) cross-fade the
    badge/chip rather than hard-swapping.
  - If a value is unchanged, do nothing (no flash) so attention is reserved for real change.
- **Reduced motion:** when `prefers-reduced-motion`, swap values instantly, drop pulses/rolls,
  keep only the dim-on-stale.

---

## 5. Prompt composer UX

The operator's single action surface — make it inviting and frictionless, placed in TIER 3 right
column as a "console".

- **Composer:**
  - Multiline `textarea`, auto-growing (1→~5 rows), mono font (operators type commands/paths).
  - **Placeholder guidance** (rotating or static, instructive): e.g.
    `Send an instruction to the swarm…  e.g. "prioritize Stage 2 FP8" · "re-run F1 on siblings" · ⌘↵ to send`.
  - **Send affordance:** a primary button labeled `Send` with a paper-plane glyph, disabled
    (dimmed) when the textarea is empty/whitespace. A hint line under it: `⌘/Ctrl + Enter to send`.
  - **Keyboard:** `Cmd/Ctrl+Enter` submits; plain `Enter` inserts a newline (don't hijack Enter —
    these are multi-line instructions). `Esc` clears focus.
- **Submit flow & confirmation:**
  - On send: optimistically append the message to the history list (dimmed, with a tiny "sending…"
    spinner), disable the button, POST `{text}`.
  - On `{ok:true}`: settle the optimistic item to normal using the server-returned `saved.ts`
    (authoritative timestamp), brief green flash + a transient toast `Instruction queued ✓`,
    clear and refocus the textarea.
  - On failure: mark the optimistic item with a red `failed — retry` affordance (click to resend),
    keep the text in the box so nothing is lost. Never silently drop an instruction.
- **History display:**
  - Data: `/api/prompts` (`{ts, text}`, newest last). Render **newest at top** (reverse) in a
    scrollable log, each entry: relative time (`2m ago`, hover = absolute `ts`) + the text
    (preserve whitespace/newlines, `white-space: pre-wrap`, mono).
  - Header: `Operator log · N`. Empty state: "No instructions sent yet — type one below."
  - When polling brings in a *new* entry not from this client, slide it in at top with a brief
    highlight so the operator notices another operator/automation acted.
  - Subtle visual distinction between operator entries and any system/test entries if detectable
    by text prefix; otherwise treat uniformly.

---

## 6. Responsive behavior

Mobile-considered but desktop-primary (this lives on a big monitor).

- **Wide (≥1100px):** centered max-1280 column. TIER 1 = 2 equal scorecards side by side.
  TIER 2 = stages (≈60% width) + jobs (≈40%). TIER 3 = docs + composer side by side.
  Pipeline rail horizontal with chevrons. Header single row.
- **Medium (700–1099px):** scorecards stay 2-up (they're the priority) but shrink hero font;
  TIER 2 and TIER 3 each collapse to a single stacked column. Pipeline rail may wrap to 2 rows.
- **Narrow (<700px):** everything single column in strict priority order: verdict bar → F1 card →
  throughput card → stages (vertical rail, down-chevrons) → jobs (cards instead of table, hide
  Node into a second line) → docs (chips wrap, 2-up) → composer → history. Header collapses:
  title on row 1, verdict pill + freshness on row 2. Sticky header still pins the verdict.
- Touch targets ≥44px (send button, chips). No horizontal scroll at any width; tables become
  stacked cards rather than overflowing.

---

## 7. Accessibility / robustness notes

- Color is never the only signal: pass/behind also carry text ("MET", "BEHIND", "BOTTLENECK")
  and glyphs (check / dot / alert).
- All live regions that update get `aria-live="polite"` on the verdict pill and freshness so a
  screen reader announces target/connection changes but isn't spammed by every digit roll.
- Parse defensively: every payload field may be empty/malformed mid-run — wrap parsing
  (`final_f1`, `fb2`, `s2rate_raw`, `s3_rate`) in try/guards; fall back to "—" + the static
  target rather than NaN or a broken layout. The dashboard must never go blank because one
  string didn't match a regex.
- Keep all assets inline; no network calls except the three same-origin API endpoints (offline-safe).
