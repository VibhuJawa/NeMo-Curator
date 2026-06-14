# Dripper × MinerU-HTML — Mission Control Visual Design System

A prescriptive, implementation-ready spec for a single self-contained `dashboard.html`
(inline CSS + vanilla JS, no build, no CDN, offline-safe). Aesthetic target:
Linear / Vercel / Grafana — dark, restrained, premium, data-dense but calm.

Everything below is exact. Use `:root` CSS custom properties verbatim.

---

## 1. Color Palette (dark theme)

### Surface elevation (background → foreground stack)
| Token | Hex | Use |
|---|---|---|
| `--bg-base` | `#0A0C10` | page background (deepest) |
| `--bg-sunken` | `#0E1117` | wells, table body, inset areas |
| `--surface-1` | `#14171F` | cards (default elevation) |
| `--surface-2` | `#1B1F2A` | raised card / hover / popovers |
| `--surface-3` | `#232836` | active row, pressed, tooltips |
| `--hairline` | `#262B36` | 1px borders, dividers |
| `--hairline-strong` | `#333A48` | card outer border, focus track |

Page uses a very subtle top glow, not a flat fill:
```css
background:
  radial-gradient(1200px 600px at 50% -10%, #11151F 0%, transparent 70%),
  var(--bg-base);
```

### Text
| Token | Hex | Contrast on `--surface-1` | Use |
|---|---|---|---|
| `--text-hi` | `#F2F4F8` | 15.0:1 | headings, primary numbers |
| `--text` | `#C7CDD9` | 9.6:1 | body |
| `--text-dim` | `#8B93A4` | 5.1:1 | labels, secondary |
| `--text-faint` | `#5C6373` | 3.0:1 | captions/units only (never <13px body) |

### Semantic (status) colors — each has a base, a soft-bg, and a border tint
| Role | Base | Soft bg (12% alpha) | Border (28%) |
|---|---|---|---|
| `--ok` (done/healthy) | `#3FB950` | `rgba(63,185,80,.12)` | `rgba(63,185,80,.28)` |
| `--run` (running/live) | `#3B82F6` | `rgba(59,130,246,.12)` | `rgba(59,130,246,.30)` |
| `--queue` (queued/pending) | `#A371F7` | `rgba(163,113,247,.12)` | `rgba(163,113,247,.28)` |
| `--warn` (bottleneck) | `#E3B341` | `rgba(227,179,65,.12)` | `rgba(227,179,65,.30)` |
| `--bad` (failed/below) | `#F85149` | `rgba(248,81,73,.12)` | `rgba(248,81,73,.30)` |
| `--accent` (brand/F1) | `#2DD4BF` | `rgba(45,212,191,.12)` | `rgba(45,212,191,.30)` |

`--accent` (teal) is the brand spine — used for the F1 target, the active nav
underline, focus rings, primary button. `--run` (blue) is reserved strictly for
live/animated items so motion reads as "this is moving right now."

### Gradients (for progress fills only — left→right)
```css
--grad-accent: linear-gradient(90deg, #14B8A6 0%, #2DD4BF 60%, #5EEAD4 100%);
--grad-run:    linear-gradient(90deg, #2563EB 0%, #3B82F6 60%, #60A5FA 100%);
--grad-ok:     linear-gradient(90deg, #2EA043 0%, #3FB950 100%);
--grad-warn:   linear-gradient(90deg, #BB8009 0%, #E3B341 100%);
```
Progress fills get a faint inner highlight: `box-shadow: inset 0 1px 0 rgba(255,255,255,.18);`

---

## 2. Typography

System stack only (no web fonts):
```css
--font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
--font-mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
```
All numeric/data uses `--font-mono` with `font-variant-numeric: tabular-nums;`
so digits never jitter during roll-ups.

### Scale (px / weight / letter-spacing / line-height)
| Token | Size | Weight | Tracking | LH | Use |
|---|---|---|---|---|---|
| `--t-display` | 30 | 650 | -0.02em | 1.1 | hero metric numbers |
| `--t-h1` | 19 | 620 | -0.01em | 1.25 | page title |
| `--t-h2` | 15 | 600 | -0.005em | 1.3 | card titles |
| `--t-body` | 14 | 450 | 0 | 1.5 | body / prompt text |
| `--t-data` | 14 | 550 | 0 | 1.4 | table cells, stat values (mono) |
| `--t-data-lg`| 22 | 600 | -0.01em | 1.2 | tile primary value (mono) |
| `--t-label` | 11.5 | 600 | 0.06em | 1.2 | UPPERCASE section/eyebrow labels |
| `--t-cap` | 12 | 500 | 0.01em | 1.3 | units, captions, timestamps |

Labels (`--t-label`) are `text-transform: uppercase;` colored `--text-dim`.
Weight note: 650/620 work via `font-weight` numeric on system fonts; if a platform
snaps to 700 that's acceptable.

---

## 3. Spacing, Radius, Border, Shadow, Layout

### Spacing scale (4px base)
`--s1:4 --s2:8 --s3:12 --s4:16 --s5:20 --s6:24 --s7:32 --s8:48`. Use only these.
Card padding = `--s5` (20px). Gap between cards = `--s5`. Section gap = `--s7`.

### Radius
`--r-sm:6 --r-md:10 --r-lg:14 --r-pill:999`. Cards `--r-lg`, controls/tiles `--r-md`,
chips/badges `--r-pill`, progress tracks `--r-pill`.

### Borders
1px solid `--hairline` for internal dividers; cards use `1px solid var(--hairline-strong)`.
Never use pure-black borders. No double borders — divider OR shadow, not both.

### Shadows (subtle, dark-theme correct — low alpha, no harsh black)
```css
--sh-1: 0 1px 2px rgba(0,0,0,.40);
--sh-2: 0 4px 16px rgba(0,0,0,.45), 0 1px 2px rgba(0,0,0,.40);
--sh-pop: 0 12px 40px rgba(0,0,0,.55);
--ring: 0 0 0 3px rgba(45,212,191,.35); /* focus */
```
Cards: `--sh-1` at rest, `--sh-2` on hover (only interactive cards animate elevation).

### Layout / grid
- Page max-width `1320px`, centered, horizontal padding `--s7` (`--s5` under 720px).
- Sticky top bar height `60px`, `backdrop-filter: blur(12px)`, bg `rgba(10,12,16,.72)`,
  bottom `1px solid var(--hairline)`.
- Body grid: 12-col CSS grid, `gap: var(--s5)`.
  - **Targets row**: two large cards, `grid-column: span 6` each (≥960px); stack to `span 12` below 880px.
  - **Stat tiles**: 4-up auto-fit, `repeat(auto-fit, minmax(180px,1fr))`.
  - **Main split**: pipeline list `span 7`, F1 journey `span 5`; stack below 900px.
  - **Jobs table**: `span 12`. **Prompt composer**: `span 12`.
- Mobile (<640px): single column, top bar wraps, tiles 2-up.

---

## 4. Component Styling

General card:
```css
.card{background:var(--surface-1);border:1px solid var(--hairline-strong);
  border-radius:var(--r-lg);padding:var(--s5);box-shadow:var(--sh-1);}
.card__head{display:flex;align-items:center;justify-content:space-between;
  margin-bottom:var(--s4);}
.card__title{font:var(--t-h2);color:var(--text-hi);}
.eyebrow{font:var(--t-label);text-transform:uppercase;color:var(--text-dim);}
```

### 4.1 Target progress bars (the two hero goals)
Card contains: eyebrow label → big mono value (`--t-display`) with unit in `--text-faint`
→ progress track → caption (start → goal).

- Track: height `10px`, radius pill, bg `--bg-sunken`, `inset 0 1px 2px rgba(0,0,0,.5)`.
- Fill: `--grad-accent` for F1, `--grad-run` for throughput; `width` = % of goal,
  transition `width 600ms cubic-bezier(.22,.61,.36,1)`.
- **Value badge**: a pill that sits on the fill's right edge (`transform:translateX(50%)`),
  bg `--surface-3`, 1px border in the metric's color, mono `--t-cap`, shows current value.
- **Threshold marker** at the goal position: a 2px vertical tick full track height,
  color `--text-dim`, with a tiny flag label "0.90" / "143" above it (`--t-cap`, `--text-dim`).
  When current ≥ goal the fill turns `--grad-ok` and badge border → `--ok`.
- F1 example: goal 0.90, current 0.8905 → fill at `(0.8905/0.95 normalized)`; render the
  track domain as `[0.80 … 0.95]` so the climb is visible and the 0.90 marker sits mid-right.
- Throughput: domain `[0 … 143]`, current 27 → ~19% fill, marker at right end (clearly far).

### 4.2 Stat tiles
Compact cards: eyebrow label (top), mono value `--t-data-lg`, delta/badge below.
```css
.tile{background:var(--surface-1);border:1px solid var(--hairline);
  border-radius:var(--r-md);padding:var(--s4);display:flex;flex-direction:column;gap:var(--s2);}
.tile__value{font-family:var(--font-mono);font-size:22px;font-weight:600;color:var(--text-hi);}
.tile__delta.up{color:var(--ok);} .tile__delta.down{color:var(--bad);}
```
Use for: current mean F1, inference pages/s, S3 rate, propagation 4.8× gain.
A thin 2px accent bar on the tile's left edge keyed to its semantic color
(`box-shadow: inset 3px 0 0 var(--accent)`).

### 4.3 Pipeline-stage list (bar per stage)
One row per stage. Grid: `[status-dot 8px] [name 1fr] [bar 200px] [value 90px mono]`.
- Stage name `--t-body` `--text`; below it a `--t-cap` `--text-faint` note ("DBSCAN", "vLLM").
- Mini bar: track `6px` pill `--bg-sunken`; fill width = `pages/s` scaled to the max stage
  (595) on a sqrt or capped-log scale so small stages stay visible — OR scale each fill to
  `min(100%, value/maxNonBottleneck)`. Fill color: `--ok` if done, `--warn` if BOTTLENECK.
- The bottleneck row (Stage 2, vLLM 27) gets `--warn` left accent, a "BOTTLENECK" chip,
  and its bar pulses (see §5). Row hover: bg `--surface-2`, radius `--r-sm`.
- Right value: `595` etc. in mono `--t-data`, unit "p/s" in `--text-faint`.

### 4.4 F1 journey chart (sparkline / step-up)
Small inline SVG, ~`100%×120px`, no library. Milestones:
`0.025 → 0.51 → 0.81 → 0.89 → 0.90(target)`.
- Render as a monotonic line+area: stroke `--accent` 2px, area fill
  `linear-gradient(180deg, rgba(45,212,191,.22), transparent)` (SVG `<linearGradient>`).
- Y domain `[0 … 1]`; dashed horizontal goal line at `0.90` in `--text-dim` with label "target 0.90".
- Dots `r=3` at each milestone, `--surface-1` fill + `--accent` stroke; last dot solid `--accent`.
- On hover of a dot show a tooltip (`--surface-3`, `--sh-pop`) "chat+pickle · 0.81".
- Draw the line with a `stroke-dasharray` reveal on first paint (700ms).

### 4.5 Status chips
```css
.chip{display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 10px;
  border-radius:var(--r-pill);font:var(--t-label);text-transform:uppercase;
  border:1px solid; background:transparent;}
```
Map: RUNNING→`--run` (+pulsing dot), DONE/COMPLETED→`--ok`, PENDING/QUEUED→`--queue`,
BOTTLENECK/WARN→`--warn`, FAILED→`--bad`. Each chip: text=base color, border=border-tint,
bg=soft-bg. Leading 6px dot in the same base color.
**Doc chips** (swarm deliverables): pill with a check glyph; present(`docs[name]==true`)→
`--ok` soft-bg + check; absent→`--surface-2` bg, `--text-faint`, no check, 0.6 opacity.

### 4.6 Live jobs table
```css
table{width:100%;border-collapse:separate;border-spacing:0;font-family:var(--font-mono);}
thead th{font:var(--t-label);text-transform:uppercase;color:var(--text-dim);
  text-align:left;padding:0 var(--s3) var(--s2);border-bottom:1px solid var(--hairline);}
tbody td{padding:var(--s3);border-bottom:1px solid var(--hairline);font:var(--t-data);color:var(--text);}
tbody tr:last-child td{border-bottom:0;}
tbody tr:hover{background:var(--surface-2);}
```
Columns: ID · Name · State(chip) · Time · Node. State cell renders a §4.5 chip.
RUNNING rows get a 2px `--run` left accent (`box-shadow: inset 2px 0 0 var(--run)`).
Empty state: centered `--text-dim` "No active jobs" with a small idle dot.
Zebra is OFF (hairlines only) — cleaner, observability-style.

### 4.7 Prompt composer + history
- History: scrollable column (max-height `260px`), each entry a left-bordered card
  (`inset 2px 0 0 var(--accent)`), `--surface-1`, padding `--s3`; timestamp in
  `--t-cap` `--text-faint` mono, text `--t-body` `--text`. Newest pinned to bottom; auto-scroll.
- Composer: `textarea` (`--surface-2`, 1px `--hairline-strong`, radius `--r-md`,
  padding `--s3`, mono `--t-body`, min-height 64px, resize vertical), placeholder
  "Send an instruction to the swarm…", focus → `--ring` + border `--accent`.
- Send button: `--accent` bg, `#04211D` text, `--r-md`, height 36px, weight 600;
  hover brighten 6%, active translateY(1px), disabled 0.45 opacity. ⌘/Ctrl+Enter submits.
- On POST success: optimistic append the entry with a 200ms fade+slide-up.

---

## 5. Motion
Global: `transition: background-color .15s, border-color .15s, box-shadow .15s, color .15s;`
Easing tokens: `--ease-out: cubic-bezier(.22,.61,.36,1)`, `--ease: cubic-bezier(.4,0,.2,1)`.

- **Progress fills / bars**: `width .6s var(--ease-out)`.
- **Number roll-up**: when a metric changes, animate value count from old→new over 500ms
  (`requestAnimationFrame`, ease-out), tabular-nums to avoid width shift. Skip if delta is 0.
- **Live pulse** (running jobs, bottleneck bar, live dot): soft breathing, NOT flashing:
  ```css
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
  .live-dot{animation:pulse 1.8s var(--ease) infinite;}
  ```
  Bottleneck bar uses a slow shimmer: a 1.2px lighter band sweeping the fill every 2.4s.
- **Card hover**: elevation `--sh-1`→`--sh-2` + `translateY(-1px)` over .15s (interactive cards only).
- **Data refresh tick**: top-bar "live" dot blips `--ok` for 400ms on each successful poll;
  on `error!==""` it goes solid `--bad` and a banner slides down.
- **Reveal**: F1 sparkline dash-reveal 700ms once; cards fade-in stagger 40ms on first load.
- `@media (prefers-reduced-motion: reduce)`: disable pulse/shimmer/roll-up/reveal; keep
  instant state changes and ≤120ms color fades.

---

## 6. Accessibility
- Contrast: all text tokens on their intended surfaces meet WCAG AA — body `--text` ≥9:1,
  labels `--text-dim` ≥5:1; `--text-faint` reserved for ≥non-essential captions only.
  Status base colors on soft-bg chips: verified ≥4.5:1 for the chip label.
- Never encode state by color alone: chips carry a text label + dot; bottleneck has the
  word "BOTTLENECK"; doc chips show check/no-check glyph; F1 marker has a numeric flag.
- Focus: every interactive element gets `outline:none; box-shadow:var(--ring);` (3px teal,
  35% alpha) — visible on all surfaces. Tab order = top bar → targets → tiles → pipeline →
  jobs → composer. Composer textarea and Send reachable; ⌘/Ctrl+Enter documented in placeholder.
- Live regions: status banner `role="status" aria-live="polite"`; prompt history list
  `aria-live="polite"` so appended ops are announced. Pulsing dots are decorative `aria-hidden`.
- Tables use real `<th scope="col">`. Progress bars use
  `role="progressbar" aria-valuenow/min/max` with `aria-label` ("Token F1: 0.8905 of 0.90 goal").
- Hit targets ≥32px height for buttons/chips that are interactive.
- Tooltips are supplementary only; never the sole source of a value.

---

## 7. Implementation notes
- Poll `/api/status` + `/api/prompts` every ~4s; diff values to trigger roll-ups only on change.
- Keep all CSS in one `<style>`; all logic in one `<script>`. No external requests.
- Parse `f1_roles`/`final_f1` as monospace fixed-column text into a small role table inside
  the F1 card (or render raw in a `--bg-sunken` `<pre>` styled mono if parsing is brittle).
- Derive throughput-target % from `s2rate_raw` (`inference_only=X pages/s`) vs 143.
- Degrade gracefully: any missing/empty field → show `—` in `--text-faint`, never blank layout.
