#!/usr/bin/env python3
"""dashboard_server.py — live FastAPI mission-control for the Dripper×MinerU pipeline.

Run:  uv run --with fastapi --with uvicorn python dashboard_server.py
Open: http://127.0.0.1:8765

Pulls live state from the Nebius cluster (squeue + log tails over SSH) on a
background refresher, serves a dark auto-refreshing dashboard, and accepts prompts
(POST /api/prompt) which are appended to prompts.jsonl for the operator to action.
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

HERE = Path(__file__).parent
PROMPTS = HERE / "prompts.jsonl"
CHATLOG = HERE / "chatlog.jsonl"
CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")
CHAT = {"sid": None, "lock": threading.Lock()}
CHAT_CTX = (
    "You are the on-dashboard co-pilot for the Dripper x MinerU-HTML pipeline. "
    "CURRENT STATUS (2026-06-13): Both targets MET — F1=0.9092 (>0.90 ✅), "
    "GPU throughput=163 p/s/node (>143 target ✅). "
    "Active work: (1) E2E v3 smoke test running — 5-job pipeline with combined "
    "GPU stage (1c+2+2b in one Slurm job, no intermediate parquet), stage 3 propagation "
    "running, F1 result expected soon. (2) LOC reduction goal: PR has 13K net new lines, "
    "target <2K. (3) Streaming improvement shipped: aftercorr Slurm deps save ~28% wall-clock "
    "at fleet scale. Hardware target: 1 CC snapshot/day on 16 GPU nodes + 40 CPU nodes. "
    "You may read files and run read-only commands. Do NOT edit files or submit/cancel jobs."
)
HOST = "nb-hel-cs-001-login-01.nvidia.com"
# Pipeline output dir — override with PIPELINE_OUTPUT env var for different runs.
# Default is the current E2E v3 run (5-job streaming pipeline).
B = os.environ.get(
    "PIPELINE_OUTPUT",
    "/lustre/fsw/portfolios/llmservice/users/vjawa/pipeline_full_e2e_v3",
)
NBX = "/tmp/nbx.sh"
REFRESH_S = 12

STATE = {
    "ts": 0,
    "queue": [],
    "fb2": "",
    "final_f1": "",
    "f1_roles": [],
    "s3_rate": "",
    "stage2_rate": "",
    "gpu_pipeline_timing": "",
    "gpu_pipeline_rate": "",
    "docs": {},
    "error": "",
}

# F1 milestones (static history) + targets
F1_JOURNEY = [("v2 bugs", 0.025), ("s3 wiring", 0.51), ("chat+pickle", 0.81)]
DOCS = [
    "OPTIMIZATION_ROADMAP.md",
    "STAGE2_GPU_PERF_PLAN.md",
    "F1_IMPROVEMENT_PLAN.md",
    "CPU_STAGES_PERF_PLAN.md",
    "STAGE3_PERF_AUDIT.md",
    "FP8_PLAN.md",
    "REDUCE_LLM_LOAD_PLAN.md",
    "STAGE3_DEEPER_PLAN.md",
    "CPU_MICROOPT_PLAN.md",
    "E2E_THROUGHPUT_MODEL.md",
]


def _ensure_nbx():
    if not Path(NBX).exists():
        Path(NBX).write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "source /Users/vjawa/Documents/codex/scripts/lib_nebius_ssh.sh\n"
            'host="$1"; shift\nnebius_ssh_command "$host" "$*"\n'
        )
        os.chmod(NBX, 0o755)


REMOTE_CMD = (
    'echo SQUEUE_START; squeue -u vjawa -h -o "%i|%j|%T|%M|%R" 2>/dev/null; echo SQUEUE_END; '
    # ── legacy experiment markers (keep for historical records) ──
    f"echo \"FB2|$(grep -oE '[0-9]+/4592 pages  [0-9.]+ pages/s' {B}/logs/fb_2.out 2>/dev/null | tail -1)\"; "
    f"echo \"S2OFFLINE|$(grep -oE 'PURE=[0-9.]+ pages/s/node' {B}/logs/atscale_self.out 2>/dev/null | tail -1)\"; "
    f'echo "EXP_BF16|$([ -f {B}/stage2_offline/metrics_stage2_shard_0000.json ] && echo done)"; '
    f'echo "EXP_FP8|$([ -f {B}/stage2_offline_fp8/metrics_stage2_shard_0000.json ] && echo done)"; '
    # ── new 5-job pipeline logs (v3 combined GPU stage) ──
    # Stage 3 rate: reads s3_0000.out (new log name from run_mineru_pipeline.sh)
    f"echo \"S3RATE|$(grep -oE '\\([0-9.]+ pages/s\\)' {B}/logs/s3_0000.out 2>/dev/null | tail -1)\"; "
    # GPU combined pipeline (1c+2+2b): sum per-GPU rates from s_gpu_0000.out
    f"echo \"GPURATE|$(grep -oE '[0-9.]+ pages/s/GPU' {B}/logs/s_gpu_0000.out 2>/dev/null | awk '{{sum+=$1}} END{{if(sum>0) print sum}}')\"; "
    # GPU ALL DONE summary line: total time + per-stage breakdown
    f"echo \"GPUDONE|$(grep 'ALL DONE' {B}/logs/s_gpu_0000.out 2>/dev/null | tail -1)\"; "
    # F1 from new Stage 4 (s4_metrics log — try both naming conventions)
    f"echo \"F1V3|$(grep -oE 'mean F1:[[:space:]]+[0-9.]+' {B}/logs/s4_metrics_*.out 2>/dev/null | tail -1)\"; "
    f'echo "F1V3ROLES_START"; grep -E "representative|singleton|sibling" {B}/logs/s4_metrics_*.out 2>/dev/null | tail -3; echo F1V3ROLES_END; '
    # Stage 4 propagation breakdown
    f'echo "PROPDIST_START"; grep -E "propagation_method|static|dynamic|fallback" {B}/logs/s4_metrics_*.out 2>/dev/null | head -8; echo PROPDIST_END; '
    # GPU pipeline metrics JSON (written by pipeline_metrics.StageMetrics)
    f"echo \"GPUJSON|$(cat {B}/stage2b/metrics_stage_gpu_pipeline_shard_0000.json 2>/dev/null | tr -d '\\n')\"; "
    # Legacy F1 fallback (old run logs)
    f"echo \"FINALF1|$(grep -E 'mean F1' {B}/logs/fb_merge_f1.out 2>/dev/null | tail -1)\"; "
    f'echo "FINALROLES_START"; grep -E "representative|singleton|sibling" {B}/logs/fb_merge_f1.out 2>/dev/null | tail -3; echo FINALROLES_END'
)


def refresh_loop():
    _ensure_nbx()
    while True:
        try:
            out = subprocess.run(
                ["bash", NBX, HOST, REMOTE_CMD], check=False, capture_output=True, text=True, timeout=40
            ).stdout
            q, in_q, roles, in_r, propdist, in_pd, in_v3r, v3roles = [], False, [], False, [], False, False, []
            for line in out.splitlines():
                if line == "SQUEUE_START":
                    in_q = True
                    continue
                if line == "SQUEUE_END":
                    in_q = False
                    continue
                if line == "FINALROLES_START":
                    in_r = True
                    continue
                if line == "FINALROLES_END":
                    in_r = False
                    continue
                if line == "F1V3ROLES_START":
                    in_v3r = True
                    continue
                if line == "F1V3ROLES_END":
                    in_v3r = False
                    continue
                if line == "PROPDIST_START":
                    in_pd = True
                    continue
                if line == "PROPDIST_END":
                    in_pd = False
                    continue
                if in_q and "|" in line:
                    p = line.split("|")
                    if len(p) >= 5:
                        q.append(
                            {
                                "id": p[0].strip(),
                                "name": p[1].strip(),
                                "state": p[2].strip(),
                                "time": p[3].strip(),
                                "node": p[4].strip(),
                            }
                        )
                elif in_r and line.strip():
                    roles.append(line.strip())
                elif in_v3r and line.strip():
                    v3roles.append(line.strip())
                elif in_pd and line.strip():
                    propdist.append(line.strip())
                elif line.startswith("FB2|"):
                    STATE["fb2"] = line[4:].strip()
                elif line.startswith("FINALF1|"):
                    v = line[8:].strip()
                    if v and not STATE.get("final_f1_v3"):
                        STATE["final_f1"] = v
                elif line.startswith("S3RATE|"):
                    v = line[7:].strip()
                    if v:
                        STATE["s3_rate"] = v
                elif line.startswith("S2RATE|"):
                    STATE["s2rate_raw"] = line[7:].strip()
                elif line.startswith("GPURATE|"):
                    v = line[8:].strip()
                    if v:
                        STATE["gpu_pipeline_rate"] = f"{v} pages/s/node (combined 1c+2+2b, kv-fp8)"
                        STATE["stage2_rate"] = f"{v} p/s/node"
                elif line.startswith("GPUDONE|"):
                    v = line[8:].strip()
                    if v:
                        STATE["gpu_pipeline_timing"] = v
                elif line.startswith("GPUJSON|"):
                    v = line[8:].strip()
                    if v:
                        try:
                            m = json.loads(v)
                            pps = m.get("pages_per_s_per_node") or m.get("pages_per_s_per_worker", 0)
                            if pps:
                                STATE["gpu_pipeline_rate"] = f"{pps:.1f} pages/s/node (combined, kv-fp8)"
                                STATE["stage2_rate"] = f"{pps:.1f} p/s/node"
                            extra = m.get("extra", {})
                            if extra.get("stage2_s"):
                                t2 = extra["stage2_s"]
                                pages = m.get("total_pages", 0)
                                pure = pages / max(t2, 1)
                                STATE["gpu_pipeline_timing"] = (
                                    f"1c={extra.get('stage1c_s', 0):.0f}s  "
                                    f"2={t2:.0f}s ({pure:.1f} p/s pure inference)  "
                                    f"2b={extra.get('stage2b_s', 0):.0f}s  "
                                    f"pages={pages:,}"
                                )
                        except Exception:
                            pass
                elif line.startswith("F1V3|"):
                    v = line[5:].strip()
                    if v:
                        STATE["final_f1"] = v
                        STATE["final_f1_v3"] = v
                elif line.startswith("S2OFFLINE|"):
                    v = line[10:].strip()
                    if v:
                        STATE["s2_offline"] = v
                        m_val = v.replace("PURE=", "").split()[0]
                        STATE["s2rate_raw"] = f"inference_only={m_val} pages/s (at-scale kv-fp8)"
                elif line.startswith("EXP_BF16|"):
                    STATE["_exp_bf16"] = line[9:].strip()
                elif line.startswith("EXP_FP8|"):
                    STATE["_exp_fp8"] = line[8:].strip()
            if v3roles:
                STATE["f1_roles"] = v3roles
            elif roles:
                STATE["f1_roles"] = roles
            if propdist:
                STATE["propdist"] = propdist
            STATE["queue"] = q
            STATE["f1_roles"] = roles
            STATE["docs"] = {d: (HERE / d).exists() for d in DOCS}
            # Experiments registry, with live done-markers overlaid.
            try:
                exps = json.loads((HERE / "experiments.json").read_text())
            except Exception:
                exps = []
            for e in exps:
                rf = e.get("result_file", "")
                if "stage2_offline_fp8" in rf and STATE.get("_exp_fp8") == "done":
                    e["status"] = "done"
                elif rf.startswith("stage2_offline/") and STATE.get("_exp_bf16") == "done":
                    e["status"] = "done"
            STATE["experiments"] = exps
            STATE.update(_compute_eta(q))
            STATE["ts"] = time.time()
            STATE["error"] = ""
        except Exception as e:
            STATE["error"] = f"{type(e).__name__}: {e}"
        time.sleep(REFRESH_S)


# E2E pipeline stages (name prefix → expected seconds for ~86k pages smoke, 1 GPU node).
# v3: 5-job pipeline — s1c+s2+s2b collapsed into s-gpu (combined GPU job).
# Actuals from 340772-340776: 1a~5min, 1b~15min, gpu~45min, s3~10min, s4~2min.
E2E_STAGES = [("s1a", 300), ("s1b", 900), ("s-gpu", 2700), ("s3", 600), ("s4", 120)]
N_E2E_STAGES = len(E2E_STAGES)


def _parse_elapsed(s):
    try:
        p = [int(x) for x in str(s).split(":")]
    except Exception:
        return 0
    if len(p) == 3:
        return p[0] * 3600 + p[1] * 60 + p[2]
    if len(p) == 2:
        return p[0] * 60 + p[1]
    return p[0] if p else 0


def _compute_eta(queue):
    """ETA for the running E2E pipeline = remaining time in the running stage +
    expected durations of all later stages (which are pending)."""
    names = {j["name"]: j for j in queue}
    # find the running E2E stage
    running_idx, running_elapsed = None, 0
    for i, (key, _exp) in enumerate(E2E_STAGES):
        for nm, j in names.items():
            if nm.startswith(key + "-") and j["state"] == "RUNNING":
                running_idx, running_elapsed = i, _parse_elapsed(j["time"])
    if running_idx is None:
        # nothing running but stages still queued? → about to start, sum all pending
        pend_idx = [i for i, (k, _e) in enumerate(E2E_STAGES) if any(nm.startswith(k + "-") for nm in names)]
        if not pend_idx:
            return {"eta_s": None, "eta_stage": "", "eta_step": ""}
        i0 = min(pend_idx)
        eta = sum(e for _k, e in E2E_STAGES[i0:])
        return {"eta_s": eta, "eta_stage": E2E_STAGES[i0][0], "eta_step": f"{i0 + 1}/{N_E2E_STAGES} queued"}
    cur_exp = E2E_STAGES[running_idx][1]
    eta = max(0, cur_exp - running_elapsed) + sum(e for _k, e in E2E_STAGES[running_idx + 1 :])
    return {
        "eta_s": eta,
        "eta_stage": E2E_STAGES[running_idx][0],
        "eta_step": f"{running_idx + 1}/{N_E2E_STAGES} running",
    }


app = FastAPI()


@app.get("/api/status")
def status():
    return JSONResponse(STATE)


@app.get("/api/prompts")
def get_prompts():
    if not PROMPTS.exists():
        return JSONResponse([])
    rows = []
    for ln in PROMPTS.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    return JSONResponse(rows[-50:])


@app.post("/api/prompt")
async def post_prompt(req: Request):
    body = await req.json()
    text = str(body.get("text", "")).strip()
    if not text:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "text": text}
    with PROMPTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return JSONResponse({"ok": True, "saved": rec})


@app.get("/api/chat/history")
def chat_history():
    if not CHATLOG.exists():
        return JSONResponse([])
    rows = []
    for ln in CHATLOG.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            pass
    return JSONResponse(rows[-100:])


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    msg = str(body.get("message", "")).strip()
    if not msg:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    if not CHAT["lock"].acquire(blocking=False):
        return JSONResponse({"ok": False, "error": "busy — a reply is still generating"}, status_code=429)
    try:
        cmd = [CLAUDE_BIN, "-p", "--output-format", "json", "--append-system-prompt", CHAT_CTX]
        if CHAT["sid"]:
            cmd += ["--resume", CHAT["sid"]]
        cmd.append(msg)
        t0 = time.time()
        proc = subprocess.run(cmd, check=False, cwd=str(HERE), capture_output=True, text=True, timeout=600)
        try:
            data = json.loads(proc.stdout)
            reply = data.get("result", "") or "(no output)"
            CHAT["sid"] = data.get("session_id") or CHAT["sid"]
            cost = data.get("total_cost_usd")
            turns = data.get("num_turns")
        except Exception:
            reply = (proc.stdout or proc.stderr or "(claude returned no parseable output)")[:4000]
            cost = turns = None
        rec = {
            "ts": time.strftime("%H:%M:%S"),
            "user": msg,
            "assistant": reply,
            "elapsed_s": round(time.time() - t0, 1),
            "cost_usd": cost,
            "turns": turns,
        }
        with CHATLOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return JSONResponse({"ok": True, **rec})
    except subprocess.TimeoutExpired:
        return JSONResponse({"ok": False, "error": "claude timed out (600s)"}, status_code=504)
    finally:
        CHAT["lock"].release()


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return CHAT_HTML


@app.get("/", response_class=HTMLResponse)
def index():
    # Prefer an external dashboard.html (owned by the design team) for hot-reload;
    # fall back to the embedded HTML if absent.
    ext = HERE / "dashboard.html"
    if ext.exists():
        return ext.read_text()
    return HTML


HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Dripper × MinerU — Mission Control</title>
<style>
:root{--bg:#0b0f1a;--panel:#121a2b;--panel2:#0e1626;--line:#1e2b45;--txt:#dce6f5;--mut:#7e8db0;
--ok:#39d98a;--run:#4aa8ff;--warn:#ffb347;--bad:#ff5d6c;--purp:#b06cff;--accent:#27e0c4}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(160deg,#070b14,#0d1424);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--txt)}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
h1{font-size:20px;margin:0;letter-spacing:.5px}
.sub{color:var(--mut);font-size:12px}
.grid{display:grid;gap:14px;grid-template-columns:1fr 1fr}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;
box-shadow:0 6px 24px rgba(0,0,0,.35)}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:1.5px;color:var(--mut);margin:0 0 12px}
.full{grid-column:1/3}
.bar{height:14px;background:var(--panel2);border-radius:8px;overflow:hidden;border:1px solid var(--line)}
.bar>span{display:block;height:100%;border-radius:8px;transition:width .6s cubic-bezier(.2,.8,.2,1)}
.row{display:flex;align-items:center;gap:10px;margin:8px 0}
.row .lab{width:130px;color:var(--mut);font-size:12px}
.row .val{margin-left:auto;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:7px}
.pulse{animation:p 1.2s ease-in-out infinite}@keyframes p{0%,100%{opacity:1}50%{opacity:.35}}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:500}
.pill{padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600}
.chip{display:inline-block;padding:3px 9px;margin:3px;border-radius:8px;font-size:11px;
border:1px solid var(--line);background:var(--panel2)}
.journey{display:flex;align-items:flex-end;gap:4px;height:90px}
.jb{flex:1;background:linear-gradient(180deg,var(--accent),#1c6;border-radius:5px 5px 0 0;
position:relative;min-height:6px}
.jb b{position:absolute;top:-18px;left:0;right:0;text-align:center;font-size:11px;color:var(--txt)}
.jb i{position:absolute;bottom:-30px;left:0;right:0;text-align:center;font-size:9px;color:var(--mut);font-style:normal}
.stage{display:flex;align-items:center;gap:10px;margin:7px 0}
.stage .nm{width:120px}.stage .pb{flex:1}
input,button{font:inherit}
#pin{width:100%;background:var(--panel2);border:1px solid var(--line);color:var(--txt);
border-radius:8px;padding:10px;resize:vertical}
#send{margin-top:8px;background:linear-gradient(90deg,var(--purp),#6c8cff);border:0;color:#fff;
padding:9px 18px;border-radius:8px;cursor:pointer;font-weight:600}
#send:hover{filter:brightness(1.1)}
.plist{max-height:150px;overflow:auto;margin-top:10px;font-size:12px}
.plist div{padding:6px 0;border-bottom:1px dashed var(--line)}
.plist .t{color:var(--mut);font-size:10px}
.flash{color:var(--accent)}
.foot{color:var(--mut);font-size:11px;margin-top:14px;text-align:center}
</style></head><body><div class=wrap>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
 <div><h1>🛰️ DRIPPER × MinerU — MISSION CONTROL</h1>
 <div class=sub>live · refresh <span id=age>—</span>s ago · <span id=err></span></div></div>
 <div style="text-align:right"><div class=sub>updated</div><div id=clock style="font-size:18px"></div></div>
</div>

<div class="card full"><h2>Targets</h2>
 <div class=row><span class=lab>① F1 &gt; 0.90</span>
   <div class=bar style=flex:1><span id=f1bar style="width:0;background:linear-gradient(90deg,#39d98a,#27e0c4)"></span></div>
   <span class=val id=f1val>—</span></div>
 <div class=row><span class=lab>② GPU 2-day/16n</span>
   <div class=bar style=flex:1><span id=gpubar style="width:0;background:linear-gradient(90deg,#ffb347,#ff5d6c)"></span></div>
   <span class=val id=gpuval>—</span></div>
 <div class=sub style=margin-top:6px>target: F1≥0.90 · GPU ≈143 pages/s/node (14% LLM coverage, 16 nodes, 2 days)</div>
</div>

<div class=grid style=margin-top:14px>
 <div class=card><h2>Pipeline stages (smoke 44k)</h2><div id=stages></div></div>
 <div class=card><h2>F1 journey</h2><div class=journey id=journey></div>
   <div class=sub style=margin-top:34px>0.025 → 0.51 → 0.81 → <span class=flash id=jnext>0.91?</span></div></div>
</div>

<div class="card full" style=margin-top:14px><h2>🔴 Live F1&gt;0.90 chain &amp; 🟣 optimization swarm</h2>
 <div id=chain class=sub></div>
 <div style=margin-top:10px id=swarm></div>
</div>

<div class="card full" style=margin-top:14px><h2>Slurm queue (live)</h2>
 <table><thead><tr><th>job</th><th>name</th><th>state</th><th>elapsed</th><th>node</th></tr></thead>
 <tbody id=q></tbody></table></div>

<div class="card full" style=margin-top:14px><h2>💬 Prompt the operator</h2>
 <textarea id=pin rows=2 placeholder="Type an instruction / hypothesis to queue (e.g. 'try FP8 next', 'lower cluster threshold to 0.9')…"></textarea>
 <button id=send>Send ▸</button> <span id=psaved class=flash></span>
 <div class=plist id=plist></div></div>

<div class=foot>Dripper×MinerU optimization · FastAPI · auto-polling /api/status</div>
</div>
<script>
const stages=[["1a feat",595,"ok"],["1b dbscan",150,"ok"],["1c prompt",88,"ok"],
 ["2 vLLM",30,"run"],["2b parse",95,"ok"],["3 propag",77,"ok"]];
const COL={ok:"#39d98a",run:"#4aa8ff",warn:"#ffb347",bad:"#ff5d6c",queue:"#7e8db0"};
const SW=[["H1 gpu-serving","OPTIMIZATION_ROADMAP.md"],["H2 fp8","FP8_PLAN.md"],
 ["H3 reduce-llm","REDUCE_LLM_LOAD_PLAN.md"],["H4 stage3-deep","STAGE3_DEEPER_PLAN.md"],
 ["H5 cpu-microopt","CPU_MICROOPT_PLAN.md"],["H6 e2e-model","E2E_THROUGHPUT_MODEL.md"],
 ["synth roadmap","OPTIMIZATION_ROADMAP.md"]];
function rstages(s){const max=600;document.getElementById('stages').innerHTML=stages.map(([n,r,st])=>
 `<div class=stage><span class=nm>${n}</span><div class="bar pb"><span style="width:${Math.min(100,r/max*100)}%;background:${COL[st]}"></span></div><span style="width:64px;text-align:right">${r} p/s</span></div>`).join('');}
function rjourney(){const J=[["v2",0.025],["s3",0.51],["chat",0.81],["fb-llm",0.91]];
 document.getElementById('journey').innerHTML=J.map(([l,v],i)=>
 `<div class=jb style="height:${v*100}%;${i==3?'opacity:.6;background:linear-gradient(180deg,#b06cff,#6c8cff)':''}"><b>${v}</b><i>${l}</i></div>`).join('');}
function num(s,re){const m=(s||'').match(re);return m?parseFloat(m[1]):null;}
async function tick(){
 let s;try{s=await (await fetch('/api/status')).json();}catch(e){return;}
 const age=Math.max(0,Math.round((Date.now()/1000)-(s.ts||0)));
 document.getElementById('age').textContent=age;
 document.getElementById('clock').textContent=new Date().toLocaleTimeString();
 document.getElementById('err').textContent=s.error?('⚠ '+s.error):'connected ✓';
 // F1 bar
 let f1=num(s.final_f1,/mean F1:\\s*([0-9.]+)/);
 if(f1==null)f1=0.81;
 document.getElementById('f1bar').style.width=Math.min(100,f1/0.90*100)+'%';
 document.getElementById('f1val').textContent=f1.toFixed(3)+(f1>=0.90?' ✅':' →0.90');
 // GPU bar — prefer new combined pipeline rate, fall back to at-scale kv-fp8 result
 let g=num(s.stage2_rate,/([0-9.]+)/)||num(s.gpu_pipeline_rate,/([0-9.]+)/)||num(s.s2rate_raw,/=([0-9.]+)/)||num(s.fb2,/([0-9.]+) pages\\/s/)||0;
 document.getElementById('gpubar').style.width=Math.min(100,g/143*100)+'%';
 const gpuLabel=g>=143?g.toFixed(0)+' / 143 p/s ✅':g>0?g.toFixed(0)+' / 143 p/s/node':'— / 143 p/s/node';
 document.getElementById('gpuval').textContent=gpuLabel;
 // chain — show v3 pipeline state
 const gpuTiming=s.gpu_pipeline_timing?('<br><span style=color:#7e8db0>⏱ '+s.gpu_pipeline_timing+'</span>'):'';
 const s3r=s.s3_rate?(' · Stage3 '+s.s3_rate):'';
 const fin=s.final_f1?('<b class=flash>'+s.final_f1+'</b>'):'<span style=color:#7e8db0>pending…</span>';
 document.getElementById('chain').innerHTML=
  `⚡ <b>E2E v3 pipeline</b> · GPU(1c+2+2b): <b>${g>0?g.toFixed(0)+' p/s/node':'running'}</b>${s3r} · F1: ${fin}`+
  gpuTiming+
  (s.f1_roles&&s.f1_roles.length?('<br><span style=color:#7e8db0>'+s.f1_roles.join(' · ')+'</span>'):'');
 // swarm
 document.getElementById('swarm').innerHTML='🟣 <b>swarm</b> '+SW.map(([n,d])=>{
   const done=s.docs&&s.docs[d];return `<span class=chip>${done?'✅':'⚙'} ${n}</span>`;}).join('');
 // queue
 document.getElementById('q').innerHTML=(s.queue||[]).map(j=>{
   const c=j.state=='RUNNING'?COL.run:COL.queue;
   return `<tr><td>${j.id}</td><td>${j.name}</td><td><span class=dot style="background:${c}"></span>${j.state}</td><td>${j.time}</td><td>${j.node}</td></tr>`;}).join('')
   ||'<tr><td colspan=5 style=color:#7e8db0>no jobs queued</td></tr>';
}
async function rprompts(){const r=await (await fetch('/api/prompts')).json();
 document.getElementById('plist').innerHTML=r.slice().reverse().map(p=>
 `<div><span class=t>${p.ts}</span><br>${p.text.replace(/</g,'&lt;')}</div>`).join('');}
document.getElementById('send').onclick=async()=>{
 const t=document.getElementById('pin').value.trim();if(!t)return;
 await fetch('/api/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
 document.getElementById('pin').value='';
 document.getElementById('psaved').textContent='queued ✓';setTimeout(()=>document.getElementById('psaved').textContent='',2000);
 rprompts();};
rjourney();rstages();tick();rprompts();setInterval(tick,4000);setInterval(rprompts,6000);
</script></body></html>"""


CHAT_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Claude · Dripper Mission Control</title>
<style>
:root{--bg:#0A0C10;--panel:#14171F;--panel2:#0E1117;--line:#222838;--txt:#e6edf7;
--mut:#7e8db0;--accent:#27e0c4;--purp:#b06cff;--user:#1b2740;--bot:#121a2b}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:radial-gradient(1200px 600px at 50% -10%,#101826,#0A0C10);
font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--txt);display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid var(--line);
background:rgba(10,12,16,.8);backdrop-filter:blur(8px);position:sticky;top:0}
header b{font-size:15px;letter-spacing:.4px}.tag{color:var(--mut);font-size:12px}
header a{margin-left:auto;color:var(--accent);text-decoration:none;font-size:13px;border:1px solid var(--line);
padding:6px 12px;border-radius:8px}header a:hover{background:var(--panel)}
#feed{flex:1;overflow:auto;padding:22px;max-width:920px;width:100%;margin:0 auto}
.msg{display:flex;gap:12px;margin:16px 0;animation:rise .25s ease}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.av{width:30px;height:30px;border-radius:8px;flex:none;display:grid;place-items:center;font-size:13px;font-weight:700}
.u .av{background:linear-gradient(135deg,#2a3c66,#1b2740);color:#bcd}
.a .av{background:linear-gradient(135deg,var(--purp),#6c8cff);color:#fff}
.bub{background:var(--bot);border:1px solid var(--line);border-radius:12px;padding:12px 14px;max-width:100%;overflow:auto}
.u .bub{background:var(--user)}
.bub pre{background:#0a0f1a;border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font-size:12.5px}
.bub code{background:#0a0f1a;padding:1px 5px;border-radius:5px}
.meta{color:var(--mut);font-size:11px;margin-top:6px}
.think{color:var(--mut);font-style:italic}
.think:after{content:'';animation:dots 1.4s steps(4,end) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
footer{border-top:1px solid var(--line);padding:14px 18px;background:rgba(10,12,16,.9)}
.box{max-width:920px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
#in{flex:1;background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:12px;
padding:12px;resize:none;font:inherit;max-height:200px;min-height:46px}
#in:focus{outline:none;border-color:var(--purp)}
#go{background:linear-gradient(135deg,var(--purp),#6c8cff);border:0;color:#fff;padding:12px 18px;
border-radius:12px;cursor:pointer;font-weight:700}#go:disabled{opacity:.5;cursor:not-allowed}
.hint{max-width:920px;margin:6px auto 0;color:var(--mut);font-size:11px}
.empty{color:var(--mut);text-align:center;margin-top:60px}
</style></head><body>
<header><b>💬 Claude</b><span class=tag>headless CLI bridge · this repo · continuous session</span>
 <a href="/">← dashboard</a></header>
<div id=feed><div class=empty>Ask anything about the pipeline, the optimization run, the code, or the targets.<br>
 e.g. <i>"summarize the optimization roadmap"</i> · <i>"what's the F1 gap and how do we close it?"</i></div></div>
<footer><div class=box>
 <textarea id=in placeholder="Message Claude…  (⌘/Ctrl+Enter to send)"></textarea>
 <button id=go>Send ▸</button></div>
 <div class=hint>Separate headless session — it can read the repo &amp; advise; it won't edit files or submit jobs unless you ask.</div>
</footer>
<script>
const feed=document.getElementById('feed'),inp=document.getElementById('in'),go=document.getElementById('go');
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function md(s){s=esc(s);
 s=s.replace(/```([\\s\\S]*?)```/g,(m,c)=>'<pre>'+c.replace(/^\\n/,'')+'</pre>');
 s=s.replace(/`([^`]+)`/g,'<code>$1</code>');
 s=s.replace(/\\*\\*([^*]+)\\*\\*/g,'<b>$1</b>');
 return s.replace(/\\n/g,'<br>');}
function add(role,html,meta){
 const wrap=document.createElement('div');wrap.className='msg '+(role=='user'?'u':'a');
 wrap.innerHTML=`<div class=av>${role=='user'?'you':'✦'}</div><div><div class=bub>${html}</div>${meta?('<div class=meta>'+meta+'</div>'):''}</div>`;
 if(feed.querySelector('.empty'))feed.innerHTML='';
 feed.appendChild(wrap);feed.scrollTop=feed.scrollHeight;return wrap;}
async function hist(){try{const r=await (await fetch('/api/chat/history')).json();
 if(r.length){feed.innerHTML='';r.forEach(m=>{add('user',md(m.user));
  add('assistant',md(m.assistant),`${m.ts} · ${m.elapsed_s||'?'}s${m.cost_usd?(' · $'+m.cost_usd.toFixed(3)):''}`);});}}catch(e){}}
async function send(){const t=inp.value.trim();if(!t)return;
 inp.value='';inp.style.height='46px';go.disabled=true;
 add('user',md(t));
 const pend=add('assistant','<span class=think>thinking</span>');
 try{const r=await (await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({message:t})})).json();
  if(r.ok){pend.querySelector('.bub').innerHTML=md(r.assistant);
   pend.querySelector('div').insertAdjacentHTML('beforeend',
    `<div class=meta>${r.ts} · ${r.elapsed_s}s${r.cost_usd?(' · $'+r.cost_usd.toFixed(3)):''}${r.turns?(' · '+r.turns+' turns'):''}</div>`);}
  else{pend.querySelector('.bub').innerHTML='<span style=color:#ff5d6c>⚠ '+esc(r.error||'error')+'</span>';}
 }catch(e){pend.querySelector('.bub').innerHTML='<span style=color:#ff5d6c>⚠ network error</span>';}
 feed.scrollTop=feed.scrollHeight;go.disabled=false;inp.focus();}
go.onclick=send;
inp.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();send();}});
inp.addEventListener('input',()=>{inp.style.height='46px';inp.style.height=Math.min(200,inp.scrollHeight)+'px';});
hist();inp.focus();
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn

    threading.Thread(target=refresh_loop, daemon=True).start()
    print("Dashboard → http://127.0.0.1:8765", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
