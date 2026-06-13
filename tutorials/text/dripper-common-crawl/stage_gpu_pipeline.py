#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Combined Stage 1c + Stage 2 + Stage 2b in a single GPU job.

Eliminates two intermediate parquet round-trips and two Slurm queue waits.
INPUT:  Stage 1b output dir. OUTPUT: combined parquet with Stage 2b schema.
RUNS ON: batch GPU partition (8xH100). Replaces JOB1c + JOB2 + JOB2b.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_metrics import StageMetrics

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "mapping_json",
    "dripper_content",
    "dripper_html",
    "dripper_error",
    "inference_time_s",
]

_STAGE1C_BINDINGS = None
_STAGE2B_BINDINGS_LOADED = False
_ITEM_ID_RE = None


def _load_stage1c_bindings():
    global _STAGE1C_BINDINGS, _ITEM_ID_RE
    import re as _re

    _ITEM_ID_RE = _re.compile(r"_item_id")
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from nemo_curator.stages.text.experimental.dripper.stage import _load_mineru_html_bindings

    _STAGE1C_BINDINGS = _load_mineru_html_bindings()


def _get_attr(case, attr: str) -> str:
    for data in (getattr(case, "process_data", None), getattr(case, "output_data", None)):
        if data is not None:
            val = getattr(data, attr, None)
            if val:
                return str(val)
    return ""


def _preprocess_one(rec: dict) -> dict:
    url = rec.get("url", "")
    html = rec.get("html") or ""
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    out = {
        k: rec.get(k, "")
        for k in [
            "url",
            "url_host_name",
            "cluster_id",
            "cluster_role",
            "warc_filename",
            "warc_record_offset",
            "warc_record_length",
        ]
    }
    out.update({"prompt": "", "item_count": 0, "simp_html": "", "map_html": "", "html": html})
    if not _STAGE1C_BINDINGS or not html.strip():
        return out
    try:
        M = _STAGE1C_BINDINGS
        case = M.case_cls(M.input_cls(raw_html=html, url=url))
        case = M.simplify_single_input(case)
        simp_html = _get_attr(case, "simpled_html")
        map_html = _get_attr(case, "map_html")
        case = M.build_prompt(case, "short_compact")
        gen_in = getattr(case, "generate_input", None)
        prompt = str(gen_in.full_prompt) if gen_in and gen_in.full_prompt else ""
        item_count = len(_ITEM_ID_RE.findall(map_html or simp_html or ""))
        out.update({"prompt": prompt, "item_count": item_count, "simp_html": simp_html, "map_html": map_html})
    except Exception as exc:
        out["prompt"] = f"ERROR:{type(exc).__name__}:{str(exc)[:100]}"
    return out


def run_stage1c(df: pd.DataFrame) -> pd.DataFrame:
    _load_stage1c_bindings()
    print(f"[gpu-pipeline] Stage 1c: preprocessing {len(df):,} pages", flush=True)
    t0 = time.perf_counter()
    results = [_preprocess_one(r) for r in df.to_dict("records")]
    elapsed = time.perf_counter() - t0
    result_df = pd.DataFrame(results)
    ok = (result_df["prompt"].astype(str).str.len() > 10).sum()
    print(f"[gpu-pipeline] Stage 1c done: {ok:,}/{len(df):,} prompts built in {elapsed:.1f}s", flush=True)
    return result_df


def _chat_format(tok, prompt: str, supports_think: list[bool]) -> str:
    msgs = [{"role": "user", "content": prompt}]
    if supports_think[0]:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            supports_think[0] = False
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_stage2_worker(
    gpu_id: int,
    slice_path: str,
    out_path: str,
    model: str,
    gpu_mem_util: float,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    max_tokens: int,
    kv_cache_dtype: str,
) -> None:
    """One GPU worker: offline-batched LLM.generate over its prompt slice."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pq.ParquetFile(slice_path).read().to_pandas()
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm_kw = dict(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    if kv_cache_dtype and kv_cache_dtype != "auto":
        llm_kw["kv_cache_dtype"] = kv_cache_dtype
    t_setup = time.perf_counter()
    llm = LLM(**llm_kw)
    setup_s = time.perf_counter() - t_setup
    rows = df.to_dict("records")
    supports_think = [True]
    prompts, samplings, ridx, results, n_trunc = [], [], [], [None] * len(rows), 0

    for i, r in enumerate(rows):
        p = str(r.get("prompt", "") or "")
        if not p or p.startswith("ERROR:"):
            results[i] = {
                **r,
                "llm_response": "",
                "dripper_error": p if p.startswith("ERROR:") else "empty_prompt",
                "inference_time_s": 0.0,
            }
            continue
        try:
            ic = int(r.get("item_count", 0) or 0)
        except (TypeError, ValueError):
            ic = 0
        max_tok = min(max_tokens, max(32, ic * 6 + 16) if ic > 0 else max_tokens)
        text = _chat_format(tok, p, supports_think)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cap = max_model_len - max_tok - 8
        if len(ids) > cap:
            ids = ids[:cap]
            n_trunc += 1
        prompts.append({"prompt_token_ids": ids})
        samplings.append(SamplingParams(temperature=0.0, max_tokens=max_tok))
        ridx.append(i)

    print(
        f"[gpu-pipeline gpu{gpu_id}] Stage 2: {len(prompts)} prompts ({n_trunc} truncated) setup={setup_s:.1f}s",
        flush=True,
    )
    t1 = time.perf_counter()
    outs = llm.generate(prompts, samplings) if prompts else []
    infer_s = time.perf_counter() - t1

    for j, o in enumerate(outs):
        i = ridx[j]
        resp = o.outputs[0].text if o.outputs else ""
        results[i] = {
            **rows[i],
            "llm_response": resp,
            "dripper_error": "" if resp else "empty_response",
            "inference_time_s": infer_s / max(len(outs), 1),
        }

    pd.DataFrame([x for x in results if x is not None]).to_parquet(out_path, index=False, compression="snappy")
    rate = len(prompts) / max(infer_s, 1e-6)
    Path(out_path + ".meta.json").write_text(
        json.dumps(
            {
                "infer_s": round(infer_s, 2),
                "setup_s": round(setup_s, 2),
                "pages": len([x for x in results if x]),
                "rate_gpu": round(rate, 2),
            }
        )
    )
    print(
        f"[gpu-pipeline gpu{gpu_id}] Stage 2 DONE {len(prompts)} pages {rate:.1f} pages/s/GPU infer={infer_s:.1f}s",
        flush=True,
    )


def run_stage2(df: pd.DataFrame, args) -> pd.DataFrame:
    """Dispatch Stage 2 across all GPUs (LPT balanced, offline batched)."""
    n_gpus = args.replicas if args.replicas > 0 else _detect_gpus()
    print(f"[gpu-pipeline] Stage 2: {len(df):,} pages over {n_gpus} GPUs", flush=True)
    tmp = Path(args.output) / "_gpu_slices"
    tmp.mkdir(parents=True, exist_ok=True)
    cost = df["prompt"].astype(str).str.len().to_numpy()
    order = sorted(range(len(df)), key=lambda i: -cost[i])
    bins: list[list[int]] = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for i in order:
        g = min(range(n_gpus), key=lambda k: load[k])
        bins[g].append(i)
        load[g] += int(cost[i])

    slice_paths, out_paths = [], []
    for g in range(n_gpus):
        sp = str(tmp / f"slice_{g}.parquet")
        op = str(tmp / f"out_{g}.parquet")
        df.iloc[bins[g]].to_parquet(sp, index=False)
        slice_paths.append(sp)
        out_paths.append(op)
    t0 = time.perf_counter()
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--gpu",
                str(g),
                "--slice",
                slice_paths[g],
                "--slice-out",
                out_paths[g],
                "--model",
                args.model,
                "--max-tokens",
                str(args.max_tokens),
                "--gpu-mem-util",
                str(args.gpu_mem_util),
                "--max-model-len",
                str(args.max_model_len),
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--kv-cache-dtype",
                args.kv_cache_dtype,
            ]
        )
        for g in range(n_gpus)
    ]
    rcs = [p.wait() for p in procs]
    print(f"[gpu-pipeline] Stage 2 workers done in {time.perf_counter() - t0:.1f}s codes={rcs}", flush=True)
    frames = [pq.ParquetFile(op).read().to_pandas() for op in out_paths if Path(op).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _detect_gpus() -> int:
    n = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS_PER_NODE", "")
    if n:
        try:
            return int(n.split(":")[-1])
        except ValueError:
            pass
    try:
        r = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=5)
        return max(1, sum(1 for ln in r.stdout.splitlines() if ln.startswith("GPU")))
    except Exception:
        return 1


_STAGE2B_W = None
_STAGE2B_M = None
_STRIP_XML = None
_LABELS_TO_WEBKIT = None
_FALLBACK_HANDLER = None


def _load_stage2b_bindings():
    global _STAGE2B_W, _STAGE2B_M, _STRIP_XML, _LABELS_TO_WEBKIT, _FALLBACK_HANDLER
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from nemo_curator.stages.text.experimental.dripper.stage import (
        _labels_to_webkit_response,
        _load_llm_web_kit_bindings,
        _load_mineru_html_bindings,
        _strip_xml_incompatible_chars,
    )

    _STAGE2B_W = _load_llm_web_kit_bindings()
    _STAGE2B_M = _load_mineru_html_bindings()
    _STRIP_XML = _strip_xml_incompatible_chars
    _LABELS_TO_WEBKIT = _labels_to_webkit_response
    try:
        _FALLBACK_HANDLER = _STAGE2B_M.get_fallback_handler("trafilatura")
    except Exception:
        _FALLBACK_HANDLER = None


def _trafilatura_content(raw_html: str, url: str) -> str:
    if not _FALLBACK_HANDLER or not _STAGE2B_M or not raw_html.strip():
        return ""
    try:
        M = _STAGE2B_M
        case = M.case_cls(M.input_cls(raw_html=raw_html, url=url))
        case = M.extract_main_html_fallback(case, fallback_handler=_FALLBACK_HANDLER)
        od = getattr(case, "output_data", None)
        if od and _STRIP_XML and isinstance(getattr(od, "main_html", None), str):
            od.main_html = _STRIP_XML(od.main_html)
        case = M.convert2content(case, output_format="mm_md")
        od = getattr(case, "output_data", None)
        return str(getattr(od, "main_content", "") or "") if od else ""
    except Exception:
        return ""


def _postprocess_one(rec: dict) -> dict:
    url = rec.get("url", "")
    raw_html = rec.get("html") or ""
    simp_html = rec.get("simp_html") or ""
    map_html = rec.get("map_html") or ""
    llm_response = rec.get("llm_response") or ""
    role = str(rec.get("cluster_role", "") or "")

    out = {
        "url": url,
        "url_host_name": rec.get("url_host_name", ""),
        "cluster_id": rec.get("cluster_id", ""),
        "cluster_role": role,
        "mapping_json": "",
        "dripper_content": "",
        "dripper_html": "",
        "dripper_error": rec.get("dripper_error", "") or "",
        "inference_time_s": rec.get("inference_time_s", 0.0),
    }

    if not _STAGE2B_W or not _STAGE2B_M or not llm_response:
        if not llm_response:
            out["dripper_error"] = out["dripper_error"] or "no_llm_response"
            out["dripper_content"] = _trafilatura_content(raw_html, url)
        return out

    M = _STAGE2B_M
    try:
        case = M.case_cls(M.input_cls(raw_html=raw_html, url=url))
        if simp_html or map_html:
            case.process_data = M.process_data_cls(simpled_html=simp_html, map_html=map_html)
        case.generate_output = M.generate_output_cls(response=llm_response)
        webkit_response: dict = {}
        try:
            case = M.parse_result(case)
            if _LABELS_TO_WEBKIT is not None:
                webkit_response = _LABELS_TO_WEBKIT(getattr(case.parse_result, "item_label", {}))
            case = M.extract_main_html_single(case)
        except Exception as exc:
            out["dripper_error"] = f"primary_failed:{type(exc).__name__}:{str(exc)[:70]}"
            if _FALLBACK_HANDLER is not None:
                try:
                    case = M.extract_main_html_fallback(case, fallback_handler=_FALLBACK_HANDLER)
                except Exception as fexc:
                    out["dripper_error"] += f"; fb:{str(fexc)[:50]}"
        od = getattr(case, "output_data", None)
        if od and _STRIP_XML and isinstance(getattr(od, "main_html", None), str):
            od.main_html = _STRIP_XML(od.main_html)
        try:
            case = M.convert2content(case, output_format="mm_md")
        except Exception as exc:
            out["dripper_error"] = out["dripper_error"] or f"convert:{type(exc).__name__}:{str(exc)[:70]}"
        od = getattr(case, "output_data", None)
        out["dripper_html"] = str(getattr(od, "main_html", "") or "") if od else ""
        out["dripper_content"] = str(getattr(od, "main_content", "") or "") if od else ""
        if not out["dripper_content"].strip():
            out["dripper_content"] = _trafilatura_content(raw_html, url)
        if role == "representative" and _STAGE2B_W is not None:
            try:
                template = _STAGE2B_W.map_parser_cls({}).parse(
                    {
                        "typical_raw_html": raw_html,
                        "typical_raw_tag_html": map_html or simp_html,
                        "llm_response": webkit_response,
                    }
                )
                out["mapping_json"] = base64.b64encode(pickle.dumps(template)).decode("ascii")
            except Exception as exc:
                out["dripper_error"] = out["dripper_error"] or f"map_parser:{type(exc).__name__}:{str(exc)[:70]}"
    except Exception as exc:
        out["dripper_error"] = f"postprocess:{type(exc).__name__}:{str(exc)[:150]}"
    return out


class _Stage2bPostprocessStage:
    """NeMo Curator ProcessingStage for Stage 2b postprocessing.

    Wraps _postprocess_one as a Curator ProcessingStage so RayDataExecutor
    distributes the CPU-bound work across all available cores.  Each Ray actor
    initialises the heavy llm-webkit + mineru-html bindings once in setup(),
    then processes batches of DocumentBatch tasks.
    """

    # Imported lazily to keep the GPU-venv import surface minimal
    _stage_cls = None

    @staticmethod
    def _build():
        """Return the concrete ProcessingStage subclass, importing Curator lazily."""
        if _Stage2bPostprocessStage._stage_cls is not None:
            return _Stage2bPostprocessStage._stage_cls

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import DocumentBatch as _DocumentBatch

        class Stage2bPostprocessStage(ProcessingStage[_DocumentBatch, _DocumentBatch]):
            name = "stage2b_postprocess"
            resources = Resources(cpus=1.0)  # one CPU core per actor
            batch_size = 128

            def num_workers(self):
                # Leave 2 CPUs free: 1 for the main process, 1 buffer
                return max(1, (os.cpu_count() or 4) - 2)

            def setup(self, _worker_metadata=None):
                # Called once per Ray actor — triggers actor mode in RayDataStageAdapter
                # and initialises the heavy bindings once per worker process.
                _load_stage2b_bindings()

            def process_batch(self, tasks):
                results = []
                for task in tasks:
                    df = task.to_pandas()
                    processed = pd.DataFrame([_postprocess_one(r) for r in df.to_dict("records")])
                    results.append(_DocumentBatch(dataset_name=task.dataset_name, data=processed))
                return results

        _Stage2bPostprocessStage._stage_cls = Stage2bPostprocessStage
        return Stage2bPostprocessStage


def run_stage2b(df: pd.DataFrame) -> pd.DataFrame:
    """Run Stage 2b postprocessing parallelised via NeMo Curator RayDataExecutor.

    Splits the DataFrame into per-CPU chunks, wraps each as a DocumentBatch,
    and executes through a ProcessingStage so RayDataExecutor distributes work
    across all available CPU cores on the GPU node.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from nemo_curator.backends.ray_data import RayDataExecutor
    from nemo_curator.tasks import DocumentBatch

    n_workers = max(1, (os.cpu_count() or 4) - 2)
    print(
        f"[gpu-pipeline] Stage 2b: postprocessing {len(df):,} pages via RayDataExecutor ({n_workers} CPU workers)",
        flush=True,
    )
    t0 = time.perf_counter()

    # Split into per-worker chunks so each actor gets a roughly equal share
    chunk = max(1, len(df) // n_workers)
    initial_tasks = [
        DocumentBatch(dataset_name="stage2b", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    stage_cls = _Stage2bPostprocessStage._build()
    executor = RayDataExecutor()
    output_tasks = executor.execute([stage_cls()], initial_tasks=initial_tasks)

    result_df = pd.concat([t.to_pandas() for t in output_tasks], ignore_index=True)
    elapsed = time.perf_counter() - t0
    content_ok = (result_df["dripper_content"].astype(str).str.len() > 5).sum()
    mapping_ok = (result_df["mapping_json"].astype(str).str.len() > 5).sum()
    print(
        f"[gpu-pipeline] Stage 2b done: content_ok={content_ok:,} mapping_ok={mapping_ok:,} "
        f"in {elapsed:.1f}s ({len(df) / max(elapsed, 1):.1f} p/s)",
        flush=True,
    )
    return result_df


def run(args):
    tracker = StageMetrics(
        "stage_gpu_pipeline",
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        n_gpus=args.replicas or _detect_gpus(),
    )
    tracker.start()
    t_total = time.perf_counter()
    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        inp = exact if exact.exists() else sorted(inp.glob("shard_*.parquet"))[0]
    all_df = pq.ParquetFile(str(inp)).read().to_pandas()
    if "cluster_role" in all_df.columns:
        rep_df = all_df[all_df["cluster_role"].isin(["representative", "singleton"])].reset_index(drop=True)
    else:
        rep_df = all_df.reset_index(drop=True)
    print(
        f"[gpu-pipeline] {len(rep_df):,} reps/singletons from {len(all_df):,} total pages "
        f"({len(rep_df) / max(len(all_df), 1) * 100:.1f}% LLM fraction)",
        flush=True,
    )

    t1c = time.perf_counter()
    rep_df = run_stage1c(rep_df)
    t1c_s = time.perf_counter() - t1c

    t2 = time.perf_counter()
    infer_df = run_stage2(rep_df, args)
    t2_s = time.perf_counter() - t2

    t2b = time.perf_counter()
    passthrough_df = rep_df[["url"] + [c for c in ["simp_html", "map_html", "html"] if c in rep_df.columns]]
    infer_df = infer_df.merge(passthrough_df, on="url", how="left", suffixes=("", "_1c"))
    for c in ["simp_html", "map_html", "html"]:
        if f"{c}_1c" in infer_df.columns:
            infer_df[c] = infer_df[c].fillna(infer_df[f"{c}_1c"])
            infer_df.drop(columns=[f"{c}_1c"], inplace=True)
    result_df = run_stage2b(infer_df)
    t2b_s = time.perf_counter() - t2b

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "pipeline_results.parquet")
    for col in OUTPUT_COLS:
        if col not in result_df.columns:
            result_df[col] = None
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    total_s = time.perf_counter() - t_total
    ok = int((result_df["dripper_content"].astype(str).str.len() > 5).sum())
    print(
        f"[gpu-pipeline] ALL DONE: {len(result_df):,} pages ok={ok} "
        f"total={total_s:.1f}s (1c={t1c_s:.1f}s 2={t2_s:.1f}s 2b={t2b_s:.1f}s) → {out_path}",
        flush=True,
    )

    tracker.finish(
        total_pages=len(result_df), errors=int((result_df["dripper_error"].astype(str).str.len() > 2).sum())
    )
    tracker.extra = {
        "stage1c_s": round(t1c_s, 1),
        "stage2_s": round(t2_s, 1),
        "stage2b_s": round(t2b_s, 1),
        "content_ok": ok,
    }
    tracker.save(args.output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--slice")
    p.add_argument("--slice-out")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--replicas", type=int, default=int(os.environ.get("N_GPU_REPLICAS", "0")))
    p.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--hf-cache", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--kv-cache-dtype", default="fp8")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_cache)

    if args.worker:
        run_stage2_worker(
            args.gpu,
            args.slice,
            args.slice_out,
            args.model,
            args.gpu_mem_util,
            args.max_model_len,
            args.max_num_seqs,
            args.max_num_batched_tokens,
            args.max_tokens,
            args.kv_cache_dtype,
        )
    else:
        if not args.input or not args.output:
            p.error("--input and --output required in main mode")
        run(args)


if __name__ == "__main__":
    main()
