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
import os
import pickle
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
_REPO_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
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

# Magic-number constants (PLR2004)
_MIN_CONTENT_LEN = 5
_MIN_ERROR_LEN = 2
_MIN_PROMPT_LEN = 10

# Single registry for lazily-loaded bindings (replaces multiple module-level globals).
_BINDINGS: dict[str, object] = {}


def _load_stage1c_bindings() -> None:
    import re as _re

    _BINDINGS["item_id_re"] = _re.compile(r"_item_id")
    from nemo_curator.stages.text.experimental.dripper.stage import _load_mineru_html_bindings

    _BINDINGS["stage1c"] = _load_mineru_html_bindings()


def _get_attr(case: object, attr: str) -> str:
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
    _b = _BINDINGS.get("stage1c")
    if not _b or not html.strip():
        return out
    try:
        case = _b.case_cls(_b.input_cls(raw_html=html, url=url))  # type: ignore[union-attr]
        case = _b.simplify_single_input(case)  # type: ignore[union-attr]
        simp_html = _get_attr(case, "simpled_html")
        map_html = _get_attr(case, "map_html")
        case = _b.build_prompt(case, "short_compact")  # type: ignore[union-attr]
        gen_in = getattr(case, "generate_input", None)
        prompt = str(gen_in.full_prompt) if gen_in and gen_in.full_prompt else ""
        _re = _BINDINGS.get("item_id_re")
        item_count = len(_re.findall(map_html or simp_html or "")) if _re else 0  # type: ignore[union-attr]
        out.update({"prompt": prompt, "item_count": item_count, "simp_html": simp_html, "map_html": map_html})
    except Exception as exc:
        out["prompt"] = f"ERROR:{type(exc).__name__}:{str(exc)[:100]}"
    return out


_STAGE_CLS_CACHE: dict = {}


def _make_stage_cls(stage_name: str, setup_fn: Callable, process_fn: Callable) -> type:
    """Build a NeMo ProcessingStage class, cached by stage_name."""
    if stage_name in _STAGE_CLS_CACHE:
        return _STAGE_CLS_CACHE[stage_name]
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.tasks import DocumentBatch as _DocumentBatch

    class _Stage(ProcessingStage[_DocumentBatch, _DocumentBatch]):
        name = stage_name
        resources = Resources(cpus=1.0)
        batch_size = 1

        def num_workers(self) -> int:
            return max(1, (os.cpu_count() or 4) - 2)

        def setup(self, _worker_metadata: object = None) -> None:
            setup_fn()

        def process(self, task: object) -> object:
            return self.process_batch([task])[0]

        def process_batch(self, tasks: list) -> list:
            return [
                _DocumentBatch(
                    dataset_name=t.dataset_name,
                    data=pd.DataFrame([process_fn(r) for r in t.to_pandas().to_dict("records")]),
                )
                for t in tasks
            ]

    _STAGE_CLS_CACHE[stage_name] = _Stage
    return _Stage


def run_stage1c(df: pd.DataFrame) -> pd.DataFrame:
    """Run Stage 1c HTML preprocessing via RayActorPoolExecutor."""
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import DocumentBatch

    n_workers = max(1, (os.cpu_count() or 4) - 2)
    t0 = time.perf_counter()
    chunk = max(1, len(df) // n_workers)
    initial_tasks = [
        DocumentBatch(dataset_name="stage1c", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    stage_cls = _make_stage_cls("stage1c_preprocess", _load_stage1c_bindings, _preprocess_one)
    pipeline = Pipeline(name="stage1c")
    pipeline.add_stage(stage_cls())
    output_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=initial_tasks) or []

    result_df = pd.concat([t.to_pandas() for t in output_tasks], ignore_index=True)
    elapsed = time.perf_counter() - t0
    ok = (result_df["prompt"].astype(str).str.len() > _MIN_PROMPT_LEN).sum()
    print(f"[gpu-pipeline] Stage 1c: {ok:,}/{len(df):,} prompts in {elapsed:.1f}s", flush=True)
    return result_df


def _chat_format(tok: object, prompt: str, supports_think: list[bool]) -> str:
    msgs = [{"role": "user", "content": prompt}]
    if supports_think[0]:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)  # type: ignore[union-attr]
        except TypeError:
            supports_think[0] = False
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # type: ignore[union-attr]


@dataclass
class _WorkerConfig:
    """GPU worker configuration (groups the 7 LLM/vLLM knobs)."""

    model: str
    gpu_mem_util: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_tokens: int
    kv_cache_dtype: str


def _build_worker_prompts(
    rows: list[dict],
    tok: object,
    max_model_len: int,
    max_tokens: int,
) -> tuple[list, list, list, list, int]:
    """Tokenize and budget prompts for offline vLLM generation.

    Returns (prompts, samplings, ridx, results, n_trunc).
    """
    from vllm import SamplingParams

    supports_think: list[bool] = [True]
    prompts: list = []
    samplings: list = []
    ridx: list = []
    results: list = [None] * len(rows)
    n_trunc = 0

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
        ids = tok(text, add_special_tokens=False)["input_ids"]  # type: ignore[operator]
        cap = max_model_len - max_tok - 8
        if len(ids) > cap:
            ids = ids[:cap]
            n_trunc += 1
        prompts.append({"prompt_token_ids": ids})
        samplings.append(SamplingParams(temperature=0.0, max_tokens=max_tok))
        ridx.append(i)

    return prompts, samplings, ridx, results, n_trunc


def run_stage2_worker(gpu_id: int, slice_path: str, out_path: str, cfg: _WorkerConfig) -> None:
    """One GPU worker: offline-batched LLM.generate over its prompt slice."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from nemo_curator.utils.vllm_utils import pick_free_port, resolve_local_model_path

    local_model = resolve_local_model_path(cfg.model)

    from transformers import AutoTokenizer
    from vllm import LLM

    df = pq.ParquetFile(slice_path).read().to_pandas()
    tok = AutoTokenizer.from_pretrained(local_model, trust_remote_code=True)
    llm_kw: dict = {
        "model": local_model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": cfg.gpu_mem_util,
        "max_model_len": cfg.max_model_len,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    if cfg.kv_cache_dtype and cfg.kv_cache_dtype != "auto":
        llm_kw["kv_cache_dtype"] = cfg.kv_cache_dtype

    t_setup = time.perf_counter()
    os.environ["MASTER_PORT"] = str(pick_free_port())
    llm = LLM(**llm_kw)
    setup_s = time.perf_counter() - t_setup

    rows = df.to_dict("records")
    prompts, samplings, ridx, results, n_trunc = _build_worker_prompts(rows, tok, cfg.max_model_len, cfg.max_tokens)

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
    print(
        f"[gpu-pipeline gpu{gpu_id}] DONE {len(prompts)} prompts ({n_trunc} trunc)"
        f" setup={setup_s:.1f}s infer={infer_s:.1f}s {rate:.1f} pages/s/GPU",
        flush=True,
    )


def run_stage2(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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

    _GPU_SLICE_COLS = ["url", "prompt", "item_count", "cluster_id", "cluster_role", "url_host_name"]
    slice_paths, out_paths = [], []
    for g in range(n_gpus):
        sp = str(tmp / f"slice_{g}.parquet")
        op = str(tmp / f"out_{g}.parquet")
        slice_df = df[[c for c in _GPU_SLICE_COLS if c in df.columns]].iloc[bins[g]]
        slice_df.to_parquet(sp, index=False)
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
    except OSError:
        return 1


def _load_stage2b_bindings() -> None:
    from nemo_curator.stages.text.experimental.dripper.stage import (
        _labels_to_webkit_response,
        _load_llm_web_kit_bindings,
        _load_mineru_html_bindings,
        _strip_xml_incompatible_chars,
    )

    _BINDINGS["stage2b_w"] = _load_llm_web_kit_bindings()
    _BINDINGS["stage2b_m"] = _load_mineru_html_bindings()
    _BINDINGS["strip_xml"] = _strip_xml_incompatible_chars
    _BINDINGS["labels_to_webkit"] = _labels_to_webkit_response
    try:
        _BINDINGS["fallback"] = _BINDINGS["stage2b_m"].get_fallback_handler("trafilatura")  # type: ignore[union-attr]
    except AttributeError:
        _BINDINGS["fallback"] = None


def _trafilatura_content(raw_html: str, url: str) -> str:
    _fallback = _BINDINGS.get("fallback")
    _b = _BINDINGS.get("stage2b_m")
    if not _fallback or not _b or not raw_html.strip():
        return ""
    try:
        case = _b.case_cls(_b.input_cls(raw_html=raw_html, url=url))  # type: ignore[union-attr]
        case = _b.extract_main_html_fallback(case, fallback_handler=_fallback)  # type: ignore[union-attr]
        od = getattr(case, "output_data", None)
        _strip_xml = _BINDINGS.get("strip_xml")
        if od and _strip_xml and isinstance(getattr(od, "main_html", None), str):
            od.main_html = _strip_xml(od.main_html)  # type: ignore[operator]
        case = _b.convert2content(case, output_format="mm_md")  # type: ignore[union-attr]
        od = getattr(case, "output_data", None)
        return str(getattr(od, "main_content", "") or "") if od else ""
    except Exception:
        return ""


def _apply_webkit_template(
    out: dict,
    role: str,
    raw_html: str,
    map_html: str,
    simp_html: str,
    webkit_response: dict,
) -> None:
    """Fill out['mapping_json'] for representative pages via map_parser."""
    _w = _BINDINGS.get("stage2b_w")
    if role != "representative" or _w is None:
        return
    try:
        template = _w.map_parser_cls({}).parse(  # type: ignore[union-attr]
            {
                "typical_raw_html": raw_html,
                "typical_raw_tag_html": map_html or simp_html,
                "llm_response": webkit_response,
            }
        )
        out["mapping_json"] = base64.b64encode(pickle.dumps(template)).decode("ascii")
    except Exception as exc:
        out["dripper_error"] = out["dripper_error"] or f"map_parser:{type(exc).__name__}:{str(exc)[:70]}"


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

    _b = _BINDINGS.get("stage2b_m")
    if not _BINDINGS.get("stage2b_w") or not _b or not llm_response:
        if not llm_response:
            out["dripper_error"] = out["dripper_error"] or "no_llm_response"
            out["dripper_content"] = _trafilatura_content(raw_html, url)
        return out

    try:
        case = _b.case_cls(_b.input_cls(raw_html=raw_html, url=url))  # type: ignore[union-attr]
        if simp_html or map_html:
            case.process_data = _b.process_data_cls(simpled_html=simp_html, map_html=map_html)  # type: ignore[union-attr]
        case.generate_output = _b.generate_output_cls(response=llm_response)  # type: ignore[union-attr]
        webkit_response: dict = {}
        try:
            case = _b.parse_result(case)  # type: ignore[union-attr]
            _labels_to_webkit = _BINDINGS.get("labels_to_webkit")
            if _labels_to_webkit is not None:
                webkit_response = _labels_to_webkit(getattr(case.parse_result, "item_label", {}))  # type: ignore[operator]
            case = _b.extract_main_html_single(case)  # type: ignore[union-attr]
        except Exception as exc:
            out["dripper_error"] = f"primary_failed:{type(exc).__name__}:{str(exc)[:70]}"
            _fallback = _BINDINGS.get("fallback")
            if _fallback is not None:
                try:
                    case = _b.extract_main_html_fallback(case, fallback_handler=_fallback)  # type: ignore[union-attr]
                except Exception as fexc:
                    out["dripper_error"] += f"; fb:{str(fexc)[:50]}"
        od = getattr(case, "output_data", None)
        _strip_xml = _BINDINGS.get("strip_xml")
        if od and _strip_xml and isinstance(getattr(od, "main_html", None), str):
            od.main_html = _strip_xml(od.main_html)  # type: ignore[operator]
        try:
            case = _b.convert2content(case, output_format="mm_md")  # type: ignore[union-attr]
        except Exception as exc:
            out["dripper_error"] = out["dripper_error"] or f"convert:{type(exc).__name__}:{str(exc)[:70]}"
        od = getattr(case, "output_data", None)
        out["dripper_html"] = str(getattr(od, "main_html", "") or "") if od else ""
        out["dripper_content"] = str(getattr(od, "main_content", "") or "") if od else ""
        if not out["dripper_content"].strip():
            out["dripper_content"] = _trafilatura_content(raw_html, url)
        _apply_webkit_template(out, role, raw_html, map_html, simp_html, webkit_response)
    except Exception as exc:
        out["dripper_error"] = f"postprocess:{type(exc).__name__}:{str(exc)[:150]}"
    return out


def run_stage2b(df: pd.DataFrame) -> pd.DataFrame:
    """Run Stage 2b postprocessing via RayActorPoolExecutor."""
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import DocumentBatch

    n_workers = max(1, (os.cpu_count() or 4) - 2)
    t0 = time.perf_counter()
    chunk = max(1, len(df) // n_workers)
    initial_tasks = [
        DocumentBatch(dataset_name="stage2b", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    stage_cls = _make_stage_cls("stage2b_postprocess", _load_stage2b_bindings, _postprocess_one)
    pipeline = Pipeline(name="stage2b")
    pipeline.add_stage(stage_cls())
    output_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=initial_tasks) or []

    result_df = pd.concat([t.to_pandas() for t in output_tasks], ignore_index=True)
    elapsed = time.perf_counter() - t0
    content_ok = (result_df["dripper_content"].astype(str).str.len() > _MIN_CONTENT_LEN).sum()
    mapping_ok = (result_df["mapping_json"].astype(str).str.len() > _MIN_CONTENT_LEN).sum()
    print(
        f"[gpu-pipeline] Stage 2b: content_ok={content_ok:,} mapping_ok={mapping_ok:,} in {elapsed:.1f}s", flush=True
    )
    return result_df


def run(args: argparse.Namespace) -> None:
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
        f"[gpu-pipeline] {len(rep_df):,}/{len(all_df):,} pages sent to LLM ({len(rep_df) / max(len(all_df), 1) * 100:.1f}%)",
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
            infer_df = infer_df.drop(columns=[f"{c}_1c"])
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
    ok = int((result_df["dripper_content"].astype(str).str.len() > _MIN_CONTENT_LEN).sum())
    print(
        f"[gpu-pipeline] ALL DONE: {len(result_df):,} pages ok={ok} "
        f"total={total_s:.1f}s (1c={t1c_s:.1f}s 2={t2_s:.1f}s 2b={t2b_s:.1f}s) → {out_path}",
        flush=True,
    )

    tracker.finish(
        total_pages=len(result_df),
        errors=int((result_df["dripper_error"].astype(str).str.len() > _MIN_ERROR_LEN).sum()),
    )
    tracker.extra = {
        "stage1c_s": round(t1c_s, 1),
        "stage2_s": round(t2_s, 1),
        "stage2b_s": round(t2b_s, 1),
        "content_ok": ok,
    }
    tracker.save(args.output)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--slice")
    p.add_argument("--slice-out")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
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
        cfg = _WorkerConfig(
            model=args.model,
            gpu_mem_util=args.gpu_mem_util,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_tokens=args.max_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
        )
        run_stage2_worker(args.gpu, args.slice, args.slice_out, cfg)
    else:
        if not args.input or not args.output:
            p.error("--input and --output required in main mode")
        run(args)


if __name__ == "__main__":
    main()
