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

"""Stage 2b: GPU vLLM inference for Stage 2a prompt shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

PROMPT_COL = "prompt"
PROMPT_CHARS_COL = "prompt_chars"
ITEM_COUNT_COL = "item_count"
REQUEST_MAX_TOKENS_COL = "request_max_tokens"
PREPROCESS_STATUS_COL = "stage2a_status"
LLM_STATUS_COL = "stage2b_status"
LLM_ERROR_COL = "stage2b_error"

RESPONSE_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("cluster_id", pa.string()),
        ("url", pa.string()),
        ("prompt_shard", pa.string()),
        ("llm_response", pa.string()),
        ("inference_time_s", pa.float64()),
        ("gpu_id", pa.int64()),
        ("prompt_chars", pa.int64()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("total_tokens", pa.int64()),
        (LLM_STATUS_COL, pa.string()),
        (LLM_ERROR_COL, pa.string()),
    ]
)


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    gpu_mem_util: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_tokens: int
    kv_cache_dtype: str


def _empty_response_table() -> pa.Table:
    return pa.Table.from_arrays([pa.array([], type=field.type) for field in RESPONSE_SCHEMA], schema=RESPONSE_SCHEMA)


def _detect_gpus() -> int:
    for name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_GPUS"):
        raw = os.environ.get(name, "")
        if raw:
            try:
                return int(raw.split(":")[-1])
            except ValueError:
                digits = "".join(ch for ch in raw if ch.isdigit())
                if digits:
                    return int(digits)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and visible not in {"NoDevFiles", "-1"}:
        return len([item for item in visible.split(",") if item.strip()])
    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=5)
    except OSError:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU"))


def _prompt_files(input_path: Path) -> list[Path]:
    files = [input_path] if input_path.is_file() else sorted(input_path.glob("prompt_*.parquet"))
    files = [path for path in files if ".tmp" not in path.name]
    if not files:
        raise FileNotFoundError(f"No prompt_*.parquet files found in {input_path}")
    return files


def _read_prompt_file_cost(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    ok_rows = 0
    prompt_chars = 0
    if rows:
        cols = [col for col in (PREPROCESS_STATUS_COL, PROMPT_CHARS_COL) if col in pf.schema_arrow.names]
        df = pf.read(columns=cols).to_pandas()
        if PREPROCESS_STATUS_COL in df.columns:
            mask = df[PREPROCESS_STATUS_COL].astype(str).eq("ok")
        else:
            mask = pd.Series(True, index=df.index)
        ok_rows = int(mask.sum())
        if PROMPT_CHARS_COL in df.columns:
            prompt_chars = int(pd.to_numeric(df.loc[mask, PROMPT_CHARS_COL], errors="coerce").fillna(0).sum())
    return {"path": str(path), "rows": rows, "ok_rows": ok_rows, "prompt_chars": prompt_chars}


def _assign_prompt_files(files: list[Path], n_gpus: int) -> list[list[dict[str, Any]]]:
    costs = [_read_prompt_file_cost(path) for path in files]
    bins: list[list[dict[str, Any]]] = [[] for _ in range(n_gpus)]
    loads = [0] * n_gpus
    for item in sorted(costs, key=lambda value: int(value["prompt_chars"]), reverse=True):
        gpu = min(range(n_gpus), key=lambda idx: loads[idx])
        bins[gpu].append(item)
        loads[gpu] += int(item["prompt_chars"])
    return bins


def _write_worker_manifest(output_dir: Path, gpu_id: int, items: list[dict[str, Any]]) -> Path:
    path = output_dir / "_gpu_manifests" / f"gpu_{gpu_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gpu_id": gpu_id, "prompt_files": items}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _response_path(output_dir: Path, prompt_path: Path) -> Path:
    stem = prompt_path.stem.removeprefix("prompt_")
    return output_dir / f"response_{stem}.parquet"


def _to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_worker_prompts(rows: list[dict[str, Any]], tok: Any, cfg: WorkerConfig) -> tuple[list[dict], list[Any], list[int], list[dict[str, Any] | None], int]:
    from vllm import SamplingParams

    supports_think = True
    prompts: list[dict] = []
    samplings: list[Any] = []
    row_indexes: list[int] = []
    results: list[dict[str, Any] | None] = [None] * len(rows)
    n_trunc = 0
    for i, row in enumerate(rows):
        prompt = str(row.get(PROMPT_COL) or "")
        if not prompt:
            results[i] = _response_row(row, "", 0.0, -1, "empty_prompt")
            continue
        item_count = max(0, _to_int(row.get(ITEM_COUNT_COL)))
        max_tokens = min(cfg.max_tokens, max(32, item_count * 6 + 16) if item_count > 0 else cfg.max_tokens)
        messages = [{"role": "user", "content": prompt}]
        if supports_think:
            try:
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                supports_think = False
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tok(text, add_special_tokens=False)["input_ids"]
        cap = cfg.max_model_len - max_tokens - 8
        if len(token_ids) > cap:
            token_ids = token_ids[:cap]
            n_trunc += 1
        prompts.append({"prompt_token_ids": token_ids})
        samplings.append(SamplingParams(temperature=0.0, max_tokens=max_tokens))
        row_indexes.append(i)
    return prompts, samplings, row_indexes, results, n_trunc


def _response_row(
    row: dict[str, Any],
    response: str,
    inference_time_s: float,
    gpu_id: int,
    error: str = "",
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "record_id": str(row.get("record_id") or ""),
        "cluster_id": str(row.get("cluster_id") or ""),
        "url": str(row.get("url") or ""),
        "prompt_shard": str(row.get("prompt_shard") or ""),
        "llm_response": response,
        "inference_time_s": float(inference_time_s),
        "gpu_id": int(gpu_id),
        "prompt_chars": _to_int(row.get(PROMPT_CHARS_COL)),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        LLM_STATUS_COL: "error" if error else "ok",
        LLM_ERROR_COL: error,
    }


def _usage_value(obj: object, name: str) -> int:
    value = getattr(obj, name, 0)
    return _to_int(value)


def _write_response_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp_{os.getpid()}.parquet")
    table = pa.Table.from_pylist(rows, schema=RESPONSE_SCHEMA) if rows else _empty_response_table()
    pq.write_table(table, str(tmp), compression="zstd")
    tmp.rename(path)


def run_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from transformers import AutoTokenizer
    from vllm import LLM

    from nemo_curator.utils.vllm_utils import pick_free_port, resolve_local_model_path

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    prompt_files = [Path(item["path"]) for item in manifest.get("prompt_files", [])]
    output_dir = Path(args.output)
    cfg = WorkerConfig(
        model=args.model,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_tokens=args.max_tokens,
        kv_cache_dtype=args.kv_cache_dtype,
    )

    local_model = resolve_local_model_path(cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(local_model, trust_remote_code=True)
    llm_kwargs: dict[str, Any] = {
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
        llm_kwargs["kv_cache_dtype"] = cfg.kv_cache_dtype
    os.environ["MASTER_PORT"] = str(pick_free_port())
    setup_t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    setup_s = time.perf_counter() - setup_t0

    total_prompts = 0
    total_rows = 0
    total_infer_s = 0.0
    total_trunc = 0
    for prompt_file in prompt_files:
        out_path = _response_path(output_dir, prompt_file)
        df = pq.ParquetFile(prompt_file).read().to_pandas()
        if PREPROCESS_STATUS_COL in df.columns:
            df = df[df[PREPROCESS_STATUS_COL].astype(str).eq("ok")].reset_index(drop=True)
        if df.empty:
            _write_response_rows(out_path, [])
            continue
        df["prompt_shard"] = prompt_file.name
        rows = df.to_dict("records")
        prompts, samplings, row_indexes, results, n_trunc = _build_worker_prompts(rows, tokenizer, cfg)
        infer_t0 = time.perf_counter()
        outputs = llm.generate(prompts, samplings) if prompts else []
        infer_s = time.perf_counter() - infer_t0
        total_prompts += len(prompts)
        total_rows += len(rows)
        total_infer_s += infer_s
        total_trunc += n_trunc
        per_prompt_s = infer_s / max(len(outputs), 1)
        for output_index, output in enumerate(outputs):
            row_index = row_indexes[output_index]
            response = output.outputs[0].text if output.outputs else ""
            usage = getattr(output, "usage", None)
            error = "" if response else "empty_response"
            results[row_index] = _response_row(
                rows[row_index],
                response,
                per_prompt_s,
                args.gpu,
                error,
                prompt_tokens=_usage_value(usage, "prompt_tokens"),
                completion_tokens=_usage_value(usage, "completion_tokens"),
            )
        _write_response_rows(out_path, [row for row in results if row is not None])
        logger.info(
            "gpu{} {} prompts={} trunc={} infer={:.1f}s -> {}",
            args.gpu,
            prompt_file.name,
            len(prompts),
            n_trunc,
            infer_s,
            out_path,
        )

    summary = {
        "gpu_id": args.gpu,
        "prompt_files": len(prompt_files),
        "rows": total_rows,
        "prompts": total_prompts,
        "truncated_prompts": total_trunc,
        "setup_s": round(setup_s, 3),
        "inference_s": round(total_infer_s, 3),
    }
    summary_path = output_dir / "_worker_summaries" / f"gpu_{args.gpu}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("gpu{} DONE prompts={} setup={:.1f}s infer={:.1f}s", args.gpu, total_prompts, setup_s, total_infer_s)


def run_driver(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_gpus = args.replicas if args.replicas > 0 else _detect_gpus()
    if n_gpus <= 0:
        raise RuntimeError("Stage 2b requires at least one visible GPU")

    prompt_files = _prompt_files(Path(args.input))
    bins = _assign_prompt_files(prompt_files, n_gpus)
    active_bins = [(gpu_id, items) for gpu_id, items in enumerate(bins) if any(int(item.get("ok_rows", 0)) > 0 for item in items)]
    manifests = [(gpu_id, _write_worker_manifest(output_dir, gpu_id, items)) for gpu_id, items in active_bins]
    logger.info(
        "Stage 2b scheduling {} prompt files across {} active GPU worker(s) on {} visible GPU(s)",
        len(prompt_files),
        len(manifests),
        n_gpus,
    )

    worker_base = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--output",
        str(output_dir),
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
    t0 = time.perf_counter()
    procs = [
        subprocess.Popen([*worker_base, "--gpu", str(gpu_id), "--manifest", str(manifest)])
        for gpu_id, manifest in manifests
    ]
    return_codes = [proc.wait() for proc in procs]
    elapsed = time.perf_counter() - t0
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"Stage 2b GPU worker failure: return_codes={return_codes}")

    worker_summaries = []
    for path in sorted((output_dir / "_worker_summaries").glob("gpu_*.json")):
        worker_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    response_files = sorted(output_dir.glob("response_*.parquet"))
    summary = {
        "input": str(Path(args.input)),
        "output": str(output_dir),
        "elapsed_s": round(elapsed, 3),
        "gpus": n_gpus,
        "prompt_files": len(prompt_files),
        "response_files": len(response_files),
        "return_codes": return_codes,
        "workers": worker_summaries,
        "input_rows": int(sum(item.get("rows", 0) for bin_items in bins for item in bin_items)),
        "input_ok_rows": int(sum(item.get("ok_rows", 0) for bin_items in bins for item in bin_items)),
        "input_prompt_chars": int(sum(item.get("prompt_chars", 0) for bin_items in bins for item in bin_items)),
        "generated_prompts": int(sum(item.get("prompts", 0) for item in worker_summaries)),
        "truncated_prompts": int(sum(item.get("truncated_prompts", 0) for item in worker_summaries)),
    }
    summary_path = output_dir / "_stage2b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "Stage 2b done in {:.1f}s response_files={} prompts={} summary={}",
        elapsed,
        len(response_files),
        summary["generated_prompts"],
        summary_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--worker", action="store_true", help="Run one GPU worker")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--manifest")
    parser.add_argument("--input", help="Stage 2a prompt directory")
    parser.add_argument("--output", required=True, help="Stage 2b response output directory")
    parser.add_argument("--replicas", type=int, default=int(os.environ.get("N_GPU_REPLICAS", "0")))
    parser.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    os.environ.setdefault("HF_HOME", args.hf_cache)

    if args.worker:
        if not args.manifest:
            parser.error("--manifest is required in --worker mode")
        run_worker(args)
    else:
        if not args.input:
            parser.error("--input is required in driver mode")
        run_driver(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
