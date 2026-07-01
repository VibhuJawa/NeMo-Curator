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

"""Stage 2b: CPU postprocessing from LLM responses (thin wrapper around DripperHTMLPostprocessStage)."""

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.experimental.dripper._layout_mapping import build_layout_mapping_data
from nemo_curator.stages.text.experimental.dripper._mapping_serialization import serialize_mapping_data
from nemo_curator.stages.text.experimental.dripper import DripperHTMLPostprocessStage
from nemo_curator.stages.text.experimental.dripper._html_compression import HTML_CHARS_COL, HTML_ZLIB_COL, get_html_from_row
from nemo_curator.stages.text.experimental.dripper.stage import (
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
)
from nemo_curator.tasks import DocumentBatch

_MIN_NONEMPTY_LEN: int = 5
_MIN_ERROR_LEN: int = 2
_NEEDS_LLM_COL = "_dripper_needs_llm"
_PRIMARY_ERROR_COL = "_dripper_primary_error"
_EMPTY_INPUT_COL = "_dripper_empty_input"


def _default_workers() -> int:
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or max(1, (os.cpu_count() or 4) - 2)))


def _init_ray_cpu(num_workers: int) -> None:
    import ray

    if not ray.is_initialized():
        ray_kwargs: dict[str, object] = {
            "num_cpus": max(1, num_workers),
            "num_gpus": 0,
            "include_dashboard": False,
        }
        if os.environ.get("RAY_TMPDIR"):
            ray_kwargs["_temp_dir"] = os.environ["RAY_TMPDIR"]
        ray.init(**ray_kwargs)


def _prepare_postprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dripper_response" not in out.columns and "llm_response" in out.columns:
        out["dripper_response"] = out["llm_response"]
    if "dripper_inference_time_s" not in out.columns and "inference_time_s" in out.columns:
        out["dripper_inference_time_s"] = out["inference_time_s"]
    if _NEEDS_LLM_COL not in out.columns:
        out[_NEEDS_LLM_COL] = out.get("dripper_response", pd.Series("", index=out.index)).astype(str).str.len() > 0
    if _PRIMARY_ERROR_COL not in out.columns:
        out[_PRIMARY_ERROR_COL] = out.get("dripper_error", pd.Series("", index=out.index)).fillna("").astype(str)
    if _EMPTY_INPUT_COL not in out.columns:
        if HTML_CHARS_COL in out.columns:
            out[_EMPTY_INPUT_COL] = pd.to_numeric(out[HTML_CHARS_COL], errors="coerce").fillna(0).le(0)
        else:
            out[_EMPTY_INPUT_COL] = [len(get_html_from_row(row)) == 0 for row in out.to_dict("records")]
    return out


def _add_mapping_json(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping_blobs: list[str] = []
    mapping_errors: list[str] = []
    bindings = _load_mineru_html_bindings()
    web_bindings = _load_llm_web_kit_bindings()
    for _, row in out.iterrows():
        role = str(row.get("cluster_role") or "")
        cluster_id = str(row.get("cluster_id") or "")
        if role and role != "representative":
            mapping_blobs.append("")
            mapping_errors.append("")
            continue
        if not cluster_id:
            mapping_blobs.append("")
            mapping_errors.append("")
            continue
        result = build_layout_mapping_data(row, bindings=bindings, web_bindings=web_bindings)
        mapping_blobs.append(serialize_mapping_data(result.mapping_data))
        mapping_errors.append(result.error)
    out["mapping_json"] = mapping_blobs
    out["mapping_error"] = mapping_errors
    return out


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if inp.is_dir():
        files = sorted(inp.glob(f"shard_{args.shard_index:04d}.parquet")) or sorted(inp.glob("*.parquet"))
        inp = files[0] if files else inp

    raw_df = pq.ParquetFile(str(inp)).read().to_pandas()
    if HTML_ZLIB_COL not in raw_df.columns:
        raise ValueError(f"{inp} is missing required HTML column: {HTML_ZLIB_COL!r}")
    df = _prepare_postprocess_input(raw_df)
    logger.info("{:,} pages to postprocess ({} workers)", len(df), args.workers)

    n_workers = args.workers
    _init_ray_cpu(n_workers)
    chunk = max(1, len(df) // n_workers)
    tasks = [
        DocumentBatch(dataset_name="stage2b", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    # Simple Curator pattern: construct library stage, build pipeline, call run()
    stage = DripperHTMLPostprocessStage(
        html_col=HTML_ZLIB_COL,
        url_col="url",
        fallback="trafilatura",
        output_format="mm_md",
        keep_intermediate=True,
        worker_count=n_workers,
    )
    pipeline = Pipeline(name="stage2b")
    pipeline.add_stage(stage)
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []

    result_df = pd.concat([t.to_pandas() for t in result_tasks], ignore_index=True) if result_tasks else df
    result_df = _add_mapping_json(result_df)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (
        f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "postprocess_results.parquet"
    )
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="zstd")
    tmp.rename(out_path)

    content_ok = int(
        (result_df["dripper_content"].astype(str).str.len() > _MIN_NONEMPTY_LEN).sum()
        if "dripper_content" in result_df.columns
        else 0
    )
    errors = int(
        (result_df["dripper_error"].astype(str).str.len() > _MIN_ERROR_LEN).sum()
        if "dripper_error" in result_df.columns
        else 0
    )
    mapping_ok = int(
        (result_df["mapping_json"].astype(str).str.len() > _MIN_NONEMPTY_LEN).sum()
        if "mapping_json" in result_df.columns
        else 0
    )
    logger.info(
        "content_ok={}/{}  mapping_ok={}  errors={}  output -> {}",
        content_ok,
        len(result_df),
        mapping_ok,
        errors,
        out_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Stage 2 output dir")
    p.add_argument("--output", required=True, help="Output dir")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=_default_workers())
    run(p.parse_args())


if __name__ == "__main__":
    main()
