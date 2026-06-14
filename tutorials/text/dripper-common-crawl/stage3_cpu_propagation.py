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

"""Stage 3: CPU template propagation for CC-scale pipeline.

Per cluster: load Stage-2b mapping_json template, propagate to siblings via
static LBP (validated clusters) then full dynamic LBP, copy GPU result for
representatives/singletons, write atomically.

Backend: RayActorPoolExecutor via NeMo Curator Pipeline.

All LBP + static/dynamic split logic lives in:
  nemo_curator.stages.text.experimental.dripper.propagation_stage
This script is a thin Slurm sharding wrapper (~200 lines).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.text.experimental.dripper.propagation_stage import (
    DripperHTMLLayoutPropagationStage,
    _cluster_static_trustworthy,
    _PropagationConfig,
    _run_content_convert,
    _run_lbp,
    _sibling_propagate,
    _StaticTrustConfig,
)
from nemo_curator.stages.text.experimental.dripper.stage import (
    _rebuild_batch,
)

OUTPUT_COLUMNS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "dripper_content",
    "dripper_html",
    "dripper_error",
    "dripper_time_s",
    "propagation_success",
    "propagation_method",  # "representative"|"singleton"|"lbp_static"|"layout_batch_parser"|"fallback"
]

_PAGES_PER_TASK = 16  # siblings per Ray actor task (PPT)


@dataclass
class _HyperParams:
    """LBP/content hyperparameters shared by stage builder and process_shard."""

    dynamic_classid_similarity_threshold: float = 0.70
    more_noise_enable: bool = True
    min_content_length_ratio: float = 0.25
    max_content_length_ratio: float = 4.0
    static_validation_min_f1: float = 0.97


@dataclass
class _ShardSpec:
    cluster_manifest_dir: str
    inference_results_dir: str
    output_dir: str
    shard_index: int
    num_shards: int


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_MANIFEST_META_COLS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]
_INFERENCE_COLS = [
    "cluster_id",
    "layout_cluster_id",
    "url",
    "llm_output_raw",
    "xpath_rules",
    "template_html",
    "inference_time_s",
    "error",
    "dripper_error",
    "dripper_content",
    "dripper_html",
    "mapping_json",
]
_NULL_VALS = ("none", "null", "nan", "")


def _load_cluster_manifest_shard(path: str) -> pd.DataFrame:
    sn = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in _MANIFEST_META_COLS if c in sn]).to_pandas()
    if "cluster_id" not in df.columns:
        df["cluster_id"] = None
    if "cluster_role" not in df.columns:
        df["cluster_role"] = "singleton"
    df["html"] = None
    if "html" in sn:
        smask = df["cluster_role"] == "sibling"
        if smask.any():
            hdf = pq.read_table(path, columns=["url", "html"]).to_pandas().drop_duplicates("url", keep="first")
            df.loc[smask, "html"] = df.loc[smask, "url"].map(hdf.set_index("url")["html"])
    return df


def _load_inference_results(path: str) -> pd.DataFrame:
    sn = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in _INFERENCE_COLS if c in sn]).to_pandas()
    if "cluster_id" not in df.columns and "layout_cluster_id" in df.columns:
        df = df.rename(columns={"layout_cluster_id": "cluster_id"})
    if "error" not in df.columns and "dripper_error" in df.columns:
        df = df.rename(columns={"dripper_error": "error"})
    return df


def _parse_mapping_json(raw: object) -> dict[str, Any] | None:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            obj = pickle.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw.strip():
        for fn in (lambda s: pickle.loads(base64.b64decode(s)), json.loads):
            try:
                obj = fn(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _parse_element_dict(element_dict_raw: str | dict) -> dict | None:
    if isinstance(element_dict_raw, dict):
        return element_dict_raw
    if not isinstance(element_dict_raw, str) or not element_dict_raw.strip():
        return None
    try:
        raw = json.loads(element_dict_raw)
        return {int(layer): {eval(k): v for k, v in layer_dict.items()} for layer, layer_dict in raw.items()}  # noqa: S307
    except (ValueError, SyntaxError):
        return None


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(tmp_path), compression="snappy")
    tmp_path.rename(out_path)


# ---------------------------------------------------------------------------
# Output-row helpers
# ---------------------------------------------------------------------------


def _output_row(row, role, html="", content="", error="", time_s=0.0, method="fallback"):
    return {
        "url": row.get("url", ""),
        "url_host_name": row.get("url_host_name", ""),
        "cluster_id": row.get("cluster_id") if role != "singleton" else None,
        "cluster_role": role,
        "dripper_content": content,
        "dripper_html": html,
        "dripper_error": error,
        "dripper_time_s": time_s,
        "propagation_success": bool(html and not error),
        "propagation_method": method,
    }


def _dispatch_cluster_rows(manifest_rows, gpu_row, mapping_data, sib_fn, use_static):
    results = []
    for row in manifest_rows:
        role = str(row.get("cluster_role", "singleton"))
        if role in ("representative", "singleton"):
            if gpu_row is not None:
                results.append(
                    _output_row(
                        row,
                        role,
                        html=gpu_row.get("dripper_html", gpu_row.get("llm_output_raw", "")),
                        content=gpu_row.get("dripper_content", ""),
                        error=gpu_row.get("error", ""),
                        time_s=gpu_row.get("inference_time_s", 0.0),
                        method=role,
                    )
                )
            else:
                results.append(_output_row(row, role, error=f"missing_gpu_result_for_{role}"))
        elif role == "sibling":
            results.append(sib_fn(row, mapping_data, use_static))
        else:
            results.append(_output_row(row, role, error=f"unknown_cluster_role={role}"))
    return results


# ---------------------------------------------------------------------------
# Ray actor stage — thin wrapper around library stage
# ---------------------------------------------------------------------------


def _build_stage3_cls(hp: _HyperParams, worker_count: int) -> type:
    """Return a ProcessingStage subclass closed over the given hyperparameters."""
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.tasks import DocumentBatch as _DocumentBatch

    _params = {
        "more_noise_enable": hp.more_noise_enable,
        "dynamic_classid_similarity_threshold": hp.dynamic_classid_similarity_threshold,
    }
    _min = hp.min_content_length_ratio
    _max = hp.max_content_length_ratio
    _f1 = hp.static_validation_min_f1
    _wc = worker_count

    # Instantiate the library stage for its bindings + memoised trust cache
    _lib_stage = DripperHTMLLayoutPropagationStage(
        dynamic_classid_similarity_threshold=hp.dynamic_classid_similarity_threshold,
        more_noise_enable=hp.more_noise_enable,
        layout_template_min_content_length_ratio=hp.min_content_length_ratio,
        layout_template_max_content_length_ratio=hp.max_content_length_ratio,
        use_static_lbp=True,
        static_validation_min_f1=hp.static_validation_min_f1,
    )

    class _Stage3PropagationStage(ProcessingStage[_DocumentBatch, _DocumentBatch]):
        name = "stage3_cpu_propagation"
        resources = Resources(cpus=1.0)
        batch_size = 1
        _initialized = False

        def num_workers(self) -> int | None:
            return _wc if _wc > 0 else None

        def setup(self, _worker_metadata: object = None) -> None:
            if self._initialized:
                return
            _lib_stage.setup()
            self._initialized = True

        def _lbp_fn(self, html, mapping_data, dynamic=True, parser_cache=None):
            return _run_lbp(_params, html, mapping_data, dynamic, _parser_cache=parser_cache)

        def _content_fn(self, main_html, url):
            return _run_content_convert(_lib_stage._bindings, main_html, url)

        def process(self, task: _DocumentBatch) -> _DocumentBatch:
            if not self._initialized:
                self.setup()
            ct = task._metadata.get("cluster_task", {})
            results = (
                self._process_cluster_task(ct)
                if ct
                else [
                    _output_row(r, str(r.get("cluster_role", "singleton")), error="missing_cluster_task")
                    for r in task.to_pandas().to_dict("records")
                ]
            )
            return _rebuild_batch(task, pd.DataFrame(results, columns=OUTPUT_COLUMNS))

        def _process_cluster_task(self, task: dict[str, Any]) -> list[dict[str, Any]]:
            manifest_rows = task["manifest_rows"]
            gpu_row = task.get("gpu_row")
            mapping_data = task.get("mapping_data")
            sib_rows = [r for r in manifest_rows if str(r.get("cluster_role", "")) == "sibling"]

            parser_cache: dict = {}
            lbp_fn_cached = lambda html, md, dynamic=True: self._lbp_fn(html, md, dynamic, parser_cache)  # noqa: E731
            trust_cfg = _StaticTrustConfig(
                memo=_lib_stage._cluster_static_ok,
                lbp_fn=lbp_fn_cached,
                content_fn=self._content_fn,
                threshold=_f1,
            )
            prop_cfg = _PropagationConfig(
                lbp_fn=lbp_fn_cached,
                content_fn=self._content_fn,
                min_ratio=_min,
                max_ratio=_max,
            )
            use_static = bool(
                sib_rows
                and mapping_data is not None
                and _cluster_static_trustworthy(task.get("cluster_id"), sib_rows, mapping_data, trust_cfg)
            )

            def sib_fn(row, md, us):
                t0 = time.perf_counter()
                html, content, error, method = _sibling_propagate(row, md, us, prop_cfg)
                return _output_row(
                    row,
                    "sibling",
                    html=html,
                    content=content,
                    error=error,
                    time_s=time.perf_counter() - t0,
                    method=method,
                )

            return _dispatch_cluster_rows(manifest_rows, gpu_row, mapping_data, sib_fn=sib_fn, use_static=use_static)

    return _Stage3PropagationStage


# ---------------------------------------------------------------------------
# GPU-result loading helpers
# ---------------------------------------------------------------------------


def _build_gpu_lookups(inference_df: pd.DataFrame) -> tuple[dict, dict]:
    by_cluster: dict[str, dict[str, Any]] = {}
    by_url: dict[str, dict[str, Any]] = {}
    for row in inference_df.to_dict("records"):
        cid = row.get("cluster_id")
        cid_s = str(cid) if cid is not None else ""
        if cid is not None and cid_s not in by_cluster:
            by_cluster[cid_s] = row
        url = str(row.get("url") or "")
        if (cid is None or cid_s.lower() in _NULL_VALS) and url and url not in by_url:
            by_url[url] = row
    return by_cluster, by_url


def _extract_manifest_ids(manifest_df: pd.DataFrame) -> tuple[set[str], set[str]]:
    records = manifest_df.to_dict("records")
    cluster_ids = {
        str(r["cluster_id"])
        for r in records
        if r.get("cluster_id") is not None and str(r["cluster_id"]).lower() not in _NULL_VALS
    }
    urls = {str(r.get("url", "")) for r in records}
    return cluster_ids, urls


def _load_gpu_df(gpu_dir: Path, shard_index: int, manifest_cluster_ids: set, manifest_urls: set) -> pd.DataFrame:
    exact_gpu = gpu_dir / f"shard_{shard_index:04d}.parquet"
    gpu_files = (
        [exact_gpu]
        if exact_gpu.exists()
        else (sorted(gpu_dir.glob("shard_*.parquet")) or sorted(gpu_dir.glob("*.parquet")))
    )
    if not gpu_files:
        msg = f"No GPU inference result files found in {gpu_dir}"
        raise FileNotFoundError(msg)
    logger.info(
        "loading GPU results for {:,} cluster_ids from {} file(s)...", len(manifest_cluster_ids), len(gpu_files)
    )
    gpu_frames = []
    for f in gpu_files:
        try:
            sdf = _load_inference_results(str(f))
            if sdf.empty:
                continue
            mask = pd.Series(False, index=sdf.index)
            if "cluster_id" in sdf.columns and manifest_cluster_ids:
                mask |= sdf["cluster_id"].astype(str).isin(manifest_cluster_ids)
            if "url" in sdf.columns and manifest_urls:
                null_cid = sdf["cluster_id"].isna() | sdf["cluster_id"].astype(str).isin(_NULL_VALS)
                mask |= null_cid & sdf["url"].astype(str).isin(manifest_urls)
            if not (filtered := sdf[mask]).empty:
                gpu_frames.append(filtered)
        except OSError as exc:
            logger.warning("could not read GPU shard {}: {}", f, exc)
    gpu_df = pd.concat(gpu_frames, ignore_index=True) if gpu_frames else pd.DataFrame()
    logger.info("{:,} relevant GPU result rows loaded", len(gpu_df))
    return gpu_df


def _build_cluster_tasks(manifest_df, cluster_gpu_lookup, singleton_gpu_lookup):
    groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        groups[str(cid) if cid is not None and str(cid).lower() not in _NULL_VALS else None].append(row)
    tasks: list[dict[str, Any]] = []
    for cid_key, rows in groups.items():
        if cid_key is None:
            tasks += [
                {
                    "cluster_id": None,
                    "manifest_rows": [r],
                    "gpu_row": singleton_gpu_lookup.get(str(r.get("url", ""))),
                    "mapping_data": None,
                }
                for r in rows
            ]
        else:
            gr = cluster_gpu_lookup.get(cid_key)
            md = _parse_mapping_json(gr.get("mapping_json") or gr.get("llm_output_raw")) if gr else None
            if md is not None:
                parsed_ed = _parse_element_dict(md.get("html_element_dict"))
                if parsed_ed is not None:
                    md = {**md, "_parsed_element_dict": parsed_ed}
            ns = [r for r in rows if str(r.get("cluster_role", "")) != "sibling"]
            sb = sorted(
                [r for r in rows if str(r.get("cluster_role", "")) == "sibling"],
                key=lambda r: len(str(r.get("html") or "")),
                reverse=True,
            )
            tasks.append(
                {"cluster_id": cid_key, "manifest_rows": ns + sb[:_PAGES_PER_TASK], "gpu_row": gr, "mapping_data": md}
            )
            for i in range(_PAGES_PER_TASK, len(sb), _PAGES_PER_TASK):
                tasks.append(
                    {
                        "cluster_id": cid_key,
                        "manifest_rows": sb[i : i + _PAGES_PER_TASK],
                        "gpu_row": None,
                        "mapping_data": md,
                    }
                )
    return tasks


def _build_doc_tasks(tasks: list[dict[str, Any]], dataset_name: str = "stage3") -> list[Any]:
    from nemo_curator.tasks import DocumentBatch

    out = []
    for t in tasks:
        df = pd.DataFrame(
            [{"url": r.get("url", ""), "cluster_role": r.get("cluster_role", "")} for r in t["manifest_rows"][:1]]
        )
        db = DocumentBatch(dataset_name=dataset_name, data=df)
        db._metadata["cluster_task"] = t
        out.append(db)
    return out


def _finalize_shard(result_df, out_path, output_dir_path, shard_index, num_shards, my_files, total_pages, t_start):
    _atomic_write_parquet(result_df, out_path)
    ns = int(result_df["propagation_success"].fillna(False).sum())
    mth = result_df["propagation_method"]
    elapsed = time.perf_counter() - t_start
    pps = total_pages / max(elapsed, 0.001)
    nf = len(result_df) - ns
    nx = int((mth == "lbp_static").sum())
    nl = int((mth == "layout_batch_parser").sum())
    nr = int((mth == "representative").sum())
    nsi = int((mth == "singleton").sum())
    metrics = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "manifest_files": len(my_files),
        "total_pages": total_pages,
        "success_pages": ns,
        "fallback_pages": nf,
        "xpath_pages": nx,
        "layout_batch_parser_pages": nl,
        "representative_pages": nr,
        "singleton_pages": nsi,
        "elapsed_s": elapsed,
        "pages_per_s": pps,
        "output_path": str(out_path),
    }
    (output_dir_path / f"metrics_shard_{shard_index:04d}.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "shard {} done  pages={:,} success={} fallback={}"
        "  xpath={} lbp={} rep={} singleton={}"
        "  elapsed={:.1f}s ({:.1f} p/s)  output={}",
        shard_index,
        total_pages,
        ns,
        nf,
        nx,
        nl,
        nr,
        nsi,
        elapsed,
        pps,
        out_path,
    )
    return metrics


# ---------------------------------------------------------------------------
# Main shard entry point
# ---------------------------------------------------------------------------


def process_shard(spec: _ShardSpec, num_workers: int, hyperparams: _HyperParams | None = None) -> dict[str, Any]:
    """Process one shard's worth of cluster assignments using RayActorPoolExecutor."""
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline

    hp = hyperparams or _HyperParams()
    shard_index, num_shards = spec.shard_index, spec.num_shards
    t_start = time.perf_counter()
    output_dir_path = Path(spec.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"shard_{shard_index:04d}.parquet"

    if out_path.exists():
        try:
            meta = pq.read_metadata(str(out_path))
            if meta.num_rows > 0:
                logger.info("SKIP shard {} — already exists ({:,} rows)", shard_index, meta.num_rows)
                return {"status": "skipped", "shard": shard_index, "rows": meta.num_rows}
            out_path.unlink(missing_ok=True)
        except OSError:
            out_path.unlink(missing_ok=True)

    manifest_dir, gpu_dir = Path(spec.cluster_manifest_dir), Path(spec.inference_results_dir)
    manifest_files = sorted(manifest_dir.glob("shard_*.parquet")) or sorted(manifest_dir.glob("*.parquet"))
    if not manifest_files:
        msg = f"No manifest shards found in {manifest_dir}"
        raise FileNotFoundError(msg)

    n = len(manifest_files)
    my_files = manifest_files[n * shard_index // num_shards : n * (shard_index + 1) // num_shards]
    if not my_files:
        logger.info("shard {}: no manifest files — writing empty shard", shard_index)
        _atomic_write_parquet(pd.DataFrame(columns=OUTPUT_COLUMNS), out_path)
        return {"status": "empty", "shard": shard_index, "rows": 0}

    manifest_df = pd.concat([_load_cluster_manifest_shard(str(f)) for f in my_files], ignore_index=True)
    logger.info("shard {}/{}: {:,} rows from {} file(s)", shard_index, num_shards, len(manifest_df), len(my_files))

    manifest_cluster_ids, manifest_urls = _extract_manifest_ids(manifest_df)
    gpu_df = _load_gpu_df(gpu_dir, shard_index, manifest_cluster_ids, manifest_urls)
    cluster_gpu_lookup, singleton_gpu_lookup = _build_gpu_lookups(gpu_df)
    del gpu_df

    tasks = _build_cluster_tasks(manifest_df, cluster_gpu_lookup, singleton_gpu_lookup)
    del manifest_df, cluster_gpu_lookup, singleton_gpu_lookup
    tasks.sort(key=lambda t: len(t["manifest_rows"]), reverse=True)  # LPT scheduling

    total_pages = sum(len(t["manifest_rows"]) for t in tasks)
    logger.info("shard {}: {:,} cluster tasks, {:,} pages", shard_index, len(tasks), total_pages)

    doc_tasks = _build_doc_tasks(tasks)
    pipeline = Pipeline(name="stage3_cpu_propagation")
    pipeline.add_stage(_build_stage3_cls(hp, worker_count=num_workers)())
    logger.info("submitting {:,} tasks to RayActorPoolExecutor ({} actors)...", len(doc_tasks), num_workers)
    t_exec = time.perf_counter()
    output_doc_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=doc_tasks) or []
    logger.info("RayActorPoolExecutor finished in {:.1f}s", time.perf_counter() - t_exec)

    frames = [t.to_pandas().reindex(columns=OUTPUT_COLUMNS) for t in output_doc_tasks]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return _finalize_shard(
        result_df, out_path, output_dir_path, shard_index, num_shards, my_files, total_pages, t_start
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_NUM_SHARDS = 80
_DEFAULT_NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "64"))


def _apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    _configs_dir = Path(__file__).parent / "configs"
    if str(_configs_dir) not in sys.path:
        sys.path.insert(0, str(_configs_dir))
    from dripper_config import DripperConfig

    cfg = DripperConfig.from_yaml(args.config)
    if args.num_shards == _DEFAULT_NUM_SHARDS:
        args.num_shards = cfg.num_shards
    if args.num_workers == _DEFAULT_NUM_WORKERS:
        stage_res = cfg.resources.get("stage3", {})
        args.num_workers = int(stage_res.get("num_workers", stage_res.get("cpus", args.num_workers)))
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: CPU template propagation for CC-scale pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to DripperConfig YAML; num_shards/num_workers read from it unless overridden",
    )
    p.add_argument("--cluster-manifest", required=True, help="cluster_assignments/ shard dir (Stage 1 output)")
    p.add_argument("--inference-results", required=True, help="gpu_results/ shard dir (Stage 2 output)")
    p.add_argument("--output-dir", required=True, help="Output dir for propagation_results/ shards")
    p.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="0-based task index (default: SLURM_ARRAY_TASK_ID)",
    )
    p.add_argument("--num-shards", type=int, default=_DEFAULT_NUM_SHARDS)
    p.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help="Ray actor count per node (default: SLURM_CPUS_PER_TASK or 64)",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return _apply_config_defaults(p.parse_args())


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    logger.info(
        "cluster_manifest={}  inference_results={}  output_dir={}  shard={}/{}  num_workers={}",
        args.cluster_manifest,
        args.inference_results,
        args.output_dir,
        args.shard_index,
        args.num_shards,
        args.num_workers,
    )
    shard_spec = _ShardSpec(
        cluster_manifest_dir=args.cluster_manifest,
        inference_results_dir=args.inference_results,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    metrics = process_shard(shard_spec, num_workers=args.num_workers)
    status = metrics.get("status", "done")
    msg = {"skipped": "already complete — skipped.", "empty": "had no input — wrote empty shard."}.get(
        status, "complete."
    )
    logger.info("Shard {} {}", args.shard_index, msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
