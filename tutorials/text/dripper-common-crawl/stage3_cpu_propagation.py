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
LBP static (validated clusters) then full dynamic LBP, copy GPU result for
representatives/singletons, write atomically.

Slurm: --array=0-79  --partition=cpu_long  --cpus-per-task=64  --mem=235G  --time=06:00:00
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

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

_WORKER_BINDINGS: Any = None
_WORKER_MINERU_BINDINGS: Any = None
_WORKER_PARAMS: dict[str, Any] = {}
_WORKER_INITIALIZED: bool = False


def _worker_init(
    dynamic_classid_similarity_threshold: float,
    more_noise_enable: bool,
    min_content_length_ratio: float,
    max_content_length_ratio: float,
    log_level: str,
) -> None:
    """Called once per worker process; imports heavy libraries."""
    global _WORKER_BINDINGS, _WORKER_MINERU_BINDINGS, _WORKER_PARAMS, _WORKER_INITIALIZED
    if _WORKER_INITIALIZED:
        return
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(processName)s %(levelname)s %(message)s",
    )
    _WORKER_PARAMS = {
        "dynamic_classid_similarity_threshold": dynamic_classid_similarity_threshold,
        "more_noise_enable": more_noise_enable,
        "min_content_length_ratio": min_content_length_ratio,
        "max_content_length_ratio": max_content_length_ratio,
    }
    try:
        from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser

        class _Bindings:
            pass

        b = _Bindings()
        b.layout_parser_cls = LayoutBatchParser
        _WORKER_BINDINGS = b
    except Exception as exc:
        logging.getLogger(__name__).warning("llm_web_kit unavailable: %s", exc)
        _WORKER_BINDINGS = None
    try:
        from mineru_html.base import MinerUHTMLCase, MinerUHTMLInput, MinerUHTMLOutput
        from mineru_html.process import convert2content

        class _MineruBindings:
            pass

        mb = _MineruBindings()
        mb.convert2content = convert2content
        mb.output_cls = MinerUHTMLOutput
        mb.case_cls = MinerUHTMLCase
        mb.input_cls = MinerUHTMLInput
        try:
            from nemo_curator.stages.text.experimental.dripper.stage import _strip_xml_incompatible_chars

            mb.strip_xml = _strip_xml_incompatible_chars
        except Exception:
            mb.strip_xml = None
        _WORKER_MINERU_BINDINGS = mb
    except Exception as exc:
        logging.getLogger(__name__).warning("mineru_html unavailable: %s", exc)
        _WORKER_MINERU_BINDINGS = None
    _WORKER_INITIALIZED = True


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _token_f1(a: str, b: str) -> float:
    """Token-multiset F1 between two texts."""
    from collections import Counter

    ca = Counter(_TOKEN_RE.findall(a.lower())) if a else Counter()
    cb = Counter(_TOKEN_RE.findall(b.lower())) if b else Counter()
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    p = common / sum(ca.values())
    r = common / sum(cb.values())
    return 2 * p * r / (p + r)


_CLUSTER_STATIC_OK: dict[str, bool] = {}  # per-worker memo: cluster_id -> bool


def _cluster_static_trustworthy(
    cluster_id: Any, sample_rows: list[dict[str, Any]], mapping_data: dict[str, Any] | None
) -> bool:
    """Return True if static LBP reproduces dynamic LBP on a sample of siblings (memoized)."""
    if mapping_data is None:
        return False
    key = str(cluster_id)
    if key in _CLUSTER_STATIC_OK:
        return _CLUSTER_STATIC_OK[key]
    K, thr = 3, _WORKER_PARAMS.get("static_validation_min_f1", 0.97)
    f1s: list[float] = []
    for row in sample_rows[:K]:
        html = _coerce_html(row.get("html", ""))
        if not html.strip():
            continue
        sh, se = _layout_batch_parser_propagate(html, mapping_data, dynamic=False)
        dh, de = _layout_batch_parser_propagate(html, mapping_data, dynamic=True)
        if not dh or de:
            continue
        if not sh or se:
            f1s.append(0.0)
            continue
        url = row.get("url", "")
        sc, _ = _convert_main_html_to_content(sh, url)
        dc, _ = _convert_main_html_to_content(dh, url)
        f1s.append(_token_f1(sc, dc))
    ok = bool(f1s) and (sum(f1s) / len(f1s) >= thr)
    _CLUSTER_STATIC_OK[key] = ok
    return ok


def _layout_batch_parser_propagate(html: str, mapping_data: dict[str, Any], dynamic: bool = True) -> tuple[str, str]:
    """Propagate template to a sibling via LayoutBatchParser; dynamic=False skips cosine matching.

    Returns (main_html_fragment, error_str).
    """
    global _WORKER_BINDINGS, _WORKER_PARAMS
    if _WORKER_BINDINGS is None:
        return "", "llm_web_kit_not_available"
    html_source = html.strip()
    if not html_source:
        return "", "empty_html"
    try:
        task_data = dict(mapping_data)
        task_data.update(
            {
                "html_source": html_source,
                "dynamic_id_enable": dynamic,
                "dynamic_classid_enable": dynamic,
                "more_noise_enable": _WORKER_PARAMS.get("more_noise_enable", True),
                "dynamic_classid_similarity_threshold": _WORKER_PARAMS.get(
                    "dynamic_classid_similarity_threshold", 0.70
                ),
            }
        )
        parts = _WORKER_BINDINGS.layout_parser_cls({}).parse(task_data)
    except Exception as exc:
        return "", f"layout_parser_error={exc!s:.200}"
    if parts.get("main_html_success") is False:
        return "", f"main_html_success_false sim={parts.get('main_html_sim', 'n/a')}"
    main_html = str(parts.get("main_html_body") or "")
    if not main_html.strip():
        return "", "layout_parser_empty_output"
    return main_html, ""


def _convert_main_html_to_content(main_html: str, url: str) -> tuple[str, str]:
    """Convert main_html to text via MinerU-HTML; falls back to lxml. Returns (content, error)."""
    global _WORKER_MINERU_BINDINGS
    if _WORKER_MINERU_BINDINGS is None:
        try:
            import lxml.html

            return lxml.html.fromstring(main_html).text_content().strip(), ""
        except Exception as exc:
            return "", f"lxml_text_fallback_error={exc!s:.100}"
    mb = _WORKER_MINERU_BINDINGS
    try:
        case = mb.case_cls(mb.input_cls(raw_html="", url=url))
        case.output_data = mb.output_cls(main_html=main_html)
        if getattr(mb, "strip_xml", None) is not None and isinstance(case.output_data.main_html, str):
            case.output_data.main_html = mb.strip_xml(case.output_data.main_html)
        result = mb.convert2content(case, output_format="mm_md")
        output = getattr(result, "output_data", None)
        content = getattr(output, "main_content", "") if output is not None else ""
        return str(content or ""), ""
    except Exception as exc:
        return "", f"content_conversion_error={exc!s:.150}"


def _process_representative_row(row: dict[str, Any]) -> dict[str, Any]:
    """Pass GPU result through unchanged for a representative row."""
    return {
        "url": row.get("url", ""),
        "url_host_name": row.get("url_host_name", ""),
        "cluster_id": row.get("cluster_id"),
        "cluster_role": "representative",
        "dripper_content": row.get("dripper_content", ""),
        "dripper_html": row.get("dripper_html", ""),
        "dripper_error": row.get("dripper_error", ""),
        "dripper_time_s": row.get("inference_time_s", 0.0),
        "propagation_success": not bool(row.get("dripper_error", "")),
        "propagation_method": "representative",
    }


def _process_singleton_row(row: dict[str, Any]) -> dict[str, Any]:
    """Pass GPU result through unchanged for a singleton row."""
    return {
        "url": row.get("url", ""),
        "url_host_name": row.get("url_host_name", ""),
        "cluster_id": None,
        "cluster_role": "singleton",
        "dripper_content": row.get("dripper_content", ""),
        "dripper_html": row.get("dripper_html", ""),
        "dripper_error": row.get("dripper_error", ""),
        "dripper_time_s": row.get("inference_time_s", 0.0),
        "propagation_success": not bool(row.get("dripper_error", "")),
        "propagation_method": "singleton",
    }


def _process_sibling_row(
    row: dict[str, Any], mapping_data: dict[str, Any] | None, use_static: bool = False
) -> dict[str, Any]:
    """Propagate template to a sibling: static LBP (if validated), then dynamic LBP."""
    url = row.get("url", "")
    url_host_name = row.get("url_host_name", "")
    cluster_id = row.get("cluster_id")
    html = _coerce_html(row.get("html", ""))
    t0 = time.perf_counter()
    method, main_html, content, error = "fallback", "", "", ""

    if mapping_data is not None:
        if use_static:
            lbp_html, lbp_err = _layout_batch_parser_propagate(html, mapping_data, dynamic=False)
            if lbp_html and not lbp_err:
                content, conv_err = _convert_main_html_to_content(lbp_html, url)
                if not conv_err:
                    main_html, method = lbp_html, "lbp_static"
                else:
                    error = conv_err
            else:
                error = lbp_err

        if not main_html:
            dyn_html, dyn_err = _layout_batch_parser_propagate(html, mapping_data, dynamic=True)
            if dyn_html and not dyn_err:
                content, conv_err = _convert_main_html_to_content(dyn_html, url)
                if not conv_err:
                    main_html, method, error = dyn_html, "layout_batch_parser", ""
                else:
                    error = conv_err or dyn_err
            elif dyn_err:
                error = f"static_failed({error}); dynamic_failed({dyn_err})" if error else dyn_err

    if not main_html:
        method = "fallback"
        if not error:
            error = "no_template_available"

    return {
        "url": url,
        "url_host_name": url_host_name,
        "cluster_id": cluster_id,
        "cluster_role": "sibling",
        "dripper_content": content,
        "dripper_html": main_html,
        "dripper_error": error,
        "dripper_time_s": time.perf_counter() - t0,
        "propagation_success": bool(main_html and not error),
        "propagation_method": method,
    }


def _make_fallback_row(row: dict[str, Any], role: str, error: str) -> dict[str, Any]:
    return {
        "url": row.get("url", ""),
        "url_host_name": row.get("url_host_name", ""),
        "cluster_id": row.get("cluster_id") if role != "singleton" else None,
        "cluster_role": role,
        "dripper_content": "",
        "dripper_html": "",
        "dripper_error": error,
        "dripper_time_s": 0.0,
        "propagation_success": False,
        "propagation_method": "fallback",
    }


def _process_cluster_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Process one cluster (representative + siblings) in a single worker call."""
    manifest_rows = task["manifest_rows"]
    gpu_row = task.get("gpu_row")
    mapping_data = task.get("mapping_data")

    sib_rows = [r for r in manifest_rows if str(r.get("cluster_role", "")) == "sibling"]
    use_static = bool(
        sib_rows
        and mapping_data is not None
        and _cluster_static_trustworthy(task.get("cluster_id"), sib_rows, mapping_data)
    )

    results = []
    for row in manifest_rows:
        role = str(row.get("cluster_role", "singleton"))
        if role in ("representative", "singleton"):
            if gpu_row is not None:
                merged = dict(row)
                merged.update(
                    {
                        "dripper_content": gpu_row.get("dripper_content", ""),
                        "dripper_html": gpu_row.get("dripper_html", gpu_row.get("llm_output_raw", "")),
                        "dripper_error": gpu_row.get("error", ""),
                        "inference_time_s": gpu_row.get("inference_time_s", 0.0),
                    }
                )
                fn = _process_representative_row if role == "representative" else _process_singleton_row
                results.append(fn(merged))
            else:
                results.append(_make_fallback_row(row, role, f"missing_gpu_result_for_{role}"))
        elif role == "sibling":
            results.append(_process_sibling_row(row, mapping_data, use_static))
        else:
            results.append(_make_fallback_row(row, role, f"unknown_cluster_role={role}"))
    return results


def _coerce_html(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return "" if raw is None else str(raw)


def _parse_mapping_json(raw: Any) -> dict[str, Any] | None:
    """Deserialise Stage-2b template: pickle+base64 first, then JSON fallback."""
    import base64
    import pickle

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
        for loader in (
            lambda s: pickle.loads(base64.b64decode(s)),
            lambda s: json.loads(s),
        ):
            try:
                obj = loader(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _load_cluster_manifest_shard(path: str) -> pd.DataFrame:
    """Load one manifest shard; html is read only for sibling rows to avoid OOM."""
    meta_cols = [
        "url",
        "url_host_name",
        "cluster_id",
        "cluster_role",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
    ]
    schema_names = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in meta_cols if c in schema_names]).to_pandas()
    if "cluster_id" not in df.columns:
        df["cluster_id"] = None
    if "cluster_role" not in df.columns:
        df["cluster_role"] = "singleton"
    if "html" in schema_names:
        sibling_mask = df["cluster_role"] == "sibling"
        if sibling_mask.any():
            html_df = pq.read_table(path, columns=["url", "html"]).to_pandas()
            html_df = html_df.drop_duplicates(subset="url", keep="first")
            df["html"] = df["url"].map(html_df.set_index("url")["html"])
            df.loc[~sibling_mask, "html"] = None
        else:
            df["html"] = None
    else:
        df["html"] = None
    return df


def _load_inference_results(path: str) -> pd.DataFrame:
    """Load GPU inference results, normalising schema variants from Stage 2."""
    cols_needed = [
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
    schema_names = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in cols_needed if c in schema_names]).to_pandas()
    if "cluster_id" not in df.columns and "layout_cluster_id" in df.columns:
        df = df.rename(columns={"layout_cluster_id": "cluster_id"})
    if "error" not in df.columns and "dripper_error" in df.columns:
        df = df.rename(columns={"dripper_error": "error"})
    return df


def _build_gpu_lookup(inference_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build cluster_id -> gpu_row dict for O(1) lookup."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in inference_df.to_dict("records"):
        cid = row.get("cluster_id")
        if cid is not None and str(cid) not in lookup:
            lookup[str(cid)] = row
    return lookup


def _build_singleton_gpu_lookup(inference_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build url -> gpu_row for singleton pages (cluster_id is NULL)."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in inference_df.to_dict("records"):
        cid = row.get("cluster_id")
        url = str(row.get("url") or "")
        if (cid is None or str(cid).lower() in ("none", "null", "nan", "")) and url:
            lookup[url] = row
    return lookup


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Write parquet atomically via a tmp file in the same directory."""
    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(tmp_path), compression="snappy")
    tmp_path.rename(out_path)


def process_shard(
    *,
    cluster_manifest_dir: str,
    inference_results_dir: str,
    output_dir: str,
    shard_index: int,
    num_shards: int,
    num_workers: int,
    dynamic_classid_similarity_threshold: float,
    more_noise_enable: bool,
    min_content_length_ratio: float,
    max_content_length_ratio: float,
    log_level: str,
    cluster_chunk_size: int,
) -> dict[str, Any]:
    """Process one shard's worth of cluster assignments."""
    t_start = time.perf_counter()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"shard_{shard_index:04d}.parquet"

    if out_path.exists():
        try:
            meta = pq.read_metadata(str(out_path))
            if meta.num_rows > 0:
                print(f"[stage3] SKIP shard {shard_index} — already exists ({meta.num_rows:,} rows)", flush=True)
                return {"status": "skipped", "shard": shard_index, "rows": meta.num_rows}
            out_path.unlink(missing_ok=True)
        except Exception:
            out_path.unlink(missing_ok=True)

    manifest_dir, gpu_dir = Path(cluster_manifest_dir), Path(inference_results_dir)
    manifest_files = sorted(manifest_dir.glob("shard_*.parquet")) or sorted(manifest_dir.glob("*.parquet"))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest shards found in {manifest_dir}")

    total_files = len(manifest_files)
    my_files = manifest_files[total_files * shard_index // num_shards : total_files * (shard_index + 1) // num_shards]
    if not my_files:
        print(f"[stage3] shard {shard_index}: no manifest files — writing empty shard", flush=True)
        _atomic_write_parquet(pd.DataFrame(columns=OUTPUT_COLUMNS), out_path)
        return {"status": "empty", "shard": shard_index, "rows": 0}

    print(f"[stage3] shard {shard_index}/{num_shards}: loading {len(my_files)} manifest file(s)...", flush=True)
    manifest_df = pd.concat([_load_cluster_manifest_shard(str(f)) for f in my_files], ignore_index=True)
    print(f"[stage3] shard {shard_index}: {len(manifest_df):,} manifest rows loaded", flush=True)

    manifest_cluster_ids: set[str] = set()
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        if cid is not None and str(cid).lower() not in ("none", "null", "nan", ""):
            manifest_cluster_ids.add(str(cid))
    manifest_urls: set[str] = {str(r.get("url", "")) for r in manifest_df.to_dict("records")}

    exact_gpu = gpu_dir / f"shard_{shard_index:04d}.parquet"
    gpu_files = (
        [exact_gpu]
        if exact_gpu.exists()
        else (sorted(gpu_dir.glob("shard_*.parquet")) or sorted(gpu_dir.glob("*.parquet")))
    )
    if not gpu_files:
        raise FileNotFoundError(f"No GPU inference result files found in {gpu_dir}")

    print(
        f"[stage3] loading GPU results for {len(manifest_cluster_ids):,} cluster_ids "
        f"from {len(gpu_files)} GPU shard file(s)...",
        flush=True,
    )
    gpu_frames = []
    for f in gpu_files:
        try:
            shard_df = _load_inference_results(str(f))
            if len(shard_df) == 0:
                continue
            mask = pd.Series(False, index=shard_df.index)
            if "cluster_id" in shard_df.columns and manifest_cluster_ids:
                mask |= shard_df["cluster_id"].astype(str).isin(manifest_cluster_ids)
            if "url" in shard_df.columns and manifest_urls:
                null_cid = shard_df["cluster_id"].isna() | shard_df["cluster_id"].astype(str).isin(
                    ("none", "null", "nan", "")
                )
                mask |= null_cid & shard_df["url"].astype(str).isin(manifest_urls)
            filtered = shard_df[mask]
            if len(filtered) > 0:
                gpu_frames.append(filtered)
        except Exception as exc:
            print(f"[stage3] WARNING: could not read GPU shard {f}: {exc}", flush=True)
    gpu_df = pd.concat(gpu_frames, ignore_index=True) if gpu_frames else pd.DataFrame()
    del gpu_frames
    print(f"[stage3] {len(gpu_df):,} relevant GPU result rows loaded", flush=True)

    cluster_gpu_lookup = _build_gpu_lookup(gpu_df)
    singleton_gpu_lookup = _build_singleton_gpu_lookup(gpu_df)
    del gpu_df

    print("[stage3] building cluster tasks...", flush=True)
    tasks: list[dict[str, Any]] = []
    cluster_groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        cid_key: str | None = (
            str(cid) if (cid is not None and str(cid).lower() not in ("none", "null", "nan", "")) else None
        )
        cluster_groups[cid_key].append(row)

    PAGES_PER_TASK = 300
    for cid_key, rows in cluster_groups.items():
        if cid_key is None:
            for row in rows:
                tasks.append(
                    {
                        "cluster_id": None,
                        "manifest_rows": [row],
                        "gpu_row": singleton_gpu_lookup.get(str(row.get("url", ""))),
                        "mapping_data": None,
                    }
                )
        else:
            gpu_row = cluster_gpu_lookup.get(cid_key)
            mapping_data = (
                _parse_mapping_json(gpu_row.get("mapping_json") or gpu_row.get("llm_output_raw"))
                if gpu_row is not None
                else None
            )
            non_sib = [r for r in rows if str(r.get("cluster_role", "")) != "sibling"]
            sib = [r for r in rows if str(r.get("cluster_role", "")) == "sibling"]
            tasks.append(
                {
                    "cluster_id": cid_key,
                    "manifest_rows": non_sib + sib[:PAGES_PER_TASK],
                    "gpu_row": gpu_row,
                    "mapping_data": mapping_data,
                }
            )
            for i in range(PAGES_PER_TASK, len(sib), PAGES_PER_TASK):
                tasks.append(
                    {
                        "cluster_id": cid_key,
                        "manifest_rows": sib[i : i + PAGES_PER_TASK],
                        "gpu_row": None,
                        "mapping_data": mapping_data,
                    }
                )

    del manifest_df, cluster_groups, cluster_gpu_lookup, singleton_gpu_lookup

    total_tasks = len(tasks)
    total_pages = sum(len(t["manifest_rows"]) for t in tasks)
    print(f"[stage3] shard {shard_index}: {total_tasks:,} cluster tasks, {total_pages:,} pages", flush=True)

    worker_initargs = (
        dynamic_classid_similarity_threshold,
        more_noise_enable,
        min_content_length_ratio,
        max_content_length_ratio,
        log_level,
    )
    all_results: list[dict[str, Any]] = []
    n_success = n_fallback = n_xpath = n_lbp = n_rep = n_singleton = pages_done = 0
    t_proc_start = time.perf_counter()
    chunk_size = max(cluster_chunk_size, 1)
    num_chunks = (total_tasks + chunk_size - 1) // chunk_size
    ctx = multiprocessing.get_context("spawn")  # avoid fork-safety issues with C extensions

    with ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx, initializer=_worker_init, initargs=worker_initargs
    ) as executor:
        for chunk_idx in range(num_chunks):
            chunk = tasks[chunk_idx * chunk_size : min((chunk_idx + 1) * chunk_size, total_tasks)]
            chunk_results: list[dict[str, Any]] = []
            for future in as_completed({executor.submit(_process_cluster_task, t): i for i, t in enumerate(chunk)}):
                try:
                    chunk_results.extend(future.result())
                except Exception as exc:
                    logger.error("Task failed: %s", exc)
            all_results.extend(chunk_results)
            for r in chunk_results:
                meth = r.get("propagation_method", "fallback")
                if r.get("propagation_success"):
                    n_success += 1
                else:
                    n_fallback += 1
                if meth in ("xpath", "lbp_static"):
                    n_xpath += 1
                elif meth == "layout_batch_parser":
                    n_lbp += 1
                elif meth == "representative":
                    n_rep += 1
                elif meth == "singleton":
                    n_singleton += 1
            pages_done += sum(len(t["manifest_rows"]) for t in chunk)
            elapsed = time.perf_counter() - t_proc_start
            print(
                f"[stage3] shard {shard_index}: chunk {chunk_idx + 1}/{num_chunks} "
                f"pages={pages_done:,}/{total_pages:,} rate={pages_done / max(elapsed, 0.001):.1f} pages/s  "
                f"success={n_success} fallback={n_fallback} xpath={n_xpath} lbp={n_lbp}",
                flush=True,
            )

    _atomic_write_parquet(pd.DataFrame(all_results, columns=OUTPUT_COLUMNS), out_path)

    elapsed_total = time.perf_counter() - t_start
    pages_per_s = total_pages / max(elapsed_total, 0.001)
    metrics = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "manifest_files": len(my_files),
        "total_pages": total_pages,
        "success_pages": n_success,
        "fallback_pages": n_fallback,
        "xpath_pages": n_xpath,
        "layout_batch_parser_pages": n_lbp,
        "representative_pages": n_rep,
        "singleton_pages": n_singleton,
        "elapsed_s": elapsed_total,
        "pages_per_s": pages_per_s,
        "output_path": str(out_path),
    }
    (output_dir_path / f"metrics_shard_{shard_index:04d}.json").write_text(json.dumps(metrics, indent=2))

    print(f"[stage3] shard {shard_index} DONE", flush=True)
    print(f"  pages:   {total_pages:,}  (success={n_success} fallback={n_fallback})", flush=True)
    print(f"  xpath:   {n_xpath}  lbp={n_lbp}  rep={n_rep}  singleton={n_singleton}", flush=True)
    print(f"  elapsed: {elapsed_total:.1f}s  ({pages_per_s:.1f} pages/s)", flush=True)
    print(f"  output:  {out_path}", flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: CPU template propagation for CC-scale pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cluster-manifest", required=True, help="cluster_assignments/ shard_NNNN.parquet dir (Stage 1 output)"
    )
    p.add_argument("--inference-results", required=True, help="gpu_results/ shard_NNNN.parquet dir (Stage 2 output)")
    p.add_argument("--output-dir", required=True, help="Output dir for propagation_results/ shard_NNNN.parquet")
    p.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="0-based task index (default: SLURM_ARRAY_TASK_ID)",
    )
    p.add_argument("--num-shards", type=int, default=80, help="Total number of array tasks (= number of CPU nodes)")
    p.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 64)),
        help="Parallel workers per node (default: SLURM_CPUS_PER_TASK or 64)",
    )
    p.add_argument(
        "--cluster-chunk-size", type=int, default=500, help="Cluster tasks per process-pool chunk (controls memory)"
    )
    p.add_argument(
        "--dynamic-classid-similarity-threshold",
        type=float,
        default=0.70,
        help="LayoutBatchParser classid similarity threshold",
    )
    p.add_argument(
        "--more-noise-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable more-noise mode in LayoutBatchParser",
    )
    p.add_argument(
        "--min-content-length-ratio",
        type=float,
        default=0.25,
        help="Minimum propagated/representative content length ratio",
    )
    p.add_argument(
        "--max-content-length-ratio",
        type=float,
        default=4.0,
        help="Maximum propagated/representative content length ratio",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    print("=" * 70, flush=True)
    print("  Stage 3: CPU Template Propagation", flush=True)
    print("=" * 70, flush=True)
    print(f"  cluster_manifest:  {args.cluster_manifest}", flush=True)
    print(f"  inference_results: {args.inference_results}", flush=True)
    print(f"  output_dir:        {args.output_dir}", flush=True)
    print(f"  shard:             {args.shard_index}/{args.num_shards}", flush=True)
    print(f"  num_workers:       {args.num_workers}", flush=True)
    print(f"  classid_threshold: {args.dynamic_classid_similarity_threshold}", flush=True)
    print(f"  content_ratio:     [{args.min_content_length_ratio}, {args.max_content_length_ratio}]", flush=True)
    print("=" * 70, flush=True)

    metrics = process_shard(
        cluster_manifest_dir=args.cluster_manifest,
        inference_results_dir=args.inference_results,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        num_workers=args.num_workers,
        dynamic_classid_similarity_threshold=args.dynamic_classid_similarity_threshold,
        more_noise_enable=args.more_noise_enable,
        min_content_length_ratio=args.min_content_length_ratio,
        max_content_length_ratio=args.max_content_length_ratio,
        log_level=args.log_level,
        cluster_chunk_size=args.cluster_chunk_size,
    )
    status = metrics.get("status", "done")
    if status == "skipped":
        print(f"[stage3] Shard {args.shard_index} already complete — skipped.", flush=True)
    elif status == "empty":
        print(f"[stage3] Shard {args.shard_index} had no input — wrote empty shard.", flush=True)
    else:
        print(f"[stage3] Shard {args.shard_index} complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
