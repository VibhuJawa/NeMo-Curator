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

Backend: RayActorPoolExecutor via NeMo Curator Pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
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


def _load_lbp_bindings() -> Any:
    try:
        from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser

        class _B:
            pass

        b = _B()
        b.layout_parser_cls = LayoutBatchParser
        return b
    except Exception as exc:
        logger.warning("llm_web_kit unavailable: %s", exc)
        return None


def _load_mineru_bindings() -> Any:
    try:
        from mineru_html.base import MinerUHTMLCase, MinerUHTMLInput, MinerUHTMLOutput
        from mineru_html.process import convert2content

        class _MB:
            pass

        mb = _MB()
        mb.convert2content = convert2content
        mb.output_cls = MinerUHTMLOutput
        mb.case_cls = MinerUHTMLCase
        mb.input_cls = MinerUHTMLInput
        try:
            from nemo_curator.stages.text.experimental.dripper.stage import _strip_xml_incompatible_chars

            mb.strip_xml = _strip_xml_incompatible_chars
        except Exception:
            mb.strip_xml = None
        return mb
    except Exception as exc:
        logger.warning("mineru_html unavailable: %s", exc)
        return None


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _token_f1(a: str, b: str) -> float:
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
    return 2 * common / (sum(ca.values()) + sum(cb.values()))


def _cluster_static_trustworthy(cluster_id, sample_rows, mapping_data, memo, lbp_fn, content_fn, threshold) -> bool:
    """Return True if static LBP reproduces dynamic LBP on K=3 sample siblings (memoized)."""
    if mapping_data is None:
        return False
    key = str(cluster_id)
    if key in memo:
        return memo[key]
    f1s = []
    for row in sample_rows[:3]:
        html = _coerce_html(row.get("html", ""))
        if not html.strip():
            continue
        sh, se = lbp_fn(html, mapping_data, dynamic=False)
        dh, de = lbp_fn(html, mapping_data, dynamic=True)
        if not dh or de:
            continue
        url = row.get("url", "")
        f1s.append(0.0 if (not sh or se) else _token_f1(content_fn(sh, url)[0], content_fn(dh, url)[0]))
    ok = bool(f1s) and (sum(f1s) / len(f1s) >= threshold)
    memo[key] = ok
    return ok


def _parse_element_dict(element_dict_raw: str | dict) -> dict | None:
    """Pre-parse html_element_dict to {int_layer: {tuple_key: value}} once per cluster."""
    if isinstance(element_dict_raw, dict):
        return element_dict_raw
    if not isinstance(element_dict_raw, str) or not element_dict_raw.strip():
        return None
    try:
        raw = json.loads(element_dict_raw)
        return {int(layer): {eval(k): v for k, v in layer_dict.items()} for layer, layer_dict in raw.items()}  # noqa: S307
    except Exception:
        return None


def _run_lbp(
    bindings: Any,
    params: dict[str, Any],
    html: str,
    mapping_data: dict[str, Any],
    dynamic: bool,
    _parser_cache: dict | None = None,
) -> tuple[str, str]:
    """Run LayoutBatchParser propagation. Returns (main_html, error).

    Uses the sim-gate bypass: always use main_html_body even when
    main_html_success=False (many siblings score 0.70-0.74, just below the
    0.75 threshold, but have valid extracted content).
    """
    if bindings is None:
        return "", "llm_web_kit_not_available"
    html_source = html.strip()
    if not html_source:
        return "", "empty_html"
    try:
        task_data = dict(mapping_data)
        if "_parsed_element_dict" in task_data:
            task_data["html_element_dict"] = task_data.pop("_parsed_element_dict")
        task_data.update(
            {
                "html_source": html_source,
                "dynamic_id_enable": dynamic,
                "dynamic_classid_enable": dynamic,
                "more_noise_enable": params.get("more_noise_enable", True),
                "dynamic_classid_similarity_threshold": params.get("dynamic_classid_similarity_threshold", 0.70),
            }
        )
        element_dict = task_data.get("html_element_dict")
        cache_key = id(element_dict) if element_dict is not None else None
        if _parser_cache is not None and cache_key is not None:
            if cache_key not in _parser_cache:
                _parser_cache[cache_key] = bindings.layout_parser_cls({})
            parser = _parser_cache[cache_key]
        else:
            parser = bindings.layout_parser_cls({})
        parts = parser.parse(task_data)
    except Exception as exc:
        return "", f"layout_parser_error={exc!s:.200}"
    main_html = str(parts.get("main_html_body") or "")
    if not main_html.strip():
        if parts.get("main_html_success") is False:
            return "", f"main_html_success_false sim={parts.get('main_html_sim', 'n/a')}"
        return "", "layout_parser_empty_output"
    return main_html, ""


_MAX_CONTENT_HTML_BYTES = 200_000


def _run_content_convert(mineru_bindings: Any, main_html: str, url: str) -> tuple[str, str]:
    if len(main_html) > _MAX_CONTENT_HTML_BYTES:
        main_html = main_html[:_MAX_CONTENT_HTML_BYTES]
    mb = mineru_bindings
    if mb is None:
        try:
            import lxml.html

            return lxml.html.fromstring(main_html).text_content().strip(), ""
        except Exception as exc:
            return "", f"lxml_text_fallback_error={exc!s:.100}"
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


def _apply_ratio_guard(
    candidate_html: str,
    candidate_content: str,
    mapping_data: dict[str, Any],
    min_ratio: float,
    max_ratio: float,
) -> tuple[str, str, str]:
    rep_len = (mapping_data or {}).get("_dripper_representative_content_len")
    if not rep_len or rep_len <= 0:
        return candidate_html, candidate_content, ""
    ratio = len(candidate_content) / rep_len
    if ratio < min_ratio:
        return "", "", f"content_length_ratio_low={ratio:.3f}"
    if ratio > max_ratio:
        return "", "", f"content_length_ratio_high={ratio:.3f}"
    return candidate_html, candidate_content, ""


def _try_lbp_once(
    html: str,
    url: str,
    mapping_data: dict[str, Any],
    method_name: str,
    dynamic: bool,
    lbp_fn: Callable,
    content_fn: Callable,
    min_ratio: float,
    max_ratio: float,
) -> tuple[str, str, str, str]:
    lbp_html, lbp_err = lbp_fn(html, mapping_data, dynamic=dynamic)
    if not lbp_html or lbp_err:
        return "", "", "", lbp_err
    raw_content, conv_err = content_fn(lbp_html, url)
    if conv_err:
        return "", "", "", conv_err
    ah, ac, ratio_err = _apply_ratio_guard(lbp_html, raw_content, mapping_data, min_ratio, max_ratio)
    return (ah, method_name, ac, "") if ah else ("", "", "", ratio_err)


def _sibling_propagate(
    row: dict[str, Any],
    mapping_data: dict[str, Any] | None,
    use_static: bool,
    lbp_fn: Callable,
    content_fn: Callable,
    min_ratio: float,
    max_ratio: float,
) -> dict[str, Any]:
    url, cluster_id = row.get("url", ""), row.get("cluster_id")
    html, t0 = _coerce_html(row.get("html", "")), time.perf_counter()
    method, main_html, content, error = "fallback", "", "", ""

    if mapping_data is not None:
        if use_static:
            main_html, method, content, error = _try_lbp_once(
                html, url, mapping_data, "lbp_static", False, lbp_fn, content_fn, min_ratio, max_ratio
            )
        if not main_html:
            dh, dm, dc, de = _try_lbp_once(
                html, url, mapping_data, "layout_batch_parser", True, lbp_fn, content_fn, min_ratio, max_ratio
            )
            if dh:
                main_html, method, content, error = dh, dm, dc, de
            elif de:
                error = f"static_failed({error}); dynamic_failed({de})" if error else de

    if not main_html:
        method, error = "fallback", error or "no_template_available"

    return {
        "url": url,
        "url_host_name": row.get("url_host_name", ""),
        "cluster_id": cluster_id,
        "cluster_role": "sibling",
        "dripper_content": content,
        "dripper_html": main_html,
        "dripper_error": error,
        "dripper_time_s": time.perf_counter() - t0,
        "propagation_success": bool(main_html and not error),
        "propagation_method": method,
    }


def _make_rep_or_singleton_row(row: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "url": row.get("url", ""),
        "url_host_name": row.get("url_host_name", ""),
        "cluster_id": row.get("cluster_id") if role == "representative" else None,
        "cluster_role": role,
        "dripper_content": row.get("dripper_content", ""),
        "dripper_html": row.get("dripper_html", ""),
        "dripper_error": row.get("dripper_error", ""),
        "dripper_time_s": row.get("inference_time_s", 0.0),
        "propagation_success": not bool(row.get("dripper_error", "")),
        "propagation_method": role,
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


def _dispatch_cluster_rows(
    manifest_rows: list[dict[str, Any]],
    gpu_row: dict[str, Any] | None,
    mapping_data: dict[str, Any] | None,
    cluster_id: Any,
    sib_fn: Callable,
    use_static: bool,
) -> list[dict[str, Any]]:
    results = []
    for row in manifest_rows:
        role = str(row.get("cluster_role", "singleton"))
        if role in ("representative", "singleton"):
            if gpu_row is not None:
                merged = {
                    **row,
                    "dripper_content": gpu_row.get("dripper_content", ""),
                    "dripper_html": gpu_row.get("dripper_html", gpu_row.get("llm_output_raw", "")),
                    "dripper_error": gpu_row.get("error", ""),
                    "inference_time_s": gpu_row.get("inference_time_s", 0.0),
                }
                results.append(_make_rep_or_singleton_row(merged, role))
            else:
                results.append(_make_fallback_row(row, role, f"missing_gpu_result_for_{role}"))
        elif role == "sibling":
            results.append(sib_fn(row, mapping_data, use_static))
        else:
            results.append(_make_fallback_row(row, role, f"unknown_cluster_role={role}"))
    return results


def _coerce_html(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return "" if raw is None else str(raw)


def _parse_mapping_json(raw: Any) -> dict[str, Any] | None:
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
        for loader in (lambda s: pickle.loads(base64.b64decode(s)), lambda s: json.loads(s)):
            try:
                obj = loader(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


def _load_cluster_manifest_shard(path: str) -> pd.DataFrame:
    _META = [
        "url",
        "url_host_name",
        "cluster_id",
        "cluster_role",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
    ]
    sn = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in _META if c in sn]).to_pandas()
    df.setdefault("cluster_id", None)
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
    _COLS = [
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
    sn = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in _COLS if c in sn]).to_pandas()
    if "cluster_id" not in df.columns and "layout_cluster_id" in df.columns:
        df = df.rename(columns={"layout_cluster_id": "cluster_id"})
    if "error" not in df.columns and "dripper_error" in df.columns:
        df = df.rename(columns={"dripper_error": "error"})
    return df


def _build_gpu_lookups(inference_df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_cluster: dict[str, dict[str, Any]] = {}
    by_url: dict[str, dict[str, Any]] = {}
    _null = ("none", "null", "nan", "")
    for row in inference_df.to_dict("records"):
        cid = row.get("cluster_id")
        cid_s = str(cid) if cid is not None else ""
        if cid is not None and cid_s not in by_cluster:
            by_cluster[cid_s] = row
        url = str(row.get("url") or "")
        if (cid is None or cid_s.lower() in _null) and url and url not in by_url:
            by_url[url] = row
    return by_cluster, by_url


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(tmp_path), compression="snappy")
    tmp_path.rename(out_path)


def _build_stage3_cls(
    *,
    dynamic_classid_similarity_threshold: float,
    more_noise_enable: bool,
    min_content_length_ratio: float,
    max_content_length_ratio: float,
    static_validation_min_f1: float,
    worker_count: int,
) -> type:
    """Return a ProcessingStage subclass closed over the given hyperparameters."""
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.stages.resources import Resources
    from nemo_curator.tasks import DocumentBatch as _DocumentBatch

    _params = {
        "more_noise_enable": more_noise_enable,
        "dynamic_classid_similarity_threshold": dynamic_classid_similarity_threshold,
    }
    _min, _max, _f1, _wc = min_content_length_ratio, max_content_length_ratio, static_validation_min_f1, worker_count

    class _Stage3PropagationStage(ProcessingStage[_DocumentBatch, _DocumentBatch]):
        name = "stage3_cpu_propagation"
        resources = Resources(cpus=1.0)
        batch_size = 1
        _lbp_bindings = None
        _mineru_bindings = None
        _cluster_static_ok: dict = {}  # noqa: RUF012
        _initialized = False

        def num_workers(self):
            return _wc if _wc > 0 else None

        def setup(self, worker_metadata=None):
            if self._initialized:
                return
            self._lbp_bindings = _load_lbp_bindings()
            self._mineru_bindings = _load_mineru_bindings()
            self._cluster_static_ok = {}
            self._initialized = True

        def _lbp_fn(self, html, mapping_data, dynamic=True, parser_cache=None):
            return _run_lbp(self._lbp_bindings, _params, html, mapping_data, dynamic, _parser_cache=parser_cache)

        def _content_fn(self, main_html, url):
            return _run_content_convert(self._mineru_bindings, main_html, url)

        def process(self, task):
            if not self._initialized:
                self.setup()
            ct = task._metadata.get("cluster_task", {})
            results = (
                self._process_cluster_task(ct)
                if ct
                else [
                    _make_fallback_row(r, str(r.get("cluster_role", "singleton")), "missing_cluster_task")
                    for r in task.to_pandas().to_dict("records")
                ]
            )
            return _DocumentBatch(
                dataset_name=task.dataset_name,
                data=pd.DataFrame(results, columns=OUTPUT_COLUMNS),
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
            )

        def _process_cluster_task(self, task):
            manifest_rows, gpu_row, mapping_data = task["manifest_rows"], task.get("gpu_row"), task.get("mapping_data")
            sib_rows = [r for r in manifest_rows if str(r.get("cluster_role", "")) == "sibling"]
            # One parser instance per cluster: _preprocess_template_data runs once, not once per sibling.
            _parser_cache: dict = {}
            lbp_fn_cached = lambda html, md, dynamic=True: self._lbp_fn(html, md, dynamic, parser_cache=_parser_cache)  # noqa: E731
            use_static = bool(
                sib_rows
                and mapping_data is not None
                and _cluster_static_trustworthy(
                    task.get("cluster_id"),
                    sib_rows,
                    mapping_data,
                    memo=self._cluster_static_ok,
                    lbp_fn=lbp_fn_cached,
                    content_fn=self._content_fn,
                    threshold=_f1,
                )
            )
            sib_fn = lambda row, md, us: _sibling_propagate(  # noqa: E731
                row,
                md,
                us,
                lbp_fn=lbp_fn_cached,
                content_fn=self._content_fn,
                min_ratio=_min,
                max_ratio=_max,
            )
            return _dispatch_cluster_rows(
                manifest_rows,
                gpu_row,
                mapping_data,
                task.get("cluster_id"),
                sib_fn=sib_fn,
                use_static=use_static,
            )

    return _Stage3PropagationStage


def _build_doc_tasks(tasks: list[dict[str, Any]], dataset_name: str = "stage3") -> list[Any]:
    from nemo_curator.tasks import DocumentBatch

    doc_batches = []
    for t in tasks:
        placeholder_df = pd.DataFrame(
            [{"url": r.get("url", ""), "cluster_role": r.get("cluster_role", "")} for r in t["manifest_rows"][:1]]
        )
        db = DocumentBatch(dataset_name=dataset_name, data=placeholder_df)
        db._metadata["cluster_task"] = t
        doc_batches.append(db)
    return doc_batches


def _finalize_shard(
    result_df, out_path, output_dir_path, shard_index, num_shards, my_files, total_pages, t_start
) -> dict[str, Any]:
    _atomic_write_parquet(result_df, out_path)
    ns = int(result_df["propagation_success"].fillna(False).sum())
    mth = result_df["propagation_method"]
    elapsed = time.perf_counter() - t_start
    pps = total_pages / max(elapsed, 0.001)
    metrics = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "manifest_files": len(my_files),
        "total_pages": total_pages,
        "success_pages": ns,
        "fallback_pages": len(result_df) - ns,
        "xpath_pages": int((mth == "lbp_static").sum()),
        "layout_batch_parser_pages": int((mth == "layout_batch_parser").sum()),
        "representative_pages": int((mth == "representative").sum()),
        "singleton_pages": int((mth == "singleton").sum()),
        "elapsed_s": elapsed,
        "pages_per_s": pps,
        "output_path": str(out_path),
    }
    (output_dir_path / f"metrics_shard_{shard_index:04d}.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[stage3] shard {shard_index} done  pages={total_pages:,} success={ns} "
        f"fallback={len(result_df) - ns}  xpath={metrics['xpath_pages']} "
        f"lbp={metrics['layout_batch_parser_pages']} rep={metrics['representative_pages']} "
        f"singleton={metrics['singleton_pages']}  elapsed={elapsed:.1f}s ({pps:.1f} p/s)  output={out_path}",
        flush=True,
    )
    return metrics


def _load_gpu_df(
    gpu_dir: Path, shard_index: int, manifest_cluster_ids: set[str], manifest_urls: set[str]
) -> pd.DataFrame:
    exact_gpu = gpu_dir / f"shard_{shard_index:04d}.parquet"
    gpu_files = (
        [exact_gpu]
        if exact_gpu.exists()
        else (sorted(gpu_dir.glob("shard_*.parquet")) or sorted(gpu_dir.glob("*.parquet")))
    )
    if not gpu_files:
        raise FileNotFoundError(f"No GPU inference result files found in {gpu_dir}")
    print(
        f"[stage3] loading GPU results for {len(manifest_cluster_ids):,} cluster_ids from {len(gpu_files)} file(s)...",
        flush=True,
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
                null_cid = sdf["cluster_id"].isna() | sdf["cluster_id"].astype(str).isin(("none", "null", "nan", ""))
                mask |= null_cid & sdf["url"].astype(str).isin(manifest_urls)
            if not (filtered := sdf[mask]).empty:
                gpu_frames.append(filtered)
        except Exception as exc:
            print(f"[stage3] WARNING: could not read GPU shard {f}: {exc}", flush=True)
    gpu_df = pd.concat(gpu_frames, ignore_index=True) if gpu_frames else pd.DataFrame()
    print(f"[stage3] {len(gpu_df):,} relevant GPU result rows loaded", flush=True)
    return gpu_df


def _build_cluster_tasks(manifest_df, cluster_gpu_lookup, singleton_gpu_lookup):
    """Group manifest rows by cluster into task dicts (PPT=16 siblings each, LPT order)."""
    PPT = 16
    _null = ("none", "null", "nan", "")
    groups = defaultdict(list)
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        groups[str(cid) if cid is not None and str(cid).lower() not in _null else None].append(row)
    tasks = []
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
            # Pre-parse html_element_dict once on driver so actors skip JSON+eval per sibling.
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
            tasks.append({"cluster_id": cid_key, "manifest_rows": ns + sb[:PPT], "gpu_row": gr, "mapping_data": md})
            for i in range(PPT, len(sb), PPT):
                tasks.append(
                    {"cluster_id": cid_key, "manifest_rows": sb[i : i + PPT], "gpu_row": None, "mapping_data": md}
                )
    return tasks


def process_shard(
    *,
    cluster_manifest_dir: str,
    inference_results_dir: str,
    output_dir: str,
    shard_index: int,
    num_shards: int,
    num_workers: int,
    dynamic_classid_similarity_threshold: float = 0.70,
    more_noise_enable: bool = True,
    min_content_length_ratio: float = 0.25,
    max_content_length_ratio: float = 4.0,
    static_validation_min_f1: float = 0.97,
) -> dict[str, Any]:
    """Process one shard's worth of cluster assignments using RayActorPoolExecutor."""
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline

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

    n = len(manifest_files)
    my_files = manifest_files[n * shard_index // num_shards : n * (shard_index + 1) // num_shards]
    if not my_files:
        print(f"[stage3] shard {shard_index}: no manifest files — writing empty shard", flush=True)
        _atomic_write_parquet(pd.DataFrame(columns=OUTPUT_COLUMNS), out_path)
        return {"status": "empty", "shard": shard_index, "rows": 0}

    manifest_df = pd.concat([_load_cluster_manifest_shard(str(f)) for f in my_files], ignore_index=True)
    print(
        f"[stage3] shard {shard_index}/{num_shards}: {len(manifest_df):,} rows from {len(my_files)} file(s)",
        flush=True,
    )

    records = manifest_df.to_dict("records")
    _null = ("none", "null", "nan", "")
    manifest_cluster_ids: set[str] = {
        str(r["cluster_id"])
        for r in records
        if r.get("cluster_id") is not None and str(r["cluster_id"]).lower() not in _null
    }
    manifest_urls: set[str] = {str(r.get("url", "")) for r in records}

    gpu_df = _load_gpu_df(gpu_dir, shard_index, manifest_cluster_ids, manifest_urls)
    cluster_gpu_lookup, singleton_gpu_lookup = _build_gpu_lookups(gpu_df)
    del gpu_df

    tasks = _build_cluster_tasks(manifest_df, cluster_gpu_lookup, singleton_gpu_lookup)
    del manifest_df, cluster_gpu_lookup, singleton_gpu_lookup
    tasks.sort(key=lambda t: len(t["manifest_rows"]), reverse=True)  # LPT: largest first

    total_pages = sum(len(t["manifest_rows"]) for t in tasks)
    print(f"[stage3] shard {shard_index}: {len(tasks):,} cluster tasks, {total_pages:,} pages", flush=True)

    hp = dict(
        dynamic_classid_similarity_threshold=dynamic_classid_similarity_threshold,
        more_noise_enable=more_noise_enable,
        min_content_length_ratio=min_content_length_ratio,
        max_content_length_ratio=max_content_length_ratio,
        static_validation_min_f1=static_validation_min_f1,
    )
    doc_tasks = _build_doc_tasks(tasks)
    pipeline = Pipeline(name="stage3_cpu_propagation")
    pipeline.add_stage(_build_stage3_cls(**hp, worker_count=num_workers)())
    print(
        f"[stage3] submitting {len(doc_tasks):,} tasks to RayActorPoolExecutor ({num_workers} actors)...", flush=True
    )
    t_exec = time.perf_counter()
    output_doc_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=doc_tasks) or []
    print(f"[stage3] RayActorPoolExecutor finished in {time.perf_counter() - t_exec:.1f}s", flush=True)

    frames = [t.to_pandas().reindex(columns=OUTPUT_COLUMNS) for t in output_doc_tasks]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return _finalize_shard(
        result_df, out_path, output_dir_path, shard_index, num_shards, my_files, total_pages, t_start
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: CPU template propagation for CC-scale pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cluster-manifest", required=True, help="cluster_assignments/ shard dir (Stage 1 output)")
    p.add_argument("--inference-results", required=True, help="gpu_results/ shard dir (Stage 2 output)")
    p.add_argument("--output-dir", required=True, help="Output dir for propagation_results/ shards")
    p.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="0-based task index (default: SLURM_ARRAY_TASK_ID)",
    )
    p.add_argument("--num-shards", type=int, default=80)
    p.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 64)),
        help="Ray actor count per node (default: SLURM_CPUS_PER_TASK or 64)",
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
    print(
        f"[stage3] cluster_manifest={args.cluster_manifest}  inference_results={args.inference_results}  "
        f"output_dir={args.output_dir}  shard={args.shard_index}/{args.num_shards}  num_workers={args.num_workers}",
        flush=True,
    )
    metrics = process_shard(
        cluster_manifest_dir=args.cluster_manifest,
        inference_results_dir=args.inference_results,
        output_dir=args.output_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        num_workers=args.num_workers,
    )
    status = metrics.get("status", "done")
    msg = {"skipped": "already complete — skipped.", "empty": "had no input — wrote empty shard."}.get(
        status, "complete."
    )
    print(f"[stage3] Shard {args.shard_index} {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
