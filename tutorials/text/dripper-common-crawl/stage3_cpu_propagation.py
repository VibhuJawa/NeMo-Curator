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

Backends:
  1. ProcessPoolExecutor (fallback): spawn-context worker pool.
  2. RayActorPoolExecutor (preferred): fixed actor pool via NeMo Curator Pipeline.

Auto-detection: Ray is used when nemo_curator.backends.ray_actor_pool is importable.
Pass --no-ray to force the ProcessPoolExecutor path.
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
from collections.abc import Callable
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

# Module-level globals for ProcessPoolExecutor workers only.
_WORKER_BINDINGS: Any = None
_WORKER_MINERU_BINDINGS: Any = None
_WORKER_PARAMS: dict[str, Any] = {}
_WORKER_INITIALIZED: bool = False
_CLUSTER_STATIC_OK: dict[str, bool] = {}


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


def _worker_init(dct: float, nme: bool, minr: float, maxr: float, f1: float, log_level: str) -> None:
    global _WORKER_BINDINGS, _WORKER_MINERU_BINDINGS, _WORKER_PARAMS, _WORKER_INITIALIZED
    if _WORKER_INITIALIZED:
        return
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO), format="%(processName)s %(levelname)s %(message)s"
    )
    _WORKER_PARAMS = {
        "dynamic_classid_similarity_threshold": dct,
        "more_noise_enable": nme,
        "min_content_length_ratio": minr,
        "max_content_length_ratio": maxr,
        "static_validation_min_f1": f1,
    }
    _WORKER_BINDINGS = _load_lbp_bindings()
    _WORKER_MINERU_BINDINGS = _load_mineru_bindings()
    _WORKER_INITIALIZED = True


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


def _process_cluster_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Process one cluster in a ProcessPoolExecutor worker."""
    manifest_rows, gpu_row, mapping_data = task["manifest_rows"], task.get("gpu_row"), task.get("mapping_data")
    sib_rows = [r for r in manifest_rows if str(r.get("cluster_role", "")) == "sibling"]

    def _lbp_fn(html, md, dynamic=True):
        return _run_lbp(_WORKER_BINDINGS, _WORKER_PARAMS, html, md, dynamic)

    def _content_fn(main_html, url):
        return _run_content_convert(_WORKER_MINERU_BINDINGS, main_html, url)

    use_static = bool(
        sib_rows
        and mapping_data is not None
        and _cluster_static_trustworthy(
            task.get("cluster_id"),
            sib_rows,
            mapping_data,
            memo=_CLUSTER_STATIC_OK,
            lbp_fn=_lbp_fn,
            content_fn=_content_fn,
            threshold=_WORKER_PARAMS.get("static_validation_min_f1", 0.97),
        )
    )

    def _sib_fn(row, md, us):
        return _sibling_propagate(
            row,
            md,
            us,
            lbp_fn=_lbp_fn,
            content_fn=_content_fn,
            min_ratio=_WORKER_PARAMS.get("min_content_length_ratio", 0.25),
            max_ratio=_WORKER_PARAMS.get("max_content_length_ratio", 4.0),
        )

    return _dispatch_cluster_rows(
        manifest_rows, gpu_row, mapping_data, task.get("cluster_id"), sib_fn=_sib_fn, use_static=use_static
    )


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
    meta_cols = [
        "url",
        "url_host_name",
        "cluster_id",
        "cluster_role",
        "warc_filename",
        "warc_record_offset",
        "warc_record_length",
    ]
    sn = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in meta_cols if c in sn]).to_pandas()
    if "cluster_id" not in df.columns:
        df["cluster_id"] = None
    if "cluster_role" not in df.columns:
        df["cluster_role"] = "singleton"
    df["html"] = None
    if "html" in sn:
        smask = df["cluster_role"] == "sibling"
        if smask.any():
            hdf = pq.read_table(path, columns=["url", "html"]).to_pandas().drop_duplicates("url", keep="first")
            df["html"] = df["url"].map(hdf.set_index("url")["html"])
            df.loc[~smask, "html"] = None
    return df


def _load_inference_results(path: str) -> pd.DataFrame:
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


def _ray_available() -> bool:
    try:
        from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor  # noqa: F401

        return True
    except Exception:
        return False


def _finalize_shard(
    result_df, out_path, output_dir_path, shard_index, num_shards, my_files, total_pages, t_start, backend
) -> dict[str, Any]:
    _atomic_write_parquet(result_df, out_path)
    ns = int(result_df["propagation_success"].fillna(False).sum())
    mth = result_df["propagation_method"]
    elapsed = time.perf_counter() - t_start
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
        "pages_per_s": total_pages / max(elapsed, 0.001),
        "output_path": str(out_path),
        "backend": backend,
    }
    (output_dir_path / f"metrics_shard_{shard_index:04d}.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[stage3] shard {shard_index} DONE ({backend})\n"
        f"  pages: {total_pages:,} (success={ns} fallback={len(result_df) - ns})\n"
        f"  xpath={metrics['xpath_pages']} lbp={metrics['layout_batch_parser_pages']} "
        f"rep={metrics['representative_pages']} singleton={metrics['singleton_pages']}\n"
        f"  elapsed={elapsed:.1f}s ({metrics['pages_per_s']:.1f} p/s)  output={out_path}",
        flush=True,
    )
    return metrics


def _load_gpu_df(
    gpu_dir: Path,
    shard_index: int,
    manifest_cluster_ids: set[str],
    manifest_urls: set[str],
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
        f"[stage3] loading GPU results for {len(manifest_cluster_ids):,} cluster_ids "
        f"from {len(gpu_files)} GPU shard file(s)...",
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
            filtered = sdf[mask]
            if not filtered.empty:
                gpu_frames.append(filtered)
        except Exception as exc:
            print(f"[stage3] WARNING: could not read GPU shard {f}: {exc}", flush=True)
    gpu_df = pd.concat(gpu_frames, ignore_index=True) if gpu_frames else pd.DataFrame()
    print(f"[stage3] {len(gpu_df):,} relevant GPU result rows loaded", flush=True)
    return gpu_df


def _build_cluster_tasks(manifest_df, cluster_gpu_lookup, singleton_gpu_lookup):
    """Group manifest rows by cluster and build task dicts.

    PPT=16: each task owns 16 siblings for optimal Ray scheduling overhead vs
    parallelism tradeoff. Siblings sorted by HTML size descending (LPT) to ensure
    heavy-HTML siblings start early.
    """
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
    dynamic_classid_similarity_threshold: float,
    more_noise_enable: bool,
    min_content_length_ratio: float,
    max_content_length_ratio: float,
    static_validation_min_f1: float,
    log_level: str,
    cluster_chunk_size: int,
    use_ray: bool | None = None,
) -> dict[str, Any]:
    """Process one shard's worth of cluster assignments.

    use_ray: True=force Ray, False=force ProcessPool, None=auto-detect.
    """
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

    records = manifest_df.to_dict("records")
    manifest_cluster_ids: set[str] = {
        str(r["cluster_id"])
        for r in records
        if r.get("cluster_id") is not None and str(r["cluster_id"]).lower() not in ("none", "null", "nan", "")
    }
    manifest_urls: set[str] = {str(r.get("url", "")) for r in records}

    gpu_df = _load_gpu_df(gpu_dir, shard_index, manifest_cluster_ids, manifest_urls)
    cluster_gpu_lookup, singleton_gpu_lookup = _build_gpu_lookups(gpu_df)
    del gpu_df

    print("[stage3] building cluster tasks...", flush=True)
    tasks = _build_cluster_tasks(manifest_df, cluster_gpu_lookup, singleton_gpu_lookup)
    del manifest_df, cluster_gpu_lookup, singleton_gpu_lookup

    # LPT sort: largest clusters first to prevent tail latency.
    tasks.sort(key=lambda t: len(t["manifest_rows"]), reverse=True)

    total_tasks = len(tasks)
    total_pages = sum(len(t["manifest_rows"]) for t in tasks)
    print(f"[stage3] shard {shard_index}: {total_tasks:,} cluster tasks, {total_pages:,} pages", flush=True)

    _want_ray = _ray_available() if use_ray is None else use_ray
    if use_ray is None:
        print(
            f"[stage3] backend auto-detect: {'RayActorPoolExecutor' if _want_ray else 'ProcessPoolExecutor'}",
            flush=True,
        )

    hp = dict(
        dynamic_classid_similarity_threshold=dynamic_classid_similarity_threshold,
        more_noise_enable=more_noise_enable,
        min_content_length_ratio=min_content_length_ratio,
        max_content_length_ratio=max_content_length_ratio,
        static_validation_min_f1=static_validation_min_f1,
    )
    base = dict(
        tasks=tasks,
        shard_index=shard_index,
        num_shards=num_shards,
        num_workers=num_workers,
        out_path=out_path,
        output_dir_path=output_dir_path,
        my_files=my_files,
        total_pages=total_pages,
        t_start=t_start,
    )

    if _want_ray:
        return _run_with_ray(**base, hp=hp)
    return _run_with_process_pool(
        **base,
        hp=hp,
        log_level=log_level,
        cluster_chunk_size=cluster_chunk_size,
        total_tasks=total_tasks,
    )


def _run_with_ray(
    *,
    tasks: list[dict[str, Any]],
    shard_index: int,
    num_shards: int,
    num_workers: int,
    hp: dict[str, Any],
    out_path: Path,
    output_dir_path: Path,
    my_files: list[Path],
    total_pages: int,
    t_start: float,
) -> dict[str, Any]:
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline

    print(f"[stage3] using RayActorPoolExecutor with {num_workers} actors", flush=True)
    doc_tasks = _build_doc_tasks(tasks)
    stage_cls = _build_stage3_cls(**hp, worker_count=num_workers)
    pipeline = Pipeline(name="stage3_cpu_propagation")
    pipeline.add_stage(stage_cls())
    print(f"[stage3] shard {shard_index}: submitting {len(doc_tasks):,} tasks to RayActorPoolExecutor...", flush=True)
    t_exec = time.perf_counter()
    output_doc_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=doc_tasks) or []
    print(
        f"[stage3] RayActorPoolExecutor finished in {time.perf_counter() - t_exec:.1f}s, collecting results...",
        flush=True,
    )
    frames = [t.to_pandas().reindex(columns=OUTPUT_COLUMNS) for t in output_doc_tasks]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    return _finalize_shard(
        result_df, out_path, output_dir_path, shard_index, num_shards, my_files, total_pages, t_start, "ray"
    )


def _run_with_process_pool(
    *,
    tasks: list[dict[str, Any]],
    shard_index: int,
    num_shards: int,
    num_workers: int,
    hp: dict[str, Any],
    log_level: str,
    cluster_chunk_size: int,
    out_path: Path,
    output_dir_path: Path,
    my_files: list[Path],
    total_tasks: int,
    total_pages: int,
    t_start: float,
) -> dict[str, Any]:
    print(f"[stage3] using ProcessPoolExecutor with {num_workers} workers", flush=True)
    worker_initargs = (
        hp["dynamic_classid_similarity_threshold"],
        hp["more_noise_enable"],
        hp["min_content_length_ratio"],
        hp["max_content_length_ratio"],
        hp["static_validation_min_f1"],
        log_level,
    )
    all_results: list[dict[str, Any]] = []
    n_success = n_fallback = n_xpath = n_lbp = pages_done = 0
    t_proc_start = time.perf_counter()
    chunk_size = max(cluster_chunk_size, 1)
    num_chunks = (total_tasks + chunk_size - 1) // chunk_size
    ctx = multiprocessing.get_context("spawn")

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
                n_success += bool(r.get("propagation_success"))
                n_fallback += not bool(r.get("propagation_success"))
                n_xpath += meth in ("xpath", "lbp_static")
                n_lbp += meth == "layout_batch_parser"
            pages_done += sum(len(t["manifest_rows"]) for t in chunk)
            elapsed = time.perf_counter() - t_proc_start
            print(
                f"[stage3] shard {shard_index}: chunk {chunk_idx + 1}/{num_chunks} "
                f"pages={pages_done:,}/{total_pages:,} rate={pages_done / max(elapsed, 0.001):.1f} pages/s  "
                f"success={n_success} fallback={n_fallback} xpath={n_xpath} lbp={n_lbp}",
                flush=True,
            )

    result_df = pd.DataFrame(all_results, columns=OUTPUT_COLUMNS)
    return _finalize_shard(
        result_df, out_path, output_dir_path, shard_index, num_shards, my_files, total_pages, t_start, "process_pool"
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
        help="Parallel workers per node (default: SLURM_CPUS_PER_TASK or 64)",
    )
    p.add_argument("--cluster-chunk-size", type=int, default=500, help="Cluster tasks per process-pool chunk")
    p.add_argument("--dynamic-classid-similarity-threshold", type=float, default=0.70)
    p.add_argument("--more-noise-enable", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-content-length-ratio", type=float, default=0.25)
    p.add_argument("--max-content-length-ratio", type=float, default=4.0)
    p.add_argument(
        "--static-validation-min-f1",
        type=float,
        default=0.97,
        help="Min token-F1 (static vs dynamic LBP on K=3 siblings) to trust static propagation.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    _ray_default = _ray_available()
    p.add_argument(
        "--use-ray",
        action=argparse.BooleanOptionalAction,
        default=_ray_default,
        help=f"Use RayActorPoolExecutor (default: {_ray_default}, auto-detected).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    be = "RayActorPoolExecutor" if args.use_ray else "ProcessPoolExecutor"
    sep = "=" * 70
    print(f"{sep}\n  Stage 3: CPU Template Propagation  [{be}]\n{sep}", flush=True)
    print(
        f"  cluster_manifest:  {args.cluster_manifest}\n"
        f"  inference_results: {args.inference_results}\n"
        f"  output_dir:        {args.output_dir}\n"
        f"  shard:             {args.shard_index}/{args.num_shards}\n"
        f"  num_workers:       {args.num_workers}\n"
        f"  classid_threshold: {args.dynamic_classid_similarity_threshold}\n"
        f"  content_ratio:     [{args.min_content_length_ratio}, {args.max_content_length_ratio}]\n"
        f"  static_val_f1:     {args.static_validation_min_f1}\n"
        f"  backend:           {be}\n{sep}",
        flush=True,
    )
    a = vars(args)
    metrics = process_shard(
        cluster_manifest_dir=a["cluster_manifest"],
        inference_results_dir=a["inference_results"],
        output_dir=a["output_dir"],
        shard_index=a["shard_index"],
        num_shards=a["num_shards"],
        num_workers=a["num_workers"],
        dynamic_classid_similarity_threshold=a["dynamic_classid_similarity_threshold"],
        more_noise_enable=a["more_noise_enable"],
        min_content_length_ratio=a["min_content_length_ratio"],
        max_content_length_ratio=a["max_content_length_ratio"],
        static_validation_min_f1=a["static_validation_min_f1"],
        log_level=a["log_level"],
        cluster_chunk_size=a["cluster_chunk_size"],
        use_ray=a["use_ray"],
    )
    status = metrics.get("status", "done")
    msg = {"skipped": "already complete — skipped.", "empty": "had no input — wrote empty shard."}.get(
        status, "complete."
    )
    print(f"[stage3] Shard {args.shard_index} {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
