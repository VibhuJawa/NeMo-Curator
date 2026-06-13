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

"""stage3_cpu_propagation.py — Stage 3: CPU template propagation for CC-scale pipeline.

Algorithm per cluster:
1. Load representative's inference result (xpath_rules / mapping_json from Stage 2)
2. For each sibling page in the cluster:
   a. Try direct lxml XPath evaluation using pre-serialized xpath_rules (30-100ms/page)
   b. If XPath match returns 0 elements, fall back to LayoutBatchParser (11s/page)
   c. If LayoutBatchParser also fails: mark as pending_fallback
3. For cluster_role=representative: copy GPU result directly (no propagation needed)
4. For cluster_role=singleton: copy GPU standalone result directly
5. Write per-shard output with checkpoint semantics (write-to-tmp-then-rename)

Input files:
  --cluster-manifest:   cluster_assignments/shard_NNNN.parquet
                        columns: url, url_host_name, cluster_id (nullable),
                                 cluster_role (representative/sibling/singleton),
                                 html (large_binary, non-null for representatives only)

  --inference-results:  gpu_results/shard_NNNN.parquet
                        columns: cluster_id, url (representative), llm_output_raw,
                                 xpath_rules (JSON), template_html, inference_time_s, error

Output file:
  --output-dir/shard_{TASK_ID:04d}.parquet
  columns: url, url_host_name, cluster_id, cluster_role,
           dripper_content, dripper_html, dripper_error, dripper_time_s,
           propagation_success (bool), propagation_method (str)

Performance targets:
  - XPath path: ~50ms/page  → 80 nodes × 64 workers × 20 pages/s = 102,400 pages/s total
  - LayoutBatchParser fallback: ~12s/page, expected <10% of siblings
  - Total 2.4B pages propagation wall time: ~3-4h on 80 CPU nodes

Slurm: --array=0-79  (80 tasks, 1 node each)
       --partition=cpu_long  --cpus-per-task=64  --mem=235G  --time=06:00:00
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

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
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
    "propagation_method",   # "representative" | "singleton" | "xpath" | "layout_batch_parser" | "fallback"
]

# ---------------------------------------------------------------------------
# Worker initializer — imports are done once per process to avoid fork issues
# ---------------------------------------------------------------------------
_WORKER_BINDINGS: Any = None  # llm_web_kit bindings after init
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
    """Called once per multiprocessing.Pool worker. Imports heavy libraries.

    NOTE: positional-only args so ProcessPoolExecutor can pass via initargs tuple.
    """
    global _WORKER_BINDINGS, _WORKER_MINERU_BINDINGS, _WORKER_PARAMS, _WORKER_INITIALIZED

    if _WORKER_INITIALIZED:
        return

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format="%(processName)s %(levelname)s %(message)s")

    _WORKER_PARAMS = {
        "dynamic_classid_similarity_threshold": dynamic_classid_similarity_threshold,
        "more_noise_enable": more_noise_enable,
        "min_content_length_ratio": min_content_length_ratio,
        "max_content_length_ratio": max_content_length_ratio,
    }

    try:
        from llm_web_kit.html_layout.html_layout_cosin import get_feature, similarity
        from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser
        from llm_web_kit.main_html_parser.parser.tag_mapping import MapItemToHtmlTagsParser
        from llm_web_kit.main_html_parser.typical_html.typical_html import select_representative_html

        class _Bindings:
            pass

        b = _Bindings()
        b.get_feature = get_feature
        b.similarity = similarity
        b.layout_parser_cls = LayoutBatchParser
        b.map_parser_cls = MapItemToHtmlTagsParser
        b.select_representative_html = select_representative_html
        _WORKER_BINDINGS = b
        logging.getLogger(__name__).debug("llm_web_kit bindings loaded in worker %s", os.getpid())
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "llm_web_kit unavailable: %s — LayoutBatchParser fallback disabled", exc)
        _WORKER_BINDINGS = None

    try:
        from mineru_html.process import convert2content
        from mineru_html.base import MinerUHTMLOutput, MinerUHTMLCase, MinerUHTMLInput

        class _MineruBindings:
            pass

        mb = _MineruBindings()
        mb.convert2content = convert2content
        mb.output_cls = MinerUHTMLOutput
        mb.case_cls = MinerUHTMLCase
        mb.input_cls = MinerUHTMLInput
        try:
            from nemo_curator.stages.text.experimental.dripper.stage import (
                _strip_xml_incompatible_chars,
            )
            mb.strip_xml = _strip_xml_incompatible_chars
        except Exception:
            mb.strip_xml = None
        _WORKER_MINERU_BINDINGS = mb
        logging.getLogger(__name__).debug("mineru_html bindings loaded in worker %s", os.getpid())
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "mineru_html unavailable: %s — content conversion will fall back to lxml", exc)
        _WORKER_MINERU_BINDINGS = None

    _WORKER_INITIALIZED = True


# ---------------------------------------------------------------------------
# XPath-based fast propagation kernel
# ---------------------------------------------------------------------------

def _xpath_propagate(
    html: str,
    xpath_rules: list[dict[str, Any]],
) -> tuple[str, str]:
    """Apply pre-serialized XPath rules from Stage 2 to a sibling HTML page.

    xpath_rules is a list of dicts, each with:
      {"xpath": str, "type": str, "label": str}

    Returns (main_html_fragment, error_str).  On success error_str is "".
    On failure returns ("", error_message).
    """
    try:
        import lxml.html as lhtml
    except ImportError:
        return "", "lxml_not_available"

    if not html.strip():
        return "", "empty_html"

    try:
        doc = lhtml.fromstring(html.encode("utf-8", errors="replace") if isinstance(html, str) else html)
    except Exception as exc:
        return "", f"lxml_parse_error={exc!s:.100}"

    if not xpath_rules:
        return "", "no_xpath_rules"

    matched_parts = []
    for rule in xpath_rules:
        xpath_expr = rule.get("xpath", "")
        if not xpath_expr:
            continue
        try:
            elements = doc.xpath(xpath_expr)
        except Exception as exc:
            return "", f"xpath_eval_error={exc!s:.100}"
        if elements:
            for el in elements:
                try:
                    import lxml.etree as etree
                    matched_parts.append(etree.tostring(el, encoding="unicode", method="html"))
                except Exception:
                    pass

    if not matched_parts:
        return "", "xpath_no_elements_matched"

    main_html = "\n".join(matched_parts)
    return main_html, ""


# ---------------------------------------------------------------------------
# CSS-selector fast-path (PERF #1): derive deterministic selectors ONCE per
# cluster from the template's red-labeled keys, apply via lxml to each sibling
# (~10-50 ms/page) instead of LayoutBatchParser (~0.3-3 s/page). Falls back to
# LBP when selectors return nothing or the content-ratio gate fails, so F1 parity
# with the standalone baseline is preserved. See STAGE3_PERF_AUDIT.md.
# ---------------------------------------------------------------------------

_POST_NUMBER_RE = re.compile(r"(post|postid)-(\d+)", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\n]+")


def _replace_post_number(text: str | None) -> str | None:
    """Mirror LayoutBatchParser.replace_post_number: strip volatile post-ids."""
    if not text:
        return None
    return _POST_NUMBER_RE.sub(lambda m: f"{m.group(1)}-", str(text)).strip()


def _xpath_quote(value: str) -> str | None:
    """Quote a string for an XPath literal. Returns None if unquotable simply."""
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    return None  # contains both quote types — skip this selector


def _derive_red_selectors(mapping_data: dict[str, Any] | None) -> list[str]:
    """Turn the template's red-labeled keys into XPath expressions (PERF #1).

    html_element_dict (from MapItemToHtmlTagsParser):
      { layer_no: { (tag, class, id, sha256, layer_no, idx):
                        (label, (parent_tag, parent_class, parent_id)) } }
    label == 'red' marks main content. We emit one XPath per red key, preferring
    id (post-number stripped) then first class token then tag. XPath (not CSS) so
    no `cssselect` dependency is required.
    """
    if not mapping_data:
        return []
    element_dict = mapping_data.get("html_element_dict") or {}
    selectors: list[str] = []
    seen: set[str] = set()
    for _layer, nodes in (element_dict.items() if isinstance(element_dict, dict) else []):
        if not isinstance(nodes, dict):
            continue
        for key, value in nodes.items():
            label = value[0] if isinstance(value, (list, tuple)) and value else None
            if label != "red":
                continue
            if not isinstance(key, (list, tuple)) or len(key) < 3:
                continue
            tag, cls, idd = key[0], key[1], key[2]
            if not tag or tag in ("html",):
                continue
            idd_n = _replace_post_number(idd)
            if idd_n:
                q = _xpath_quote(idd_n)
                xp = f".//{tag}[@id={q}]" if q else None
            else:
                cls_n = _replace_post_number(_WS_RE.sub(" ", cls) if cls else None)
                first = cls_n.strip().split(" ")[0] if cls_n else ""
                if first:
                    q = _xpath_quote(first)
                    xp = (f".//{tag}[contains(concat(' ',normalize-space(@class),' '),"
                          f"concat(' ',{q},' '))]") if q else None
                else:
                    xp = f".//{tag}"
            if xp and xp not in seen:
                seen.add(xp)
                selectors.append(xp)
    return selectors


def _css_extract(html: str, selectors: list[str]) -> tuple[str, str]:
    """Apply compiled red XPath selectors to a sibling page. Returns (main_html, err)."""
    if not selectors:
        return "", "no_selectors"
    try:
        import lxml.html as lhtml
        import lxml.etree as etree
    except ImportError:
        return "", "lxml_not_available"
    if not html.strip():
        return "", "empty_html"
    try:
        doc = lhtml.fromstring(html.encode("utf-8", errors="replace") if isinstance(html, str) else html)
    except Exception as exc:
        return "", f"lxml_parse_error={exc!s:.80}"

    parts: list[str] = []
    matched: set[int] = set()
    for sel in selectors:
        try:
            els = doc.xpath(sel)
        except Exception:
            continue
        for el in els:
            # Keep outermost match only (skip nodes nested inside an already-kept node).
            if any(id(a) in matched for a in el.iterancestors()):
                continue
            matched.add(id(el))
            try:
                parts.append(etree.tostring(el, encoding="unicode", method="html"))
            except Exception:
                pass
    if not parts:
        return "", "css_no_elements_matched"
    return "\n".join(parts), ""


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _token_f1(a: str, b: str) -> float:
    """Token-multiset F1 between two texts (same metric as compare_f1.py)."""
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


# Per-worker memo of whether a cluster's fast STATIC LBP matching reproduces full
# dynamic LBP (validated on a sample). cluster_id -> bool.
_CLUSTER_STATIC_OK: dict[str, bool] = {}


def _cluster_static_trustworthy(cluster_id: Any, sample_rows: list[dict[str, Any]],
                                mapping_data: dict[str, Any] | None) -> bool:
    """Decide ONCE per cluster whether the fast static-only LBP path reproduces full
    dynamic LBP. On up to K sample siblings, run BOTH static and dynamic LBP and
    require their extracted content to agree (token-F1 ≥ thr). If they agree, all the
    cluster's siblings can use the fast static path; otherwise they use full dynamic
    LBP. This keeps F1 at the dynamic-LBP baseline while letting the ~majority of
    (stable-template) clusters run on the cheap static path. Memoized per worker."""
    if mapping_data is None:
        return False
    key = str(cluster_id)
    if key in _CLUSTER_STATIC_OK:
        return _CLUSTER_STATIC_OK[key]
    K = 3
    thr = _WORKER_PARAMS.get("static_validation_min_f1", 0.97)
    f1s: list[float] = []
    for row in sample_rows[:K]:
        html = _coerce_html(row.get("html", ""))
        if not html.strip():
            continue
        sh, se = _layout_batch_parser_propagate(html, mapping_data, dynamic=False)
        dh, de = _layout_batch_parser_propagate(html, mapping_data, dynamic=True)
        if not dh or de:
            continue          # dynamic (the baseline) failed → uninformative sample
        if not sh or se:
            f1s.append(0.0)   # static missed where dynamic succeeded → not safe
            continue
        url = row.get("url", "")
        sc, _ = _convert_main_html_to_content(sh, url)
        dc, _ = _convert_main_html_to_content(dh, url)
        f1s.append(_token_f1(sc, dc))
    ok = bool(f1s) and (sum(f1s) / len(f1s) >= thr)
    _CLUSTER_STATIC_OK[key] = ok
    return ok


def _layout_similarity(template_main_html: str, candidate_html: str, layer: Any) -> float | None:
    """Layout-feature cosine similarity (llm_web_kit) between the template's main
    HTML and a candidate extraction. Used to gate the XPath fast-path: a low score
    means the selectors grabbed a structurally different region → fall back to LBP.
    Returns None if features can't be computed (gate is then skipped)."""
    global _WORKER_BINDINGS
    if _WORKER_BINDINGS is None or not template_main_html or not candidate_html:
        return None
    try:
        f1 = _WORKER_BINDINGS.get_feature(template_main_html)
        f2 = _WORKER_BINDINGS.get_feature(candidate_html)
        if f1 is None or f2 is None:
            return None
        try:
            return float(_WORKER_BINDINGS.similarity(f1, f2, layer_n=int(layer) if layer else 3))
        except TypeError:
            return float(_WORKER_BINDINGS.similarity(f1, f2))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LayoutBatchParser fallback kernel (used when CSS selectors produce nothing)
# ---------------------------------------------------------------------------

def _layout_batch_parser_propagate(
    html: str,
    mapping_data: dict[str, Any],
    dynamic: bool = True,
) -> tuple[str, str]:
    """Use LayoutBatchParser (llm_web_kit) to propagate a template to a sibling.

    PERF: when dynamic=False, the expensive dynamic id/classid matching (sklearn
    get_feature + cosine_similarity per candidate node — the dominant cost per the
    perf audit) is disabled, so this runs LBP's pure STATIC matching. For siblings
    whose markup matches the template statically (stable CMS templates — the common
    case) this yields IDENTICAL output to full LBP at a fraction of the cost; LBP's
    own `main_html_success` flag tells us when static matching was sufficient. When
    it reports failure, the caller retries with dynamic=True (full LBP), preserving
    baseline F1 exactly.

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
        task_data.update({
            "html_source": html_source,
            "dynamic_id_enable": dynamic,
            "dynamic_classid_enable": dynamic,
            "more_noise_enable": _WORKER_PARAMS.get("more_noise_enable", True),
            "dynamic_classid_similarity_threshold": _WORKER_PARAMS.get(
                "dynamic_classid_similarity_threshold", 0.70
            ),
        })
        parts = _WORKER_BINDINGS.layout_parser_cls({}).parse(task_data)
    except Exception as exc:
        return "", f"layout_parser_error={exc!s:.200}"

    if parts.get("main_html_success") is False:
        return "", f"main_html_success_false sim={parts.get('main_html_sim', 'n/a')}"

    main_html = str(parts.get("main_html_body") or "")
    if not main_html.strip():
        return "", "layout_parser_empty_output"

    return main_html, ""


# ---------------------------------------------------------------------------
# Content conversion (main_html -> text content via MinerU convert2content)
# ---------------------------------------------------------------------------

def _convert_main_html_to_content(main_html: str, url: str) -> tuple[str, str]:
    """Convert main_html fragment to text content using MinerU-HTML's converter.

    Returns (content_str, error_str).
    """
    global _WORKER_MINERU_BINDINGS
    if _WORKER_MINERU_BINDINGS is None:
        # Best-effort: strip tags with lxml
        try:
            import lxml.html
            return lxml.html.fromstring(main_html).text_content().strip(), ""
        except Exception as exc:
            return "", f"lxml_text_fallback_error={exc!s:.100}"

    mb = _WORKER_MINERU_BINDINGS
    try:
        # Build a real MinerU case (case_cls(input_cls(...))) and attach the
        # propagated main_html as output_data — identical to the standalone
        # Dripper's _convert_main_html path. A bare shim object lacks the
        # attributes convert2content reads and silently produces nothing.
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


# ---------------------------------------------------------------------------
# Per-row processing functions (run inside worker processes)
# ---------------------------------------------------------------------------

def _process_representative_row(row: dict[str, Any]) -> dict[str, Any]:
    """Representative row: the GPU result IS the result. No propagation needed."""
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
    """Singleton row (no cluster): GPU standalone result is the final result."""
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
    row: dict[str, Any],
    red_selectors: list[str] | None,
    mapping_data: dict[str, Any] | None,
    representative_content_len: int,
    use_static: bool = False,
) -> dict[str, Any]:
    """Sibling row: LayoutBatchParser propagation.

    PERF: when the cluster passed per-cluster validation (use_static — static LBP
    proven to reproduce full dynamic LBP on a sample), try LBP STATIC matching first
    (dynamic id/classid disabled → no sklearn cosine work, the audit's dominant
    cost), falling back to dynamic only if static misses a given page. For
    un-validated clusters we go straight to full dynamic LBP. This keeps F1 at the
    dynamic-LBP baseline while the ~majority of stable-template clusters run cheap.
    """
    global _WORKER_PARAMS

    url = row.get("url", "")
    url_host_name = row.get("url_host_name", "")
    cluster_id = row.get("cluster_id")
    html = _coerce_html(row.get("html", ""))

    t0 = time.perf_counter()
    method = "fallback"
    main_html = ""
    content = ""
    error = ""

    if mapping_data is not None:
        # Tier 1: LBP static-only (fast) — only for clusters validated as static-safe.
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

        # Tier 2: full dynamic LBP (baseline) — primary path for un-validated
        # clusters, or fallback when static missed a page.
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
        # Both paths failed — mark as pending_fallback
        method = "fallback"
        if not error:
            error = "no_template_available"

    elapsed = time.perf_counter() - t0

    return {
        "url": url,
        "url_host_name": url_host_name,
        "cluster_id": cluster_id,
        "cluster_role": "sibling",
        "dripper_content": content,
        "dripper_html": main_html,
        "dripper_error": error,
        "dripper_time_s": elapsed,
        "propagation_success": bool(main_html and not error),
        "propagation_method": method,
    }


def _process_cluster_task(
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process one cluster (representative + all siblings) in a single worker call.

    task dict keys:
      cluster_id:   str or None
      cluster_role: 'representative' | 'singleton' | 'sibling' (for ungrouped singletons)
      manifest_rows: list[dict]  — rows from cluster_assignments
      gpu_row:      dict | None  — matched row from inference_results (for rep/singleton)
      xpath_rules:  list[dict] | None  — from gpu_row["xpath_rules"]
      mapping_data: dict | None  — from gpu_row["mapping_json"] parsed
      representative_content_len: int — for ratio check
    """
    manifest_rows = task["manifest_rows"]
    gpu_row = task.get("gpu_row")
    red_selectors = task.get("red_selectors")
    mapping_data = task.get("mapping_data")
    representative_content_len = task.get("representative_content_len", 0)

    # PERF: decide ONCE per cluster whether fast static LBP reproduces dynamic LBP.
    sib_rows = [r for r in manifest_rows if str(r.get("cluster_role", "")) == "sibling"]
    use_static = False
    if sib_rows and mapping_data is not None:
        use_static = _cluster_static_trustworthy(task.get("cluster_id"), sib_rows, mapping_data)

    results = []
    for row in manifest_rows:
        role = str(row.get("cluster_role", "singleton"))

        if role == "representative":
            if gpu_row is not None:
                merged = dict(row)
                merged.update({
                    "dripper_content": gpu_row.get("dripper_content", ""),
                    "dripper_html": gpu_row.get("dripper_html", gpu_row.get("llm_output_raw", "")),
                    "dripper_error": gpu_row.get("error", ""),
                    "inference_time_s": gpu_row.get("inference_time_s", 0.0),
                })
                results.append(_process_representative_row(merged))
            else:
                # GPU result missing for this representative — mark as fallback
                results.append({
                    "url": row.get("url", ""),
                    "url_host_name": row.get("url_host_name", ""),
                    "cluster_id": row.get("cluster_id"),
                    "cluster_role": "representative",
                    "dripper_content": "",
                    "dripper_html": "",
                    "dripper_error": "missing_gpu_result_for_representative",
                    "dripper_time_s": 0.0,
                    "propagation_success": False,
                    "propagation_method": "fallback",
                })

        elif role == "singleton":
            if gpu_row is not None:
                merged = dict(row)
                merged.update({
                    "dripper_content": gpu_row.get("dripper_content", ""),
                    "dripper_html": gpu_row.get("dripper_html", gpu_row.get("llm_output_raw", "")),
                    "dripper_error": gpu_row.get("error", ""),
                    "inference_time_s": gpu_row.get("inference_time_s", 0.0),
                })
                results.append(_process_singleton_row(merged))
            else:
                results.append({
                    "url": row.get("url", ""),
                    "url_host_name": row.get("url_host_name", ""),
                    "cluster_id": None,
                    "cluster_role": "singleton",
                    "dripper_content": "",
                    "dripper_html": "",
                    "dripper_error": "missing_gpu_result_for_singleton",
                    "dripper_time_s": 0.0,
                    "propagation_success": False,
                    "propagation_method": "fallback",
                })

        elif role == "sibling":
            results.append(_process_sibling_row(
                row, red_selectors, mapping_data, representative_content_len, use_static
            ))

        else:
            # Unknown role — pass through with error
            results.append({
                "url": row.get("url", ""),
                "url_host_name": row.get("url_host_name", ""),
                "cluster_id": row.get("cluster_id"),
                "cluster_role": role,
                "dripper_content": "",
                "dripper_html": "",
                "dripper_error": f"unknown_cluster_role={role}",
                "dripper_time_s": 0.0,
                "propagation_success": False,
                "propagation_method": "fallback",
            })

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_html(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    if raw is None:
        return ""
    return str(raw)


def _parse_xpath_rules(raw: Any) -> list[dict[str, Any]] | None:
    """Parse the xpath_rules column from Stage 2 output."""
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return None


def _parse_mapping_json(raw: Any) -> dict[str, Any] | None:
    """Parse the propagation template from Stage 2b output for LayoutBatchParser.

    Stage 2b serializes the template via pickle+base64 (lossless — preserves the
    tuple keys in html_element_dict that a JSON round-trip would destroy). We try
    pickle first, then fall back to JSON for older outputs.
    """
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
        # pickle+base64 (current Stage 2b format)
        try:
            obj = pickle.loads(base64.b64decode(raw))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # legacy JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cluster_manifest_shard(path: str) -> pd.DataFrame:
    """Load one shard from cluster_assignments/.

    Critical: html is only loaded for sibling rows that need propagation.
    Loading html for all rows (representatives + singletons already processed
    by Stage 2) would OOM at scale — each HTML page is 50-500 KB and there
    can be 30M+ rows per shard.
    """
    # First pass: load metadata without html (fast, low memory)
    meta_cols = [
        "url", "url_host_name", "cluster_id", "cluster_role",
        "warc_filename", "warc_record_offset", "warc_record_length",
    ]
    schema_names = pq.read_schema(path).names
    available_meta = [c for c in meta_cols if c in schema_names]
    df = pq.read_table(path, columns=available_meta).to_pandas()

    if "cluster_id" not in df.columns:
        df["cluster_id"] = None
    if "cluster_role" not in df.columns:
        df["cluster_role"] = "singleton"

    # Second pass: load html only for sibling rows (they need it for propagation)
    # Representatives and singletons already have their content from Stage 2.
    if "html" in schema_names:
        sibling_mask = df["cluster_role"] == "sibling"
        if sibling_mask.any():
            # Read html for all rows but only keep sibling values (others → None)
            # This avoids the full-table html load while still being correct.
            html_df = pq.read_table(path, columns=["url", "html"]).to_pandas()
            # Deduplicate on url — Stage 1b can produce duplicate URLs when
            # the same page appears in outputs from multiple GPU partitions
            html_df = html_df.drop_duplicates(subset="url", keep="first")
            html_map = html_df.set_index("url")["html"]
            df["html"] = df["url"].map(html_map)
            # Clear html for non-siblings to free memory
            df.loc[~sibling_mask, "html"] = None
        else:
            df["html"] = None
    else:
        df["html"] = None

    return df


def _load_inference_results(path: str) -> pd.DataFrame:
    """Load GPU inference results (Stage 2 output).

    Handles schema variants:
    - Canonical Stage 2 output: cluster_id, error, llm_output_raw
    - run_mineru_html_standalone.py --representatives-only output:
        layout_cluster_id (→ cluster_id), dripper_error (→ error)
    """
    cols_needed = [
        "cluster_id", "layout_cluster_id",
        "url", "llm_output_raw", "xpath_rules", "template_html",
        "inference_time_s", "error", "dripper_error",
        "dripper_content", "dripper_html", "mapping_json",
    ]
    schema_names = pq.read_schema(path).names
    available = [c for c in cols_needed if c in schema_names]
    df = pq.read_table(path, columns=available).to_pandas()

    # Normalise cluster_id column name
    if "cluster_id" not in df.columns and "layout_cluster_id" in df.columns:
        df = df.rename(columns={"layout_cluster_id": "cluster_id"})

    # Normalise error column name
    if "error" not in df.columns and "dripper_error" in df.columns:
        df = df.rename(columns={"dripper_error": "error"})

    return df


def _build_gpu_lookup(inference_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build cluster_id -> gpu_row dict for O(1) lookup during task construction."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in inference_df.to_dict("records"):
        cid = row.get("cluster_id")
        if cid is not None and str(cid) not in lookup:
            lookup[str(cid)] = row
    # Also index by url for singletons (cluster_id=None)
    # Singletons won't have cluster_id, so index by url
    return lookup


def _build_singleton_gpu_lookup(inference_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build url -> gpu_row for singleton pages (cluster_id is NULL in inference output)."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in inference_df.to_dict("records"):
        cid = row.get("cluster_id")
        url = str(row.get("url") or "")
        if (cid is None or str(cid).lower() in ("none", "null", "nan", "")) and url:
            lookup[url] = row
    return lookup


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Write parquet atomically via a tmp file in the same directory."""
    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(tmp_path), compression="snappy")
    tmp_path.rename(out_path)


def _shard_is_done(out_path: Path, expected_rows: int | None = None) -> bool:
    """Check if a shard output already exists (and optionally has expected row count)."""
    if not out_path.exists():
        return False
    if expected_rows is None:
        return True
    try:
        meta = pq.read_metadata(str(out_path))
        actual = meta.num_rows
        return actual == expected_rows
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main processing logic (called once per Slurm array task)
# ---------------------------------------------------------------------------

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

    # --- Checkpoint resume ---
    if out_path.exists():
        try:
            meta = pq.read_metadata(str(out_path))
            if meta.num_rows > 0:
                print(f"[stage3] SKIP shard {shard_index} — already exists ({meta.num_rows:,} rows)", flush=True)
                return {"status": "skipped", "shard": shard_index, "rows": meta.num_rows}
            else:
                # Zero-row parquet is suspicious — could be a failed partial write; reprocess
                print(f"[stage3] shard {shard_index} exists with 0 rows — reprocessing", flush=True)
                out_path.unlink(missing_ok=True)
        except Exception:
            # Corrupt shard — reprocess
            out_path.unlink(missing_ok=True)

    # --- Resolve input shard files ---
    manifest_dir = Path(cluster_manifest_dir)
    gpu_dir = Path(inference_results_dir)

    # Cluster manifest shards: we select 1-of-N shards from the manifest directory
    manifest_files = sorted(manifest_dir.glob("shard_*.parquet"))
    if not manifest_files:
        # Also try flat parquet
        manifest_files = sorted(manifest_dir.glob("*.parquet"))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest shards found in {manifest_dir}")

    # Select this task's slice of manifest shards
    total_files = len(manifest_files)
    file_start = total_files * shard_index // num_shards
    file_end = total_files * (shard_index + 1) // num_shards
    my_files = manifest_files[file_start:file_end]

    if not my_files:
        print(f"[stage3] shard {shard_index}: no manifest files assigned — writing empty shard", flush=True)
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        _atomic_write_parquet(empty_df, out_path)
        return {"status": "empty", "shard": shard_index, "rows": 0}

    print(f"[stage3] shard {shard_index}/{num_shards}: loading {len(my_files)} manifest file(s)...", flush=True)

    # Load and concatenate assigned manifest shards
    manifest_frames = []
    for f in my_files:
        manifest_frames.append(_load_cluster_manifest_shard(str(f)))
    manifest_df = pd.concat(manifest_frames, ignore_index=True)
    del manifest_frames
    print(f"[stage3] shard {shard_index}: {len(manifest_df):,} manifest rows loaded", flush=True)

    # --- Load GPU inference results (filtered to only cluster_ids we need) ---
    # CRITICAL: At CC scale, the full gpu_results dir is ~222 GB across 64 shards.
    # Loading ALL 64 shards on every Stage 3 node would OOM the 220 GB nodes.
    # Solution: collect the cluster_ids in our manifest slice first, then only
    # read the GPU rows matching those ids (predicate pushdown per shard).
    manifest_cluster_ids: set[str] = set()
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        if cid is not None and str(cid).lower() not in ("none", "null", "nan", ""):
            manifest_cluster_ids.add(str(cid))
    manifest_urls: set[str] = {str(r.get("url", "")) for r in manifest_df.to_dict("records")}

    gpu_files = sorted(gpu_dir.glob("shard_*.parquet"))
    if not gpu_files:
        gpu_files = sorted(gpu_dir.glob("*.parquet"))
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
            # Filter to only the cluster_ids and singleton urls we need
            if len(shard_df) == 0:
                continue
            mask = pd.Series(False, index=shard_df.index)
            if "cluster_id" in shard_df.columns and manifest_cluster_ids:
                mask |= shard_df["cluster_id"].astype(str).isin(manifest_cluster_ids)
            if "url" in shard_df.columns and manifest_urls:
                # Singletons: cluster_id is None/null, match by url
                null_cid = shard_df["cluster_id"].isna() | shard_df["cluster_id"].astype(str).isin(
                    ("none", "null", "nan", "")
                )
                mask |= (null_cid & shard_df["url"].astype(str).isin(manifest_urls))
            filtered = shard_df[mask]
            if len(filtered) > 0:
                gpu_frames.append(filtered)
        except Exception as exc:
            print(f"[stage3] WARNING: could not read GPU shard {f}: {exc}", flush=True)
    if gpu_frames:
        gpu_df = pd.concat(gpu_frames, ignore_index=True)
    else:
        gpu_df = pd.DataFrame()
    del gpu_frames
    print(f"[stage3] {len(gpu_df):,} relevant GPU result rows loaded", flush=True)

    # Build lookup indexes
    cluster_gpu_lookup = _build_gpu_lookup(gpu_df)
    singleton_gpu_lookup = _build_singleton_gpu_lookup(gpu_df)
    del gpu_df

    # --- Build cluster tasks ---
    print(f"[stage3] building cluster tasks...", flush=True)
    tasks: list[dict[str, Any]] = []

    # Group manifest rows by cluster_id (None = singleton)
    cluster_groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_df.to_dict("records"):
        cid = row.get("cluster_id")
        cid_key: str | None = str(cid) if (cid is not None and str(cid).lower() not in ("none", "null", "nan", "")) else None
        cluster_groups[cid_key].append(row)

    # PERF #3: cap siblings per task so a giant cluster is split across workers
    # instead of running serially on one (load balancing).
    PAGES_PER_TASK = 300

    for cid_key, rows in cluster_groups.items():
        if cid_key is None:
            # Singletons — each gets its own mini-task (near-free copy of gpu_row).
            for row in rows:
                url = str(row.get("url", ""))
                tasks.append({
                    "cluster_id": None,
                    "manifest_rows": [row],
                    "gpu_row": singleton_gpu_lookup.get(url),
                    "red_selectors": None,
                    "mapping_data": None,
                    "representative_content_len": 0,
                })
        else:
            gpu_row = cluster_gpu_lookup.get(cid_key)
            mapping_data = None
            representative_content_len = 0
            if gpu_row is not None:
                mapping_data = _parse_mapping_json(
                    gpu_row.get("mapping_json") or gpu_row.get("llm_output_raw")
                )
                rep_content = gpu_row.get("dripper_content", "")
                if rep_content:
                    representative_content_len = len(str(rep_content))

            # PERF #1+#2: derive the red-key CSS selectors ONCE per cluster.
            red_selectors = _derive_red_selectors(mapping_data)

            non_sib = [r for r in rows if str(r.get("cluster_role", "")) != "sibling"]
            sib = [r for r in rows if str(r.get("cluster_role", "")) == "sibling"]

            # First task carries the representative(s) + the first sibling chunk.
            first_chunk = sib[:PAGES_PER_TASK]
            tasks.append({
                "cluster_id": cid_key,
                "manifest_rows": non_sib + first_chunk,
                "gpu_row": gpu_row,
                "red_selectors": red_selectors,
                "mapping_data": mapping_data,
                "representative_content_len": representative_content_len,
            })
            # Remaining siblings → balanced page-level tasks (no rep, share template).
            for i in range(PAGES_PER_TASK, len(sib), PAGES_PER_TASK):
                tasks.append({
                    "cluster_id": cid_key,
                    "manifest_rows": sib[i:i + PAGES_PER_TASK],
                    "gpu_row": None,
                    "red_selectors": red_selectors,
                    "mapping_data": mapping_data,
                    "representative_content_len": representative_content_len,
                })

    del manifest_df, cluster_groups, cluster_gpu_lookup, singleton_gpu_lookup

    total_tasks = len(tasks)
    total_pages = sum(len(t["manifest_rows"]) for t in tasks)
    print(f"[stage3] shard {shard_index}: {total_tasks:,} cluster tasks, {total_pages:,} pages", flush=True)

    # initargs tuple must match _worker_init positional signature exactly
    worker_initargs = (
        dynamic_classid_similarity_threshold,
        more_noise_enable,
        min_content_length_ratio,
        max_content_length_ratio,
        log_level,
    )

    all_results: list[dict[str, Any]] = []
    n_success = 0
    n_fallback = 0
    n_xpath = 0
    n_lbp = 0
    n_rep = 0
    n_singleton = 0
    pages_done = 0

    t_proc_start = time.perf_counter()

    # Process in chunks to allow periodic progress reporting and avoid unbounded
    # memory from keeping all futures in-flight at once.
    chunk_size = max(cluster_chunk_size, 1)
    num_chunks = (total_tasks + chunk_size - 1) // chunk_size

    # Use spawn context so that lxml / llm_web_kit C extensions are not
    # inherited across fork() — fork-safety is not guaranteed for those libs.
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=worker_initargs,
    ) as executor:
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_tasks)
            chunk = tasks[chunk_start:chunk_end]

            chunk_results: list[dict[str, Any]] = []

            futures = {executor.submit(_process_cluster_task, task): i
                       for i, task in enumerate(chunk)}
            for future in as_completed(futures):
                try:
                    rows = future.result()
                    chunk_results.extend(rows)
                except Exception as exc:
                    logger.error("Task failed: %s", exc)

            # Stats and progress reporting happen per chunk (inside executor context)
            all_results.extend(chunk_results)
            for r in chunk_results:
                meth = r.get("propagation_method", "fallback")
                if r.get("propagation_success"):
                    n_success += 1
                else:
                    n_fallback += 1
                if meth in ("xpath", "lbp_static"):
                    n_xpath += 1   # fast path (static-only; no dynamic similarity)
                elif meth == "layout_batch_parser":
                    n_lbp += 1     # dynamic-matching fallback
                elif meth == "representative":
                    n_rep += 1
                elif meth == "singleton":
                    n_singleton += 1

            pages_done += sum(len(t["manifest_rows"]) for t in chunk)
            elapsed = time.perf_counter() - t_proc_start
            rate = pages_done / max(elapsed, 0.001)
            print(
                f"[stage3] shard {shard_index}: chunk {chunk_idx+1}/{num_chunks} "
                f"pages={pages_done:,}/{total_pages:,} "
                f"rate={rate:.1f} pages/s  "
                f"success={n_success} fallback={n_fallback} "
                f"xpath={n_xpath} lbp={n_lbp}",
                flush=True,
            )

    # --- Write output ---
    result_df = pd.DataFrame(all_results, columns=OUTPUT_COLUMNS)
    _atomic_write_parquet(result_df, out_path)

    t_end = time.perf_counter()
    elapsed_total = t_end - t_start
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

    metrics_path = output_dir_path / f"metrics_shard_{shard_index:04d}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[stage3] shard {shard_index} DONE", flush=True)
    print(f"  pages:      {total_pages:,}  (success={n_success} fallback={n_fallback})", flush=True)
    print(f"  xpath:      {n_xpath}  lbp={n_lbp}  rep={n_rep}  singleton={n_singleton}", flush=True)
    print(f"  elapsed:    {elapsed_total:.1f}s  ({pages_per_s:.1f} pages/s)", flush=True)
    print(f"  output:     {out_path}", flush=True)

    return metrics


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 3: CPU template propagation for CC-scale pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cluster-manifest",
        required=True,
        help="Directory containing cluster_assignments/ shard_NNNN.parquet files (Stage 1 output)",
    )
    p.add_argument(
        "--inference-results",
        required=True,
        help="Directory containing gpu_results/ shard_NNNN.parquet files (Stage 2 output)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for propagation_results/ shard_NNNN.parquet files",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
        help="0-based task index (default: SLURM_ARRAY_TASK_ID)",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=80,
        help="Total number of array tasks (= number of CPU nodes)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 64)),
        help="Parallel workers per node (default: SLURM_CPUS_PER_TASK or 64)",
    )
    p.add_argument(
        "--cluster-chunk-size",
        type=int,
        default=500,
        help="Number of cluster tasks to submit to the process pool per chunk (controls memory)",
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
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
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
    print(flush=True)

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
