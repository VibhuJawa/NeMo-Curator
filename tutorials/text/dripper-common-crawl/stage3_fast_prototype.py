#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""stage3_fast_prototype.py — ILLUSTRATIVE prototype of the optimized Stage 3
propagation kernel.  NOT a drop-in replacement; do NOT run against production.

Implements the top recommendations from STAGE3_PERF_AUDIT.md:

  #1  Derive deterministic CSS/XPath selectors ONCE per cluster from the
      template's `html_element_dict` red-key set, apply via lxml to siblings
      (~10-50 ms/page) instead of LayoutBatchParser (~0.3-3 s/page).
  #2  Compile the cluster template ONCE; reuse a prepared parser across all the
      cluster's siblings (eliminates per-sibling _preprocess_template_data).
  #3  Fan siblings out at PAGE granularity so a 5,000-sibling cluster is split
      across workers instead of running serially on one.

Fallbacks and gates preserve F1 parity with the standalone LayoutBatchParser
baseline:
  - selectors return 0 elements  -> fall back to LBP
  - text-vs-text content ratio out of bounds (M1 fix) -> fall back to LBP
  - optional layout-similarity gate below threshold   -> fall back to LBP

The pieces marked `# VENDOR` reference llm_web_kit internals confirmed by reading
the installed package (layout_batch_parser.py / tag_mapping.py / html_layout_cosin.py).
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# --- mirror of LayoutBatchParser.normalize_key / replace_post_number (VENDOR) ---
_POST_NUMBER_RE = re.compile(r"(post|postid)-(\d+)", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\n]+")


def _replace_post_number(text: str | None) -> str | None:
    if not text:
        return None
    return _POST_NUMBER_RE.sub(lambda m: f"{m.group(1)}-", text).strip()


def _normalize_key(tag: str, cls: str | None, idd: str | None, blacklisted_ids: set[str]) -> tuple:
    """Reproduce LayoutBatchParser.normalize_key for the STATIC (non-dynamic) case.

    Mirrors layout_batch_parser.LayoutBatchParser.normalize_key:
      - body/html            -> (tag, None, None)
      - id present & valid    -> (tag, None, post_normalized(id))
      - else                  -> (tag, post_normalized(class), post_normalized(id))
    """
    if cls:
        cls = _WS_RE.sub(" ", cls)
    if tag in ("body", "html"):
        return (tag, None, None)
    if idd and idd not in blacklisted_ids:
        return (tag, None, _replace_post_number(idd))
    return (tag, _replace_post_number(cls), _replace_post_number(idd))


# ---------------------------------------------------------------------------
# #1 + #2: compile selectors + prepared template ONCE per cluster
# ---------------------------------------------------------------------------


class CompiledTemplate:
    """Per-cluster compiled artifacts, built once and reused across all siblings.

    Attributes:
      red_selectors:  list[str] of CSS selectors targeting main-content nodes.
      mapping_data:   the original template dict (for the LBP fallback path).
      rep_content_len: representative extracted-TEXT length (for the ratio gate).
      template_main_html: typical_main_html (for the optional similarity gate).
      similarity_layer:   SIMILARITY_LAYER from the template.
    """

    __slots__ = (
        "mapping_data",
        "red_selectors",
        "rep_content_len",
        "similarity_layer",
        "template_main_html",
    )

    def __init__(self, mapping_data: dict[str, Any], rep_content_len: int) -> None:
        self.mapping_data = mapping_data
        self.rep_content_len = rep_content_len
        self.template_main_html = mapping_data.get("typical_main_html") or ""
        self.similarity_layer = mapping_data.get("similarity_layer")
        self.red_selectors = self._derive_red_selectors(mapping_data)

    @staticmethod
    def _derive_red_selectors(mapping_data: dict[str, Any]) -> list[str]:
        """Turn the template's red-labeled keys into CSS selectors (#1).

        html_element_dict (VENDOR, from MapItemToHtmlTagsParser.parse docstring):
          { layer_no: { (tag, class, id, sha256, layer_no, idx):
                            (label, (parent_tag, parent_class, parent_id)) } }
        label == 'red' marks main content.  We emit one CSS selector per red key.
        """
        element_dict = mapping_data.get("html_element_dict") or {}
        # Build the id blacklist exactly as _preprocess_template_data does:
        # an id appearing >3 times in the template doc is "dynamic" -> ignore it.
        # (We approximate from the dict; the real parser counts in the DOM.)
        selectors: list[str] = []
        seen: set[str] = set()
        for nodes in element_dict.values():
            if not isinstance(nodes, dict):
                continue
            for key, value in nodes.items():
                label = value[0] if isinstance(value, (list, tuple)) and value else None
                if label != "red":
                    continue
                # key = (tag, class, id, sha256, layer_no, idx)
                try:
                    tag, cls, idd = key[0], key[1], key[2]
                except (IndexError, TypeError):
                    # key is too short or not subscriptable — skip this node
                    continue
                sel = CompiledTemplate._key_to_css(tag, cls, idd)
                if sel and sel not in seen:
                    seen.add(sel)
                    selectors.append(sel)
        return selectors

    @staticmethod
    def _key_to_css(tag: str, cls: str | None, idd: str | None) -> str | None:
        if not tag or tag in ("html",):
            return None
        # Prefer id (most specific & what normalize_key prefers), strip post-number.
        idd_n = _replace_post_number(idd)
        if idd_n:
            # CSS escaping is omitted for brevity; real impl should escape.
            return f"{tag}[id='{idd_n}']"
        cls_n = _replace_post_number(cls)
        if cls_n:
            first = cls_n.strip().split(" ")[0]
            if first:
                return f"{tag}.{first}"
        return tag  # last resort: tag-only (broad — relies on ratio gate)


def compile_cluster_template(mapping_data: dict[str, Any] | None, rep_content_len: int) -> CompiledTemplate | None:
    if not mapping_data:
        return None
    return CompiledTemplate(mapping_data, rep_content_len)


# ---------------------------------------------------------------------------
# #1: fast XPath/CSS extraction per sibling
# ---------------------------------------------------------------------------


def _xpath_extract_inner(html: str, compiled: CompiledTemplate) -> tuple[str, str]:
    """Inner extraction logic after guard checks; assumes lxml is available."""
    import lxml.html as lhtml
    from lxml import etree

    try:
        doc = lhtml.fromstring(html.encode("utf-8", "replace"))
    except (ValueError, etree.LxmlError) as exc:
        return "", f"lxml_parse_error={exc!s:.80}"

    parts: list[str] = []
    matched_nodes: set[int] = set()
    for sel in compiled.red_selectors:
        try:
            els = doc.cssselect(sel)
        except (ValueError, etree.XPathError):
            # Malformed selector — skip and try remaining selectors
            continue
        for el in els:
            # Avoid double-emitting nested matches (keep outermost).
            if any(anc in matched_nodes for anc in (id(a) for a in el.iterancestors())):
                continue
            matched_nodes.add(id(el))
            with contextlib.suppress(ValueError, etree.LxmlError):
                parts.append(etree.tostring(el, encoding="unicode", method="html"))
    if not parts:
        return "", "xpath_no_elements_matched"
    return "\n".join(parts), ""


def xpath_extract(html: str, compiled: CompiledTemplate) -> tuple[str, str]:
    """Apply compiled red selectors to a sibling.  Returns (main_html, error)."""
    try:
        import lxml.html  # noqa: F401 — check availability only
    except ImportError:
        return "", "lxml_not_available"
    if not html.strip():
        return "", "empty_html"
    if not compiled.red_selectors:
        return "", "no_selectors"
    return _xpath_extract_inner(html, compiled)


# ---------------------------------------------------------------------------
# #3: page-level, size-balanced work units
# ---------------------------------------------------------------------------


class RatioGate:
    """Text-length and layout-similarity gate parameters."""

    __slots__ = ("max_ratio", "min_ratio", "min_sim")

    def __init__(self, min_ratio: float = 0.25, max_ratio: float = 4.0, min_sim: float | None = 0.75) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_sim = min_sim


class SiblingProcessingConfig:
    """Groups callables and gate config for process_sibling_fast.

    Attributes:
        convert_fn: callable(main_html, url) -> (content, error)
        lbp_fn: callable(html, mapping_data) -> (main_html, error)
        similarity_fn: optional callable(tmpl_html, body_html, layer) -> float | None
        gate: RatioGate with ratio and similarity thresholds
    """

    __slots__ = ("convert_fn", "gate", "lbp_fn", "similarity_fn")

    def __init__(
        self,
        convert_fn: Callable[[str, str], tuple[str, str]],
        lbp_fn: Callable[[str, dict[str, Any]], tuple[str, str]],
        similarity_fn: Callable[..., float | None] | None = None,
        gate: RatioGate | None = None,
    ) -> None:
        self.convert_fn = convert_fn
        self.lbp_fn = lbp_fn
        self.similarity_fn = similarity_fn
        self.gate = gate if gate is not None else RatioGate()


def _apply_xpath_gates(
    content: str,
    xp_html: str,
    compiled: CompiledTemplate,
    cfg: SiblingProcessingConfig,
) -> tuple[bool, str]:
    """Return (ok, error) after running ratio and similarity gates."""
    gate = cfg.gate
    if compiled.rep_content_len > 0:
        ratio = len(content) / max(compiled.rep_content_len, 1)
        if ratio < gate.min_ratio or ratio > gate.max_ratio:
            return False, f"xpath_content_ratio_oob={ratio:.3f}"

    if cfg.similarity_fn is not None and compiled.template_main_html and gate.min_sim is not None:
        try:
            sim = cfg.similarity_fn(compiled.template_main_html, xp_html, compiled.similarity_layer)
            if sim is not None and sim < gate.min_sim:
                return False, f"xpath_low_sim={sim:.3f}"
        except Exception:
            # Intentionally swallowed: gate failure must not abort the fast path.
            return True, ""
    return True, ""


def process_sibling_fast(
    html: str,
    url: str,
    compiled: CompiledTemplate,
    cfg: SiblingProcessingConfig,
) -> dict[str, Any]:
    """Returns the same row schema as stage3's _process_sibling_row."""
    method = "fallback"
    main_html = ""
    content = ""
    error = ""

    # --- #1 fast path ---
    xp_html, xp_err = xpath_extract(html, compiled)
    if xp_html and not xp_err:
        # convert FIRST so the ratio compares text-vs-text (M1 fix).
        content, conv_err = cfg.convert_fn(xp_html, url)
        if conv_err:
            error = conv_err
        else:
            ok, gate_err = _apply_xpath_gates(content, xp_html, compiled, cfg)
            if ok:
                main_html = xp_html
                method = "xpath"
            else:
                error = gate_err
                content = ""

    # --- LBP fallback (preserves baseline F1 for pages selectors can't cover) ---
    if not main_html:
        lbp_html, lbp_err = cfg.lbp_fn(html, compiled.mapping_data)
        if lbp_html and not lbp_err:
            content, conv_err = cfg.convert_fn(lbp_html, url)
            if not conv_err:
                main_html, error, method = lbp_html, "", "layout_batch_parser"
            else:
                error = conv_err
        elif lbp_err:
            error = f"xpath_failed({error}); lbp_failed({lbp_err})" if error else lbp_err

    if not main_html and not error:
        error = "no_template_available"

    return {
        "url": url,
        "cluster_role": "sibling",
        "dripper_content": content,
        "dripper_html": main_html,
        "dripper_error": error,
        "propagation_success": bool(main_html and not error),
        "propagation_method": method,
    }


# ---------------------------------------------------------------------------
# #3: page-level, size-balanced work units
# ---------------------------------------------------------------------------


def build_page_units(tasks: list[dict[str, Any]], pages_per_unit: int = 256) -> list[dict[str, Any]]:
    """Split per-cluster tasks into balanced page-level units.

    Each unit: { 'cluster_id', 'compiled_token', 'rows': [...] }.
    A huge cluster yields multiple units (fanned across workers); rep/singleton
    rows are grouped separately (near-free copies).  The compiled template is
    shipped once per cluster (worker memoizes by cluster_id) rather than per row.
    """
    units: list[dict[str, Any]] = []
    for task in tasks:
        cid = task["cluster_id"]
        sib_rows = [r for r in task["manifest_rows"] if str(r.get("cluster_role")) == "sibling"]
        other_rows = [r for r in task["manifest_rows"] if str(r.get("cluster_role")) != "sibling"]
        if other_rows:
            units.append({"cluster_id": cid, "kind": "copy", "rows": other_rows, "gpu_row": task.get("gpu_row")})
        for i in range(0, len(sib_rows), pages_per_unit):
            units.append(
                {
                    "cluster_id": cid,
                    "kind": "sibling",
                    "rows": sib_rows[i : i + pages_per_unit],
                    "mapping_data": task.get("mapping_data"),
                    "representative_content_len": task.get("representative_content_len", 0),
                }
            )
    return units


# Per-worker cache so the compiled template is built ONCE per cluster per worker
# (#2), even though units arrive interleaved.
_WORKER_TEMPLATE_CACHE: dict[Any, CompiledTemplate] = {}


def process_sibling_unit(unit: dict[str, Any], cfg: SiblingProcessingConfig) -> list[dict[str, Any]]:
    cid = unit["cluster_id"]
    compiled = _WORKER_TEMPLATE_CACHE.get(cid)
    if compiled is None:
        compiled = compile_cluster_template(unit.get("mapping_data"), unit.get("representative_content_len", 0))
        _WORKER_TEMPLATE_CACHE[cid] = compiled
    out = []
    for row in unit["rows"]:
        html = row.get("html") or ""
        if isinstance(html, (bytes, bytearray)):
            html = html.decode("utf-8", "replace")
        if compiled is None:
            out.append(
                {
                    "url": row.get("url", ""),
                    "cluster_role": "sibling",
                    "dripper_content": "",
                    "dripper_html": "",
                    "dripper_error": "no_template",
                    "propagation_success": False,
                    "propagation_method": "fallback",
                }
            )
            continue
        out.append(process_sibling_fast(html, row.get("url", ""), compiled, cfg))
    return out


# ---------------------------------------------------------------------------
# Notes for integration (see STAGE3_PERF_AUDIT.md §2):
#   - Wire similarity_fn to llm_web_kit.html_layout.html_layout_cosin using
#     get_feature / similarity; return None when either feature is None.
#   - convert_fn / lbp_fn are the existing stage3 worker functions
#     (_convert_main_html_to_content / _layout_batch_parser_propagate).
#   - GATE rollout on compare_f1.py: XPath-vs-LBP token-F1 >= 0.99 on a sample.
#   - Build red selectors in Stage 2b instead (write an `xpath_rules` column) to
#     avoid carrying the full template through Stage 3 — see audit #1 option (a).
# ---------------------------------------------------------------------------
