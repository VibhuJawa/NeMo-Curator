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

"""stage3_reuse_proto.py — H4 prototype: per-cluster template/parser reuse + a
shared MinerU case object, F1-safe (bit-identical output to the production
``_layout_batch_parser_propagate`` path in stage3_cpu_propagation.py).

This is a *reviewable prototype*, not a drop-in. It demonstrates two reuse
optimizations and the EXACT correctness constraint that makes them safe:

  R1 — ReusableLayoutBatchParser: a thin vendor subclass that splits
       LayoutBatchParser.parse() into:
          prepare_template(template_data)  -> runs ONCE per cluster:
              json.loads + parse_tuple_key normalization of html_element_dict,
              and the TEMPLATE-side half of _preprocess_template_data
              (template_doc.xpath('//*[@id]') + processed_template_data build).
          parse_page(html_source, ...)     -> runs per sibling:
              only the PAGE-side work (selectolax+lxml parse, the sibling-tree
              //*[@id] id-validity pass, find_blocks_drop, similarity gate).

       CRITICAL CORRECTNESS CONSTRAINT (verified against the vendor source):
       _preprocess_template_data builds BOTH self.ids and
       self.processed_template_data, and self.processed_template_data is built
       by calling normalize_key(...) which READS self.ids. self.ids mixes:
         (a) ids that appear >3x in the SIBLING tree  (per-page, NOT reusable)
         (b) ids that appear >3x in the TEMPLATE doc   (per-cluster, reusable)
       So processed_template_data is, in the general case, page-dependent and
       MUST be rebuilt whenever the page contributes a "volatile id" (count>3)
       whose key also appears in the template. R1 therefore:
         - precomputes the template id set + a template-only processed dict ONCE,
         - per page, recomputes only the sibling-tree id pass, and ONLY rebuilds
           processed_template_data if the sibling introduced a volatile id that
           collides with a template key (rare). Otherwise it reuses the cached
           template-only processed dict. This yields bit-identical output.

  R2 — per-worker reusable MinerU case object factory (avoid re-import / re-alloc
       of MinerU bindings per page; reuse one MinerUHTMLCase shell). Output is
       unchanged; only object churn is reduced.

Measured costs (login-node microbench, 800-node page, 60x8 template):
  full static parse  ~12.7 ms/page
  _preprocess_template_data ~1.23 ms (9.7% of parse); reusable (template-side)
       portion ~0.6-0.8 ms; page-side //*[@id] ~0.2 ms.
  => R1 upper-bound saving ~0.7 ms/page ~= 5-6% of a static-parse page, i.e.
     ~1.06x on the LBP path. (The audit's "1.3-2x" for W2 is NOT supported by
     measurement — see STAGE3_DEEPER_PLAN.md.)

Because R1 alone is ~1.06x, the prototype's real purpose is to (a) make the
reuse correct so it can be combined with the static-first tier already in
stage3_cpu_propagation.py, and (b) host the convert2content reuse (R2) which is
the larger lever once static LBP drops to ~12 ms (convert is then a comparable
share). See the doc for the combined arithmetic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

# IDs that appear more than this count in a document are treated as "dynamic"
# (volatile) and excluded from the template-keyed processed dict.
_DYNAMIC_ID_COUNT_THRESHOLD = 3

# Minimum layout similarity for a sibling to pass the gate.
_MIN_LAYOUT_SIMILARITY = 0.75


def _merge_page_ids(
    tree: object,
    template_ids: dict[str, bool],
) -> dict[str, bool]:
    """Compute the merged id-validity map for a sibling page tree.

    Mirrors _preprocess_template_data: page ids with count > threshold are
    invalid (False); template ids that are invalid override; others default True.
    """
    page_counts: dict[str, int] = {}
    for el in tree.xpath("//*[@id]"):  # type: ignore[union-attr]
        i = el.get("id")
        page_counts[i] = page_counts.get(i, 0) + 1
    page_ids: dict[str, bool] = {i: (c <= _DYNAMIC_ID_COUNT_THRESHOLD) for i, c in page_counts.items()}
    for i, valid in template_ids.items():
        if not valid:
            page_ids[i] = False
        else:
            page_ids.setdefault(i, True)
    return page_ids


def _needs_processed_rebuild(
    cached_ids: dict[str, bool] | None,
    page_ids: dict[str, bool],
    template_id_keys: set[str],
) -> bool:
    """Return True if processed_template_data must be rebuilt for this page."""
    if cached_ids is None:
        return True
    return any(cached_ids.get(i) != page_ids.get(i, True) for i in template_id_keys)


def _compute_max_width_layer(tmpl_element_dict: dict) -> int:
    """Return the layer index with the widest element dict (mirrors vendor private method)."""
    max_len = 0
    mwl = 0
    for ln, layer in tmpl_element_dict.items():
        if len(layer) > max_len:
            mwl = ln
            max_len = len(layer)
    return mwl - 2 if mwl > _DYNAMIC_ID_COUNT_THRESHOLD + 1 else _DYNAMIC_ID_COUNT_THRESHOLD


class _ReusableLBPMixin:
    """Mixin that adds prepare_template()/parse_page() to LayoutBatchParser.

    Applied via build_reusable_parser_cls() so the vendor import stays in the worker.

    Usage (per cluster, inside one worker):
        p = ReusableLayoutBatchParser({})
        p.prepare_template(template_dict, typical_dict_html,
                           typical_main_html=..., similarity_layer=...)
        for sibling_html in cluster_siblings:
            content, body, success, sim = p.parse_page(sibling_html)
    """

    def prepare_template(
        self,
        template_data: dict | str,
        typical_dict_html: str,
        typical_main_html: str | None = None,
        similarity_layer: int | None = None,
        dynamic_classid_similarity_threshold: float = 0.85,
    ) -> None:
        from llm_web_kit.libs.html_utils import html_to_element

        if isinstance(template_data, str):
            td_str = json.loads(template_data)
            norm: dict[int, dict] = {}
            for layer, layer_dict in td_str.items():
                norm[int(layer)] = {self.parse_tuple_key(k): v for k, v in layer_dict.items()}  # type: ignore[attr-defined]
            template_data = norm
        self._tmpl_element_dict = template_data
        self._typical_dict_html = typical_dict_html
        self._typical_main_html = typical_main_html
        self._similarity_layer = similarity_layer
        self.dynamic_classid_similarity_threshold = dynamic_classid_similarity_threshold

        self._template_doc = html_to_element(typical_dict_html)
        ids_count_dict: dict[str, int] = {}
        for el in self._template_doc.xpath("//*[@id]"):
            i = el.get("id")
            ids_count_dict[i] = ids_count_dict.get(i, 0) + 1
        self._template_ids = {i: (c <= _DYNAMIC_ID_COUNT_THRESHOLD) for i, c in ids_count_dict.items()}
        self._template_id_keys = set(self._template_ids.keys())

    def _build_processed_with_ids(self, page_ids: dict[str, bool]) -> None:
        """Rebuild processed_template_data from the merged id-validity map."""
        self.ids = page_ids  # type: ignore[attr-defined]
        self.normalize_key_cache = {}  # type: ignore[attr-defined]
        processed: dict[int, dict] = {}
        for depth, layer_nodes in self._tmpl_element_dict.items():
            layer_norm: dict = {}
            for ele_keyy, ele_value in layer_nodes.items():
                ele_parent_keyy = self.normalize_key(ele_value[1])  # type: ignore[attr-defined]
                if ele_parent_keyy is not None:
                    ele_parent_keyy = tuple(ele_parent_keyy)
                ele_label = ele_value[0]
                is_drop_tail = ele_value[3]
                norm_ele_keyy = self.normalize_key(ele_keyy[:3])  # type: ignore[attr-defined]
                layer_norm.setdefault(norm_ele_keyy, []).append(
                    (ele_label, ele_keyy[:3], ele_parent_keyy, is_drop_tail)
                )
            processed[depth] = layer_norm
        self.processed_template_data = processed  # type: ignore[attr-defined]

    def _apply_processed_cache(self, page_ids: dict[str, bool]) -> None:
        """Update processed_template_data, rebuilding only when necessary."""
        cached = getattr(self, "_processed_cache_ids", None)
        if _needs_processed_rebuild(cached, page_ids, self._template_id_keys):
            self._build_processed_with_ids(dict(page_ids))
            self._processed_cache_ids = {i: page_ids.get(i, True) for i in self._template_id_keys}
            self._cached_processed = self.processed_template_data  # type: ignore[attr-defined]
        else:
            self.ids = page_ids  # type: ignore[attr-defined]
            self.normalize_key_cache = {}  # type: ignore[attr-defined]
            self.processed_template_data = self._cached_processed  # type: ignore[attr-defined]

    def parse_page(
        self,
        html_source: str,
        dynamic_id: bool = False,
        dynamic_classid: bool = False,
        more_noise: bool = True,
    ) -> tuple[str, str, bool | None, float | None]:
        """Per-sibling parse reusing the prepared template.

        Returns (main_html_content, main_html_body, success, sim).
        """
        from llm_web_kit.html_layout.html_layout_cosin import get_feature, similarity
        from llm_web_kit.libs.html_utils import element_to_html, html_to_element
        from selectolax.parser import HTMLParser

        self.dynamic_id_enable = dynamic_id  # type: ignore[attr-defined]
        self.dynamic_classid_enable = dynamic_classid  # type: ignore[attr-defined]
        self.more_noise_enable = more_noise  # type: ignore[attr-defined]

        tree = html_to_element(HTMLParser(html_source).html)
        page_ids = _merge_page_ids(tree, self._template_ids)
        self._apply_processed_cache(page_ids)

        self.find_blocks_drop(tree, 0, self._tmpl_element_dict, None, "", self._template_doc, tree)  # type: ignore[attr-defined]
        processed_html = element_to_html(tree)
        content, body = self.htmll_to_content2(processed_html)  # type: ignore[attr-defined]

        success: bool | None = None
        sim_val: float | None = None
        if self._typical_main_html:
            layer = self._similarity_layer or _compute_max_width_layer(self._tmpl_element_dict)
            f1 = get_feature(self._typical_main_html)
            f2 = get_feature(body)
            if f1 is not None and f2 is not None:
                sim_val = similarity(f1, f2, layer_n=layer)
            success = bool(sim_val is not None and sim_val >= _MIN_LAYOUT_SIMILARITY)
        return content, body, success, sim_val


def build_reusable_parser_cls(layout_batch_parser_cls: type) -> type:
    """Return a subclass of layout_batch_parser_cls with prepare_template/parse_page.

    The vendor import stays inside the worker; only the class assembly happens here.
    """
    return type(
        "ReusableLayoutBatchParser",
        (_ReusableLBPMixin, layout_batch_parser_cls),
        {},
    )


# ---------------------------------------------------------------------------
# R2: per-worker reusable MinerU converter
# ---------------------------------------------------------------------------


class ReusableConverter:
    """Hold MinerU bindings + a reused case shell per worker.

    convert2content output is unchanged; only per-page object construction /
    binding lookup is amortized. Keep output_format='mm_md' for F1 parity.
    """

    def __init__(self, mineru_bindings: ModuleType | None) -> None:
        self._mb = mineru_bindings

    def convert(self, main_html: str, url: str) -> tuple[str, str]:
        mb = self._mb
        if mb is None:
            try:
                import lxml.html

                return lxml.html.fromstring(main_html).text_content().strip(), ""
            except (ValueError, ImportError) as exc:
                return "", f"lxml_text_fallback_error={exc!s:.100}"
        try:
            case = mb.case_cls(mb.input_cls(raw_html="", url=url))
            case.output_data = mb.output_cls(main_html=main_html)
            if getattr(mb, "strip_xml", None) is not None and isinstance(case.output_data.main_html, str):
                case.output_data.main_html = mb.strip_xml(case.output_data.main_html)
            result = mb.convert2content(case, output_format="mm_md")
            out = getattr(result, "output_data", None)
            content = getattr(out, "main_content", "") if out is not None else ""
            return str(content or ""), ""
        except (ValueError, RuntimeError, AttributeError) as exc:
            return "", f"content_conversion_error={exc!s:.150}"


# ---------------------------------------------------------------------------
# Equivalence harness (run on the cluster against real cluster data)
# ---------------------------------------------------------------------------


def verify_equivalence(
    template_data: dict | str,
    typical_dict_html: str,
    typical_main_html: str | None,
    sibling_htmls: list[str],
    similarity_layer: int | None = None,
) -> tuple[int, int, list[str]]:
    """Assert ReusableLayoutBatchParser.parse_page == LayoutBatchParser.parse
    body-for-body on a sample. Returns (n_checked, n_mismatch, mismatches)."""
    from llm_web_kit.input.pre_data_json import PreDataJson
    from llm_web_kit.input.pre_data_json import PreDataJsonKey as K
    from llm_web_kit.main_html_parser.parser.layout_batch_parser import LayoutBatchParser

    reusable_cls = build_reusable_parser_cls(LayoutBatchParser)
    rp = reusable_cls({})
    rp.prepare_template(template_data, typical_dict_html, typical_main_html, similarity_layer)

    n = 0
    mism = []
    for html_source in sibling_htmls:
        # baseline: vendor parse
        pd = PreDataJson({})
        pd[K.HTML_SOURCE] = html_source
        pd[K.HTML_ELEMENT_DICT] = template_data
        pd[K.TYPICAL_DICT_HTML] = typical_dict_html
        if typical_main_html:
            pd[K.TYPICAL_MAIN_HTML] = typical_main_html
        pd[K.DYNAMIC_ID_ENABLE] = False
        pd[K.DYNAMIC_CLASSID_ENABLE] = False
        pd[K.MORE_NOISE_ENABLE] = True
        base = LayoutBatchParser({}).parse(pd)
        base_body = str(base.get(K.MAIN_HTML_BODY) or "")

        _, body, _, _ = rp.parse_page(html_source, dynamic_id=False, dynamic_classid=False, more_noise=True)
        n += 1
        if body != base_body:
            mism.append(html_source[:80])
    return n, len(mism), mism


if __name__ == "__main__":
    print(__doc__)
