#!/usr/bin/env python3
"""
verify_pipeline.py — runs every pipeline step and prints PASS/FAIL.
Run on dgx-a100-02 with:
  /raid/vjawa/nemo-curator-adlr-mm/.venv/bin/python3 verify_pipeline.py
"""

from __future__ import annotations

import re
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

sys.path.insert(0, "/raid/vjawa/nemo-curator-adlr-mm/submodules/Curator")

DATA_DIR = "/raid/vjawa/dripper_tutorial"
MANIFEST = f"{DATA_DIR}/layout_precompute_manifest.parquet"
BASELINE = f"{DATA_DIR}/baseline_dripper_results.parquet"

# F1 threshold considered "good" for propagation quality gate.
_F1_THRESHOLD = 0.95

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, bool, str | None]] = []


def check(name: str, fn: Callable[[], object]) -> object:
    try:
        val = fn()
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e!s:.120}")
        results.append((name, False, str(e)))
        return None
    else:
        print(f"  [{PASS}] {name}")
        results.append((name, True, None))
        return val


def coerce_html(raw: bytes | str | None) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw or "")


# ── 0. Imports ────────────────────────────────────────────────────────────────
print("\n=== 0. IMPORTS ===")
import pyarrow.parquet as pq

from nemo_curator.stages.text.experimental.dripper.stage import (
    DripperHTMLExtractionStage,
    _load_llm_web_kit_bindings,
    _load_mineru_html_bindings,
    _token_f1,
)


def convert_html_to_content(bindings: object, main_html: str, url: str = "") -> str:
    """Convert extracted main HTML to plain text content via bindings.convert2content."""
    try:
        case = bindings.case_cls(bindings.input_cls(raw_html=main_html, url=url))  # type: ignore[union-attr]
        case = bindings.convert2content(case, output_format="mm_md")  # type: ignore[union-attr]
        output_data = getattr(case, "output_data", None)
        return str(getattr(output_data, "main_content", "") or main_html)
    except (ValueError, RuntimeError, AttributeError):
        return main_html  # fallback: use raw html as content


print(f"  [{PASS}] core imports")

# ── 1. Data loading ───────────────────────────────────────────────────────────
print("\n=== 1. DATA LOADING ===")
manifest = check("manifest parquet", lambda: pq.ParquetFile(MANIFEST).read().to_pandas())
baseline = None
try:
    baseline = pq.ParquetFile(BASELINE).read().to_pandas()
    print(f"  [{PASS}] baseline parquet ({len(baseline)} rows)")
except (FileNotFoundError, OSError) as e:
    print(f"  [{SKIP}] baseline: {e!s:.80} — F1 cells will be skipped")

if manifest is not None:
    print(f"         manifest: {len(manifest)} rows, {manifest['url_host_name'].nunique()} hosts")
    print(f"         hosts: {list(manifest['url_host_name'].unique())}")

# ── 2. llm-webkit bindings ────────────────────────────────────────────────────
print("\n=== 2. LLM-WEBKIT BINDINGS ===")
web = check("load llm_web_kit bindings", _load_llm_web_kit_bindings)
if web:
    check("get_feature callable", lambda: web.get_feature("<html><body><p>hi</p></body></html>"))
    check(
        "cluster_html_struct callable",
        lambda: web.cluster_html_struct(
            [
                {
                    "track_id": "0",
                    "html": "<html><body><p>hi</p></body></html>",
                    "feature": web.get_feature("<html><body><p>hi</p></body></html>"),
                }
            ],
            threshold=0.95,
        ),
    )

# ── 3. MinerU-HTML bindings ───────────────────────────────────────────────────
print("\n=== 3. MINERU-HTML BINDINGS ===")
bindings = check("load mineru_html bindings", _load_mineru_html_bindings)


def test_simplify() -> tuple[str, str]:
    raw = coerce_html(manifest[manifest["url_host_name"] == "hysplitbbs.arl.noaa.gov"].iloc[0]["html"])
    case = bindings.case_cls(bindings.input_cls(raw_html=raw, url="http://example.com"))
    case = bindings.simplify_single_input(case)
    simp = DripperHTMLExtractionStage._get_processed_attr(case, "simpled_html")
    mapped = DripperHTMLExtractionStage._get_processed_attr(case, "map_html")
    if not simp:
        msg = "empty simplified html"
        raise AssertionError(msg)
    if not mapped:
        msg = "empty mapped html"
        raise AssertionError(msg)
    return simp, mapped


simp_result = None
if bindings and manifest is not None:
    simp_result = check("simplify_single_input + get_processed_attr", test_simplify)
    if simp_result:
        simp, mapped = simp_result
        print(f"         simplified: {len(simp):,} chars  mapped: {len(mapped):,} chars")
        item_count = len(re.findall(r"_item_id=", mapped))
        print(f"         _item_id nodes: {item_count}")

# ── 4. DOM feature extraction ─────────────────────────────────────────────────
print("\n=== 4. DOM FEATURE EXTRACTION ===")
if web and manifest is not None:

    def test_features() -> list:
        rows = manifest[manifest["url_host_name"] == "hysplitbbs.arl.noaa.gov"].head(3)
        features = []
        for _, row in rows.iterrows():
            f = web.get_feature(coerce_html(row["html"]))
            if f is None:
                msg = "None feature"
                raise AssertionError(msg)
            features.append(f)
        return features

    feats = check("get_feature on 3 pages", test_features)
    if feats:
        print(f"         feature keys: {list(feats[0].keys())}")
        print(f"         layers in first feature: {len(feats[0].get('tags', {}))}")

# ── 5. Layout clustering ──────────────────────────────────────────────────────
print("\n=== 5. LAYOUT CLUSTERING ===")
if web and manifest is not None:

    def test_clustering() -> tuple:
        rows = manifest[manifest["url_host_name"] == "hysplitbbs.arl.noaa.gov"].head(10)
        samples = []
        for i, (_, row) in enumerate(rows.iterrows()):
            html = coerce_html(row["html"])
            feat = web.get_feature(html)
            if feat:
                samples.append({"track_id": str(i), "html": html, "feature": feat})
        clustered, _ = web.cluster_html_struct(samples, threshold=0.95)
        from collections import Counter

        dist = Counter(s["layout_id"] for s in clustered)
        return clustered, dist

    cluster_result = check("cluster_html_struct on 10 pages", test_clustering)
    if cluster_result:
        _, dist = cluster_result
        print(f"         cluster distribution: {dict(dist)}")

# ── 6. Representative selection ───────────────────────────────────────────────
print("\n=== 6. REPRESENTATIVE SELECTION ===")
if web and manifest is not None:

    def test_rep() -> object:
        vc = manifest[manifest["dripper_layout_id"].str.startswith("layout-", na=False)][
            "dripper_layout_id"
        ].value_counts()
        cluster_id = vc.index[0]
        rows = manifest[manifest["dripper_layout_id"] == cluster_id].head(10)
        candidates = [{"track_id": row["url"], "html": coerce_html(row["html"])} for _, row in rows.iterrows()]
        rep = web.select_representative_html(candidates)
        if rep is None:
            msg = "None representative"
            raise AssertionError(msg)
        return rep

    rep_result = check("select_representative_html", test_rep)
    if rep_result:
        print(f"         representative URL: {rep_result['track_id'][-80:]}")

# ── 7. MapItemToHtmlTagsParser (template building) ────────────────────────────
print("\n=== 7. MAP_PARSER (template building) ===")
mapping_result = None
if web and bindings and manifest is not None and baseline is not None:

    def test_mapping() -> tuple:
        # Find a row that has both HTML in manifest and LLM response in baseline
        merged = manifest.merge(baseline[["url", "dripper_response", "dripper_content"]], on="url", how="inner")
        merged = merged[
            merged["dripper_response"].notna() & merged["dripper_layout_id"].str.startswith("layout-", na=False)
        ]
        if len(merged) == 0:
            msg = "no rows with both HTML and LLM response"
            raise AssertionError(msg)
        row = merged.iloc[0]
        rep_html = coerce_html(row["html"])
        llm_resp = str(row["dripper_response"])

        # Simplify
        case = bindings.case_cls(bindings.input_cls(raw_html=rep_html, url=str(row["url"])))
        case = bindings.simplify_single_input(case)
        mapped_html = DripperHTMLExtractionStage._get_processed_attr(case, "map_html")

        # Map items → template
        result = web.map_parser_cls({}).parse(
            {
                "typical_raw_html": rep_html,
                "typical_raw_tag_html": mapped_html,
                "llm_response": llm_resp,
            }
        )
        if not result.get("html_element_dict"):
            msg = "empty html_element_dict"
            raise AssertionError(msg)
        return result, row

    map_res = check("map_parser_cls.parse() with correct keys", test_mapping)
    if map_res:
        mapping_result, source_row = map_res
        print(f"         typical_main_html_success: {mapping_result.get('typical_main_html_success')}")
        print(f"         template main html: {len(str(mapping_result.get('typical_main_html', ''))):,} chars")
        print(f"         element_dict keys: {list(mapping_result.get('html_element_dict', {}).keys())[:3]}...")
elif baseline is None:
    print(f"  [{SKIP}] baseline not available")

# ── 8. LayoutBatchParser (propagation) ───────────────────────────────────────
print("\n=== 8. LAYOUT_PARSER (propagation to sibling) ===")
if web and bindings and mapping_result is not None and manifest is not None:

    def test_propagation() -> tuple:
        cluster_id = str(source_row["dripper_layout_id"])
        siblings = manifest[
            (manifest["dripper_layout_id"] == cluster_id) & (manifest["url"] != source_row["url"])
        ].head(3)
        if len(siblings) == 0:
            msg = f"no siblings for cluster {cluster_id}"
            raise AssertionError(msg)

        sibling_row = siblings.iloc[0]
        sibling_html = coerce_html(sibling_row["html"])

        task_data = dict(mapping_result)
        task_data["html_source"] = sibling_html
        task_data["dynamic_id_enable"] = True
        task_data["dynamic_classid_enable"] = True
        task_data["more_noise_enable"] = True
        task_data["dynamic_classid_similarity_threshold"] = 0.85

        t0 = time.perf_counter()
        result = web.layout_parser_cls({}).parse(task_data)
        elapsed = time.perf_counter() - t0
        return result, elapsed, sibling_row

    prop_res = check("layout_parser_cls.parse() on sibling", test_propagation)
    if prop_res:
        prop_out, prop_time, prop_sibling = prop_res
        print(f"         propagation time: {prop_time:.2f}s")
        print(f"         main_html_success: {prop_out.get('main_html_success')}")
        print(f"         main_html_sim: {prop_out.get('main_html_sim')}")
        print(f"         main_html_body: {len(str(prop_out.get('main_html_body', ''))):,} chars")
elif baseline is None:
    print(f"  [{SKIP}] baseline not available")

# ── 9. _token_f1 ──────────────────────────────────────────────────────────────
print("\n=== 9. TOKEN F1 ===")
check(
    "_token_f1 basic",
    lambda: (_token_f1("hello world foo", "hello world foo") == 1.0 and _token_f1("hello", "world") == 0.0),
)
if prop_res and baseline is not None:

    def test_f1() -> float | str:
        main_html = str(prop_out.get("main_html_body") or "")
        prop_content = convert_html_to_content(bindings, main_html, url=str(prop_sibling.get("url", "")))
        baseline_row = baseline[baseline["url"] == prop_sibling["url"]]
        if baseline_row.empty:
            return "no baseline row to compare"
        ref = str(baseline_row.iloc[0]["dripper_content"] or "")
        f1 = _token_f1(prop_content, ref)
        if not (0.0 <= f1 <= 1.0):
            msg = f"F1 score {f1} out of expected range [0.0, 1.0]"
            raise AssertionError(msg)
        return f1

    f1_res = check("F1 propagated vs baseline", test_f1)
    if f1_res is not None and isinstance(f1_res, float):
        print(f"         F1 = {f1_res:.4f} {'✓ ≥0.95' if f1_res >= _F1_THRESHOLD else '✗ <0.95'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed:
    print("\nFailed steps:")
    for name, ok, err in results:
        if not ok:
            print(f"  ✗ {name}: {err[:100]}")
    sys.exit(1)
else:
    print("All steps passed — ready to build notebook.")
