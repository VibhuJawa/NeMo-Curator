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

"""Pure-Python regression tests for the MinerU-HTML clustering + propagation tutorial.

These tests cover the dependency-free helpers of the 7-stage CC-scale extraction
pipeline that lives under ``tutorials/text/dripper-common-crawl/``. They deliberately
do NOT require the optional ``mineru_html`` / ``llm_web_kit`` packages, nor any
GPU / Ray / vLLM access: the heavy imports in the stage scripts all live inside
worker-init functions, so importing the modules themselves is safe.

They lock in the four correctness invariants of the pipeline:
  #1  Stage 3 reads Stage 2b output (the pickled mapping), not the raw Stage 2 output.
  #2  Stage 2b builds content via the standalone parse_result -> extract_main_html_single
      -> convert2content path (no nonexistent ``main_html_body`` map_parser key).
  #3  Stage 2 applies the tokenizer chat template (``enable_thinking=False``).
  #4  The propagation template is serialized with pickle+base64 so the tuple keys in
      ``html_element_dict`` survive (a JSON round-trip would stringify them).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import pickle
from pathlib import Path
from types import ModuleType

import pytest

# tests/stages/text/experimental/dripper/ -> repo root is five parents up.
_REPO_ROOT = Path(__file__).resolve().parents[5]
_TUTORIAL_DIR = _REPO_ROOT / "tutorials" / "text" / "dripper-common-crawl"


def _load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _TUTORIAL_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage3 = _load_module("stage3_cpu_propagation", "stage3_cpu_propagation.py")
compare_f1 = _load_module("compare_f1", "compare_f1.py")


def _read(filename: str) -> str:
    return (_TUTORIAL_DIR / filename).read_text()


class TestParseMappingJson:
    """stage3._parse_mapping_json (bug #4 regression: tuple keys must survive)."""

    def test_pickle_base64_tuple_keys_round_trip(self):
        """The propagation template's html_element_dict has TUPLE KEYS.

        A JSON round-trip would stringify them and break LayoutBatchParser;
        pickle+base64 must preserve them exactly (bug #4).
        """
        template = {
            "html_element_dict": {
                ("div", "class", "content"): "node-a",
                ("p",): "node-b",
                ("span", "id"): 42,
            },
            "scalar": "value",
            "nested": {("k1", "k2"): [1, 2, 3]},
        }
        encoded = base64.b64encode(pickle.dumps(template)).decode("ascii")

        out = stage3._parse_mapping_json(encoded)
        assert out == template
        keys = list(out["html_element_dict"].keys())
        assert all(isinstance(k, tuple) for k in keys)
        assert ("div", "class", "content") in out["html_element_dict"]
        assert ("p",) in out["html_element_dict"]

    def test_raw_bytes_pickle(self):
        template = {"html_element_dict": {("a", "b"): 1}}
        out = stage3._parse_mapping_json(pickle.dumps(template))
        assert out == template
        assert ("a", "b") in out["html_element_dict"]

    def test_plain_dict_passthrough(self):
        d = {"a": 1, "b": {"c": 2}}
        assert stage3._parse_mapping_json(d) is d

    def test_legacy_json_string(self):
        d = {"foo": "bar", "n": 3}
        assert stage3._parse_mapping_json(json.dumps(d)) == d

    def test_none(self):
        assert stage3._parse_mapping_json(None) is None

    def test_nan(self):
        assert stage3._parse_mapping_json(float("nan")) is None

    def test_garbage_string(self):
        assert stage3._parse_mapping_json("!!!not-valid-anything!!!") is None

    def test_empty_string(self):
        assert stage3._parse_mapping_json("") is None

    def test_json_list_is_rejected(self):
        # A mapping must decode to a dict, not a list.
        assert stage3._parse_mapping_json(json.dumps([1, 2, 3])) is None


class TestParseXpathRules:
    """stage3._parse_xpath_rules."""

    def test_list_passthrough(self):
        rules = [{"xpath": "//div", "type": "t", "label": "l"}]
        assert stage3._parse_xpath_rules(rules) is rules

    def test_json_string(self):
        rules = [{"xpath": "//p"}]
        assert stage3._parse_xpath_rules(json.dumps(rules)) == rules

    def test_bytes(self):
        rules = [{"xpath": "//span"}]
        assert stage3._parse_xpath_rules(json.dumps(rules).encode("utf-8")) == rules

    def test_none(self):
        assert stage3._parse_xpath_rules(None) is None

    def test_nan(self):
        assert stage3._parse_xpath_rules(float("nan")) is None

    def test_garbage(self):
        assert stage3._parse_xpath_rules("not json at all {[") is None

    def test_json_dict_is_rejected(self):
        # xpath_rules must be a list, not a dict.
        assert stage3._parse_xpath_rules(json.dumps({"a": 1})) is None

    def test_empty_string(self):
        assert stage3._parse_xpath_rules("") is None


class TestCoerceHtml:
    """stage3._coerce_html."""

    def test_bytes_to_str(self):
        assert stage3._coerce_html(b"<html>hi</html>") == "<html>hi</html>"

    def test_bytearray_to_str(self):
        assert stage3._coerce_html(bytearray(b"abc")) == "abc"

    def test_none_to_empty(self):
        assert stage3._coerce_html(None) == ""

    def test_str_passthrough(self):
        assert stage3._coerce_html("<p>x</p>") == "<p>x</p>"

    def test_invalid_utf8_replaced(self):
        # Decode errors -> replacement, never raises.
        out = stage3._coerce_html(b"\xff\xfeabc")
        assert isinstance(out, str)
        assert "abc" in out


class TestF1:
    """compare_f1.tokenize / compare_f1.f1."""

    def test_tokenize_basic(self):
        assert compare_f1.tokenize("Hello, World!") == {"hello": 1, "world": 1}

    def test_tokenize_empty(self):
        assert compare_f1.tokenize("") == {}
        assert compare_f1.tokenize(None) == {}

    def test_tokenize_lowercases_and_counts(self):
        assert compare_f1.tokenize("a A a") == {"a": 3}

    def test_identical_is_one(self):
        assert compare_f1.f1("the quick brown fox", "the quick brown fox") == 1.0

    def test_disjoint_is_zero(self):
        assert compare_f1.f1("alpha beta", "gamma delta") == 0.0

    def test_both_empty_is_one(self):
        assert compare_f1.f1("", "") == 1.0

    def test_one_empty_is_zero(self):
        assert compare_f1.f1("something here", "") == 0.0
        assert compare_f1.f1("", "something here") == 0.0

    def test_partial_overlap_harmonic(self):
        # pred = {a,b,c}, ref = {a,b,d}; common = 2 -> P = R = 2/3 -> F1 = 2/3.
        got = compare_f1.f1("a b c", "a b d")
        assert got == pytest.approx(2.0 / 3.0)

    def test_partial_overlap_asymmetric(self):
        # pred = {a,b,c,d}, ref = {a,b}; common = 2 -> P = 0.5, R = 1.0.
        got = compare_f1.f1("a b c d", "a b")
        p, r = 0.5, 1.0
        assert got == pytest.approx(2 * p * r / (p + r))

    def test_multiset_repeats_count(self):
        # pred = {a:2,b:1}, ref = {a:1,b:1}; common = min(2,1)+min(1,1) = 2.
        got = compare_f1.f1("a a b", "a b")
        p, r = 2.0 / 3.0, 1.0
        assert got == pytest.approx(2 * p * r / (p + r))


class TestStage2bSerializationGuards:
    """Source guards on the Stage 2b postprocess script."""

    def test_bug4_pickle_base64_serialization(self):
        """Bug #4: template serialized via base64.b64encode(pickle.dumps(...))."""
        src = _read("stage2b_cpu_postprocess.py")
        assert "base64.b64encode(pickle.dumps(" in src

    def test_bug4_no_sanitize_jsondumps_template_path(self):
        """Bug #4: the lossy json.dumps(_sanitize(template)) path must be gone."""
        src = _read("stage2b_cpu_postprocess.py")
        assert "_sanitize" not in src
        assert "json.dumps(template" not in src

    def test_bug2_no_main_html_body_key(self):
        """Bug #2: Stage 2b must not read the nonexistent map_parser main_html_body key."""
        src = _read("stage2b_cpu_postprocess.py")
        assert "main_html_body" not in src

    def test_bug2_uses_standalone_extraction_path(self):
        """Bug #2: content built via parse_result -> extract_main_html_single -> convert2content."""
        src = _read("stage2b_cpu_postprocess.py")
        assert "parse_result" in src
        assert "extract_main_html_single" in src
        assert "convert2content" in src


class TestStage2ChatTemplateGuards:
    """Source guards on the Stage 2 offline inference script."""

    def test_bug3_applies_chat_template(self):
        """Bug #3: Stage 2 must apply the chat template (enable_thinking=False)."""
        src = _read("stage2_gpu_inference_offline.py")
        assert "apply_chat_template" in src
        assert "enable_thinking" in src

    def test_bug3_loads_tokenizer(self):
        src = _read("stage2_gpu_inference_offline.py")
        assert "AutoTokenizer" in src
