#!/usr/bin/env python3
"""
test_pipeline_correctness.py — pure-Python regression + correctness tests for the
7-stage MinerU-HTML CC-scale extraction pipeline.

These tests deliberately do NOT require the optional `mineru_html` /
`llm_web_kit` packages, nor any GPU/Ray/vLLM/Slurm access. The heavy imports in
the stage modules live inside worker-init functions (`_worker_init` /
`_init_worker` / inside Ray deployment `__init__`), so importing the modules
themselves is safe.

They lock in the four bug fixes found during the audit:
  #1  Stage 3 reads stage2b output (mapping_json), not raw stage2.
  #2  Stage 2b uses the standalone parse_result→extract_main_html_single→
      convert2content path (no nonexistent `main_html_body` map_parser key).
  #3  Stage 2 applies the tokenizer chat template (enable_thinking=False).
  #4  The propagation template is serialized pickle+base64 (tuple keys survive),
      not json.dumps(_sanitize(...)).

Run:  python3 -m pytest test_pipeline_correctness.py -v
"""

from __future__ import annotations

import base64
import importlib.util
import json
import pickle
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Module loading helpers (load by path; heavy deps are lazy inside workers)
# ---------------------------------------------------------------------------
def _load_module(name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage3 = _load_module("stage3_cpu_propagation", "stage3_cpu_propagation.py")
compare_f1 = _load_module("compare_f1", "compare_f1.py")


def _read(filename: str) -> str:
    return (HERE / filename).read_text()


# ===========================================================================
# stage3 _parse_mapping_json  (bug #4 regression: tuple keys must survive)
# ===========================================================================
class TestParseMappingJson:
    def test_pickle_base64_tuple_keys_round_trip(self) -> None:
        """The propagation template's html_element_dict has TUPLE KEYS. A JSON
        round-trip would stringify them and break LayoutBatchParser. pickle+base64
        must preserve them exactly (bug #4)."""
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
        if out != template:
            msg = f"decoded dict does not match original; got {out!r}"
            raise AssertionError(msg)
        # The tuple keys must remain tuples, not stringified.
        keys = list(out["html_element_dict"].keys())
        if not all(isinstance(k, tuple) for k in keys):
            msg = "html_element_dict keys are not all tuples"
            raise AssertionError(msg)
        if ("div", "class", "content") not in out["html_element_dict"]:
            msg = "expected tuple key ('div', 'class', 'content') missing"
            raise AssertionError(msg)
        if ("p",) not in out["html_element_dict"]:
            msg = "expected tuple key ('p',) missing"
            raise AssertionError(msg)

    def test_raw_bytes_pickle(self) -> None:
        template = {"html_element_dict": {("a", "b"): 1}}
        out = stage3._parse_mapping_json(pickle.dumps(template))
        if out != template:
            msg = f"decoded dict does not match; got {out!r}"
            raise AssertionError(msg)
        if ("a", "b") not in out["html_element_dict"]:
            msg = "expected tuple key ('a', 'b') missing"
            raise AssertionError(msg)

    def test_plain_dict_passthrough(self) -> None:
        d = {"a": 1, "b": {"c": 2}}
        if stage3._parse_mapping_json(d) is not d:
            msg = "plain dict should be returned as-is"
            raise AssertionError(msg)

    def test_legacy_json_string(self) -> None:
        d = {"foo": "bar", "n": 3}
        if stage3._parse_mapping_json(json.dumps(d)) != d:
            msg = "JSON string should decode to the original dict"
            raise AssertionError(msg)

    def test_none(self) -> None:
        if stage3._parse_mapping_json(None) is not None:
            msg = "None input should return None"
            raise AssertionError(msg)

    def test_nan(self) -> None:
        if stage3._parse_mapping_json(float("nan")) is not None:
            msg = "NaN input should return None"
            raise AssertionError(msg)

    def test_garbage_string(self) -> None:
        if stage3._parse_mapping_json("!!!not-valid-anything!!!") is not None:
            msg = "garbage string should return None"
            raise AssertionError(msg)

    def test_empty_string(self) -> None:
        if stage3._parse_mapping_json("") is not None:
            msg = "empty string should return None"
            raise AssertionError(msg)

    def test_json_list_is_rejected(self) -> None:
        # mapping_json must decode to a dict, not a list.
        if stage3._parse_mapping_json(json.dumps([1, 2, 3])) is not None:
            msg = "JSON list should be rejected (must decode to dict)"
            raise AssertionError(msg)


# ===========================================================================
# stage3 _parse_xpath_rules
# ===========================================================================
class TestParseXpathRules:
    def test_list_passthrough(self) -> None:
        rules = [{"xpath": "//div", "type": "t", "label": "l"}]
        if stage3._parse_xpath_rules(rules) is not rules:
            msg = "list should be returned as-is"
            raise AssertionError(msg)

    def test_json_string(self) -> None:
        rules = [{"xpath": "//p"}]
        if stage3._parse_xpath_rules(json.dumps(rules)) != rules:
            msg = "JSON string should decode to the original list"
            raise AssertionError(msg)

    def test_bytes(self) -> None:
        rules = [{"xpath": "//span"}]
        if stage3._parse_xpath_rules(json.dumps(rules).encode("utf-8")) != rules:
            msg = "UTF-8 bytes should decode to the original list"
            raise AssertionError(msg)

    def test_none(self) -> None:
        if stage3._parse_xpath_rules(None) is not None:
            msg = "None input should return None"
            raise AssertionError(msg)

    def test_nan(self) -> None:
        if stage3._parse_xpath_rules(float("nan")) is not None:
            msg = "NaN input should return None"
            raise AssertionError(msg)

    def test_garbage(self) -> None:
        if stage3._parse_xpath_rules("not json at all {[") is not None:
            msg = "garbage string should return None"
            raise AssertionError(msg)

    def test_json_dict_is_rejected(self) -> None:
        # xpath_rules must be a list, not a dict.
        if stage3._parse_xpath_rules(json.dumps({"a": 1})) is not None:
            msg = "JSON dict should be rejected (must decode to list)"
            raise AssertionError(msg)

    def test_empty_string(self) -> None:
        if stage3._parse_xpath_rules("") is not None:
            msg = "empty string should return None"
            raise AssertionError(msg)


# ===========================================================================
# stage3 _coerce_html
# ===========================================================================
class TestCoerceHtml:
    def test_bytes_to_str(self) -> None:
        if stage3._coerce_html(b"<html>hi</html>") != "<html>hi</html>":
            msg = "bytes should decode to str"
            raise AssertionError(msg)

    def test_bytearray_to_str(self) -> None:
        if stage3._coerce_html(bytearray(b"abc")) != "abc":
            msg = "bytearray should decode to str"
            raise AssertionError(msg)

    def test_none_to_empty(self) -> None:
        if stage3._coerce_html(None) != "":
            msg = "None should return empty string"
            raise AssertionError(msg)

    def test_str_passthrough(self) -> None:
        if stage3._coerce_html("<p>x</p>") != "<p>x</p>":
            msg = "str should be returned as-is"
            raise AssertionError(msg)

    def test_invalid_utf8_replaced(self) -> None:
        # decode errors -> replacement, never raises
        out = stage3._coerce_html(b"\xff\xfeabc")
        if not isinstance(out, str):
            msg = "result should be str even for invalid UTF-8"
            raise TypeError(msg)
        if "abc" not in out:
            msg = "ASCII portion 'abc' should survive replacement decoding"
            raise AssertionError(msg)


# ===========================================================================
# compare_f1.tokenize / f1
# ===========================================================================
class TestF1:
    def test_tokenize_basic(self) -> None:
        if compare_f1.tokenize("Hello, World!") != {"hello": 1, "world": 1}:
            msg = "tokenize should lowercase and strip punctuation"
            raise AssertionError(msg)

    def test_tokenize_empty(self) -> None:
        if compare_f1.tokenize("") != {}:
            msg = "empty string should tokenize to empty dict"
            raise AssertionError(msg)
        if compare_f1.tokenize(None) != {}:
            msg = "None should tokenize to empty dict"
            raise AssertionError(msg)

    def test_tokenize_lowercases_and_counts(self) -> None:
        if compare_f1.tokenize("a A a") != {"a": 3}:
            msg = "tokenize should count all occurrences case-insensitively"
            raise AssertionError(msg)

    def test_identical_is_one(self) -> None:
        if compare_f1.f1("the quick brown fox", "the quick brown fox") != 1.0:
            msg = "identical strings should have F1 = 1.0"
            raise AssertionError(msg)

    def test_disjoint_is_zero(self) -> None:
        if compare_f1.f1("alpha beta", "gamma delta") != 0.0:
            msg = "disjoint strings should have F1 = 0.0"
            raise AssertionError(msg)

    def test_both_empty_is_one(self) -> None:
        if compare_f1.f1("", "") != 1.0:
            msg = "both empty should have F1 = 1.0"
            raise AssertionError(msg)

    def test_one_empty_is_zero(self) -> None:
        if compare_f1.f1("something here", "") != 0.0:
            msg = "one empty string should have F1 = 0.0"
            raise AssertionError(msg)
        if compare_f1.f1("", "something here") != 0.0:
            msg = "one empty string should have F1 = 0.0"
            raise AssertionError(msg)

    def test_partial_overlap_harmonic(self) -> None:
        # pred = {a,b,c}, ref = {a,b,d}; common = 2
        # precision = 2/3, recall = 2/3, F1 = 2PR/(P+R) = 2/3
        got = compare_f1.f1("a b c", "a b d")
        if got != pytest.approx(2.0 / 3.0):
            msg = f"expected F1 ≈ 2/3, got {got}"
            raise AssertionError(msg)

    def test_partial_overlap_asymmetric(self) -> None:
        # pred = {a,b,c,d} (4 toks), ref = {a,b} (2 toks); common = 2
        # precision = 2/4 = 0.5, recall = 2/2 = 1.0
        # F1 = 2*0.5*1.0 / (0.5+1.0) = 1.0/1.5 = 2/3
        got = compare_f1.f1("a b c d", "a b")
        p, r = 0.5, 1.0
        if got != pytest.approx(2 * p * r / (p + r)):
            msg = f"expected F1 ≈ 2/3, got {got}"
            raise AssertionError(msg)

    def test_multiset_repeats_count(self) -> None:
        # pred = {a:2,b:1}, ref = {a:1,b:1}; common = min(2,1)+min(1,1) = 2
        # precision = 2/3, recall = 2/2 = 1.0
        got = compare_f1.f1("a a b", "a b")
        p, r = 2.0 / 3.0, 1.0
        if got != pytest.approx(2 * p * r / (p + r)):
            msg = f"expected F1 ≈ 2/3, got {got}"
            raise AssertionError(msg)


# ===========================================================================
# Source-text regression guards (grep-based, dependency-free)
# ===========================================================================
class TestPipelineWiringGuards:
    def test_bug1_stage3_reads_stage2b_not_stage2(self) -> None:
        """Bug #1: Stage 3 --inference-results must point at STAGE2B_OUT."""
        sh = _read("run_mineru_pipeline.sh")
        if "--inference-results '${STAGE2B_OUT}'" not in sh:
            msg = "Stage 3 must read STAGE2B_OUT (has mapping_json), not STAGE2_OUT"
            raise AssertionError(msg)
        if "--inference-results '${STAGE2_OUT}'" in sh:
            msg = "Stage 3 must NOT read the raw STAGE2_OUT (no mapping_json there)"
            raise AssertionError(msg)


class TestStage2bSerializationGuards:
    def test_bug4_pickle_base64_serialization(self) -> None:
        """Bug #4: template serialized via base64.b64encode(pickle.dumps(...))."""
        src = _read("stage2b_cpu_postprocess.py")
        if "base64.b64encode(pickle.dumps(" not in src:
            msg = "Stage 2b must serialize the template via pickle+base64 (tuple keys)"
            raise AssertionError(msg)

    def test_bug4_no_sanitize_jsondumps_template_path(self) -> None:
        """Bug #4: the lossy json.dumps(_sanitize(template)) path must be gone."""
        src = _read("stage2b_cpu_postprocess.py")
        if "_sanitize" in src:
            msg = "Stage 2b must not use a _sanitize() helper for the template"
            raise AssertionError(msg)
        # No json.dumps of the template object (the only json-serialized template
        # path was the buggy one). pickle is the serializer now.
        if "json.dumps(template" in src:
            msg = "Stage 2b must not use json.dumps(template ...)"
            raise AssertionError(msg)

    def test_bug2_no_main_html_body_key(self) -> None:
        """Bug #2: Stage 2b must not read the nonexistent map_parser
        `main_html_body` key; content comes from the standalone path."""
        src = _read("stage2b_cpu_postprocess.py")
        if "main_html_body" in src:
            msg = "Stage 2b must not read template['main_html_body'] (does not exist)"
            raise AssertionError(msg)

    def test_bug2_uses_standalone_extraction_path(self) -> None:
        """Bug #2: content built via parse_result -> extract_main_html_single ->
        convert2content (the standalone Dripper path)."""
        src = _read("stage2b_cpu_postprocess.py")
        if "parse_result" not in src:
            msg = "Stage 2b must use parse_result"
            raise AssertionError(msg)
        if "extract_main_html_single" not in src:
            msg = "Stage 2b must use extract_main_html_single"
            raise AssertionError(msg)
        if "convert2content" not in src:
            msg = "Stage 2b must use convert2content"
            raise AssertionError(msg)


class TestStage2ChatTemplateGuards:
    def test_bug3_applies_chat_template(self) -> None:
        """Bug #3: Stage 2 must apply the tokenizer chat template before
        engine.generate (raw prompt -> degenerate 'mainmainmain' output)."""
        src = _read("stage2_gpu_inference.py")
        if "apply_chat_template" not in src:
            msg = "Stage 2 must apply the chat template, not feed the raw prompt"
            raise AssertionError(msg)
        if "enable_thinking" not in src:
            msg = "Stage 2 chat template must pass enable_thinking (=False) like standalone"
            raise AssertionError(msg)

    def test_bug3_loads_tokenizer(self) -> None:
        src = _read("stage2_gpu_inference.py")
        if "AutoTokenizer" not in src:
            msg = "Stage 2 must load AutoTokenizer"
            raise AssertionError(msg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
