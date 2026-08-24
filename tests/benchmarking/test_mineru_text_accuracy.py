# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "benchmarking" / "scripts" / "mineru_text_accuracy.py"
SPEC = importlib.util.spec_from_file_location("mineru_text_accuracy", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_token_overlap_f1() -> None:
    assert MODULE.token_overlap_f1("one two two", "one two") == 0.8
    assert MODULE.token_overlap_f1("", "") == 1.0
    assert MODULE.token_overlap_f1("one", "two") == 0.0
