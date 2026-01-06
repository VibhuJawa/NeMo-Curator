# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Code Annotation Library for NeMo Curator.

This package provides fast code annotation capabilities including:
- Basic statistics (byte counts, line stats, pattern detection)
- Language detection via hyperpolyglot
- Software metrics via rust-code-analysis
- BPE tokenization
- Decontamination via n-gram matching

Example usage:
    >>> from code_annotation import compute_basic_stats, compute_language_detection
    >>> codes = ["def hello(): print('world')", "int main() { return 0; }"]
    >>> stats = compute_basic_stats(codes)
    >>> print(stats[0]["num_bytes"])
    28
    >>> filenames = ["hello.py", "main.c"]
    >>> langs = compute_language_detection(codes, filenames)
    >>> print(langs[0]["language"])
    Python
"""

from code_annotation._code_annotation import (
    compute_basic_stats,
    compute_decontamination,
    compute_language_detection,
    compute_ngram_matches,
    compute_opencoder_rs,
    compute_tokenization,
)

# Try to import software_metrics if available (requires feature flag during build)
try:
    from code_annotation._code_annotation import compute_software_metrics

    HAS_SOFTWARE_METRICS = True
except ImportError:
    HAS_SOFTWARE_METRICS = False

    def compute_software_metrics(*args, **kwargs):
        raise ImportError(
            "compute_software_metrics requires the 'software_metrics' feature. "
            "Rebuild with: maturin develop --features software_metrics"
        )


__all__ = [
    "compute_basic_stats",
    "compute_language_detection",
    "compute_software_metrics",
    "compute_tokenization",
    "compute_opencoder_rs",
    "compute_decontamination",
    "compute_ngram_matches",
    "HAS_SOFTWARE_METRICS",
]
