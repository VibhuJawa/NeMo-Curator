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
"""

from code_annotation._code_annotation import (
    CODE_COL_NAME,
    COMPRESSED_SRC_COL_NAME,
    FILENAME_COL_NAME,
    LANGUAGE_COL_NAME,
    TOKENS_COL_NAME,
    annotate,
)

__all__ = [
    "annotate",
    "CODE_COL_NAME",
    "COMPRESSED_SRC_COL_NAME",
    "FILENAME_COL_NAME",
    "LANGUAGE_COL_NAME",
    "TOKENS_COL_NAME",
]
