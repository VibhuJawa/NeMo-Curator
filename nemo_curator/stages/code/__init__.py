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

"""Code annotation stages for NeMo Curator.

This module provides processing stages for code data curation:
- Language detection (hyperpolyglot-based)
- Code quality metrics (rust-code-analysis-based)
- License detection (scancode-based)
- Quality signals (OpenCoder-based)
"""

from nemo_curator.stages.code.code_metrics import CodeMetricsStage
from nemo_curator.stages.code.language_id import LanguageIdentificationStage
from nemo_curator.stages.code.license_detection import LicenseDetectionStage
from nemo_curator.stages.code.quality_signals import CodeQualitySignalsStage

__all__ = [
    "CodeMetricsStage",
    "CodeQualitySignalsStage",
    "LanguageIdentificationStage",
    "LicenseDetectionStage",
]
