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

"""Dripper HTML main-content extraction pipeline and stage exports."""

from nemo_curator.stages.text.experimental.dripper.pipeline import DripperCommonCrawlPipeline
from nemo_curator.stages.text.experimental.dripper.stages.clustering import DripperHTMLLayoutClusteringStage
from nemo_curator.stages.text.experimental.dripper.stages.grouping import HostDomainGroupingStage
from nemo_curator.stages.text.experimental.dripper.stages.inference import DripperHTMLInferenceStage
from nemo_curator.stages.text.experimental.dripper.stages.layout_finalize import DripperHTMLLayoutFinalizeStage
from nemo_curator.stages.text.experimental.dripper.stages.layout_plan import DripperHTMLLayoutPlanStage
from nemo_curator.stages.text.experimental.dripper.stages.postprocess import DripperHTMLPostprocessStage
from nemo_curator.stages.text.experimental.dripper.stages.preprocess import DripperHTMLPreprocessStage

__all__ = [
    "DripperCommonCrawlPipeline",
    "DripperHTMLInferenceStage",
    "DripperHTMLLayoutClusteringStage",
    "DripperHTMLLayoutFinalizeStage",
    "DripperHTMLLayoutPlanStage",
    "DripperHTMLPostprocessStage",
    "DripperHTMLPreprocessStage",
    "HostDomainGroupingStage",
]
