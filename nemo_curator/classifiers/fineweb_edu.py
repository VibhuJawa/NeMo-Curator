# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"
import torch
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoModelForSequenceClassification

from nemo_curator.classifiers.base import (
    DistributedDataClassifier,
    _get_suggest_memory_for_classifier,
    _run_classifier_helper,
)
from nemo_curator.datasets import DocumentDataset

FINEWEB_EDU_IDENTIFIER = "HuggingFaceTB/fineweb-edu-classifier"


class FinewebEduModel(HFModel):
    def __init__(self, path_or_name, max_mem_gb, autocast=False):
        self.path_or_name = path_or_name
        self.autocast = autocast
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()
        super().__init__(path_or_name=path_or_name, max_mem_gb=max_mem_gb)

    def load_model(self, device="cuda"):
        model = AutoModelForSequenceClassification.from_pretrained(self.path_or_name)
        model = model.to(device)
        model = self.configure_forward(model, self.autocast)
        return model

    @staticmethod
    def configure_forward(model, autocast=True):
        original_forward = model.forward

        def custom_forward(*args, **kwargs):
            if autocast:
                with torch.autocast(device_type="cuda"):
                    output = original_forward(*args, **kwargs)
            return output.logits.squeeze(-1).float()

        model.forward = custom_forward
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


class FineWebEduClassifier(DistributedDataClassifier):
    def __init__(
        self,
        filter_by=None,
        batch_size=256,
        text_field: str = "text",
        pred_column="fineweb-edu-score",
        int_column="fineweb-edu-score-int",
        max_chars=-1,
        device_type="cuda",
        autocast=True,
        max_mem_gb=None,
    ):
        model = FinewebEduModel(
            path_or_name=FINEWEB_EDU_IDENTIFIER,
            autocast=autocast,
            max_mem_gb=max_mem_gb,
        )

        self.text_field = text_field
        self.int_column = int_column
        super().__init__(
            model=model,
            filter_by=filter_by,
            batch_size=batch_size,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            labels=None,
            out_dim=1,
        )

    def _run_classifier(self, dataset: DocumentDataset):
        print("Starting Fineweb EDU classifier inference", flush=True)
        ddf = dataset.df

        pipe = op.Sequential(
            op.Tokenizer(
                self.model,
                cols=[self.text_field],
                tokenizer_type="sentencepiece",
                max_length=self.model.max_seq_length(),
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col=self.pred_column,
            ),
            keep_cols=ddf.columns.tolist(),
        )
        ddf = pipe(ddf)
        # Go from list to scalar
        ddf[self.pred_column] = ddf[self.pred_column].list.get(0)
        ddf[self.int_column] = (
            ddf[self.pred_column].clip(lower=0, upper=5).round().astype(int)
        )
        return DocumentDataset(ddf)
