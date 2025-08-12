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

from collections.abc import Callable
from dataclasses import dataclass

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.text.modifiers.doc_modifier import DocumentModifier
from ray_curator.tasks import DocumentBatch


@dataclass
class Modify(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    The module responsible for modifying the text of the records in the dataset.
    It accepts an arbitrary function that accepts a text field and returns a modified text field.
    It also accepts a DocumentModifier object, in which case the score_fn will be the score_document method of the DocumentFilter.

    Args:
        modifier_fn (Callable | DocumentModifier | list[DocumentModifier]): The score function or the DocumentModifier object (or list of DocumentModifiers). If it is a DocumentModifier object, the score_fn will be the score_document method of the DocumentModifier.
        text_field (str | list[str]): The field (or list of fields) the documents will be read from.

    """

    modifier_fn: Callable[[str], float | str] | DocumentModifier | list[DocumentModifier]
    text_field: str | list[str] = "text"
    _name: str = "modifier_fn"

    def __post_init__(self):
        self.modifier_fn = _validate_and_normalize_modifiers(self.modifier_fn, self.text_field)
        self.text_field = _create_text_fields(self.text_field, self.modifier_fn)
        self._name = _get_modifier_stage_name(self.modifier_fn)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        for modifier_fn in self.modifier_fn:
            if isinstance(modifier_fn, DocumentModifier) and hasattr(modifier_fn, "model_check_or_download"):
                modifier_fn.model_check_or_download()

    def setup(self, _: WorkerMetadata | None = None) -> None:
        for modifier_fn in self.modifier_fn:
            if isinstance(modifier_fn, DocumentModifier) and hasattr(modifier_fn, "load_model"):
                modifier_fn.load_model()

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Applies the scoring to a dataset

        Args:
            batch (DocumentBatch): The batch to apply the module to

        Returns:
            DocumentBatch: A batch with the new score

        """

        df = batch.to_pandas()

        for modifier_fn_i, text_field_i in zip(self.modifier_fn, self.text_field, strict=True):
            inner_modifier_fn = (
                modifier_fn_i.modify_document if isinstance(modifier_fn_i, DocumentModifier) else modifier_fn_i
            )
            df[text_field_i] = df[text_field_i].apply(inner_modifier_fn)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def _modifier_name(x: DocumentModifier | Callable) -> str:
    return x.name if isinstance(x, DocumentModifier) else x.__name__


def _get_modifier_stage_name(modifiers: list[DocumentModifier | Callable]) -> str:
    """
    Derive the stage name from the provided modifiers.
    """
    return (
        _modifier_name(modifiers[0])
        if len(modifiers) == 1
        else "modifier_chain_of_" + "_".join(_modifier_name(m) for m in modifiers)
    )


def _validate_and_normalize_modifiers(
    _modifier: DocumentModifier | Callable | list[DocumentModifier | Callable],
    text_field: str | list[str] | None,
) -> list[DocumentModifier | Callable]:
    """
    Validate inputs and normalize the modifier(s) to a list.
    """
    if text_field is None:
        msg = "Text field cannot be None"
        raise ValueError(msg)

    modifiers: list[DocumentModifier | Callable] = _modifier if isinstance(_modifier, list) else [_modifier]
    if not modifiers:
        msg = "modifier_fn list cannot be empty"
        raise ValueError(msg)
    if any(not (isinstance(m, DocumentModifier) or callable(m)) for m in modifiers):
        msg = "Each modifier must be a DocumentModifier or callable"
        raise TypeError(msg)
    if len(modifiers) == 1 and isinstance(text_field, list) and len(text_field) > 1:
        msg = f"More text fields than modifiers provided: {text_field}"
        raise ValueError(msg)

    return modifiers


def _create_text_fields(text_field: str | list[str], modifiers: list[DocumentModifier | Callable]) -> list[str]:
    """
    Create/expand text fields to match the number of modifiers.
    """
    if isinstance(text_field, list):
        if len(text_field) == len(modifiers):
            return text_field
        elif len(text_field) == 1:
            return text_field * len(modifiers)
        else:
            msg = (
                f"Number of text fields ({len(text_field)}) must be 1 or equal to number of "
                f"modifiers ({len(modifiers)})"
            )
            raise ValueError(msg)
    else:
        return [text_field] * len(modifiers)
