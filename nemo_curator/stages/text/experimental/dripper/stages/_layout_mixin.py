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

"""Row-keying helpers shared by the clustering and layout-template stages.

``DripperHTMLLayoutClusteringStage`` and ``DripperHTMLLayoutTemplateStage`` derive the
same per-row host key, feature HTML, and page-signature key.  Both expose identical
``host_col``/``url_col``/``*_html_col``/``item_count_col`` fields; they differ only in the
*name* of the feature-source field (``layout_feature_source`` vs
``layout_template_feature_source``).  The mixin reconciles that single difference through
the ``_feature_source`` property, which each concrete stage implements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_curator.stages.text.experimental.dripper.stages._layout_utils import _layout_page_signature_key
from nemo_curator.stages.text.experimental.dripper.stages._utils import _coerce_html, _url_host_key

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd


class _LayoutRowKeyMixin:
    """Shared row-keying logic for the Dripper clustering and layout-template stages.

    Concrete stages provide the dataclass fields these methods read
    (``host_col``, ``url_col``, ``simplified_html_col``, ``mapped_html_col``,
    ``html_col``, ``item_count_col``, ``layout_page_signature_mode``,
    ``layout_exact_query_value_keys``) plus the ``_feature_source`` property.
    """

    # Provided by the concrete stage dataclasses; annotated here for readers/type checkers.
    host_col: str | None
    url_col: str | None
    simplified_html_col: str
    mapped_html_col: str
    html_col: str
    item_count_col: str
    layout_page_signature_mode: str
    layout_exact_query_value_keys: str | Iterable[str] | None

    @property
    def _feature_source(self) -> str:
        """Feature-source mode (``raw_html``/``simpled_html``/``mapped_html``).

        Each concrete stage maps its own field onto this name so the shared
        ``_row_feature_html`` can stay field-name agnostic.
        """
        raise NotImplementedError

    def _row_host_key(self, row: pd.Series) -> str:
        if self.host_col and self.host_col in row:
            host_key = _url_host_key(row.get(self.host_col))
            if host_key:
                return host_key
        return _url_host_key(row.get(self.url_col) if self.url_col else None)

    def _row_feature_html(self, row: pd.Series) -> str:
        if self._feature_source == "simpled_html":
            return _coerce_html(row.get(self.simplified_html_col, ""))
        if self._feature_source == "mapped_html":
            return _coerce_html(row.get(self.mapped_html_col, ""))
        return _coerce_html(row.get(self.html_col, ""))

    def _layout_page_signature_key(self, row: pd.Series) -> str:
        return _layout_page_signature_key(
            row.get(self.url_col) if self.url_col else None,
            row.get(self.item_count_col) if self.item_count_col in row else None,
            self.layout_page_signature_mode,
            exact_query_value_keys=self.layout_exact_query_value_keys,
        )
