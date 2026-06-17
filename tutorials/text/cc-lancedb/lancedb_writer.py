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

"""CC-specific LanceDB schemas and re-export of LanceDBWriter from NeMo Curator.

Each row stores the HTML bytes ONCE plus the output of all three extractors
side-by-side, enabling direct comparison without re-fetching.
"""

import pyarrow as pa

from nemo_curator.stages.text.io.writer import LanceDBWriter

# URL index table: url + WARC coordinates + HTML bytes + 3 extractor outputs
# Note: LanceDBWriter.__post_init__ auto-injects lance-encoding:blob on large_binary fields.
LANCEDB_URL_INDEX_SCHEMA = pa.schema(
    [
        pa.field("cc_url", pa.string()),
        pa.field("cc_snapshot_id", pa.string()),
        pa.field("warc_filename", pa.string()),
        pa.field("warc_record_offset", pa.int32()),
        pa.field("warc_record_length", pa.int32()),
        pa.field("cc_html_bytes", pa.large_binary()),
        pa.field("cc_extracted_text_trafilatura", pa.large_string()),
        pa.field("cc_extracted_text_justext", pa.large_string()),
        pa.field("cc_extracted_text_resiliparse", pa.large_string()),
        pa.field("content_digest", pa.string()),
        pa.field("url_host_name", pa.string()),
    ]
)

__all__ = ["LANCEDB_URL_INDEX_SCHEMA", "LanceDBWriter"]
