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

import pytest

from nemo_curator.utils.uri import redact_uri_identity, validate_credential_free_uri_identity


@pytest.mark.parametrize(
    "value",
    [
        "/indexes/part-?.parquet",
        "simplecache::/indexes/part-?.parquet",
        "s3://[2001:db8::1]:9000/images",
        "simplecache::s3://[2001:db8::1]:9000/images",
    ],
)
def test_credential_free_uri_accepts_local_globs_and_ipv6_chains(value: str) -> None:
    assert validate_credential_free_uri_identity(value, "dataset URI") == value
    assert redact_uri_identity(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "simplecache::s3://dummy-user:dummy-pass@bucket/images",
        "zip://archive.zip::simplecache::s3://dummy-user:dummy-pass@bucket/images",
    ],
)
def test_credential_free_uri_rejects_userinfo_at_every_chain_depth(value: str) -> None:
    with pytest.raises(ValueError, match="URI userinfo") as raised:
        validate_credential_free_uri_identity(value, "dataset URI")

    assert "dummy-pass" not in str(raised.value)


@pytest.mark.parametrize(
    "value",
    [
        "s3://bucket/images?download=1",
        "simplecache::s3://bucket/images?download=1",
        "s3://bucket/images#snapshot",
        "simplecache::s3://bucket/images#snapshot",
    ],
)
def test_credential_free_uri_rejects_all_uri_queries_and_fragments(value: str) -> None:
    with pytest.raises(ValueError, match="persisted URI identities must remain stable and credential-free"):
        validate_credential_free_uri_identity(value, "dataset URI")


def test_credential_free_uri_rejects_empty_chain_component() -> None:
    with pytest.raises(ValueError, match="empty fsspec chain components"):
        validate_credential_free_uri_identity("simplecache::::s3://bucket/images", "dataset URI")


def test_uri_redaction_recurses_through_fsspec_chain() -> None:
    value = "simplecache::s3://dummy-user:dummy-pass@bucket/images?dummy-token=value#fragment"

    assert redact_uri_identity(value) == "simplecache::s3://bucket/images"
