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

"""Credential-free URI identities for persisted configuration metadata."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def validate_credential_free_uri_identity(value: str, name: str) -> str:
    """Reject URI components that can carry credentials or ephemeral secrets."""

    if not isinstance(value, str) or not value:
        msg = f"{name} must be a non-empty URI or path identity"
        raise ValueError(msg)
    try:
        parsed = urlsplit(value)
        has_userinfo = parsed.username is not None or parsed.password is not None
    except ValueError as exc:
        msg = f"{name} must be a valid URI or path identity"
        raise ValueError(msg) from exc
    is_uri = bool(parsed.scheme or parsed.netloc)
    if has_userinfo or (is_uri and (parsed.query or parsed.fragment)):
        msg = (
            f"{name} must not contain URI userinfo, query, or fragment components; "
            "supply credentials through the process environment or storage options"
        )
        raise ValueError(msg)
    return value


def redact_uri_identity(value: str) -> str:
    """Remove credential-bearing URI components from defensive diagnostics."""

    try:
        parsed = urlsplit(value)
        if not parsed.scheme and not parsed.netloc:
            return value
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "<redacted-invalid-uri>"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc if parsed.netloc else "", parsed.path, "", ""))
