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


def _split_fsspec_chain(value: str) -> tuple[str, ...]:
    """Split ``::`` chains without treating an IPv6 address as a delimiter."""

    parts = []
    start = 0
    bracket_depth = 0
    index = 0
    while index < len(value):
        character = value[index]
        if character == "[":
            bracket_depth += 1
        elif character == "]" and bracket_depth:
            bracket_depth -= 1
        elif bracket_depth == 0 and value.startswith("::", index):
            parts.append(value[start:index])
            index += 2
            start = index
            continue
        index += 1
    parts.append(value[start:])
    return tuple(parts)


def validate_credential_free_uri_identity(value: str, name: str) -> str:
    """Reject URI components that can carry credentials or ephemeral secrets."""

    if not isinstance(value, str) or not value:
        msg = f"{name} must be a non-empty URI or path identity"
        raise ValueError(msg)
    for position, component in enumerate(_split_fsspec_chain(value), start=1):
        if not component:
            msg = f"{name} must not contain empty fsspec chain components"
            raise ValueError(msg)
        try:
            parsed = urlsplit(component)
            has_userinfo = parsed.username is not None or parsed.password is not None
            _ = parsed.port
        except ValueError as exc:
            msg = f"{name} contains invalid syntax in fsspec chain component {position}"
            raise ValueError(msg) from exc
        is_uri = bool(parsed.scheme or parsed.netloc)
        if has_userinfo:
            msg = (
                f"{name} fsspec chain component {position} must not contain URI userinfo; "
                "supply credentials through the process environment or storage options"
            )
            raise ValueError(msg)
        if is_uri and (parsed.query or parsed.fragment):
            msg = (
                f"{name} fsspec chain component {position} must not contain a URI query or fragment; "
                "persisted URI identities must remain stable and credential-free, so supply credentials "
                "and backend options through the process environment or storage options"
            )
            raise ValueError(msg)
    return value


def _redact_uri_component(value: str) -> str:
    if not value:
        return "<redacted-invalid-uri>"
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


def redact_uri_identity(value: str) -> str:
    """Remove credential-bearing URI components from defensive diagnostics."""

    return "::".join(_redact_uri_component(component) for component in _split_fsspec_chain(value))
