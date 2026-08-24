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

from unittest import mock

import pytest

from nemo_curator.core.utils import get_free_port, ignore_ray_head_node


def test_get_free_port_probes_all_ipv4_interfaces() -> None:
    first_port = 23456
    fake_socket = mock.MagicMock()
    fake_socket.__enter__.return_value = fake_socket
    fake_socket.bind.side_effect = [OSError, None]

    with mock.patch("nemo_curator.core.utils.socket.socket", return_value=fake_socket):
        assert get_free_port(first_port) == first_port + 1

    assert fake_socket.bind.call_args_list == [
        mock.call(("0.0.0.0", first_port)),  # noqa: S104
        mock.call(("0.0.0.0", first_port + 1)),  # noqa: S104
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        *[(v, True) for v in ("1", "true", "TRUE", "yes", " 1 ")],
    ],
)
def test_ignore_ray_head_node_env_parsing(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool) -> None:
    if value is None:
        monkeypatch.delenv("CURATOR_IGNORE_RAY_HEAD_NODE", raising=False)
    else:
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", value)
    assert ignore_ray_head_node() is expected
