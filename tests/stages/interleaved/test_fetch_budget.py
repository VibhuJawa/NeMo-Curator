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

"""Budget arithmetic for the cluster-wide fetch budget. Runs without Ray."""

import pytest

from nemo_curator.stages.interleaved.fetch_budget import FetchBudget, is_overload_error


def test_share_is_invariant_to_worker_count():
    budget = FetchBudget(total=64, minimum=8)
    shares = []
    for index in range(8):
        shares.append(budget.register(f"worker-{index}"))
    assert shares == [64, 32, 21, 16, 12, 10, 9, 8]
    assert budget.total == 64
    assert budget.worker_count * budget.share() <= budget.total


def test_registration_is_idempotent():
    budget = FetchBudget(total=64, minimum=8)
    assert budget.register("worker-a") == 64
    assert budget.register("worker-a") == 64
    assert budget.worker_count == 1


def test_unregister_returns_the_share_to_survivors():
    budget = FetchBudget(total=64, minimum=8)
    budget.register("worker-a")
    budget.register("worker-b")
    budget.unregister("worker-b")
    assert budget.share() == 64
    budget.unregister("worker-b")
    assert budget.worker_count == 1


def test_every_worker_keeps_at_least_one_request():
    budget = FetchBudget(total=4, minimum=1)
    for index in range(16):
        budget.register(f"worker-{index}")
    assert budget.share() == 1


def test_overload_halves_the_budget_down_to_the_floor():
    budget = FetchBudget(total=64, minimum=8)
    budget.register("worker-a")
    assert [budget.report(overloaded=True) for _ in range(4)] == [32, 16, 8, 8]
    assert budget.total == 8


def test_recovery_is_gradual_and_capped_at_the_initial_total():
    budget = FetchBudget(total=1024, minimum=64)
    budget.register("worker-a")
    budget.report(overloaded=True)
    assert budget.total == 512
    assert [budget.report(overloaded=False) for _ in range(3)] == [544, 578, 614]
    for _ in range(64):
        budget.report(overloaded=False)
    assert budget.total == 1024


def test_recovery_from_the_floor_completes_in_a_bounded_number_of_reports():
    """An additive step would need ~1000 reports here, i.e. never within a run."""
    budget = FetchBudget(total=1024, minimum=8)
    budget.register("worker-a")
    for _ in range(8):
        budget.report(overloaded=True)
    assert budget.total == 8

    reports = 0
    while budget.total < 1024:
        budget.report(overloaded=False)
        reports += 1
        assert reports < 100, f"recovery stalled at {budget.total} after {reports} reports"


def test_recovery_after_one_throttle_is_bounded_at_any_scale():
    for total in (1024, 4096, 65_536):
        budget = FetchBudget(total=total, minimum=8)
        budget.register("worker-a")
        budget.report(overloaded=True)
        reports = 0
        while budget.total < total:
            budget.report(overloaded=False)
            reports += 1
        assert reports == 12


def test_shrunk_budget_is_still_divided_among_workers():
    budget = FetchBudget(total=64, minimum=8)
    budget.register("worker-a")
    budget.register("worker-b")
    assert budget.report(overloaded=True) == 16


@pytest.mark.parametrize(("total", "minimum"), [(16, 0), (16, 32), (0, 0), (-1, 1)])
def test_invalid_bounds_are_rejected(total: int, minimum: int):
    with pytest.raises(ValueError, match="minimum must be in"):
        FetchBudget(total=total, minimum=minimum)


@pytest.mark.parametrize(
    "message",
    [
        "Service error: 503 Slow Down",
        "Please reduce your request rate.",
        "ServiceUnavailable: retry later",
        "SERVICEUNAVAILABLE",
    ],
)
def test_overload_signals_are_detected(message: str):
    assert is_overload_error(RuntimeError(message)) is True


@pytest.mark.parametrize("message", ["404 Not Found", "connection reset by peer", ""])
def test_unrelated_errors_are_not_overload(message: str):
    assert is_overload_error(RuntimeError(message)) is False


def test_overload_is_detected_through_the_exception_chain():
    error = RuntimeError("lance take failed")
    error.__cause__ = ConnectionError("503 Slow Down")
    assert is_overload_error(error) is True
