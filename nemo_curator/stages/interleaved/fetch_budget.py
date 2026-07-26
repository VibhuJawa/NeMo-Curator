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

"""Cluster-wide in-flight request budget for object-store fetches.

Sizing IO concurrency per worker makes the number of in-flight requests grow
with the worker count, so scaling a pipeline out increases the pressure it puts
on shared object storage. The budget here is owned by one named Ray actor and
divided among the workers registered with it, so the cluster-wide total stays
flat as workers come and go.

The actor is scoped to the run, not detached. Registrations are only released by
``teardown()``, and workers that are killed (timeout, OOM, node loss) never get
there; a budget that outlived the job would accumulate those dead registrations
across runs and silently shrink every share towards the per-worker floor. A
genuinely persistent budget needs heartbeats and TTL expiry, which is a separate
change.
"""

from __future__ import annotations

import os
import socket

import ray
from loguru import logger

BUDGET_ACTOR_NAME = "curator_lance_fetch_budget"
BUDGET_ACTOR_NAMESPACE = "nemo_curator"

_CALL_TIMEOUT_SECONDS = 30.0
# Recover 1/16th of the current allowance per clean report: ~12 reports back to
# full after a halving, slow enough not to walk straight back into a throttle.
_RECOVERY_DIVISOR = 16
_OVERLOAD_MARKERS = ("503", "reduce your request rate", "serviceunavailable")


def worker_id() -> str:
    """Identity of the calling worker process, stable across repeated setups."""
    return f"{socket.gethostname()}:{os.getpid()}"


def is_overload_error(error: BaseException) -> bool:
    """Whether an exception, or anything it chains to, is a store overload signal."""
    messages: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    text = " ".join(messages).lower()
    return any(marker in text for marker in _OVERLOAD_MARKERS)


class FetchBudget:
    """Split a cluster-wide in-flight request allowance across registered workers.

    Deliberately free of Ray so the arithmetic can be exercised directly.
    """

    def __init__(self, total: int, minimum: int) -> None:
        if not 0 < minimum <= total:
            msg = "minimum must be in (0, total]"
            raise ValueError(msg)
        self._maximum = total
        self._minimum = minimum
        self._total = total
        self._workers: set[str] = set()

    @property
    def total(self) -> int:
        """Current cluster-wide allowance, independent of the worker count."""
        return self._total

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def share(self) -> int:
        """Per-worker allowance; every worker keeps at least one in-flight request.

        That floor is deliberate, and it means a cluster with more workers than
        budget does exceed the configured total. Handing a worker a share of zero
        would stall it outright, which is the worse failure.
        """
        return max(1, self._total // max(1, len(self._workers)))

    def register(self, worker_id: str) -> int:
        """Claim a share for ``worker_id``; repeat registrations are idempotent."""
        self._workers.add(worker_id)
        return self.share()

    def unregister(self, worker_id: str) -> None:
        """Return a departing worker's share to the pool."""
        self._workers.discard(worker_id)

    def report(self, *, overloaded: bool) -> int:
        """Halve the allowance when the store pushes back, otherwise recover a slice of it.

        Recovery is proportional rather than additive so that it takes a bounded
        number of reports at any scale: an additive step would leave a budget
        halved from a large total effectively pinned there for the whole run.
        """
        if overloaded:
            self._total = max(self._minimum, self._total // 2)
        else:
            self._total = min(self._maximum, self._total + max(1, self._total // _RECOVERY_DIVISOR))
        return self.share()


@ray.remote(num_cpus=0)
class FetchBudgetActor:
    """Named owner of a :class:`FetchBudget`, shared by every worker in a run."""

    def __init__(self, total: int, minimum: int) -> None:
        self._budget = FetchBudget(total, minimum)

    def register(self, worker_id: str) -> int:
        return self._budget.register(worker_id)

    def unregister(self, worker_id: str) -> None:
        self._budget.unregister(worker_id)

    def report(self, overloaded: bool) -> int:
        return self._budget.report(overloaded=overloaded)


def _actor() -> ray.actor.ActorHandle:
    return ray.get_actor(name=BUDGET_ACTOR_NAME, namespace=BUDGET_ACTOR_NAMESPACE)


def register_worker(total: int, minimum: int, worker_id: str) -> int | None:
    """Return this worker's share of the shared budget, or ``None`` if it is unreachable.

    Budget faults never fail a run: callers fall back to their configured
    concurrency and simply run without the cluster-wide cap.
    """
    try:
        actor = FetchBudgetActor.options(  # type: ignore[attr-defined]
            name=BUDGET_ACTOR_NAME,
            namespace=BUDGET_ACTOR_NAMESPACE,
            get_if_exists=True,
        ).remote(total, minimum)
        return int(ray.get(actor.register.remote(worker_id), timeout=_CALL_TIMEOUT_SECONDS))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Fetch budget unavailable; falling back to local IO concurrency: {exc}")
        return None


def report_outcome(*, overloaded: bool) -> int | None:
    """Report a fetch outcome and return the updated share, or ``None`` if unreachable."""
    try:
        return int(ray.get(_actor().report.remote(overloaded), timeout=_CALL_TIMEOUT_SECONDS))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Fetch budget update failed; keeping current IO concurrency: {exc}")
        return None


def unregister_worker(worker_id: str) -> None:
    """Release this worker's share so surviving workers can reclaim it."""
    try:
        ray.get(_actor().unregister.remote(worker_id), timeout=_CALL_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Fetch budget release failed: {exc}")
