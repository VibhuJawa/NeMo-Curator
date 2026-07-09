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

from __future__ import annotations

from types import SimpleNamespace

import pytest

rapidsmpf = pytest.importorskip("rapidsmpf")
pytest.importorskip("cudf")
_RAPIDS_MPF_RELEASE = tuple(int(component) for component in rapidsmpf.__version__.split(".")[:2])
if _RAPIDS_MPF_RELEASE != (26, 6):
    pytest.skip("GPU Lance shuffle tests require RAPIDS-MPF 26.06", allow_module_level=True)

from rapidsmpf.integrations.ray import RapidsMPFActor  # noqa: E402

from nemo_curator.stages.interleaved import gpu_lance_shuffle_actor as gpu_lance_actor_module  # noqa: E402
from nemo_curator.stages.interleaved import rapidsmpf_2606_shuffler as shuffler_module  # noqa: E402
from nemo_curator.stages.interleaved.rapidsmpf_2606_shuffler import (  # noqa: E402
    GpuLanceRapidsMPFShuffler,
)

pytestmark = pytest.mark.gpu


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def _uninitialized_actor() -> GpuLanceRapidsMPFShuffler:
    actor = object.__new__(GpuLanceRapidsMPFShuffler)
    actor._comm = SimpleNamespace(logger=_Logger())
    return actor


def test_shuffler_imports_the_rapidsmpf_2606_actor_api() -> None:
    assert _RAPIDS_MPF_RELEASE == (26, 6)
    assert issubclass(GpuLanceRapidsMPFShuffler, RapidsMPFActor)


def test_actor_initializes_memory_and_shares_statistics_before_base_init(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []
    statistics = SimpleNamespace(enabled=True)
    memory_resource = object()

    class FakeBufferResource:
        def __init__(
            self,
            device_mr: object,
            *,
            memory_available: object,
            statistics: object,
        ) -> None:
            assert device_mr is memory_resource
            assert memory_available is None
            self.statistics = statistics
            events.append("buffer_resource")

    def fake_statistics(*, enable: bool) -> object:
        assert enable is True
        return statistics

    def fake_pool_memory_resource(
        _upstream: object,
        *,
        initial_pool_size: int,
        maximum_pool_size: None,
    ) -> object:
        assert initial_pool_size == 1024
        assert maximum_pool_size is None
        return object()

    def fake_rmm_resource_adaptor(_pool: object) -> object:
        return memory_resource

    def fake_base_init(self: RapidsMPFActor, nranks: int, actor_statistics: object) -> None:
        assert nranks == 2
        assert hasattr(self, "mr")
        assert hasattr(self, "br")
        self._stats = actor_statistics
        events.append("base_actor")

    monkeypatch.setattr(shuffler_module, "Statistics", fake_statistics)
    monkeypatch.setattr(shuffler_module.rmm.mr, "CudaMemoryResource", lambda: object())
    monkeypatch.setattr(shuffler_module.rmm.mr, "PoolMemoryResource", fake_pool_memory_resource)
    monkeypatch.setattr(shuffler_module, "RmmResourceAdaptor", fake_rmm_resource_adaptor)
    monkeypatch.setattr(
        shuffler_module.rmm.mr,
        "set_current_device_resource",
        lambda mr: events.append(("current_memory_resource", mr)),
    )
    monkeypatch.setattr(shuffler_module, "BufferResource", FakeBufferResource)
    monkeypatch.setattr(RapidsMPFActor, "__init__", fake_base_init)

    actor = GpuLanceRapidsMPFShuffler(
        nranks=2,
        total_nparts=4,
        shuffle_on=["key"],
        rmm_pool_size=1024,
        spill_memory_limit=None,
        enable_statistics=True,
    )

    assert events == [
        ("current_memory_resource", memory_resource),
        "buffer_resource",
        "base_actor",
    ]
    assert actor.shuffler is None
    assert actor.br.statistics is actor.statistics


def test_setup_worker_constructs_the_2606_shuffler_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    actor = _uninitialized_actor()
    actor.total_nparts = 17
    actor.br = object()
    setup_addresses: list[bytes] = []
    constructor_calls: list[tuple[object, int, int, object]] = []
    created_shuffler = object()

    def fake_setup_worker(_self: RapidsMPFActor, root_address: bytes) -> None:
        setup_addresses.append(root_address)

    def fake_shuffler(
        comm: object,
        operation_id: int,
        *,
        total_num_partitions: int,
        br: object,
    ) -> object:
        constructor_calls.append((comm, operation_id, total_num_partitions, br))
        return created_shuffler

    monkeypatch.setattr(RapidsMPFActor, "setup_worker", fake_setup_worker)
    monkeypatch.setattr(shuffler_module, "Shuffler", fake_shuffler)

    actor.setup_worker(b"root-address")

    assert setup_addresses == [b"root-address"]
    assert constructor_calls == [(actor.comm, 0, 17, actor.br)]
    assert actor.shuffler is created_shuffler


def test_insert_finished_is_one_global_signal() -> None:
    actor = _uninitialized_actor()
    calls = 0

    class FakeShuffler:
        def insert_finished(self) -> None:
            nonlocal calls
            calls += 1

    actor.shuffler = FakeShuffler()

    actor.insert_finished()

    assert calls == 1
    assert actor.comm.logger.messages == ["Insert finished"]


def test_cleanup_is_safe_before_communicator_setup() -> None:
    actor = object.__new__(GpuLanceRapidsMPFShuffler)
    actor.enable_statistics = True
    actor.mr = object()
    actor.shuffler = None
    actor._comm = None
    actor._rank = -1

    actor.cleanup()

    assert actor.shuffler is None


def test_gpu_lance_actor_uses_2606_completion_and_extraction_methods() -> None:
    implementation = gpu_lance_actor_module._actor_implementation()

    assert issubclass(implementation, GpuLanceRapidsMPFShuffler)
    extraction_names = set(implementation._extract_from.__code__.co_names)
    completion_names = set(implementation._finish_return_shuffle.__code__.co_names)
    assert {"wait", "local_partitions", "extract"} <= extraction_names
    assert "wait_any" not in extraction_names
    assert "insert_finished" in completion_names


def test_gpu_lance_return_shuffle_sends_one_global_completion_signal() -> None:
    implementation = gpu_lance_actor_module._actor_implementation()
    actor = object.__new__(implementation)
    completion_calls = 0

    class FakeShuffler:
        def insert_finished(self) -> None:
            nonlocal completion_calls
            completion_calls += 1

    actor._rank = 0
    actor._return_shuffler = FakeShuffler()
    actor._extract_from = lambda _shuffler: iter(())

    with pytest.raises(RuntimeError, match="expected only its return partition"):
        actor._finish_return_shuffle()

    assert completion_calls == 1


def test_gpu_lance_cleanup_retries_failed_shutdown_before_marking_cleaned() -> None:
    implementation = gpu_lance_actor_module._actor_implementation()
    actor = object.__new__(implementation)

    class FakeExecutor:
        calls = 0

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait is True
            assert cancel_futures is True
            self.calls += 1

    class FailOnceShuffler:
        calls = 0

        def shutdown(self) -> None:
            self.calls += 1
            if self.calls == 1:
                msg = "return shuffle shutdown failed"
                raise RuntimeError(msg)

    class FakeShuffler:
        calls = 0

        def shutdown(self) -> None:
            self.calls += 1

    payload_executor = FakeExecutor()
    return_shuffler = FailOnceShuffler()
    request_shuffler = FakeShuffler()
    actor._cleaned = False
    actor._indexes = {0: object()}
    actor._origins = {0: object()}
    actor._document_datasets = {("documents", 1): object()}
    actor._payload_executor = payload_executor
    actor._image_dataset = object()
    actor._return_shuffler = return_shuffler
    actor.shuffler = request_shuffler
    actor.enable_statistics = False

    with pytest.raises(RuntimeError, match="return shuffle shutdown failed"):
        actor.cleanup()

    assert actor._cleaned is False
    assert payload_executor.calls == 1
    assert actor._payload_executor is None
    assert return_shuffler.calls == 1
    assert actor._return_shuffler is return_shuffler
    assert request_shuffler.calls == 1
    assert actor.shuffler is None
    assert actor._indexes == {}
    assert actor._origins == {}
    assert actor._document_datasets == {}
    assert actor._image_dataset is None

    actor.cleanup()
    actor.cleanup()

    assert actor._cleaned is True
    assert payload_executor.calls == 1
    assert return_shuffler.calls == 2
    assert request_shuffler.calls == 1
