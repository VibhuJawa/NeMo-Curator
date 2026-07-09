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

from unittest import mock

import pytest

from nemo_curator.backends.ray_actor_pool import executor as executor_module
from nemo_curator.backends.ray_actor_pool.executor import RayActorPoolExecutor, _parse_runtime_env
from nemo_curator.backends.ray_actor_pool.utils import calculate_optimal_actors_for_stage
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask


class TestRayActorPoolExecutor:
    def test_parse_runtime_env(self):
        # With noset defined we should override it to be empty
        with_noset_defined = {"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": mock.ANY}}
        assert _parse_runtime_env(with_noset_defined) == {
            "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""}
        }

        # we overwrite when config env_var is not provided
        without_env_var = {"some_other_key": "some_other_value"}
        assert _parse_runtime_env(without_env_var) == {
            "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": ""},
            "some_other_key": "some_other_value",
        }

    @pytest.mark.parametrize(
        ("available_cpus", "expected_actors", "expected_warning"),
        [
            (8.0, 4, None),
            (2.0, 2, "requires 4 actors from num_workers(), but only 2 fit"),
        ],
    )
    def test_calculate_optimal_actors_respects_explicit_num_workers(
        self, available_cpus: float, expected_actors: int, expected_warning: str | None
    ) -> None:
        stage = _stage_with_num_workers(num_workers=4, cpus=1.0, batch_size=10)

        with (
            mock.patch(
                "nemo_curator.backends.ray_actor_pool.utils.get_available_cpu_gpu_resources",
                return_value=(available_cpus, 0.0),
            ),
            mock.patch("nemo_curator.backends.ray_actor_pool.utils.logger.warning") as mock_warning,
        ):
            assert calculate_optimal_actors_for_stage(stage, num_tasks=1) == expected_actors

        if expected_warning is None:
            mock_warning.assert_not_called()
        else:
            mock_warning.assert_called_once()
            assert expected_warning in mock_warning.call_args.args[0]

    def test_rapidsmpf_setup_failure_force_kills_without_teardown(self) -> None:
        executor = RayActorPoolExecutor(show_progress=False)
        stage = mock.Mock(name="shuffle-stage")
        stage.name = "shuffle-stage"
        stage.resources = Resources(cpus=1.0, gpus=1.0)
        actors = [mock.Mock(name="actor-0"), mock.Mock(name="actor-1")]
        actors[0].setup_root.remote.return_value = mock.sentinel.root_ref
        for actor in actors:
            actor.setup.remote.return_value = mock.sentinel.setup_ref
        actor_factory = mock.Mock()
        actor_factory.remote.side_effect = actors

        with (
            mock.patch.object(executor_module.ShuffleStageAdapter, "options", return_value=actor_factory),
            mock.patch.object(
                executor_module.ray,
                "get",
                side_effect=[b"root-address", RuntimeError("worker setup failed")],
            ),
            mock.patch.object(executor_module.ray, "kill") as ray_kill,
            mock.patch.object(executor, "_cleanup_rapidsmpf_actors") as cleanup,
            pytest.raises(RuntimeError, match="worker setup failed"),
        ):
            executor._create_rapidsmpf_actors(stage, num_actors=2, num_tasks=4)

        cleanup.assert_not_called()
        for actor in actors:
            actor.teardown.remote.assert_not_called()
        assert ray_kill.call_args_list == [
            mock.call(actors[0], no_restart=True),
            mock.call(actors[1], no_restart=True),
        ]

    def test_shuffle_window_failure_force_kills_without_teardown(self) -> None:
        executor = RayActorPoolExecutor(show_progress=False)
        stage = mock.Mock(name="shuffle-stage")
        stage.name = "shuffle-stage"
        stage.resources = Resources(cpus=1.0, gpus=1.0)
        stage.ray_stage_spec.return_value = {RayStageSpecKeys.IS_SHUFFLE_STAGE: True}
        stage._shuffle_task_window_size = 4
        actors = [mock.Mock(name="actor-0"), mock.Mock(name="actor-1")]
        tasks = [EmptyTask()]
        events: list[str] = []

        def record_force_kill(actor: mock.Mock, *, no_restart: bool) -> None:
            assert no_restart is True
            events.append(f"force-kill:{actors.index(actor)}")

        with (
            mock.patch.object(executor_module, "register_loguru_serializer"),
            mock.patch.object(executor_module, "execute_setup_on_node"),
            mock.patch.object(executor_module, "calculate_optimal_actors_for_stage", return_value=2),
            mock.patch.object(executor_module.ray, "init"),
            mock.patch.object(executor_module.ray, "shutdown", side_effect=lambda: events.append("ray.shutdown")),
            mock.patch.object(executor, "_create_rapidsmpf_actors", return_value=actors),
            mock.patch.object(
                executor,
                "_process_shuffle_stage_with_rapidsmpf_actors",
                side_effect=RuntimeError("window failed"),
            ),
            mock.patch.object(
                executor, "_cleanup_rapidsmpf_actors", side_effect=lambda _actors: events.append("cleanup")
            ) as cleanup,
            mock.patch.object(
                executor_module.ray,
                "kill",
                side_effect=record_force_kill,
            ) as ray_kill,
            pytest.raises(RuntimeError, match="window failed"),
        ):
            executor.execute([stage], initial_tasks=tasks)

        cleanup.assert_not_called()
        for actor in actors:
            actor.teardown.remote.assert_not_called()
        assert ray_kill.call_args_list == [
            mock.call(actors[0], no_restart=True),
            mock.call(actors[1], no_restart=True),
        ]
        assert events == ["force-kill:0", "force-kill:1", "ray.shutdown"]

    def test_shuffle_success_gracefully_cleans_actors_before_ray_shutdown(self) -> None:
        executor = RayActorPoolExecutor(show_progress=False)
        stage = mock.Mock(name="shuffle-stage")
        stage.name = "shuffle-stage"
        stage.resources = Resources(cpus=1.0, gpus=1.0)
        stage.ray_stage_spec.return_value = {RayStageSpecKeys.IS_SHUFFLE_STAGE: True}
        stage._shuffle_task_window_size = 4
        actors = [mock.sentinel.actor_0, mock.sentinel.actor_1]
        tasks = [EmptyTask()]
        events: list[str] = []

        with (
            mock.patch.object(executor_module, "register_loguru_serializer"),
            mock.patch.object(executor_module, "execute_setup_on_node"),
            mock.patch.object(executor_module, "calculate_optimal_actors_for_stage", return_value=2),
            mock.patch.object(executor_module.ray, "init"),
            mock.patch.object(executor_module.ray, "shutdown", side_effect=lambda: events.append("ray.shutdown")),
            mock.patch.object(executor, "_create_rapidsmpf_actors", return_value=actors),
            mock.patch.object(executor, "_process_shuffle_stage_with_rapidsmpf_actors", return_value=tasks),
            mock.patch.object(
                executor, "_cleanup_rapidsmpf_actors", side_effect=lambda _actors: events.append("cleanup")
            ) as cleanup,
            mock.patch.object(executor, "_force_kill_rapidsmpf_actors") as force_kill,
        ):
            assert executor.execute([stage], initial_tasks=tasks) == tasks

        cleanup.assert_called_once_with(actors)
        force_kill.assert_not_called()
        assert events == ["cleanup", "ray.shutdown"]

    def test_successful_rapidsmpf_cleanup_tears_down_then_force_kills(self) -> None:
        executor = RayActorPoolExecutor(show_progress=False)
        actors = [mock.Mock(name="actor-0"), mock.Mock(name="actor-1")]
        teardown_refs = [mock.sentinel.teardown_0, mock.sentinel.teardown_1]
        for actor, teardown_ref in zip(actors, teardown_refs, strict=True):
            actor.teardown.remote.return_value = teardown_ref

        with (
            mock.patch.object(executor_module.ray, "get") as ray_get,
            mock.patch.object(executor, "_force_kill_rapidsmpf_actors") as force_kill,
        ):
            executor._cleanup_rapidsmpf_actors(actors)

        ray_get.assert_called_once_with(
            teardown_refs,
            timeout=executor_module._ACTOR_TEARDOWN_TIMEOUT_SECONDS,
        )
        for actor in actors:
            actor.teardown.remote.assert_called_once_with()
        force_kill.assert_called_once_with(actors)

    def test_cleanup_force_kills_all_actors_when_graceful_teardown_fails(self) -> None:
        executor = RayActorPoolExecutor(show_progress=False)
        actors = [mock.Mock(name="actor-0"), mock.Mock(name="actor-1")]
        teardown_refs = [mock.sentinel.teardown_0, mock.sentinel.teardown_1]
        for actor, teardown_ref in zip(actors, teardown_refs, strict=True):
            actor.teardown.remote.return_value = teardown_ref

        with (
            mock.patch.object(executor_module.ray, "get", side_effect=RuntimeError("teardown failed")) as ray_get,
            mock.patch.object(executor_module.ray, "kill") as ray_kill,
        ):
            executor._cleanup_rapidsmpf_actors(actors)

        ray_get.assert_called_once_with(
            teardown_refs,
            timeout=executor_module._ACTOR_TEARDOWN_TIMEOUT_SECONDS,
        )
        assert ray_kill.call_args_list == [
            mock.call(actors[0], no_restart=True),
            mock.call(actors[1], no_restart=True),
        ]


def _stage_with_num_workers(*, num_workers: int, cpus: float, batch_size: int) -> mock.Mock:
    stage = mock.Mock()
    stage.name = "stage"
    stage.resources = Resources(cpus=cpus, gpus=0.0)
    stage.batch_size = batch_size
    stage.num_workers.return_value = num_workers
    return stage
