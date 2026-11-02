import asyncio

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import nexus.runner.hytuner as hytuner
import nexus.runner.single as single

from nexus.gateway.session_manager import session_manager
from nexus.gateway.trajectory import Chain, Trajectory, TrajectoryNode
from nexus.gateway.trajectory_store import InMemoryTrajectoryRecordStore


def _store(namespace: str = 'runner-test') -> InMemoryTrajectoryRecordStore:
    return InMemoryTrajectoryRecordStore(namespace)


def _trajectory() -> Trajectory:
    assistant = {'role': 'assistant', 'content': 'ok'}
    return Trajectory(
        session_id='7_uid_0_0',
        nodes={
            0: TrajectoryNode(
                node_id=0,
                message_id=0,
                parent=None,
                generated_by='llm',
                children=[],
                original_message=assistant,
            )
        },
        roots=[0],
        chains={'main': Chain(chain_id='main', roots=[0])},
    )


class FakeSessionManager:
    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory
        self.active = True
        self.end_calls: list[tuple[str, str | None, dict[str, Any]]] = []
        self.abort_calls: list[str] = []
        self.publications_finished = False
        self.last_main_call_truncated = False

    def get_session(self, _trajectory_id: str):
        return self if self.active else None

    async def finish_pending_publications(self) -> None:
        self.publications_finished = True

    def has_session(self, _trajectory_id: str) -> bool:
        return self.active

    async def abort_in_flight(self, trajectory_id: str) -> None:
        self.abort_calls.append(trajectory_id)

    def end_session(self, trajectory_id: str, termination_reason: str | None = None, **facts: Any) -> Trajectory:
        assert self.publications_finished
        self.end_calls.append((trajectory_id, termination_reason, facts))
        self.active = False
        self.trajectory.state = 'completed'
        self.trajectory.termination_reason = termination_reason
        return self.trajectory


def _use_session_manager(monkeypatch: pytest.MonkeyPatch, fake: FakeSessionManager) -> None:
    monkeypatch.setattr(session_manager, 'has_session', fake.has_session)
    monkeypatch.setattr(session_manager, 'get_session', fake.get_session)
    monkeypatch.setattr(session_manager, 'abort_in_flight', fake.abort_in_flight)
    monkeypatch.setattr(session_manager, 'end_session', fake.end_session)
    monkeypatch.setattr(
        hytuner,
        'get_agent_gateway_scaffold_policy',
        lambda _agent_name, configured_policy=None: configured_policy or 'nexus_main_agent',
    )


class TestLegacyHyTunerContract:
    async def test_run_defaults_to_non_gateway_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        imported_modules: list[str] = []

        async def fake_single_run(run_input: hytuner.HyTunerRunInput) -> single.RunOutput:
            assert run_input.openai_clients['main']['name'] == 'hytuner'
            return single.RunOutput(
                processor_results={
                    'hytuner': {
                        'prompt_ids': [90],
                        'response_ids': [10, 11],
                        'response_mask': [1, 1],
                    }
                }
            )

        monkeypatch.setattr(hytuner.single, 'run', fake_single_run)
        monkeypatch.setattr(hytuner.Registry, 'import_module', imported_modules.append)

        output = await hytuner.run(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            validate=False,
        )

        assert isinstance(output, hytuner.HyTunerRunOutput)
        assert output.segments[0].prompt_ids == [90]
        assert output.segments[0].response_ids == [10, 11]
        assert imported_modules == [
            'nexus.trace.processors.hytuner',
            'nexus.extensions.openai_clients.hytuner',
        ]

    async def test_non_gateway_applies_per_input_binary_reward_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def fake_single_run(_run_input: hytuner.HyTunerRunInput) -> single.RunOutput:
            return single.RunOutput(
                task_output=SimpleNamespace(score=0.75),
                processor_results={
                    'hytuner': {
                        'prompt_ids': [90],
                        'response_ids': [10],
                        'response_mask': [1],
                    }
                },
            )

        monkeypatch.setattr(hytuner.single, 'run', fake_single_run)
        monkeypatch.setattr(hytuner.Registry, 'import_module', lambda _module: None)

        output = await hytuner.run(
            hytuner.HyTunerRunInput(
                task_name='task',
                task_config={
                    'hytuner_reward_policy': {
                        'mode': 'binary',
                        'threshold': 1.0,
                    }
                },
            ),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            validate=False,
        )

        assert isinstance(output, hytuner.HyTunerRunOutput)
        assert output.reward_score == 0.0

    async def test_failed_legacy_run_preserves_task_termination_reason(self, monkeypatch):
        async def fake_single_run(_run_input):
            return single.RunOutput(
                success=False,
                error=single.ErrorInfo(type='RuntimeError', message='verifier failed'),
                processor_results={
                    'hytuner': {
                        'prompt_ids': [90],
                        'response_ids': [10, 11],
                        'response_mask': [1, 1],
                        'termination_reason': 'REWARD_ERROR',
                    }
                },
            )

        monkeypatch.setattr(hytuner.single, 'run', fake_single_run)
        monkeypatch.setattr(hytuner.Registry, 'import_module', lambda _module: None)

        output = await hytuner.run(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            validate=False,
        )

        assert output.success is False
        assert output.termination_reason == 'REWARD_ERROR'

    @pytest.mark.parametrize('use_gateway', [False, True])
    async def test_runner_forwards_token_budget_include(self, monkeypatch, use_gateway):
        captured = {}

        async def fake_single_run(run_input, hooks=None):
            del hooks
            captured.update(run_input.openai_clients['main']['config']['client_args'])
            return single.RunOutput()

        monkeypatch.setattr(hytuner.single, 'run', fake_single_run)
        monkeypatch.setattr(hytuner.Registry, 'import_module', lambda _name: None)
        monkeypatch.setattr(
            hytuner,
            'get_agent_gateway_scaffold_policy',
            lambda _agent_name, configured_policy=None: configured_policy or 'nexus_main_agent',
        )

        include = ['agent_output', 'compaction_output']
        if use_gateway:
            run_input = hytuner.HyTunerRunInput(task_name='task', token_budget_config={'token_budget_include': include})
            path_kwargs = {
                'tokenizer': object(),
                'server_manager': object(),
                'use_gateway': True,
                'record_store': _store(),
                'trajectory_id': '7_uid_0_0',
            }
        else:
            run_input = hytuner.HyTunerRunInput(task_name='task')
            path_kwargs = {
                'tokenizer': None,
                'server_manager': None,
                'context_management': {'token_budget_include': include},
            }
        await hytuner.run(
            run_input,
            sampling_params={},
            tool_parser=None,
            validate=False,
            **path_kwargs,
        )

        assert captured['token_budget_include'] == ['agent_output', 'compaction_output']
        assert 'max_prompt_length' not in captured

    async def test_non_gateway_keeps_segment_routing_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        routes = np.array([[1, 2]])

        async def fake_run(_run_input: hytuner.HyTunerRunInput) -> single.RunOutput:
            return single.RunOutput(
                processor_results={
                    'hytuner': {
                        'segments': [
                            {
                                'prompt_ids': [10],
                                'raw_input_ids': [10, 20],
                                'response_ids': [20],
                                'response_mask': [1],
                                'sidechain_info': {'kind': 'worker'},
                                'routed_experts': routes,
                            }
                        ]
                    }
                }
            )

        monkeypatch.setattr(hytuner.single, 'run', fake_run)
        monkeypatch.setattr(hytuner.Registry, 'import_module', lambda _name: None)

        output = await hytuner.run(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            validate=False,
        )

        assert output.segments[0].sidechain_info == {'kind': 'worker'}
        np.testing.assert_array_equal(output.segments[0].routed_experts, routes)


class TestGatewayHyTunerContract:
    async def test_gateway_executor_requires_trajectory_id(self):
        with pytest.raises(ValueError, match='trajectory_id'):
            await hytuner._execute_gateway(
                hytuner.HyTunerRunInput(task_name='task'),
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                record_store=_store(),
            )

    async def test_run_returns_canonical_trajectory_with_execution_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trajectory = _trajectory()
        stats = [{'input_tokens': 100, 'output_tokens': 20}]

        async def fake_execute(run_input: hytuner.HyTunerRunInput, **_kwargs: Any):
            run_input.compaction_manager = SimpleNamespace(compaction_stats=stats)
            return trajectory, single.RunOutput(
                success=False,
                error=single.ErrorInfo(type='RuntimeError', message='task failed'),
            )

        monkeypatch.setattr(hytuner, '_execute_gateway', fake_execute)

        output = await hytuner.run(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            use_gateway=True,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
        )

        assert output is trajectory
        assert output.extra['hytuner'] == {
            'success': False,
            'request_id': trajectory.session_id,
            'error_type': 'RuntimeError',
            'error_message': 'task failed',
        }
        assert output.extra['compaction_stats'] == stats
        stats[0]['input_tokens'] = 999
        assert output.extra['compaction_stats'][0]['input_tokens'] == 100

    @pytest.mark.parametrize('agent_name', ['hycode', 'hyness'])
    async def test_scoreable_hycode_exit_aborts_orphan_without_losing_reward(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agent_name: str,
    ) -> None:
        class PendingSessionManager(FakeSessionManager):
            def __init__(self, trajectory: Trajectory):
                super().__init__(trajectory)
                self.in_flight = True

            async def abort_in_flight(self, trajectory_id: str) -> None:
                await super().abort_in_flight(trajectory_id)
                self.in_flight = False

            async def finish_pending_publications(self) -> None:
                if self.in_flight:
                    raise RuntimeError('session still has an in-flight request')
                await super().finish_pending_publications()

        trajectory = _trajectory()
        gateway = PendingSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)

        async def fake_run(_run_input: Any, hooks: single.RunHooks) -> single.RunOutput:
            result = single.RunOutput(
                task_output=SimpleNamespace(score=0.75),
                agent_output=SimpleNamespace(
                    termination_reason='ENV_DONE',
                    scoreable_process_exit=True,
                    hycode_exit_name='SIGKILL',
                    exit_code=-9,
                ),
            )
            context = SimpleNamespace(get_runtime_id=lambda _name: None)
            await single._invoke_post_run_hook(hooks, 'run_output', hooks.on_run_output, context, result)
            await single._invoke_post_run_hook(hooks, 'context_closing', hooks.on_context_closing, context)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)

        output, _ = await hytuner._execute_gateway(
            hytuner.HyTunerRunInput(task_name='task', agent_name=agent_name),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
        )

        assert gateway.abort_calls == [trajectory.session_id]
        assert gateway.end_calls[0][1] == 'ENV_DONE'
        assert output.termination_reason == 'ENV_DONE'
        assert output.final_reward == 0.75
        assert output.extra['hycode_exit_name'] == 'SIGKILL'
        assert output.extra['hycode_exit_code'] == -9

    async def test_gateway_applies_per_input_binary_reward_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trajectory = _trajectory()
        gateway = FakeSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)

        async def fake_run(_run_input: Any, hooks: single.RunHooks) -> single.RunOutput:
            result = single.RunOutput(task_output=SimpleNamespace(score=0.75))
            context = SimpleNamespace(get_runtime_id=lambda _name: None)
            await single._invoke_post_run_hook(hooks, 'run_output', hooks.on_run_output, context, result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)

        output, _ = await hytuner._execute_gateway(
            hytuner.HyTunerRunInput(
                task_name='task',
                task_config={
                    'hytuner_reward_policy': {
                        'mode': 'binary',
                        'threshold': 1.0,
                    }
                },
            ),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
        )

        assert output.final_reward == 0.0

    async def test_failed_gateway_run_preserves_task_termination_reason(self, monkeypatch):
        trajectory = _trajectory()
        gateway = FakeSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)

        async def fake_run(_run_input, hooks=None):
            hooks.span_processors[0].termination_reason = 'REWARD_ERROR'
            result = single.RunOutput(
                success=False,
                error=single.ErrorInfo(type='RuntimeError', message='verifier failed'),
            )
            await hooks.on_run_output(SimpleNamespace(get_runtime_id=lambda _name: None), result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)

        output = await hytuner.run(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            use_gateway=True,
            record_store=_store(),
            trajectory_id='7_uid_0_0',
        )

        assert output.termination_reason == 'REWARD_ERROR'
        assert gateway.end_calls[0][1] == 'REWARD_ERROR'

    async def test_gateway_rejects_reserved_hytuner_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_run(_run_input, hooks=None):
            result = single.RunOutput()
            await hooks.on_run_output(SimpleNamespace(get_runtime_id=lambda _name: None), result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)

        collision = _trajectory()
        collision.extra['hytuner'] = {'owner': 'task'}
        _use_session_manager(monkeypatch, FakeSessionManager(collision))
        with pytest.raises(ValueError, match=r"trajectory\.extra\['hytuner'\] is reserved"):
            await hytuner.run(
                hytuner.HyTunerRunInput(task_name='task'),
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                use_gateway=True,
                record_store=_store(),
                trajectory_id='7_uid_0_0',
            )

    async def test_gateway_requires_injected_store(self) -> None:
        with pytest.raises(ValueError, match='record_store'):
            await hytuner.run(
                hytuner.HyTunerRunInput(task_name='task'),
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                use_gateway=True,
                trajectory_id='7_uid_0_0',
            )

    @pytest.mark.parametrize('validate', [False, True])
    async def test_public_run_passes_gateway_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        validate: bool,
    ) -> None:
        captured: dict[str, Any] = {}
        record_store = _store()

        async def fake_gateway(run_input: hytuner.HyTunerRunInput, **kwargs: Any):
            captured['task_config'] = dict(run_input.task_config)
            captured.update(kwargs)
            return _trajectory(), single.RunOutput()

        monkeypatch.setattr(hytuner, '_execute_gateway', fake_gateway)
        await hytuner.run(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer='tokenizer',
            server_manager='server',
            sampling_params={},
            tool_parser='parser',
            use_gateway=True,
            record_store=record_store,
            trajectory_id='7_uid_0_0',
            validate=validate,
        )

        assert captured['tokenizer'] == 'tokenizer'
        assert captured['server_manager'] == 'server'
        assert captured['tool_parser'] == 'parser'
        assert captured['record_store'] is record_store
        assert captured['task_config']['is_validation'] is validate

    @pytest.mark.parametrize(
        ('agent_name', 'expected_policy'),
        [('main_agent', 'nexus_main_agent'), ('hycode', 'hycode')],
    )
    async def test_execute_gateway_configures_canonical_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agent_name: str,
        expected_policy: str,
    ) -> None:
        trajectory = _trajectory()
        gateway = FakeSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)
        captured: dict[str, Any] = {}
        record_store = _store()
        monkeypatch.setattr(
            hytuner,
            'get_agent_gateway_scaffold_policy',
            lambda agent_name, configured_policy=None: (
                configured_policy or {'main_agent': 'nexus_main_agent', 'hycode': 'hycode'}[agent_name]
            ),
        )

        async def fake_run(run_input: hytuner.HyTunerRunInput, hooks: single.RunHooks | None = None):
            captured.update(run_input.openai_clients['main']['config']['client_args'])
            result = single.RunOutput()
            await hooks.on_run_output(SimpleNamespace(get_runtime_id=lambda _name: None), result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)

        output, _ = await hytuner._execute_gateway(
            hytuner.HyTunerRunInput(task_name='task', agent_name=agent_name),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            record_store=record_store,
            trajectory_id=trajectory.session_id,
        )

        assert output is trajectory
        assert captured['trajectory_id'] == trajectory.session_id
        assert captured['record_store'] is record_store
        assert captured['scaffold_policy'] == expected_policy
        assert gateway.end_calls[0][1] == 'ENV_DONE'

    async def test_execute_gateway_waits_for_children_before_finalization(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trajectory = _trajectory()
        gateway = FakeSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)
        completion_order: list[str] = []

        async def child(agent_id: str, delay: float) -> None:
            await asyncio.sleep(delay)
            trajectory.chains[agent_id] = Chain(chain_id=agent_id, kind='subagent', parent_chain_id='main')
            completion_order.append(agent_id)

        async def fake_run(_run_input: Any, hooks: single.RunHooks | None = None):
            await asyncio.gather(child('agent-1', 0.01), child('agent-2', 0))
            result = single.RunOutput()
            await hooks.on_run_output(SimpleNamespace(get_runtime_id=lambda _name: None), result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)

        result, _ = await hytuner._execute_gateway(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
        )

        assert completion_order == ['agent-2', 'agent-1']
        assert list(result.chains) == ['main', 'agent-2', 'agent-1']

    async def test_unexpected_exception_finalizes_the_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gateway = FakeSessionManager(_trajectory())
        _use_session_manager(monkeypatch, gateway)

        async def fail_run(_run_input: Any, hooks: Any = None):
            del hooks
            raise RuntimeError('boom')

        monkeypatch.setattr(hytuner.single, 'run', fail_run)

        with pytest.raises(RuntimeError, match='boom'):
            await hytuner._execute_gateway(
                hytuner.HyTunerRunInput(task_name='task'),
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                record_store=_store(),
                trajectory_id='7_uid_0_0',
            )

        assert gateway.abort_calls == ['7_uid_0_0']
        assert gateway.end_calls[0][1] == 'UNHANDLED_ERROR'

    async def test_successful_run_still_cancels_orphaned_completions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An agent that stops itself mid-request still yields a task output, so the run
        # resolves to ENV_DONE rather than to any failure reason.
        trajectory = _trajectory()
        gateway = FakeSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)

        async def fake_run(_run_input: Any, hooks: single.RunHooks | None = None):
            result = single.RunOutput()
            assert hooks is not None
            await hooks.on_run_output(SimpleNamespace(get_runtime_id=lambda _name: None), result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)

        result, _ = await hytuner._execute_gateway(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
        )

        assert gateway.abort_calls == [trajectory.session_id]
        assert gateway.end_calls[0][1] == 'ENV_DONE'
        assert result is not None

    async def test_cleanup_exception_after_finalization_returns_trajectory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trajectory = _trajectory()
        gateway = FakeSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)
        result = single.RunOutput()

        async def fail_during_context_cleanup(_run_input: Any, hooks: single.RunHooks):
            context = SimpleNamespace(get_runtime_id=lambda _name: None)
            await hooks.on_run_output(context, result)
            raise ConnectionError('sandbox stop timed out during DNS resolution')

        monkeypatch.setattr(hytuner.single, 'run', fail_during_context_cleanup)

        finalized, returned_result = await hytuner._execute_gateway(
            hytuner.HyTunerRunInput(task_name='task'),
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
        )

        assert finalized is trajectory
        assert returned_result is result
        assert gateway.end_calls[0][1] == 'ENV_DONE'

    async def test_publication_failure_is_not_swallowed_by_run_hooks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gateway = FakeSessionManager(_trajectory())
        _use_session_manager(monkeypatch, gateway)

        async def fail_publications() -> None:
            raise RuntimeError('store unavailable')

        async def fake_run(_run_input, hooks):
            result = single.RunOutput()
            context = SimpleNamespace(get_runtime_id=lambda _name: None)
            await single._invoke_post_run_hook(hooks, 'run_output', hooks.on_run_output, context, result)
            await single._invoke_post_run_hook(hooks, 'context_closing', hooks.on_context_closing, context)
            return result

        monkeypatch.setattr(gateway, 'finish_pending_publications', fail_publications)
        monkeypatch.setattr(hytuner.single, 'run', fake_run)
        with pytest.raises(RuntimeError, match='store unavailable'):
            await hytuner._execute_gateway(
                hytuner.HyTunerRunInput(task_name='task'),
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                record_store=_store(),
                trajectory_id='7_uid_0_0',
            )
        assert not gateway.end_calls

    async def test_gateway_rejects_context_management_kwarg(self) -> None:
        with pytest.raises(ValueError, match='does not take context_management'):
            await hytuner._execute_gateway(
                hytuner.HyTunerRunInput(task_name='task', agent_name='main_agent'),
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                record_store=_store(),
                trajectory_id='7_uid_0_0',
                context_management={'token_budget': 10},
            )

    async def test_gateway_rejects_segment_shaping(self) -> None:
        with pytest.raises(NotImplementedError, match='segment_shaping_config'):
            await hytuner._execute_gateway(
                hytuner.HyTunerRunInput(task_name='task'),
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                record_store=_store(),
                trajectory_id='7_uid_0_0',
                segment_shaping_config={'policy': 'tool_error_turn_penalty_only'},
            )

    async def test_gateway_reads_row_token_budget_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        trajectory = _trajectory()
        _use_session_manager(monkeypatch, FakeSessionManager(trajectory))
        captured: dict[str, Any] = {}

        async def fake_run(run_input: hytuner.HyTunerRunInput, hooks: single.RunHooks | None = None):
            captured.update(run_input.openai_clients['main']['config']['client_args'])
            result = single.RunOutput()
            await hooks.on_run_output(SimpleNamespace(get_runtime_id=lambda _name: None), result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)
        run_input = hytuner.HyTunerRunInput(
            task_name='task',
            token_budget_config={'token_budget': 50, 'token_budget_include': ['agent_output']},
        )

        output = await hytuner.run(
            run_input,
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            use_gateway=True,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
        )

        assert captured['token_budget'] == 50
        assert captured['token_budget_include'] == ['agent_output']
        assert output.extra['hytuner']['token_budget'] == 50

    async def test_gateway_rejects_unsupported_main_agent_compaction_options(self) -> None:
        run_input = hytuner.HyTunerRunInput(
            task_name='task',
            agent_name='main_agent',
            agent_config={'compaction': {'inject_trajectory_budget_state': True}},
        )
        with pytest.raises(NotImplementedError, match='agent_config.compaction.inject_trajectory_budget_state'):
            await hytuner._execute_gateway(
                run_input,
                tokenizer=None,
                server_manager=None,
                sampling_params={},
                tool_parser=None,
                record_store=_store(),
                trajectory_id='7_uid_0_0',
            )

    def test_token_budget_config_requires_agent_output(self) -> None:
        with pytest.raises(ValueError, match='agent_output'):
            hytuner.HyTunerRunInput(
                task_name='task', token_budget_config={'token_budget_include': ['tool_observation']}
            )

    async def test_gateway_uses_the_trainer_model_length_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        trajectory = _trajectory()
        gateway = FakeSessionManager(trajectory)
        _use_session_manager(monkeypatch, gateway)
        captured: dict[str, Any] = {}

        async def fake_run(run_input: hytuner.HyTunerRunInput, hooks: single.RunHooks | None = None):
            captured.update(run_input.openai_clients['main']['config']['client_args'])
            result = single.RunOutput()
            await hooks.on_run_output(SimpleNamespace(get_runtime_id=lambda _name: None), result)
            return result

        monkeypatch.setattr(hytuner.single, 'run', fake_run)
        run_input = hytuner.HyTunerRunInput(task_name='task')

        await hytuner._execute_gateway(
            run_input,
            tokenizer=None,
            server_manager=None,
            sampling_params={},
            tool_parser=None,
            record_store=_store(),
            trajectory_id=trajectory.session_id,
            max_model_len=80,
        )

        assert captured['max_model_len'] == 80
