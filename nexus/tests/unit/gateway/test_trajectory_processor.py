import subprocess
import sys

from types import SimpleNamespace

import pytest

from _fakes import memory_store

import nexus.gateway.client as gateway_client
import nexus.gateway.nexus_run_tracker as nexus_run_tracker
import nexus.gateway.trajectory_processor as trajectory_processor

from nexus.core.openai_client import OpenAIConfig
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.session_manager import GatewaySessionManager
from nexus.gateway.trajectory import Trajectory
from nexus.gateway.trajectory_processor import GatewayTrajectoryProcessor
from nexus.runner.single import ProcessorConfig, RunInput, run
from nexus.trace.events import SpanEndEvent
from nexus.trace.types import Span, Status


class StubProcessor:
    is_token_level = False

    async def create_completion(self, **_kwargs):
        return SimpleNamespace(
            message={
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'call-1',
                        'type': 'function',
                        'function': {'name': 'search', 'arguments': '{}'},
                    }
                ],
            },
            prompt_ids=None,
            raw_prompt_ids=None,
            token_ids=None,
            log_probs=None,
            routed_experts=None,
            stop_reason='tool_calls',
            usage=None,
            id=None,
            created=None,
            model=None,
            object=None,
            system_fingerprint=None,
            service_tier=None,
            choice_logprobs=None,
            choice_provider_specific_fields=None,
            extra_fields={},
            completion_text=None,
            prompt_render=None,
        )


def _span(span_type: str, *, outputs=None, termination_reason=None, status='OK') -> Span:
    attributes = {'type': span_type}
    if outputs is not None:
        attributes['outputs'] = outputs
    if termination_reason is not None:
        attributes['agent.termination_reason'] = termination_reason
    return Span(
        trace_id='trace',
        span_id=f'{span_type}-span',
        parent_span_id='',
        name=span_type,
        start_time_unix_nano=1,
        end_time_unix_nano=2,
        attributes=attributes,
        status=Status(code=status),
    )


@pytest.fixture
def manager(monkeypatch):
    value = GatewaySessionManager()
    monkeypatch.setattr(gateway_client, 'session_manager', value)
    monkeypatch.setattr(trajectory_processor, 'session_manager', value)
    monkeypatch.setattr(nexus_run_tracker, 'session_manager', value)
    return value


class TestGatewayTrajectoryProcessor:
    def test_gateway_objects_are_lazily_resolved_after_importing_nexus(self):
        subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import nexus\n'
                    'from nexus.core import openai_client_registry\n'
                    'from nexus.trace.processors.base import span_processor_registry\n'
                    "assert not openai_client_registry.get_entry('gateway').is_loaded\n"
                    "assert not span_processor_registry.get_entry('gateway_trajectory').is_loaded\n"
                    "assert openai_client_registry.get('gateway').__name__ == 'GatewayOpenAIClient'\n"
                    "assert span_processor_registry.get('gateway_trajectory').__name__ "
                    "== 'GatewayTrajectoryProcessor'\n"
                ),
            ],
            check=True,
        )

    async def test_single_runner_returns_the_processor_result(self, manager):
        class BackendClient:
            async def create_completion(self, **_kwargs):
                message = SimpleNamespace(
                    content='done',
                    reasoning_content=None,
                    provider_specific_fields=None,
                    tool_calls=None,
                )
                choice = SimpleNamespace(message=message, finish_reason='stop', logprobs=None)
                return SimpleNamespace(choices=[choice], usage=None, model='fake')

        gateway = GatewayOpenAIClient(
            OpenAIConfig(
                client_args={
                    'openai_client': BackendClient(),
                    'trajectory_id': 'run-1',
                    'record_store': memory_store('run-1'),
                    'scaffold_policy': 'nexus_main_agent',
                },
                request_args={},
            )
        )
        result = await run(
            RunInput(
                run_id='run-1',
                task_name='identity',
                task_config={},
                agent_name='main_agent',
                agent_config={'user_prompt': 'say done', 'tools': [], 'max_iterations': 1},
                openai_clients={'main': {'instance': gateway}},
                processors={
                    'gateway': ProcessorConfig(
                        name='gateway_trajectory',
                        args={'trajectory_id': 'run-1'},
                    )
                },
            )
        )

        assert result.success
        assert result.processor_results['gateway']['session_id'] == 'run-1'
        assert manager.active_count == 0

    async def test_returns_enriched_canonical_trajectory(self, manager):
        session = manager.get_or_create(
            'run-1',
            processor=StubProcessor(),
            record_store=memory_store('run-1'),
            scaffold_policy='generic',
        )
        user = {'role': 'user', 'content': 'question'}
        completion = await session.create_completion([user])
        observation = {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'result'}
        processor = GatewayTrajectoryProcessor(trajectory_id='run-1')

        processor.on_end(
            _span(
                'agent',
                outputs={'main_trajectory': [user, completion.message, observation]},
                termination_reason='MAX_STEPS',
            ),
            SpanEndEvent(),
        )
        processor.on_end(
            _span('task', outputs={'score': 0.75, 'extra_fields': {'benchmark': 'swe'}}),
            SpanEndEvent(),
        )
        processor.finalize()
        await processor.afinalize()

        result = processor.get_result()
        assert result is not None
        trajectory = Trajectory.from_dump(result)
        assert trajectory.termination_reason == 'MAX_STEPS'
        assert trajectory.final_reward == 0.75
        assert trajectory.to_messages() == [user, completion.message, observation]
        assert trajectory.extra['benchmark'] == 'swe'
        assert trajectory.extra['tool_stats']['search']['success'] == 1
        assert not manager.has_session('run-1')

    async def test_successful_run_still_aborts_in_flight_requests(self, manager):
        class Session:
            def __init__(self, trajectory_id):
                self.trajectory_id = trajectory_id
                self.record_store = memory_store(trajectory_id)
                self.aborted = False
                self.last_main_call_truncated = False

            async def abort_in_flight(self):
                self.aborted = True
                return []

            async def finish_pending_publications(self):
                return None

            def end(self, **_kwargs):
                return Trajectory(session_id=self.trajectory_id)

        session = Session('traj')
        manager._sessions = {'traj': session}
        processor = GatewayTrajectoryProcessor(trajectory_id='traj')

        await processor.afinalize()

        assert session.aborted
        assert processor.get_result() is not None

    async def test_sessions_are_isolated_and_failed_finalization_is_removed(self, manager):
        class Session:
            def __init__(self, trajectory_id, error=None):
                self.trajectory_id = trajectory_id
                self.error = error
                self.record_store = memory_store(trajectory_id)
                self.last_main_call_truncated = False

            async def abort_in_flight(self):
                return []

            async def finish_pending_publications(self):
                return None

            def end(self, **_kwargs):
                if self.error is not None:
                    raise self.error
                return Trajectory(session_id=self.trajectory_id)

        manager._sessions = {
            'broken': Session('broken', RuntimeError('materialization failed')),
            'healthy': Session('healthy'),
        }
        broken = GatewayTrajectoryProcessor(trajectory_id='broken')
        healthy = GatewayTrajectoryProcessor(trajectory_id='healthy')

        with pytest.raises(RuntimeError, match='materialization failed'):
            await broken.afinalize()
        await healthy.afinalize()

        assert manager.active_count == 0
        result = healthy.get_result()
        assert result is not None
        assert result['session_id'] == 'healthy'
        assert broken.get_result() is None
