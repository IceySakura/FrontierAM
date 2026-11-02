from __future__ import annotations

import asyncio

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core.openai_client import ContextLengthExceededError, OpenAICompatibleClient, OpenAIConfig
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.projects.hy_live import agent as agent_module
from nexus.extensions.projects.hy_live import health as health_module
from nexus.extensions.projects.hy_live import hylive_shell as shell_module
from nexus.extensions.projects.hy_live.hylive_shell import LiveShell
from nexus.extensions.projects.hy_live.trajectory_processor import MainTrajectoryProcessor
from nexus.trace import Span, SpanEndEvent, SpanProcessor, SpanStartEvent, processor_context


class TracedPolicyClient(OpenAICompatibleClient):
    def __init__(self, responses: list[dict[str, Any] | Exception], *, client_key: str = 'main') -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self._client_key = client_key
        self.responses = list(responses)
        self.requests: list[list[dict[str, Any]]] = []

    async def _create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        self.requests.append(deepcopy(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ExtendedChatCompletion.model_validate({
            'id': 'test-response',
            'created': 1,
            'model': 'test-model',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': response}],
        })


async def test_processor_captures_original_response_and_isolates_snapshots() -> None:
    response = {
        'role': 'assistant',
        'content': '  reply\n',
        'reasoning_content': 'original reasoning',
        'tool_calls': [{'id': 'bad-call', 'type': 'function', 'function': {'name': 'unknown', 'arguments': '{broken'}}],
    }
    client = TracedPolicyClient([response])
    capture = MainTrajectoryProcessor(client_key='main')
    messages = [{'role': 'system', 'content': 'base'}, {'role': 'user', 'content': 'question'}]
    with processor_context([capture]):
        assert capture.snapshot_result() == [], 'No request must not invent a trajectory'
        returned = await client.create_completion(messages)
        expected = [*deepcopy(messages), returned.choices[0].message.model_dump()]
        assert capture.snapshot_result() == expected, 'Capture raw output before the agent can reject or edit it'
        returned.choices[0].message.content = 'rewritten'
        returned.choices[0].message.tool_calls = None
        messages[0]['content'] = 'rewritten system'
        snapshot = capture.snapshot_result()
        snapshot[0]['content'] = 'consumer mutation'
        assert capture.snapshot_result() == expected, 'Snapshots must not alias either input, response or caller data'
    assert capture.get_result() == expected, 'Finalization must preserve the last captured call'


@pytest.mark.parametrize('client_key', ['external', 'user_simulator', 'other-main'])
async def test_processor_ignores_other_clients(client_key: str) -> None:
    capture = MainTrajectoryProcessor(client_key='main')
    client = TracedPolicyClient([{'role': 'assistant', 'content': 'not policy'}], client_key=client_key)
    with processor_context([capture]):
        await client.create_completion([{'role': 'user', 'content': 'private sidechain'}])
    assert capture.get_result() == [], 'Never mix backend or simulator histories into the main trajectory'


async def test_processor_uses_latest_actual_request_without_merging_rewritten_branches() -> None:
    client = TracedPolicyClient([
        {'role': 'assistant', 'content': 'first'},
        {'role': 'assistant', 'content': 'last'},
    ])
    capture = MainTrajectoryProcessor(client_key='main')
    with processor_context([capture]):
        await client.create_completion([{'role': 'system', 'content': 'old'}])
        actual_messages = [{'role': 'system', 'content': 'new'}, {'role': 'user', 'content': 'new context'}]
        returned = await client.create_completion(actual_messages)
        assert capture.snapshot_result() == [*actual_messages, returned.choices[0].message.model_dump()], (
            'The message view is the latest real call, not a concatenation of incompatible branches'
        )


async def test_processor_keeps_failed_request_without_fabricating_response() -> None:
    client = TracedPolicyClient([
        {'role': 'assistant', 'content': 'first'},
        RuntimeError('provider failed'),
    ])
    capture = MainTrajectoryProcessor(client_key='main')
    with processor_context([capture]):
        first = await client.create_completion([{'role': 'user', 'content': 'first question'}])
        messages = [
            {'role': 'user', 'content': 'first question'},
            first.choices[0].message.model_dump(),
            {'role': 'user', 'content': 'failing question'},
        ]
        with pytest.raises(RuntimeError, match='provider failed'):
            await client.create_completion(messages)
        assert capture.snapshot_result() == messages, 'Expose the failed request but never invent its assistant answer'


async def test_processors_are_isolated_between_concurrent_runs() -> None:
    async def collect(content: str) -> list[dict[str, Any]]:
        capture = MainTrajectoryProcessor(client_key='main')
        client = TracedPolicyClient([{'role': 'assistant', 'content': content}])
        with processor_context([capture]):
            await asyncio.sleep(0)
            await client.create_completion([{'role': 'user', 'content': content}])
            await asyncio.sleep(0)
            return capture.snapshot_result()

    first, second = await asyncio.gather(collect('first'), collect('second'))
    assert [message['content'] for message in first] == ['first', 'first'], 'Do not collect another run using main'
    assert [message['content'] for message in second] == ['second', 'second'], 'Each run owns its collector'


class WarmRuntime:
    async def run_command(self, command: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(return_code=0, stdout='', stderr='', status='done')

    def on(self, event: str, handler: Any) -> None:
        return None


def _demo_context(client: Any) -> SimpleNamespace:
    async def get_runtime() -> Any:
        return WarmRuntime()

    return SimpleNamespace(run_id='test', get_openai_client=lambda name: client, get_runtime=get_runtime)


@pytest.mark.parametrize('client_key', ['main', 'custom_policy'])
async def test_agent_output_uses_processor_not_loop_messages(
    monkeypatch: pytest.MonkeyPatch,
    client_key: str,
) -> None:
    client = TracedPolicyClient(
        [{'role': 'assistant', 'content': 'original', 'reasoning_content': 'raw reasoning'}], client_key=client_key
    )
    monkeypatch.setattr(agent_module, 'require_context', lambda: _demo_context(client))
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: None)
    original_dump = shell_module._dump_message

    def rewrite_message(message: Any) -> dict[str, Any]:
        dumped = original_dump(message)
        dumped['content'] = 'agent rewritten'
        return dumped

    monkeypatch.setattr(shell_module, '_dump_message', rewrite_message)
    result = await agent_module.live_agent(
        agent_module.LiveAgentAgentInput(
            prompt='question',
            policy_client_name=client_key,
        )
    )
    assert result.summary == 'agent rewritten', 'The agent may process the response for its own control flow'
    assert result.main_trajectory[-1]['content'] == 'original', 'Output must come from the observed completion'
    assert result.main_trajectory[-1]['reasoning_content'] == 'raw reasoning', 'Retain model reasoning'


class PhaseRecorder(SpanProcessor):
    """Record task-phase spans and the policy calls, in the order they are observed."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.phase_durations: dict[str, float] = {}

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        if span.name == 'openai_completion':
            self.events.append('policy_call_started')

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.attributes.get('type') != 'task_phase':
            return
        self.events.append(f'phase_ended:{span.name}')
        self.phase_durations[span.name] = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9


async def test_warm_up_is_reported_as_the_sandbox_init_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TracedPolicyClient([{'role': 'assistant', 'content': 'answer'}])
    monkeypatch.setattr(agent_module, 'require_context', lambda: _demo_context(client))
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: None)
    recorder = PhaseRecorder()

    with processor_context([recorder]):
        await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='question'))

    # hytuner reads this phase as sandbox_init_time; bringing the sandbox up is the only episode
    # time that falls outside the conversation trace, so without the phase it is unattributed.
    assert 'task.init.get_runtime' in recorder.phase_durations
    assert recorder.events[0] == 'phase_ended:task.init.get_runtime', (
        'The sandbox must be up before the policy speaks, so the phase has to close first'
    )


async def test_context_without_a_sandbox_runtime_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TracedPolicyClient([])
    context = SimpleNamespace(run_id='test', get_openai_client=lambda name: client)
    monkeypatch.setattr(agent_module, 'require_context', lambda: context)
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: None)
    with pytest.raises(RuntimeError, match='get_runtime'):
        await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='question'))
    assert client.requests == [], 'Fail before the policy speaks instead of running a chain without a backend'


def _install_policy(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    client = TracedPolicyClient(responses)
    span_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(agent_module, 'require_context', lambda: _demo_context(client))
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(
        agent_module,
        'update_span',
        lambda span_id, *, attributes: span_updates.append({'span_id': span_id, 'attributes': attributes}),
    )
    return span_updates


async def test_an_episode_that_escapes_sets_the_unhandled_termination_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    span_updates = _install_policy(monkeypatch, [])

    async def escape(_shell: LiveShell, **_kwargs: Any) -> Any:
        raise RuntimeError('the shell broke')

    monkeypatch.setattr(LiveShell, 'run', escape)
    with pytest.raises(RuntimeError, match='the shell broke'):
        await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='question'))
    assert span_updates[-1]['attributes']['hytuner.termination_reason'] == 'UNHANDLED_ERROR'


async def test_a_cerebellum_that_keeps_failing_ends_the_episode_as_an_environment_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cerebellum hands one failed call back to itself as ``[失败]``; a second failure ends the episode."""
    span_updates = _install_policy(monkeypatch, [RuntimeError('provider failed'), RuntimeError('provider failed')])

    result = await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='question'))

    assert result.episode_termination == 'ENV_ERROR'
    assert span_updates[-1]['attributes']['hytuner.termination_reason'] == 'ENV_ERROR'


async def test_format_violation_ends_as_a_scoreable_termination(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {'role': 'assistant', 'content': '<tool_call>delegate</tool_call>', 'reasoning_content': 'raw reasoning'}
    span_updates = _install_policy(monkeypatch, [raw])

    result = await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='question'))

    assert result.episode_termination == health_module.FORMAT_ERROR_TERMINATION
    assert span_updates[-1]['attributes']['hytuner.termination_reason'] == health_module.FORMAT_ERROR_TERMINATION, (
        'A policy contract violation must reach the trainer as scoreable, not as a masked UNHANDLED_ERROR'
    )
    assert result.main_trajectory[-1]['content'] == raw['content'], 'The sampled response stays in the trajectory'


async def test_cerebellum_context_overflow_ends_as_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    span_updates = _install_policy(monkeypatch, [ContextLengthExceededError('prompt too long')])

    result = await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='question'))

    assert result.episode_termination == health_module.TRUNCATION_TERMINATION
    assert span_updates[-1]['attributes']['hytuner.termination_reason'] == 'TRUNCATION'


@pytest.mark.parametrize('cancel', [False, True])
async def test_span_failure_does_not_replace_original_exception_or_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cancel: bool,
) -> None:
    client = TracedPolicyClient([])
    original_error = asyncio.CancelledError('cancelled') if cancel else RuntimeError('original failure')

    async def fail_episode(_shell: LiveShell, **kwargs: Any) -> None:
        raise original_error

    def fail_span_update(*args: Any, **kwargs: Any) -> None:
        raise ValueError('span update failed')

    monkeypatch.setattr(agent_module, 'require_context', lambda: _demo_context(client))
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(agent_module, 'update_span', fail_span_update)
    monkeypatch.setattr(LiveShell, 'run', fail_episode)
    with pytest.raises(type(original_error)) as error:
        await agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='question'))
    assert error.value is original_error
    assert 'Failed to record interrupted cerebellum termination' in caplog.text


async def test_processor_preserves_start_snapshot_when_client_mutates_request() -> None:
    capture = MainTrajectoryProcessor(client_key='main')

    class MutatingPolicyClient(TracedPolicyClient):
        async def _create_completion(
            self,
            messages: list[dict[str, Any]],
            model: str | None = None,
            **kwargs: Any,
        ) -> ExtendedChatCompletion:
            assert capture.snapshot_result() == messages, 'Capture the request before client execution'
            messages[0]['content'] = 'mutated inside client'
            return await super()._create_completion(messages, model, **kwargs)

    client = MutatingPolicyClient([{'role': 'assistant', 'content': 'answer'}])
    messages = [{'role': 'user', 'content': 'original request'}]
    with processor_context([capture]):
        await client.create_completion(messages)
    assert capture.get_result()[0]['content'] == 'original request', 'Use on_start, not post-call mutable inputs'


async def test_agent_cancelled_during_completion_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    unblock = asyncio.Event()

    class WaitingPolicyClient(TracedPolicyClient):
        async def _create_completion(
            self,
            messages: list[dict[str, Any]],
            model: str | None = None,
            **kwargs: Any,
        ) -> ExtendedChatCompletion:
            started.set()
            await unblock.wait()
            raise AssertionError('The test must cancel the pending completion')

    client = WaitingPolicyClient([])
    monkeypatch.setattr(agent_module, 'require_context', lambda: _demo_context(client))
    monkeypatch.setattr(agent_module, 'get_current_span_id', lambda: None)
    pending = asyncio.create_task(agent_module.live_agent(agent_module.LiveAgentAgentInput(prompt='pending question')))
    await asyncio.wait_for(started.wait(), timeout=2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
