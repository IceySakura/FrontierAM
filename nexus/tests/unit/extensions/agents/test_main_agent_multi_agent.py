"""Behavioral tests for main-agent subagent orchestration."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import agent_registry
from nexus.core.context import ExecutionContext
from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig
from nexus.extensions.agents.main.agent import MainAgentInput, MainExitStatus, run_main_agent
from nexus.extensions.agents.main.multi_agent import (
    AgentStatus,
    MultiAgentConfig,
    SubagentConfig,
    SubagentController,
    SubagentRunFailed,
)


class _FakeMessage:
    def __init__(self, content: str, *, tool_calls: list[Any] | None = None) -> None:
        self.content = content
        self.refusal = None
        self.tool_calls = tool_calls

    def model_dump(self, *, exclude_none: bool) -> dict[str, Any]:
        tool_calls = None
        if self.tool_calls is not None:
            tool_calls = [
                {
                    'id': tool_call.id,
                    'type': 'function',
                    'function': {
                        'name': tool_call.function.name,
                        'arguments': tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]
        message = {'role': 'assistant', 'content': self.content, 'tool_calls': tool_calls}
        return {key: value for key, value in message.items() if not exclude_none or value is not None}


class _BlockingClient(OpenAICompatibleClient):
    """Return sequenced answers while allowing the first inference to be paused."""

    supports_request_meta = True

    def __init__(self) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.first_request_started = asyncio.Event()
        self.release_first_request = asyncio.Event()
        self.requests: list[list[dict[str, Any]]] = []

    async def _create_completion(
        self, messages: list[dict[str, Any]], model: Any = None, **kwargs: Any
    ) -> SimpleNamespace:
        del model, kwargs
        request_index = len(self.requests)
        self.requests.append(deepcopy(messages))
        if request_index == 0:
            self.first_request_started.set()
            await self.release_first_request.wait()
        content = 'draft answer' if request_index == 0 else 'revised answer'
        return SimpleNamespace(choices=[SimpleNamespace(message=_FakeMessage(content), finish_reason='stop')])


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class _RegisteredMainAgentClient(OpenAICompatibleClient):
    """Drive a root through registered control tools and one forked child."""

    supports_request_meta = True

    def __init__(self, role: str = 'root', requests: list[list[dict[str, Any]]] | None = None) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self._role = role
        self.requests = [] if requests is None else requests
        self.child_clients: list[_RegisteredMainAgentClient] = []

    def fork_for_agent(self, *, agent_id: str, parent_agent_id: str | None) -> _RegisteredMainAgentClient:
        assert agent_id == 'agent-1'
        assert parent_agent_id == 'main'
        child = type(self)(role='child')
        self.child_clients.append(child)
        return child

    async def _create_completion(
        self, messages: list[dict[str, Any]], model: Any = None, **kwargs: Any
    ) -> SimpleNamespace:
        del model
        self.requests.append(deepcopy(messages))
        if self._role == 'child':
            message = _FakeMessage('child result')
        elif len(self.requests) == 1:
            assert {schema['function']['name'] for schema in kwargs['tools']} == {'spawn_agent', 'wait_agent'}
            message = _FakeMessage(
                '',
                tool_calls=[_tool_call('spawn-1', 'spawn_agent', '{"prompt": "check the calculation"}')],
            )
        elif len(self.requests) == 2:
            message = _FakeMessage(
                '',
                tool_calls=[_tool_call('wait-1', 'wait_agent', '{"agent_ids": ["agent-1"]}')],
            )
        else:
            message = _FakeMessage('root result')
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')])


class _InterimMainWorkClient(_RegisteredMainAgentClient):
    """Emit one content-only root turn before observing the spawned child."""

    async def _create_completion(
        self, messages: list[dict[str, Any]], model: Any = None, **kwargs: Any
    ) -> SimpleNamespace:
        if self._role == 'child':
            return await super()._create_completion(messages, model=model, **kwargs)
        self.requests.append(deepcopy(messages))
        if len(self.requests) == 1:
            message = _FakeMessage(
                '',
                tool_calls=[_tool_call('spawn-1', 'spawn_agent', '{"prompt": "check the calculation"}')],
            )
        elif len(self.requests) == 2:
            message = _FakeMessage('<main_work>14</main_work>')
        elif len(self.requests) == 3:
            message = _FakeMessage(
                '',
                tool_calls=[_tool_call('wait-1', 'wait_agent', '{"agent_ids": ["agent-1"]}')],
            )
        else:
            message = _FakeMessage('root result')
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')])


def _config() -> MultiAgentConfig:
    return MultiAgentConfig(
        max_agents=2,
        max_active_agents=2,
        subagent=SubagentConfig(system_prompt='worker', tools=['worker/tool']),
    )


@pytest.mark.parametrize(
    'invalid_timeout',
    [True, False, '1', '1.5', float('nan'), float('inf'), float('-inf'), 0, -1],
)
def test_multi_agent_config_rejects_non_strict_or_nonpositive_child_timeout(invalid_timeout: Any) -> None:
    with pytest.raises(ValueError, match='child_timeout_seconds'):
        MultiAgentConfig(
            child_timeout_seconds=invalid_timeout,
            subagent=SubagentConfig(system_prompt='worker', tools=['worker/tool']),
        )


@pytest.mark.parametrize('valid_timeout', [1, 1.5])
def test_multi_agent_config_accepts_finite_positive_numeric_child_timeout(valid_timeout: int | float) -> None:
    config = MultiAgentConfig(
        child_timeout_seconds=valid_timeout,
        subagent=SubagentConfig(system_prompt='worker', tools=['worker/tool']),
    )

    assert config.child_timeout_seconds == float(valid_timeout)


async def test_registered_main_agent_executes_spawn_and_wait_tools() -> None:
    """The registered wrapper resolves and executes the control tools end to end."""
    client = _RegisteredMainAgentClient()
    agent_input = MainAgentInput(
        user_prompt='solve collaboratively',
        tools=['multi_agent/spawn_agent', 'multi_agent/wait_agent'],
        multi_agent=MultiAgentConfig(
            max_agents=1,
            max_active_agents=1,
            subagent=SubagentConfig(system_prompt='worker', tools=['base/finish']),
        ),
    )

    async with ExecutionContext(openai_clients={'main': client}):
        output = await agent_registry.get('main_agent')(agent_input)

    assert output.final_answer == 'root result'
    assert output.multi_agent_summary is not None
    assert output.multi_agent_summary['completed_agents'] == 1
    assert output.multi_agent_summary['agents'][0]['final_answer'] == 'child result'
    assert len(client.child_clients) == 1
    assert client.requests[1][-2]['role'] == 'tool'
    assert 'agent-1' in client.requests[1][-2]['content']
    assert client.requests[1][-1]['role'] == 'user'
    assert 'wait_agent' in client.requests[1][-1]['content']
    assert 'spawn_agent' in client.requests[1][-1]['content']


async def test_root_content_is_intermediate_until_spawned_child_is_observed() -> None:
    client = _InterimMainWorkClient()
    agent_input = MainAgentInput(
        user_prompt='do independent work while the child runs',
        tools=['multi_agent/spawn_agent', 'multi_agent/wait_agent'],
        multi_agent=MultiAgentConfig(
            max_agents=1,
            max_active_agents=1,
            subagent=SubagentConfig(system_prompt='worker', tools=['base/finish']),
        ),
    )

    async with ExecutionContext(openai_clients={'main': client}):
        output = await run_main_agent(agent_input)

    assert output.final_answer == 'root result'
    assert len(client.requests) == 4
    assert client.requests[2][-2] == {'role': 'assistant', 'content': '<main_work>14</main_work>'}
    assert client.requests[2][-1]['role'] == 'user'
    assert 'wait_agent' in client.requests[2][-1]['content']
    assert any(message.get('content') == '<main_work>14</main_work>' for message in output.main_trajectory)


def test_main_agent_import_registers_only_spawn_and_wait_control_tools() -> None:
    """Importing the main agent's tool module exposes only the supported control tools."""
    repository_root = Path(__file__).resolve().parents[4]
    environment = {**os.environ, 'PYTHONPATH': str(repository_root)}
    command = [
        sys.executable,
        '-c',
        (
            'from nexus.core import EntryNotFoundError, tool_registry; '
            'import nexus.extensions.agents.main.tools; '
            "registered = [tool_registry.get(name).__tool_meta__.name for name in "
            "('multi_agent/spawn_agent', 'multi_agent/wait_agent')]; "
            "\ntry:\n tool_registry.get('multi_agent/send_message'); registered.append('unexpected_send_message')"
            "\nexcept EntryNotFoundError:\n registered.append('send_message_absent')"
            "\nprint(','.join(registered))"
        ),
    ]

    completed = subprocess.run(command, check=True, capture_output=True, env=environment, text=True)

    assert completed.stdout.strip() == 'spawn_agent,wait_agent,send_message_absent'


def test_hytuner_fork_shares_runtime_but_resets_conversation_state() -> None:
    """A HyTuner child reuses immutable runtime inputs without parent history."""
    from nexus.extensions.openai_clients.hytuner import HyTunerClient

    tokenizer = object()
    server_manager = object()
    config = OpenAIConfig(
        client_args={'tokenizer': tokenizer, 'server_manager': server_manager},
        request_args={'sampling_params': {'temperature': 0.7}},
    )
    parent = HyTunerClient(config)
    parent.last_messages.append({'role': 'user', 'content': 'parent context'})
    child = parent.fork_for_agent(agent_id='agent-1', parent_agent_id='main')

    assert child is not parent
    assert parent._client_key == 'main'
    assert child._client_key == 'main:agent-1'
    assert child.config is parent.config
    assert child.tokenizer is tokenizer
    assert child.server_manager is server_manager
    assert child.request_id != parent.request_id
    assert child.last_messages == []
    assert child.last_messages is not parent.last_messages


@pytest.mark.parametrize(
    ('exit_status', 'is_success'),
    [
        (MainExitStatus.FINISHED.value, True),
        (MainExitStatus.REFUSED.value, True),
        (MainExitStatus.MAX_ITERATIONS.value, False),
        (MainExitStatus.TOOL_ERROR.value, False),
        (MainExitStatus.TOOL_EXECUTION_ERROR.value, False),
        (MainExitStatus.INFERENCE_ERROR.value, False),
        (MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value, False),
        (MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value, False),
        (MainExitStatus.INFERENCE_TIMEOUT.value, False),
        (MainExitStatus.OBSERVATION_TIMEOUT.value, False),
        (MainExitStatus.RUNTIME_ERROR.value, False),
    ],
)
async def test_finish_classifies_every_main_agent_exit_status(exit_status: str, is_success: bool) -> None:
    """Only normal child exits allow the root terminal barrier to succeed."""

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        return SimpleNamespace(final_answer='child result', exit_status=exit_status)

    controller = SubagentController(_config(), run_child=run_child)
    spawned = await controller.spawn('classify child exit')

    if is_success:
        result = await controller.finish()
        assert [item.agent_id for item in result.completed] == [spawned.agent_id]
    else:
        with pytest.raises(SubagentRunFailed) as error:
            await controller.finish()
        assert error.value.result.failed[0].agent_id == spawned.agent_id
        assert error.value.result.failed[0].exit_status == exit_status


async def test_run_main_agent_rejects_multi_agent_when_client_cannot_fork() -> None:
    """A root must fail before inference when its client cannot isolate children."""
    client = _BlockingClient()
    agent_input = MainAgentInput(
        user_prompt='delegate safely',
        tools=['multi_agent/spawn_agent', 'multi_agent/wait_agent'],
        multi_agent=_config(),
    )

    async with ExecutionContext(openai_clients={'main': client}):
        with pytest.raises(TypeError, match='fork_for_agent'):
            await run_main_agent(agent_input)


async def test_spawn_rejects_blank_prompt() -> None:
    """A child task must be explicit before it consumes capacity."""

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        return SimpleNamespace(final_answer='done', exit_status='finished')

    controller = SubagentController(_config(), run_child=run_child)

    with pytest.raises(ValueError, match='non-empty'):
        await controller.spawn('   ')

    assert controller.summary()['spawned_agents'] == 0


async def test_spawn_enforces_total_and_active_capacity() -> None:
    """The controller independently bounds total and concurrent children."""
    child_started = asyncio.Event()
    release_child = asyncio.Event()

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        child_started.set()
        await release_child.wait()
        return SimpleNamespace(final_answer='done', exit_status='finished')

    active_limited = SubagentController(
        MultiAgentConfig(
            max_agents=2,
            max_active_agents=1,
            subagent=SubagentConfig(system_prompt='worker', tools=['worker/tool']),
        ),
        run_child=run_child,
    )
    await active_limited.spawn('first')
    await child_started.wait()
    with pytest.raises(RuntimeError, match='max_active_agents'):
        await active_limited.spawn('second')
    release_child.set()
    await active_limited.finish()

    async def run_completed_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        return SimpleNamespace(final_answer='done', exit_status='finished')

    total_limited = SubagentController(
        MultiAgentConfig(
            max_agents=1,
            max_active_agents=1,
            subagent=SubagentConfig(system_prompt='worker', tools=['worker/tool']),
        ),
        run_child=run_completed_child,
    )
    first = await total_limited.spawn('first')
    await total_limited.wait([first.agent_id])
    with pytest.raises(RuntimeError, match='max_agents'):
        await total_limited.spawn('second')


async def test_duplicate_wait_ids_are_rejected_and_repeated_finish_is_idempotent() -> None:
    """Repeated close is safe while ambiguous duplicate wait targets are rejected."""

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        return SimpleNamespace(final_answer='done', exit_status='finished')

    controller = SubagentController(_config(), run_child=run_child)
    spawned = await controller.spawn('finish once')
    with pytest.raises(ValueError, match='duplicates'):
        await controller.wait([spawned.agent_id, spawned.agent_id])

    first = await controller.finish()
    second = await controller.finish()

    assert first.model_dump(mode='json') == second.model_dump(mode='json')


async def test_unknown_child_id_is_rejected_by_wait() -> None:
    """Wait cannot silently target an absent child."""

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        return SimpleNamespace(final_answer='done', exit_status='finished')

    controller = SubagentController(_config(), run_child=run_child)
    with pytest.raises(KeyError, match='unknown agent_id'):
        await controller.wait(['agent-404'])


async def test_wait_respects_any_and_all_completion_modes() -> None:
    """Any returns after one child, while all remains pending for every child."""
    releases = {'first': asyncio.Event(), 'second': asyncio.Event()}

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity
        await releases[prompt].wait()
        return SimpleNamespace(final_answer=prompt, exit_status='finished')

    controller = SubagentController(_config(), run_child=run_child)
    first = await controller.spawn('first')
    second = await controller.spawn('second')
    wait_any_task = asyncio.create_task(controller.wait([first.agent_id, second.agent_id], return_when='any'))
    releases['first'].set()
    any_result = await wait_any_task

    assert [item.agent_id for item in any_result.completed] == [first.agent_id]
    assert any_result.pending == [second.agent_id]

    wait_all_task = asyncio.create_task(controller.wait([first.agent_id, second.agent_id], return_when='all'))
    await asyncio.sleep(0)
    assert not wait_all_task.done()
    releases['second'].set()
    all_result = await wait_all_task

    assert [item.agent_id for item in all_result.completed] == [first.agent_id, second.agent_id]
    assert all_result.pending == []


async def test_wait_timeout_does_not_cancel_child() -> None:
    """A wait timeout reports pending state without changing child lifecycle."""
    child_started = asyncio.Event()
    release_child = asyncio.Event()

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        child_started.set()
        await release_child.wait()
        return SimpleNamespace(final_answer='done', exit_status='finished')

    controller = SubagentController(_config(), run_child=run_child)
    spawned = await controller.spawn('wait without cancellation')
    await child_started.wait()
    result = await controller.wait([spawned.agent_id], timeout_seconds=0.001)

    assert result.timed_out is True
    assert result.pending == [spawned.agent_id]
    assert controller.get_record(spawned.agent_id).task.done() is False

    release_child.set()
    await controller.finish()


async def test_abort_cancellation_still_cancels_the_child() -> None:
    """Cancelling cleanup cannot leave an active child behind."""
    child_started = asyncio.Event()
    never_release = asyncio.Event()

    async def run_child(identity: Any, prompt: str) -> None:
        del identity, prompt
        child_started.set()
        await never_release.wait()

    controller = SubagentController(_config(), run_child=run_child)
    await controller.spawn('block cleanup')
    await child_started.wait()
    abort_task = asyncio.create_task(controller.abort())
    await asyncio.sleep(0)
    abort_task.cancel()
    await asyncio.gather(abort_task, return_exceptions=True)
    await asyncio.sleep(0)

    assert controller.summary()['pending_agents'] == 0


async def test_finish_waits_for_running_child_instead_of_cancelling_it() -> None:
    """A normal root completion keeps a running child alive until it finishes."""
    release_child = asyncio.Event()
    child_started = asyncio.Event()

    async def run_child(identity: Any, prompt: str) -> SimpleNamespace:
        del identity, prompt
        child_started.set()
        await release_child.wait()
        return SimpleNamespace(final_answer='done', exit_status='finished')

    controller = SubagentController(_config(), run_child=run_child)
    spawned = await controller.spawn('solve 1 + 2')
    await child_started.wait()
    finish_task = asyncio.create_task(controller.finish())
    await asyncio.sleep(0)

    assert not finish_task.done()
    release_child.set()
    result = await finish_task

    assert [item.agent_id for item in result.completed] == [spawned.agent_id]
    assert result.failed == []
    assert result.pending == []
    assert controller.summary()['cancelled_agents'] == 0


async def test_abort_cancels_running_children_and_leaves_no_tasks() -> None:
    """Only the abort path cancels a child that has not reached a terminal state."""
    child_started = asyncio.Event()
    never_release = asyncio.Event()

    async def run_child(identity: Any, prompt: str) -> None:
        del identity, prompt
        child_started.set()
        await never_release.wait()

    controller = SubagentController(_config(), run_child=run_child)
    spawned = await controller.spawn('wait forever')
    await child_started.wait()
    await controller.abort()

    assert controller.summary()['pending_agents'] == 0
    assert controller.summary()['cancelled_agents'] == 1
    assert controller.get_record(spawned.agent_id).task.done()


async def test_finish_raises_when_any_child_fails() -> None:
    """A failed child makes the root lifecycle fail atomically after waiting."""

    async def run_child(identity: Any, prompt: str) -> None:
        del identity, prompt
        raise LookupError('missing source')

    controller = SubagentController(_config(), run_child=run_child)
    await controller.spawn('collect sources')

    with pytest.raises(SubagentRunFailed) as error:
        await controller.finish()

    assert error.value.result.failed[0].status == AgentStatus.FAILED
