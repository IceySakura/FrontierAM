from typing import Any

import pytest

from nexus.extensions.agents.pi_agent import agent


@pytest.mark.asyncio
async def test_run_pi_invokes_runtime_with_expected_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured['target'] = target
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.25,
        }

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    result = await agent.run_pi(agent.PiInput(user_prompt='hello', working_dir='/workspace'))

    assert result.success is True
    assert result.output == 'done'
    assert captured['target'].__name__ == 'setup_and_run_pi'
    assert captured['version'] == '0.84.1'
    assert captured['model_id'] == 'nexus-proxy-model'
    assert captured['output_format'] == 'json'
    # Every built-in tool, including the read-only ones pi ships disabled.
    assert captured['tools'] == ['read', 'bash', 'edit', 'write', 'grep', 'find', 'ls']


@pytest.mark.asyncio
async def test_run_pi_defaults_to_an_isolated_benchmark_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """A default run must depend on the task prompt alone, not on repo contents."""
    captured: dict[str, Any] = {}

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured.update(kwargs)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 0.1}

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    await agent.run_pi(agent.PiInput(user_prompt='hello', working_dir='/workspace'))

    assert captured['load_context_files'] is False
    assert captured['trust_project_files'] is False
    assert captured['load_extensions'] is False


@pytest.mark.asyncio
async def test_run_pi_forwards_development_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured.update(kwargs)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 0.1}

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    await agent.run_pi(
        agent.PiInput(
            user_prompt='hello',
            working_dir='/workspace',
            load_context_files=True,
            trust_project_files=True,
            tools=['read'],
            extra_env={'MY_TOKEN': 'x'},
        )
    )

    assert captured['load_context_files'] is True
    assert captured['trust_project_files'] is True
    assert captured['tools'] == ['read']
    assert captured['extra_env'] == {'MY_TOKEN': 'x'}


@pytest.mark.asyncio
async def test_run_pi_allows_disabling_all_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit empty list must stay empty rather than fall back to the defaults."""
    captured: dict[str, Any] = {}

    async def fake_call_runtime_agent(target, /, **kwargs):
        captured.update(kwargs)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 0.1}

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    await agent.run_pi(agent.PiInput(user_prompt='hello', working_dir='/workspace', tools=[]))

    assert captured['tools'] == []


@pytest.mark.asyncio
async def test_run_pi_surfaces_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_runtime_agent(target, /, **kwargs):
        return {
            'success': True,
            'output': '{"type":"agent_end"}',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
            'trace': {'turns': 2, 'tool_calls': 1, 'event_counts': {'agent_end': 1}},
        }

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    result = await agent.run_pi(agent.PiInput(user_prompt='hello', working_dir='/workspace'))

    assert result.trace is not None
    assert result.trace['turns'] == 2
    assert result.trace['tool_calls'] == 1


@pytest.mark.asyncio
async def test_run_pi_reports_failure_without_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_runtime_agent(target, /, **kwargs):
        return {
            'success': False,
            'output': '',
            'stderr': 'Failed to install pi: boom',
            'exit_code': -1,
            'execution_time': 0.5,
        }

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    result = await agent.run_pi(agent.PiInput(user_prompt='hello', working_dir='/workspace'))

    assert result.success is False
    assert result.exit_code == -1
    assert 'boom' in result.stderr
    assert result.trace is None


@pytest.mark.asyncio
async def test_run_pi_warns_when_thinking_without_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """pi silently ignores --thinking unless the model advertises reasoning."""

    async def fake_call_runtime_agent(target, /, **kwargs):
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'execution_time': 0.1}

    monkeypatch.setattr(agent, 'call_runtime_agent', fake_call_runtime_agent)
    with caplog.at_level('WARNING'):
        await agent.run_pi(
            agent.PiInput(user_prompt='hello', working_dir='/workspace', thinking='high', reasoning=False)
        )

    assert 'not advertised as reasoning-capable' in caplog.text


def test_run_pi_kwargs_match_the_runtime_signature() -> None:
    """run_pi passes kwargs across an RPC boundary, so a typo in a keyword only
    surfaces as a TypeError inside the sandbox. Pin the contract here instead."""
    import inspect
    import re

    from pathlib import Path

    from nexus.extensions.agents.pi_agent.runtime_setup import setup_and_run_pi

    accepted = set(inspect.signature(setup_and_run_pi).parameters)
    source = Path(agent.__file__).read_text()
    call_block = source.split('call_runtime_agent(', 1)[1].split(')\n', 1)[0]
    passed = set(re.findall(r'^\s{8}(\w+)=', call_block, re.M))

    assert passed - accepted == set(), f'setup_and_run_pi does not accept: {sorted(passed - accepted)}'
    # 'task' is passed positionally; everything else must be forwarded.
    assert accepted - passed - {'task'} == set(), f'not forwarded: {sorted(accepted - passed - {"task"})}'
