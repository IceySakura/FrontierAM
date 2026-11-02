"""``terminal_task`` has to report the agent wall it enforces itself.

The task wraps the agent in ``asyncio.wait_for(timeout=agent_timeout)`` and then swallows
the timeout on purpose, so the tests still run and a partially finished job can earn its
score. Nothing recorded that the wall had fired, though: the trajectory reached hytuner
with an empty ``termination_reason`` and fell back to ``ENV_DONE``, the same label a run
that finished on its own gets. Measured over 9,087 terminus2 trajectories, the only
reasons ever observed were ``ENV_DONE`` and ``UNHANDLED_ERROR`` -- ``TASK_TIMEOUT`` never
appeared, while swe_agent, which reports the same way from inside its own agent, showed
it on 4.86% of trajectories.

The agent itself (terminus2) reports its exit status, but only on the path where it
returns; ``wait_for`` cancels it, so that report never runs. The wall belongs to the task
and only the task can report it.
"""

from __future__ import annotations

import asyncio

import pytest

from nexus.core.context import ExecutionContext, _current_context
from nexus.extensions.tasks.terminal import task as terminal_task_module
from nexus.extensions.tasks.terminal.task import TerminalTaskInput, terminal_task
from nexus.runtime.providers.base import RuntimeProvider


class _StubProvider(RuntimeProvider):
    async def acquire(self, name: str, **params):
        raise AssertionError('the test pre-populates the runtime cache; acquire should not run')

    async def _release_impl(self, name: str) -> None:
        return None


class _StubCommandResult:
    return_code = 0


class _StubRuntime:
    async def run_command(self, command: str, timeout: int):
        return _StubCommandResult()


def _run_task(monkeypatch: pytest.MonkeyPatch, *, agent, agent_timeout: int) -> dict[str, object]:
    """Drive terminal_task with scoring stubbed, returning the span attributes it wrote."""
    recorded: dict[str, object] = {}

    async def _tests(_task_input):
        return 1, {}

    monkeypatch.setattr(terminal_task_module, 'run_terminal_tests_in_runtime', _tests)
    monkeypatch.setattr(terminal_task_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(
        terminal_task_module,
        'update_span',
        lambda span_id, attributes: recorded.update(attributes),
    )

    task_input = TerminalTaskInput(
        message='do the thing',
        agent_timeout=agent_timeout,
        run_agent=agent,
    )

    ctx = ExecutionContext(runtime_provider=_StubProvider(), task_name='terminal_task')
    ctx._runtimes['main'] = _StubRuntime()

    token = _current_context.set(ctx)
    try:
        asyncio.run(terminal_task(task_input))
    finally:
        _current_context.reset(token)
    return recorded


def test_the_agent_wall_is_reported_as_task_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reported bug: this used to reach hytuner as ENV_DONE, indistinguishable from a
    # run the agent ended itself.
    async def _agent_that_overruns(**_kwargs):
        await asyncio.sleep(10)

    recorded = _run_task(monkeypatch, agent=_agent_that_overruns, agent_timeout=0)

    assert recorded['hytuner.termination_reason'] == 'TASK_TIMEOUT'


def test_an_agent_that_finishes_in_time_reports_no_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    # The task must not label anything itself here: terminus2 reports its own exit status,
    # and a reason written by the task would overwrite it on the same span.
    async def _agent_that_finishes(**_kwargs):
        return None

    recorded = _run_task(monkeypatch, agent=_agent_that_finishes, agent_timeout=30)

    assert 'hytuner.termination_reason' not in recorded
