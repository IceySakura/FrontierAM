"""A smoke task: run one conversation and report what the mechanism did.

Exists to check the plumbing rather than a model's work. A real recipe scores whether
the worker solved something; this one scores whether the pieces reached each other --
did a turn happen, did the worker read its user memory, did it run a skill it was
asked for, did it call an MCP tool.

That distinction is the whole design. The requirement text and the checks are separate
inputs: a run states what it wants the worker to do, and separately what it expects to
observe. Nothing here knows about search, or skills, or memory in particular -- a check
names a tool or a string, and the task looks for it.

What a check can look at:

``tool``
    A tool name the worker used, matched as a prefix of what the trajectory recorded.
    ``mcp__search`` matches ``mcp__search__web_search``.
``says``
    Text in the worker's own output, for a run whose prompt asks it to echo something.
``command``
    A shell command run in the sandbox after the conversation, expected to exit 0.
    This is how a check reaches something the conversation did not report -- a file
    the worker wrote, a registration ``claude mcp add`` left behind.

The score is the share of checks that passed, so a partial result is still a number
and the failing ones are named in the output.
"""

import json
import logging

from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserOutput
from nexus.extensions.tasks.hitl.trace import labels


logger = logging.getLogger(__name__)


TASK_NAME = 'hitl/smoke_task'

DEFAULT_COMMAND_TIMEOUT_SECONDS = 60


class SmokeCheck(BaseModel):
    """One thing a run expects to observe.

    Exactly one of the three is set. ``name`` is what the report calls it, so a
    failing check reads as a sentence rather than a pattern.
    """

    name: str
    tool: str = ''
    says: str = ''
    command: str = ''

    def kind(self) -> Literal['tool', 'says', 'command']:
        if self.tool:
            return 'tool'
        if self.says:
            return 'says'
        if self.command:
            return 'command'
        raise ValueError(f'check {self.name!r} sets none of tool, says or command')


class SmokeResult(BaseModel):
    """What one check found."""

    name: str
    kind: str
    passed: bool
    detail: str = ''


class SmokeTaskInput(TaskRunInput, WithAgentRunner):
    """What the run asks for, and what it expects to see."""

    message: str
    """What the user asks the worker for. Reaches the agent as its ``user_prompt``."""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    """The observations this run expects, as :class:`SmokeCheck` fields."""
    working_dir: str = '/workspace'
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS


class SmokeTaskOutput(TaskRunOutput, WithScore):
    """The share of checks that passed, and which ones did not."""

    all_passed: bool = False
    results: list[SmokeResult] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    """Every tool name the worker's trajectories recorded, for a run whose check
    failed and needs to say what the worker did use instead."""
    mode: str = ''
    """How the user was driven, from ``SmokeAgentOutput``.

    Reported because the rest of these are not comparable across modes: a replay makes
    no user-side model call, so its ``turns`` counts script steps.
    """
    turns: int = 0
    worker_rounds: int = 0
    agent_error: str = ''


@task_registry.register(TASK_NAME)
async def run_smoke_task(task_input: SmokeTaskInput) -> SmokeTaskOutput:
    """Run one conversation, then look for what the run said it expected.

    A conversation that failed is still checked: a ``command`` check can find what a
    failed run left behind, which is often the thing being debugged.
    """
    checks = [SmokeCheck.model_validate(raw) for raw in task_input.checks]
    if not checks:
        raise ValueError('a smoke run needs at least one check; otherwise it reports nothing')

    agent_output = await task_input.run_agent(
        user_prompt=task_input.message,
        working_dir=task_input.working_dir,
    )
    if not isinstance(agent_output, AgentWithUserOutput):
        raise TypeError(
            f'the smoke task reads {AgentWithUserOutput.__name__} fields, and this run produced '
            f'{type(agent_output).__name__}. Set agent_name to one that returns it.'
        )
    # Before the checks: a command check may start a process that makes its own
    # requests, which would then count as the conversation's.
    tools_used = _tools_used()
    said = _said()

    results: list[SmokeResult] = []
    for check in checks:
        kind = check.kind()
        if kind == 'tool':
            matched = sorted(name for name in tools_used if name.startswith(check.tool))
            results.append(
                SmokeResult(
                    name=check.name,
                    kind=kind,
                    passed=bool(matched),
                    detail=', '.join(matched) if matched else f'no tool starts with {check.tool!r}',
                )
            )
        elif kind == 'says':
            found = check.says in said
            results.append(
                SmokeResult(
                    name=check.name,
                    kind=kind,
                    passed=found,
                    detail='' if found else f'{check.says!r} is not in what the worker said',
                )
            )
        else:
            results.append(await _run_check_command(check, task_input))

    passed = [result for result in results if result.passed]
    for result in results:
        logger.info(
            '[hitl/smoke] %s %s (%s) %s',
            'PASS' if result.passed else 'FAIL',
            result.name,
            result.kind,
            result.detail,
        )

    return SmokeTaskOutput(
        all_passed=len(passed) == len(results),
        score=len(passed) / len(results),
        results=results,
        tools_used=sorted(tools_used),
        mode=_mode_of(agent_output),
        turns=agent_output.turns,
        worker_rounds=agent_output.worker_rounds,
        agent_error=_agent_error(agent_output),
    )


def _mode_of(agent_output: AgentWithUserOutput) -> str:
    """How the user was driven, or '' for an agent that does not say.

    Read off the field rather than the class: a task should not have to import the
    smoke agent to run against it, and an agent that drives the user one way only has
    nothing to report.
    """
    mode = agent_output.model_dump().get('mode')
    return mode if isinstance(mode, str) else ''


async def _run_check_command(check: SmokeCheck, task_input: SmokeTaskInput) -> SmokeResult:
    """Run one check's command in the sandbox the conversation used.

    A command that cannot be run at all fails the check rather than raising: the
    other checks have already cost a conversation.
    """
    from nexus.runtime.executors.models.command import RunCommandRequest

    try:
        runtime = await require_context().get_runtime()
        answer = await runtime.call(
            RunCommandRequest(
                command=check.command,
                cwd=task_input.working_dir,
                timeout=float(task_input.command_timeout),
            )
        )
    except Exception as exc:  # noqa: BLE001 - a failed check is a result, not a raise
        return SmokeResult(name=check.name, kind='command', passed=False, detail=f'{type(exc).__name__}: {exc}')

    output = _trimmed(f'{answer.stdout or ""}{answer.stderr or ""}')
    if answer.return_code == 0:
        return SmokeResult(name=check.name, kind='command', passed=True, detail=output)
    return SmokeResult(
        name=check.name,
        kind='command',
        passed=False,
        detail=f'{answer.status} exit {answer.return_code}: {output}',
    )


def _trimmed(text: str) -> str:
    """One line a log can carry, from however much a command wrote."""
    return ' '.join(text.split())[:400]


def _tools_used() -> set[str]:
    """Every tool name the run-level chat tracker recorded, whoever called it.

    Read off the exported trajectories rather than the agent output: the generic
    agent output carries no trajectories at all, and a report is what the user was
    shown, which a backend may summarise.

    Not filtered by label. A user-side tool and a worker-side one are both evidence
    that a mechanism worked, and a check names the tool it wants.
    """
    names: set[str] = set()
    for trajectory in labels.run_level_trajectories().get('trajectories') or []:
        messages = trajectory.get('messages') if isinstance(trajectory, dict) else trajectory
        for message in messages or []:
            if not isinstance(message, dict):
                continue
            for call in message.get('tool_calls') or []:
                if not isinstance(call, dict):
                    continue
                name = (call.get('function') or {}).get('name')
                if isinstance(name, str) and name:
                    names.add(name)
    return names


def _said() -> str:
    """Every recorded message, serialised, as one string to search.

    Read off the exported trajectories for the reason :func:`_tools_used` gives. Whole
    messages rather than their text: a ``says`` check names a string it expects
    somewhere in the run, and a tool's arguments and results are part of that.
    """
    parts: list[str] = []
    for trajectory in labels.run_level_trajectories().get('trajectories') or []:
        messages = trajectory.get('messages') if isinstance(trajectory, dict) else trajectory
        parts.append(json.dumps(messages, ensure_ascii=False, default=str))
    return '\n'.join(parts)


def _agent_error(agent_output: AgentWithUserOutput) -> str:
    """What the agent reported going wrong, or '' when it did not."""
    return '' if agent_output.success else str(agent_output.stderr or '')


__all__ = ['TASK_NAME', 'SmokeCheck', 'SmokeResult', 'SmokeTaskInput', 'SmokeTaskOutput', 'run_smoke_task']
