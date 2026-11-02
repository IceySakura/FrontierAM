"""A parallel smoke task: several worker sessions at once, checked for independence.

Exists to check in a real sandbox what the unit tests check with fakes -- that two
sessions running at the same time keep their conversations, their trace labels and
their token accounting apart. Every mechanism it exercises is already covered by
``tests/.../actor/test_run_turns.py``, ``trace/test_span_labels.py`` and
``actor/labelling/test_handler_and_proxy.py``; what those cannot reach is real Claude
Code processes behind real per-session proxies talking to a real gateway.

The checks a run can ask for, beyond the three ``hitl/smoke_task`` offers:

``label``
    Text that must appear in one label's trajectories, and text that must not. Two
    sessions told different things must not be able to quote each other.
``overlap``
    How many sessions were running at once. Read off what ``run_turns`` recorded --
    ``BatchTurnResult.inflight`` names the sessions still working when one finished --
    rather than from wall-clock timing, which cannot tell a slow turn from a
    concurrent one.
``reacted``
    Whether a session was answered the moment its turn finished, while its neighbours
    were still on theirs. This is the affordance ``run_turns`` yielding in completion
    order exists for: act on the first result without waiting for the batch to drain.

Two things are asserted for every run rather than being checks, because they hold for
any correct run whatever it asked for: nothing exported unattributed, and one token
label per session plus the user's.
"""

import json
import logging

from typing import Any, Literal, cast

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


TASK_NAME = 'hitl/parallel_task'

DEFAULT_COMMAND_TIMEOUT_SECONDS = 60

CheckKind = Literal['label', 'overlap', 'reacted', 'tool', 'says', 'command']


class ParallelCheck(BaseModel):
    """One thing a run expects to observe.

    Exactly one of ``label``, ``overlap``, ``reacted``, ``tool``, ``says`` and
    ``command`` is set. ``holds`` and ``lacks`` belong to ``label``.
    """

    name: str
    label: str = ''
    holds: str = ''
    lacks: str = ''
    overlap: int = 0
    reacted: bool = False
    tool: str = ''
    says: str = ''
    command: str = ''

    def kind(self) -> CheckKind:
        if self.label:
            return 'label'
        if self.overlap:
            return 'overlap'
        if self.reacted:
            return 'reacted'
        if self.tool:
            return 'tool'
        if self.says:
            return 'says'
        if self.command:
            return 'command'
        raise ValueError(f'check {self.name!r} sets none of label, overlap, reacted, tool, says or command')


class ParallelResult(BaseModel):
    """What one check found."""

    name: str
    kind: str
    passed: bool
    detail: str = ''


class ParallelTaskInput(TaskRunInput, WithAgentRunner):
    """What the run drives, and what it expects to see."""

    message: str = ''
    """The opening the user is given. A replay drives the workers from its script, so
    this is context rather than instruction."""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    """The observations this run expects, as :class:`ParallelCheck` fields."""
    working_dir: str = '/workspace'
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS


class ParallelTaskOutput(TaskRunOutput, WithScore):
    """The share of checks that passed, and what the sessions did."""

    all_passed: bool = False
    results: list[ParallelResult] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    session_ids: dict[str, str] = Field(default_factory=dict)
    """Each session's own id. Two sessions sharing one would mean one process served
    both, which every other check would still pass."""
    peak_inflight: int = 0
    """The most sessions recorded running at once."""
    unattributed: int = 0
    """Trajectories that carry no label. Any number above zero means something made
    requests from outside every labelled scope."""
    usage_labels: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    turns: int = 0
    worker_rounds: int = 0
    agent_error: str = ''


@task_registry.register(TASK_NAME)
async def run_parallel_task(task_input: ParallelTaskInput) -> ParallelTaskOutput:
    """Run several sessions at once, then look for what the run said it expected.

    A conversation that failed is still checked: what the sessions left behind is
    often the thing being debugged.
    """
    checks = [ParallelCheck.model_validate(raw) for raw in task_input.checks]
    if not checks:
        raise ValueError('a parallel run needs at least one check; otherwise it reports nothing')

    agent_output = await task_input.run_agent(
        user_prompt=task_input.message,
        working_dir=task_input.working_dir,
    )
    if not isinstance(agent_output, AgentWithUserOutput):
        raise TypeError(
            f'the parallel task reads {AgentWithUserOutput.__name__} fields, and this run produced '
            f'{type(agent_output).__name__}. Set agent_name to one that returns it.'
        )

    # Read before any command check: a command may start something that makes its own
    # requests, which would then count as the conversation's.
    exported = labels.run_level_trajectories()
    tools_used = _tools_used(exported)
    said = _said()
    reported = agent_output.model_dump()
    overlaps = _overlaps(reported)

    results: list[ParallelResult] = []
    for check in checks:
        kind = check.kind()
        if kind == 'label':
            results.append(_check_label(check, exported))
        elif kind == 'overlap':
            results.append(_check_overlap(check, overlaps))
        elif kind == 'reacted':
            results.append(_check_reacted(check, _followed(reported)))
        elif kind == 'tool':
            matched = sorted(name for name in tools_used if name.startswith(check.tool))
            results.append(
                ParallelResult(
                    name=check.name,
                    kind=kind,
                    passed=bool(matched),
                    detail=', '.join(matched) if matched else f'no tool starts with {check.tool!r}',
                )
            )
        elif kind == 'says':
            found = check.says in said
            results.append(
                ParallelResult(
                    name=check.name,
                    kind=kind,
                    passed=found,
                    detail='' if found else f'{check.says!r} is not in what the workers said',
                )
            )
        else:
            results.append(await _run_check_command(check, task_input))

    passed = [result for result in results if result.passed]
    for result in results:
        logger.info(
            '[hitl/parallel] %s %s (%s) %s',
            'PASS' if result.passed else 'FAIL',
            result.name,
            result.kind,
            result.detail,
        )

    return ParallelTaskOutput(
        all_passed=len(passed) == len(results),
        score=len(passed) / len(results),
        results=results,
        sessions=_strings(reported.get('sessions')),
        session_ids=_id_map(reported.get('session_ids')),
        peak_inflight=peak_inflight(overlaps),
        unattributed=labels.unattributed_count(exported),
        usage_labels=sorted(_usage_labels()),
        tools_used=sorted(tools_used),
        turns=agent_output.turns,
        worker_rounds=agent_output.worker_rounds,
        agent_error=_agent_error(agent_output),
    )


# -- the two checks this task adds ------------------------------------------


def _check_label(check: ParallelCheck, exported: dict[str, Any]) -> ParallelResult:
    """Whether one label's trajectories hold what they should and nothing they should not."""
    trajectories = labels.filter_by_label(exported, check.label).get('trajectories') or []
    if not trajectories:
        return ParallelResult(
            name=check.name,
            kind='label',
            passed=False,
            detail=f'nothing exported under {check.label!r}; there is: {", ".join(_exported_labels(exported)) or "(none)"}',
        )

    text = '\n'.join(_trajectory_text(trajectory) for trajectory in trajectories)
    missing = check.holds and check.holds not in text
    leaked = check.lacks and check.lacks in text
    if missing and leaked:
        detail = f'{check.holds!r} is absent and {check.lacks!r} leaked in'
    elif missing:
        detail = f'{check.holds!r} is not in what {check.label} recorded'
    elif leaked:
        detail = f'{check.lacks!r} reached {check.label}, which never saw it'
    else:
        detail = f'{len(trajectories)} trajectory(ies), {len(text)} characters'
    return ParallelResult(name=check.name, kind='label', passed=not (missing or leaked), detail=detail)


def _check_overlap(check: ParallelCheck, overlaps: list[dict[str, Any]]) -> ParallelResult:
    """Whether enough sessions were ever recorded running at once."""
    if not overlaps:
        return ParallelResult(
            name=check.name,
            kind='overlap',
            passed=False,
            detail='the agent recorded no turns, so nothing says what ran together',
        )
    peak = peak_inflight(overlaps)
    return ParallelResult(
        name=check.name,
        kind='overlap',
        passed=peak >= check.overlap,
        detail=f'{peak} session(s) ran at once, wanted {check.overlap}',
    )


def _check_reacted(check: ParallelCheck, followed: list[dict[str, Any]]) -> ParallelResult:
    """Whether a follow-up went out while other sessions were still working.

    A follow-up sent after the batch drained is a correct send but not the affordance
    being checked: the point is that a finished session can be answered without waiting
    for its neighbours.
    """
    if not followed:
        return ParallelResult(
            name=check.name,
            kind='reacted',
            passed=False,
            detail='no follow-up was sent; the script asks for none, or no turn completed',
        )
    reactive = [item for item in followed if _strings(item.get('while_running'))]
    if not reactive:
        return ParallelResult(
            name=check.name,
            kind='reacted',
            passed=False,
            detail=f'{len(followed)} follow-up(s), all sent after the batch had drained',
        )
    named = ', '.join(
        f'{item.get("session")} while {"/".join(_strings(item.get("while_running")))} ran' for item in reactive
    )
    return ParallelResult(name=check.name, kind='reacted', passed=True, detail=named)


def peak_inflight(overlaps: list[dict[str, Any]]) -> int:
    """The most sessions running at once, over every recorded turn.

    A turn's own session is running while it finishes, so it counts alongside the ones
    still named in ``inflight``.
    """
    return max((len(_strings(item.get('inflight'))) + 1 for item in overlaps), default=0)


# -- reading what the run left behind ---------------------------------------


def _overlaps(reported: dict[str, Any]) -> list[dict[str, Any]]:
    found = reported.get('overlaps')
    return [item for item in found if isinstance(item, dict)] if isinstance(found, list) else []


def _followed(reported: dict[str, Any]) -> list[dict[str, Any]]:
    found = reported.get('followed')
    return [item for item in found if isinstance(item, dict)] if isinstance(found, list) else []


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, (list, tuple, set)) else []


def _id_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, str)}


def _exported_labels(exported: dict[str, Any]) -> list[str]:
    return sorted(labels.group_by_label(exported))


def _trajectory_text(trajectory: object) -> str:
    """Everything one trajectory holds, as one string to search.

    Tool arguments are included: what a session was told arrives as a call's argument
    on the user's side and as a message on the worker's.
    """
    parts: list[str] = []
    for message in _messages_of(trajectory):
        parts.append(_content_text(message.get('content')))
        calls = message.get('tool_calls')
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get('function')
            arguments = function.get('arguments') if isinstance(function, dict) else None
            if isinstance(arguments, str):
                parts.append(arguments)
    return '\n'.join(part for part in parts if part)


def _messages_of(trajectory: object) -> list[dict[str, Any]]:
    """One trajectory's messages, whichever of the two exported shapes it is.

    ``branch_last`` exports a dict carrying them; ``global_first`` a bare list.
    """
    found: object = trajectory
    if isinstance(trajectory, dict):
        found = cast('dict[str, Any]', trajectory).get('messages')
    if not isinstance(found, list):
        return []
    return [cast('dict[str, Any]', message) for message in found if isinstance(message, dict)]


def _content_text(content: object) -> str:
    """A message's text, whether it arrived as a string or as typed blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = cast('dict[str, Any]', block).get('text')
        if isinstance(text, str):
            parts.append(text)
    return '\n'.join(parts)


def _usage_labels() -> set[str]:
    """The labels the usage processor counted, or none when it is not installed."""
    from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor
    from nexus.trace.context import get_processor

    processor = get_processor(HITLUsageProcessor)
    if processor is None:
        return set()
    by_label = processor.get_result().get('by_label')
    return set(by_label) if isinstance(by_label, dict) else set()


def _tools_used(exported: dict[str, Any]) -> set[str]:
    """Every tool name the run-level chat tracker recorded, whoever called it."""
    names: set[str] = set()
    trajectories = exported.get('trajectories')
    for trajectory in trajectories if isinstance(trajectories, list) else []:
        for message in _messages_of(trajectory):
            calls = message.get('tool_calls')
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get('function')
                name = function.get('name') if isinstance(function, dict) else None
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
    return '' if agent_output.success else str(agent_output.stderr or '')


async def _run_check_command(check: ParallelCheck, task_input: ParallelTaskInput) -> ParallelResult:
    """Run one check's command in the sandbox the sessions used.

    A command that cannot be run at all fails the check rather than raising: the other
    checks have already cost a run.
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
        return ParallelResult(name=check.name, kind='command', passed=False, detail=f'{type(exc).__name__}: {exc}')

    output = _trimmed(f'{answer.stdout or ""}{answer.stderr or ""}')
    if answer.return_code == 0:
        return ParallelResult(name=check.name, kind='command', passed=True, detail=output)
    return ParallelResult(
        name=check.name,
        kind='command',
        passed=False,
        detail=f'{answer.status} exit {answer.return_code}: {output}',
    )


def _trimmed(text: str) -> str:
    """One line a log can carry, from however much a command wrote."""
    return ' '.join(text.split())[:400]


__all__ = [
    'TASK_NAME',
    'CheckKind',
    'ParallelCheck',
    'ParallelResult',
    'ParallelTaskInput',
    'ParallelTaskOutput',
    'peak_inflight',
    'run_parallel_task',
]
