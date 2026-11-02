"""The parallel agent: several worker sessions, driven from a script.

The user is replaced by a script here, as in ``recipe/smoke`` -- but where that one
drives a single session, this drives several at once through
:meth:`SessionManager.run_turns`. What the run is for is what happens between the
sessions, so no model decides which of them gets addressed: a step names its
recipients, and the script is the same on every run.

What it records for the task to check:

``session_ids``
    Each session's own id. Two sessions sharing one would mean a single process served
    both, which the label checks would not notice.
``overlaps``
    Per finished turn, which sessions were still running. This is what
    ``BatchTurnResult.inflight`` reports, and it is the run's evidence of concurrency --
    wall-clock timing cannot tell a slow turn from a simultaneous one.
"""

import asyncio
import logging
import time

from typing import Any

from pydantic import BaseModel, Field

from nexus.core import agent_registry, require_context
from nexus.extensions.tasks.hitl.actor.base import Session, SessionManager, SessionMessage
from nexus.extensions.tasks.hitl.actor.factory import make_worker_manager
from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserInput, AgentWithUserOutput
from nexus.extensions.tasks.hitl.trace import labels


logger = logging.getLogger(__name__)


AGENT_NAME = 'hitl/parallel_agent'

DEFAULT_SESSIONS = ('alpha', 'beta')


class ParallelSend(BaseModel):
    """One message to one session, inside a step."""

    to: str
    message: str
    follow_up: list[str] = Field(default_factory=list)
    """Messages sent to this session the moment its turn finishes.

    Sent from inside the loop over ``run_turns`` rather than after it, so this session
    starts its next turn while the others are still on their first. A session that is
    not ``completed`` gets none of them: there is nothing to continue.
    """


class ParallelStep(BaseModel):
    """One step of the script: a set of sessions addressed at once.

    Every send in a step starts before any of them finishes.
    """

    parallel: list[ParallelSend] = Field(default_factory=list)


class ParallelAgentInput(AgentWithUserInput):
    """The production input plus the sessions to open and the script to run."""

    sessions: list[str] = Field(default_factory=lambda: list(DEFAULT_SESSIONS))
    """Names to open, one worker each. Each becomes the label ``worker/<name>``."""
    script: list[dict[str, Any]] = Field(default_factory=list)
    """The steps to run, as :class:`ParallelStep` fields."""
    shared_workspace: bool = False
    """Whether the sessions share ``working_dir``.

    Off by default: two Claude Code processes writing one directory make a crossed
    file indistinguishable from a crossed conversation, and the crossing is what a run
    is looking for. Each session otherwise works in ``<working_dir>/<name>``.
    """


class ParallelAgentOutput(AgentWithUserOutput):
    """The production output plus what the sessions did."""

    sessions: list[str] = Field(default_factory=list)
    session_ids: dict[str, str] = Field(default_factory=dict)
    workspaces: dict[str, str] = Field(default_factory=dict)
    overlaps: list[dict[str, Any]] = Field(default_factory=list)
    followed: list[dict[str, Any]] = Field(default_factory=list)
    """Each follow-up sent from inside the batch loop, against who was still running
    when it went out. A ``while_running`` that is never populated means every follow-up
    waited for the batch to drain."""
    steps_run: int = 0
    replies: dict[str, str] = Field(default_factory=dict)
    """Each session's last reply, so a run can see what they answered without opening
    the trajectories."""
    reports: dict[str, str] = Field(default_factory=dict)
    """Each session's last turn as ``SessionManager.report`` rendered it, which is the
    path a user-facing run reads."""


def check_sessions(names: list[str]) -> list[str]:
    """Refuse a session list nothing could be attributed from."""
    if not names:
        raise ValueError('a parallel run needs at least one session')
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f'session names must differ; repeated: {repeated}')
    for name in names:
        labels.validate_name(name, kind='session name')
    return names


def check_script(steps: list[ParallelStep], *, sessions: list[str]) -> None:
    """Refuse a script that addresses nobody, or somebody twice in one step.

    ``run_turns`` refuses a repeated session as well; refusing here names the step it
    was in, which the run can act on before spending a sandbox.
    """
    if not steps:
        raise ValueError('a parallel run needs at least one step')
    known = set(sessions)
    for index, step in enumerate(steps, start=1):
        if not step.parallel:
            raise ValueError(f'step {index} addresses nobody; give it at least one send')
        addressed = [send.to for send in step.parallel]
        unknown = sorted({name for name in addressed if name not in known})
        if unknown:
            raise ValueError(f'step {index} addresses {unknown}, which this run does not open. It has: {sessions}')
        repeated = sorted({name for name in addressed if addressed.count(name) > 1})
        if repeated:
            raise ValueError(f'step {index} addresses {repeated} more than once; a session takes one turn at a time')


def workspace_of(session: str, *, working_dir: str, shared: bool) -> str:
    """Where ``session`` works."""
    return working_dir if shared else f'{working_dir.rstrip("/")}/{session}'


@agent_registry.register(AGENT_NAME)
async def run_parallel_agent(agent_input: ParallelAgentInput) -> ParallelAgentOutput:
    """Open every session, run the script, and report what each of them did.

    Reports rather than raises for everything after assembly, as the other agents do:
    a run whose sessions fell over still has checks worth running against what they
    left behind.
    """
    started = time.monotonic()
    error = ''
    steps_run = 0
    manager: SessionManager | None = None
    names: list[str] = []
    workspaces: dict[str, str] = {}
    session_ids: dict[str, str] = {}
    overlaps: list[dict[str, Any]] = []
    followed: list[dict[str, Any]] = []
    replies: dict[str, str] = {}
    reports: dict[str, str] = {}
    rounds = 0

    try:
        context = require_context()
        labels.require_labelling_tracker()
        names = check_sessions(list(agent_input.sessions))
        steps = [ParallelStep.model_validate(raw) for raw in agent_input.script]
        check_script(steps, sessions=names)

        manager = make_worker_manager(
            agent_input.worker_backend,
            default_working_dir=agent_input.working_dir,
            default_config=agent_input.worker_config,
            openai_client_name=agent_input.worker_client_name,
        )
        await manager.setup()

        workspaces = {
            name: workspace_of(name, working_dir=agent_input.working_dir, shared=agent_input.shared_workspace)
            for name in names
        }
        await _make_workspaces(context, sorted(set(workspaces.values())))

        opened: dict[str, Session] = {}
        for name in names:
            opened[name] = await manager.start(name=name, workspace=workspaces[name])
            session_ids[name] = opened[name].session_id
        logger.info('[parallel] opened %d session(s): %s', len(opened), session_ids)

        for index, step in enumerate(steps, start=1):
            follow_ups = {send.to: list(send.follow_up) for send in step.parallel if send.follow_up}
            turns = [(opened[send.to], SessionMessage(content=send.message, kind='message')) for send in step.parallel]
            async for event in manager.run_turns(turns):
                rounds += 1
                name = event.session.name
                overlaps.append({
                    'step': index,
                    'session': name,
                    'inflight': sorted(event.inflight),
                    'kind': event.kind,
                })
                if event.result.has_output:
                    replies[name] = event.result.output
                reports[name] = manager.report(event.result)
                logger.info(
                    '[parallel] step %d: %s finished (%s), still running: %s',
                    index,
                    name,
                    event.kind,
                    sorted(event.inflight) or 'nobody',
                )
                # Sent from inside the loop: this session's next turn starts while the
                # others are still on their first, which is the affordance
                # ``run_turns`` yielding in completion order exists for.
                if event.kind != 'completed':
                    continue
                for text in follow_ups.pop(name, []):
                    result = await manager.send(event.session, SessionMessage(content=text, kind='message'))
                    rounds += 1
                    followed.append({
                        'step': index,
                        'session': name,
                        'while_running': sorted(event.inflight),
                        'kind': result.kind,
                    })
                    if result.has_output:
                        replies[name] = result.output
                    reports[name] = manager.report(result)
            steps_run = index
    except Exception as exc:  # noqa: BLE001 - a failed run still reports what it managed
        error = f'{type(exc).__name__}: {exc}'
        logger.exception('[parallel] agent failed')
    finally:
        if manager is not None:
            try:
                await manager.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown must not mask the outcome
                logger.warning('[parallel] shutdown failed: %s', exc)

    completed = not error and steps_run == len(agent_input.script)
    return ParallelAgentOutput(
        success=completed,
        output='\n\n'.join(f'[{name}] {reply}' for name, reply in sorted(replies.items())),
        stderr=error,
        exit_code=1 if error else 0,
        execution_time=time.monotonic() - started,
        turns=steps_run,
        worker_rounds=rounds,
        reached_max_turns=False,
        run_meta={
            'status': 'error' if error else ('completed' if completed else 'max_turns'),
            'turns': steps_run,
            'messages_sent': rounds,
        },
        sessions=names,
        session_ids=session_ids,
        workspaces=workspaces,
        overlaps=overlaps,
        followed=followed,
        steps_run=steps_run,
        replies=replies,
        reports=reports,
    )


async def _make_workspaces(context: Any, paths: list[str]) -> None:
    """Create each session's directory before its worker starts in it."""
    from nexus.runtime.executors.models.command import RunCommandRequest

    runtime = await context.get_runtime()
    await asyncio.gather(*(runtime.call(RunCommandRequest(command=f'mkdir -p {path}', timeout=60.0)) for path in paths))


__all__ = [
    'AGENT_NAME',
    'DEFAULT_SESSIONS',
    'ParallelAgentInput',
    'ParallelAgentOutput',
    'ParallelSend',
    'ParallelStep',
    'check_script',
    'check_sessions',
    'run_parallel_agent',
    'workspace_of',
]
