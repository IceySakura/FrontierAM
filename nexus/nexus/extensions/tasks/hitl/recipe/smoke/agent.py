"""The smoke agent: the same conversation, driven two ways.

``agents/agent_with_user`` is the production loop. This one exists because the two
things a smoke run needs are exactly what a production loop must not have: a way to
make the user say a particular thing at a particular turn, and a way to skip the model
entirely.

``mode='live'``
    The user simulator runs for real, and ``injections`` puts a line into its own
    history at a named turn. Tests what a prompt causes: whether the user, told it has
    a search tool, actually reaches for it.

``mode='replay'``
    No model call at all. ``script`` lists the tool calls to make, in order, and each
    is handed to the same dispatcher a live turn uses. Tests the worker-side chain
    deterministically -- the same input every run, which is what a bug like a turn
    timing out needs.

Either way the worker is a real Claude Code in a real sandbox: what is replaced is the
user, never the thing being checked.

What is shared with the production agent, rather than copied:
:class:`~nexus.extensions.tasks.hitl.agents.agent_with_user.RunState` and
``build_handlers``, so sending, accepting and reporting are one implementation. What is
written out here is the turn loop, because that is what differs.
"""

import json
import logging
import time

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from nexus.core import agent_registry, require_context
from nexus.core.types.message import ChatCompletionMessageToolCall
from nexus.extensions.tasks.hitl.actor import mcp
from nexus.extensions.tasks.hitl.actor.base import SessionManager
from nexus.extensions.tasks.hitl.actor.factory import make_worker_manager
from nexus.extensions.tasks.hitl.actor.rendering import WORKER_REPORT_PROTOCOL
from nexus.extensions.tasks.hitl.agents.agent_with_user import (
    WORKER_NAME,
    AgentWithUserInput,
    AgentWithUserOutput,
    RunState,
    accepted_text,
    build_handlers,
)
from nexus.extensions.tasks.hitl.trace import labels
from nexus.extensions.tasks.hitl.user_agent import queue as queue_mod
from nexus.extensions.tasks.hitl.user_agent import skills as skills_mod
from nexus.extensions.tasks.hitl.user_agent.compact import Budget, compact
from nexus.extensions.tasks.hitl.user_agent.dispatch import (
    Dispatcher,
    call_name,
    check_handlers,
    read_response,
    tool_message,
)
from nexus.extensions.tasks.hitl.user_agent.language import detect
from nexus.extensions.tasks.hitl.user_agent.personas import get_persona, load_persona_dirs
from nexus.extensions.tasks.hitl.user_agent.prompts import (
    NUDGE,
    build_opening_message,
    build_system_prompt,
    environment_facts,
    reminder,
    reminder_message,
    worker_command_lines,
    worker_skills_section,
)
from nexus.extensions.tasks.hitl.user_agent.reminders import Reminders
from nexus.extensions.tasks.hitl.user_agent.skills import catalogue
from nexus.extensions.tasks.hitl.user_agent.tools import toolset


logger = logging.getLogger(__name__)


AGENT_NAME = 'hitl/smoke_agent'

Mode = Literal['live', 'replay']

USER_LABEL = labels.user_label()


class ScriptedTurn(BaseModel):
    """One turn of a replay: the tool to call and what to call it with."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Injection(BaseModel):
    """A line put into the user's history before a given live turn.

    ``turn`` is 1-based, matching what the log prints. A turn beyond the cap never
    fires, which :func:`check_injections` refuses rather than silently dropping.
    """

    turn: int
    text: str


class SmokeAgentInput(AgentWithUserInput):
    """The production input plus how this run is driven."""

    mode: Mode = 'live'
    injections: list[dict[str, Any]] = Field(default_factory=list)
    """Lines to put into the user's own history, as :class:`Injection` fields.
    ``live`` only."""
    script: list[dict[str, Any]] = Field(default_factory=list)
    """The tool calls to make, as :class:`ScriptedTurn` fields. ``replay`` only."""
    persona_dirs: list[str] = Field(default_factory=list)
    """Directories holding persona files, read before ``persona`` is resolved.

    Declared here rather than inherited: the generic agent does not take it, so a
    smoke run naming a persona from outside the package needs it. The generator reads
    the same directories to refuse an unknown name before the batch starts.
    """


class SmokeAgentOutput(AgentWithUserOutput):
    """The production output plus what drove this run.

    ``mode`` is here because the numbers are not comparable across modes: a replay
    makes no user-side model call, so its ``turns`` counts script steps and its usage
    summary holds nothing for the user.
    """

    mode: str = 'live'
    scripted_steps: int = 0
    injections_fired: int = 0


def check_injections(injections: Sequence[Injection], *, cap: int) -> None:
    """Refuse an injection no turn will reach, or two on one turn.

    Both would leave a check looking for behaviour nothing asked for, and report a
    failing mechanism when the ask never arrived.
    """
    turns = [injection.turn for injection in injections]
    late = sorted({turn for turn in turns if turn > cap})
    if late:
        raise ValueError(f'injections name turn(s) {late}, past the cap of {cap}; they would never fire')
    early = sorted({turn for turn in turns if turn < 1})
    if early:
        raise ValueError(f'injections name turn(s) {early}; turns are 1-based')
    repeated = sorted({turn for turn in turns if turns.count(turn) > 1})
    if repeated:
        raise ValueError(f'injections name turn(s) {repeated} more than once; put both lines in one injection')


def scripted_call(step: ScriptedTurn, *, index: int) -> ChatCompletionMessageToolCall:
    """``step`` as the tool call a dispatcher takes.

    The same shape a provider returns, so the dispatcher, the handlers and the history
    entry are the live path's.
    """
    from openai.types.chat.chat_completion_message_tool_call import Function

    return ChatCompletionMessageToolCall(
        id=f'replay_{index}',
        type='function',
        function=Function(name=step.tool, arguments=json.dumps(step.args)),
    )


def check_script(script: Sequence[ScriptedTurn], *, offered: Sequence[str]) -> None:
    """Refuse a scripted call the tool set does not offer.

    The dispatcher reports an unknown tool as content, which a model reads and works
    around. A script cannot read anything, so the step is simply lost.
    """
    if not script:
        raise ValueError('a replay needs at least one scripted turn')
    unknown = sorted({step.tool for step in script if step.tool not in set(offered)})
    if unknown:
        raise ValueError(f'the script calls {unknown}, which this run does not offer. It has: {sorted(offered)}')


@agent_registry.register(AGENT_NAME)
async def run_smoke_agent(agent_input: SmokeAgentInput) -> SmokeAgentOutput:
    """Drive one worker, live or from a script, and report what happened.

    Reports rather than raises for everything after assembly, as the production agent
    does: a smoke run whose conversation fell over still has checks worth running
    against what it left behind.
    """
    started = time.monotonic()
    queue = queue_mod.from_task_input(
        requirements=agent_input.requirements,
        user_prompt=agent_input.user_prompt,
    )

    error = ''
    turns = 0
    compactions = 0
    fired = 0
    stopped = False
    state: RunState | None = None
    manager: SessionManager | None = None
    script: list[ScriptedTurn] = []
    try:
        context = require_context()
        labels.require_labelling_tracker()
        load_persona_dirs(agent_input.persona_dirs)
        natural_language = detect('\n'.join(queue.segments))
        persona = get_persona(agent_input.persona).for_language(natural_language.code)
        skills = skills_mod.load_user_skills(external=agent_input.skill_dirs)
        if agent_input.mcp_servers:
            # Resolved in the sandbox: a bare interpreter here would be the host's,
            # and only the venv the run built carries this package.
            servers = await mcp.resolve_in_sandbox(await context.get_runtime(), agent_input.mcp_servers)
            loaded = await context.load_mcp_tools(extra_servers=servers)
            logger.info('[smoke] mcp servers=%d tools=%s', len(servers), sorted(loaded))
        tools = toolset.build(
            toolset.tool_names(persona.tools, skills, agent_input.extra_tools),
            skills=skills,
        )

        cap = agent_input.max_turns
        injections = [Injection.model_validate(raw) for raw in agent_input.injections]
        script = [ScriptedTurn.model_validate(raw) for raw in agent_input.script]
        _check_mode(agent_input.mode, injections=injections, script=script)
        if agent_input.mode == 'replay':
            check_script(script, offered=tools.schema_names)
        else:
            check_injections(injections, cap=cap)
        by_turn: Mapping[int, str] = {injection.turn: injection.text for injection in injections}

        manager = make_worker_manager(
            agent_input.worker_backend,
            default_working_dir=agent_input.working_dir,
            default_config=agent_input.worker_config,
            openai_client_name=agent_input.worker_client_name,
        )
        await manager.setup()
        session = await manager.start(name=WORKER_NAME, workspace=agent_input.working_dir)
        await skills_mod.install_skills(await context.get_runtime(), skills)

        state = RunState(
            queue=queue,
            manager=manager,
            session=session,
            natural_language=natural_language,
        )
        handlers = {name: handler for name, handler in build_handlers(state).items() if name in tools.schema_names}
        check_handlers(handlers, set(tools.schema_names))
        dispatcher = Dispatcher(executor=tools.executor, handlers=handlers)

        client = labels.enable_labels(context.get_openai_client(agent_input.user_client_name))
        history = [
            {
                'role': 'system',
                'content': build_system_prompt(
                    persona,
                    usage_docs=(*tools.usage_docs, ('worker_report', WORKER_REPORT_PROTOCOL)),
                    environment=environment_facts(
                        working_dir=agent_input.working_dir, workers=[WORKER_NAME], product=manager.product
                    ),
                    worker_skills=worker_skills_section(manager.commands, prefix=manager.command_prefix),
                    may_edit=tools.may_edit,
                ),
            },
            {
                'role': 'user',
                'content': build_opening_message(
                    queue.current,
                    language=state.natural_language.name,
                    catalogue=catalogue(skills),
                ),
            },
        ]
        context.tool_state.update(toolset.tool_state(tools, working_dir=agent_input.working_dir))

        reminders = Reminders(
            recalls=persona.recalls,
            light=agent_input.reminder_light,
            heavy_every=agent_input.reminder_heavy_every,
            light_every=agent_input.reminder_light_every,
            catalogue=catalogue(skills),
            worker_commands=worker_command_lines(manager.commands, prefix=manager.command_prefix),
        )
        budget = Budget(window=agent_input.context_window, pct=agent_input.compact_pct)

        logger.info(
            '[smoke] mode=%s persona=%s cap=%d script=%d injections=%d %s',
            agent_input.mode,
            persona.name,
            cap,
            len(script),
            len(injections),
            tools.describe(),
        )

        with labels.labelled(client, USER_LABEL):
            steps = len(script) if agent_input.mode == 'replay' else cap
            for turn in range(1, min(steps, cap) + 1):
                turns = turn

                if agent_input.mode == 'replay':
                    calls = [scripted_call(script[turn - 1], index=turn)]
                    response = None
                    logger.info('[smoke] turn %d replays %s', turn, script[turn - 1].tool)
                else:
                    if text := by_turn.get(turn):
                        history.append(reminder_message(text))
                        fired += 1
                        logger.info('[smoke] turn %d injected: %s', turn, text)
                    response = await client.create_completion(messages=history, tools=list(tools.schemas))
                    message, calls = read_response(response)
                    history.append(message)

                if not calls:
                    history.append({'role': 'user', 'content': reminder(NUDGE)})
                    continue

                # Every call is answered before anything else joins the history.
                appended: list[dict[str, Any]] = []
                for call in calls:
                    result = await dispatcher.dispatch(call)
                    history.append(tool_message(call, result.content))
                    appended.extend(result.messages)
                    recall = reminders.after_tool(call_name(call))
                    if recall:
                        appended.append(reminder_message(recall))
                history.extend(appended)

                if state.stopped:
                    stopped = True
                    break

                # Only a live turn has a reported prompt size to compare against, and
                # only a live turn appends enough to need compacting.
                if response is not None and budget.exceeded(response):
                    history = await compact(
                        history,
                        client=client,
                        queue=queue,
                        language=state.natural_language.name,
                        prompt=agent_input.compact_prompt,
                    )
                    compactions += 1
                    reminders.compacted()
    except Exception as exc:  # noqa: BLE001 - a failed run still reports what it managed
        error = f'{type(exc).__name__}: {exc}'
        logger.exception('[smoke] agent failed')
    finally:
        if manager is not None:
            try:
                await manager.shutdown()
            except Exception as exc:  # noqa: BLE001 - shutdown must not mask the outcome
                logger.warning('[smoke] worker shutdown failed: %s', exc)

    # A replay ends when its script does, which is a finished run rather than a run
    # that ran out of turns.
    completed = stopped or (agent_input.mode == 'replay' and not error and turns == len(script))
    return SmokeAgentOutput(
        success=error == '' and completed,
        output=accepted_text(state.accepted) if state else '',
        stderr=error,
        exit_code=1 if error else 0,
        execution_time=time.monotonic() - started,
        turns=turns,
        worker_rounds=state.worker_rounds if state else 0,
        reached_max_turns=not completed and not error,
        run_meta={
            'status': 'error' if error else ('completed' if completed else 'max_turns'),
            'stop_reason': state.stop_reason if state else '',
            'completed_steps': len(state.accepted) if state else 0,
            'total_steps': queue.total,
            'turns': turns,
            'messages_sent': state.messages_sent if state else 0,
            'compactions': compactions,
        },
        mode=agent_input.mode,
        scripted_steps=len(script),
        injections_fired=fired,
    )


def _check_mode(mode: str, *, injections: Sequence[Injection], script: Sequence[ScriptedTurn]) -> None:
    """Refuse a field the other mode reads, which this one would ignore."""
    if mode == 'replay' and injections:
        raise ValueError('injections are read in live mode only; a replay has no model to instruct')
    if mode == 'live' and script:
        raise ValueError('script is read in replay mode only; set mode="replay" to run it')


__all__ = [
    'AGENT_NAME',
    'Injection',
    'Mode',
    'ScriptedTurn',
    'SmokeAgentInput',
    'SmokeAgentOutput',
    'check_injections',
    'check_script',
    'run_smoke_agent',
    'scripted_call',
]
