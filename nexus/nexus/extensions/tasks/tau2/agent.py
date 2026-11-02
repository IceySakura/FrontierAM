"""Tau2 agent — bridges nexus OpenAI clients to tau2-bench's ``completion_fn``.

This agent assembles tau2's LLMAgent/LLMSoloAgent + UserSimulator + Orchestrator,
runs the full simulation loop, and returns the raw result for the task layer
to evaluate.

Key bridge::

    nexus OpenAICompatibleClient.create_completion()  →  tau2 completion_fn(**kwargs)

The tau2 ``completion_fn`` signature: ``async def completion_fn(**kwargs) -> ModelResponse``
where kwargs include model, messages, tools, tool_choice, etc.
"""

import functools
import logging
import os
import sys

from typing import Any

import litellm

from litellm import register_model

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    agent_registry,
    require_context,
)
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


@functools.cache
def _configure_loguru() -> None:
    """Suppress tau2's verbose loguru DEBUG output (executed once).

    tau2 uses loguru (not stdlib logging), and hytuner bypasses tau2's
    run_tasks() entry point which normally configures loguru to INFO level.
    """
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level='INFO')


# Mapping from tau2 TerminationReason to hytuner TerminationReason.
# tau2 uses its own enum (user_stop, agent_stop, max_steps, etc.),
# which must be aligned with hytuner's enum for correct metric computation
# (e.g. complete_ratio relies on ENV_DONE / SUCCESS).
_TAU2_TO_HYTUNER_TERMINATION_REASON: dict[str, str] = {
    'user_stop': 'ENV_DONE',
    'agent_stop': 'ENV_DONE',
    'max_steps': 'MAX_STEPS',
    'too_many_errors': 'ENV_ERROR',
    'agent_error': 'ENV_ERROR',
    'user_error': 'ENV_ERROR',
}


def _map_termination_reason(tau2_reason: Any) -> str:
    """Map a tau2 termination reason to hytuner's TerminationReason string.

    tau2's ``TerminationReason`` is a ``(str, Enum)`` with lowercase values::

        USER_STOP  = "user_stop"
        AGENT_STOP = "agent_stop"
        ...

    Because it inherits ``str``, both ``.value`` and ``str(member)`` return
    the lowercase value (e.g. ``"user_stop"``).  We use ``.value`` directly.
    """
    if tau2_reason is None:
        return ''

    # tau2 TerminationReason is (str, Enum): .value gives the lowercase string key
    key = tau2_reason.value if hasattr(tau2_reason, 'value') else str(tau2_reason)
    mapped = _TAU2_TO_HYTUNER_TERMINATION_REASON.get(key)
    if mapped is not None:
        return mapped

    logger.warning(f'Unknown tau2 termination reason: {tau2_reason!r} (key={key!r})')
    return key


# Suppress litellm verbose output
litellm.suppress_debug_info = True  # ty: ignore[invalid-assignment]

# Register a dummy model so tau2's cost calculation doesn't fail
# when we use "dummy" as model name (actual calls go through nexus client)
register_model({
    'dummy': {
        'input_cost_per_token': 0.0,
        'output_cost_per_token': 0.0,
        'litellm_provider': 'openai',
        'mode': 'chat',
    },
})


class Tau2AgentInput(AgentRunInput):
    """Input for tau2 agent — all tau2-specific config is passed through from the task."""

    solo_mode: bool = False
    max_steps: int = 200
    max_errors: int = 10
    seed: int | None = None
    add_thinking_tool: bool = False
    task_data: dict[str, Any] | None = None  # tau2 Task dict, always provided by generator
    db_path: str | None = None  # per-task DB path (e.g. for RL data)
    strip_initial_assistant_message: bool = False  # Strip tau2's default greeting from agent messages (for training)


class Tau2AgentOutput(AgentRunOutput):
    """Output from tau2 agent — raw simulation result for task-layer evaluation.

    The ``simulation`` and ``tau2_task`` objects are passed as-is
    (arbitrary_types_allowed from base class).
    """

    simulation: Any = None
    tau2_task: Any = None
    domain: str = ''
    solo_mode: bool = False
    termination_reason: str | None = None
    trajectory: list[dict[str, Any]] = []
    error: str | None = None


def make_env_constructor(domain: str, db_path: str):
    """Create a tau2 environment constructor that loads a per-task DB.

    Returns a callable ``constructor(**kwargs)`` that builds a tau2 environment
    with the specified DB injected.  This factory is shared by:

    - The agent layer (``_build_environment``) for running simulations.
    - The task layer (``evaluate_simulation``'s ``environment_constructor``).
    """
    from tau2.registry import registry as tau2_registry

    env_fn = tau2_registry.get_env_constructor(domain)

    def constructor(**kwargs):
        kwargs.setdefault('db', _load_domain_db(domain, db_path))
        return env_fn(**kwargs)

    return constructor


def _build_environment(domain: str, db_path: str, solo_mode: bool):
    """Build a tau2 environment with a per-task DB."""
    constructor = make_env_constructor(domain, db_path)
    kwargs: dict[str, Any] = {}
    if solo_mode:
        kwargs['solo_mode'] = solo_mode
    return constructor(**kwargs)


def _resolve_db_path(db_path: str) -> str:
    """Resolve a relative *db_path* against the ``TAU2_DATA_DIR`` environment variable.

    Raises ``ValueError`` if *db_path* is an absolute path (all db_path values
    must be relative and resolved via ``TAU2_DATA_DIR``).
    """
    if os.path.isabs(db_path):
        raise ValueError(
            f'db_path must be a relative path, got absolute path: {db_path}. '
            'Use a relative path and set TAU2_DATA_DIR to the base directory.'
        )
    data_dir = os.getenv('TAU2_DATA_DIR')
    if not data_dir:
        raise ValueError(
            'TAU2_DATA_DIR environment variable is required to resolve db_path. '
            f'Cannot resolve relative db_path: {db_path}'
        )
    return os.path.join(data_dir, db_path)


def _load_domain_db(domain: str, db_path: str):
    """Load a domain-specific DB (FlightDB / RetailDB / TelecomDB) from *db_path*.

    Each domain has its own DB class; the base class provides ``load()``.
    If *db_path* is relative, it is resolved against ``TAU2_DATA_DIR``.
    """
    db_path = _resolve_db_path(db_path)

    if domain == 'airline':
        from tau2.domains.airline.data_model import FlightDB

        return FlightDB.load(db_path)
    if domain == 'retail':
        from tau2.domains.retail.data_model import RetailDB

        return RetailDB.load(db_path)
    if domain == 'telecom':
        from tau2.domains.telecom.data_model import TelecomDB

        return TelecomDB.load(db_path)
    raise ValueError(f'Custom db_path is not supported for domain: {domain}')


# The default greeting message tau2's Orchestrator injects as the first
# assistant turn when message_history is empty (non-solo mode).
_TAU2_DEFAULT_GREETING = 'Hi! How can I help you today?'


def _make_nexus_completion_fn(client, *, strip_initial_assistant: bool = False):
    """Wrap a nexus ``OpenAICompatibleClient`` into a tau2-compatible async ``completion_fn``.

    Converts the nexus ``ExtendedChatCompletion`` response to a litellm
    ``ModelResponse``, including ``reasoning_content`` injection.

    If *strip_initial_assistant* is True, the tau2 default greeting
    (``"Hi! How can I help you today?"``) that the Orchestrator prepends
    after the system message is removed before forwarding to the nexus
    client.  This is useful during training where the greeting is not
    generated by the model and would pollute the training signal.
    """
    from litellm.main import ModelResponse

    async def completion_fn(**kwargs):
        messages = kwargs.pop('messages', [])
        kwargs.pop('model', None)  # discard tau2's "dummy" model; nexus client uses its own
        kwargs.pop('num_retries', None)  # litellm-specific, not needed by nexus

        # Optionally strip tau2's default initial assistant greeting.
        # The message sequence from tau2 looks like:
        #   [system, assistant("Hi! How can I help you today?"), user(...), ...]
        # During training we want to remove that assistant message so the model
        # only sees messages it actually generated.
        if strip_initial_assistant and len(messages) >= 2:
            second = messages[1]
            if (
                second.get('role') == 'assistant'
                and isinstance(second.get('content'), str)
                and second['content'].strip() == _TAU2_DEFAULT_GREETING
            ):
                messages = [messages[0]] + messages[2:]

        # tau2's to_litellm_messages adds a non-standard top-level 'name' field
        # to each tool_call dict.  HyTunerClient's history consistency check
        # compares messages by value, so the extra field would cause a mismatch.
        # Strip it here before forwarding to the nexus client.
        for msg in messages:
            for tc in msg.get('tool_calls') or []:
                tc.pop('name', None)

        # Call nexus client
        response = await client.create_completion(messages=messages, **kwargs)

        if response is None:
            logger.error(
                '[tau2_agent] create_completion returned None! '
                f'messages count={len(messages)}, kwargs keys={list(kwargs.keys())}'
            )
            raise RuntimeError(
                'HyTunerClient.create_completion() returned None. '
                'Check vLLM server_manager.generate() and token_output.'
            )

        # Defensive check: ensure choices exist and have a message
        if not response.choices or response.choices[0].message is None:
            logger.error(f'[tau2_agent] create_completion returned empty choices or None message! response={response}')
            raise RuntimeError(
                'HyTunerClient.create_completion() returned empty choices. The model may have generated 0 tokens.'
            )

        # Convert nexus ExtendedChatCompletion
        resp_data = response.model_dump()
        resp_data['model'] = 'dummy'
        model_response = ModelResponse(**resp_data)

        # Ensure tool_calls is [] instead of None.
        # tau2's agenerate (llm_utils:309) iterates tool_calls and crashes with
        # "'NoneType' object is not iterable" when it's None.
        # We cannot modify the external tau2 package, so normalise here.
        for choice in model_response.choices:
            if choice.message.tool_calls is None:
                choice.message.tool_calls = []

        # Inject reasoning_content from ExtendedMessage into litellm Message
        # (litellm's Message lacks this field, but tau2 reads it for chain-of-thought)
        for src, dst in zip(response.choices, model_response.choices, strict=False):
            dst.message.reasoning_content = getattr(src.message, 'reasoning_content', None)

        return model_response

    return completion_fn


def _report_termination_span(termination_reason: str | None) -> None:
    """Report termination reason to HyTuner via span attributes.

    This enables HyTuner's metrics processor to correctly compute
    complete_ratio, truncation_ratio, and other training metrics.
    """
    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                '_hytuner_termination_reason': termination_reason or 'UNKNOWN',
            },
        )


@agent_registry.register('tau2_agent')
async def run_tau2_agent(agent_input: Tau2AgentInput) -> Tau2AgentOutput:
    """Run a tau2-bench simulation and return raw result for task-layer evaluation.

    Steps:
        1. Create tau2-compatible completion functions from nexus OpenAI clients.
        2. Load tau2 task + environment.
        3. Assemble LLMAgent/LLMSoloAgent + UserSimulator.
        4. Run the Orchestrator simulation loop.
    """
    _configure_loguru()

    from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
    from tau2.environment.tool import Tool
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import DummyUser, UserSimulator

    # 1. Get nexus context and clients
    ctx = require_context()
    agent_client = ctx.get_openai_client('main')  # required
    agent_completion_fn = _make_nexus_completion_fn(
        agent_client,
        strip_initial_assistant=agent_input.strip_initial_assistant_message,
    )

    # User client — only needed for multi-turn mode
    user_completion_fn = None
    if not agent_input.solo_mode:
        try:
            user_client = ctx.get_openai_client('user')
            user_completion_fn = _make_nexus_completion_fn(user_client)
        except KeyError:
            logger.warning("No 'user' OpenAI client configured. UserSimulator will use litellm with default settings.")

    # 2. Load tau2 task and environment
    from tau2.data_model.tasks import Task

    if agent_input.task_data is not None:
        task = Task.model_validate(agent_input.task_data)
    else:
        raise ValueError('task_data is required. Ensure generator provides task_data for every task.')

    # Extract domain from task_data
    try:
        domain = agent_input.task_data['user_scenario']['instructions']['domain']
    except (KeyError, TypeError):
        task_id = agent_input.task_data.get('id', 'unknown') if agent_input.task_data else 'unknown'
        raise ValueError(
            f"Task {task_id} is missing 'user_scenario.instructions.domain'. "
            'Each task must specify its domain in the task data.'
        ) from None

    if agent_input.db_path is None:
        raise ValueError('db_path is required. Ensure generator provides db_path for every task.')

    environment = _build_environment(domain, agent_input.db_path, agent_input.solo_mode)
    agent_policy_doc = environment.get_policy()
    tools: list[Tool] = environment.get_tools()
    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = []

    # Optionally add thinking tool
    if agent_input.add_thinking_tool:

        def think(thoughts: str):
            """Use this tool to think. The thoughts will be visible in the history.
            Only use this tool when necessary.
            """
            return 'Your thoughts are recorded. Please continue your work.'

        from tau2.environment.tool import as_tool

        tools.append(as_tool(think))

    # 3. Create agent and user
    # Model name is "dummy" because actual API calls go through the nexus completion_fn
    if agent_input.solo_mode:
        agent = LLMSoloAgent(
            tools=tools + user_tools,
            domain_policy=agent_policy_doc,
            llm='dummy',
            task=task,
            completion_fn=agent_completion_fn,  # ty: ignore[unknown-argument]
        )
        user = DummyUser()
    else:
        agent = LLMAgent(
            tools=tools,
            domain_policy=agent_policy_doc,
            llm='dummy',
            completion_fn=agent_completion_fn,  # ty: ignore[unknown-argument]
        )
        if user_completion_fn is not None:
            # User also uses nexus client via completion_fn
            user = UserSimulator(
                tools=user_tools if len(user_tools) > 0 else None,
                instructions=str(task.user_scenario),
                llm='dummy',
                completion_fn=user_completion_fn,  # ty: ignore[unknown-argument]
            )
        else:
            # Fallback: user uses litellm directly (needs OPENAI_API_KEY env)
            user = UserSimulator(
                tools=user_tools if len(user_tools) > 0 else None,
                instructions=str(task.user_scenario),
                llm='gpt-4.1',
            )

    # 4. Run orchestrator
    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=agent_input.max_steps,
        max_errors=agent_input.max_errors,
        seed=agent_input.seed,
        solo_mode=agent_input.solo_mode,
    )

    error = None
    trajectory = []
    termination_reason = None

    try:
        simulation = await orchestrator.arun()  # ty: ignore[unresolved-attribute]
        trajectory = [msg.model_dump() for msg in simulation.messages]
        termination_reason = _map_termination_reason(simulation.termination_reason)
    except ContextLengthExceededError as e:
        logger.warning(f'Tau2 context length exceeded, stopping gracefully: {e}')
        trajectory = [msg.model_dump() for msg in orchestrator.get_trajectory()]
        termination_reason = 'TRUNCATION'
        error = str(e)
        _report_termination_span(termination_reason)
        return Tau2AgentOutput(
            simulation=None,
            tau2_task=task,
            domain=domain,
            solo_mode=agent_input.solo_mode,
            termination_reason=termination_reason,
            trajectory=trajectory,
            error=error,
        )
    except Exception as e:
        logger.error(f'Tau2 simulation error: {type(e).__name__}: {e}')
        trajectory = [msg.model_dump() for msg in orchestrator.get_trajectory()]
        termination_reason = termination_reason or 'UNHANDLED_ERROR'
        error = str(e)
        _report_termination_span(termination_reason)
        return Tau2AgentOutput(
            simulation=None,
            tau2_task=task,
            domain=domain,
            solo_mode=agent_input.solo_mode,
            termination_reason=termination_reason,
            trajectory=trajectory,
            error=error,
        )

    logger.info(f'Tau2 simulation finished: termination={termination_reason}')
    _report_termination_span(termination_reason)

    return Tau2AgentOutput(
        simulation=simulation,
        tau2_task=task,
        domain=domain,
        solo_mode=agent_input.solo_mode,
        termination_reason=termination_reason,
        trajectory=trajectory,
        error=error,
    )
