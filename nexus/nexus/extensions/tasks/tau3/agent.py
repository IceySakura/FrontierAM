"""Tau3 agent — bridges nexus OpenAI clients to tau3-bench via litellm CustomLLM.

Instead of reimplementing tau3's internal wiring, this agent uses tau3's own
``build_*`` and ``run_simulation`` APIs:

    1. Register nexus OpenAI clients as a litellm CustomLLM provider.
    2. Build a ``TextRunConfig`` pointing to ``nexus/agent`` and ``nexus/user``.
    3. Use ``build_text_orchestrator()`` to construct the full orchestrator.
    4. Use ``run_simulation()`` to execute + evaluate in one step.

Key bridge::

    tau3 generate() → litellm.completion(model="nexus/agent")
        → _NexusLLMProvider.completion()
            → nexus OpenAICompatibleClient.create_completion()
"""

import asyncio
import concurrent.futures
import functools
import logging
import os
import sys
import threading

from typing import Any

import litellm

from litellm import CustomLLM, ModelResponse, register_model

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
    """Suppress tau3's verbose loguru DEBUG output (executed once)."""
    from loguru import logger as loguru_logger  # ty: ignore[unresolved-import]

    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level='INFO')


@functools.cache
def _patch_get_response_cost() -> None:
    """Patch tau2's get_response_cost to silently return 0 for nexus/ models.

    tau2 calls ``litellm.completion_cost()`` after every API call, which
    raises for custom ``nexus/agent`` and ``nexus/user`` models and logs
    an ERROR via loguru on each invocation — flooding the output with
    thousands of irrelevant lines.  This patch makes the function skip
    the cost lookup for nexus-provider models.
    """
    import tau2.utils.llm_utils as llm_utils  # ty: ignore[unresolved-import]

    _original = llm_utils.get_response_cost

    # litellm strips the "nexus/" provider prefix from model names in
    # ModelResponse, so the model field is just "agent" or "user".
    _nexus_models = frozenset({'agent', 'user', 'nexus/agent', 'nexus/user'})

    def _quiet_get_response_cost(response):  # noqa: ANN001
        model = getattr(response, 'model', '') or ''
        if model in _nexus_models:
            return 0.0
        return _original(response)

    llm_utils.get_response_cost = _quiet_get_response_cost  # ty: ignore[invalid-assignment]


# Mapping from tau3 TerminationReason to hytuner TerminationReason.
_TAU3_TO_HYTUNER_TERMINATION_REASON: dict[str, str] = {
    'user_stop': 'ENV_DONE',
    'agent_stop': 'ENV_DONE',
    'max_steps': 'MAX_STEPS',
    'too_many_errors': 'ENV_ERROR',
    'agent_error': 'ENV_ERROR',
    'user_error': 'ENV_ERROR',
    'timeout': 'ENV_ERROR',
    'infrastructure_error': 'ENV_ERROR',
    'context_window_exceeded': 'TRUNCATION',
    'unexpected_error': 'ENV_ERROR',
}


def _map_termination_reason(tau3_reason: Any) -> str:
    """Map a tau3 termination reason to hytuner's TerminationReason string."""
    if tau3_reason is None:
        return ''
    key = tau3_reason.value if hasattr(tau3_reason, 'value') else str(tau3_reason)
    mapped = _TAU3_TO_HYTUNER_TERMINATION_REASON.get(key)
    if mapped is not None:
        return mapped
    logger.warning(f'Unknown tau3 termination reason: {tau3_reason!r} (key={key!r})')
    return key


# Suppress litellm verbose output
litellm.suppress_debug_info = True  # type: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# CustomLLM provider: route litellm calls through nexus OpenAI clients
# ---------------------------------------------------------------------------


# Dedicated thread pool for tau3 orchestrator.run() calls.
# CRITICAL: Must NOT use asyncio's default executor (loop.run_in_executor(None, ...))
# because httpx/anyio internally uses the default executor for DNS resolution etc.
# If tau3 tasks fill all default-executor slots, httpx cannot resolve DNS → deadlock.
# max_workers=256 is safe: ThreadPoolExecutor creates threads lazily on demand,
# so only the actually needed threads are created (= --max-concurrent).
_tau3_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=256,
    thread_name_prefix='tau3',
)


class _NexusLLMProvider(CustomLLM):
    """A litellm CustomLLM that delegates to nexus OpenAI clients.

    Registered as the ``nexus`` provider so that tau3's ``generate()`` calls
    ``litellm.completion(model="nexus/agent", ...)`` route through nexus.
    """

    _active_clients: dict[str, Any] = {}
    _ref_count: int = 0
    _lock: threading.Lock = threading.Lock()
    _main_loop: asyncio.AbstractEventLoop | None = None
    # Per-request timeout in seconds. 0 or None means no per-request timeout
    # (rely on per-run global timeout instead). Override via TAU3_REQUEST_TIMEOUT.
    _request_timeout: float | None = float(os.getenv('TAU3_REQUEST_TIMEOUT', '0')) or None
    # Max retries per single LLM call. Default 5.
    _max_call_retries: int = int(os.getenv('TAU3_MAX_CALL_RETRIES', '5'))

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ) -> ModelResponse:
        """Synchronous completion — bridge async nexus client to sync tau3."""
        if headers is None:
            headers = {}
        # Extract the role from model name: "nexus/agent" → "agent", "nexus/user" → "user".
        # litellm strips the provider prefix before calling this handler, so model may be
        # just "agent" or "user" (without "nexus/"). Use the model string as-is in that case.
        role = model.split('/', 1)[-1] if '/' in model else model
        nexus_client = self._active_clients.get(role)
        if nexus_client is None:
            raise RuntimeError(
                f'No nexus client registered for role "{role}". Available: {list(self._active_clients.keys())}'
            )

        # Log request details
        n_messages = len(messages)
        last_role = messages[-1].get('role', '?') if messages else '?'
        n_tools = len(optional_params.get('tools', []))
        last_content = messages[-1].get('content', '') if messages else ''
        content_preview = (
            (last_content[:200] + '...') if isinstance(last_content, str) and len(last_content) > 200 else last_content
        )
        logger.debug(
            f'[tau3-llm] → {role} | messages={n_messages} (last={last_role}) | '
            f'tools={n_tools} | content_preview={content_preview!r}'
        )

        # Build kwargs for nexus client
        kwargs: dict[str, Any] = {}
        if optional_params.get('tools'):
            kwargs['tools'] = optional_params['tools']
        if optional_params.get('tool_choice'):
            kwargs['tool_choice'] = optional_params['tool_choice']
        for key in ('temperature', 'top_p', 'max_tokens', 'max_completion_tokens', 'reasoning_effort'):
            if key in optional_params:
                kwargs[key] = optional_params[key]

        # Bridge async nexus client to sync context with retry.
        # Each individual LLM call gets up to _max_call_retries attempts with
        # _request_timeout per attempt. This handles transient API errors,
        # timeouts, and empty responses without restarting the whole simulation.
        import time as _time

        response = None
        for attempt in range(1, self._max_call_retries + 1):
            try:
                call_start = _time.time()
                main_loop = _NexusLLMProvider._main_loop
                if main_loop is not None and main_loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        nexus_client.create_completion(messages=messages, **kwargs),
                        main_loop,
                    )
                    response = future.result(timeout=self._request_timeout)
                else:
                    try:
                        asyncio.get_running_loop()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            response = pool.submit(
                                asyncio.run,
                                nexus_client.create_completion(messages=messages, **kwargs),
                            ).result(timeout=self._request_timeout)
                    except RuntimeError:
                        response = asyncio.run(nexus_client.create_completion(messages=messages, **kwargs))
                call_elapsed = _time.time() - call_start

                if response is None:
                    raise RuntimeError('nexus client create_completion() returned None.')
                # ExtendedChatCompletion is guaranteed after None check, but type inference sees unknown
                if not response.choices or response.choices[0].message is None:  # ty: ignore[possibly-missing-attribute]
                    raise RuntimeError('nexus client create_completion() returned empty choices.')

                break  # Success
            except Exception as e:
                call_elapsed = _time.time() - call_start
                if attempt < self._max_call_retries:
                    wait = min(2**attempt, 30)
                    logger.warning(
                        f'[tau3-llm] {role} call attempt {attempt}/{self._max_call_retries} failed '
                        f'after {call_elapsed:.1f}s: {type(e).__name__}: {e}. Retrying in {wait}s...'
                    )
                    _time.sleep(wait)
                else:
                    logger.error(
                        f'[tau3-llm] {role} call failed after {self._max_call_retries} attempts '
                        f'(last: {call_elapsed:.1f}s): {type(e).__name__}: {e}'
                    )
                    raise

        # Convert nexus ExtendedChatCompletion → litellm ModelResponse
        resp_data = response.model_dump()  # ty: ignore[possibly-missing-attribute]
        resp_data['model'] = model
        result = ModelResponse(**resp_data)

        # Normalise tool_calls: None → [] (tau3 iterates over it).
        for choice in result.choices:
            if choice.message.tool_calls is None:
                choice.message.tool_calls = []

        # Log response details
        msg = result.choices[0].message if result.choices else None
        resp_content = (msg.content or '')[:200] if msg else ''
        n_tool_calls = len(msg.tool_calls) if msg and msg.tool_calls else 0
        [tc.function.name for tc in msg.tool_calls] if msg and msg.tool_calls else []
        usage = getattr(result, 'usage', None)
        tokens_in = getattr(usage, 'prompt_tokens', '?') if usage else '?'
        tokens_out = getattr(usage, 'completion_tokens', '?') if usage else '?'
        if n_tool_calls > 0:
            # Log tool call details
            tool_args_preview = []
            for tc in msg.tool_calls[:3]:  # max 3 to avoid log flood  # ty: ignore[not-subscriptable, possibly-missing-attribute]
                args_str = tc.function.arguments or ''
                args_preview = (args_str[:100] + '...') if len(args_str) > 100 else args_str
                tool_args_preview.append(f'{tc.function.name}({args_preview})')
            logger.debug(
                f'[tau3-llm] ← {role} | {call_elapsed:.1f}s | tool_calls={n_tool_calls}: {", ".join(tool_args_preview)} | '
                f'tokens={tokens_in}/{tokens_out}'
            )
        else:
            content_preview = (resp_content[:200] + '...') if len(resp_content) > 200 else resp_content
            logger.debug(
                f'[tau3-llm] ← {role} | {call_elapsed:.1f}s | content={content_preview!r} | tokens={tokens_in}/{tokens_out}'
            )

        return result


# Register the custom provider with litellm (once at import time)
_nexus_provider = _NexusLLMProvider()
litellm.custom_provider_map = [  # type: ignore[assignment]
    {'provider': 'nexus', 'custom_handler': _nexus_provider},
]

# Register dummy pricing to avoid litellm cost-lookup errors
register_model({
    'nexus/agent': {
        'input_cost_per_token': 0.0,
        'output_cost_per_token': 0.0,
        'litellm_provider': 'openai',
        'mode': 'chat',
    },
    'nexus/user': {
        'input_cost_per_token': 0.0,
        'output_cost_per_token': 0.0,
        'litellm_provider': 'openai',
        'mode': 'chat',
    },
})


class Tau3AgentInput(AgentRunInput):
    """Input for tau3 agent — all tau3-specific config is passed through from the task."""

    domain: str = ''  # required — domain name (airline, retail, telecom, banking_knowledge)
    solo_mode: bool = False
    max_steps: int = 200
    max_errors: int = 10
    seed: int | None = None
    add_thinking_tool: bool = False
    task_data: dict[str, Any] | None = None
    retrieval_config: str | None = None  # retrieval strategy for banking_knowledge
    retrieval_config_kwargs: dict[str, Any] | None = None


class Tau3AgentOutput(AgentRunOutput):
    """Output from tau3 agent — raw simulation result for task-layer evaluation."""

    simulation: Any = None
    tau3_task: Any = None
    domain: str = ''
    solo_mode: bool = False
    termination_reason: str | None = None
    trajectory: list[dict[str, Any]] = []
    error: str | None = None
    # Rich simulation metadata (from SimulationRun)
    duration: float | None = None  # simulation wall-clock time in seconds
    agent_cost: float | None = None  # agent LLM cost
    user_cost: float | None = None  # user simulator LLM cost
    simulation_seed: int | None = None


def _report_termination_span(termination_reason: str | None) -> None:
    """Report termination reason via span attributes."""
    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={'_hytuner_termination_reason': termination_reason or 'UNKNOWN'},
        )


@agent_registry.register('tau3_agent')
async def run_tau3_agent(agent_input: Tau3AgentInput) -> Tau3AgentOutput:
    """Run a tau3-bench simulation using tau3's own build/run APIs.

    Uses tau3's layered architecture:
        Layer 2: ``build_text_orchestrator(TextRunConfig, task)``
        Layer 1: ``run_simulation(orchestrator)``
    """
    _configure_loguru()
    _patch_get_response_cost()

    from tau2.data_model.simulation import TextRunConfig  # ty: ignore[unresolved-import]
    from tau2.data_model.tasks import Task  # ty: ignore[unresolved-import]
    from tau2.runner.build import build_text_orchestrator  # ty: ignore[unresolved-import]

    # 1. Validate input
    if agent_input.task_data is None:
        raise ValueError('task_data is required. Ensure generator provides task_data for every task.')
    if not agent_input.domain:
        raise ValueError('domain is required. Set it explicitly in task config.')

    domain = agent_input.domain
    task = Task.model_validate(agent_input.task_data)

    # 2. Register nexus clients with the custom provider (ref-counted for concurrency)
    ctx = require_context()
    agent_client = ctx.get_openai_client('main')
    with _NexusLLMProvider._lock:
        _NexusLLMProvider._active_clients['agent'] = agent_client
        _NexusLLMProvider._ref_count += 1
        # Store the main event loop so completion() can dispatch async calls
        # back to it via run_coroutine_threadsafe (avoids httpx connection leaks).
        _NexusLLMProvider._main_loop = asyncio.get_event_loop()

    if not agent_input.solo_mode:
        try:
            user_client = ctx.get_openai_client('user')
            _NexusLLMProvider._active_clients['user'] = user_client
        except KeyError:
            logger.warning("No 'user' OpenAI client configured. UserSimulator will use litellm defaults.")

    # 3. Build TextRunConfig — routes LLM calls through "nexus/agent" and "nexus/user"
    user_llm = 'nexus/user' if 'user' in _NexusLLMProvider._active_clients else 'gpt-4.1'

    config_kwargs: dict[str, Any] = {
        'domain': domain,
        'agent': 'llm_agent_solo' if agent_input.solo_mode else 'llm_agent',
        'llm_agent': 'nexus/agent',
        'llm_args_agent': {},
        'user': 'dummy_user' if agent_input.solo_mode else 'user_simulator',
        'llm_user': user_llm,
        'llm_args_user': {},
        'max_steps': agent_input.max_steps,
        'max_errors': agent_input.max_errors,
    }
    if agent_input.retrieval_config is not None and domain == 'banking_knowledge':
        config_kwargs['retrieval_config'] = agent_input.retrieval_config
    if agent_input.retrieval_config_kwargs is not None and domain == 'banking_knowledge':
        config_kwargs['retrieval_config_kwargs'] = agent_input.retrieval_config_kwargs

    text_config = TextRunConfig(**config_kwargs)

    # Optionally add thinking tool
    if agent_input.add_thinking_tool:
        logger.info('add_thinking_tool is set but not yet supported via TextRunConfig; ignoring.')

    # 4. Build orchestrator and run simulation (sync — in a thread)
    error = None
    trajectory = []
    termination_reason = None
    simulation = None

    logger.debug(
        f'[tau3-sim] Starting: domain={domain}, task_id={task.id}, '
        f'solo={agent_input.solo_mode}, max_steps={agent_input.max_steps}, '
        f'retrieval={agent_input.retrieval_config or "none"}'
    )

    try:
        orchestrator = build_text_orchestrator(
            text_config,
            task,
            seed=agent_input.seed,
        )

        loop = asyncio.get_event_loop()
        simulation = await loop.run_in_executor(_tau3_executor, orchestrator.run)
        trajectory = [msg.model_dump() for msg in simulation.messages]  # ty: ignore[not-iterable]
        termination_reason = _map_termination_reason(simulation.termination_reason)

        # Log simulation summary
        n_messages = len(simulation.messages)  # ty: ignore[invalid-argument-type]
        n_agent = sum(1 for m in simulation.messages if getattr(m, 'role', '') == 'assistant')  # ty: ignore[not-iterable]
        n_user = sum(1 for m in simulation.messages if getattr(m, 'role', '') == 'user')  # ty: ignore[not-iterable]
        n_tool = sum(1 for m in simulation.messages if getattr(m, 'role', '') == 'tool')  # ty: ignore[not-iterable]
        dur = getattr(simulation, 'duration', None)
        dur_str = f'{dur:.1f}s' if dur else '?'
        logger.info(
            f'[tau3-sim] Done: task_id={task.id}, termination={termination_reason}, '
            f'msgs={n_messages} (agent={n_agent}, user={n_user}, tool={n_tool}), '
            f'duration={dur_str}'
        )
    except ContextLengthExceededError as e:
        logger.warning(f'Tau3 context length exceeded, stopping gracefully: {e}')
        termination_reason = 'TRUNCATION'
        error = str(e)
    except Exception as e:
        logger.error(f'[tau3-sim] Simulation failed for task_id={task.id}: {type(e).__name__}: {e}')
        termination_reason = 'UNHANDLED_ERROR'
        error = str(e)
    finally:
        with _NexusLLMProvider._lock:
            _NexusLLMProvider._ref_count -= 1
            if _NexusLLMProvider._ref_count <= 0:
                _NexusLLMProvider._active_clients.clear()
                _NexusLLMProvider._ref_count = 0

    _report_termination_span(termination_reason)
    if error is None:
        logger.info(f'Tau3 simulation finished: termination={termination_reason}')

    return Tau3AgentOutput(
        simulation=simulation,
        tau3_task=task,
        domain=domain,
        solo_mode=agent_input.solo_mode,
        termination_reason=termination_reason,
        trajectory=trajectory,
        error=error,
        duration=getattr(simulation, 'duration', None) if simulation else None,
        agent_cost=getattr(simulation, 'agent_cost', None) if simulation else None,
        user_cost=getattr(simulation, 'user_cost', None) if simulation else None,
        simulation_seed=getattr(simulation, 'seed', None) if simulation else None,
    )
