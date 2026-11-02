"""Tau3 benchmark task — runs a customer-service simulation and evaluates the result.

The task delegates execution to the tau3 agent, then uses tau3's evaluator
to compute a reward score (0.0–1.0).
"""

import asyncio
import concurrent.futures
import json
import logging
import os

from types import MethodType
from typing import Any

from openai import AsyncOpenAI

from nexus.core import (
    AgentRunOutput,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)


logger = logging.getLogger(__name__)


_NL_ASSERTION_SYSTEM_PROMPT = """
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually.

FORMAT
- Your response should be a JSON object with the following fields:
- `reasoning`: a short explanation for your classification
- `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
- `expectedOutcome`: repeat the expectation from the input that you are grading

Example response structure:
{
    "results": [
        {
            "expectedOutcome": "<one of the expected outcomes from the input>",
            "reasoning": "<reasoning trace>",
            "metExpectation": <false or true>
        }
    ]
}
"""

_NL_ASSERTION_PATCHED = False


def _patch_taiji_openapi_port(nexus_client: Any) -> int | None:
    """Patch a task-local TaijiClient instance to use :1081/openapi."""
    try:
        from nexus.extensions.openai_clients.taiji import TaijiClient
    except ImportError:
        return None

    if not isinstance(nexus_client, TaijiClient) or nexus_client.url_pattern != 'openapi':
        return None
    if getattr(nexus_client, '_tau3_openapi_port_patched', False):
        return int(getattr(nexus_client, '_tau3_openapi_port', 1081))

    openapi_port = int(os.getenv('TAU3_TAIJI_OPENAPI_PORT', '1081'))

    def _create_openai_client_with_port(self: Any, host: str, port: int) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f'http://{host}:{openapi_port}/openapi',
            api_key='dummy',
            timeout=self.timeout,
            default_headers={
                'Authorization': self.authorization,
                'wsid': self.wsid,
            },
        )

    nexus_client._create_openai_client = MethodType(_create_openai_client_with_port, nexus_client)
    nexus_client._tau3_openapi_port_patched = True
    nexus_client._tau3_openapi_port = openapi_port
    return openapi_port


def _patch_taiji_clients_in_context() -> None:
    ctx = require_context()
    for client_name, client in (ctx.openai_clients or {}).items():
        openapi_port = _patch_taiji_openapi_port(client)
        if openapi_port is not None:
            logger.debug('Patched tau3 Taiji client %s to use openapi port %s', client_name, openapi_port)


class Tau3TaskInput(TaskRunInput, WithAgentRunner):
    """Input configuration for a single tau3 benchmark task."""

    domain: str = ''  # required — domain name (airline, retail, telecom, banking_knowledge)
    task_split: str = 'base'  # task split: test | train | base
    solo_mode: bool = False
    max_steps: int = 200
    max_errors: int = 10
    seed: int | None = None
    evaluation_type: str = 'all'
    add_thinking_tool: bool = False
    task_data: dict[str, Any] | None = None
    retrieval_config: str | None = None  # retrieval strategy for banking_knowledge
    retrieval_config_kwargs: dict[str, Any] | None = None
    # Named OpenAI client in ExecutionContext used for NL_ASSERTION evaluation.
    # Defaults to "eval"; falls back to "user" then "main" if not present.
    eval_client_name: str = 'eval'


class Tau3TaskOutput(TaskRunOutput, WithScore):
    """Output of a tau3 benchmark task, including score and diagnostics."""

    reward_info: dict[str, Any] | None = None
    termination_reason: str | None = None
    trajectory: list[dict[str, Any]] = []
    error: str | None = None
    # Rich simulation metadata
    duration: float | None = None
    agent_cost: float | None = None
    user_cost: float | None = None


def _build_nl_assertion_messages(trajectory: list[Any], nl_assertions: list[str]) -> list[dict[str, str]]:
    trajectory_str = '\n'.join([f'{message.role}: {message.content}' for message in trajectory])
    user_prompt = f"""
conversation:
{trajectory_str}

expectedOutcomes:
{nl_assertions}
"""
    return [
        {'role': 'system', 'content': _NL_ASSERTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


def _resolve_eval_client_name() -> str:
    ctx = require_context()
    configured = (ctx.task_config or {}).get('eval_client_name')
    candidates = [configured, 'eval', 'user', 'main']
    for candidate in candidates:
        if candidate and candidate in (ctx.openai_clients or {}):
            return str(candidate)
    raise KeyError(
        f'No eval client available for tau3 NL_ASSERTION. Configured={configured!r}, '
        f'available={list((ctx.openai_clients or {}).keys())}'
    )


def _run_completion_in_thread(client: Any, messages: list[dict[str, str]]) -> Any:
    return asyncio.run(client.create_completion(messages=messages))


def _run_completion_sync(client: Any, messages: list[dict[str, str]]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(client.create_completion(messages=messages))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_completion_in_thread, client, messages).result()


def _registry_evaluate_nl_assertions(cls: type[Any], trajectory: list[Any], nl_assertions: list[str]) -> list[Any]:
    from tau2.data_model.simulation import NLAssertionCheck  # ty: ignore[unresolved-import]

    ctx = require_context()
    client_name = _resolve_eval_client_name()
    client = ctx.get_openai_client(client_name)
    messages = _build_nl_assertion_messages(trajectory, nl_assertions)
    model_name = client.config.request_args.get('model') or client.config.client_args.get('model', '')

    logger.debug(
        '[tau3-eval] Running NL_ASSERTION with client=%s model=%s assertions=%s messages=%s',
        client_name,
        model_name,
        len(nl_assertions),
        len(messages),
    )

    response = _run_completion_sync(client, messages)
    if response is None or not response.choices or response.choices[0].message is None:
        raise RuntimeError(f'NL_ASSERTION evaluator got empty response from client {client_name!r}')

    content = response.choices[0].message.content or ''
    try:
        result_data = json.loads(content)
    except json.JSONDecodeError as exc:
        preview = content[:500]
        logger.error('[tau3-eval] Invalid NL_ASSERTION JSON from client=%s: %r', client_name, preview)
        raise RuntimeError(f'NL_ASSERTION evaluator returned invalid JSON: {exc}: {preview!r}') from exc

    results = result_data.get('results', [])
    logger.debug('[tau3-eval] Parsed NL_ASSERTION results: client=%s count=%s', client_name, len(results))
    return [
        NLAssertionCheck(
            nl_assertion=result['expectedOutcome'],
            met=result['metExpectation'],
            justification=result['reasoning'],
        )
        for result in results
    ]


def _ensure_registry_backed_nl_assertions() -> None:
    global _NL_ASSERTION_PATCHED
    if _NL_ASSERTION_PATCHED:
        return

    import tau2.evaluator.evaluator_nl_assertions as nl_eval_module  # ty: ignore[unresolved-import]

    nl_eval_module.NLAssertionsEvaluator.evaluate_nl_assertions = classmethod(_registry_evaluate_nl_assertions)  # ty: ignore[invalid-assignment]
    nl_eval_module.FullDuplexNLAssertionsEvaluator.evaluate_nl_assertions = classmethod(  # ty: ignore[invalid-assignment]
        _registry_evaluate_nl_assertions
    )
    _NL_ASSERTION_PATCHED = True
    logger.debug('Patched tau2 NL_ASSERTION evaluator to use nexus client registry')


@task_registry.register('tau3')
async def run_tau3_task(task_input: Tau3TaskInput) -> Tau3TaskOutput:
    """Run a tau3-bench simulation and evaluate the result."""
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # ty: ignore[unresolved-import]

    task_data = task_input.task_data or {}
    task_id = task_data.get('id', 'unknown')

    logger.debug(f'Running tau3 task: task_id={task_id}, solo_mode={task_input.solo_mode}')
    _patch_taiji_clients_in_context()

    # 1. Call the agent to run the simulation
    agent_output: AgentRunOutput = await task_input.run_agent(
        domain=task_input.domain,
        solo_mode=task_input.solo_mode,
        max_steps=task_input.max_steps,
        max_errors=task_input.max_errors,
        seed=task_input.seed,
        add_thinking_tool=task_input.add_thinking_tool,
        task_data=task_data,
        retrieval_config=task_input.retrieval_config,
        retrieval_config_kwargs=task_input.retrieval_config_kwargs,
    )

    simulation = getattr(agent_output, 'simulation', None)
    tau3_task = getattr(agent_output, 'tau3_task', None)
    domain = getattr(agent_output, 'domain', '')
    solo_mode = getattr(agent_output, 'solo_mode', task_input.solo_mode)
    termination_reason = getattr(agent_output, 'termination_reason', None)
    trajectory = getattr(agent_output, 'trajectory', [])
    error = getattr(agent_output, 'error', None)

    # 2. Evaluate the simulation (task layer responsibility)
    score = 0.0
    reward_info = None

    if simulation is not None and tau3_task is not None and error is None:
        _ensure_registry_backed_nl_assertions()
        try:
            eval_type = EvaluationType(task_input.evaluation_type)

            # Auto-upgrade to ALL_WITH_NL_ASSERTIONS when the task's reward_basis
            # includes NL_ASSERTION. Without this, evaluate_simulation raises
            # because EvaluationType.ALL skips NL assertion evaluation.
            if eval_type == EvaluationType.ALL and tau3_task.evaluation_criteria is not None:
                basis_values = {
                    b.value if hasattr(b, 'value') else str(b)
                    for b in (tau3_task.evaluation_criteria.reward_basis or [])
                }
                if 'NL_ASSERTION' in basis_values:
                    logger.debug('Task reward_basis includes NL_ASSERTION, upgrading eval to all_with_nl_assertions')
                    eval_type = EvaluationType.ALL_WITH_NL_ASSERTIONS

            # Build env_kwargs for evaluate_simulation (needed for banking_knowledge)
            env_kwargs: dict[str, Any] = {}
            if task_input.retrieval_config is not None and domain == 'banking_knowledge':
                env_kwargs['retrieval_variant'] = task_input.retrieval_config
                env_kwargs['task'] = tau3_task
                if task_input.retrieval_config_kwargs:
                    env_kwargs['retrieval_kwargs'] = task_input.retrieval_config_kwargs

            reward_result = evaluate_simulation(
                simulation=simulation,
                task=tau3_task,
                evaluation_type=eval_type,
                solo_mode=solo_mode,
                domain=domain,
                env_kwargs=env_kwargs if env_kwargs else None,  # ty: ignore[invalid-argument-type]
            )
            score = reward_result.reward
            reward_info = reward_result.model_dump()
        except Exception as e:
            logger.error(f'Tau3 evaluation error: {type(e).__name__}: {e}')
            error = str(e)

    logger.info(f'Tau3 task completed: score={score}, termination={termination_reason}')

    return Tau3TaskOutput(
        score=score,
        reward_info=reward_info,
        termination_reason=termination_reason,
        trajectory=trajectory,
        error=error,
        duration=getattr(agent_output, 'duration', None),
        agent_cost=getattr(agent_output, 'agent_cost', None),
        user_cost=getattr(agent_output, 'user_cost', None),
    )
