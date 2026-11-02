import logging

from typing import Any, Literal

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithMainTrajectory,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.base.evaluation import evaluate_output
from nexus.extensions.tasks.stem_mas.evaluator import (
    GeneralEvaluator,
    HLEEvaluator,
    Lean4VerifierEvaluator,
    OSSEvaluator,
    OlympiadEvaluator,
    ResearchEvaluator,
    RewardSystemEvaluator,
)
from nexus.extensions.tasks.stem_mas.solvers import solve_with_official_pipeline
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)


def _extract_runtime_id_from_url(url: str) -> str:
    """Extract the runtime host id from a runtime provider base URL."""
    host = url.strip()
    if '://' in host:
        host = host.split('://', 1)[1]
    host = host.split('/', 1)[0].split(':', 1)[0]
    return host.split('.', 1)[0] if host else ''


def _extract_runtime_id(runtime: Any) -> str | None:
    if hasattr(runtime, 'config') and hasattr(runtime.config, 'base_url'):
        return _extract_runtime_id_from_url(str(runtime.config.base_url)) or None
    return None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get('text') or part.get('content')
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(part))
        return '\n'.join(part for part in parts if part).strip()
    return str(content).strip()


def _extract_trajectory_prediction(main_trajectory: list[dict]) -> str:
    for msg in reversed(main_trajectory):
        if msg.get('role') != 'assistant':
            continue
        content = _content_to_text(msg.get('content'))
        if content:
            return content
    logger.warning('No assistant content found in main trajectory; using empty prediction')
    return ''


def _extract_prediction(agent_output: object) -> str:
    """Extract prediction text from different agent output types.

    Supports:
    - ``MainAgentOutput`` with ``final_answer``
    - Any ``WithMainTrajectory`` agent (latest non-empty assistant content)
    - ``InstalledAgentOutput`` (claude_code, codex, gemini_cli) with ``output``
    """
    success = getattr(agent_output, 'success', True)
    if success is False:
        exit_code = getattr(agent_output, 'exit_code', None)
        stderr = (getattr(agent_output, 'stderr', '') or '').strip()
        if len(stderr) > 1000:
            stderr = stderr[:1000] + '...'
        detail = f'exit_code={exit_code}'
        if stderr:
            detail = f'{detail}, stderr={stderr}'
        raise RuntimeError(f'Agent execution failed: {detail}')
    if hasattr(agent_output, 'final_answer'):
        final_answer = _content_to_text(agent_output.final_answer)
        if final_answer:
            return final_answer
    if isinstance(agent_output, WithMainTrajectory):
        return _extract_trajectory_prediction(agent_output.main_trajectory)
    if hasattr(agent_output, 'output'):
        output = _content_to_text(agent_output.output)
        if output:
            return output
        logger.warning('Agent output is empty for %s; using empty prediction', type(agent_output).__name__)
        return ''
    raise RuntimeError(f'Cannot extract prediction from agent output: {type(agent_output)}')


class StemMasTaskInput(TaskRunInput, WithAgentRunner):
    """stem_mas task input."""

    question: str
    prompt: list[dict[str, Any]] | None = None
    answer: str
    run_oracle: bool = False
    metadata: dict[str, Any] | None = None
    evaluator_type: Literal[
        'hle', 'general', 'none', 'olympiad', 'research', 'oss', 'reward_system', 'lean4_verifier'
    ] = 'general'
    evaluator_none_score: float = 0.0
    needs_runtime: bool = True

    # official solver config
    official_solver_pipeline: str | None = None
    official_solver_command: str | None = None
    official_solver_timeout_seconds: int = 1800
    official_solver_extra: dict[str, Any] | None = None

    # reward_system evaluator config
    reward_system_user_token: str | None = None
    reward_system_exp_id: str | None = None
    reward_system_router_id: str | None = None
    reward_system_extra: dict[str, Any] | None = None
    reward_system_public_test: dict[str, Any] | None = None
    reward_system_ci_test: bool = False

    # lean4 verifier evaluator config
    lean4_verifier_url: str | None = None
    lean4_verifier_benchmark: str | None = None
    lean4_verifier_task_id: str | None = None
    lean4_verifier_manifest: str | None = None
    lean4_verifier_timeout_seconds: int = 300
    lean4_verifier_candidate_kind: Literal['proof_body'] = 'proof_body'
    lean4_verifier_candidate_policy: Literal['proof_body', 'target_declaration_proof'] = 'proof_body'
    lean4_verifier_forbid_sorry: bool = True


class StemMasTaskOutput(TaskRunOutput, WithScore):
    """stem_mas task output."""

    prediction: str
    answer: str
    judge_details: dict | None = None  # Judge details (prompt, response, parsed_result)


@task_registry.register('stem_mas')
async def stem_mas_task(task_input: StemMasTaskInput) -> StemMasTaskOutput:
    """Execute a stem_mas task where the agent solves one normalized benchmark question."""
    if task_input.needs_runtime:
        ctx = require_context()
        runtime = await ctx.get_runtime('main')
        logger.debug(f'Runtime initialized: {runtime}')
        runtime_id = _extract_runtime_id(runtime)
        span_id = get_current_span_id()
        if span_id and runtime_id:
            update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})
    else:
        logger.debug('Runtime initialization skipped by task_config.needs_runtime=false')
    # Execute
    if task_input.run_oracle:
        prediction = str(task_input.answer)
        logger.debug('Oracle mode: bypassed agent')
    elif task_input.official_solver_pipeline:
        metadata = task_input.metadata or {}
        benchmark = str(task_input.lean4_verifier_benchmark or '')
        task_id = str(task_input.lean4_verifier_task_id or '')
        result = await solve_with_official_pipeline(
            pipeline=task_input.official_solver_pipeline,
            benchmark=benchmark,
            task_id=task_id,
            question=task_input.question,
            reference=task_input.answer,
            metadata=metadata,
            extra=task_input.official_solver_extra,
            command=task_input.official_solver_command,
            timeout_seconds=task_input.official_solver_timeout_seconds,
        )
        prediction = result.prediction
        logger.info(
            'Official solver completed: pipeline=%s benchmark=%s task_id=%s stderr_len=%d',
            task_input.official_solver_pipeline,
            benchmark,
            task_id,
            len(result.stderr),
        )
    else:
        prompt = task_input.prompt or [{'role': 'user', 'content': task_input.question}]
        agent_output = await task_input.run_agent(
            user_prompt=task_input.question,
            prompt=prompt,
        )
        prediction = _extract_prediction(agent_output)

    prediction_val = prediction.strip()

    # Evaluate
    if task_input.evaluator_type == 'none':
        score = task_input.evaluator_none_score
        judge_details = None
        logger.info('Skipping evaluator, using default score')
    else:
        if task_input.evaluator_type == 'reward_system':
            if not task_input.reward_system_router_id:
                raise ValueError('reward_system_router_id is required for reward_system evaluator')
            evaluator = RewardSystemEvaluator(
                user_token=task_input.reward_system_user_token,
                exp_id=task_input.reward_system_exp_id or '',
                router_id=task_input.reward_system_router_id,
                extra=task_input.reward_system_extra,
                public_test=task_input.reward_system_public_test,
                ci_test=task_input.reward_system_ci_test,
            )
            logger.debug('Using RewardSystemEvaluator for evaluation: router_id=%s', task_input.reward_system_router_id)
        elif task_input.evaluator_type == 'general':
            evaluator = GeneralEvaluator()
            logger.debug('Using GeneralEvaluator for evaluation')
        elif task_input.evaluator_type == 'hle':
            evaluator = HLEEvaluator()
            logger.debug('Using HLEEvaluator for evaluation')
        elif task_input.evaluator_type == 'olympiad':
            evaluator = OlympiadEvaluator()
            logger.debug('Using OlympiadEvaluator for evaluation')
        elif task_input.evaluator_type == 'research':
            evaluator = ResearchEvaluator()
            logger.debug('Using ResearchEvaluator for evaluation')
        elif task_input.evaluator_type == 'oss':
            evaluator = OSSEvaluator()
            logger.debug('Using OSSEvaluator for evaluation')
        elif task_input.evaluator_type == 'lean4_verifier':
            if not task_input.lean4_verifier_benchmark or not task_input.lean4_verifier_task_id:
                raise ValueError('lean4_verifier_benchmark and lean4_verifier_task_id are required')
            evaluator = Lean4VerifierEvaluator(
                url=task_input.lean4_verifier_url,
                benchmark=task_input.lean4_verifier_benchmark,
                task_id=task_input.lean4_verifier_task_id,
                expected_manifest_version=task_input.lean4_verifier_manifest,
                timeout_seconds=task_input.lean4_verifier_timeout_seconds,
                candidate_kind=task_input.lean4_verifier_candidate_kind,
                candidate_policy=task_input.lean4_verifier_candidate_policy,
                forbid_sorry=task_input.lean4_verifier_forbid_sorry,
            )
            logger.debug(
                'Using Lean4VerifierEvaluator: benchmark=%s task_id=%s',
                task_input.lean4_verifier_benchmark,
                task_input.lean4_verifier_task_id,
            )
        else:
            raise ValueError(f'Unsupported evaluator_type: {task_input.evaluator_type}')

        score, _, _ = await evaluate_output(
            prediction=prediction_val,
            reference=task_input.answer,
            question=task_input.question,
            evaluator=evaluator,
        )
        judge_details = evaluator.get_last_judge_result()

    logger.info(
        f'Question: {task_input.question} | Prediction: {prediction_val} | Reference: {task_input.answer} | Score: {score}'
    )

    return StemMasTaskOutput(
        score=score,
        prediction=prediction_val,
        answer=task_input.answer,
        judge_details=judge_details,
    )
