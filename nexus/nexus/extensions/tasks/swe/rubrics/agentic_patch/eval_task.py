"""Rubric evaluation task: score a patch against rubrics using an LLM judge.

Pure LLM call - no runtime/sandbox needed. Takes rubrics_yaml + patch_text,
outputs per-item scores and a normalized [0, 1] score.
"""

import logging

import yaml

from pydantic import Field, ValidationError

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.swe.rubrics.agentic_patch.prompts import EVAL_SYSTEM_PROMPT, EVAL_USER_PROMPT
from nexus.extensions.tasks.swe.rubrics.shared.schema import (
    RubricSchema,
    RubricScores,
    parse_llm_scores_response,
)


logger = logging.getLogger(__name__)


MAX_EVAL_RETRIES = 3


async def score_patch_against_rubrics(
    *,
    rubrics_yaml: str,
    patch_text: str,
    problem_statement: str,
    eval_client_name: str = 'rubric_eval',
    max_retries: int = MAX_EVAL_RETRIES,
) -> RubricScores:
    """Score a patch against rubrics using an LLM judge.

    This is the core scoring function, reused by eval_task, integrated task, and RubricEvaluator.
    Retries up to max_retries times if the LLM response cannot be parsed into scores.

    Args:
        rubrics_yaml: YAML string of rubric definitions
        patch_text: The code patch to evaluate
        problem_statement: PR description / problem context
        eval_client_name: Which openai_client to use for evaluation
        max_retries: Max attempts if score parsing fails (default: 3)

    Returns:
        RubricScores with per-item scores and normalized aggregate score
    """
    ctx = require_context()
    client = ctx.get_openai_client(eval_client_name)

    # Parse rubrics to get weights
    rubrics = RubricSchema.from_yaml(rubrics_yaml)
    weights = rubrics.get_weights()

    if not weights:
        logger.warning('[rubric_eval] No rubric items found in YAML')
        return RubricScores(score=0.0)

    # Build evaluation prompt
    user_prompt = EVAL_USER_PROMPT.format(
        problem_statement=problem_statement,
        patch_text=patch_text,
        rubrics_yaml=rubrics_yaml,
    )

    messages: list[dict[str, str]] = [
        {'role': 'system', 'content': EVAL_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    last_raw_response = ''
    for attempt in range(1, max_retries + 1):
        response = await client.create_completion(messages=messages)
        raw_response = response.choices[0].message.content or ''
        last_raw_response = raw_response

        logger.debug(f'[rubric_eval] Attempt {attempt}/{max_retries}, response: {raw_response[:500]}')

        item_scores = parse_llm_scores_response(raw_response)
        if item_scores:
            scores = RubricScores.compute(item_scores, weights)
            scores.raw_response = raw_response
            return scores

        logger.warning(
            f'[rubric_eval] Attempt {attempt}/{max_retries}: failed to parse scores from response: {raw_response[:200]}'
        )

    # All retries exhausted, return zero scores with last raw response
    logger.error(f'[rubric_eval] All {max_retries} attempts failed to parse scores')
    scores = RubricScores.compute({}, weights)
    scores.raw_response = last_raw_response
    return scores


# ============== Input/Output ==============


class SweRubricEvalInput(TaskRunInput, WithAgentRunner):
    """Input for rubric evaluation task. No runtime needed.

    Uses WithAgentRunner so the runner injects openai_clients into ExecutionContext.
    The run_agent callable is not used.
    """

    rubrics_yaml: str = Field(description='Rubric YAML content')
    patch_text: str = Field(description='Code patch to evaluate')
    problem_statement: str = Field(default='', description='PR description / problem context')
    eval_client_name: str = Field(default='rubric_eval', description='Which openai_client to use')


class SweRubricEvalOutput(TaskRunOutput, WithScore):
    """Output from rubric evaluation task."""

    # score inherited from WithScore, normalized to [0, 1]
    item_scores: dict[str, float] = Field(default_factory=dict, description='Per-item binary scores')
    weighted_sum: float = Field(default=0.0, description='Raw weighted sum')
    max_possible_score: float = Field(default=0.0, description='Maximum possible weighted sum')
    raw_response: str = Field(default='', description='Raw LLM response for debugging')
    error: str | None = None


# ============== Task ==============


@task_registry.register('swe_rubric_eval_task')
async def run_swe_rubric_eval_task(task_input: SweRubricEvalInput) -> SweRubricEvalOutput:
    """Evaluate a patch against rubrics using an LLM judge.

    Flow: validate inputs -> call LLM -> parse scores -> compute weighted average
    """
    # --- Input validation (graceful fail) ---
    if not task_input.patch_text:
        return SweRubricEvalOutput(score=0.0, error='missing patch_text')

    if not task_input.rubrics_yaml:
        return SweRubricEvalOutput(score=0.0, error='missing rubrics_yaml')

    # Validate rubrics YAML format
    try:
        RubricSchema.from_yaml(task_input.rubrics_yaml)
    except (yaml.YAMLError, ValidationError, ValueError) as e:
        return SweRubricEvalOutput(score=0.0, error=f'invalid rubrics_yaml: {e}')

    # --- Score ---
    try:
        scores = await score_patch_against_rubrics(
            rubrics_yaml=task_input.rubrics_yaml,
            patch_text=task_input.patch_text,
            problem_statement=task_input.problem_statement,
            eval_client_name=task_input.eval_client_name,
        )

        return SweRubricEvalOutput(
            score=scores.score,
            item_scores=scores.item_scores,
            weighted_sum=scores.weighted_sum,
            max_possible_score=scores.max_possible_score,
            raw_response=scores.raw_response,
        )

    except KeyError as e:
        return SweRubricEvalOutput(score=0.0, error=f'openai_client not found: {e}')
    except Exception as e:
        logger.exception(f'[rubric_eval] Evaluation failed: {e}')
        return SweRubricEvalOutput(score=0.0, error=f'evaluation failed: {e}')
