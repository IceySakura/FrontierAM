import asyncio
import json
import logging
import os
import re

from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx
import pandas as pd

from pydantic import BaseModel, Field

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import EvaluationDetails, Evaluator
from nexus.extensions.agents.base.schema import ToolChoice
from nexus.extensions.tasks.hy3_search.prompt import (
    JUDGE_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT,
    JUDGE_SEMANTIC_EQUIVALENCE_TOOL,
    REWARD_SERVICE_PROMPT_TEMPLATE,
)
from nexus.extensions.tasks.hy3_search.utils import (
    extract_error_context,
    extract_reward_score,
    flat_string_to_dataframe,
    get_judge_use_tool_calling_default,
    norm_column,
    parse_router_id,
    unified_to_dataframe,
)
from nexus.extensions.tasks.hy3_search.wide_search_evaluation.data_loader import (
    WideSearchQuery,
    WideSearchResponse,
)
from nexus.extensions.tasks.hy3_search.wide_search_evaluation.evaluation import (
    EvaluationResult as WideSearchEvalResult,
)
from nexus.extensions.tasks.hy3_search.wide_search_evaluation.evaluation import (
    evaluate_single_query,
)
from nexus.extensions.tasks.hy3_search.wide_search_evaluation.metric_utils import LLMFunc


logger = logging.getLogger(__name__)


async def judge_semantic_match(
    question: str,
    predicted: str,
    expected: str,
    *,
    judge_prompt_template: str | None = None,
    judge_system_prompt: str | None = None,
    use_tool_calling: bool | None = None,
    max_input_length: int = 4096,
) -> float:
    """
    Use LLM to judge semantic equivalence between predicted and expected answers.

    Args:
        question: The question being evaluated.
        predicted: The predicted answer.
        expected: The expected/reference answer.
        judge_prompt_template: Optional custom prompt template. Available placeholders:
            - {prompt} or {question}: The question being evaluated
            - {response_reference} or {expected} or {ref_answer}: The expected/reference answer
            - {response} or {predicted} or {model_response}: The predicted answer
        judge_system_prompt: Optional custom system prompt. If not provided, uses default.
        use_tool_calling: If True, use tool calling for structured output. If False, use
            direct text generation and regex parsing (for templates like REWARD_SERVICE_PROMPT_TEMPLATE).
        max_input_length: Maximum length for predicted/expected strings (default: 4096).

    Returns:
        1.0 if semantically equivalent, 0.0 otherwise.

    Raises:
        RuntimeError: If LLM call fails or returns invalid response.
    """
    # Get use_tool_calling from environment variable if not explicitly provided
    if use_tool_calling is None:
        use_tool_calling = get_judge_use_tool_calling_default()

    logger.debug(f'use_tool_calling: {use_tool_calling}')
    # Truncate inputs to max_input_length
    predicted_truncated = predicted[-max_input_length:] if len(predicted) > max_input_length else predicted
    expected_truncated = expected[-max_input_length:] if len(expected) > max_input_length else expected

    # Use custom system prompt if provided, otherwise use default
    # When use_tool_calling=False, default to simple prompt (aligned with build_judge_prompt.py)
    if judge_system_prompt is not None:
        system_prompt = judge_system_prompt
    elif not use_tool_calling:
        system_prompt = 'You are a helpful assistant.'
    else:
        system_prompt = JUDGE_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT

    if judge_prompt_template is not None:
        # Provide both canonical names and aliases for flexibility
        user_content = judge_prompt_template.format(
            prompt=question,
            question=question,
            response_reference=expected_truncated,
            expected=expected_truncated,
            ref_answer=expected_truncated,
            response=predicted_truncated,
            predicted=predicted_truncated,
            model_response=predicted_truncated,
        )
    elif not use_tool_calling:
        user_content = REWARD_SERVICE_PROMPT_TEMPLATE.format(
            prompt=question,
            ref_answer=expected_truncated,
            model_response=predicted_truncated,
        )
    else:
        user_content = f'QUESTION: {question}\n\nPREDICTED: "{predicted}"\n\nEXPECTED: "{expected}"'

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_content},
    ]

    ctx = require_context()
    client = ctx.get_openai_client('judge')

    if not use_tool_calling:
        # Direct text generation mode - parse result using regex
        response = await client.create_completion(messages=messages)
        content = response.choices[0].message.content or ''

        # Parse "是否一致": 1 pattern from response
        is_match = re.search(r'"是否一致":\s*1', content) is not None
        logger.info(f'LLM judge (text mode): is_match={is_match}')
        return 1.0 if is_match else 0.0

    # Tool calling mode (default)
    response = await client.create_completion(
        messages=messages,
        tools=[JUDGE_SEMANTIC_EQUIVALENCE_TOOL],
        tool_choice=ToolChoice.REQUIRED,
    )

    # Extract tool call (no defensive fallbacks - fail fast if malformed)
    if not (tool_calls := response.choices[0].message.tool_calls):
        raise RuntimeError('LLM judge returned no tool call')

    try:
        result = json.loads(tool_calls[0].function.arguments)
        is_match = result['is_equivalent']
        reason = result.get('reason', 'No reason provided')

        logger.info(f'LLM judge: {is_match} - {reason}')
        return 1.0 if is_match else 0.0

    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f'Invalid LLM judge response: {e}') from e


class EvalMode(str, Enum):
    EXACT_MATCH = 'exact_match'
    LLM_JUDGE = 'llm_judge'
    REWARD_SERVICE = 'reward_service'
    WIDESEARCH = 'wide_search'


# Reward service defaults (single source of truth)
REWARD_SERVICE_DEFAULT_TIMEOUT = 1800.0
REWARD_SERVICE_DEFAULT_EXP_ID = 'nexus_eval'


class RewardServiceConfig(BaseModel, frozen=True):
    """Immutable configuration for reward service API evaluation."""

    url: str
    token: str
    router_id: str
    exp_id: str = REWARD_SERVICE_DEFAULT_EXP_ID
    timeout: float = REWARD_SERVICE_DEFAULT_TIMEOUT
    extra: dict[str, Any] = Field(default_factory=dict)


async def exact_match_eval(question: str, predicted: str, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


@dataclass
class WideSearchEvaluationDetails(EvaluationDetails):
    """Detailed evaluation metrics for wide_search mode."""

    exact_match: float = 0.0
    precision_by_row: float = 0.0
    recall_by_row: float = 0.0
    f1_by_row: float = 0.0
    precision_by_item: float = 0.0
    recall_by_item: float = 0.0
    f1_by_item: float = 0.0


def _make_sync_judge(client) -> LLMFunc:
    """Wrap an async Nexus OpenAI client into a synchronous LLM callable."""

    def sync_judge(messages: list[dict]) -> str | None:
        try:
            response = asyncio.run(client.create_completion(messages=messages))
            return response.choices[0].message.content
        except Exception as exc:
            logger.warning(f'Judge LLM call failed: {exc}')
            return None

    return sync_judge


def _build_evaluation_config(exp_df: pd.DataFrame) -> dict:
    """Build a minimal evaluation config when none is provided.

    Falls back to heuristic unique-column detection and exact_match for all
    non-key columns.  Prefer supplying an explicit evaluation config via
    ``SearchTaskInput.evaluation`` for WideSearch datasets that ship one.
    """
    required = exp_df.columns.tolist()
    unique_columns = [col for col in required if 'id' in col.lower()] or required
    eval_pipeline = {
        col: {'preprocess': ['norm_str'], 'metric': ['exact_match']} for col in required if col not in unique_columns
    }
    return {'required': required, 'unique_columns': unique_columns, 'eval_pipeline': eval_pipeline}


async def _wide_search_eval_with_details(
    question: str,
    predicted: str,
    expected: str,
    evaluation: dict | None = None,
    instance_id: str = 'unknown',
) -> tuple[float, WideSearchEvaluationDetails]:
    """Evaluate predicted and expected data for wide_search mode.

    Delegates to ``wide_search_evaluation.evaluation.evaluate_single_query`` which
    implements the full WideSearch evaluation pipeline (per-column preprocessing,
    diverse metric functions, LLM-based primary-key alignment).

    Args:
        question: The original question (used for logging).
        predicted: Agent's text response (should contain a Markdown table).
        expected: Ground truth serialised as a flat_string
            (``col=val | col=val || col=val | col=val``).
        evaluation: Optional evaluation config dict with keys ``required``,
            ``unique_columns``, and ``eval_pipeline``.  When ``None``, a minimal
            config is auto-inferred from the ground-truth columns.
        instance_id: Instance identifier used for logging.

    Returns:
        ``(f1_by_item, WideSearchEvaluationDetails)``
    """
    detail = WideSearchEvaluationDetails()

    # Parse prediction
    try:
        pred_df = unified_to_dataframe(predicted)
    except Exception as exc:
        logger.warning(f'Failed to parse prediction table: {exc}')
        detail.msg = f'failed to parse prediction: {exc}'
        return 0.0, detail

    # Parse ground truth
    try:
        exp_df = flat_string_to_dataframe(expected)
    except Exception as exc:
        logger.warning(f'Failed to parse ground truth table: {exc}')
        detail.msg = f'failed to parse ground truth: {exc}'
        return 0.0, detail

    # Normalise column names on both DataFrames
    pred_df.columns = [norm_column(c) for c in pred_df.columns]
    exp_df.columns = [norm_column(c) for c in exp_df.columns]

    # Build evaluation config if not supplied
    if evaluation is None:
        evaluation = _build_evaluation_config(exp_df)

    ws_query = WideSearchQuery(
        instance_id=instance_id,
        query=question,
        evaluation=evaluation,
        answer=exp_df,
        language='zh',
    )
    ws_response = WideSearchResponse(instance_id=instance_id, response=predicted)

    # Obtain a sync LLM wrapper from the judge client (optional)
    llm_func: LLMFunc = None
    try:
        ctx = require_context()
        judge_client = ctx.get_openai_client('judge')
        llm_func = _make_sync_judge(judge_client)
        logger.debug('Judge client available: LLM-based evaluation steps enabled')
    except Exception:
        logger.debug('No judge client configured: LLM-based evaluation steps disabled')

    # Run synchronous evaluation pipeline in a thread executor
    loop = asyncio.get_running_loop()
    eval_result: WideSearchEvalResult = await loop.run_in_executor(
        None,
        lambda: evaluate_single_query(
            query=ws_query,
            response=ws_response,
            llm_func=llm_func,
            response_df=pred_df,
        ),
    )

    detail.exact_match = eval_result.score
    detail.precision_by_row = eval_result.precision_by_row
    detail.recall_by_row = eval_result.recall_by_row
    detail.f1_by_row = eval_result.f1_by_row
    detail.precision_by_item = eval_result.precision_by_item
    detail.recall_by_item = eval_result.recall_by_item
    detail.f1_by_item = eval_result.f1_by_item
    detail.msg = eval_result.msg

    return eval_result.f1_by_item, detail


async def wide_search_eval(question: str, predicted: str, expected: str) -> float:
    """Evaluate predicted and expected data for wide_search mode and return scalar score only."""
    score, _ = await _wide_search_eval_with_details(question=question, predicted=predicted, expected=expected)
    return score


async def llm_judge_eval(
    question: str,
    predicted: str,
    expected: str,
    *,
    judge_prompt_template: str | None = None,
    judge_system_prompt: str | None = None,
    use_tool_calling: bool | None = None,
    max_input_length: int = 4096,
) -> float:
    """LLM-based semantic evaluation with exact match fallback."""
    try:
        return await judge_semantic_match(
            question,
            predicted,
            expected,
            judge_prompt_template=judge_prompt_template,
            judge_system_prompt=judge_system_prompt,
            use_tool_calling=use_tool_calling,
            max_input_length=max_input_length,
        )
    except Exception as e:
        logger.warning(f'LLM judge failed ({e}), falling back to exact match')
        return await exact_match_eval(question, predicted, expected)


async def reward_service_eval(
    question: str,
    predicted: str,
    expected: str,
    *,
    config: RewardServiceConfig,
) -> float:
    """
    Evaluate using the reward service API.

    This is a cost-effective alternative to LLM judge that uses the reward
    service platform for semantic equivalence evaluation.

    Args:
        question: The question being evaluated.
        predicted: The predicted answer.
        expected: The expected/reference answer.
        config: Reward service configuration.

    Returns:
        Score from 0.0 to 1.0 based on reward service evaluation.

    Raises:
        RuntimeError: If reward service call fails or returns invalid response.
    """
    class1, class2, router_version = parse_router_id(config.router_id)

    json_data = {
        'data_source': 'reward_system',
        'messages': [
            {'content': '', 'role': 'system'},
            {'content': question, 'role': 'user'},
            {'content': predicted, 'role': 'assistant'},
        ],
        'reward_model': {'ground_truth': expected},
        'extra': {
            'answer_pattern': '',
            'process_truncate': True,
            'ref_answer': expected,
            'short_answer_pattern': '',
            **config.extra,
        },
        'ref_answer': expected,
        'exp_id': config.exp_id,
        'class1': class1,
        'class2': class2,
        'router_version': router_version,
    }

    headers = {'token': config.token, 'Content-Type': 'application/json'}

    logger.debug('Calling reward service: %s (router: %s)', config.url, config.router_id)

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(config.url, json=json_data, headers=headers)
            response.raise_for_status()
            return extract_reward_score(response.json())
    except httpx.HTTPStatusError as e:
        error_context = extract_error_context(e.response)
        details = f'question={question[:200]!r} | predicted={predicted[:200]!r} | expected={expected[:200]!r}'
        raise RuntimeError(f'Reward service HTTP {e.response.status_code}{error_context} | {details}') from e
    except httpx.RequestError as e:
        error_context = extract_error_context(getattr(e, 'response', None))
        details = f'question={question[:200]!r} | predicted={predicted[:200]!r} | expected={expected[:200]!r}'
        raise RuntimeError(f'Reward service request failed: {e} | {error_context} | {details}') from e
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        details = f'question={question[:200]!r} | predicted={predicted[:200]!r} | expected={expected[:200]!r}'
        raise RuntimeError(f'Invalid reward service response format: {e} | {details}') from e


class SearchEvaluator(Evaluator):
    """
    Polymorphic evaluator that selects strategy based on configuration.
    Encapsulates the 'how' of evaluation.

    Supports four evaluation modes:
    - EXACT_MATCH: Simple string comparison
    - LLM_JUDGE: Use LLM for semantic equivalence (higher cost)
    - REWARD_SERVICE: Use reward service API for evaluation (lower cost)
    - WIDESEARCH: Structured data comparison for wide search tasks
    """

    def __init__(
        self,
        mode: EvalMode,
        *,
        # LLM Judge configuration
        judge_prompt_template: str | None = None,
        judge_system_prompt: str | None = None,
        judge_use_tool_calling: bool | None = None,
        judge_max_input_length: int = 4096,
        # Reward Service configuration (resolved from env if not provided)
        reward_config: RewardServiceConfig | None = None,
        # WideSearch evaluation configuration
        widesearch_evaluation: dict | None = None,
        instance_id: str = 'unknown',
    ):
        self.mode = mode
        self.instance_id = instance_id
        # LLM Judge params
        self.judge_prompt_template = judge_prompt_template
        self.judge_system_prompt = judge_system_prompt
        self.judge_use_tool_calling = judge_use_tool_calling
        self.judge_max_input_length = judge_max_input_length
        # Reward Service config — auto-resolve from env vars if not explicitly provided
        self.reward_config = reward_config if reward_config is not None else self._resolve_reward_config(mode)
        # WideSearch config
        self.widesearch_evaluation = widesearch_evaluation

    @staticmethod
    def _resolve_reward_config(mode: EvalMode) -> RewardServiceConfig | None:
        """Resolve reward service config from environment variables when mode requires it."""
        if mode != EvalMode.REWARD_SERVICE:
            return None
        url = os.environ.get('REWARD_SERVICE_URL')
        token = os.environ.get('REWARD_SERVICE_TOKEN')
        router_id = os.environ.get('REWARD_SERVICE_ROUTER_ID')
        if not url or not token or not router_id:
            return None
        kwargs: dict[str, Any] = {'url': url, 'token': token, 'router_id': router_id}
        exp_id = os.environ.get('REWARD_SERVICE_EXP_ID')
        if exp_id:
            kwargs['exp_id'] = exp_id
        timeout = os.environ.get('REWARD_SERVICE_TIMEOUT')
        if timeout:
            kwargs['timeout'] = float(timeout)
        return RewardServiceConfig(**kwargs)

    async def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> float:
        logger.debug(f'Evaluating with {self.mode.value}')
        normalized_question = question or ''

        if self.mode == EvalMode.LLM_JUDGE:
            return await llm_judge_eval(
                normalized_question,
                prediction,
                reference,
                judge_prompt_template=self.judge_prompt_template,
                judge_system_prompt=self.judge_system_prompt,
                use_tool_calling=self.judge_use_tool_calling,
                max_input_length=self.judge_max_input_length,
            )
        if self.mode == EvalMode.REWARD_SERVICE:
            if not self.reward_config:
                raise ValueError(
                    'RewardServiceConfig is required for REWARD_SERVICE eval mode. '
                    'Set REWARD_SERVICE_URL, REWARD_SERVICE_TOKEN, and REWARD_SERVICE_ROUTER_ID env vars.'
                )
            return await reward_service_eval(
                normalized_question,
                prediction,
                reference,
                config=self.reward_config,
            )
        if self.mode == EvalMode.WIDESEARCH:
            score, _ = await _wide_search_eval_with_details(
                normalized_question,
                prediction,
                reference,
                evaluation=self.widesearch_evaluation,
                instance_id=self.instance_id,
            )
            return score

        return await exact_match_eval(normalized_question, prediction, reference)

    async def evaluate_with_details(
        self,
        prediction: str,
        reference: str,
        question: str | None = None,
    ) -> tuple[float, EvaluationDetails | None]:
        """Evaluate and return optional structured details for downstream consumers."""
        normalized_question = question or ''

        if self.mode == EvalMode.WIDESEARCH:
            score, details = await _wide_search_eval_with_details(
                normalized_question,
                prediction,
                reference,
                evaluation=self.widesearch_evaluation,
                instance_id=self.instance_id,
            )
            return score, details

        logger.warning(f'evaluate_with_details for {self.mode} is not supported yet')
        score = await self.evaluate(prediction=prediction, reference=reference, question=normalized_question)
        return score, None
