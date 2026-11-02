from __future__ import annotations

import asyncio
import json
import logging
import os
import re

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import httpx

from pydantic import BaseModel, Field

from nexus.core import require_context
from nexus.extensions.agents.base.evaluation import EvaluationDetails, Evaluator
from nexus.extensions.tasks.search.deepsearch_qa_evaluation.evaluation import (
    deepsearch_qa_eval,
)
from nexus.extensions.tasks.search.draco_rubric_evaluation.evaluation import (
    draco_rubric_eval,
)
from nexus.extensions.tasks.search.prompt import (
    JUDGE_SEMANTIC_EQUIVALENCE_PROMPT_TEMPLATE,
    JUDGE_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT,
)
from nexus.extensions.tasks.search.utils import (
    extract_error_context,
    extract_reward_score,
    flat_string_to_dataframe,
    norm_column,
    parse_router_id,
    unified_to_dataframe,
)
from nexus.extensions.tasks.search.wide_search_evaluation.metric_utils import LLMFunc
from nexus.utils.rate_limiter import get_rate_limiter


if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__)
JUDGE_XML_TAG_TEMPLATE = r'<{tag}>\s*(.*?)\s*</{tag}>'


def _extract_xml_tag(content: str, tag: str) -> str | None:
    """Extract a single XML tag value from judge output."""
    match = re.search(JUDGE_XML_TAG_TEMPLATE.format(tag=tag), content, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


JUDGE_DEFAULT_MAX_RETRIES = 3
JUDGE_DEFAULT_RETRY_DELAY = 1.0
JUDGE_CALL_TIMEOUT_SEC = float(os.environ.get('WIDESEARCH_JUDGE_TIMEOUT_SEC', '300'))  # 单次 judge 调用超时
# 每个进程内 WideSearch judge 并发上限：削平 rollout 结束时的请求尖峰，避免
# 端点过载触发 litellm 退避风暴（与 judge 模型自身的服务端并发无关）。
JUDGE_MAX_CONCURRENT = int(os.environ.get('WIDESEARCH_JUDGE_MAX_CONCURRENT', '1024'))


async def judge_semantic_match(
    question: str,
    predicted: str,
    expected: str,
    *,
    judge_prompt_template: str | None = None,
    judge_system_prompt: str | None = None,
    max_retries: int = JUDGE_DEFAULT_MAX_RETRIES,
    retry_delay: float = JUDGE_DEFAULT_RETRY_DELAY,
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
        max_retries: Max attempts on transient LLM failures (invalid XML, empty response).
        retry_delay: Base delay between retries in seconds (multiplied by attempt number).
    Returns:
        1.0 if semantically equivalent, 0.0 otherwise.

    Raises:
        RuntimeError: If LLM call fails after all retries.
    """
    system_prompt = JUDGE_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT
    if judge_system_prompt:
        system_prompt = (
            f'{JUDGE_SEMANTIC_EQUIVALENCE_SYSTEM_PROMPT}\n\n# 自定义评估指令\n\n{judge_system_prompt.strip()}'
        )
    prompt_template = judge_prompt_template or JUDGE_SEMANTIC_EQUIVALENCE_PROMPT_TEMPLATE
    user_content = prompt_template.format(
        prompt=question,
        question=question,
        response_reference=expected,
        expected=expected,
        ref_answer=expected,
        response=predicted,
        predicted=predicted,
        model_response=predicted,
    )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_content},
    ]

    ctx = require_context()
    client = ctx.get_openai_client('judge')

    last_error: RuntimeError | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            response = await client.create_completion(messages=messages)
            content = response.choices[0].message.content or ''

            is_equivalent = _extract_xml_tag(content, 'is_equivalent')
            reason = _extract_xml_tag(content, 'reason') or 'No reason provided'
            if is_equivalent not in {'0', '1', 'yes', 'no', 'true', 'false'}:
                raise RuntimeError(f'Invalid LLM judge XML response: {content[:300]!r}')

            is_match = is_equivalent in {'1', 'yes', 'true'}
            logger.info(f'LLM judge: {is_match} - {reason}')
            return 1.0 if is_match else 0.0
        except Exception as e:
            last_error = RuntimeError(str(e)) if not isinstance(e, RuntimeError) else e
            if attempt < max_retries:
                logger.warning(f'LLM judge attempt {attempt}/{max_retries} failed, retrying: {e}')
                await asyncio.sleep(retry_delay * attempt)

    raise last_error  # type: ignore[misc]


class EvalMode(str, Enum):
    EXACT_MATCH = 'exact_match'
    LLM_JUDGE = 'llm_judge'
    DEEPSEARCH_QA = 'deepsearch_qa'
    DRACO_RUBRIC_PER_CRITERION = 'draco_rubric_per_criterion'
    DRACO_RUBRIC_ONESHOT = 'draco_rubric_oneshot'
    REWARD_SERVICE = 'reward_service'
    WIDESEARCH = 'wide_search'
    INFERENCE_ONLY = 'inference_only'


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
    max_retries: int = JUDGE_DEFAULT_MAX_RETRIES
    retry_delay: float = JUDGE_DEFAULT_RETRY_DELAY
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


def _make_async_judge(client) -> LLMFunc:
    """Wrap an async Nexus OpenAI client into an async LLM callable for WideSearch.

    Unlike the previous sync bridge (``run_in_executor`` + ``run_coroutine_threadsafe``
    + blocking ``future.result``), this runs the judge call directly on the event
    loop so many queries can be scored concurrently without occupying executor
    threads or contending for the GIL with the rollout loop.

    Three safeguards address the timeout storm seen in RL training:
      * a process-local semaphore bounds concurrent judge calls (anti burst);
      * ``asyncio.wait_for`` enforces a per-call wall-clock timeout that *cleanly*
        cancels the underlying coroutine (including any litellm backoff sleep),
        releasing the slot instead of leaking the connection;
      * a small bounded retry loop on top of the client's own retries.
    """
    limiter = get_rate_limiter('widesearch_judge', max_concurrent=JUDGE_MAX_CONCURRENT)

    async def async_judge(messages: list[dict]) -> str | None:
        for attempt in range(1, JUDGE_DEFAULT_MAX_RETRIES + 1):
            try:
                async with limiter:
                    response = await asyncio.wait_for(
                        client.create_completion(messages=messages),
                        timeout=JUDGE_CALL_TIMEOUT_SEC,
                    )
                return response.choices[0].message.content
            except asyncio.TimeoutError:
                logger.warning(
                    f'Judge LLM timed out after {JUDGE_CALL_TIMEOUT_SEC}s '
                    f'(attempt {attempt}/{JUDGE_DEFAULT_MAX_RETRIES})'
                )
            except Exception as exc:
                logger.warning(f'Judge LLM call failed (attempt {attempt}/{JUDGE_DEFAULT_MAX_RETRIES}): {exc}')
            if attempt < JUDGE_DEFAULT_MAX_RETRIES:
                await asyncio.sleep(JUDGE_DEFAULT_RETRY_DELAY * attempt)
        return None

    return async_judge


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
    from nexus.extensions.tasks.search.wide_search_evaluation.data_loader import (
        WideSearchQuery,
        WideSearchResponse,
    )
    from nexus.extensions.tasks.search.wide_search_evaluation.evaluation import (
        EvaluationResult,
        evaluate_single_query,
    )

    detail = WideSearchEvaluationDetails()

    # Parse prediction
    try:
        pred_df = unified_to_dataframe(predicted)
    except Exception as exc:
        logger.warning(f'Failed to parse prediction table: {exc}')
        detail.status = 'failed'
        detail.msg = f'Failed to parse prediction table: {exc}'
        return 0.0, detail

    # Parse ground truth
    try:
        exp_df = flat_string_to_dataframe(expected)
    except Exception as exc:
        logger.warning(f'Failed to parse ground truth table: {exc}')
        detail.status = 'failed'
        detail.msg = f'Failed to parse ground truth table: {exc}'
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

    # Obtain an async LLM wrapper from the judge client (optional)
    llm_func: LLMFunc = None
    try:
        ctx = require_context()
        judge_client = ctx.get_openai_client('judge')
        llm_func = _make_async_judge(judge_client)
        logger.debug('Judge client available: LLM-based evaluation steps enabled')
    except Exception:
        logger.debug('No judge client configured: LLM-based evaluation steps disabled')

    # Run the evaluation pipeline directly on the event loop; judge calls are
    # awaited cooperatively so many queries score concurrently without threads.
    eval_result: EvaluationResult = await evaluate_single_query(
        query=ws_query,
        response=ws_response,
        llm_func=llm_func,
        response_df=pred_df,
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
) -> float:
    """LLM-based semantic evaluation."""
    return await judge_semantic_match(
        question,
        predicted,
        expected,
        judge_prompt_template=judge_prompt_template,
        judge_system_prompt=judge_system_prompt,
    )


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
        for attempt in range(1, max(1, config.max_retries) + 1):
            try:
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    response = await client.post(config.url, json=json_data, headers=headers)
                    response.raise_for_status()
                    return extract_reward_score(response.json())
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt < config.max_retries and (
                    not isinstance(e, httpx.HTTPStatusError) or e.response.status_code >= 500
                ):
                    logger.warning(f'Reward service attempt {attempt}/{config.max_retries} failed, retrying: {e}')
                    await asyncio.sleep(config.retry_delay * attempt)
                    continue
                raise
        raise RuntimeError('Reward service retries exhausted without a response')
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
        # Deprecated compatibility flag. XML prompt judging is always used.
        judge_use_tool_calling: bool | None = None,
        # Reward Service configuration (resolved from env if not provided)
        reward_config: RewardServiceConfig | None = None,
        # WideSearch evaluation configuration
        widesearch_evaluation: dict | None = None,
        instance_id: str = 'unknown',
        # DRACO Rubric configuration
        draco_batch_size: int = 0,
        draco_max_concurrency: int = 10,
        draco_max_response_length: int = 65536,
    ):
        self.mode = mode
        self.instance_id = instance_id
        # LLM Judge params
        self.judge_prompt_template = judge_prompt_template
        self.judge_system_prompt = judge_system_prompt
        if judge_use_tool_calling is not None:
            logger.info('judge_use_tool_calling is deprecated and ignored; using XML prompt judge')
        # Reward Service config — auto-resolve from env vars if not explicitly provided
        self.reward_config = reward_config if reward_config is not None else self._resolve_reward_config(mode)
        # WideSearch config
        self.widesearch_evaluation = widesearch_evaluation
        # DRACO Rubric config
        self.draco_batch_size = draco_batch_size
        self.draco_max_concurrency = draco_max_concurrency
        self.draco_max_response_length = draco_max_response_length

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
            )
        if self.mode == EvalMode.DEEPSEARCH_QA:
            score, _ = await deepsearch_qa_eval(
                normalized_question,
                prediction,
                reference,
                judge_prompt_template=self.judge_prompt_template,
                judge_system_prompt=self.judge_system_prompt,
            )
            return score
        if self.mode in (EvalMode.DRACO_RUBRIC_PER_CRITERION, EvalMode.DRACO_RUBRIC_ONESHOT):
            # Per-Criterion: batch_size=0，逐条并行评估
            # One-Shot: batch_size=draco_batch_size，批量打包评估（默认 9999 即全部打包为一次调用）
            effective_batch_size = (
                0
                if self.mode == EvalMode.DRACO_RUBRIC_PER_CRITERION
                else (self.draco_batch_size if self.draco_batch_size > 0 else 9999)
            )
            score, _ = await draco_rubric_eval(
                normalized_question,
                prediction,
                reference,
                judge_system_prompt=self.judge_system_prompt,
                max_response_length=self.draco_max_response_length,
                batch_size=effective_batch_size,
                max_concurrency=self.draco_max_concurrency,
            )
            return score
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

        if self.mode == EvalMode.DEEPSEARCH_QA:
            score, details = await deepsearch_qa_eval(
                normalized_question,
                prediction,
                reference,
                judge_prompt_template=self.judge_prompt_template,
                judge_system_prompt=self.judge_system_prompt,
            )
            return score, details

        if self.mode in (EvalMode.DRACO_RUBRIC_PER_CRITERION, EvalMode.DRACO_RUBRIC_ONESHOT):
            effective_batch_size = (
                0
                if self.mode == EvalMode.DRACO_RUBRIC_PER_CRITERION
                else (self.draco_batch_size if self.draco_batch_size > 0 else 9999)
            )
            score, details = await draco_rubric_eval(
                normalized_question,
                prediction,
                reference,
                judge_system_prompt=self.judge_system_prompt,
                max_response_length=self.draco_max_response_length,
                batch_size=effective_batch_size,
                max_concurrency=self.draco_max_concurrency,
            )
            return score, details

        logger.warning(f'evaluate_with_details for {self.mode} is not supported yet')
        score = await self.evaluate(prediction=prediction, reference=reference, question=normalized_question)
        return score, None
