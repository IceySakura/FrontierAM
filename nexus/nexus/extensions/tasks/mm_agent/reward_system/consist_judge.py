"""Consistency judge using Taiji API for reward evaluation.

Uses an external LLM (via Taiji API) to judge whether the model's response
is consistent with the reference answer, based on REWARD_SERVICE_PROMPT_TEMPLATE.
"""

import json
import logging
import os

from nexus.extensions.tasks.mm_agent.reward_system.taiji_api import get_model_response


logger = logging.getLogger(__name__)

# Default judge model(s), can be overridden via CONSIST_JUDGE_MODEL env (comma-separated)
DEFAULT_JUDGE_MODEL = 'rm_gpt-oss_gy_a800_new_inter,rm_gpt-oss_gy_a800_new_outer'

REWARD_SERVICE_PROMPT_TEMPLATE = """
【评估任务】
针对输入的[问题]，判断[参考答案]和[模型回复]的**答案总结**是否一致，一致输出1，不一致输出0。

【任务输入】
[问题] = {prompt}

[参考答案] = {ref_answer}

[模型回复] = {model_response}

【评估步骤】
- step1：针对输入的[问题]，把[参考答案]的核心结论归纳总结成一个简洁的**答案总结**，注意：**答案总结**务必完整、清晰、具体、明确；
- step2：针对输入的[问题]，将[模型回复]的核心结论归纳总结成一个简洁的**答案总结**，注意：**答案总结**务必完整、清晰、具体、明确，如果[模型回复]本身没给出明确答案，例如只给了代码，没显示地给出具体的执行结果，其**答案总结**直接设置为**没有明确答案**，一致性直接判为不一致；
- step3：判断[参考答案]的**答案总结**和[模型回复]的**答案总结**的核心语义是否一致，一致输出1，不一致输出0；

【一致性判断标准】
1. 如果两个**答案总结**的核心语义完全等价，则判为一致；
2. 如果两个**答案总结**的表达形式不同，但针对[问题]而言，两者回答[问题]的语义相同，也判为一致；
3. 如果两个**答案总结**在关键数据或结论存在差异，则判为不一致；
4. 如果两个**答案总结**任意一方为**没有明确答案**，则直接判为不一致；

【注意事项】
- 如果[模型回复]中只给出了代码，但是没有显式地给出执行结果，一致性直接判为不一致；

【任务输出】
严格按JSON格式输出：
{{
    "参考答案的答案总结": <抽取或总结[参考答案]中针对[问题]的**答案总结**>,
    "模型回复的答案总结"：<抽取或总结[模型回复]中针对[问题]的**答案总结**>,
    "分析": <简要说明判断依据>,
    "是否一致": <根据“一致性判断标准”判断**参考答案的答案总结**和**模型回复的答案总结**是否一致，一致输出1，不一致输出0>
}}

【最终输出】
""".strip()


def _extract_json_objects(text: str) -> list[dict]:
    """Extract all top-level JSON objects from ``text`` in order.

    Uses ``json.JSONDecoder().raw_decode`` from each ``{`` position rather
    than a regex. The real JSON parser correctly handles braces that appear
    *inside* string values (e.g. the ``}`` in ``{"分析": "按 f(x)} 推导"}``),
    multiple concatenated JSON objects, and reasoning prose before/after the
    object. On a successful decode we advance past the parsed object so its
    inner braces are never re-examined as candidate starts.

    Args:
        text: Arbitrary text that may contain zero or more JSON objects.

    Returns:
        The decoded ``dict`` objects, in order of appearance. Non-dict
        top-level JSON values (e.g. arrays, numbers) are skipped.
    """
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    idx = 0
    n = len(text)
    while idx < n:
        brace = text.find('{', idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except ValueError:
            # Not a valid JSON object starting here; advance one char and retry.
            idx = brace + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = end
    return objects


def _parse_judge_response(response_text: str | None) -> tuple[float, dict]:
    """Parse the judge model response for the V1 JSON prompt template.

    The V1 template instructs the judge to return strict JSON with the
    following keys (in Chinese):

        {
            "参考答案的答案总结": ...,
            "模型回复的答案总结": ...,
            "分析": ...,
            "是否一致": 0 or 1
        }

    The judge may emit extra reasoning text (and even a separate reasoning
    JSON object) before/after the answer, so we decode every top-level JSON
    object and use the LAST one carrying the ``是否一致`` verdict. If no such
    object is found, ``ValueError`` is raised so the caller can retry.

    Args:
        response_text: Raw text response from the judge model.

    Returns:
        A tuple of (score, details_dict).
        score is 1.0 if consistent, 0.0 otherwise.
    """
    if response_text is None:
        # Treat empty/None judge output as a parse failure so the caller
        # can retry and ultimately fall back to the -100 sentinel instead
        # of a misleading 0.0 score.
        raise ValueError('Judge model returned None response')

    text = response_text.strip()

    # Decode every top-level JSON object (string-literal aware), then pick
    # the LAST one carrying the "是否一致" verdict — the judge often emits a
    # reasoning JSON before the final answer object.
    parsed: dict | None = None
    for obj in reversed(_extract_json_objects(text)):
        if '是否一致' in obj:
            parsed = obj
            break

    if parsed is None:
        raise ValueError(f'Judge model returned invalid response: {text[:300]!r}')

    try:
        is_consistent = int(parsed['是否一致'])
    except (TypeError, ValueError) as e:
        raise ValueError(f'Judge model returned non-integer 是否一致: {parsed["是否一致"]!r}') from e

    if is_consistent not in (0, 1):
        raise ValueError(f'Judge model returned out-of-range 是否一致: {is_consistent}')

    return float(is_consistent), {
        'ref_summary': parsed.get('参考答案的答案总结', ''),
        'model_summary': parsed.get('模型回复的答案总结', ''),
        'analysis': parsed.get('分析', ''),
        'is_consistent': is_consistent,
        'raw_response': response_text,
        'parse_method': 'v1_json',
    }


async def consist_judge(
    question: str,
    response: str,
    target_answer: str,
    max_retry: int = 2,
) -> tuple[float, dict]:
    """Judge consistency between model response and reference answer.

    Uses an external LLM via Taiji API to evaluate whether the model's response
    is semantically consistent with the reference answer.

    Args:
        question: The original question.
        response: The model's response/prediction.
        target_answer: The ground truth/reference answer.
        max_retry: Maximum number of retries for the judge API call.

    Returns:
        A tuple of (score, details).
        score: 1.0 if consistent, 0.0 if not.
        details: Dictionary with evaluation details.
    """
    judge_model = [
        m.strip() for m in os.environ.get('CONSIST_JUDGE_MODEL', DEFAULT_JUDGE_MODEL).split(',') if m.strip()
    ]

    # Number of additional attempts when the judge's response cannot be
    # parsed into a valid verdict. 0 -> only a single call, no parse-retry.
    try:
        parse_retries = int(os.environ.get('CONSIST_JUDGE_PARSE_RETRIES', '2'))
    except (TypeError, ValueError):
        parse_retries = 2
    if parse_retries < 0:
        parse_retries = 0

    # Build the judge prompt using the V1 strict-JSON template.
    judge_prompt = REWARD_SERVICE_PROMPT_TEMPLATE.format(
        prompt=question,
        ref_answer=target_answer,
        model_response=response,
    )

    last_err: Exception | None = None
    total_attempts = parse_retries + 1
    for attempt in range(total_attempts):
        try:
            judge_response = await get_model_response(
                img_path=None,
                input_text=judge_prompt,
                max_token=1024,
                max_retry=1,
                MODEL=judge_model,
            )
            score, details = _parse_judge_response(judge_response)
            details['judge_model'] = judge_model
            details['parse_attempts'] = attempt + 1
            details['question'] = question
            details['target_answer'] = target_answer
            details['model_response'] = response
            return score, details
        except ValueError as e:
            # Parse failure: retry by re-calling the judge model.
            last_err = e
            logger.warning(f'Consist judge parse failed (attempt {attempt + 1}/{total_attempts}): {e}')
            continue
        except Exception as e:
            logger.error(f'Consist judge failed: {e}')
            return -100, {
                'error': str(e),
                'judge_model': judge_model,
                'question': question,
                'target_answer': target_answer,
                'model_response': response,
            }

    logger.error(f'Consist judge parse failed after {total_attempts} attempts: {last_err}')
    return -100, {
        'error': f'parse_failed_after_{total_attempts}_attempts: {last_err}',
        'judge_model': judge_model,
        'question': question,
        'target_answer': target_answer,
        'model_response': response,
    }
