"""Shared LLM-based answer verification for answer-only datasets."""

from __future__ import annotations

import json
import logging


logger = logging.getLogger(__name__)

JUDGE_PROMPT = (
    '你是一个严格的答案验证专家。请验证以下模型答案是否正确。\n\n'
    '标准答案：{ground_truth}\n\n'
    '模型答案：{model_answer}\n\n'
    '请仔细比较模型的答案与标准答案。验证标准：\n'
    '1. 数值答案：数值必须基本一致（允许小数点后两位误差，如3.92和3.94可视为正确）\n'
    '2. 文本答案：核心信息必须一致，表达方式可以不同\n'
    '3. 表格答案：数据内容必须一致，格式可以不同\n'
    '4. 无法判断或答案不明确的，视为错误\n\n'
    '请严格按照以下JSON格式输出验证结果，不要添加任何其他内容：\n\n'
    '{{"answer_1": 0}}\n\n'
    '请将answer_1的值设置为0（错误）或1（正确）。\n\n'
    '重要：只输出JSON格式，不要有任何解释、说明或其他文字。确保JSON格式完全正确。'
)

MAX_RETRIES = 10


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


async def evaluate(
    model_output: str,
    task: dict,
    llm,
    *,
    dataset_name: str = 'answer_judge',
) -> dict:
    """Evaluate *model_output* against the task ``answer`` via LLM judge."""
    ground_truth = task.get('answer', '')

    if not model_output or not model_output.strip():
        return {
            'score': 0,
            'rationale': 'No model output',
            'requirement_status': [],
            'judge_prompt': [],
        }

    prompt = JUDGE_PROMPT.format(
        ground_truth=ground_truth,
        model_answer=model_output,
    )
    messages = [{'role': 'user', 'content': prompt}]

    for attempt in range(MAX_RETRIES):
        try:
            raw = await llm.chat(messages)
            if isinstance(raw, dict):
                raw = raw.get('response', '')
            raw = _strip_markdown_fences(raw)
            result = json.loads(raw)
            score = int(result.get('answer_1', 0))
            return {
                'score': score,
                'rationale': f'LLM judge: answer_1={score}',
                'requirement_status': [],
                'judge_prompt': messages,
            }
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                '%s judge JSON parse failed (attempt %s/%s): %s',
                dataset_name,
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                '%s judge call failed (attempt %s/%s): %s',
                dataset_name,
                attempt + 1,
                MAX_RETRIES,
                exc,
            )

    return {
        'score': 0,
        'rationale': f'Judge failed after {MAX_RETRIES} attempts',
        'requirement_status': [],
        'judge_prompt': messages,
    }
