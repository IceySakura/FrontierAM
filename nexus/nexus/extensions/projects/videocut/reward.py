"""VideoCut reward scoring using LLM-as-a-judge.

    final_score = 0.5 * acc_reward + 0.5 * tool_reward

where:
- acc_reward (0 or 1): semantic correctness judged by an LLM
- tool_reward (0 or 1): bonus for using tools when the answer is correct

The final score is in [0, 1].
"""

import logging
import re

from typing import Any

from nexus.core import OpenAICompatibleClient


logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Your task is to determine if a model's answer is semantically "
    'equivalent to a provided standard answer, given a specific question.\n'
    "Your evaluation must be strict. The model's answer is only correct if it fully matches the "
    'meaning of the standard answer.\n'
    'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". '
    'Do not provide any explanation or other text.'
)

_JUDGE_USER_TEMPLATE = (
    "I will provide a question, a standard answer, and a model's answer. "
    "You must evaluate if the model's answer is correct.\n\n"
    '---\n'
    '**Example 1:**\n'
    '[Question]: What activity is happening in the video between 10s and 20s?\n'
    '[Standard Answer]: The person is running.\n'
    "[Model's Answer]: running\n"
    '[Your Judgement]: CORRECT\n'
    '---\n'
    '**Example 2:**\n'
    '[Question]: What color is the car shown at the beginning of the video?\n'
    '[Standard Answer]: The car is red.\n'
    "[Model's Answer]: blue\n"
    '[Your Judgement]: INCORRECT\n'
    '---\n'
    '**Task:**\n'
    '[Question]: {question}\n'
    '[Standard Answer]: {ground_truth}\n'
    "[Model's Answer]: {answer}\n"
    '[Your Judgement]:'
)


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    """Extract the model's final answer from the last assistant message.

    The standard OpenAI tool-calling path expects the last assistant turn to be
    plain text.

    Returns:
        (answer_text, is_format_error)
    """
    text = solution_str.strip()
    is_format_error = False

    if not text:
        return '', True

    if not text or len(text) >= 1000:
        is_format_error = True

    return text, is_format_error


def _count_tool_calls_in_trajectory(trajectory: list[dict[str, Any]]) -> tuple[int, int]:
    """Count successful and failed tool calls in the trajectory.

    Native OpenAI tool-calling stores results as ``role='tool'`` messages.

    Returns:
        (n_success, n_failed) — counts of successful and failed tool calls.
    """
    n_success = 0
    n_failed = 0
    for msg in trajectory:
        role = msg.get('role')
        meta = msg.get('meta') or {}
        if role == 'tool':
            if meta.get('error'):
                n_failed += 1
            else:
                n_success += 1
    return n_success, n_failed


def _extract_option_letter(answer: str) -> str | None:
    """Extract a multiple-choice option letter from an answer if present."""
    patterns = [
        r'^\s*([A-Z])(?:[\.\):]|\s|$)',
        r'\boption\s+([A-Z])\b',
        r'\banswer\s+is\s+([A-Z])\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


class ScoreResult:
    """Result of compute_score."""

    __slots__ = ('score', 'acc_reward', 'tool_reward')

    def __init__(self, score: float, acc_reward: float, tool_reward: float) -> None:
        self.score = score
        self.acc_reward = acc_reward
        self.tool_reward = tool_reward

    @property
    def correct(self) -> bool:
        """Whether the judge considered the answer correct."""
        return self.acc_reward > 0.5


def _compute_tool_reward(
    acc_reward: float,
    trajectory: list[dict[str, Any]] | None,
    solution_str: str,
) -> float:
    """Compute tool_reward: bonus for using video_cut_tool when the answer is correct."""
    if trajectory is not None:
        n_success, n_failed = _count_tool_calls_in_trajectory(trajectory)
        has_tool = n_success > 0
    else:
        has_tool = False
        n_success, n_failed = (1 if has_tool else 0), 0

    return max(0.0, 1.0 - 0.3 * n_failed) if has_tool and acc_reward > 0.5 else 0.0


def _extract_judgement_text(response: Any) -> str:
    """Extract judge text from the first choice message content."""
    try:
        content = response.choices[0].message.content
    except Exception:
        return ''

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get('text') for item in content if isinstance(item, dict) and isinstance(item.get('text'), str)]
        return '\n'.join(parts).strip()
    return ''


async def compute_score(
    solution_str: str,
    ground_truth: str,
    question: str,
    judge_client: OpenAICompatibleClient,
    trajectory: list[dict[str, Any]] | None = None,
) -> ScoreResult:
    """Compute the VideoCut reward score using LLM-as-a-judge.

    Args:
        solution_str: Last assistant response text.
        ground_truth: Reference answer string.
        question: Original question text (used for judge context).
        judge_client: OpenAI-compatible client to use as the LLM judge.
        trajectory: Full conversation trajectory. When provided, tool usage is detected
            from the trajectory rather than from solution_str.

    Returns:
        ScoreResult with:
        - score: 0.5 * acc_reward + 0.5 * tool_reward  (in [0, 1])
        - acc_reward: 1.0 if judge says CORRECT, else 0.0
        - tool_reward: 1.0 if tools used and answer correct, else 0.0
        - correct: True if acc_reward > 0.5
    """
    answer_text, _ = _extract_answer(solution_str)

    # --- Accuracy via LLM judge ---
    user_prompt = _JUDGE_USER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer_text,
    )

    acc_reward = 0.0
    try:
        response = await judge_client.create_completion(
            messages=[
                {'role': 'system', 'content': _JUDGE_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        judgement = _extract_judgement_text(response)

        if re.search(r'\bINCORRECT\b', judgement, re.IGNORECASE):
            acc_reward = 0.0
        elif re.search(r'\bCORRECT\b', judgement, re.IGNORECASE):
            acc_reward = 1.0
        else:
            logger.warning(
                f'VideoCut judge: unexpected response "{judgement}" '
                f'(answer="{answer_text[:100]}", gt="{ground_truth[:100]}")'
            )
    except Exception as exc:
        logger.warning(f'VideoCut judge: LLM call failed: {exc}')

    # Penalise excessively long answers (judge hacking)
    if len(answer_text) >= 1000:
        acc_reward = 0.0

    tool_reward = _compute_tool_reward(acc_reward, trajectory, solution_str)
    final_score = 0.5 * acc_reward + 0.5 * tool_reward

    logger.debug(f'VideoCut reward: acc={acc_reward:.1f}, tool={tool_reward:.2f} → score={final_score:.3f}')

    return ScoreResult(
        score=final_score,
        acc_reward=acc_reward,
        tool_reward=tool_reward,
    )


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip()
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer[1:-1].strip()
    return re.sub(r'\s+', ' ', answer.rstrip('.')).strip().lower()


def compute_rule_score(
    solution_str: str,
    ground_truth: str,
    trajectory: list[dict[str, Any]] | None = None,
) -> ScoreResult:
    """Compute a rule-based reward for multiple-choice answers.

    Uses the same scoring formula as :func:`compute_score`:
        ``final_score = 0.5 * acc_reward + 0.5 * tool_reward``

    Args:
        solution_str: Last assistant response text.
        ground_truth: Reference answer string.
        trajectory: Full conversation trajectory (for tool-usage detection).

    Returns:
        ScoreResult matching the same interface as :func:`compute_score`.
    """
    answer_text, _ = _extract_answer(solution_str)
    pred_option = _extract_option_letter(answer_text)
    gt_option = _extract_option_letter(ground_truth)

    if pred_option and gt_option:
        acc_reward = 1.0 if pred_option == gt_option else 0.0
    else:
        acc_reward = 1.0 if _normalize_answer(answer_text) == _normalize_answer(ground_truth) else 0.0

    tool_reward = _compute_tool_reward(acc_reward, trajectory, solution_str)
    final_score = 0.5 * acc_reward + 0.5 * tool_reward

    logger.debug(
        f'VideoCut rule reward: acc={acc_reward:.1f}, '
        f'tool={tool_reward:.2f}, answer="{answer_text}", gt="{ground_truth}" → score={final_score:.3f}'
    )

    return ScoreResult(
        score=final_score,
        acc_reward=acc_reward,
        tool_reward=tool_reward,
    )
