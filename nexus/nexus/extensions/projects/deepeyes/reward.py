"""DeepEyes reward scoring using LLM-as-a-judge.

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
    '[Question]: Is the countertop tan or blue?\n'
    '[Standard Answer]: The countertop is tan.\n'
    "[Model's Answer]: tan\n"
    '[Your Judgement]: CORRECT\n'
    '---\n'
    '**Example 2:**\n'
    '[Question]: Is the man phone both blue and closed?\n'
    '[Standard Answer]: Yes, the man phone is both blue and closed.\n'
    "[Model's Answer]: No.\n"
    '[Your Judgement]: INCORRECT\n'
    '---\n'
    '**Task:**\n'
    '[Question]: {question}\n'
    '[Standard Answer]: {ground_truth}\n'
    "[Model's Answer]: {answer}\n"
    '[Your Judgement]:'
)


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    """Extract the model's answer text and detect format errors.

    Returns:
        (answer_text, is_format_error)
    """
    is_format_error = False

    # Check <think> tag pairing
    if solution_str.count('<think>') != solution_str.count('</think>'):
        is_format_error = True

    # Get content after </think>
    if '</think>' in solution_str:
        predict_no_think = solution_str.split('</think>')[-1].strip()
    else:
        predict_no_think = solution_str.strip()

    # Check <answer> tag pairing
    if predict_no_think.count('<answer>') != predict_no_think.count('</answer>'):
        is_format_error = True

    # Strategy 1: extract from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Penalise judge-hacking (excessively long answers)
        if len(answer_text) >= 1000:
            is_format_error = True
        return answer_text, is_format_error

    # No <answer> tags — format error
    is_format_error = True

    # Strategy 2: content after last </tool_response>
    tool_response_match = re.search(
        r'</tool_response>\s*assistant\s*\n(.*?)$',
        predict_no_think,
        re.DOTALL | re.MULTILINE,
    )
    if tool_response_match:
        return tool_response_match.group(1).strip(), is_format_error

    # Strategy 3: strip tool tags after </think>
    if '</think>' in solution_str:
        remaining = predict_no_think
        remaining = re.sub(r'<tool_call>.*?</tool_call>', '', remaining, flags=re.DOTALL)
        remaining = re.sub(r'<tool_response>.*?</tool_response>', '', remaining, flags=re.DOTALL)
        remaining = re.sub(r'\b(user|assistant)\b', '', remaining)
        answer_text = remaining.strip()
        if answer_text:
            return answer_text, is_format_error

    # Strategy 4: full solution as last resort
    return solution_str.strip(), is_format_error


def _count_tool_calls_in_trajectory(trajectory: list[dict[str, Any]]) -> tuple[int, int]:
    """Count successful and failed tool calls in the trajectory.

    Returns:
        (n_success, n_failed) — counts of tool messages with and without meta.error.
    """
    n_success = 0
    n_failed = 0
    for msg in trajectory:
        if msg.get('role') == 'tool':
            meta = msg.get('meta') or {}
            if meta.get('error'):
                n_failed += 1
            else:
                n_success += 1
    return n_success, n_failed


def _count_tool_records(tool_records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count successful and failed tool calls from context-updated tool records."""
    n_success = 0
    n_failed = 0
    for record in tool_records:
        if record.get('success') and not record.get('error'):
            n_success += 1
        else:
            n_failed += 1
    return n_success, n_failed


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
    tool_records: list[dict[str, Any]] | None = None,
) -> float:
    """Compute tool_reward shared by both scoring functions."""
    if tool_records is not None:
        n_success, n_failed = _count_tool_records(tool_records)
        has_tool = n_success > 0
    elif trajectory is not None:
        n_success, n_failed = _count_tool_calls_in_trajectory(trajectory)
        has_tool = n_success > 0
    else:
        has_tool = bool(
            re.search(r'<tool_call>.*?</tool_call>', solution_str, re.DOTALL)
            or re.search(r'<tool_response>.*?</tool_response>', solution_str, re.DOTALL)
        )
        n_success, n_failed = (1 if has_tool else 0), 0

    return max(0.0, 1.0 - 0.3 * n_failed) if has_tool and acc_reward > 0.5 else 0.0


async def compute_score(
    solution_str: str,
    ground_truth: str,
    question: str,
    judge_client: OpenAICompatibleClient,
    trajectory: list[dict[str, Any]] | None = None,
    tool_records: list[dict[str, Any]] | None = None,
) -> ScoreResult:
    """Compute the DeepEyes reward score.

    Args:
        solution_str: Last assistant response text (used for answer extraction and format check).
        ground_truth: Reference answer string.
        question: Original question text (used for judge context).
        judge_client: OpenAI-compatible client to use as the LLM judge.
        trajectory: Full conversation trajectory. When provided, tool usage is detected
            from the trajectory (assistant tool_calls + tool messages) rather than from
            solution_str. Defaults to None (falls back to regex on solution_str).
        tool_records: Structured tool execution records collected via context_updates.
            When provided, this takes precedence over trajectory-based detection.

    Returns:
        ScoreResult with:
        - score: 0.5 * acc_reward + 0.5 * tool_reward  (in [0, 1])
        - acc_reward: 1.0 if judge says CORRECT, else 0.0
        - tool_reward: 1.0 if tools used and answer correct, else 0.0
        - correct: True if acc_reward > 0.5 (i.e. judge said CORRECT)
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
        judgement = response.choices[0].message.content or ''
        judgement = judgement.strip()

        if re.search(r'\bCORRECT\b', judgement, re.IGNORECASE):
            acc_reward = 1.0
        elif re.search(r'\bINCORRECT\b', judgement, re.IGNORECASE):
            acc_reward = 0.0
        else:
            logger.warning(
                f'DeepEyes judge: unexpected response "{judgement}" '
                f'(answer="{answer_text[:100]}", gt="{ground_truth[:100]}")'
            )
    except Exception as exc:
        logger.warning(f'DeepEyes judge: LLM call failed: {exc}')

    # Penalise excessively long answers (judge hacking)
    if len(answer_text) >= 1000:
        acc_reward = 0.0

    tool_reward = _compute_tool_reward(acc_reward, trajectory, solution_str, tool_records=tool_records)
    final_score = 0.5 * acc_reward + 0.5 * tool_reward

    logger.debug(f'DeepEyes reward: acc={acc_reward:.1f}, tool={tool_reward:.2f} → score={final_score:.3f}')

    return ScoreResult(
        score=final_score,
        acc_reward=acc_reward,
        tool_reward=tool_reward,
    )


def _extract_boxed_content(text: str) -> str | None:
    r"""Extract the content inside the last ``\boxed{...}`` in *text*.

    Handles nested braces so that ``\boxed{f(x) = \frac{1}{2}}`` is correctly
    extracted as ``f(x) = \frac{1}{2}``.
    """
    # Find the last occurrence of \boxed{
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return None
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    # Strip whitespace and common wrappers
    answer = answer.strip()
    # Remove surrounding $ signs (LaTeX inline math)
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer[1:-1].strip()
    # Remove trailing period / percent duplicates
    return answer.rstrip('.')


def compute_rule_score(
    solution_str: str,
    ground_truth: str,
    trajectory: list[dict[str, Any]] | None = None,
    tool_records: list[dict[str, Any]] | None = None,
) -> ScoreResult:
    r"""Compute a rule-based reward by extracting the answer from ``\boxed{}``.

    Uses the same scoring formula as :func:`compute_score`:
        ``final_score = 0.5 * acc_reward + 0.5 * tool_reward``

    Args:
        solution_str: Last assistant response text.
        ground_truth: Reference answer string.
        trajectory: Full conversation trajectory (for tool-usage detection).
        tool_records: Structured tool execution records collected via context_updates.
            When provided, this takes precedence over trajectory-based detection.

    Returns:
        ScoreResult matching the same interface as :func:`compute_score`.
    """
    boxed = _extract_boxed_content(solution_str)

    acc_reward = 1.0 if boxed is not None and _normalize_answer(boxed) == _normalize_answer(ground_truth) else 0.0

    tool_reward = _compute_tool_reward(acc_reward, trajectory, solution_str, tool_records=tool_records)
    final_score = 0.5 * acc_reward + 0.5 * tool_reward

    logger.debug(
        f'DeepEyes rule reward: acc={acc_reward:.1f}, '
        f'tool={tool_reward:.2f} boxed="{boxed}", gt="{ground_truth}" → score={final_score:.3f}'
    )

    return ScoreResult(
        score=final_score,
        acc_reward=acc_reward,
        tool_reward=tool_reward,
    )
