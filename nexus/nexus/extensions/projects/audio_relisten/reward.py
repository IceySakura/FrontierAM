"""Audio Re-Listen reward scoring.

Mirrors the DeepEyes formula:

    final_score = 0.5 * acc_reward + 0.5 * tool_reward

where
- acc_reward (0 or 1): correctness — exact-match for multi-choice, or
  LLM-as-a-judge for open-ended.
- tool_reward (0 or 1): bonus when the answer is correct AND the model
  invoked the audio_relisten tool at least once.

This intentionally keeps the same shape as DeepEyes so the same training /
evaluation harness can be reused without changes.
"""

from __future__ import annotations

import logging
import re

from typing import Any

from nexus.core import OpenAICompatibleClient


logger = logging.getLogger(__name__)


_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Your task is to determine if a model's answer is semantically "
    'equivalent to a provided standard answer for an audio-comprehension question.\n'
    "Your evaluation must be strict. The model's answer is only correct if it fully matches the "
    'meaning of the standard answer.\n'
    'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". '
    'Do not provide any explanation or other text.'
)

_JUDGE_USER_TEMPLATE = (
    "I will provide a question, a standard answer, and a model's answer. "
    "You must evaluate if the model's answer is correct.\n\n"
    '---\n'
    '[Question]: {question}\n'
    '[Standard Answer]: {ground_truth}\n'
    "[Model's Answer]: {answer}\n"
    '[Your Judgement]:'
)


class ScoreResult:
    """Result of compute_score / compute_rule_score."""

    __slots__ = ('score', 'acc_reward', 'tool_reward')

    def __init__(self, score: float, acc_reward: float, tool_reward: float) -> None:
        self.score = score
        self.acc_reward = acc_reward
        self.tool_reward = tool_reward

    @property
    def correct(self) -> bool:
        return self.acc_reward > 0.5


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    """Extract the model's final answer from the assistant text.

    Returns ``(answer_text, is_structured)``. ``is_structured`` is True when
    the answer was extracted from a clean wrapper (``<answer>``, ``\\boxed{}``)
    so the caller can apply tighter matching rules.

    Tries, in order:
      1. ``<answer>...</answer>`` tags (structured)
      2. ``\\boxed{...}`` (structured)
      3. Content after ``</think>`` (unstructured)
      4. Whole solution string (unstructured)
    """
    # Strategy 1: <answer> tag
    m = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if m:
        return m.group(1).strip(), True

    # Strategy 2: \boxed{...}
    boxed = _extract_boxed_content(solution_str)
    if boxed is not None:
        return boxed.strip(), True

    # Strategy 3: after </think>
    if '</think>' in solution_str:
        return solution_str.split('</think>')[-1].strip(), False

    return solution_str.strip(), False


def _extract_boxed_content(text: str) -> str | None:
    """Extract the content inside the last ``\\boxed{...}`` in *text* (handling nested braces)."""
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
    return text[start : i - 1]


# ---------------------------------------------------------------------------
# Tool usage detection
# ---------------------------------------------------------------------------


def _count_tool_calls_in_trajectory(trajectory: list[dict[str, Any]]) -> tuple[int, int]:
    """Count successful and failed tool calls by inspecting tool messages."""
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
    n_success = 0
    n_failed = 0
    for record in tool_records:
        if record.get('success') and not record.get('error'):
            n_success += 1
        else:
            n_failed += 1
    return n_success, n_failed


def _compute_tool_reward(
    acc_reward: float,
    trajectory: list[dict[str, Any]] | None,
    tool_records: list[dict[str, Any]] | None,
) -> float:
    """Tool reward: 1.0 (minus a small failure penalty) if the answer is right and a tool was used."""
    if tool_records is not None:
        n_success, n_failed = _count_tool_records(tool_records)
    elif trajectory is not None:
        n_success, n_failed = _count_tool_calls_in_trajectory(trajectory)
    else:
        return 0.0

    if n_success == 0 or acc_reward <= 0.5:
        return 0.0
    return max(0.0, 1.0 - 0.3 * n_failed)


# ---------------------------------------------------------------------------
# Rule-based scoring (multi-choice / exact-match)
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower()).rstrip('.')


def _match_multi_choice(
    answer: str,
    ground_truth: str,
    options: list[str] | None,
    *,
    is_structured: bool,
) -> bool:
    """Match ``answer`` against ``ground_truth`` for a multi-choice question.

    Accepts either:
      - a verbatim option-letter (e.g. ``"A"`` / ``"A. Apples"``);
      - a paraphrase that exactly matches one of the options when normalized.

    The matcher intentionally does NOT do loose substring containment over the
    full assistant turn — the model often lists every option in its reasoning,
    which would otherwise produce false-positive matches. When ``is_structured``
    is False (no ``<answer>`` / ``\\boxed{}`` wrapper), we are conservative and
    only accept exact matches; loose containment is reserved for short strings
    that came out of a structured wrapper.
    """
    a_norm = _normalize(answer)
    gt_norm = _normalize(ground_truth)

    if not a_norm:
        return False
    if a_norm == gt_norm:
        return True

    # Letter-style answer (A / B / C / D, possibly wrapped in punctuation).
    # Look at the FIRST standalone letter token — the wrapped answer should be short.
    letter_match = re.match(r'^[\(\[\s]*([A-Za-z])[\)\]\.\:\s]', answer + ' ')
    if not letter_match and re.fullmatch(r'\s*[A-Za-z]\s*', answer):
        # Fallback: a single letter alone (e.g. "A").
        letter_match = re.match(r'\s*([A-Za-z])', answer)
    if options and letter_match:
        idx = ord(letter_match.group(1).upper()) - ord('A')
        if 0 <= idx < len(options) and _normalize(options[idx]) == gt_norm:
            return True

    # Substring containment is only safe when the answer came from a structured
    # wrapper, is reasonably long (so single letters don't trigger), and is
    # short enough that listing multiple options is unlikely.
    return bool(is_structured and 4 <= len(a_norm) <= 200 and (gt_norm in a_norm or a_norm in gt_norm))


def compute_rule_score(
    solution_str: str,
    ground_truth: str,
    *,
    options: list[str] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    tool_records: list[dict[str, Any]] | None = None,
) -> ScoreResult:
    """Rule-based reward used when ``ground_truth`` is a fixed string / option."""
    answer, is_structured = _extract_answer(solution_str)
    correct = _match_multi_choice(answer, ground_truth, options, is_structured=is_structured)
    acc_reward = 1.0 if correct else 0.0
    tool_reward = _compute_tool_reward(acc_reward, trajectory, tool_records)
    final = 0.5 * acc_reward + 0.5 * tool_reward
    logger.debug(
        'audio_relisten rule reward: acc=%.1f tool=%.2f answer=%r gt=%r → score=%.3f',
        acc_reward,
        tool_reward,
        answer,
        ground_truth,
        final,
    )
    return ScoreResult(score=final, acc_reward=acc_reward, tool_reward=tool_reward)


# ---------------------------------------------------------------------------
# LLM-as-a-judge scoring
# ---------------------------------------------------------------------------


async def compute_score(
    solution_str: str,
    ground_truth: str,
    question: str,
    judge_client: OpenAICompatibleClient,
    *,
    trajectory: list[dict[str, Any]] | None = None,
    tool_records: list[dict[str, Any]] | None = None,
) -> ScoreResult:
    """LLM-as-a-judge reward (used when answers are open-ended)."""
    answer, _ = _extract_answer(solution_str)
    user_prompt = _JUDGE_USER_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
    )

    acc_reward = 0.0
    try:
        response = await judge_client.create_completion(
            messages=[
                {'role': 'system', 'content': _JUDGE_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        judgement = (response.choices[0].message.content or '').strip()
        if re.search(r'\bCORRECT\b', judgement, re.IGNORECASE) and not re.search(
            r'\bINCORRECT\b', judgement, re.IGNORECASE
        ):
            acc_reward = 1.0
        elif re.search(r'\bINCORRECT\b', judgement, re.IGNORECASE):
            acc_reward = 0.0
        else:
            logger.warning('audio_relisten judge: unexpected response %r', judgement)
    except Exception as exc:
        logger.warning('audio_relisten judge: LLM call failed: %s', exc)

    tool_reward = _compute_tool_reward(acc_reward, trajectory, tool_records)
    final = 0.5 * acc_reward + 0.5 * tool_reward
    logger.debug(
        'audio_relisten model reward: acc=%.1f tool=%.2f answer=%r gt=%r → score=%.3f',
        acc_reward,
        tool_reward,
        answer,
        ground_truth,
        final,
    )
    return ScoreResult(score=final, acc_reward=acc_reward, tool_reward=tool_reward)
