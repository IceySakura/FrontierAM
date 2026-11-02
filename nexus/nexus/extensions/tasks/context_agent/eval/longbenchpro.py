"""LongBench-Pro evaluator — LLM-based consistency check.

Unlike AA-LCR (where ``answer`` is a list of *alternative* acceptable
answers, any of which counts as correct), LongBench-Pro's ``answer`` is a
list of the *components* of a single gold answer (e.g. a ranking order, a
set of typo corrections, a list of clause numbers). The candidate must be
consistent with the **whole** gold answer, so the list is joined into one
composite official answer and judged once.

Models are prompted (``question_thinking``) to emit their final answer
after an ``[Answer]`` / ``[答案]`` marker; the LLM judge is robust to the
surrounding reasoning text.

Data schema
-----------
Each task has::

    {
        "context_messages": list,
        "context_question": str,
        "messages": list,
        "answer": list[str],   # components of ONE gold answer
        "metadata": {
            "task_id": str,
            "dataset_name": "LongBenchPro",
            "sub_category": str,   # primary_task, e.g. "T1. Retrieval & Ranking"
            "language": "en" | "zh",
        },
    }
"""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# Strip the model's reasoning and keep only what follows the answer marker.
_ANSWER_MARKERS = [
    r'\[Answer\]',
    r'\[答案\]',
    r'The correct answer is',
]

_JUDGE_PROMPT = (
    'You are grading an answer to a long-context question.\n'
    'The CANDIDATE ANSWER is CORRECT only if it is consistent with the '
    'complete OFFICIAL ANSWER below. The official answer may consist of '
    'several components (e.g. a ranked list, a set of items, multiple '
    'corrections); ALL components must be present and, where the question '
    'asks for an order/ranking, in the correct order. Ignore differences in '
    'formatting, surrounding explanation, or reasoning text.\n\n'
    'QUESTION (for reference only): {question}\n\n'
    'OFFICIAL ANSWER (all components, one per line):\n{official_answer}\n\n'
    'CANDIDATE ANSWER TO ASSESS:\n{candidate_answer}\n\n'
    'Reply only with CORRECT or INCORRECT.'
)


def extract_final_answer(response: str) -> str:
    """Return the text following the last answer marker, if any.

    Falls back to the full response when no marker is present.
    """
    last_idx = -1
    last_end = -1
    for marker in _ANSWER_MARKERS:
        for m in re.finditer(marker, response, flags=re.IGNORECASE):
            if m.start() > last_idx:
                last_idx = m.start()
                last_end = m.end()
    if last_idx >= 0:
        return response[last_end:].strip()
    return response.strip()


async def _judge_once(
    question: str,
    official_answer: str,
    candidate_answer: str,
    llm,
) -> bool | None:
    """Ask the LLM whether *candidate_answer* matches *official_answer*.

    Returns ``True`` / ``False``, or ``None`` on parse failure.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        official_answer=official_answer,
        candidate_answer=candidate_answer,
    )
    for attempt in range(MAX_RETRIES):
        try:
            raw = await llm.chat([{'role': 'user', 'content': prompt}])
            if isinstance(raw, dict):
                raw = raw.get('response') or raw.get('content') or ''
            verdict = raw.strip().upper()
            if 'CORRECT' in verdict and 'INCORRECT' not in verdict:
                return True
            if 'INCORRECT' in verdict:
                return False
            logger.warning(
                'LongBenchPro judge returned ambiguous verdict (attempt %s/%s): %r',
                attempt + 1,
                MAX_RETRIES,
                raw.strip(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                'LongBenchPro judge call failed (attempt %s/%s): %s',
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
    return None


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* against the composite gold ``answer`` list.

    The ``answer`` list components are joined into a single official answer;
    the candidate is correct (score=1) only if the LLM judges it consistent
    with that complete answer.
    """
    if not model_output or not model_output.strip():
        return {
            'score': 0,
            'rationale': 'No model output',
            'requirement_status': [],
            'judge_prompt': None,
        }

    question = task.get('context_question', '')
    answers = task.get('answer', [])
    if not isinstance(answers, list):
        answers = [str(answers)]

    if not answers:
        return {
            'score': 0,
            'rationale': 'No official answer provided',
            'requirement_status': [],
            'judge_prompt': None,
        }

    official_answer = '\n'.join(str(a) for a in answers)
    candidate = extract_final_answer(model_output)

    result = await _judge_once(question, official_answer, candidate, llm)

    if result is True:
        return {
            'score': 1,
            'rationale': f'LLM judge: CORRECT (gold: {official_answer[:80]}…)',
            'requirement_status': [],
            'judge_prompt': None,
        }
    if result is False:
        return {
            'score': 0,
            'rationale': f'LLM judge: INCORRECT (gold: {official_answer[:80]}…)',
            'requirement_status': [],
            'judge_prompt': None,
        }
    return {
        'score': 0,
        'rationale': 'LLM judge: verdict unavailable after retries',
        'requirement_status': [],
        'judge_prompt': None,
    }
