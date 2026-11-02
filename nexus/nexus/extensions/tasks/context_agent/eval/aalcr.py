"""AA-LCR evaluator — LLM-based equality check (official method).

Data schema
-----------
Each task has::

    {
        "context_messages": list,
        "context_question": str,
        "messages": list,
        "answer": list[str],   # e.g. ["1. Airline Industry (12)\\n2. ..."]
        "metadata": {
            "task_id": str,
            "dataset_name": "AALCR",
            "sub_category": str,   # e.g. "Academia"
        },
    }

``answer`` is a **list of strings**, where each string is one
acceptable reference answer.  The candidate is judged CORRECT if the
LLM deems it consistent with **any** of the official answers.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)

MAX_RETRIES = 3

_JUDGE_PROMPT = (
    'Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.\n'
    'For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.\n\n'
    'The question, for reference only: {question}\n'
    'The OFFICIAL ANSWER: {official_answer}\n'
    'CANDIDATE ANSWER TO ASSESS: {candidate_answer}\n\n'
    'Reply only with CORRECT or INCORRECT.'
)


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
                'AALCR judge returned ambiguous verdict (attempt %s/%s): %r',
                attempt + 1,
                MAX_RETRIES,
                raw.strip(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                'AALCR judge call failed (attempt %s/%s): %s',
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
    return None


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* against the ground-truth ``answer`` list.

    The candidate is marked correct (score=1) if the LLM judges it
    consistent with **any** of the official answers.

    Uses the caller-provided Judge so Nexus ``openai_clients.judge`` controls
    the model and request settings.
    """
    if not model_output or not model_output.strip():
        return {
            'score': 0,
            'rationale': 'No model output',
            'requirement_status': [],
            'judge_prompt': None,
        }

    question = task.get('context_question', '')
    answers: list[str] = task.get('answer', [])

    for official in answers:
        result = await _judge_once(question, official, model_output, llm)
        if result is True:
            return {
                'score': 1,
                'rationale': f'LLM judge: CORRECT (matched official answer: {official[:80]}…)',
                'requirement_status': [],
                'judge_prompt': None,
            }

    return {
        'score': 0,
        'rationale': (
            'LLM judge: INCORRECT against all official answers' if answers else 'No official answers provided'
        ),
        'requirement_status': [],
        'judge_prompt': None,
    }
