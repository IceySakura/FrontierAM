"""oolong evaluator — rule-based scoring ported from the official oolong code.

The scoring logic is taken verbatim (in spirit) from the official oolong
repository and wrapped into the framework's ``evaluate`` interface.

Data schema (converted to the MRCR-style task layout)
-----------------------------------------------------
Each task has::

    {
        "context_messages": list,
        "context_question": str,
        "messages": list,
        "answer": str,   # original oolong answer, a stringified python list,
                         # e.g. "['incorrect']", "[41714]",
                         #      "[datetime.date(2023, 2, 25)]"
        "metadata": {
            "task_id": str,
            "dataset_name": str,            # oolong subset, e.g. "spam"
            "sub_category": str,
            "oolong_task_group": str,       # "counting" | "timeline" | "user"
            "oolong_task": str,             # e.g. "TASK_TYPE.MOST_FREQ"
            "oolong_answer_type": str,      # e.g. "ANSWER_TYPE.NUMERIC"
            "oolong_context_window_id": int,
            ...
        },
    }

Scoring is purely rule-based, so the injected ``llm`` argument is unused.

Two scoring paths from the official code are preserved:

* ``score_synth`` — for the classification / aggregate-statistics subsets
  (counting / timeline / user tasks with LABEL / COMPARISON / NUMERIC /
  USER / MONTH_YEAR / DATE answers). This is the path used by every subset
  in the current dataset.
* ``score_dnd`` — for the ``\\boxed{}``-style "dnd" subset. Kept as a
  fallback in case such data is added later.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import re

from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)

# dateutil is optional; only needed for DATE answers. Declared once as ``Any``
# and then filled in, rather than bound by the import itself, so the absent
# case is a plain ``None`` instead of a type conflicting with the module.
_dateutil_parser: Any = None
with contextlib.suppress(Exception):
    import dateutil.parser as _dt_parser

    _dateutil_parser = _dt_parser

# oolong classification subsets present in the converted data.
DATASET_NAMES: tuple[str, ...] = (
    'agnews',
    'app_reviews',
    'formality',
    'imdb',
    'metaphors',
    'multinli',
    'negation',
    'spam',
    'trec_coarse',
    'yahoo',
)

_COMPARISON_PHRASES = ('more common', 'less common', 'same frequency')


# ---------------------------------------------------------------------------
# synth scoring (official ``synth_attempt_answer_parse`` + ``synth_process_response``)
# ---------------------------------------------------------------------------
def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Best-effort parse of a model response into a candidate answer.

    Ported from the official oolong ``synth_attempt_answer_parse``. Returns
    ``(candidate_answer, parse_confidence)`` where confidence is one of
    ``low`` / ``med`` / ``high`` / ``vhigh``.
    """
    parse_confidence = 'low'
    if ':' not in answer:  # bad start
        if len(answer) < 20:  # it's short, return the whole thing
            return answer, parse_confidence
        return answer.split()[-1], parse_confidence

    candidate_answer = answer.split(':')[-1].strip()
    candidate_answer = candidate_answer.replace('*', '')  # models like bolding
    candidate_answer = candidate_answer.replace('[', '')
    candidate_answer = candidate_answer.replace(']', '')  # answer in []
    parse_confidence = 'med'
    if 'User:' in answer or 'Answer:' in answer or 'Date:' in answer or 'Label' in answer:
        parse_confidence = 'high'
    if len(candidate_answer) < 20:
        parse_confidence = 'vhigh'
    elif 'more common' in candidate_answer:
        candidate_answer = 'more common'
    elif 'less common' in candidate_answer:
        candidate_answer = 'less common'
    elif 'same frequency' in candidate_answer:
        candidate_answer = 'same frequency'

    return candidate_answer, parse_confidence


def _parse_gold(answer_str: str):
    """Parse the stored oolong answer string into a gold value."""
    if 'datetime' not in answer_str:
        return ast.literal_eval(answer_str)[0]
    return datetime.strptime(answer_str, '[datetime.date(%Y, %m, %d)]')


def score_synth(output: str, answer_str: str, answer_type: str) -> tuple[float, str, object, object]:
    """Official synth scoring.

    Returns ``(score, parse_confidence, gold, trimmed_output)``.
    """
    gold = _parse_gold(answer_str)
    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)

    score: float = 0.0
    if str(trimmed_output) == str(gold):
        score = 1.0
    elif str(trimmed_output) in _COMPARISON_PHRASES:
        # account for these being slightly different wordings
        if str(trimmed_output) in str(gold):
            score = 1.0
    elif answer_type == 'ANSWER_TYPE.NUMERIC':  # partial credit for numbers
        try:
            trimmed_output = int(trimmed_output)
            gold = int(gold)
            score = 0.75 ** (abs(gold - trimmed_output))
        except Exception:  # noqa: BLE001
            parse_confidence = 'low'  # didn't parse as a number, bad sign
    elif answer_type == 'ANSWER_TYPE.DATE':
        if _dateutil_parser is None:
            parse_confidence = 'low'
        else:
            try:
                trimmed_output = _dateutil_parser.parse(trimmed_output)
                score = float(trimmed_output == gold)
            except Exception:  # noqa: BLE001
                parse_confidence = 'low'  # didn't parse as a date, bad sign

    return float(score), parse_confidence, gold, trimmed_output


# ---------------------------------------------------------------------------
# dnd scoring (official ``dnd_*`` helpers) — fallback for \boxed{} answers
# ---------------------------------------------------------------------------
def dnd_parse_answer(answer: str):
    """Parse the answer into int, list[str], or str (official logic)."""
    try:
        return int(answer)
    except ValueError:
        pass
    if ',' in answer:
        return [item.strip() for item in answer.split(',') if item.strip()]
    return answer


def dnd_parse_response(answer: str) -> tuple[object, str]:
    match = re.search(r'\\boxed\{\\text\{([^}]*)\}\}', answer) or re.search(r'\\boxed[\{]+([^}]*)[\}]+', answer)
    if match:
        answer = match.group(1)
    else:
        return answer, 'low'
    return dnd_parse_answer(answer), 'high'


def score_dnd(output: str, answer_str: str) -> tuple[float, str, object, object]:
    """Official dnd scoring.

    Returns ``(score, parse_confidence, gold, trimmed_output)``.
    """
    gold = dnd_parse_answer(answer_str)
    trimmed_output, parse_confidence = dnd_parse_response(output)

    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed_output, int):
        score = 0.75 ** abs(gold - trimmed_output)
    elif isinstance(gold, str) and isinstance(trimmed_output, str):
        score = float(gold.strip().lower() == trimmed_output.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        score = len(overlap) / len(gold) if gold else 0.0

    return float(score), parse_confidence, gold, trimmed_output


def _is_dnd(metadata: dict) -> bool:
    """Heuristic: route to dnd scoring only for dnd-style data."""
    markers = (
        str(metadata.get('dataset_name', '')),
        str(metadata.get('sub_category', '')),
        str(metadata.get('oolong_task_group', '')),
        str(metadata.get('oolong_task', '')),
    )
    return any('dnd' in m.lower() for m in markers)


# ---------------------------------------------------------------------------
# framework entry point
# ---------------------------------------------------------------------------
async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* against the oolong ground-truth answer.

    Rule-based; *llm* is unused. Returns the framework result dict.
    """
    if not model_output or not model_output.strip():
        return {
            'score': 0,
            'rationale': 'No model output',
            'requirement_status': [],
            'judge_prompt': None,
        }

    answer_str = task.get('answer', '')
    metadata = task.get('metadata', {}) or {}
    answer_type = metadata.get('oolong_answer_type', '')

    try:
        if _is_dnd(metadata):
            score, confidence, gold, parsed = score_dnd(model_output, answer_str)
            mode = 'dnd'
        else:
            score, confidence, gold, parsed = score_synth(model_output, answer_str, answer_type)
            mode = 'synth'
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            'oolong scoring failed (answer=%r, answer_type=%r): %s',
            answer_str,
            answer_type,
            exc,
        )
        return {
            'score': 0,
            'rationale': f'oolong scoring error: {exc}',
            'requirement_status': [],
            'judge_prompt': None,
        }

    rationale = (
        f'oolong[{mode}] answer_type={answer_type or "n/a"}; '
        f'parsed={parsed!r} vs gold={gold!r}; '
        f'parse_confidence={confidence}; score={score:.4f}'
    )
    return {
        'score': score,
        'rationale': rationale,
        'requirement_status': [],
        'judge_prompt': None,
    }
