"""GraphWalks evaluator — strict final-answer extraction and set F1.

This is aligned with the standalone GraphWalks harness so scores remain
comparable.  The final non-empty response line must contain the complete
``Final Answer: [...]`` marker; a missing closing bracket is a format error.
Nodes are comma-separated, case-sensitive strings.  Duplicate predictions do
not change the score because precision and recall are computed over sets.
"""

from __future__ import annotations

import os
import re


_FINAL_ANSWER_RE = re.compile(r'Final Answer:\s*\[(.*)\]')


def _lenient_default() -> bool:
    return os.environ.get('GRAPHWALKS_LENIENT_EXTRACT', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _parse_node_list(inner: str) -> list[str]:
    """Split a ``Final Answer: [...]`` payload on commas, stripping quotes."""
    items: list[str] = []
    for raw in inner.split(','):
        token = raw.strip().strip('\'"')
        if token:
            items.append(token)
    return items


def extract_final_answer(
    response: str,
    *,
    strict_last_line: bool = True,
) -> tuple[list[str], bool]:
    """Return ``(nodes, format_error)`` from a model response."""
    if not (response or '').strip():
        return [], True
    lines = [line for line in response.splitlines() if line.strip()]
    last_line = lines[-1] if lines else ''
    match = _FINAL_ANSWER_RE.search(last_line)
    if match:
        return _parse_node_list(match.group(1)), False
    if strict_last_line:
        return [], True
    matches = list(_FINAL_ANSWER_RE.finditer(response))
    if not matches:
        return [], True
    return _parse_node_list(matches[-1].group(1)), False


def set_f1(prediction: list[str], gold: list[str]) -> dict[str, int | float]:
    """Compute set F1. Two empty sets score 1; exactly one empty set scores 0."""
    predicted = set(prediction)
    expected = set(gold)
    overlap = len(predicted & expected)
    n_gold = len(expected)
    n_pred = len(predicted)
    recall = overlap / n_gold if n_gold else 0.0
    precision = overlap / n_pred if n_pred else 0.0
    if recall + precision:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 1.0 if not expected and not predicted else 0.0
    return {
        'n_overlap': overlap,
        'n_gold': n_gold,
        'n_pred': n_pred,
        'precision': round(precision, 6),
        'recall': round(recall, 6),
        'f1': round(f1, 6),
    }


def _gold_nodes(task: dict) -> list[str]:
    nodes = task.get('answer_nodes')
    if isinstance(nodes, list) and nodes:
        return [str(node).strip() for node in nodes if str(node).strip()]
    answer = task.get('answer') or ''
    return [part.strip() for part in str(answer).split(',') if part.strip()]


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Score *model_output* against the gold node set without calling an LLM."""
    gold = _gold_nodes(task)
    prediction, format_error = extract_final_answer(
        model_output or '',
        strict_last_line=not _lenient_default(),
    )
    metrics = set_f1(prediction, gold)
    score = metrics['f1']

    if format_error and not gold:
        detail = (
            "no parseable 'Final Answer: [...]' on the last line, and the gold "
            'set is empty -> counted correct (empty == empty)'
        )
    elif format_error:
        detail = (
            "no parseable 'Final Answer: [...]' on the last line (truncated or missing closing bracket) -> format error"
        )
    else:
        detail = (
            f'predicted {metrics["n_pred"]} node(s), '
            f'{metrics["n_overlap"]} of {metrics["n_gold"]} gold node(s) matched '
            f'(P={metrics["precision"]:.3f} R={metrics["recall"]:.3f})'
        )

    return {
        'score': score,
        'rationale': f'GraphWalks set F1 = {score:.4f} ({detail}).',
        'requirement_status': [],
        'judge_prompt': None,
    }
