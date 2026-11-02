"""MRCR (Multi-Round Coreference Resolution) evaluator.

Covers both MRCR_400 and MRCR_800 with the same logic.
Uses the official scoring: the response must start with the random prefix,
then similarity is measured via ``SequenceMatcher``.

Data schema
-----------
Each task has::

    {
        "context_messages": list,
        "context_question": str,
        "messages": list,
        "answer": str,   # long text starting with a random prefix
        "metadata": {
            "task_id": str,
            "dataset_name": "MRCR_400" | "MRCR_800",
            "random_string_to_prepend": str,
        },
    }
"""

from __future__ import annotations

from difflib import SequenceMatcher


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Official MRCR scoring: prefix check + sequence similarity."""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* against the ground-truth text."""
    answer = task.get('answer', '')
    rsp = task.get('metadata', {}).get('random_string_to_prepend', '')
    score = grade(model_output, answer, rsp)

    return {
        'score': score,
        'rationale': (
            f'Prefix {"present" if model_output.startswith(rsp) else "missing"}; sequence similarity = {score:.4f}.'
        ),
        'requirement_status': [],
        'judge_prompt': None,
    }
