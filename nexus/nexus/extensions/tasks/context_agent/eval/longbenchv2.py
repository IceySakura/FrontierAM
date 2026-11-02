"""LongBenchV2 evaluator — multiple-choice exact match.

Uses the official ``extract_answer`` logic from
`THUDM/LongBench <https://github.com/THUDM/LongBench/blob/main/pred.py>`_.

Data schema
-----------
Each task has::

    {
        "context_messages": list,
        "context_question": str,
        "messages": list,
        "answer": str,   # single letter, e.g. "D"
        "metadata": {
            "task_id": str,
            "context_category": "LongBenchV2",
            "sub_category": str,
            "difficulty": str,   # "easy" | "medium" | "hard"
            "length": str,       # "short" | "medium" | "long"
        },
    }
"""

from __future__ import annotations

import re


def extract_answer(response: str) -> str | None:
    """Extract the chosen letter (A-D) from a model response.

    Ported from the official LongBenchV2 evaluation code
    (``THUDM/LongBench pred.py``).
    """
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', response)
    if match:
        return match.group(1)
    return None


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* against the ground-truth choice letter."""
    gold = task.get('answer', '')
    pred = extract_answer(model_output)
    correct = pred is not None and pred.upper() == gold.upper()

    return {
        'score': 1 if correct else 0,
        'rationale': (
            f"Extracted '{pred}' from model output; gold answer is '{gold}'. {'Match' if correct else 'Mismatch'}."
        ),
        'requirement_status': [],
        'judge_prompt': None,
    }
