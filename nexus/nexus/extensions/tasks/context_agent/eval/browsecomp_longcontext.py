"""BrowseCompLongContext evaluator — LLM-based grading (official method).

Ported from OpenAI's BrowseComp benchmark to the unified evaluator
interface. The candidate response is graded by an LLM judge using the
official BrowseComp grader template, which extracts the final answer and
decides ``correct: yes|no`` against the gold ``answer``.

Reference: https://openai.com/index/browsecomp/

Data schema
-----------
Each task has::

    {
        "context_messages": list,
        "context_question": str,
        "messages": list,
        "answer": str,   # single gold answer, e.g. "Amr Zaki"
        "metadata": {
            "task_id": str,
            "context_category": "BrowseCompLongContext",
            "sub_category": "BrowseCompLongContext",
            "dataset_name": "BrowseCompLongContext",
            "language": "en" | "zh",
            ...
        },
    }
"""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


async def _grade_once(question: str, correct_answer: str, response: str, llm) -> str | None:
    """Run the official grader prompt once; return ``"yes"``/``"no"`` or ``None``."""
    prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    messages = [{'role': 'user', 'content': prompt}]

    raw = await llm.chat(messages)
    if isinstance(raw, dict):
        raw = raw.get('response', '')
    raw = raw or ''

    match = re.search(r'correct:\s*(yes|no)', raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


async def evaluate(model_output: str, task: dict, llm) -> dict:
    """Evaluate *model_output* against the gold ``answer`` via the LLM judge.

    The candidate is correct (score=1) only if the grader judges the
    extracted final answer to match the gold answer.
    """
    if not model_output or not model_output.strip():
        return {
            'score': 0,
            'rationale': 'No model output',
            'requirement_status': [],
            'judge_prompt': None,
        }

    question = task.get('context_question', '')
    answer = task.get('answer', '')
    if isinstance(answer, list):
        answer = '\n'.join(str(a) for a in answer)
    answer = str(answer)

    if not answer.strip():
        return {
            'score': 0,
            'rationale': 'No official answer provided',
            'requirement_status': [],
            'judge_prompt': None,
        }

    verdict: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            verdict = await _grade_once(question, answer, model_output, llm)
            if verdict is not None:
                break
            logger.warning(
                'BrowseCompLongContext grader returned no verdict (attempt %s/%s)',
                attempt + 1,
                MAX_RETRIES,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                'BrowseCompLongContext grader call failed (attempt %s/%s): %s',
                attempt + 1,
                MAX_RETRIES,
                exc,
            )

    if verdict == 'yes':
        return {
            'score': 1,
            'rationale': f'LLM grader: CORRECT (gold: {answer[:80]}…)',
            'requirement_status': [],
            'judge_prompt': None,
        }
    if verdict == 'no':
        return {
            'score': 0,
            'rationale': f'LLM grader: INCORRECT (gold: {answer[:80]}…)',
            'requirement_status': [],
            'judge_prompt': None,
        }
    return {
        'score': 0,
        'rationale': 'LLM grader: verdict unavailable after retries',
        'requirement_status': [],
        'judge_prompt': None,
    }
