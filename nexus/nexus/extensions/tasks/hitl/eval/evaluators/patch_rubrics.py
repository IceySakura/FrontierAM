"""An LLM judge scoring a patch against per-instance rubrics.

Needs no sandbox: the patch text and the rubrics are the whole input, so this runs
wherever the host runs. The rubrics arrive with the task input; a run whose dataset
carries none gets an incomplete result rather than a lookup.

The rubric items and their weights come from the SWE task, and so does the parsing of
the judge's reply. The judging instruction is this package's own: a chain run hands the
worker a requirement statement rather than the reference implementation, so an item
demanding a name that statement never gave is scored on behaviour instead.
"""

import logging

from nexus.core import require_context
from nexus.extensions.tasks.hitl.eval.result import EvaluationResult
from nexus.extensions.tasks.hitl.utils.patch import normalize_patch
from nexus.extensions.tasks.swe.rubrics.shared.schema import (
    RubricSchema,
    RubricScores,
    parse_llm_scores_response,
)


logger = logging.getLogger(__name__)


MAX_EVAL_RETRIES = 3

EVAL_SYSTEM_PROMPT = """\
You are a rubric based evaluator for software-engineering agent's generated patch. Use the provided rubric to evaluate the generated
patch.
Inputs (provided later):
- PR_DESCRIPTION: problem + expected behavior.
- RUBRIC: dictionary of n rubric items for an ideal patch with their ids as keys and descriptions as values.
- PATCH: the model's predicted code patch
Your job:
1. Analyze the rubric and the patch to evaluate the SWE-agent's generated patch.
2. Emit a score for each rubric item. The score should be a binary score of 1 if the patch satisfies the rubric item
and 0 otherwise.

The rubric was written with the reference implementation in hand, so an item may demand a
name the PR_DESCRIPTION never gave. The patch author only ever saw the PR_DESCRIPTION.
Take these two steps for every item, in this order.

Step 1. List the identifiers the item demands: file paths, class, interface, function,
method, field, constant, config key and environment variable names, type annotations,
and any literal string the item quotes as required output. For each one, decide whether
the PR_DESCRIPTION names it. The PR_DESCRIPTION names it when it writes that identifier
or literal, and when it asks for it in words that leave no room for another choice. The
PR_DESCRIPTION does not name it when it only describes the behaviour in prose, and does
not name it when it asks for a different value than the item demands.

Step 2. When the PR_DESCRIPTION names every identifier the item demands, score the item
as it is written. Otherwise drop the identifiers it did not name and score what is left,
which is the behaviour: a patch that reaches the same observable outcome under another
name, in another file, or through an equivalent structure scores 1.

Report both the score and this decision for each item. Return one JSON object.
JSON format:
{
"<rubric_id_1>": {"score": <0 or 1>, "unnamed_identifiers": [<identifiers the PR_DESCRIPTION did not name>], "why": "<one line>"},
"<rubric_id_2>": {"score": <0 or 1>, "unnamed_identifiers": [], "why": "<one line>"},
...
"<rubric_id_n>": {"score": <0 or 1>, "unnamed_identifiers": [], "why": "<one line>"}
}"""

EVAL_USER_PROMPT = """\
PR_DESCRIPTION:
{problem_statement}

PATCH:
{patch_text}

RUBRIC:
{rubrics_yaml}

Please evaluate the PATCH using the rubric and return the scores in JSON format.
SCORE
"""


async def _score_patch_against_rubrics(
    *,
    rubrics_yaml: str,
    patch_text: str,
    problem_statement: str,
    eval_client_name: str,
    max_retries: int = MAX_EVAL_RETRIES,
) -> RubricScores:
    """Ask the judge for per-item scores and weight them by the rubric.

    Retries while the reply parses to no scores at all, which is the shape a judge
    produces when it answers in prose.
    """
    client = require_context().get_openai_client(eval_client_name)

    rubrics = RubricSchema.from_yaml(rubrics_yaml)
    weights = rubrics.get_weights()
    if not weights:
        logger.warning('[hitl/swe/eval] no rubric items in the YAML')
        return RubricScores(score=0.0)

    messages = [
        {'role': 'system', 'content': EVAL_SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': EVAL_USER_PROMPT.format(
                problem_statement=problem_statement,
                patch_text=patch_text,
                rubrics_yaml=rubrics_yaml,
            ),
        },
    ]

    raw_response = ''
    for attempt in range(1, max_retries + 1):
        response = await client.create_completion(messages=messages)
        raw_response = response.choices[0].message.content or ''

        item_scores = parse_llm_scores_response(raw_response)
        if item_scores:
            scores = RubricScores.compute(item_scores, weights)
            scores.raw_response = raw_response
            return scores

        logger.warning(
            '[hitl/swe/eval] attempt %d/%d parsed no scores from: %s',
            attempt,
            max_retries,
            raw_response[:200],
        )

    logger.error('[hitl/swe/eval] all %d attempts parsed no scores', max_retries)
    scores = RubricScores.compute({}, weights)
    scores.raw_response = raw_response
    return scores


async def run_swe_patch_rubrics_eval(
    *,
    patch_text: str,
    problem_statement: str,
    rubrics_yaml: str,
    eval_client_name: str = 'rubric_eval',
) -> EvaluationResult:
    """Score a patch against SWE rubrics."""

    if not rubrics_yaml:
        return EvaluationResult(
            name='patch_rubrics',
            completed=False,
            score=0.0,
            error='No rubrics available.',
        )

    try:
        normalized_patch = normalize_patch(
            patch_text or '',
            max_file_chars=20_000,
            max_total_chars=300_000,
        )
        scores = await _score_patch_against_rubrics(
            rubrics_yaml=rubrics_yaml,
            patch_text=normalized_patch,
            problem_statement=problem_statement,
            eval_client_name=eval_client_name,
        )
        return EvaluationResult(
            name='patch_rubrics',
            completed=True,
            score=float(scores.score),
            details={
                'item_scores': scores.model_dump() if hasattr(scores, 'model_dump') else {},
            },
        )
    except Exception as exc:
        logger.exception('[hitl/swe/eval] patch_rubrics evaluation failed')
        return EvaluationResult(
            name='patch_rubrics',
            completed=False,
            score=0.0,
            error=f'{type(exc).__name__}: {exc}',
        )


__all__ = ['run_swe_patch_rubrics_eval']
