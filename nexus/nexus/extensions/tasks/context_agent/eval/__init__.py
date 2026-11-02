"""Scoring — one evaluator per benchmark, routed by ``metadata.dataset_name``.

The harness runs rollouts and scores them in separate steps, so this package is
never imported by a rollout. Grade a finished run with::

    python contextagent/judge.py --predictions outputs/.../predictions.jsonl \\
        --input data/LongBenchV2.files.contextagent.jsonl

**The routing key is ``metadata.dataset_name``, not the filename**, and the two
disagree often enough to be worth stating: ``CLB_200`` carries ``"CLB"``,
``doc_longer_406`` carries ``"DocLonger"``, ``table_longer_66`` carries
``"TableMixLonger"``, and both ``CtxAgent1000`` and ``surge_eval_200_with_docs``
carry ``"surge_docs"`` — the last two are genuinely indistinguishable here, which
is harmless because they grade identically.

Three grading styles are in play, and mixing them in one average is misleading:

*Rule-based* (no judge model, deterministic, free)
    ``LongBenchV2`` matches a regex for the chosen letter — strictly the official
    one, so "The answer is D" scores 0 where "The correct answer is (D)" scores 1.
    ``MRCR_800`` is a prefix gate then a difflib ratio, so it yields a genuine
    float. ``oolong`` parses the answer out and gives partial credit for numeric
    distance. ``GraphWalks`` strictly extracts ``Final Answer: [...]`` from the
    last line and computes set F1 over the predicted and gold node sets.

*Reference-graded* (a judge model compares against a gold answer)
    ``AALCR`` is correct if it matches **any** of N official answers;
    ``LongBenchPro`` joins its answer list and demands **all** components;
    ``BrowseCompLongContext`` uses the official BrowseComp grader prompt;
    ``TableMixV2`` and ``DocLonger`` use the generic answer judge. The four
    ``ComplexLong*`` sets share one judge whose comparison rule is selected from
    ``metadata.answer_form`` — scalar, set, ordered sequence or integer — since
    that choice alone decides whether a given answer is right.

*Rubric-graded* (a judge model checks a criteria list; ``answer`` is ignored)
    ``CLB``, ``CLBLife``, ``outside_L3_high``, ``Pluto20``, ``SurgeOODV21``,
    ``surge_docs``, ``GDPpdf``, and ``TableMixLonger`` (which brings its own
    checklist template).

*Deliverable-graded* (a judge agent checks criteria against actual files)
    ``GDPval``. The Judge receives a private Runtime-side copy of the rollout's
    ``workspace/`` and can inspect spreadsheets, presentations, documents, PDFs,
    and text files with the path-native tools.

AALCR and LongBenchPro retain their benchmark-specific prompts and answer-list
semantics while using the Judge client configured for the run.

Text-only rubric evaluators still see only ``model_output``. GDPval is the
exception: it uses :mod:`.deliverable_judge`.
"""

from __future__ import annotations

import logging

from collections.abc import Awaitable, Callable

from . import (
    aalcr,
    answer_judge,
    browsecomp_longcontext,
    complexlong,
    deliverable_judge,
    graphwalks,
    longbenchpro,
    longbenchv2,
    mrcr,
    oolong,
    rm_system,
    rubrics_judge,
    tablemixlonger,
)


logger = logging.getLogger(__name__)

EvalFn = Callable[..., Awaitable[dict]]


async def _answer(model_output: str, task: dict, llm) -> dict:
    """The generic answer judge, tagged with the dataset name for the log."""
    name = (task.get('metadata') or {}).get('dataset_name') or 'answer_judge'
    return await answer_judge.evaluate(model_output, task, llm, dataset_name=name)


# dataset_name -> evaluator. Several names share one entry; that is not an
# oversight, it is what the upstream one-line re-export modules encoded.
EVALUATORS: dict[str, EvalFn] = {
    # rule-based
    'GraphWalks': graphwalks.evaluate,
    'LongBenchV2': longbenchv2.evaluate,
    'MRCR_400': mrcr.evaluate,
    'MRCR_800': mrcr.evaluate,
    'oolong': oolong.evaluate,
    # reference-graded
    'AALCR': aalcr.evaluate,
    'LongBenchPro': longbenchpro.evaluate,
    'BrowseCompLongContext': browsecomp_longcontext.evaluate,
    'TableMixV2': _answer,
    'DocLonger': _answer,
    'ComplexLongTable': complexlong.evaluate,
    'ComplexLongGraph': complexlong.evaluate,
    'ComplexLongTree': complexlong.evaluate,
    'ComplexLongDoc': complexlong.evaluate,
    # rubric-graded
    'CLB': rubrics_judge.evaluate,
    'CLBLife': rubrics_judge.evaluate,
    'outside_L3_high': rubrics_judge.evaluate,
    'Pluto20': rubrics_judge.evaluate,
    # Ships a non-empty `answer` AND rubrics; the answer is deliberately
    # ignored, matching upstream.
    'SurgeOODV21': rubrics_judge.evaluate,
    'surge_docs': rubrics_judge.evaluate,
    'GDPpdf': rubrics_judge.evaluate_accuracy,
    'TableMixLonger': tablemixlonger.evaluate,
    # deliverable-graded
    'GDPval': deliverable_judge.evaluate,
}

DELIVERABLE_AWARE: frozenset[EvalFn] = frozenset({deliverable_judge.evaluate})

# How each is graded, for the report and for refusing to average across styles.
STYLES: dict[str, str] = {
    'GraphWalks': 'rule',
    'LongBenchV2': 'rule',
    'MRCR_400': 'rule',
    'MRCR_800': 'rule',
    'oolong': 'rule',
    'AALCR': 'reference',
    'LongBenchPro': 'reference',
    'BrowseCompLongContext': 'reference',
    'TableMixV2': 'reference',
    'DocLonger': 'reference',
    'ComplexLongTable': 'reference',
    'ComplexLongGraph': 'reference',
    'ComplexLongTree': 'reference',
    'ComplexLongDoc': 'reference',
    'CLB': 'rubric',
    'CLBLife': 'rubric',
    'outside_L3_high': 'rubric',
    'Pluto20': 'rubric',
    'SurgeOODV21': 'rubric',
    'surge_docs': 'rubric',
    'GDPval': 'rubric',
    'GDPpdf': 'rubric',
    'TableMixLonger': 'rubric',
}

# Graded without a judge model, so they need no --judge-model and cost nothing.
RULE_BASED: frozenset[str] = frozenset(name for name, style in STYLES.items() if style == 'rule')


def dataset_name(task: dict) -> str:
    return (task.get('metadata') or {}).get('dataset_name') or ''


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


async def evaluate(
    model_output: str,
    task: dict,
    llm,
    backend: str = 'auto',
    *,
    runtime_context=None,
    working_dir: str | None = None,
    staged_files: list[dict] | None = None,
    artifact_max_files: int = 10_000,
    artifact_max_bytes: int = 1024 * 1024 * 1024,
) -> dict:
    """Score *model_output* for *task*.

    ``backend``: ``auto`` routes on ``metadata.dataset_name``; ``rubrics``,
    ``rubric_accuracy``, ``answer``, ``deliverable`` and ``rm_system`` force
    one judge regardless.

    Returns ``{"score", "rationale", "requirement_status", ...}``. Rubric
    evaluators also return ``criteria_passed`` and ``criteria_total``.
    ``score`` is ``0``/``1`` for most, a float for MRCR and oolong.
    """
    if backend == 'rm_system':
        return await rm_system.evaluate(model_output, task, llm)
    if backend == 'rubrics':
        return await rubrics_judge.evaluate(model_output, task, llm)
    if backend == 'rubric_accuracy':
        return await rubrics_judge.evaluate_accuracy(model_output, task, llm)
    if backend == 'answer':
        return await _answer(model_output, task, llm)
    if backend == 'deliverable':
        return await deliverable_judge.evaluate(
            model_output,
            task,
            llm,
            runtime_context=runtime_context,
            working_dir=working_dir,
            staged_files=staged_files,
            max_files=artifact_max_files,
            max_bytes=artifact_max_bytes,
        )

    name = dataset_name(task)
    evaluator = EVALUATORS.get(name)
    if evaluator is not None:
        if evaluator in DELIVERABLE_AWARE:
            return await evaluator(
                model_output,
                task,
                llm,
                runtime_context=runtime_context,
                working_dir=working_dir,
                staged_files=staged_files,
                max_files=artifact_max_files,
                max_bytes=artifact_max_bytes,
            )
        return await evaluator(model_output, task, llm)

    # Unregistered corpus: infer from which reference the row carries. Rubrics
    # win when both are present, matching the datasets that ship both.
    has_rubrics = _has_value(task.get('rubrics'))
    has_answer = _has_value(task.get('answer'))
    if has_rubrics:
        if has_answer:
            logger.warning(
                'no evaluator for dataset_name=%r and the row has both rubrics and an answer; grading on the rubrics',
                name,
            )
        return await rubrics_judge.evaluate(model_output, task, llm)
    if has_answer:
        return await _answer(model_output, task, llm)
    return {
        'score': None,
        'rationale': (
            f'nothing to grade against: dataset_name={name!r} has no evaluator, '
            'and the row carries neither rubrics nor an answer'
        ),
        'requirement_status': [],
    }


__all__ = [
    'EVALUATORS',
    'DELIVERABLE_AWARE',
    'RULE_BASED',
    'STYLES',
    'dataset_name',
    'evaluate',
    'rm_system',
]
