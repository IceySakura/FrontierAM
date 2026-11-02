"""Evaluating a HITL run: one module per evaluator, plus how they combine.

``evaluators/`` holds the evaluators, each taking what it alone needs -- a prepared
sandbox for the official unit tests, a model client for the rubrics judge -- and
returning an :class:`EvaluationResult`. :mod:`aggregate` turns a mapping of their
results into the run's score and is independent of which ones ran, so adding an
evaluator does not touch it. :mod:`sandbox` prepares the second runtime the official
tests need and releases it afterwards.

A recipe decides which evaluators to run and assembles their inputs.
"""

from nexus.extensions.tasks.hitl.eval.aggregate import (
    SWEScorePolicy,
    SWEUnitScorePolicy,
    aggregate_swe_evaluations,
)
from nexus.extensions.tasks.hitl.eval.evaluators.patch_rubrics import run_swe_patch_rubrics_eval
from nexus.extensions.tasks.hitl.eval.evaluators.reward_hacking import detect_worker_reward_hacking
from nexus.extensions.tasks.hitl.eval.evaluators.unit_test import run_swe_unit_test_eval
from nexus.extensions.tasks.hitl.eval.result import EvaluationResult
from nexus.extensions.tasks.hitl.eval.sandbox import eval_script_of, run_unit_eval_in_sandbox


__all__ = [
    'EvaluationResult',
    'SWEScorePolicy',
    'SWEUnitScorePolicy',
    'aggregate_swe_evaluations',
    'detect_worker_reward_hacking',
    'eval_script_of',
    'run_swe_patch_rubrics_eval',
    'run_swe_unit_test_eval',
    'run_unit_eval_in_sandbox',
]
