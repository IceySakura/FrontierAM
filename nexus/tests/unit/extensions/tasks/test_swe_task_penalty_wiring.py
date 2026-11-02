"""Config-matrix tests over run_swe_task's detection / monitoring / penalty wiring.

apply_swe_penalties used to sit inside the ``enable_output_monitoring`` branch, so a run
that set ``enable_anti_hack`` and ``hack_reward_override`` with monitoring off ran the
detector, mirrored its flags, and then silently kept the full reward. These tests drive
``run_swe_task`` rather than the penalty helper, because only the task can show which
branch the penalty is wired into.

The task is driven through its ``patch_text`` shortcut, which skips runtime provisioning
and the agent, leaving the grade-then-penalise tail these cases care about.
"""

from __future__ import annotations

import asyncio

import pytest

from nexus.core.context import ExecutionContext, _current_context
from nexus.extensions.tasks.swe import task as swe_task
from nexus.extensions.tasks.swe.task import SwebenchTaskInput, run_swe_task
from nexus.runtime.providers.base import RuntimeProvider


CLEAN_FLAGS = {
    'reward_hack_mask': False,
    'reward_hack_rules': [],
    'reward_hack_attempted': False,
    'reward_hack_effective': False,
}

HACK_FLAGS = {
    'reward_hack_mask': True,
    'reward_hack_rules': ['online_answer_fetch'],
    'reward_hack_attempted': True,
    'reward_hack_effective': True,
}

CODESWITCH_FINDING = {'codeswitch': {'is_codeswitch': True}}


class _StubProvider(RuntimeProvider):
    async def acquire(self, name: str, **params):
        raise AssertionError('the test pre-populates the runtime cache; acquire should not run')

    async def _release_impl(self, name: str) -> None:
        return None


class _StubRuntime:
    """Stands in for an acquired runtime. Deliberately has no ``config`` attribute."""


class _StubHandler:
    def get_repo_language(self, instance):
        return 'py'

    async def eval_runtime_init(self, instance, runtime):
        return None


class _StubProcessor:
    def snapshot_result(self):
        return {'messages': []}


def _run_task(
    monkeypatch: pytest.MonkeyPatch,
    *,
    detector_flags: dict | None = None,
    monitor_findings: dict | None = None,
    graded_score: float = 1.0,
    **task_kwargs,
):
    """Drive run_swe_task with the runtime, agent, detector and monitors stubbed out."""
    monkeypatch.setattr(swe_task, 'get_handler', lambda _dataset_type: _StubHandler())
    monkeypatch.setattr(swe_task, 'get_processor', lambda _cls: _StubProcessor())
    monkeypatch.setattr(swe_task, '_chat_messages', lambda _snapshot: [{'role': 'user', 'content': 'hi'}])
    monkeypatch.setattr(
        swe_task,
        'detect_reward_hacking',
        lambda *args, **kwargs: dict(detector_flags if detector_flags is not None else CLEAN_FLAGS),
    )
    monkeypatch.setattr(swe_task, 'run_swe_monitors', lambda _messages: dict(monitor_findings or {}))

    async def _grade(*args, **kwargs):
        return {
            'score': graded_score,
            'resolved': graded_score >= 1.0,
            'completed': True,
            'report_json': '{}',
            'test_output': '',
            'apply_log': '',
        }

    monkeypatch.setattr(swe_task, 'evaluate_and_grade_in_runtime', _grade)

    async def _unused_agent(*args, **kwargs):
        raise AssertionError('patch_text short-circuits inference; the agent should not run')

    task_input = SwebenchTaskInput(
        instance_id='django__django-1',
        instance={'instance_id': 'django__django-1', 'repo': 'django/django', 'base_commit': 'a' * 40},
        patch_text='diff --git a/a.py b/a.py',
        run_agent=_unused_agent,
        **task_kwargs,
    )

    ctx = ExecutionContext(runtime_provider=_StubProvider(), task_name='swe_task')
    ctx._runtimes['eval'] = _StubRuntime()

    token = _current_context.set(ctx)
    try:
        return asyncio.run(run_swe_task(task_input))
    finally:
        _current_context.reset(token)


# ============== The regression this file exists for ==============


def test_a_penalty_applies_with_output_monitoring_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reported bug: monitoring defaults to off, and the override used to be gated on it.
    output = _run_task(
        monkeypatch,
        detector_flags=HACK_FLAGS,
        enable_reward_hacking_detection=True,
        enable_anti_hack=True,
        hack_reward_override=-1.0,
    )

    assert output.score == -1.0
    assert output.termination_reason == 'CODE_REWARD_HACKING'
    assert output.extra_fields['reward_extra_info']['extra_reward_metrics']['hack_penalized'] == 1.0


def test_the_same_penalty_applies_with_output_monitoring_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monitoring must not change the anti-hack outcome in either position.
    output = _run_task(
        monkeypatch,
        detector_flags=HACK_FLAGS,
        enable_reward_hacking_detection=True,
        enable_output_monitoring=True,
        enable_anti_hack=True,
        hack_reward_override=-1.0,
    )

    assert output.score == -1.0
    assert output.termination_reason == 'CODE_REWARD_HACKING'


# ============== Anti-hack dispositions ==============


def test_a_hack_without_an_override_is_relabelled_for_the_maskout(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _run_task(
        monkeypatch,
        detector_flags=HACK_FLAGS,
        enable_reward_hacking_detection=True,
        enable_anti_hack=True,
    )

    assert output.score == 1.0  # masking, not penalising: hytuner drops the sample instead
    assert output.termination_reason == 'CODE_REWARD_HACKING'


def test_an_override_without_anti_hack_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    # hack_reward_override is documented as requiring enable_anti_hack.
    output = _run_task(
        monkeypatch,
        detector_flags=HACK_FLAGS,
        enable_reward_hacking_detection=True,
        hack_reward_override=-1.0,
    )

    assert output.score == 1.0
    assert output.termination_reason == 'ENV_DONE'


def test_validation_reports_the_earned_score_rather_than_the_override(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _run_task(
        monkeypatch,
        detector_flags=HACK_FLAGS,
        enable_reward_hacking_detection=True,
        enable_anti_hack=True,
        hack_reward_override=-1.0,
        is_validation=True,
    )

    assert output.score == 1.0
    assert output.termination_reason == 'CODE_REWARD_HACKING'


def test_an_honest_run_is_untouched_by_a_configured_override(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _run_task(
        monkeypatch,
        detector_flags=CLEAN_FLAGS,
        enable_reward_hacking_detection=True,
        enable_anti_hack=True,
        hack_reward_override=-1.0,
    )

    assert output.score == 1.0
    assert output.termination_reason == 'ENV_DONE'
    assert output.extra_fields['reward_hack_mask'] is False


# ============== Monitoring still gates its own findings ==============


def test_a_quality_penalty_needs_monitoring_to_produce_its_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    # Moving the penalty call out of the monitoring branch must not let the quality
    # penalties act on runs that never ran the monitors: their keys stay absent.
    output = _run_task(
        monkeypatch,
        monitor_findings=CODESWITCH_FINDING,
        codeswitch_penalty=0.3,
        codeswitch_penalty_reward_threshold=1.0,
    )

    assert output.score == 1.0
    assert 'codeswitch' not in output.extra_fields


def test_a_quality_penalty_applies_once_monitoring_reports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _run_task(
        monkeypatch,
        monitor_findings=CODESWITCH_FINDING,
        enable_output_monitoring=True,
        codeswitch_penalty=0.3,
        codeswitch_penalty_reward_threshold=1.0,
    )

    assert output.score == pytest.approx(0.7)
    assert output.extra_fields['codeswitch']['penalized'] is True


def test_a_hack_override_outranks_a_quality_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _run_task(
        monkeypatch,
        detector_flags=HACK_FLAGS,
        monitor_findings=CODESWITCH_FINDING,
        enable_reward_hacking_detection=True,
        enable_output_monitoring=True,
        enable_anti_hack=True,
        hack_reward_override=-1.0,
        codeswitch_penalty=0.3,
    )

    assert output.score == -1.0


# ============== Everything off ==============


def test_the_default_config_leaves_the_score_and_extra_fields_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    # apply_swe_penalties now runs unconditionally, so the all-defaults run has to stay a no-op.
    output = _run_task(monkeypatch, graded_score=0.5)

    assert output.score == 0.5
    assert output.termination_reason == 'ENV_DONE'
    assert output.extra_fields == {}
