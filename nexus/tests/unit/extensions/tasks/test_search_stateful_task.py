"""``search_stateful_task`` must not pay a reward call for a rollout that was cut short.

The task reads whatever the agent produced and sends it straight to the configured
evaluator. When the rollout stopped because it ran out of room, or out of time, that
"answer" is only the fragment the model had emitted when the limit hit, so the score the
reward system returns describes nothing about the policy — yet it still centres the GRPO
group its siblings sit in. ``search`` and ``mm_agent`` already short-circuit these; this
task did not.

The cutoff arrives as ``exit_status``, in either of two vocabularies: ``main_agent`` hands
back its own lowercase value, other agents the uppercase ``TerminationReason``.
"""

from __future__ import annotations

import asyncio

from typing import Any

import pytest

from nexus.core.agent import AgentRunOutput, WithMainTrajectory
from nexus.core.context import ExecutionContext, _current_context
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.codex.agent import CodexOutput
from nexus.extensions.agents.main.agent import MainExitStatus
from nexus.extensions.tasks.search_stateful import task as task_module
from nexus.extensions.tasks.search_stateful.task import (
    SearchStatefulTaskInput,
    _extract_prediction,
    search_stateful_task,
)
from nexus.runtime.providers.base import RuntimeProvider


QUESTION = 'How many moons does Jupiter have?'
REFERENCE = '95'
PARTIAL_ANSWER = 'Jupiter has at least ninety-five confirmed moons, of which the four'


class _StubProvider(RuntimeProvider):
    async def acquire(self, name: str, **params):
        raise AssertionError('the test pre-populates the runtime cache; acquire should not run')

    async def _release_impl(self, name: str) -> None:
        return None


class _StubRuntime:
    pass


class FakeAgentOutput(AgentRunOutput, WithMainTrajectory):
    final_answer: str | None = None
    # ``Any`` on purpose: a declared ``str`` would let pydantic coerce an Enum member into
    # its value, hiding whether the task unwraps enums itself.
    exit_status: Any = None


def _run_task(
    monkeypatch: pytest.MonkeyPatch,
    agent_output: Any,
    **task_kwargs: Any,
) -> tuple[Any, dict[str, object], list[str]]:
    """Drive the task with one canned agent result.

    Returns the task output, the span attributes it wrote, and the predictions handed to
    the evaluator (empty when evaluation was skipped).
    """
    recorded_attrs: dict[str, object] = {}
    judged: list[str] = []

    async def fake_evaluate_output(*, prediction, reference, question, evaluator, **_kwargs):
        judged.append(prediction)
        return 1.0, None, True

    monkeypatch.setattr(task_module, 'evaluate_output', fake_evaluate_output)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(
        task_module,
        'update_span',
        lambda span_id, attributes: recorded_attrs.update(attributes),
    )

    async def fake_run_agent(*, user_prompt, **_kwargs):
        assert user_prompt == QUESTION
        return agent_output

    task_input = SearchStatefulTaskInput(
        question=QUESTION,
        answer=REFERENCE,
        evaluator_type='general',
        run_agent=fake_run_agent,
        **task_kwargs,
    )

    ctx = ExecutionContext(runtime_provider=_StubProvider(), task_name='search_stateful')
    ctx._runtimes['main'] = _StubRuntime()

    token = _current_context.set(ctx)
    try:
        output = asyncio.run(search_stateful_task(task_input))
    finally:
        _current_context.reset(token)
    return output, recorded_attrs, judged


@pytest.mark.parametrize(
    ('exit_status', 'expected_reason'),
    [
        # main_agent's own vocabulary: one sampling call hit max_tokens, and the prompt
        # outgrowing the model window mid-run. This is what it actually returns — a str.
        (MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value, 'OUTPUT_LENGTH_EXCEEDED'),
        (MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value, 'TRUNCATION'),
        # Out of time rather than out of room: a sampling call and an observation that
        # never came back. The fragment is just as unjudgeable.
        (MainExitStatus.INFERENCE_TIMEOUT.value, 'TASK_TIMEOUT'),
        (MainExitStatus.OBSERVATION_TIMEOUT.value, 'TASK_TIMEOUT'),
        # Agents that report the canonical reason directly.
        (TerminationReason.OUTPUT_LENGTH_EXCEEDED, 'OUTPUT_LENGTH_EXCEEDED'),
        (TerminationReason.TRUNCATION, 'TRUNCATION'),
        (TerminationReason.TASK_TIMEOUT, 'TASK_TIMEOUT'),
        # Unwrapped enum members, in case an agent stops calling ``.value``.
        (MainExitStatus.OUTPUT_LENGTH_EXCEEDED, 'OUTPUT_LENGTH_EXCEEDED'),
        (MainExitStatus.CONTEXT_LENGTH_EXCEEDED, 'TRUNCATION'),
        (MainExitStatus.INFERENCE_TIMEOUT, 'TASK_TIMEOUT'),
    ],
)
def test_a_cut_off_rollout_is_never_sent_to_the_reward_system(
    monkeypatch: pytest.MonkeyPatch,
    exit_status: Any,
    expected_reason: str,
) -> None:
    output, attrs, judged = _run_task(
        monkeypatch,
        FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': PARTIAL_ANSWER}],
            final_answer=PARTIAL_ANSWER,
            exit_status=exit_status,
        ),
    )

    assert judged == []
    assert output.score == 0.0
    assert output.judge_details is None
    assert attrs['hytuner.termination_reason'] == expected_reason
    # The fragment is still reported so the run stays debuggable.
    assert output.prediction == PARTIAL_ANSWER


def test_a_cutoff_inside_the_reasoning_block_reports_no_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    # max_tokens hit before the model left its think block: the turn carries no content,
    # so ``final_answer`` is None. Stringifying it used to make the prediction 'None'.
    output, _attrs, judged = _run_task(
        monkeypatch,
        FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'reasoning_content': PARTIAL_ANSWER}],
            final_answer=None,
            exit_status=MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value,
        ),
    )

    assert judged == []
    assert output.score == 0.0
    assert output.prediction == ''


def test_a_missing_final_answer_is_not_stringified_into_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same None, but the agent finished cleanly, so the reward call still happens. It must
    # not be asked to grade the word 'None'.
    _output, _attrs, judged = _run_task(
        monkeypatch,
        FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': ''}],
            final_answer=None,
            exit_status=MainExitStatus.FINISHED.value,
        ),
    )

    assert judged == ['']


@pytest.mark.parametrize(
    'exit_status',
    [
        MainExitStatus.FINISHED.value,
        MainExitStatus.MAX_ITERATIONS.value,
        MainExitStatus.TOOL_ERROR.value,
        TerminationReason.SUCCESS,
        None,
        'SOME_FUTURE_REASON',
    ],
)
def test_every_other_outcome_still_reaches_the_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    exit_status: Any,
) -> None:
    # Only the two length cutoffs skip scoring. An unrecognized status must not match one.
    output, attrs, judged = _run_task(
        monkeypatch,
        FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': PARTIAL_ANSWER}],
            final_answer=PARTIAL_ANSWER,
            exit_status=exit_status,
        ),
    )

    assert judged == [PARTIAL_ANSWER]
    assert output.score == 1.0
    assert 'hytuner.termination_reason' not in attrs


def test_oracle_mode_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    # No agent runs, so there is no exit status to inspect.
    output, _attrs, judged = _run_task(
        monkeypatch,
        FakeAgentOutput(main_trajectory=[], exit_status=None),
        run_oracle=True,
    )

    assert judged == [REFERENCE]
    assert output.prediction == REFERENCE


def _codex_output(**overrides: object) -> CodexOutput:
    values = {
        'success': True,
        'output': 'answer',
        'stderr': '',
        'exit_code': 0,
        'execution_time': 1.0,
        'exit_status': 'completed',
    }
    values.update(overrides)
    return CodexOutput(**values)


def test_extract_prediction_reads_successful_codex_output() -> None:
    assert _extract_prediction(_codex_output()) == 'answer'


def test_extract_prediction_rejects_failed_codex_output() -> None:
    with pytest.raises(RuntimeError, match='Agent execution failed: exit_code=1, stderr=boom'):
        _extract_prediction(_codex_output(success=False, output='', stderr='boom', exit_code=1, exit_status='error'))


@pytest.mark.asyncio
async def test_search_stateful_task_accepts_successful_codex_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Context:
        async def get_runtime(self, name: str):
            return object()

        def get_runtime_id(self, name: str):
            return None

    async def run_agent(**kwargs: object) -> CodexOutput:
        return _codex_output(output='final answer')

    monkeypatch.setattr('nexus.extensions.tasks.search_stateful.task.require_context', lambda: _Context())

    result = await search_stateful_task(
        SearchStatefulTaskInput(
            question='question',
            answer='answer',
            run_agent=run_agent,
            evaluator_type='none',
            evaluator_none_score=0.5,
        )
    )

    assert result.prediction == 'final answer'
    assert result.score == 0.5


@pytest.mark.asyncio
async def test_truncated_codex_output_uses_default_score_without_reward_system(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Context:
        async def get_runtime(self, name: str):
            return object()

        def get_runtime_id(self, name: str):
            return None

    async def run_agent(**kwargs: object) -> CodexOutput:
        return _codex_output(
            success=False,
            output='unfinished reasoning',
            exit_code=1,
            termination_reason='TRUNCATION',
            exit_status='truncation',
        )

    async def fail_if_evaluated(**kwargs: object):
        raise AssertionError('truncated output must not be sent to the reward system')

    monkeypatch.setattr('nexus.extensions.tasks.search_stateful.task.require_context', lambda: _Context())
    monkeypatch.setattr('nexus.extensions.tasks.search_stateful.task.evaluate_output', fail_if_evaluated)
    monkeypatch.setattr('nexus.extensions.tasks.search_stateful.task.get_current_span_id', lambda: None)

    result = await search_stateful_task(
        SearchStatefulTaskInput(
            question='question',
            answer='answer',
            run_agent=run_agent,
            evaluator_type='reward_system',
            reward_system_router_id='hle-default-v1',
            reward_system_extra={'default_truncate_score': 0.25},
        )
    )

    assert result.score == 0.25
    assert result.prediction == 'unfinished reasoning'
    assert result.judge_details == {
        'skipped': 'agent_truncation',
        'termination_reason': 'TRUNCATION',
        'score': 0.25,
    }
