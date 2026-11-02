import pytest

from nexus.core.agent import AgentRunOutput, WithMainTrajectory
from nexus.extensions.agents.base.evaluation import EvaluationDetails
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.main.agent import MainExitStatus
from nexus.extensions.tasks.hy3_search import task as hy3_search_task_module
from nexus.extensions.tasks.search import task as search_task_module
from nexus.extensions.tasks.search.evaluator import EvalMode


QUESTION = 'question'
ANSWER = 'answer'
WHAT_HAPPENED_QUESTION = 'What happened?'
REFERENCE_ANSWER = 'Reference Answer'
PARTIAL_ANSWER = 'partial'
RECOVERY_QUESTION = 'Need recovery'
RECOVERED_ANSWER = 'Recovered answer from last turn'
JUDGED_QUESTION = 'Need judged answer'
AGENT_ANSWER = 'Agent answer'
WIDE_SEARCH_EVALUATION = {
    'required': ['company'],
    'unique_columns': ['company'],
    'eval_pipeline': {'company': {'metric': ['exact_match']}},
}


class FakeAgentOutput(AgentRunOutput, WithMainTrajectory):
    final_answer: str | None = None
    exit_status: TerminationReason | str | None = None


@pytest.mark.parametrize(
    'requested_mode',
    [EvalMode.LLM_JUDGE, EvalMode.EXACT_MATCH, EvalMode.REWARD_SERVICE],
)
def test_resolve_eval_mode_keeps_requested_mode_for_normal_samples(requested_mode: EvalMode) -> None:
    async def fake_run_agent(*, user_prompt, **kwargs):
        raise AssertionError('run_agent should not be called in this unit test')

    task_input = search_task_module.SearchTaskInput(
        question=QUESTION,
        answer=ANSWER,
        eval_mode=requested_mode,
        run_agent=fake_run_agent,
    )

    resolved = search_task_module._resolve_eval_mode(task_input)

    assert resolved == requested_mode


@pytest.mark.parametrize(
    'requested_mode',
    [EvalMode.LLM_JUDGE, EvalMode.EXACT_MATCH, EvalMode.REWARD_SERVICE],
)
def test_resolve_eval_mode_detects_wide_search_from_dataset_evaluation(requested_mode: EvalMode) -> None:
    async def fake_run_agent(*, user_prompt, **kwargs):
        raise AssertionError('run_agent should not be called in this unit test')

    task_input = search_task_module.SearchTaskInput(
        question=QUESTION,
        answer=ANSWER,
        evaluation=WIDE_SEARCH_EVALUATION,
        eval_mode=requested_mode,
        run_agent=fake_run_agent,
    )

    resolved = search_task_module._resolve_eval_mode(task_input)

    assert resolved == EvalMode.WIDESEARCH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('task_module', 'task_fn', 'task_input_cls'),
    [
        pytest.param(
            search_task_module, search_task_module.search_task, search_task_module.SearchTaskInput, id='search'
        ),
        pytest.param(
            hy3_search_task_module,
            hy3_search_task_module.hy3_search_task,
            hy3_search_task_module.Hy3SearchTaskInput,
            id='hy3_search',
        ),
    ],
)
@pytest.mark.parametrize(
    'exit_status',
    [TerminationReason.TRUNCATION, MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value],
)
async def test_training_length_exceeded_short_circuits_evaluation(
    monkeypatch, task_module, task_fn, task_input_cls, exit_status
) -> None:
    async def fake_run_agent(*, user_prompt, **kwargs):
        assert user_prompt == WHAT_HAPPENED_QUESTION
        return FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': PARTIAL_ANSWER}],
            final_answer=PARTIAL_ANSWER,
            exit_status=exit_status,
        )

    async def fail_evaluate_output(**kwargs):
        raise AssertionError('evaluate_output should not be called when training length exceeded short-circuits')

    monkeypatch.setattr(task_module, 'evaluate_output', fail_evaluate_output)

    output = await task_fn(
        task_input_cls(
            question=WHAT_HAPPENED_QUESTION,
            answer=REFERENCE_ANSWER,
            training_length_exceeded=True,
            run_agent=fake_run_agent,
        )
    )

    assert output.score == 0.0
    assert output.prediction == PARTIAL_ANSWER
    assert output.answer == 'reference answer'
    assert output.trajectory == [{'role': 'assistant', 'content': PARTIAL_ANSWER}]


@pytest.mark.asyncio
async def test_search_task_soft_eval_uses_last_message_when_final_answer_missing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run_agent(*, user_prompt, **kwargs):
        assert user_prompt == RECOVERY_QUESTION
        return FakeAgentOutput(
            main_trajectory=[
                {'role': 'assistant', 'content': RECOVERED_ANSWER},
            ],
            final_answer=None,
            exit_status=TerminationReason.SUCCESS,
        )

    class FakeEvaluator:
        def __init__(self, mode, **kwargs):
            captured['mode'] = mode
            captured['kwargs'] = kwargs

    async def fake_evaluate_output(*, prediction, reference, question, evaluator, need_details):
        captured['prediction'] = prediction
        captured['reference'] = reference
        captured['question'] = question
        captured['need_details'] = need_details
        return 1.0, EvaluationDetails(), True

    monkeypatch.setattr(search_task_module, 'SearchEvaluator', FakeEvaluator)
    monkeypatch.setattr(search_task_module, 'evaluate_output', fake_evaluate_output)

    output = await search_task_module.search_task(
        search_task_module.SearchTaskInput(
            question=RECOVERY_QUESTION,
            answer=REFERENCE_ANSWER,
            eval_mode=EvalMode.LLM_JUDGE,
            soft_eval=True,
            soft_score=0.6,
            run_agent=fake_run_agent,
        )
    )

    assert captured['mode'] == EvalMode.LLM_JUDGE
    assert captured['prediction'] == RECOVERED_ANSWER
    assert captured['reference'] == 'reference answer'
    assert output.prediction == RECOVERED_ANSWER
    assert output.score == 0.6
    assert output.judge_status == 'success'


@pytest.mark.asyncio
@pytest.mark.parametrize('eval_mode', [EvalMode.LLM_JUDGE, EvalMode.REWARD_SERVICE])
async def test_search_task_marks_judge_failed_when_details_report_failure(monkeypatch, eval_mode: EvalMode) -> None:
    async def fake_run_agent(*, user_prompt, **kwargs):
        return FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': AGENT_ANSWER}],
            final_answer=AGENT_ANSWER,
            exit_status=TerminationReason.SUCCESS,
        )

    class FakeEvaluator:
        def __init__(self, mode, **kwargs):
            self.mode = mode

    async def fake_evaluate_output(*, prediction, reference, question, evaluator, need_details):
        return 0.0, EvaluationDetails(status='failed', msg='judge backend unavailable'), True

    monkeypatch.setattr(search_task_module, 'SearchEvaluator', FakeEvaluator)
    monkeypatch.setattr(search_task_module, 'evaluate_output', fake_evaluate_output)

    output = await search_task_module.search_task(
        search_task_module.SearchTaskInput(
            question=JUDGED_QUESTION,
            answer=REFERENCE_ANSWER,
            eval_mode=eval_mode,
            run_agent=fake_run_agent,
        )
    )

    assert output.score == 0.0
    assert output.prediction == AGENT_ANSWER
    assert output.judge_status == 'failed'
    assert output.evaluation_details == {'status': 'failed', 'msg': 'judge backend unavailable'}


@pytest.mark.asyncio
async def test_search_task_marks_judge_failed_when_prediction_is_empty(monkeypatch) -> None:
    async def fake_run_agent(*, user_prompt, **kwargs):
        return FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': ''}],
            final_answer='',
            exit_status=TerminationReason.SUCCESS,
        )

    async def fail_evaluate_output(**kwargs):
        raise AssertionError('evaluate_output should not be called for an empty prediction')

    monkeypatch.setattr(search_task_module, 'evaluate_output', fail_evaluate_output)

    output = await search_task_module.search_task(
        search_task_module.SearchTaskInput(
            question=JUDGED_QUESTION,
            answer=REFERENCE_ANSWER,
            eval_mode=EvalMode.LLM_JUDGE,
            soft_eval=False,
            run_agent=fake_run_agent,
        )
    )

    assert output.score == 0.0
    assert output.prediction == ''
    assert output.judge_status == 'failed'
    assert output.evaluation_details == {
        'status': 'failed',
        'msg': 'Prediction is empty; judge evaluation skipped.',
    }


@pytest.mark.asyncio
async def test_search_task_wide_search_uses_last_assistant_message_and_overrides_eval_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run_agent(*, user_prompt, **kwargs):
        return FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': 'company=acme'}],
            final_answer=None,
            exit_status=TerminationReason.MAX_ITERATIONS,
        )

    class FakeEvaluator:
        def __init__(self, mode, **kwargs):
            captured['mode'] = mode
            captured['kwargs'] = kwargs

    async def fake_evaluate_output(*, prediction, reference, question, evaluator, need_details):
        captured['prediction'] = prediction
        captured['reference'] = reference
        captured['question'] = question
        captured['need_details'] = need_details
        return 1.0, EvaluationDetails(status='success'), True

    monkeypatch.setattr(search_task_module, 'SearchEvaluator', FakeEvaluator)
    monkeypatch.setattr(search_task_module, 'evaluate_output', fake_evaluate_output)

    output = await search_task_module.search_task(
        search_task_module.SearchTaskInput(
            question='Find a table',
            answer='company=acme',
            eval_mode=EvalMode.LLM_JUDGE,
            evaluation=WIDE_SEARCH_EVALUATION,
            run_agent=fake_run_agent,
        )
    )

    assert captured['mode'] == EvalMode.WIDESEARCH
    assert captured['prediction'] == 'company=acme'
    assert captured['reference'] == 'company=acme'
    assert output.prediction == 'company=acme'
    assert output.score == 1.0
    assert output.judge_status == 'success'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('requested_mode', 'expected_judge_status'),
    [
        (EvalMode.EXACT_MATCH, None),
        (EvalMode.REWARD_SERVICE, 'success'),
    ],
    ids=['exact-match', 'reward-service'],
)
async def test_search_task_normal_sample_uses_requested_eval_mode(
    monkeypatch,
    requested_mode: EvalMode,
    expected_judge_status: str | None,
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_agent(*, user_prompt, **kwargs):
        return FakeAgentOutput(
            main_trajectory=[{'role': 'assistant', 'content': AGENT_ANSWER}],
            final_answer=AGENT_ANSWER,
            exit_status=TerminationReason.SUCCESS,
        )

    class FakeEvaluator:
        def __init__(self, mode, **kwargs):
            captured['mode'] = mode

    async def fake_evaluate_output(*, prediction, reference, question, evaluator, need_details):
        captured['prediction'] = prediction
        captured['reference'] = reference
        return 1.0, EvaluationDetails(status='success'), True

    async def fake_generate_final_answer(*, trajectory, return_type):
        captured['generated_from_trajectory'] = trajectory
        captured['return_type'] = return_type
        return AGENT_ANSWER

    monkeypatch.setattr(search_task_module, 'SearchEvaluator', FakeEvaluator)
    monkeypatch.setattr(search_task_module, 'evaluate_output', fake_evaluate_output)
    monkeypatch.setattr(search_task_module, 'generate_final_answer', fake_generate_final_answer)

    output = await search_task_module.search_task(
        search_task_module.SearchTaskInput(
            question=JUDGED_QUESTION,
            answer=REFERENCE_ANSWER,
            eval_mode=requested_mode,
            run_agent=fake_run_agent,
        )
    )

    assert captured['mode'] == requested_mode
    if requested_mode == EvalMode.EXACT_MATCH:
        assert captured['generated_from_trajectory'] == [{'role': 'assistant', 'content': AGENT_ANSWER}]
        assert captured['return_type'] == 'str'
        assert captured['prediction'] == AGENT_ANSWER.lower()
    else:
        assert captured['prediction'] == AGENT_ANSWER
    assert captured['reference'] == 'reference answer'
    assert output.score == 1.0
    assert output.judge_status == expected_judge_status
