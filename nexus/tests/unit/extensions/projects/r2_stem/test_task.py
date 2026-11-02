import asyncio
import json
import sys

from types import SimpleNamespace

import pytest

import nexus.extensions.projects.r2_stem.task as task_module

from nexus.extensions.projects.r2_stem.evaluator.common import load_ref_answer
from nexus.extensions.projects.r2_stem.task import (
    ScienceAgentTaskInput,
    _agent_succeeded,
    _agent_wall_time_limit,
    _build_agent_prompt,
    _build_verifier_observability,
    _cutoff_termination_reason,
    _extract_answer,
    _extract_hycode_answer,
    _normalize_evaluation_method,
    _resolve_agent_name,
    run_science_agent_task,
)


def test_cutoff_reason_separates_output_limit_from_context_limit() -> None:
    from nexus.extensions.projects.r2_stem.agent import MiniSWEAgentStemExitStatus

    assert _cutoff_termination_reason(SimpleNamespace(exit_status='truncation')) == 'TRUNCATION'
    assert (
        _cutoff_termination_reason(SimpleNamespace(exit_status=MiniSWEAgentStemExitStatus.OUTPUT_LENGTH_EXCEEDED))
        == 'OUTPUT_LENGTH_EXCEEDED'
    )
    # An agent that reports the canonical reason directly is trusted over its exit status.
    assert (
        _cutoff_termination_reason(SimpleNamespace(termination_reason='OUTPUT_LENGTH_EXCEEDED', exit_status='error'))
        == 'OUTPUT_LENGTH_EXCEEDED'
    )
    assert _cutoff_termination_reason(SimpleNamespace(exit_status='submitted')) is None
    assert _cutoff_termination_reason(SimpleNamespace(exit_status=None)) is None


def test_cutoff_reason_covers_an_agent_that_ran_out_of_time() -> None:
    from nexus.extensions.projects.r2_stem.agent import MiniSWEAgentStemExitStatus

    assert _cutoff_termination_reason(SimpleNamespace(exit_status='timeout')) == 'TASK_TIMEOUT'
    assert _cutoff_termination_reason(SimpleNamespace(exit_status=MiniSWEAgentStemExitStatus.TIMEOUT)) == 'TASK_TIMEOUT'


async def _unused_agent(**kwargs):
    pytest.fail('this input is only used to read configuration')


def test_agent_wall_time_limit_leaves_room_for_the_agent_to_stop_itself() -> None:
    task_input = ScienceAgentTaskInput(question='q', agent_timeout=18000, run_agent=_unused_agent)

    assert _agent_wall_time_limit(task_input) == 17400.0


def test_agent_wall_time_limit_caps_the_reserve_on_a_short_timeout() -> None:
    """A reserve larger than the timeout would leave 0, which agents read as 'no limit'."""
    task_input = ScienceAgentTaskInput(question='q', agent_timeout=60, run_agent=_unused_agent)

    assert _agent_wall_time_limit(task_input) == 54.0


def test_sandbox_evaluation_names_are_supported() -> None:
    assert _normalize_evaluation_method('unit_cases') == 'unit_cases'
    assert _normalize_evaluation_method('verifier_image') == 'verifier_image'
    with pytest.raises(ValueError, match='Unsupported evaluation_method'):
        _normalize_evaluation_method('terminal_bench')


def test_verifier_observability_is_bounded_and_numeric() -> None:
    result = _build_verifier_observability(
        {
            'results': {'tool': {'name': 'pytest'}, 'summary': {'tests': 8, 'passed': 6, 'failed': 1, 'skipped': 1}},
            'nexus_verifier': {
                'reward': 1.0,
                'return_code': 0,
                'status': 'finished',
                'duration_seconds': 12.5,
                'log_tail': 'must not cross into extra_fields',
                'log_truncated': True,
            },
        },
        attempted=True,
        scoring_failed=False,
        total_duration_seconds=20.0,
    )

    metrics = result['reward_extra_info']['extra_reward_metrics']
    assert metrics == {
        'verifier_attempted': 1.0,
        'verifier_completed': 1.0,
        'verifier_scoring_failed': 0.0,
        'verifier_reward_present': 1.0,
        'verifier_graded_cutoff': 0.0,
        'verifier_total_duration_s': 20.0,
        'verifier_test_duration_s': 12.5,
        'verifier_setup_artifact_duration_s': 7.5,
        'verifier_nonzero_return_code': 0.0,
        'verifier_log_truncated': 1.0,
        'verifier_tests_total': 8.0,
        'verifier_tests_passed': 6.0,
        'verifier_tests_failed': 1.0,
        'verifier_tests_skipped': 1.0,
        'verifier_tests_pass_ratio': 0.75,
    }
    assert result['verifier_summary']['tests'] == {'tests': 8, 'passed': 6, 'failed': 1, 'skipped': 1}
    assert 'log_tail' not in result['verifier_summary']


def test_verifier_observability_tolerates_missing_and_malformed_fields() -> None:
    missing = _build_verifier_observability(
        None,
        attempted=True,
        scoring_failed=True,
        total_duration_seconds=float('nan'),
    )
    assert missing == {
        'reward_extra_info': {
            'extra_reward_metrics': {
                'verifier_attempted': 1.0,
                'verifier_completed': 0.0,
                'verifier_scoring_failed': 1.0,
                'verifier_reward_present': 0.0,
                'verifier_graded_cutoff': 0.0,
            }
        },
        'verifier_summary': {'attempted': True, 'completed': False, 'scoring_failed': True},
    }

    malformed = _build_verifier_observability(
        {
            'summary': {'tests': 'bad', 'passed': None},
            'nexus_verifier': {
                'reward': object(),
                'return_code': 'signal',
                'duration_seconds': -2,
            },
        },
        attempted=True,
        scoring_failed=False,
        total_duration_seconds=1,
    )
    metrics = malformed['reward_extra_info']['extra_reward_metrics']
    assert metrics['verifier_reward_present'] == 0.0
    assert metrics['verifier_test_duration_s'] == 0.0
    assert metrics['verifier_setup_artifact_duration_s'] == 1.0
    assert malformed['verifier_summary']['return_code'] == 'signal'


def test_verifier_observability_reads_a_bare_summary_and_says_why_scoring_stopped() -> None:
    failed = _build_verifier_observability(
        {'summary': {'tests': 4, 'passed': 1}},
        attempted=True,
        scoring_failed=True,
        total_duration_seconds=3.0,
        error='verifier_image_error:' + 'x' * 2000,
        graded_cutoff_reason='TRUNCATION',
    )
    metrics = failed['reward_extra_info']['extra_reward_metrics']
    assert metrics['verifier_tests_pass_ratio'] == 0.25
    assert metrics['verifier_graded_cutoff'] == 1.0
    assert metrics['verifier_scoring_failed'] == 1.0
    summary = failed['verifier_summary']
    assert summary['scoring_failed'] is True
    assert summary['graded_cutoff_reason'] == 'TRUNCATION'
    assert summary['error'].startswith('verifier_image_error:')
    assert len(summary['error']) == task_module.MAX_SUMMARY_ERROR_CHARS

    skipped = _build_verifier_observability(
        {'skipped_evaluator_reason': 'TASK_TIMEOUT', 'exit_status': 'timeout'},
        attempted=False,
        scoring_failed=False,
        total_duration_seconds=None,
    )
    assert skipped['verifier_summary']['skipped_reason'] == 'TASK_TIMEOUT'
    assert skipped['reward_extra_info']['extra_reward_metrics']['verifier_attempted'] == 0.0


def test_only_terminus_text_evaluation_gets_answer_file_prompt() -> None:
    question = 'Solve the problem.'

    terminus_prompt = _build_agent_prompt(
        question=question,
        evaluation_method='llm_judge',
        agent_name='terminus2',
        working_dir='/work/project',
    )
    assert question in terminus_prompt
    assert 'write exactly one output file: /work/project/answer.txt' in terminus_prompt

    assert (
        _build_agent_prompt(
            question=question,
            evaluation_method='unit_cases',
            agent_name='terminus2',
            working_dir='/work/project',
        )
        == question
    )
    assert (
        _build_agent_prompt(
            question=question,
            evaluation_method='verifier_image',
            agent_name='terminus2',
            working_dir='/work/project',
        )
        == question
    )
    assert (
        _build_agent_prompt(
            question=question,
            evaluation_method='llm_judge',
            agent_name='mini_swe_agent_stem',
            working_dir='/work/project',
        )
        == question
    )


@pytest.mark.asyncio
async def test_mini_swe_extracts_submission() -> None:
    output = SimpleNamespace(submission='  final answer  ', main_trajectory=[])

    answer = await _extract_answer(
        agent_name='mini_swe_agent_stem',
        agent_output=output,
        runtime=SimpleNamespace(),
        evaluation_method='llm_judge',
        working_dir='/workspace',
        test_timeout=120,
    )

    assert answer == 'final answer'


@pytest.mark.asyncio
async def test_mini_swe_falls_back_to_last_assistant_message() -> None:
    output = SimpleNamespace(
        submission='',
        main_trajectory=[{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': ' answer '}],
    )

    answer = await _extract_answer(
        agent_name='mini_swe_agent_stem',
        agent_output=output,
        runtime=SimpleNamespace(),
        evaluation_method='llm_judge',
        working_dir='/workspace',
        test_timeout=120,
    )

    assert answer == 'answer'


@pytest.mark.asyncio
async def test_terminus_llm_answer_uses_post_agent_capture_script(monkeypatch) -> None:
    calls = []

    async def fake_capture_answer_file(*, runtime, answer_path, test_timeout):
        calls.append((runtime, answer_path, test_timeout))
        return 'captured answer'

    monkeypatch.setattr(task_module, 'capture_answer_file', fake_capture_answer_file)
    runtime = SimpleNamespace()

    answer = await _extract_answer(
        agent_name='terminus2',
        agent_output=SimpleNamespace(),
        runtime=runtime,
        evaluation_method='llm_judge',
        working_dir='/workspace/task',
        test_timeout=321,
    )

    assert answer == 'captured answer'
    assert calls == [(runtime, '/workspace/task/answer.txt', 321)]


def test_hycode_result_event_provides_the_answer() -> None:
    output = '\n'.join(
        [
            '{"type":"assistant","message":{"content":"let me check the panel"}}',
            '{"type":"result","subtype":"success","is_error":false,"result":"  GAPDH  "}',
        ]
    )
    assert _extract_hycode_answer(output) == 'GAPDH'


def test_hycode_last_result_event_wins() -> None:
    output = '\n'.join(
        [
            '{"type":"result","subtype":"success","result":"first"}',
            '{"type":"result","subtype":"success","result":"second"}',
        ]
    )
    assert _extract_hycode_answer(output) == 'second'


def test_hycode_empty_result_falls_back_to_newest_assistant_text() -> None:
    output = '\n'.join(
        [
            '{"type":"assistant","message":{"content":"early guess"}}',
            '{"type":"assistant","message":{"content":"final guess"}}',
            '{"type":"result","subtype":"error_max_turns","is_error":true,"result":""}',
        ]
    )
    assert _extract_hycode_answer(output) == 'final guess'


def test_hycode_reads_assistant_content_blocks() -> None:
    output = (
        '{"type":"assistant","message":{"content":'
        '[{"type":"text","text":"answer: "},{"type":"tool_use","id":"x"},{"type":"text","text":"42"}]}}'
    )
    assert _extract_hycode_answer(output) == 'answer: 42'


def test_hycode_thinking_only_turn_is_used_instead_of_an_earlier_line() -> None:
    # The shape 5 of 32 real L1 sessions ended on: the model omitted its think-end token, so
    # the backend reported the whole turn as reasoning and the turn carried no text block.
    # Reaching past it would judge a mid-run debugging line instead of what the model said.
    output = '\n'.join(
        [
            '{"type":"assistant","message":{"id":"m1","content":'
            '[{"type":"text","text":"cv2 is not available. I will remove cv2 usage."}]}}',
            '{"type":"assistant","message":{"id":"m2","content":'
            '[{"type":"thinking","thinking":"Final answer: {\\"answer_id\\": \\"IFITM3\\"}"}]}}',
            '{"type":"result","subtype":"success","is_error":false,"result":""}',
        ]
    )
    assert _extract_hycode_answer(output) == 'Final answer: {"answer_id": "IFITM3"}'


def test_hycode_prefers_text_over_thinking_within_one_turn() -> None:
    output = (
        '{"type":"assistant","message":{"content":'
        '[{"type":"thinking","thinking":"weighing the options"},{"type":"text","text":"LST1"}]}}'
    )
    assert _extract_hycode_answer(output) == 'LST1'


def test_hycode_result_event_still_outranks_a_thinking_only_turn() -> None:
    output = '\n'.join(
        [
            '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"maybe PV"}]}}',
            '{"type":"result","subtype":"success","result":"PV"}',
        ]
    )
    assert _extract_hycode_answer(output) == 'PV'


def test_hycode_tool_use_only_turn_does_not_clear_the_answer() -> None:
    # A tool call carries nothing for the judge, so it must not displace the real answer.
    output = '\n'.join(
        [
            '{"type":"assistant","message":{"content":[{"type":"text","text":"LST1"}]}}',
            '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"x","name":"Bash"}]}}',
        ]
    )
    assert _extract_hycode_answer(output) == 'LST1'


def test_hycode_reads_assistant_content_placed_on_the_event() -> None:
    assert _extract_hycode_answer('{"type":"assistant","content":"top level"}') == 'top level'


def test_hycode_skips_lines_that_are_not_json_events() -> None:
    output = '\n'.join(
        [
            'npm WARN deprecated something',
            '',
            '[]',
            '{"type":"result","result":"clean"}',
        ]
    )
    assert _extract_hycode_answer(output) == 'clean'


def test_hycode_blank_assistant_events_do_not_clear_the_answer() -> None:
    output = '\n'.join(
        [
            '{"type":"assistant","message":{"content":"real answer"}}',
            '{"type":"assistant","message":{"content":"   "}}',
        ]
    )
    assert _extract_hycode_answer(output) == 'real answer'


def test_hycode_output_without_usable_text_yields_empty_answer() -> None:
    assert _extract_hycode_answer('') == ''
    assert _extract_hycode_answer('{"type":"system","subtype":"init"}') == ''
    assert _extract_hycode_answer('{"type":"result","result":null}') == ''


def test_hycode_tolerates_a_very_long_event_line() -> None:
    filler = 'x' * 100_000
    output = '\n'.join(
        [
            f'{{"type":"assistant","message":{{"content":"{filler}"}}}}',
            '{"type":"result","subtype":"success","result":"short answer"}',
        ]
    )
    assert _extract_hycode_answer(output) == 'short answer'


def test_hycode_text_judging_asks_for_an_answer_only_final_message() -> None:
    question = 'Name the control gene.'
    prompt = _build_agent_prompt(
        question=question,
        evaluation_method='rmsys',
        agent_name='hycode',
        working_dir='/app',
    )
    assert question in prompt
    # The session ends on the first turn without a tool call, so the prompt has to state that
    # rule as well as what the final message should hold. Asserting only the latter let a
    # wording through that never told the model how to end, which stranded 31% of real
    # sessions on a mid-work line.
    assert 'every response must include at least one tool call' in prompt
    assert 'A response without a tool call ends the session' in prompt
    assert 'the entire content of that message' in prompt

    assert (
        _build_agent_prompt(
            question=question,
            evaluation_method='unit_cases',
            agent_name='hycode',
            working_dir='/app',
        )
        == question
    )


@pytest.mark.asyncio
async def test_hycode_extract_answer_reads_stream_json_output() -> None:
    output = SimpleNamespace(
        output='{"type":"result","subtype":"success","result":"GAPDH"}',
        termination_reason='ENV_DONE',
    )
    answer = await _extract_answer(
        agent_name='hycode',
        agent_output=output,
        runtime=SimpleNamespace(),
        evaluation_method='rmsys',
        working_dir='/app',
        test_timeout=120,
    )
    assert answer == 'GAPDH'


def test_hycode_success_status_comes_from_termination_reason() -> None:
    assert _agent_succeeded('hycode', 'ENV_DONE')
    assert not _agent_succeeded('hycode', 'MAX_STEPS')
    assert not _agent_succeeded('hycode', 'FORMAT_ERROR')


def test_running_out_of_turns_is_judged_not_cut_off_for_either_agent() -> None:
    # Both harnesses treat a turn-limit stop the same way: it is not a length cutoff, so the
    # run is judged on whatever it produced instead of taking a blanket zero.
    assert _cutoff_termination_reason(SimpleNamespace(termination_reason='MAX_STEPS')) is None
    assert _cutoff_termination_reason(SimpleNamespace(exit_status='max_iterations')) is None
    assert _cutoff_termination_reason(SimpleNamespace(termination_reason='ENV_DONE')) is None
    assert _cutoff_termination_reason(SimpleNamespace(termination_reason='FORMAT_ERROR')) is None


def test_truncation_is_a_cutoff_for_either_agent() -> None:
    # Mini-SWE spells it as a lowercase exit status, HyCode as a canonical termination reason.
    assert (
        _cutoff_termination_reason(SimpleNamespace(exit_status=SimpleNamespace(value='truncation'))) == 'TRUNCATION'
    )
    assert _cutoff_termination_reason(SimpleNamespace(termination_reason='TRUNCATION')) == 'TRUNCATION'


def test_success_status_depends_on_agent() -> None:
    assert _agent_succeeded('mini_swe_agent_stem', 'submitted')
    assert not _agent_succeeded('mini_swe_agent_stem', 'completed')
    assert _agent_succeeded('terminus2', 'completed')
    assert not _agent_succeeded('terminus2', 'submitted')


def test_agent_name_comes_from_task_config_without_execution_context() -> None:
    assert _resolve_agent_name('terminus2', None) == 'terminus2'
    assert _resolve_agent_name('  terminus2  ', '') == 'terminus2'


def test_agent_name_falls_back_to_runner_value() -> None:
    assert _resolve_agent_name('', 'mini_swe_agent_stem') == 'mini_swe_agent_stem'


def test_agent_name_rejects_task_config_contradicting_runner() -> None:
    with pytest.raises(ValueError, match='does not match the configured agent_name'):
        _resolve_agent_name('terminus2', 'mini_swe_agent_stem')


def test_agent_name_is_required() -> None:
    with pytest.raises(ValueError, match='requires task_config.agent_name'):
        _resolve_agent_name('', None)


def test_agent_name_must_be_supported() -> None:
    with pytest.raises(ValueError, match='Unsupported agent: claude_code'):
        _resolve_agent_name('claude_code', None)


@pytest.mark.asyncio
async def test_mini_swe_unit_cases_uploads_tests_only_after_agent(monkeypatch) -> None:
    events = []
    runtime = SimpleNamespace()

    class FakeProvider:
        def configure(self, name, **kwargs):
            events.append(('configure', name, kwargs))

        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        assert candidate_runtime is runtime
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        assert events[-1] == 'hide_verifier'
        events.append(('agent', kwargs))
        return SimpleNamespace(
            exit_status=SimpleNamespace(value='submitted'),
            iterations=3,
            raw_terminal_output='terminal output',
            error=None,
            error_type=None,
            error_traceback=None,
        )

    async def fake_unit_cases_evaluate(**kwargs):
        assert events[-1][0] == 'agent'
        events.append(('upload_and_run_tests', kwargs))
        return 1.0, {'results': []}

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'unit_cases_evaluate', fake_unit_cases_evaluate)

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Solve it.',
            agent_name='mini_swe_agent_stem',
            evaluation_method='unit_cases',
            task_dir='/host/task',
            run_agent=fake_agent,
        )
    )

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names == ['get_runtime', 'hide_verifier', 'agent', 'upload_and_run_tests', 'release']
    assert output.score == 1.0
    assert output.answer is None


@pytest.mark.asyncio
async def test_terminus_verifier_image_runs_only_after_agent(monkeypatch) -> None:
    events = []
    runtime = SimpleNamespace()

    class FakeProvider:
        def configure(self, name, **kwargs):
            events.append(('configure', name, kwargs))

        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        assert candidate_runtime is runtime
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        assert events[-1] == 'hide_verifier'
        events.append(('agent', kwargs))
        return SimpleNamespace(exit_status=SimpleNamespace(value='completed'), n_iterations=5)

    async def fake_verifier_image_evaluate(**kwargs):
        assert events[-1][0] == 'agent'
        assert kwargs['agent_runtime_name'] == 'main'
        assert kwargs['verifier_runtime_name'] == 'verifier'
        assert kwargs['verifier_mounts'][0].source == '/app/output.patch'
        events.append(('run_isolated_verifier', kwargs))
        return 0.75, {
            'summary': {'tests': 4, 'passed': 3, 'failed': 1},
            'nexus_verifier': {
                'reward': 0.75,
                'return_code': 0,
                'status': 'finished',
                'duration_seconds': 1.5,
                'log_tail': 'pytest output',
                'log_truncated': False,
            },
        }

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'verifier_image_evaluate', fake_verifier_image_evaluate)

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Create the patch.',
            agent_name='terminus2',
            evaluation_method='verifier_image',
            task_dir='/host/task',
            sample_id='sample-1',
            verifier_image='registry/verifier@sha256:' + 'a' * 64,
            verifier_mounts=[{'source': '/app/output.patch'}],
            run_agent=fake_agent,
        )
    )

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names == ['get_runtime', 'hide_verifier', 'agent', 'run_isolated_verifier', 'release']
    assert output.score == 0.75
    assert output.answer is None
    assert output.test_results['summary'] == {'tests': 4, 'passed': 3, 'failed': 1}
    metrics = output.extra_fields['reward_extra_info']['extra_reward_metrics']
    assert metrics['verifier_completed'] == 1.0
    assert metrics['verifier_reward_present'] == 1.0
    assert metrics['verifier_tests_pass_ratio'] == 0.75
    assert output.extra_fields['verifier_summary']['reward'] == 0.75
    assert 'log_tail' not in output.extra_fields['verifier_summary']


_PASSING_VERIFIER_RESULT = (
    1.0,
    {
        'results': {'summary': {'tests': 2, 'passed': 2, 'failed': 0}},
        'nexus_verifier': {
            'reward': 1.0,
            'return_code': 0,
            'status': 'finished',
            'duration_seconds': 3.0,
            'log_tail': '',
            'log_truncated': False,
        },
    },
)


def _cut_off_verifier_harness(
    monkeypatch, *, exit_status=None, agent_output=None, verifier_error=None, verifier_result=None, tool_parser=None
):
    """Drive the task with an agent that stops with ``exit_status`` (or returns ``agent_output``)
    and a stub verifier. ``tool_parser`` goes on the rollout client, as the HyTuner runner builds it."""
    events = []
    span_attributes: dict[str, object] = {}
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name):
            events.append('release')

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()
        openai_clients = {'main': SimpleNamespace(config=SimpleNamespace(client_args={'tool_parser': tool_parser}))}

        async def get_runtime(self, name):
            events.append('get_runtime')
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        events.append('agent')
        if agent_output is not None:
            return agent_output
        return SimpleNamespace(exit_status=SimpleNamespace(value=exit_status), n_iterations=5)

    async def fake_verifier_image_evaluate(**kwargs):
        events.append('verify')
        if verifier_error is not None:
            raise verifier_error
        return verifier_result if verifier_result is not None else _PASSING_VERIFIER_RESULT

    async def fail_text_judging(**kwargs):
        pytest.fail('a cut-off rollout must never reach text judging')

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'verifier_image_evaluate', fake_verifier_image_evaluate)
    monkeypatch.setattr(task_module, 'capture_answer_file', fail_text_judging)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fail_text_judging)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(task_module, 'update_span', lambda span_id, attributes: span_attributes.update(attributes))
    return events, span_attributes, fake_agent


def _cut_off_task_input(run_agent, *, agent_name='terminus2', evaluation_method='verifier_image', score_truncated=True):
    return ScienceAgentTaskInput(
        question='Write the results file.',
        agent_name=agent_name,
        ref_answer='reference',
        evaluation_method=evaluation_method,
        working_dir='/app',
        sample_id='sample-1',
        verifier_image='registry/verifier@sha256:' + 'a' * 64,
        verifier_mounts=[{'source': '/app/results.json'}],
        verifier_score_truncated=score_truncated,
        run_agent=run_agent,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('exit_status', 'reason'),
    [('truncation', 'TRUNCATION'), ('output_length_exceeded', 'OUTPUT_LENGTH_EXCEEDED')],
)
async def test_cut_off_verifier_rollout_is_graded_when_the_row_asks(monkeypatch, exit_status, reason) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(monkeypatch, exit_status=exit_status)

    output = await run_science_agent_task(_cut_off_task_input(agent))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    # The rollout still trains as a cut-off; only its score now comes from the verifier.
    assert span_attributes['hytuner.termination_reason'] == reason
    assert output.error == f'agent_exit_status:{exit_status}'
    metrics = output.extra_fields['reward_extra_info']['extra_reward_metrics']
    assert metrics['verifier_graded_cutoff'] == 1.0
    assert metrics['verifier_tests_pass_ratio'] == 1.0
    assert output.extra_fields['verifier_summary']['graded_cutoff_reason'] == reason


@pytest.mark.asyncio
async def test_hycode_truncation_on_a_verifier_row_is_graded(monkeypatch) -> None:
    # HyCode reports the canonical reason instead of an exit status.
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=SimpleNamespace(termination_reason='TRUNCATION', num_turns=40)
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert span_attributes['hytuner.termination_reason'] == 'TRUNCATION'
    assert output.extra_fields['verifier_summary']['graded_cutoff_reason'] == 'TRUNCATION'


@pytest.mark.asyncio
async def test_cut_off_verifier_rollout_scores_zero_unseen_by_default(monkeypatch) -> None:
    events, _, agent = _cut_off_verifier_harness(monkeypatch, exit_status='truncation')

    output = await run_science_agent_task(_cut_off_task_input(agent, score_truncated=False))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'release']
    assert output.score == 0.0
    assert output.test_results == {'skipped_evaluator_reason': 'TRUNCATION', 'exit_status': 'truncation'}
    assert output.extra_fields['verifier_summary']['skipped_reason'] == 'TRUNCATION'
    assert output.extra_fields['reward_extra_info']['extra_reward_metrics']['verifier_graded_cutoff'] == 0.0


@pytest.mark.asyncio
async def test_timed_out_verifier_rollout_that_passes_everything_reports_env_done(monkeypatch) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(monkeypatch, exit_status='timeout')

    output = await run_science_agent_task(_cut_off_task_input(agent))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    # Every test passed, so the work was finished even though the clock was not. Keeping
    # TASK_TIMEOUT would have HyTuner mask the row and throw the grade away.
    assert span_attributes['hytuner.termination_reason'] == 'ENV_DONE'
    assert output.extra_fields['verifier_summary']['graded_cutoff_reason'] == 'TASK_TIMEOUT'


@pytest.mark.asyncio
async def test_timed_out_verifier_rollout_short_of_full_marks_stays_a_timeout(monkeypatch) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, exit_status='timeout', verifier_result=(0.6, {'nexus_verifier': {'reward': 0.6}})
    )

    output = await run_science_agent_task(_cut_off_task_input(agent))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 0.6
    # A partial pass does not show the work was finished, so the reason stands and HyTuner
    # applies its own maskout.
    assert span_attributes['hytuner.termination_reason'] == 'TASK_TIMEOUT'


@pytest.mark.asyncio
async def test_env_error_that_passes_everything_reports_env_done(monkeypatch) -> None:
    # ENV_ERROR is not a cut-off, so it reaches the verifier already. It is masked by HyTuner
    # though, so without this the grade it earned would be thrown away.
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=SimpleNamespace(termination_reason='ENV_ERROR', num_turns=12)
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert span_attributes['hytuner.termination_reason'] == 'ENV_DONE'


@pytest.mark.asyncio
async def test_env_error_short_of_full_marks_stays_an_env_error(monkeypatch) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch,
        agent_output=SimpleNamespace(termination_reason='ENV_ERROR', num_turns=12),
        verifier_result=(0.6, {'nexus_verifier': {'reward': 0.6}}),
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 0.6
    assert 'hytuner.termination_reason' not in span_attributes


@pytest.mark.asyncio
async def test_max_steps_keeps_its_reason_because_hytuner_trains_it(monkeypatch) -> None:
    # MAX_STEPS is absent from MASKOUT_TERMINATION_REASONS, so its real score already trains
    # and there is nothing to rescue by relabelling.
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=SimpleNamespace(termination_reason='MAX_STEPS', num_turns=200)
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert 'hytuner.termination_reason' not in span_attributes


@pytest.mark.asyncio
async def test_unhandled_error_is_not_rescued_by_a_passing_grade(monkeypatch) -> None:
    # Masked, but it names a failure nobody explained; a passing grade is not evidence that
    # the failure was harmless.
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=SimpleNamespace(termination_reason='UNHANDLED_ERROR', num_turns=3)
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert 'hytuner.termination_reason' not in span_attributes


def _real_hycode_output(
    termination_reason: str,
    hycode_exit_name: str | None = None,
    *,
    output: str = '',
    result_subtype: str | None = None,
):
    """Build the output run_hycode returns, including its lowercased display exit_status."""
    from nexus.extensions.agents.hycode.agent import HyCodeOutput

    return HyCodeOutput(
        success=True,
        output=output,
        stderr='',
        exit_code=0,
        execution_time=1.0,
        exit_status=(hycode_exit_name or termination_reason).lower(),
        termination_reason=termination_reason,
        result_subtype=result_subtype,
        num_turns=12,
        scoreable_process_exit=hycode_exit_name is not None,
        hycode_exit_name=hycode_exit_name,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('hycode_exit_name', [None, 'SIGKILL'])
async def test_a_clean_hycode_finish_is_not_recorded_as_a_failed_exit(monkeypatch, hycode_exit_name) -> None:
    # exit_status is 'env_done' (or 'sigkill' for a scoreable process exit); success is decided
    # on the canonical ENV_DONE that HyTuner trains on.
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=_real_hycode_output('ENV_DONE', hycode_exit_name)
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert output.error is None
    assert 'hytuner.termination_reason' not in span_attributes


@pytest.mark.asyncio
async def test_real_hycode_env_error_that_passes_everything_reports_env_done(monkeypatch) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=_real_hycode_output('ENV_ERROR')
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert output.error == 'agent_exit_status:env_error'
    assert span_attributes['hytuner.termination_reason'] == 'ENV_DONE'
    assert span_attributes['hytuner.cleared_termination_reason'] == 'ENV_ERROR'


# HyTuner's HY4 wrapper names only the opening token.
_HY4_PARSER = SimpleNamespace(tool_calls_start_token='<tool_calls:6124c78e>')
# What the HY4 parser hands back as plain text when the model drops the <arg_value> open tag.
_UNPARSED_TOOL_CALL = (
    '<tool_calls:6124c78e><tool_call:6124c78e>Bash<arg_key:6124c78e>command</arg_key:6124c78e>'
    'ls -la</arg_value:6124c78e></tool_call:6124c78e></tool_calls:6124c78e>'
)
# Seen in task 7160533: the reply HyCode settled on held only the tail of the block.
_UNPARSED_TOOL_CALL_TAIL = 'Inspect data structure</arg_value:6124c78e></tool_call:6124c78e></tool_calls:6124c78e>'
# A block max_tokens cut off before its closing token.
_CUT_OFF_TOOL_CALL = (
    '<tool_calls:6124c78e><tool_call:6124c78e>Bash<arg_key:6124c78e>command</arg_key:6124c78e>'
    '<arg_value:6124c78e>cd /app && python3 -c "'
)


def _hycode_stream(*events: dict) -> str:
    return '\n'.join(json.dumps(event) for event in events)


def _hycode_assistant(block: dict) -> dict:
    return {'type': 'assistant', 'message': {'role': 'assistant', 'content': [block]}}


def _hycode_result(text: str) -> dict:
    return {'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 12, 'result': text}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('stream', 'tool_parser'),
    [
        pytest.param(_hycode_stream(_hycode_result(_UNPARSED_TOOL_CALL)), _HY4_PARSER, id='result-text'),
        pytest.param(
            _hycode_stream(
                _hycode_assistant({'type': 'thinking', 'thinking': 'List the files.' + _UNPARSED_TOOL_CALL}),
                _hycode_result(''),
            ),
            _HY4_PARSER,
            id='thinking-only-final-turn',
        ),
        pytest.param(
            _hycode_stream(_hycode_result(_UNPARSED_TOOL_CALL_TAIL)), _HY4_PARSER, id='only-the-tail-of-the-block'
        ),
        pytest.param(
            _hycode_stream(_hycode_result(_UNPARSED_TOOL_CALL)),
            SimpleNamespace(tool_calls_end_token='</tool_calls:6124c78e>'),
            id='parser-names-its-closing-token',
        ),
    ],
)
async def test_hycode_ending_on_an_unparsed_tool_call_reports_format_error(monkeypatch, stream, tool_parser) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch,
        agent_output=_real_hycode_output('ENV_DONE', output=stream, result_subtype='success'),
        tool_parser=tool_parser,
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.exit_status == 'format_error'
    assert output.error == 'agent_exit_status:format_error'
    assert span_attributes['hytuner.termination_reason'] == 'FORMAT_ERROR'
    assert span_attributes['hytuner.cleared_termination_reason'] == 'ENV_DONE'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('agent_output', 'tool_parser'),
    [
        pytest.param(
            _real_hycode_output('ENV_DONE', output=_hycode_stream(_hycode_result('Done.')), result_subtype='success'),
            _HY4_PARSER,
            id='clean-finish',
        ),
        pytest.param(
            _real_hycode_output(
                'ENV_DONE',
                output=_hycode_stream(
                    _hycode_assistant({'type': 'text', 'text': _UNPARSED_TOOL_CALL}),
                    _hycode_assistant({'type': 'text', 'text': 'Done.'}),
                    _hycode_result('Done.'),
                ),
                result_subtype='success',
            ),
            _HY4_PARSER,
            id='only-an-earlier-turn-held-markup',
        ),
        pytest.param(
            _real_hycode_output(
                'ENV_DONE', output=_hycode_stream(_hycode_result(_CUT_OFF_TOOL_CALL)), result_subtype='success'
            ),
            _HY4_PARSER,
            id='block-cut-off-before-its-closing-token',
        ),
        pytest.param(
            _real_hycode_output(
                'ENV_DONE', output=_hycode_stream(_hycode_result(_UNPARSED_TOOL_CALL)), result_subtype='success'
            ),
            None,
            id='no-tool-parser',
        ),
        pytest.param(
            _real_hycode_output(
                'ENV_DONE', output=_hycode_stream(_hycode_result(_UNPARSED_TOOL_CALL)), result_subtype='success'
            ),
            SimpleNamespace(),
            id='parser-names-no-tool-call-tokens',
        ),
        pytest.param(
            _real_hycode_output(
                'ENV_DONE',
                'SIGKILL',
                output=_hycode_stream(_hycode_assistant({'type': 'text', 'text': _UNPARSED_TOOL_CALL})),
            ),
            _HY4_PARSER,
            id='process-exit-without-a-result-event',
        ),
        pytest.param(
            _real_hycode_output(
                'MAX_STEPS',
                output=_hycode_stream(_hycode_result(_UNPARSED_TOOL_CALL)),
                result_subtype='error_max_turns',
            ),
            _HY4_PARSER,
            id='agent-named-another-reason',
        ),
    ],
)
async def test_other_hycode_runs_are_not_relabelled_format_error(monkeypatch, agent_output, tool_parser) -> None:
    _, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=agent_output, tool_parser=tool_parser
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='hycode'))

    assert output.exit_status != 'format_error'
    assert 'hytuner.termination_reason' not in span_attributes


def _pausable_context(monkeypatch):
    """Share one task context with the HyTuner client's pause helpers, as a training run does."""
    from nexus.extensions.openai_clients import hytuner as hytuner_client

    monkeypatch.setattr(task_module, '_PAUSED_DEADLINE_POLL_SECONDS', 0.02)
    ctx = task_module.require_context()
    monkeypatch.setattr(task_module, 'require_context', lambda: ctx)
    monkeypatch.setattr(hytuner_client, 'require_context', lambda: ctx)
    return ctx, hytuner_client


async def _parked_for(hytuner_client, seconds: float) -> None:
    """Park like a model call held at a partial-rollout boundary."""
    paused = hytuner_client._pause_task_timeout('parked-call')
    await asyncio.sleep(seconds)
    hytuner_client._restore_task_timeout(paused, seconds, 'parked-call')


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason='the HyTuner client pauses the runner deadline through asyncio.Timeout'
)
@pytest.mark.asyncio
async def test_hycode_backstop_does_not_count_partial_rollout_pauses(monkeypatch) -> None:
    from nexus.utils.timeout import async_timeout

    _, span_attributes, _ = _cut_off_verifier_harness(monkeypatch)
    ctx, hytuner_client = _pausable_context(monkeypatch)

    async def parked_agent(**kwargs):
        await _parked_for(hytuner_client, 1.2)
        await asyncio.sleep(0.1)
        return _real_hycode_output('ENV_DONE')

    task_input = _cut_off_task_input(parked_agent, agent_name='hycode').model_copy(update={'agent_timeout': 1})
    async with async_timeout(30) as runner_timeout:
        ctx.timeout_handle = runner_timeout
        output = await run_science_agent_task(task_input)

    assert output.exit_status == 'env_done'
    assert output.error is None
    assert 'hytuner.termination_reason' not in span_attributes


@pytest.mark.asyncio
async def test_hycode_backstop_still_fires_on_active_time(monkeypatch) -> None:
    from nexus.utils.timeout import async_timeout

    _cut_off_verifier_harness(monkeypatch, verifier_result=(0.6, {'nexus_verifier': {'reward': 0.6}}))
    ctx, _ = _pausable_context(monkeypatch)

    async def slow_agent(**kwargs):
        await asyncio.sleep(5)

    task_input = _cut_off_task_input(slow_agent, agent_name='hycode').model_copy(update={'agent_timeout': 1})
    async with async_timeout(30) as runner_timeout:
        ctx.timeout_handle = runner_timeout
        output = await run_science_agent_task(task_input)

    assert output.exit_status == 'timeout'
    assert output.extra_fields['verifier_summary']['graded_cutoff_reason'] == 'TASK_TIMEOUT'


@pytest.mark.asyncio
async def test_other_agents_keep_the_wall_clock_backstop(monkeypatch) -> None:
    from nexus.utils.timeout import async_timeout

    _cut_off_verifier_harness(monkeypatch)
    ctx, hytuner_client = _pausable_context(monkeypatch)

    async def parked_agent(**kwargs):
        await _parked_for(hytuner_client, 1.2)
        return SimpleNamespace(exit_status=SimpleNamespace(value='completed'), n_iterations=5)

    task_input = _cut_off_task_input(parked_agent).model_copy(update={'agent_timeout': 1})
    async with async_timeout(30) as runner_timeout:
        ctx.timeout_handle = runner_timeout
        output = await run_science_agent_task(task_input)

    assert output.exit_status == 'timeout'


@pytest.mark.asyncio
async def test_an_agent_that_raised_reports_unhandled_error(monkeypatch) -> None:
    # Without a reason of its own the rollout reaches HyTuner as ENV_DONE, which is not masked,
    # so AGENT_FAILURE_SCORE would train as a real -300 on a rollout that never ran.
    async def blows_up(**kwargs):
        raise RuntimeError('sandbox refused the agent')

    events, span_attributes, _ = _cut_off_verifier_harness(monkeypatch, exit_status='submitted')

    output = await run_science_agent_task(_cut_off_task_input(blows_up))

    assert 'verify' not in events
    assert output.score == task_module.AGENT_FAILURE_SCORE
    assert output.exit_status == 'agent_error'
    assert span_attributes['hytuner.termination_reason'] == 'UNHANDLED_ERROR'
    assert output.test_results == {'skipped_evaluator_reason': 'AGENT_ERROR', 'exit_status': 'agent_error'}


@pytest.mark.asyncio
async def test_mini_swe_timeout_is_rescued_the_same_way_hycode_is(monkeypatch) -> None:
    from nexus.extensions.projects.r2_stem.agent import MiniSWEAgentStemExitStatus

    # Mini-SWE spells the ending as a short lowercase status, HyCode as a canonical reason.
    # Both have to end up at ENV_DONE or the harnesses would train on different rules.
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch,
        agent_output=SimpleNamespace(exit_status=MiniSWEAgentStemExitStatus.TIMEOUT, iterations=7, submission=''),
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='mini_swe_agent_stem'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert span_attributes['hytuner.termination_reason'] == 'ENV_DONE'


@pytest.mark.asyncio
async def test_a_lowercase_exit_status_is_never_matched_by_case_folding(monkeypatch) -> None:
    # 'env_error' upper-cases into _FULL_PASS_CLEARS_REASON, but it is not what any agent
    # publishes, so rewriting on it would relabel a rollout by name collision. The comparison
    # is deliberately case-sensitive; this pins that.
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch, agent_output=SimpleNamespace(exit_status='env_error', iterations=7, submission='')
    )

    output = await run_science_agent_task(_cut_off_task_input(agent, agent_name='mini_swe_agent_stem'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    assert 'hytuner.termination_reason' not in span_attributes


@pytest.mark.asyncio
async def test_full_marks_never_rewrites_a_reason_hytuner_already_trains(monkeypatch) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(monkeypatch, exit_status='truncation')

    output = await run_science_agent_task(_cut_off_task_input(agent))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == 1.0
    # TRUNCATION is not masked, so it already trains on its real score; relabelling would
    # only lose the diagnosis.
    assert span_attributes['hytuner.termination_reason'] == 'TRUNCATION'


@pytest.mark.asyncio
async def test_timed_out_verifier_rollout_scores_zero_unseen_by_default(monkeypatch) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(monkeypatch, exit_status='timeout')

    output = await run_science_agent_task(_cut_off_task_input(agent, score_truncated=False))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'release']
    assert output.score == 0.0
    assert span_attributes['hytuner.termination_reason'] == 'TASK_TIMEOUT'


@pytest.mark.asyncio
async def test_cancelled_agent_on_a_verifier_row_is_graded_when_the_row_asks(monkeypatch) -> None:
    # The outer wait_for fired, so run_agent never returned and the sandbox may still be
    # changing. The row opted into grading whatever the run had reached by then.
    async def never_returns(**kwargs):
        await asyncio.sleep(3600)

    events, span_attributes, _ = _cut_off_verifier_harness(monkeypatch, exit_status='timeout')

    task_input = _cut_off_task_input(never_returns)
    output = await run_science_agent_task(task_input.model_copy(update={'agent_timeout': 1}))

    assert 'verify' in events
    assert output.score == 1.0
    assert output.exit_status == 'timeout'
    # Full marks, so the same rule applies as to a self-reported timeout.
    assert span_attributes['hytuner.termination_reason'] == 'ENV_DONE'
    assert output.extra_fields['verifier_summary']['graded_cutoff_reason'] == 'TASK_TIMEOUT'


@pytest.mark.asyncio
async def test_cancelled_agent_keeps_the_failure_score_by_default(monkeypatch) -> None:
    async def never_returns(**kwargs):
        await asyncio.sleep(3600)

    events, _, _ = _cut_off_verifier_harness(monkeypatch, exit_status='timeout')

    task_input = _cut_off_task_input(never_returns, score_truncated=False)
    output = await run_science_agent_task(task_input.model_copy(update={'agent_timeout': 1}))

    assert 'verify' not in events
    assert output.score == task_module.AGENT_FAILURE_SCORE
    assert output.test_results == {'skipped_evaluator_reason': 'AGENT_TIMEOUT', 'exit_status': 'timeout'}


@pytest.mark.asyncio
async def test_text_judged_rollout_ignores_the_truncation_flag(monkeypatch) -> None:
    events, _, agent = _cut_off_verifier_harness(monkeypatch, exit_status='truncation')

    output = await run_science_agent_task(_cut_off_task_input(agent, evaluation_method='rmsys'))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'release']
    assert output.score == 0.0
    assert output.test_results == {'skipped_evaluator_reason': 'TRUNCATION', 'exit_status': 'truncation'}


@pytest.mark.asyncio
async def test_graded_cut_off_that_cannot_be_verified_is_a_reward_error(monkeypatch) -> None:
    events, span_attributes, agent = _cut_off_verifier_harness(
        monkeypatch,
        exit_status='truncation',
        verifier_error=RuntimeError('could not snapshot verifier artifacts: sandbox died'),
    )

    output = await run_science_agent_task(_cut_off_task_input(agent))

    assert events == ['get_runtime', 'hide_verifier', 'agent', 'verify', 'release']
    assert output.score == task_module.EVALUATION_FAILURE_SCORE
    assert span_attributes['hytuner.termination_reason'] == 'REWARD_ERROR'
    assert output.extra_fields['reward_extra_info']['extra_reward_metrics']['verifier_scoring_failed'] == 1.0
    summary = output.extra_fields['verifier_summary']
    assert summary['scoring_failed'] is True
    assert 'could not snapshot verifier artifacts' in summary['error']


@pytest.mark.asyncio
async def test_terminus_llm_captures_answer_releases_runtime_then_judges(monkeypatch) -> None:
    events = []
    runtime = SimpleNamespace()

    class FakeProvider:
        def configure(self, name, **kwargs):
            events.append(('configure', name, kwargs))

        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    class FakeDetails:
        def to_dict(self):
            return {'verdict': True}

    class FakeEvaluator:
        async def evaluate_with_details(self, *, prediction, reference, question):
            assert events[-1] == ('release', 'main')
            events.append(('judge', prediction, reference, question))
            return 1.0, FakeDetails()

    async def fake_hide_verifier(candidate_runtime):
        assert candidate_runtime is runtime
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        assert '/science/work/answer.txt' in kwargs['user_prompt']
        events.append(('agent', kwargs))
        return SimpleNamespace(exit_status=SimpleNamespace(value='completed'), n_iterations=4)

    async def fake_capture_answer_file(*, runtime, answer_path, test_timeout):
        events.append(('capture_answer', answer_path, test_timeout))
        return 'final scientific answer'

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'capture_answer_file', fake_capture_answer_file)
    monkeypatch.setattr(task_module, 'build_evaluator', lambda method: FakeEvaluator())

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Solve it.',
            agent_name='terminus2',
            ref_answer='reference',
            evaluation_method='llm_judge',
            working_dir='/science/work',
            run_agent=fake_agent,
        )
    )

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names == ['get_runtime', 'hide_verifier', 'agent', 'capture_answer', 'release', 'judge']
    assert output.answer == 'final scientific answer'
    assert output.score == 1.0


@pytest.mark.asyncio
async def test_truncation_skips_text_evaluator_and_scores_zero(monkeypatch) -> None:
    events = []
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        assert candidate_runtime is runtime
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        events.append(('agent', kwargs))
        return SimpleNamespace(exit_status=SimpleNamespace(value='truncation'), n_iterations=5)

    async def fail_capture_answer_file(**kwargs):
        pytest.fail('truncated rollouts must not capture answer files')

    async def fail_rmsys_evaluate(**kwargs):
        pytest.fail('truncated rollouts must not call reward-system evaluation')

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'capture_answer_file', fail_capture_answer_file)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fail_rmsys_evaluate)

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Solve it.',
            agent_name='terminus2',
            ref_answer='reference',
            evaluation_method='rmsys',
            working_dir='/science/work',
            run_agent=fake_agent,
        )
    )

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names == ['get_runtime', 'hide_verifier', 'agent', 'release']
    assert output.score == 0.0
    assert output.answer == ''
    assert output.exit_status == 'truncation'
    assert output.error == 'agent_exit_status:truncation'
    assert output.test_results == {'skipped_evaluator_reason': 'TRUNCATION', 'exit_status': 'truncation'}


def _hycode_context(events, runtime):
    class FakeProvider:
        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    return FakeContext()


@pytest.mark.asyncio
async def test_hycode_env_done_sends_the_stream_json_answer_to_reward_system(monkeypatch) -> None:
    events = []
    runtime = SimpleNamespace()
    judged = {}

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        events.append(('agent', kwargs))
        return SimpleNamespace(
            output='\n'.join(
                [
                    '{"type":"assistant","message":{"content":"inspecting the panel"}}',
                    '{"type":"result","subtype":"success","is_error":false,"result":"GAPDH","num_turns":7}',
                ]
            ),
            termination_reason='ENV_DONE',
            num_turns=7,
        )

    async def fake_rmsys_evaluate(**kwargs):
        events.append('judge')
        judged.update(kwargs)
        return 1.0

    monkeypatch.setattr(task_module, 'require_context', lambda: _hycode_context(events, runtime))
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fake_rmsys_evaluate)

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Name the control gene.',
            agent_name='hycode',
            ref_answer='GAPDH',
            evaluation_method='rmsys',
            working_dir='/app',
            reward_system_router_id='ScienceHYA3B--6',
            run_agent=fake_agent,
        )
    )

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names == ['get_runtime', 'hide_verifier', 'agent', 'release', 'judge']
    assert judged['model_answer'] == 'GAPDH'
    # The harness-specific contract reaches the agent but must not reach the judge, or the
    # reward system would score HyCode rollouts against a different question than Mini-SWE.
    assert 'every response must include at least one tool call' in events[2][1]['user_prompt']
    assert judged['question'] == 'Name the control gene.'
    assert judged['router_id'] == 'ScienceHYA3B--6'
    assert output.answer == 'GAPDH'
    assert output.extra_fields == {}
    assert output.score == 1.0
    assert output.exit_status == 'ENV_DONE'
    assert output.iterations == 7


@pytest.mark.asyncio
async def test_hycode_max_steps_skips_the_evaluator_and_scores_zero(monkeypatch) -> None:
    events = []
    runtime = SimpleNamespace()

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        events.append(('agent', kwargs))
        return SimpleNamespace(
            output='{"type":"result","subtype":"error_max_turns","is_error":true,"num_turns":200}',
            termination_reason='MAX_STEPS',
        )

    async def fail_rmsys_evaluate(**kwargs):
        pytest.fail('cut-off rollouts must not call reward-system evaluation')

    monkeypatch.setattr(task_module, 'require_context', lambda: _hycode_context(events, runtime))
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fail_rmsys_evaluate)

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Name the control gene.',
            agent_name='hycode',
            ref_answer='GAPDH',
            evaluation_method='rmsys',
            working_dir='/app',
            run_agent=fake_agent,
        )
    )

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names == ['get_runtime', 'hide_verifier', 'agent', 'release']
    # Running out of turns is judged like any other stop; this run scores 0 only because its
    # closing event carries no text to judge, and it stays in the batch as a usable negative
    # rather than dropping to EVALUATION_FAILURE_SCORE.
    assert output.score == 0.0
    assert output.answer == ''
    assert output.exit_status == 'MAX_STEPS'
    assert output.test_results == {'skipped_evaluator_reason': 'EMPTY_ANSWER', 'exit_status': 'MAX_STEPS'}


@pytest.mark.asyncio
async def test_mini_swe_empty_answer_keeps_the_evaluation_failure_score(monkeypatch) -> None:
    from nexus.extensions.projects.r2_stem.agent import MiniSWEAgentStemExitStatus

    events = []
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        events.append(('agent', kwargs))
        return SimpleNamespace(
            exit_status=MiniSWEAgentStemExitStatus.MAX_ITERATIONS,
            iterations=50,
            submission='',
        )

    async def fail_rmsys_evaluate(**kwargs):
        pytest.fail('an empty answer must not reach the reward system')

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fail_rmsys_evaluate)

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Name the control gene.',
            agent_name='mini_swe_agent_stem',
            ref_answer='GAPDH',
            evaluation_method='rmsys',
            working_dir='/science/work',
            run_agent=fake_agent,
        )
    )

    # The 0-instead-of-EVALUATION_FAILURE_SCORE rule is scoped to hycode, so running out of
    # iterations with nothing to judge still scores the way mini-swe was tuned against.
    assert output.score == task_module.EVALUATION_FAILURE_SCORE
    assert output.answer == ''
    assert output.exit_status == 'max_iterations'
    assert output.judge_error == 'rmsys_error:rmsys evaluation requires a non-empty answer'
    assert output.test_results is None


@pytest.mark.asyncio
async def test_output_length_exceeded_skips_evaluator_and_reports_its_own_reason(monkeypatch) -> None:
    from nexus.extensions.projects.r2_stem.agent import MiniSWEAgentStemExitStatus

    events = []
    span_attributes: dict[str, object] = {}
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        events.append(('agent', kwargs))
        return SimpleNamespace(
            exit_status=MiniSWEAgentStemExitStatus.OUTPUT_LENGTH_EXCEEDED,
            iterations=3,
            submission='half of a sent',
        )

    async def fail_rmsys_evaluate(**kwargs):
        pytest.fail('rollouts cut off by max_tokens must not call reward-system evaluation')

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fail_rmsys_evaluate)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(task_module, 'update_span', lambda span_id, attributes: span_attributes.update(attributes))

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Solve it.',
            agent_name='mini_swe_agent_stem',
            ref_answer='reference',
            evaluation_method='rmsys',
            working_dir='/science/work',
            run_agent=fake_agent,
        )
    )

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names == ['get_runtime', 'hide_verifier', 'agent', 'release']
    assert output.score == 0.0
    assert output.answer == ''
    assert output.exit_status == 'output_length_exceeded'
    # The training side must be able to tell this apart from a context-window cutoff.
    assert span_attributes == {'hytuner.termination_reason': 'OUTPUT_LENGTH_EXCEEDED'}
    assert output.test_results == {
        'skipped_evaluator_reason': 'OUTPUT_LENGTH_EXCEEDED',
        'exit_status': 'output_length_exceeded',
    }


@pytest.mark.asyncio
async def test_agent_that_stops_on_its_own_budget_scores_zero_and_reports_a_timeout(monkeypatch) -> None:
    """An agent that respects its wall-clock budget returns, so the task can explain it.

    This is the path the outer ``asyncio.wait_for`` used to take instead, and cancelling
    the agent left HyTuner with no termination reason at all — it read the rollout as
    ENV_DONE and scored it as a scoring failure.
    """
    from nexus.extensions.projects.r2_stem.agent import MiniSWEAgentStemExitStatus

    events = []
    span_attributes: dict[str, object] = {}
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def fake_agent(**kwargs):
        events.append(('agent', kwargs))
        return SimpleNamespace(exit_status=MiniSWEAgentStemExitStatus.TIMEOUT, iterations=7, submission='')

    async def fail_rmsys_evaluate(**kwargs):
        pytest.fail('rollouts that ran out of time must not call reward-system evaluation')

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'rmsys_evaluate', fail_rmsys_evaluate)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(task_module, 'update_span', lambda span_id, attributes: span_attributes.update(attributes))

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Solve it.',
            agent_name='mini_swe_agent_stem',
            ref_answer='reference',
            evaluation_method='rmsys',
            agent_timeout=18000,
            run_agent=fake_agent,
        )
    )

    agent_kwargs = next(event[1] for event in events if event[0] == 'agent')
    # The agent has to know the budget to be able to stop inside it.
    assert agent_kwargs['wall_time_limit_seconds'] == 17400.0
    assert output.score == 0.0
    assert output.exit_status == 'timeout'
    assert span_attributes == {'hytuner.termination_reason': 'TASK_TIMEOUT'}
    assert output.test_results == {'skipped_evaluator_reason': 'TASK_TIMEOUT', 'exit_status': 'timeout'}


@pytest.mark.asyncio
async def test_agent_timeout_keeps_failure_score_and_skips_sandbox_evaluator(monkeypatch) -> None:
    events = []
    span_attributes: dict[str, object] = {}
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def slow_agent(**kwargs):
        await asyncio.sleep(60)

    async def fail_unit_cases_evaluate(**kwargs):
        pytest.fail('timed-out rollouts must not run the sandbox evaluator')

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'unit_cases_evaluate', fail_unit_cases_evaluate)
    monkeypatch.setattr(task_module, 'get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(task_module, 'update_span', lambda span_id, attributes: span_attributes.update(attributes))

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Solve it.',
            agent_name='mini_swe_agent_stem',
            evaluation_method='unit_cases',
            task_dir='/host/task',
            agent_timeout=0,
            run_agent=slow_agent,
        )
    )

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        'get_runtime',
        'hide_verifier',
        'release',
    ]
    # Reaching this branch means the agent's own budget failed to contain the run, so the
    # failure score stays: it is a misconfiguration and should stay visible in failed_ratio.
    assert output.score == task_module.AGENT_FAILURE_SCORE
    assert output.exit_status == 'timeout'
    # The cancelled agent published nothing, so without this the rollout reads as ENV_DONE.
    assert span_attributes == {'hytuner.termination_reason': 'TASK_TIMEOUT'}
    assert output.test_results == {'skipped_evaluator_reason': 'AGENT_TIMEOUT', 'exit_status': 'timeout'}
    assert output.agent_error is not None and 'timed out' in output.agent_error


@pytest.mark.asyncio
async def test_agent_error_keeps_failure_score_and_skips_host_evaluator(monkeypatch) -> None:
    events = []
    runtime = SimpleNamespace()

    class FakeProvider:
        async def release(self, name):
            events.append(('release', name))

    class FakeContext:
        agent_name = None
        runtime_provider = FakeProvider()

        async def get_runtime(self, name):
            events.append(('get_runtime', name))
            return runtime

        def get_runtime_id(self, target):
            return None

    async def fake_hide_verifier(candidate_runtime):
        events.append('hide_verifier')

    async def failing_agent(**kwargs):
        raise RuntimeError('sandbox connection lost')

    def fail_build_evaluator(method):
        pytest.fail('failed rollouts must not build a judge')

    monkeypatch.setattr(task_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(task_module, 'hide_verifier_from_agent', fake_hide_verifier)
    monkeypatch.setattr(task_module, 'build_evaluator', fail_build_evaluator)

    output = await run_science_agent_task(
        ScienceAgentTaskInput(
            question='Solve it.',
            agent_name='mini_swe_agent_stem',
            ref_answer='reference',
            evaluation_method='llm_judge',
            run_agent=failing_agent,
        )
    )

    assert [event if isinstance(event, str) else event[0] for event in events] == [
        'get_runtime',
        'hide_verifier',
        'release',
    ]
    assert output.score == task_module.AGENT_FAILURE_SCORE
    assert output.exit_status == 'agent_error'
    assert output.test_results == {'skipped_evaluator_reason': 'AGENT_ERROR', 'exit_status': 'agent_error'}
    assert output.agent_error_type == 'RuntimeError'


def test_ref_answer_prefers_task_config(tmp_path) -> None:
    (tmp_path / 'ref_answer.txt').write_text('file reference', encoding='utf-8')

    ref_answer = load_ref_answer(reference='config reference', task_dir=str(tmp_path))

    assert ref_answer == 'config reference'


def test_ref_answer_loads_from_task_dir(tmp_path) -> None:
    ref_answer_path = tmp_path / 'ref_answer.txt'
    ref_answer_path.write_text('  host-side reference\n', encoding='utf-8')

    ref_answer = load_ref_answer(reference=None, task_dir=str(tmp_path))

    assert ref_answer == 'host-side reference'


def test_ref_answer_normalizes_structured_config() -> None:
    ref_answer = load_ref_answer(reference={'answer': 'value'}, task_dir='')

    assert ref_answer == '{"answer": "value"}'


def test_ref_answer_is_required(tmp_path) -> None:
    with pytest.raises(ValueError, match='task_config.task_dir/ref_answer.txt'):
        load_ref_answer(reference=None, task_dir=str(tmp_path))
