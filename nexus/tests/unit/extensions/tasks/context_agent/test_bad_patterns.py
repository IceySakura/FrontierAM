"""Exercise real detectors through task scoring, without external Judges or runtimes."""

from __future__ import annotations

import copy

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from pydantic import ValidationError

from nexus.core import ExecutionContext
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.tasks.context_agent import bad_patterns, general_task
from nexus.extensions.tasks.context_agent.bad_patterns import ToolCallBadPatternPolicy, apply_bad_pattern_score
from nexus.extensions.tasks.context_agent.enterprisebench_native import task as native
from nexus.runtime.providers.local import LocalRuntimeProvider


HY_END = '</think:6124c78e>'
POLICY: dict[str, Any] = {
    'id': 'test-think',
    'penalty': 0.5,
    'reward_threshold': 0.5,
    'patterns': {
        'think_protocol_leak_hy': {'enabled': True, 'penalize': True},
        'think_protocol_leak_all_family': {'enabled': True, 'penalize': False},
    },
}


def agent_output(text='The correct answer is (D)' + HY_END, **kwargs):
    return SimpleNamespace(
        main_trajectory=[{'role': 'user', 'content': 'Choose an option.'}, {'role': 'assistant', 'content': text}],
        tool_schemas=[],
        final_answer=text,
        exit_status=kwargs.pop('exit_status', TerminationReason.SUCCESS),
        **kwargs,
    )


@pytest.mark.parametrize(('score', 'expected'), [(1.0, 0.5), (0.5, 0.0), (0.4, 0.4), (0.0, 0.0)])
@pytest.mark.parametrize('as_dict', [False, True])
def test_penalty_threshold_floor_and_preserved_metrics(score, expected, as_dict):
    agent = agent_output()
    extra = {
        'reward_extra_info': {'tool_name_resolution_error_count': 2, 'extra_reward_metrics': {'answer_score': 1.0}}
    }
    original = copy.deepcopy(extra)
    result, fields = apply_bad_pattern_score(
        score, vars(agent) if as_dict else agent, ToolCallBadPatternPolicy(**POLICY), extra_fields=extra
    )
    assert result == expected
    assert extra == original
    assert fields['reward_extra_info']['tool_name_resolution_error_count'] == 2
    assert fields['reward_extra_info']['extra_reward_metrics']['answer_score'] == 1.0
    assert fields['tool_call_bad_pattern']['original_score'] == score
    assert fields['tool_call_bad_pattern']['penalty'] == pytest.approx(score - expected)


@pytest.mark.parametrize('text', ['Ordinary answer.', 'Quoted </think> example.'])
def test_clean_and_monitor_only_rules_keep_reward(text):
    score, extra = apply_bad_pattern_score(1.0, agent_output(text), ToolCallBadPatternPolicy(**POLICY))
    assert score == 1.0
    assert extra['tool_call_bad_pattern']['penalized'] is False
    metrics = extra['reward_extra_info']['extra_reward_metrics']
    assert metrics['tool_call_bad_pattern/think_protocol_leak_hy/hit'] == 0.0
    assert metrics['tool_call_bad_pattern/think_protocol_leak_all_family/hit'] == float('</think>' in text)


def test_disabled_policy_does_not_run_detection(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('disabled policy must not scan')

    monkeypatch.setattr(bad_patterns, 'detect_tool_call_bad_patterns', fail)
    assert apply_bad_pattern_score(1.0, agent_output(), None) == (1.0, {})


def test_rule_disabled_and_multiple_hits_do_not_stack():
    policy = ToolCallBadPatternPolicy.model_validate(
        {
            **POLICY,
            'max_hits': 1,
            'patterns': {'think_protocol_leak_hy': {}, 'think_protocol_leak_all_family': {'enabled': False}},
        }
    )
    agent = agent_output()
    agent.main_trajectory += [{'role': 'assistant', 'content': HY_END + '</think>'}]
    score, extra = apply_bad_pattern_score(1.0, agent, policy)
    detection = extra['tool_call_bad_pattern']
    assert score == 0.5
    assert detection['issue_counts'] == {'think_protocol_leak_hy': 2}
    assert len(detection['hits']) == 1
    assert detection['hits_truncated'] is True


def test_truncated_final_turn_protocol_leak_is_monitor_only():
    score, extra = apply_bad_pattern_score(
        1.0, agent_output(exit_status=TerminationReason.OUTPUT_LENGTH_EXCEEDED), ToolCallBadPatternPolicy(**POLICY)
    )
    assert score == 1.0
    assert extra['tool_call_bad_pattern']['issue_count'] == 0
    assert extra['tool_call_bad_pattern']['nonissue_evidence_count'] > 0


@pytest.mark.parametrize('field', ['reasoning_content', 'tool_calls'])
def test_reasoning_and_tool_arguments_are_scanned(field):
    agent = agent_output('Done.')
    agent.main_trajectory[-1][field] = (
        HY_END
        if field == 'reasoning_content'
        else [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command":"' + HY_END + '"}'}}
        ]
    )
    score, extra = apply_bad_pattern_score(1.0, agent, ToolCallBadPatternPolicy(**POLICY))
    assert score == 0.5
    assert extra['tool_call_bad_pattern']['issue_counts']['think_protocol_leak_hy'] == 1


def test_actual_tool_schema_is_used():
    agent = agent_output('')
    agent.tool_schemas = [
        {
            'type': 'function',
            'function': {
                'name': 'read_lines',
                'parameters': {'type': 'object', 'properties': {'start': {'type': 'integer'}}},
            },
        }
    ]
    agent.main_trajectory[-1]['tool_calls'] = [
        {'id': 'c1', 'type': 'function', 'function': {'name': 'read_lines', 'arguments': '{"start":"wrong"}'}}
    ]
    policy = ToolCallBadPatternPolicy.model_validate({'id': 'args', 'penalty': 0.5, 'patterns': {'tool_call_args_invalid': {}}})
    score, extra = apply_bad_pattern_score(1.0, agent, policy)
    assert score == 0.5
    assert extra['tool_call_bad_pattern']['schema_registry_tool_count'] == 1


def test_detection_failure_keeps_score_and_records_error(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('test detector failure')

    monkeypatch.setattr(bad_patterns, 'detect_tool_call_bad_patterns', fail)
    score, extra = apply_bad_pattern_score(1.0, agent_output(), ToolCallBadPatternPolicy(**POLICY))
    assert score == 1.0
    assert extra['tool_call_bad_pattern']['error_type'] == 'RuntimeError'
    assert 'reward_extra_info' not in extra


def test_missing_trajectory_is_not_reported_as_clean():
    score, extra = apply_bad_pattern_score(1.0, SimpleNamespace(), ToolCallBadPatternPolicy(**POLICY))
    assert score == 1.0
    assert 'main_trajectory' in extra['tool_call_bad_pattern']['error']


@pytest.mark.parametrize(
    ('input_cls', 'required'),
    [
        (general_task.ContextAgentGeneralTaskInput, {'context_question': 'Read.'}),
        (native.EnterpriseBenchNativeTaskInput, {}),
    ],
)
def test_task_policy_is_opt_in_and_validated(input_cls, required):
    assert input_cls(run_agent=AsyncMock(), **required).tool_call_bad_pattern_policy is None
    configured = input_cls(run_agent=AsyncMock(), tool_call_bad_pattern_policy=ToolCallBadPatternPolicy.model_validate(POLICY), **required)
    assert configured.tool_call_bad_pattern_policy.penalty == 0.5
    with pytest.raises(ValidationError, match='unknown bad patterns'):
        input_cls(
            run_agent=AsyncMock(), tool_call_bad_pattern_policy={'id': 'bad', 'patterns': {'typo': {}}}, **required
        )


@pytest.mark.parametrize(('policy', 'expected'), [(None, 1.0), (POLICY, 0.5)])
async def test_general_task_preserves_judge_evidence(monkeypatch, policy, expected):
    monkeypatch.setattr(general_task, 'resolve_judge_llm', lambda ctx: None)
    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await general_task.run_context_agent_general_task(
            general_task.ContextAgentGeneralTaskInput(
                run_agent=AsyncMock(return_value=agent_output()),
                context_question='Choose an option.',
                dataset_name='LongBenchV2',
                answer='D',
                tool_call_bad_pattern_policy=policy,
            )
        )
    assert output.score == expected
    assert output.answer_score == 1.0
    assert output.answer_judge['score'] == 1.0
    assert output.termination_reason == 'ENV_DONE'


async def test_general_task_skipped_judge_still_records_detection(monkeypatch):
    evaluator = AsyncMock()
    monkeypatch.setattr(general_task, 'evaluate', evaluator)
    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await general_task.run_context_agent_general_task(
            general_task.ContextAgentGeneralTaskInput(
                run_agent=AsyncMock(return_value=agent_output(exit_status=TerminationReason.OUTPUT_LENGTH_EXCEEDED)),
                context_question='Choose an option.',
                dataset_name='LongBenchV2',
                answer='D',
                tool_call_bad_pattern_policy=ToolCallBadPatternPolicy.model_validate(POLICY),
            )
        )
    assert output.score == 0.0
    assert output.evaluation_skipped
    assert output.termination_reason == 'TRUNCATION'
    assert output.extra_fields['tool_call_bad_pattern']['nonissue_evidence_count'] > 0
    evaluator.assert_not_awaited()


@pytest.mark.parametrize(('policy', 'expected'), [(None, 1.0), (POLICY, 0.5)])
async def test_native_enterprise_keeps_verifier_reward_and_penalizes_score(monkeypatch, tmp_path, policy, expected):
    runtime = SimpleNamespace(run_command=AsyncMock(), invoke=AsyncMock(return_value={'uid': 1000, 'gid': 1000}))
    ctx = SimpleNamespace(
        runtime_provider=object(),
        task_config={},
        task_files=None,
        run_id='test',
        tool_state={},
        get_runtime=AsyncMock(return_value=runtime),
    )
    monkeypatch.setattr(native, 'require_context', lambda: ctx)
    monkeypatch.setattr(native, '_resolve_task_dir', AsyncMock(return_value=(tmp_path, None)))
    monkeypatch.setattr(
        native, '_build_enterprise_task_spec', lambda *args: SimpleNamespace(name='test', instruction='Do it.')
    )
    for name in ['_apply_environment_uploads', '_boot_image_services', '_persist_agent_deliverable']:
        monkeypatch.setattr(native, name, AsyncMock())
    monkeypatch.setattr(native, 'get_current_span_id', lambda: None)
    monkeypatch.setattr(native, 'persist_workspace_evidence', AsyncMock(return_value={}))
    records = Mock(return_value={})
    monkeypatch.setattr(native, 'persist_run_records', records)
    monkeypatch.setattr(
        native,
        '_run_enterprise_verifier',
        AsyncMock(
            return_value=(
                {'reward': 1.0},
                {},
                '',
                {'reward': {'score': 1.0, 'criteria': [{'name': 'one', 'value': 1.0}]}},
                [],
                '',
                None,
            )
        ),
    )
    output = await native.enterprisebench_native_task(
        native.EnterpriseBenchNativeTaskInput(
            run_agent=AsyncMock(return_value=agent_output()),
            artifact_root=str(tmp_path),
            workdir='/workdir',
            judge_config={'criteria': [{'name': 'one', 'description': 'Task completed.'}]},
            tool_call_bad_pattern_policy=policy,
        )
    )
    assert output.score == expected
    assert output.reward == 1.0
    assert output.rewards['reward'] == 1.0
    assert output.raw_reward == 1.0
    assert output.sample_accuracy == 1.0
    assert records.call_args.kwargs['result_summary']['score'] == expected
    assert records.call_args.kwargs['result_summary']['raw_reward'] == 1.0
    assert output.criteria_passed == 1
    assert output.termination_reason == 'ENV_DONE'


@pytest.mark.parametrize('zero_tool_errors', [False, True])
async def test_general_bad_patterns_preserve_agentif_metrics_and_tool_zeroing(monkeypatch, zero_tool_errors):
    monkeypatch.setattr(general_task, 'resolve_judge_llm', lambda ctx: None)
    monkeypatch.setattr(general_task, 'score_constraints', AsyncMock(return_value={'binary': 1, 'ok': True}))
    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await general_task.run_context_agent_general_task(
            general_task.ContextAgentGeneralTaskInput(
                run_agent=AsyncMock(return_value=agent_output(tool_name_resolution_error_count=1)),
                context_question='Choose an option.',
                dataset_name='LongBenchV2',
                answer='D',
                constraint_rubrics=[{'description': 'Follow instructions.'}],
                zero_score_on_tool_name_resolution_error=zero_tool_errors,
                tool_call_bad_pattern_policy=ToolCallBadPatternPolicy.model_validate(POLICY),
            )
        )
    assert output.score == (0.0 if zero_tool_errors else 0.5)
    assert output.answer_score == output.constraint_score == 1.0
    info = output.extra_fields['reward_extra_info']
    assert info['tool_name_resolution_zeroed'] == float(zero_tool_errors)
    assert info['extra_reward_metrics']['answer_score'] == 1.0
    assert info['extra_reward_metrics']['constraint_score'] == 1.0
    assert info['extra_reward_metrics']['tool_call_bad_pattern/think_protocol_leak_hy/hit'] == 1.0


def test_hytuner_binary_reward_transform_preserves_bad_pattern_penalty():
    from nexus.runner.hytuner import HyTunerRewardPolicy, _apply_hytuner_reward_policy

    score, extra = apply_bad_pattern_score(1.0, agent_output(), ToolCallBadPatternPolicy(**POLICY))
    transformed = _apply_hytuner_reward_policy(score, HyTunerRewardPolicy(mode='binary', threshold=0.9), extra)
    assert transformed == 0.5
