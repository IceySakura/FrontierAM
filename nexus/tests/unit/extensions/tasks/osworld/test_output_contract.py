"""The stage contract is task-local and preserves raw execution evidence."""

import pytest

from pydantic import ValidationError

from nexus.extensions.tasks.os_world.outputs import (
    EVALUABLE_AGENT_STATUSES,
    AgentExitStatus,
    OSWorldAgentOutput,
    OSWorldJudgeOutput,
    normalize_agent_output,
    task_status,
)


@pytest.mark.parametrize('status', list(AgentExitStatus))
def test_agent_semantic_code_matrix(status):
    code = 0 if status in EVALUABLE_AGENT_STATUSES else -1
    value = OSWorldAgentOutput(exit_status=status, exit_code=code, error='fixture error', process_exit_code=137)
    assert value.model_dump(mode='json')['exit_status'] == status.value
    assert value.process_exit_code == 137
    with pytest.raises(ValidationError):
        OSWorldAgentOutput(exit_status=status, exit_code=-1 if code == 0 else 0, error='fixture error')


def test_task_stage_priority_and_infeasible_conclusion():
    agent = OSWorldAgentOutput(
        exit_status='finished', exit_code=0, conclusion='infeasible', cc_return_code=0, main_trajectory=[{'id': 1}]
    )
    judge = OSWorldJudgeOutput(exit_status='finished', exit_code=0, result={'score': 1})
    assert task_status(agent, judge) == 'success'
    assert agent.model_dump()['main_trajectory'] == [{'id': 1}]
    assert task_status(agent, None) == 'judge_error'
    bad_judge = OSWorldJudgeOutput(exit_status='judge_error', exit_code=-1, error='OCR failed')
    assert task_status(agent, bad_judge) == 'judge_error'
    failed = OSWorldAgentOutput(exit_status='inference_error', exit_code=-1, error='Request timed out')
    assert task_status(failed, bad_judge) == 'agent_error'


def test_invalid_results_require_error_and_judge_codes_are_consistent():
    with pytest.raises(ValidationError):
        OSWorldAgentOutput(exit_status='unknown_error', exit_code=-1)
    with pytest.raises(ValidationError):
        OSWorldJudgeOutput(exit_status='finished', exit_code=-1, error='failure')
    with pytest.raises(ValidationError):
        OSWorldJudgeOutput(exit_status='judge_error', exit_code=-1)


@pytest.mark.parametrize(
    ('raw', 'expected', 'code'),
    [
        ({'success': True, 'exit_code': 1, 'exit_status': 'max_iterations'}, 'max_iterations', 0),
        (
            {'success': True, 'exit_code': -1, 'exit_status': 'max_iterations', 'termination_reason': 'cc_timeout'},
            'agent_timeout',
            0,
        ),
        ({'success': True, 'exit_code': -1, 'exit_status': 'budget_exhausted'}, 'agent_timeout', 0),
        ({'exit_status': 'infeasible'}, 'finished', 0),
        ({'success': False, 'stderr': 'Failed to install Claude Code: npm unavailable'}, 'environment_error', -1),
        ({'success': False, 'stderr': 'Claude Code exited without writing an exit code'}, 'runtime_error_by_env', -1),
        ({'exit_status': 'incomplete', 'terminal_finish_reason': 'max_tokens'}, 'output_length_exceeded', 0),
        ({'success': False}, 'unknown_error', -1),
    ],
)
def test_stage_classifier(raw, expected, code):
    output = normalize_agent_output(raw)
    assert output.exit_status == expected
    assert output.exit_code == code
    assert output.process_exit_code == raw.get('exit_code')


def test_parent_error_is_not_overwritten_by_other_session():
    import json

    raw = '\n'.join(
        json.dumps(event)
        for event in [
            {'type': 'system', 'subtype': 'init', 'session_id': 'root'},
            {'type': 'result', 'session_id': 'root', 'is_error': True, 'result': 'Request timed out'},
            {'type': 'result', 'session_id': 'child', 'parent_tool_use_id': 'tool', 'result': 'TASK_DONE'},
            {'type': 'result', 'session_id': 'other', 'result': 'TASK_DONE'},
        ]
    )
    output = normalize_agent_output({'success': False, 'exit_code': 1, 'output': raw}, harness='cc')
    assert output.exit_status == 'inference_error'
    assert output.error == 'Request timed out'
    assert output.exit_code == -1


def test_model_prose_is_not_an_error_signal():
    result = normalize_agent_output({
        'exit_status': 'finished',
        'success': True,
        'exit_code': 0,
        'final_answer': 'Earlier request timed out; context window exceeded is an example.',
    })
    assert result.exit_status == 'finished'


def test_processor_summary_keeps_metrics_without_inventing_disabled_trackers():
    from nexus.extensions.tasks.os_world.outputs import summarize_processors

    assert summarize_processors({'chat': {'messages': ['large']}}) == {}
    counters = {'main': {'completion_tokens': 2**54 + 1, 'total_tokens_per_trajectory': [123, 456]}}
    anomaly = {
        'main': {
            'total_requests': 7,
            'total_steps': 5,
            'error_requests': 2,
            'anomaly_counts': {'HALLUCINATED_TOOL': 2},
            'requests': [{'text': 'trace'}],
        },
        'aux': {'total_requests': 1, 'total_steps': 1, 'requests': []},
    }
    result = summarize_processors({'token_counter': counters, 'bridge_toolcall_anomaly': anomaly})
    assert result['token_counter'] == counters
    assert result['bridge_toolcall_anomaly'] == {
        client: {k: v for k, v in entry.items() if k != 'requests'} for client, entry in anomaly.items()
    }
    assert 'requests' in anomaly['main']
    assert 'execution_timeout_tracker' not in result


@pytest.mark.parametrize('atif_enabled', [False, True])
def test_processor_summary_preserves_bridge_trajectory_payloads(atif_enabled):
    from nexus.extensions.tasks.os_world.outputs import summarize_processors

    chat = {'trajectories': [[{'role': 'assistant', 'content': '完整对话' * 10000}]], 'tools': []}
    atif = {'atif': {'steps': [{'role': 'agent', 'message': 'full trace'}], 'extra': {}}}
    source = {'_default_chat_trace': chat, '_default_trace': {'path': 'incremental.jsonl'}}
    if atif_enabled:
        source['_default_atif_trace'] = atif
    result = summarize_processors(source)
    assert result['_default_chat_trace'] is chat
    assert '_default_trace' not in result
    if atif_enabled:
        assert result['_default_atif_trace'] is atif
    else:
        assert '_default_atif_trace' not in result
    assert source['_default_trace'] == {'path': 'incremental.jsonl'}


CC_OUTPUT_LIMIT_ERROR = (
    "API Error: Claude's response exceeded the 64000 output token maximum. "
    'To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.'
)


@pytest.mark.parametrize('limit', ['64000', '32000', '64,000'])
def test_cc_root_output_limit_is_evaluable_and_preserves_cli_code(limit):
    import json

    error = CC_OUTPUT_LIMIT_ERROR.replace('64000', limit)
    raw = '\n'.join(
        json.dumps(event)
        for event in [
            {'type': 'system', 'subtype': 'init', 'session_id': 'root'},
            {'type': 'assistant', 'session_id': 'root', 'message': {'content': 'partial answer'}},
            {
                'type': 'result',
                'session_id': 'root',
                'subtype': 'error_during_execution',
                'is_error': True,
                'result': error,
                'terminal_reason': 'completed',
            },
        ]
    )
    output = normalize_agent_output(
        {
            'success': False,
            'exit_status': 'tool_error',
            'exit_code': 1,
            'cc_return_code': 1,
            'output': raw,
        },
        harness='cc',
    )
    assert output.exit_status == 'output_length_exceeded'
    assert output.exit_code == 0
    assert output.process_exit_code == output.cc_return_code == 1
    assert output.error == error
    assert output.conclusion is None
    assert output.model_dump()['output'] == raw


@pytest.mark.parametrize('source', ['assistant', 'child', 'other', 'recovered', 'cleanup', 'unrelated_error'])
def test_cc_output_limit_requires_current_root_error(source):
    import json

    root_error = source in {'cleanup', 'unrelated_error'}
    events = [
        {'type': 'system', 'subtype': 'init', 'session_id': 'root'},
        {
            'type': 'result' if source != 'assistant' else 'assistant',
            'session_id': 'child' if source == 'child' else 'other' if source == 'other' else 'root',
            **({'parent_tool_use_id': 'tool'} if source == 'child' else {}),
            'is_error': source != 'assistant',
            'result': CC_OUTPUT_LIMIT_ERROR,
        },
    ]
    if source != 'cleanup':
        events.append({
            'type': 'result',
            'session_id': 'root',
            'is_error': root_error,
            'result': 'API Error: connection refused' if root_error else 'TASK_DONE',
        })
    output = normalize_agent_output(
        {
            'success': not root_error,
            'exit_status': 'tool_error' if root_error else 'finished',
            'exit_code': 1 if root_error else 0,
            'output': '\n'.join(map(json.dumps, events)),
            **({'failure_stage': 'cleanup', 'stderr': 'termination failed'} if source == 'cleanup' else {}),
        },
        harness='cc',
    )
    assert output.exit_status == (
        'runtime_error_by_env' if source == 'cleanup' else 'inference_error' if root_error else 'finished'
    )


def test_codex_api_error_takes_priority_over_generic_infrastructure_label():
    output = normalize_agent_output({
        'success': False,
        'exit_code': 1,
        'failure_kind': 'infrastructure',
        'terminal_api_error': 'maximum context length exceeded',
    })
    assert output.exit_status == 'context_length_exceeded'
    assert output.exit_code == 0
    assert output.process_exit_code == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('agent_name', ['cua_gpt_agent', 'cua_anthropic_agent'])
@pytest.mark.parametrize('outcome', ['finished', 'length', 'context', 'inference', 'retry', 'unrelated'])
async def test_bare_observation_preserves_client_and_classifies_actual_provider_outcome(agent_name, outcome):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.core.openai_client import ContextLengthExceededError
    from nexus.extensions.tasks.os_world.outputs import run_task_agent

    error = ContextLengthExceededError('context full') if outcome == 'context' else ConnectionError('upstream closed')
    client = SimpleNamespace(
        create_completion=AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(finish_reason='length' if outcome == 'length' else 'stop')]
            )
        )
    )
    if outcome in {'context', 'inference', 'retry', 'unrelated'}:
        client.create_completion.side_effect = error
    ctx = ExecutionContext(task_name='os_world', agent_name=agent_name)
    ctx.openai_clients = {'main': client}
    originals = ctx.openai_clients

    async def agent(**kwargs):
        try:
            await ctx.get_openai_client().create_completion(messages=[], model='fixture')
        except Exception:
            if outcome == 'retry':
                client.create_completion.side_effect = None
                await ctx.get_openai_client().create_completion(messages=[], model='fixture')
            elif outcome == 'unrelated':
                raise ValueError('later tool error') from None
            else:
                raise
        ctx.tool_state['os_world_last_action'] = {'action_type': 'DONE'}
        return {'exit_status': 'finished', 'trajectory': [{'event': 'retained'}]}

    async with ctx:
        if outcome in {'inference', 'unrelated'}:
            with pytest.raises(ConnectionError if outcome == 'inference' else ValueError):
                await run_task_agent(agent)
            if outcome == 'inference':
                assert ctx.tool_state['osworld_agent_output'].exit_status == 'inference_error'
            else:
                assert 'osworld_agent_output' not in ctx.tool_state
        else:
            output = await run_task_agent(agent)
            assert output.exit_status == {
                'length': 'output_length_exceeded',
                'context': 'context_length_exceeded',
            }.get(outcome, 'finished')
            assert output.exit_code == 0
            if outcome == 'length':
                assert 'os_world_last_action' not in ctx.tool_state
            if outcome != 'context':
                assert output.model_dump()['trajectory'] == [{'event': 'retained'}]
        assert ctx.openai_clients is originals
        assert ctx.get_openai_client() is client


@pytest.mark.asyncio
async def test_output_hook_is_explicit_and_task_scoped():
    from types import SimpleNamespace

    from nexus.extensions.tasks.os_world.outputs import OSWorldOutputContractHook

    hook = OSWorldOutputContractHook()
    output = SimpleNamespace(agent_output={'legacy': True}, task_output={'agent_output': {'exit_code': 0}})
    await hook.on_run_output(SimpleNamespace(task_name='other'), output)
    assert output.agent_output == {'legacy': True}
    await hook.on_run_output(SimpleNamespace(task_name='os_world'), output)
    assert output.agent_output == {'exit_code': 0}


def test_oom_requires_explicit_runtime_reason_not_a_cli_code():
    from nexus.extensions.tasks.os_world.outputs import classify_exception

    exc = RuntimeError('killed')
    assert classify_exception(exc, stage='agent').exit_status == 'unknown_error'
    exc.reason = 'Process.OOMKilled'
    result = classify_exception(exc, stage='agent')
    assert result.exit_status == 'runtime_error_by_agent'
    assert result.exit_code == 0
    assert normalize_agent_output({'success': False, 'exit_code': 137}).exit_code == -1


@pytest.mark.parametrize('confirmed', [True, False])
def test_codex_budget_requires_confirmed_cleanup(confirmed):
    result = normalize_agent_output({
        'exit_code': -1,
        'success': False,
        'failure_kind': 'infrastructure',
        'failure_stage': 'timeout',
        'budget_stop_confirmed': confirmed,
        'error': 'Codex execution time budget exhausted',
    })
    assert result.exit_status == ('agent_timeout' if confirmed else 'runtime_error_by_env')
    assert result.exit_code == (0 if confirmed else -1)


def test_cleanup_error_cannot_be_masked_by_context_error():
    import json

    output = normalize_agent_output(
        {
            'success': False,
            'failure_stage': 'cleanup',
            'stderr': 'termination failed',
            'output': json.dumps({'type': 'result', 'is_error': True, 'result': 'context_length_exceeded'}),
        },
        harness='cc',
    )
    assert output.exit_status == 'runtime_error_by_env'
    assert output.exit_code == -1
    assert output.error == 'termination failed'


@pytest.mark.asyncio
async def test_anthropic_retry_envelope_keeps_original_provider_error():
    from contextlib import suppress
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.core.openai_client import ContextLengthExceededError
    from nexus.extensions.tasks.os_world.outputs import run_task_agent

    ctx = ExecutionContext(task_name='os_world', agent_name='cua_anthropic_agent')
    ctx.openai_clients = {
        'main': SimpleNamespace(create_completion=AsyncMock(side_effect=ContextLengthExceededError('context full')))
    }

    async def agent(**kwargs):
        with suppress(ContextLengthExceededError):
            await ctx.get_openai_client().create_completion(messages=[])
        raise RuntimeError('Anthropic API failed too many times: context full')

    async with ctx:
        result = await run_task_agent(agent)
    assert result.exit_status == 'context_length_exceeded'
    assert result.error == 'ContextLengthExceededError: context full'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'mode',
    [
        'gpt_bare',
        'claude_bare',
        'gpt_codex',
        'claude_cc',
        'hunyuan_codex',
        'hunyuan_cc',
    ],
)
@pytest.mark.parametrize('status', list(AgentExitStatus))
async def test_six_mode_stage_matrix_preserves_validity_and_judge_boundary(tmp_path, monkeypatch, mode, status):
    import json

    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.data.artifacts import RunArtifacts
    from nexus.extensions.tasks.os_world.task import OSWorldTaskInput, run_os_world
    from nexus.runner.single import RunInput

    for key in (
        'WEDATA_KEY',
        'AGS_SECRET_ID',
        'AGS_SECRET_KEY',
        'AGS_OSWORLD_TOOL_ID',
        'NEXUS_API_KEY',
        'NEXUS_BASE_URL',
        'NEXUS_MODEL',
        'HUNYUAN_BASE_URL',
        'HUNYUAN_MODEL',
    ):
        monkeypatch.setenv(key, 'fixture')
    root = Path(__file__).resolve().parents[5]
    config = RunInput.from_file(root / 'nexus/extensions/tasks/os_world/configs' / f'osworld_{mode}.yaml')
    task_file = tmp_path / 'case.json'
    task_file.write_text(json.dumps({'id': 'matrix', 'instruction': 'fixture', 'config': [], 'evaluator': {}}))
    monkeypatch.setattr(RunArtifacts, 'publish', AsyncMock(side_effect=ConnectionError('injected upload failure')))
    evaluator = AsyncMock(return_value={'score': 0})
    monkeypatch.setattr('nexus.extensions.tasks.os_world.task.evaluate_os_world', evaluator)
    valid = status in EVALUABLE_AGENT_STATUSES
    stage = OSWorldAgentOutput(
        exit_status=status,
        exit_code=0 if valid else -1,
        error=None if status == AgentExitStatus.FINISHED else 'actual fixture error',
        process_exit_code=137,
    )
    ctx = ExecutionContext(task_name='os_world', agent_name=config.agent_name)
    ctx._runtimes['main'] = SimpleNamespace(drain_trace=AsyncMock())

    async def agent(**kwargs):
        return stage

    async with ctx:
        inp = OSWorldTaskInput(
            task_file=str(task_file),
            run_agent=agent,
            run_setup=False,
            include_initial_screenshot=False,
            skip_gui_start=True,
            setup_wait_time=0,
        )
        if valid:
            output = await run_os_world(inp)
            assert output.status == 'success'
            assert output.score == 0
            assert output.agent_output.exit_status == status
            assert output.agent_output.process_exit_code == 137
            assert output.judge_output.exit_code == 0
            assert output.artifacts['upload_failed']
            evaluator.assert_awaited_once()
        else:
            with pytest.raises(RuntimeError, match='actual fixture error'):
                await run_os_world(inp)
            assert ctx.tool_state['osworld_partial_task_output']['status'] == 'agent_error'
            assert ctx.tool_state['osworld_partial_task_output']['agent_output']['exit_status'] == status
            assert ctx.tool_state['osworld_partial_task_output']['judge_output'] is None
            evaluator.assert_not_awaited()
