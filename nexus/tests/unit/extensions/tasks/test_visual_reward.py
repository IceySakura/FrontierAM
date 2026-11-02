"""Visual task reward gating and the request consumed by the reward platform."""

import base64
import copy

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pydantic import ValidationError

from nexus.core import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentOutput
from nexus.extensions.tasks.visual import reward
from nexus.extensions.tasks.visual.task import VisualTaskInput, visual_task
from nexus.runner.hytuner import _task_reward_or_scoreable_zero
from nexus.runner.single import RunInput, run
from nexus.utils.reward_system import RewardSystemClient


ROUTER = 'MultimodalReward-GeneralV2-0'


@pytest.fixture
def make_reward_task():
    def make(output=None, **kwargs):
        values = {
            'user_prompt': 'Question',
            'reward_system_router_id': ROUTER,
            'run_agent': AsyncMock(
                return_value=output if output is not None else SimpleNamespace(final_answer='Answer')
            ),
        }
        return VisualTaskInput(**(values | kwargs))

    return make


@pytest.fixture
def reward_client(monkeypatch):
    client = SimpleNamespace(get_score=AsyncMock(return_value=1.0))
    monkeypatch.setattr(reward, 'RewardSystemClient', lambda **kwargs: client)
    return client


async def test_no_router_skips_scoring_even_with_reference_and_failed_agent(monkeypatch):
    constructor = AsyncMock(side_effect=AssertionError('must not construct reward client'))
    monkeypatch.setattr(reward, 'RewardSystemClient', constructor)
    agent = AsyncMock(return_value=SimpleNamespace(exit_status='runtime_error'))
    async with ExecutionContext():
        result = await visual_task(VisualTaskInput(user_prompt='Question', answer='Secret reference', run_agent=agent))
    assert result.score is None
    assert result.evaluation_details is None
    assert agent.await_args.kwargs == {'user_prompt': 'Question', 'history': None, 'upload_workspace': None}
    constructor.assert_not_called()


@pytest.mark.parametrize('router_id', ['', 'invalid', 'MM--0', 'MM-Rule-', '-Rule-0'])
def test_invalid_router_is_rejected_before_running_agent(router_id):
    with pytest.raises(ValidationError, match='class1-class2-router_version'):
        VisualTaskInput(user_prompt='Question', reward_system_router_id=router_id, run_agent=AsyncMock())


async def test_runner_sends_original_multimodal_question_reference_and_metadata_to_reward(tmp_path, monkeypatch):
    from nexus.extensions.agents.visual_agent import agent as agent_module

    image = tmp_path / 'source.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    prompt = [
        {'type': 'text', 'text': 'Before'},
        {'type': 'image_url', 'image_url': {'url': str(image), 'detail': 'high'}},
        {'type': 'text', 'text': 'After'},
    ]
    history = [{'role': 'assistant', 'content': 'Previous observation'}]
    original = copy.deepcopy(prompt)
    captured = []

    async def request(client, data):
        captured.append(data)
        return {'data': {'reward': {'error_code': 0, 'final_score': 0.75}}}

    async def delegate(main_input):
        assert 'Secret reference' not in str(main_input.model_dump())
        return MainAgentOutput(
            main_trajectory=[{'role': 'tool', 'content': 'Tool chatter must not enter reward input'}],
            exit_status='finished',
            final_answer='<think>Reasoning</think>Final response',
        )

    monkeypatch.setenv('NEXUS_REWARD_SYSTEM_TOKEN', 'unit-test-token')
    monkeypatch.setenv('NEXUS_REWARD_SYSTEM_EXP_ID', 'visual-unit-test')
    monkeypatch.setattr(RewardSystemClient, '_request', request)
    monkeypatch.setattr(agent_module, 'run_main_agent', delegate)
    run_input = RunInput(
        task_name='visual',
        task_config={
            'user_prompt': prompt,
            'history': history,
            'reward_system_router_id': ROUTER,
            'answer': ' Secret reference ',
            'reward_system_extra': {'checklist_mode': 'all', 'length_penalty_beta': 0.1},
            'public_test': {'input': 'example'},
        },
        agent_name='visual_agent',
    )
    result = await run(run_input)
    assert result.success, result.error
    assert result.task_output.score == 0.75
    assert _task_reward_or_scoreable_zero(result, False) == 0.75
    assert result.agent_output.final_answer == '<think>Reasoning</think>Final response'
    assert prompt == original
    assert run_input.task_config['user_prompt'] == original
    (data,) = captured
    assert (data['class1'], data['class2'], data['router_version']) == ('MultimodalReward', 'GeneralV2', '0')
    assert data['exp_id'] == 'visual-unit-test'
    assert data['ref_answer'] == 'Secret reference'
    assert data['extra'] == {'checklist_mode': 'all', 'length_penalty_beta': 0.1}
    assert data['public_test'] == {'input': 'example'}
    assert data['length_penalty_beta'] == 0.1
    assert data['messages'][:2] == [{'role': 'system', 'content': ''}, *history]
    parts = data['messages'][2]['content']
    assert [part['type'] for part in parts] == ['text', 'image_url', 'text']
    assert parts[0] == original[0]
    assert parts[2] == original[2]
    assert parts[1]['image_url']['detail'] == 'high'
    assert parts[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert base64.b64decode(parts[1]['image_url']['url'].split(',', 1)[1]) == image.read_bytes()
    assert data['messages'][3:] == [{'role': 'assistant', 'content': 'Final response'}]


@pytest.mark.parametrize('strip', [True, [ROUTER]])
async def test_text_only_router_strips_media_in_prompt_and_history_without_reading_files(
    strip, make_reward_task, reward_client
):
    client = reward_client
    media = [{'type': 'text', 'text': 'Question'}, {'type': 'image_url', 'image_url': {'url': '/missing.png'}}]
    task = make_reward_task(
        user_prompt=media,
        history=[{'role': 'assistant', 'content': media}],
        reward_system_strip_multimodal_content=strip,
    )
    result = await visual_task(task)
    assert result.score == 1.0
    assert 'log_summary' not in client.get_score.await_args.kwargs
    assert client.get_score.await_args.kwargs['messages'] == [
        {'role': 'system', 'content': ''},
        {'role': 'assistant', 'content': 'Question'},
        {'role': 'user', 'content': 'Question'},
        {'role': 'assistant', 'content': 'Answer'},
    ]
    assert task.user_prompt == media


@pytest.mark.parametrize('judge_score', [None, -100.0, float('nan'), float('inf')])
async def test_unusable_reward_is_masked_and_reported_to_hytuner(
    monkeypatch, judge_score, make_reward_task, reward_client
):
    client = reward_client
    client.get_score.return_value = judge_score
    monkeypatch.setattr(reward, 'get_current_span_id', lambda: 'task-span')
    recorded = []
    monkeypatch.setattr(reward, 'update_span', lambda span_id, *, attributes: recorded.append(attributes))
    task = make_reward_task()
    result = await visual_task(task)
    assert result.score == -100.0
    assert result.evaluation_details['error']
    assert recorded[-1]['hytuner.termination_reason'] == 'REWARD_ERROR'
    assert recorded[-1]['hytuner.reward_time'] >= 0


@pytest.mark.parametrize(
    ('exit_status', 'score'),
    [('inference_error', -100.0), ('tool_execution_error', -100.0), ('max_iterations', 0.0), ('tool_error', 0.0)],
)
async def test_abnormal_exit_skips_reward_request(exit_status, score, make_reward_task, reward_client):
    client = reward_client
    client.get_score.side_effect = AssertionError('must not be called')
    task = make_reward_task(output=SimpleNamespace(final_answer='Partial', exit_status=exit_status))
    result = await visual_task(task)
    assert result.score == score
    client.get_score.assert_not_called()


@pytest.mark.parametrize(
    'part',
    [
        {'type': 'file', 'file': {'url': '/host/input.pdf'}},
        {'type': 'file', 'file': {'file_data': '/host/input.pdf'}},
        {'type': 'video_url', 'video_url': {'url': '/host/input.mp4'}},
    ],
)
@pytest.mark.parametrize('in_history', [False, True])
async def test_reward_local_file_video_is_rejected_before_agent(part, in_history):
    with pytest.raises(ValidationError, match='local paths cannot be read'):
        await run(
            RunInput(
                task_name='visual',
                agent_name='visual_agent',
                task_config={
                    'user_prompt': 'Question' if in_history else [part],
                    'history': [{'role': 'assistant', 'content': [part]}] if in_history else None,
                    'reward_system_router_id': ROUTER,
                },
            )
        )


async def test_media_failure_prevents_reward_and_model(monkeypatch, main_delegate, reward_client):
    from nexus.extensions.agents.visual_agent.resources import images

    monkeypatch.setattr(images, 'upload_images_to_runtime', AsyncMock(side_effect=RuntimeError('upload failed')))
    monkeypatch.setattr(images, 'require_context', lambda: SimpleNamespace(runtime_provider=object()))
    result = await run(
        RunInput(
            task_name='visual',
            agent_name='visual_agent',
            task_config={
                'user_prompt': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QUJD'}}],
                'reward_system_router_id': ROUTER,
            },
        )
    )
    assert result.success is False
    main_delegate.assert_not_called()
    reward_client.get_score.assert_not_called()


def test_oversized_reward_image_and_raw_base64_have_clear_errors(tmp_path):
    path = tmp_path / 'large.png'
    with path.open('wb') as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='64 MiB'):
        reward._image_url(str(path))
    with pytest.raises(ValueError, match='wrap raw base64'):
        reward._image_url('A' * 5000)


async def test_reward_errors_include_diagnostics_without_signed_urls(caplog, make_reward_task, reward_client):
    client = reward_client
    client.get_score.side_effect = ConnectionError('bad request https://example.com/a?secret=value')
    result = await visual_task(make_reward_task(output=SimpleNamespace(final_answer='ok')))
    assert result.score == -100.0
    client.get_score.assert_awaited_once()
    assert 'bad request' in result.evaluation_details['error']
    assert 'secret=value' not in result.evaluation_details['error']
    assert 'bad request' in caplog.text
    assert 'secret=value' not in caplog.text


@pytest.mark.parametrize('file_data', ['/host/secret.pdf', '', None])
def test_file_id_cannot_bypass_reward_file_source_validation(file_data):
    with pytest.raises(ValidationError):
        VisualTaskInput(
            user_prompt=[{'type': 'file', 'file': {'file_id': 'id', 'file_data': file_data}}],
            reward_system_router_id=ROUTER,
            run_agent=AsyncMock(),
        )


@pytest.mark.parametrize(
    ('source', 'error'),
    [
        ('/nonexistent/image.png', 'readable local file'),
        ('data:image/png;base64,', 'base64'),
        ('data:image/png;base64,not-base64!', 'base64'),
    ],
)
async def test_reward_rejects_invalid_image_before_request(source, error, make_reward_task, reward_client):
    client = reward_client
    with pytest.raises(ValueError, match=error):
        await visual_task(
            make_reward_task(
                user_prompt=[{'type': 'image_url', 'image_url': {'url': source}}],
                output=SimpleNamespace(final_answer='ok'),
            )
        )
    client.get_score.assert_not_called()


@pytest.mark.parametrize('router_id', [None, ROUTER])
@pytest.mark.parametrize('parallel', [False, True])
async def test_transport_failure_stops_visual_agent_and_skips_reward(monkeypatch, tmp_path, router_id, parallel):
    from nexus.core import OpenAICompatibleClient, OpenAIConfig, ToolExecutor
    from nexus.core.types import ExtendedChatCompletion
    from nexus.trace import processor_context
    from nexus.trace.processors.hytuner import HytunerProcessor

    class Client(OpenAICompatibleClient):
        def __init__(self):
            super().__init__(OpenAIConfig(client_args={}, request_args={}))
            self.calls = 0

        async def _create_completion(self, messages, model=None, **kwargs):
            self.calls += 1
            message = (
                {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'code',
                            'type': 'function',
                            'function': {'name': 'code_interpreter', 'arguments': '{"code":"print(42)"}'},
                        }
                    ],
                }
                if self.calls == 1
                else {'role': 'assistant', 'content': 'Guessed answer'}
            )
            return ExtendedChatCompletion.model_validate({
                'id': str(self.calls),
                'created': 0,
                'model': 'local-test',
                'choices': [
                    {'index': 0, 'message': message, 'finish_reason': 'tool_calls' if self.calls == 1 else 'stop'}
                ],
            })

    monkeypatch.delenv('NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE', raising=False)
    # The mock replaces the whole ToolExecutor entry point, so it raises what
    # the executor's RUNTIME dispatch branch surfaces for a dead sandbox
    # transport: the dedicated RuntimeTransportError (the raw OS error is
    # wrapped exactly there, so a LOCAL/MCP tool's own connection errors are
    # not misread as transport death).
    from nexus.runtime.client.base import RuntimeTransportError

    transport = AsyncMock(side_effect=RuntimeTransportError('runtime connection lost'))
    monkeypatch.setattr(ToolExecutor, 'parallel_call' if parallel else 'call', transport)
    scoring = AsyncMock(side_effect=AssertionError('Transport failures must not be scored'))
    monkeypatch.setattr(reward, 'RewardSystemClient', lambda **kwargs: SimpleNamespace(get_score=scoring))
    client = Client()
    processor = HytunerProcessor(file_path=str(tmp_path / 'trace.json'), trajectory_id='transport-failure')
    with processor_context([processor]):
        result = await run(
            RunInput(
                task_name='visual',
                agent_name='visual_agent',
                task_config={'user_prompt': 'Compute', 'reward_system_router_id': router_id},
                agent_config={'parallel_tool_calls': parallel},
                openai_clients={'main': {'instance': client}},
            )
        )
    assert result.success, result.error
    assert result.agent_output.exit_status == 'tool_execution_error'
    assert result.task_output.score == (None if router_id is None else -100.0)
    assert processor.termination_reason == 'ENV_ERROR'
    assert client.calls == 1
    transport.assert_awaited_once()
    scoring.assert_not_called()


@pytest.mark.parametrize('setting', ['missing-token', 'nan', '0', '-1', 'invalid'])
async def test_reward_configuration_fails_before_generation(monkeypatch, setting):
    monkeypatch.delenv('NEXUS_REWARD_SYSTEM_TOKEN', raising=False)
    monkeypatch.delenv('nexus_reward_system_token', raising=False)
    monkeypatch.delenv('REWARD_PLATFORM_TIMEOUT', raising=False)
    if setting != 'missing-token':
        monkeypatch.setenv('NEXUS_REWARD_SYSTEM_TOKEN', 'test-token')
        monkeypatch.setenv('REWARD_PLATFORM_TIMEOUT', setting)
    generate = AsyncMock()
    with pytest.raises(ValueError, match='user_token|REWARD_PLATFORM_TIMEOUT|could not convert'):
        await visual_task(VisualTaskInput(user_prompt='Question', reward_system_router_id=ROUTER, run_agent=generate))
    generate.assert_not_called()


@pytest.mark.parametrize('timeout', [None, '12.5'])
async def test_reward_uses_one_request_with_bounded_timeout(monkeypatch, timeout, make_reward_task):
    monkeypatch.setenv('NEXUS_REWARD_SYSTEM_TOKEN', 'test-token')
    monkeypatch.delenv('REWARD_PLATFORM_TIMEOUT', raising=False)
    if timeout:
        monkeypatch.setenv('REWARD_PLATFORM_TIMEOUT', timeout)
    client = reward.create_reward_client()
    assert client.timeout == (float(timeout) if timeout else 180)
    assert client.max_retries == 1
    request = AsyncMock(side_effect=TimeoutError('timed out'))
    monkeypatch.setattr(client, '_request', request)
    monkeypatch.setattr(reward, 'create_reward_client', lambda: client)
    result = await visual_task(make_reward_task())
    assert result.score == -100
    request.assert_awaited_once()


async def test_reward_programming_error_is_not_masked(make_reward_task, reward_client):
    client = reward_client
    client.get_score.side_effect = TypeError('bad argument')
    with pytest.raises(TypeError, match='bad argument'):
        await visual_task(make_reward_task())


@pytest.mark.parametrize('suffix', ['.bin', '.txt', '.jpg'])
def test_reward_image_mime_uses_bytes_over_suffix(tmp_path, suffix):
    path = tmp_path / ('image' + suffix)
    path.write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    assert reward._image_url(str(path)).startswith('data:image/png;base64,')


@pytest.mark.parametrize(
    ('output', 'expected'),
    [
        ({'final_answer': '<think>reason</think>final', 'output': 'ignored'}, 'final'),
        ({'final_answer': '', 'output': 'ignored'}, ''),
        ({'final_answer': None, 'main_trajectory': [{'role': 'assistant', 'content': 'last answer'}]}, 'last answer'),
        ({'main_trajectory': [{'role': 'assistant', 'reasoning_content': 'reason</think>answer'}]}, 'answer'),
        ({'main_trajectory': [{'role': 'tool', 'content': 'observation'}], 'output': 'fallback'}, 'fallback'),
        (
            {
                'main_trajectory': [{'role': 'assistant', 'content': 'intermediate', 'tool_calls': [{}]}],
                'output': 'fallback',
            },
            'fallback',
        ),
        ({'output': [{'type': 'text', 'text': 'part one'}, {'type': 'text', 'text': 'part two'}]}, 'part one part two'),
        ({}, ''),
    ],
)
async def test_reward_prediction_fallback_reaches_request(output, expected, make_reward_task, reward_client):
    await visual_task(make_reward_task(output=SimpleNamespace(**output)))
    assert reward_client.get_score.await_args.kwargs['messages'][-1] == {'role': 'assistant', 'content': expected}


@pytest.mark.parametrize(
    'source',
    [
        'data:image/png;base64,iVBORw0KGgo=',
        'https://example.com/image.png',
        'http://example-1250000000.cos-internal.ap-guangzhou.tencentcos.cn/a.png?q-signature=test-only&x=%2F',
    ],
)
async def test_reward_preserves_inline_and_remote_images(source, make_reward_task, reward_client, monkeypatch):
    def unexpected(path):
        raise AssertionError('Remote and inline images must not open local files')

    monkeypatch.setattr('pathlib.Path.open', unexpected)
    prompt = [
        {'type': 'text', 'text': 'Before'},
        {'type': 'image_url', 'image_url': {'url': source, 'detail': 'high'}},
        {'type': 'text', 'text': 'After'},
    ]
    original = copy.deepcopy(prompt)
    await visual_task(make_reward_task(user_prompt=prompt))
    assert reward_client.get_score.await_args.kwargs['messages'][1] == {'role': 'user', 'content': original}
    assert prompt == original


PNG_URI = 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode()


def _tool_trajectory(count):
    trajectory = []
    for i in range(count):
        trajectory.append({
            'role': 'assistant',
            'content': None,
            'tool_calls': [{'id': f'c{i}', 'type': 'function', 'function': {'name': 'plot', 'arguments': '{}'}}],
        })
        trajectory.append({
            'role': 'tool',
            'tool_call_id': f'c{i}',
            'content': [{'type': 'text', 'text': 'ok'}, {'type': 'image_url', 'image_url': {'url': PNG_URI}}],
        })
    return trajectory


def test_tool_image_limit_requires_the_trajectory_switch():
    with pytest.raises(ValidationError, match='reward_system_include_trajectory'):
        VisualTaskInput(user_prompt='Q', reward_system_max_tool_images=1, run_agent=AsyncMock())


async def test_trajectory_keeps_only_the_latest_tool_images(make_reward_task, reward_client):
    history = [{'role': 'assistant', 'content': 'Previous'}]
    messages, (_, details) = await _score_trajectory(
        make_reward_task,
        reward_client,
        [*_tool_trajectory(3), {'role': 'assistant', 'content': 'Closing', 'reasoning_content': 'private'}],
        history=history,
        task_config={'reward_system_max_tool_images': 1},
    )
    roles = ['system', 'assistant', 'user'] + ['assistant', 'tool'] * 3 + ['assistant']
    assert [m['role'] for m in messages] == roles
    assert messages[:3] == [{'role': 'system', 'content': ''}, *history, {'role': 'user', 'content': 'Question'}]
    assert messages[-1]['content'] == 'Answer'
    has_image = [any(p['type'] == 'image_url' for p in m['content']) for m in messages if m['role'] == 'tool']
    assert has_image == [False, False, True]
    assert details['tool_images_sent'] == 1
    assert details['trajectory_messages_sent'] == 6
    assert reward_client.get_score.await_args.kwargs['log_summary'] is True


async def _score_trajectory(
    make_reward_task, reward_client, generated, *, history=None, task_config=None, **output_fields
):
    history = history or []
    output = SimpleNamespace(**{
        'final_answer': 'Answer',
        'main_trajectory': [
            {'role': 'system', 'content': 'S'},
            *history,
            {'role': 'user', 'content': 'Question'},
            *generated,
        ],
        'generated_message_start': len(history) + 2,
        **output_fields,
    })
    task = make_reward_task(
        output=output, history=history, reward_system_include_trajectory=True, **(task_config or {})
    )
    result = await reward.score_visual_response(task, output, reward_client)
    reward_client.get_score.assert_awaited_once()
    return reward_client.get_score.await_args.kwargs['messages'], result


async def test_trajectory_filters_runtime_messages(make_reward_task, reward_client):
    from nexus.extensions.compaction.runtime_state import (
        build_compaction_resume_message,
        build_trajectory_budget_state_message,
    )

    budget = build_trajectory_budget_state_message(token_budget=1000, token_consumed=50, completed_compactions=0)
    call, response = _tool_trajectory(1)
    messages, _ = await _score_trajectory(
        make_reward_task,
        reward_client,
        [budget, call, response, build_compaction_resume_message(), {'role': 'assistant', 'content': 'Answer'}],
    )
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert messages[2]['tool_calls'][0]['id'] == messages[3]['tool_call_id']


@pytest.mark.parametrize('start', [None, 1, 3])
async def test_trajectory_rejects_missing_or_invalid_boundary(make_reward_task, reward_client, start):
    output = SimpleNamespace(
        final_answer='Answer',
        generated_message_start=start,
        main_trajectory=[{'role': 'system'}, {'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'}],
    )
    task = make_reward_task(reward_system_include_trajectory=True)
    with pytest.raises(ValueError, match='boundary|generated_message_start'):
        await reward.score_visual_response(task, output, reward_client)
    reward_client.get_score.assert_not_awaited()


async def test_trajectory_requires_agent_output(make_reward_task, reward_client):
    output = SimpleNamespace(final_answer='Answer', main_trajectory=[])
    task = make_reward_task(reward_system_include_trajectory=True)
    with pytest.raises(ValueError, match='non-empty main_trajectory'):
        await reward.score_visual_response(task, output, reward_client)
    reward_client.get_score.assert_not_awaited()


async def test_trajectory_allows_direct_answer(make_reward_task, reward_client):
    messages, (_, details) = await _score_trajectory(
        make_reward_task,
        reward_client,
        [{'role': 'assistant', 'content': '<think>private</think>Answer'}],
    )
    assert messages == [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': 'Question'},
        {'role': 'assistant', 'content': 'Answer'},
    ]
    assert details['trajectory_messages_sent'] == 0


@pytest.mark.parametrize('content', ['observation', []])
async def test_trajectory_ending_with_tool_preserves_content_and_pair(make_reward_task, reward_client, content):
    call, response = _tool_trajectory(1)
    response['content'] = content
    messages, _ = await _score_trajectory(make_reward_task, reward_client, [call, response])
    assert messages[2]['tool_calls'][0]['id'] == messages[3]['tool_call_id']
    assert messages[3]['content'] == content
    assert messages[-1] == {'role': 'assistant', 'content': 'Answer'}


@pytest.mark.parametrize(
    ('kind', 'exit_status'),
    [
        ('orphan', 'finished'),
        ('missing', 'finished'),
        ('wrong_id', 'finished'),
        ('duplicate', 'finished'),
        ('missing', 'refused'),
    ],
)
async def test_trajectory_rejects_unpaired_tools(make_reward_task, reward_client, kind, exit_status):
    call, response = _tool_trajectory(1)
    generated = [call, response]
    if kind == 'orphan':
        generated = [response]
    elif kind == 'missing':
        generated = [call]
    elif kind == 'wrong_id':
        response['tool_call_id'] = 'unknown'
    else:
        generated.append(response)
    if exit_status == 'refused':
        generated.append({'role': 'assistant', 'refusal': 'Request refused.'})
    with pytest.raises(ValueError, match='tool'):
        await _score_trajectory(make_reward_task, reward_client, generated, exit_status=exit_status)
    reward_client.get_score.assert_not_awaited()


async def test_refusal_with_unexecuted_tool_calls_is_scored(make_reward_task, reward_client):
    prior = _tool_trajectory(1)
    refusal = {
        'role': 'assistant',
        'refusal': 'Request refused.',
        'tool_calls': [
            {'id': 'unexecuted', 'type': 'function', 'function': {'name': 'must_not_run', 'arguments': '{}'}}
        ],
    }
    generated = [*prior, refusal]
    original = copy.deepcopy(generated)
    messages, (score, _) = await _score_trajectory(
        make_reward_task,
        reward_client,
        generated,
        exit_status='refused',
        final_answer='Request refused.',
    )
    assert score == 1.0
    assert messages[-1] == {'role': 'assistant', 'content': 'Request refused.'}
    assert len(messages) == 5
    assert all(call['id'] != 'unexecuted' for message in messages for call in message.get('tool_calls', []))
    assert messages[2]['tool_calls'][0]['id'] == messages[3]['tool_call_id']
    assert generated == original


@pytest.mark.parametrize('as_parts', [False, True])
async def test_trajectory_removes_intermediate_thinking(make_reward_task, reward_client, monkeypatch, as_parts):
    monkeypatch.setenv('THINK_END_TOKEN', '</custom-think>')
    call, response = _tool_trajectory(1)
    call['reasoning_content'] = 'separate private'
    call['content'] = (
        [{'type': 'text', 'text': 'private'}, {'type': 'text', 'text': ' more</custom-think>Plotting'}]
        if as_parts
        else 'private</custom-think>Plotting'
    )
    original = copy.deepcopy(call)
    messages, _ = await _score_trajectory(make_reward_task, reward_client, [call, response])
    assert 'reasoning_content' not in messages[2]
    assert reward.extract_text_from_content(messages[2]['content']).strip() == 'Plotting'
    assert call == original


@pytest.mark.parametrize(('strip', 'limit'), [(False, 0), (False, 1), (False, 5), (True, 1)])
async def test_trajectory_counts_individual_images_and_strips_consistently(
    make_reward_task, reward_client, strip, limit
):
    call, response = _tool_trajectory(1)
    image = response['content'][-1]
    response['content'] = [{'type': 'text', 'text': 'observation'}, image, image, image]
    messages, (_, details) = await _score_trajectory(
        make_reward_task,
        reward_client,
        [call, response],
        task_config={'reward_system_max_tool_images': limit, 'reward_system_strip_multimodal_content': strip},
    )
    content = messages[3]['content']
    if strip:
        assert content == 'observation'
        assert details['tool_images_sent'] == 0
    else:
        assert sum(p['type'] == 'image_url' for p in content) == min(limit, 3)
        assert details['tool_images_sent'] == min(limit, 3)


def test_trajectory_switch_requires_router():
    with pytest.raises(ValidationError, match='requires reward_system_router_id'):
        VisualTaskInput(user_prompt='Q', reward_system_include_trajectory=True, run_agent=AsyncMock())


async def test_trajectory_preserves_parallel_pairs_and_hook_images(make_reward_task, reward_client):
    first, first_response, second, second_response = _tool_trajectory(2)
    first['tool_calls'].extend(second['tool_calls'])
    image = {'type': 'image_url', 'image_url': {'url': PNG_URI}}
    hook_message = {'role': 'user', 'content': [image]}
    messages, (_, details) = await _score_trajectory(
        make_reward_task,
        reward_client,
        [first, second_response, first_response, hook_message],
    )
    assert {call['id'] for call in messages[2]['tool_calls']} == {message['tool_call_id'] for message in messages[3:5]}
    assert messages[5] == hook_message
    assert details['tool_images_sent'] == 0
