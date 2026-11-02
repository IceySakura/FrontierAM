from __future__ import annotations

import base64
import struct
import zlib

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from pydantic import BaseModel, Field

from nexus.core import OpenAIConfig, ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentInput, PostIterationHookResult, run_main_agent
from nexus.extensions.openai_clients.openai_responses import (
    OpenAIResponsesClient,
    OpenAIResponsesRequestError,
)
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.provider_call_capture import ProviderCallCaptureProcessor


_STEP_TOOL = 'integration/responses_step'
_SHELL_TOOL = 'integration/responses_shell'
_PATCH_TOOL = 'integration/responses_apply_patch'
_COMPUTER_TOOL = 'integration/responses_computer'
_GUI_STATE_KEY = 'responses_four_round_gui_state'
_GUI_ACTIONS_KEY = 'responses_four_round_gui_actions'


class _StepInput(BaseModel):
    step: int


class _ShellInput(BaseModel):
    commands: list[str]
    max_output_length: int | None = None
    timeout_ms: int | None = None


class _PatchInput(BaseModel):
    type: str
    path: str
    diff: str | None = None


class _ComputerInput(BaseModel):
    actions: list[dict[str, Any]]
    pending_safety_checks: list[dict[str, Any]] = Field(default_factory=list)
    raw_item: dict[str, Any]


@tool_registry.register(
    _STEP_TOOL,
    meta={
        'name': 'integration_step',
        'description': 'Advance the deterministic Responses integration state machine.',
        'tool_type': ToolType.LOCAL,
    },
)
async def _integration_step_tool(params: _StepInput, context: ToolContext) -> ToolResponse:
    del context
    return ToolResponse(output=[{'type': 'text', 'text': f'step={params.step}'}], meta={'success': True})


@tool_registry.register(
    _SHELL_TOOL,
    meta={
        'name': 'shell',
        'description': 'Execute the deterministic shell step.',
        'tool_type': ToolType.LOCAL,
    },
)
async def _integration_shell_tool(params: _ShellInput, context: ToolContext) -> ToolResponse:
    del context
    return ToolResponse(
        output=[{'type': 'text', 'text': f'stdout:{" ".join(params.commands)}'}],
        meta={'success': True, 'exit_code': 0},
    )


@tool_registry.register(
    _PATCH_TOOL,
    meta={
        'name': 'apply_patch',
        'description': 'Apply the deterministic patch step.',
        'tool_type': ToolType.LOCAL,
    },
)
async def _integration_patch_tool(params: _PatchInput, context: ToolContext) -> ToolResponse:
    del context
    return ToolResponse(
        output=[{'type': 'text', 'text': f'{params.type}:{params.path}'}],
        meta={'success': True},
    )


@tool_registry.register(
    _COMPUTER_TOOL,
    meta={
        'name': 'computer',
        'description': 'Execute one deterministic GUI state transition.',
        'tool_type': ToolType.LOCAL,
    },
)
async def _integration_computer_tool(params: _ComputerInput, context: ToolContext) -> ToolResponse:
    state = int(context.tool_state.get(_GUI_STATE_KEY, 0))
    expected_actions = [
        {'type': 'click', 'x': 20, 'y': 30, 'button': 'left'},
        {'type': 'type', 'text': 'CODEX'},
        {'type': 'keypress', 'keys': ['ENTER']},
    ]
    assert params.actions == [expected_actions[state]]
    actions = [*context.tool_state.get(_GUI_ACTIONS_KEY, []), params.actions[0]]
    return ToolResponse(
        output=[{'type': 'text', 'text': f'gui-state={state + 1}'}],
        meta={'success': True},
        context_updates={_GUI_STATE_KEY: state + 1, _GUI_ACTIONS_KEY: actions},
    )


def _png_data_url(red: int, green: int, blue: int) -> str:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body) & 0xFFFFFFFF)

    header = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(bytes([0, red, green, blue]))
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', pixels) + chunk(b'IEND', b'')
    return f'data:image/png;base64,{base64.b64encode(png).decode()}'


_GUI_SCREENSHOTS = [
    _png_data_url(255, 0, 0),
    _png_data_url(0, 255, 0),
    _png_data_url(0, 0, 255),
    _png_data_url(255, 255, 0),
]


async def _four_round_gui_observation(ctx: Any, hook_context: Any) -> PostIterationHookResult:
    assert ctx.tool_state[_GUI_STATE_KEY] == hook_context.iteration
    return PostIterationHookResult(
        extra_messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Current desktop screenshot.'},
                    {'type': 'image_url', 'image_url': {'url': _GUI_SCREENSHOTS[hook_context.iteration]}},
                ],
            }
        ]
    )


def _reasoning(index: int) -> dict[str, Any]:
    return {
        'type': 'reasoning',
        'id': f'rs_{index}',
        'encrypted_content': f'encrypted-{index}',
        'summary': [{'type': 'summary_text', 'text': f'reasoning-{index}'}],
    }


def _response(index: int, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'id': f'resp_{index}',
        'object': 'response',
        'created_at': index,
        'model': 'gpt-integration',
        'status': 'completed',
        'output': output,
        'usage': {
            'input_tokens': 10 * index,
            'output_tokens': 5 * index,
            'total_tokens': 15 * index,
            'output_tokens_details': {'reasoning_tokens': index},
        },
    }


class _ScriptedResponsesCreate:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses = [
            _response(
                1,
                [
                    _reasoning(1),
                    {
                        'type': 'function_call',
                        'id': 'fc_1',
                        'call_id': 'call_1',
                        'name': 'integration_step',
                        'arguments': '{"step":1}',
                        'status': 'completed',
                    },
                ],
            ),
            _response(
                2,
                [
                    _reasoning(2),
                    {
                        'type': 'function_call',
                        'id': 'fc_2',
                        'call_id': 'call_2',
                        'name': 'integration_step',
                        'arguments': '{"step":2}',
                        'status': 'completed',
                    },
                ],
            ),
            _response(
                3,
                [
                    _reasoning(3),
                    {
                        'type': 'function_call',
                        'id': 'fc_3',
                        'call_id': 'call_3',
                        'name': 'integration_step',
                        'arguments': '{"step":3}',
                        'status': 'completed',
                    },
                ],
            ),
            _response(
                4,
                [
                    _reasoning(4),
                    {
                        'type': 'message',
                        'id': 'msg_4',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'FOUR_ROUND_OK'}],
                    },
                ],
            ),
        ]

    async def __call__(self, **request: Any) -> dict[str, Any]:
        self.requests.append(deepcopy(request))
        return deepcopy(self.responses[len(self.requests) - 1])


class _RetryFirstResponsesCreate(_ScriptedResponsesCreate):
    def __init__(self) -> None:
        super().__init__()
        self.transport_attempts = 0

    async def __call__(self, **request: Any) -> dict[str, Any]:
        self.transport_attempts += 1
        if self.transport_attempts == 1:
            raise httpx.ConnectError('retry the first transport attempt')
        return await super().__call__(**request)


class _ProviderRejectionError(RuntimeError):
    status_code = 422
    body = {'error': {'type': 'invalid_request_error', 'message': 'scripted provider rejection'}}


class _TerminalErrorResponsesCreate:
    async def __call__(self, **request: Any) -> dict[str, Any]:
        del request
        raise _ProviderRejectionError('scripted provider rejection')


def _item_types(request: dict[str, Any]) -> list[str]:
    return [item['type'] for item in request['input'] if isinstance(item, dict) and 'type' in item]


@pytest.mark.integration
async def test_main_agent_responses_preserves_four_round_function_history() -> None:
    scripted_create = _ScriptedResponsesCreate()
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'integration-secret',
                'base_url': 'https://responses.invalid/v1',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={
                'model': 'gpt-integration',
                'store': False,
                'reasoning': {'effort': 'high'},
            },
        )
    )
    client.client = SimpleNamespace(responses=SimpleNamespace(create=scripted_create))
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')

    async with context:
        output = await run_main_agent(
            MainAgentInput(
                system_prompt='Follow the deterministic four-round tool sequence.',
                user_prompt=[
                    {'type': 'text', 'text': 'Inspect the initial screenshot and complete all three steps.'},
                    {
                        'type': 'image_url',
                        'image_url': {'url': 'data:image/png;base64,aW5pdGlhbC1wbmc='},
                    },
                ],
                tools=[_STEP_TOOL],
                max_iterations=4,
                single_tool_timeout=10,
                inference_time_timeout=30,
                observation_time_timeout=30,
            )
        )

    assert output.exit_status == 'finished'
    assert output.iterations == 4
    assert output.final_answer == 'FOUR_ROUND_OK'
    assert len(scripted_create.requests) == 4

    first_user = next(item for item in scripted_create.requests[0]['input'] if item.get('role') == 'user')
    assert first_user['content'] == [
        {'type': 'input_text', 'text': 'Inspect the initial screenshot and complete all three steps.'},
        {'type': 'input_image', 'image_url': 'data:image/png;base64,aW5pdGlhbC1wbmc='},
    ]

    for request_index, request in enumerate(scripted_create.requests[1:], start=1):
        assert _item_types(request)[-3:] == [
            'reasoning',
            'function_call',
            'function_call_output',
        ]
        reasoning_items = [item for item in request['input'] if item.get('type') == 'reasoning']
        assert [item['id'] for item in reasoning_items] == [f'rs_{index}' for index in range(1, request_index + 1)]
        assert [item['encrypted_content'] for item in reasoning_items] == [
            f'encrypted-{index}' for index in range(1, request_index + 1)
        ]
        function_output = next(
            item
            for item in request['input']
            if item.get('type') == 'function_call_output' and item.get('call_id') == f'call_{request_index}'
        )
        assert function_output['output'] == f'step={request_index}'


@pytest.mark.integration
async def test_main_agent_responses_preserves_four_round_computer_screenshots() -> None:
    actions = [
        {'type': 'click', 'x': 20, 'y': 30, 'button': 'left'},
        {'type': 'type', 'text': 'CODEX'},
        {'type': 'keypress', 'keys': ['ENTER']},
    ]
    scripted_create = _ScriptedResponsesCreate()
    scripted_create.responses = [
        _response(
            index,
            [
                _reasoning(index),
                {
                    'type': 'computer_call',
                    'id': f'computer_{index}',
                    'call_id': f'computer_call_{index}',
                    'actions': [action],
                    'pending_safety_checks': [{'id': 'safe-1'}] if index == 1 else [],
                    'status': 'completed',
                },
            ],
        )
        for index, action in enumerate(actions, start=1)
    ] + [
        _response(
            4,
            [
                _reasoning(4),
                {
                    'type': 'message',
                    'id': 'gui_done',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': 'GUI_FOUR_ROUND_OK'}],
                },
            ],
        )
    ]
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'integration-secret',
                'base_url': 'https://responses.invalid/v1',
                'max_retries': 1,
            },
            request_args={'model': 'gpt-integration', 'store': False},
        )
    )
    client.client = SimpleNamespace(responses=SimpleNamespace(create=scripted_create))
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')

    async with context:
        output = await run_main_agent(
            MainAgentInput(
                system_prompt='Use the computer exactly once per turn.',
                user_prompt=[
                    {'type': 'text', 'text': 'Complete the progressive GUI.'},
                    {'type': 'image_url', 'image_url': {'url': _GUI_SCREENSHOTS[0]}},
                ],
                tools=[_COMPUTER_TOOL],
                tool_schema_overrides=[{'type': 'computer'}],
                post_iteration_hook=(
                    'tests.integration.openai_clients.test_openai_responses_four_round::_four_round_gui_observation'
                ),
                max_iterations=4,
                single_tool_timeout=10,
                inference_time_timeout=30,
                observation_time_timeout=30,
            )
        )

    assert output.exit_status == 'finished'
    assert output.iterations == 4
    assert output.final_answer == 'GUI_FOUR_ROUND_OK'
    assert context.tool_state[_GUI_STATE_KEY] == 3
    assert context.tool_state[_GUI_ACTIONS_KEY] == actions
    assert len(scripted_create.requests) == 4
    assert all(request['tools'] == [{'type': 'computer'}] for request in scripted_create.requests)

    for request_index, request in enumerate(scripted_create.requests[1:], start=1):
        computer_output = next(
            item
            for item in request['input']
            if item.get('type') == 'computer_call_output' and item.get('call_id') == f'computer_call_{request_index}'
        )
        assert computer_output['output'] == {
            'type': 'computer_screenshot',
            'image_url': _GUI_SCREENSHOTS[request_index],
        }
        if request_index == 1:
            assert computer_output['acknowledged_safety_checks'] == [{'id': 'safe-1'}]
        assert not any(
            item.get('role') == 'user'
            and item.get('content')
            == [
                {'type': 'input_text', 'text': 'Current desktop screenshot.'},
            ]
            for item in request['input']
        )


@pytest.mark.integration
async def test_provider_capture_records_one_complete_call_per_responses_round() -> None:
    scripted_create = _ScriptedResponsesCreate()
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'tracker-integration-secret',
                'base_url': 'https://responses.invalid/v1',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'gpt-integration', 'store': False},
        )
    )
    client.client = SimpleNamespace(responses=SimpleNamespace(create=scripted_create))
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')

    with processor_context([tracker, provider_capture]):
        async with context:
            output = await run_main_agent(
                MainAgentInput(
                    system_prompt='Complete exactly three tools before the final answer.',
                    user_prompt='Run the four-round tracker integration.',
                    tools=[_STEP_TOOL],
                    max_iterations=4,
                    single_tool_timeout=10,
                    inference_time_timeout=30,
                    observation_time_timeout=30,
                )
            )

    assert output.exit_status == 'finished'
    result = tracker.get_result()
    assert result['trajectories']
    assert 'calls' not in result
    calls = provider_capture.get_result()
    assert len(calls) == 4
    assert [call['provider'] for call in calls] == ['openai'] * 4
    assert [call['protocol'] for call in calls] == ['responses'] * 4
    assert [call['provider_io']['attempts'] for call in calls] == [[{'attempt': 1, 'status': 'success'}]] * 4
    for index, call in enumerate(calls, start=1):
        assert call['status']['code'] == 'OK'
        assert 'input' not in call
        assert 'output' not in call
        assert call['provider_io']['request']['input']
        assert call['provider_io']['response']['id'] == f'resp_{index}'
        assert call['duration_ms'] is not None
    assert 'tracker-integration-secret' not in str(calls)


@pytest.mark.integration
async def test_provider_capture_keeps_retries_inside_one_four_round_call_record() -> None:
    scripted_create = _RetryFirstResponsesCreate()
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'tracker-retry-secret',
                'base_url': 'https://responses.invalid/v1',
                'capture_raw_io': True,
                'max_retries': 2,
            },
            request_args={'model': 'gpt-integration', 'store': False},
        )
    )
    client.client = SimpleNamespace(responses=SimpleNamespace(create=scripted_create))
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')

    with processor_context([tracker, provider_capture]):
        async with context:
            output = await run_main_agent(
                MainAgentInput(
                    system_prompt='Complete exactly three tools before the final answer.',
                    user_prompt='Run the four-round retry integration.',
                    tools=[_STEP_TOOL],
                    max_iterations=4,
                    single_tool_timeout=10,
                    inference_time_timeout=30,
                    observation_time_timeout=30,
                )
            )

    assert output.exit_status == 'finished'
    assert scripted_create.transport_attempts == 5
    assert 'calls' not in tracker.get_result()
    calls = provider_capture.get_result()
    assert len(calls) == 4
    assert all('input' not in call and 'output' not in call for call in calls)
    assert calls[0]['provider_io']['attempts'] == [
        {
            'attempt': 1,
            'status': 'error',
            'error_type': 'ConnectError',
            'message': 'retry the first transport attempt',
        },
        {'attempt': 2, 'status': 'success'},
    ]
    assert all(len(call['provider_io']['attempts']) == 1 for call in calls[1:])


@pytest.mark.integration
async def test_provider_capture_keeps_terminal_error_without_assistant_output() -> None:
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'tracker-error-secret',
                'base_url': 'https://responses.invalid/v1',
                'capture_raw_io': True,
                'max_retries': 2,
            },
            request_args={'model': 'gpt-integration', 'store': False},
        )
    )
    client.client = SimpleNamespace(responses=SimpleNamespace(create=_TerminalErrorResponsesCreate()))
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()

    with processor_context([tracker, provider_capture]):
        with pytest.raises(OpenAIResponsesRequestError, match='status 422'):
            await client.create_completion([{'role': 'user', 'content': 'trigger terminal provider error'}])

    result = tracker.get_result()
    # A terminal provider error yields no assistant output, so chat_tracker exports no
    # trajectory leaf at all. Diagnostics live only in the provider_call capture below.
    assert result['trajectories'] == []
    assert 'calls' not in result
    calls = provider_capture.get_result()
    assert len(calls) == 1
    call = calls[0]
    assert call['status']['code'] == 'ERROR'
    assert 'input' not in call
    assert 'output' not in call
    assert call['error']['message']
    assert call['provider_io']['attempts'] == [
        {
            'attempt': 1,
            'status': 'error',
            'error_type': '_ProviderRejectionError',
            'message': 'scripted provider rejection',
            'status_code': 422,
            'response': {
                'body': {
                    'error': {
                        'type': 'invalid_request_error',
                        'message': 'scripted provider rejection',
                    }
                },
                'status_code': 422,
            },
        }
    ]
    assert call['provider_io']['error'] == {
        'type': '_ProviderRejectionError',
        'message': 'scripted provider rejection',
        'status_code': 422,
    }
    assert 'tracker-error-secret' not in str(calls)
