from __future__ import annotations

import asyncio
import os

from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from pydantic import BaseModel, Field

from nexus.core import OpenAIConfig, ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentInput, PostIterationHookResult, run_main_agent
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
from nexus.runtime.providers.docker import DockerRuntimeProvider
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.integration.openai_clients.test_openai_responses_four_round import (
    _ScriptedResponsesCreate,
    _reasoning,
    _response,
)
from tests.provider_call_capture import ProviderCallCaptureProcessor


_RUN_GUI_ENV = 'NEXUS_RUN_GUI_INTEGRATION'
_REAL_COMPUTER_TOOL = 'integration/responses_real_gui_computer'
_GUI_FILE = '/tmp/nexus_responses_real_gui_four_round.txt'
_GUI_TEXT = 'RESPONSES_REAL_GUI_FOUR_ROUND_OK'
_GUI_ACTIONS_STATE = 'responses_real_gui_actions'


class _DelayedScriptedResponsesCreate(_ScriptedResponsesCreate):
    """Model stub whose latency exceeds the runtime server's keep-alive window."""

    async def __call__(self, **request: Any) -> dict[str, Any]:
        await asyncio.sleep(6)
        return await super().__call__(**request)


class _RealComputerInput(BaseModel):
    actions: list[dict[str, Any]]
    pending_safety_checks: list[dict[str, Any]] = Field(default_factory=list)
    raw_item: dict[str, Any]


@tool_registry.register(
    _REAL_COMPUTER_TOOL,
    meta={
        'name': 'computer',
        'description': 'Execute official Responses computer actions against the integration desktop.',
        'tool_type': ToolType.LOCAL,
    },
)
async def _real_computer_tool(params: _RealComputerInput, context: ToolContext) -> ToolResponse:
    runtime = await require_context().get_runtime('main')
    recorded_actions = list(context.tool_state.get(_GUI_ACTIONS_STATE, []))

    for action in params.actions:
        action_type = action.get('type')
        if action_type == 'click':
            raw_button = str(action.get('button') or 'left')
            if raw_button not in {'left', 'right', 'middle'}:
                raise ValueError(f'Unsupported real GUI integration mouse button: {raw_button!r}')
            response = await runtime.gui_click(
                int(action['x']),
                int(action['y']),
                button=cast(Literal['left', 'right', 'middle'], raw_button),
            )
        elif action_type == 'type':
            response = await runtime.gui_type(str(action['text']))
        elif action_type == 'keypress':
            keys = [str(key) for key in action.get('keys') or []]
            if len(keys) != 1:
                raise ValueError(f'The real GUI integration expects one key per keypress action, got {keys!r}')
            response = await runtime.gui_press_key(keys[0].title())
        else:
            raise ValueError(f'Unsupported real GUI integration action: {action_type!r}')
        if not response.success:
            raise RuntimeError(f'Real GUI action failed: {action!r}: {response.error}')
        recorded_actions.append(action)
        await asyncio.sleep(0.2)

    return ToolResponse(
        output=[{'type': 'text', 'text': f'Executed {len(params.actions)} real GUI action(s).'}],
        meta={'success': True},
        context_updates={_GUI_ACTIONS_STATE: recorded_actions},
    )


async def _real_gui_screenshot_observation(ctx: Any, hook_context: Any) -> PostIterationHookResult:
    del hook_context
    runtime = await ctx.get_runtime('main')
    screenshot = await runtime.gui_screenshot()
    if not screenshot.success or not screenshot.image:
        raise RuntimeError(f'Real GUI screenshot failed: {screenshot.error}')
    return PostIterationHookResult(
        extra_messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Current desktop screenshot.'},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/{screenshot.format};base64,{screenshot.image}'},
                    },
                ],
            }
        ]
    )


def _require_gui_runtime() -> None:
    if os.getenv(_RUN_GUI_ENV) != '1':
        pytest.skip(f'set {_RUN_GUI_ENV}=1 to run Docker GUI integration tests')


@pytest.mark.integration
@pytest.mark.gui
async def test_openai_responses_real_gui_completes_four_provider_rounds() -> None:
    _require_gui_runtime()
    actions = [
        {'type': 'click', 'x': 220, 'y': 170, 'button': 'left'},
        {'type': 'type', 'text': f"printf %s '{_GUI_TEXT}' > {_GUI_FILE}"},
        {'type': 'keypress', 'keys': ['ENTER']},
    ]
    scripted_create = _DelayedScriptedResponsesCreate()
    scripted_create.responses = [
        _response(
            index,
            [
                _reasoning(index),
                {
                    'type': 'computer_call',
                    'id': f'real_gui_computer_{index}',
                    'call_id': f'real_gui_call_{index}',
                    'actions': [action],
                    'pending_safety_checks': [{'id': 'real-gui-safe'}] if index == 1 else [],
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
                    'id': 'real_gui_done',
                    'role': 'assistant',
                    'status': 'completed',
                    'content': [{'type': 'output_text', 'text': _GUI_TEXT}],
                },
            ],
        )
    ]
    client = OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': 'real-gui-integration-secret',
                'base_url': 'https://responses.invalid/v1',
                'capture_raw_io': True,
                'max_retries': 1,
            },
            request_args={'model': 'gpt-real-gui-integration', 'store': False},
        )
    )
    client.client = SimpleNamespace(responses=SimpleNamespace(create=scripted_create))
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    provider = DockerRuntimeProvider(
        image=os.getenv(
            'NEXUS_GUI_BASE_IMAGE',
            'mirrors.tencent.com/hunyuan_yanguan/nexus-ubuntu-session:latest',
        ),
        runtime_image=os.getenv('NEXUS_GUI_NEXUS_RUNTIME_IMAGE', 'nexus-runtime:1.5.2'),
        force_rebuild=False,
        exposed_ports=[5900, 6080],
        print_port_mappings=False,
        request_timeout=300,
    )
    context = ExecutionContext(
        runtime_provider=provider,
        openai_clients={'main': client},
        run_id='responses-real-gui-four-round',
        agent_name='main_agent',
    )

    with processor_context([tracker, provider_capture]):
        async with context:
            runtime = await context.get_runtime('main')
            started = await runtime.gui_start()
            assert started.success, started.error
            launched = await runtime.gui_launch_app('xterm -title NexusResponsesE2E -geometry 86x24+80+70')
            assert launched.success, launched.error
            await asyncio.sleep(1)
            initial_screenshot = await runtime.gui_screenshot()
            assert initial_screenshot.success
            assert initial_screenshot.image

            output = await run_main_agent(
                MainAgentInput(
                    system_prompt='Execute exactly one official computer action per turn.',
                    user_prompt=[
                        {'type': 'text', 'text': 'Create the requested file through the visible terminal.'},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': (f'data:image/{initial_screenshot.format};base64,{initial_screenshot.image}')
                            },
                        },
                    ],
                    tools=[_REAL_COMPUTER_TOOL],
                    tool_schema_overrides=[{'type': 'computer'}],
                    post_iteration_hook=(
                        'tests.integration.gui_agent.test_openai_responses_real_gui_four_round'
                        '::_real_gui_screenshot_observation'
                    ),
                    max_iterations=4,
                    single_tool_timeout=30,
                    inference_time_timeout=60,
                    observation_time_timeout=60,
                )
            )
            verification = await runtime.run_command(f'cat {_GUI_FILE}')

    assert output.exit_status == 'finished'
    assert output.iterations == 4
    assert output.final_answer == _GUI_TEXT
    assert context.tool_state[_GUI_ACTIONS_STATE] == actions
    assert verification.return_code == 0
    assert verification.stdout is not None
    assert verification.stdout.strip() == _GUI_TEXT
    assert len(scripted_create.requests) == 4
    assert all(request['tools'] == [{'type': 'computer'}] for request in scripted_create.requests)

    for request_index, request in enumerate(scripted_create.requests[1:], start=1):
        computer_output = next(
            item
            for item in request['input']
            if item.get('type') == 'computer_call_output' and item.get('call_id') == f'real_gui_call_{request_index}'
        )
        image_url = computer_output['output']['image_url']
        assert image_url.startswith('data:image/png;base64,')
        assert len(image_url) > 1_000
    assert 'calls' not in tracker.get_result()
    assert len(provider_capture.get_result()) == 4
