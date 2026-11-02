from __future__ import annotations

import asyncio
import base64
import json
import os

from pathlib import Path
from typing import Any

import pytest

from pydantic import BaseModel

from nexus.core import OpenAIConfig, ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentInput, run_main_agent
from nexus.extensions.openai_clients.vertex import VertexClient
from nexus.runtime.providers.docker import DockerRuntimeProvider
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.e2e._llm_account import AccountLookupError, SQLiteModelAccount, load_sqlite_model_account
from tests.provider_call_capture import ProviderCallCaptureProcessor


_ACCOUNT_DB_ENV = 'NEXUS_LLM_E2E_ACCOUNT_DB'
_ACCOUNT_RUN_ID_ENV = 'NEXUS_LLM_E2E_ACCOUNT_RUN_ID'
_RUN_ENV = 'NEXUS_RUN_REAL_VERTEX_GUI_E2E'
_OUTPUT_ENV = 'NEXUS_VERTEX_GUI_E2E_OUTPUT'
_TOOL_NAME = 'e2e/vertex_real_gui_computer'
_ACTION_STATE = 'vertex_real_gui_actions'
_GUI_FILE = '/tmp/nexus_real_vertex_gui_four_round.txt'
_GUI_MARKER = 'NEXUS_REAL_VERTEX_GUI_FOUR_ROUND_OK'
_NATIVE_TOOL = {
    'type': 'computer_20251124',
    'name': 'computer',
    'display_width_px': 1920,
    'display_height_px': 1080,
    'display_number': 1,
}


class _VertexComputerInput(BaseModel):
    action: str
    coordinate: list[int] | None = None
    text: str | None = None


@tool_registry.register(
    _TOOL_NAME,
    meta={
        'name': 'computer',
        'description': 'Execute one native Anthropic computer action against the integration desktop.',
        'tool_type': ToolType.LOCAL,
    },
)
async def _vertex_computer_tool(params: _VertexComputerInput, context: ToolContext) -> ToolResponse:
    runtime = await require_context().get_runtime('main')
    action = params.action
    if action in {'left_click', 'click'}:
        if params.coordinate is None or len(params.coordinate) != 2:
            raise ValueError(f'{action} requires a two-element coordinate')
        response = await runtime.gui_click(params.coordinate[0], params.coordinate[1], button='left')
    elif action == 'type':
        if params.text is None:
            raise ValueError('type requires text')
        response = await runtime.gui_type(params.text)
    elif action in {'key', 'keypress'}:
        if params.text is None:
            raise ValueError(f'{action} requires text')
        response = await runtime.gui_press_key(params.text.title())
    else:
        raise ValueError(f'Unsupported native Anthropic computer action: {action!r}')
    if not response.success:
        raise RuntimeError(f'GUI action failed: {response.error}')

    actions = list(context.tool_state.get(_ACTION_STATE, []))
    actions.append(params.model_dump(mode='json', exclude_none=True))
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Executed native computer action {action}.'}],
        meta={'success': True},
        context_updates={_ACTION_STATE: actions},
    )


@pytest.fixture(scope='module')
def sqlite_model_account() -> SQLiteModelAccount:
    if os.getenv(_RUN_ENV) != '1':
        pytest.skip(f'set {_RUN_ENV}=1 to run the real Vertex Anthropic Docker GUI E2E')
    raw_path = os.environ.get(_ACCOUNT_DB_ENV)
    if not raw_path:
        pytest.fail(f'{_ACCOUNT_DB_ENV} is required when {_RUN_ENV}=1')
    assert raw_path is not None
    try:
        return load_sqlite_model_account(raw_path, run_id=os.environ.get(_ACCOUNT_RUN_ID_ENV))
    except AccountLookupError as error:
        pytest.fail(str(error))


def _client(account: SQLiteModelAccount) -> VertexClient:
    model = os.environ.get('NEXUS_VERTEX_ANTHROPIC_E2E_MODEL', 'claude-opus-4-8')
    timeout = float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300'))
    return VertexClient(
        OpenAIConfig(
            client_args={
                'protocol': 'anthropic',
                'api_key': account.credential,
                'base_url': account.base_url,
                'gateway_routing': {
                    'provider': 'google',
                    'model': model,
                    'api': 'claude_api',
                    'timeout': int(timeout),
                },
                'capture_raw_io': True,
                'max_retries': 2,
                'timeout': timeout,
            },
            request_args={
                'model': model,
                'max_tokens': 4096,
                'thinking': {'type': 'adaptive'},
                'reasoning_effort': 'high',
            },
        )
    )


def _provider() -> DockerRuntimeProvider:
    return DockerRuntimeProvider(
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


def _write_artifacts(
    *,
    account: SQLiteModelAccount,
    tracker_result: dict[str, Any],
    provider_calls: list[dict[str, Any]],
    agent_output: dict[str, Any],
    initial_png: bytes,
    final_png: bytes,
) -> None:
    output_dir = Path(os.environ.get(_OUTPUT_ENV, 'outputs/vertex_gui_e2e')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'account_db_env': _ACCOUNT_DB_ENV,
        'account_run_id': account.run_id,
        'client': 'vertex',
        'provider': 'vertex',
        'protocol': 'anthropic',
        'model': os.environ.get('NEXUS_VERTEX_ANTHROPIC_E2E_MODEL', 'claude-opus-4-8'),
        'thinking': {'type': 'adaptive'},
        'tools': [_NATIVE_TOOL],
        'runtime_provider': 'docker',
        'max_iterations': 4,
    }
    (output_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'agent_output.json').write_text(
        json.dumps(agent_output, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (output_dir / 'chat_tracker.json').write_text(
        json.dumps(tracker_result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (output_dir / 'provider_calls.json').write_text(
        json.dumps(provider_calls, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (output_dir / 'initial.png').write_bytes(initial_png)
    (output_dir / 'final.png').write_bytes(final_png)


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.gui
async def test_real_vertex_anthropic_main_agent_completes_four_round_docker_gui(
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _client(sqlite_model_account)
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(
        runtime_provider=_provider(),
        openai_clients={'main': client},
        run_id='real-vertex-anthropic-main-agent-gui-four-round',
        agent_name='main_agent',
    )

    with processor_context([tracker, provider_capture]):
        async with context:
            runtime = await context.get_runtime('main')
            started = await runtime.gui_start()
            assert started.success, started.error
            launched = await runtime.gui_launch_app('xterm -title NexusRealVertexE2E -geometry 86x24+80+70')
            assert launched.success, launched.error
            await asyncio.sleep(1)
            initial = await runtime.gui_screenshot()
            assert initial.success, initial.error
            assert initial.image

            command = f"printf %s '{_GUI_MARKER}' > {_GUI_FILE}"
            output = await run_main_agent(
                MainAgentInput(
                    system_prompt=(
                        'You are executing a strict native Anthropic computer-tool protocol. '
                        'Return exactly one computer action in each of the first three model responses. '
                        'Response 1: call computer with action=left_click and coordinate=[220,170]. '
                        f'Response 2: call computer with action=type and text exactly: {command} '
                        'Response 3: call computer with action=key and text=ENTER. '
                        f'Response 4: use no tool and reply exactly {_GUI_MARKER}.'
                    ),
                    user_prompt=[
                        {'type': 'text', 'text': 'Execute the four-response native GUI protocol now.'},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/{initial.format};base64,{initial.image}'},
                        },
                    ],
                    tools=[_TOOL_NAME],
                    tool_schema_overrides=[_NATIVE_TOOL],
                    post_iteration_hook=(
                        'tests.integration.gui_agent.test_openai_responses_real_gui_four_round'
                        '::_real_gui_screenshot_observation'
                    ),
                    max_iterations=4,
                    single_tool_timeout=60,
                    inference_time_timeout=float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300')) * 4,
                    observation_time_timeout=180,
                )
            )
            verification = await runtime.run_command(f'cat {_GUI_FILE}')
            final = await runtime.gui_screenshot()
            assert final.success, final.error
            assert final.image

    assert output.exit_status == 'finished'
    assert output.iterations == 4
    assert output.final_answer is not None
    assert output.final_answer.strip() == _GUI_MARKER
    assert verification.return_code == 0
    assert (verification.stdout or '').strip() == _GUI_MARKER

    actions = context.tool_state[_ACTION_STATE]
    assert actions[:2] == [
        {'action': 'left_click', 'coordinate': [220, 170]},
        {'action': 'type', 'text': command},
    ]
    assert actions[2]['action'] == 'key'
    assert actions[2]['text'].lower() in {'enter', 'return'}

    tracker_result = tracker.get_result()
    assert 'calls' not in tracker_result
    calls = provider_capture.get_result()
    assert len(calls) == 4
    assert [call['provider'] for call in calls] == ['vertex'] * 4
    assert [call['protocol'] for call in calls] == ['anthropic'] * 4
    assert all(call['status']['code'] == 'OK' for call in calls)
    assert all(call['provider_io']['attempts'][-1]['status'] == 'success' for call in calls)
    assert all(call['provider_io']['request']['tools'] == [_NATIVE_TOOL] for call in calls)
    assert all(call['provider_io']['request']['thinking'] == {'type': 'adaptive'} for call in calls)

    for request_index, call in enumerate(calls[1:], start=1):
        messages = call['provider_io']['request']['messages']
        assistant_blocks = [
            block
            for message in messages
            if message['role'] == 'assistant' and isinstance(message.get('content'), list)
            for block in message['content']
        ]
        replayed_signatures = {block.get('signature') for block in assistant_blocks if block.get('signature')}
        replayed_tool_uses = {block.get('id') for block in assistant_blocks if block.get('type') == 'tool_use'}
        tool_results = [
            block
            for message in messages
            if message['role'] == 'user' and isinstance(message.get('content'), list)
            for block in message['content']
            if block.get('type') == 'tool_result'
        ]
        for previous_call in calls[:request_index]:
            for block in previous_call['provider_io']['response']['content']:
                if block.get('type') == 'thinking' and block.get('signature'):
                    assert block['signature'] in replayed_signatures
                if block.get('type') == 'tool_use':
                    assert block['id'] in replayed_tool_uses
                    assert any(result.get('tool_use_id') == block['id'] for result in tool_results)

    serialized_trace = json.dumps({'chat': tracker_result, 'provider_calls': calls}, ensure_ascii=False)
    assert sqlite_model_account.credential not in serialized_trace
    _write_artifacts(
        account=sqlite_model_account,
        tracker_result=tracker_result,
        provider_calls=calls,
        agent_output=output.model_dump(mode='json'),
        initial_png=base64.b64decode(initial.image),
        final_png=base64.b64decode(final.image),
    )
