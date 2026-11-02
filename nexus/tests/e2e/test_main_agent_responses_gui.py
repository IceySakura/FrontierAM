from __future__ import annotations

import asyncio
import base64
import json
import os

from pathlib import Path

import pytest

from nexus.core import OpenAIConfig
from nexus.core.context import ExecutionContext
from nexus.extensions.agents.main.agent import MainAgentInput, run_main_agent
from nexus.extensions.openai_clients.gateway_routing import merge_gateway_routing
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
from nexus.runtime.providers.docker import DockerRuntimeProvider
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.e2e._llm_account import AccountLookupError, SQLiteModelAccount, load_sqlite_model_account
from tests.integration.gui_agent.test_openai_responses_real_gui_four_round import (
    _GUI_ACTIONS_STATE,
    _REAL_COMPUTER_TOOL,
)
from tests.provider_call_capture import ProviderCallCaptureProcessor


_ACCOUNT_DB_ENV = 'NEXUS_LLM_E2E_ACCOUNT_DB'
_ACCOUNT_RUN_ID_ENV = 'NEXUS_LLM_E2E_ACCOUNT_RUN_ID'
_RUN_ENV = 'NEXUS_RUN_REAL_RESPONSES_GUI_E2E'
_OUTPUT_ENV = 'NEXUS_RESPONSES_GUI_E2E_OUTPUT'
_GUI_FILE = '/tmp/nexus_real_responses_gui_four_round.txt'
_GUI_MARKER = 'NEXUS_REAL_RESPONSES_GUI_FOUR_ROUND_OK'


@pytest.fixture(scope='module')
def sqlite_model_account() -> SQLiteModelAccount:
    if os.getenv(_RUN_ENV) != '1':
        pytest.skip(f'set {_RUN_ENV}=1 to run the real Responses Docker GUI E2E')
    raw_path = os.environ.get(_ACCOUNT_DB_ENV)
    if not raw_path:
        pytest.fail(f'{_ACCOUNT_DB_ENV} is required when {_RUN_ENV}=1')
    assert raw_path is not None
    try:
        return load_sqlite_model_account(raw_path, run_id=os.environ.get(_ACCOUNT_RUN_ID_ENV))
    except AccountLookupError as error:
        pytest.fail(str(error))


def _client(account: SQLiteModelAccount) -> OpenAIResponsesClient:
    model = os.environ.get('NEXUS_RESPONSES_E2E_MODEL', 'gpt-5.6-sol')
    timeout = float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300'))
    routed_api_key = merge_gateway_routing(
        account.credential,
        {
            'provider': 'openai',
            'model': model,
            'timeout': int(timeout),
        },
    )
    return OpenAIResponsesClient(
        OpenAIConfig(
            client_args={
                'api_key': routed_api_key,
                'base_url': account.base_url,
                'capture_raw_io': True,
                'ensure_v1_path': True,
                'max_retries': 2,
                'timeout': timeout,
            },
            request_args={
                'model': model,
                'max_output_tokens': 2048,
                'store': False,
                'include': ['reasoning.encrypted_content'],
                'reasoning': {'effort': 'high', 'summary': 'auto'},
                'text': {'verbosity': 'low'},
                'parallel_tool_calls': False,
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
    tracker_result: dict,
    provider_calls: list[dict],
    agent_output: dict,
    initial_png: bytes,
    final_png: bytes,
) -> None:
    output_dir = Path(os.environ.get(_OUTPUT_ENV, 'outputs/responses_gui_e2e')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'account_db_env': _ACCOUNT_DB_ENV,
        'account_run_id': account.run_id,
        'client': 'openai_responses',
        'provider': 'openai',
        'protocol': 'responses',
        'model': os.environ.get('NEXUS_RESPONSES_E2E_MODEL', 'gpt-5.6-sol'),
        'store': False,
        'include': ['reasoning.encrypted_content'],
        'tools': [{'type': 'computer'}],
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
async def test_real_responses_main_agent_completes_four_round_docker_gui(
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _client(sqlite_model_account)
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(
        runtime_provider=_provider(),
        openai_clients={'main': client},
        run_id='real-responses-main-agent-gui-four-round',
        agent_name='main_agent',
    )

    with processor_context([tracker, provider_capture]):
        async with context:
            runtime = await context.get_runtime('main')
            started = await runtime.gui_start()
            assert started.success, started.error
            launched = await runtime.gui_launch_app('xterm -title NexusRealResponsesE2E -geometry 86x24+80+70')
            assert launched.success, launched.error
            await asyncio.sleep(1)
            initial = await runtime.gui_screenshot()
            assert initial.success, initial.error
            assert initial.image

            command = f"printf %s '{_GUI_MARKER}' > {_GUI_FILE}"
            output = await run_main_agent(
                MainAgentInput(
                    system_prompt=(
                        'You are executing a strict real Responses computer-tool protocol. '
                        'Return exactly one computer action in each of the first three model responses, never batch '
                        'actions, and never request an extra screenshot or wait. '
                        'Response 1: click the visible terminal at x=220, y=170 with the left button. '
                        f'Response 2: type exactly this text into the terminal: {command} '
                        'Response 3: press only ENTER. '
                        f'Response 4: use no tool and reply exactly {_GUI_MARKER}.'
                    ),
                    user_prompt=[
                        {'type': 'text', 'text': 'Execute the four-response GUI protocol now.'},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/{initial.format};base64,{initial.image}'},
                        },
                    ],
                    tools=[_REAL_COMPUTER_TOOL],
                    tool_schema_overrides=[{'type': 'computer'}],
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

    tracker_result = tracker.get_result()
    assert 'calls' not in tracker_result
    calls = provider_capture.get_result()
    _write_artifacts(
        account=sqlite_model_account,
        tracker_result=tracker_result,
        provider_calls=calls,
        agent_output=output.model_dump(mode='json'),
        initial_png=base64.b64decode(initial.image),
        final_png=base64.b64decode(final.image),
    )
    assert output.exit_status == 'finished', (
        f'exit_status={output.exit_status!r}, iterations={output.iterations}, final_answer={output.final_answer!r}'
    )
    assert output.iterations == 4
    assert output.final_answer is not None
    assert output.final_answer.strip() == _GUI_MARKER
    assert verification.return_code == 0
    assert verification.stdout is not None
    assert verification.stdout.strip() == _GUI_MARKER

    actions = context.tool_state[_GUI_ACTIONS_STATE]
    assert [action['type'] for action in actions] == ['click', 'type', 'keypress']
    assert {key: actions[0].get(key) for key in ('type', 'x', 'y', 'button')} == {
        'type': 'click',
        'x': 220,
        'y': 170,
        'button': 'left',
    }
    assert {key: actions[1].get(key) for key in ('type', 'text')} == {'type': 'type', 'text': command}
    assert actions[2]['keys'] == ['ENTER']

    assert len(calls) == 4
    assert [call['provider'] for call in calls] == ['openai'] * 4
    assert [call['protocol'] for call in calls] == ['responses'] * 4
    assert all(call['status']['code'] == 'OK' for call in calls)
    assert all(call['provider_io']['attempts'][-1]['status'] == 'success' for call in calls)
    assert all(call['provider_io']['request']['tools'] == [{'type': 'computer'}] for call in calls)
    assert all(call['provider_io']['request']['store'] is False for call in calls)
    assert all(call['provider_io']['request']['include'] == ['reasoning.encrypted_content'] for call in calls)

    for request_index, call in enumerate(calls[1:], start=1):
        request_input = call['provider_io']['request']['input']
        previous_reasoning = [
            item
            for previous_call in calls[:request_index]
            for item in previous_call['provider_io']['response']['output']
            if item.get('type') == 'reasoning'
        ]
        assert all(item in request_input for item in previous_reasoning)
        for previous_call in calls[:request_index]:
            computer_call = next(
                output_item
                for output_item in previous_call['provider_io']['response']['output']
                if output_item.get('type') == 'computer_call'
            )
            assert computer_call in request_input
            computer_output = next(
                item
                for item in request_input
                if item.get('type') == 'computer_call_output' and item.get('call_id') == computer_call['call_id']
            )
            image_url = computer_output['output']['image_url']
            assert image_url.startswith('data:image/png;base64,')
            assert len(image_url) > 1_000

    serialized_trace = json.dumps({'chat': tracker_result, 'provider_calls': calls}, ensure_ascii=False)
    assert sqlite_model_account.credential not in serialized_trace
