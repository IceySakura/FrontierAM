from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from nexus.core import OpenAIConfig
from nexus.core.context import ExecutionContext
from nexus.core.openai_client import OpenAICompatibleClient
from nexus.core.types import ExtendedChatCompletion, ExtendedChoice, ExtendedMessage, ExtendedUsage
from nexus.extensions.agents.claude_code.agent import (
    ClaudeCodeInput,
    ClaudeCodeInteractiveInput,
    run_claude_code,
    run_claude_code_interactive,
)
from nexus.extensions.openai_clients.vertex import VertexClient
from nexus.runtime.providers.docker import DockerRuntimeProvider
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider
from nexus.runtime.server.services.anthropic_proxy.models import MessagesResponse
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.e2e._llm_account import AccountLookupError, SQLiteModelAccount, load_sqlite_model_account
from tests.provider_call_capture import ProviderCallCaptureProcessor


_ACCOUNT_DB_ENV = 'NEXUS_LLM_E2E_ACCOUNT_DB'
_ACCOUNT_RUN_ID_ENV = 'NEXUS_LLM_E2E_ACCOUNT_RUN_ID'
_LIVE_TEXT_PROMPT_MARKER = 'Nexus verification case 516'
_LIVE_TEXT_RESULT = '714'
_NATIVE_RESPONSE_MARKER = 'NEXUS_CC_NATIVE_BODY=318'
_FALLBACK_RESPONSE_MARKER = 'NEXUS_CC_FALLBACK_BODY=919'
_FOUR_ROUND_MARKER = 'NEXUS_CC_FOUR_ROUND_OK=428'
_REAL_FOUR_ROUND_MARKER = 'NEXUS_CC_REAL_OPUS_FOUR_ROUND_OK=637'
_REAL_FOUR_ROUND_FILE = '/workspace/nexus_cc_real_opus_four_round_steps.txt'
_REAL_FOUR_ROUND_OUTPUT_ENV = 'NEXUS_CLAUDE_CODE_FOUR_ROUND_OUTPUT'
_INTERACTIVE_TIMEOUT_MARKER = 'NEXUS_CC_INTERACTIVE_TIMEOUT_OK=842'
_INTERACTIVE_FIRST_QUESTION = 'Which implementation should be used first?'
_INTERACTIVE_SECOND_QUESTION = 'Which implementation should be used second?'


class _RecordingVertexClient(VertexClient):
    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)
        self.completions: list[ExtendedChatCompletion] = []
        self.attempts: list[dict[str, Any]] = []
        self.in_flight: set[asyncio.Task[Any]] = set()

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        attempt: dict[str, Any] = {'completion': None, 'error': None}
        self.attempts.append(attempt)
        task = asyncio.current_task()
        if task is not None:
            self.in_flight.add(task)
        try:
            completion = await super()._create_completion(messages=messages, model=model, **kwargs)
        except BaseException as error:
            attempt['error'] = type(error).__name__
            raise
        else:
            attempt['completion'] = completion
            self.completions.append(completion)
            return completion
        finally:
            if task is not None:
                self.in_flight.discard(task)

    async def settle_attempts(self, timeout: float = 5.0) -> None:
        """Finish recorder-owned work after Claude Code and its runtime have exited."""
        tasks = set(self.in_flight)
        if not tasks:
            return

        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class _NativeResponseSentinelClient(OpenAICompatibleClient):
    """Fake host client whose OpenAI fallback text differs from its native body."""

    def __init__(self) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.calls: list[list[Any]] = []

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        self.calls.append(messages)
        native_body = {
            'id': 'msg-native-sentinel',
            'type': 'message',
            'role': 'assistant',
            'model': str(model or 'claude-test'),
            'content': [{'type': 'text', 'text': _NATIVE_RESPONSE_MARKER}],
            'stop_reason': 'end_turn',
            'stop_sequence': None,
            'usage': {'input_tokens': 1, 'output_tokens': 1},
        }
        return ExtendedChatCompletion(
            id='chatcmpl-native-sentinel',
            choices=[
                ExtendedChoice(
                    finish_reason='stop',
                    index=0,
                    message=ExtendedMessage(role='assistant', content=_FALLBACK_RESPONSE_MARKER),
                )
            ],
            created=0,
            model=str(model or 'claude-test'),
            usage=ExtendedUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            provider_specific_fields={
                'native_response': {'protocol': 'anthropic', 'body': native_body},
            },
        )


class _FourRoundNativeClaudeCodeClient(OpenAICompatibleClient):
    """Script four native Anthropic turns while recording CC's returned tool results."""

    def __init__(self) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.calls: list[dict[str, Any]] = []
        self.business_calls: list[dict[str, Any]] = []

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        call_index = len(self.calls) + 1
        is_business_call = bool(kwargs.get('tools'))
        business_index = len(self.business_calls) + 1 if is_business_call else None
        if business_index is not None and business_index > 4:
            raise AssertionError(f'Claude Code made unexpected business provider call {business_index}')

        call = {
            'messages': messages,
            'kwargs': kwargs,
            'kind': 'business' if is_business_call else 'preflight',
            'business_index': business_index,
        }
        self.calls.append(call)
        if is_business_call:
            self.business_calls.append(call)
        model_name = str(model or 'claude-test')
        content: list[dict[str, Any]] = []
        stop_reason = 'end_turn'
        if business_index is None:
            content.append({'type': 'text', 'text': 'English'})
        elif business_index < 4:
            content.extend(
                [
                    {
                        'type': 'thinking',
                        'thinking': f'Execute deterministic Bash step {business_index}.',
                        'signature': f'cc-four-round-signature-{business_index}',
                    },
                    {
                        'type': 'tool_use',
                        'id': f'cc-four-round-tool-{business_index}',
                        'name': 'Bash',
                        'input': {
                            'command': (
                                f"printf '%s\\n' 'cc-round-{business_index}' >> /workspace/nexus_cc_four_round_steps.txt"
                            ),
                            'description': f'Record deterministic round {business_index}',
                        },
                    },
                ]
            )
            stop_reason = 'tool_use'
        else:
            content.append({'type': 'text', 'text': _FOUR_ROUND_MARKER})

        native_body = {
            'id': f'msg-cc-four-round-{call_index}',
            'type': 'message',
            'role': 'assistant',
            'model': model_name,
            'content': content,
            'stop_reason': stop_reason,
            'stop_sequence': None,
            'usage': {'input_tokens': call_index, 'output_tokens': call_index},
        }
        provider_call = {
            'provider': 'scripted',
            'protocol': 'anthropic',
            'model': model_name,
            'endpoint': 'in-process://claude-code-four-round',
            'kind': call['kind'],
            'business_index': business_index,
            'request': {'messages': messages},
            'response': native_body,
            'attempts': [{'attempt': 1, 'status': 'success'}],
        }
        completion = ExtendedChatCompletion(
            id=f'chatcmpl-cc-four-round-{call_index}',
            choices=[
                ExtendedChoice(
                    finish_reason='tool_calls' if business_index is not None and business_index < 4 else 'stop',
                    index=0,
                    message=ExtendedMessage(role='assistant', content=_FALLBACK_RESPONSE_MARKER),
                )
            ],
            created=call_index,
            model=model_name,
            usage=ExtendedUsage(
                prompt_tokens=call_index,
                completion_tokens=call_index,
                total_tokens=call_index * 2,
            ),
            provider_specific_fields={
                'native_response': {'protocol': 'anthropic', 'body': native_body},
                'provider_call': provider_call,
            },
        )
        self._record_provider_call(provider_call)
        return completion


class _InteractiveHookTimeoutClient(OpenAICompatibleClient):
    """Script two AskUserQuestion calls so the PTY timeout path is deterministic."""

    def __init__(self) -> None:
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self.calls: list[dict[str, Any]] = []
        self.business_calls: list[dict[str, Any]] = []

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        is_business_call = bool(kwargs.get('tools'))
        business_index = len(self.business_calls) + 1 if is_business_call else None
        if business_index is not None and business_index > 3:
            raise AssertionError(f'Claude Code made unexpected interactive provider call {business_index}')

        call = {
            'messages': messages,
            'kwargs': kwargs,
            'kind': 'business' if is_business_call else 'preflight',
            'business_index': business_index,
        }
        self.calls.append(call)
        if is_business_call:
            self.business_calls.append(call)

        content: list[dict[str, Any]]
        stop_reason = 'end_turn'
        if business_index is None:
            content = [{'type': 'text', 'text': 'English'}]
        elif business_index in (1, 2):
            question = _INTERACTIVE_FIRST_QUESTION if business_index == 1 else _INTERACTIVE_SECOND_QUESTION
            content = [
                {
                    'type': 'tool_use',
                    'id': f'cc-interactive-question-{business_index}',
                    'name': 'AskUserQuestion',
                    'input': {
                        'questions': [
                            {
                                'question': question,
                                'header': 'Choice',
                                'options': [
                                    {'label': 'Alpha', 'description': 'Use the Alpha implementation.'},
                                    {'label': 'Beta', 'description': 'Use the Beta implementation.'},
                                ],
                                'multiSelect': False,
                            }
                        ]
                    },
                }
            ]
            stop_reason = 'tool_use'
        else:
            content = [{'type': 'text', 'text': _INTERACTIVE_TIMEOUT_MARKER}]

        model_name = str(model or 'claude-test')
        native_body = {
            'id': f'msg-cc-interactive-{len(self.calls)}',
            'type': 'message',
            'role': 'assistant',
            'model': model_name,
            'content': content,
            'stop_reason': stop_reason,
            'stop_sequence': None,
            'usage': {'input_tokens': len(self.calls), 'output_tokens': len(self.calls)},
        }
        return ExtendedChatCompletion(
            id=f'chatcmpl-cc-interactive-{len(self.calls)}',
            choices=[
                ExtendedChoice(
                    finish_reason='tool_calls' if stop_reason == 'tool_use' else 'stop',
                    index=0,
                    message=ExtendedMessage(role='assistant', content=_FALLBACK_RESPONSE_MARKER),
                )
            ],
            created=len(self.calls),
            model=model_name,
            usage=ExtendedUsage(
                prompt_tokens=len(self.calls),
                completion_tokens=len(self.calls),
                total_tokens=len(self.calls) * 2,
            ),
            provider_specific_fields={'native_response': {'protocol': 'anthropic', 'body': native_body}},
        )


@pytest.fixture
def sqlite_model_account() -> SQLiteModelAccount:
    raw_path = os.environ.get(_ACCOUNT_DB_ENV)
    if not raw_path:
        pytest.skip(f'{_ACCOUNT_DB_ENV} is required for the live Claude Code E2E test')
    assert raw_path is not None
    try:
        return load_sqlite_model_account(raw_path, run_id=os.environ.get(_ACCOUNT_RUN_ID_ENV))
    except AccountLookupError as error:
        pytest.fail(str(error))


def _make_client(account: SQLiteModelAccount) -> _RecordingVertexClient:
    model = os.environ.get('NEXUS_CLAUDE_CODE_E2E_MODEL', 'claude-opus-4-8')
    timeout = float(os.environ.get('NEXUS_CLAUDE_CODE_E2E_TIMEOUT', '600'))
    return _RecordingVertexClient(
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


def _make_provider(timeout: float) -> DockerRuntimeProvider | GongfengRuntimeProvider:
    provider_name = os.environ.get('NEXUS_CLAUDE_CODE_E2E_RUNTIME_PROVIDER', 'docker').strip().lower()
    if provider_name == 'gongfeng':
        api_base_url = os.environ.get('SWE_RUNTIME_BASE_URL', '').strip()
        token = (
            os.environ.get('SWE_RUNTIME_TOKEN', '').strip()
            or os.environ.get('GONGFENG_RUNTIME_TOKEN', '').strip()
            or os.environ.get('GONGFENG_TOKEN', '').strip()
        )
        if not api_base_url or not token:
            pytest.skip(
                'Gongfeng Claude Code E2E requires SWE_RUNTIME_BASE_URL and '
                'SWE_RUNTIME_TOKEN (or GONGFENG_RUNTIME_TOKEN/GONGFENG_TOKEN)'
            )
        return GongfengRuntimeProvider(
            api_base_url=api_base_url,
            token=token,
            image=os.environ.get(
                'NEXUS_CLAUDE_CODE_E2E_IMAGE',
                'mirrors.tencent.com/hunyuan_yanguan/nexus-ubuntu-session:latest',
            ),
            runtime_image=os.environ.get('NEXUS_CLAUDE_CODE_E2E_RUNTIME_IMAGE', 'current'),
            resource_spec={'cpu': 2, 'memory': 10},
            request_timeout=timeout,
            queue_wait_max=min(timeout, 600),
            purpose='test',
            print_port_mappings=False,
        )
    if provider_name != 'docker':
        raise ValueError(f'Unsupported NEXUS_CLAUDE_CODE_E2E_RUNTIME_PROVIDER: {provider_name!r}')
    if shutil.which('docker') is None:
        pytest.skip('Docker is unavailable; set NEXUS_CLAUDE_CODE_E2E_RUNTIME_PROVIDER=gongfeng for remote E2E')
    return DockerRuntimeProvider(
        image=os.environ.get(
            'NEXUS_CLAUDE_CODE_E2E_IMAGE',
            'mirrors.tencent.com/hunyuan_yanguan/nexus-ubuntu-session:latest',
        ),
        runtime_image=os.environ.get('NEXUS_CLAUDE_CODE_E2E_RUNTIME_IMAGE', 'nexus-runtime:1.5.2'),
        force_rebuild=False,
        print_port_mappings=False,
        request_timeout=timeout,
    )


def _safe_runtime_diagnostic(output_success: bool, exit_code: int, stderr: str) -> str:
    return f'Claude Code success={output_success}, exit_code={exit_code}, stderr_chars={len(stderr)}'


def _provider_call(completion: ExtendedChatCompletion) -> dict[str, Any]:
    provider_call = (completion.provider_specific_fields or {}).get('provider_call')
    assert isinstance(provider_call, dict)
    return provider_call


def _write_real_four_round_artifacts(
    *,
    account: SQLiteModelAccount,
    output: dict[str, Any],
    provider_calls: list[dict[str, Any]],
    tracker_result: dict[str, Any],
) -> None:
    output_dir = Path(os.environ.get(_REAL_FOUR_ROUND_OUTPUT_ENV, 'outputs/claude_code_real_opus_four_round')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'account_db_env': _ACCOUNT_DB_ENV,
        'account_run_id': account.run_id,
        'agent': 'claude_code',
        'client': 'vertex',
        'provider': 'vertex',
        'protocol': 'anthropic',
        'model': os.environ.get('NEXUS_CLAUDE_CODE_E2E_MODEL', 'claude-opus-4-8'),
        'thinking': {'type': 'adaptive'},
        'reasoning_effort': 'high',
        'runtime_provider': 'docker',
        'expected_provider_rounds': 4,
    }
    artifacts = {
        'config.json': config,
        'agent_output.json': output,
        'provider_calls.json': provider_calls,
        'chat_tracker.json': tracker_result,
    }
    serialized = json.dumps(artifacts, ensure_ascii=False)
    assert account.credential not in serialized
    for filename, body in artifacts.items():
        (output_dir / filename).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')


@pytest.mark.e2e
async def test_claude_code_uses_native_anthropic_response_through_proxy() -> None:
    timeout = float(os.environ.get('NEXUS_CLAUDE_CODE_E2E_TIMEOUT', '600'))
    client = _NativeResponseSentinelClient()
    context = ExecutionContext(
        runtime_provider=_make_provider(timeout),
        runtime_configs={'main': {'worker_venv': True}},
        openai_clients={'main': client},
        run_id=f'cc-native-sentinel-{uuid.uuid4().hex[:8]}',
        agent_name='claude_code',
    )

    async with context:
        output = await run_claude_code(
            ClaudeCodeInput(
                user_prompt=f'Reply with exactly {_NATIVE_RESPONSE_MARKER} and no other text. Do not use tools.',
                working_dir='/workspace',
                timeout=timeout,
                version=os.environ.get('NEXUS_CLAUDE_CODE_E2E_VERSION', 'latest'),
                conversion_log_level='WARNING',
                thinking_preset='bedrock-claude',
            )
        )

    assert output.success, _safe_runtime_diagnostic(output.success, output.exit_code, output.stderr)
    assert output.exit_code == 0
    assert output.output.strip() == _NATIVE_RESPONSE_MARKER
    assert _FALLBACK_RESPONSE_MARKER not in output.output
    assert client.calls, 'Claude Code did not make a host LLM call through AnthropicProxy and RSIOLLMBridge'


@pytest.mark.e2e
async def test_claude_code_native_anthropic_tools_complete_four_provider_rounds() -> None:
    timeout = float(os.environ.get('NEXUS_CLAUDE_CODE_E2E_TIMEOUT', '600'))
    client = _FourRoundNativeClaudeCodeClient()
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(
        runtime_provider=_make_provider(timeout),
        runtime_configs={'main': {'worker_venv': True}},
        openai_clients={'main': client},
        run_id=f'cc-native-four-round-{uuid.uuid4().hex[:8]}',
        agent_name='claude_code',
    )

    with processor_context([tracker, provider_capture]):
        async with context:
            output = await run_claude_code(
                ClaudeCodeInput(
                    user_prompt=(
                        'Follow all provider tool requests. After all steps, print only the exact marker '
                        f'{_FOUR_ROUND_MARKER}.'
                    ),
                    working_dir='/workspace',
                    timeout=timeout,
                    version=os.environ.get('NEXUS_CLAUDE_CODE_E2E_VERSION', 'latest'),
                    conversion_log_level='WARNING',
                    thinking_preset='bedrock-claude',
                )
            )

    assert output.success, _safe_runtime_diagnostic(output.success, output.exit_code, output.stderr)
    assert output.exit_code == 0
    assert output.output.strip() == _FOUR_ROUND_MARKER
    assert _FALLBACK_RESPONSE_MARKER not in output.output
    assert len(client.business_calls) == 4
    assert any(call['kind'] == 'preflight' for call in client.calls)

    for request_index, call in enumerate(client.business_calls[1:], start=1):
        messages = call['messages']
        assistant_messages = [message for message in messages if message.get('role') == 'assistant']
        assert assistant_messages[-1]['thinking_blocks'][-1]['signature'] == (
            f'cc-four-round-signature-{request_index}'
        )
        tool_messages = [message for message in messages if message.get('role') == 'tool']
        assert tool_messages[-1]['tool_call_id'] == f'cc-four-round-tool-{request_index}'

    assert 'calls' not in tracker.get_result()
    calls = provider_capture.get_result()
    assert len(calls) == len(client.calls)
    business_calls = [call for call in calls if call['provider_io']['kind'] == 'business']
    preflight_calls = [call for call in calls if call['provider_io']['kind'] == 'preflight']
    assert len(business_calls) == 4
    assert preflight_calls
    assert [call['provider'] for call in calls] == ['scripted'] * len(calls)
    assert [call['protocol'] for call in calls] == ['anthropic'] * len(calls)
    assert all(call['provider_io']['attempts'] == [{'attempt': 1, 'status': 'success'}] for call in calls)


@pytest.mark.e2e
async def test_claude_code_interactive_hook_timeout_keeps_the_real_pty_session() -> None:
    """A real Claude Code TUI survives one timed-out Hook and answers the next one."""
    timeout = float(os.environ.get('NEXUS_CLAUDE_CODE_E2E_TIMEOUT', '600'))
    hook_wait_timeout = float(os.environ.get('NEXUS_CLAUDE_CODE_INTERACTIVE_E2E_HOOK_WAIT', '3'))
    hook_command_timeout = float(
        os.environ.get(
            'NEXUS_CLAUDE_CODE_INTERACTIVE_E2E_HOOK_COMMAND_TIMEOUT',
            str(max(hook_wait_timeout + 10, 30)),
        )
    )
    client = _InteractiveHookTimeoutClient()
    requests: list[Any] = []
    first_provider_cancelled = False

    async def get_user_input(request: Any) -> str:
        nonlocal first_provider_cancelled
        requests.append(request)
        if len(requests) == 1:
            try:
                await asyncio.sleep(hook_wait_timeout * 3)
            except asyncio.CancelledError:
                first_provider_cancelled = True
                raise
            return 'This answer arrived too late and must not be used.'
        return 'Alpha'

    context = ExecutionContext(
        runtime_provider=_make_provider(timeout),
        runtime_configs={'main': {'worker_venv': True}},
        openai_clients={'main': client},
        run_id=f'cc-interactive-timeout-{uuid.uuid4().hex[:8]}',
        agent_name='claude_code_interactive',
    )

    async with context:
        output = await run_claude_code_interactive(
            ClaudeCodeInteractiveInput(
                user_prompt='Follow the scripted provider tool calls and finish with its exact marker.',
                get_user_input=get_user_input,
                working_dir='/workspace',
                timeout=timeout,
                max_turns=1,
                version=os.environ.get('NEXUS_CLAUDE_CODE_E2E_VERSION', 'latest'),
                conversion_log_level='WARNING',
                thinking_preset='bedrock-claude',
                hook_wait_timeout=hook_wait_timeout,
                hook_command_timeout=hook_command_timeout,
            )
        )

    assert output.success, _safe_runtime_diagnostic(output.success, output.exit_code, output.stderr)
    assert output.exit_code == 0
    assert output.output.strip() == _INTERACTIVE_TIMEOUT_MARKER
    assert first_provider_cancelled is True
    assert len(requests) == 2
    assert [request.hook['input']['questions'][0]['question'] for request in requests] == [
        _INTERACTIVE_FIRST_QUESTION,
        _INTERACTIVE_SECOND_QUESTION,
    ]
    assert [request.metadata['hook_timeout_seconds'] for request in requests] == [
        hook_wait_timeout,
        hook_wait_timeout,
    ]

    requested = [event for event in output.events if event.get('type') == 'pty_interaction_requested']
    resolved = [event for event in output.events if event.get('type') == 'pty_interaction_resolved']
    assert [event['interaction_id'] for event in requested] == [
        'cc-interactive-question-1',
        'cc-interactive-question-2',
    ]
    assert [event['interaction_id'] for event in resolved] == [
        'cc-interactive-question-1',
        'cc-interactive-question-2',
    ]
    assert resolved[0]['resolution'] in {'unattended', 'expired'}
    assert resolved[1]['resolution'] == 'user'
    assert {event['turn_id'] for event in requested + resolved} == {1}
    assert len({event['claude_pid'] for event in requested + resolved}) == 1
    assert len(client.business_calls) == 3


@pytest.mark.e2e
@pytest.mark.llm
async def test_claude_code_text_through_anthropic_proxy(sqlite_model_account: SQLiteModelAccount) -> None:
    client = _make_client(sqlite_model_account)
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    timeout = float(os.environ.get('NEXUS_CLAUDE_CODE_E2E_TIMEOUT', '600'))
    context = ExecutionContext(
        runtime_provider=_make_provider(timeout),
        runtime_configs={'main': {'worker_venv': True}},
        openai_clients={'main': client},
        run_id=f'cc-text-e2e-{uuid.uuid4().hex[:8]}',
        agent_name='claude_code',
    )

    with processor_context([tracker]):
        async with context:
            output = await run_claude_code(
                ClaudeCodeInput(
                    user_prompt=(
                        f'{_LIVE_TEXT_PROMPT_MARKER}: what is (37 * 19) + 11? '
                        'Give the numerical result in one short sentence. Do not use tools.'
                    ),
                    working_dir='/workspace',
                    timeout=timeout,
                    version=os.environ.get('NEXUS_CLAUDE_CODE_E2E_VERSION', 'latest'),
                    conversion_log_level='WARNING',
                    thinking_preset='bedrock-claude',
                )
            )

    await client.settle_attempts()
    assert output.success, _safe_runtime_diagnostic(output.success, output.exit_code, output.stderr)
    assert output.exit_code == 0
    # The deterministic sentinel E2E proves exact proxy selection; this live request checks provider reachability.
    assert _LIVE_TEXT_RESULT in output.output
    assert client.completions, 'Claude Code did not reach the host Vertex client through the Anthropic proxy'
    assert all((attempt['completion'] is None) != (attempt['error'] is None) for attempt in client.attempts), (
        'Every Vertex attempt must have exactly one terminal outcome'
    )
    successful_completions = [attempt['completion'] for attempt in client.attempts if attempt['completion'] is not None]
    assert len(client.completions) == len(successful_completions)
    assert Counter(map(id, client.completions)) == Counter(map(id, successful_completions))

    provider_calls = []
    for completion in client.completions:
        native_response = (completion.provider_specific_fields or {}).get('native_response')
        assert isinstance(native_response, dict)
        assert native_response.get('protocol') == 'anthropic'
        native_body = native_response.get('body')
        assert isinstance(native_body, dict)
        MessagesResponse.model_validate(native_body)

        provider_call = (completion.provider_specific_fields or {}).get('provider_call')
        assert provider_call is not None
        assert provider_call['provider'] == 'vertex'
        assert provider_call['protocol'] == 'anthropic'
        assert provider_call['attempts'][-1]['status'] == 'success'
        assert provider_call.get('request')
        assert provider_call.get('response')
        provider_calls.append(provider_call)

    assert all(
        provider_call['request'].get('thinking') == {'type': 'adaptive'}
        and provider_call['request'].get('output_config') == {'effort': 'high'}
        for provider_call in provider_calls
    ), 'Not every successful Claude Code provider call requested adaptive high-effort thinking'

    tracker_result = tracker.get_result()
    assert tracker_result['trajectories'], 'chat_tracker did not record the Claude Code LLM call'
    tracker_json = json.dumps(tracker_result, ensure_ascii=False)
    assert _LIVE_TEXT_PROMPT_MARKER in tracker_json
    assert _LIVE_TEXT_RESULT in tracker_json
    assert sqlite_model_account.credential not in json.dumps(provider_calls, ensure_ascii=False)
    assert sqlite_model_account.credential not in tracker_json


@pytest.mark.e2e
@pytest.mark.llm
async def test_claude_code_real_opus_tools_complete_four_provider_rounds(
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _make_client(sqlite_model_account)
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    timeout = float(os.environ.get('NEXUS_CLAUDE_CODE_E2E_TIMEOUT', '600'))
    context = ExecutionContext(
        runtime_provider=_make_provider(timeout),
        runtime_configs={'main': {'worker_venv': True}},
        openai_clients={'main': client},
        run_id=f'cc-real-opus-four-round-{uuid.uuid4().hex[:8]}',
        agent_name='claude_code',
    )

    commands = [
        f"printf '%s\\n' 'round-1' > {_REAL_FOUR_ROUND_FILE}",
        f"printf '%s\\n' 'round-2' >> {_REAL_FOUR_ROUND_FILE}",
        f"printf '%s\\n' 'round-3' >> {_REAL_FOUR_ROUND_FILE}",
    ]
    protocol = ' '.join(
        [
            'Follow this exact four-response protocol; never combine steps.',
            f'Response 1: call Bash exactly once with command: {commands[0]}',
            f'Response 2, only after receiving response 1 tool_result: call Bash exactly once with command: {commands[1]}',
            f'Response 3, only after receiving response 2 tool_result: call Bash exactly once with command: {commands[2]}',
            'Do not use any other tool and do not add text to responses 1 through 3.',
            f'Response 4, only after receiving response 3 tool_result: use no tool and output exactly {_REAL_FOUR_ROUND_MARKER}',
        ]
    )

    with processor_context([tracker, provider_capture]):
        async with context:
            runtime = await context.get_runtime('main')
            output = await run_claude_code(
                ClaudeCodeInput(
                    user_prompt=protocol,
                    working_dir='/workspace',
                    timeout=timeout,
                    version=os.environ.get('NEXUS_CLAUDE_CODE_E2E_VERSION', 'latest'),
                    conversion_log_level='WARNING',
                    thinking_preset='bedrock-claude',
                )
            )
            verification = await runtime.run_command(f'cat {_REAL_FOUR_ROUND_FILE}')

    await client.settle_attempts()
    assert output.success, _safe_runtime_diagnostic(output.success, output.exit_code, output.stderr)
    assert output.exit_code == 0
    assert output.output.strip() == _REAL_FOUR_ROUND_MARKER
    assert verification.return_code == 0
    assert verification.stdout is not None
    assert verification.stdout.splitlines() == ['round-1', 'round-2', 'round-3']

    provider_calls = [_provider_call(completion) for completion in client.completions]
    business_calls = [call for call in provider_calls if call.get('request', {}).get('tools')]
    assert len(business_calls) == 4
    assert all(call['provider'] == 'vertex' and call['protocol'] == 'anthropic' for call in business_calls)
    assert all(call['attempts'][-1]['status'] == 'success' for call in business_calls)
    assert all(
        call['request'].get('thinking') == {'type': 'adaptive'}
        and call['request'].get('output_config') == {'effort': 'high'}
        for call in business_calls
    )

    tool_use_ids: list[str] = []
    for index, call in enumerate(business_calls[:3]):
        response_content = call['response']['content']
        thinking_blocks = [item for item in response_content if item.get('type') == 'thinking']
        tool_uses = [item for item in response_content if item.get('type') == 'tool_use']
        assert all(item.get('signature') for item in thinking_blocks)
        assert len(tool_uses) == 1
        assert tool_uses[0]['name'] == 'Bash'
        assert tool_uses[0]['input']['command'] == commands[index]
        tool_use_ids.append(tool_uses[0]['id'])

    final_content = business_calls[-1]['response']['content']
    assert not any(item.get('type') == 'tool_use' for item in final_content)
    assert any(
        item.get('type') == 'text' and item.get('text', '').strip() == _REAL_FOUR_ROUND_MARKER for item in final_content
    )

    for request_index, tool_use_id in enumerate(tool_use_ids, start=1):
        request_messages = business_calls[request_index]['request']['messages']
        previous_thinking = [
            item for item in business_calls[request_index - 1]['response']['content'] if item.get('type') == 'thinking'
        ]
        replayed_thinking = [
            item
            for message in request_messages
            if message.get('role') == 'assistant' and isinstance(message.get('content'), list)
            for item in message['content']
            if item.get('type') == 'thinking'
        ]
        for thinking in previous_thinking:
            assert thinking in replayed_thinking
        tool_results = [
            item
            for message in request_messages
            if message.get('role') == 'user' and isinstance(message.get('content'), list)
            for item in message['content']
            if item.get('type') == 'tool_result'
        ]
        assert any(item.get('tool_use_id') == tool_use_id for item in tool_results)

    tracker_result = tracker.get_result()
    assert 'calls' not in tracker_result
    tracked_business_calls = [
        call for call in provider_capture.get_result() if call.get('provider_io', {}).get('request', {}).get('tools')
    ]
    assert len(tracked_business_calls) == 4
    assert sqlite_model_account.credential not in json.dumps(provider_calls, ensure_ascii=False)
    assert sqlite_model_account.credential not in json.dumps(tracker_result, ensure_ascii=False)
    _write_real_four_round_artifacts(
        account=sqlite_model_account,
        output=output.model_dump(mode='json'),
        provider_calls=business_calls,
        tracker_result=tracker_result,
    )
