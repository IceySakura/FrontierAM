from __future__ import annotations

import asyncio
import json
import os

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pydantic import BaseModel, Field

from nexus.core import (
    OpenAICompatibleClient,
    OpenAIConfig,
    ToolContext,
    ToolResponse,
    ToolType,
    tool_registry,
)
from nexus.core.context import ExecutionContext
from nexus.core.types import ExtendedChatCompletion, ExtendedMessage
from nexus.extensions.agents.main.agent import (
    MainAgentInput,
    PostIterationHookContext,
    PostIterationHookResult,
    run_main_agent,
)
from nexus.extensions.openai_clients.gateway_routing import merge_gateway_routing
from nexus.extensions.openai_clients.litellm import LiteLLMClient
from nexus.extensions.openai_clients.openai_responses import OpenAIResponsesClient
from nexus.extensions.openai_clients.vertex import VertexClient
from nexus.extensions.openai_clients.vertex_anthropic import VertexAnthropicClient
from nexus.trace import processor_context
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from tests.e2e._llm_account import AccountLookupError, SQLiteModelAccount, load_sqlite_model_account
from tests.provider_call_capture import ProviderCallCaptureProcessor


_ACCOUNT_DB_ENV = 'NEXUS_LLM_E2E_ACCOUNT_DB'
_ACCOUNT_RUN_ID_ENV = 'NEXUS_LLM_E2E_ACCOUNT_RUN_ID'
_PROTOCOL_TOOL = 'e2e/main_agent_protocol_step'
_PROTOCOL_TOOL_STATE = 'main_agent_protocol_step'
_PROTOCOL_FINAL = 'NEXUS_MAIN_AGENT_FOUR_ROUND_OK'
_OUTPUT_ENV = 'NEXUS_MAIN_AGENT_E2E_OUTPUT'
_XHIGH_OUTPUT_ENV = 'NEXUS_RESPONSES_XHIGH_E2E_OUTPUT'
_LIVE_ROUNDS = [
    ('Compute (37 * 19) + 11.', 'NEXUS_LIVE_FOUR_ROUND_1=714'),
    ('Compute 7^222 modulo 1000.', 'NEXUS_LIVE_FOUR_ROUND_2=49'),
    ('Compute the sum of k^2 for integer k from 1 through 100.', 'NEXUS_LIVE_FOUR_ROUND_3=338350'),
    (
        'Add the three numerical answers from the prior rounds: 714, 49, and 338350.',
        'NEXUS_LIVE_FOUR_ROUND_4=339113',
    ),
]


@dataclass(frozen=True)
class _Case:
    name: str
    provider: str
    protocol: str
    thinking_kind: str
    prompt: str
    expected_answer: str


_CASES = [
    _Case(
        'vertex_anthropic',
        'vertex',
        'anthropic',
        'signed_thinking',
        'Compute (37 * 19) + 11. Return exactly NEXUS_TEXT_THINKING_OK=714',
        'NEXUS_TEXT_THINKING_OK=714',
    ),
    _Case(
        'vertex_openai',
        'vertex',
        'openai',
        'openai_thinking',
        'Compute 7^222 modulo 1000. Return exactly NEXUS_TEXT_THINKING_OK=49',
        'NEXUS_TEXT_THINKING_OK=49',
    ),
    _Case(
        'openai_responses',
        'openai',
        'responses',
        'reasoning_items',
        'Compute 7^222 modulo 1000. Return exactly NEXUS_TEXT_THINKING_OK=49',
        'NEXUS_TEXT_THINKING_OK=49',
    ),
    _Case(
        'litellm_openai',
        'litellm',
        'openai',
        'openai_thinking',
        'Compute 7^222 modulo 1000. Return exactly NEXUS_TEXT_THINKING_OK=49',
        'NEXUS_TEXT_THINKING_OK=49',
    ),
    _Case(
        'litellm_anthropic',
        'litellm',
        'anthropic',
        'litellm_signed_thinking',
        'Compute (37 * 19) + 11. Return exactly NEXUS_TEXT_THINKING_OK=714',
        'NEXUS_TEXT_THINKING_OK=714',
    ),
]
_STREAM_CASES = [case for case in _CASES if case.name != 'openai_responses']


def _records_provider_calls(case: _Case) -> bool:
    return not case.name.startswith('litellm_')


class _ProtocolStepInput(BaseModel):
    step: int = Field(ge=1, le=3)
    checksum: int


@tool_registry.register(
    _PROTOCOL_TOOL,
    meta={
        'name': 'protocol_step',
        'description': 'Advance the four-round E2E protocol by exactly one sequential step.',
        'tool_type': ToolType.LOCAL,
    },
)
async def _protocol_step_tool(params: _ProtocolStepInput, context: ToolContext) -> ToolResponse:
    expected = int(context.tool_state.get(_PROTOCOL_TOOL_STATE, 0)) + 1
    if params.step != expected:
        raise ValueError(f'Expected protocol step {expected}, got {params.step}')
    expected_checksums = {1: 49, 2: 714, 3: 830}
    if params.checksum != expected_checksums[expected]:
        raise ValueError(f'Protocol step {expected} checksum is incorrect')
    next_instruction = {
        1: 'Call protocol_step once with step=2 and checksum=714 on the next turn.',
        2: 'Call protocol_step once with step=3 and checksum=830 on the next turn.',
        3: f'Do not call any tool again. Reply exactly {_PROTOCOL_FINAL} on the next turn.',
    }[expected]
    return ToolResponse(
        output=[{'type': 'text', 'text': f'Protocol step {expected} accepted. {next_instruction}'}],
        meta={'success': True, 'step': expected},
        context_updates={_PROTOCOL_TOOL_STATE: expected},
    )


async def _force_responses_four_round_tool_choice(
    _context: object,
    hook_context: PostIterationHookContext,
) -> PostIterationHookResult:
    iteration = hook_context.iteration
    return PostIterationHookResult(next_completion_kwargs={'tool_choice': 'required' if iteration < 3 else 'none'})


async def _force_litellm_openai_four_round_tool_choice(
    _context: object,
    hook_context: PostIterationHookContext,
) -> PostIterationHookResult:
    iteration = hook_context.iteration
    tool_choice: str | dict[str, object] = (
        {'type': 'function', 'function': {'name': 'protocol_step'}} if iteration < 3 else 'none'
    )
    return PostIterationHookResult(next_completion_kwargs={'tool_choice': tool_choice})


class _RecordingClient(OpenAICompatibleClient):
    def __init__(self, inner: OpenAICompatibleClient) -> None:
        super().__init__(inner.config)
        self.inner = inner
        self.completions: list[ExtendedChatCompletion] = []

    async def _create_completion(
        self,
        messages: list[Any],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        completion = await self.inner._create_completion(messages=messages, model=model, **kwargs)  # noqa: SLF001
        self.completions.append(completion)
        return completion


def _completion_shapes(completions: list[ExtendedChatCompletion]) -> list[dict[str, Any]]:
    return [
        {
            'finish_reason': completion.choices[0].finish_reason,
            'has_content': bool(completion.choices[0].message.content),
            'has_refusal': bool(completion.choices[0].message.refusal),
            'tool_names': [tool_call.function.name for tool_call in completion.choices[0].message.tool_calls or []],
            'reasoning_types': [item.get('type') for item in completion.choices[0].message.reasoning_items or []],
            'response_output_types': [
                item.get('type')
                for item in (completion.choices[0].message.provider_specific_fields or {}).get(
                    'responses_output_items', []
                )
            ],
        }
        for completion in completions
    ]


@pytest.fixture(scope='module')
def sqlite_model_account() -> SQLiteModelAccount:
    raw_path = os.environ.get(_ACCOUNT_DB_ENV)
    if not raw_path:
        pytest.skip(f'{_ACCOUNT_DB_ENV} is required for live text-thinking E2E tests')
    assert raw_path is not None
    try:
        return load_sqlite_model_account(raw_path, run_id=os.environ.get(_ACCOUNT_RUN_ID_ENV))
    except AccountLookupError as error:
        pytest.fail(str(error))


@pytest.fixture(autouse=True)
async def _drain_litellm_logging_worker() -> AsyncIterator[None]:
    yield
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    await GLOBAL_LOGGING_WORKER.flush()
    await GLOBAL_LOGGING_WORKER.stop()


def _model(case: _Case, account: SQLiteModelAccount) -> str:
    if case.name in {'vertex_openai', 'openai_responses', 'litellm_openai'}:
        if case.name == 'vertex_openai':
            return os.environ.get('NEXUS_VERTEX_OPENAI_E2E_MODEL', 'gpt-5.6-sol')
        if case.name == 'litellm_openai':
            return os.environ.get('NEXUS_LITELLM_OPENAI_E2E_MODEL', 'gpt-5.6-sol')
        return os.environ.get('NEXUS_RESPONSES_E2E_MODEL', 'gpt-5.6-sol')
    return os.environ.get('NEXUS_VERTEX_ANTHROPIC_E2E_MODEL', 'claude-opus-4-8')


def _make_client(
    case: _Case,
    account: SQLiteModelAccount,
    *,
    preserve_cache_task_id: bool = False,
) -> OpenAICompatibleClient:
    model = _model(case, account)
    timeout = float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300'))
    if case.name == 'vertex_anthropic':
        return VertexClient(
            OpenAIConfig(
                client_args={
                    'protocol': 'anthropic',
                    'api_key': account.api_key if preserve_cache_task_id else account.credential,
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

    if case.name == 'vertex_openai':
        gateway_provider = os.environ.get('NEXUS_VERTEX_OPENAI_E2E_GATEWAY_PROVIDER', 'openai').strip()
        gateway_routing: dict[str, str | int] = {
            'provider': gateway_provider,
            'model': model,
            'timeout': int(timeout),
        }
        gateway_api = os.environ.get('NEXUS_VERTEX_OPENAI_E2E_GATEWAY_API', '').strip()
        if gateway_api:
            gateway_routing['api'] = gateway_api
        client_args: dict[str, Any] = {
            'protocol': 'openai',
            'api_key': account.credential,
            'base_url': os.environ.get('NEXUS_VERTEX_OPENAI_E2E_BASE_URL', f'{account.base_url}/v1'),
            'capture_raw_io': True,
            'max_retries': 2,
            'timeout': timeout,
        }
        if gateway_provider:
            client_args['gateway_routing'] = gateway_routing
        return VertexClient(
            OpenAIConfig(
                client_args=client_args,
                request_args={
                    'model': model,
                    'max_completion_tokens': 1024,
                    'reasoning_effort': 'high',
                },
            )
        )

    if case.name == 'litellm_openai':
        routed_api_key = merge_gateway_routing(
            account.credential,
            {
                'provider': 'openai',
                'model': model,
                'timeout': int(timeout),
            },
        )
        return LiteLLMClient(
            OpenAIConfig(
                client_args={
                    'api_key': routed_api_key,
                    'base_url': f'{account.base_url}/v1',
                    'max_retries': 2,
                    'timeout': timeout,
                },
                request_args={
                    'model': f'openai/{model}',
                    'max_completion_tokens': 1024,
                    'reasoning_effort': 'high',
                },
            )
        )

    if case.name == 'litellm_anthropic':
        routed_api_key = merge_gateway_routing(
            account.credential,
            {
                'provider': 'anthropic',
                'model': model,
                'timeout': int(timeout),
            },
        )
        return LiteLLMClient(
            OpenAIConfig(
                client_args={
                    'api_key': routed_api_key,
                    'base_url': account.base_url,
                    'max_retries': 2,
                    'timeout': timeout,
                },
                request_args={
                    'model': f'anthropic/{model}',
                    'max_tokens': 4096,
                    'thinking': {'type': 'adaptive'},
                    'reasoning_effort': 'high',
                },
            )
        )

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
                'max_output_tokens': 1024,
                'store': False,
                'include': ['reasoning.encrypted_content'],
                'reasoning': {'effort': 'high', 'summary': 'auto'},
                'text': {'verbosity': 'low'},
            },
        )
    )


def _write_main_agent_artifacts(
    *,
    case: _Case,
    account: SQLiteModelAccount,
    client: OpenAICompatibleClient,
    tracker_result: dict[str, Any],
    provider_calls: list[dict[str, Any]],
    output: Any,
) -> None:
    output_dir = Path(os.environ.get(_OUTPUT_ENV, 'outputs/main_agent_client_matrix')).resolve() / case.name
    output_dir.mkdir(parents=True, exist_ok=True)
    client_args = client.config.client_args
    config = {
        'account_db_env': _ACCOUNT_DB_ENV,
        'account_run_id': account.run_id,
        'client': case.name,
        'provider': case.provider,
        'protocol': case.protocol,
        'model': _model(case, account),
        'base_url': str(client_args.get('base_url') or ''),
        'capture_raw_io': client_args.get('capture_raw_io'),
        'request_args': client.config.request_args,
        'max_iterations': 4,
        'tool': 'protocol_step',
    }
    agent_output = output.model_dump(mode='json')
    serialized = json.dumps(
        {
            'config': config,
            'tracker': tracker_result,
            'provider_calls': provider_calls,
            'agent_output': agent_output,
        },
        ensure_ascii=False,
    )
    assert account.credential not in serialized
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


def _write_xhigh_artifacts(
    *,
    account: SQLiteModelAccount,
    completion: ExtendedChatCompletion,
    tracker_result: dict[str, Any],
    provider_calls: list[dict[str, Any]],
) -> None:
    output_dir = Path(os.environ.get(_XHIGH_OUTPUT_ENV, 'outputs/responses_xhigh_e2e')).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'account_db_env': _ACCOUNT_DB_ENV,
        'account_run_id': account.run_id,
        'client': 'openai_responses',
        'provider': 'openai',
        'protocol': 'responses',
        'model': os.environ.get('NEXUS_RESPONSES_E2E_MODEL', 'gpt-5.6-sol'),
        'reasoning': {'effort': 'xhigh', 'summary': 'auto'},
        'store': False,
    }
    bodies = {
        'config.json': config,
        'completion.json': completion.model_dump(mode='json'),
        'chat_tracker.json': tracker_result,
        'provider_calls.json': provider_calls,
    }
    serialized = json.dumps(bodies, ensure_ascii=False)
    assert account.credential not in serialized
    for filename, body in bodies.items():
        (output_dir / filename).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')


def _assert_all_history_replayed(
    case: _Case,
    provider_calls: list[dict[str, Any]],
    *,
    require_thinking: bool = True,
) -> None:
    for request_index in range(1, len(provider_calls)):
        request = provider_calls[request_index]['request']
        prior_calls = provider_calls[:request_index]

        if case.thinking_kind == 'reasoning_items':
            request_input = request['input']
            prior_output_items = [item for prior_call in prior_calls for item in prior_call['response']['output']]
            replayed_call_ids = {
                item['call_id'] for item in request_input if item.get('type') == 'function_call_output'
            }
            for item in prior_output_items:
                if item.get('type') in {'reasoning', 'function_call'}:
                    assert item in request_input
                if item.get('type') == 'function_call':
                    assert item['call_id'] in replayed_call_ids
            if request.get('tools'):
                tool = request['tools'][0]
                assert tool['type'] == 'function'
                assert tool['name'] == 'protocol_step'
                assert 'function' not in tool
            continue

        messages = request['messages']
        if case.thinking_kind == 'litellm_signed_thinking':
            expected_signatures = {
                block.get('signature')
                for prior_call in prior_calls
                for choice in prior_call['response']['choices']
                for block in choice['message'].get('thinking_blocks') or []
                if block.get('signature')
            }
            replayed_signatures = {
                block.get('signature')
                for message in messages
                if message['role'] == 'assistant'
                for block in message.get('thinking_blocks') or []
                if block.get('signature')
            }
            expected_call_ids = {
                tool_call['id']
                for prior_call in prior_calls
                for choice in prior_call['response']['choices']
                for tool_call in choice['message'].get('tool_calls') or []
            }
            assistant_call_ids = {
                tool_call['id']
                for message in messages
                if message['role'] == 'assistant'
                for tool_call in message.get('tool_calls') or []
            }
            tool_result_ids = {message.get('tool_call_id') for message in messages if message['role'] == 'tool'}
            if require_thinking:
                assert expected_signatures
            assert replayed_signatures == expected_signatures
            assert assistant_call_ids == expected_call_ids
            assert tool_result_ids == expected_call_ids
            continue

        if case.thinking_kind == 'signed_thinking':
            assistant_blocks = [
                block
                for message in messages
                if message['role'] == 'assistant' and isinstance(message.get('content'), list)
                for block in message['content']
            ]
            user_blocks = [
                block
                for message in messages
                if message['role'] == 'user' and isinstance(message.get('content'), list)
                for block in message['content']
            ]
            replayed_signatures = {block.get('signature') for block in assistant_blocks if block.get('signature')}
            replayed_tool_uses = {block.get('id') for block in assistant_blocks if block.get('type') == 'tool_use'}
            replayed_tool_results = {
                block.get('tool_use_id') for block in user_blocks if block.get('type') == 'tool_result'
            }
            for prior_call in prior_calls:
                for block in prior_call['response']['content']:
                    if block.get('type') == 'thinking' and block.get('signature'):
                        assert block['signature'] in replayed_signatures
                    if block.get('type') == 'tool_use':
                        assert block['id'] in replayed_tool_uses
                        assert block['id'] in replayed_tool_results
            continue

        expected_call_ids = {
            tool_call['id']
            for prior_call in prior_calls
            for choice in prior_call['response']['choices']
            for tool_call in choice['message'].get('tool_calls') or []
        }
        assistant_call_ids = {
            tool_call['id']
            for message in messages
            if message['role'] == 'assistant'
            for tool_call in message.get('tool_calls') or []
        }
        tool_result_ids = {message.get('tool_call_id') for message in messages if message['role'] == 'tool'}
        assert assistant_call_ids == expected_call_ids
        assert tool_result_ids == expected_call_ids


def _assert_thinking(case: _Case, message: ExtendedMessage, completion: ExtendedChatCompletion) -> None:
    if case.thinking_kind in {'signed_thinking', 'litellm_signed_thinking'}:
        blocks = message.thinking_blocks or []
        assert blocks, 'Vertex Anthropic response did not contain thinking_blocks'
        assert any(block.type == 'thinking' and getattr(block, 'signature', None) for block in blocks)
        return

    reasoning_tokens = 0
    if completion.usage and completion.usage.completion_tokens_details:
        reasoning_tokens = completion.usage.completion_tokens_details.reasoning_tokens or 0

    if case.thinking_kind == 'reasoning_items':
        items = message.reasoning_items or []
        assert any(item.get('type') == 'reasoning' for item in items) or reasoning_tokens > 0
        return

    assert message.thinking_blocks or message.reasoning_items or message.reasoning_content or reasoning_tokens > 0


@pytest.mark.e2e
@pytest.mark.llm
async def test_legacy_vertex_anthropic_alias_reaches_real_provider(
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    base_client = _make_client(_CASES[0], sqlite_model_account)
    with pytest.warns(DeprecationWarning, match='vertex_anthropic.*deprecated'):
        client = VertexAnthropicClient(base_client.config)
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()

    with processor_context([tracker, provider_capture]):
        completion = await client.create_completion(
            messages=[
                {'role': 'system', 'content': 'Follow the requested output format exactly.'},
                {
                    'role': 'user',
                    'content': 'Compute (37 * 19) + 11. Reply exactly NEXUS_VERTEX_ANTHROPIC_ALIAS_OK=714.',
                },
            ]
        )

    assert completion.choices[0].message.content is not None
    assert completion.choices[0].message.content.strip() == 'NEXUS_VERTEX_ANTHROPIC_ALIAS_OK=714'
    assert 'calls' not in tracker.get_result()
    calls = provider_capture.get_result()
    assert len(calls) == 1
    call = calls[0]
    assert call['provider'] == 'vertex'
    assert call['protocol'] == 'anthropic'
    assert call['provider_io']['request']['thinking'] == {'type': 'adaptive'}
    assert call['provider_io']['request']['output_config'] == {'effort': 'high'}
    assert call['provider_io']['attempts'][-1]['status'] == 'success'
    assert sqlite_model_account.credential not in json.dumps(call, ensure_ascii=False)


@pytest.mark.e2e
@pytest.mark.llm
async def test_vertex_anthropic_mid_conversation_system_preserves_cached_prefix(
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _make_client(
        _CASES[0],
        sqlite_model_account,
        preserve_cache_task_id=True,
    )
    stable_system = 'Follow the requested output format exactly. ' + 'This is stable prompt-cache validation context. ' * 500
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': stable_system},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Reply exactly NEXUS_VERTEX_CACHE_ROUND_1.',
                    'cache_control': {'type': 'ephemeral'},
                }
            ],
        },
    ]

    first = await client.create_completion(
        messages=messages,
        max_tokens=64,
        thinking=None,
        reasoning_effort=None,
    )
    assert first.choices[0].message.content is not None
    assert 'NEXUS_VERTEX_CACHE_ROUND_1' in first.choices[0].message.content
    await asyncio.sleep(2)
    messages.extend([
        first.choices[0].message.model_dump(mode='json', exclude_none=True),
        {'role': 'user', 'content': 'Reply exactly NEXUS_VERTEX_CACHE_ROUND_2.'},
        {'role': 'system', 'content': 'Runtime reminder: keep the reply concise.'},
    ])

    second_attempts: list[ExtendedChatCompletion] = []
    for attempt in range(3):
        if attempt:
            await asyncio.sleep(2)
        second = await client.create_completion(
            messages=messages,
            max_tokens=64,
            thinking=None,
            reasoning_effort=None,
        )
        second_attempts.append(second)
        if second.usage is not None and second.usage.cache_read_input_tokens > 0:
            break

    assert second.choices[0].message.content is not None
    assert 'NEXUS_VERTEX_CACHE_ROUND_2' in second.choices[0].message.content
    assert first.usage is not None
    assert first.usage.cache_creation_input_tokens + first.usage.cache_read_input_tokens > 0
    assert second.usage is not None
    assert second.usage.cache_read_input_tokens > 0

    first_call = (first.provider_specific_fields or {})['provider_call']
    second_calls = [(completion.provider_specific_fields or {})['provider_call'] for completion in second_attempts]
    assert first_call['request']['system'] == stable_system
    assert all(call['request']['system'] == stable_system for call in second_calls)
    assert all(
        call['request']['messages'][-1]
        == {
            'role': 'system',
            'content': 'Runtime reminder: keep the reply concise.',
        }
        for call in second_calls
    )
    assert sqlite_model_account.credential not in json.dumps([first_call, *second_calls], ensure_ascii=False)


@pytest.mark.e2e
@pytest.mark.llm
async def test_openai_responses_accepts_real_xhigh_reasoning_effort(
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _make_client(_CASES[2], sqlite_model_account)
    client.config.request_args['reasoning'] = {'effort': 'xhigh', 'summary': 'auto'}
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()

    with processor_context([tracker, provider_capture]):
        completion = await client.create_completion(
            messages=[
                {'role': 'system', 'content': 'Solve the arithmetic and follow the requested concise format.'},
                {'role': 'user', 'content': 'What is (37 * 19) + 11? Include the numerical answer 714.'},
            ]
        )

    message = completion.choices[0].message
    assert message.content is not None
    assert '714' in message.content
    reasoning_tokens = 0
    if completion.usage and completion.usage.completion_tokens_details:
        reasoning_tokens = completion.usage.completion_tokens_details.reasoning_tokens or 0
    assert any(item.get('type') == 'reasoning' for item in message.reasoning_items or []) or reasoning_tokens > 0

    tracker_result = tracker.get_result()
    assert 'calls' not in tracker_result
    calls = provider_capture.get_result()
    assert len(calls) == 1
    call = calls[0]
    assert call['provider'] == 'openai'
    assert call['protocol'] == 'responses'
    assert call['provider_io']['request']['reasoning'] == {'effort': 'xhigh', 'summary': 'auto'}
    assert call['provider_io']['attempts'][-1]['status'] == 'success'
    assert call['provider_io']['response']['status'] == 'completed'
    assert sqlite_model_account.credential not in json.dumps(call, ensure_ascii=False)
    _write_xhigh_artifacts(
        account=sqlite_model_account,
        completion=completion,
        tracker_result=tracker_result,
        provider_calls=calls,
    )


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.parametrize('case', _CASES, ids=lambda case: case.name)
async def test_main_agent_text_thinking(
    case: _Case,
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _RecordingClient(_make_client(case, sqlite_model_account))
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')

    async with context:
        output = await run_main_agent(
            MainAgentInput(
                system_prompt=(
                    'You are a helpful math assistant. Work out the result carefully and follow the requested output '
                    'format exactly.'
                ),
                user_prompt=case.prompt,
                tools=[],
                max_iterations=1,
                inference_time_timeout=float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300')),
            )
        )

    assert output.exit_status == 'finished'
    assert output.iterations == 1
    assert output.final_answer is not None
    assert case.expected_answer in output.final_answer
    assert len(client.completions) == 1

    completion = client.completions[0]
    message = completion.choices[0].message
    _assert_thinking(case, message, completion)

    assistant_message = output.main_trajectory[-1]
    assert assistant_message['role'] == 'assistant'
    assert case.expected_answer in assistant_message['content']
    if message.thinking_blocks:
        assert assistant_message.get('thinking_blocks')
    if message.reasoning_items:
        assert assistant_message.get('reasoning_items')
    if message.reasoning_content:
        assert assistant_message.get('reasoning_content')

    if _records_provider_calls(case):
        provider_call = (completion.provider_specific_fields or {}).get('provider_call')
        assert provider_call is not None
        assert provider_call['provider'] == case.provider
        assert provider_call['protocol'] == case.protocol
        assert provider_call['attempts'][-1]['status'] == 'success'
        assert provider_call.get('request')
        assert provider_call.get('response')
        assert not provider_call['request'].get('tools')
        assert sqlite_model_account.credential not in json.dumps(provider_call, ensure_ascii=False)


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.parametrize('case', _STREAM_CASES, ids=lambda case: case.name)
async def test_main_agent_real_stream_smoke(
    case: _Case,
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _RecordingClient(_make_client(case, sqlite_model_account))
    if case.name.startswith('litellm_'):
        client.config.client_args['stream'] = True
    else:
        client.config.request_args['stream'] = True
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()

    with processor_context([tracker, provider_capture]):
        async with ExecutionContext(openai_clients={'main': client}, agent_name='main_agent'):
            output = await run_main_agent(
                MainAgentInput(
                    system_prompt='Follow the requested output format exactly.',
                    user_prompt='Compute (37 * 19) + 11. Reply exactly NEXUS_REAL_STREAM_OK=714 and no other text.',
                    tools=[],
                    max_iterations=1,
                    inference_time_timeout=float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300')),
                )
            )

    assert output.exit_status == 'finished'
    assert output.iterations == 1
    assert output.final_answer is not None
    assert output.final_answer.strip() == 'NEXUS_REAL_STREAM_OK=714'

    tracker_result = tracker.get_result()
    assert tracker_result['trajectories']
    assert 'calls' not in tracker_result
    assert len(client.completions) == 1
    calls = provider_capture.get_result()
    if not _records_provider_calls(case):
        assert calls == []
        completion_text = client.completions[0].choices[0].message.content
        assert completion_text is not None
        assert completion_text.strip() == 'NEXUS_REAL_STREAM_OK=714'
        return

    assert len(calls) == 1
    call = calls[0]
    assert call['provider'] == case.provider
    assert call['protocol'] == case.protocol
    assert call['provider_io']['attempts'][-1]['status'] == 'success'
    assert call['provider_io']['request']['stream'] is True
    response = call['provider_io']['response']
    if case.name == 'vertex_anthropic':
        assert response['raw_sse_events']
    else:
        assert response['stream_chunks']
    request = call['provider_io']['request']
    if case.protocol == 'anthropic':
        assert request['thinking'] == {'type': 'adaptive'}
    else:
        assert request['reasoning_effort'] == 'high'
    assert sqlite_model_account.credential not in json.dumps(call, ensure_ascii=False)


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.parametrize('case', _CASES, ids=lambda case: case.name)
async def test_openai_client_text_thinking_replays_four_live_rounds(
    case: _Case,
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _make_client(case, sqlite_model_account)
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    messages: list[dict[str, Any]] = [
        {
            'role': 'system',
            'content': (
                'You are validating a four-round protocol. Think carefully, preserve prior reasoning state, and reply '
                'with exactly the marker requested in the latest user message.'
            ),
        }
    ]
    completions: list[ExtendedChatCompletion] = []

    with processor_context([tracker, provider_capture]):
        async with ExecutionContext(openai_clients={'main': client}, agent_name='main_agent'):
            for round_index, (calculation, expected) in enumerate(_LIVE_ROUNDS, start=1):
                messages.append({
                    'role': 'user',
                    'content': (
                        f'This is protocol round {round_index} of 4. {calculation} '
                        f'Reply with exactly {expected} and no other text.'
                    ),
                })
                completion = await client.create_completion(messages=messages)
                completions.append(completion)
                message = completion.choices[0].message
                assert message.content is not None
                assert message.content.strip() == expected
                messages.append(message.model_dump(mode='json', exclude_none=True))

    assert len(completions) == 4
    first_message = completions[0].choices[0].message
    _assert_thinking(case, first_message, completions[0])
    if case.thinking_kind == 'reasoning_items':
        assert any(item.get('encrypted_content') for item in first_message.reasoning_items or [])

    tracker_result = tracker.get_result()
    assert tracker_result['trajectories']
    assert 'calls' not in tracker_result
    calls = provider_capture.get_result()
    if not _records_provider_calls(case):
        assert calls == []
        assert sqlite_model_account.credential not in json.dumps(tracker_result, ensure_ascii=False)
        return

    assert len(calls) == 4
    assert [call['provider'] for call in calls] == [case.provider] * 4
    assert [call['protocol'] for call in calls] == [case.protocol] * 4
    assert all(call['provider_io']['attempts'][-1]['status'] == 'success' for call in calls)
    assert all(call['provider_io'].get('request') and call['provider_io'].get('response') for call in calls)
    _assert_all_history_replayed(case, [call['provider_io'] for call in calls])
    second_request = calls[1]['provider_io']['request']
    if case.thinking_kind == 'signed_thinking':
        assistant_messages = [message for message in second_request['messages'] if message['role'] == 'assistant']
        thinking_blocks = [
            block
            for message in assistant_messages
            for block in message['content']
            if isinstance(block, dict) and block.get('type') == 'thinking'
        ]
        assert any(block.get('signature') for block in thinking_blocks)
    elif case.thinking_kind == 'litellm_signed_thinking':
        assistant_messages = [message for message in second_request['messages'] if message['role'] == 'assistant']
        thinking_blocks = [
            block
            for message in assistant_messages
            for block in message.get('thinking_blocks') or []
            if isinstance(block, dict) and block.get('type') == 'thinking'
        ]
        assert any(block.get('signature') for block in thinking_blocks)
    elif case.thinking_kind == 'reasoning_items':
        reasoning_items = [item for item in second_request['input'] if item.get('type') == 'reasoning']
        assert any(item.get('encrypted_content') for item in reasoning_items)
    else:
        assistant_messages = [message for message in second_request['messages'] if message['role'] == 'assistant']
        if first_message.reasoning_content:
            assert any(
                message.get('reasoning_content') == first_message.reasoning_content for message in assistant_messages
            )
        else:
            reasoning_tokens = 0
            if completions[0].usage and completions[0].usage.completion_tokens_details:
                reasoning_tokens = completions[0].usage.completion_tokens_details.reasoning_tokens or 0
            assert reasoning_tokens > 0
            assert any(message.get('content') == first_message.content for message in assistant_messages)
    assert sqlite_model_account.credential not in json.dumps(calls, ensure_ascii=False)


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.parametrize('case', _CASES, ids=lambda case: case.name)
async def test_main_agent_tool_loop_replays_four_live_rounds(
    case: _Case,
    sqlite_model_account: SQLiteModelAccount,
) -> None:
    client = _RecordingClient(_make_client(case, sqlite_model_account))
    post_iteration_hook = None
    if case.name == 'openai_responses':
        client.config.request_args['tool_choice'] = 'required'
        post_iteration_hook = 'tests.e2e.test_main_agent_text_thinking::_force_responses_four_round_tool_choice'
    elif case.name == 'litellm_openai':
        client.config.request_args['tool_choice'] = {
            'type': 'function',
            'function': {'name': 'protocol_step'},
        }
        post_iteration_hook = 'tests.e2e.test_main_agent_text_thinking::_force_litellm_openai_four_round_tool_choice'
    if case.name == 'vertex_openai':
        # The gateway's Chat Completions route rejects reasoning_effort together with function tools for GPT 5.6.
        # The Responses case below is the supported thinking+tools path; keep this case focused on Vertex's OpenAI
        # adapter and ordinary Main Agent tool replay.
        client.config.request_args['reasoning_effort'] = 'none'
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    provider_capture = ProviderCallCaptureProcessor()
    context = ExecutionContext(openai_clients={'main': client}, agent_name='main_agent')
    first_step_instruction = 'On round 1, call protocol_step exactly once with step=1 and checksum=49. '

    with processor_context([tracker, provider_capture]):
        async with context:
            output = await run_main_agent(
                MainAgentInput(
                    system_prompt=(
                        'You are validating a strict four-round Main Agent protocol. '
                        f'{first_step_instruction}'
                        'After each successful tool result, call it exactly once with the next integer. '
                        f'After step 3, call no tool and reply exactly {_PROTOCOL_FINAL}. '
                        'Never skip, repeat, combine, or parallelize steps.'
                    ),
                    user_prompt='Begin the strict four-round protocol now.',
                    tools=[_PROTOCOL_TOOL],
                    post_iteration_hook=post_iteration_hook,
                    max_iterations=4,
                    single_tool_timeout=30,
                    inference_time_timeout=float(os.environ.get('NEXUS_LLM_E2E_TIMEOUT', '300')) * 4,
                    observation_time_timeout=120,
                )
            )

    assert output.exit_status == 'finished', json.dumps(_completion_shapes(client.completions), ensure_ascii=False)
    assert output.iterations == 4
    assert output.final_answer is not None
    assert output.final_answer.strip() == _PROTOCOL_FINAL
    assert context.tool_state[_PROTOCOL_TOOL_STATE] == 3

    tracker_result = tracker.get_result()
    assert tracker_result['trajectories']
    assert 'calls' not in tracker_result
    assert len(client.completions) == 4
    calls = provider_capture.get_result()
    first_completion = client.completions[0]
    if not _records_provider_calls(case):
        assert calls == []
        if case.name != 'litellm_openai':
            _assert_thinking(case, first_completion.choices[0].message, first_completion)
        messages = output.main_trajectory
        assistant_call_ids = {
            tool_call['id']
            for message in messages
            if message['role'] == 'assistant'
            for tool_call in message.get('tool_calls') or []
        }
        tool_result_ids = {message.get('tool_call_id') for message in messages if message['role'] == 'tool'}
        assert len(assistant_call_ids) == 3
        assert tool_result_ids == assistant_call_ids
        assert sqlite_model_account.credential not in json.dumps(tracker_result, ensure_ascii=False)
        _write_main_agent_artifacts(
            case=case,
            account=sqlite_model_account,
            client=client,
            tracker_result=tracker_result,
            provider_calls=calls,
            output=output,
        )
        return

    assert len(calls) == 4
    assert [call['provider'] for call in calls] == [case.provider] * 4
    assert [call['protocol'] for call in calls] == [case.protocol] * 4
    assert all(call['status']['code'] == 'OK' for call in calls)
    assert all(call['provider_io']['attempts'][-1]['status'] == 'success' for call in calls)
    assert all(call['provider_io'].get('request') and call['provider_io'].get('response') for call in calls)
    _assert_all_history_replayed(case, [call['provider_io'] for call in calls])

    if case.name not in {'vertex_openai', 'openai_responses', 'litellm_openai'}:
        _assert_thinking(case, first_completion.choices[0].message, first_completion)
    second_request = calls[1]['provider_io']['request']
    if case.thinking_kind == 'signed_thinking':
        assistant_messages = [message for message in second_request['messages'] if message['role'] == 'assistant']
        thinking_blocks = [
            block
            for message in assistant_messages
            for block in message['content']
            if isinstance(block, dict) and block.get('type') == 'thinking'
        ]
        assert any(block.get('signature') for block in thinking_blocks)
        assert any(
            isinstance(block, dict) and block.get('type') == 'tool_use'
            for message in assistant_messages
            for block in message['content']
        )
        assert any(
            isinstance(block, dict) and block.get('type') == 'tool_result'
            for message in second_request['messages']
            if message['role'] == 'user' and isinstance(message.get('content'), list)
            for block in message['content']
        )
    elif case.thinking_kind == 'litellm_signed_thinking':
        assistant_messages = [message for message in second_request['messages'] if message['role'] == 'assistant']
        assert any(
            block.get('signature')
            for message in assistant_messages
            for block in message.get('thinking_blocks') or []
            if isinstance(block, dict) and block.get('type') == 'thinking'
        )
        assert any(message.get('tool_calls') for message in assistant_messages)
        assert any(message['role'] == 'tool' for message in second_request['messages'])
    elif case.thinking_kind == 'reasoning_items':
        replayed_reasoning = [item for item in second_request['input'] if item.get('type') == 'reasoning']
        assert all(item.get('encrypted_content') for item in replayed_reasoning)
        assert any(item.get('type') == 'function_call' for item in second_request['input'])
        assert any(item.get('type') == 'function_call_output' for item in second_request['input'])
        assert second_request['tools'][0]['type'] == 'function'
        assert second_request['tools'][0]['name'] == 'protocol_step'
        assert 'function' not in second_request['tools'][0]
    else:
        assistant_messages = [message for message in second_request['messages'] if message['role'] == 'assistant']
        assert any(message.get('tool_calls') for message in assistant_messages)
        if case.name == 'vertex_openai':
            assert second_request['reasoning_effort'] == 'none'
        elif case.name == 'litellm_openai':
            # The gateway accepts the requested effort with tools but may omit reasoning fields from an individual
            # tool-call response. Text and direct-round cases above require real reasoning output for this client.
            assert second_request['reasoning_effort'] == 'high'
        else:
            assert any(message.get('reasoning_content') for message in assistant_messages) or (
                first_completion.usage is not None
                and first_completion.usage.completion_tokens_details is not None
                and (first_completion.usage.completion_tokens_details.reasoning_tokens or 0) > 0
            )
        assert any(message['role'] == 'tool' for message in second_request['messages'])

    assert sqlite_model_account.credential not in json.dumps(calls, ensure_ascii=False)
    _write_main_agent_artifacts(
        case=case,
        account=sqlite_model_account,
        client=client,
        tracker_result=tracker_result,
        provider_calls=calls,
        output=output,
    )
