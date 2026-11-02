import asyncio
import logging
import pickle

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from _fakes import memory_store
from tests.unit.gateway.helpers import latest_node_id

from nexus.core import ContextLengthExceededError
from nexus.gateway.backend import BackendResult, HyTunerServerBackend
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.processor import Processor
from nexus.gateway.session import GatewaySession as _GatewaySession
from nexus.gateway.trajectory_assembler import TrajectoryAssembler


def _gateway_session(*args, **kwargs):
    kwargs.setdefault('scaffold_policy', 'generic')
    kwargs.setdefault('record_store', memory_store(str(args[0])))
    return _GatewaySession(*args, **kwargs)


def _segment_kind(node_list) -> str:
    boundary = node_list.response_nodes[0].segment_boundary
    return boundary.kind if boundary is not None and boundary.kind is not None else 'conversation'


def _response_axis(node_list) -> tuple[list[int], list[int]]:
    token_ids: list[int] = []
    response_mask: list[int] = []
    last_index = node_list.nodes.index(node_list.last_response_node)
    for index, node in enumerate(node_list.nodes[node_list.first_response_node_index : last_index + 1]):
        account = node.token_account
        if account is None or account.token_ids is None:
            continue
        start = account.gen_start if index == 0 else 0
        owned = account.token_ids[start:]
        generated_start = account.gen_start - start
        token_ids.extend(owned)
        response_mask.extend(
            [0] * generated_start + [1] * account.gen_len + [0] * (len(owned) - generated_start - account.gen_len)
        )
    return token_ids, response_mask


class ByteTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del kwargs
        rendered = ''
        for message in messages:
            prefix = {'system': 'S:', 'user': 'U:', 'assistant': 'A:', 'tool': 'T:'}[message['role']]
            content = message.get('content') or ''
            if message['role'] == 'system' and tools:
                names = ','.join(tool['function']['name'] for tool in tools)
                content += f'|tools:{names}'
            rendered += ('\n' if rendered else '') + prefix + content
        if add_generation_prompt:
            rendered += ('\n' if rendered else '') + 'A:'
        return rendered

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return bytes(token_ids).decode()


class RecordingTemplateTokenizer(ByteTokenizer):
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        self.calls.append({'tools': tools, **kwargs})
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **kwargs,
        )


class QueuedBackend:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.prompts = []
        self.execution_requests = []

    async def create_completion(self, request_id, **kwargs):
        del request_id
        self.prompts.append(kwargs['prompt_ids'])
        self.execution_requests.append(kwargs['request_kwargs'])
        text = next(self._responses)
        return BackendResult(token_ids=list(text.encode()), log_probs=[-0.1] * len(text))


class YieldingQueuedBackend(QueuedBackend):
    async def create_completion(self, request_id, **kwargs):
        await asyncio.sleep(0)
        return await super().create_completion(request_id, **kwargs)


class MissingTokenBackend:
    async def create_completion(self, _request_id, **_kwargs):
        return BackendResult(message={'role': 'assistant', 'content': 'wrong lane'})


class WaitReportingBackend:
    async def create_completion(self, _request_id, **_kwargs):
        return BackendResult(token_ids=[ord('a')], extra_fields={'_abort_wait_time': 0.25})


class AbortResumeServerManager:
    def __init__(self) -> None:
        self.first_attempt_finished = asyncio.Event()
        self.resume = asyncio.Event()
        self.attempts = []

    async def generate_tokens(self, **_kwargs):
        self.attempts.append({'token_ids': [ord('h')], 'stop_reason': 'aborted'})
        self.first_attempt_finished.set()
        await self.resume.wait()
        self.attempts.append({'token_ids': [ord('i')], 'stop_reason': 'stop'})
        return SimpleNamespace(
            token_ids=[ord('h'), ord('i')],
            log_probs=[-0.1, -0.2],
            routed_experts=None,
            stop_reason='stop',
            extra_fields={},
        )


class LastAwareTokenizer(ByteTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **kwargs,
        )
        return rendered if add_generation_prompt else rendered + '|last'


class TrailingAssistantTokenizer(ByteTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del tools, kwargs
        rendered = ''
        for message in messages:
            prefix = {'system': 'S:', 'user': 'U:', 'assistant': 'A:', 'tool': 'T:'}[message['role']]
            suffix = '|end' if message['role'] == 'assistant' else ''
            rendered += ('\n' if rendered else '') + prefix + (message.get('content') or '') + suffix
        if add_generation_prompt:
            rendered += ('\n' if rendered else '') + 'A:'
        return rendered


class RetokenizingAssistantTokenizer(TrailingAssistantTokenizer):
    def encode(self, text, add_special_tokens=False):
        token_ids = super().encode(text, add_special_tokens=add_special_tokens)
        marker = b'first|end'
        marker_start = text.encode().find(marker)
        if marker_start >= 0:
            token_ids[marker_start + len(b'first') - 1] = 999
        return token_ids


class ZeroTokenClosureTokenizer(TrailingAssistantTokenizer):
    def encode(self, text, add_special_tokens=False):
        return super().encode(text.replace('|end', ''), add_special_tokens=add_special_tokens)


class BoundaryMergeTokenizer(ByteTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del tools, kwargs
        rendered = ''.join(
            ('|' if message['role'] == 'assistant' else '') + (message.get('content') or '') for message in messages
        )
        return rendered + ('|' if add_generation_prompt else '')

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        token_ids = []
        index = 0
        while index < len(text):
            if text.startswith('ab', index):
                token_ids.append(999)
                index += 2
            else:
                token_ids.extend(text[index].encode())
                index += 1
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return ''.join('ab' if token == 999 else chr(token) for token in token_ids)


class StructuredMergeTokenizer(ByteTokenizer):
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        data = text.encode()
        token_ids = []
        index = 0
        while index < len(data):
            if data[index : index + 2] == b'\nU':
                token_ids.append(999)
                index += 2
            else:
                token_ids.append(data[index])
                index += 1
        return token_ids

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return ''.join('\nU' if token == 999 else chr(token) for token in token_ids)


class AliasedTokenTokenizer(ByteTokenizer):
    def decode(self, token_ids, skip_special_tokens=False):
        aliases = {1: 'x', 2: 'x', 1001: 'a', 1002: 'b', 1003: 'c'}
        if len(token_ids) == 1 and token_ids[0] in aliases:
            return aliases[token_ids[0]]
        return super().decode(token_ids, skip_special_tokens=skip_special_tokens)


class ImplicitSpecialTokenizer(ByteTokenizer):
    def encode(self, text):
        return [777, *text.encode()]


class StrippingTokenizer(ByteTokenizer):
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text.replace('|end', '') if skip_special_tokens else text


class ConditionalAssistantTokenizer(ByteTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        assert not tokenize
        del tools, kwargs
        rendered = ''
        for index, message in enumerate(messages):
            role = message['role']
            content = message.get('content') or ''
            if role == 'assistant':
                if index == len(messages) - 1:
                    content = f'<think></think>{content}'
                rendered += f'A:{content}|end\n'
            else:
                prefix = {'system': 'S:', 'user': 'U:', 'tool': 'T:'}[role]
                rendered += f'{prefix}{content}\n'
        if add_generation_prompt:
            rendered += 'A:'
        return rendered


class HiddenTerminalTokenizer(ByteTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        assert not tokenize
        del tools, kwargs
        rendered = ''
        for message in messages:
            content = message.get('content') or ''
            if message['role'] == 'assistant':
                rendered += f'A:{content}|A\n'
            else:
                rendered += f'U:{content}\n'
        if add_generation_prompt:
            rendered += 'A:'
        return rendered


class ReformattingAssistantTokenizer(ByteTokenizer):
    """History re-render normalizes assistant content inside the block (drops spaces)."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del tools, kwargs
        rendered = ''
        for message in messages:
            prefix = {'system': 'S:', 'user': 'U:', 'assistant': 'A:', 'tool': 'T:'}[message['role']]
            content = message.get('content') or ''
            if message['role'] == 'assistant':
                content = content.replace(' ', '') + '|end'
            rendered += ('\n' if rendered else '') + prefix + content
        if add_generation_prompt:
            rendered += ('\n' if rendered else '') + 'A:'
        return rendered


class InterleavedReasoningTokenizer(ByteTokenizer):
    """Keep reasoning only for assistant messages after the latest user query."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del tools, kwargs
        last_user = max((index for index, message in enumerate(messages) if message['role'] == 'user'), default=-1)
        rendered = ''
        for index, message in enumerate(messages):
            content = message.get('content') or ''
            if message['role'] == 'assistant' and content.startswith('<think>') and '</think>' in content:
                _, answer = content.removeprefix('<think>').split('</think>', 1)
                content = content if index > last_user else answer
            prefix = {'system': 'S:', 'user': 'U:', 'assistant': 'A:', 'tool': 'T:'}[message['role']]
            rendered += ('\n' if rendered else '') + prefix + content
        if add_generation_prompt:
            rendered += ('\n' if rendered else '') + 'A:'
        return rendered


class WireQueuedBackend:
    def __init__(self, responses: list[tuple[str, str]]):
        self._responses = iter(responses)
        self.prompts: list[list[int]] = []

    async def create_completion(self, request_id: str, **kwargs: Any) -> BackendResult:
        del request_id
        self.prompts.append(kwargs['prompt_ids'])
        sampled, wire = next(self._responses)
        return BackendResult(
            token_ids=list(sampled.encode()),
            log_probs=[-0.1] * len(sampled),
            wire_text=wire,
        )


class TokenQueuedBackend:
    def __init__(self, responses: list[tuple[list[int], str]]):
        self._responses = iter(responses)
        self.prompts: list[list[int]] = []

    async def create_completion(self, request_id: str, **kwargs: Any) -> BackendResult:
        del request_id
        self.prompts.append(kwargs['prompt_ids'])
        token_ids, wire_text = next(self._responses)
        return BackendResult(token_ids=token_ids, log_probs=[-0.1] * len(token_ids), wire_text=wire_text)


class RecordingToolParser:
    def __init__(self) -> None:
        self.token_ids: list[int] | None = None

    def extract_tool_calls(self, token_ids: list[int]) -> tuple[None, str, list[dict[str, Any]]]:
        self.token_ids = token_ids
        return (
            None,
            'parsed|end',
            [{'id': 'call-1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}],
        )


class FallbackToolParser:
    def __init__(self, raises: bool) -> None:
        self._raises = raises

    def extract_tool_calls(self, token_ids: list[int]) -> tuple[None, str, list[dict[str, Any]]]:
        del token_ids
        if self._raises:
            raise ValueError('not a tool call')
        return None, 'parser text', []


class RequestToolParser:
    think_start_token = '<think>'
    think_end_token = '</think>'

    def __init__(self) -> None:
        self.request = None

    async def extract_tool_calls(self, token_ids, request=None):
        del token_ids
        self.request = request
        return (
            'raw-tool',
            '<think>reason</think>answer',
            [SimpleNamespace(name='lookup', arguments='{}')],
        )


class StructuredParser:
    generation_prompt = 'A:'

    def inject_tool_prompt(self, messages, tools):
        del tools
        return messages

    def parse(self, messages, *, add_generation_prompt, is_first_msg):
        del is_first_msg
        if len(messages) < 2:
            raise ValueError('Partial histories are not supported')
        text = ''.join(getattr(self, f'parse_{message["role"]}')(message) for message in messages)
        return text + (self.generation_prompt if add_generation_prompt else '')

    def parse_system(self, message):
        return f'S:{message["content"]}\n'

    def parse_user(self, message):
        return f'U:{message["content"]}\n'


class InconsistentStructuredParser(StructuredParser):
    def parse(self, messages, *, add_generation_prompt, is_first_msg):
        del messages, is_first_msg
        return 'opaque' + (self.generation_prompt if add_generation_prompt else '')


class ContextSensitiveParser:
    generation_prompt = '|'

    def inject_tool_prompt(self, messages, tools):
        del tools
        return messages

    def parse(self, messages, *, add_generation_prompt, is_first_msg):
        del is_first_msg
        text = ''.join(getattr(self, f'parse_{message["role"]}')(message) for message in messages)
        if len(messages) > 1 and messages[-1]['role'] == 'assistant':
            text = 'X' + text
        return text + (self.generation_prompt if add_generation_prompt else '')

    def parse_user(self, message):
        return message['content']

    def parse_assistant(self, message):
        return '|' + message['content']

    def parse_tool(self, message):
        return message['content']


class TestWireText:
    async def test_backend_without_wire_text_uses_clean_decode_for_message(self) -> None:
        processor = Processor(QueuedBackend(['answer|end']), tokenizer=StrippingTokenizer())

        result = await processor.create_completion('clean-fallback', [{'role': 'user', 'content': 'answer'}])

        assert result.completion_text == 'answer|end'
        assert result.message == {'role': 'assistant', 'content': 'answer'}

    async def test_tool_parser_receives_provider_wire_tokens(self) -> None:
        parser = RecordingToolParser()
        processor = Processor(
            WireQueuedBackend([('call|end', 'call')]),
            tokenizer=ByteTokenizer(),
            tool_parser=parser,
        )

        result = await processor.create_completion(
            'wire-parser',
            [{'role': 'user', 'content': 'use the tool'}],
            request_kwargs={'tools': [{'type': 'function', 'function': {'name': 'lookup'}}]},
        )

        assert parser.token_ids == list(b'call|end')
        assert result.completion_text == 'call|end'
        assert result.message['content'] == 'parsed'
        assert result.message['raw_response'] == 'call'
        assert result.message['tool_calls'][0]['id'] == 'call-1'

    async def test_tool_parser_preserves_parsed_plain_text(self) -> None:
        processor = Processor(
            WireQueuedBackend([('body<special>|end', 'body<special>')]),
            tokenizer=ByteTokenizer(),
            tool_parser=FallbackToolParser(False),
        )

        result = await processor.create_completion(
            'wire-fallback',
            [{'role': 'user', 'content': 'answer'}],
            request_kwargs={'tools': [{'type': 'function', 'function': {'name': 'lookup'}}]},
        )

        assert result.completion_text == 'body<special>|end'
        assert result.message == {
            'role': 'assistant',
            'content': 'parser text',
            'raw_response': 'body<special>',
        }

    async def test_tool_parser_errors_fail_loudly(self) -> None:
        processor = Processor(
            WireQueuedBackend([('body<special>|end', 'body<special>')]),
            tokenizer=ByteTokenizer(),
            tool_parser=FallbackToolParser(True),
        )

        with pytest.raises(ValueError, match='not a tool call'):
            await processor.create_completion(
                'wire-fallback',
                [{'role': 'user', 'content': 'answer'}],
                request_kwargs={'tools': [{'type': 'function', 'function': {'name': 'lookup'}}]},
            )

    async def test_output_length_preserves_raw_partial_response_when_tool_parsing_fails(self) -> None:
        class LengthBackend:
            async def create_completion(self, _request_id: str, **_kwargs: Any) -> BackendResult:
                return BackendResult(
                    token_ids=list(b'partial tool call'),
                    wire_text='partial tool call',
                    stop_reason='length',
                )

        class FailingToolParser:
            async def extract_tool_calls(self, _token_ids, request=None):
                del request
                raise ValueError('unclosed tool call')

        processor = Processor(LengthBackend(), tokenizer=ByteTokenizer(), tool_parser=FailingToolParser())

        result = await processor.create_completion(
            'partial-tool',
            [{'role': 'user', 'content': 'question'}],
        )

        assert result.stop_reason == 'length'
        assert result.message['content'] == 'partial tool call'
        assert result.message['raw_response'] == 'partial tool call'

    async def test_tool_parser_receives_request_context_and_preserves_raw_fields(self) -> None:
        parser = RequestToolParser()
        tools = [{'type': 'function', 'function': {'name': 'lookup'}}]
        processor = Processor(
            WireQueuedBackend([('sampled', 'visible')]),
            tokenizer=ByteTokenizer(),
            tool_parser=parser,
            apply_chat_template_kwargs={'reasoning_effort': 'high'},
        )

        result = await processor.create_completion(
            'request-context',
            [{'role': 'user', 'content': 'use the tool'}],
            request_kwargs={'tools': tools},
        )

        assert parser.request.tools == tools
        assert parser.request.chat_template_kwargs == {'is_training': True, 'reasoning_effort': 'high'}
        tool_call_id = result.message['tool_calls'][0]['id']
        assert tool_call_id.startswith('call_')
        assert result.message == {
            'role': 'assistant',
            'content': 'answer',
            'reasoning_content': 'reason',
            'raw_response': 'visible',
            'raw_tool': 'raw-tool',
            'tool_calls': [
                {
                    'id': tool_call_id,
                    'type': 'function',
                    # Stored message keeps the parser's JSON string arguments (the OpenAI Function
                    # model nexus round-trips through requires a str); str->dict happens only in a
                    # render-time copy inside _render_chat.
                    'function': {'name': 'lookup', 'arguments': '{}'},
                }
            ],
        }


class ArgItemsTokenizer:
    """Tokenizer whose template renders tool-call arguments structurally (like HunYuan):
    it iterates ``arguments.items()``, so a JSON-string arguments value raises AttributeError."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del tools, kwargs
        rendered = ''
        for message in messages:
            rendered += ('\n' if rendered else '') + message['role'] + ':' + (message.get('content') or '')
            for tool in message.get('tool_calls') or []:
                args = tool['function']['arguments']
                # This is the operation that crashes on a str: str has no .items().
                rendered += ' ' + tool['function']['name'] + '(' + ','.join(f'{k}={v}' for k, v in args.items()) + ')'
        if add_generation_prompt:
            rendered += ('\n' if rendered else '') + 'assistant:'
        return rendered

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return bytes(token_ids).decode(errors='ignore')


class JsonArgToolParser:
    """Parser that returns arguments as a JSON string, like the real hunyuan_v21 parser."""

    think_start_token = '<think>'
    think_end_token = '</think>'

    async def extract_tool_calls(self, token_ids, request=None):
        del token_ids, request
        return ('raw', 'call', [SimpleNamespace(name='calculate', arguments='{"expr": "1+1"}')])


class TestToolCallArgumentNormalization:
    """Regression: tool-call arguments have two consumers with conflicting type needs. Chat
    templates that render tool calls structurally (HunYuan iterates arguments.items()) require a
    DICT, but nexus round-trips the assistant message through an OpenAI ``Function`` pydantic
    model whose ``arguments`` field is a STRING. The gateway keeps ``arguments`` as the parser's
    JSON string in the STORED message (Function-model-safe) and normalizes str -> dict for
    rendering at a SINGLE chokepoint — ``GatewayOpenAIClient._create_completion`` (the interface
    implementation every generation request reaches, including callers that bypass the public
    ``create_completion`` override). The processor/render path assumes dict arguments and does no
    second conversion. Getting this wrong broke ~99% of nexus/calculator rows in the gateway R3
    A/B run — first via 'str has no attribute items', then (after an over-correcting source-level
    fix that mutated the stored message) via 'Input should be a valid string ... input_type=dict'."""

    def test_parse_result_keeps_arguments_as_string(self) -> None:
        # The stored message must keep the parser's string arguments so nexus's Function model
        # (arguments: str) validates it — the fix must NOT convert at the source.
        _content, tool_calls = Processor._normalize_tool_parse_result(
            (None, 'text', [SimpleNamespace(name='calc', arguments='{"expr": "1+1"}')]),
            'text',
        )
        assert tool_calls is not None
        assert tool_calls[0]['function']['arguments'] == '{"expr": "1+1"}'

    def test_parse_result_assigns_nonempty_unique_tool_call_ids(self) -> None:
        _content, tool_calls = Processor._normalize_tool_parse_result(
            (
                None,
                'text',
                [
                    SimpleNamespace(name='first', arguments='{}'),
                    SimpleNamespace(name='second', arguments='{}'),
                ],
            ),
            'text',
        )

        assert tool_calls is not None
        tool_call_ids = [tool_call['id'] for tool_call in tool_calls]
        assert all(tool_call_id.startswith('call_') for tool_call_id in tool_call_ids)
        assert len(set(tool_call_ids)) == len(tool_call_ids)


class TestGenerationBudgets:
    async def test_token_budget_does_not_charge_the_initial_prompt(self) -> None:
        backend = QueuedBackend(['answer'])
        session = _gateway_session(
            'budget',
            Processor(backend, tokenizer=ByteTokenizer()),
            metadata={'token_budget_config': {'token_budget': 10}},
        )

        await session.create_completion(
            [{'role': 'user', 'content': 'non-empty prompt'}],
            request_kwargs={'max_tokens': 100},
        )

        assert backend.execution_requests[0]['max_tokens'] == 10

    async def test_token_consumption_counts_only_a_strict_prompt_extension(self) -> None:
        backend = QueuedBackend(['a', 'b'])
        session = _gateway_session(
            'budget',
            Processor(backend, tokenizer=ByteTokenizer()),
            metadata={'token_budget_config': {'token_budget': 20}},
        )
        first_messages = [{'role': 'user', 'content': 'q'}]
        first = await session.create_completion(first_messages, request_kwargs={'max_tokens': 100})
        second_messages = [*first_messages, first.message, {'role': 'user', 'content': 'next'}]
        before = session.segment_consumed_tokens()
        await session.update_token_consumption(second_messages)
        extension = session.segment_consumed_tokens() - before
        await session.create_completion(second_messages, request_kwargs={'max_tokens': 100})

        assert extension > 0
        assert backend.execution_requests[1]['max_tokens'] == 20 - 1 - extension

    async def test_rewritten_prompt_starts_a_free_token_budget_baseline(self) -> None:
        backend = QueuedBackend(['a', 'b'])
        session = _gateway_session(
            'budget',
            Processor(backend, tokenizer=ByteTokenizer()),
            metadata={'token_budget_config': {'token_budget': 20}},
        )
        await session.create_completion(
            [{'role': 'user', 'content': 'first prompt'}],
            request_kwargs={'max_tokens': 100},
        )
        consumed = session.segment_consumed_tokens()

        rewritten = [{'role': 'user', 'content': 'other prompt'}]
        await session.update_token_consumption(rewritten)
        await session.create_completion(rewritten, request_kwargs={'max_tokens': 100})

        assert session.segment_consumed_tokens() == consumed + 1
        assert backend.execution_requests[1]['max_tokens'] == 20 - consumed

    async def test_interleaved_chains_keep_separate_prompt_ledgers(self) -> None:
        backend = QueuedBackend(['a', 'b', 'c'])
        session = _gateway_session(
            'budget',
            Processor(backend, tokenizer=ByteTokenizer()),
            scaffold_policy='nexus_main_agent',
            metadata={'token_budget_config': {'token_budget': 20}},
        )
        main_messages = [{'role': 'user', 'content': 'main'}]
        main = await session.create_completion(main_messages, request_kwargs={'max_tokens': 100})
        main_consumed = session.segment_consumed_tokens()

        worker_meta = {'declared_chain': {'chain_id': 'worker', 'parent_chain_id': 'main', 'kind': 'worker'}}
        await session.create_completion(
            [{'role': 'user', 'content': 'worker'}],
            request_kwargs={'max_tokens': 100},
            request_meta=worker_meta,
        )

        assert session.segment_consumed_tokens() == main_consumed
        assert session.segment_consumed_tokens(worker_meta) == 1

        next_messages = [*main_messages, main.message, {'role': 'user', 'content': 'next'}]
        await session.update_token_consumption(next_messages)
        extension = session.segment_consumed_tokens() - main_consumed
        await session.create_completion(next_messages, request_kwargs={'max_tokens': 100})

        assert extension > 0
        assert backend.execution_requests[2]['max_tokens'] == 20 - 2 - extension

    async def test_max_model_length_is_a_trajectory_budget(self) -> None:
        backend = QueuedBackend(['first', 'second'])
        session = _gateway_session(
            'budget',
            Processor(backend, tokenizer=ByteTokenizer()),
            metadata={'token_budget_config': {'max_model_len': 40}},
        )
        first_prompt = [{'role': 'user', 'content': 'question'}]

        first = await session.create_completion(first_prompt)
        second_prompt = [*first_prompt, first.message, {'role': 'user', 'content': 'next'}]
        await session.create_completion(second_prompt)

        assert backend.execution_requests[0]['max_tokens'] == 40 - len(backend.prompts[0])
        assert backend.execution_requests[1]['max_tokens'] == 40 - len(backend.prompts[1])

    async def test_explicit_per_call_limit_is_not_rewritten(self) -> None:
        backend = QueuedBackend(['answer'])
        session = _gateway_session(
            'budget',
            Processor(backend, tokenizer=ByteTokenizer()),
            metadata={'token_budget_config': {'max_model_len': 40}},
        )

        await session.create_completion(
            [{'role': 'user', 'content': 'question'}],
            request_kwargs={'max_tokens': 7},
            uses_default_max_tokens=False,
        )

        assert backend.execution_requests[0]['max_tokens'] == 7

    async def test_compaction_starts_a_new_prompt_baseline(self) -> None:
        backend = QueuedBackend(['first', 'summary'])
        session = _gateway_session(
            'budget',
            Processor(backend, tokenizer=ByteTokenizer()),
        )
        first_prompt = [{'role': 'user', 'content': 'a long original question'}]

        await session.create_completion(first_prompt, request_kwargs={'max_tokens': 30})
        await session.create_completion(
            [{'role': 'user', 'content': 'short summary'}],
            request_kwargs={'max_tokens': 30},
        )

        assert backend.execution_requests[0]['max_tokens'] == 30
        assert backend.execution_requests[1]['max_tokens'] == 30


class TestEndpointComposition:
    async def test_gateway_commits_one_completion_after_endpoint_internal_abort_resume(self) -> None:
        server_manager = AbortResumeServerManager()
        processor = Processor(HyTunerServerBackend(server_manager), tokenizer=ByteTokenizer())
        record_store = memory_store('abort-resume')
        session = _gateway_session('abort-resume', processor, record_store=record_store)
        user = {'role': 'user', 'content': 'question'}

        completion_task = asyncio.create_task(session.create_completion([user]))
        await server_manager.first_attempt_finished.wait()

        active = session.get_trajectory()
        assert active.state == 'active'
        assert [node for node in active.iter_nodes() if node.generated_by == 'llm'] == []

        server_manager.resume.set()
        completion = await completion_task
        await session.finish_pending_publications()
        final = session.end('ENV_DONE')

        assert completion.message == {'role': 'assistant', 'content': 'hi'}
        assert len(server_manager.attempts) == 2
        llm_nodes = [node for node in final.iter_nodes() if node.generated_by == 'llm']
        assert len(llm_nodes) == 1
        assert llm_nodes[0].token_account.generated_ids.tolist() == [ord('h'), ord('i')]
        payload = await record_store.get_many([llm_nodes[0].token_account.payload_ref], fields={'logprobs'})
        np.testing.assert_array_equal(payload[0]['logprobs'], np.array([-0.1, -0.2], dtype=np.float32))


class TestRequestFacts:
    async def test_backend_mutation_does_not_change_the_recorded_request(self) -> None:
        class MutatingBackend:
            async def create_completion(self, _request_id, **kwargs):
                kwargs['request_kwargs']['provider_extension']['value'] = 2
                return BackendResult(token_ids=[ord('a')], log_probs=[-0.1])

        session = _gateway_session('request-snapshot', Processor(MutatingBackend(), tokenizer=ByteTokenizer()))
        request_kwargs = {'max_tokens': 1, 'provider_extension': {'value': 1}}

        await session.create_completion(
            [{'role': 'user', 'content': 'question'}],
            request_kwargs=request_kwargs,
            uses_default_max_tokens=False,
        )

        await session.finish_pending_publications()
        llm = next(node for node in session.end().iter_nodes() if node.generated_by == 'llm')
        assert request_kwargs['provider_extension']['value'] == 1
        assert llm.request_kwargs == {'max_tokens': 1, 'provider_extension': {'value': 1}}


class TestEmptyTokenCompletion:
    async def test_complete_empty_generation_is_recorded(self) -> None:
        session = _gateway_session('empty', Processor(QueuedBackend(['']), tokenizer=ByteTokenizer()))

        result = await session.create_completion([{'role': 'user', 'content': 'question'}])

        assert result.message == {'role': 'assistant', 'content': ''}
        assert result.recorded
        await session.finish_pending_publications()
        assert len(list(session.end().iter_nodes())) == 2


class TestProviderTiming:
    async def test_abort_wait_is_not_counted_as_inference_time(self, monkeypatch) -> None:
        clock = iter([10.0, 10.75])
        monkeypatch.setattr('nexus.gateway.processor.time', SimpleNamespace(perf_counter=lambda: next(clock)))
        processor = Processor(WaitReportingBackend(), tokenizer=ByteTokenizer())

        result = await processor.create_completion('request', [{'role': 'user', 'content': 'question'}])

        assert result.inference_time == 0.5

    async def test_reported_and_wall_inference_times_are_both_preserved(self, monkeypatch) -> None:
        class ReportingBackend:
            async def create_completion(self, *_args, **_kwargs):
                return BackendResult(
                    token_ids=[1],
                    stop_reason='stop',
                    inference_time=0.25,
                )

        clock = iter([10.0, 10.75])
        monkeypatch.setattr('nexus.gateway.processor.time', SimpleNamespace(perf_counter=lambda: next(clock)))
        processor = Processor(ReportingBackend(), tokenizer=ByteTokenizer())

        result = await processor.create_completion('request', [{'role': 'user', 'content': 'question'}])

        assert result.inference_time == pytest.approx(0.25)
        assert result.wall_inference_time == pytest.approx(0.75)

    async def test_session_records_reported_and_wall_inference_times(self, monkeypatch) -> None:
        class ReportingBackend:
            async def create_completion(self, *_args, **_kwargs):
                return BackendResult(
                    token_ids=[ord('a')],
                    stop_reason='stop',
                    inference_time=0.25,
                )

        # Only the first two reads belong to the inference window under test. The
        # publication drain (session._finish_pending_publications) times its own store
        # wait through the same clock, so a finite iterator would raise StopIteration
        # there rather than measure anything. Serve the pair, then stay monotonic.
        inference_clock = iter([10.0, 10.75])
        latest = [10.75]

        def perf_counter() -> float:
            value = next(inference_clock, None)
            if value is not None:
                latest[0] = value
            return latest[0]

        monkeypatch.setattr('nexus.gateway.processor.time', SimpleNamespace(perf_counter=perf_counter))
        session = _gateway_session(
            'timing',
            Processor(ReportingBackend(), tokenizer=ByteTokenizer()),
        )

        await session.create_completion([{'role': 'user', 'content': 'question'}])
        await session.finish_pending_publications()
        llm_node = next(node for node in session.end().iter_nodes() if node.generated_by == 'llm')

        assert llm_node.provider_facts is not None
        assert llm_node.provider_facts.inference_time == pytest.approx(0.25)
        assert llm_node.provider_facts.wall_inference_time == pytest.approx(0.75)


class TestPromptCacheOwnership:
    async def test_context_sensitive_history_reuses_via_block_verification(self) -> None:
        # The live-turn think frame only ever decorates a terminal assistant, so the
        # history rendering is byte-stable and block verification authorizes reuse
        # where the removed standalone-render guard used to refuse it.
        backend = WireQueuedBackend([('first|end', 'first'), ('second|end', 'second')])
        processor = Processor(backend, tokenizer=ConditionalAssistantTokenizer())
        session = _gateway_session('conditional-assistant', processor, scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}
        tools_a = [{'type': 'function', 'function': {'name': 'a'}}]

        first = await session.create_completion([system, user], request_kwargs={'tools': tools_a})
        assert first.message == {'role': 'assistant', 'content': 'first'}
        await session.create_completion(
            [system, user, first.message, {'role': 'user', 'content': 'next'}],
            request_kwargs={'tools': tools_a},
        )

        assert len(backend.prompts) == 2
        first_stream = backend.prompts[0] + list(b'first|end')
        assert backend.prompts[1][: len(first_stream)] == first_stream
        assert len(session.get_trajectory().to_segment_node_lists()) == 1

    async def test_cached_ownership_uses_structured_spans(self) -> None:
        backend = QueuedBackend(['a', 'done'])
        processor = Processor(
            backend,
            tokenizer=BoundaryMergeTokenizer(),
            template_parser=ContextSensitiveParser(),
        )
        session = _gateway_session('structured-cache', processor, scaffold_policy='generic')
        user = {'role': 'user', 'content': 'q'}

        first = await session.create_completion([user])
        tool = {'role': 'tool', 'content': 'ab'}
        await session.create_completion([user, first.message, tool])

        assert backend.prompts[1] == backend.prompts[0] + list(b'a') + [999, ord('|')]
        tool_node = next(node for node in session.get_trajectory().iter_nodes() if node.original_message == tool)
        assert tool_node.templated_message == 'ab'
        assert tool_node.token_account.token_ids.tolist() == [999]

    async def test_unlocatable_seam_keeps_the_sampled_stream(self) -> None:
        # The retry cursor sampled a terminator ('|B') that the re-render does not
        # reproduce anywhere ('|A' fixed frame). The canonical tail is still
        # derivable, so continuation falls back to the legacy empty seam.
        backend = WireQueuedBackend([('x|A', 'x'), ('x|B', 'x'), ('next|A', 'next')])
        processor = Processor(backend, tokenizer=HiddenTerminalTokenizer())
        session = _gateway_session('physical-cache', processor, scaffold_policy='generic')
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([user])
        await session.create_completion([user])
        history = [user, first.message, {'role': 'user', 'content': 'next'}]
        assert session.segment_assembler.frozen_prefix(history) is not None

        await session.create_completion(history)

        assert len(backend.prompts) == 3
        previous_stream = backend.prompts[1] + list(b'x|B')
        assert backend.prompts[2] == previous_stream + list(b'U:next\nA:')
        assert all(
            _segment_kind(node_list) == 'conversation' for node_list in session.get_trajectory().to_segment_node_lists()
        )

    async def test_interleaved_reasoning_rejects_notification_that_rewrites_frozen_history(self) -> None:
        backend = QueuedBackend(['<think>spawn reasoning</think>spawn', '<think>done reasoning</think>done', 'ack'])
        session = _gateway_session(
            'interleaved-reasoning',
            Processor(backend, tokenizer=InterleavedReasoningTokenizer()),
            scaffold_policy='generic',
        )
        user = {'role': 'user', 'content': 'question'}

        spawned = await session.create_completion([user])
        work = {'role': 'tool', 'content': 'work'}
        done = await session.create_completion([user, spawned.message, work])
        notification = {'role': 'user', 'content': '<task-notification>result</task-notification>'}

        with pytest.raises(ValueError, match='legacy-compatible continuation tail'):
            await session.create_completion([user, spawned.message, work, done.message, notification])

        assert len(backend.prompts) == 2

    async def test_unlocatable_seam_does_not_reset_the_token_budget(self) -> None:
        backend = WireQueuedBackend([('x|B', 'x'), ('unused|A', 'unused')])
        session = _gateway_session(
            'physical-budget',
            Processor(backend, tokenizer=HiddenTerminalTokenizer()),
            scaffold_policy='generic',
        )
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([user], request_kwargs={'max_tokens': 10})
        history = [user, first.message, {'role': 'user', 'content': 'next'}]

        with pytest.raises(ContextLengthExceededError):
            await session.create_completion(history, request_kwargs={'max_tokens': 10})

        assert len(backend.prompts) == 1

    async def test_normalized_reformat_splices_the_sampled_stream(self, caplog) -> None:
        # The re-render loses the sampled spaces inside the assistant block (a
        # lossless-normalization difference): semantic authorization resends the
        # sampled bytes verbatim and the stream stays continuous in one segment.
        backend = WireQueuedBackend([('{"a": 1}TAIL', '{"a": 1}TAIL'), ('done|end', 'done')])
        processor = Processor(backend, tokenizer=ReformattingAssistantTokenizer())
        session = _gateway_session('reformat-splice', processor, scaffold_policy='generic')
        user = {'role': 'user', 'content': 'q'}

        first = await session.create_completion([user])
        assert first.message['content'] == '{"a": 1}TAIL'
        with caplog.at_level(logging.WARNING, logger='nexus.gateway.processor'):
            await session.create_completion([user, first.message, {'role': 'user', 'content': 'next'}])

        sampled = list(b'{"a": 1}TAIL')
        assert backend.prompts[1] == backend.prompts[0] + sampled + list(b'|end\nU:next\nA:')
        assert len(session.get_trajectory().to_segment_node_lists()) == 1
        assert any('resending sampled bytes' in record.getMessage() for record in caplog.records)

    async def test_retry_realizations_resume_from_the_cursor(self) -> None:
        # Two same-prompt retries decode to the same text with different sampled IDs.
        # The chain has exactly one live realization (the cursor), so the next call
        # deterministically extends the latest retry's physical stream.
        backend = TokenQueuedBackend([([1], 'x'), ([2], 'x'), ([3], 'next')])
        processor = Processor(backend, tokenizer=AliasedTokenTokenizer())
        session = _gateway_session('retry-cursor', processor, scaffold_policy='generic')
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([user])
        await session.create_completion([user])
        history = [user, first.message, {'role': 'user', 'content': 'next'}]
        await session.create_completion(history)

        assert len(backend.prompts) == 3
        assert 2 in backend.prompts[2]
        assert 1 not in backend.prompts[2]

    async def test_prompt_cache_never_crosses_declared_chains(self) -> None:
        backend = TokenQueuedBackend([([1001], 'a'), ([1002], 'b'), ([1003], 'c')])
        processor = Processor(backend, tokenizer=AliasedTokenTokenizer())
        session = _gateway_session('chain-cache', processor, scaffold_policy='nexus_main_agent')
        user = {'role': 'user', 'content': 'x'}
        question = {'role': 'user', 'content': 'q'}
        meta_a = {'declared_chain': {'chain_id': 'A', 'kind': 'worker'}}
        meta_b = {'declared_chain': {'chain_id': 'B', 'kind': 'worker'}}

        first = await session.create_completion([user], request_meta=meta_a)
        second = await session.create_completion([user, first.message, question], request_meta=meta_b)
        await session.create_completion(
            [user, first.message, question, second.message, {'role': 'user', 'content': 'z'}],
            request_meta=meta_a,
        )

        assert backend.prompts[1] == list(b'U:x\nA:a\nU:q\nA:')
        assert backend.prompts[2] == backend.prompts[0] + [1001] + list(b'\nU:q\nA:b\nU:z\nA:')
        assert 1002 not in backend.prompts[2]

    async def test_tool_change_restarts_the_stream(self) -> None:
        # Tools render into the frozen prefix and sit outside message equality,
        # so a tools change refuses the splice and restarts with a fresh render.
        backend = TokenQueuedBackend([([1001], 'a'), ([1002], 'b'), ([1003], 'c')])
        processor = Processor(backend, tokenizer=AliasedTokenTokenizer())
        session = _gateway_session('tool-cache', processor, scaffold_policy='generic')
        user = {'role': 'user', 'content': 'x'}
        tools_a = [{'type': 'function', 'function': {'name': 'a'}}]
        tools_b = [{'type': 'function', 'function': {'name': 'b'}}]

        first = await session.create_completion([user], request_kwargs={'tools': tools_a})
        await session.create_completion(
            [user, first.message, {'role': 'user', 'content': 'next'}],
            request_kwargs={'tools': tools_b},
        )

        assert 1001 not in backend.prompts[1]
        assembler = session.segment_assembler
        node_id = latest_node_id(assembler)
        boundary = assembler.trajectory.nodes[node_id].segment_boundary
        assert boundary is not None
        assert 'render_divergence' in boundary.causes
        assert boundary.evidence['divergence'] == 'context-change'


class TestFrozenPrefixSelection:
    async def test_continuing_an_older_sibling_resends_its_sampled_stream(self) -> None:
        backend = QueuedBackend(['a', 'b', 'c', 'd'])
        session = _gateway_session('frozen-older', Processor(backend, tokenizer=ByteTokenizer()))
        user = {'role': 'user', 'content': 'x'}
        first = await session.create_completion([user])
        history = [user, first.message, {'role': 'user', 'content': 'q'}]
        older = await session.create_completion(history)
        await session.create_completion(history)

        continued = [*history, older.message, {'role': 'user', 'content': 'r'}]
        frozen = session.segment_assembler.frozen_prefix(continued)
        await session.create_completion(continued)

        assert frozen is not None

        assert frozen.count == 4
        assert backend.prompts[3][: len(backend.prompts[1]) + 1] == backend.prompts[1] + list(b'b')

    async def test_a_separate_tree_in_the_context_does_not_hide_the_main_path(self) -> None:
        backend = QueuedBackend(['a', 'b', 'summary', 'd'])
        session = _gateway_session('frozen-summary-tree', Processor(backend, tokenizer=ByteTokenizer()))
        user = {'role': 'user', 'content': 'x'}
        first = await session.create_completion([user])
        history = [user, first.message, {'role': 'user', 'content': 'q'}]
        second = await session.create_completion(history)
        await session.create_completion([{'role': 'user', 'content': 'summarize'}])

        frozen = session.segment_assembler.frozen_prefix([*history, second.message, {'role': 'user', 'content': 'r'}])

        assert frozen is not None

        assert frozen.count == 4

    async def test_a_subagent_reuses_the_system_node_it_shares_with_the_main_agent(self) -> None:
        backend = QueuedBackend(['a', 'b'])
        session = _gateway_session(
            'frozen-shared-system',
            Processor(backend, tokenizer=ByteTokenizer()),
            scaffold_policy='nexus_main_agent',
        )
        system = {'role': 'system', 'content': 's'}
        sub_meta = {'declared_chain': {'chain_id': 'sub', 'kind': 'worker'}}
        await session.create_completion([system, {'role': 'user', 'content': 'main'}])
        sub_task = {'role': 'user', 'content': 'sub'}
        reply = await session.create_completion([system, sub_task], request_meta=sub_meta)

        assembler = session.segment_assembler
        messages = [system, sub_task, reply.message, {'role': 'user', 'content': 'go'}]
        frozen = assembler.frozen_prefix(messages, context=assembler.extract_context(messages, request_meta=sub_meta))

        assert frozen is not None

        assert frozen.count == 3
        assert assembler._runtime_state.facts.by_node[frozen.node_ids[0]].context_key[1] == 'main'

    async def test_inputs_left_by_a_failed_recording_are_not_reused(self, monkeypatch) -> None:
        backend = QueuedBackend(['a', 'b'])
        session = _gateway_session('frozen-failed-record', Processor(backend, tokenizer=ByteTokenizer()))
        assembler = session.segment_assembler
        user = {'role': 'user', 'content': 'x'}
        first = await session.create_completion([user])
        history = [user, first.message, {'role': 'user', 'content': 'q'}]

        def fail(*_args, **_kwargs):
            raise RuntimeError('recording failed after the inputs were attached')

        monkeypatch.setattr(assembler, '_request_boundary', fail)
        with pytest.raises(RuntimeError, match='recording failed'):
            await session.create_completion(history)
        monkeypatch.undo()

        frozen = assembler.frozen_prefix(history)
        assert frozen is not None
        assert frozen.count == 2

    async def test_only_prefixes_sent_with_the_same_tools_are_reused(self) -> None:
        backend = QueuedBackend(['a', 'b', 'c'])
        session = _gateway_session('frozen-tools', Processor(backend, tokenizer=ByteTokenizer()))
        user = {'role': 'user', 'content': 'x'}
        tools_a = {'tools': [{'type': 'function', 'function': {'name': 'a'}}]}
        tools_b = {'tools': [{'type': 'function', 'function': {'name': 'b'}}]}
        first = await session.create_completion([user], request_kwargs=tools_a)
        history = [user, first.message, {'role': 'user', 'content': 'q'}]
        second = await session.create_completion(history, request_kwargs=tools_b)

        assembler = session.segment_assembler
        back_to_a = [*history, second.message, {'role': 'user', 'content': 'r'}]
        frozen = assembler.frozen_prefix(back_to_a, tools=tools_a['tools'])

        assert frozen is not None

        assert frozen.count == 2
        frozen_b = assembler.frozen_prefix(back_to_a, tools=tools_b['tools'])
        assert frozen_b is not None
        assert frozen_b.count == 4


class TestFrozenPrefixAnchor:
    async def test_recording_follows_the_frozen_node_among_identical_siblings(self) -> None:
        # A prefilled assistant turn and a later identical sample are same-identity,
        # same-token siblings; the request resends the sample, so it is recorded there.
        backend = QueuedBackend(['b', 'a', 'c'])
        session = _gateway_session('frozen-anchor', Processor(backend, tokenizer=ByteTokenizer()))
        assembler = session.segment_assembler
        user = {'role': 'user', 'content': 'x'}
        prefilled = {'role': 'assistant', 'content': 'a'}

        await session.create_completion([user, prefilled, {'role': 'user', 'content': 'q'}])
        sampled = await session.create_completion([user])
        history = [user, sampled.message, {'role': 'user', 'content': 'r'}]
        frozen = assembler.frozen_prefix(history)
        await session.create_completion(history)

        assert frozen is not None
        output_id = max(assembler.trajectory.nodes)
        path = assembler._node_path(assembler.trajectory.nodes[output_id].parent)
        assert tuple(path[: frozen.count]) == frozen.node_ids
        assert assembler.trajectory.nodes[path[1]].generated_by == 'llm'

    async def test_concurrent_requests_record_under_their_shared_frozen_prefix(self) -> None:
        backend = YieldingQueuedBackend(['a', 'b', 'c'])
        session = _gateway_session('frozen-concurrent', Processor(backend, tokenizer=ByteTokenizer()))
        user = {'role': 'user', 'content': 'x'}

        first = await session.create_completion([user])
        history = [user, first.message, {'role': 'user', 'content': 'q'}]
        await asyncio.gather(session.create_completion(history), session.create_completion(history))

        nodes = session.segment_assembler.trajectory.nodes
        outputs = sorted(node_id for node_id, node in nodes.items() if node.generated_by == 'llm')[1:]
        assert backend.prompts[1] == backend.prompts[2]
        assert nodes[outputs[0]].parent == nodes[outputs[1]].parent


class TestPromptRendering:
    def test_text_content_blocks_are_flattened_only_for_rendering(self) -> None:
        processor = Processor(QueuedBackend([]), tokenizer=ByteTokenizer())
        messages = [
            {
                'role': 'system',
                'content': [
                    {'type': 'text', 'text': 'first', 'cache_control': {'type': 'ephemeral'}},
                    {'type': 'text', 'text': 'second'},
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'reminder\n'},
                    {'type': 'text', 'text': 'question'},
                ],
            },
        ]

        rendered = processor._render_chat(messages, None, add_generation_prompt=True)

        assert rendered == 'S:firstsecond\nU:reminder\nquestion\nA:'
        assert isinstance(messages[0]['content'], list)
        assert isinstance(messages[1]['content'], list)

    def test_template_receives_exact_empty_tools_and_training_mode(self) -> None:
        tokenizer = RecordingTemplateTokenizer()
        processor = Processor(QueuedBackend([]), tokenizer=tokenizer)

        processor._render_chat(
            [{'role': 'user', 'content': 'question'}],
            [],
            add_generation_prompt=True,
        )

        assert tokenizer.calls == [{'tools': [], 'is_training': True}]

    def test_training_mode_can_be_overridden_explicitly(self) -> None:
        tokenizer = RecordingTemplateTokenizer()
        processor = Processor(
            QueuedBackend([]),
            tokenizer=tokenizer,
            apply_chat_template_kwargs={'is_training': False},
        )

        processor._render_chat(
            [{'role': 'user', 'content': 'question'}],
            None,
            add_generation_prompt=True,
        )

        assert tokenizer.calls == [{'tools': None, 'is_training': False}]

    async def test_nodes_preserve_template_tokens_and_segment_masks(self):
        processor = Processor(QueuedBackend(['first', 'second']), tokenizer=ByteTokenizer())
        session = _gateway_session('rendered', processor, scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        first_user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, first_user])
        second_user = {'role': 'user', 'content': 'next'}
        await session.create_completion([system, first_user, first.message, second_user])

        trajectory = session.get_trajectory()
        nodes = list(trajectory.iter_nodes())
        assert [node.original_message for node in nodes] == [
            system,
            first_user,
            first.message,
            second_user,
            {'role': 'assistant', 'content': 'second'},
        ]
        assert [node.templated_message for node in nodes] == [
            'S:sys',
            '\nU:question',
            '\nA:first',
            '\nU:next',
            '\nA:second',
        ]
        assert [node.token_account.token_ids.tolist() for node in nodes] == [
            list(b'S:sys'),
            list(b'\nU:question'),
            list(b'\nA:first'),
            list(b'\nU:next'),
            list(b'\nA:second'),
        ]

        response_ids, response_mask = _response_axis(trajectory.to_segment_node_lists()[0])
        assert response_ids == list(b'first\nU:next\nA:second')
        assert response_mask == [1] * len(b'first') + [0] * len(b'\nU:next\nA:') + [1] * len(b'second')

        restored = type(trajectory).from_dump(trajectory.dump())
        assert [node.templated_message for node in restored.iter_nodes()] == [
            'S:sys',
            '\nU:question',
            '\nA:first',
            '\nU:next',
            '\nA:second',
        ]

    async def test_canonical_tokenization_wins_over_span_boundaries(self):
        """The engine must see the whole-text (canonical) tokenization: a BPE merge across
        a span boundary without a special-token barrier happens exactly as the template's
        full render would tokenize. The straddling token is attributed to the earlier
        span (boundaries stay text-accurate; id grouping is bookkeeping only)."""
        backend = QueuedBackend(['answer'])
        processor = Processor(backend, tokenizer=BoundaryMergeTokenizer())
        session = _gateway_session('message-boundaries', processor, scaffold_policy='generic')
        messages = [
            {'role': 'user', 'content': 'a'},
            {'role': 'tool', 'content': 'b'},
        ]

        await session.create_completion(messages)

        assert backend.prompts == [[999, ord('|')]]
        input_nodes = list(session.get_trajectory().iter_nodes())[:2]
        assert [node.templated_message for node in input_nodes] == ['a', 'b']
        assert [node.token_account.token_ids.tolist() for node in input_nodes] == [[999], []]

    async def test_appended_messages_join_the_canonical_stream(self):
        backend = QueuedBackend(['r', 'done'])
        processor = Processor(backend, tokenizer=BoundaryMergeTokenizer())
        session = _gateway_session('appended-boundaries', processor, scaffold_policy='generic')
        user = {'role': 'user', 'content': 'q'}

        first = await session.create_completion([user])
        await session.create_completion([
            user,
            first.message,
            {'role': 'tool', 'content': 'a'},
            {'role': 'user', 'content': 'b'},
        ])

        old_stream = backend.prompts[0] + list(b'r')
        # The appended tail ('a' + 'b' + generation '|') is tokenized canonically as one
        # text, so the cross-span merge 'ab' -> 999 happens exactly as a full render would.
        assert backend.prompts[1] == old_stream + [999, ord('|')]
        response_ids, response_mask = _response_axis(session.get_trajectory().to_segment_node_lists()[0])
        assert response_ids == [ord('r'), 999, ord('|')] + list(b'done')
        assert response_mask == [1, 0, 0] + [1] * len(b'done')

    async def test_new_message_prefix_is_not_mistaken_for_closure(self):
        backend = QueuedBackend(['a', 'done'])
        processor = Processor(backend, tokenizer=BoundaryMergeTokenizer())
        session = _gateway_session('closure-collision', processor, scaffold_policy='generic')
        user = {'role': 'user', 'content': 'q'}

        first = await session.create_completion([user])
        next_message = {'role': 'tool', 'content': 'a'}
        await session.create_completion([user, first.message, next_message])

        nodes = list(session.get_trajectory().iter_nodes())
        first_node = next(node for node in nodes if node.original_message == first.message)
        next_node = next(node for node in nodes if node.original_message == next_message)
        assert first_node.templated_message == '|a'
        assert first_node.token_account.token_ids.tolist() == list(b'|a')
        assert next_node.templated_message == 'a'
        assert next_node.token_account.token_ids.tolist() == [ord('a')]


class TestToolRenderFork:
    async def test_tool_template_change_forks_in_assembly(self):
        backend = QueuedBackend(['first', 'second'])
        session = _gateway_session('tools', Processor(backend, tokenizer=ByteTokenizer()), scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}
        tools_a = [{'type': 'function', 'function': {'name': 'a'}}]
        tools_b = [{'type': 'function', 'function': {'name': 'b'}}]

        first = await session.create_completion([system, user], request_kwargs={'tools': tools_a})
        next_user = {'role': 'user', 'content': 'next'}
        await session.create_completion(
            [system, user, first.message, next_user],
            request_kwargs={'tools': tools_b},
        )

        trajectory = session.get_trajectory()
        system_nodes = [node for node in trajectory.iter_nodes() if node.original_message['role'] == 'system']
        assert [node.templated_message for node in system_nodes] == ['S:sys|tools:a', 'S:sys|tools:b']
        assert len({node.message_id for node in system_nodes}) == 1
        first_message_nodes = [
            node for node in trajectory.iter_nodes() if node.original_message.get('content') == 'first'
        ]
        assert {node.generated_by for node in first_message_nodes} == {'client', 'llm'}
        assert len({node.message_id for node in first_message_nodes}) == 1
        node_lists = trajectory.to_segment_node_lists()
        assert [_segment_kind(node_list) for node_list in node_lists] == ['conversation', 'render_restart']
        assert len(trajectory.roots) == 2
        assert node_lists[1].response_nodes[0].segment_boundary.causes == ('render_divergence',)


class TestLegacySeamCompat:
    async def test_hycode_blank_tool_content_reuses_sampled_tokens(self, monkeypatch):
        monkeypatch.setenv('GATEWAY_LEGACY_SEAM_COMPAT', '1')
        tool_call = {'id': 'call-1', 'type': 'function', 'function': {'name': 'Bash', 'arguments': '{}'}}

        class ToolParser:
            async def extract_tool_calls(self, token_ids):
                text = bytes(token_ids).decode()
                return text, '\n', [tool_call]

        sampled = '<tool_call>{}</tool_call>\n'
        backend = QueuedBackend([sampled, 'done'])
        processor = Processor(backend, tokenizer=ByteTokenizer(), tool_parser=ToolParser())
        session = _gateway_session('hycode-blank', processor, scaffold_policy='hycode')
        user = {'role': 'user', 'content': 'calculate'}
        first = await session.create_completion([user])
        replay = {**first.message, 'content': None}
        observation = {'role': 'tool', 'tool_call_id': first.message['tool_calls'][0]['id'], 'content': '42'}

        await session.create_completion([user, replay, observation])

        assert backend.prompts[1] == backend.prompts[0] + list(sampled.encode()) + list(b'\nT:42\nA:')
        trajectory = session.get_trajectory()
        assert len(trajectory.to_segment_node_lists()) == 1
        assert all(len(node.children) <= 1 for node in trajectory.iter_nodes())
        assert first.message['content'] == '\n'

    async def test_compat_uses_legacy_canonical_subtraction_for_a_direct_extension(self, monkeypatch):
        monkeypatch.setenv('GATEWAY_LEGACY_SEAM_COMPAT', '1')
        backend = QueuedBackend(['first', 'second'])
        processor = Processor(backend, tokenizer=TrailingAssistantTokenizer())
        session = _gateway_session('legacy-boundary', processor)
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, user])
        await session.create_completion([system, user, first.message, {'role': 'user', 'content': 'next'}])

        node_lists = session.get_trajectory().to_segment_node_lists()
        assert len(node_lists) == 1
        assert _response_axis(node_lists[0]) == (
            list(b'first\nU:next\nA:second'),
            [1] * len(b'first') + [0] * len(b'\nU:next\nA:') + [1] * len(b'second'),
        )

    async def test_compat_does_not_insert_an_inferred_seam(self, monkeypatch, caplog):
        monkeypatch.setenv('GATEWAY_LEGACY_SEAM_COMPAT', '1')
        backend = WireQueuedBackend([('value with spaceTAIL', 'value with spaceTAIL'), ('done|end', 'done')])
        processor = Processor(backend, tokenizer=ReformattingAssistantTokenizer())
        session = _gateway_session('legacy-seam', processor)
        user = {'role': 'user', 'content': 'q'}

        first = await session.create_completion([user])
        with caplog.at_level(logging.WARNING, logger='nexus.gateway.processor'):
            await session.create_completion([user, first.message, {'role': 'user', 'content': 'next'}])

        frozen_stream = backend.prompts[0] + list(b'value with spaceTAIL')
        assert backend.prompts[1] == frozen_stream + list(b'\nU:next\nA:')
        assert any('No Worries' in record.getMessage() for record in caplog.records)


class TestTemplateClosure:
    async def test_historical_assistant_closure_extends_the_generated_node(self):
        backend = QueuedBackend(['first', 'second'])
        processor = Processor(backend, tokenizer=TrailingAssistantTokenizer())
        session = _gateway_session('assistant-tail', processor, scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, user])
        await session.create_completion(
            [system, user, first.message, {'role': 'user', 'content': 'next'}],
        )

        trajectory = session.get_trajectory()
        first_nodes = [node for node in trajectory.iter_nodes() if node.original_message.get('content') == 'first']
        assert len(first_nodes) == 1
        account = first_nodes[0].token_account
        assert account is not None
        assert first_nodes[0].generated_by == 'llm'
        # The template closure is owned by the following span, never by the frozen
        # generated node; the physical stream is identical either way.
        assert first_nodes[0].templated_message == '\nA:first'
        assert account.token_ids.tolist() == list(b'\nA:first')
        assert account.gen_start == len(b'\nA:')
        assert account.gen_len == len(b'first')
        assert account.generated_ids.tolist() == list(b'first')
        assert account.trailing_ids.size == 0
        next_node = next(node for node in trajectory.iter_nodes() if node.original_message.get('content') == 'next')
        assert next_node.templated_message == '|end\nU:next'
        node_list = trajectory.to_segment_node_lists()[0]
        assert _response_axis(node_list) == (
            list(b'first|end\nU:next\nA:second'),
            [1] * len(b'first') + [0] * len(b'|end\nU:next\nA:') + [1] * len(b'second'),
        )

        restored_trajectories = (
            type(trajectory).from_dump(trajectory.dump()),
            pickle.loads(pickle.dumps(trajectory)),
        )
        for restored in restored_trajectories:
            restored_first = next(
                node for node in restored.iter_nodes() if node.original_message.get('content') == 'first'
            )
            assert restored_first.token_account is not None
            assert restored_first.token_account.token_ids.tolist() == list(b'\nA:first')
            assert restored_first.token_account.generated_ids.tolist() == list(b'first')
            assert _response_axis(restored.to_segment_node_lists()[0]) == _response_axis(node_list)

    async def test_boundary_retokenization_preserves_the_sampled_stream(self):
        backend = QueuedBackend(['first', 'second'])
        processor = Processor(backend, tokenizer=RetokenizingAssistantTokenizer())
        session = _gateway_session('assistant-retokenized', processor, scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, user])
        await session.create_completion(
            [system, user, first.message, {'role': 'user', 'content': 'next'}],
        )

        trajectory = session.get_trajectory()
        first_nodes = [node for node in trajectory.iter_nodes() if node.original_message.get('content') == 'first']
        assert len(first_nodes) == 1
        assert first_nodes[0].generated_by == 'llm'
        account = first_nodes[0].token_account
        assert account is not None
        assert account.generated_ids.tolist() == list(b'first')
        # Frozen IDs are never re-encoded: the sampled span stays intact and the
        # closure text is encoded as the head of the following span instead.
        assert account.trailing_ids.size == 0
        old_stream = backend.prompts[0] + list(b'first')
        assert backend.prompts[1][: len(old_stream)] == old_stream
        assert 999 not in backend.prompts[1]
        node_list = trajectory.to_segment_node_lists()[0]
        assert _response_axis(node_list) == (
            list(b'first|end\nU:next\nA:second'),
            [1] * len(b'first') + [0] * len(b'|end\nU:next\nA:') + [1] * len(b'second'),
        )

    async def test_zero_token_closure_extends_only_the_templated_message(self):
        backend = QueuedBackend(['first', 'second'])
        processor = Processor(backend, tokenizer=ZeroTokenClosureTokenizer())
        session = _gateway_session('zero-token-closure', processor, scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, user])
        await session.create_completion(
            [system, user, first.message, {'role': 'user', 'content': 'next'}],
        )

        trajectory = session.get_trajectory()
        first_nodes = [node for node in trajectory.iter_nodes() if node.original_message.get('content') == 'first']
        assert len(first_nodes) == 1
        # The zero-token closure text belongs to the following span; the generated
        # node keeps exactly what was rendered for it at its own call.
        assert first_nodes[0].templated_message == '\nA:first'
        assert first_nodes[0].token_account is not None
        assert first_nodes[0].token_account.token_ids.tolist() == list(b'\nA:first')
        next_node = next(node for node in trajectory.iter_nodes() if node.original_message.get('content') == 'next')
        assert next_node.templated_message == '|end\nU:next'
        assert next_node.token_account is not None
        assert next_node.token_account.token_ids.tolist() == list(b'\nU:next')
        assert len(trajectory.to_segment_node_lists()) == 1


class TestObservationOwnership:
    async def test_observation_tokens_come_from_the_next_rendered_prompt(self):
        processor = Processor(QueuedBackend(['first', 'second']), tokenizer=ByteTokenizer())
        session = _gateway_session('observation', processor, scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, user])
        observation = {'role': 'tool', 'content': 'result'}
        session.record_observation('result', parent_messages=[system, user, first.message])
        await session.create_completion(
            [system, user, first.message, observation, {'role': 'user', 'content': 'next'}],
        )

        trajectory = session.get_trajectory()
        node_list = trajectory.to_segment_node_lists()[0]
        assert _response_axis(node_list) == (
            list(b'first\nT:result\nU:next\nA:second'),
            [1] * len(b'first') + [0] * len(b'\nT:result\nU:next\nA:') + [1] * len(b'second'),
        )
        restored = type(trajectory).from_dump(trajectory.dump())
        assert _response_axis(restored.to_segment_node_lists()[0]) == _response_axis(node_list)

    async def test_assistant_closure_precedes_an_explicit_observation(self):
        processor = Processor(QueuedBackend(['first', 'second']), tokenizer=TrailingAssistantTokenizer())
        session = _gateway_session('closure-observation', processor, scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, user])
        observation = {'role': 'tool', 'content': 'result'}
        session.record_observation('result', parent_messages=[system, user, first.message])
        await session.create_completion(
            [system, user, first.message, observation, {'role': 'user', 'content': 'next'}],
        )

        assert _response_axis(session.get_trajectory().to_segment_node_lists()[0]) == (
            list(b'first|end\nT:result\nU:next\nA:second'),
            [1] * len(b'first') + [0] * len(b'|end\nT:result\nU:next\nA:') + [1] * len(b'second'),
        )

    def test_explicit_observation_token_ids_are_rejected(self):
        session = _gateway_session(
            'observation-token-ids',
            Processor(QueuedBackend([]), tokenizer=ByteTokenizer()),
            scaffold_policy='generic',
        )

        with pytest.raises(ValueError, match='next rendered prompt'):
            session.record_observation('result', token_ids=[1, 2, 3])

    async def test_terminal_observation_has_no_response_occurrence(self):
        session = _gateway_session(
            'terminal-observation',
            Processor(QueuedBackend(['first']), tokenizer=ByteTokenizer()),
            scaffold_policy='generic',
        )
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        first = await session.create_completion([system, user])
        observation = {'role': 'tool', 'content': 'result'}
        session.record_observation(message=observation, parent_messages=[system, user, first.message])

        trajectory = session.get_trajectory()
        observation_node = next(node for node in trajectory.iter_nodes() if node.original_message == observation)
        assert observation_node.token_account is None
        assert _response_axis(trajectory.to_segment_node_lists()[0]) == (
            list(b'first'),
            [1] * len(b'first'),
        )
        assert trajectory.to_messages() == [system, user, first.message, observation]


class TestRenderRestarts:
    async def test_switching_back_to_an_old_template_starts_another_segment(self):
        backend = QueuedBackend(['first', 'second', 'third'])
        session = _gateway_session('tools', Processor(backend, tokenizer=ByteTokenizer()), scaffold_policy='generic')
        system = {'role': 'system', 'content': 'sys'}
        first_user = {'role': 'user', 'content': 'q1'}
        tools_a = [{'type': 'function', 'function': {'name': 'a'}}]
        tools_b = [{'type': 'function', 'function': {'name': 'b'}}]

        first = await session.create_completion([system, first_user], request_kwargs={'tools': tools_a})
        second_user = {'role': 'user', 'content': 'q2'}
        second = await session.create_completion(
            [system, first_user, first.message, second_user],
            request_kwargs={'tools': tools_b},
        )
        third_user = {'role': 'user', 'content': 'q3'}
        await session.create_completion(
            [system, first_user, first.message, second_user, second.message, third_user],
            request_kwargs={'tools': tools_a},
        )

        node_lists = session.get_trajectory().to_segment_node_lists()
        by_response = {node_list.response_nodes[-1].original_message['content']: node_list for node_list in node_lists}
        assert set(by_response) == {'first', 'second', 'third'}
        assert _segment_kind(by_response['first']) == 'conversation'
        assert _segment_kind(by_response['second']) == 'render_restart'
        assert _segment_kind(by_response['third']) == 'render_restart'


class TestPromptRenderConstruction:
    async def test_non_monotonic_template_uses_lossless_span_fallback(self):
        backend = QueuedBackend(['answer'])
        session = _gateway_session('unstable', Processor(backend, tokenizer=LastAwareTokenizer()))

        await session.create_completion([{'role': 'user', 'content': 'question'}])

        assert backend.prompts == [list(b'U:question\nA:')]
        user_node = next(
            node for node in session.get_trajectory().iter_nodes() if node.original_message['role'] == 'user'
        )
        output_node = next(
            node for node in session.get_trajectory().iter_nodes() if node.original_message['role'] == 'assistant'
        )
        assert user_node.templated_message == ''
        assert output_node.templated_message == 'U:question\nA:answer'

    async def test_structured_parser_does_not_require_prefix_rerenders(self):
        backend = QueuedBackend(['answer'])
        processor = Processor(
            backend,
            tokenizer=ByteTokenizer(),
            template_parser=StructuredParser(),
        )
        session = _gateway_session('structured', processor)
        system = {'role': 'system', 'content': 'sys'}
        user = {'role': 'user', 'content': 'question'}

        await session.create_completion([system, user])

        nodes = list(session.get_trajectory().iter_nodes())
        assert [node.templated_message for node in nodes] == [
            'S:sys\n',
            'U:question\n',
            'A:answer',
        ]

    async def test_structured_spans_use_the_same_token_construction(self):
        backend = QueuedBackend(['answer'])
        tokenizer = StructuredMergeTokenizer()
        processor = Processor(
            backend,
            tokenizer=tokenizer,
            template_parser=StructuredParser(),
        )
        session = _gateway_session('structured-boundaries', processor)
        messages = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'question'},
        ]

        await session.create_completion(messages)

        spans = ['S:sys\n', 'U:question\n', 'A:']
        expected = tokenizer.encode(''.join(spans), add_special_tokens=False)
        assert backend.prompts == [expected]
        # The '\n'+'U' merge across the span boundary happens exactly as a full render would.
        assert 999 in expected

    async def test_inconsistent_structured_spans_fail_before_inference(self):
        backend = QueuedBackend(['unused'])
        processor = Processor(
            backend,
            tokenizer=ByteTokenizer(),
            template_parser=InconsistentStructuredParser(),
        )

        with pytest.raises(ValueError, match='Template parser spans do not reconstruct'):
            await processor.create_completion(
                'inconsistent-structured',
                [
                    {'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': 'question'},
                ],
            )

        assert backend.prompts == []

    async def test_tokenizer_must_disable_implicit_special_tokens(self):
        backend = QueuedBackend(['unused'])
        processor = Processor(backend, tokenizer=ImplicitSpecialTokenizer())

        with pytest.raises(TypeError, match='add_special_tokens'):
            await processor.create_completion('implicit-specials', [{'role': 'user', 'content': 'question'}])
        assert backend.prompts == []

    async def test_token_processor_rejects_a_message_lane_backend_result(self):
        processor = Processor(MissingTokenBackend(), tokenizer=ByteTokenizer())

        with pytest.raises(RuntimeError, match='Token-level backend returned no token IDs'):
            await processor.create_completion('missing-token-ids', [{'role': 'user', 'content': 'question'}])

    async def test_client_token_count_accepts_prompt_render(self):
        client = object.__new__(GatewayOpenAIClient)
        client._processor = Processor(QueuedBackend([]), tokenizer=ByteTokenizer())
        client._trajectory_id = 'traj-count-render'  # no session exists -> canonical fallback
        client._model_registry = None
        client._metadata = None
        client._agent_request_meta = {}

        result = await client._count_tokens(messages=[{'role': 'user', 'content': 'question'}])
        without_generation = await client._count_tokens(
            messages=[{'role': 'user', 'content': 'question'}],
            add_generation_prompt=False,
        )

        assert result.input_tokens == len(b'U:question\nA:')
        assert without_generation.input_tokens == len(b'U:question')


class _ExtractOnlyAdapter:
    """An adapter that reads media but cannot expand it.

    Gateway needs all three contract methods; this fixture is the incomplete shape
    those loud-failure tests exercise.
    """

    def __init__(self, media):
        self._media = media

    async def process_multi_modal_info(self, messages):
        # The fixtures pass the same async extractor the plugin would implement.
        return await self._media(messages)

    def get_placeholder_token_ids(self):
        return {'images': 1}


class TestPromptRenderRequirements:
    def test_token_recording_requires_prompt_render(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')

        with pytest.raises(ValueError, match='PromptRender is required'):
            assembler.record_completion(
                messages=[{'role': 'user', 'content': 'question'}],
                output_message={'role': 'assistant', 'content': 'answer'},
                token_training={
                    'prompt_ids': [1],
                    'raw_prompt_ids': [1],
                    'response_ids': [2],
                },
            )

    def test_token_recording_rejects_a_missing_raw_prompt(self):
        assembler = TrajectoryAssembler(scaffold_policy='generic')
        processor = Processor(QueuedBackend([]), tokenizer=ByteTokenizer())
        prompt_render, _ = processor._build_prompt_render(
            [{'role': 'user', 'content': 'question'}],
            None,
            text='U:question\nA:',
            add_generation_prompt=True,
        )

        with pytest.raises(ValueError, match='raw_prompt_ids are required'):
            assembler.record_completion(
                messages=[{'role': 'user', 'content': 'question'}],
                output_message={'role': 'assistant', 'content': 'answer'},
                completion_text='answer',
                prompt_render=prompt_render,
                token_training={
                    'prompt_ids': prompt_render.token_ids,
                    'raw_prompt_ids': [],
                    'response_ids': [1],
                },
            )

    @pytest.mark.asyncio
    async def test_vision_extractor_unknown_modality_fails_loudly(self):
        async def vision_info(messages):
            del messages
            return {'depth': ['unsupported']}

        processor = Processor(
            QueuedBackend([]),
            tokenizer=ByteTokenizer(),
            mm_plugin=_ExtractOnlyAdapter(vision_info),
        )
        user = {
            'role': 'user',
            'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}],
        }

        with pytest.raises(NotImplementedError, match='prompt expansion encodes images only'):
            await processor.render_and_tokenize([user], None)

    @pytest.mark.asyncio
    async def test_vision_extractor_non_mapping_fails_loudly(self):
        async def vision_info(messages):
            del messages
            return ['unsupported']

        processor = Processor(
            QueuedBackend([]),
            tokenizer=ByteTokenizer(),
            mm_plugin=_ExtractOnlyAdapter(vision_info),
        )
        user = {
            'role': 'user',
            'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}],
        }

        with pytest.raises(AttributeError):
            await processor.render_and_tokenize([user], None)

    @pytest.mark.asyncio
    async def test_video_payload_still_raises(self):
        async def vision_info(messages):
            del messages
            return {'videos': ['one-video']}

        processor = Processor(
            QueuedBackend([]),
            tokenizer=ByteTokenizer(),
            mm_plugin=_ExtractOnlyAdapter(vision_info),
        )
        user = {'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': 'x'}}]}

        with pytest.raises(NotImplementedError, match='prompt expansion encodes images only'):
            await processor.render_and_tokenize([user], None)


class MergedToolBlockTokenizer(ByteTokenizer):
    """Consecutive tool results share one U[...] container; its closer sits after the last result."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del kwargs, tools
        rendered = ''
        index = 0
        while index < len(messages):
            message = messages[index]
            if message['role'] == 'tool':
                group = []
                while index < len(messages) and messages[index]['role'] == 'tool':
                    group.append(messages[index].get('content') or '')
                    index += 1
                rendered += 'U[' + '|'.join(f'R<{content}>' for content in group) + ']'
                continue
            prefix = {'system': 'S:', 'user': 'U:', 'assistant': 'A:'}[message['role']]
            rendered += prefix + (message.get('content') or '') + ';'
            index += 1
        if add_generation_prompt:
            rendered += 'A:'
        return rendered


class HistoryRewritingTokenizer(ByteTokenizer):
    """Rewrites non-terminal assistant content; divergence is not a tail relocation."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        assert not tokenize
        del kwargs, tools
        rendered = ''
        for index, message in enumerate(messages):
            prefix = {'system': 'S:', 'user': 'U:', 'assistant': 'A:', 'tool': 'T:'}[message['role']]
            content = message.get('content') or ''
            if message['role'] == 'assistant' and index < len(messages) - 1:
                content = content.replace('<think>', '')
            rendered += prefix + content + ';'
        if add_generation_prompt:
            rendered += 'A:'
        return rendered


class EmptySpanRejectingTokenizer(HistoryRewritingTokenizer):
    def encode(self, text, add_special_tokens=False):
        if not text:
            raise ValueError('empty attribution span')
        return super().encode(text, add_special_tokens=add_special_tokens)


class DigramMergeTokenizer(MergedToolBlockTokenizer):
    """One-shot encoding merges '>|' into a single id; span-wise encoding keeps the bytes apart."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        index = 0
        while index < len(text):
            if text[index : index + 2] == '>|':
                ids.append(999)
                index += 2
            else:
                ids.append(ord(text[index]))
                index += 1
        return ids

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return ''.join('>|' if token == 999 else chr(token) for token in token_ids)


class TestMergedToolBlockRelocation:
    @pytest.mark.asyncio
    async def test_consecutive_tool_results_relocate_the_shared_closure(self):
        processor = Processor(QueuedBackend(['done']), tokenizer=MergedToolBlockTokenizer())
        messages = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'call'},
            {'role': 'tool', 'content': 'a'},
            {'role': 'tool', 'content': 'b'},
        ]

        raw_prompt_ids, prompt_ids, _, render = await processor.render_and_tokenize(messages, None)

        assert render.text == 'U:hi;A:call;U[R<a>|R<b>]A:'
        assert render.templated_messages == ['U:hi;', 'A:call;', 'U[R<a>', '|R<b>]']
        assert render.generation_prompt == 'A:'
        assert raw_prompt_ids.tolist() == prompt_ids.tolist() == list(b'U:hi;A:call;U[R<a>|R<b>]A:')

    async def test_backend_receives_the_relocated_prompt_ids(self):
        backend = QueuedBackend(['done'])
        processor = Processor(backend, tokenizer=MergedToolBlockTokenizer())
        messages = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'call'},
            {'role': 'tool', 'content': 'a'},
            {'role': 'tool', 'content': 'b'},
        ]

        result = await processor.create_completion('merged-tools', messages)

        assert backend.prompts[0] == list(b'U:hi;A:call;U[R<a>|R<b>]A:')
        assert result.prompt_ids.tolist() == backend.prompts[0]

    @pytest.mark.asyncio
    async def test_three_consecutive_tool_results_relocate_iteratively(self):
        processor = Processor(QueuedBackend([]), tokenizer=MergedToolBlockTokenizer())
        messages = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'call'},
            {'role': 'tool', 'content': 'a'},
            {'role': 'tool', 'content': 'b'},
            {'role': 'tool', 'content': 'c'},
        ]

        _, _, _, render = await processor.render_and_tokenize(messages, None)

        assert render.templated_messages == ['U:hi;', 'A:call;', 'U[R<a>', '|R<b>', '|R<c>]']
        assert ''.join(render.templated_messages) + render.generation_prompt == render.text

    @pytest.mark.asyncio
    async def test_messages_after_the_tool_group_append_normally(self):
        processor = Processor(QueuedBackend([]), tokenizer=MergedToolBlockTokenizer())
        messages = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'call'},
            {'role': 'tool', 'content': 'a'},
            {'role': 'tool', 'content': 'b'},
            {'role': 'user', 'content': 'next'},
        ]

        _, _, _, render = await processor.render_and_tokenize(messages, None)

        assert render.templated_messages == ['U:hi;', 'A:call;', 'U[R<a>', '|R<b>]', 'U:next;']

    @pytest.mark.asyncio
    async def test_history_rewrite_uses_lossless_span_fallback(self):
        processor = Processor(QueuedBackend([]), tokenizer=HistoryRewritingTokenizer())
        messages = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '<think>x'},
            {'role': 'user', 'content': 'next'},
        ]

        _, _, _, render = await processor.render_and_tokenize(messages, None)

        assert render.text == 'U:q;A:x;U:next;A:'
        assert render.templated_messages == ['', '', 'U:q;A:x;U:next;']
        assert [ids.tolist() for ids in render.message_token_ids] == [[], [], list(b'U:q;A:x;U:next;')]
        assert render.generation_prompt == 'A:'

    async def test_attribution_encoding_failure_keeps_the_canonical_prompt(self):
        backend = QueuedBackend(['answer'])
        session = _gateway_session('attribution-encoding', Processor(backend, tokenizer=EmptySpanRejectingTokenizer()))
        messages = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '<think>x'},
            {'role': 'user', 'content': 'next'},
        ]

        await session.create_completion(messages)

        assert backend.prompts == [list(b'U:q;A:x;U:next;A:')]
        nodes = list(session.get_trajectory().iter_nodes())
        assert [node.templated_message for node in nodes] == [
            '',
            '',
            'U:q;A:x;U:next;',
            'A:answer',
        ]

    async def test_cached_prefix_stays_frozen_while_the_new_tail_relocates(self):
        backend = QueuedBackend(['ok1', 'ok2'])
        session = _gateway_session(
            'merged-cache',
            Processor(backend, tokenizer=MergedToolBlockTokenizer()),
            scaffold_policy='generic',
        )
        first_messages = [{'role': 'user', 'content': 'hi'}]
        first = await session.create_completion(first_messages)
        second_messages = [
            *first_messages,
            first.message,
            {'role': 'tool', 'content': 'a'},
            {'role': 'tool', 'content': 'b'},
        ]

        await session.create_completion(second_messages)

        assert backend.prompts[1] == list(b'U:hi;A:ok1;U[R<a>|R<b>]A:')
        frozen = backend.prompts[0] + list(b'ok1')
        assert backend.prompts[1][: len(frozen)] == frozen
        node_lists = session.get_trajectory().to_segment_node_lists()
        assert len(node_lists) == 1
        assert len(node_lists[0].response_nodes) == 2

    async def test_cross_call_closure_rewrite_fails_when_legacy_tail_is_not_derivable(self):
        backend = QueuedBackend(['ok1', 'ok2', 'ok3'])
        session = _gateway_session(
            'merged-guard',
            Processor(backend, tokenizer=MergedToolBlockTokenizer()),
            scaffold_policy='generic',
        )
        first_messages = [{'role': 'user', 'content': 'hi'}]
        first = await session.create_completion(first_messages)
        one_tool = [*first_messages, first.message, {'role': 'tool', 'content': 'a'}]
        await session.create_completion(one_tool)
        two_tools = [*one_tool, {'role': 'tool', 'content': 'b'}]

        with pytest.raises(ValueError, match='legacy-compatible continuation tail'):
            await session.create_completion(two_tools)

        assert len(backend.prompts) == 2
        assert all(
            _segment_kind(node_list) == 'conversation' for node_list in session.get_trajectory().to_segment_node_lists()
        )

    @pytest.mark.asyncio
    async def test_canonical_encoding_is_authoritative_across_the_relocated_boundary(self):
        """The '>|' digram straddles the relocated tool-block boundary: the canonical
        (whole-text) merge must reach the engine; the straddling token is attributed to
        the earlier span and the concatenation stays equal to the canonical encode."""
        tokenizer = DigramMergeTokenizer()
        processor = Processor(QueuedBackend([]), tokenizer=tokenizer)
        messages = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'call'},
            {'role': 'tool', 'content': 'a'},
            {'role': 'tool', 'content': 'b'},
        ]

        _, _, _, render = await processor.render_and_tokenize(messages, None)

        assert render.token_ids.tolist() == tokenizer.encode(render.text)
        assert 999 in render.token_ids
        flattened = [token for ids in render.message_token_ids for token in ids]
        assert render.token_ids.tolist() == flattened + render.generation_prompt_ids.tolist()


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit_resume', [False, True])
async def test_codex_native_summary_and_resume_keep_canonical_token_compositions(explicit_resume):
    backend = QueuedBackend(['before', 'summary', 'done'])
    session = _gateway_session('codex-native', Processor(backend, tokenizer=ByteTokenizer()), scaffold_policy='codex')
    initial = [{'role': 'user', 'content': 'solve'}]
    before = await session.create_completion(initial)
    await session.create_completion(
        [*initial, before.message, {'role': 'user', 'content': 'summarize'}],
        request_meta={
            'scaffold': 'codex',
            'codex': {'request_kind': 'compaction'},
            'invocation_kind': 'summary_inference',
        },
    )
    await session.create_completion(
        [{'role': 'user', 'content': 'resume from summary'}],
        request_meta={'invocation_kind': 'compaction_resume'} if explicit_resume else None,
    )
    await session.finish_pending_publications()
    trajectory = session.end()
    compositions = trajectory.to_segment_node_lists()

    assert [_segment_kind(composition) for composition in compositions] == [
        'conversation',
        'summary_inference',
        'compaction_resume' if explicit_resume else 'render_restart',
    ]
    assert [_response_axis(composition)[0] for composition in compositions] == [
        list(text.encode()) for text in ['before', 'summary', 'done']
    ]
    for composition, prompt_ids in zip(compositions, backend.prompts, strict=True):
        token_ids = [token_id for node in composition.nodes for token_id in node.token_account.token_ids]
        assert token_ids == prompt_ids + _response_axis(composition)[0]
    assert len(compositions) == 3
    assert all(composition.response_nodes for composition in compositions)


class ThinkPromptTokenizer(ByteTokenizer):
    """Open a think block for the reply unless reasoning_effort is no_think, which renders it closed."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None, **kwargs):
        closed = '<think></think>' if kwargs.get('reasoning_effort') == 'no_think' else ''
        messages = [
            {**message, 'content': closed + (message.get('content') or '')}
            if message['role'] == 'assistant'
            else message
            for message in messages
        ]
        rendered = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, tools=tools
        )
        return rendered + (closed or '<think>') if add_generation_prompt else rendered


class RawTextParser:
    think_start_token = '<think>'
    think_end_token = '</think>'

    async def extract_tool_calls(self, token_ids, request=None):
        del request
        text = bytes(token_ids).decode()
        return text, text, []


class TestReasoningSplit:
    async def test_open_think_block_splits_reasoning(self) -> None:
        processor = Processor(
            QueuedBackend(['reason</think>answer']),
            tokenizer=ThinkPromptTokenizer(),
            tool_parser=RawTextParser(),
            apply_chat_template_kwargs={'reasoning_effort': 'high'},
        )

        result = await processor.create_completion('open', [{'role': 'user', 'content': 'q'}])

        assert (result.message['reasoning_content'], result.message['content']) == ('reason', 'answer')

    async def test_template_closed_think_block_keeps_sampled_think_end_in_content(self) -> None:
        processor = Processor(
            QueuedBackend(['answer</think>tail']),
            tokenizer=ThinkPromptTokenizer(),
            tool_parser=RawTextParser(),
            apply_chat_template_kwargs={'reasoning_effort': 'no_think'},
        )

        result = await processor.create_completion('closed', [{'role': 'user', 'content': 'q'}])

        assert result.message['content'] == 'answer</think>tail'
        assert 'reasoning_content' not in result.message

    async def test_prompt_without_think_markup_still_splits(self) -> None:
        # Templates that open thinking implicitly put no think markup in the prompt.
        processor = Processor(
            QueuedBackend(['reason</think>answer']),
            tokenizer=ByteTokenizer(),
            tool_parser=RawTextParser(),
        )

        result = await processor.create_completion('implicit', [{'role': 'user', 'content': 'q'}])

        assert (result.message['reasoning_content'], result.message['content']) == ('reason', 'answer')

    async def test_sampled_think_end_re_renders_without_splice_warning(self, caplog) -> None:
        processor = Processor(
            QueuedBackend(['answer</think>tail', 'done']),
            tokenizer=ThinkPromptTokenizer(),
            tool_parser=RawTextParser(),
            apply_chat_template_kwargs={'reasoning_effort': 'no_think'},
        )
        session = _gateway_session('think-end-round-trip', processor)
        user = {'role': 'user', 'content': 'q'}

        first = await session.create_completion([user])
        with caplog.at_level(logging.WARNING, logger='nexus.gateway.processor'):
            await session.create_completion([user, first.message, {'role': 'user', 'content': 'next'}])

        assert not any('No Worries' in record.getMessage() for record in caplog.records)
