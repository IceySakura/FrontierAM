"""Claude Code -> anthropic_proxy -> llm_handler -> litellm Gemini replay of thinking summaries.

Gemini signs at message level (``provider_specific_fields.thought_signatures``) and on
functionCall parts (encoded in ``tool_call.id``), never on the summary itself. The
anthropic_proxy therefore hands Claude Code a thinking block with ``signature=''``,
which Claude Code echoes back on the next turn. These tests pin down what reaches
Gemini for each thinking preset.
"""

from __future__ import annotations

import asyncio

import pytest

from litellm.llms.vertex_ai.gemini.transformation import _gemini_convert_messages_with_history

from nexus.runtime.server.services.anthropic_proxy.converter import (
    convert_messages_to_openai,
    convert_openai_content_to_anthropic,
)
from nexus.runtime.server.services.anthropic_proxy.models import Message, get_thinking_config
from nexus.runtime.server.services.llm_handler import LLMRequestHandler


TOOL_CALL_ID = 'call_1__thought__U0lHX0ZD'
SUMMARY = 'SUMMARY TEXT'
CONTENT = 'Let me check.'


def _gemini_response_message() -> dict:
    return {
        'role': 'assistant',
        'content': CONTENT,
        'reasoning_content': SUMMARY,
        'thinking_blocks': [{'type': 'thinking', 'thinking': SUMMARY}],
        'provider_specific_fields': {'thought_signatures': ['SIG_TEXT']},
        'tool_calls': [
            {
                'id': TOOL_CALL_ID,
                'type': 'function',
                'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'},
            }
        ],
    }


class _FakeRuntime:
    def __init__(self) -> None:
        self.emitted: list = []

    async def emit(self, event, data):  # noqa: ANN001, ANN202
        self.emitted.append((event, data))

    def on(self, event, handler):  # noqa: ANN001, ANN202
        pass

    def off(self, event, handler):  # noqa: ANN001, ANN202
        pass


class _GeminiClient:
    class Config:
        request_args: dict = {}
        client_args: dict = {}

    config = Config()

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def create_completion(self, **kwargs):  # noqa: ANN003, ANN202
        self.requests.append(kwargs)
        await asyncio.sleep(0)
        return {'choices': [{'message': _gemini_response_message(), 'finish_reason': 'tool_calls'}]}


def _model_parts(openai_messages: list[dict]) -> list[dict]:
    contents = _gemini_convert_messages_with_history(messages=openai_messages, model='gemini-4-argon')
    model_turns = [c for c in contents if c['role'] == 'model']
    assert len(model_turns) == 1
    return list(model_turns[0]['parts'])


async def _round_trip(preset: str) -> list[dict]:
    """Return the Gemini parts of the replayed assistant turn on the second request."""
    cfg = get_thinking_config(preset)
    client = _GeminiClient()
    handler = LLMRequestHandler(runtime=_FakeRuntime(), openai_client=client, preserve_thinking=True)

    user = Message(role='user', content='weather in Paris?')
    await handler.handle_llm_request({'request_id': 'r1', 'messages': convert_messages_to_openai([user], cfg)})

    # What Claude Code receives and stores in its own history.
    blocks = convert_openai_content_to_anthropic(_gemini_response_message(), thinking_config=cfg)
    assistant = Message(role='assistant', content=[b.model_dump() for b in blocks])
    thinking = [b for b in assistant.model_dump()['content'] if b['type'] == 'thinking']
    assert thinking
    assert thinking[0]['signature'] == ''

    tool_result = Message(role='user', content=[{'type': 'tool_result', 'tool_use_id': TOOL_CALL_ID, 'content': '18C'}])
    second = convert_messages_to_openai([user, assistant, tool_result], cfg)
    await handler.handle_llm_request({'request_id': 'r2', 'messages': second})

    assert len(client.requests) == 2
    return _model_parts(client.requests[1]['messages'])


def _text_parts(parts: list[dict]) -> list[dict]:
    return [p for p in parts if 'text' in p and not p.get('thought')]


@pytest.mark.asyncio
@pytest.mark.parametrize('preset', ['hunyuan', 'bedrock-claude'])
async def test_summary_is_never_replayed_as_visible_text(preset: str) -> None:
    parts = await _round_trip(preset)

    assert all(p.get('thoughtSignature') != '' for p in parts)
    assert [p['text'] for p in _text_parts(parts)] == [CONTENT]


@pytest.mark.asyncio
@pytest.mark.parametrize('preset', ['hunyuan', 'bedrock-claude'])
async def test_text_and_function_call_signatures_are_restored(preset: str) -> None:
    parts = await _round_trip(preset)

    text = _text_parts(parts)
    assert text == [{'text': CONTENT, 'thoughtSignature': 'SIG_TEXT'}]
    fn = [p for p in parts if 'function_call' in p]
    assert len(fn) == 1
    assert fn[0]['thoughtSignature'] == 'U0lHX0ZD'


@pytest.mark.asyncio
async def test_hunyuan_replays_summary_as_thought_part() -> None:
    parts = await _round_trip('hunyuan')

    assert {'thought': True, 'text': SUMMARY} in parts


@pytest.mark.asyncio
async def test_bedrock_claude_drops_unsigned_summary() -> None:
    parts = await _round_trip('bedrock-claude')

    assert all(SUMMARY not in (p.get('text') or '') for p in parts)
