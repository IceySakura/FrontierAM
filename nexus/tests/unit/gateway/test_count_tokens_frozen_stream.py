"""Design law: ``count_tokens`` counts the prompt the session would actually send.

The compaction trigger compares ``current_prompt_tokens`` against a budget, so the
count must measure real serving cost — the frozen sampled stream plus the rendered
tail, exactly like the request path. A fresh canonical render is the wrong ruler:
whenever an emission violates ``render(parse(x)) == x`` (e.g. an invalid tool_call
block the parser drops) the re-render undercounts the bytes the next request will
resend verbatim.
"""

from __future__ import annotations

import re

from collections.abc import AsyncIterator
from typing import Any

import pytest

from _fakes import FakeTokenizer, RecordingEndpoint, memory_store

from nexus.core.openai_client import OpenAIConfig
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.session_manager import session_manager


class JunkDroppingToolParser:
    """Parser whose content excises ``<junk>…</junk>`` blocks (round-trip violation)."""

    def __init__(self, tokenizer: FakeTokenizer):
        self._tokenizer = tokenizer

    async def extract_tool_calls(self, token_ids: list[int], request: Any = None) -> tuple[str, str, list[Any]]:
        del request
        text = self._tokenizer.decode(token_ids)
        return text, re.sub(r'<junk>.*?</junk>', '', text, flags=re.S), []


@pytest.fixture(autouse=True)
async def _clean_singleton() -> AsyncIterator[None]:
    session_manager.end_all()
    yield
    session = session_manager.get_session('traj-count')
    if session is not None:
        await session.finish_pending_publications()
    session_manager.end_all()


def _build_client(tokenizer: FakeTokenizer, endpoint: RecordingEndpoint) -> GatewayOpenAIClient:
    return GatewayOpenAIClient(
        OpenAIConfig(
            client_args={
                'tokenizer': tokenizer,
                'server_manager': endpoint,
                'tool_parser': JunkDroppingToolParser(tokenizer),
                'trajectory_id': 'traj-count',
                'record_store': memory_store('traj-count'),
                'scaffold_policy': 'generic',
                'metadata': {'trajectory_id': 'traj-count'},
            },
            request_args={'sampling_params': {'max_tokens': 100, 'temperature': 1.0}},
        )
    )


class TestCountTokensCountsTheFrozenStream:
    async def test_count_equals_next_request_prompt_length_under_parse_loss(self):
        tokenizer = FakeTokenizer()
        endpoint = RecordingEndpoint(tokenizer, ['PLAN<junk>DROPPED BY PARSER</junk>DONE', 'FINAL'])
        client = _build_client(tokenizer, endpoint)

        messages: list[dict] = [{'role': 'user', 'content': 'q'}]
        completion = await client.create_completion(messages=list(messages))
        assistant = completion.choices[0].message.model_dump(exclude_none=True)
        assert '<junk>' not in (assistant.get('content') or '')  # parse loss happened
        messages += [assistant, {'role': 'tool', 'content': 'obs'}]

        counted = await client.count_tokens(messages=list(messages))
        second = await client.create_completion(messages=list(messages))
        del second

        actual_next_prompt = endpoint.calls[1]['prompt_ids']
        assert counted.input_tokens == len(actual_next_prompt)
        # The canonical re-render is strictly shorter (the junk bytes exist only in
        # the frozen stream) — the old counting would have returned this number.
        canonical = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        assert counted.input_tokens > len(tokenizer.encode(canonical))

    async def test_count_without_session_falls_back_to_canonical_render(self):
        tokenizer = FakeTokenizer()
        client = _build_client(tokenizer, RecordingEndpoint(tokenizer, []))

        messages = [{'role': 'user', 'content': 'q'}]
        counted = await client.count_tokens(messages=list(messages))

        canonical = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        assert counted.input_tokens == len(tokenizer.encode(canonical))

    async def test_update_token_consumption_accounts_the_frozen_stream(self):
        tokenizer = FakeTokenizer()
        endpoint = RecordingEndpoint(tokenizer, ['PLAN<junk>DROPPED BY PARSER</junk>DONE', 'FINAL'])
        client = _build_client(tokenizer, endpoint)

        messages: list[dict] = [{'role': 'user', 'content': 'q'}]
        completion = await client.create_completion(messages=list(messages))
        messages += [completion.choices[0].message.model_dump(exclude_none=True), {'role': 'tool', 'content': 'obs'}]

        await client.update_token_consumption(list(messages))
        state = client._session()._token_budget_state
        accounted = [list(ids) for ids in state._accounted_prompt_ids.values()]
        assert len(accounted) == 1

        second = await client.create_completion(messages=list(messages))
        del second
        # The accounting ruler must equal the stream the next request actually sent
        # (frozen splice), not a fresh canonical render of the message view.
        assert accounted[0] == endpoint.calls[1]['prompt_ids']

    async def test_count_normalizes_string_arguments_without_mutating_history(self):
        tokenizer = FakeTokenizer()
        client = _build_client(tokenizer, RecordingEndpoint(tokenizer, []))

        string_args = [
            {'role': 'user', 'content': 'q'},
            {
                'role': 'assistant',
                'content': 'x',
                'tool_calls': [{'id': '1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"a": 1}'}}],
            },
            {'role': 'tool', 'content': 'obs'},
        ]
        dict_args = [dict(m) for m in string_args]
        dict_args[1] = {
            **dict_args[1],
            'tool_calls': [{'id': '1', 'type': 'function', 'function': {'name': 'f', 'arguments': {'a': 1}}}],
        }

        counted = await client.count_tokens(messages=string_args)

        assert counted.input_tokens == (await client.count_tokens(messages=dict_args)).input_tokens
        # Read-only: the caller's history must keep its wire-format string.
        assert string_args[1]['tool_calls'][0]['function']['arguments'] == '{"a": 1}'
