"""HyTunerClient flags a completion that began a tool call its parser could not extract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import UNPARSED_TOOL_CALL_FIELD, OpenAIConfig
from nexus.extensions.openai_clients.hytuner import HyTunerClient


_PARSED_CALL = '<tool_calls>bash</tool_calls>'
_MALFORMED_CALL = 'run it<tool_calls>code<arg_key>code<arg_key>'


class CharacterTokenizer:
    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        rendered = ''.join(f'<{message["role"]}>{message.get("content") or ""}' for message in messages)
        return rendered + ('<assistant>' if kwargs.get('add_generation_prompt') else '')

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return ''.join(chr(token_id) for token_id in token_ids)


class ServerManager:
    def __init__(self, completion: str, stop_reason: str) -> None:
        self.completion = completion
        self.stop_reason = stop_reason

    async def generate(self, **kwargs: Any) -> SimpleNamespace:
        del kwargs
        token_ids = [ord(character) for character in self.completion]
        return SimpleNamespace(
            token_ids=token_ids,
            log_probs=[-0.1] * len(token_ids),
            stop_reason=self.stop_reason,
            extra_fields={},
        )


class MarkerToolParser:
    """Parses only the exact well-formed call; anything else comes back as raw text."""

    think_start_token = '<think>'
    think_end_token = '</think>'
    tool_calls_start_token: str | None = '<tool_calls>'

    async def extract_tool_calls(self, token_ids: list[int], request: Any = None) -> tuple[str, str, list]:
        del request
        text = ''.join(chr(token_id) for token_id in token_ids)
        if text == _PARSED_CALL:
            return text, '', [SimpleNamespace(model_dump=lambda: {'name': 'bash', 'arguments': '{}'})]
        return text, text, []


class UnnamedMarkerToolParser(MarkerToolParser):
    tool_calls_start_token = None


def _build_client(
    completion: str,
    *,
    parser: MarkerToolParser | None = None,
    stop_reason: str = 'stop',
    **client_args: Any,
) -> HyTunerClient:
    client = HyTunerClient(
        OpenAIConfig(
            client_args={
                'tokenizer': CharacterTokenizer(),
                'tool_parser': parser or MarkerToolParser(),
                'token_budget_include': ['agent_output'],
                **client_args,
            },
            request_args={},
        )
    )
    client.server_manager = ServerManager(completion, stop_reason)
    return client


def _flagged(completion: Any) -> bool:
    provider_fields = completion.choices[0].provider_specific_fields or {}
    return provider_fields.get(UNPARSED_TOOL_CALL_FIELD, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('parser', 'completion', 'flagged'),
    [
        pytest.param(MarkerToolParser(), _MALFORMED_CALL, True, id='rejected-tool-call'),
        pytest.param(MarkerToolParser(), 'the answer is 42', False, id='plain-answer'),
        pytest.param(MarkerToolParser(), _PARSED_CALL, False, id='parsed-tool-call'),
        pytest.param(UnnamedMarkerToolParser(), _MALFORMED_CALL, False, id='parser-without-marker'),
    ],
)
@pytest.mark.parametrize('model', [None, 'judge'], ids=['main', 'auxiliary'])
async def test_client_flags_a_completion_that_left_a_tool_call_unparsed(
    parser: MarkerToolParser, completion: str, flagged: bool, model: str | None
) -> None:
    client = _build_client(completion, parser=parser, auxiliary_model_names=['judge'])

    response = await client.create_completion([{'role': 'user', 'content': 'q'}], model=model)

    assert _flagged(response) is flagged
    if flagged:
        assert response.choices[0].finish_reason == 'stop'
        assert response.choices[0].message.tool_calls is None
