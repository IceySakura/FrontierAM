import pytest

from _fakes import FakeTokenizer, RecordingEndpoint, memory_store

from nexus.core import UNPARSED_TOOL_CALL_FIELD, OpenAIConfig
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.session_manager import session_manager


_PARSED_CALL = '<tool_calls>bash</tool_calls>'


class _MarkerToolParser:
    """Parses only the exact well-formed call; anything else comes back as raw text."""

    tool_calls_start_token = '<tool_calls>'

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self._tokenizer = tokenizer

    async def extract_tool_calls(self, token_ids, request=None):  # noqa: ANN001, ANN201
        del request
        text = self._tokenizer.decode(token_ids)
        if text == _PARSED_CALL:
            return text, '', [{'name': 'bash', 'arguments': '{}'}]
        return text, text, []


class _UnnamedMarkerToolParser(_MarkerToolParser):
    tool_calls_start_token = None


async def _first_choice(parser_cls: type[_MarkerToolParser], completion: str, trajectory_id: str):  # noqa: ANN202
    tokenizer = FakeTokenizer()
    gateway = GatewayOpenAIClient(
        OpenAIConfig(
            client_args={
                'tokenizer': tokenizer,
                'server_manager': RecordingEndpoint(tokenizer, [completion]),
                'tool_parser': parser_cls(tokenizer),
                'trajectory_id': trajectory_id,
                'record_store': memory_store(trajectory_id),
                'scaffold_policy': 'generic',
            },
            request_args={'sampling_params': {'max_tokens': 100}},
        )
    )
    try:
        response = await gateway.create_completion([{'role': 'user', 'content': 'q'}])
        return response.choices[0]
    finally:
        await gateway._session().finish_pending_publications()
        session_manager.end_session(trajectory_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('parser_cls', 'completion', 'flagged'),
    [
        pytest.param(_MarkerToolParser, 'thinking <tool_calls>bash', True, id='rejected-tool-call'),
        pytest.param(_MarkerToolParser, 'the answer is 42', False, id='plain-answer'),
        pytest.param(_MarkerToolParser, _PARSED_CALL, False, id='parsed-tool-call'),
        pytest.param(_UnnamedMarkerToolParser, 'thinking <tool_calls>bash', False, id='parser-without-marker'),
    ],
)
async def test_gateway_flags_a_completion_that_left_a_tool_call_unparsed(
    parser_cls: type[_MarkerToolParser], completion: str, flagged: bool, request: pytest.FixtureRequest
) -> None:
    choice = await _first_choice(parser_cls, completion, f'unparsed-tool-call-{request.node.callspec.id}')

    provider_fields = choice.provider_specific_fields or {}
    assert provider_fields.get(UNPARSED_TOOL_CALL_FIELD, False) is flagged
    assert choice.message.provider_specific_fields is None or (
        UNPARSED_TOOL_CALL_FIELD not in choice.message.provider_specific_fields
    )
