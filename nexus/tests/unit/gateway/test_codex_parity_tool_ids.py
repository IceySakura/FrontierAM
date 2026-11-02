import pytest

from pydantic import BaseModel

from nexus.core import OpenAIConfig
from nexus.extensions.openai_clients.hytuner import HyTunerClient
from nexus.gateway.processor import Processor


class ParsedFunction(BaseModel):
    name: str
    arguments: str


class ToolParser:
    async def extract_tool_calls(self, token_ids, request=None):
        return ('raw', '', [ParsedFunction(name='exec_command', arguments='{}') for _ in token_ids])


@pytest.mark.asyncio
async def test_parity_tool_ids_match_across_clients_and_remain_unique_across_turns(monkeypatch):
    monkeypatch.setenv('NEXUS_DETERMINISTIC_TOOL_CALL_IDS', '1')
    parser = ToolParser()
    legacy = HyTunerClient(OpenAIConfig(client_args={'tool_parser': parser}, request_args={}))
    gateway = Processor(backend=None, tool_parser=parser)

    for token_ids, expected in [
        ([1, 2], ['call_000000000000000000000000', 'call_000000000000000000000001']),
        ([3], ['call_000000000000000000000002']),
    ]:
        legacy_calls = (await legacy._parse_tool_calls(token_ids))[2]
        gateway_calls = (await gateway._parse_tool_calls(token_ids, '', None))[2]
        assert [call['id'] for call in legacy_calls] == expected
        assert [call['id'] for call in gateway_calls] == expected

    fresh = HyTunerClient(OpenAIConfig(client_args={'tool_parser': parser}, request_args={}))
    assert (await fresh._parse_tool_calls([1]))[2][0]['id'] == 'call_000000000000000000000000'


@pytest.mark.asyncio
async def test_random_ids_remain_default(monkeypatch):
    monkeypatch.delenv('NEXUS_DETERMINISTIC_TOOL_CALL_IDS', raising=False)
    parser = ToolParser()
    legacy = HyTunerClient(OpenAIConfig(client_args={'tool_parser': parser}, request_args={}))
    gateway = Processor(backend=None, tool_parser=parser)

    legacy_call = (await legacy._parse_tool_calls([1]))[2][0]['id']
    gateway_call = (await gateway._parse_tool_calls([1], '', None))[2][0]['id']
    assert legacy_call != gateway_call
    assert legacy_call != 'call_000000000000000000000000'
    assert gateway_call != 'call_000000000000000000000000'
