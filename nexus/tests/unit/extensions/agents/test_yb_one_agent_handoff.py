import asyncio

from types import SimpleNamespace

import pytest

from nexus.extensions.agents.yb_one_agent.agent import ExitStatus, _agent_loop, _agent_loop_stream
from nexus.extensions.agents.yb_one_agent.prompt_builder import (
    DEFAULT_TOOLS,
    build_system_prompt,
    build_tool_schemas,
)


class _HandoffClient:
    async def create_completion(self, **kwargs):
        del kwargs
        tool_call = SimpleNamespace(
            id='handoff-call',
            function=SimpleNamespace(
                name='handoff',
                arguments='{"subagent_type":"expert_agent"}',
            ),
        )
        message = SimpleNamespace(
            content='',
            reasoning_content=None,
            thinking_blocks=None,
            tool_calls=[tool_call],
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _NeverExecuteTools:
    async def call(self, name, args):
        pytest.fail(f'tool executor must not run after handoff: {name}({args})')


class _StreamingClient:
    config = SimpleNamespace(request_args={'model': 'test-model'})

    def get_effective_request_args(self):
        return {'model': 'test-model'}

    def _apply_enforced_request_args(self, request_args):
        return request_args

    class _Completions:
        async def create(self, **kwargs):
            del kwargs

            async def chunks():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content='done',
                                reasoning_content=None,
                                thinking_blocks=None,
                                tool_calls=None,
                            )
                        )
                    ],
                    usage=None,
                    provider_specific_fields={'inference_time': 0.25},
                )

            return chunks()

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))


def test_handoff_schema_matches_router_sft_contract():
    [schema] = build_tool_schemas(tools={'handoff'})

    assert schema == {
        'type': 'function',
        'function': {
            'name': 'handoff',
            'description': (
                'Hand off the current request to a more capable specialist model. '
                'Call this when the request is complex enough that a stronger model is required to answer it reliably.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'subagent_type': {
                        'type': 'string',
                        'description': 'The specialist to route to. Use "expert_agent" for the stronger model.',
                        'enum': ['expert_agent'],
                    }
                },
                'required': ['subagent_type'],
            },
        },
    }


def test_default_tools_preserve_standard_delegate_guidance():
    prompt = build_system_prompt(tools=DEFAULT_TOOLS, skills=[])
    implicit_default_prompt = build_system_prompt(tools=None, skills=[])
    schema_names = [schema['function']['name'] for schema in build_tool_schemas(DEFAULT_TOOLS)]

    assert 'delegate' in DEFAULT_TOOLS
    assert 'handoff' not in DEFAULT_TOOLS
    assert '使用 `delegate`' in prompt
    assert '- 使用 `handoff`' not in prompt
    assert '- 使用 `handoff`' not in implicit_default_prompt
    assert 'delegate' in schema_names
    assert 'handoff' not in schema_names


def test_router_tool_selection_renders_only_handoff_guidance():
    tools = (set(DEFAULT_TOOLS) - {'delegate'}) | {'handoff'}
    prompt = build_system_prompt(tools=tools, skills=[])
    schema_names = [schema['function']['name'] for schema in build_tool_schemas(tools)]

    assert '- 使用 `handoff`：当你认为用户query你无法处理时，使用handoff工具将请求路由到专家agent' in prompt
    assert '使用 `delegate`' not in prompt
    assert '### delegate工具' not in prompt
    assert 'delegate' not in schema_names
    assert schema_names[-1] == 'handoff'


@pytest.mark.asyncio
async def test_handoff_stops_first_step_without_tool_execution():
    history = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'complex request'},
    ]

    events = [
        event
        async for event in _agent_loop(
            history=history,
            executor=_NeverExecuteTools(),
            client=_HandoffClient(),
            tools=build_tool_schemas(tools={'handoff'}),
        )
    ]

    assert events[-1]['type'] == 'done'
    assert events[-1]['exit_status'] == ExitStatus.HANDOFF.value
    assert events[-1]['iterations'] == 1
    assert len(history) == 3
    assert history[-1] == {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {
                'type': 'function',
                'function': {
                    'name': 'handoff',
                    'arguments': {'subagent_type': 'expert_agent'},
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_non_streaming_inference_timeout_interrupts_unreturned_call():
    class HangingClient:
        async def create_completion(self, **kwargs):
            del kwargs
            await asyncio.Event().wait()

    history = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'request'},
    ]
    events = [
        event
        async for event in _agent_loop(
            history=history,
            executor=_NeverExecuteTools(),
            client=HangingClient(),
            tools=[],
            max_iterations=1,
            completion_retry_on_tool_error=0,
            llm_retry_on_error=0,
            inference_time_timeout=0.01,
        )
    ]

    assert events[-1]['type'] == 'done'
    assert events[-1]['exit_status'] == ExitStatus.TIMEOUT.value


@pytest.mark.asyncio
async def test_streaming_loop_uses_reported_inference_time():
    history = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'request'},
    ]

    events = [
        event
        async for event in _agent_loop_stream(
            history=history,
            executor=_NeverExecuteTools(),
            client=_StreamingClient(),
            tools=[],
            completion_retry_on_tool_error=0,
        )
    ]

    assert events[-1]['type'] == 'done'
    assert events[-1]['exit_status'] == ExitStatus.FINISHED.value
    assert events[-1]['inference_time'] == pytest.approx(0.25)
    assert not any(message['role'] == 'tool' for message in history)
