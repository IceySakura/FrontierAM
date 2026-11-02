"""Check mode-aware reasoning conversion and Context Agent judging semantics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.core import ExecutionContext, OpenAIConfig
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.context_agent.loop import run_tool_call_loop
from nexus.extensions.openai_clients import hytuner
from nexus.extensions.tasks.context_agent import general_task
from nexus.runtime.providers.local import LocalRuntimeProvider


THINK_BEGIN = '<think:6124c78e>'
THINK_END = '</think:6124c78e>'
TOOL_BLOCK = (
    '<tool_calls:6124c78e><tool_call:6124c78e>bash'
    '<arg_key:6124c78e>command</arg_key:6124c78e>'
    '<arg_value:6124c78e>cat document.md</arg_value:6124c78e>'
    '</tool_call:6124c78e></tool_calls:6124c78e>'
)
TOOLS = [{'type': 'function', 'function': {'name': 'bash', 'parameters': {'type': 'object'}}}]
PARSED_CALLS = [
    {'id': 'call_test', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command":"cat document.md"}'}}
]


class _Tokenizer:
    def __init__(self, output: str) -> None:
        self.output = output

    def apply_chat_template(self, messages: list[dict], *, add_generation_prompt: bool, **kwargs: Any) -> str:
        suffix = THINK_BEGIN if kwargs.get('reasoning_effort', 'no_think') != 'no_think' else ''
        return 'prompt' + (suffix if add_generation_prompt else '')

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(text.encode())

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return self.output


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch):
    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    # Keep the test independent of the sandbox's thread wakeup support.
    monkeypatch.setattr(hytuner.asyncio, 'to_thread', run_inline)
    monkeypatch.setattr(hytuner, 'get_current_span_id', lambda: None)

    def build(output: str, others: str, *, effort: str = 'high', calls=None, stop_reason: str = 'stop'):
        client = hytuner.HyTunerClient(
            OpenAIConfig(
                client_args={
                    'tokenizer': _Tokenizer(output),
                    'tool_parser': SimpleNamespace(think_start_token=THINK_BEGIN, think_end_token=THINK_END),
                    'apply_chat_template_kwargs': {'reasoning_effort': effort},
                },
                request_args={},
            )
        )
        monkeypatch.setattr(client, '_parse_tool_calls', AsyncMock(return_value=(output, others, calls or [])))
        client._cached_generate = AsyncMock(
            return_value=SimpleNamespace(
                token_ids=list(output.encode()),
                log_probs=None,
                stop_reason=stop_reason,
                extra_fields={},
            )
        )
        return client

    return build


@pytest.mark.parametrize('auxiliary', [False, True])
@pytest.mark.parametrize('explicit_start', [False, True])
async def test_unclosed_thinking_preserves_reasoning_and_stops(
    make_client: Any, auxiliary: bool, explicit_start: bool
) -> None:
    reasoning = 'I need to inspect the file first.\n' + TOOL_BLOCK
    raw = (THINK_BEGIN if explicit_start else '') + reasoning
    client = make_client(raw, raw)
    completion_fn = client._auxiliary_completion if auxiliary else client._create_completion
    completion = await completion_fn([{'role': 'user', 'content': 'Read the document.'}], tools=TOOLS)
    choice = completion.choices[0]
    assert choice.finish_reason == 'stop'
    assert choice.message.content == ''
    assert choice.message.model_dump(exclude_none=True)['content'] == ''
    assert choice.message.reasoning_content == reasoning
    assert not choice.message.tool_calls
    assert choice.message.raw_response == raw
    if not auxiliary:
        assert client.last_messages[-1]['content'] == ''
        assert client.last_tokens[-len(raw.encode()) :] == list(raw.encode())
        assert completion.usage.completion_tokens_details.reasoning_tokens == len(raw.encode())
        assert completion.usage.completion_tokens_details.text_tokens == 0


@pytest.mark.parametrize('auxiliary', [False, True])
@pytest.mark.parametrize('stop_reason', ['stop', 'length'])
async def test_exhausted_token_budget_keeps_length_status(make_client: Any, auxiliary: bool, stop_reason: str) -> None:
    raw = 'unfinished thinking'
    client = make_client(raw, raw, stop_reason=stop_reason)
    completion_fn = client._auxiliary_completion if auxiliary else client._create_completion
    completion = await completion_fn(
        [{'role': 'user', 'content': 'Read.'}], sampling_params={'max_tokens': len(raw.encode())}
    )
    assert completion.choices[0].finish_reason == 'length'
    assert completion.choices[0].message.content == ''
    assert completion.choices[0].message.reasoning_content == raw


@pytest.mark.parametrize('auxiliary', [False, True])
async def test_closed_reasoning_preserves_valid_tool_calls(make_client: Any, auxiliary: bool) -> None:
    raw = 'Read the file.' + THINK_END + TOOL_BLOCK
    client = make_client(raw, 'Read the file.' + THINK_END, calls=PARSED_CALLS)
    completion_fn = client._auxiliary_completion if auxiliary else client._create_completion
    completion = await completion_fn([{'role': 'user', 'content': 'Read.'}], tools=TOOLS)
    assert completion.choices[0].finish_reason == 'tool_calls'
    assert completion.choices[0].message.tool_calls[0].function.name == 'bash'
    assert completion.choices[0].message.reasoning_content == 'Read the file.'


@pytest.mark.parametrize('auxiliary', [False, True])
@pytest.mark.parametrize('effort', ['high', 'no_think'])
async def test_final_answer_and_no_think_are_preserved(make_client: Any, auxiliary: bool, effort: str) -> None:
    raw = ('Reasoning.' + THINK_END if effort == 'high' else '') + 'Final answer.'
    client = make_client(raw, raw, effort=effort)
    completion_fn = client._auxiliary_completion if auxiliary else client._create_completion
    completion = await completion_fn([{'role': 'user', 'content': 'Read.'}])
    assert completion.choices[0].finish_reason == 'stop'
    assert completion.choices[0].message.content == 'Final answer.'


async def test_no_think_answer_reaches_the_judge(make_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = 'The correct answer is (D)'
    client = make_client(raw, raw, effort='no_think')
    completion = await client._create_completion([{'role': 'user', 'content': 'Choose an option.'}])
    execute_tool = AsyncMock(side_effect=AssertionError('No parsed tool calls were returned'))
    loop_result = await run_tool_call_loop(
        client=SimpleNamespace(create_completion=AsyncMock(return_value=completion)),
        conversation=[],
        tool_schemas=TOOLS,
        max_tool_rounds=20,
        execute_tool=execute_tool,
    )
    assert loop_result.exit_status == TerminationReason.SUCCESS
    assert loop_result.final_answer == raw
    execute_tool.assert_not_called()
    evaluate = AsyncMock(wraps=general_task.evaluate)
    monkeypatch.setattr(general_task, 'evaluate', evaluate)
    monkeypatch.setattr(general_task, 'resolve_judge_llm', lambda *_args: None)
    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await general_task.run_context_agent_general_task(
            general_task.ContextAgentGeneralTaskInput(
                run_agent=AsyncMock(return_value=loop_result),
                context_question='Choose an option.',
                dataset_name='LongBenchV2',
                answer='D',
            )
        )
    assert output.score == 1.0
    assert output.termination_reason == 'ENV_DONE'
    assert output.evaluation_skipped is False
    evaluate.assert_awaited_once()
    assert evaluate.call_args.args[0] == raw


async def test_format_error_status_does_not_skip_judge_for_existing_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluate = AsyncMock(wraps=general_task.evaluate)
    monkeypatch.setattr(general_task, 'evaluate', evaluate)
    monkeypatch.setattr(general_task, 'resolve_judge_llm', lambda *_args: None)
    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await general_task.run_context_agent_general_task(
            general_task.ContextAgentGeneralTaskInput(
                run_agent=AsyncMock(
                    return_value=SimpleNamespace(
                        final_answer='The correct answer is (D)', exit_status=TerminationReason.FORMAT_ERROR
                    )
                ),
                context_question='Choose an option.',
                dataset_name='LongBenchV2',
                answer='D',
            )
        )
    assert output.score == 1.0
    assert output.evaluation_skipped is False
    evaluate.assert_awaited_once()


@pytest.mark.parametrize('auxiliary', [False, True])
async def test_thinking_without_parser_boundary_preserves_reasoning(make_client: Any, auxiliary: bool) -> None:
    raw = 'Only reasoning.' + THINK_END
    client = make_client(raw, 'Only reasoning.')
    completion_fn = client._auxiliary_completion if auxiliary else client._create_completion
    completion = await completion_fn([{'role': 'user', 'content': 'Read.'}])
    assert completion.choices[0].finish_reason == 'stop'
    assert completion.choices[0].message.content == ''
    assert completion.choices[0].message.reasoning_content == 'Only reasoning.'


@pytest.mark.parametrize('auxiliary', [False, True])
@pytest.mark.parametrize('effort', ['high', 'no_think'])
async def test_client_preserves_parser_tool_calls_without_reasoning_boundary(
    make_client: Any, auxiliary: bool, effort: str
) -> None:
    client = make_client(TOOL_BLOCK, '', effort=effort, calls=PARSED_CALLS)
    completion_fn = client._auxiliary_completion if auxiliary else client._create_completion
    completion = await completion_fn([{'role': 'user', 'content': 'Read.'}], tools=TOOLS)
    assert completion.choices[0].finish_reason == 'tool_calls'
    assert completion.choices[0].message.reasoning_content is None
    assert completion.choices[0].message.tool_calls[0].function.name == 'bash'


async def test_unclosed_marker_in_parser_content_only_changes_token_accounting(make_client: Any) -> None:
    raw = THINK_BEGIN + 'Read the file.' + TOOL_BLOCK
    client = make_client(raw, raw, calls=PARSED_CALLS)
    completion = await client._create_completion([{'role': 'user', 'content': 'Read.'}], tools=TOOLS)
    assert completion.choices[0].finish_reason == 'tool_calls'
    assert completion.choices[0].message.tool_calls[0].function.name == 'bash'
    assert completion.usage.completion_tokens_details.reasoning_tokens == len(raw.encode())
    assert completion.usage.completion_tokens_details.text_tokens == 0


async def test_reasoning_only_stop_is_an_empty_final_answer(make_client: Any) -> None:
    client = make_client('Still thinking.', 'Still thinking.')
    completion = await client._create_completion([{'role': 'user', 'content': 'Read.'}])
    execute_tool = AsyncMock()
    result = await run_tool_call_loop(
        client=SimpleNamespace(create_completion=AsyncMock(return_value=completion)),
        conversation=[],
        tool_schemas=TOOLS,
        max_tool_rounds=20,
        execute_tool=execute_tool,
    )
    assert result.exit_status == TerminationReason.FORMAT_ERROR
    assert result.final_answer is None
    execute_tool.assert_not_called()


@pytest.mark.parametrize(
    ('stop_reason', 'expected_exit_status', 'expected_termination_reason'),
    [
        ('stop', TerminationReason.FORMAT_ERROR, 'FORMAT_ERROR'),
        ('length', TerminationReason.OUTPUT_LENGTH_EXCEEDED, 'TRUNCATION'),
    ],
)
async def test_reasoning_only_history_keeps_content_and_termination_reason(
    make_client: Any,
    monkeypatch: pytest.MonkeyPatch,
    stop_reason: str,
    expected_exit_status: TerminationReason,
    expected_termination_reason: str,
) -> None:
    reasoning = 'Still thinking.'
    client = make_client(reasoning, reasoning, stop_reason=stop_reason)
    completion = await client._create_completion([{'role': 'user', 'content': 'Read.'}])
    assert client.last_messages[-1]['content'] == ''
    assert client.last_messages[-1]['reasoning_content'] == reasoning

    execute_tool = AsyncMock()
    loop_result = await run_tool_call_loop(
        client=SimpleNamespace(create_completion=AsyncMock(return_value=completion)),
        conversation=[],
        tool_schemas=TOOLS,
        max_tool_rounds=20,
        execute_tool=execute_tool,
    )
    assert loop_result.exit_status == expected_exit_status
    execute_tool.assert_not_called()

    evaluate = AsyncMock(side_effect=AssertionError('Reasoning without an answer must not reach the Judge'))
    monkeypatch.setattr(general_task, 'evaluate', evaluate)
    async with ExecutionContext(openai_clients={}, runtime_provider=LocalRuntimeProvider()):
        output = await general_task.run_context_agent_general_task(
            general_task.ContextAgentGeneralTaskInput(
                run_agent=AsyncMock(return_value=loop_result),
                context_question='Choose an option.',
                dataset_name='LongBenchV2',
                answer='D',
            )
        )
    assert output.score == 0.0
    assert output.termination_reason == expected_termination_reason
    evaluate.assert_not_awaited()
