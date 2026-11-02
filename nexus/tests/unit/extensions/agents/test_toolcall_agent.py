from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import BaseModel

import pytest

from nexus.core import (
    ContextLengthExceededError,
    ToolContext,
    ToolExecutor,
    ToolResponse,
    ToolType,
    tool_registry,
)
from nexus.extensions.agents.base.errors import NonRetryableProviderError
from nexus.extensions.agents.base.prompt import TOOL_CALL_REQUIRED
from nexus.extensions.agents.base.schema import TerminationReason, ToolChoice
from nexus.extensions.agents.base.toolcall import ToolCallAgent
from nexus.extensions.agents.base import utils as base_utils


from nexus.extensions.agents.base.evaluation import EvaluationDetails, evaluate_output

from nexus.extensions.agents.base.tools import FinishInput, TodoItem, WriteTodosInput, finish_tool, write_todos_tool


DEFAULT_TOOL_ARGUMENTS = '{"value": "done"}'
FINISH_TOOL_NAME = 'finish'
FINAL_ANSWER = 'final answer'
PLAIN_ANSWER = 'final answer'
TOOL_DECISION = 'tool decision'
HIDDEN_REASONING = 'hidden reasoning'
MESSAGE_SIGNATURE = 'signed-message'
NO_TOOL_TEXT = 'no tool'
ARGUMENT_PARSE_ERROR_PREFIX = f'Argument parse error: {FINISH_TOOL_NAME}:'
RUNTIME_TOOL_ERROR_PREFIX = f'An error occurred when calling tool `{FINISH_TOOL_NAME}`:'
NON_RETRYABLE_TOOL_ERROR_PREFIX = f'An unrecoverable error occurred when calling tool `{FINISH_TOOL_NAME}`:'
UNKNOWN_TOOL_ERROR_PREFIX = f"Tool '{FINISH_TOOL_NAME}' not found."


QUEUED_TASK = 'queued'
ACTIVE_TASK = 'active'
NEXT_STEP_TASK = 'next step'
DONE_MESSAGE = 'done'


class _ToolCallInput(BaseModel):
    value: str


@tool_registry.register(
    'tests/toolcall_finish',
    meta={
        'name': 'finish_impl',
        'description': 'Finish-like tool for ToolCallAgent tests',
        'tool_type': ToolType.LOCAL,
    },
)
async def _toolcall_finish(params: _ToolCallInput, context: ToolContext) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': params.value}])


def _make_finish_tool_call(arguments: str = DEFAULT_TOOL_ARGUMENTS):
    """Create a tool call object compatible with ChatCompletionMessageToolCall."""
    return SimpleNamespace(
        id='call-1',
        type='function',
        function=SimpleNamespace(name=FINISH_TOOL_NAME, arguments=arguments),
    )


def _build_raising_client(error: Exception):
    class _RaisingClient:
        async def create_completion(self, **kwargs):
            raise error

    return _RaisingClient()


@pytest.mark.asyncio
async def test_ask_llm_sets_context_length_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'nexus.extensions.agents.base.toolcall.require_context',
        lambda: SimpleNamespace(openai_client=_build_raising_client(ContextLengthExceededError('too long'))),
    )
    agent = ToolCallAgent(executor=ToolExecutor([]))

    message = await agent.ask_llm()

    assert message.content is None
    assert message.tool_calls is None
    assert agent.exit_status == TerminationReason.TRUNCATION


@pytest.mark.asyncio
async def test_ask_llm_sets_format_error_for_bad_tool_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'nexus.extensions.agents.base.toolcall.require_context',
        lambda: SimpleNamespace(openai_client=_build_raising_client(KeyError('missing field'))),
    )
    agent = ToolCallAgent(executor=ToolExecutor([]))

    await agent.ask_llm()

    assert agent.exit_status == TerminationReason.FORMAT_ERROR
    assert agent.tool_error_count == 1


@pytest.mark.asyncio
async def test_ask_llm_sets_unhandled_error_for_other_client_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        'nexus.extensions.agents.base.toolcall.require_context',
        lambda: SimpleNamespace(openai_client=_build_raising_client(RuntimeError('client boom'))),
    )
    agent = ToolCallAgent(executor=ToolExecutor([]))

    await agent.ask_llm()

    assert agent.exit_status == TerminationReason.UNHANDLED_ERROR
    assert agent.tool_error_count == 1


@pytest.mark.asyncio
async def test_ask_llm_sets_provider_error_for_non_retryable_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'nexus.extensions.agents.base.toolcall.require_context',
        lambda: SimpleNamespace(openai_client=_build_raising_client(NonRetryableProviderError('no_budget_error'))),
    )
    agent = ToolCallAgent(executor=ToolExecutor([]))

    await agent.ask_llm()

    assert agent.exit_status == TerminationReason.PROVIDER_ERROR
    assert agent.tool_error_count == 1


@pytest.mark.asyncio
async def test_ask_llm_sets_provider_error_for_marker_in_generic_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'nexus.extensions.agents.base.toolcall.require_context',
        lambda: SimpleNamespace(openai_client=_build_raising_client(RuntimeError('no_budget_error'))),
    )
    agent = ToolCallAgent(executor=ToolExecutor([]))

    await agent.ask_llm()

    assert agent.exit_status == TerminationReason.PROVIDER_ERROR
    assert agent.tool_error_count == 1


@pytest.mark.asyncio
async def test_think_stores_reasoning_and_signature_in_message_dict() -> None:
    """ExtendedMessage fields are preserved when stored as dict in messages."""
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
    from nexus.core.types import ExtendedMessage

    agent = ToolCallAgent(executor=ToolExecutor([]))
    message = ExtendedMessage(
        content=TOOL_DECISION,
        reasoning_content=HIDDEN_REASONING,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id='call-1',
                type='function',
                function=Function(name=FINISH_TOOL_NAME, arguments=DEFAULT_TOOL_ARGUMENTS),
            )
        ],
    )

    async def fake_ask_llm(**kwargs):
        return message

    agent.ask_llm = fake_ask_llm  # type: ignore[method-assign]

    should_act = await agent.think()

    assert should_act is True
    assert agent.messages[-1]['reasoning_content'] == HIDDEN_REASONING
    assert agent.messages[-1]['content'] == TOOL_DECISION


@pytest.mark.asyncio
async def test_think_requests_retry_when_tool_call_is_required() -> None:
    from nexus.core.types import ExtendedMessage

    agent = ToolCallAgent(executor=ToolExecutor([]), tool_choice=ToolChoice.REQUIRED)
    agent.ask_llm = AsyncMock(return_value=ExtendedMessage(content=NO_TOOL_TEXT))  # type: ignore[method-assign]

    should_act = await agent.think()

    assert should_act is True
    assert agent.messages[-1]['role'] == 'user'
    assert agent.messages[-1]['content'] == TOOL_CALL_REQUIRED


@pytest.mark.asyncio
async def test_think_finishes_when_auto_mode_returns_plain_answer() -> None:
    from nexus.core.types import ExtendedMessage

    agent = ToolCallAgent(executor=ToolExecutor([]), tool_choice=ToolChoice.AUTO)
    agent.ask_llm = AsyncMock(return_value=ExtendedMessage(content=PLAIN_ANSWER))  # type: ignore[method-assign]

    should_act = await agent.think()

    assert should_act is False
    assert agent.exit_status == TerminationReason.SUCCESS
    assert agent.messages[-1]['content'] == PLAIN_ANSWER


@pytest.mark.asyncio
async def test_execute_tool_accepts_schema_alias_and_marks_finish_success() -> None:
    executor = ToolExecutor([_toolcall_finish])
    executor.set_tool_override('tests/toolcall_finish', name=FINISH_TOOL_NAME)
    executor.call = AsyncMock(return_value=ToolResponse(output=[{'type': 'text', 'text': FINAL_ANSWER}]))  # type: ignore[method-assign]
    agent = ToolCallAgent(executor=executor, finish_tool_names=[FINISH_TOOL_NAME])

    await agent.execute_tool(_make_finish_tool_call(f'{{"value": "{FINAL_ANSWER}"}}'))

    assert agent.exit_status == TerminationReason.SUCCESS
    assert agent.messages[-1]['role'] == 'tool'
    assert agent.messages[-1]['name'] == FINISH_TOOL_NAME
    assert agent.messages[-1]['content'] == FINAL_ANSWER
    executor.call.assert_awaited_once_with(FINISH_TOOL_NAME, {'value': FINAL_ANSWER})


@pytest.mark.asyncio
async def test_execute_tool_records_argument_parse_error_for_training() -> None:
    executor = ToolExecutor([_toolcall_finish])
    executor.set_tool_override('tests/toolcall_finish', name=FINISH_TOOL_NAME)
    agent = ToolCallAgent(executor=executor, finish_tool_names=[FINISH_TOOL_NAME], is_training=True)

    await agent.execute_tool(_make_finish_tool_call('{"value": }'))

    assert agent.exit_status == TerminationReason.FORMAT_ERROR
    assert agent.tool_error_count == 1
    assert str(agent.messages[-1]['content']).startswith(ARGUMENT_PARSE_ERROR_PREFIX)


@pytest.mark.asyncio
async def test_execute_tool_records_runtime_error_but_keeps_inference_running() -> None:
    executor = ToolExecutor([_toolcall_finish])
    executor.set_tool_override('tests/toolcall_finish', name=FINISH_TOOL_NAME)
    executor.call = AsyncMock(side_effect=RuntimeError('boom'))  # type: ignore[method-assign]
    agent = ToolCallAgent(executor=executor, finish_tool_names=[FINISH_TOOL_NAME], is_training=False)

    await agent.execute_tool(_make_finish_tool_call())

    assert agent.exit_status is None
    assert agent.tool_error_count == 1
    memory_text = str(agent.messages[-1]['content'])
    assert memory_text.startswith(RUNTIME_TOOL_ERROR_PREFIX)
    assert 'Traceback:\n' in memory_text


@pytest.mark.asyncio
async def test_execute_tool_stops_inference_for_non_retryable_provider_error() -> None:
    executor = ToolExecutor([_toolcall_finish])
    executor.set_tool_override('tests/toolcall_finish', name=FINISH_TOOL_NAME)
    executor.call = AsyncMock(side_effect=NonRetryableProviderError('no_budget_error'))  # type: ignore[method-assign]
    agent = ToolCallAgent(executor=executor, finish_tool_names=[FINISH_TOOL_NAME], is_training=False)

    await agent.execute_tool(_make_finish_tool_call())

    assert agent.exit_status == TerminationReason.PROVIDER_ERROR
    assert agent.tool_error_count == 1
    assert str(agent.messages[-1]['content']).startswith(NON_RETRYABLE_TOOL_ERROR_PREFIX)


@pytest.mark.asyncio
async def test_execute_tool_rejects_unknown_tool_name() -> None:
    agent = ToolCallAgent(executor=ToolExecutor([]), is_training=True)

    await agent.execute_tool(_make_finish_tool_call())

    assert agent.exit_status == TerminationReason.FORMAT_ERROR
    assert agent.tool_error_count == 1
    assert str(agent.messages[-1]['content']).startswith(UNKNOWN_TOOL_ERROR_PREFIX)


### Test Evaluation ###
class _EvaluatorReturnsScore:
    async def evaluate(self, prediction, reference, question=None) -> float:
        return 1.7

    async def evaluate_with_details(self, prediction, reference, question=None):
        return 1.7, EvaluationDetails()


class _EvaluatorRaisesValueError:
    async def evaluate(self, prediction, reference, question=None) -> float:
        raise ValueError('bad score')

    async def evaluate_with_details(self, prediction, reference, question=None):
        raise ValueError('bad score')


class _EvaluatorCrashes:
    async def evaluate(self, prediction, reference, question=None) -> float:
        raise RuntimeError('boom')

    async def evaluate_with_details(self, prediction, reference, question=None):
        raise RuntimeError('boom')


class _EvaluatorRaisesTypeError:
    async def evaluate(self, prediction, reference, question=None) -> float:
        raise TypeError('wrong type')

    async def evaluate_with_details(self, prediction, reference, question=None):
        raise TypeError('wrong type')


class _EvaluatorOnlyEvaluate:
    def __init__(self) -> None:
        self.called = False

    async def evaluate(self, prediction, reference, question=None) -> float:
        self.called = True
        return -0.2

    async def evaluate_with_details(self, prediction, reference, question=None):
        raise AssertionError('evaluate_with_details should not be called when need_details=False')


def _assert_failed_details(score: float, details: EvaluationDetails | None) -> None:
    assert score == 0.0
    assert details is not None
    assert details.status == 'failed'
    assert details.msg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('prediction', 'reference'),
    [
        ('prediction', None),
        ('   ', 'reference'),
        ('prediction', '   '),
        (None, 'reference'),
    ],
)
async def test_evaluate_output_returns_failed_details_for_guardrail_cases(
    prediction,
    reference,
) -> None:
    score, details, is_valid_reward = await evaluate_output(
        prediction=prediction,
        reference=reference,
        evaluator=_EvaluatorReturnsScore(),
        need_details=True,
    )

    _assert_failed_details(score, details)
    # Guardrail/data issues yield a genuine 0.0 score, not an infra failure.
    assert is_valid_reward is True


@pytest.mark.asyncio
async def test_evaluate_output_clamps_successful_scores() -> None:
    score, details, is_valid_reward = await evaluate_output(
        prediction='prediction',
        reference='reference',
        evaluator=_EvaluatorReturnsScore(),
        need_details=True,
    )

    assert score == 1.0
    assert details == EvaluationDetails()
    assert is_valid_reward is True


@pytest.mark.asyncio
async def test_evaluate_output_clamps_scores_without_details_via_evaluate_only() -> None:
    evaluator = _EvaluatorOnlyEvaluate()

    score, details, is_valid_reward = await evaluate_output(
        prediction='prediction',
        reference='reference',
        evaluator=evaluator,
        need_details=False,
    )

    assert evaluator.called is True
    assert score == 0.0
    assert details is None
    # A non-sentinel negative score is clamped but still a valid reward.
    assert is_valid_reward is True


@pytest.mark.asyncio
@pytest.mark.parametrize('evaluator', [_EvaluatorRaisesValueError(), _EvaluatorRaisesTypeError()])
async def test_evaluate_output_returns_failed_details_for_invalid_score_errors(evaluator) -> None:
    score, details, is_valid_reward = await evaluate_output(
        prediction='prediction',
        reference='reference',
        evaluator=evaluator,
        need_details=True,
    )

    _assert_failed_details(score, details)
    # Evaluator exceptions are infra failures, not genuine 0.0 scores.
    assert is_valid_reward is False


@pytest.mark.asyncio
async def test_evaluate_output_returns_failed_details_for_unhandled_exception() -> None:
    score, details, is_valid_reward = await evaluate_output(
        prediction='prediction',
        reference='reference',
        evaluator=_EvaluatorCrashes(),
        need_details=True,
    )

    _assert_failed_details(score, details)
    assert is_valid_reward is False


@pytest.mark.asyncio
async def test_evaluate_output_keeps_legacy_none_details_when_need_details_disabled() -> None:
    score, details, is_valid_reward = await evaluate_output(
        prediction=None,
        reference='reference',
        evaluator=_EvaluatorReturnsScore(),
        need_details=False,
    )

    assert score == 0.0
    assert details is None
    assert is_valid_reward is True


### Test Tools ###
@pytest.mark.asyncio
async def test_write_todos_tool_prefers_in_progress_task() -> None:
    response = await write_todos_tool(
        WriteTodosInput(
            todos=[
                TodoItem(content=QUEUED_TASK, status='pending'),
                TodoItem(content=ACTIVE_TASK, status='in_progress'),
            ]
        ),
        ToolContext(tool_configs={}),
    )

    assert response.context_updates['active_todo'] == ACTIVE_TASK
    assert response.meta['success'] is True
    assert response.context_updates['todos'] == [
        {'content': QUEUED_TASK, 'status': 'pending'},
        {'content': ACTIVE_TASK, 'status': 'in_progress'},
    ]
    assert QUEUED_TASK in response.output[0]['text']
    assert ACTIVE_TASK in response.output[0]['text']


@pytest.mark.asyncio
async def test_write_todos_tool_handles_pending_only_and_empty_list() -> None:
    pending_response = await write_todos_tool(
        WriteTodosInput(todos=[TodoItem(content=NEXT_STEP_TASK, status='pending')]),
        ToolContext(tool_configs={}),
    )
    empty_response = await write_todos_tool(
        WriteTodosInput(todos=[]),
        ToolContext(tool_configs={}),
    )

    assert pending_response.context_updates['active_todo'].endswith(NEXT_STEP_TASK)
    assert empty_response.context_updates['active_todo'].startswith('No active task')


@pytest.mark.asyncio
async def test_finish_tool_marks_finished_and_returns_message() -> None:
    response = await finish_tool(FinishInput(message=DONE_MESSAGE), ToolContext(tool_configs={}))

    assert response.meta == {'finished': True}
    assert response.output == [{'type': 'text', 'text': DONE_MESSAGE}]


### Test Utils ###
def test_parse_func_arguments_handles_dict_json_and_invalid_payloads() -> None:
    assert base_utils.parse_func_arguments(None) == {}
    assert base_utils.parse_func_arguments({'a': 1}) == {'a': 1}
    assert base_utils.parse_func_arguments('{"a": 1}') == {'a': 1}

    with pytest.raises(ValueError):
        base_utils.parse_func_arguments('{"a": }')

    with pytest.raises(ValueError):
        base_utils.parse_func_arguments('["a"]')


def test_format_to_user_message_and_extract_text_from_content_cover_mixed_content() -> None:
    message = base_utils.format_to_user_message(
        [
            {'role': 'system', 'content': 'rules'},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'part1'}, 'part2']},
        ],
        separator='\n\n',
    )

    assert message == {'role': 'user', 'content': 'System: rules\n\nAssistant: part1 part2'}
    assert base_utils.extract_text_from_content([{'type': 'text', 'text': 'hello'}, 'world']) == 'hello world'


def test_extract_reasoning_and_signature_reads_both_signature_sources() -> None:
    message_with_thinking_block = SimpleNamespace(
        reasoning_content='reasoning',
        thinking_blocks=[SimpleNamespace(signature='sig-1')],
        provider_specific_fields=None,
    )
    provider_fields_message = SimpleNamespace(
        reasoning_content='reasoning',
        thinking_blocks=[],
        provider_specific_fields={'reasoningContentBlocks': [{'reasoningText': {'signature': 'sig-2'}}]},
    )

    assert base_utils.extract_reasoning_and_signature(message_with_thinking_block) == ('reasoning', 'sig-1')
    assert base_utils.extract_reasoning_and_signature(provider_fields_message) == ('reasoning', 'sig-2')


def test_prepare_messages_for_thinking_converts_only_anthropic_assistant_messages() -> None:
    messages = [
        {'role': 'assistant', 'content': 'answer', 'reasoning_content': 'hidden', 'signature': 'sig'},
        {'role': 'user', 'content': 'question'},
    ]

    converted = base_utils.prepare_messages_for_thinking(messages, 'api_aws_third_anthropic/model')

    assert messages[0]['content'] == 'answer'
    assert converted[0]['content'] == [
        {'type': 'thinking', 'thinking': 'hidden', 'signature': 'sig'},
        {'type': 'text', 'text': 'answer'},
    ]
    assert converted[1] == {'role': 'user', 'content': 'question'}
