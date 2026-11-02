"""Tests for the search_s2t_rl_yb_glue agent loop and its search tool pool.

Focus areas: tool and inference failures must not change the training signal the
reward gates are calibrated against (every tool failure is fed back as an error
observation and the loop continues; every inference failure propagates so the
runner keeps deciding ``termination_reason``), plus the ordering guarantees around
tool-call validation — the per-call subquery cap is applied first, then the single
search slot is spent, and only then does ToolExecutor.call validate, so an invalid
call still costs the quota and still pushes a sibling call over budget.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Pre-populate nexus.extensions in sys.modules with a plain module to prevent
# its __init__.py (which imports heavy optional deps like pandas) from running.
_EXT_KEY = 'nexus.extensions'
if _EXT_KEY not in sys.modules:
    _stub = types.ModuleType(_EXT_KEY)
    import nexus as _nexus_pkg

    _stub.__path__ = [p + '/extensions' for p in _nexus_pkg.__path__]  # type: ignore[union-attr]
    sys.modules[_EXT_KEY] = _stub

from nexus.core import ContextLengthExceededError  # noqa: E402
from nexus.core.context import ExecutionContext  # noqa: E402
from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig  # noqa: E402
from nexus.core.tool import ToolExecutor, ToolResponse  # noqa: E402
from nexus.core.types import ExtendedChatCompletion  # noqa: E402
from nexus.extensions.projects.search_s2t_rl_yb_glue.agent import (  # noqa: E402
    SearchS2TExitStatus,
    SearchS2TYBAgentInput,
    run_search_s2t_agent_yb_glue,
)
from nexus.extensions.projects.search_s2t_rl_yb_glue.tools import (  # noqa: E402
    _get_pooled_http_client,
    aclose_pooled_http_clients,
)
from nexus.trace import processor_context  # noqa: E402
from nexus.trace.processors.base import SpanProcessor  # noqa: E402


class _MockOpenAIClient(OpenAICompatibleClient):
    """Mock client with an exact-enough count_tokens for the budget fitter."""

    def __init__(self):
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self._responses: list[Any] = []
        self._error: BaseException | None = None

    async def _create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        raise NotImplementedError

    def set_responses(self, *responses: Any) -> None:
        self._responses = list(responses)

    def set_error(self, error: BaseException) -> None:
        self._error = error

    async def create_completion(self, messages, **kwargs):
        if self._error is not None:
            raise self._error
        return self._responses.pop(0)

    async def count_tokens(self, *, messages, tools=None, add_generation_prompt=True):
        return SimpleNamespace(input_tokens=sum(len(str(message)) for message in messages) // 4)


def _make_tool_call(call_id: str, name: str, arguments: Any):
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return tool_call


def _make_message(content: str | None = None, tool_calls: list | None = None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    def _dump(**kwargs):
        dumped: dict[str, Any] = {'role': 'assistant'}
        if content:
            dumped['content'] = content
        if tool_calls:
            dumped['tool_calls'] = [
                {'id': tc.id, 'function': {'name': tc.function.name, 'arguments': tc.function.arguments}}
                for tc in tool_calls
            ]
        return dumped

    message.model_dump = _dump
    return message


def _make_response(message):
    return MagicMock(choices=[MagicMock(message=message)])


def _agent_input(**overrides: Any) -> SearchS2TYBAgentInput:
    defaults: dict[str, Any] = {
        'user_prompt': '珠峰有多高？',
        'max_iterations': 2,
        'max_search_calls': 1,
        'max_subqueries_per_call': 3,
    }
    defaults.update(overrides)
    return SearchS2TYBAgentInput(**defaults)


def _search_response(*, result_count: int = 2, subquery_count: int = 1, **meta: Any) -> ToolResponse:
    return ToolResponse(
        output=[{'type': 'text', 'text': '[web_doc 1 begin]...[web_doc 1 end]'}],
        meta={'success': True, 'result_count': result_count, 'subquery_count': subquery_count, **meta},
    )


def _last_tool_message(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    return trajectory[-1]


class TestToolCallErrors:
    """Bad tool calls become error observations; only quota accounting changes."""

    @pytest.mark.parametrize('arguments', ['123', '"a string"', '[1, 2]'])
    async def test_non_object_arguments_fail_the_run(self, arguments):
        # Valid JSON that is not an object used to raise AttributeError out of the
        # loop. It still fails the run (runner: UNHANDLED_ERROR, masked), only with
        # a message that says what happened.
        client = _MockOpenAIClient()
        client.set_responses(_make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', arguments)])))

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                with pytest.raises(TypeError, match='non-object arguments'):
                    await run_search_s2t_agent_yb_glue(_agent_input())

        assert tool_call.await_count == 0

    async def test_unparseable_arguments_are_malformed_and_the_loop_continues(self):
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', 'not json')])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert tool_call.await_count == 0
        assert output.malformed_tool_calls == 1
        assert output.search_count == 0
        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert _last_tool_message(output.main_trajectory[:-1])['tool_call_id'] == 'c1'

    @pytest.mark.parametrize('arguments', [{'query': 'not a list'}, {'query': []}, {}])
    async def test_invalid_arguments_still_consume_the_search_quota(self, arguments):
        # ToolExecutor.call validates the arguments before reaching the backend, so
        # the slot is spent on a call that never searched. Sparing it would let a
        # sibling call in the same message run and lift the trajectory out of the
        # -1.0 format gate, so the wasted quota is load-bearing.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', arguments)])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert output.malformed_tool_calls == 0
        assert output.search_count == 1
        assert output.valid_search_count == 0
        assert output.rejected_tool_calls == 0
        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert 'Error' in _last_tool_message(output.main_trajectory[:-1])['content']

    async def test_invalid_sibling_call_still_exhausts_the_search_budget(self):
        # Two web_search calls in one message: the first carries a non-list query and
        # fails validation inside ToolExecutor.call, having already spent the single
        # search slot. The second is well-formed but must still be rejected over
        # budget, because rejected_tool_calls > 0 is what rule_based_judge turns into
        # the -1.0 format penalty.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(
                _make_message(
                    tool_calls=[
                        _make_tool_call('c1', 'web_search', {'query': 'not a list'}),
                        _make_tool_call('c2', 'web_search', {'query': ['珠峰']}),
                    ]
                )
            ),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            output = await run_search_s2t_agent_yb_glue(_agent_input())

        tool_messages = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert [m['tool_call_id'] for m in tool_messages] == ['c1', 'c2']
        assert output.search_count == 1
        assert output.valid_search_count == 0
        assert output.rejected_tool_calls == 1
        assert output.overlimit_search_calls == 1

    async def test_blank_queries_still_reach_the_search_tool(self):
        # The input model only constrains list length, so an all-blank query list
        # passes validation and spends the search slot, exactly as before.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['   ', '\n']})])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                tool_call.return_value = _search_response(result_count=0, subquery_count=0, empty=True)
                output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert tool_call.await_count == 1
        assert output.search_count == 1
        assert output.valid_search_count == 0
        assert output.malformed_tool_calls == 0

    async def test_unknown_tool_name_is_an_error_observation(self):
        # A name outside SEARCH_TOOL_NAMES is not search accounting at all: the
        # executor cannot resolve it, the name error becomes an observation and the
        # search slot stays available.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'picture_search', {'query': ['珠峰']})])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert output.malformed_tool_calls == 0
        assert output.search_count == 0
        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert 'Error' in _last_tool_message(output.main_trajectory[:-1])['content']

    async def test_legacy_search_alias_still_spends_the_search_slot(self):
        # `search` is absent from the model-visible schema and ToolExecutor cannot
        # resolve it, so it can only ever produce an error observation. It still has
        # to count as a search call: see the sibling case below for why dropping it
        # from SEARCH_TOOL_NAMES would move the reward.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'search', {'query': ['珠峰']})])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert output.search_count == 1
        assert output.valid_search_count == 0
        assert output.malformed_tool_calls == 0
        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert 'Error' in _last_tool_message(output.main_trajectory[:-1])['content']

    async def test_legacy_alias_sibling_call_still_exhausts_the_search_budget(self):
        # Hallucinated `search` followed by a well-formed `web_search` in one message.
        # The alias spends the only slot, so the second call must be rejected over
        # budget and keep the trajectory on the -1.0 format gate.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(
                _make_message(
                    tool_calls=[
                        _make_tool_call('c1', 'search', {'query': ['珠峰']}),
                        _make_tool_call('c2', 'web_search', {'query': ['珠峰']}),
                    ]
                )
            ),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            output = await run_search_s2t_agent_yb_glue(_agent_input())

        tool_messages = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert [m['tool_call_id'] for m in tool_messages] == ['c1', 'c2']
        assert output.search_count == 1
        assert output.valid_search_count == 0
        assert output.rejected_tool_calls == 1
        assert output.overlimit_search_calls == 1

    async def test_non_object_arguments_under_an_unknown_tool_name_keep_going(self):
        # The non-object guard only fires for search names. A hallucinated name has
        # always reached ToolExecutor.call and come back as a name-resolution
        # observation, so it must not turn into an UNHANDLED_ERROR maskout.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'picture_search', 123)])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert output.search_count == 0
        assert output.malformed_tool_calls == 0
        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert 'Error' in _last_tool_message(output.main_trajectory[:-1])['content']

    async def test_sibling_tool_calls_each_get_a_paired_observation(self):
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(
                _make_message(
                    tool_calls=[
                        _make_tool_call('c1', 'web_search', 'not json'),
                        _make_tool_call('c2', 'web_search', {'query': ['珠峰']}),
                    ]
                )
            ),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                tool_call.return_value = _search_response()
                output = await run_search_s2t_agent_yb_glue(_agent_input())

        tool_messages = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert [m['tool_call_id'] for m in tool_messages] == ['c1', 'c2']
        assert output.malformed_tool_calls == 1
        assert output.search_count == 1


class TestSubqueryCap:
    async def test_overlong_query_list_is_truncated_not_rejected(self):
        # The tool input model caps the list at 3, so validating before slicing
        # would turn this into a format error instead of a truncated search.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(
                _make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['a', 'b', 'c', 'd']})])
            ),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                tool_call.return_value = _search_response(subquery_count=3)
                output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert tool_call.await_args is not None
        assert tool_call.await_args.args[1]['query'] == ['a', 'b', 'c']
        assert output.subquery_count == 4
        assert output.executed_subquery_count == 3
        assert output.overlimit_subquery_count == 1
        assert output.malformed_tool_calls == 0
        assert output.exit_status == SearchS2TExitStatus.FINISHED.value


class TestHappyPath:
    async def test_search_then_answer(self):
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['珠峰 高度']})])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                tool_call.return_value = _search_response(result_count=3)
                output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert output.search_count == 1
        assert output.valid_search_count == 1
        assert output.web_len == 3
        assert output.iterations == 2
        assert output.final_answer == '珠峰高 8848.86 米。'


class TestEnvironmentErrors:
    """Search failures keep feeding the model, so the trajectory still trains."""

    async def test_search_give_up_after_retries_keeps_going(self):
        # The tool never raises: an exhausted retry budget comes back as a normal
        # ToolResponse carrying give_up plus the fixed empty-result text.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['珠峰']})])),
            _make_response(_make_message(content='抱歉，我没有查到相关资料。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                tool_call.return_value = ToolResponse(
                    output=[{'type': 'text', 'text': '未查询到对应搜索结果'}],
                    meta={
                        'success': False,
                        'give_up': True,
                        'empty': True,
                        'result_count': 0,
                        'subquery_count': 1,
                        'error': 'HTTP 500',
                    },
                )
                output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert output.search_count == 1
        # A non-empty query did reach the tool, so the attempt stays valid and the
        # required-search gate is unlocked even though the backend gave up.
        assert output.valid_search_count == 1
        assert output.final_answer == '抱歉，我没有查到相关资料。'

    async def test_tool_exception_becomes_an_error_observation(self):
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['珠峰']})])),
            _make_response(_make_message(content='抱歉，我没有查到相关资料。')),
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                tool_call.side_effect = RuntimeError('backend exploded')
                output = await run_search_s2t_agent_yb_glue(_agent_input())

        assert output.exit_status == SearchS2TExitStatus.FINISHED.value
        assert output.tool_call_count == 0
        assert 'backend exploded' in _last_tool_message(output.main_trajectory[:-1])['content']


class TestInferenceErrors:
    """Every inference failure propagates so the runner classifies the run."""

    @pytest.mark.parametrize(
        'error',
        [
            ContextLengthExceededError('too long'),
            asyncio.TimeoutError(),
            RuntimeError('engine died'),
        ],
    )
    async def test_inference_failures_propagate(self, error):
        # The runner maps a propagating TimeoutError to TASK_TIMEOUT and anything
        # else to UNHANDLED_ERROR, both of which HyTuner masks out. Handling them
        # here would keep the run successful and spend a reward judge call on it.
        client = _MockOpenAIClient()
        client.set_error(error)

        async with ExecutionContext(openai_clients={'main': client}):
            with pytest.raises(type(error)):
                await run_search_s2t_agent_yb_glue(_agent_input())

    async def test_cancellation_is_not_swallowed(self):
        # Rollout abort and the runner-level total timeout both arrive this way.
        client = _MockOpenAIClient()
        client.set_error(asyncio.CancelledError())

        async with ExecutionContext(openai_clients={'main': client}):
            with pytest.raises(asyncio.CancelledError):
                await run_search_s2t_agent_yb_glue(_agent_input())


class TestBudgetViolation:
    async def test_tool_call_after_budget_exhausted_is_ignored(self):
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['珠峰']})]))
        )

        async with ExecutionContext(openai_clients={'main': client}):
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                output = await run_search_s2t_agent_yb_glue(_agent_input(max_iterations=1))

        assert tool_call.await_count == 0
        assert output.exit_status == SearchS2TExitStatus.TOOL_BUDGET_VIOLATION.value
        assert output.rejected_tool_calls == 1
        assert output.overlimit_search_calls == 1


class _SpanRecorder(SpanProcessor):
    """Collects (name, type, attributes) for every span that ends."""

    def __init__(self):
        self.spans: list[tuple[str, Any, dict[str, Any]]] = []

    def on_end(self, span, event) -> None:
        self.spans.append((span.name, span.attributes.get('type'), dict(span.attributes)))

    def agent_attributes(self) -> dict[str, Any]:
        return next(attributes for _, span_type, attributes in self.spans if span_type == 'agent')

    def span_types(self) -> list[Any]:
        return [span_type for _, span_type, _ in self.spans]


class TestSpanReporting:
    async def test_agent_span_carries_observation_time_but_no_termination_reason(self):
        # HytunerProcessor sums agent.observation_time off span attributes. It also
        # reads agent.termination_reason, which stays unwritten on purpose: setting
        # it would take masking decisions away from the runner.
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['珠峰']})])),
            _make_response(_make_message(content='珠峰高 8848.86 米。')),
        )
        recorder = _SpanRecorder()

        async with ExecutionContext(openai_clients={'main': client}):
            with processor_context([recorder]):
                with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                    tool_call.return_value = _search_response()
                    await run_search_s2t_agent_yb_glue(_agent_input())

        attributes = recorder.agent_attributes()
        assert 'agent.termination_reason' not in attributes
        assert attributes['agent.exit_status'] == SearchS2TExitStatus.FINISHED.value
        assert attributes['agent.iterations'] == 2
        assert attributes['agent.search_count'] == 1
        assert attributes['agent.observation_time'] > 0

        # agent_inference / agent_observation are the span types HytunerProcessor
        # turns into timeline segments.
        span_types = recorder.span_types()
        assert span_types.count('agent_inference') == 2
        assert span_types.count('agent_observation') == 1

    async def test_observation_span_is_named_after_the_tool(self):
        client = _MockOpenAIClient()
        client.set_responses(
            _make_response(_make_message(tool_calls=[_make_tool_call('c1', 'web_search', {'query': ['珠峰']})])),
            _make_response(_make_message(content='done')),
        )
        recorder = _SpanRecorder()

        async with ExecutionContext(openai_clients={'main': client}):
            with processor_context([recorder]):
                with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as tool_call:
                    tool_call.return_value = _search_response()
                    await run_search_s2t_agent_yb_glue(_agent_input())

        observation = next(attrs for _, span_type, attrs in recorder.spans if span_type == 'agent_observation')
        assert observation['tool_name'] == 'web_search'


class TestPooledHttpClient:
    async def test_same_config_reuses_one_client_per_loop(self):
        try:
            first = _get_pooled_http_client(
                timeout=30.0,
                connect_timeout=5.0,
                max_connections=128,
                max_keepalive_connections=64,
                keepalive_expiry=60.0,
            )
            second = _get_pooled_http_client(
                timeout=30.0,
                connect_timeout=5.0,
                max_connections=128,
                max_keepalive_connections=64,
                keepalive_expiry=60.0,
            )
            other_timeout = _get_pooled_http_client(
                timeout=10.0,
                connect_timeout=5.0,
                max_connections=128,
                max_keepalive_connections=64,
                keepalive_expiry=60.0,
            )

            assert first is second
            assert other_timeout is not first
        finally:
            await aclose_pooled_http_clients()

        assert first.is_closed
        assert other_timeout.is_closed

    async def test_closed_client_is_recreated(self):
        client = _get_pooled_http_client(
            timeout=30.0,
            connect_timeout=5.0,
            max_connections=8,
            max_keepalive_connections=4,
            keepalive_expiry=1.0,
        )
        await client.aclose()

        try:
            replacement = _get_pooled_http_client(
                timeout=30.0,
                connect_timeout=5.0,
                max_connections=8,
                max_keepalive_connections=4,
                keepalive_expiry=1.0,
            )
            assert replacement is not client
            assert not replacement.is_closed
        finally:
            await aclose_pooled_http_clients()
