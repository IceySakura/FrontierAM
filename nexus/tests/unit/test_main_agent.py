"""Tests for main_agent: schema, utils, parallel_call, processor, integration."""

from __future__ import annotations

import asyncio
import json
import sys
import types

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic import BaseModel

from nexus.core.context import ExecutionContext
from nexus.core.tool import (
    ToolArgumentValidationError,
    ToolContext,
    ToolExecutor,
    ToolExposure,
    ToolMeta,
    ToolNameResolutionError,
    ToolResponse,
    ToolSearchConfig,
    ToolType,
)


# Pre-populate nexus.extensions in sys.modules with a plain module to prevent
# its __init__.py (which imports heavy optional deps like pandas) from running.
_EXT_KEY = 'nexus.extensions'
if _EXT_KEY not in sys.modules:
    _stub = types.ModuleType(_EXT_KEY)
    import nexus as _nexus_pkg

    _stub.__path__ = [p + '/extensions' for p in _nexus_pkg.__path__]  # type: ignore[union-attr]
    sys.modules[_EXT_KEY] = _stub

from nexus.extensions.agents.main.agent import (  # noqa: E402
    MainAgentInput,
    MainAgentOutput,
    MainAgentToolOverride,
    MainExitStatus,
    _LoopState,
    _build_tool_response_message,
    _install_compaction_manager,
    _prepare_messages_for_inference,
    _resolve_tool_error_continue,
    _sanitize_unparseable_tool_args,
)
from nexus.extensions.agents.main.processor import MainAgentMetricsProcessor  # noqa: E402
from nexus.extensions.agents.main.utils import (  # noqa: E402
    TOOL_RESPONSE_TRUNCATION_MARKER,
    extract_tool_output_content,
    extract_tool_output_text,
    maybe_truncate,
    minimum_tool_response_token_budget,
    truncate_tool_response_by_tokens,
)
from nexus.runtime.client.base import RuntimeTransportError  # noqa: E402
from nexus.trace import processor_context  # noqa: E402


# ---------------------------------------------------------------------------
# MainAgentInput validation
# ---------------------------------------------------------------------------


class TestMainAgentInput:
    def test_minimal(self):
        inp = MainAgentInput(user_prompt='hello', tools=['a', 'b'])
        assert inp.tools == ['a', 'b']
        assert inp.max_iterations == 30
        assert inp.parallel_tool_calls is False

    def test_full(self):
        inp = MainAgentInput(
            user_prompt='hello',
            tools=['x'],
            max_iterations=10,
            parallel_tool_calls=True,
            invalid_tool_args_retries=2,
            tool_call_error_retries=1,
            tool_output_truncation=500,
            tool_overrides=[
                {
                    'tool_name': 'x',
                    'name': 'ToolX',
                    'description': 'override',
                    'params': {'value': 'value desc'},
                }
            ],
        )
        assert inp.max_iterations == 10
        assert inp.invalid_tool_args_retries == 2
        assert inp.tool_call_error_retries == 1
        assert inp.tool_overrides is not None
        assert isinstance(inp.tool_overrides[0], MainAgentToolOverride)
        assert inp.tool_overrides[0].tool_name == 'x'

    def test_prompt_content_parts(self):
        parts = [{'type': 'text', 'text': 'hello from parts'}]
        inp = MainAgentInput(
            user_prompt=parts,
            system_prompt=parts,
            tools=['x'],
        )
        assert isinstance(inp.user_prompt, list)
        assert isinstance(inp.system_prompt, list)

    def test_image_history_limit_defaults_to_none(self):
        inp = MainAgentInput(user_prompt='hello', tools=['a'])
        assert inp.image_history_limit is None

    def test_multi_agent_configuration_is_optional(self):
        from nexus.extensions.agents.main.multi_agent import MultiAgentConfig, SubagentConfig

        config = MultiAgentConfig(
            max_agents=2,
            max_active_agents=1,
            subagent=SubagentConfig(system_prompt='worker', tools=['worker/tool']),
        )
        inp = MainAgentInput(user_prompt='delegate', tools=[], multi_agent=config)

        assert inp.multi_agent is config

    @pytest.mark.parametrize('max_tool_response_tokens', [0, -1])
    def test_max_tool_response_tokens_must_be_positive(self, max_tool_response_tokens):
        with pytest.raises(ValueError, match='greater than 0'):
            MainAgentInput(
                user_prompt='hello',
                tools=['a'],
                max_tool_response_tokens=max_tool_response_tokens,
            )

    def test_tool_error_continue_defaults_to_false(self):
        inp = MainAgentInput(user_prompt='hello', tools=['a'])
        assert inp.tool_error_continue is False
        assert inp.wrap_failed_tool_response_as_tool_error is False


class TestBuildToolResponseMessage:
    def test_default_preserves_tool_reported_failure_payload(self):
        state = _LoopState()
        agent_input = MainAgentInput(user_prompt='hello', tools=['a'])
        resp = ToolResponse(output=[{'type': 'text', 'text': 'backend failed'}], meta={'success': False})

        message = _build_tool_response_message('call-1', resp, 'backend failed', agent_input, state)

        assert message == {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': 'backend failed',
        }
        assert state.tool_error_count == 0

    def test_enabled_wraps_string_tool_reported_failure(self):
        state = _LoopState()
        agent_input = MainAgentInput(
            user_prompt='hello',
            tools=['a'],
            wrap_failed_tool_response_as_tool_error=True,
        )
        resp = ToolResponse(output=[{'type': 'text', 'text': 'backend failed'}], meta={'success': False})

        message = _build_tool_response_message('call-1', resp, 'backend failed', agent_input, state)

        assert message['content'] == 'Tool error: backend failed'
        assert state.tool_error_count == 1

    def test_enabled_prefixes_multimodal_tool_reported_failure(self):
        state = _LoopState()
        agent_input = MainAgentInput(
            user_prompt='hello',
            tools=['a'],
            wrap_failed_tool_response_as_tool_error=True,
        )
        payload = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]
        resp = ToolResponse(output=payload, meta={'success': False})

        message = _build_tool_response_message('call-1', resp, payload, agent_input, state)

        assert message['content'] == [
            {'type': 'text', 'text': 'Tool error: tool reported failure'},
            *payload,
        ]
        assert state.tool_error_count == 1


# ---------------------------------------------------------------------------
# tool_error_continue resolution (config field + env var override)
# ---------------------------------------------------------------------------


class TestResolveToolErrorContinue:
    _ENV = 'NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE'

    def _input(self, **kw):
        return MainAgentInput(user_prompt='hello', tools=['a'], **kw)

    def test_defaults_to_false(self, monkeypatch):
        monkeypatch.delenv(self._ENV, raising=False)
        assert _resolve_tool_error_continue(self._input()) is False

    def test_config_field_used_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(self._ENV, raising=False)
        assert _resolve_tool_error_continue(self._input(tool_error_continue=True)) is True

    @pytest.mark.parametrize('val', ['1', 'true', 'TRUE', 'yes', 'y', 'on'])
    def test_env_truthy_overrides(self, monkeypatch, val):
        monkeypatch.setenv(self._ENV, val)
        assert _resolve_tool_error_continue(self._input()) is True

    @pytest.mark.parametrize('val', ['0', 'false', 'no', 'n', 'off'])
    def test_env_falsy_overrides_config(self, monkeypatch, val):
        # env var falsy beats config field True
        monkeypatch.setenv(self._ENV, val)
        assert _resolve_tool_error_continue(self._input(tool_error_continue=True)) is False

    def test_env_unrecognized_falls_back_to_config(self, monkeypatch):
        monkeypatch.setenv(self._ENV, 'maybe')
        assert _resolve_tool_error_continue(self._input(tool_error_continue=True)) is True
        assert _resolve_tool_error_continue(self._input(tool_error_continue=False)) is False


class TestSanitizeUnparseableToolArgs:
    def test_wraps_invalid_json_arguments(self):
        msg = {
            'role': 'assistant',
            'tool_calls': [
                {'id': '', 'function': {'name': 'f', 'arguments': '{"a": 1'}, 'type': 'function'},
            ],
        }
        _sanitize_unparseable_tool_args(msg)
        wrapped = json.loads(msg['tool_calls'][0]['function']['arguments'])
        assert wrapped == {'_unparseable_arguments': '{"a": 1'}

    def test_leaves_valid_json_arguments_untouched(self):
        msg = {
            'role': 'assistant',
            'tool_calls': [
                {'id': '', 'function': {'name': 'f', 'arguments': '{"a": 1}'}, 'type': 'function'},
            ],
        }
        _sanitize_unparseable_tool_args(msg)
        assert msg['tool_calls'][0]['function']['arguments'] == '{"a": 1}'

    def test_no_tool_calls_is_noop(self):
        msg = {'role': 'assistant', 'content': 'hi'}
        _sanitize_unparseable_tool_args(msg)  # should not raise
        assert msg == {'role': 'assistant', 'content': 'hi'}

    def test_tool_search_config(self):
        inp = MainAgentInput(
            user_prompt='hello',
            tools=['x'],
            tool_search={
                'enabled': True,
                'exposure_overrides': {'x': 'deferred'},
            },
        )

        assert inp.tool_search is not None
        assert isinstance(inp.tool_search, ToolSearchConfig)
        assert inp.tool_search.enabled is True
        assert inp.tool_search.exposure_overrides['x'] == ToolExposure.DEFERRED


# ---------------------------------------------------------------------------
# MainExitStatus
# ---------------------------------------------------------------------------


class TestExitStatus:
    def test_all_statuses_have_termination_reason(self):
        for status in MainExitStatus:
            assert status.termination_reason != 'UNKNOWN'

    def test_unknown_exit_status_maps_to_unknown_reason(self):
        assert MainExitStatus.termination_reason_for('unknown') == 'UNKNOWN'


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


class TestMaybeTruncate:
    def test_none_limit(self):
        assert maybe_truncate('abcdef', None) == 'abcdef'

    def test_short_content(self):
        assert maybe_truncate('abc', 10) == 'abc'

    def test_exact_limit(self):
        assert maybe_truncate('abcd', 4) == 'abcd'

    def test_truncation(self):
        content = 'A' * 100
        result = maybe_truncate(content, 20)
        assert '[truncated]' in result
        assert result.startswith('A' * 10)
        assert result.endswith('A' * 10)
        assert len(result) < 100

    def test_tool_response_token_budget_boundary(self):
        minimum_budget = minimum_tool_response_token_budget(list)

        with pytest.raises(ValueError, match=rf'must be at least {minimum_budget}'):
            truncate_tool_response_by_tokens('tool result', minimum_budget - 1, list, ''.join)

        content = 'H' + ('x' * minimum_budget) + 'T'
        truncated, *_ = truncate_tool_response_by_tokens(content, minimum_budget, list, ''.join)

        assert truncated == f'H{TOOL_RESPONSE_TRUNCATION_MARKER}T'


class TestPrepareMessagesForInference:
    def test_none_limit_returns_original_messages(self):
        messages = [{'role': 'user', 'content': 'hello'}]
        assert _prepare_messages_for_inference(messages, None) is messages

    def test_omits_older_images_without_mutating_trajectory(self):
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'old'},
                    {'type': 'image_url', 'image_url': {'url': 'data:old'}},
                ],
            },
            {'role': 'assistant', 'content': 'middle'},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'new'},
                    {'type': 'image_url', 'image_url': {'url': 'data:new'}},
                ],
            },
        ]

        prepared = _prepare_messages_for_inference(messages, 1)

        assert prepared is not messages
        assert prepared[0]['content'][1] == {
            'type': 'text',
            'text': '[previous image omitted from inference history]',
        }
        assert prepared[2]['content'][1]['image_url']['url'] == 'data:new'
        assert messages[0]['content'][1]['image_url']['url'] == 'data:old'

    def test_zero_limit_omits_all_images(self):
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:old'}},
                ],
            }
        ]

        prepared = _prepare_messages_for_inference(messages, 0)

        assert prepared[0]['content'] == [{'type': 'text', 'text': '[previous image omitted from inference history]'}]


class TestExtractToolOutputText:
    def test_string(self):
        assert extract_tool_output_text('hello') == 'hello'

    def test_text_parts(self):
        output = [{'type': 'text', 'text': 'line1'}, {'type': 'text', 'text': 'line2'}]
        assert extract_tool_output_text(output) == 'line1\nline2'

    def test_image_placeholder(self):
        output = [{'type': 'image_url', 'image_url': {'url': 'data:...'}}]
        assert extract_tool_output_text(output) == '[image]'

    def test_non_list(self):
        assert extract_tool_output_text(42) == '42'

    def test_mixed(self):
        output = [{'type': 'text', 'text': 'hi'}, {'type': 'image_url'}]
        result = extract_tool_output_text(output)
        assert 'hi' in result
        assert '[image]' in result


class TestExtractToolOutputContent:
    def test_string(self):
        assert extract_tool_output_content('hello') == 'hello'

    def test_text_parts_flatten_to_string(self):
        output = [{'type': 'text', 'text': 'line1'}, {'type': 'text', 'text': 'line2'}]
        assert extract_tool_output_content(output) == 'line1\nline2'

    def test_multimodal_parts_are_preserved(self):
        output = [
            {'type': 'text', 'text': 'zoomed'},
            {'type': 'image_url', 'image_url': {'url': 'data:...'}},
        ]
        assert extract_tool_output_content(output) == output


# ---------------------------------------------------------------------------
# ToolExecutor.parallel_call
# ---------------------------------------------------------------------------


class _DummyInput(BaseModel):
    value: str = 'default'


class _RequiredInput(BaseModel):
    value: str


def _make_tool_func(
    name: str,
    parallel_func=None,
    registry_name: str | None = None,
    input_model: type[BaseModel] = _DummyInput,
) -> Any:
    """Create a mock tool func with proper __tool_meta__."""

    async def _tool(params: Any, context: ToolContext) -> ToolResponse:
        return ToolResponse(output=[{'type': 'text', 'text': f'{name}:{params.value}'}])

    _tool.__tool_input_model__ = input_model
    _tool.__tool_meta__ = ToolMeta(
        name=name,
        description=f'Tool {name}',
        tool_type=ToolType.LOCAL,
        input_model=input_model,
        func=_tool,
        parallel_func=parallel_func,
    )
    _tool.nexus_meta = {'type': 'tool', 'name': registry_name or name}
    return _tool


class TestParallelCall:
    async def test_empty_calls(self):
        executor = ToolExecutor([])
        result = await executor.parallel_call([])
        assert result == []

    async def test_no_parallel_func(self):
        """Tools without parallel_func fallback to individual call()."""
        tool_a = _make_tool_func('alpha')
        tool_b = _make_tool_func('beta')
        executor = ToolExecutor([tool_a, tool_b])

        ctx = ExecutionContext(openai_clients={})
        async with ctx:
            results = await executor.parallel_call(
                [
                    ('alpha', {'value': '1'}),
                    ('beta', {'value': '2'}),
                ]
            )

        assert len(results) == 2
        assert 'alpha:1' in extract_tool_output_text(results[0].output)
        assert 'beta:2' in extract_tool_output_text(results[1].output)

    async def test_with_parallel_func(self):
        """Tools sharing a parallel_func are batched together."""
        call_log: list[list[tuple[str, Any, Any]]] = []

        async def batch_func(tool_calls: list[tuple[str, BaseModel, ToolContext]]) -> list[ToolResponse]:
            call_log.append([(n, p, c) for n, p, c in tool_calls])
            return [
                ToolResponse(output=[{'type': 'text', 'text': f'batch:{name}:{params.value}'}])
                for name, params, _ctx in tool_calls
            ]

        tool_a = _make_tool_func('aa', parallel_func=batch_func)
        tool_b = _make_tool_func('bb', parallel_func=batch_func)
        tool_c = _make_tool_func('cc')  # no parallel_func
        executor = ToolExecutor([tool_a, tool_b, tool_c])

        ctx = ExecutionContext(openai_clients={})
        async with ctx:
            results = await executor.parallel_call(
                [
                    ('aa', {'value': 'x'}),
                    ('cc', {'value': 'y'}),
                    ('bb', {'value': 'z'}),
                ]
            )

        assert len(results) == 3
        # aa and bb should have been batched
        assert len(call_log) == 1
        assert len(call_log[0]) == 2

        # Order preserved
        assert 'batch:aa:x' in extract_tool_output_text(results[0].output)
        assert 'cc:y' in extract_tool_output_text(results[1].output)
        assert 'batch:bb:z' in extract_tool_output_text(results[2].output)

    async def test_unknown_tool_raises(self):
        executor = ToolExecutor([])
        with pytest.raises(KeyError, match='not found'):
            await executor.parallel_call([('nonexistent', {})])

    async def test_unknown_tool_error_lists_schema_names(self):
        tool_a = _make_tool_func('alpha', registry_name='pkg/alpha')
        executor = ToolExecutor([tool_a])
        executor.set_tool_override('pkg/alpha', name='AlphaTool')

        with pytest.raises(KeyError) as exc_info:
            await executor.call('missing_tool', {})

        message = str(exc_info.value)
        assert "Tool 'missing_tool' not found" in message
        assert 'AlphaTool' in message
        assert 'pkg/alpha' not in message

    def test_argument_validation_error_uses_schema_name_and_parameter_summary(self):
        tool_a = _make_tool_func('RequiredTool', registry_name='pkg/required', input_model=_RequiredInput)
        executor = ToolExecutor([tool_a])

        with pytest.raises(ToolArgumentValidationError) as exc_info:
            executor.validate_tool_call('RequiredTool', {})

        message = exc_info.value.message_for_model
        assert "Invalid arguments for tool 'RequiredTool'" in message
        assert 'value: Field required' in message
        assert 'required=[value: string]' in message
        assert 'pkg/required' not in message
        assert 'errors.pydantic.dev' not in message

    async def test_schema_name_override_dispatch(self):
        tool_a = _make_tool_func('alpha', registry_name='pkg/alpha')
        executor = ToolExecutor([tool_a])
        executor.set_tool_override('pkg/alpha', name='AlphaTool')

        schemas = executor.get_tool_schemas()
        assert schemas[0]['function']['name'] == 'AlphaTool'

        ctx = ExecutionContext(openai_clients={})
        async with ctx:
            resp = await executor.call('AlphaTool', {'value': '1'})

        assert 'alpha:1' in extract_tool_output_text(resp.output)

        with pytest.raises(KeyError, match='not found'):
            executor.set_tool_override('alpha', name='AlphaTool')

    async def test_default_schema_name_uses_tool_meta_name(self):
        tool_a = _make_tool_func('alpha', registry_name='pkg/alpha')
        executor = ToolExecutor([tool_a])

        schemas = executor.get_tool_schemas()
        assert schemas[0]['function']['name'] == 'alpha'

        ctx = ExecutionContext(openai_clients={})
        async with ctx:
            # Call by schema-visible meta name
            resp = await executor.call('alpha', {'value': '1'})
            assert 'alpha:1' in extract_tool_output_text(resp.output)

            # Canonical registry name remains valid as internal fallback
            resp2 = await executor.call('pkg/alpha', {'value': '2'})
            assert 'alpha:2' in extract_tool_output_text(resp2.output)

    async def test_registry_name_calling_fallback(self):
        tool_a = _make_tool_func('alpha', registry_name='pkg/alpha')
        executor = ToolExecutor([tool_a])
        executor.set_tool_override('pkg/alpha', name='AlphaTool')

        ctx = ExecutionContext(openai_clients={})
        async with ctx:
            # schema-visible name works
            resp = await executor.call('AlphaTool', {'value': '1'})
            assert 'alpha:1' in extract_tool_output_text(resp.output)

            # canonical registry name is also accepted as fallback
            resp2 = await executor.call('pkg/alpha', {'value': '2'})
            assert 'alpha:2' in extract_tool_output_text(resp2.output)

    def test_duplicate_schema_name_override_raises(self):
        tool_a = _make_tool_func('alpha')
        tool_b = _make_tool_func('beta')
        executor = ToolExecutor([tool_a, tool_b])
        executor.set_tool_override('alpha', name='Shared')
        executor.set_tool_override('beta', name='Shared')

        with pytest.raises(ValueError, match='Duplicate schema tool name'):
            executor.get_tool_schemas()


# ---------------------------------------------------------------------------
# MainAgentMetricsProcessor
# ---------------------------------------------------------------------------

from nexus.trace.events import SpanEndEvent, SpanStartEvent  # noqa: E402
from nexus.trace.types import Span  # noqa: E402


def _make_span(name: str, start_ns: int, end_ns: int, attributes: dict | None = None) -> Span:
    return Span(
        trace_id='trace-1',
        span_id=f'span-{name}-{start_ns}',
        parent_span_id='parent-1',
        name=name,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        attributes=attributes or {},
    )


class TestMainAgentMetricsProcessor:
    @pytest.mark.parametrize('outer_count', [None, 9])
    def test_empty_outer_span_preserves_latest_inner_counters(self, outer_count):
        proc = MainAgentMetricsProcessor()
        event = SpanEndEvent(span_id='agent')
        for count in (2, 5):
            proc.on_end(_make_span('agent.main_agent', 0, 1, attributes={
                'type': 'agent', 'agent.iterations': count,
            }), event)
        attrs = {'type': 'agent'}
        if outer_count is not None:
            attrs['agent.iterations'] = outer_count
        proc.on_end(_make_span('agent.visual_agent', 0, 2, attributes=attrs), event)
        assert proc.get_result()['agent.iterations'] == (5 if outer_count is None else outer_count)

    def test_empty(self):
        proc = MainAgentMetricsProcessor()
        result = proc.get_result()
        assert result['agent.total_inference_time'] == 0.0
        assert result['agent.total_observation_time'] == 0.0
        assert result['agent.inference_times'] == []
        assert result['agent.observation_times'] == []

    def test_inference_and_observation_timing(self):
        proc = MainAgentMetricsProcessor()
        event = SpanEndEvent(span_id='s1')

        # Simulate 2 inference spans (1s each) and 2 observation spans (0.5s each)
        ns = 1_000_000_000
        proc.on_end(_make_span('agent.inference', 0, 1 * ns), event)
        proc.on_end(_make_span('agent.observation', 1 * ns, int(1.5 * ns)), event)
        proc.on_end(_make_span('agent.inference', 2 * ns, 3 * ns), event)
        proc.on_end(_make_span('agent.observation', 3 * ns, int(3.5 * ns)), event)

        result = proc.get_result()
        assert result['agent.inference_times'] == pytest.approx([1.0, 1.0])
        assert result['agent.observation_times'] == pytest.approx([0.5, 0.5])
        assert result['agent.total_inference_time'] == pytest.approx(2.0)
        assert result['agent.total_observation_time'] == pytest.approx(1.0)

    def test_agent_span_attributes(self):
        proc = MainAgentMetricsProcessor()
        event = SpanEndEvent(span_id='s-main')

        agent_span = _make_span(
            'agent.main_agent',
            0,
            5_000_000_000,
            attributes={
                'type': 'agent',
                'agent.exit_status': 'finished',
                'agent.termination_reason': 'ENV_DONE',
                'agent.iterations': 3,
                'agent.tool_call_count': 5,
                'agent.tool_error_count': 0,
                'unrelated_key': 'ignored',
            },
        )
        proc.on_end(agent_span, event)

        result = proc.get_result()
        assert result['agent.exit_status'] == 'finished'
        assert result['agent.iterations'] == 3
        assert result['agent.tool_call_count'] == 5
        assert 'unrelated_key' not in result

    def test_ignores_other_spans(self):
        proc = MainAgentMetricsProcessor()
        event = SpanEndEvent(span_id='s1')
        proc.on_end(_make_span('openai_completion', 0, 1_000_000_000), event)
        proc.on_end(_make_span('tool_call', 0, 500_000_000), event)

        result = proc.get_result()
        assert result['agent.inference_times'] == []
        assert result['agent.observation_times'] == []

    def test_iteration_callback_uses_outer_loop_index(self):
        class _HookProc(MainAgentMetricsProcessor):
            def __init__(self):
                super().__init__()
                self.iterations: list[int] = []

            def on_iteration_start(self, iteration: int) -> None:
                self.iterations.append(iteration)

        proc = _HookProc()
        start_event = SpanStartEvent(parent_span_id='parent', name='agent.inference')

        # Same outer iteration can produce multiple inference spans due to retries.
        proc.on_start(_make_span('agent.inference', 0, 1, attributes={'iteration': 1}), start_event)
        proc.on_start(_make_span('agent.inference', 1, 2, attributes={'iteration': 1}), start_event)
        proc.on_start(_make_span('agent.inference', 2, 3, attributes={'iteration': 2}), start_event)

        assert proc.iterations == [1, 2]


# ---------------------------------------------------------------------------
# Integration: mock client + mock tools
# ---------------------------------------------------------------------------

from nexus.core.openai_client import OpenAICompatibleClient, OpenAIConfig, TokenizerPanicError  # noqa: E402
from nexus.core.types import ExtendedChatCompletion  # noqa: E402


class _MockOpenAIClient(OpenAICompatibleClient):
    """Thin mock that satisfies the pydantic type constraint."""

    def __init__(self):
        super().__init__(OpenAIConfig(client_args={}, request_args={}))
        self._create_completion_impl: Any = None

    async def _create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        raise NotImplementedError

    def set_responses(self, *responses: Any) -> None:
        it = iter(responses)

        async def _impl(messages, **kwargs):
            return next(it)

        self._create_completion_impl = _impl

    def set_side_effect(self, exc: Exception) -> None:
        async def _impl(messages, **kwargs):
            raise exc

        self._create_completion_impl = _impl

    async def create_completion(self, messages, **kwargs):
        return await self._create_completion_impl(messages, **kwargs)


def _make_mock_message(
    content: str | None = None,
    tool_calls: list | None = None,
    refusal: str | None = None,
):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.refusal = refusal

    def _dump(**kwargs):
        d: dict[str, Any] = {'role': 'assistant'}
        if content:
            d['content'] = content
        if refusal:
            d['refusal'] = refusal
        if tool_calls:
            d['tool_calls'] = [
                {
                    'id': tc.id,
                    'function': {'name': tc.function.name, 'arguments': tc.function.arguments},
                }
                for tc in tool_calls
            ]
        return d

    msg.model_dump = _dump
    return msg


def _make_mock_tool_call(call_id: str, name: str, args: dict):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


async def test_run_main_agent_passes_tool_search_config_to_executor():
    from nexus.extensions.agents.main.agent import run_main_agent

    mock_message = MagicMock()
    mock_message.tool_calls = None
    mock_message.content = 'done'
    mock_message.model_dump.return_value = {'role': 'assistant', 'content': 'done'}
    mock_client = _MockOpenAIClient()
    mock_client.set_responses(MagicMock(choices=[MagicMock(message=mock_message)]))

    tool_func = _make_tool_func('alpha')

    ctx = ExecutionContext(openai_clients={'main': mock_client})
    async with ctx:
        with patch('nexus.extensions.agents.main.agent.tool_registry.get', return_value=tool_func):
            with patch('nexus.extensions.agents.main.agent.ToolExecutor') as executor_cls:
                executor = MagicMock()
                executor.get_tool_schemas.return_value = []
                executor_cls.return_value = executor

                await run_main_agent(
                    MainAgentInput(
                        user_prompt='hello',
                        tools=['alpha'],
                        tool_search={'enabled': True, 'exposure_overrides': {'alpha': 'deferred'}},
                    )
                )

    _, kwargs = executor_cls.call_args
    assert kwargs['allowed_tool_names'] == ['alpha']
    assert kwargs['tool_search_config'] is not None
    assert kwargs['tool_search_config'].enabled is True


async def test_run_main_agent_rejects_tool_search_with_schema_overrides():
    from nexus.extensions.agents.main.agent import run_main_agent

    mock_client = _MockOpenAIClient()

    ctx = ExecutionContext(openai_clients={'main': mock_client})
    async with ctx:
        with patch.object(ExecutionContext, 'load_mcp_tools', new_callable=AsyncMock) as load_mcp_tools:
            with patch('nexus.extensions.agents.main.agent.ToolExecutor') as executor_cls:
                with pytest.raises(ValueError, match='tool_schema_overrides cannot be used'):
                    await run_main_agent(
                        MainAgentInput(
                            user_prompt='hello',
                            tools=['alpha'],
                            mcp_servers=[{'name': 'will-not-load'}],
                            tool_search={'enabled': True, 'exposure_overrides': {'missing': 'deferred'}},
                            tool_schema_overrides=[
                                {
                                    'type': 'function',
                                    'function': {
                                        'name': 'alpha',
                                        'description': 'override',
                                        'parameters': {'type': 'object', 'properties': {}},
                                    },
                                }
                            ],
                        )
                    )

    load_mcp_tools.assert_not_awaited()
    executor_cls.assert_not_called()


async def test_main_agent_calls_deferred_tool_through_tool_call():
    from nexus.core import tool_registry
    from nexus.extensions.agents.main.agent import run_main_agent

    deferred_tool = _make_tool_func('deferred_tool')
    original_get = tool_registry.get

    def _get_tool(tool_name: str):
        if tool_name == 'deferred_tool':
            return deferred_tool
        return original_get(tool_name)

    search_call = MagicMock()
    search_call.id = 'call_search'
    search_call.function.name = 'tool_search'
    search_call.function.arguments = json.dumps({'query': 'deferred', 'max_results': 10})

    tool_call = MagicMock()
    tool_call.id = 'call_tool'
    tool_call.function.name = 'tool_call'
    tool_call.function.arguments = json.dumps(
        {
            'tool_name': 'deferred_tool',
            'params': {'value': 'ok'},
        }
    )

    msg_search = MagicMock()
    msg_search.tool_calls = [search_call]
    msg_search.content = None
    msg_search.model_dump.return_value = {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_search',
                'type': 'function',
                'function': {'name': 'tool_search', 'arguments': search_call.function.arguments},
            }
        ],
    }

    msg_call = MagicMock()
    msg_call.tool_calls = [tool_call]
    msg_call.content = None
    msg_call.model_dump.return_value = {
        'role': 'assistant',
        'tool_calls': [
            {
                'id': 'call_tool',
                'type': 'function',
                'function': {'name': 'tool_call', 'arguments': tool_call.function.arguments},
            }
        ],
    }

    msg_done = MagicMock()
    msg_done.tool_calls = None
    msg_done.content = 'done'
    msg_done.model_dump.return_value = {'role': 'assistant', 'content': 'done'}

    mock_client = _MockOpenAIClient()
    mock_client.set_responses(
        MagicMock(choices=[MagicMock(message=msg_search)]),
        MagicMock(choices=[MagicMock(message=msg_call)]),
        MagicMock(choices=[MagicMock(message=msg_done)]),
    )

    ctx = ExecutionContext(openai_clients={'main': mock_client})
    async with ctx:
        with patch('nexus.extensions.agents.main.agent.tool_registry.get', side_effect=_get_tool):
            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='hello',
                    tools=['deferred_tool'],
                    max_iterations=3,
                    tool_search={
                        'enabled': True,
                        'exposure_overrides': {'deferred_tool': 'deferred'},
                    },
                )
            )

    assert output.exit_status == MainExitStatus.FINISHED.value
    assert output.final_answer == 'done'
    assert any(
        message.get('role') == 'tool' and message.get('content') == 'deferred_tool:ok'
        for message in output.main_trajectory
    )


async def test_run_main_agent_injects_tool_search_reminder_into_first_user_message():
    from nexus.core import tool_registry
    from nexus.extensions.agents.main.agent import run_main_agent

    captured_messages: list[Any] = []

    async def _capture(messages, **kwargs):
        captured_messages.append(messages)
        response = MagicMock()
        response.choices = [MagicMock()]
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = 'done'
        msg.model_dump.return_value = {'role': 'assistant', 'content': 'done'}
        response.choices[0].message = msg
        return response

    mock_client = _MockOpenAIClient()
    mock_client._create_completion_impl = _capture

    deferred_tool = _make_tool_func('deferred_tool')
    original_get = tool_registry.get

    def _get_tool(tool_name: str):
        if tool_name == 'deferred_tool':
            return deferred_tool
        return original_get(tool_name)

    ctx = ExecutionContext(openai_clients={'main': mock_client})
    async with ctx:
        with patch('nexus.extensions.agents.main.agent.tool_registry.get', side_effect=_get_tool):
            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='hello',
                    tools=['deferred_tool'],
                    max_iterations=1,
                    tool_search={
                        'enabled': True,
                        'exposure_overrides': {'deferred_tool': 'deferred'},
                    },
                )
            )

    assert output.exit_status == MainExitStatus.FINISHED.value
    # 首轮推理拿到的 messages:第二条是 user,其 content 末尾应含 reminder
    first_messages = captured_messages[0]
    user_msg = next(m for m in first_messages if m.get('role') == 'user')
    assert isinstance(user_msg['content'], str)
    assert user_msg['content'].startswith('hello')
    assert '<system-reminder>' in user_msg['content']
    assert 'deferred_tool' in user_msg['content']


async def _run_compaction_request_meta_case(run_agent, agent_input):
    from nexus.extensions.compaction import CompactionResult

    invalid_call = MagicMock()
    invalid_call.id = 'invalid'
    invalid_call.function.name = 'echo'
    invalid_call.function.arguments = '{'
    valid_call = _make_mock_tool_call('valid', 'echo', {'value': 'hi'})
    responses = iter(
        [
            MagicMock(choices=[MagicMock(message=_make_mock_message(tool_calls=[invalid_call]))]),
            MagicMock(choices=[MagicMock(message=_make_mock_message(tool_calls=[valid_call]))]),
            MagicMock(choices=[MagicMock(message=_make_mock_message(content='Done!'))]),
        ]
    )
    request_meta: list[dict[str, Any]] = []
    mock_client = _MockOpenAIClient()

    async def _capture(messages, **kwargs):
        request_meta.append(kwargs['request_meta'])
        return next(responses)

    mock_client._create_completion_impl = _capture
    compaction_manager = MagicMock()
    compaction_manager.maybe_compact = AsyncMock(
        return_value=CompactionResult(
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'summary', '_is_compacted_summary': True},
            ],
            compacted=True,
        )
    )
    ctx = ExecutionContext(openai_clients={'main': mock_client}, compaction_manager=compaction_manager)

    async with ctx:
        with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'echo:hi'}])
            output = await run_agent(agent_input)

    return output, request_meta


class TestMainAgentIntegration:
    async def test_next_completion_kwargs_merge_with_authoritative_request_meta(self):
        from nexus.extensions.agents.main.agent import PostIterationHookResult, run_main_agent
        from nexus.extensions.compaction import CompactionResult

        first_call = _make_mock_tool_call('first', 'echo', {'value': 'hi'})
        invalid_call = MagicMock()
        invalid_call.id = 'invalid'
        invalid_call.function.name = 'echo'
        invalid_call.function.arguments = '{'
        responses = iter(
            [
                MagicMock(choices=[MagicMock(message=_make_mock_message(tool_calls=[first_call]))]),
                MagicMock(choices=[MagicMock(message=_make_mock_message(tool_calls=[invalid_call]))]),
                MagicMock(choices=[MagicMock(message=_make_mock_message(content='Done!'))]),
            ]
        )
        captured: list[dict[str, Any]] = []
        mock_client = _MockOpenAIClient()

        async def capture_completion(messages, **kwargs):
            del messages
            captured.append(kwargs)
            return next(responses)

        async def hook(_ctx, _tool_messages):
            return PostIterationHookResult(
                next_completion_kwargs={
                    'sampling_params': {'bad_words': ['tool']},
                    'request_meta': {
                        'hook_fact': 'kept',
                        'iteration': -1,
                        'retry': -1,
                        'invocation_kind': 'hook_override',
                    },
                }
            )

        mock_client._create_completion_impl = capture_completion
        compaction_manager = MagicMock()
        compaction_manager.maybe_compact = AsyncMock(
            return_value=CompactionResult(
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'summary', '_is_compacted_summary': True},
                ],
                compacted=True,
            )
        )
        ctx = ExecutionContext(openai_clients={'main': mock_client}, compaction_manager=compaction_manager)

        async with ctx:
            with (
                patch('nexus.extensions.agents.main.agent.import_from_path', return_value=hook),
                patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call,
            ):
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'echo:hi'}])
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        post_iteration_hook='test::hook',
                        invalid_tool_args_retries=1,
                        max_iterations=2,
                    )
                )

        assert output.exit_status == MainExitStatus.FINISHED.value
        assert 'sampling_params' not in captured[0]
        assert captured[1]['sampling_params'] == {'bad_words': ['tool']}
        assert 'sampling_params' not in captured[2]
        assert [item['request_meta'] for item in captured] == [
            {'iteration': 1, 'retry': 0},
            {
                'hook_fact': 'kept',
                'iteration': 2,
                'retry': 0,
                'invocation_kind': 'compaction_resume',
            },
            {'iteration': 2, 'retry': 1, 'invocation_kind': 'compaction_resume'},
        ]

    async def test_request_meta_marks_retries_and_post_compaction_resume(self):
        from nexus.extensions.agents.main.agent import run_main_agent

        output, request_meta = await _run_compaction_request_meta_case(
            run_main_agent,
            MainAgentInput(user_prompt='test', tools=[], invalid_tool_args_retries=1, max_iterations=2),
        )

        assert output.exit_status == MainExitStatus.FINISHED.value
        assert request_meta == [
            {'iteration': 1, 'retry': 0},
            {'iteration': 1, 'retry': 1},
            {'iteration': 2, 'retry': 0, 'invocation_kind': 'compaction_resume'},
        ]

    async def test_simple_finish_no_tools(self):
        """LLM returns content without tool_calls → exit finished."""
        mock_client = _MockOpenAIClient()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = _make_mock_message(content='Done!')
        mock_client.set_responses(response)

        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            with processor_context([proc]):
                from nexus.extensions.agents.main.agent import run_main_agent

                agent_input = MainAgentInput(
                    user_prompt='hello',
                    tools=[],
                )
                output = await run_main_agent(agent_input)

        assert isinstance(output, MainAgentOutput)
        assert output.exit_status == 'finished'
        assert output.final_answer == 'Done!'
        assert output.iterations == 1

        # Verify processor collected metrics
        result = proc.get_result()
        assert result['agent.exit_status'] == 'finished'
        assert result['agent.iterations'] == 1
        assert len(result['agent.inference_times']) == 1
        assert result['agent.inference_times'][0] > 0

    async def test_refusal_is_returned_without_empty_assistant(self):
        mock_client = _MockOpenAIClient()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = _make_mock_message(refusal='I cannot help with that request.')
        mock_client.set_responses(response)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(MainAgentInput(user_prompt='hello', tools=[]))

        assert output.exit_status == 'refused'
        assert output.final_answer == 'I cannot help with that request.'
        assert output.main_trajectory[-1] == {
            'role': 'assistant',
            'refusal': 'I cannot help with that request.',
        }

    async def test_refusal_never_executes_provider_tool_calls(self):
        tool_call = MagicMock()
        tool_call.id = 'call_after_refusal'
        tool_call.function.name = 'must_not_run'
        tool_call.function.arguments = '{}'
        mock_client = _MockOpenAIClient()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = _make_mock_message(
            refusal='Request refused.',
            tool_calls=[tool_call],
        )
        mock_client.set_responses(response)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(MainAgentInput(user_prompt='hello', tools=[]))

        assert output.exit_status == MainExitStatus.REFUSED.value
        assert output.final_answer == 'Request refused.'
        assert all(message['role'] != 'tool' for message in output.main_trajectory)

    async def test_empty_response_is_inference_error_and_is_not_appended(self):
        mock_client = _MockOpenAIClient()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = _make_mock_message()
        mock_client.set_responses(response)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(MainAgentInput(user_prompt='hello', tools=[]))

        assert output.exit_status == MainExitStatus.INFERENCE_ERROR.value
        assert output.final_answer is None
        assert [message['role'] for message in output.main_trajectory] == ['system', 'user']

    async def test_span_update_failure_does_not_fail_agent(self):
        """Span attribute update failure should not break final output."""
        mock_client = _MockOpenAIClient()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = _make_mock_message(content='Done!')
        mock_client.set_responses(response)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with (
                patch('nexus.extensions.agents.main.agent.get_current_span_id', return_value='span-x'),
                patch('nexus.extensions.agents.main.agent.update_span', side_effect=RuntimeError('trace down')),
            ):
                output = await run_main_agent(MainAgentInput(user_prompt='hello', tools=[]))

        assert output.exit_status == 'finished'
        assert output.final_answer == 'Done!'

    async def test_tool_overrides_apply_to_llm_schemas(self):
        mock_client = _MockOpenAIClient()
        captured_tools: list[Any] = []

        async def _capture_tools(messages, **kwargs):
            captured_tools.append(kwargs.get('tools'))
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = _make_mock_message(content='Done!')
            return response

        mock_client._create_completion_impl = _capture_tools
        tool_func = _make_tool_func('alpha', registry_name='pkg/alpha')

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch('nexus.extensions.agents.main.agent.tool_registry.get', return_value=tool_func):
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='hello',
                        tools=['pkg/alpha'],
                        tool_overrides=[
                            {
                                'tool_name': 'pkg/alpha',
                                'name': 'AlphaTool',
                                'description': 'Overridden description',
                                'params': {'value': 'Overridden param description'},
                            }
                        ],
                    )
                )

        assert output.exit_status == 'finished'
        assert len(captured_tools) == 1
        schema = captured_tools[0][0]
        assert schema['function']['name'] == 'AlphaTool'
        assert schema['function']['description'] == 'Overridden description'
        assert schema['function']['parameters']['properties']['value']['description'] == 'Overridden param description'

    async def test_tool_call_then_finish(self):
        """LLM calls a tool, then finishes on second turn."""
        tc = _make_mock_tool_call('tc1', 'test_integ/echo', {'value': 'hi'})
        msg1 = _make_mock_message(tool_calls=[tc])
        msg2 = _make_mock_message(content='All done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg1)]),
            MagicMock(choices=[MagicMock(message=msg2)]),
        )

        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            with processor_context([proc]):
                from nexus.extensions.agents.main.agent import run_main_agent

                with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                    mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'echo:hi'}])

                    agent_input = MainAgentInput(
                        user_prompt='test',
                        tools=[],
                    )
                    output = await run_main_agent(agent_input)

        assert output.exit_status == 'finished'
        assert output.iterations == 2

        result = proc.get_result()
        assert result['agent.tool_call_count'] == 1
        assert len(result['agent.inference_times']) == 2
        assert len(result['agent.observation_times']) == 1

    async def test_max_tool_response_tokens_truncates_tool_observation(self):
        tool_call = _make_mock_tool_call('tc1', 'test_integ/echo', {'value': 'hi'})
        mock_client = _MockOpenAIClient()
        mock_client.tokenizer = SimpleNamespace(
            encode=lambda text, add_special_tokens=False: list(text),
            decode=lambda token_ids, skip_special_tokens=False: ''.join(token_ids),
        )
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=_make_mock_message(tool_calls=[tool_call]))]),
            MagicMock(choices=[MagicMock(message=_make_mock_message(content='All done'))]),
        )
        budget = minimum_tool_response_token_budget(list)

        async with ExecutionContext(openai_clients={'main': mock_client}):
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': f'H{"middle" * 20}T'}])
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_tool_response_tokens=budget,
                    )
                )

        tool_message = next(message for message in output.main_trajectory if message.get('role') == 'tool')
        assert tool_message['content'] == f'H{TOOL_RESPONSE_TRUNCATION_MARKER}T'

    async def test_tool_response_tokenizer_panic_is_converted(self):
        class PanicException(BaseException):
            pass

        PanicException.__module__ = 'pyo3_runtime'

        tool_call = _make_mock_tool_call('tc1', 'test_integ/echo', {'value': 'hi'})
        mock_client = _MockOpenAIClient()
        mock_client.tokenizer = SimpleNamespace(
            encode=lambda text, **_: list(text),
            decode=lambda token_ids, **_: ''.join(token_ids),
        )
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=_make_mock_message(tool_calls=[tool_call]))]),
        )
        budget = minimum_tool_response_token_budget(list)

        async with ExecutionContext(openai_clients={'main': mock_client}):
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'tool output'}])
                with (
                    patch(
                        'nexus.extensions.agents.main.agent.truncate_tool_response_by_tokens',
                        side_effect=PanicException('NormalizedString bad split'),
                    ),
                    pytest.raises(TokenizerPanicError, match='Failed to tokenize tool result'),
                ):
                    await run_main_agent(
                        MainAgentInput(
                            user_prompt='test',
                            tools=[],
                            max_tool_response_tokens=budget,
                        )
                    )

    async def test_termination_requested_tool_response_finishes_without_second_inference(self):
        """A tool response with termination_requested should finish the loop like OSWorld DONE/FAIL markers."""
        done_tc = _make_mock_tool_call('tc1', 'test_integ/done', {'value': 'DONE'})
        followup_tc = _make_mock_tool_call('tc2', 'test_integ/echo', {'value': 'after'})
        msg1 = _make_mock_message(tool_calls=[done_tc, followup_tc])

        mock_client = _MockOpenAIClient()
        inference_count = 0

        async def _completion(messages, **kwargs):
            nonlocal inference_count
            inference_count += 1
            return MagicMock(choices=[MagicMock(message=msg1)])

        mock_client._create_completion_impl = _completion
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = [
                    ToolResponse(output=[{'type': 'text', 'text': 'DONE'}], meta={'termination_requested': True}),
                    ToolResponse(output=[{'type': 'text', 'text': 'after'}]),
                ]

                output = await run_main_agent(MainAgentInput(user_prompt='test', tools=[]))

        assert output.exit_status == 'finished'
        assert output.iterations == 1
        assert inference_count == 1
        assert mock_call.await_count == 2
        tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
        assert [msg['tool_call_id'] for msg in tool_messages] == ['tc1', 'tc2']

    async def test_multimodal_tool_output_is_preserved_in_history(self):
        tc = _make_mock_tool_call('tc1', 'test_integ/zoom', {'value': 'hi'})
        msg1 = _make_mock_message(tool_calls=[tc])
        msg2 = _make_mock_message(content='All done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg1)]),
            MagicMock(choices=[MagicMock(message=msg2)]),
        )

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                multimodal_output = [
                    {'type': 'text', 'text': 'zoomed'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,abc'}},
                ]
                mock_call.return_value = ToolResponse(output=multimodal_output)

                output = await run_main_agent(MainAgentInput(user_prompt='test', tools=[]))

        tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
        assert len(tool_messages) == 1
        assert tool_messages[0]['content'] == multimodal_output
        assert 'meta' not in tool_messages[0]

    async def test_multimodal_tool_output_disabled_flattens_to_text(self):
        """multimodal_tool_output=False should fall back to extract_tool_output_text."""
        tc = _make_mock_tool_call('tc1', 'test_integ/zoom', {'value': 'hi'})
        msg1 = _make_mock_message(tool_calls=[tc])
        msg2 = _make_mock_message(content='All done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg1)]),
            MagicMock(choices=[MagicMock(message=msg2)]),
        )

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                multimodal_output = [
                    {'type': 'text', 'text': 'zoomed'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,abc'}},
                ]
                mock_call.return_value = ToolResponse(output=multimodal_output)

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        multimodal_tool_output=False,
                    )
                )

        tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
        assert len(tool_messages) == 1
        # With multimodal_tool_output=False, content should be a flat string with image placeholder
        assert tool_messages[0]['content'] == 'zoomed\n[image]'

    async def test_max_iterations(self):
        """Agent hits max_iterations."""
        tc = _make_mock_tool_call('tc1', 'dummy', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()

        async def _always_tool(messages, **kwargs):
            return MagicMock(choices=[MagicMock(message=msg_with_tool)])

        mock_client._create_completion_impl = _always_tool

        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            with processor_context([proc]):
                from nexus.extensions.agents.main.agent import run_main_agent

                with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                    mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'ok'}])

                    agent_input = MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_iterations=2,
                    )
                    output = await run_main_agent(agent_input)

        assert output.exit_status == 'max_iterations'
        assert output.iterations == 2

        result = proc.get_result()
        assert result['agent.exit_status'] == 'max_iterations'
        assert result['agent.termination_reason'] == 'MAX_STEPS'

    async def test_inference_timeout(self):
        """Inference should stop with timeout when LLM call exceeds remaining budget."""
        mock_client = _MockOpenAIClient()

        async def _slow_completion(messages, **kwargs):
            await asyncio.sleep(0.05)
            return MagicMock(choices=[MagicMock(message=_make_mock_message(content='late'))])

        mock_client._create_completion_impl = _slow_completion
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    inference_time_timeout=0.01,
                )
            )

        assert output.exit_status == 'inference_timeout'
        assert output.iterations == 1

    async def test_inference_timeout_stops_immediately_after_post_hoc_charge(self):
        """A completed response that exhausts the budget must not be processed."""
        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(
                choices=[MagicMock(message=_make_mock_message(content='late'))],
                provider_specific_fields={'inference_time': 0.02},
            )
        )
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    inference_time_timeout=0.01,
                )
            )

        assert output.exit_status == 'inference_timeout'
        assert output.final_answer is None

    async def test_provider_timeout_is_not_misclassified_as_inference_budget_timeout(self):
        """A provider's own timeout remains an inference error while budget remains."""
        mock_client = _MockOpenAIClient()
        mock_client.set_side_effect(TimeoutError('provider request timed out'))
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    inference_time_timeout=10,
                )
            )

        assert output.exit_status == 'inference_error'

    async def test_local_one_agent_inference_timeout_stops_immediately_after_post_hoc_charge(
        self,
        monkeypatch,
        tmp_path,
    ):
        """local_one_agent must also reject a response that exhausts the budget."""
        from nexus.extensions.agents.local_one_agent import agent as local_agent

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(
                choices=[MagicMock(message=_make_mock_message(content='late'))],
                provider_specific_fields={'inference_time': 0.02},
            )
        )
        mock_executor = MagicMock()
        mock_executor.get_model_tool_schemas.return_value = []
        monkeypatch.setattr(local_agent, '_provision_env', AsyncMock())
        monkeypatch.setattr(local_agent, '_make_local_tool_executor', lambda *args, **kwargs: mock_executor)
        ctx = ExecutionContext(
            openai_clients={'main': mock_client},
            task_config={
                'local_context': {
                    'workspace': {'path': str(tmp_path), 'scope': 'conversation'},
                    'permission': {'level': 'basic'},
                    'runtime': {'os': 'macos', 'arch': 'x86_64', 'shell': 'bash'},
                }
            },
        )

        async with ctx:
            output = await local_agent.run_local_one_agent(
                local_agent.LocalOneAgentInput(
                    user_prompt='test',
                    tools=[],
                    inference_time_timeout=0.01,
                )
            )

        assert output.exit_status == 'inference_timeout'
        assert output.final_answer is None

    async def test_compaction_inference_timeout(self):
        """A self-summary completion shares the main agent's inference budget."""
        from nexus.utils.timeout import InferenceTimeoutError

        tool_call = _make_mock_tool_call('tc1', 'dummy', {'value': 'one'})
        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=_make_mock_message(tool_calls=[tool_call]))]))
        compaction_manager = MagicMock()
        compaction_manager.maybe_compact = AsyncMock(side_effect=InferenceTimeoutError('budget exhausted'))
        ctx = ExecutionContext(
            openai_clients={'main': mock_client},
            compaction_manager=compaction_manager,
        )

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'ok'}])
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        inference_time_timeout=1,
                        max_iterations=2,
                    )
                )

        assert output.exit_status == 'inference_timeout'
        assert output.iterations == 1
        compaction_manager.maybe_compact.assert_awaited_once()
        inference_budget = compaction_manager.maybe_compact.await_args.kwargs['inference_budget']
        assert inference_budget.total_timeout == 1

    async def test_inference_timeout_reconciles_completed_turns_post_hoc(self, monkeypatch):
        """Completed turns use reported inference time; an overlong next turn still times out."""
        from contextlib import asynccontextmanager

        from nexus.extensions.agents.main import agent as main_agent
        from nexus.utils import timeout as timeout_utils

        tc1 = _make_mock_tool_call('tc1', 'dummy', {'value': 'one'})
        tc2 = _make_mock_tool_call('tc2', 'dummy', {'value': 'two'})
        responses = iter(
            [
                MagicMock(
                    choices=[MagicMock(message=_make_mock_message(tool_calls=[tc1]))],
                    provider_specific_fields={'inference_time': 0.01},
                ),
                MagicMock(
                    choices=[MagicMock(message=_make_mock_message(tool_calls=[tc2]))],
                    provider_specific_fields={'inference_time': 0.01},
                ),
            ]
        )
        mock_client = _MockOpenAIClient()

        async def _completion(messages, **kwargs):
            del messages, kwargs
            await asyncio.sleep(0.02)
            try:
                return next(responses)
            except StopIteration:
                await asyncio.sleep(1)

        mock_client._create_completion_impl = _completion
        ctx = ExecutionContext(openai_clients={'main': mock_client})
        remaining_budgets: list[float | None] = []
        original_with_timeout = timeout_utils.with_timeout

        @asynccontextmanager
        async def _recording_timeout(timeout_sec):
            remaining_budgets.append(timeout_sec)
            async with original_with_timeout(timeout_sec):
                yield

        monkeypatch.setattr(timeout_utils, 'with_timeout', _recording_timeout)

        async with ctx:
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'ok'}])
                output = await asyncio.wait_for(
                    main_agent.run_main_agent(
                        MainAgentInput(
                            user_prompt='test',
                            tools=[],
                            inference_time_timeout=0.08,
                            max_iterations=3,
                        )
                    ),
                    timeout=0.25,
                )

        assert output.exit_status == 'inference_timeout'
        assert output.iterations == 3
        assert mock_call.await_count == 2
        assert [budget for budget in remaining_budgets if budget is not None] == pytest.approx([0.08, 0.07, 0.06])

    async def test_inference_runtime_error(self):
        """Runtime failure from create_completion should map to inference_error."""
        mock_client = _MockOpenAIClient()
        mock_client.set_side_effect(RuntimeError('gateway failure'))
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='test',
                    tools=[],
                )
            )

        assert output.exit_status == 'inference_error'
        assert output.iterations == 1

    async def test_invalid_inference_response_shape(self):
        """Malformed model response should map to inference_error."""
        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[]))
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(MainAgentInput(user_prompt='test', tools=[]))

        assert output.exit_status == 'inference_error'
        assert output.iterations == 1

    async def test_observation_timeout(self):
        """Observation should stop with timeout when tool execution exceeds remaining budget."""
        tc = _make_mock_tool_call('tc1', 'slow_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_with_tool)]))
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            async def _slow_tool_call(name, params):
                await asyncio.sleep(0.05)
                return ToolResponse(output=[{'type': 'text', 'text': 'ok'}])

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = _slow_tool_call
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        observation_time_timeout=0.01,
                    )
                )

        assert output.exit_status == 'observation_timeout'
        assert output.iterations == 1

    async def test_tool_call_error_exhausted(self):
        """Agent exits with tool_execution_error when tool-call execution retries are exhausted."""
        tc = _make_mock_tool_call('tc1', 'bad_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_with_tool)]))

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = RuntimeError('tool exploded')

                agent_input = MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    tool_call_error_retries=1,
                )
                output = await run_main_agent(agent_input)

        assert output.exit_status == 'tool_execution_error'
        assert output.iterations == 1
        assert mock_call.await_count == 2

    async def test_tool_error_does_not_use_tool_call_error_retries(self):
        """Tool-call schema/name errors are model-correctable format errors, not execution retry errors."""
        tc = _make_mock_tool_call('tc1', 'missing_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_with_tool)]))

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = ToolNameResolutionError('missing_tool', ['known_tool'])

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        tool_call_error_retries=2,
                    )
                )

        assert output.exit_status == 'tool_error'
        assert output.iterations == 1
        assert mock_call.await_count == 1
        tool_messages = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert len(tool_messages) == 1
        assert "Tool error: Tool 'missing_tool' not found" in tool_messages[0]['content']

    async def test_parallel_tool_error_does_not_use_tool_call_error_retries(self):
        """Parallel tool-call schema/name errors should also skip execution retries."""
        tc = _make_mock_tool_call('tc1', 'missing_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_with_tool)]))

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'parallel_call', new_callable=AsyncMock) as mock_parallel_call:
                mock_parallel_call.side_effect = ToolNameResolutionError('missing_tool', ['known_tool'])

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        parallel_tool_calls=True,
                        tool_call_error_retries=2,
                    )
                )

        assert output.exit_status == 'tool_error'
        assert output.iterations == 1
        assert mock_parallel_call.await_count == 1
        tool_messages = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert len(tool_messages) == 1
        assert "Tool error: Tool 'missing_tool' not found" in tool_messages[0]['content']

    async def test_tool_call_error_continue_recovers_after_exhausted_retries(self, monkeypatch):
        """continue-on-tool-error feeds tool_execution_error back so the agent can recover."""
        monkeypatch.delenv('NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE', raising=False)
        tc = _make_mock_tool_call('tc1', 'bad_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])
        msg_done = _make_mock_message(content='Done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_with_tool)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = RuntimeError('tool exploded')

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_iterations=3,
                        tool_error_continue=True,
                    )
                )

        assert output.exit_status == 'finished'
        assert output.iterations == 2
        assert output.final_answer == 'Done'
        assert mock_call.await_count == 1
        tool_messages = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert len(tool_messages) == 1
        assert tool_messages[0]['tool_call_id'] == 'tc1'
        assert 'tool exploded' in tool_messages[0]['content']

    async def test_tool_os_connection_error_stays_recoverable(self, monkeypatch):
        """A LOCAL/MCP tool's own ConnectionResetError is a recoverable tool error.

        The OS exceptions that reveal a dead sandbox transport can also come
        from a tool's business logic (e.g. a local tool hitting an external
        search service). Only the executor's RUNTIME dispatch branch wraps
        them into RuntimeTransportError, so the raw OS error seen here must
        not veto continue-on-tool-error.
        """
        monkeypatch.delenv('NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE', raising=False)
        tc = _make_mock_tool_call('tc1', 'web_search', {'query': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])
        msg_done = _make_mock_message(content='Done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_with_tool)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = ConnectionResetError('outer service reset the connection')

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_iterations=3,
                        tool_error_continue=True,
                    )
                )

        assert output.exit_status == 'finished'
        assert output.iterations == 2  # the run continued into the next turn
        assert output.final_answer == 'Done'
        assert mock_call.await_count == 1

    @pytest.mark.parametrize('tool_error_continue', [False, True])
    async def test_runtime_transport_error_is_fatal(self, monkeypatch, tool_error_continue):
        """RuntimeTransportError (dead sandbox transport) vetoes continue-on-tool-error."""
        monkeypatch.delenv('NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE', raising=False)
        tc = _make_mock_tool_call('tc1', 'code_interpreter', {'code': '1'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_with_tool)]))

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = RuntimeTransportError('sandbox connection lost')

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_iterations=3,
                        tool_error_continue=tool_error_continue,
                    )
                )

        assert output.exit_status == 'tool_execution_error'
        assert output.iterations == 1  # transport death never reaches a scoring turn
        assert mock_call.await_count == 1

    async def test_parallel_tool_os_connection_error_stays_recoverable(self, monkeypatch):
        """Parallel branch: a raw OS connection error is recoverable with continue enabled."""
        monkeypatch.delenv('NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE', raising=False)
        tc = _make_mock_tool_call('tc1', 'web_search', {'query': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])
        msg_done = _make_mock_message(content='Done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_with_tool)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'parallel_call', new_callable=AsyncMock) as mock_parallel_call:
                mock_parallel_call.side_effect = ConnectionResetError('outer service reset the connection')

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_iterations=3,
                        parallel_tool_calls=True,
                        tool_error_continue=True,
                    )
                )

        assert output.exit_status == 'finished'
        assert output.iterations == 2
        assert output.final_answer == 'Done'
        assert mock_parallel_call.await_count == 1

    @pytest.mark.parametrize('tool_error_continue', [False, True])
    async def test_parallel_runtime_transport_error_is_fatal(self, monkeypatch, tool_error_continue):
        """Parallel branch: RuntimeTransportError stays fatal despite continue-on-tool-error."""
        monkeypatch.delenv('NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE', raising=False)
        tc = _make_mock_tool_call('tc1', 'code_interpreter', {'code': '1'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_with_tool)]))

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'parallel_call', new_callable=AsyncMock) as mock_parallel_call:
                mock_parallel_call.side_effect = RuntimeTransportError('sandbox connection lost')

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_iterations=3,
                        parallel_tool_calls=True,
                        tool_error_continue=tool_error_continue,
                    )
                )

        assert output.exit_status == 'tool_execution_error'
        assert output.iterations == 1
        assert mock_parallel_call.await_count == 1

    async def test_tool_call_error_retries_recover(self):
        """Tool-call retry can recover within one iteration."""
        tc = _make_mock_tool_call('tc1', 'flaky_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])
        msg_done = _make_mock_message(content='Done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_with_tool)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = [
                    RuntimeError('temporary error'),
                    ToolResponse(output=[{'type': 'text', 'text': 'ok'}]),
                ]

                agent_input = MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    tool_call_error_retries=1,
                )
                output = await run_main_agent(agent_input)

        assert output.exit_status == 'finished'
        assert output.iterations == 2
        assert mock_call.await_count == 2

    async def test_tool_output_render_error(self):
        """Unexpected tool output shape error should map to tool_execution_error."""
        tc = _make_mock_tool_call('tc1', 'bad_output_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_with_tool)]))
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        class _BadText:
            pass

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': _BadText()}])

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                    )
                )

        assert output.exit_status == 'tool_execution_error'
        assert output.iterations == 1

    async def test_tool_output_render_error_continue_recovers(self, monkeypatch):
        """continue-on-tool-error also covers output rendering failures."""
        monkeypatch.delenv('NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE', raising=False)
        tc = _make_mock_tool_call('tc1', 'bad_output_tool', {'value': 'x'})
        msg_with_tool = _make_mock_message(tool_calls=[tc])
        msg_done = _make_mock_message(content='Done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_with_tool)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        class _BadText:
            pass

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': _BadText()}])

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=[],
                        max_iterations=3,
                        tool_error_continue=True,
                    )
                )

        assert output.exit_status == 'finished'
        assert output.iterations == 2
        assert output.final_answer == 'Done'
        assert mock_call.await_count == 1
        tool_messages = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert len(tool_messages) == 1
        assert tool_messages[0]['tool_call_id'] == 'tc1'
        assert 'Tool error' in tool_messages[0]['content']

    async def test_no_tool_calls_finishes_even_with_retry_enabled(self):
        """No tool calls should finish immediately, even if invalid_tool_args_retries > 0."""
        msg_empty = _make_mock_message(content='hmm')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_empty)]))

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            agent_input = MainAgentInput(
                user_prompt='test',
                tools=[],
                invalid_tool_args_retries=1,
            )
            output = await run_main_agent(agent_input)

        assert output.exit_status == 'finished'
        assert output.iterations == 1

    async def test_valid_json_invalid_tool_args_retry_with_observation(self):
        """Valid JSON that fails input_model validation should be fed back for model correction."""
        bad_tc = _make_mock_tool_call('tc1', 'RequiredTool', {})
        good_tc = _make_mock_tool_call('tc2', 'RequiredTool', {'value': 'ok'})

        msg_bad = _make_mock_message(tool_calls=[bad_tc])
        msg_good = _make_mock_message(tool_calls=[good_tc])
        msg_done = _make_mock_message(content='Done')

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_bad)]),
            MagicMock(choices=[MagicMock(message=msg_good)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )
        tool_func = _make_tool_func('RequiredTool', registry_name='pkg/required', input_model=_RequiredInput)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch('nexus.extensions.agents.main.agent.tool_registry.get', return_value=tool_func):
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=['pkg/required'],
                        invalid_tool_args_retries=1,
                    )
                )

        assert output.exit_status == 'finished'
        assert output.iterations == 3

        tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
        assert len(tool_messages) == 2
        assert tool_messages[0]['tool_call_id'] == 'tc1'
        assert "Tool error: Invalid arguments for tool 'RequiredTool'" in tool_messages[0]['content']
        assert 'value: Field required' in tool_messages[0]['content']
        assert 'errors.pydantic.dev' not in tool_messages[0]['content']
        assert tool_messages[1]['tool_call_id'] == 'tc2'
        assert tool_messages[1]['content'] == 'RequiredTool:ok'

    async def test_valid_json_invalid_tool_args_without_retry_exits_tool_error(self):
        """Default retry budget remains compatible: validation error terminates as tool_error."""
        bad_tc = _make_mock_tool_call('tc1', 'RequiredTool', {})
        msg_bad = _make_mock_message(tool_calls=[bad_tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_bad)]))
        tool_func = _make_tool_func('RequiredTool', registry_name='pkg/required', input_model=_RequiredInput)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch('nexus.extensions.agents.main.agent.tool_registry.get', return_value=tool_func):
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=['pkg/required'],
                    )
                )

        assert output.exit_status == 'tool_error'
        assert output.iterations == 1
        tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
        assert len(tool_messages) == 1
        assert "Tool error: Invalid arguments for tool 'RequiredTool'" in tool_messages[0]['content']

    async def test_invalid_tool_args_skip_other_tool_calls_in_same_message(self):
        """Pre-validation should avoid partial execution when one of several tool calls is invalid."""
        bad_tc = _make_mock_tool_call('tc1', 'RequiredTool', {})
        good_tc = _make_mock_tool_call('tc2', 'OptionalTool', {'value': 'ok'})
        msg_bad = _make_mock_message(tool_calls=[bad_tc, good_tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_bad)]))
        tool_funcs = {
            'pkg/required': _make_tool_func('RequiredTool', registry_name='pkg/required', input_model=_RequiredInput),
            'pkg/optional': _make_tool_func('OptionalTool', registry_name='pkg/optional'),
        }

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch('nexus.extensions.agents.main.agent.tool_registry.get', side_effect=tool_funcs.__getitem__):
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='test',
                        tools=['pkg/required', 'pkg/optional'],
                    )
                )

        assert output.exit_status == 'tool_error'
        tool_messages = [msg for msg in output.main_trajectory if msg.get('role') == 'tool']
        assert [msg['tool_call_id'] for msg in tool_messages] == ['tc1', 'tc2']
        assert "Tool error: Invalid arguments for tool 'RequiredTool'" in tool_messages[0]['content']
        assert 'not executed because another tool call' in tool_messages[1]['content']
        assert tool_messages[1]['content'] != 'OptionalTool:ok'

    async def test_retry_on_bad_json(self):
        """Agent retries when tool call arguments are invalid JSON, then recovers."""
        bad_tc = MagicMock()
        bad_tc.id = 'tc1'
        bad_tc.function.name = 'some_tool'
        bad_tc.function.arguments = '{invalid json}'

        good_tc = _make_mock_tool_call('tc2', 'some_tool', {'value': 'ok'})

        msg_bad = _make_mock_message(tool_calls=[bad_tc])
        msg_good = _make_mock_message(tool_calls=[good_tc])
        msg_done = _make_mock_message(content='Done')

        mock_client = _MockOpenAIClient()
        # Iteration 1: bad JSON → retry → good → execute tool
        # Iteration 2: no tool calls → finished
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_bad)]),
            MagicMock(choices=[MagicMock(message=msg_good)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )

        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.return_value = ToolResponse(output=[{'type': 'text', 'text': 'ok'}])

                agent_input = MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    invalid_tool_args_retries=1,
                )
                output = await run_main_agent(agent_input)

        # Iteration 1 recovered from bad JSON; iteration 2 no tool calls → finished
        assert output.exit_status == 'finished'
        assert output.iterations == 2
        # Verify the tool was actually called (iteration 1 recovered successfully)
        mock_call.assert_called_once()

    async def test_invalid_json_exhausted_exits_tool_error(self):
        """Unparseable tool-call arguments are format errors when retries are exhausted."""
        bad_tc = MagicMock()
        bad_tc.id = 'tc1'
        bad_tc.function.name = 'some_tool'
        bad_tc.function.arguments = '{invalid json}'
        msg_bad = _make_mock_message(tool_calls=[bad_tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(MagicMock(choices=[MagicMock(message=msg_bad)]))

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='test',
                    tools=[],
                )
            )

        assert output.exit_status == 'tool_error'
        assert MainExitStatus(output.exit_status).termination_reason == 'FORMAT_ERROR'
        assert output.iterations == 1

    @pytest.mark.parametrize(
        'message',
        [
            _make_mock_message(content='partial reasoning'),
            _make_mock_message(tool_calls=[_make_mock_tool_call('tc1', 'some_tool', {'value': 'partial'})]),
            _make_mock_message(refusal='partial refusal'),
            _make_mock_message(),
        ],
    )
    async def test_output_length_exceeded_stops_before_tool_handling(self, message):
        response = MagicMock(choices=[MagicMock(message=message, finish_reason='length')])
        mock_client = _MockOpenAIClient()
        mock_client._create_completion_impl = AsyncMock(return_value=response)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    invalid_tool_args_retries=2,
                )
            )

        assert output.exit_status == MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value
        assert MainExitStatus(output.exit_status).termination_reason == 'OUTPUT_LENGTH_EXCEEDED'
        assert output.main_trajectory[-1] == message.model_dump(exclude_none=True)
        assert mock_client._create_completion_impl.await_count == 1

    async def test_output_length_exceeded_takes_precedence_over_completion_parsing(self):
        """A provider max_tokens stop stays distinct even when tool-call JSON is incomplete."""
        bad_tool_call = MagicMock()
        bad_tool_call.id = 'tc1'
        bad_tool_call.function.name = 'some_tool'
        bad_tool_call.function.arguments = '{incomplete json'
        message = _make_mock_message(content='partial reasoning', tool_calls=[bad_tool_call])
        response = MagicMock(choices=[MagicMock(message=message, finish_reason='length')])
        mock_client = _MockOpenAIClient()
        mock_client._create_completion_impl = AsyncMock(return_value=response)

        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            from nexus.extensions.agents.main.agent import run_main_agent

            output = await run_main_agent(
                MainAgentInput(
                    user_prompt='test',
                    tools=[],
                    invalid_tool_args_retries=2,
                )
            )

        assert output.exit_status == MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value
        assert MainExitStatus(output.exit_status).termination_reason == 'OUTPUT_LENGTH_EXCEEDED'
        assert output.final_answer == 'partial reasoning'
        assert output.main_trajectory[-1] == message.model_dump(exclude_none=True)
        assert mock_client._create_completion_impl.await_count == 1

    async def test_context_length_exceeded(self):
        """Agent catches ContextLengthExceededError → context_length_exceeded."""
        from nexus.core import ContextLengthExceededError

        mock_client = _MockOpenAIClient()
        mock_client.set_side_effect(ContextLengthExceededError('too long'))

        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': mock_client})

        async with ctx:
            with processor_context([proc]):
                from nexus.extensions.agents.main.agent import run_main_agent

                agent_input = MainAgentInput(
                    user_prompt='test',
                    tools=[],
                )
                output = await run_main_agent(agent_input)

        assert output.exit_status == 'context_length_exceeded'

        result = proc.get_result()
        assert result['agent.termination_reason'] == 'TRUNCATION'


# ---------------------------------------------------------------------------
# End-to-end: continue-on-tool-error switch (NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE)
# ---------------------------------------------------------------------------


class TestToolErrorContinueEndToEnd:
    """A bad tool call (unresolvable name) should abort by default, but with the
    switch enabled it is fed back to the model as an observation so the run can
    recover instead of being discarded as FORMAT_ERROR."""

    _ENV = 'NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE'

    @staticmethod
    def _client_bad_then_done():
        """Turn 1: call a non-existent tool. Turn 2: finish with plain text."""
        bad_tc = _make_mock_tool_call('c1', 'ghost_tool', {'value': 'x'})
        msg_bad = _make_mock_message(tool_calls=[bad_tc])
        msg_done = _make_mock_message(content='done')
        client = _MockOpenAIClient()
        client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_bad)]),
            MagicMock(choices=[MagicMock(message=msg_done)]),
        )
        return client

    async def _run(self, client, **agent_kwargs):
        from nexus.extensions.agents.main.agent import run_main_agent

        # A single real registered tool ('alpha') so the executor has a non-empty
        # schema set; 'ghost_tool' then fails name resolution during validation.
        alpha = _make_tool_func('alpha')
        ctx = ExecutionContext(openai_clients={'main': client})
        async with ctx:
            with patch('nexus.extensions.agents.main.agent.tool_registry.get', return_value=alpha):
                return await run_main_agent(
                    MainAgentInput(user_prompt='hi', tools=['alpha'], max_iterations=5, **agent_kwargs)
                )

    @staticmethod
    def _tool_error_messages(output):
        return [
            m for m in output.main_trajectory if m.get('role') == 'tool' and 'Tool error' in str(m.get('content', ''))
        ]

    async def test_aborts_with_format_error_by_default(self, monkeypatch):
        monkeypatch.delenv(self._ENV, raising=False)
        output = await self._run(self._client_bad_then_done())

        assert output.exit_status == MainExitStatus.TOOL_ERROR.value
        assert MainExitStatus(output.exit_status).termination_reason == 'FORMAT_ERROR'
        assert output.iterations == 1  # second (recovery) turn never requested
        assert output.final_answer is None
        # the validation error was recorded but the run still terminated
        assert len(self._tool_error_messages(output)) == 1

    async def test_continues_and_finishes_when_enabled_via_config(self, monkeypatch):
        monkeypatch.delenv(self._ENV, raising=False)
        output = await self._run(self._client_bad_then_done(), tool_error_continue=True)

        assert output.exit_status == MainExitStatus.FINISHED.value
        assert output.final_answer == 'done'
        assert output.iterations == 2  # recovered on the next turn
        # the error was fed back as a tool observation the model could react to
        err_msgs = self._tool_error_messages(output)
        assert len(err_msgs) == 1
        assert err_msgs[0]['tool_call_id'] == 'c1'
        # history stays well-formed: the bad assistant call has a matching tool reply
        roles = [m['role'] for m in output.main_trajectory]
        assert roles == ['system', 'user', 'assistant', 'tool', 'assistant']

    async def test_env_var_overrides_disabled_config(self, monkeypatch):
        monkeypatch.setenv(self._ENV, '1')
        # config left at default False — env var should force continue
        output = await self._run(self._client_bad_then_done())

        assert output.exit_status == MainExitStatus.FINISHED.value
        assert output.final_answer == 'done'

    async def test_env_var_can_force_disable_over_enabled_config(self, monkeypatch):
        monkeypatch.setenv(self._ENV, 'false')
        output = await self._run(self._client_bad_then_done(), tool_error_continue=True)

        assert output.exit_status == MainExitStatus.TOOL_ERROR.value
        assert output.final_answer is None

    async def test_invalid_json_loop_keeps_counts_consistent(self, monkeypatch):
        """Repeated unparseable-args turns in continue mode must keep
        tool_error_count <= tool_call_count (each batch counts as attempted)."""
        monkeypatch.delenv(self._ENV, raising=False)

        def _bad_json_msg(call_id):
            tc = MagicMock()
            tc.id = call_id
            tc.function.name = 'some_tool'
            tc.function.arguments = '{invalid json}'  # unparseable
            return _make_mock_message(tool_calls=[tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=_bad_json_msg('c1'))]),
            MagicMock(choices=[MagicMock(message=_bad_json_msg('c2'))]),
            MagicMock(choices=[MagicMock(message=_make_mock_message(content='done'))]),
        )

        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            with processor_context([proc]):
                from nexus.extensions.agents.main.agent import run_main_agent

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='hi',
                        tools=[],
                        max_iterations=5,
                        tool_error_continue=True,
                    )
                )

        assert output.exit_status == MainExitStatus.FINISHED.value
        assert output.iterations == 3
        result = proc.get_result()
        # both bad batches counted as attempted AND errored (was 0 vs 2 before the fix)
        assert result['agent.tool_call_count'] == 2
        assert result['agent.tool_error_count'] == 2
        assert result['agent.tool_error_count'] <= result['agent.tool_call_count']

    async def test_partial_bad_json_batch_counts_only_malformed_as_error(self, monkeypatch):
        """A batch with one unparseable + one valid call counts both as attempted
        but only the malformed one as an error."""
        monkeypatch.delenv(self._ENV, raising=False)

        bad_tc = MagicMock()
        bad_tc.id = 'bad1'
        bad_tc.function.name = 'some_tool'
        bad_tc.function.arguments = '{bad json'  # unparseable
        good_tc = _make_mock_tool_call('good1', 'some_tool', {'value': 'ok'})  # valid JSON
        msg_mixed = _make_mock_message(tool_calls=[bad_tc, good_tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_mixed)]),
            MagicMock(choices=[MagicMock(message=_make_mock_message(content='done'))]),
        )

        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            with processor_context([proc]):
                from nexus.extensions.agents.main.agent import run_main_agent

                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='hi',
                        tools=[],
                        max_iterations=5,
                        tool_error_continue=True,
                    )
                )

        assert output.exit_status == MainExitStatus.FINISHED.value
        assert output.iterations == 2
        result = proc.get_result()
        assert result['agent.tool_call_count'] == 2  # both attempted
        assert result['agent.tool_error_count'] == 1  # only the malformed one
        # both calls still got a tool response (history well-formed)
        tool_msgs = [m for m in output.main_trajectory if m.get('role') == 'tool']
        assert {m['tool_call_id'] for m in tool_msgs} == {'bad1', 'good1'}

    async def test_validation_error_continue_emits_observation_span(self, monkeypatch):
        """The Case-B continue path emits an agent.observation span so the
        recoverable-error iteration still shows up in the trace/metrics
        (it otherwise skips the tool-execution span)."""
        monkeypatch.delenv(self._ENV, raising=False)
        from nexus.extensions.agents.main.agent import run_main_agent

        alpha = _make_tool_func('alpha')
        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': self._client_bad_then_done()})
        async with ctx:
            with processor_context([proc]):
                with patch('nexus.extensions.agents.main.agent.tool_registry.get', return_value=alpha):
                    output = await run_main_agent(
                        MainAgentInput(
                            user_prompt='hi',
                            tools=['alpha'],
                            max_iterations=5,
                            tool_error_continue=True,
                        )
                    )

        assert output.exit_status == MainExitStatus.FINISHED.value
        result = proc.get_result()
        # iter 1 hit a bad tool name (no tool executed) but still recorded an
        # observation span; iter 2 finished with plain text (no observation).
        assert len(result['agent.observation_times']) == 1
        assert len(result['agent.inference_times']) == 2

    async def test_invalid_json_continue_emits_observation_span(self, monkeypatch):
        """The Case-A (unparseable args) continue path also emits an
        agent.observation span so that iteration stays visible in trace/metrics."""
        monkeypatch.delenv(self._ENV, raising=False)
        from nexus.extensions.agents.main.agent import run_main_agent

        bad_tc = MagicMock()
        bad_tc.id = 'b1'
        bad_tc.function.name = 'some_tool'
        bad_tc.function.arguments = '{bad json'  # unparseable
        msg_bad = _make_mock_message(tool_calls=[bad_tc])

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(
            MagicMock(choices=[MagicMock(message=msg_bad)]),
            MagicMock(choices=[MagicMock(message=_make_mock_message(content='done'))]),
        )

        proc = MainAgentMetricsProcessor()
        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            with processor_context([proc]):
                output = await run_main_agent(
                    MainAgentInput(
                        user_prompt='hi',
                        tools=[],
                        max_iterations=5,
                        tool_error_continue=True,
                    )
                )

        assert output.exit_status == MainExitStatus.FINISHED.value
        result = proc.get_result()
        assert len(result['agent.observation_times']) == 1
        assert len(result['agent.inference_times']) == 2


class TestMainAgentCompactionInstall:
    @staticmethod
    def _input(**compaction: Any) -> MainAgentInput:
        return MainAgentInput(tools=[], user_prompt='hi', compaction=compaction)

    async def test_ratio_uses_the_client_sequence_limit(self) -> None:
        client = _MockOpenAIClient()
        client.max_model_len = 100
        ctx = ExecutionContext(openai_clients={'main': client})

        async with ctx:
            _install_compaction_manager(
                self._input(compaction_condition='remaining_tokens_exhausted', condition={'ratio': 0.5})
            )

        assert ctx.compaction_manager._condition.ratio_base == 100

    async def test_ratio_requires_a_client_sequence_limit(self) -> None:
        ctx = ExecutionContext(openai_clients={'main': _MockOpenAIClient()})

        async with ctx:
            with pytest.raises(ValueError, match='max_generation_length'):
                _install_compaction_manager(self._input(condition={'ratio': 0.5}))

    async def test_rejects_a_runner_installed_manager(self) -> None:
        ctx = ExecutionContext(openai_clients={'main': _MockOpenAIClient()}, compaction_manager=MagicMock())

        async with ctx:
            with pytest.raises(ValueError, match='runner-installed compaction manager'):
                _install_compaction_manager(self._input())


# ---------------------------------------------------------------------------
# Tool calls the client flagged as unparsed (UNPARSED_TOOL_CALL_FIELD)
# ---------------------------------------------------------------------------

_MALFORMED_CALL_TEXT = 'Run it.<tool_calls><tool_call>code<arg_key>code<arg_key>code</arg_key></tool_call></tool_calls>'


def _unparsed_tool_call_completion(*, finish_reason: str = 'stop', content: str = _MALFORMED_CALL_TEXT):
    from nexus.core import UNPARSED_TOOL_CALL_FIELD
    from nexus.core.types import ExtendedChoice, ExtendedMessage

    return ExtendedChatCompletion(
        id='unparsed-tool-call',
        created=0,
        choices=[
            ExtendedChoice(
                finish_reason=finish_reason,
                index=0,
                message=ExtendedMessage(role='assistant', content=content, tool_calls=None),
                provider_specific_fields={UNPARSED_TOOL_CALL_FIELD: True},
            )
        ],
    )


class TestMainAgentUnparsedToolCall:
    async def test_unparsed_tool_call_exits_tool_error_without_retry_or_execution(self):
        from nexus.extensions.agents.main.agent import run_main_agent

        mock_client = _MockOpenAIClient()
        calls: list[dict[str, Any]] = []

        async def completion(messages, **kwargs):
            calls.append(kwargs)
            return _unparsed_tool_call_completion()

        mock_client._create_completion_impl = completion
        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            with patch.object(ToolExecutor, 'call', new_callable=AsyncMock) as mock_call:
                output = await run_main_agent(
                    MainAgentInput(user_prompt='hello', tools=[], invalid_tool_args_retries=2, tool_error_continue=True)
                )

        assert output.exit_status == MainExitStatus.TOOL_ERROR.value
        assert MainExitStatus.termination_reason_for(output.exit_status) == 'FORMAT_ERROR'
        assert output.main_trajectory[-1] == {'role': 'assistant', 'content': _MALFORMED_CALL_TEXT}
        assert len(calls) == 1
        mock_call.assert_not_called()

    async def test_output_length_takes_priority_over_unparsed_tool_call(self):
        from nexus.extensions.agents.main.agent import run_main_agent

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(_unparsed_tool_call_completion(finish_reason='length'))
        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            output = await run_main_agent(MainAgentInput(user_prompt='hello', tools=[]))

        assert output.exit_status == MainExitStatus.OUTPUT_LENGTH_EXCEEDED.value

    async def test_unparsed_tool_call_with_empty_content_is_not_an_inference_error(self):
        from nexus.extensions.agents.main.agent import run_main_agent

        mock_client = _MockOpenAIClient()
        mock_client.set_responses(_unparsed_tool_call_completion(content=''))
        ctx = ExecutionContext(openai_clients={'main': mock_client})
        async with ctx:
            output = await run_main_agent(MainAgentInput(user_prompt='hello', tools=[]))

        assert output.exit_status == MainExitStatus.TOOL_ERROR.value
        assert output.main_trajectory[-1]['role'] == 'assistant'
