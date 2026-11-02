"""Exercise routed requests through real clients, processors and trajectory assembly."""

from __future__ import annotations

import asyncio
import copy
import json

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from _fakes import FakeTokenizer, RecordingEndpoint

from nexus.core import ContextLengthExceededError
from nexus.core.context import ExecutionContext
from nexus.core.openai_client import OpenAIConfig, apply_enforced_request_args
from nexus.extensions.agents.main.agent import MainAgentInput, _install_compaction_manager
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.policies.hycode import COMPACT_TRIGGER_MARKER
from nexus.gateway.session_manager import session_manager
from nexus.runtime.server.services.codex_compaction import CODEX_TURN_METADATA_HEADER
from nexus.runtime.server.services.llm_handler import LLMRequestHandler
from nexus.runtime.server.services.responses_proxy.server import ResponsesProxy


class MessageService:
    def __init__(self, name: str, *, blocked: bool = False) -> None:
        self.name = name
        self.config = OpenAIConfig(client_args={}, request_args={})
        self.calls: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()

    async def create_completion(self, **kwargs: Any) -> SimpleNamespace:
        model = kwargs.pop('model', None)
        kwargs = apply_enforced_request_args(
            {'model': model, **self.config.request_args, **kwargs}, self.config.enforced_request_args
        )
        self.calls.append(copy.deepcopy(kwargs))
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(
            model='same-server-model',
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.name), finish_reason='stop')],
        )


class OffsetTokenizer(FakeTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [value + 1000 for value in super().encode(text, add_special_tokens)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return super().decode([value - 1000 for value in token_ids], skip_special_tokens)


@pytest.fixture(autouse=True)
async def clean_sessions() -> AsyncIterator[None]:
    yield
    for session in list(session_manager._sessions.values()):
        await session.abort_in_flight()
        await session.finish_pending_publications()
    session_manager.end_all()


def message_model(service: MessageService, model: str) -> dict[str, Any]:
    service.config.request_args['model'] = model
    return {'openai_client': service}


def build_client(
    model_configs: dict[str, dict[str, Any]],
    *,
    request_args: dict[str, Any] | None = None,
    **config: Any,
) -> GatewayOpenAIClient:
    return GatewayOpenAIClient(
        OpenAIConfig(
            client_args={
                'trajectory_id': 'multi-test',
                'scaffold_policy': 'nexus_main_agent',
                'models': [{'id': model_id, **model_config} for model_id, model_config in model_configs.items()],
                'default_model_id': 'main',
                'compaction_model_id': 'subagent',
                **config,
            },
            request_args=(
                {'max_tokens': 100, 'sampling_params': {'max_tokens': 100}} if request_args is None else request_args
            ),
        )
    )


@pytest.mark.parametrize('default_model_id', ['main', 'subagent'])
async def test_session_segment_counters_use_explicit_model(default_model_id: str) -> None:
    service = MessageService('unused')
    client = build_client(
        {'main': message_model(service, 'a5b'), 'subagent': message_model(service, 'a3b')},
        default_model_id=default_model_id,
    )
    session = client._session()
    registry = client._model_registry
    assert registry.default_model_id == default_model_id, 'Keep the configured default model ID'
    binding = registry.get(default_model_id)
    session.set_segment_consumed_tokens(3, binding=binding)
    assert session.segment_consumed_tokens(binding=binding) == 3, 'Read and write the selected model counter'
    session.set_segment_consumed_tokens(0, binding=binding)
    assert session.segment_consumed_tokens(binding=binding) == 0, 'Reset the same counter'


@pytest.mark.parametrize(
    ('field', 'session_limit', 'model_limit'),
    [
        ('max_model_len', None, 40),
        ('max_model_len', None, 60),
        ('max_model_len', None, None),
        ('token_budget', 40, 60),
        ('token_budget', 60, 40),
        ('token_budget', 40, None),
        ('token_budget', None, 40),
        ('token_budget', None, None),
    ],
)
async def test_model_budget_uses_only_the_designated_configuration_source(
    field: str, session_limit: int | None, model_limit: int | None
) -> None:
    tokenizer = FakeTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['ok'])
    client = build_client(
        {
            'main': {
                'tokenizer': tokenizer,
                'server_manager': endpoint,
                field: model_limit,
            },
            'subagent': message_model(MessageService('unused'), 'a3b'),
        },
        token_budget=session_limit,
    )
    await client.create_completion([{'role': 'user', 'content': 'q'}])
    if field == 'max_model_len':
        expected = 100 if model_limit is None else min(100, model_limit - len(endpoint.calls[0]['prompt_ids']))
        assert client.max_model_len == model_limit, 'The public model limit must agree with the selected model'
    else:
        expected = 100 if session_limit is None else min(100, session_limit)
    assert endpoint.max_tokens_per_call == [expected], 'Ignore limits configured at the wrong scope'


async def test_fork_compaction_ratio_uses_default_model_length() -> None:
    tokenizer = FakeTokenizer()
    endpoint = RecordingEndpoint(tokenizer, [])
    client = build_client(
        {
            'main': {
                'tokenizer': tokenizer,
                'server_manager': endpoint,
                'max_model_len': 40,
            },
            'subagent': {
                'tokenizer': tokenizer,
                'server_manager': endpoint,
                'max_model_len': 80,
            },
        },
    )
    child = client.fork_for_agent(agent_id='child', parent_agent_id='main')
    assert client.max_model_len == 40, 'Root must report its own model limit'
    assert child.max_model_len == 40, 'Forks use the registry default model'
    async with ExecutionContext(openai_clients={'main': child}) as context:
        agent_input = MainAgentInput(
            tools=[],
            user_prompt='q',
            compaction={'compaction_condition': 'remaining_tokens_exhausted', 'condition': {'ratio': 0.5}},
        )
        _install_compaction_manager(agent_input)
        assert context.compaction_manager._condition.ratio_base == 40, 'Use the default model capacity'


@pytest.mark.parametrize('explicit', [False, True])
async def test_model_budget_consumption_tightens_default_and_explicit_followup_requests(explicit: bool) -> None:
    tokenizer = FakeTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['ab', 'cdef'])
    client = build_client(
        {
            'main': {'tokenizer': tokenizer, 'server_manager': endpoint},
            'subagent': message_model(MessageService('unused'), 'a3b'),
        },
        token_budget=6,
    )
    messages = [{'role': 'user', 'content': 'q'}]
    response = await client.create_completion(messages)
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    call_args = {'sampling_params': {'max_tokens': 5}} if explicit else {}
    response = await client.create_completion(messages, **call_args)
    assert endpoint.max_tokens_per_call == [6, 4], 'Default and explicit requests must honor the remaining budget'
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    with pytest.raises(ContextLengthExceededError):
        await client.create_completion(messages)
    assert len(endpoint.calls) == 2, 'Exhaustion must stop before a third backend call'


async def test_model_default_budget_still_deducts_prompt_growth() -> None:
    tokenizer = FakeTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['ab', 'cd'])
    client = build_client(
        {
            'main': {
                'tokenizer': tokenizer,
                'server_manager': endpoint,
                'max_model_len': 500,
            },
            'subagent': message_model(MessageService('unused'), 'a3b'),
        },
    )
    messages = [{'role': 'user', 'content': 'q'}]
    response = await client.create_completion(messages)
    messages += [response.choices[0].message.model_dump(exclude_none=True), {'role': 'user', 'content': 'next'}]
    await client.create_completion(messages)
    prompt_growth = len(endpoint.calls[1]['prompt_ids']) - len(endpoint.calls[0]['prompt_ids'])
    assert prompt_growth > 0, 'The followup must extend the prompt'
    assert endpoint.max_tokens_per_call == [100, 100 - prompt_growth], 'Policy merging must preserve context accounting'


@pytest.mark.parametrize(
    'components',
    [None, ['agent_output'], ['agent_output', 'tool_observation'], ['agent_output', 'compaction_output']],
)
async def test_shared_budget_uses_global_components_and_deduplicates_observations(
    components: list[str] | None,
) -> None:
    tokenizer, summary_tokenizer = FakeTokenizer(), OffsetTokenizer()
    main = RecordingEndpoint(tokenizer, ['AB', 'C', 'D'])
    summary = RecordingEndpoint(summary_tokenizer, ['SUM'])
    client = build_client(
        {
            'main': {
                'tokenizer': tokenizer,
                'server_manager': main,
                'token_budget_include': ['agent_output'],
            },
            'subagent': {
                'tokenizer': summary_tokenizer,
                'server_manager': summary,
                'token_budget_include': ['agent_output'],
            },
        },
        token_budget=500,
        token_budget_include=components,
        request_args={'sampling_params': {'max_tokens': 2000}},
    )
    messages = [{'role': 'user', 'content': 'q'}]
    response = await client.create_completion(messages)
    messages += [response.choices[0].message.model_dump(exclude_none=True), {'role': 'tool', 'content': 'obs'}]
    prompt_tokens = (await client.count_tokens(messages=messages)).input_tokens
    observation_tokens = prompt_tokens - len(main.calls[0]['prompt_ids']) - 2
    assert observation_tokens > 0, 'The tool continuation must add tokens'
    for _ in range(2):
        await client.update_token_consumption(messages)
    response = await client.create_completion(messages)
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    observation_charge = observation_tokens if components is None or 'tool_observation' in components else 0
    assert main.max_tokens_per_call == [500, 498 - observation_charge], 'Charge the tool once, before dispatch'
    await client.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
    assert summary.max_tokens_per_call == [497 - observation_charge], 'Summary must see the other model consumption'
    client.curr_segment_consumed_tokens = 0
    await client.create_completion(
        [{'role': 'user', 'content': 'summary and task'}], request_meta={'invocation_kind': 'compaction_resume'}
    )
    summary_charge = 3 if components is None or 'compaction_output' in components else 0
    assert main.max_tokens_per_call[-1] == 497 - observation_charge - summary_charge, (
        'Resuming after compaction must preserve the shared budget and global summary policy'
    )


async def test_message_model_usage_does_not_consume_shared_token_budget() -> None:
    class UsageService(MessageService):
        async def create_completion(self, **kwargs: Any) -> SimpleNamespace:
            response = await super().create_completion(**kwargs)
            response.usage = {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}
            return response

    tokenizer = FakeTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['AB', 'CDEF'])
    service = UsageService('summary')
    client = build_client(
        {
            'main': {'tokenizer': tokenizer, 'server_manager': endpoint},
            'subagent': message_model(service, 'a3b'),
        },
        token_budget=6,
    )
    messages = [{'role': 'user', 'content': 'q'}]
    response = await client.create_completion(messages)
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    await client.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
    response = await client.create_completion(messages)
    assert endpoint.max_tokens_per_call == [6, 4], 'Message usage is outside the existing token accounting contract'
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    with pytest.raises(ContextLengthExceededError):
        await client.create_completion(messages)
    await client.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
    assert len(service.calls) == 2, 'Message calls remain allowed after token budget exhaustion'


@pytest.mark.parametrize('cancel_pending', [False, True])
async def test_shared_budget_counts_completed_concurrent_requests_without_reservation(cancel_pending: bool) -> None:
    class BlockingEndpoint(RecordingEndpoint):
        def __init__(self, tokenizer: FakeTokenizer) -> None:
            super().__init__(tokenizer, ['ABCD', 'Z'])
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.requested_limits: list[int] = []

        async def generate_tokens(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            self.requested_limits.append(kwargs['sampling_params']['max_tokens'])
            self.started.set()
            await self.release.wait()
            return await super().generate_tokens(*args, **kwargs)

    tokenizer = FakeTokenizer()
    main, child = BlockingEndpoint(tokenizer), RecordingEndpoint(tokenizer, ['EFGH'])
    client = build_client(
        {
            'main': {'tokenizer': tokenizer, 'server_manager': main},
            'subagent': {'tokenizer': tokenizer, 'server_manager': child},
        },
        token_budget=6,
    )
    messages = [{'role': 'user', 'content': 'q'}]
    pending = asyncio.create_task(client.create_completion(messages))
    try:
        await asyncio.wait_for(main.started.wait(), 2)
        child_client = client.fork_for_agent(agent_id='child', parent_agent_id='main')
        await child_client.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
        if cancel_pending:
            pending.cancel()
        main.release.set()
        if cancel_pending:
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            await pending
    finally:
        main.release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    assert main.requested_limits == [6], 'The first request sees the initial budget'
    assert child.max_tokens_per_call == [6], 'In-flight requests do not reserve the budget'
    if cancel_pending:
        await client.create_completion(messages)
        assert main.requested_limits == [6, 2], 'Cancelled output must not consume the shared budget'
    else:
        with pytest.raises(ContextLengthExceededError):
            await client.create_completion(messages)
        assert main.requested_limits == [6], 'Both completed outputs must count before the next request'


def chain_meta(agent_id: str) -> dict[str, Any]:
    return {
        'agent_id': agent_id,
        'parent_agent_id': None if agent_id == 'main' else 'main',
        'declared_chain': {
            'chain_id': agent_id,
            'kind': 'main' if agent_id == 'main' else 'subagent',
            'parent_chain_id': None if agent_id == 'main' else 'main',
        },
        'iteration': 0,
        'retry': 0,
    }


async def test_concurrent_models_keep_routes_and_one_trajectory() -> None:
    main = MessageService('main-answer', blocked=True)
    child = MessageService('child-answer')
    client = build_client({'main': message_model(main, 'a5b'), 'subagent': message_model(child, 'a3b')})
    messages = [{'role': 'user', 'content': 'question'}]
    pending = asyncio.create_task(client.create_completion(messages, request_meta=chain_meta('main')))
    try:
        await asyncio.wait_for(main.started.wait(), 2)
        response = await client.create_completion(
            messages, request_meta={**chain_meta('child'), 'invocation_kind': 'summary_inference'}
        )
        assert response.choices[0].message.content == 'child-answer', 'Child used the wrong backend'
    finally:
        main.release.set()
        parent_response = await pending
    assert parent_response.choices[0].message.content == 'main-answer', 'Await changed the parent model'
    assert main.calls[0]['model'] == 'a5b', 'The selected endpoint must use its configured request model'
    assert child.calls[0]['model'] == 'a3b', 'Child must use its endpoint model'
    trajectory = client._session().get_trajectory()
    facts = {
        node.provider_facts.model_id: node.provider_facts
        for node in trajectory.iter_nodes()
        if node.generated_by == 'llm'
    }
    assert set(facts) == {'main', 'subagent'}, 'Both models must be recorded in one trajectory'
    assert facts['subagent'].model == 'same-server-model', 'Keep the actual server model independently'
    assert trajectory.chains['child'].parent_chain_id == 'main', 'Keep the declared parent relationship'


async def test_cancelled_main_request_does_not_cancel_child_model() -> None:
    main, child = MessageService('main', blocked=True), MessageService('child')
    client = build_client({'main': message_model(main, 'a5b'), 'subagent': message_model(child, 'a3b')})
    pending = asyncio.create_task(client.create_completion([{'role': 'user', 'content': 'main'}]))
    await asyncio.wait_for(main.started.wait(), 2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    child_client = client.fork_for_agent(agent_id='child', parent_agent_id='main')
    response = await child_client.create_completion(
        [{'role': 'user', 'content': 'child'}], request_meta={'invocation_kind': 'summary_inference'}
    )
    assert response.choices[0].message.content == 'child', 'Cancellation must not poison the shared session'
    session = client._session()
    await session.finish_pending_publications()
    assert session.get_trajectory().to_messages('child')[-1]['content'] == 'child', 'Keep the surviving model output'


async def test_child_generation_preserves_main_frozen_stream() -> None:
    tokenizer, child_tokenizer = FakeTokenizer(), OffsetTokenizer()
    main, child = RecordingEndpoint(tokenizer, ['parent', 'continued']), RecordingEndpoint(child_tokenizer, ['child'])
    client = build_client({
        'main': {'tokenizer': tokenizer, 'server_manager': main},
        'subagent': {
            'tokenizer': child_tokenizer,
            'server_manager': child,
        },
    })
    child_client = client.fork_for_agent(agent_id='child', parent_agent_id='main')
    messages = [{'role': 'user', 'content': 'q'}]
    response = await client.create_completion(messages)
    original_stream = main.calls[0]['prompt_ids'] + tokenizer.encode('parent')
    await child_client.create_completion(
        [{'role': 'user', 'content': 'subtask'}], request_meta={'invocation_kind': 'summary_inference'}
    )
    messages += [response.choices[0].message.model_dump(exclude_none=True), {'role': 'user', 'content': 'continue'}]
    counted = await client.count_tokens(messages=messages)
    await client.create_completion(messages)
    assert main.calls[1]['prompt_ids'][: len(original_stream)] == original_stream, (
        'Child must not replace the main prefix'
    )
    assert counted.input_tokens == len(main.calls[1]['prompt_ids']), 'Counting must reuse the main sampled stream'


async def test_summary_override_does_not_change_normal_route() -> None:
    main, child = MessageService('main-answer'), MessageService('summary')
    client = build_client({'main': message_model(main, 'a5b'), 'subagent': message_model(child, 'a3b')})
    messages = [{'role': 'user', 'content': 'q'}]
    summary = await client.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
    resumed = await client.create_completion(messages, request_meta={'invocation_kind': 'compaction_resume'})
    assert summary.choices[0].message.content == 'summary', 'Summary must use the override model'
    assert resumed.choices[0].message.content == 'main-answer', 'Resume must use the ordinary model'


@pytest.mark.parametrize('policy_name', ['claude_code', 'hycode'])
@pytest.mark.parametrize('token_level', [False, True])
async def test_policy_inferred_summary_routes_before_inference(
    policy_name: str, token_level: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if token_level:
        tokenizer, summary_tokenizer = FakeTokenizer(), OffsetTokenizer()
        main = RecordingEndpoint(tokenizer, ['continued'])
        summary = RecordingEndpoint(summary_tokenizer, ['summary'])
        models = {
            'main': {'tokenizer': tokenizer, 'server_manager': main},
            'subagent': {'tokenizer': summary_tokenizer, 'server_manager': summary},
        }
    else:
        main, summary = MessageService('continued'), MessageService('summary')
        models = {'main': message_model(main, 'a5b'), 'subagent': message_model(summary, 'a3b')}
    models['main']['request_args'] = {'sampling_params': {'temperature': 0.2}}
    models['subagent']['request_args'] = {'sampling_params': {'temperature': 0.7}}
    tools = [{'type': 'function', 'function': {'name': 'calculator', 'parameters': {'type': 'object'}}}]
    client = build_client(models, scaffold_policy=policy_name, token_budget=50, token_budget_include=['agent_output'])
    session = client._session()
    classify = Mock(wraps=session.segment_assembler.extract_context)
    monkeypatch.setattr(session.segment_assembler, 'extract_context', classify)
    resolve = client._resolve_request_kwargs
    resolved_models = []

    def resolve_after_classification(*args: Any, **kwargs: Any) -> Any:
        assert classify.call_count == len(resolved_models) + 1, 'Classify once before resolving request parameters'
        resolved_models.append(kwargs['binding'].model_id)
        return resolve(*args, **kwargs)

    monkeypatch.setattr(client, '_resolve_request_kwargs', resolve_after_classification)
    execute = session.create_completion

    async def execute_resolved_request(**kwargs: Any) -> Any:
        assert resolved_models, 'Select the model and resolve parameters before entering Session'
        assert kwargs['binding'] is client._model_registry.get(resolved_models[-1]), (
            'Session receives the selected binding'
        )
        return await execute(**kwargs)

    monkeypatch.setattr(session, 'create_completion', execute_resolved_request)
    messages = [{'role': 'user', 'content': COMPACT_TRIGGER_MARKER + '\nSummarize the completed work.'}]
    response = await client.create_completion(messages, tools=tools, temperature=0.9)

    assert classify.call_count == 1, 'Session must reuse the classification from the client'
    assert classify.call_args.kwargs['tools'] == tools, 'Classification uses the caller tools'
    assert not main.calls, 'Policy-inferred summaries must bypass the main endpoint'
    assert len(summary.calls) == 1, 'Summary inference must reach the mapped endpoint'
    assert response.choices[0].message.content == 'summary', 'Use the summary model response'
    sampling = summary.calls[0]['sampling_params'] if token_level else summary.calls[0]
    assert sampling['temperature'] == (0.7 if token_level else 0.9), 'Apply the selected model backend contract'
    if token_level:
        assert session._token_budget_state._token_consumed == 0, 'Policy summary classification also governs budget'

    history = messages + [response.choices[0].message.model_dump(exclude_none=True)]
    history.append({'role': 'user', 'content': 'Continue the original task.'})
    continued = await client.create_completion(history, tools=tools)
    assert continued.choices[0].message.content == 'continued', 'Historical markers must not reroute ordinary calls'
    assert len(main.calls) == len(summary.calls) == 1, 'Keep the invocation route request-local'
    assert classify.call_count == 2, 'Each completion is classified once'
    assert resolved_models == ['subagent', 'main'], 'Resolve each request using its already selected model'
    sampling = main.calls[0]['sampling_params'] if token_level else main.calls[0]
    assert sampling['temperature'] == 0.2, 'Resume with the default model sampling parameters'
    nodes = [node for node in session.get_trajectory().iter_nodes() if node.generated_by == 'llm']
    assert [node.provider_facts.model_id for node in nodes] == ['subagent', 'main'], 'Record the actual model IDs'
    assert nodes[0].segment_boundary.kind == 'summary_inference', 'Routing and segmentation must agree'


async def test_backend_mutation_does_not_change_model_request_defaults() -> None:
    class MutatingService(MessageService):
        async def create_completion(self, **kwargs: Any) -> SimpleNamespace:
            response = await super().create_completion(**kwargs)
            kwargs['extra_body']['options'].append('backend mutation')
            return response

    service = MutatingService('main')
    defaults = {'extra_body': {'options': ['configured']}}
    client = build_client({
        'main': {**message_model(service, 'a5b'), 'request_args': defaults},
        'subagent': message_model(MessageService('child'), 'a3b'),
    })
    defaults['extra_body']['options'].append('caller mutation')
    for _ in range(2):
        await client.create_completion([{'role': 'user', 'content': 'q'}])
    assert [call['extra_body']['options'] for call in service.calls] == [['configured'], ['configured']], (
        'Construction-time and per-request copies must protect model defaults'
    )


@pytest.mark.parametrize('same_tokenizer', [False, True])
async def test_model_change_rerenders_prefix_and_splits_segments(same_tokenizer: bool) -> None:
    main_tokenizer = FakeTokenizer()
    child_tokenizer = main_tokenizer if same_tokenizer else OffsetTokenizer()
    main = RecordingEndpoint(main_tokenizer, ['parent'])
    child = RecordingEndpoint(child_tokenizer, ['child'])
    client = build_client({
        'main': {'tokenizer': main_tokenizer, 'server_manager': main},
        'subagent': {
            'tokenizer': child_tokenizer,
            'server_manager': child,
        },
    })
    messages = [{'role': 'user', 'content': 'q'}]
    response = await client.create_completion(messages)
    messages += [response.choices[0].message.model_dump(exclude_none=True), {'role': 'user', 'content': 'continue'}]
    await client.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
    expected = child_tokenizer.encode(
        '<user>q</user>\n<assistant>parent</assistant>\n<user>continue</user>\n<assistant>'
    )
    assert child.calls[0]['prompt_ids'] == expected, 'Foreign model tokens must be rerendered'
    session = client._session()
    assert client.curr_segment_consumed_tokens == 6, 'A model switch must not combine segment counters'
    assert session._token_budget_state._token_consumed == 11, 'Both outputs must accumulate in the shared state'
    client.curr_segment_consumed_tokens = 0
    assert client.curr_segment_consumed_tokens == 0, 'Reset the default model counter'
    assert session._token_budget_state._token_consumed == 11, 'Segment reset must not change global consumption'
    await session.finish_pending_publications()
    segments = session.get_trajectory().to_segment_node_lists()
    assert len(segments) == 2, 'A model switch must split even identical token contracts'
    assert [{node.provider_facts.model_id for node in segment.response_nodes} for segment in segments] == [
        {'main'},
        {'subagent'},
    ], 'Each segment must have only one generating model'
    assert 'model_change' in segments[1].response_nodes[0].segment_boundary.causes, 'Record the model switch cause'


@pytest.mark.parametrize('subagent_model_id', [None, 'calculator'])
async def test_forks_use_subagent_model_and_share_session(subagent_model_id: str | None) -> None:
    main, worker, summary = MessageService('main-answer'), MessageService('child-answer'), MessageService('summary')
    client = build_client(
        {
            'main': message_model(main, 'Qwen3-32B'),
            'calculator': message_model(worker, 'Qwen3-8B'),
            'summary': message_model(summary, 'Qwen3-30B-A3B'),
        },
        subagent_model_id=subagent_model_id,
        compaction_model_id='summary',
    )
    child = client.fork_for_agent(agent_id='child', parent_agent_id='main')
    grandchild = child.fork_for_agent(agent_id='grandchild', parent_agent_id='child')
    messages = [{'role': 'user', 'content': 'q'}]
    for fork in (client, child, grandchild):
        response = await fork.create_completion(
            messages, request_meta=chain_meta('scaffold-child' if fork is client else 'main')
        )
        expected = 'child-answer' if fork is not client and subagent_model_id is not None else 'main-answer'
        assert response.choices[0].message.content == expected, 'Request identity must not override the client binding'
        assert fork._session() is client._session(), 'Forks share the task session'
    response = await child.create_completion(messages, request_meta={'invocation_kind': 'summary_inference'})
    assert response.choices[0].message.content == 'summary', 'Compaction takes precedence over subagent identity'
    response = await child.create_completion(messages, request_meta={'invocation_kind': 'compaction_resume'})
    expected = 'child-answer' if subagent_model_id is not None else 'main-answer'
    assert response.choices[0].message.content == expected, 'Resume with the configured subagent model'
    await client.create_completion(messages)
    models = [
        node.provider_facts.model_id
        for node in client._session().get_trajectory().iter_nodes()
        if node.generated_by == 'llm'
    ]
    child_model = subagent_model_id or 'main'
    assert models == ['main', child_model, child_model, 'summary', child_model, 'main'], (
        'Keep every route request-local'
    )
    assert 'declared_chain' not in client._session().get_trajectory().extra, 'Chain identity is request-local'


@pytest.mark.parametrize(
    ('override', 'enforced', 'expected'),
    [
        (None, {}, 'Qwen3-4B-Instruct'),
        ({'model': 'Qwen3-8B'}, {}, 'Qwen3-8B'),
        ({'model': 'Qwen3-8B'}, {'model': 'Qwen3-30B-A3B'}, 'Qwen3-30B-A3B'),
    ],
)
async def test_request_model_policy_does_not_change_model_id(
    override: dict[str, Any] | None, enforced: dict[str, Any], expected: str
) -> None:
    main, child = MessageService('main-answer'), MessageService('child-answer')
    receive = Mock(wraps=main.create_completion)
    main.create_completion = receive
    client = build_client(
        {'main': message_model(main, 'a5b'), 'subagent': message_model(child, 'a3b')},
        request_args={'model': 'Qwen3-4B-Instruct'},
    )
    client.config.enforced_request_args.update(enforced)
    response = await client.create_completion([{'role': 'user', 'content': 'q'}], _request_args_override=override)
    assert response.choices[0].message.content == 'main-answer', 'Request model parameters must not select a processor'
    assert receive.call_args.kwargs['model'] == expected, 'Apply model policy only when constructing the request'
    assert 'model_id' not in receive.call_args.kwargs, 'Routing IDs must not reach the backend'
    assert not child.calls, 'Request parameters must not change the selected endpoint'
    node = next(node for node in client._session().get_trajectory().iter_nodes() if node.generated_by == 'llm')
    assert node.provider_facts.model_id == 'main', 'Keep the selected model ID independent of request parameters'


@pytest.mark.parametrize('subagent_model_id', [None, 'subagent'])
async def test_fork_token_accounting_follows_subagent_model(subagent_model_id: str | None) -> None:
    tokenizer = FakeTokenizer()
    child_tokenizer = OffsetTokenizer()
    main = RecordingEndpoint(tokenizer, ['ab', 'cdef', 'z'] if subagent_model_id is None else ['ab', 'z'])
    child_endpoint = RecordingEndpoint(child_tokenizer, ['cdef'])
    client = build_client(
        {
            'main': {'tokenizer': tokenizer, 'server_manager': main, 'max_model_len': 400, 'priority': 3},
            'subagent': {
                'tokenizer': child_tokenizer,
                'server_manager': child_endpoint,
                'max_model_len': 800,
                'priority': 7,
            },
        },
        token_budget=10,
        subagent_model_id=subagent_model_id,
    )
    child = client.fork_for_agent(agent_id='child', parent_agent_id='main')
    assert child.tokenizer is (child_tokenizer if subagent_model_id is not None else tokenizer), (
        'Use the child tokenizer'
    )
    assert child.max_model_len == (800 if subagent_model_id is not None else 400), 'Use the child model capacity'
    messages = [{'role': 'user', 'content': 'q'}]
    response = await client.create_completion(messages)
    await child.create_completion(messages)
    assert main.max_tokens_per_call[0] == 10, 'Use the global budget'
    assert main.calls[0]['priority'] == 3, 'Use the main model default priority'
    child_call = child_endpoint.calls[0] if subagent_model_id is not None else main.calls[1]
    assert child_call['sampling_params']['max_tokens'] == 8, 'Share the remaining global budget'
    assert child_call['priority'] == (7 if subagent_model_id is not None else 3), 'Use the bound model default priority'
    assert client.curr_segment_consumed_tokens == 2, 'Keep the parent context counter'
    assert child.curr_segment_consumed_tokens == 4, 'Keep the child context counter'
    child.curr_segment_consumed_tokens = 0
    assert client.curr_segment_consumed_tokens == 2, 'Resetting the child must not reset the parent'
    await child.update_token_consumption(messages)
    counted = await child.count_tokens(messages=messages)
    assert counted.input_tokens == len(child_call['prompt_ids']), 'Counting uses the same model as child inference'
    messages.append(response.choices[0].message.model_dump(exclude_none=True))
    await client.create_completion(messages, priority=0)
    assert main.max_tokens_per_call[-1] == 4, 'A segment reset must not reset global consumption'
    assert main.calls[-1]['priority'] == 0, 'An explicit per-call priority overrides the model default'


async def test_token_fork_declares_distinct_direct_call_ordinals() -> None:
    tokenizer = FakeTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['first', 'second', 'third'])
    client = build_client({
        'main': {'tokenizer': tokenizer, 'server_manager': endpoint},
        'subagent': message_model(MessageService('summary'), 'a3b'),
    })
    child = client.fork_for_agent(agent_id='child', parent_agent_id='main')
    messages = [{'role': 'user', 'content': 'q'}]
    first = await child.create_completion(messages)
    messages += [first.choices[0].message.model_dump(exclude_none=True), {'role': 'user', 'content': 'next'}]
    second = await child.create_completion(messages, request_meta={'iteration': 7, 'retry': 2})
    messages += [second.choices[0].message.model_dump(exclude_none=True), {'role': 'user', 'content': 'last'}]
    await child.create_completion(messages)
    assert [call['request_id'] for call in endpoint.calls] == [
        'multi-test:child:0:0:completion',
        'multi-test:child:7:2:completion',
        'multi-test:child:8:0:completion',
    ], 'Direct fork calls need unique ordinals while harness ordinals remain unchanged'


async def test_fork_observation_keeps_child_context_after_parent_completes() -> None:
    client = build_client({
        'main': message_model(MessageService('main'), 'a5b'),
        'subagent': message_model(MessageService('child'), 'a3b'),
    })
    child = client.fork_for_agent(agent_id='child', parent_agent_id='main')
    messages = [{'role': 'user', 'content': 'child task'}]
    response = await child.create_completion(messages)
    parents = messages + [response.choices[0].message.model_dump(exclude_none=True)]
    await client.create_completion([{'role': 'user', 'content': 'main task'}])
    child.record_observation(content='child result', parent_messages=parents)
    trajectory = client._session().get_trajectory()
    assert trajectory.to_messages('child')[-1]['content'] == 'child result', (
        'Observation must belong to the child chain'
    )
    assert all(message.get('content') != 'child result' for message in trajectory.to_messages('main')), (
        'Latest active chain must not steal child observations'
    )


async def test_observation_identity_ignores_task_metadata_for_routing() -> None:
    client = GatewayOpenAIClient(
        OpenAIConfig(
            request_args={},
            client_args={
                'trajectory_id': 'legacy-observation',
                'scaffold_policy': 'nexus_main_agent',
                'openai_client': MessageService('child'),
                'metadata': {'validate': True},
            },
        )
    )
    completion = await client.create_completion(
        [{'role': 'user', 'content': 'child task'}], request_meta=chain_meta('child')
    )
    history = [{'role': 'user', 'content': 'child task'}, completion.choices[0].message.model_dump(exclude_none=True)]
    client.record_observation(content='tool result', parent_messages=history, request_meta=chain_meta('child'))
    assert client._session().get_trajectory().to_messages('child')[-1]['content'] == 'tool result', (
        'An unforked client places the observation on the chain its request_meta names'
    )


@pytest.mark.parametrize('with_registry', [False, True])
async def test_downstream_model_is_a_backend_request_parameter(with_registry: bool) -> None:
    main = MessageService('main')
    main.config.request_args['model'] = 'Qwen3-4B-Instruct'
    receive = Mock(wraps=main.create_completion)
    main.create_completion = receive
    if with_registry:
        client = build_client({
            'main': {'openai_client': main},
            'subagent': message_model(MessageService('child'), 'a3b'),
        })
    else:
        client = GatewayOpenAIClient(
            OpenAIConfig(
                client_args={
                    'trajectory_id': 'single-model-pin',
                    'scaffold_policy': 'nexus_main_agent',
                    'openai_client': main,
                },
                request_args={},
            )
        )
    await client.create_completion([{'role': 'user', 'content': 'q'}], model='caller-model')
    assert receive.call_args.kwargs.get('model') == 'caller-model', 'Forward the downstream model, not the model ID'
    assert 'model_id' not in receive.call_args.kwargs, 'Consume routing IDs before the backend call'
    assert main.calls[0]['model'] == 'Qwen3-4B-Instruct', 'The backend owns its request model'
    if not with_registry:
        node = next(node for node in client._session().get_trajectory().iter_nodes() if node.generated_by == 'llm')
        assert node.provider_facts.model_id is None, 'Legacy trajectories have no registered model ID'


async def test_extra_body_model_does_not_change_registered_model_route() -> None:
    main, child = MessageService('main'), MessageService('child')
    client = build_client({'main': message_model(main, 'a5b'), 'subagent': message_model(child, 'a3b')})
    await client.create_completion(
        [{'role': 'user', 'content': 'q'}],
        request_meta={'invocation_kind': 'summary_inference'},
        extra_body={'model': 'a5b'},
    )
    assert not main.calls, 'Wire parameters must not change the selected endpoint'
    assert child.calls[0]['extra_body']['model'] == 'a5b', 'Forward the backend request parameter unchanged'
    node = next(node for node in client._session().get_trajectory().iter_nodes() if node.generated_by == 'llm')
    assert node.provider_facts.model_id == 'subagent', 'Record the registry ID, not the backend model parameter'


async def test_responses_proxy_routes_compaction_before_inference() -> None:
    main, child = MessageService('main-answer'), MessageService('child-answer')
    client = build_client({'main': message_model(main, 'a5b'), 'subagent': message_model(child, 'a3b')})
    handler = LLMRequestHandler(runtime=object(), openai_client=client)

    async def forward(data: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        return await handler._produce_response('proxy-request', data, rsio_mode=False)

    proxy = ResponsesProxy(sio=object(), codex_version='0.147.0')
    proxy.bridge = SimpleNamespace(request=forward)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy._create_app()), base_url='http://proxy'
    ) as connection:
        for model, headers, expected in [
            ('subagent', {}, 'main-answer'),
            ('main', {CODEX_TURN_METADATA_HEADER: json.dumps({'request_kind': 'compaction'})}, 'child-answer'),
            ('main', {}, 'main-answer'),
        ]:
            response = await connection.post(
                '/v1/responses',
                headers=headers,
                json={
                    'model': model,
                    'input': [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'task'}]}],
                },
            )
            assert response.status_code == 200, response.text
            assert expected in response.text, 'Responses output must come from the selected model'
    nodes = [node for node in client._session().get_trajectory().iter_nodes() if node.generated_by == 'llm']
    assert [node.provider_facts.model_id for node in nodes] == ['main', 'subagent', 'main'], (
        'Do not persist the summary override'
    )
    assert nodes[1].segment_boundary.kind == 'summary_inference', 'Classify the summary before recording'
