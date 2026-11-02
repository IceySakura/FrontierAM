"""Paired Legacy/Gateway accounting through actual token-level client requests."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from _fakes import FakeTokenizer, RecordingEndpoint, StubToolParser
from test_token_budget_parity import _NoGlueTokenizer

from nexus.core import ContextLengthExceededError, OpenAIConfig
from nexus.extensions.openai_clients.hytuner import HyTunerClient
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.session_manager import session_manager


ALL_COMPONENTS = ['agent_output', 'tool_observation', 'compaction_output']
QUESTION = [{'role': 'user', 'content': 'q'}]
OBSERVATION = {'role': 'tool', 'content': 'obs'}


@pytest.fixture(autouse=True)
async def isolated_sessions(monkeypatch):
    monkeypatch.setenv('GATEWAY_LEGACY_SEAM_COMPAT', '1')
    session_manager.end_all()
    yield
    session = session_manager.get_session('observation-budget')
    if session is not None:
        await session.finish_pending_publications()
    session_manager.end_all()


def build_pair(*, budget=1000, components=None, scripted=None, tokenizer_type=_NoGlueTokenizer):
    clients = []
    for client_type in (HyTunerClient, GatewayOpenAIClient):
        tokenizer = tokenizer_type()
        endpoint = RecordingEndpoint(tokenizer, list(scripted or ['AB', 'CD', 'EF', 'GH']))
        client = client_type(
            OpenAIConfig(
                client_args={
                    'tokenizer': tokenizer,
                    'tool_parser': StubToolParser(tokenizer),
                    'server_manager': endpoint,
                    'trajectory_id': 'observation-budget',
                    'scaffold_policy': 'generic',
                    'max_model_len': 10000,
                    'token_budget': budget,
                    'token_budget_include': ALL_COMPONENTS if components is None else components,
                },
                request_args={'sampling_params': {'max_tokens': 2000, 'temperature': 1.0}},
            )
        )
        clients.append((client, endpoint))
    return clients


def consumed(client):
    if isinstance(client, HyTunerClient):
        return client.token_consumed, client.curr_segment_consumed_tokens
    state = client._session()._token_budget_state
    return state._token_consumed, sum(state._segment_consumed_tokens.values())


async def complete(client, messages, **kwargs):
    result = await client.create_completion(copy.deepcopy(messages), **kwargs)
    return [*messages, result.choices[0].message.model_dump(exclude_none=True)]


@pytest.mark.parametrize('explicit_updates', [0, 1, 2])
@pytest.mark.parametrize(
    'components',
    [
        ALL_COMPONENTS,
        ['agent_output'],
        ['agent_output', 'tool_observation'],
        ['agent_output', 'compaction_output'],
    ],
)
async def test_native_tool_append_and_manager_updates_have_identical_budget(explicit_updates, components):
    results = []
    for client, endpoint in build_pair(components=components):
        messages = await complete(client, QUESTION)
        messages.append(dict(OBSERVATION))
        original = copy.deepcopy(messages)
        for _ in range(explicit_updates):
            await client.update_token_consumption(copy.deepcopy(messages))
        await complete(client, messages)
        assert messages == original
        results.append((endpoint.max_tokens_per_call, consumed(client), endpoint.calls[-1]['prompt_ids']))

    assert results[0] == results[1]
    # The exact rendered observation includes the tool and next-assistant headers.
    observation_tokens = len('<tool>obs<assistant>')
    assert results[0][1] == (
        (4 if 'agent_output' in components else 0) + (observation_tokens if 'tool_observation' in components else 0),
        4 + observation_tokens,
    )


async def test_observation_budget_exhausts_before_the_same_dispatch_and_deduplicates_rejection():
    results = []
    for client, endpoint in build_pair(budget=30):
        messages = await complete(client, QUESTION)
        messages.append(dict(OBSERVATION))
        messages = await complete(client, messages)
        messages.append(dict(OBSERVATION))
        with pytest.raises(ContextLengthExceededError):
            await complete(client, messages)
        after_rejection = consumed(client)
        with pytest.raises(ContextLengthExceededError):
            await complete(client, messages)
        assert consumed(client) == after_rejection
        results.append((endpoint.max_tokens_per_call, consumed(client)))

    assert results[0] == results[1] == ([30, 8], (44, 44))


async def test_legacy_seam_counts_rendered_template_tokens_as_well_as_tool_content():
    results = []
    for client, endpoint in build_pair(tokenizer_type=FakeTokenizer):
        messages = [*(await complete(client, QUESTION)), dict(OBSERVATION)]
        await complete(client, messages)
        suffix_tokens = len(endpoint.calls[1]['prompt_ids']) - len(endpoint.calls[0]['prompt_ids']) - 2
        results.append((endpoint.max_tokens_per_call, consumed(client), suffix_tokens, endpoint.calls[1]['prompt_ids']))
    assert results[0] == results[1]
    assert results[0][2] == len('<tool>obs</tool>\n<assistant>')
    assert results[0][1] == (4 + results[0][2], 4 + results[0][2])


@pytest.mark.parametrize('explicit_update', [False, True])
async def test_initial_history_with_tool_results_is_not_consumed(explicit_update):
    history = [*QUESTION, {'role': 'assistant', 'content': 'old call'}, dict(OBSERVATION)]
    results = []
    for client, endpoint in build_pair():
        if explicit_update:
            await client.update_token_consumption(copy.deepcopy(history))
        await complete(client, history)
        results.append((endpoint.max_tokens_per_call, consumed(client)))
    assert results[0] == results[1] == ([1000], (2, 2))


@pytest.mark.parametrize(
    'suffix',
    [
        [{'role': 'user', 'content': 'notification'}],
        [dict(OBSERVATION), {'role': 'user', 'content': 'notification'}],
    ],
)
async def test_user_and_mixed_appends_are_not_tool_observations(suffix):
    results = []
    for client, endpoint in build_pair():
        messages = [*(await complete(client, QUESTION)), *copy.deepcopy(suffix)]
        await complete(client, messages)
        results.append((endpoint.max_tokens_per_call, consumed(client)))
    assert results[0] == results[1]
    assert results[0][1] == (4, 4)


@pytest.mark.parametrize('changed_message', [0, 1])
async def test_rewritten_history_with_tool_tail_does_not_charge_replayed_input(changed_message):
    results = []
    for client, endpoint in build_pair():
        messages = await complete(client, QUESTION)
        messages[changed_message] = {**messages[changed_message], 'content': 'replacement prompt'}
        messages.append(dict(OBSERVATION))
        await complete(client, messages)
        results.append((endpoint.max_tokens_per_call, consumed(client)))
    assert results[0] == results[1] == ([1000, 998], (4, 4))


@pytest.mark.parametrize('invocation_kind', ['summary_inference', 'compaction_resume'])
async def test_gateway_compaction_invocations_do_not_charge_an_appended_tool(invocation_kind):
    # This generic invocation contract is independent of scaffold-specific Legacy
    # prefix invalidation, which is outside the tool-observation accounting change.
    client, endpoint = build_pair()[1]
    messages = [*(await complete(client, QUESTION)), dict(OBSERVATION)]
    await complete(client, messages, request_meta={'invocation_kind': invocation_kind})
    assert (endpoint.max_tokens_per_call, consumed(client)) == ([1000, 998], (4, 4))


@pytest.mark.parametrize('components', [ALL_COMPONENTS, ['agent_output', 'tool_observation']])
async def test_compaction_preserves_aggregate_budget_and_charges_new_window_tools(components):
    results = []
    for client, endpoint in build_pair(components=components):
        messages = [*(await complete(client, QUESTION)), dict(OBSERVATION)]
        messages = await complete(client, messages)
        await complete(
            client,
            [*messages, {'role': 'user', 'content': 'summarize'}],
            tools=[],
            request_meta={'invocation_kind': 'summary_inference'},
        )
        resumed = await complete(
            client,
            [{'role': 'user', 'content': 'summary and task'}],
            request_meta={'invocation_kind': 'compaction_resume'},
        )
        resumed.append(dict(OBSERVATION))
        # Reuse the endpoint with one additional deterministic completion.
        endpoint._queue.append('IJ')
        await complete(client, resumed)
        results.append((endpoint.max_tokens_per_call, consumed(client)))
    assert results[0] == results[1]
    assert results[0][0] == (
        [1000, 978, 976, 974, 952] if 'compaction_output' in components else [1000, 978, 976, 976, 954]
    )


@pytest.mark.parametrize('manager_update_before_retry', [False, True])
async def test_backend_failure_retries_same_tool_history_without_double_charging(manager_update_before_retry):
    results = []
    for client, endpoint in build_pair():
        messages = [*(await complete(client, QUESTION)), dict(OBSERVATION)]
        original_generate = endpoint.generate
        if isinstance(client, GatewayOpenAIClient):
            state = client._session()._token_budget_state
            successful_prompt = copy.deepcopy(state._accounted_prompt_ids)
            initial_prompt_lens = dict(state._initial_prompt_lens)

        async def fail_generation(**_kwargs):
            raise RuntimeError('injected backend failure')

        endpoint.generate = endpoint.generate_tokens = fail_generation
        with pytest.raises(RuntimeError, match='injected backend failure'):
            await complete(client, messages)
        after_failure = consumed(client)
        if isinstance(client, GatewayOpenAIClient):
            assert state._accounted_prompt_ids.keys() == successful_prompt.keys()
            for context_key, prompt_ids in successful_prompt.items():
                np.testing.assert_array_equal(state._accounted_prompt_ids[context_key], prompt_ids)
            assert state._initial_prompt_lens == initial_prompt_lens
        if manager_update_before_retry:
            await client.update_token_consumption(copy.deepcopy(messages))
            assert consumed(client) == after_failure
        endpoint.generate = endpoint.generate_tokens = original_generate
        await complete(client, messages)
        results.append((after_failure, endpoint.max_tokens_per_call, consumed(client)))
    assert results[0] == results[1] == ((22, 22), [1000, 978], (24, 24))
