"""Gateway per-turn token-budget contract (TokenBudgetState clamp semantics).

History (2026-07-21, Trajectory-contract merge): this file used to A/B the gateway
budget against legacy ``HyTunerClient`` turn by turn. After the merge the two paths
tokenize differently — legacy does incremental tokenize (reuses the previous turn's
REAL generated token ids, closing glue included), while the gateway PromptRender
re-renders from messages and owns ``message_token_ids``. Under a fake char-level
template those two accountings of the assistant closing glue (``</assistant>\\n``)
diverge by construction, so byte-for-byte legacy parity is no longer a meaningful
assertion (user decision: accept glue semantics, real E2E is the end-to-end check).

The gateway takes the minimum of every configured generation limit:

1. ``max_tokens``, with prompt growth deducted for a default segment budget;
2. the selected model's remaining context space, ``max_model_len - len(prompt_ids)``;
3. the remaining trajectory ``token_budget``, including observations charged before dispatch.

An explicit per-call ``max_tokens`` bounds only its own request and cannot widen the
configured cap. Every available model and trajectory limit applies even when
``max_tokens`` is absent or explicit.

Exhaustion raises ``ContextLengthExceededError``.
"""

from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from _fakes import RecordingEndpoint, memory_store

from nexus.core import ContextLengthExceededError
from nexus.core.openai_client import OpenAIConfig
from nexus.gateway.client import GatewayOpenAIClient
from nexus.gateway.policies.hycode import COMPACT_SUMMARY_MARKER, COMPACT_TRIGGER_MARKER
from nexus.gateway.session_manager import session_manager
from nexus.gateway.token_budget import TokenBudgetState
from nexus.utils.token_budget_config import TokenBudgetComponent


BASE_SAMPLING = {'max_tokens': 100, 'temperature': 1.0}
USER_MESSAGE = {'role': 'user', 'content': 'What is 1+1?'}
TOOL_MESSAGE = {'role': 'tool', 'content': 'observation: 2'}


class _NoGlueTokenizer:
    """Char-level tokenizer whose template adds NO per-message closing glue.

    The gateway PromptRender re-renders history from messages; a template that
    appended assistant-closing glue would double-count it against a completion
    that already carries its own glue (see module docstring). A glue-free template
    keeps the prompt growth deterministic so the budget formula is what's tested.
    """

    eos_token = '§'

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        del tokenize, tools, kwargs
        parts = [f'<{m.get("role")}>{m.get("content") or ""}' for m in messages]
        if add_generation_prompt:
            parts.append('<assistant>')
        return ''.join(parts)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return ''.join(chr(i) for i in token_ids)


class _InterruptibleEndpoint(RecordingEndpoint):
    def __init__(self, tokenizer: _NoGlueTokenizer, scripted: list[str]):
        super().__init__(tokenizer, scripted)
        self.failure: BaseException | None = None
        self.attempted_limits: list[int | None] = []

    async def generate_tokens(self, *args: Any, **kwargs: Any) -> Any:
        self.attempted_limits.append(kwargs['sampling_params'].get('max_tokens'))
        if self.failure is not None:
            failure = self.failure
            self.failure = None
            raise failure
        return await super().generate_tokens(*args, **kwargs)


@pytest.fixture(autouse=True)
async def _clean_singleton() -> AsyncIterator[None]:
    """The session_manager is a process-wide singleton — isolate tests."""
    session_manager.end_all()
    yield
    session = session_manager.get_session('traj-budget')
    if session is not None:
        await session.finish_pending_publications()
    session_manager.end_all()


def _build_gateway_client(
    tokenizer: _NoGlueTokenizer,
    endpoint: RecordingEndpoint,
    *,
    max_model_len: int | None = None,
    token_budget: int | None = None,
    token_budget_include: list[str] | None = None,
    sampling_params: dict | None = None,
    scaffold_policy: str = 'generic',
) -> GatewayOpenAIClient:
    # max_model_len / token_budget live at the client_args top level exactly as
    # _run_gateway passes them; the client's _session() folds them into the
    # session's token_budget_config metadata.
    return GatewayOpenAIClient(
        OpenAIConfig(
            client_args={
                'tokenizer': tokenizer,
                'server_manager': endpoint,
                'trajectory_id': 'traj-budget',
                'record_store': memory_store('traj-budget'),
                'scaffold_policy': scaffold_policy,
                'metadata': {'trajectory_id': 'traj-budget'},
                'max_model_len': max_model_len,
                'token_budget': token_budget,
                'token_budget_include': token_budget_include,
            },
            request_args={'sampling_params': dict(sampling_params if sampling_params is not None else BASE_SAMPLING)},
        )
    )


async def _drive_turns(client: Any, num_turns: int, per_turn_kwargs: list[dict] | None = None) -> None:
    """Run a user -> assistant -> tool -> assistant ... flow for ``num_turns`` turns."""
    messages: list[dict] = [dict(USER_MESSAGE)]
    for turn_index in range(num_turns):
        kwargs = dict((per_turn_kwargs or [{}] * num_turns)[turn_index])
        completion = await client.create_completion(messages=list(messages), **kwargs)
        assistant_message = completion.choices[0].message.model_dump(exclude_none=True)
        messages = messages + [assistant_message, dict(TOOL_MESSAGE)]


SCRIPTED_TEXTS = ['A' * 10, 'B' * 8, 'C' * 5]


@pytest.mark.parametrize(
    ('components', 'expected_limits'),
    [(['agent_output'], [100, 100, 98]), (None, [100, 97, 95])],
)
async def test_hycode_summary_and_resume_use_distinct_output_budget_components(components, expected_limits):
    """A marker-only summary must honor component selection, unlike its ordinary continuation."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['SUM', 'OK', 'X'])
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=100,
        token_budget_include=components,
        sampling_params={'max_tokens': 2000},
        scaffold_policy='hycode',
    )

    await client.create_completion([{'role': 'user', 'content': COMPACT_TRIGGER_MARKER}])
    await client.create_completion([{'role': 'user', 'content': COMPACT_SUMMARY_MARKER + '. Continue.'}])
    await client.create_completion([{'role': 'user', 'content': 'A new context'}])

    assert endpoint.max_tokens_per_call == expected_limits
    trajectory = client._session().segment_assembler.freeze()
    generated = [node for node in trajectory.nodes.values() if node.generated_by == 'llm']
    assert generated[0].segment_boundary.kind == 'summary_inference'
    assert generated[1].segment_boundary.kind == 'compaction_resume'
    assert client._session()._token_budget_state.segment_consumed_tokens(('hycode', 'main', False)) == 6


@pytest.mark.parametrize('include_observations', [False, True])
async def test_hycode_summary_control_suffix_is_not_a_tool_observation(include_observations):
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['AA', 'SUM', 'OK'])
    components = ['agent_output', 'tool_observation'] if include_observations else ['agent_output']
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=300,
        token_budget_include=components,
        sampling_params={'max_tokens': 2000},
        scaffold_policy='hycode',
    )
    messages = [{'role': 'user', 'content': 'Work'}]
    first = await client.create_completion(messages)
    summary_messages = messages + [
        first.choices[0].message.model_dump(exclude_none=True),
        {'role': 'tool', 'content': 'O'},
        {'role': 'user', 'content': COMPACT_TRIGGER_MARKER},
    ]
    await client.create_completion(summary_messages)
    await client.create_completion([{'role': 'user', 'content': 'Compacted history'}])

    # Match Legacy's role-aware accounting: mixed tool/control input is not a
    # tool-only continuation, and summary output is excluded independently.
    assert endpoint.max_tokens_per_call == [300, 298, 298]


async def test_hycode_child_does_not_inherit_parent_summary_output_classification():
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['SUM', 'OK', 'X'])
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=100,
        token_budget_include=['agent_output'],
        sampling_params={'max_tokens': 2000},
        scaffold_policy='hycode',
    )
    trigger = {'role': 'user', 'content': COMPACT_TRIGGER_MARKER}
    summary = await client.create_completion([trigger])
    await client.create_completion(
        [
            trigger,
            summary.choices[0].message.model_dump(exclude_none=True),
            {'role': 'user', 'content': 'Child task'},
        ],
        request_meta={'agent_id': 'child-1'},
    )
    await client.create_completion([{'role': 'user', 'content': 'Main task resumes'}])

    assert endpoint.max_tokens_per_call == [100, 100, 98]
    trajectory = client._session().segment_assembler.freeze()
    child_output = next(node for node in trajectory.nodes.values() if node.original_message.get('content') == 'OK')
    assert child_output.segment_boundary is None


async def test_failed_hycode_summary_is_retryable_without_committing_consumption():
    tokenizer = _NoGlueTokenizer()
    endpoint = _InterruptibleEndpoint(tokenizer, ['SUM', 'OK'])
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=100,
        token_budget_include=['agent_output'],
        sampling_params={'max_tokens': 2000},
        scaffold_policy='hycode',
    )
    messages = [{'role': 'user', 'content': COMPACT_TRIGGER_MARKER}]
    endpoint.failure = RuntimeError('summary backend failure')
    with pytest.raises(RuntimeError, match='summary backend failure'):
        await client.create_completion(messages)
    assert client._session().segment_assembler.freeze().nodes == {}

    await client.create_completion(messages)
    await client.create_completion([{'role': 'user', 'content': 'Continue'}])

    assert endpoint.attempted_limits == [100, 100, 100]
    assert client._session()._token_budget_state.segment_consumed_tokens(('hycode', 'main', False)) == 5


async def test_base_budget_is_response_length_minus_used():
    """max_model_len/token_budget both unset: turn 1 gets the full response_length;
    later turns shrink by tokens already generated (used_tokens)."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, list(SCRIPTED_TEXTS))
    client = _build_gateway_client(tokenizer, endpoint)

    await _drive_turns(client, 3)

    assert endpoint.max_tokens_per_call[0] == BASE_SAMPLING['max_tokens']
    # Turn 2/3 shrink monotonically as prior generations consume the budget.
    assert endpoint.max_tokens_per_call[1] < BASE_SAMPLING['max_tokens']
    assert endpoint.max_tokens_per_call[2] < endpoint.max_tokens_per_call[1]


async def test_max_model_len_preserves_smaller_response_budget():
    """Legacy response_length remains the cap when context space is larger."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, list(SCRIPTED_TEXTS))
    client = _build_gateway_client(tokenizer, endpoint, max_model_len=400)

    await _drive_turns(client, 1)

    assert endpoint.max_tokens_per_call[0] == BASE_SAMPLING['max_tokens']


async def test_max_model_len_caps_larger_response_budget():
    """Legacy response_length is reduced when the context has less space."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, list(SCRIPTED_TEXTS))
    client = _build_gateway_client(tokenizer, endpoint, max_model_len=80)

    await _drive_turns(client, 1)

    initial_prompt_len = len(endpoint.calls[0]['prompt_ids'])
    assert endpoint.max_tokens_per_call[0] == 80 - initial_prompt_len


async def test_max_model_len_supplies_unified_budget_without_response_limit():
    """Unified mode derives max_tokens from the remaining context space."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, list(SCRIPTED_TEXTS))
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        max_model_len=400,
        sampling_params={'temperature': 1.0},
    )

    await _drive_turns(client, 1)

    initial_prompt_len = len(endpoint.calls[0]['prompt_ids'])
    assert endpoint.max_tokens_per_call[0] == 400 - initial_prompt_len


async def test_token_budget_clamps_first_turn():
    """token_budget set: turn 1's budget is clamped to token_budget."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, list(SCRIPTED_TEXTS))
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=30,
        token_budget_include=['agent_output'],
    )

    await _drive_turns(client, 3)

    assert endpoint.max_tokens_per_call[0] == 30


def test_compaction_output_can_be_excluded_from_budget():
    state = TokenBudgetState(
        token_budget=20,
        token_budget_include=['agent_output', 'tool_observation'],
    )
    state.record_generation(
        'summary',
        raw_prompt_ids=[1, 2],
        prompt_ids=[1, 2],
        generated_ids=[3, 4, 5, 6],
        output_component=TokenBudgetComponent.COMPACTION_OUTPUT,
    )

    request = state.prepare_generation(
        'summary',
        raw_prompt_ids=[1, 2, 3, 4, 5, 6],
        prompt_ids=[1, 2, 3, 4, 5, 6],
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
    )

    assert request['max_tokens'] == 20
    assert state.segment_consumed_tokens('summary') == 4


@pytest.mark.parametrize('explicit_prompt_update', [False, True])
def test_external_observation_accounting_survives_prompt_rewrite(explicit_prompt_update: bool):
    """Classified tool accounting and explicit prompt updates survive a prompt rewrite."""
    state = TokenBudgetState(
        token_budget=20,
        token_budget_include=['agent_output', 'tool_observation'],
    )
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])

    # The next request extends the sampled stream with a three-token external observation.
    extended_prompt = [1, 2, 3, 4, 5, 6, 7]
    if explicit_prompt_update:
        state.update_prompt('main', raw_prompt_ids=extended_prompt, prompt_ids=extended_prompt)
    else:
        state.account_tool_observation_suffix('main', extended_prompt, [TOOL_MESSAGE], 3)
    extended_request = state.prepare_generation(
        'main',
        raw_prompt_ids=extended_prompt,
        prompt_ids=extended_prompt,
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
    )
    # 20 - two agent tokens - three already-accounted observation tokens.
    assert extended_request['max_tokens'] == 15
    state.record_generation('main', raw_prompt_ids=extended_prompt, prompt_ids=extended_prompt, generated_ids=[8])

    # Compaction replaces the conversation prompt. Its output is excluded from
    # this budget, while the external observation must remain charged.
    state.record_generation(
        'main',
        raw_prompt_ids=[20, 21],
        prompt_ids=[20, 21],
        generated_ids=[22, 23],
        output_component=TokenBudgetComponent.COMPACTION_OUTPUT,
    )
    request = state.prepare_generation(
        'main',
        raw_prompt_ids=[20, 21, 22, 23],
        prompt_ids=[20, 21, 22, 23],
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
    )

    # 20 - two agent tokens - three observation tokens - one agent token.
    assert request['max_tokens'] == 14


async def test_gateway_commits_tool_observation_before_prompt_rewrite():
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['A' * 10, 'B' * 8, 'summary'])
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=100,
        token_budget_include=['agent_output', 'tool_observation'],
    )

    messages = [dict(USER_MESSAGE)]
    first = await client.create_completion(messages=list(messages))
    messages += [first.choices[0].message.model_dump(exclude_none=True), dict(TOOL_MESSAGE)]
    await client.create_completion(messages=list(messages))

    first_output_ids = tokenizer.encode('A' * 10)
    second_output_ids = tokenizer.encode('B' * 8)
    committed_prefix = endpoint.calls[0]['prompt_ids'] + first_output_ids
    second_prompt = endpoint.calls[1]['prompt_ids']
    assert second_prompt[: len(committed_prefix)] == committed_prefix
    tool_observation_ids = second_prompt[len(committed_prefix) :]

    await client.create_completion(
        [{'role': 'user', 'content': 'compacted context'}],
        request_meta={'invocation_kind': 'summary_inference'},
    )

    expected_remaining = 100 - len(first_output_ids) - len(tool_observation_ids) - len(second_output_ids)
    assert endpoint.max_tokens_per_call[2] == expected_remaining


async def test_gateway_user_notification_is_not_a_tool_observation():
    """Match Legacy: a user notification grows the context without a tool charge."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['A' * 10, 'B' * 8])
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=80,
        token_budget_include=['agent_output', 'tool_observation'],
        sampling_params={'max_tokens': 2000},
    )

    messages = [dict(USER_MESSAGE)]
    first = await client.create_completion(messages=list(messages))
    first_output_ids = tokenizer.encode('A' * 10)
    committed_prefix = endpoint.calls[0]['prompt_ids'] + first_output_ids
    notified = messages + [
        first.choices[0].message.model_dump(exclude_none=True),
        {'role': 'user', 'content': '<task-notification>done</task-notification>'},
    ]

    await client.create_completion(messages=notified)

    second_prompt = endpoint.calls[1]['prompt_ids']
    assert second_prompt[: len(committed_prefix)] == committed_prefix
    external_observation_ids = second_prompt[len(committed_prefix) :]
    assert external_observation_ids
    assert endpoint.max_tokens_per_call[1] == 80 - len(first_output_ids)


@pytest.mark.parametrize(
    ('token_budget_include', 'pending_remaining', 'committed_remaining'),
    [
        (['agent_output', 'tool_observation'], 15, 14),
        (['agent_output'], 18, 17),
    ],
)
def test_charged_observations_do_not_commit_prompt_or_charge_retries_twice(
    token_budget_include: list[str],
    pending_remaining: int,
    committed_remaining: int,
):
    """Retries must not charge observations twice or ignore the configured components."""
    state = TokenBudgetState(token_budget=20, token_budget_include=token_budget_include)
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])
    committed_prefix = [1, 2, 3, 4]
    extended_prompt = committed_prefix + [5, 6, 7]

    for _ in range(2):
        state.account_tool_observation_suffix('main', extended_prompt, [TOOL_MESSAGE], 3)
        request = state.prepare_generation(
            'main',
            raw_prompt_ids=extended_prompt,
            prompt_ids=extended_prompt,
            request_kwargs={'max_tokens': 100},
            uses_default_max_tokens=True,
        )
        assert request['max_tokens'] == pending_remaining
        assert state.segment_consumed_tokens('main') == 5
        np.testing.assert_array_equal(state._accounted_prompt_ids['main'], committed_prefix)

    unchanged = state.prepare_generation(
        'main',
        raw_prompt_ids=committed_prefix,
        prompt_ids=committed_prefix,
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
    )
    assert unchanged['max_tokens'] == pending_remaining

    state.record_generation('main', raw_prompt_ids=extended_prompt, prompt_ids=extended_prompt, generated_ids=[8])
    next_prompt = extended_prompt + [8]
    request = state.prepare_generation(
        'main',
        raw_prompt_ids=next_prompt,
        prompt_ids=next_prompt,
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
    )
    assert request['max_tokens'] == committed_remaining
    assert state.segment_consumed_tokens('main') == 6


@pytest.mark.parametrize(
    ('candidate', 'expected_remaining'),
    [
        ([1, 2, 3, 4], 18),
        ([1, 2, 3, 4, 5, 6, 7], 18),
        ([9, 10], 18),
        ([1, 2], 18),
        ([], 18),
    ],
)
def test_prompt_preview_classification_preserves_committed_state(candidate: list[int], expected_remaining: int):
    state = TokenBudgetState(token_budget=20)
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])
    kwargs = {'max_tokens': 100, 'temperature': 0.5}

    for _ in range(2):
        prepared = state.prepare_generation(
            'main',
            raw_prompt_ids=candidate,
            prompt_ids=candidate,
            request_kwargs=kwargs,
            uses_default_max_tokens=True,
        )
        assert prepared['max_tokens'] == expected_remaining
        assert state.segment_consumed_tokens('main') == 2

    assert kwargs == {'max_tokens': 100, 'temperature': 0.5}
    committed = [1, 2, 3, 4]
    unchanged = state.prepare_generation(
        'main',
        raw_prompt_ids=committed,
        prompt_ids=committed,
        request_kwargs=kwargs,
        uses_default_max_tokens=True,
    )
    assert unchanged['max_tokens'] == 18


@pytest.mark.parametrize(
    ('components', 'expected_pending', 'expected_remaining'),
    [
        (['agent_output', 'tool_observation'], 18, 14),
        (['agent_output'], 18, 17),
    ],
)
def test_explicit_update_between_prepare_and_record_does_not_double_charge(
    components: list[str], expected_pending: int, expected_remaining: int
):
    state = TokenBudgetState(token_budget=20, token_budget_include=components)
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])
    extended = [1, 2, 3, 4, 5, 6, 7]

    prepared = state.prepare_generation(
        'main',
        raw_prompt_ids=extended,
        prompt_ids=extended,
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
    )
    assert prepared['max_tokens'] == expected_pending
    state.update_prompt('main', raw_prompt_ids=extended, prompt_ids=extended)
    state.update_prompt('main', raw_prompt_ids=extended, prompt_ids=extended)
    state.record_generation('main', raw_prompt_ids=extended, prompt_ids=extended, generated_ids=[8])

    completed = extended + [8]
    following = state.prepare_generation(
        'main',
        raw_prompt_ids=completed,
        prompt_ids=completed,
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
    )
    assert following['max_tokens'] == expected_remaining
    assert state.segment_consumed_tokens('main') == 6


@pytest.mark.parametrize(('budget', 'expected_limit'), [(20, 13), (10, 5)])
def test_raw_observation_delta_and_expanded_context_length_are_distinct(budget: int, expected_limit: int):
    state = TokenBudgetState(token_budget=budget)
    initial_raw = [1, 2]
    initial_expanded = [101, 102, 103, 104, 105]
    state.record_generation('main', raw_prompt_ids=initial_raw, prompt_ids=initial_expanded, generated_ids=[3, 4])

    # Raw grows by 2 output + 3 observation tokens; expanded length is independently 12.
    extended_raw = [1, 2, 3, 4, 5, 6, 7]
    extended_expanded = list(range(201, 213))
    state.update_prompt('main', raw_prompt_ids=extended_raw, prompt_ids=extended_expanded)
    prepared = state.prepare_generation(
        'main',
        raw_prompt_ids=extended_raw,
        prompt_ids=extended_expanded,
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=True,
        max_model_len=25,
    )
    assert prepared['max_tokens'] == expected_limit
    state.record_generation('main', raw_prompt_ids=extended_raw, prompt_ids=extended_expanded, generated_ids=[8])
    assert state.segment_consumed_tokens('main') == 6


def test_rejected_rewrite_preserves_old_context_baseline_until_commit():
    state = TokenBudgetState(token_budget=100)
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])

    with pytest.raises(ContextLengthExceededError):
        state.prepare_generation(
            'main',
            raw_prompt_ids=[9],
            prompt_ids=list(range(10)),
            request_kwargs={'max_tokens': 100},
            uses_default_max_tokens=True,
            max_model_len=10,
        )

    committed = [1, 2, 3, 4]
    old_request = state.prepare_generation(
        'main',
        raw_prompt_ids=committed,
        prompt_ids=committed,
        request_kwargs={'max_tokens': 7},
        uses_default_max_tokens=True,
        max_model_len=10,
    )
    assert old_request['max_tokens'] == 5

    # A successful explicit replacement commits its own expanded baseline.
    state.update_prompt('main', raw_prompt_ids=[9], prompt_ids=[109])
    rewritten = state.prepare_generation(
        'main',
        raw_prompt_ids=[9],
        prompt_ids=[109],
        request_kwargs={'max_tokens': 7},
        uses_default_max_tokens=True,
        max_model_len=10,
    )
    assert rewritten['max_tokens'] == 7
    assert state.segment_consumed_tokens('main') == 2


def test_model_length_is_request_local_while_token_budget_is_shared() -> None:
    state = TokenBudgetState(token_budget=20)
    main_context, child_context = ('main', 'conversation'), ('child', 'conversation')
    main_prompt, child_prompt = [1, 2], [101, 102]
    main_request = state.prepare_generation(
        main_context, main_prompt, main_prompt, {'max_tokens': 100}, True, max_model_len=10
    )
    assert main_request['max_tokens'] == 8, 'Use the main model capacity for this request'
    state.record_generation(main_context, main_prompt, main_prompt, [3, 4, 5])
    child_request = state.prepare_generation(
        child_context, child_prompt, child_prompt, {'max_tokens': 100}, True, max_model_len=6
    )
    assert child_request['max_tokens'] == 4, 'Use the child model capacity without changing the main context'
    state.record_generation(child_context, child_prompt, child_prompt, [103])
    main_continuation = [1, 2, 3, 4, 5]
    following = state.prepare_generation(main_context, main_continuation, main_continuation, {'max_tokens': 100}, True)
    assert following['max_tokens'] == 16, 'An omitted model length must not reuse an earlier request limit'
    assert state._token_consumed == 4, 'Both providers consume the same budget state'
    assert not hasattr(state, '_max_model_len'), 'Model capacity must not be stored in token budget state'


@pytest.mark.parametrize('failure_type', [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize('retry_before_rewrite', [False, True])
async def test_failed_tool_extension_keeps_its_charge_across_retry_and_rewrite(
    failure_type: type[BaseException], retry_before_rewrite: bool
):
    tokenizer = _NoGlueTokenizer()
    scripted = ['A' * 10, 'B' * 8, 'summary'] if retry_before_rewrite else ['A' * 10, 'summary']
    endpoint = _InterruptibleEndpoint(tokenizer, scripted)
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        token_budget=100,
        token_budget_include=['agent_output', 'tool_observation'],
    )
    messages = [dict(USER_MESSAGE)]
    first = await client.create_completion(messages=list(messages))
    extended = messages + [
        first.choices[0].message.model_dump(exclude_none=True),
        dict(TOOL_MESSAGE),
    ]

    endpoint.failure = failure_type('scripted endpoint interruption')
    with pytest.raises(failure_type):
        await client.create_completion(messages=extended)
    failed_limit = endpoint.attempted_limits[-1]

    # Dispatch failure leaves the prompt baseline untouched, but the observation
    # was already delivered and remains charged, even if compaction follows.
    expected_after_rewrite = failed_limit
    if retry_before_rewrite:
        await client.create_completion(messages=extended)
        assert endpoint.attempted_limits[-1] == failed_limit
        first_prefix = endpoint.calls[0]['prompt_ids'] + tokenizer.encode('A' * 10)
        retried_prompt = endpoint.calls[1]['prompt_ids']
        assert retried_prompt[: len(first_prefix)] == first_prefix
        observation_tokens = len(retried_prompt) - len(first_prefix)
        assert failed_limit == 90 - observation_tokens
        expected_after_rewrite -= 8

    await client.create_completion(
        [{'role': 'user', 'content': 'compacted context'}],
        request_meta={'invocation_kind': 'summary_inference'},
    )
    assert endpoint.max_tokens_per_call[-1] == expected_after_rewrite


def test_charged_observations_can_exhaust_budget_without_committing_prompt():
    state = TokenBudgetState(token_budget=5, token_budget_include=['agent_output', 'tool_observation'])
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])
    extended_prompt = [1, 2, 3, 4, 5, 6, 7]
    committed_prompt = list(state._accounted_prompt_ids['main'])

    for _ in range(2):
        state.account_tool_observation_suffix('main', extended_prompt, [TOOL_MESSAGE], 3)
        with pytest.raises(ContextLengthExceededError):
            state.prepare_generation(
                'main',
                raw_prompt_ids=extended_prompt,
                prompt_ids=extended_prompt,
                request_kwargs={'max_tokens': 100},
                uses_default_max_tokens=True,
            )
        np.testing.assert_array_equal(state._accounted_prompt_ids['main'], committed_prompt)
        assert state.segment_consumed_tokens('main') == 5


async def test_explicit_per_call_max_tokens_bounds_only_its_request():
    """An explicit per-call max_tokens replaces the cumulative segment computation."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, list(SCRIPTED_TEXTS))
    client = _build_gateway_client(tokenizer, endpoint)

    per_turn_kwargs = [{}, {'max_tokens': 1000, 'sampling_params': {'max_tokens': 7}}, {}]
    await _drive_turns(client, 3, per_turn_kwargs)

    assert endpoint.max_tokens_per_call[1] == 7


async def test_explicit_per_call_max_tokens_cannot_widen_configured_cap():
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, list(SCRIPTED_TEXTS))
    client = _build_gateway_client(tokenizer, endpoint)

    per_turn_kwargs = [{}, {'sampling_params': {'max_tokens': 1000}}, {}]
    await _drive_turns(client, 3, per_turn_kwargs)

    assert endpoint.max_tokens_per_call[1] == BASE_SAMPLING['max_tokens']


def test_explicit_max_tokens_respects_model_length_and_trajectory_budget():
    state = TokenBudgetState(token_budget=5)
    prompt = [1, 2, 3, 4]

    params = state.prepare_generation(
        'summary',
        raw_prompt_ids=prompt,
        prompt_ids=prompt,
        request_kwargs={'max_tokens': 100},
        uses_default_max_tokens=False,
        max_model_len=10,
    )
    assert params['max_tokens'] == 5

    state.record_generation('main', raw_prompt_ids=[9], prompt_ids=[9], generated_ids=[1, 2, 3, 4, 5])
    with pytest.raises(ContextLengthExceededError):
        state.prepare_generation(
            'summary',
            raw_prompt_ids=prompt,
            prompt_ids=prompt,
            request_kwargs={'max_tokens': 100},
            uses_default_max_tokens=False,
            max_model_len=10,
        )


@pytest.mark.parametrize('uses_default_max_tokens', [False, True])
@pytest.mark.parametrize(
    ('request_limit', 'model_length', 'token_budget', 'default_limit', 'explicit_limit'),
    [
        pytest.param(9, 40, 40, 3, 9, id='request-is-smallest'),
        pytest.param(100, 10, 40, 2, 2, id='model-is-smallest'),
        pytest.param(100, 40, 5, 3, 3, id='trajectory-is-smallest'),
        pytest.param(9, None, None, 3, 9, id='request-only'),
        pytest.param(None, 10, None, 2, 2, id='model-only'),
        pytest.param(None, None, 5, 3, 3, id='trajectory-only'),
        pytest.param(None, 10, 40, 2, 2, id='no-request-model-is-smallest'),
        pytest.param(None, 40, 5, 3, 3, id='no-request-trajectory-is-smallest'),
        pytest.param(9, 10, None, 2, 2, id='no-trajectory'),
        pytest.param(9, None, 5, 3, 3, id='no-model'),
        pytest.param(None, None, None, None, None, id='no-limits'),
    ],
)
def test_generation_honors_every_available_limit_after_prompt_growth(
    uses_default_max_tokens: bool,
    request_limit: int | None,
    model_length: int | None,
    token_budget: int | None,
    default_limit: int | None,
    explicit_limit: int | None,
) -> None:
    state = TokenBudgetState(token_budget=token_budget)
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])
    prompt = np.arange(1, 9)
    request = {'max_tokens': request_limit, 'temperature': 0.5}

    prepared = state.prepare_generation(
        'main',
        raw_prompt_ids=prompt,
        prompt_ids=prompt,
        request_kwargs=request,
        uses_default_max_tokens=uses_default_max_tokens,
        max_model_len=model_length,
    )

    expected_limit = default_limit if uses_default_max_tokens else explicit_limit
    assert prepared == {'max_tokens': expected_limit, 'temperature': 0.5}, 'Every configured limit must apply'
    assert request == {'max_tokens': request_limit, 'temperature': 0.5}, 'Preparing must preserve caller parameters'
    assert state.segment_consumed_tokens('main') == 2, 'Preparing must not charge candidate prompt growth'
    np.testing.assert_array_equal(state._accounted_prompt_ids['main'], [1, 2, 3, 4])


@pytest.mark.parametrize('uses_default_max_tokens', [False, True])
@pytest.mark.parametrize(
    ('request_kwargs', 'model_length', 'token_budget'),
    [
        pytest.param({'max_tokens': 0}, None, None, id='zero-request'),
        pytest.param({'max_tokens': -1}, None, None, id='negative-request'),
        pytest.param({}, 8, None, id='full-context'),
        pytest.param({}, 7, None, id='overflowed-context'),
        pytest.param({}, None, 2, id='exhausted-trajectory'),
        pytest.param({}, None, 1, id='overdrawn-trajectory'),
        pytest.param({'max_tokens': 100}, 100, 2, id='trajectory-exhausted-with-other-limits'),
    ],
)
def test_any_exhausted_limit_rejects_generation(
    uses_default_max_tokens: bool,
    request_kwargs: dict[str, int],
    model_length: int | None,
    token_budget: int | None,
) -> None:
    state = TokenBudgetState(token_budget=token_budget)
    state.record_generation('main', raw_prompt_ids=[1, 2], prompt_ids=[1, 2], generated_ids=[3, 4])
    prompt = np.arange(1, 9)

    with pytest.raises(ContextLengthExceededError, match='Gateway context length exceeded'):
        state.prepare_generation(
            'main',
            raw_prompt_ids=prompt,
            prompt_ids=prompt,
            request_kwargs=request_kwargs,
            uses_default_max_tokens=uses_default_max_tokens,
            max_model_len=model_length,
        )

    assert state.segment_consumed_tokens('main') == 2, 'Rejected requests must preserve committed consumption'
    np.testing.assert_array_equal(state._accounted_prompt_ids['main'], [1, 2, 3, 4])


async def test_trajectory_budget_without_other_limits_stops_before_dispatch() -> None:
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['abc', 'unused'])
    client = _build_gateway_client(tokenizer, endpoint, token_budget=3, sampling_params={'temperature': 1.0})

    with pytest.raises(ContextLengthExceededError):
        await _drive_turns(client, 2)

    assert endpoint.max_tokens_per_call == [3], 'A trajectory-only budget must limit output and stop the next call'


async def test_budget_exhaustion_raises_on_same_turn():
    """When the running prompt exceeds the budget, the next turn raises before dispatch."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['A' * 10, 'B' * 8])
    client = _build_gateway_client(tokenizer, endpoint, sampling_params={'max_tokens': 12, 'temperature': 1.0})

    with pytest.raises(ContextLengthExceededError):
        await _drive_turns(client, 2)

    # Turn 1 dispatched; turn 2 raised before reaching the endpoint.
    assert len(endpoint.max_tokens_per_call) == 1


async def test_failed_budget_check_does_not_rebase_followup_prompt():
    """A rejected prompt is not a committed prefix for a later notification."""
    tokenizer = _NoGlueTokenizer()
    endpoint = RecordingEndpoint(tokenizer, ['A' * 10, 'unexpected'])
    client = _build_gateway_client(
        tokenizer,
        endpoint,
        sampling_params={'max_tokens': 12, 'temperature': 1.0},
    )

    messages = [dict(USER_MESSAGE)]
    completion = await client.create_completion(messages=list(messages))
    assistant = completion.choices[0].message.model_dump(exclude_none=True)
    running = messages + [assistant, {'role': 'tool', 'content': 'background command is running'}]

    with pytest.raises(ContextLengthExceededError):
        await client.create_completion(messages=running)

    notified = running + [{'role': 'user', 'content': '<task-notification>completed</task-notification>'}]
    with pytest.raises(ContextLengthExceededError):
        await client.create_completion(messages=notified)

    assert len(endpoint.max_tokens_per_call) == 1
