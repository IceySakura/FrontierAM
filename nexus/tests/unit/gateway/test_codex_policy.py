from __future__ import annotations

import json

import pytest

from _fakes import memory_store

from nexus.gateway.processor import Processor
from nexus.gateway.scaffold_policy import InputMessage, RequestEvent, build_scaffold_policy
from nexus.gateway.session import GatewaySession
from nexus.gateway.trajectory_assembler import TrajectoryAssembler
from nexus.gateway.types import BackendResult
from nexus.runtime.server.services.codex_compaction import DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX
from nexus.runtime.server.services.responses_proxy.codex import build_codex_request_meta
from nexus.runtime.server.services.responses_proxy.converter import responses_to_openai


def _request_meta(version: str, **codex: object) -> dict:
    return {
        'request_id': f'provider-{version}',
        'scaffold': 'codex',
        'scaffold_version': version,
        'codex': codex,
        'codex_metadata_source': 'turn_metadata',
    }


@pytest.mark.parametrize('version', ['0.146.1', '0.147.0', '0.148.0'])
def test_registered_codex_policy_uses_structured_thread_identity_across_versions(version: str) -> None:
    policy = build_scaffold_policy('codex')
    request_meta = _request_meta(
        version,
        request_kind='turn',
        session_id='root-thread',
        thread_id='child-thread',
        window_id='child-thread:0',
        parent_thread_id='root-thread',
        thread_source='subagent',
        subagent_kind='thread_spawn',
    )

    facts = policy.inspect_event(RequestEvent(({'role': 'user', 'content': 'child prompt'},), None, request_meta))

    assert facts.context_key == ('codex', 'child-thread', 'child-thread:0')
    assert facts.membership is not None
    assert facts.membership.chain_id == 'child-thread'
    assert facts.membership.parent_chain_id == 'main'
    assert facts.membership.detector == 'codex_turn_metadata'
    assert facts.child_prompts[0].anchor == InputMessage(0)
    assert facts.child_prompts[0].value == 'child prompt'


def test_thread_spawn_metadata_normalizes_to_trainable_subagent() -> None:
    policy = build_scaffold_policy('codex')
    request_meta = _request_meta(
        '0.147.0',
        request_kind='turn',
        session_id='root-thread',
        thread_id='child-thread',
        window_id='child-thread:0',
        parent_thread_id='root-thread',
        thread_source='subagent',
        subagent_kind='thread_spawn',
    )

    facts = policy.inspect_event(RequestEvent(({'role': 'user', 'content': 'child'},), None, request_meta))

    assert facts.membership is not None
    assert facts.membership.kind == 'subagent'
    assert facts.membership.parent_chain_id == 'main'
    assert facts.declared_claim is not None
    assert facts.declared_claim.kind == 'subagent'
    attributes = dict(facts.membership.attributes)
    assert attributes['codex_subagent_kind'] == 'thread_spawn'
    assert attributes['parent_chain_id'] == 'main'


def test_unknown_codex_sidechain_kind_is_not_promoted_to_trainable_subagent() -> None:
    policy = build_scaffold_policy('codex')
    request_meta = _request_meta(
        '0.147.0',
        request_kind='turn',
        session_id='root-thread',
        thread_id='review-thread',
        window_id='review-thread:0',
        parent_thread_id='root-thread',
        thread_source='subagent',
        subagent_kind='review',
    )

    facts = policy.inspect_event(RequestEvent(({'role': 'user', 'content': 'review'},), None, request_meta))

    assert facts.membership is not None
    assert facts.membership.kind == 'review'
    assert dict(facts.membership.attributes)['codex_subagent_kind'] == 'review'


def test_legacy_prompt_fallback_marks_compaction_without_codex_metadata() -> None:
    policy = build_scaffold_policy('codex')
    prompt = f'{DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX} Preserve all decisions.'
    request_meta = {
        'request_id': 'provider-legacy',
        'scaffold': 'codex',
        'scaffold_version': '0.114.0',
        'codex': {},
        'codex_metadata_source': 'none',
    }

    facts = policy.inspect_event(RequestEvent(({'role': 'user', 'content': prompt},), None, request_meta))

    assert facts.context_key == ('codex', 'main', False)
    assert facts.membership is None
    assert len(facts.boundary_facts) == 1
    boundary = facts.boundary_facts[0]
    assert boundary.rule_name == 'codex_compaction_prompt_fallback'
    assert boundary.segment_kind == 'summary_inference'
    assert boundary.cause == 'compaction'
    assert boundary.at_request_edge is True


def test_explicit_turn_metadata_wins_over_compaction_shaped_prompt() -> None:
    policy = build_scaffold_policy('codex')
    prompt = f'{DEFAULT_CODEX_COMPACTION_PROMPT_PREFIX} This is ordinary user content.'
    request_meta = _request_meta(
        '0.147.0',
        request_kind='turn',
        session_id='root-thread',
        thread_id='root-thread',
        window_id='root-thread:0',
    )

    facts = policy.inspect_event(RequestEvent(({'role': 'user', 'content': prompt},), None, request_meta))

    assert facts.context_key == ('codex', 'root-thread', 'root-thread:0')
    assert facts.boundary_facts == ()


def test_incomplete_structured_identity_stays_on_main_chain_without_guessing() -> None:
    policy = build_scaffold_policy('codex')
    request_meta = _request_meta('0.147.0', request_kind='turn', thread_source='subagent')

    facts = policy.inspect_event(RequestEvent(({'role': 'user', 'content': 'work'},), None, request_meta))

    assert facts.context_key == ('codex', 'main', False)
    assert facts.membership is None
    assert facts.declared_claim is None
    assert facts.parent_claim is None


def test_duplicate_execution_keeps_only_the_continued_response() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='codex', session_id='duplicate-execution')
    meta = _request_meta(
        '0.160.0',
        request_kind='turn',
        session_id='root-thread',
        thread_id='root-thread',
        window_id='root-thread:0',
    )
    prompt = [{'role': 'user', 'content': 'Fix the bug.'}]
    lost = {'role': 'assistant', 'content': 'never delivered'}
    delivered = {'role': 'assistant', 'content': 'delivered'}
    assembler.record_completion(prompt, lost, request_meta=meta)
    assembler.record_completion(prompt, delivered, request_meta=meta)
    assembler.record_completion(
        [*prompt, delivered, {'role': 'user', 'content': 'Continue.'}],
        {'role': 'assistant', 'content': 'done'},
        request_meta=meta,
    )

    trajectory = assembler.freeze(state='completed')

    selection = {
        node.original_message['content']: node.selection
        for node in trajectory.nodes.values()
        if node.generated_by == 'llm'
    }
    assert selection['never delivered'] == 'rejected_sibling'
    assert selection['delivered'] == 'selected'
    assert len(trajectory.to_segment_node_lists()) == 1


@pytest.mark.parametrize('version', ['0.146.1', '0.147.0'])
def test_structured_compaction_is_a_summary_boundary(version: str) -> None:
    policy = build_scaffold_policy('codex')
    request_meta = _request_meta(
        version,
        request_kind='compaction',
        session_id='root-thread',
        thread_id='root-thread',
        window_id='root-thread:0',
    )

    facts = policy.inspect_event(RequestEvent(({'role': 'user', 'content': 'summarize'},), None, request_meta))

    assert len(facts.boundary_facts) == 1
    boundary = facts.boundary_facts[0]
    assert boundary.rule_name == 'codex_compaction_metadata'
    assert boundary.segment_kind == 'summary_inference'
    assert boundary.from_request_context is True


@pytest.mark.asyncio
async def test_gateway_backend_uses_session_id_and_preserves_provider_request_metadata() -> None:
    class Backend:
        def __init__(self) -> None:
            self.request_id = None
            self.request_meta = None

        async def create_completion(self, request_id, *, request_meta, **_kwargs):
            self.request_id = request_id
            self.request_meta = request_meta
            return BackendResult(message={'role': 'assistant', 'content': 'done'}, stop_reason='stop')

    backend = Backend()
    session = GatewaySession('trajectory-1', Processor(backend), memory_store('trajectory-1'), scaffold_policy='codex')

    await session.create_completion(
        [{'role': 'user', 'content': 'work'}],
        request_meta=_request_meta(
            '0.147.0',
            request_kind='turn',
            session_id='root-thread',
            thread_id='root-thread',
            window_id='root-thread:0',
        ),
    )

    await session.finish_pending_publications()
    assert backend.request_id == 'trajectory-1'
    assert backend.request_meta['request_id'] == 'provider-0.147.0'


def test_subagent_v1_completion_notification_joins_the_structured_child_chain() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='codex', session_id='trajectory-1')
    child_prompt = 'Inspect the bug without editing files.'
    spawn = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {
                'type': 'function',
                'id': 'call-spawn-1',
                'function': {
                    'name': 'multi_agent_v1__spawn_agent',
                    'arguments': json.dumps({'message': child_prompt}),
                },
            }
        ],
    }
    main_meta = _request_meta(
        '0.147.0',
        request_kind='turn',
        session_id='root-thread',
        thread_id='root-thread',
        window_id='root-thread:0',
    )
    main_meta['codex_tool_identities'] = {
        'multi_agent_v1__spawn_agent': {
            'tool_type': 'function',
            'namespace': 'multi_agent_v1',
            'name': 'spawn_agent',
        }
    }
    assembler.record_completion(
        [{'role': 'user', 'content': 'Fix the bug.'}],
        spawn,
        request_meta=main_meta,
    )

    child_meta = _request_meta(
        '0.147.0',
        request_kind='turn',
        session_id='root-thread',
        thread_id='child-thread',
        window_id='child-thread:0',
        parent_thread_id='root-thread',
        thread_source='subagent',
        subagent_kind='thread_spawn',
    )
    assembler.record_completion(
        [{'role': 'user', 'content': child_prompt}],
        {'role': 'assistant', 'content': 'The bug is in the parser.'},
        request_meta=child_meta,
    )

    notification = {
        'role': 'user',
        'content': (
            '<subagent_notification>\n'
            '{"agent_path":"child-thread","status":{"completed":"The bug is in the parser."}}\n'
            '</subagent_notification>'
        ),
    }
    assembler.record_completion(
        [{'role': 'user', 'content': 'Fix the bug.'}, spawn, notification],
        {'role': 'assistant', 'content': 'I will apply the fix.'},
        request_meta=main_meta,
    )

    trajectory = assembler.freeze(state='completed')
    child = trajectory.chains['child-thread']
    assert child.spawn is not None
    assert child.spawn.tool_use_id == 'call-spawn-1'
    assert child.join is not None
    assert child.join.kind == 'notify'
    assert trajectory.nodes[child.join.node].original_message == notification


def test_v1_text_items_spawn_correlates_with_child_wire_prompt() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='codex', session_id='text-item-spawn')
    main_meta = _request_meta('0.147.0', session_id='root-thread', thread_id='root-thread', window_id='root-thread:0')
    assembler.record_completion(
        [{'role': 'user', 'content': 'Delegate the calculation.'}],
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'spawn-text-items',
                    'type': 'function',
                    'function': {
                        'name': 'spawn_agent',
                        'arguments': json.dumps({
                            'items': [
                                {'type': 'text', 'text': 'Compute 1 + 1.'},
                                {'type': 'text', 'text': 'Return just the number.'},
                            ]
                        }),
                    },
                }
            ],
        },
        request_meta=main_meta,
    )
    child_request = responses_to_openai({
        'input': [
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    {'type': 'input_text', 'text': 'Compute 1 + 1.'},
                    {'type': 'input_text', 'text': 'Return just the number.'},
                ],
            }
        ]
    })
    assembler.record_completion(
        child_request['messages'],
        {'role': 'assistant', 'content': '2'},
        request_meta=_request_meta(
            '0.147.0',
            session_id='root-thread',
            thread_id='child-thread',
            window_id='child-thread:0',
            parent_thread_id='root-thread',
            thread_source='subagent',
            subagent_kind='thread_spawn',
        ),
    )

    child = assembler.freeze(state='completed').chains['child-thread']

    assert child.spawn is not None
    assert child.spawn.tool_use_id == 'spawn-text-items'


@pytest.mark.parametrize('argument_field', ['message', 'items'])
def test_v1_spawn_results_bind_identical_prompts_to_concrete_agent_ids(argument_field: str) -> None:
    assembler = TrajectoryAssembler(scaffold_policy='codex', session_id='wire-spawns')
    request_body = {
        'tools': [
            {
                'type': 'namespace',
                'name': 'multi_agent_v1',
                'tools': [{'type': 'function', 'name': 'spawn_agent'}],
            }
        ],
        'input': [{'type': 'message', 'role': 'user', 'content': 'Delegate the calculations.'}],
    }
    tool_identities = {}
    converted = responses_to_openai(request_body, collect_tool_name_mappings=tool_identities)
    main_meta = build_codex_request_meta(
        request_body,
        {},
        configured_version='0.147.0',
        request_id='parent-spawns',
        compaction_evidence=None,
        tool_identities=tool_identities,
    )
    main_meta['codex'] = {'session_id': 'root-thread', 'thread_id': 'root-thread', 'window_id': 'root-thread:0'}
    # Identical task text leaves prompt correlation ambiguous; only the spawn results disambiguate.
    arguments = (
        {'message': 'Compute the result.'}
        if argument_field == 'message'
        else {'items': [{'type': 'text', 'text': 'Compute the result.'}]}
    )
    spawn = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
            {
                'id': call_id,
                'type': 'function',
                'function': {'name': 'multi_agent_v1__spawn_agent', 'arguments': json.dumps(arguments)},
            }
            for call_id in ('spawn-first', 'spawn-failed', 'spawn-last')
        ],
    }
    assembler.record_completion(converted['messages'], spawn, request_meta=main_meta)
    for child_id in ('child-last', 'child-first'):
        assembler.record_completion(
            [{'role': 'user', 'content': 'Compute the result.'}],
            {'role': 'assistant', 'content': '42'},
            request_meta=_request_meta(
                '0.147.0',
                session_id='root-thread',
                thread_id=child_id,
                window_id=f'{child_id}:0',
                parent_thread_id='root-thread',
                thread_source='subagent',
                subagent_kind='thread_spawn',
            ),
        )
    for call_id, result in (
        ('spawn-first', {'agent_id': 'child-first', 'nickname': 'Ada'}),
        ('spawn-failed', 'Agent limit reached'),
        ('spawn-last', {'agent_id': 'child-last', 'nickname': 'Grace'}),
    ):
        request_body['input'].extend([
            {
                'type': 'function_call',
                'call_id': call_id,
                'namespace': 'multi_agent_v1',
                'name': 'spawn_agent',
                'arguments': json.dumps(arguments),
            },
            {
                'type': 'function_call_output',
                'call_id': call_id,
                'output': [{'type': 'input_text', 'text': json.dumps(result)}],
            },
        ])
    converted = responses_to_openai(request_body, collect_tool_name_mappings=tool_identities)
    assembler.record_completion(
        converted['messages'], {'role': 'assistant', 'content': 'Done.'}, request_meta=main_meta
    )

    trajectory = assembler.freeze(state='completed')

    assert set(trajectory.chains) == {'main', 'child-first', 'child-last'}
    assert trajectory.chains['child-first'].spawn.tool_use_id == 'spawn-first'
    assert trajectory.chains['child-last'].spawn.tool_use_id == 'spawn-last'
