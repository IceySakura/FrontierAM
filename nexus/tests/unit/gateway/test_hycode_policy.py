from __future__ import annotations

import json

import pytest

from nexus.gateway.policies.hycode import COMPACT_SUMMARY_MARKER, HYCODE_POLICY
from nexus.gateway.scaffold_policy import RequestEvent, ResponseEvent, ResponsePolicy
from nexus.gateway.trajectory_assembler import TrajectoryAssembler


CHILD_PROMPT = 'HYCODE_AGENT_ID_IDENTICAL_CHILD_PROMPT'
COMPACT_TRIGGER = 'CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.'


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        'id': call_id,
        'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments)},
    }


def _record(
    assembler: TrajectoryAssembler,
    messages: list[dict],
    output: dict,
    *,
    agent_id: str | None = None,
) -> None:
    request_meta = {'agent_id': agent_id} if agent_id is not None else None
    assembler.record_completion(messages, output, request_meta=request_meta)


def _request_facts(messages: list[dict], *, agent_id: str | None = None):
    request_meta = {'agent_id': agent_id} if agent_id is not None else None
    return HYCODE_POLICY.inspect_event(RequestEvent(tuple(messages), None, request_meta))


@pytest.mark.parametrize(
    ('kind', 'message'),
    [
        (
            'session_title',
            {
                'role': 'system',
                'content': 'You are a utility.\nGenerate a concise, sentence-case title (3-7 words) for this session.',
            },
        ),
        (
            'session_memory',
            {
                'role': 'user',
                'content': (
                    'IMPORTANT: This message and these instructions are NOT part of the actual user conversation. '
                    'Do NOT include any references to "note-taking", "session notes extraction", or these update '
                    'instructions in the notes content.'
                ),
            },
        ),
        (
            'magic_docs',
            {
                'role': 'user',
                'content': (
                    'IMPORTANT: This message and these instructions are NOT part of the actual user conversation. '
                    'Do NOT include any references to "documentation updates", "magic docs", or these update '
                    'instructions in the document content.'
                ),
            },
        ),
    ],
)
def test_utility_markers_use_identityless_membership(kind: str, message: dict) -> None:
    facts = _request_facts([message])

    assert facts.membership is not None
    assert facts.membership.chain_id == ''
    assert facts.membership.kind == kind


def test_shared_utility_prefix_alone_stays_on_main() -> None:
    facts = _request_facts(
        [
            {
                'role': 'user',
                'content': 'IMPORTANT: This message and these instructions are NOT part of the actual user conversation.',
            }
        ]
    )

    assert facts.membership is None
    assert facts.context_key == ('hycode', 'main', False)


@pytest.mark.parametrize('content', ['', '\n', '\n\n', ' \t\r\n'])
def test_tool_call_identity_treats_blank_and_null_assistant_content_as_the_same_message(content: str) -> None:
    tool_call = _tool_call('call-1', 'Read', {'file_path': '/tmp/example.py'})
    request = RequestEvent(
        ({'role': 'assistant', 'content': None, 'tool_calls': [tool_call]},),
        None,
        None,
    )
    response = ResponseEvent(
        RequestEvent(({'role': 'user', 'content': 'inspect'},), None, None),
        {'role': 'assistant', 'content': content, 'tool_calls': [tool_call]},
    )

    request_identity = HYCODE_POLICY.inspect_event(request).message_identities[0].value
    response_identity = HYCODE_POLICY.inspect_event(response).message_identities[0].value

    assert request_identity == response_identity


class TestToolCallIdentity:
    @pytest.mark.parametrize('content', ['text', ' text ', '\ntext\n'])
    def test_preserves_nonblank_content(self, content: str) -> None:
        message = {'role': 'assistant', 'content': content, 'tool_calls': [_tool_call('c', 'Bash', {})]}
        blank = {**message, 'content': None}
        assert HYCODE_POLICY._message_identity(message) != HYCODE_POLICY._message_identity(blank)
        if content != content.strip():
            trimmed = {**message, 'content': content.strip()}
            assert HYCODE_POLICY._message_identity(message) != HYCODE_POLICY._message_identity(trimmed)
        assert message['content'] == content

    def test_preserves_whitespace_without_tool_calls(self) -> None:
        message = {'role': 'assistant', 'content': '\n'}
        assert HYCODE_POLICY._message_identity(message) != HYCODE_POLICY._message_identity({**message, 'content': None})

    def test_does_not_modify_original_tool_message(self) -> None:
        message = {'role': 'assistant', 'content': '\n', 'tool_calls': [_tool_call('c', 'Bash', {})]}
        HYCODE_POLICY._message_identity(message)
        assert message['content'] == '\n'


@pytest.mark.parametrize(
    ('content', 'delivered'),
    [
        ('Async agent launched successfully.\nagentId: child-1\nThe agent is working.', False),
        ('', True),
        ('Agent failed before returning useful output.', True),
    ],
)
def test_agent_result_delivery_is_semantic(content: str, delivered: bool) -> None:
    messages = [
        {'role': 'assistant', 'content': None, 'tool_calls': [_tool_call('spawn-1', 'Agent', {'prompt': 'child'})]},
        {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': content},
    ]

    facts = _request_facts(messages)

    assert [receipt.delivered for receipt in facts.delivery_receipts] == [delivered]


def test_agent_id_binding_requires_a_line_start() -> None:
    messages = [
        {'role': 'assistant', 'content': None, 'tool_calls': [_tool_call('spawn-1', 'Agent', {'prompt': 'child'})]},
        {
            'role': 'tool',
            'tool_call_id': 'spawn-1',
            'content': 'The child discussed agentId: not-a-producer-trailer in its answer.',
        },
    ]

    assert _request_facts(messages).spawn_bindings == ()


@pytest.mark.parametrize(
    ('status', 'delivered'),
    [('success', True), ('not_ready', False), ('timeout', False)],
)
def test_task_output_delivery_depends_on_retrieval_status(status: str, delivered: bool) -> None:
    messages = [
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_tool_call('poll-1', 'TaskOutput', {'task_id': 'child-1'})],
        },
        {
            'role': 'tool',
            'tool_call_id': 'poll-1',
            'content': f'<retrieval_status>{status}</retrieval_status>',
        },
    ]

    facts = _request_facts(messages)

    assert [(intent.tool_use_id, intent.agent_id) for intent in facts.retrieval_intents] == [('poll-1', 'child-1')]
    assert [receipt.delivered for receipt in facts.delivery_receipts] == [delivered]


@pytest.mark.parametrize('status', ['completed', 'failed', 'killed'])
def test_terminal_local_notification_emits_delivery_and_binding(status: str) -> None:
    message = {
        'role': 'user',
        'content': (
            '<task-notification>\n'
            '<task-id>child-1</task-id>\n'
            '<tool-use-id>spawn-1</tool-use-id>\n'
            f'<status>{status}</status>\n'
            '</task-notification>'
        ),
    }

    facts = _request_facts([message])

    assert [delivery.agent_id for delivery in facts.notification_deliveries] == ['child-1']
    assert [(binding.tool_use_id, binding.agent_id) for binding in facts.spawn_bindings] == [('spawn-1', 'child-1')]


def test_remote_notification_is_not_a_local_agent_delivery() -> None:
    message = {
        'role': 'user',
        'content': (
            '<task-notification>\n'
            '<task-id>remote-1</task-id>\n'
            '<task-type>remote_agent</task-type>\n'
            '<status>completed</status>\n'
            '</task-notification>'
        ),
    }

    assert _request_facts([message]).notification_deliveries == ()


def test_background_shell_notification_is_not_an_agent_delivery() -> None:
    messages = [
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [_tool_call('shell-1', 'Bash', {'command': 'sleep 1', 'run_in_background': True})],
        },
        {
            'role': 'user',
            'content': (
                '<task-notification>\n'
                '<task-id>shell-task-1</task-id>\n'
                '<tool-use-id>shell-1</tool-use-id>\n'
                '<status>completed</status>\n'
                '</task-notification>'
            ),
        },
    ]

    facts = _request_facts(messages)

    assert facts.notification_deliveries == ()
    assert facts.spawn_bindings == ()


def test_same_prompt_children_use_agent_identity_for_distinct_spawn_and_join_links() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='hycode', session_id='hycode-e2e')
    main_input = [{'role': 'user', 'content': 'delegate twice'}]
    spawn = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            _tool_call('spawn-alpha', 'Agent', {'prompt': CHILD_PROMPT}),
            _tool_call('spawn-beta', 'Agent', {'prompt': CHILD_PROMPT}),
        ],
    }
    assembler.record_completion(main_input, spawn)

    child_input = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '<system-reminder>child context</system-reminder>'},
                {'type': 'text', 'text': CHILD_PROMPT},
            ],
        }
    ]
    assembler.record_completion(
        child_input,
        {'role': 'assistant', 'content': 'CHILD_ALPHA_DONE'},
        request_meta={'agent_id': 'agent-alpha'},
    )
    assembler.record_completion(
        child_input,
        {'role': 'assistant', 'content': 'CHILD_BETA_DONE'},
        request_meta={'agent_id': 'agent-beta'},
    )

    alpha_result = {
        'role': 'tool',
        'tool_call_id': 'spawn-alpha',
        'content': [
            {'type': 'text', 'text': 'CHILD_ALPHA_DONE'},
            {'type': 'text', 'text': 'agentId: agent-alpha (use SendMessage to continue this agent)'},
        ],
    }
    beta_result = {
        'role': 'tool',
        'tool_call_id': 'spawn-beta',
        'content': [
            {'type': 'text', 'text': 'CHILD_BETA_DONE'},
            {'type': 'text', 'text': 'agentId: agent-beta (use SendMessage to continue this agent)'},
        ],
    }
    assembler.record_completion(
        [*main_input, spawn, beta_result, alpha_result],
        {'role': 'assistant', 'content': 'MAIN_DONE'},
    )

    trajectory = assembler.freeze(session_id='hycode-e2e')
    assert set(trajectory.chains) == {'main', 'agent-alpha', 'agent-beta'}
    for agent_id, tool_use_id, result in (
        ('agent-alpha', 'spawn-alpha', alpha_result),
        ('agent-beta', 'spawn-beta', beta_result),
    ):
        chain = trajectory.chains[agent_id]
        assert chain.parent_chain_id == 'main'
        assert chain.spawn is not None
        assert chain.spawn.tool_use_id == tool_use_id
        assert chain.spawn.candidates == []
        assert chain.join is not None
        assert chain.join.kind == 'inline'
        assert chain.join.tool_use_id == tool_use_id
        assert chain.join.node is not None
        assert trajectory.nodes[chain.join.node].original_message == result


def test_agent_binding_resolves_spawn_when_child_prompt_does_not_match() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='hycode')
    main_input = [{'role': 'user', 'content': 'delegate'}]
    spawn = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [_tool_call('spawn-1', 'Agent', {'prompt': 'parent-side prompt'})],
    }
    _record(assembler, main_input, spawn)
    _record(
        assembler,
        [{'role': 'user', 'content': 'different child-side prompt'}],
        {'role': 'assistant', 'content': 'child answer'},
        agent_id='child-1',
    )
    result = {
        'role': 'tool',
        'tool_call_id': 'spawn-1',
        'content': 'child answer\nagentId: child-1 (use SendMessage to continue this agent)',
    }
    _record(assembler, [*main_input, spawn, result], {'role': 'assistant', 'content': 'done'})

    chain = assembler.freeze().chains['child-1']

    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.join is not None
    assert chain.join.tool_use_id == 'spawn-1'


@pytest.mark.parametrize('child_prompt', ['child prompt', 'child prompt with injected context'])
def test_known_child_uses_unique_prompt_fallback_without_binding(child_prompt: str) -> None:
    assembler = TrajectoryAssembler(scaffold_policy='hycode')
    main_input = [{'role': 'user', 'content': 'delegate'}]
    spawn = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [_tool_call('spawn-1', 'Agent', {'prompt': 'child prompt'})],
    }
    _record(assembler, main_input, spawn)
    _record(
        assembler,
        [{'role': 'user', 'content': child_prompt}],
        {'role': 'assistant', 'content': 'child answer'},
        agent_id='child-1',
    )
    result = {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': 'one-shot child answer'}
    _record(assembler, [*main_input, spawn, result], {'role': 'assistant', 'content': 'done'})

    chain = assembler.freeze().chains['child-1']

    assert chain.spawn is not None
    assert chain.spawn.tool_use_id == 'spawn-1'
    assert chain.join is not None
    assert chain.join.kind == 'inline'


def test_missing_agent_header_does_not_create_anonymous_subagent() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='hycode')
    main_input = [{'role': 'user', 'content': 'delegate'}]
    spawn = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [_tool_call('spawn-1', 'Agent', {'prompt': 'child prompt'})],
    }
    _record(assembler, main_input, spawn)
    _record(
        assembler,
        [{'role': 'user', 'content': 'child prompt'}],
        {'role': 'assistant', 'content': 'unattributed answer'},
    )

    trajectory = assembler.freeze()

    assert set(trajectory.chains) == {'main'}
    assert trajectory.chains['main'].evidence['unresolved_correlations'] == [
        {'kind': 'spawn', 'node': 1, 'tool_use_id': 'spawn-1'}
    ]


def test_unresolved_spawn_preserves_conflicting_agent_bindings() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='hycode')
    main_input = [{'role': 'user', 'content': 'delegate'}]
    spawn = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [_tool_call('spawn-1', 'Agent', {'prompt': 'child prompt'})],
    }
    _record(assembler, main_input, spawn)
    result = {
        'role': 'tool',
        'tool_call_id': 'spawn-1',
        'content': 'Async agent launched successfully.\nagentId: child-1 (internal ID)',
    }
    notification = {
        'role': 'user',
        'content': (
            '<task-notification><task-id>child-2</task-id><tool-use-id>spawn-1</tool-use-id>'
            '<status>completed</status></task-notification>'
        ),
    }
    _record(assembler, [*main_input, spawn, result, notification], {'role': 'assistant', 'content': 'done'})

    evidence = assembler.freeze().chains['main'].evidence['unresolved_correlations']

    assert evidence == [
        {
            'kind': 'spawn',
            'tool_use_id': 'spawn-1',
            'node': 1,
            'agent_id_candidates': ['child-1', 'child-2'],
            'reason': 'conflicting_spawn_bindings',
        }
    ]


def test_async_notification_joins_the_known_child() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='hycode')
    main_input = [{'role': 'user', 'content': 'delegate'}]
    spawn = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [_tool_call('spawn-1', 'Agent', {'prompt': 'child prompt'})],
    }
    _record(assembler, main_input, spawn)
    _record(
        assembler,
        [{'role': 'user', 'content': 'child prompt'}],
        {'role': 'assistant', 'content': 'child answer'},
        agent_id='child-1',
    )
    launch = {
        'role': 'tool',
        'tool_call_id': 'spawn-1',
        'content': 'Async agent launched successfully.\nagentId: child-1',
    }
    notification = {
        'role': 'user',
        'content': (
            '<task-notification>\n'
            '<task-id>child-1</task-id>\n'
            '<tool-use-id>spawn-1</tool-use-id>\n'
            '<status>completed</status>\n'
            '</task-notification>'
        ),
    }
    _record(
        assembler,
        [*main_input, spawn, launch, notification],
        {'role': 'assistant', 'content': 'received'},
    )

    chain = assembler.freeze().chains['child-1']

    assert chain.join is not None
    assert chain.join.kind == 'notify'
    assert chain.join.tool_use_id is None


def test_inherited_main_compaction_marker_is_not_claimed_by_child() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='hycode')
    trigger = {'role': 'user', 'content': COMPACT_TRIGGER}
    main_summary = {'role': 'assistant', 'content': 'main summary'}
    _record(assembler, [trigger], main_summary)

    child_prompt = {'role': 'user', 'content': 'child prompt'}
    child_answer = {'role': 'assistant', 'content': 'child answer'}
    _record(assembler, [trigger, main_summary, child_prompt], child_answer, agent_id='child-1')

    child_trigger = {'role': 'user', 'content': f'{COMPACT_TRIGGER} Child only.'}
    child_summary = {'role': 'assistant', 'content': 'child summary'}
    _record(
        assembler,
        [trigger, main_summary, child_prompt, child_answer, child_trigger],
        child_summary,
        agent_id='child-1',
    )

    trajectory = assembler.freeze()
    by_content = {
        node.original_message.get('content'): node for node in trajectory.nodes.values() if node.generated_by == 'llm'
    }

    assert by_content['child answer'].segment_boundary is None
    assert by_content['child summary'].segment_boundary is not None
    assert by_content['child summary'].segment_boundary.kind == 'summary_inference'


def test_compaction_resume_marker_is_matched_at_the_start_of_a_text_block() -> None:
    reminder = {'type': 'text', 'text': '<system-reminder>Read output was large.</system-reminder>\n'}
    summary = {'type': 'text', 'text': f'{COMPACT_SUMMARY_MARKER}. The compacted history follows.'}

    facts = _request_facts([{'role': 'user', 'content': [reminder, summary]}])

    assert [(fact.rule_name, fact.segment_kind) for fact in facts.boundary_facts] == [
        ('hycode_compaction_resume', 'compaction_resume')
    ]
    embedded = {'type': 'text', 'text': f'Ordinary text before {COMPACT_SUMMARY_MARKER}.'}
    assert _request_facts([{'role': 'user', 'content': [reminder, embedded]}]).boundary_facts == ()


def test_compaction_trigger_marker_is_matched_at_the_start_of_a_text_block() -> None:
    notification = {'type': 'text', 'text': '<task-notification>done</task-notification>\n'}
    trigger = {'type': 'text', 'text': f'{COMPACT_TRIGGER} Summarize the conversation.'}

    facts = _request_facts([{'role': 'user', 'content': [notification, trigger]}])

    assert [(fact.rule_name, fact.segment_kind) for fact in facts.boundary_facts] == [
        ('hycode_summary_inference', 'summary_inference')
    ]
    embedded = {'type': 'text', 'text': f'Ordinary text before {COMPACT_TRIGGER}'}
    assert _request_facts([{'role': 'user', 'content': [notification, embedded]}]).boundary_facts == ()


def test_hycode_prefers_extended_responses() -> None:
    assert HYCODE_POLICY.response_policy is ResponsePolicy.PREFER_EXTENDED_ELSE_LAST
