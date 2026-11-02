"""WorkBuddy Gateway policy facts and trajectory assembly from request identity headers."""

from __future__ import annotations

import json

import pytest

from nexus.gateway.policies.workbuddy import (
    MAIN_CONVERSATION_ID_KEY,
    RETRACTED_TASK_NOTIFICATION_KEY,
    WORKBUDDY_POLICY,
    build_workbuddy_request_meta,
)
from nexus.gateway.scaffold_policy import InvocationKind, RequestEvent, ResponseEvent, ResponsePolicy
from nexus.gateway.trajectory_assembler import TrajectoryAssembler


MAIN_ID = 'main-conversation'
CHILD_ID = 'child-conversation'
TITLE_ID = 'title-conversation'
CHILD_PROMPT = 'Read chapter-1.txt and report its ANCHOR_1 line.'
MAIN_SYSTEM = {'role': 'system', 'content': 'This conversation is powered by general-model'}
CHILD_SYSTEM = {'role': 'system', 'content': 'You are CodeBuddy Code.'}
SUMMARY_SYSTEM = {'role': 'system', 'content': 'You are a helpful AI assistant tasked with summarizing conversations.'}


def _meta(
    purpose: str,
    conversation_id: str = MAIN_ID,
    *,
    agent_type: str | None = 'main',
    parent_conversation_id: str | None = None,
    invocation_kind: str | None = None,
) -> dict:
    identity = {'agent_type': agent_type, 'agent_purpose': purpose, 'conversation_id': conversation_id}
    if parent_conversation_id is not None:
        identity['parent_conversation_id'] = parent_conversation_id
    meta = {'workbuddy': {key: value for key, value in identity.items() if value is not None}}
    if invocation_kind is not None:
        meta['invocation_kind'] = invocation_kind
    return meta


def _child_meta(**kwargs) -> dict:
    return _meta('subagent:general-purpose', CHILD_ID, agent_type='subagent', parent_conversation_id=MAIN_ID, **kwargs)


def _with_main(meta: dict) -> dict:
    """request_meta as WorkBuddyOpenAIProxy stamps it once the main conversation has been seen."""
    return {**meta, MAIN_CONVERSATION_ID_KEY: MAIN_ID}


def _agent_call(call_id: str, task: str) -> dict:
    return {'id': call_id, 'type': 'function', 'function': {'name': 'Agent', 'arguments': json.dumps({'prompt': task})}}


def _user(text: str) -> dict:
    return {'role': 'user', 'content': [{'type': 'text', 'text': text}]}


def _generated_call(call_id: str, name: str, arguments: dict, content: str = 'thinking</think:6124c78e>') -> dict:
    """Assistant message in the shape the Gateway returns to WorkBuddy."""
    return {
        'role': 'assistant',
        'content': content,
        'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}
        ],
    }


def _echo(message: dict) -> dict:
    """The same assistant message as WorkBuddy sends it back in the next request."""
    echoed = {
        'role': 'assistant',
        'content': [{'annotations': [], 'text': message['content'], 'type': 'text'}],
        'messageId': 'workbuddy-message-id',
    }
    if message.get('tool_calls'):
        echoed['tool_calls'] = [
            {**call, 'function': {**call['function'], 'arguments': json.loads(call['function']['arguments'])}}
            for call in message['tool_calls']
        ]
    return echoed


def _request_facts(messages: list[dict], meta: dict | None):
    return WORKBUDDY_POLICY.inspect_event(RequestEvent(tuple(messages), None, meta))


def test_header_projection_keeps_only_non_empty_identity_headers() -> None:
    meta = build_workbuddy_request_meta({
        'X-Agent-Purpose': 'conversation',
        'X-Conversation-Id': MAIN_ID,
        'x-parent-conversation-id': '',
        'X-Request-Id': 'request-1',
        'Authorization': 'Bearer secret',
        'X-Stainless-Os': 'Linux',
    })

    assert meta == {
        'request_id': 'request-1',
        'workbuddy': {'agent_purpose': 'conversation', 'conversation_id': MAIN_ID, 'request_id': 'request-1'},
    }
    assert build_workbuddy_request_meta({'Authorization': 'Bearer secret'}) is None


def test_echoed_text_blocks_and_structured_arguments_match_generated_message() -> None:
    generated = _generated_call('call-1', 'Bash', {'command': 'python calc.py', 'description': 'run'})
    request_identity = _request_facts([_echo(generated)], _meta('conversation')).message_identities[0].value
    response = ResponseEvent(RequestEvent((_user('go'),), None, _meta('conversation')), generated)
    response_identity = WORKBUDDY_POLICY.inspect_event(response).message_identities[0].value

    assert request_identity == response_identity


def test_title_purpose_is_an_identityless_session_title_chain() -> None:
    facts = _request_facts([_user('<session>task</session>')], _meta('conversation_topic', TITLE_ID))

    assert facts.membership is not None
    assert (facts.membership.chain_id, facts.membership.kind) == ('', 'session_title')
    assert facts.context_key == ('workbuddy', 'utility:session_title', True)


def test_child_purpose_declares_conversation_chain_and_parent() -> None:
    facts = _request_facts([CHILD_SYSTEM, _user(f'<user_query>{CHILD_PROMPT}</user_query>')], _child_meta())

    assert facts.membership is not None
    assert (facts.membership.chain_id, facts.membership.kind) == (CHILD_ID, 'subagent')
    assert dict(facts.membership.attributes) == {'parent_conversation_id': MAIN_ID, 'subagent_type': 'general-purpose'}
    assert facts.parent_claim is not None
    assert facts.parent_claim.parent_chain_id == MAIN_ID
    assert [candidate.value for candidate in facts.child_prompts] == [CHILD_PROMPT]


@pytest.mark.parametrize(
    ('meta', 'parent_chain_id'),
    [
        (_with_main(_child_meta()), 'main'),
        (_child_meta(), MAIN_ID),
        (
            _with_main(
                _meta('subagent:general-purpose', 'grandchild', agent_type='subagent', parent_conversation_id=CHILD_ID)
            ),
            CHILD_ID,
        ),
    ],
)
def test_child_of_the_main_conversation_claims_the_main_chain(meta: dict, parent_chain_id: str) -> None:
    facts = _request_facts([CHILD_SYSTEM, _user(f'<user_query>{CHILD_PROMPT}</user_query>')], meta)

    assert facts.parent_claim is not None
    assert facts.parent_claim.parent_chain_id == parent_chain_id


@pytest.mark.parametrize(
    ('meta', 'error'),
    [
        (None, 'has no request_meta.workbuddy'),
        ({'workbuddy': {'agent_type': 'main', 'conversation_id': MAIN_ID}}, 'missing the x-agent-purpose'),
        (_meta('context_summary_manual'), "Unknown WorkBuddy x-agent-purpose='context_summary_manual'"),
        (_meta('subagent:general-purpose'), "x-agent-purpose='subagent:general-purpose' conflicts"),
        (_meta('conversation', agent_type='subagent', parent_conversation_id=MAIN_ID), 'conflicts with'),
        (_meta('conversation', agent_type='subagent'), "expected x-agent-type='main'"),
    ],
)
def test_unregistered_or_inconsistent_identity_fails_loudly(meta: dict | None, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _request_facts([_user('task')], meta)


def test_summary_purpose_declares_summary_inference_boundary() -> None:
    meta = _meta('context_summary_max_token', invocation_kind='summary_inference')
    facts = _request_facts([SUMMARY_SYSTEM, _user('history'), _user('Summarize the conversation.')], meta)

    assert facts.invocation_kind == InvocationKind.SUMMARY_INFERENCE
    assert [(fact.rule_name, fact.segment_kind, fact.anchor.index) for fact in facts.boundary_facts] == [
        ('workbuddy_summary_inference', 'summary_inference', 2)
    ]


@pytest.mark.parametrize(
    ('purpose', 'invocation_kind'),
    [('context_summary_max_token', 'compaction_resume'), ('conversation', 'summary_inference')],
)
def test_invocation_kind_must_agree_with_summary_purpose(purpose: str, invocation_kind: str) -> None:
    with pytest.raises(ValueError, match='summary'):
        _request_facts([_user('task')], _meta(purpose, invocation_kind=invocation_kind))


@pytest.mark.parametrize(
    ('result', 'delivered'),
    [
        # wb 5.7/5.9 headless runs a run_in_background child to completion and returns its reply.
        ('73\n\n[Agent ID: agent-f071b51069f949ce]', True),
        (
            'Spawned successfully. agent_id: general-purpose-1@team name: general-purpose-1 team_name: team '
            'task_id: task-1 The agent is now running and will receive instructions via mailbox.',
            False,
        ),
    ],
)
def test_background_agent_delivery_follows_the_tool_result(result: str, delivered: bool) -> None:
    spawn = _generated_call('spawn-1', 'Agent', {'prompt': CHILD_PROMPT, 'run_in_background': True})
    messages = [_user('delegate'), _echo(spawn), {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': result}]

    facts = _request_facts(messages, _meta('conversation'))

    assert [(intent.tool_use_id, intent.child_prompt) for intent in facts.spawn_intents] == [('spawn-1', CHILD_PROMPT)]
    assert [(receipt.tool_use_id, receipt.delivered) for receipt in facts.delivery_receipts] == [('spawn-1', delivered)]


def test_retracted_task_notification_declares_history_rewrite_boundary() -> None:
    meta = {**_meta('conversation'), RETRACTED_TASK_NOTIFICATION_KEY: True}

    facts = _request_facts([_user('task'), {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'bg-done'}], meta)

    assert [(fact.rule_name, fact.segment_kind, fact.cause, fact.anchor.index) for fact in facts.boundary_facts] == [
        ('workbuddy_task_notification_retracted', 'render_restart', 'history_rewrite', 1)
    ]


def test_child_summary_stays_on_the_child_chain() -> None:
    meta = _meta(
        'context_summary_max_token',
        CHILD_ID,
        agent_type='subagent',
        parent_conversation_id=MAIN_ID,
        invocation_kind='summary_inference',
    )

    facts = _request_facts([SUMMARY_SYSTEM, _user('child history'), _user('Summarize.')], meta)

    assert facts.membership is not None
    assert facts.membership.chain_id == CHILD_ID
    assert facts.child_prompts == ()
    assert facts.boundary_facts[0].segment_kind == 'summary_inference'


def test_workbuddy_prefers_extended_responses() -> None:
    assert WORKBUDDY_POLICY.response_policy is ResponsePolicy.PREFER_EXTENDED_ELSE_LAST


def _record_title(assembler: TrajectoryAssembler) -> None:
    assembler.record_completion(
        [{'role': 'system', 'content': 'Generate a concise title'}, _user('<session>task</session>')],
        {'role': 'assistant', 'content': 'Task title'},
        request_meta=_meta('conversation_topic', TITLE_ID),
    )


@pytest.mark.parametrize('run_in_background', [False, True])
def test_child_reply_resolves_spawn_and_inline_join_on_main(run_in_background: bool) -> None:
    assembler = TrajectoryAssembler(scaffold_policy='workbuddy')
    _record_title(assembler)
    main_input = [MAIN_SYSTEM, _user('<user_query>Use one Agent child.</user_query>')]
    arguments = {'subagent_type': 'general-purpose', 'prompt': CHILD_PROMPT, 'run_in_background': run_in_background}
    spawn = _generated_call('spawn-1', 'Agent', arguments)
    assembler.record_completion(main_input, spawn, request_meta=_meta('conversation'))

    child_input = [CHILD_SYSTEM, _user(f'<user_query>{CHILD_PROMPT}</user_query>')]
    read = _generated_call('read-1', 'Read', {'file_path': '/workspace/chapter-1.txt'})
    assembler.record_completion(child_input, read, request_meta=_child_meta())
    read_result = {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'ANCHOR_1: approved'}
    child_answer = {'role': 'assistant', 'content': 'ANCHOR_1: approved'}
    assembler.record_completion([*child_input, _echo(read), read_result], child_answer, request_meta=_child_meta())

    spawn_result = {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': 'ANCHOR_1: approved\n\n[Agent ID: agent-1]'}
    assembler.record_completion(
        [*main_input, _echo(spawn), spawn_result],
        {'role': 'assistant', 'content': 'ANCHOR_1: approved'},
        request_meta=_meta('conversation'),
    )

    trajectory = assembler.freeze()
    child = trajectory.chains[CHILD_ID]
    title_chain_ids = [chain_id for chain_id, chain in trajectory.chains.items() if chain.kind == 'session_title']
    spawn_node = next(node for node in trajectory.nodes.values() if node.original_message == spawn)
    assert len(title_chain_ids) == 1
    assert trajectory.chains[title_chain_ids[0]].parent_chain_id is None
    assert child.parent_chain_id == 'main'
    assert child.spawn is not None
    assert (child.spawn.node, child.spawn.tool_use_id) == (spawn_node.node_id, 'spawn-1')
    assert child.join is not None
    assert (child.join.kind, child.join.tool_use_id) == ('inline', 'spawn-1')
    assert trajectory.nodes[child.join.node].original_message == spawn_result
    assert spawn_node.children == [child.join.node]


def test_summary_and_resume_boundaries_pair_on_the_main_chain() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='workbuddy')
    main_input = [MAIN_SYSTEM, _user('<user_query>Review the files.</user_query>')]
    read = _generated_call('read-1', 'Read', {'file_path': '/workspace/chapter-1.txt'}, content='read chapter 1')
    assembler.record_completion(main_input, read, request_meta=_meta('conversation'))
    read_result = {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'ANCHOR_1'}

    summary_input = [SUMMARY_SYSTEM, main_input[1], _echo(read), read_result, _user('Summarize the conversation.')]
    summary = {'role': 'assistant', 'content': 'Read chapter-1; ANCHOR_1 retained.'}
    assembler.record_completion(
        summary_input,
        summary,
        request_meta=_meta('context_summary_max_token', invocation_kind='summary_inference'),
    )

    resume_input = [
        MAIN_SYSTEM,
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': f'<conversation_history_summary>\n{summary["content"]}'},
                {'type': 'text', 'text': '<user_query>Please continue.</user_query>'},
            ],
        },
    ]
    resumed = _generated_call('read-2', 'Read', {'file_path': '/workspace/chapter-2.txt'}, content='read chapter 2')
    assembler.record_completion(
        resume_input, resumed, request_meta=_meta('conversation', invocation_kind='compaction_resume')
    )
    continued = {'role': 'assistant', 'content': 'ANCHOR_1 and ANCHOR_2'}
    assembler.record_completion(
        [*resume_input, _echo(resumed), {'role': 'tool', 'tool_call_id': 'read-2', 'content': 'ANCHOR_2'}],
        continued,
        request_meta=_meta('conversation'),
    )

    trajectory = assembler.freeze()
    boundaries = {
        node.original_message['content']: node.segment_boundary
        for node in trajectory.nodes.values()
        if node.generated_by == 'llm'
    }
    assert set(trajectory.chains) == {'main'}
    assert boundaries[read['content']] is None
    assert boundaries[summary['content']].kind == 'summary_inference'
    assert boundaries[summary['content']].evidence['boundary_rules'] == ['workbuddy_summary_inference']
    assert boundaries[resumed['content']].kind == 'compaction_resume'
    assert boundaries[resumed['content']].evidence['boundary_rules'] == ['workbuddy_compaction_resume']
    assert boundaries[continued['content']] is None


def test_retracted_task_notification_forks_into_a_history_rewrite_segment() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='workbuddy')
    main_input = [MAIN_SYSTEM, _user('<user_query>Run the background test.</user_query>')]
    foreground = _generated_call('fg-1', 'Bash', {'command': 'sleep 30; echo fg-done'}, content='foreground')
    assembler.record_completion(main_input, foreground, request_meta=_meta('conversation'))
    foreground_result = {'role': 'tool', 'tool_call_id': 'fg-1', 'content': 'fg-done'}
    history = [*main_input, _echo(foreground), foreground_result]

    notification = _user('<task-notification>\n<task-id>HEevJO</task-id>\n</task-notification>')
    read = _generated_call('read-1', 'TaskOutput', {'task_id': 'HEevJO', 'block': False}, content='read it')
    assembler.record_completion([*history, notification], read, request_meta=_meta('conversation'))
    # wb drops the notification from the next request's history.
    answer = {'role': 'assistant', 'content': '\\boxed{bg-done}'}
    assembler.record_completion(
        [*history, _echo(read), {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'bg-done'}],
        answer,
        request_meta={**_meta('conversation'), RETRACTED_TASK_NOTIFICATION_KEY: True},
    )

    trajectory = assembler.freeze()
    generated = {
        node.original_message.get('content'): node for node in trajectory.nodes.values() if node.generated_by == 'llm'
    }
    fork = next(node for node in trajectory.nodes.values() if node.original_message == foreground_result)
    boundary = generated[answer['content']].segment_boundary
    assert len(fork.children) == 2
    assert generated[read['content']].segment_boundary is None
    assert boundary is not None
    assert (boundary.kind, boundary.causes) == ('render_restart', ('history_rewrite',))
    assert boundary.evidence['boundary_rules'] == ['workbuddy_task_notification_retracted']


def test_main_parent_survives_the_same_prompt_spawned_by_another_child() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='workbuddy')
    review, coordinate = 'Review the code', 'Coordinate a review'
    assembler.record_completion(
        [MAIN_SYSTEM, _user('<user_query>Delegate to a reviewer and a coordinator.</user_query>')],
        {
            'role': 'assistant',
            'content': 'delegate',
            'tool_calls': [_agent_call('spawn-a', review), _agent_call('spawn-b', coordinate)],
        },
        request_meta=_with_main(_meta('conversation')),
    )
    reviewer = [CHILD_SYSTEM, _user(f'<user_query>{review}</user_query>')]

    def child_meta(conversation_id: str, parent: str, purpose: str = 'subagent:general-purpose') -> dict:
        return _with_main(_meta(purpose, conversation_id, agent_type='subagent', parent_conversation_id=parent))

    assembler.record_completion(
        reviewer, {'role': 'assistant', 'content': 'Review A complete.'}, request_meta=child_meta('child-a', MAIN_ID)
    )
    # A coordinator type has its own system prompt; general-purpose children cannot spawn Agents.
    assembler.record_completion(
        [
            {'role': 'system', 'content': 'You coordinate code reviews.'},
            _user(f'<user_query>{coordinate}</user_query>'),
        ],
        {'role': 'assistant', 'content': 'delegate another review', 'tool_calls': [_agent_call('spawn-c', review)]},
        request_meta=child_meta('child-b', MAIN_ID, 'subagent:coordinator'),
    )
    assembler.record_completion(
        reviewer, {'role': 'assistant', 'content': 'Review C complete.'}, request_meta=child_meta('child-c', 'child-b')
    )

    chains = assembler.freeze().chains
    assert (chains['child-a'].parent_chain_id, chains['child-a'].spawn.tool_use_id) == ('main', 'spawn-a')
    assert (chains['child-b'].parent_chain_id, chains['child-b'].spawn.tool_use_id) == ('main', 'spawn-b')
    assert (chains['child-c'].parent_chain_id, chains['child-c'].spawn.tool_use_id) == ('child-b', 'spawn-c')


def test_summary_copy_of_agent_result_does_not_move_the_join() -> None:
    assembler = TrajectoryAssembler(scaffold_policy='workbuddy')
    main_input = [MAIN_SYSTEM, _user('<user_query>Use one Agent child.</user_query>')]
    spawn = _generated_call('spawn-1', 'Agent', {'prompt': CHILD_PROMPT})
    assembler.record_completion(main_input, spawn, request_meta=_meta('conversation'))
    assembler.record_completion(
        [CHILD_SYSTEM, _user(f'<user_query>{CHILD_PROMPT}</user_query>')],
        {'role': 'assistant', 'content': 'child answer'},
        request_meta=_child_meta(),
    )
    spawn_result = {'role': 'tool', 'tool_call_id': 'spawn-1', 'content': 'child answer'}
    assembler.record_completion(
        [*main_input, _echo(spawn), spawn_result],
        {'role': 'assistant', 'content': 'main answer'},
        request_meta=_meta('conversation'),
    )
    assembler.record_completion(
        [SUMMARY_SYSTEM, main_input[1], _echo(spawn), spawn_result, _user('Summarize.')],
        {'role': 'assistant', 'content': 'summary'},
        request_meta=_meta('context_summary_max_token', invocation_kind='summary_inference'),
    )

    trajectory = assembler.freeze()
    join_node = trajectory.chains[CHILD_ID].join.node

    assert trajectory.nodes[trajectory.nodes[join_node].parent].original_message == spawn
