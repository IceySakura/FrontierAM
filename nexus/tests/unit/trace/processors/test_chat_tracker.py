from __future__ import annotations

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor
from nexus.trace.types import Span


def _completion_span(inputs: dict, outputs: dict | None, span_id: str) -> Span:
    return Span(
        trace_id='trace',
        span_id=span_id,
        parent_span_id='',
        name='openai_completion',
        start_time_unix_nano=0,
        attributes={'inputs': inputs, 'outputs': outputs},
    )


def _track_completion(tracker: ChatTrackerProcessor, inputs: dict, outputs: dict | None, span_id: str) -> None:
    tracker.on_end(_completion_span(inputs, outputs, span_id), SpanEndEvent())


def test_branch_last_merges_assistant_tool_call_output_with_history() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'report env'}
    tools = [{'type': 'function', 'function': {'name': 'exec'}}]

    first_tool_call = {
        'id': 'tooluse_abc',
        'function': {'name': 'exec', 'arguments': '{"command": "env | sort"}'},
        'type': 'function',
        'index': 0,
    }
    replayed_tool_call = {
        'id': 'tooluseabc',
        'type': 'function',
        'function': {'name': 'exec', 'arguments': '{"command":"env | sort"}'},
    }

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {'tools': tools}},
        {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [first_tool_call]}}]},
        'first',
    )
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {'role': 'assistant', 'content': None, 'tool_calls': [replayed_tool_call]},
                {'role': 'tool', 'tool_call_id': 'tooluseabc', 'content': 'PATH=/usr/bin'},
            ],
            'kwargs': {'tools': tools},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'second',
    )

    result = tracker.get_result()

    assert len(result['trajectories']) == 1
    messages = result['trajectories'][0]['messages']
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert messages[2]['tool_calls'][0]['id'] == 'tooluseabc'
    assert messages[3]['tool_call_id'] == 'tooluseabc'
    assert messages[-1]['content'] == 'done'


def test_same_tool_call_id_merges_after_bash_cwd_normalization() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'check version'}
    produced_tool_call = {
        'id': 'call-version',
        'type': 'function',
        'function': {
            'name': 'Bash',
            'arguments': '{"command":"cd /workspace && ./executable version","description":"Check version"}',
        },
        'index': 0,
    }
    replayed_tool_call = {
        'id': 'call-version',
        'type': 'function',
        'function': {
            'name': 'Bash',
            'arguments': '{"command":"./executable version","description":"Check version"}',
        },
    }

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'tool_calls': [produced_tool_call]}}]},
        'first',
    )
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {'role': 'assistant', 'content': '', 'tool_calls': [replayed_tool_call]},
                {'role': 'tool', 'tool_call_id': 'call-version', 'content': 'v1.0.0'},
            ],
            'kwargs': {},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'second',
    )

    assert len(tracker.get_result()['trajectories']) == 1


def test_same_tool_call_id_merges_after_edit_default_injection() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'update file'}
    produced_tool_call = {
        'id': 'call-edit',
        'type': 'function',
        'function': {
            'name': 'Edit',
            'arguments': '{"file_path":"/workspace/a.py","old_string":"old","new_string":"new"}',
        },
    }
    replayed_tool_call = {
        'id': 'call-edit',
        'type': 'function',
        'function': {
            'name': 'Edit',
            'arguments': '{"replace_all":false,"file_path":"/workspace/a.py","old_string":"old","new_string":"new"}',
        },
    }

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [produced_tool_call]}}]},
        'first',
    )
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {'role': 'assistant', 'content': '', 'tool_calls': [replayed_tool_call]},
                {'role': 'tool', 'tool_call_id': 'call-edit', 'content': 'updated'},
            ],
            'kwargs': {},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'second',
    )

    assert len(tracker.get_result()['trajectories']) == 1


def test_same_tool_call_id_with_different_function_names_remains_separate() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'inspect workspace'}
    bash_call = {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': 'Bash', 'arguments': '{"command":"pwd"}'},
    }
    read_call = {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': 'Read', 'arguments': '{"file_path":"/workspace/a.py"}'},
    }

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [bash_call]}}]},
        'first',
    )
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {'role': 'assistant', 'content': '', 'tool_calls': [read_call]},
                {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'contents'},
            ],
            'kwargs': {},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'second',
    )

    assert len(tracker.get_result()['trajectories']) == 2


def test_tree_keeps_incremental_input_nodes_and_only_the_leaf_history() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'report env'}
    assistant = {'role': 'assistant', 'content': 'checking'}
    tool = {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'PATH=/usr/bin'}
    first_tools = [{'type': 'function', 'function': {'name': 'exec'}}]
    second_tools = [{'type': 'function', 'function': {'name': 'exec'}, 'description': 'latest'}]

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {'tools': first_tools, 'tool_choice': 'auto'}},
        {'choices': [{'message': assistant, 'finish_reason': 'stop'}]},
        'first',
    )
    _track_completion(
        tracker,
        {'messages': [system, user, assistant, tool], 'kwargs': {'tools': second_tools, 'tool_choice': 'required'}},
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}]},
        'second',
    )

    path = []
    node = tracker.root
    while node.children:
        assert len(node.children) == 1
        node = node.children[0]
        path.append(node)

    assert [node.message['role'] for node in path] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert path[3].message == tool

    llm_nodes = [node for node in path if node.generated_by == 'llm']
    assert len(llm_nodes) == 2
    assert llm_nodes[0].generation_detail == {'finish_reason': 'stop', 'tool_choice': 'auto'}
    assert llm_nodes[1].generation_detail == {
        'input_messages': [system, user, assistant, tool],
        'output_message': {'role': 'assistant', 'content': 'done'},
        'finish_reason': 'stop',
        'tool_choice': 'required',
        'tools': second_tools,
    }
    assert llm_nodes[1].generation_detail['output_message'] is llm_nodes[1].message
    assert 'input_messages' not in llm_nodes[0].generation_detail
    assert 'input_messages' in llm_nodes[1].generation_detail
    assert all('output' not in node.generation_detail for node in llm_nodes)


def test_dump_trajectories_handles_conversation_deeper_than_recursion_limit() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    messages = [{'role': 'user', 'content': 'Complete a long-running task.'}]
    turn_count = 600

    for turn in range(turn_count):
        assistant = {'role': 'assistant', 'content': f'step {turn}'}
        _track_completion(
            tracker,
            {'messages': list(messages), 'kwargs': {}},
            {'choices': [{'message': assistant}]},
            f'turn-{turn}',
        )
        messages.append(assistant)
        if turn < turn_count - 1:
            messages.append({'role': 'tool', 'tool_call_id': f'call-{turn}', 'content': f'output {turn}'})

    result = tracker.get_result()

    assert len(messages) == turn_count * 2
    assert result['trajectories'][0]['messages'] == messages


def test_dump_trajectories_preserves_branch_insertion_order() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')

    for branch in ('first', 'second'):
        _track_completion(
            tracker,
            {'messages': [{'role': 'user', 'content': branch}], 'kwargs': {}},
            {'choices': [{'message': {'role': 'assistant', 'content': f'{branch} response'}}]},
            branch,
        )

    trajectories = tracker.get_result()['trajectories']

    assert [trajectory['messages'][-1]['content'] for trajectory in trajectories] == [
        'first response',
        'second response',
    ]


def test_missing_assistant_output_does_not_create_trajectory() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    user = {'role': 'user', 'content': 'trigger provider error'}

    _track_completion(tracker, {'messages': [user], 'kwargs': {}}, None, 'error')

    assert tracker.root.children == []
    assert tracker.get_result()['trajectories'] == []


def test_failed_retries_before_success_export_only_successful_trajectory() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    user = {'role': 'user', 'content': 'retry provider call'}
    inputs = {'messages': [user], 'kwargs': {}}

    _track_completion(tracker, inputs, None, 'retry-1')
    _track_completion(tracker, inputs, None, 'retry-2')
    _track_completion(
        tracker,
        inputs,
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'success',
    )

    trajectories = tracker.get_result()['trajectories']
    assert len(trajectories) == 1
    assert trajectories[0]['messages'] == [user, {'role': 'assistant', 'content': 'done'}]


def test_output_without_role_preserves_previous_export_behavior() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    user = {'role': 'user', 'content': 'hello'}

    _track_completion(
        tracker,
        {'messages': [user], 'kwargs': {}},
        {'choices': [{'message': {'content': 'done'}}]},
        'missing-role',
    )

    output_node = tracker.root.children[0].children[0]
    assert output_node.message == {'role': 'assistant', 'content': 'done'}
    assert tracker.get_result()['trajectories'][0]['messages'] == [user, {'content': 'done'}]


def test_tool_only_null_content_exports_empty_string_content_key() -> None:
    """Tool-only assistant turns must keep ``content: ''`` (SFT schema), not omit the key."""
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    user = {'role': 'user', 'content': 'run'}
    tool_call = {
        'id': 'call_abc',
        'type': 'function',
        'function': {'name': 'exec', 'arguments': '{}'},
    }

    _track_completion(
        tracker,
        {'messages': [user], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [tool_call]}}]},
        'tool-only',
    )

    asst = tracker.get_result()['trajectories'][0]['messages'][-1]
    assert 'content' in asst
    assert asst['content'] == ''
    assert asst['tool_calls'] == [tool_call]


def test_empty_and_null_assistant_content_do_not_merge_without_tool_calls() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'report env'}

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': ''}}]},
        'first',
    )
    _track_completion(
        tracker,
        {'messages': [system, user, {'role': 'assistant', 'content': None}], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'second',
    )

    result = tracker.get_result()

    assert len(result['trajectories']) == 2


def test_finish_reason_recorded_in_extra_turns() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'hello'}

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {'tool_choice': 'auto'}},
        {'choices': [{'message': {'role': 'assistant', 'content': 'hi'}, 'finish_reason': 'stop'}]},
        'first',
    )

    traj = tracker.get_result()['trajectories'][0]
    # messages stay standard OpenAI-compatible: no metadata keys injected.
    assistant_msg = traj['messages'][-1]
    assert assistant_msg['role'] == 'assistant'
    assert 'finish_reason' not in assistant_msg
    assert 'tool_choice' not in assistant_msg
    # per-turn metadata lives under extra.turns, aligned via index.
    turns = traj['extra']['turns']
    assert len(turns) == 1
    assert turns[0]['index'] == 2  # system(0), user(1), assistant(2)
    assert turns[0]['response']['finish_reason'] == 'stop'
    assert turns[0]['request']['tool_choice'] == 'auto'
    assert traj['messages'][turns[0]['index']]['role'] == 'assistant'


def test_finish_reason_recorded_for_every_turn() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'do task'}
    a1 = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{}'}}],
    }
    tool1 = {'role': 'tool', 'tool_call_id': 'c1', 'content': 'output1'}

    # turn 1 -> tool_calls
    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {'tool_choice': 'auto'}},
        {'choices': [{'message': a1, 'finish_reason': 'tool_calls'}]},
        'first',
    )
    # turn 2 -> stop (final answer)
    _track_completion(
        tracker,
        {'messages': [system, user, a1, tool1], 'kwargs': {'tool_choice': 'auto'}},
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}]},
        'second',
    )

    traj = tracker.get_result()['trajectories'][0]
    roles = [m['role'] for m in traj['messages']]
    assert roles == ['system', 'user', 'assistant', 'tool', 'assistant']

    turns = traj['extra']['turns']
    # both assistant turns recorded, in order, with correct indices
    assert len(turns) == 2
    assert turns[0]['index'] == 2
    assert turns[0]['response']['finish_reason'] == 'tool_calls'
    assert turns[1]['index'] == 4
    assert turns[1]['response']['finish_reason'] == 'stop'


def test_missing_finish_reason_does_not_add_turn() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'hello'}

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': 'hi'}}]},
        'first',
    )

    traj = tracker.get_result()['trajectories'][0]
    assistant_msg = traj['messages'][-1]
    assert assistant_msg['role'] == 'assistant'
    assert 'finish_reason' not in assistant_msg
    # no finish_reason and no tool_choice -> no extra block at all
    assert 'extra' not in traj


def test_provider_call_attributes_do_not_expand_chat_tracker_contract() -> None:
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    span = _completion_span(
        {'messages': [{'role': 'user', 'content': 'hello'}], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'default-off',
    )
    span.attributes['provider_call'] = {
        'provider': 'openai',
        'protocol': 'responses',
        'model': 'gpt-test',
        'endpoint': 'https://provider.test/v1/responses',
        'attempts': [{'attempt': 1, 'status': 'success'}],
    }

    tracker.on_end(span, SpanEndEvent())

    assert 'calls' not in tracker.get_result()
    assert 'calls' not in tracker.snapshot_result()
    assert tracker.get_result()['trajectories']


def test_reasoning_id_merges_response_recorded_before_tool_calls_restored() -> None:
    """Responses path: same response seen without then with tool_calls must not fork.

    Repro of the Codex 0.114 + gpt-5.6-sol shape (R2_codex0114_evohard3): the response is first
    recorded straight off ``choices[0].message`` — text only, ``reasoning_content: ''``, no
    ``tool_calls`` — and then again from the next request's replayed history, where the Responses
    ``function_call`` item has become ``tool_calls``. ``_tool_call_identities`` returns None for
    the first shape, so only the reasoning-item id can bridge them.

    Before the fix this produced 2 trajectories, the first truncated at the fork and showing an
    assistant turn with no tool_calls despite the model having called one — a corrupt training
    sample that teaches the model to stop where it should have used a tool.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are a coding agent.'}
    user = {'role': 'user', 'content': 'build the dashboard'}
    reasoning_item = {
        'type': 'reasoning',
        'id': 'rs_095761da3dadebf0006a9848de2dbc8196a73ce5a60d9b0397',
        'encrypted_content': 'gAAAAAB-opaque',
        'summary': [],
    }
    tool_call = {
        'id': 'call_xBUIRU8nAdtGCqzocVfS9VaJ',
        'type': 'function',
        'function': {'name': 'update_plan', 'arguments': '{"plan":[]}'},
    }
    assistant_text = 'I will build it module by module.'

    # Shape 1: raw response — no tool_calls yet, reasoning_content present but empty.
    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {'tool_choice': 'auto'}},
        {
            'choices': [
                {
                    'finish_reason': 'stop',
                    'message': {
                        'role': 'assistant',
                        'content': assistant_text,
                        'reasoning_content': '',
                        'reasoning_items': [reasoning_item],
                    },
                }
            ]
        },
        'first',
    )
    # Shape 2: replayed history — tool_calls restored, reasoning_content dropped.
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {
                    'role': 'assistant',
                    'content': assistant_text,
                    'reasoning_items': [reasoning_item],
                    'tool_calls': [tool_call],
                },
                {'role': 'tool', 'tool_call_id': tool_call['id'], 'content': 'plan updated'},
            ],
            'kwargs': {'tool_choice': 'auto'},
        },
        {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'done'}}]},
        'second',
    )

    trajectories = tracker.get_result()['trajectories']
    assert len(trajectories) == 1
    messages = trajectories[0]['messages']
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'tool', 'assistant']
    # The merged assistant turn must keep the tool call the model actually made.
    assert messages[2]['tool_calls'] == [tool_call]


def test_reasoning_id_does_not_merge_turns_with_different_content() -> None:
    """A shared reasoning id must not collapse genuinely different assistant turns.

    Guards the fast path from over-merging: role/content still have to match, mirroring how
    _tool_call_identities keeps the role/content check after an id hit.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'go'}
    reasoning_item = {'type': 'reasoning', 'id': 'rs_shared', 'summary': []}

    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {
            'choices': [
                {'message': {'role': 'assistant', 'content': 'first answer', 'reasoning_items': [reasoning_item]}}
            ]
        },
        'first',
    )
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {'role': 'assistant', 'content': 'DIFFERENT answer', 'reasoning_items': [reasoning_item]},
            ],
            'kwargs': {},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'second'}}]},
        'second',
    )

    assert len(tracker.get_result()['trajectories']) == 2


def test_snapshot_without_reasoning_id_still_merges_when_tool_calls_arrive() -> None:
    """No reasoning id anywhere: the structural anchor must still merge the two shapes.

    This is the exact shape found at every divergence point of R4 (post-_reasoning_identities
    rerun): the bare snapshot carries neither ``reasoning_items`` ids nor ``tool_calls``, so the
    id-based fast path cannot fire and the tree still forked. Merging is driven instead by the
    structure -- same parent, same role/content, and the snapshot is a childless leaf missing
    exactly the tool_calls.

    An earlier version of this test asserted 2 trajectories, on the mistaken assumption that a
    missing reasoning id should preserve the fork. Verified against the exports: bare assistant
    messages appear only as trajectory tails (349 occurrences, 0 mid-conversation), and each such
    tail matched a longer trajectory that differed only by the tool_calls -- i.e. these are always
    the same response, never two turns.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'go'}
    tool_call = {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': 'Bash', 'arguments': '{"command":"pwd"}'},
    }
    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {
            'choices': [
                {'message': {'role': 'assistant', 'content': 'same text', 'reasoning_items': [{'type': 'reasoning'}]}}
            ]
        },
        'first',
    )
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {
                    'role': 'assistant',
                    'content': 'same text',
                    'reasoning_items': [{'type': 'reasoning'}],
                    'tool_calls': [tool_call],
                },
                {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'ok'},
            ],
            'kwargs': {},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
        'second',
    )

    trajectories = tracker.get_result()['trajectories']
    assert len(trajectories) == 1
    messages = trajectories[0]['messages']
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert messages[2]['tool_calls'] == [tool_call]


def test_snapshot_with_existing_continuation_is_not_upgraded() -> None:
    """A bare assistant that already has a continuation is a real turn boundary, not a snapshot.

    Guards the structural merge: only childless leaves may be upgraded, so a genuine
    ``finish_reason='stop'`` turn that the agent continued from keeps its own branch.
    """
    tracker = ChatTrackerProcessor(tools_mode='branch_last')
    system = {'role': 'system', 'content': 'You are an agent.'}
    user = {'role': 'user', 'content': 'go'}
    tool_call = {
        'id': 'call-9',
        'type': 'function',
        'function': {'name': 'Bash', 'arguments': '{"command":"ls"}'},
    }

    # Turn 1 ends with a plain assistant answer.
    _track_completion(
        tracker,
        {'messages': [system, user], 'kwargs': {}},
        {'choices': [{'message': {'role': 'assistant', 'content': 'shared'}}]},
        'first',
    )
    # The agent continues from it, so that node gains a child and stops being a snapshot.
    _track_completion(
        tracker,
        {
            'messages': [system, user, {'role': 'assistant', 'content': 'shared'}, {'role': 'user', 'content': 'more'}],
            'kwargs': {},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'second'}}]},
        'second',
    )
    # A later same-text assistant WITH tool_calls must not collapse into that settled node.
    _track_completion(
        tracker,
        {
            'messages': [
                system,
                user,
                {'role': 'assistant', 'content': 'shared', 'tool_calls': [tool_call]},
                {'role': 'tool', 'tool_call_id': 'call-9', 'content': 'ok'},
            ],
            'kwargs': {},
        },
        {'choices': [{'message': {'role': 'assistant', 'content': 'third'}}]},
        'third',
    )

    assert len(tracker.get_result()['trajectories']) == 2
