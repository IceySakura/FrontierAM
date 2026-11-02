import json

import pytest

from nexus.extensions.agents.claude_code.stream_json import ParentSessionState, parse_transcript


def text(value, *, session='root', parent=None):
    return {
        'type': 'assistant',
        'session_id': session,
        'parent_tool_use_id': parent,
        'message': {'content': [{'type': 'text', 'text': value}]},
    }


@pytest.mark.parametrize('result', [False, True])
def test_parent_selection_preserves_child_trajectory(result):
    events = [
        text('unrelated before init', session='other'),
        {'type': 'system', 'subtype': 'init', 'session_id': 'root'},
        text('TASK_INFEASIBLE'),
    ]
    if result:
        events.append({'type': 'result', 'session_id': 'root', 'subtype': 'success', 'result': 'root result'})
    events += [
        text('child waiting', parent='tool-1'),
        {
            'type': 'result',
            'session_id': 'root',
            'parent_tool_use_id': 'tool-1',
            'subtype': 'error_max_turns',
            'result': 'child result',
        },
        text('foreign', session='other'),
    ]
    trajectory, answer, subtype = parse_transcript('\n'.join(map(json.dumps, events)))
    assert [x['text'] for x in trajectory] == [
        'unrelated before init',
        'TASK_INFEASIBLE',
        'child waiting',
        'foreign',
    ]
    assert answer == ('root result' if result else 'TASK_INFEASIBLE')
    assert subtype == ('success' if result else None)


def test_late_assistant_cannot_override_authoritative_root_result():
    events = [
        {'type': 'result', 'session_id': 'root', 'subtype': 'error_during_execution', 'result': 'API failure'},
        text('TASK_DONE'),
    ]
    _, answer, subtype = parse_transcript('\n'.join(map(json.dumps, events)))
    assert answer == 'API failure'
    assert subtype == 'error_during_execution'


def test_child_only_output_is_not_parent_answer():
    trajectory, answer, subtype = parse_transcript(json.dumps(text('TASK_DONE', parent='child')))
    assert len(trajectory) == 1
    assert answer is subtype is None


@pytest.mark.parametrize('value', ['mentions TASK_DONE inline.', '> TASK_DONE', '```text\nTASK_DONE\n```'])
def test_nonterminal_marker_does_not_select_drain(value):
    state = ParentSessionState('root')
    state.feed(text(value))
    assert state.terminal_marker is None


def test_pending_tools_prevent_terminal_selection():
    state = ParentSessionState('root')
    event = text('TASK_DONE')
    event['message']['content'].append({'type': 'tool_use', 'id': 'one', 'name': 'Bash', 'input': {}})
    state.feed(event)
    assert state.pending_tools == {'one'}
    assert state.terminal_marker is None
