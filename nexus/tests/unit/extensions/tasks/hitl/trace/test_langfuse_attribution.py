"""Placing an interleaved LLM call with the agent it belongs to.

A worker and the subagents it spawns share one trace label and their calls arrive
interleaved. Attribution reads the tool_call ids of a call's history as a sequence and
matches the record whose own sequence it begins with, falling back to the user messages
for a call that has issued no tool yet.
"""

from nexus.extensions.tasks.hitl.trace.langfuse.attribution import (
    LangfuseAgentRecord,
    has_tool_call,
    record_for,
    tool_call_ids,
    user_fingerprints,
)


def assistant(*call_ids: str) -> dict:
    """An assistant message issuing the given tool_call ids."""
    return {
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': 'Bash', 'arguments': '{}'}} for call_id in call_ids
        ],
    }


def user(text: str) -> dict:
    return {'role': 'user', 'content': text}


def system(text: str = 'sys') -> dict:
    return {'role': 'system', 'content': text}


def record(name: str, messages: list[dict]) -> LangfuseAgentRecord:
    """A record whose identity already covers one call's history."""
    made = LangfuseAgentRecord(name=name)
    made.extend_identity(messages)
    return made


def test_tool_call_ids_reads_only_the_assistant_side_in_order() -> None:
    """A tool result repeats an id its assistant message already carries."""
    history = [
        system(),
        user('go'),
        assistant('a', 'b'),
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'x'},
        {'role': 'tool', 'tool_call_id': 'b', 'content': 'y'},
        assistant('c'),
    ]
    assert tool_call_ids(history) == ('a', 'b', 'c')


def test_a_call_is_placed_with_the_record_its_ids_extend() -> None:
    records = [
        record('main', [system(), user('build'), assistant('m1')]),
        record('sub', [system(), user('explore'), assistant('s1')]),
    ]
    call = [
        system(),
        user('explore'),
        assistant('s1'),
        {'role': 'tool', 'tool_call_id': 's1', 'content': 'x'},
        assistant('s2'),
    ]

    assert record_for(call, records).name == 'sub'


def test_a_call_with_no_id_yet_is_placed_by_its_user_message() -> None:
    """A subagent's opening call has issued nothing, so its request identifies it."""
    records = [record('main', [system(), user('build'), assistant('m1')])]
    opening = [system(), user('explore the tree')]

    assert record_for(opening, records) is None, 'a new opening belongs to no existing record'

    records.append(record('sub', opening))
    second = [system(), user('explore the tree'), assistant('s1')]
    assert record_for(second, records).name == 'sub', "the subagent's own second call rejoins it"


def test_the_longest_matching_record_wins() -> None:
    """One agent recorded at two points matches twice; the later point is right."""
    early = record('early', [system(), user('go'), assistant('a')])
    late = record(
        'late',
        [system(), user('go'), assistant('a'), {'role': 'tool', 'tool_call_id': 'a', 'content': 'x'}, assistant('b')],
    )
    call = [system(), user('go'), assistant('a'), assistant('b'), assistant('c')]

    assert record_for(call, [early, late]).name == 'late'


def test_a_retry_does_not_shorten_a_records_identity() -> None:
    """A retried call re-sends a shorter history; the record keeps the longer one.

    Shortening would let the record match every agent that shares only its opening.
    """
    made = record('agent', [system(), user('go'), assistant('a'), assistant('b')])
    made.extend_identity([system(), user('go'), assistant('a')])

    assert made.tool_call_ids == ('a', 'b')


def test_the_first_call_of_the_first_agent_matches_nothing() -> None:
    assert record_for([system(), user('build')], []) is None


def test_user_fingerprints_ignores_non_user_messages() -> None:
    history = [system(), user('one'), assistant('a'), user('two')]
    assert len(user_fingerprints(history)) == 2


def _reply(*, tool_calls: bool) -> dict:
    """An outputs payload whose assistant response does or does not issue a tool call."""
    message = {'role': 'assistant', 'content': None if tool_calls else 'a title'}
    if tool_calls:
        message['tool_calls'] = [{'id': 'x', 'type': 'function', 'function': {'name': 'Bash', 'arguments': '{}'}}]
    return {'choices': [{'message': message}]}


def test_a_call_is_an_agent_turn_when_its_history_carries_an_id() -> None:
    assert has_tool_call([system(), user('go'), assistant('a')], _reply(tool_calls=False))


def test_a_call_is_an_agent_turn_when_its_response_issues_a_tool() -> None:
    assert has_tool_call([system(), user('go')], _reply(tool_calls=True))


def test_a_prose_only_reply_with_no_prior_tool_is_not_an_agent_turn() -> None:
    """A title or a summary answers in prose and issues nothing -- a utility call."""
    assert not has_tool_call([system(), user('generate a title')], _reply(tool_calls=False))
