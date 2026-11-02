"""The four loop parts, and the invariants a hand-written loop would break.

What these cover is not "does the function return the right string" but the three
ways a loop goes wrong silently:

* a tool call left unanswered -- an assistant message with N calls and fewer than N
  ``role: tool`` replies is a malformed conversation, and the endpoint rejects it on
  the *next* request, several turns after the mistake;
* compaction dropping an assistant while keeping the tool results that answered it,
  which is the same malformation arriving from the other direction;
* ``kw_only`` on the result type -- without it positional matching binds the wrong
  field and the type can never gain a field.
"""

import asyncio
import json

from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.core import ToolResponse
from nexus.extensions.tasks.hitl.user_agent import compact as compact_mod
from nexus.extensions.tasks.hitl.user_agent import queue as queue_mod
from nexus.extensions.tasks.hitl.user_agent.dispatch import (
    Dispatcher,
    ExtendedToolResponse,
    call_args,
    call_id,
    call_name,
    check_handlers,
    content_of,
    read_response,
    text_of,
    tool_message,
    with_error,
)
from nexus.extensions.tasks.hitl.user_agent.prompts import append_reminder, reminder


def make_call(name: str, args: Any, call_ref: str = 'c1') -> Any:
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_ref, function=SimpleNamespace(name=name, arguments=raw))


class Message:
    def __init__(self, calls: list[Any], content: str | None = None) -> None:
        self.tool_calls = calls
        self.content = content
        self.role = 'assistant'

    def model_dump(self, **_: Any) -> dict[str, Any]:
        dumped: dict[str, Any] = {'role': 'assistant'}
        if self.content is not None:
            dumped['content'] = self.content
        if self.tool_calls:
            dumped['tool_calls'] = [{'id': c.id, 'function': {'name': c.function.name}} for c in self.tool_calls]
        return dumped


def response_with(calls: list[Any], content: str | None = None) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=Message(calls, content))])


# -- reading a response --------------------------------------------------


def test_a_response_becomes_a_history_entry_and_its_calls() -> None:
    """Both shapes a turn arrives in, since the loop branches on which it got.

    The prose-only turn must read as "no calls" rather than crash: that is the
    branch that nudges instead of advancing the run.
    """
    message, calls = read_response(response_with([make_call('read', {'path': '/a'})]))
    prose, no_calls = read_response(response_with([], content='I think I am done.'))

    assert message['role'] == 'assistant'
    assert len(calls) == 1
    assert call_name(calls[0]) == 'read'
    assert call_args(calls[0]) == {'path': '/a'}
    assert no_calls == []
    assert prose['content'] == 'I think I am done.'


def test_malformed_arguments_raise_rather_than_reading_as_empty() -> None:
    """Returning {} would run the tool as though the model passed nothing.

    Absent arguments are the case that legitimately *is* empty, so the two are
    asserted together -- collapsing them is what makes the raise look optional.
    """
    with pytest.raises(json.JSONDecodeError):
        call_args(make_call('read', '{not json'))

    assert call_args(make_call('accept', '')) == {}


def test_a_tool_message_carries_the_id_of_the_call_it_answers() -> None:
    call = make_call('read', {}, 'call_abc')

    assert tool_message(call, 'result')['tool_call_id'] == 'call_abc'


# -- the result type -----------------------------------------------------


def test_the_result_type_is_keyword_only_and_frozen() -> None:
    """Both consequences of one declaration, which is what a rewrite would drop.

    Without ``kw_only``, ``case ExtendedToolResponse(x)`` binds ``tool_call_id``,
    so a loop author writing the natural thing sends a call id to the worker as
    the message body; and a subclass adding a field raises at import once a base
    field has a default. Without ``frozen`` a handler can rewrite a result after
    the dispatcher built it.
    """

    @dataclass(frozen=True, kw_only=True)
    class WithMore(ExtendedToolResponse):
        extra: str

    def match_positionally(result: ExtendedToolResponse) -> None:
        match result:
            case ExtendedToolResponse(_):
                pass

    assert ExtendedToolResponse.__match_args__ == ()
    assert WithMore(tool_call_id='c', name='n', content='x', extra='e').extra == 'e'

    with pytest.raises(TypeError, match='positional sub-patterns'):
        match_positionally(ExtendedToolResponse(tool_call_id='c1', name='read', content='body'))

    with pytest.raises(FrozenInstanceError):
        ExtendedToolResponse(tool_call_id='c1', name='read', content='body').content = 'rewritten'  # type: ignore[misc]


def test_a_failed_call_is_distinguishable_from_one_that_returned_nothing() -> None:
    failed = ExtendedToolResponse(tool_call_id='c', name='n', content='<error>x</error>')
    empty = ExtendedToolResponse(tool_call_id='c', name='n', content='', response=ToolResponse(output=[]))

    assert failed.failed
    assert not empty.failed


# -- dispatch ------------------------------------------------------------


class Executor:
    """The parts of ToolExecutor dispatch uses."""

    def __init__(self, *, result: Any = 'ok', raises: Exception | None = None, invalid: Exception | None = None):
        self.result = result
        self.raises = raises
        self.invalid = invalid
        self.validated: list[tuple[str, dict]] = []
        self.called: list[tuple[str, dict]] = []

    def validate_tool_call(self, name: str, params: dict) -> str:
        self.validated.append((name, params))
        if self.invalid is not None:
            raise self.invalid
        return name

    async def call(self, name: str, params: dict) -> ToolResponse:
        self.called.append((name, params))
        if self.raises is not None:
            raise self.raises
        return ToolResponse(output=[{'type': 'text', 'text': self.result}])


def dispatch(executor: Any, call: Any, handlers: dict | None = None) -> ExtendedToolResponse:
    return asyncio.run(Dispatcher(executor=executor, handlers=handlers or {}).dispatch(call))


def test_arguments_are_validated_before_the_tool_runs() -> None:
    """This is where the Literal types on a tool's input model take effect."""
    executor = Executor()

    dispatch(executor, make_call('read', {'path': '/a'}))

    assert executor.validated == [('read', {'path': '/a'})]
    assert executor.called == [('read', {'path': '/a'})]


def test_a_rejected_argument_stops_the_tool_from_running() -> None:
    executor = Executor(invalid=ValueError("intent must be one of ('request', 'answer')"))

    result = dispatch(executor, make_call('send_message', {'intent': 'invented'}))

    assert executor.called == [], 'the body must not run on arguments the model got wrong'
    assert 'intent must be one of' in result.content
    assert result.failed


@pytest.mark.parametrize(
    'call',
    [make_call('read', '{not json'), make_call('read', {'path': '/a'})],
    ids=['bad arguments', 'good arguments'],
)
def test_every_call_is_answered_whatever_happens(call: Any) -> None:
    """The invariant: N calls in, N tool messages out, or the next request 400s."""
    result = dispatch(Executor(), call)

    assert result.tool_call_id == call_id(call)
    assert result.content, 'an empty answer is still an answer, but there must be one'


def test_a_raising_tool_becomes_something_the_model_can_read() -> None:
    result = dispatch(Executor(raises=RuntimeError('no such file')), make_call('read', {'path': '/x'}))

    assert 'no such file' in result.content
    assert result.failed


def test_a_handler_can_replace_what_the_model_sees() -> None:
    async def handler(call: Any, response: ToolResponse, args: dict) -> ExtendedToolResponse:
        return ExtendedToolResponse(
            tool_call_id=call_id(call), name='accept', content='Accepted.', response=response
        )

    result = dispatch(Executor(), make_call('accept', {'reason': 'satisfied'}), {'accept': handler})

    assert result.content == 'Accepted.'


def test_a_handler_can_ask_for_messages_to_be_appended() -> None:
    """How the next requirement reaches the model."""

    async def handler(call: Any, response: ToolResponse, args: dict) -> ExtendedToolResponse:
        return ExtendedToolResponse(
            tool_call_id=call_id(call),
            name='accept',
            content='Accepted.',
            response=response,
            messages=({'role': 'user', 'content': 'NEXT REQUIREMENT'},),
        )

    result = dispatch(Executor(), make_call('accept', {}), {'accept': handler})

    assert result.messages[0]['content'] == 'NEXT REQUIREMENT'


def test_a_raising_handler_does_not_lose_the_tool_result() -> None:
    async def handler(call: Any, response: ToolResponse, args: dict) -> ExtendedToolResponse:
        raise RuntimeError('handler bug')

    result = dispatch(Executor(result='the worker replied'), make_call('send_message', {}), {'send_message': handler})

    assert 'the worker replied' in result.content, 'the tool ran; its result must survive the handler'
    assert 'handler bug' in result.content


def test_a_tool_without_a_handler_passes_its_result_through() -> None:
    result = dispatch(Executor(result='file contents'), make_call('read', {'path': '/a'}))

    assert result.content == 'file contents'
    assert not result.messages


def test_a_handler_for_a_tool_nobody_offers_is_refused_at_assembly() -> None:
    """Otherwise the tool behaves as though no handler existed -- and a handler
    that was meant to end the run means the run never ends. The matching case is
    asserted alongside, since a check that refuses everything also 'passes'."""

    async def handler(call: Any, response: ToolResponse, args: dict) -> ExtendedToolResponse:
        raise AssertionError('never runs')

    check_handlers({'read': handler}, {'read', 'bash'})

    with pytest.raises(ValueError, match='accept'):
        check_handlers({'accept': handler}, {'read', 'bash'})


def test_a_tool_response_reads_as_the_text_the_model_sees() -> None:
    """A part carrying no text still has to say something arrived: dropping it
    would leave an empty tool message with no explanation of why."""
    assert text_of(ToolResponse(output=[{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}])) == 'a\nb'
    assert text_of(ToolResponse(output=[])) == ''
    assert text_of(ToolResponse(output=[{'type': 'image_url', 'image_url': {'url': 'x'}}])) == '[image]'


def test_an_image_reaches_the_model_as_an_image_rather_than_as_its_base64() -> None:
    """What the history carries for a tool that returns something other than text.

    Rendering the part with ``str()`` put its whole payload in the text: measured
    at 240,077 characters for one 1280x720 screenshot, which the model still could
    not see. Every observation tool under ``extensions/tools`` -- ``browser``,
    ``desktop``, ``mobile`` -- answers this way, so a run that holds one depends on
    this.
    """
    url = f'data:image/png;base64,{"A" * 4000}'
    shot = ToolResponse(output=[{'type': 'image_url', 'image_url': {'url': url}}])

    content = content_of(shot)
    assert isinstance(content, list), 'a non-text result keeps its parts'
    assert [part['type'] for part in content] == ['image_url']
    assert content[0]['image_url']['url'] == url, 'and the payload is not truncated'

    call = make_call('screenshot', {})
    assert tool_message(call, content)['content'] == content

    mixed = ToolResponse(output=[{'type': 'text', 'text': 'here'}, {'type': 'image_url', 'image_url': {'url': url}}])
    mixed_content = content_of(mixed)
    assert isinstance(mixed_content, list)
    assert [part['type'] for part in mixed_content] == ['text', 'image_url']

    text_only = ToolResponse(output=[{'type': 'text', 'text': 'plain'}])
    assert content_of(text_only) == 'plain', 'a text result stays a string'


def test_an_error_appends_to_a_tool_result_whichever_form_it_is_in() -> None:
    """The handler-failure path appends to the result it already had.

    An f-string would render a content-part list as its repr, putting the base64
    back into the text on the one path that is already going wrong.
    """
    assert with_error('out', 'boom') == 'out\n\n<error>\nboom\n</error>'
    assert with_error('', 'boom') == '<error>\nboom\n</error>'

    parts = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}}]
    appended = with_error(parts, 'boom')
    assert isinstance(appended, list)
    assert [part['type'] for part in appended] == ['image_url', 'text']
    assert appended[0] == parts[0], 'the image is untouched'
    assert 'boom' in appended[-1]['text']


# -- reminders -----------------------------------------------------------


def test_a_reminder_rides_on_the_result_and_is_tagged_or_absent() -> None:
    """As a separate user message a reminder outlives its turn, which is what the
    legacy recipe's five hardcoded dedup markers exist to clean up. Nothing to say
    must therefore add nothing at all, not an empty tagged block."""
    combined = append_reminder('the worker changed three files', 'check it yourself')

    assert reminder('go verify') == '<system-reminder>\ngo verify\n</system-reminder>'
    assert combined.startswith('the worker changed three files')
    assert '<system-reminder>' in combined
    assert reminder('   ') == ''
    assert append_reminder('result', '  ') == 'result'


# -- compaction ----------------------------------------------------------


class Summariser:
    """Records what it was asked, answers with a fixed account."""

    def __init__(self, account: str = 'what I asked for and what I got') -> None:
        self.account = account
        self.prompts: list[str] = []

    async def create_completion(self, *, messages: list[dict]) -> Any:
        self.prompts.append(str(messages[0]['content']))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.account))])


def a_turn(history: list[dict[str, Any]], call_ref: str) -> list[dict[str, Any]]:
    """One whole turn appended: an assistant with a call, then the call's reply."""
    return [
        *history,
        {'role': 'assistant', 'tool_calls': [{'id': call_ref, 'function': {'name': 'send_message'}}]},
        {'role': 'tool', 'tool_call_id': call_ref, 'content': 'a worker report'},
    ]


def long_history(turns: int = 3) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = [
        {'role': 'system', 'content': 'persona'},
        {'role': 'user', 'content': 'what you want'},
    ]
    for n in range(turns):
        history = a_turn(history, f'c{n}')
    return history


def test_the_compacted_history_is_a_request_the_provider_accepts() -> None:
    """Keeping message structure means keeping it well-formed, and three runs died
    on that: a tool reply whose call was summarised away (0818b), then a history
    ending on the assistant summary (0818c). One user message has neither problem.
    """
    queue = queue_mod.Requirements.build(['first thing', 'second thing'])

    compacted = asyncio.run(compact_mod.compact(long_history(), client=Summariser(), queue=queue, language='English'))

    assert [m['role'] for m in compacted] == ['system', 'user']
    assert not any(m.get('tool_calls') for m in compacted)
    assert compacted[-1]['content'].strip()


def test_the_requirements_and_their_state_come_from_the_run_not_the_account() -> None:
    """The account is the model's own prose and drifts; what has been accepted must
    not. Every requirement is listed, including ones not asked for yet, so the
    simulator can see the whole arc rather than only where it is.
    """
    queue = queue_mod.Requirements.build(['wire pusher', 'add echo', 'template env'])
    queue.accept('the SDK is declared')

    body = asyncio.run(compact_mod.compact(long_history(), client=Summariser(), queue=queue, language='English'))[-1]['content']

    for requirement in ('wire pusher', 'add echo', 'template env'):
        assert requirement in body
    assert 'the SDK is declared' in body
    assert 'working on this one now' in body


def test_a_single_requirement_run_is_not_numbered() -> None:
    """``swe_task`` turns one problem statement into a one-requirement queue, where
    "1 in total, and you are on number 1" says nothing.
    """
    single = queue_mod.Requirements.build(['make leaf nodes execute'])

    body = asyncio.run(compact_mod.compact(long_history(), client=Summariser(), queue=single, language='English'))[-1]['content']

    assert 'make leaf nodes execute' in body
    assert 'in total' not in body


def test_a_failed_summary_still_leaves_a_usable_history() -> None:
    """The history is already over the window, so returning it unchanged is not an
    option. What survives is the requirement and an instruction to go ask.
    """

    class Failing:
        async def create_completion(self, *, messages: list[dict]) -> Any:
            raise RuntimeError('provider down')

    queue = queue_mod.Requirements.build(['the one thing'])

    compacted = asyncio.run(compact_mod.compact(long_history(), client=Failing(), queue=queue, language='English'))

    assert [m['role'] for m in compacted] == ['system', 'user']
    assert 'the one thing' in compacted[-1]['content']
    assert compact_mod.FAILED_INSTRUCTION in compacted[-1]['content']


def test_the_summary_prompt_carries_the_tool_calls_and_their_replies() -> None:
    """What the user did is in its tool calls, not its prose -- a transcript of only
    the text would ask the model to account for a conversation it cannot see.
    """
    summariser = Summariser()
    queue = queue_mod.Requirements.build(['a thing'])

    asyncio.run(compact_mod.compact(long_history(), client=summariser, queue=queue, language='English'))

    prompt = summariser.prompts[0]
    assert 'send_message' in prompt
    assert 'a worker report' in prompt
    assert 'persona' not in prompt, 'the system prompt is not part of the conversation to account for'


@pytest.mark.parametrize(
    ('window', 'pct', 'reported', 'expected'),
    [
        (12000, 50, 5000, False),
        (12000, 50, 7000, True),
        (0, 50, 999999, False),
        (12000, 50, 0, False),
    ],
    ids=['under', 'over', 'disabled', 'provider reported nothing'],
)
def test_the_budget_reads_the_size_the_provider_reported(window: int, pct: int, reported: int, expected: bool) -> None:
    """A reported zero is a field left unfilled, not a measurement: a request cannot
    cost nothing. Treating it as one would compact on every turn.
    """
    budget = compact_mod.Budget(window=window, pct=pct)
    response = {'usage': {'prompt_tokens': reported, 'completion_tokens': 10}}

    assert budget.exceeded(response) is expected
