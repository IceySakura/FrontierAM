"""Projecting a worker turn into the report the user simulator reads.

Each backend fills what it can observe, so the projection lives on the manager
rather than on the result: dispatching by result class made the PTY backend inherit
print-mode's "not separable" note, which was false -- its steps *are* separable, the
event buffer was simply empty.

The three v1 defects still pinned here, all of which produced a plausible export:

1. **PTY degraded silently.** ``_format_worker_turn`` (``swe_chain/handlers.py:210``)
   reads ``raw['assistant_messages']``, which the PTY backend does not record, so
   the user lost the claim-vs-commentary split with no indication.
2. **The shape varied with message count.** With one message v1 returned a bare
   string (``:222``), so whether the user saw tags at all depended on how many
   messages the worker happened to send that turn.
3. **The tags were never explained.** ``final_response`` / ``intermediate_steps``
   / ``worker_question`` / ``worker_progress`` appear nowhere in
   ``swe_chain/prompts.py`` or ``swe/prompts/``.
"""

import pytest

from nexus.core import UserInputRequest
from nexus.extensions.tasks.hitl.actor.base import TurnCost, TurnResult
from nexus.extensions.tasks.hitl.actor.cost import render_cost
from nexus.extensions.tasks.hitl.actor.factory import make_worker_manager
from nexus.extensions.tasks.hitl.actor.print_claude_code.manager import PrintClaudeCodeSessionManager
from nexus.extensions.tasks.hitl.actor.pty_claude_code.manager import PtyClaudeCodeSessionManager
from nexus.extensions.tasks.hitl.actor.rendering import (
    DETAIL_HEAD,
    DETAIL_TAIL,
    SECTION_LIMIT,
    WORKER_REPORT_PROTOCOL,
    render_questions,
    render_worker_turn,
    split_messages,
)


BACKENDS = ('print_claude_code', 'pty_claude_code')


@pytest.fixture
def printer() -> PrintClaudeCodeSessionManager:
    return PrintClaudeCodeSessionManager()


@pytest.fixture
def pty() -> PtyClaudeCodeSessionManager:
    return PtyClaudeCodeSessionManager()


def question(hint: str = 'Which auth flow?') -> UserInputRequest:
    return UserInputRequest(hint=hint, source='hook')


def section(rendered: str, tag: str) -> str:
    """The body of one tagged section, or '' when absent."""
    open_tag, close_tag = f'<{tag}>', f'</{tag}>'
    if open_tag not in rendered:
        return ''
    return rendered.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()


def spoke(*texts: str) -> list[dict[str, object]]:
    """Step entries for a turn that only said things."""
    return [{'type': 'text', 'text': text} for text in texts]


# --- every backend projects, and none of them by inheritance --------------


@pytest.mark.parametrize('backend', BACKENDS)
def test_every_backend_can_project_a_turn(backend: str) -> None:
    """A backend with no projector cannot be constructed at all.

    ``report`` is abstract, so this is enforced by the class, not by review.
    """
    manager = make_worker_manager(backend)

    rendered = manager.report(TurnResult(kind='completed', output='did the thing'))

    assert section(rendered, 'final_response') == 'did the thing'


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_turn_with_nothing_in_it_still_reports_a_stable_shape(backend: str) -> None:
    """v1's shape varied with message count; the section set must not."""
    rendered = make_worker_manager(backend).report(TurnResult(kind='completed'))

    assert '<final_response>' in rendered


# --- print mode: splits on the steps it records ---------------------------


def test_a_multi_message_turn_separates_the_claim_from_the_commentary(printer) -> None:
    result = TurnResult(
        kind='completed',
        output='patched it',
        raw={'steps': spoke('let me look', 'editing now', 'patched it')},
    )

    rendered = printer.report(result)

    assert section(rendered, 'final_response') == 'patched it'
    assert section(rendered, 'worker_progress') == 'let me look\n\nediting now'


def test_print_mode_reports_the_tools_between_the_messages(printer) -> None:
    """The defect: stream-json carries tool_use blocks and only text was read.

    A worker that names four files it looked at is making a claim; the calls are what
    lets the user tell that from a summary of a file it never opened.
    """
    result = TurnResult(
        kind='completed',
        raw={
            'steps': [
                {'type': 'text', 'text': 'checking the config'},
                {'type': 'tool_use', 'name': 'Read', 'input': {'file_path': 'config/broadcasting.php'}},
                {'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'composer --version'}},
                {'type': 'text', 'text': 'the driver falls back to null'},
            ]
        },
    )

    progress = section(printer.report(result), 'worker_progress')

    assert progress == 'checking the config\n\n[used Read: config/broadcasting.php]\n[used Bash: composer --version]'


def test_a_single_message_turn_still_carries_the_final_response_tag(printer) -> None:
    """v1 returned a bare string here, so the user's input shape depended on
    how many messages the worker sent."""
    result = TurnResult(kind='completed', output='only', raw={'steps': spoke('only')})

    rendered = printer.report(result)

    assert section(rendered, 'final_response') == 'only'
    assert '<worker_progress>' not in rendered, 'nothing came before it'


def test_blank_messages_do_not_become_empty_steps(printer) -> None:
    result = TurnResult(kind='completed', output='real', raw={'steps': spoke('', '   ', 'real')})

    rendered = printer.report(result)

    assert section(rendered, 'final_response') == 'real'
    assert '<worker_progress>' not in rendered


@pytest.mark.parametrize('bad', [None, 'not a list', 42, [], ['', '  '], [{'type': 'unknown'}]])
def test_unusable_steps_fall_back_rather_than_crash(printer, bad) -> None:
    result = TurnResult(kind='completed', output='combined', raw={'steps': bad})

    rendered = printer.report(result)

    assert section(rendered, 'final_response') == 'combined'
    assert 'no separable messages' in section(rendered, 'note')


# --- the worker asking a question ----------------------------------------


def test_a_pending_question_is_not_presented_as_a_result(printer) -> None:
    """The worker's last message *is* the question, so it is progress, not a claim."""
    result = TurnResult(
        kind='needs_input',
        output='I got as far as the schema',
        raw={'steps': spoke('looked around', 'I got as far as the schema')},
        user_input_request=question('Which auth flow?'),
    )

    rendered = printer.report(result)

    assert section(rendered, 'worker_question') == 'Which auth flow?'
    assert section(rendered, 'worker_progress') == 'looked around\n\nI got as far as the schema'
    assert '<final_response>' not in rendered, 'a turn that asks has no claimed result'


def test_progress_keeps_the_order_the_turn_happened_in(printer) -> None:
    """The last thing said comes last.

    It was prepended before, so a turn that asked put its closing line above the
    steps that led to it -- an order the worker never produced.
    """
    result = TurnResult(
        kind='needs_input',
        raw={
            'steps': [
                {'type': 'text', 'text': 'first'},
                {'type': 'tool_use', 'name': 'Read', 'input': {'file_path': 'a.py'}},
                {'type': 'text', 'text': 'last'},
            ]
        },
        user_input_request=question('Which one?'),
    )

    assert section(printer.report(result), 'worker_progress') == 'first\n\n[used Read: a.py]\n\nlast'


def test_a_question_with_no_progress_omits_the_progress_section(printer) -> None:
    result = TurnResult(kind='needs_input', output='', user_input_request=question())

    rendered = printer.report(result)

    assert '<worker_progress>' not in rendered
    assert section(rendered, 'worker_question')


def test_a_question_without_a_hint_still_says_something(printer) -> None:
    result = TurnResult(kind='needs_input', output='', user_input_request=UserInputRequest(source='hook'))

    assert section(printer.report(result), 'worker_question') == 'Worker requested input.'


# --- PTY: richer material, and it must be used ---------------------------


def test_pty_splits_from_its_own_events_instead_of_degrading(pty) -> None:
    """The defect: PTY has no assistant_messages, so v1 reported no progress at all."""
    result = TurnResult(
        kind='completed',
        output='patched it',
        raw={
            'recent_events': [
                {'type': 'text', 'text': 'let me look'},
                {'type': 'tool_use', 'name': 'Read'},
                {'type': 'text', 'text': 'patched it'},
            ]
        },
    )

    rendered = pty.report(result)

    assert section(rendered, 'final_response') == 'patched it'
    assert section(rendered, 'worker_progress') == 'let me look\n\n[used Read]'


def test_the_claim_comes_from_output_not_the_summarised_step(pty) -> None:
    """The defect: PTY events cut assistant text to 350/150, and the claim came from there.

    27 of 28 reports in run 0910a carried the marker inside <final_response>, and the
    user spent rounds asking the worker to repeat what it had already said.
    """
    whole = 'the middle of this is what the user needs'
    result = TurnResult(
        kind='completed',
        output=whole,
        raw={'recent_events': [{'type': 'text', 'text': 'the middle\n...[truncated]...\nneeds'}]},
    )

    assert section(pty.report(result), 'final_response') == whole


def test_pty_reports_tool_names_once_each_in_order(pty) -> None:
    result = TurnResult(
        kind='completed',
        raw={
            'recent_events': [
                {'type': 'text', 'text': 'a'},
                {'type': 'tool_use', 'name': 'Read'},
                {'type': 'tool_use', 'name': 'Edit'},
                {'type': 'tool_use', 'name': 'Read'},
                {'type': 'text', 'text': 'b'},
            ]
        },
    )

    assert section(pty.report(result), 'worker_progress') == 'a\n\n[used Read]\n[used Edit]\n[used Read]'


def test_a_tool_argument_is_cut_to_the_length_a_report_shows(pty) -> None:
    """A command or a path can run to any length; the report cannot.

    The limits are the PTY proxy's own (``interactive/events.py``), so a report is the
    same length whichever backend produced it.
    """
    long_command = 'pytest ' + 'tests/deep/path/' * 40 + 'test_end.py'
    result = TurnResult(
        kind='completed',
        raw={
            'recent_events': [
                {'type': 'text', 'text': 'a'},
                {'type': 'tool_use', 'name': 'Bash', 'input': {'command': long_command}},
                {'type': 'text', 'text': 'b'},
            ]
        },
    )

    progress = section(pty.report(result), 'worker_progress')

    assert len(long_command) > DETAIL_HEAD + DETAIL_TAIL, 'the sample has to be long enough to cut'
    assert long_command not in progress
    assert 'pytest tests/deep/path/' in progress, 'the head survives'
    assert 'test_end.py' in progress, 'the tail survives, so the point of the command is visible'
    assert len(progress) < len(long_command)


def test_a_tool_nobody_declared_a_field_for_reports_its_name_alone(pty) -> None:
    """An MCP server's tool has argument names this module cannot know.

    Naming one at random is as likely to surface a flag as the thing it acted on.
    """
    result = TurnResult(
        kind='completed',
        raw={
            'recent_events': [
                {'type': 'tool_use', 'name': 'mcp__search__query', 'input': {'q': 'anything', 'limit': 5}},
                {'type': 'text', 'text': 'done'},
            ]
        },
    )

    assert section(pty.report(result), 'worker_progress') == '[used mcp__search__query]'


def test_pty_does_not_leak_the_screen_or_the_event_cursor(pty) -> None:
    """A pyte render of the viewport is more than a person reading a reply sees.

    Both keys stay in ``raw`` for logging; neither reaches the simulator.
    """
    result = TurnResult(
        kind='completed',
        raw={
            'latest_screen': 'SECRET-VIEWPORT-CONTENT',
            'event_cursor': {'latest_seq': 41, 'after_seq': 7},
            'recent_events': [{'type': 'text', 'text': 'done'}],
        },
    )

    rendered = pty.report(result)

    assert 'SECRET-VIEWPORT-CONTENT' not in rendered
    assert 'latest_seq' not in rendered
    assert '41' not in rendered


def test_pty_with_no_text_events_does_not_borrow_print_modes_note(pty) -> None:
    """The defect inheritance caused: PTY was told its steps were 'not separable'.

    They are separable -- the event buffer held no text this turn. It still reports
    the tools it saw, which print mode could not have.
    """
    result = TurnResult(
        kind='completed',
        output='combined',
        raw={'recent_events': [{'type': 'tool_use', 'name': 'Bash'}]},
    )

    rendered = pty.report(result)

    assert section(rendered, 'final_response') == 'combined'
    assert section(rendered, 'worker_progress') == '[used Bash]'
    assert 'not separable' not in rendered
    assert 'no message events' in section(rendered, 'note')


def test_pty_honours_a_pending_question_like_every_other_backend(pty) -> None:
    result = TurnResult(
        kind='needs_input',
        raw={'recent_events': [{'type': 'text', 'text': 'got this far'}]},
        user_input_request=question('Which one?'),
    )

    rendered = pty.report(result)

    assert section(rendered, 'worker_question') == 'Which one?'
    assert '<final_response>' not in rendered


# --- the cost a turn carries, on every backend ---------------------------


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_turn_carries_its_cost_after_the_report(backend: str) -> None:
    turn_cost = TurnCost(duration_s=200.0, context_tokens=180_000)

    rendered = make_worker_manager(backend).report(TurnResult(kind='completed', output='did the thing', cost=turn_cost))

    assert render_cost(turn_cost) in rendered
    assert rendered.index('did the thing') < rendered.index(render_cost(turn_cost)), 'the cost follows the turn'


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_turn_without_a_cost_adds_no_line(backend: str) -> None:
    rendered = make_worker_manager(backend).report(TurnResult(kind='completed', output='did the thing'))

    assert 'tokens of context' not in rendered


# --- the properties reminders read ---------------------------------------


def test_awaiting_answer_and_has_output() -> None:
    blocked = TurnResult(kind='needs_input', output='partial', user_input_request=question())
    plain = TurnResult(kind='completed', output='done')
    empty = TurnResult(kind='terminal', output='   ')

    assert blocked.awaiting_answer
    assert blocked.has_output
    assert not plain.awaiting_answer
    assert plain.has_output
    assert not empty.has_output


# --- the tags are documented for the simulator's prompt ------------------


@pytest.mark.parametrize(
    'tag',
    ['final_response', 'worker_question', 'worker_progress', 'note'],
)
def test_every_tag_the_renderer_can_emit_is_explained(tag: str) -> None:
    """v1 shipped these tags with no explanation anywhere in its prompts."""
    assert f'<{tag}>' in WORKER_REPORT_PROTOCOL


def test_the_protocol_tells_the_user_a_claim_is_not_evidence() -> None:
    assert 'claim to verify' in WORKER_REPORT_PROTOCOL


# --- the shared helpers --------------------------------------------------


def test_split_messages_distinguishes_unusable_from_single() -> None:
    assert split_messages(['a', 'b']) == ('b', ('a',))
    assert split_messages(['only']) == ('only', ())
    assert split_messages([]) is None
    assert split_messages('nope') is None
    assert split_messages(None) is None


def test_render_worker_turn_is_the_shared_vocabulary() -> None:
    """Both projectors end here, so the section set cannot drift between them."""
    rendered = render_worker_turn(final='done', steps=('a',), note='n')

    for tag in ('final_response', 'worker_progress', 'note'):
        assert f'<{tag}>' in rendered


# --- the options a question offered ---------------------------------------


ASKED = {
    'name': 'AskUserQuestion',
    'input': {
        'questions': [
            {
                'header': 'M2M count',
                'question': 'How should the number of picked records be controlled?',
                'options': [
                    {
                        'label': '`lower`/`upper` bounds (Recommended)',
                        'description': 'Reuses the vocabulary _integer already uses.',
                        'preview': 'lower: 1\nupper: 3',
                    },
                    {
                        'label': 'A single `length` key',
                        'description': '`length` already means string length for Char, so it would carry two meanings.',
                    },
                ],
            }
        ]
    },
}


def test_a_question_carries_the_options_it_offered(printer) -> None:
    """Without them the question reads as "you decide", and that is what gets answered.

    The trade-off the worker wrote into each description is the only thing that
    makes the choice a choice.
    """
    asked = UserInputRequest(hint='Claude Code asks for user input.', source='hook', hook=ASKED)
    result = TurnResult(kind='needs_input', output='', user_input_request=asked)

    body = section(printer.report(result), 'worker_question')

    assert '`lower`/`upper` bounds (Recommended)' in body
    assert 'would carry two meanings' in body
    assert 'lower: 1' in body


def test_a_question_without_options_still_renders(printer) -> None:
    asked = UserInputRequest(
        hint='ignored', source='hook', hook={'input': {'questions': [{'header': 'H', 'question': 'Q?'}]}}
    )

    body = section(printer.report(TurnResult(kind='needs_input', user_input_request=asked)), 'worker_question')

    assert 'H: Q?' in body


def test_a_deferred_tool_that_is_not_a_question_falls_back_to_its_hint(printer) -> None:
    """``ExitPlanMode`` has a plan rather than questions, and upstream words it."""
    asked = UserInputRequest(hint='wants to exit plan mode', source='hook', hook={'input': {'plan': 'do the thing'}})

    body = section(printer.report(TurnResult(kind='needs_input', user_input_request=asked)), 'worker_question')

    assert body == 'wants to exit plan mode'


def test_multi_select_says_so() -> None:
    rendered = render_questions([{'question': 'Q?', 'multiSelect': True, 'options': [{'label': 'A'}]}])

    assert 'several answers are allowed' in rendered


def test_nothing_to_render_yields_nothing() -> None:
    for questions in (None, {}, [], [{'header': 'no text at all'}][0:0]):
        assert render_questions(questions) == ''


def test_a_section_is_capped_so_one_turn_cannot_fill_the_context() -> None:
    """The backstop on how much of a turn reaches the user.

    A worker is an agent rather than a person pasting, so nothing it normally says
    comes near this: across 678 reports from the recorded batches the largest was
    17,899 characters. What the cap is for is one turn that emits a whole build log,
    which otherwise reaches the user verbatim -- there is no other limit on this path.
    """
    log = 'X' * (SECTION_LIMIT * 4)

    report = render_worker_turn(final=log)

    assert log not in report, 'the section is cut'
    assert len(report) < SECTION_LIMIT + 200, 'and the cut bounds the whole report'
    assert 'characters omitted' in report, 'a silent cut would read as a complete report'


def test_a_report_of_ordinary_size_is_left_alone() -> None:
    """The measured 99th percentile passes through whole."""
    said = 'y' * 12_268

    report = render_worker_turn(final=said)

    assert said in report
    assert 'omitted' not in report
