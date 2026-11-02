"""Which reminder a turn earns, and where it lands.

The cases here are the ones the measured runs (0812e/f/g, nine trajectories)
showed mattering: every run opens with two or three messages to the worker, so a
heavy reminder at each would repeat the checklist while the system prompt is
still nearby; and one run looked 35 times in a row, where a reminder is the only
thing holding the persona.
"""

from nexus.extensions.tasks.hitl.user_agent.prompts import worker_command_lines, worker_skills_section
from nexus.extensions.tasks.hitl.user_agent.prompts.messages import reminder_message
from nexus.extensions.tasks.hitl.user_agent.reminders import (
    HEAVY_PREFACE,
    LIGHT_TEXT,
    Reminders,
    heavy_text,
)


RECALLS = ('you hand over one thing at a time', 'you check before you agree')
WORKER_COMMANDS = worker_command_lines({'verify': 'run it from clean', 'code-review': 'read it over'}, prefix='/')


def levels(reminders: Reminders, names: list[str]) -> list[str]:
    """The reminder each call earns, as 'H'/'l' plus the turn, '' when silent."""
    out: list[str] = []
    for name in names:
        text = reminders.after_tool(name)
        if not text:
            continue
        out.append(('H' if text.startswith(HEAVY_PREFACE) else 'l') + str(reminders.turn))
    return out


def test_a_message_to_the_worker_earns_the_heavy_one() -> None:
    reminders = Reminders(recalls=RECALLS)

    assert levels(reminders, ['send_message']) == ['H1']


def test_a_second_message_inside_the_cooldown_is_downgraded_not_skipped() -> None:
    """All nine measured runs open goal then answer, so this is the common path."""
    reminders = Reminders(recalls=RECALLS)

    assert levels(reminders, ['send_message', 'send_message']) == ['H1', 'l2']


def test_the_heavy_one_returns_once_its_interval_has_passed() -> None:
    reminders = Reminders(recalls=RECALLS, heavy_every=10, light_every=15)
    names = ['send_message'] + ['read'] * 10 + ['send_message']

    assert levels(reminders, names) == ['H1', 'H12']


def test_looking_stays_quiet_until_the_conversation_has_run_on() -> None:
    """A run that opens by looking is not reminded on its first call.

    The measured runs all open with a message to the worker, so this only shows
    up as a spent reminder while the system prompt is still the nearest thing in
    the history.
    """
    reminders = Reminders(recalls=RECALLS, light_every=15)

    assert levels(reminders, ['read'] * 14) == []
    assert levels(reminders, ['read']) == ['l15']


def test_a_long_stretch_of_looking_is_reminded_at_the_interval() -> None:
    """The 35-call stretch measured on run 0812f is what the light one is for."""
    reminders = Reminders(recalls=RECALLS, light_every=15)

    assert levels(reminders, ['send_message'] + ['read'] * 35) == ['H1', 'l16', 'l31']


def test_a_reminder_of_either_level_defers_the_next_light_one() -> None:
    heavy_first = Reminders(recalls=RECALLS, light_every=15)
    light_first = Reminders(recalls=(), light_every=15)

    assert levels(heavy_first, ['send_message'] + ['read'] * 14) == ['H1']
    assert levels(light_first, ['send_message'] + ['read'] * 14) == ['l1']


def test_a_persona_with_nothing_to_recall_still_gets_the_light_one() -> None:
    """Otherwise a persona declaring no recall lines would go unreminded."""
    reminders = Reminders(recalls=())

    assert levels(reminders, ['send_message']) == ['l1']


def test_a_run_may_replace_the_light_text() -> None:
    reminders = Reminders(recalls=(), light='remember what you came for')

    assert reminders.after_tool('send_message') == 'remember what you came for'


def test_an_empty_light_text_silences_it_rather_than_sending_a_blank() -> None:
    reminders = Reminders(recalls=(), light='   ')

    assert reminders.after_tool('send_message') == ''


def test_the_heavy_one_lists_every_recall_the_persona_declared() -> None:
    text = heavy_text(RECALLS)

    for line in RECALLS:
        assert f'- {line}' in text


def test_the_heavy_one_keeps_the_order_the_persona_declared() -> None:
    text = heavy_text(('first', 'second', 'third'))

    assert text.index('first') < text.index('second') < text.index('third')


def test_blank_recall_lines_are_dropped() -> None:
    assert heavy_text(('kept', '   ')).count('- ') == 1
    assert heavy_text(('', '  ')) == ''


# -- the two skill lists a heavy reminder carries --------------------------


def test_the_heavy_one_carries_both_skill_lists() -> None:
    """The opening message names the user's skills and the system prompt names the
    worker's; a run that compacts keeps neither."""
    text = heavy_text(RECALLS, catalogue='- browser: drive a browser', worker_commands=WORKER_COMMANDS)

    assert '- browser: drive a browser' in text
    assert '`/verify`' in text
    assert '`/code-review`' in text


def test_worker_skills_alone_are_enough_for_a_heavy_reminder() -> None:
    """A persona with no recalls and no skills of its own still has these to be told."""
    reminders = Reminders(recalls=(), worker_commands=WORKER_COMMANDS)

    assert '`/verify`' in reminders.after_tool('send_message')


def test_a_worker_with_no_skills_adds_nothing() -> None:
    assert heavy_text(RECALLS, worker_commands='') == heavy_text(RECALLS)
    assert heavy_text((), worker_commands='   ') == ''


def test_the_lines_are_the_ones_the_system_prompt_lists() -> None:
    """Two lists that drift leave the user with a name the guard refuses."""
    commands = {'verify': 'run it from clean'}
    section = worker_skills_section(commands, prefix='/')

    for line in worker_command_lines(commands, prefix='/').splitlines():
        assert line in section


def test_a_harness_with_no_command_form_lists_nothing() -> None:
    assert worker_command_lines({'verify': 'run it from clean'}, prefix='') == ''


def test_a_reminder_is_its_own_user_message() -> None:
    """Not appended to a tool result: one measured ``read`` answered with 20036
    characters, and a line after that is not read."""
    entry = reminder_message(LIGHT_TEXT)

    assert entry['role'] == 'user'
    assert entry['content'].startswith('<system-reminder>')
    assert LIGHT_TEXT in entry['content']


def test_nothing_to_say_yields_no_message_at_all() -> None:
    assert reminder_message('') == {}
    assert reminder_message('   ') == {}
