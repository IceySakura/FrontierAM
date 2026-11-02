"""The messages a run appends to the user's history.

Each closes on an instruction for the turn that follows it. The first task and every
later one get different wording.
"""

from nexus.extensions.tasks.hitl.user_agent.reminders import SKILL_PREFACE
from nexus.extensions.tasks.hitl.utils.text import section, tagged


REMINDER_TAG = 'system-reminder'

# Said to a turn that produced prose and no tool call.
NUDGE = 'That turn made no tool call. Take an action now.'

# Closes the first task's message.
SCOPE_INSTRUCTION = (
    'IMPORTANT: the worker has not read this task and never will. It knows the '
    'workspace and what you have told it. Everything you want has to be asked for.'
)

# Closes every later requirement's message.
CONTINUITY_INSTRUCTION = (
    'This requirement is added to the ones already accepted, not a page turn past them. '
    'Keep how you talk, the language you write in, and what you found out about the code '
    'and about the worker; the worker still holds the full history of the session. Watch '
    'for the new requirement breaking an earlier one. When the two conflict, follow the '
    'new one.'
)

# Closes the first task's message, and said again after every message the worker
# receives. ``{name}`` is the run's language, read from the task by
# ``user_agent.language.detect``.
LANGUAGE_INSTRUCTION = 'IMPORTANT: write every message to the worker in {name}.'

# Sent after `send_message` delivers and the worker comes back with output rather than
# a question of its own. ``handover_reminder`` joins the three below into one message.

# Where the run stands. ``{request}`` is the queue position, ``{handover}`` the Nth
# time work has come back on that request.
HANDOVER_STATE = 'The worker has handed work back on request {request}. This is the {handover} time.'

# What to look at.
HANDOVER_LOOK = (
    'Look at what changed rather than at what the worker said about it. Cover the parts '
    'your request was about and anything else that changed.'
)

HANDOVER_REPLY = 'Then reply in your own words.'

# Ordinals for ``{handover}``, which counts from 1.
_ORDINALS = ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth')


def _ordinal(count: int) -> str:
    """``count`` as a word, falling back to digits with a suffix."""
    if 1 <= count <= len(_ORDINALS):
        return _ORDINALS[count - 1]
    suffix = 'th' if count % 100 in (11, 12, 13) else {1: 'st', 2: 'nd', 3: 'rd'}.get(count % 10, 'th')
    return f'{count}{suffix}'


def handover_reminder(*, request: int, handover: int) -> str:
    """What a handover is told: where the run stands, what to look at, then reply."""
    return '\n'.join([
        HANDOVER_STATE.format(request=request, handover=_ordinal(handover)),
        HANDOVER_LOOK,
        HANDOVER_REPLY,
    ])


def reminder(text: str) -> str:
    """``text`` wrapped in the reminder tag, or '' when there is nothing to say."""
    return tagged(REMINDER_TAG, text)


def append_reminder(content: str, text: str) -> str:
    """``content`` with a reminder after it, sharing that content's lifetime."""
    wrapped = reminder(text)
    if not wrapped:
        return content
    if not content.strip():
        return wrapped
    return f'{content}\n\n{wrapped}'


def reminder_message(text: str) -> dict[str, str]:
    """``text`` as its own history entry, or ``{}`` when there is nothing to say.

    Goes in as ``role: user`` rather than riding on a tool's content.
    """
    wrapped = reminder(text)
    if not wrapped:
        return {}
    return {'role': 'user', 'content': wrapped}


def language_instruction(name: str) -> str:
    """The language rule, naming the language the run writes in."""
    return LANGUAGE_INSTRUCTION.format(name=name)


def build_opening_message(requirement: str, *, language: str, catalogue: str = '') -> str:
    """The first user message: the task, what came with it, and the first move."""
    if not requirement.strip():
        raise ValueError('the opening message needs a requirement')
    parts = [section('## Your task', requirement)]
    parts.append(reminder(SCOPE_INSTRUCTION))
    if catalogue.strip():
        parts.append(skill_reminder(catalogue))
    parts.append(reminder(language_instruction(language)))
    return '\n\n'.join(parts)


def build_requirement_message(requirement: str) -> str:
    """A later requirement, arriving as something newly assigned to the user."""
    if not requirement.strip():
        raise ValueError('a requirement message needs a requirement')
    return '\n\n'.join([
        section('## Your new requirement', requirement),
        reminder(CONTINUITY_INSTRUCTION),
    ])


def skill_reminder(catalogue: str) -> str:
    """The skill menu, wrapped in a reminder tag."""
    if not catalogue.strip():
        return ''
    return reminder(f'{SKILL_PREFACE}\n{catalogue.strip()}')


__all__ = [
    'CONTINUITY_INSTRUCTION',
    'HANDOVER_LOOK',
    'HANDOVER_REPLY',
    'HANDOVER_STATE',
    'LANGUAGE_INSTRUCTION',
    'NUDGE',
    'REMINDER_TAG',
    'SCOPE_INSTRUCTION',
    'append_reminder',
    'build_opening_message',
    'build_requirement_message',
    'handover_reminder',
    'language_instruction',
    'reminder',
    'reminder_message',
    'skill_reminder',
]
