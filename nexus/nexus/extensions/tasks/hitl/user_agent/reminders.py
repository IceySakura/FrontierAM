"""Reminding the user of its own persona, at two levels of detail.

A system prompt loses force as the conversation grows, so the persona is put back
within reach periodically. This asks for no action and cannot fail, which is what
separates it from a guard (a guard refuses a call, see
``recipe/swe_chain_v2/handlers.py``).

* **light** -- one prepared sentence pointing back at the persona. No specifics.
* **heavy** -- the persona's own lines, the skills the user can read, and the skills
  the worker can be asked to run.

When each one fires:

* a heavy one after a message to a worker, which is the turn a worker's output enters
  the history;
* a heavy one inside its cooldown is downgraded to light rather than skipped;
* a light one elsewhere, once the conversation has run on without any reminder.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from nexus.extensions.tasks.hitl.user_agent.tools.orchestration import WORKER_FACING_TOOLS


LIGHT_TEXT = 'You are a user simulator. By default you avoid giving the worker an implementation, a fix or a diagnosis.'
"""Prepared, and overridable per run -- see :class:`Reminders`.

Carries the role and the wording rule every persona shares. How much is said at once is
per-persona, so it stays in ``recalls``. The language a run writes in has its own
reminder, sent after every message the worker receives.
"""

HEAVY_PREFACE = 'You work in the following ways.'
SKILL_PREFACE = 'Available skills you can pull up for yourself via invoke_skill:'
WORKER_SKILL_PREFACE = "The worker's available skills are as followed:"

# Turns between heavy reminders.
HEAVY_EVERY = 10

# Turns without any reminder before a light one fires.
LIGHT_EVERY = 15


def heavy_text(
    recalls: Sequence[str],
    *,
    preface: str = HEAVY_PREFACE,
    catalogue: str = '',
    worker_commands: str = '',
) -> str:
    """The checklist a heavy reminder carries, or '' when it would carry nothing.

    ``recalls`` are the persona's own one-liners, kept in the order it declared them.
    ``catalogue`` is the skill menu. ``worker_commands`` is
    :func:`~nexus.extensions.tasks.hitl.user_agent.prompts.worker_skills.worker_command_lines`,
    which the system prompt lists as well.
    """
    lines = [line.strip() for line in recalls if line.strip()]
    parts = []
    if lines:
        listed = '\n'.join(f'- {line}' for line in lines)
        parts.append(f'{preface}\n{listed}')
    if catalogue.strip():
        parts.append(f'{SKILL_PREFACE}\n{catalogue.strip()}')
    if worker_commands.strip():
        parts.append(f'{WORKER_SKILL_PREFACE}\n{worker_commands.strip()}')
    return '\n\n'.join(parts)


@dataclass
class Reminders:
    """Which reminder a turn earns, and the counters that decide it.

    Held by the run rather than passed around. The decision needs how long it has been
    since the last one.

    ``light``, the two intervals and ``worker_tools`` are constructor arguments so a
    run can tune them without a new class.
    """

    recalls: tuple[str, ...] = ()
    light: str = LIGHT_TEXT
    heavy_every: int = HEAVY_EVERY
    light_every: int = LIGHT_EVERY
    catalogue: str = ''
    """The skill menu, carried by every heavy reminder."""

    worker_commands: str = ''
    """The worker's skills, one line each, carried by every heavy reminder."""

    worker_tools: frozenset[str] = WORKER_FACING_TOOLS
    """Schema names of the tools that put something in front of a worker.

    The heavy reminder fires after one of these. A name outside this set counts as the
    user looking at something on its own. A simulator that names its sending tool
    differently and leaves this at the default gets light reminders only, and silently.
    """

    turn: int = 0
    last_any: int = 0
    """Turn of the last reminder of either level. Starts at 0, not at negative
    infinity: the run has just begun and the system prompt is as near as a reminder
    would be."""

    last_heavy: int = field(default=-(10**9))
    """Unlike ``last_any``, seeded low so the first message to a worker earns the heavy
    one."""

    def after_tool(self, name: str) -> str:
        """The reminder text for the turn just taken, or '' to stay quiet.

        Advances the turn counter. Called once per tool call.
        """
        self.turn += 1
        if name in self.worker_tools:
            return self._heavy() or self._light(force=True)
        return self._light()

    def compacted(self) -> None:
        """Called when the history was replaced, which drops every reminder in it.

        Both cooldowns are cleared, and the next turn earns a heavy one.
        """
        self.last_heavy = -(10**9)
        self.last_any = 0
        self.turn = 0

    def _heavy(self) -> str:
        if self.turn - self.last_heavy < self.heavy_every:
            return ''
        text = heavy_text(self.recalls, catalogue=self.catalogue, worker_commands=self.worker_commands)
        if not text:
            return ''
        self.last_heavy = self.turn
        self.last_any = self.turn
        return text

    def _light(self, *, force: bool = False) -> str:
        """The light text, when due. ``force`` is the downgrade of a heavy one."""
        if not force and self.turn - self.last_any < self.light_every:
            return ''
        if not self.light.strip():
            return ''
        self.last_any = self.turn
        return self.light


__all__ = [
    'HEAVY_EVERY',
    'HEAVY_PREFACE',
    'LIGHT_EVERY',
    'LIGHT_TEXT',
    'SKILL_PREFACE',
    'WORKER_SKILL_PREFACE',
    'Reminders',
    'heavy_text',
]
