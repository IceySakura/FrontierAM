"""The ``Persona`` type and the process-wide set of them.

A persona is everything a run says to the user on its behalf: the sides of itself it
is told to attend to, the lines put back in front of it as the conversation runs on,
and the tools it holds.

Every field is data, so a persona is a file rather than a module -- see
:mod:`.loader` for the format and ``HITL_USER_PERSONA_DIRS`` for where a run puts its
own. :func:`register_persona` remains for one built in a process, such as a test.
"""

import logging

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from nexus.extensions.tasks.hitl.utils.text import section


logger = logging.getLogger(__name__)


class PersonaError(ValueError):
    """A persona is declared in a way that cannot be assembled."""


@dataclass(frozen=True)
class Persona:
    """Who the user is, and what it is told to attend to as the run goes on."""

    name: str

    prompt: str
    """Which sides of this to attend to: how it goes about asking, and what it looks
    at in work handed back. Rendered as one system prompt section by :meth:`section`.
    """

    tools: tuple[str, ...] = ()

    recalls: tuple[str, ...] = field(default_factory=tuple)
    """One line each, put back in front of the user as the conversation runs on.

    A phrasing of what ``prompt`` already says rather than an addition to it. Read by
    ``user_agent.reminders``; no recalls means every reminder is the light one.
    """

    reads_memory: bool = True
    """Whether this persona wrote the preference note the worker was given.

    The note is the run's (``worker_config['user_memory']``), and a persona that reads
    it is told it wrote it. False for a user with no prior working relationship: being
    handed preferences it is supposed to have written is a contradiction it has to
    resolve on its own.
    """

    variants: Mapping[str, str] = field(default_factory=dict)
    """Extra prompt text per natural language, keyed by language code.

    Filled from the ``persona.<code>.md`` files beside a persona written as a
    directory. :meth:`for_language` is what puts one in front of the model. Wrapped
    read-only in ``__post_init__``: a registered persona is shared by every run in the
    process, and the rest of this class is frozen against exactly that.
    """

    def __post_init__(self) -> None:
        if not self.name:
            raise PersonaError('persona name must not be empty')
        if not self.prompt.strip():
            raise PersonaError(f'persona {self.name!r} has an empty prompt')
        duplicates = sorted({name for name in self.tools if list(self.tools).count(name) > 1})
        if duplicates:
            raise PersonaError(f'persona {self.name!r} declares tools more than once: {duplicates}')
        if any(not line.strip() for line in self.recalls):
            raise PersonaError(f'persona {self.name!r} has a blank line in recalls')
        object.__setattr__(self, 'variants', MappingProxyType(dict(self.variants)))

    def for_language(self, code: str) -> 'Persona':
        """This persona with the variant for ``code`` appended, or itself."""
        extra = self.variants.get(code, '').strip()
        if not extra:
            return self
        return replace(self, prompt=f'{self.prompt.strip()}\n\n{extra}')

    def section(self, heading: str = '## Who you are') -> str:
        """This persona's part of the system prompt."""
        return section(heading, self.prompt.strip())


DEFAULT_TOOLS = ('send_message', 'accept', 'bash', 'read')
"""What a persona holds unless it says otherwise: speak, accept, and look.

``read`` is ``str_replace_editor``'s ``view`` without the commands that write, so a
user gets the line range, the directory listing and the image without a way to
change the work. A run that wants it to be able to change the work asks for
``str_replace_editor`` by name; see ``ToolSet.may_edit`` and ``prompts.identity``.

``invoke_skill`` is absent on purpose -- it is added by ``toolset.tool_names``
when a run loaded any skills, so a persona listing it would be declaring a tool
that exists only sometimes. The full catalogue is ``toolset.CATALOGUE``, which
is a different question from this one.
"""

_personas: dict[str, Persona] = {}
_builtins_loaded = False


def register_persona(persona: Persona) -> Persona:
    """Add ``persona`` to the process-wide set, refusing to shadow another."""
    if persona.name in _personas:
        raise PersonaError(f'persona {persona.name!r} is already registered')
    _personas[persona.name] = persona
    return persona


def get_persona(name: str) -> Persona:
    """The registered persona ``name``, loading the built-in files on first use."""
    _load_builtins()
    persona = _personas.get(name)
    if persona is None:
        available = ', '.join(sorted(_personas)) or '(none)'
        raise PersonaError(f'unknown persona {name!r}. Registered: {available}')
    return persona


def registered_personas() -> tuple[str, ...]:
    """Every registered name, so a test can check them all without listing them."""
    _load_builtins()
    return tuple(sorted(_personas))


def load_persona_dirs(dirs: Sequence[str | Path]) -> tuple[str, ...]:
    """Register every persona in ``dirs``, returning the names added or replaced.

    How a run brings its own: ``HITL_USER_PERSONA_DIRS`` names the directories, and a
    file there may carry a built-in's name to take its place. Registering rather than
    returning them keeps one lookup for the agent, whatever a persona came from.
    """
    from nexus.extensions.tasks.hitl.user_agent.personas.loader import load_personas

    _load_builtins()
    external = load_personas(builtin_dir=None, external_dirs=[Path(entry) for entry in dirs])
    replaced = sorted(set(external) & set(_personas))
    if replaced:
        logger.info('personas replaced by %s: %s', [str(entry) for entry in dirs], replaced)
    _personas.update(external)
    return tuple(sorted(external))


def _load_builtins() -> None:
    """Read the built-in persona files once."""
    from nexus.extensions.tasks.hitl.user_agent.personas.loader import load_personas

    global _builtins_loaded
    if _builtins_loaded:
        return
    # Set before reading, so a file that raises does not leave the next call retrying
    # into the same failure with half the set registered.
    _builtins_loaded = True
    for persona in load_personas().values():
        _personas.setdefault(persona.name, persona)


__all__ = [
    'DEFAULT_TOOLS',
    'Persona',
    'PersonaError',
    'get_persona',
    'load_persona_dirs',
    'register_persona',
    'registered_personas',
]
