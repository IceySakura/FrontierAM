"""Reading a persona from a file, so adding one does not mean writing code.

A persona is a markdown file: YAML frontmatter for the declarations, the body for the
prompt the user reads. Every field is data -- ``Persona`` holds strings and tuples of
strings and nothing callable -- so a file carries the whole of one.

    ---
    name: default
    tools: [send_message, accept, bash, read]
    recalls:
      - "Asking: say one thing at a time."
    ---

    You are a working engineer who needs a change to a codebase you rely on.

The list fields are the ones that go wrong: a bare ``- Asking: ...`` is a YAML mapping,
not a string, so :func:`_string_tuple` refuses it by name instead of letting a dict
reach the prompt.
"""

import logging

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from nexus.extensions.tasks.hitl.user_agent.personas.registry import (
    DEFAULT_TOOLS,
    Persona,
    PersonaError,
)
from nexus.utils.skill_parser import extract_frontmatter


logger = logging.getLogger(__name__)


PERSONA_SUFFIX = '.md'

BASE_STEM = 'persona'
"""The file a persona written as a directory keeps its frontmatter and body in.

Beside it, ``persona.<code>.md`` holds the text for one natural language. The
directory's own name is the persona's name.
"""

BUILTIN_DIR = Path(__file__).parent / 'builtin'

# Frontmatter keys a file may set, by the ``Persona`` field each becomes. Listed so an
# unknown key is a refusal rather than a value silently dropped.
_SCALAR_FIELDS = ('name',)
_BOOL_FIELDS = ('reads_memory',)
_LIST_FIELDS = ('tools', 'recalls')


def read_persona(path: Path) -> Persona:
    """One persona, written either as a file or as a directory.

    A directory keeps its frontmatter and body in ``persona.md`` and its per-language
    text in ``persona.<code>.md``. A file is the whole persona on its own.

    ``name`` defaults to the file's stem or the directory's name, so ``reviewer.md``
    needs no frontmatter at all beyond what it wants to change. Declaring it and
    disagreeing with that name is refused: the name is how a run asks for it, and two
    answers would make which one wins invisible.

    ``tools`` defaults to :data:`~.registry.DEFAULT_TOOLS`.
    """
    if path.is_dir():
        return _read_directory(path)
    if not path.is_file():
        raise PersonaError(f'no persona file at {path}')
    return _read_file(path, name=path.stem)


def _read_directory(parent: Path) -> Persona:
    """A persona directory: its base file, and one variant per language beside it."""
    base = parent / f'{BASE_STEM}{PERSONA_SUFFIX}'
    if not base.is_file():
        raise PersonaError(f'persona directory {parent} has no {BASE_STEM}{PERSONA_SUFFIX}')
    variants = {}
    for path in sorted(parent.glob(f'{BASE_STEM}.*{PERSONA_SUFFIX}')):
        code = path.name[len(BASE_STEM) + 1 : -len(PERSONA_SUFFIX)]
        body = path.read_text(encoding='utf-8').strip()
        if not body:
            raise PersonaError(f'{path}: a language variant with nothing in it')
        variants[code] = body
    return replace(_read_file(base, name=parent.name), variants=variants)


def _read_file(path: Path, *, name: str) -> Persona:
    """One persona file, carrying ``name`` when its frontmatter declares none."""
    fields, body = _split(path)

    unknown = sorted(set(fields) - set(_SCALAR_FIELDS) - set(_BOOL_FIELDS) - set(_LIST_FIELDS))
    if unknown:
        known = ', '.join(sorted((*_SCALAR_FIELDS, *_BOOL_FIELDS, *_LIST_FIELDS)))
        raise PersonaError(f'{path}: unknown frontmatter {unknown}; a persona takes {known}')

    declared = fields.get('name')
    if declared is not None and not isinstance(declared, str):
        raise PersonaError(f'{path}: frontmatter name must be text, got {type(declared).__name__}')
    declared = (declared or '').strip()
    if declared and declared != name:
        raise PersonaError(f'{path}: frontmatter name {declared!r} does not match {name!r}')
    if not name:
        raise PersonaError(f'{path}: a persona needs a name')

    if not body.strip():
        raise PersonaError(f'{path}: the body is the prompt, and it is empty')

    tools = _string_tuple(fields, 'tools', path) or DEFAULT_TOOLS
    return Persona(
        name=name,
        prompt=body.strip(),
        tools=tools,
        recalls=_string_tuple(fields, 'recalls', path),
        reads_memory=_flag(fields, 'reads_memory', path, default=True),
    )


def load_personas(
    *,
    builtin_dir: Path | None = BUILTIN_DIR,
    external_dirs: Sequence[Path] = (),
) -> dict[str, Persona]:
    """Personas from the built-in directory, then from each external one in order.

    An external file may replace a built-in of the same name, which is logged: a run
    saying what the default persona should be for its own topic is the point of the
    external directories. Two external directories offering one name raises instead --
    there is no order between them to prefer.
    """
    personas: dict[str, Persona] = {}
    if builtin_dir is not None and builtin_dir.is_dir():
        for path in _persona_files(builtin_dir):
            persona = read_persona(path)
            personas[persona.name] = persona

    seen: dict[str, Path] = {}
    for parent in external_dirs:
        if not parent.is_dir():
            raise PersonaError(f'persona directory not found: {parent}')
        for path in _persona_files(parent):
            persona = read_persona(path)
            if persona.name in seen:
                raise PersonaError(f'persona {persona.name!r} is defined twice: {seen[persona.name]} and {path}')
            seen[persona.name] = path
            if persona.name in personas:
                logger.info('persona %s: %s replaces the built-in one', persona.name, path)
            personas[persona.name] = persona
    return personas


def _persona_files(parent: Path) -> Iterable[Path]:
    """Every persona under ``parent``: each ``.md`` file, and each persona directory.

    A directory counts as one only when it holds the base file. A persona directory
    alongside anything else -- ``__pycache__``, a fixtures folder -- would otherwise
    make every one of those a persona that fails to read.
    """
    return sorted(
        path
        for path in parent.iterdir()
        if (path.is_file() and path.suffix == PERSONA_SUFFIX)
        or (path.is_dir() and (path / f'{BASE_STEM}{PERSONA_SUFFIX}').is_file())
    )


def _split(path: Path) -> tuple[dict[str, Any], str]:
    """The frontmatter mapping and the body after it.

    A file with no frontmatter is a prompt and nothing else, which is a usable
    persona: the name comes from the filename and every other field has a default.

    ``extract_frontmatter`` returns None for a block with no lines in it as well as
    for a file with no delimiters, so an empty block is found here rather than left
    for its delimiters to reach the prompt.
    """
    text = path.read_text(encoding='utf-8')
    block = extract_frontmatter(text)
    if block is None:
        opening, _, rest = text.partition('---\n')
        if opening.strip() or not rest.startswith('---\n'):
            return {}, text
        return {}, rest[len('---\n') :]
    try:
        fields = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise PersonaError(f'{path}: frontmatter is not valid YAML: {exc}') from exc
    if fields is None:
        fields = {}
    if not isinstance(fields, dict):
        raise PersonaError(f'{path}: frontmatter must be a mapping, got {type(fields).__name__}')
    lines = text.split('\n')
    return fields, '\n'.join(lines[len(block.split('\n')) + 2 :])


def _flag(fields: Mapping[str, Any], key: str, path: Path, *, default: bool) -> bool:
    """One boolean field. ``yes`` and ``no`` are YAML booleans, so both forms work."""
    raw = fields.get(key)
    if raw is None:
        return default
    if not isinstance(raw, bool):
        raise PersonaError(f'{path}: frontmatter {key} must be true or false, got {type(raw).__name__}')
    return raw


def _string_tuple(fields: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    """One list field, refusing anything that is not a list of strings.

    An entry holding a colon parses as a mapping unless it is quoted, which is what
    goes wrong most: ``- Asking: one thing`` becomes ``{'Asking': 'one thing'}``. The
    refusal says so rather than reporting a type.
    """
    raw = fields.get(key)
    if raw is None:
        return ()
    if isinstance(raw, str):
        raise PersonaError(f'{path}: frontmatter {key} must be a list, not one string')
    if not isinstance(raw, list):
        raise PersonaError(f'{path}: frontmatter {key} must be a list, got {type(raw).__name__}')
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            first = next(iter(item), '')
            raise PersonaError(
                f'{path}: {key} entry {index} parsed as a mapping, so it holds an unquoted colon. '
                f'Wrap it in quotes: - "{first}: ..."'
            )
        if not isinstance(item, str):
            raise PersonaError(f'{path}: {key} entry {index} must be text, got {type(item).__name__}')
    return tuple(item.strip() for item in raw if item.strip())


__all__ = [
    'BASE_STEM',
    'BUILTIN_DIR',
    'PERSONA_SUFFIX',
    'load_personas',
    'read_persona',
]
