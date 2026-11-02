"""Skills a user simulator can invoke: a catalogue plus their bodies and files.

A skill is a directory holding ``SKILL.md`` and whatever the body refers to
(scripts, fixtures). The frontmatter is read for two different purposes:
``description`` goes into the catalogue the simulator sees every round, and the
body is returned only when the simulator invokes that skill by name.

The frontmatter itself is located and validated by
:mod:`nexus.utils.skill_parser`, whose ``parse_skill_file`` is not used directly
because it is async and returns no body, while this runs on the host before the
first await and needs the body.

Loading happens in two halves because the two halves have different paths
available. Scanning, validating and reading bodies runs on the host before the
sandbox exists, so it uses ``local_dir``. Invoking happens later and refers to
``sandbox_dir``, which is where :func:`upload_requests` puts the files.
"""

import logging

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nexus.utils.skill_parser import (
    MAX_DESCRIPTION_LEN,
    MAX_NAME_LEN,
    SKILLS_FILENAME,
    extract_frontmatter,
    sanitize_single_line,
    validate_field,
)


logger = logging.getLogger(__name__)


SKILL_FILE = SKILLS_FILENAME
# Absolute on purpose. ``upload_file`` resolves a non-absolute path against
# NEXUS_WORKDIR or the runtime's cwd and never expands ``~``
# (nexus/runtime/executors/file.py:34-44), so a '~/...' root would create a
# directory literally named '~'. Callers whose sandbox uses a different home
# pass ``root``.
SANDBOX_ROOT = '/root/.hitl_user/skills'


class SkillError(ValueError):
    """A skill directory cannot be used as declared."""


@dataclass(frozen=True)
class Skill:
    """One skill, with the fields each of the two halves needs.

    ``local_dir`` is only meaningful while packing, ``sandbox_dir`` only after
    the files land in the runtime. ``files`` excludes ``SKILL.md`` because its
    content is already in ``body``.
    """

    name: str
    description: str
    body: str
    local_dir: Path
    sandbox_dir: str
    files: tuple[str, ...] = ()
    origin: str = 'builtin'

    def render(self) -> str:
        """The text an invoking simulator receives."""
        parts = [f'<skill name="{self.name}">', self.body.strip(), '</skill>']
        if self.files:
            listing = '\n'.join(self.files)
            parts += [
                '',
                f'<skill_files dir="{self.sandbox_dir}">',
                listing,
                '</skill_files>',
            ]
        return '\n'.join(parts)


def sandbox_dir_for(name: str, *, root: str = SANDBOX_ROOT) -> str:
    if not root.startswith('/'):
        raise SkillError(f'skill sandbox root must be absolute, got {root!r}')
    return f'{root.rstrip("/")}/{name}'


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """The frontmatter fields and the body that follows them.

    The block is located and parsed by :mod:`nexus.utils.skill_parser`, so a
    folded scalar or an escaped quote reads here the same as it does for every
    other consumer of ``SKILL.md``. Text with no block yields an empty mapping
    and itself.
    """
    block = extract_frontmatter(text)
    if block is None:
        return {}, text
    try:
        fields = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        raise SkillError(f'frontmatter is not valid YAML: {exc}') from exc
    if not isinstance(fields, dict):
        raise SkillError('frontmatter must be a YAML mapping')
    lines = text.split('\n')
    # extract_frontmatter joins the lines between the two delimiters, so the
    # closing one sits just past as many lines as the block holds.
    closing = 1 + len(block.split('\n'))
    return fields, '\n'.join(lines[closing + 1 :])


def _field(fields: Mapping[str, Any], key: str, limit: int, manifest: Path) -> str:
    """One frontmatter field, collapsed to a line and checked for length.

    An empty ``key:`` parses as ``None``, which counts as absent rather than as
    a type error.
    """
    raw = fields.get(key) or ''
    if not isinstance(raw, str):
        raise SkillError(f'{manifest}: frontmatter {key} must be text, got {type(raw).__name__}')
    value = sanitize_single_line(raw)
    try:
        validate_field(value, limit, key)
    except ValueError as exc:
        raise SkillError(f'{manifest}: frontmatter {exc}') from exc
    return value


def read_skill(directory: Path, *, origin: str = 'builtin', root: str = SANDBOX_ROOT) -> Skill:
    """Read one skill directory.

    Raises :class:`SkillError` for anything that would show up later as a skill
    the simulator can see in the catalogue but cannot invoke.
    """
    manifest = directory / SKILL_FILE
    if not manifest.is_file():
        raise SkillError(f'{directory}: no {SKILL_FILE}')

    try:
        fields, body = split_frontmatter(manifest.read_text(encoding='utf-8'))
    except SkillError as exc:
        raise SkillError(f'{manifest}: {exc}') from exc
    name = _field(fields, 'name', MAX_NAME_LEN, manifest)
    description = _field(fields, 'description', MAX_DESCRIPTION_LEN, manifest)
    if name != directory.name:
        raise SkillError(f'{manifest}: frontmatter name {name!r} does not match directory {directory.name!r}')
    if not body.strip():
        raise SkillError(f'{manifest}: body is empty')

    files = tuple(
        sorted(
            path.relative_to(directory).as_posix()
            for path in directory.rglob('*')
            if path.is_file() and path.name != SKILL_FILE
        )
    )
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        local_dir=directory,
        sandbox_dir=sandbox_dir_for(name, root=root),
        files=files,
        origin=origin,
    )


def _skill_dirs(parent: Path) -> Iterator[Path]:
    for child in sorted(parent.iterdir()):
        if child.is_dir() and (child / SKILL_FILE).is_file():
            yield child


def load_skills(
    *,
    builtin_dir: Path | None = None,
    external_dirs: Sequence[Path] = (),
    root: str = SANDBOX_ROOT,
) -> dict[str, Skill]:
    """Load built-in skills, then external ones.

    An external skill may replace a built-in one of the same name, which is
    logged. Two external directories offering the same name is a configuration
    error and raises.
    """
    skills: dict[str, Skill] = {}

    if builtin_dir is not None and builtin_dir.is_dir():
        for directory in _skill_dirs(builtin_dir):
            skill = read_skill(directory, origin='builtin', root=root)
            skills[skill.name] = skill

    seen_external: dict[str, Path] = {}
    for parent in external_dirs:
        if not parent.is_dir():
            raise SkillError(f'external skill directory not found: {parent}')
        for directory in _skill_dirs(parent):
            skill = read_skill(directory, origin='external', root=root)
            if skill.name in seen_external:
                raise SkillError(
                    f'skill {skill.name!r} is defined by two external directories: '
                    f'{seen_external[skill.name]} and {directory}'
                )
            seen_external[skill.name] = directory
            if skill.name in skills:
                logger.info('skill %s: external %s replaces the built-in one', skill.name, directory)
            skills[skill.name] = skill

    return skills


def catalogue(skills: Mapping[str, Skill]) -> str:
    """The menu the simulator is shown: one line per skill, no bodies."""
    if not skills:
        return ''
    lines = [f'- {name}: {skills[name].description}' for name in sorted(skills)]
    return '\n'.join(lines)


def upload_requests(skills: Iterable[Skill]) -> list:
    """File items placing every skill's directory at its ``sandbox_dir``.

    Built with the same helper the SkillsBench task uses to install skills for a
    worker, so text arrives as text and binaries as base64.
    """
    from nexus.extensions.tasks.skills.utils import read_local_directory_to_file_items

    items: list = []
    for skill in skills:
        items += read_local_directory_to_file_items(skill.local_dir, skill.sandbox_dir)
    return items


__all__ = [
    'MAX_DESCRIPTION_LEN',
    'MAX_NAME_LEN',
    'SANDBOX_ROOT',
    'SKILL_FILE',
    'Skill',
    'SkillError',
    'catalogue',
    'load_skills',
    'read_skill',
    'sandbox_dir_for',
    'split_frontmatter',
    'upload_requests',
]
