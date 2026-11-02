"""Skill loading, and placing a skill's files in the sandbox."""

import logging

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hitl.user_agent.skills.loader import (
    SANDBOX_ROOT,
    Skill,
    SkillError,
    catalogue,
    load_skills,
    read_skill,
    upload_requests,
)


logger = logging.getLogger(__name__)

BUILTIN_DIR = Path(__file__).resolve().parent


def load_user_skills(*, builtin_dir: Path | None = BUILTIN_DIR, external: Sequence[str] = ()) -> dict[str, Skill]:
    """Built-in skills plus any directories a run was pointed at."""
    return load_skills(
        builtin_dir=builtin_dir,
        external_dirs=[Path(entry).expanduser().resolve() for entry in external if entry.strip()],
    )


async def install_skills(runtime: Any, skills: Mapping[str, Skill]) -> None:
    """Place every skill's files where its body says they are.

    Raises when any upload failed, naming the paths.
    """
    if not skills:
        return
    items = upload_requests(skills.values())
    if not items:
        return
    response = await runtime.upload_file(files=items)
    failed = [result for result in response.results if not result.success]
    if failed:
        detail = '; '.join(f'{result.path}: {result.message}' for result in failed)
        raise RuntimeError(f'could not place skill files in the sandbox: {detail}')
    logger.info('placed %d skill files for %d skills', len(items), len(skills))


__all__ = [
    'BUILTIN_DIR',
    'SANDBOX_ROOT',
    'Skill',
    'SkillError',
    'catalogue',
    'install_skills',
    'load_skills',
    'load_user_skills',
    'read_skill',
    'upload_requests',
]
