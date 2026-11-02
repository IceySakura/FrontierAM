"""The skill directories a worker harness reads, placed in the sandbox.

Where they go is the harness's to declare (``SessionManager.skills_path``); which
directories go there is the run's, passed through ``worker_config`` as
``worker_skills``. A run naming none, or a harness declaring no path, places nothing.

The home directory is read sandbox-side and the files are uploaded to the absolute
path that gives, so ``upload_file`` never has to expand ``~`` or resolve against a cwd
it does not know (``nexus/runtime/executors/file.py``).
"""

import logging
import os

from collections.abc import Mapping, Sequence
from pathlib import Path

from nexus.extensions.tasks.hitl.utils import coerce
from nexus.runtime import Runtime


logger = logging.getLogger(__name__)


CONFIG_KEY = 'worker_skills'


def skill_dirs(config: Mapping[str, object]) -> list[str]:
    """The local skill directories this run was configured with."""
    entries = coerce.optional_str_list(config.get(CONFIG_KEY)) or []
    return [entry for entry in entries if entry.strip()]


def sandbox_home() -> str:
    """Sandbox-side: the worker's own home directory."""
    return os.environ.get('HOME', '/root')


async def place_worker_skills(runtime: Runtime, config: Mapping[str, object], *, skills_path: str) -> None:
    """Upload this run's skill directories into the sandbox, if it has any.

    Must run before the harness starts: Claude Code reads the skills it offers when a
    session starts.

    Raises ``RuntimeError`` when the home directory cannot be read or an upload fails.
    A worker whose skills silently never arrived still advertises them through
    ``worker_commands``, so the user simulator asks for a command that does nothing.
    """
    directories = skill_dirs(config)
    if not directories or not skills_path:
        return

    try:
        home = await runtime.invoke(sandbox_home)
    except Exception as exc:
        raise RuntimeError(f'could not read the sandbox home directory to place skills: {exc}') from exc

    root = f'{str(home).rstrip("/")}/{skills_path.strip("/")}'
    items = _file_items(directories, root)
    if not items:
        raise RuntimeError(f'skill directories {directories} hold no files')

    response = await runtime.upload_file(files=items)
    failed = [result for result in response.results if not result.success]
    if failed:
        detail = '; '.join(f'{result.path}: {result.message}' for result in failed)
        raise RuntimeError(f'could not place worker skill files in the sandbox: {detail}')
    logger.info('placed %d skill files for %d skills at %s', len(items), len(directories), root)


def _file_items(directories: Sequence[str], root: str) -> list:
    """File items placing each directory at ``<root>/<basename>/``.

    Uses the helper the SkillsBench task installs worker skills with, so text arrives
    as text and a binary as base64 -- a skill carrying a font or an image is a skill
    like any other.
    """
    from nexus.extensions.tasks.skills.utils import read_local_directory_to_file_items

    items: list = []
    for entry in directories:
        local = Path(entry).expanduser()
        items += read_local_directory_to_file_items(local, f'{root}/{local.name}')
    return items


__all__ = ['CONFIG_KEY', 'place_worker_skills', 'sandbox_home', 'skill_dirs']
