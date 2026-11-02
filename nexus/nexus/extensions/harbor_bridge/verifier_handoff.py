#!/usr/bin/env python3
"""How Harbor Bridge hands something it built over to a verifier.

Every handoff takes the same three steps: the payload is written once into the Host-private
trial directory, uploaded to a fixed path under the container's agent log dir, and announced
through an environment variable the task may not redefine. The modules above this one decide
*what* is handed over; this one only performs the handoff.

The fixed address is the reason to keep those steps together. A task free to point
``T3_AGENT_RESULT_PATH`` at a file it wrote itself would be grading its own homework, so the
address is not a convention each caller re-implements but part of what a handoff means here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexus.extensions.harbor_bridge.upload import raise_for_upload_failures


def write_for_verifier(trial_paths: Any, filename: str, payload: Any) -> Path:
    """Persist one payload in the Host-private Harbor trial directory."""
    path = trial_paths.agent_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return path


def verifier_path_for(environment_os: Any, filename: str) -> str:
    """Return the address a verifier reads this file from, for the target environment OS."""
    from harbor.models.trial.paths import EnvironmentPaths

    return str(EnvironmentPaths.for_os(environment_os).agent_dir / filename)


async def upload_to_verifier(
    runtime: Any,
    host_path: Path,
    *,
    environment_os: Any,
    filename: str,
    label: str,
) -> str:
    """Place one Host file at that address, in a Shared or Separate verifier runtime."""
    from nexus.runtime.executors.file import FileItem

    container_path = verifier_path_for(environment_os, filename)
    response = await runtime.upload_file(
        files=[FileItem(path=container_path, content=host_path.read_text(encoding='utf-8'), encoding='utf8')],
        overwrite=True,
    )
    raise_for_upload_failures(response, label)
    return container_path


def verifier_env_with_reserved_path(
    verifier_env: dict[str, str],
    *,
    environment_os: Any,
    filename: str,
    env_name: str,
) -> dict[str, str]:
    """Announce the address, refusing a task's attempt to name a different one."""
    container_path = verifier_path_for(environment_os, filename)
    configured = verifier_env.get(env_name)
    if configured is not None and configured != container_path:
        raise ValueError(f'{env_name} is reserved by Harbor Bridge and must be {container_path!r}, got {configured!r}')
    return {**verifier_env, env_name: container_path}


__all__ = [
    'upload_to_verifier',
    'verifier_env_with_reserved_path',
    'verifier_path_for',
    'write_for_verifier',
]
