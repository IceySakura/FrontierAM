"""Environment handed to the tools that run arbitrary code.

``bash``, ``curl`` and the ``python`` REPL are real subprocesses, so whatever
is in this process's environment is one ``env`` call away from the model. The
harness is normally started with endpoint credentials there, and a rollout
that prints them has leaked them into every trace anyone later reads.

So the child gets a filtered copy: everything except the variables whose *name*
suggests a secret. Name-matching rather than value-matching is the point — a
value-based scrub has to be told what the secret is, while a new
``FOO_API_TOKEN`` is caught by this the day it is introduced.

The model-facing ``bash`` and ``python`` tools additionally use
``main_image_env``. It removes the Nexus Nix bin directory from ``PATH`` so
both tools resolve the Python supplied by the main sandbox image instead of
the interpreter used to run the Runtime server or Worker.

This is not a sandbox. A tool that can run ``bash`` can read the API key out of
a config file, and nothing here stops it. It only ensures the harness itself is
not the one handing the key over.
"""

from __future__ import annotations

import os
import pwd
import shutil

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from nexus.utils.shell import get_clean_sandbox_env


# Substring match, case-insensitive, against the variable name.
_SECRET_MARKERS: tuple[str, ...] = (
    'key',
    'token',
    'secret',
    'password',
    'passwd',
    'credential',
    'auth',
)

# Kept despite matching a marker above: these change how a subprocess behaves
# rather than granting access to anything, and dropping them breaks real
# commands (``curl`` needs the CA bundle to verify TLS at all).
_KEEP: frozenset[str] = frozenset({
    'SSL_CERT_FILE',
    'SSL_CERT_DIR',
    'REQUESTS_CA_BUNDLE',
    'CURL_CA_BUNDLE',
    'GIT_SSL_CAINFO',
})


def is_secret_name(name: str) -> bool:
    """Whether *name* looks like it holds a credential."""
    if name.upper() in _KEEP:
        return False
    low = name.lower()
    return any(marker in low for marker in _SECRET_MARKERS)


def scrub_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return *env* minus credential-looking variables."""
    source = os.environ if env is None else env
    return {key: value for key, value in source.items() if not is_secret_name(key)}


def main_image_env() -> dict[str, str]:
    """Return a secret-scrubbed environment for main-image commands.

    Runtime and Worker processes prepend ``/nix/runtime/nix-env/bin`` to
    ``PATH``. Removing that entry is what makes child commands resolve the
    Python, pip, curl, and other tools supplied by the main sandbox image.
    """
    env = scrub_env(get_clean_sandbox_env())
    identity = _EXECUTION_IDENTITY.get()
    if identity is not None:
        account = pwd.getpwuid(identity['uid'])
        env.update(HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name)
        # Worker-specific Python/shell/cache settings may point into /root.
        for key in (
            'PYTHONPATH',
            'PYTHONHOME',
            'VIRTUAL_ENV',
            'BASH_ENV',
            'ENV',
            'XDG_CACHE_HOME',
            'XDG_CONFIG_HOME',
            'XDG_DATA_HOME',
            'PIP_TARGET',
            'PIP_PREFIX',
        ):
            env.pop(key, None)
    return env


def resolve_main_image_python(env: Mapping[str, str] | None = None) -> str:
    """Resolve the Python interpreter exposed by the main sandbox image."""
    command_env = main_image_env() if env is None else env
    path = command_env.get('PATH', '')
    for executable in ('python', 'python3'):
        resolved = shutil.which(executable, path=path)
        if resolved is not None:
            return resolved
    raise RuntimeError('Main sandbox image does not provide python or python3 on PATH')


_EXECUTION_IDENTITY: ContextVar[dict[str, int] | None] = ContextVar('context_agent_execution_identity', default=None)


@contextmanager
def execution_identity(identity: Mapping[str, Any] | None):
    """Scope subprocess credentials to this tool call, including to_thread work."""
    value = None if identity is None else {'uid': int(identity['uid']), 'gid': int(identity['gid'])}
    if value is not None and (value['uid'] <= 0 or value['gid'] < 0):
        raise ValueError('Agent sandbox identity must have a non-root UID and nonnegative GID')
    token = _EXECUTION_IDENTITY.set(value)
    try:
        yield
    finally:
        _EXECUTION_IDENTITY.reset(token)


def subprocess_identity_args() -> dict[str, Any]:
    identity = _EXECUTION_IDENTITY.get()
    if identity is None:
        return {}
    return {'user': identity['uid'], 'group': identity['gid'], 'extra_groups': [], 'umask': 0o022}


def sandbox_command(command: list[str]) -> list[str]:
    """Prevent sudo/setuid binaries from regaining privileges in agent children."""
    if _EXECUTION_IDENTITY.get() is None:
        return command
    executable = shutil.which('setpriv', path=main_image_env().get('PATH', ''))
    if executable is None:
        raise RuntimeError('setpriv is required for unprivileged agent execution')
    return [executable, '--no-new-privs', '--', *command]
