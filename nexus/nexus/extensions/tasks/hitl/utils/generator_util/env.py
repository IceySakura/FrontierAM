"""Reading a generator's environment, and loading its dataset.

A launch script is a shell script, so a variable's value is whatever someone typed.
Each reader here refuses a value it cannot mean rather than falling back to the
default, which is what turns a typo into a run that finishes and reports a number.
"""

import json
import os

from collections.abc import Iterable
from pathlib import Path
from typing import Any


class EnvValueError(ValueError):
    """An environment variable is set to something it cannot mean."""


TRUE_VALUES = ('1', 'true', 'yes', 'on')
FALSE_VALUES = ('0', 'false', 'no', 'off')


def _env_int(key: str, default: int = 0) -> int:
    """The integer ``key`` is set to, or ``default`` when it is unset.

    Raises :class:`EnvValueError` for a value that is not an integer.
    """
    raw = os.environ.get(key, '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise EnvValueError(f'{key}={raw!r} is not an integer') from None


def _env_bool(key: str, default: bool = False) -> bool:
    """Whether ``key`` is set to a true value, or ``default`` when it is unset.

    Raises :class:`EnvValueError` for anything outside :data:`TRUE_VALUES` and
    :data:`FALSE_VALUES`.
    """
    raw = os.environ.get(key, '').strip().lower()
    if not raw:
        return default
    if raw in TRUE_VALUES:
        return True
    if raw in FALSE_VALUES:
        return False
    raise EnvValueError(f'{key}={raw!r} is not a boolean; use one of {[*TRUE_VALUES, *FALSE_VALUES]}')


def _env_choice(key: str, default: str, choices: Iterable[str]) -> str:
    """The value of ``key``, which must name one of ``choices``.

    For a variable whose value the run then looks up. Raises :class:`EnvValueError`
    for a name outside ``choices``.
    """
    allowed = tuple(choices)
    raw = os.environ.get(key, '').strip()
    if not raw:
        return default
    if raw not in allowed:
        raise EnvValueError(f'{key}={raw!r} is not one of {sorted(allowed)}')
    return raw


def _env_list(key: str) -> list[str]:
    """The comma-separated items ``key`` is set to, blanks dropped."""
    raw = os.environ.get(key, '').strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(',') if item.strip()]


def _require_env(key: str, msg: str) -> str:
    """The value of ``key``, raising ``ValueError(msg)`` when it is unset."""
    value = os.environ.get(key, '').strip()
    if not value:
        raise ValueError(msg)
    return value


def _load_dataset(path: str) -> list[dict[str, Any]]:
    """The instances in a ``.jsonl`` file, or in a JSON list or single object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Dataset not found: {path}')
    instances: list[dict[str, Any]] = []
    if p.suffix == '.jsonl':
        with open(p, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line:
                    instances.append(json.loads(line))
    else:
        data = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict):
            instances = [data]
    return instances


__all__ = [
    'FALSE_VALUES',
    'TRUE_VALUES',
    'EnvValueError',
    '_env_bool',
    '_env_choice',
    '_env_int',
    '_env_list',
    '_load_dataset',
    '_require_env',
]
