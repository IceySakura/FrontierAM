"""Removed dataset root materialization helpers.

Dataset items now expose files through ``task_files``. Use
``nexus.dataset.task_files.task_files_from_spec`` for direct access.
"""

from __future__ import annotations

from typing import Any


def has_root_references(_value: Any) -> bool:
    """Old ``@/`` root references are no longer supported."""

    return False


def materialize_root(*_args: Any, **_kwargs: Any) -> None:
    """Fail fast for the removed root materialization API."""

    raise RuntimeError('Dataset root materialization was removed; use task_files instead')


def resolve_root_references(value: Any, *_args: Any, **_kwargs: Any) -> Any:
    """Old ``@/`` root references are no longer resolved."""

    return value
