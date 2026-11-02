"""HyCode/OpenCC exits that remain eligible for task scoring."""

from __future__ import annotations


SCOREABLE_EXIT_NAMES = {
    -9: 'SIGKILL',
    1: 'GENERAL_ERROR',
    143: 'SIGTERM',
    # Host got no exit_code key while the run completed. Not -1: the sandbox writes that on the
    # infra early-returns in runtime_setup.py, which stay out of this table (UNHANDLED_ERROR).
    -5: 'EXIT_MISSING',
}
SCOREABLE_EXIT_CODES = frozenset(SCOREABLE_EXIT_NAMES)


def is_scoreable_exit(exit_code: object) -> bool:
    return isinstance(exit_code, int) and exit_code in SCOREABLE_EXIT_CODES


def scoreable_exit_name(exit_code: object) -> str | None:
    if not isinstance(exit_code, int):
        return None
    return SCOREABLE_EXIT_NAMES.get(exit_code)


__all__ = ['SCOREABLE_EXIT_CODES', 'SCOREABLE_EXIT_NAMES', 'is_scoreable_exit', 'scoreable_exit_name']
