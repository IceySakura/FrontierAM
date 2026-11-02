"""Host platform predicates shared across Nexus modules."""

import sys


def is_windows() -> bool:
    """Return whether the current interpreter runs on native Windows."""
    return sys.platform == 'win32'
