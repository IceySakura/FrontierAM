"""Shared command helpers for OSWorld-compatible local runtime actions."""

from __future__ import annotations

import shlex

from nexus.runtime.utils.gui import session_user_command


SHELL_OPERATORS = {'|', '&&', '||', ';', '>', '>>', '<', '2>&1'}


def quote_command_part(part: str) -> str:
    """Quote one command-list part while preserving shell operators used by OSWorld JSON."""
    if part in SHELL_OPERATORS:
        return part
    return shlex.quote(part)


def command_to_shell(command: str | list[str]) -> str:
    """Convert an OSWorld command field into a shell command string."""
    if isinstance(command, list):
        return ' '.join(quote_command_part(str(part)) for part in command)
    return command


def osworld_user_command(command: str, *, user: str = 'user', home_dir: str = '/home/user') -> str:
    """Wrap a shell command so local OSWorld actions use the standard /home/user account."""
    return session_user_command(command, user=user, home_dir=home_dir, missing_user_policy='fallback')


__all__ = ['command_to_shell', 'osworld_user_command', 'quote_command_part']
