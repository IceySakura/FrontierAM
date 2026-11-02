"""Outbound guard: which messages may be sent to a worker harness.

A harness command is a message whose first token names it -- ``/verify``, or
``/verify the auth module`` with an instruction after it. Only the first token is
read, so a command word later in a sentence is prose.

Three refusal classes: a name the run does not have, and the two sets of names the
manager declares -- ``denied_commands`` and ``unsupported_commands``. Malformed
near-misses -- a backtick-wrapped name, a leading path, the prefix alone -- are
refused with what to write instead.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType


WORKER_COMMANDS_KEY = 'worker_commands'
"""``worker_config`` key holding the skills a run installed, name -> what it covers."""

BUILTIN_COMMANDS_KEY = 'builtin_commands'
"""``worker_config`` key naming which of the harness's own commands a run offers.

A list of names. Absent or empty offers none of them. What each name means is the
backend's own ``BUILTIN_COMMANDS``.
"""


def offered_builtins(available: Mapping[str, str], names: Collection[str]) -> Mapping[str, str]:
    """The entries of ``available`` that ``names`` asks for.

    Raises for a name outside ``available``, which the harness answers
    ``Unknown command`` to.
    """
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f'unknown built-in command(s) {unknown}; this harness has {sorted(available)}')
    return MappingProxyType({name: available[name] for name in sorted(names)})


# Stripped from a token's tail before it is matched.
_TRAILING = '.,;:!?)]}\'"`'


@dataclass(frozen=True)
class OutgoingCheck:
    """Verdict for one outbound message."""

    allowed: bool
    reason: str = ''

    def __bool__(self) -> bool:
        return self.allowed


_ALLOWED = OutgoingCheck(allowed=True)


def first_token(text: str) -> str:
    """The leading whitespace-delimited token, lowercased, without its tail marks."""
    stripped = text.strip()
    if not stripped:
        return ''
    return stripped.split(maxsplit=1)[0].casefold().rstrip(_TRAILING)


def command_name(text: str, *, prefix: str) -> str:
    """The command ``text`` names, or '' when its first token is not one.

    A leading path returns what follows the prefix (``/tmp/x.py`` -> ``tmp/x.py``),
    which the harness would read as a command name.
    """
    if not prefix:
        return ''
    token = first_token(text)
    if not token.startswith(prefix) or token == prefix:
        return ''
    return token[len(prefix) :]


def check_outgoing(
    text: str,
    *,
    allowed: Collection[str] = frozenset(),
    prefix: str = '',
    denied: Collection[str] = frozenset(),
    unsupported: Collection[str] = frozenset(),
) -> OutgoingCheck:
    """Whether ``text`` may be sent, given what this harness accepts.

    ``allowed`` holds the command names this run has, without the prefix. ``prefix``
    is ``SessionManager.command_prefix``; empty means the harness has no command form.
    ``denied`` and ``unsupported`` are the manager's own attributes of those names.

    A refusal says what was not sent and what there is instead. It does not explain
    the harness: an unregistered name reaches the user as a message it can rewrite,
    and why the host stopped it is this module's concern, not the user's.

    An unregistered name is refused rather than sent because Claude Code answers one
    locally without producing an assistant turn, so nothing signals completion and the
    turn waits out its timeout -- 900 seconds and the session, on run 0821a.
    """
    stripped = text.strip()
    quoted = _quoted_command(stripped, prefix=prefix)
    if quoted:
        return OutgoingCheck(
            allowed=False,
            reason=(f'Not sent. Write it as {prefix}{quoted} with no backticks, as the first thing in the message.'),
        )

    if not prefix:
        if stripped.startswith('/'):
            return OutgoingCheck(
                allowed=False,
                reason='Not sent: this worker has no commands. Say what you want instead.',
            )
        return _ALLOWED

    name = command_name(stripped, prefix=prefix)
    if not name:
        if stripped == prefix:
            return OutgoingCheck(
                allowed=False,
                reason=f'Not sent: {prefix} on its own names no command.{_have(allowed, prefix)}',
            )
        return _ALLOWED

    if name in denied or name in unsupported:
        return OutgoingCheck(
            allowed=False,
            reason=f'Not sent: {prefix}{name} cannot be sent to this worker.{_have(allowed, prefix)}',
        )
    if name not in allowed:
        return OutgoingCheck(allowed=False, reason=_unknown_reason(name, allowed=allowed, prefix=prefix))
    return _ALLOWED


def _quoted_command(text: str, *, prefix: str) -> str:
    """The command name inside leading backticks, or '' when there are none."""
    if not prefix or not text.startswith('`'):
        return ''
    closing = text.find('`', 1)
    inner = (text[1:closing] if closing > 1 else text[1:]).strip()
    return command_name(inner, prefix=prefix)


def _have(allowed: Collection[str], prefix: str) -> str:
    """What this worker does take, as a clause to append to a refusal."""
    if not allowed:
        return ' It has no commands; say what you want instead.'
    return ' It has: ' + ', '.join(f'{prefix}{known}' for known in sorted(allowed))


def _unknown_reason(name: str, *, allowed: Collection[str], prefix: str) -> str:
    """What was not sent, and what there is instead."""
    if '/' in name or '.' in name:
        return (
            f'Not sent: a message starting with {prefix} is read as a command, and {prefix}{name} is '
            'not one. Put the path later in the sentence.'
        )
    return f"Not sent: {prefix}{name} is not one of this worker's commands.{_have(allowed, prefix)}"


__all__ = [
    'BUILTIN_COMMANDS_KEY',
    'WORKER_COMMANDS_KEY',
    'OutgoingCheck',
    'check_outgoing',
    'command_name',
    'first_token',
    'offered_builtins',
]
