"""Codex interactive agent — runs the native Codex TUI behind a PTY.

Unlike the ``codex`` agent (``codex exec --ephemeral``, one shot then exit), this
keeps one Codex TUI alive across turns so context, workspace, and terminal state
survive, and so the interactive-only code path is exercised.

Topology (verified end to end before implementation):

    PTY ── codex TUI ──► in-session relay ──► codex app-server
                              │
                              └── taps the TUI's own downstream traffic

The relay forwards every TUI byte verbatim. The first prompt is a native TUI CLI
argument; follow-ups use the PTY. Captured app-server requests
are answered with a method-specific JSON-RPC response on that same WebSocket
connection. The relay also reads the same ``turn/started`` /
``turn/completed`` notifications the TUI receives, which makes turn boundaries
trustworthy without screen scraping.

Modules:
  * ``appserver.py`` — app-server process + relay + frame decoding/response sending
  * ``boundary.py``  — turn boundary adapter: thread/turn correlation, terminal states
  * ``screen.py``    — thin wrapper over the shared ``VirtualScreen`` (pyte)
  * ``events.py``    — bounded event buffer with cursor commit
  * ``session.py``   — PTY driver and session state machine
  * ``runtime.py``   — worker-side registry and the four runtime entry points
"""

from __future__ import annotations


# 0.131.0 is the first verified stable release where both the TUI remote endpoint
# and app-server listener support the unix:// topology used here. Newer stable
# releases are allowed; compatibility is verified through 0.156.1, while 0.155.1
# remains the default.
INTERACTIVE_DEFAULT_CODEX_VERSION = '0.155.1'
INTERACTIVE_MIN_CODEX_VERSION = '0.131.0'


class CodexVersionUnsupportedError(RuntimeError):
    """Raised when the configured Codex cannot support the interactive topology."""


def validate_interactive_version(version: str) -> None:
    """Require the first protocol-compatible stable release or newer."""
    from packaging.version import InvalidVersion, Version

    try:
        parsed = Version(version)
    except InvalidVersion:
        raise CodexVersionUnsupportedError(
            f'codex_interactive requires an official stable Codex version >= '
            f'{INTERACTIVE_MIN_CODEX_VERSION}; got {version!r}'
        ) from None
    supported = (
        not parsed.is_prerelease
        and not parsed.is_devrelease
        and not parsed.is_postrelease
        and parsed.local is None
        and parsed >= Version(INTERACTIVE_MIN_CODEX_VERSION)
    )
    if not supported:
        raise CodexVersionUnsupportedError(
            f'codex_interactive supports official stable Codex releases >= '
            f'{INTERACTIVE_MIN_CODEX_VERSION}; got {version!r}. Versions below 0.131 lack the required '
            'unix:// remote topology, and prerelease, postrelease, dev, or locally tagged builds are rejected.'
        )


__all__ = [
    'INTERACTIVE_DEFAULT_CODEX_VERSION',
    'INTERACTIVE_MIN_CODEX_VERSION',
    'CodexVersionUnsupportedError',
    'validate_interactive_version',
]
