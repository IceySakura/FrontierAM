"""Local CDP origin compatibility without permitting arbitrary web origins."""

import shlex

from pathlib import Path


LOCAL_CDP_ORIGINS = ','.join(
    f'http://{host}:{port}' for port in (9222, 1337) for host in ('127.0.0.1', 'localhost', '[::1]')
)
LOCAL_CDP_ORIGIN_ARG = '--remote-allow-origins=' + LOCAL_CDP_ORIGINS


def add_local_cdp_origins(command: str | list) -> str | list:
    try:
        parts = [str(p) for p in command] if isinstance(command, list) else shlex.split(command)
    except ValueError:
        return command
    if not parts or Path(parts[0]).name not in {
        'google-chrome',
        'google-chrome-stable',
        'chromium',
        'chromium-browser',
    }:
        return command
    if not any(p.startswith('--remote-debugging-port=') for p in parts):
        return command
    if any(p.startswith('--remote-allow-origins') for p in parts):
        return command
    if isinstance(command, list):
        return [*parts, LOCAL_CDP_ORIGIN_ARG]
    # Do not append options to an unrelated command after shell operators.
    if any(p in {';', '&&', '||', '|', '&'} for p in parts):
        return command
    return command + ' ' + shlex.quote(LOCAL_CDP_ORIGIN_ARG)
