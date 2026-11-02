"""Credential-free, task-scoped network environment for OSWorld guest tools."""

from collections.abc import Mapping
from typing import Any


OSWORLD_PROXY_SERVER = 'http://127.0.0.1:18888'
_PROXY_ENV_KEYS = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY')
_LOOPBACK_HOSTS = ('localhost', '127.0.0.1', '::1', '[::1]')


def guest_proxy_env(existing: Mapping[str, str]) -> dict[str, str]:
    """Route tools through the guest proxy, never the runner's host proxy.

    Call only for tasks whose proxy was enabled by the OSWorld task adapter.
    Preserve explicit bypasses and always keep the local Bridge/CDP direct.
    No upstream proxy address or credentials are passed to Claude Code.
    """
    bypasses = dict.fromkeys(_LOOPBACK_HOSTS)
    for key in ('no_proxy', 'NO_PROXY'):
        for host in existing.get(key, '').split(','):
            if host.strip():
                bypasses.setdefault(host.strip(), None)
    no_proxy = ','.join(bypasses)
    return {
        **dict.fromkeys(_PROXY_ENV_KEYS, OSWORLD_PROXY_SERVER),
        'no_proxy': no_proxy,
        'NO_PROXY': no_proxy,
    }


def proxy_setup_failure_message(result: Mapping[str, Any]) -> str:
    """Expose useful failure stages without leaking proxy credentials or commands."""
    upload = result.get('upload')
    upload = upload if isinstance(upload, Mapping) else {}

    def scalar(value: object) -> object:
        return value if isinstance(value, (bool, int)) else None

    # Never log arbitrary stdout/stderr: an upstream proxy may echo its URL,
    # credentials or configuration. Recognize only fixed, non-sensitive causes.
    stderr = str(result.get('stderr') or '')
    reason = next(
        (text for text in ('tinyproxy: not found', 'No space left on device', 'Connection refused') if text in stderr),
        'proxy setup did not become ready',
    )
    stage = result.get('stage')
    if stage not in ('upload', 'install', 'launch', 'probe', 'ready'):
        stage = 'unknown'
    return (
        f'OSWorld proxy setup failed: stage={stage} '
        f'upload_success={scalar(upload.get("success"))} '
        f'upload_status_code={scalar(upload.get("status_code"))} '
        f'command_return_code={scalar(result.get("return_code"))}; {reason}'
    )
