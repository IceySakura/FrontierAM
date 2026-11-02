"""Explicit host egress policy for HTTP and WebSocket clients."""

from __future__ import annotations

import os

from urllib.request import getproxies, proxy_bypass

from yarl import URL


def host_trust_env() -> bool:
    return os.getenv('NEXUS_HOST_TRUST_ENV', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def host_session_kwargs() -> dict:
    return {'trust_env': True} if host_trust_env() else {}


def websocket_proxy(url: str) -> str | None:
    """Resolve WS(S) proxies only when enabled, respecting NO_PROXY."""
    if not host_trust_env():
        return None
    target = URL(url)
    if proxy_bypass(target.raw_authority):
        return None
    scheme = 'wss' if target.scheme in {'https', 'wss'} else 'ws'
    proxies = getproxies()
    return proxies.get(scheme) or proxies.get('https' if scheme == 'wss' else 'http')
