"""CUA-Gym mock web app startup helpers."""

from __future__ import annotations

import re
import shlex

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nexus.extensions.tasks.cua_gym.materializer import detect_placeholders, load_env_replacements
from nexus.extensions.tasks.cua_gym.setup import cua_gym_user_command


MockHubMode = Literal['disabled', 'external', 'local']
URL_PLACEHOLDER_PATTERN = re.compile(r'^__([A-Z0-9_]+(?:URL|HOST))__$')


@dataclass(frozen=True)
class MockHubResult:
    replacements: dict[str, str] = field(default_factory=dict)
    started_apps: dict[str, str] = field(default_factory=dict)
    uploaded_apps: list[str] = field(default_factory=list)
    missing_placeholders: list[str] = field(default_factory=list)


def url_placeholders(bundle_dir: str | Path) -> list[str]:
    """Return URL/HOST placeholders that need mock app endpoints."""
    return [token for token in detect_placeholders(bundle_dir) if URL_PLACEHOLDER_PATTERN.match(token)]


def placeholder_to_app_name(placeholder: str) -> str:
    """Map a CUA-Gym URL placeholder to its mock hub app directory name."""
    name = placeholder.strip('_').lower()
    if name.startswith('cua_gym_'):
        name = name.removeprefix('cua_gym_')
    for suffix in ('_base_url', '_url', '_host'):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name.endswith('_mock'):
        name = f'{name}_mock'
    return name


async def _start_local_app(
    runtime: Any,
    *,
    hub_root: str,
    app_name: str,
    port: int,
    timeout: int,
    display: str,
) -> str:
    app_dir = f'{hub_root.rstrip("/")}/websites/{app_name}'
    url = f'http://127.0.0.1:{port}'
    app_dir_q = shlex.quote(app_dir)
    log_q = shlex.quote(f'/tmp/cua-gym-hub-{app_name}.log')
    command = (
        f'test -d {app_dir_q} || {{ echo "CUA-Gym-Hub app is missing from image: {app_name}" >&2; exit 1; }}; '
        f'cd {app_dir_q} && '
        'test -d node_modules || { echo "CUA-Gym-Hub app is not prebuilt: missing node_modules" >&2; exit 1; } && '
        'test -d dist || { echo "CUA-Gym-Hub app is not prebuilt: missing dist" >&2; exit 1; } && '
        'mkdir -p .mock-states .mock-files && '
        f'(nohup npm run preview -- --host 127.0.0.1 --port {port} >{log_q} 2>&1 &)'
    )
    response = await runtime.run_command(command=cua_gym_user_command(command, display=display), timeout=timeout)
    if getattr(response, 'return_code', None) != 0:
        raise RuntimeError(f'Failed to start CUA-Gym mock app {app_name}: {getattr(response, "stderr", "")}')

    health_command = (
        f'for i in $(seq 1 30); do '
        f'(curl -fsS {shlex.quote(url)}/go?sid=nexus-health >/dev/null 2>&1 || '
        f'curl -fsS {shlex.quote(url)}/ >/dev/null 2>&1) && exit 0; '
        'sleep 0.5; '
        'done; '
        f'tail -100 {log_q} 2>/dev/null; exit 1'
    )
    health = await runtime.run_command(command=cua_gym_user_command(health_command, display=display), timeout=timeout)
    if getattr(health, 'return_code', None) != 0:
        raise RuntimeError(f'CUA-Gym mock app {app_name} did not become healthy: {getattr(health, "stdout", "")}')
    return url


async def prepare_mock_hub(
    runtime: Any,
    *,
    bundle_dir: str | Path,
    mode: MockHubMode = 'local',
    hub_root: str = '/opt/cua-gym-hub',
    start_port: int = 18000,
    env_file: str | Path | None = None,
    timeout: int = 60,
    display: str = ':0',
) -> MockHubResult:
    """Resolve mock URL placeholders and optionally start required hub apps."""
    placeholders = url_placeholders(bundle_dir)
    if not placeholders:
        return MockHubResult()

    replacements = load_env_replacements(env_file)
    missing = [placeholder for placeholder in placeholders if placeholder not in replacements]
    if mode == 'external':
        return MockHubResult(replacements=replacements, missing_placeholders=missing)
    if mode == 'disabled':
        return MockHubResult(replacements=replacements, missing_placeholders=missing)

    app_ports: dict[str, int] = {}
    started_apps: dict[str, str] = {}
    uploaded_apps: list[str] = []
    for placeholder in placeholders:
        if placeholder in replacements:
            continue
        app_name = placeholder_to_app_name(placeholder)
        if app_name not in app_ports:
            app_ports[app_name] = start_port + len(app_ports)
            started_apps[app_name] = await _start_local_app(
                runtime,
                hub_root=hub_root,
                app_name=app_name,
                port=app_ports[app_name],
                timeout=timeout,
                display=display,
            )
        endpoint = started_apps[app_name]
        replacements[placeholder] = endpoint.removeprefix('http://') if placeholder.endswith('_HOST__') else endpoint

    return MockHubResult(
        replacements=replacements,
        started_apps=started_apps,
        uploaded_apps=uploaded_apps,
        missing_placeholders=[],
    )


__all__ = [
    'MockHubMode',
    'MockHubResult',
    'placeholder_to_app_name',
    'prepare_mock_hub',
    'url_placeholders',
]
