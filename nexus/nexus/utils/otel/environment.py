from __future__ import annotations

import os

from collections.abc import Mapping
from urllib.parse import quote

from nexus.utils.otel.auto import is_enabled


_RUNTIME_EXPORTER_ENDPOINT_ENV = 'NEXUS_OTEL_RUNTIME_EXPORTER_OTLP_ENDPOINT'
_RUNTIME_PROCESS_ROLES = frozenset({'runtime-base', 'runtime-worker'})


def is_managed_otel_env_key(name: str) -> bool:
    normalized = name.upper()
    return normalized.startswith('OTEL_') or normalized.startswith('NEXUS_OTEL_')


def _render_resource_attributes(attributes: Mapping[str, str]) -> str:
    return ','.join(f'{quote(key, safe="")}={quote(value, safe="")}' for key, value in attributes.items())


def managed_otel_env(
    *,
    process_role: str,
    service_name: str,
    identifiers: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if not is_enabled():
        return {}

    overlay = {key: value for key, value in os.environ.items() if is_managed_otel_env_key(key)}
    if process_role in _RUNTIME_PROCESS_ROLES and (
        runtime_endpoint := os.getenv(_RUNTIME_EXPORTER_ENDPOINT_ENV, '').strip()
    ):
        overlay['OTEL_EXPORTER_OTLP_ENDPOINT'] = runtime_endpoint
    overlay['NEXUS_OTEL_PROCESS_ROLE'] = process_role
    overlay['OTEL_SERVICE_NAME'] = service_name
    overlay['OTEL_PROPAGATORS'] = 'tracecontext'

    if identifiers:
        inherited = overlay.get('OTEL_RESOURCE_ATTRIBUTES', '').strip().strip(',')
        rendered = _render_resource_attributes(identifiers)
        overlay['OTEL_RESOURCE_ATTRIBUTES'] = ','.join(value for value in (inherited, rendered) if value)

    return overlay


def merge_managed_otel_env(
    environment: Mapping[str, str] | None,
    *,
    process_role: str,
    service_name: str,
    identifiers: Mapping[str, str] | None = None,
) -> dict[str, str] | None:
    overlay = managed_otel_env(
        process_role=process_role,
        service_name=service_name,
        identifiers=identifiers,
    )
    if not overlay:
        return dict(environment) if environment is not None else None
    merged = {key: value for key, value in (environment or {}).items() if not is_managed_otel_env_key(key)}
    merged.update(overlay)
    return merged
