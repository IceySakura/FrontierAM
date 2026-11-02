from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nexus.distributed.config.model import DistributedConfig, RunTraceSinkConfig


@dataclass(frozen=True, slots=True)
class RunTraceBackendMetadata:
    name: str
    kind: str
    display_name: str
    description: str | None
    user_selectable: bool
    self_grantable: bool
    system_managed: bool
    required: bool


_SYSTEM_BACKENDS = {
    'artifact': RunTraceBackendMetadata(
        name='artifact',
        kind='artifact',
        display_name='Artifact Storage',
        description=None,
        user_selectable=False,
        self_grantable=False,
        system_managed=True,
        required=True,
    ),
    'local': RunTraceBackendMetadata(
        name='local',
        kind='local',
        display_name='Local Storage',
        description=None,
        user_selectable=False,
        self_grantable=False,
        system_managed=True,
        required=False,
    ),
}


def run_trace_backend_catalog(config: DistributedConfig) -> tuple[RunTraceBackendMetadata, ...]:
    catalog = dict(_SYSTEM_BACKENDS)
    for name, backend in config.run_trace_sinks.items():
        system_backend = _SYSTEM_BACKENDS.get(name)
        catalog[name] = _metadata_from_config(name, backend, system_backend=system_backend)
    return tuple(catalog[name] for name in sorted(catalog))


def _metadata_from_config(
    name: str,
    backend: RunTraceSinkConfig,
    *,
    system_backend: RunTraceBackendMetadata | None,
) -> RunTraceBackendMetadata:
    if system_backend is not None:
        return RunTraceBackendMetadata(
            name=name,
            kind=backend.kind if 'kind' in backend.model_fields_set else system_backend.kind,
            display_name=backend.display_name or system_backend.display_name,
            description=backend.description,
            user_selectable=False,
            self_grantable=False,
            system_managed=True,
            required=system_backend.required,
        )
    return RunTraceBackendMetadata(
        name=name,
        kind=backend.kind,
        display_name=backend.display_name or name,
        description=backend.description,
        user_selectable=backend.user_selectable,
        self_grantable=backend.self_grantable,
        system_managed=backend.system_managed,
        required=backend.required,
    )


__all__ = ['RunTraceBackendMetadata', 'run_trace_backend_catalog']
