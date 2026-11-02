"""Adapters from the compiled Harbor Compose contract to runtime topology types."""

from __future__ import annotations

from typing import Literal

from nexus.extensions.harbor_bridge.compose.models import ComposePlan
from nexus.runtime.topology import ComposeTopologySpec, TopologyServiceSpec


def topology_spec_from_plan(
    plan: ComposePlan,
    *,
    agent_transport: Literal['none', 'socket-v1'] = 'none',
) -> ComposeTopologySpec:
    """Convert one validated task plan into the provider-neutral runtime spec."""
    return ComposeTopologySpec(
        definition=plan.definition,
        definition_sha256=plan.definition_sha256,
        primary_service=plan.primary_service,
        startup_timeout_sec=plan.startup_timeout_sec,
        startup_jitter_max=plan.startup_jitter_max,
        start_retry_attempts=plan.start_retry_attempts,
        start_retry_backoff_sec=plan.start_retry_backoff_sec,
        start_retry_jitter_max=plan.start_retry_jitter_max,
        shutdown_timeout_sec=plan.shutdown_timeout_sec,
        services={
            name: TopologyServiceSpec(
                image=summary.image,
                one_shot=summary.one_shot,
                has_healthcheck=summary.has_healthcheck,
                depends_on=tuple(summary.depends_on),
            )
            for name, summary in plan.services.items()
        },
        bind_paths=tuple(plan.bind_paths),
        primary_environment=dict(plan.environment),
        agent_transport=agent_transport,
    )


__all__ = ['topology_spec_from_plan']
