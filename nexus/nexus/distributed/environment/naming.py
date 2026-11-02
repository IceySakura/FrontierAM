from __future__ import annotations

import re

from nexus.distributed.utils.identity import canonical_sha256


def dns_name(value: str, *, fallback: str) -> str:
    normalized = re.sub(r'[^a-z0-9-]+', '-', value.lower()).strip('-')
    if not normalized:
        return fallback
    if len(normalized) <= 63:
        return normalized
    suffix = canonical_sha256(value)[:8]
    return f'{normalized[:54].rstrip("-")}-{suffix}'


def allocation_resource_name(allocation_id: str) -> str:
    normalized = re.sub(r'[^a-z0-9-]+', '-', allocation_id.lower()).strip('-') or 'allocation'
    suffix = canonical_sha256(allocation_id)[:8]
    return dns_name(f'nexus-{normalized[:44]}-{suffix}', fallback='nexus-allocation')


def allocation_resource_names(allocation_id: str) -> tuple[str, str]:
    workload_name = allocation_resource_name(allocation_id)
    return workload_name, dns_name(f'{workload_name}-headless', fallback='nexus-allocation')


__all__ = ['allocation_resource_name', 'allocation_resource_names', 'dns_name']
