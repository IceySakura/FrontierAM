"""Built-in request_rate policy selection and snapshot consumption.

This module never imports or calls a concrete policy plugin or signal source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from nexus.distributed.errors import DomainValidationError
from nexus.distributed.persistence.orm import PolicyScopeRow
from nexus.distributed.policies.models import (
    REQUEST_RATE_KIND,
    PolicyBinding,
    PolicyExecutorConfig,
    RequestPolicySelection,
    RequestRateDecision,
)
from nexus.distributed.repositories.policy import policy_scope_key


RequestPolicyMode = Literal['enforce', 'shadow', 'disabled']


def request_budget_key(selection: RequestPolicySelection) -> str:
    return policy_scope_key(REQUEST_RATE_KIND, selection.binding, selection.scope)


def resolve_request_selection(
    config: PolicyExecutorConfig,
    requested: Mapping[str, Any] | RequestPolicySelection | None,
    *,
    environment_id: str,
    owner_principal_id: str | None,
    spec_type: str,
) -> RequestPolicySelection | None:
    if spec_type == 'nexus_batch':
        return None
    parsed = (
        requested
        if isinstance(requested, RequestPolicySelection) or requested is None
        else RequestPolicySelection.model_validate(requested)
    )
    required = config.request_defaults.get(environment_id)
    if required is not None and parsed is not None and parsed != required:
        raise DomainValidationError('request_policy_required')
    selection = parsed or required
    if selection is None:
        return None
    if not _binding_permits(
        config.bindings.get(selection.binding),
        selection=selection,
        environment_id=environment_id,
        owner_principal_id=owner_principal_id,
    ):
        raise DomainValidationError('request_policy_not_permitted')
    return selection


def validate_manager_request_policy(
    config: PolicyExecutorConfig,
    selection: Mapping[str, Any] | RequestPolicySelection | None,
    *,
    environment_id: str | None = None,
    owner_principal_id: str | None = None,
) -> RequestPolicySelection | None:
    if selection is None:
        return None
    parsed = (
        selection if isinstance(selection, RequestPolicySelection) else RequestPolicySelection.model_validate(selection)
    )
    binding = config.bindings.get(parsed.binding)
    if binding is None or binding.kind != REQUEST_RATE_KIND or parsed.scope not in binding.scopes:
        raise ValueError('invalid manager request_policy binding')
    if environment_id is not None and environment_id not in binding.environments:
        raise ValueError('manager request_policy binding does not permit environment')
    if owner_principal_id is not None and binding.principals and owner_principal_id not in binding.principals:
        raise ValueError('manager request_policy binding does not permit principal')
    return parsed


def _binding_permits(
    binding: PolicyBinding | None,
    *,
    selection: RequestPolicySelection,
    environment_id: str,
    owner_principal_id: str | None,
) -> bool:
    return not (
        binding is None
        or binding.kind != REQUEST_RATE_KIND
        or environment_id not in binding.environments
        or (binding.principals and owner_principal_id not in binding.principals)
        or selection.scope not in binding.scopes
    )


@dataclass(frozen=True, slots=True)
class RequestRateSnapshotReading:
    mode: RequestPolicyMode
    config_revision: int
    config_hash: str
    ready: bool
    allow: bool
    rate_per_minute: float
    snapshot_version: int | None
    valid_until: datetime | None
    reason_code: str


def consume_request_rate_snapshot(
    selection: RequestPolicySelection,
    config: PolicyExecutorConfig,
    row: PolicyScopeRow | None,
    *,
    now: datetime,
    fingerprint: str | None = None,
) -> RequestRateSnapshotReading:
    binding = config.bindings.get(selection.binding)
    if binding is None or binding.kind != REQUEST_RATE_KIND or selection.scope not in binding.scopes:
        raise ValueError('request_policy_binding_missing')
    digest = fingerprint or binding.fingerprint(config.imports)
    decision, snapshot_version, valid_until, reason = _parse_fresh_snapshot(row, digest=digest, now=now)
    ready = decision is not None and snapshot_version is not None and valid_until is not None
    if binding.mode in {'shadow', 'disabled'}:
        return RequestRateSnapshotReading(
            mode=binding.mode,
            config_revision=binding.revision,
            config_hash=digest,
            ready=ready,
            allow=True if decision is None else decision.allow,
            rate_per_minute=0.0 if decision is None else decision.rate_per_minute,
            snapshot_version=snapshot_version,
            valid_until=valid_until,
            reason_code=reason if ready else f'policy_{binding.mode}',
        )
    return RequestRateSnapshotReading(
        mode='enforce',
        config_revision=binding.revision,
        config_hash=digest,
        ready=ready,
        allow=bool(ready and decision is not None and decision.allow),
        rate_per_minute=decision.rate_per_minute if (ready and decision is not None and decision.allow) else 0.0,
        snapshot_version=snapshot_version if ready else None,
        valid_until=valid_until if ready else None,
        reason_code=reason if ready else 'policy_unavailable',
    )


def _parse_fresh_snapshot(
    row: PolicyScopeRow | None,
    *,
    digest: str,
    now: datetime,
) -> tuple[RequestRateDecision | None, int | None, datetime | None, str]:
    if (
        row is None
        or row.config_hash != digest
        or row.valid_until is None
        or row.valid_until <= now
        or not isinstance(row.snapshot_json, Mapping)
    ):
        return None, None, None, 'policy_unavailable'
    snapshot = row.snapshot_json
    try:
        decision = RequestRateDecision.model_validate(snapshot['payload'])
        version = int(snapshot['snapshot_version'])
        reason = str(snapshot.get('reason_code') or 'policy_available')
    except (KeyError, TypeError, ValueError):
        return None, None, None, 'policy_unavailable'
    return decision, version, row.valid_until, reason


def configured_rate_per_minute(binding: PolicyBinding | None, scope: str) -> float | None:
    if binding is None:
        return None
    source_cfg = binding.scopes[scope].source_config if scope in binding.scopes else {}
    for candidate in (
        source_cfg.get('quota_qpm'),
        source_cfg.get('rate_per_minute'),
        binding.params.get('quota_qpm'),
        binding.params.get('rate_per_minute'),
    ):
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None
