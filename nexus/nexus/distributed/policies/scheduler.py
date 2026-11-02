"""Built-in snapshot consumption; this module never imports or calls a policy plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nexus.distributed.errors import DomainValidationError
from nexus.distributed.persistence.orm import PolicyScopeRow
from nexus.distributed.policies.models import PolicyExecutorConfig, PolicySelection, SchedulerDecision
from nexus.distributed.repositories.policy import policy_scope_key


def resolve_selection(
    config: PolicyExecutorConfig,
    requested: dict | None,
    *,
    environment_id: str,
    owner_principal_id: str | None,
    spec_type: str,
) -> PolicySelection | None:
    if spec_type == 'nexus_batch':
        return None
    selection = PolicySelection.model_validate(requested) if requested else config.defaults.get(environment_id)
    required = config.defaults.get(environment_id)
    if required is not None and selection is not None:
        if selection.binding != required.binding:
            raise DomainValidationError('scheduler_policy_required')
        selection = selection.model_copy(update={'scopes': sorted(set(selection.scopes) | set(required.scopes))})
    if selection is None:
        return None
    binding = config.bindings.get(selection.binding)
    if (
        binding is None
        or binding.kind != 'scheduler'
        or environment_id not in binding.environments
        or (binding.principals and owner_principal_id not in binding.principals)
        or not set(selection.scopes) <= binding.scopes.keys()
    ):
        raise DomainValidationError('scheduler_policy_not_permitted')
    return selection


def selection_keys(selection: PolicySelection, config: PolicyExecutorConfig) -> list[str]:
    binding = config.bindings[selection.binding]
    return [policy_scope_key(binding.kind, selection.binding, scope) for scope in selection.scopes]


@dataclass
class SchedulerPolicyEffect:
    allow: bool = True
    reason: str = 'policy_available'
    delay_seconds: float = 30
    preferred_managers: list[str] = field(default_factory=list)
    valid_until: datetime | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def consume_snapshot(
    selection: PolicySelection,
    config: PolicyExecutorConfig,
    rows: dict[str, PolicyScopeRow],
    *,
    now: datetime,
    fingerprint: str | None = None,
) -> SchedulerPolicyEffect:
    binding = config.bindings[selection.binding]
    effect = SchedulerPolicyEffect(delay_seconds=binding.max_defer_seconds)
    if binding.mode == 'disabled':
        return effect
    digest = fingerprint or binding.fingerprint(config.imports)
    versions = {}
    preferences = []
    for scope, key in zip(selection.scopes, selection_keys(selection, config), strict=True):
        row = rows.get(key)
        snapshot = row.snapshot_json if row is not None else None
        decision = None
        ready = (
            row is not None
            and row.config_hash == digest
            and row.valid_until is not None
            and row.valid_until > now
            and snapshot is not None
        )
        if ready and snapshot is not None:
            try:
                decision = SchedulerDecision.model_validate(snapshot['payload'])
            except (ValueError, KeyError, TypeError):
                ready = False
        if not ready:
            previously_denied = snapshot is not None and snapshot.get('payload', {}).get('allow') is False
            if binding.on_unavailable == 'defer' or previously_denied:
                effect.allow = False
                effect.reason = 'policy_unavailable'
            continue
        assert row is not None
        assert row.valid_until is not None
        assert snapshot is not None
        assert decision is not None
        versions[scope] = snapshot['snapshot_version']
        effect.valid_until = min(effect.valid_until or row.valid_until, row.valid_until)
        if not decision.allow:
            effect.allow = False
            effect.reason = snapshot['reason_code']
        if decision.preferred_managers:
            preferences.append(decision.preferred_managers)
    # A binding has one ranking source. Conflicting scope preferences fall back to existing order.
    if preferences and all(preference == preferences[0] for preference in preferences):
        effect.preferred_managers = preferences[0]
    effect.evidence = {
        'binding': selection.binding,
        'scopes': selection.scopes,
        'mode': binding.mode,
        'config_revision': binding.revision,
        'snapshot_versions': versions,
        'decision': 'allow' if effect.allow else 'defer',
        'reason_code': effect.reason,
    }
    if binding.mode == 'shadow':
        effect.allow = True
        effect.preferred_managers = []
        effect.valid_until = None
    return effect
