"""Nonsecret, immutable source bindings reported with Single running state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nexus.distributed.errors import DomainConflictError
from nexus.distributed.trace.models import BackendName, Identity, TraceSourceIdentity
from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql.sink_source import normalize_parent_created_at


class BoundTraceSource(TraceSourceIdentity):
    backend: BackendName
    principal_scope_id: str = Field(pattern=r'^[0-9a-f]{32}$')


class TraceSourceBinding(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    version: Literal[1] = 1
    run_id: Identity
    run_span_id: Identity
    assignment_id: Identity
    attempt_id: Identity
    selected_sinks: tuple[BackendName, ...] = Field(min_length=1, max_length=32)
    sources: tuple[BoundTraceSource, ...] = Field(default=(), max_length=32)
    # Boundless prunes day partitions on this key and readers cannot derive it. The artifact
    # envelope that used to be its only carrier is published at close, which left a running
    # Execution unreadable; the binding is reported with the running state instead. One
    # Execution has exactly one key -- retries share the parent -- hence a single field
    # rather than one per source. Absent for classic TDSQL, whose tables have no such column.
    parent_created_at: str | None = Field(default=None, min_length=1, max_length=32)

    @model_validator(mode='after')
    def validate_sources(self) -> TraceSourceBinding:
        names = [source.backend for source in self.sources]
        if len(set(self.selected_sinks)) != len(self.selected_sinks) or len(set(names)) != len(names):
            raise ValueError('duplicate trace backend')
        if not set(names).issubset(self.selected_sinks):
            raise ValueError('unselected trace source')
        return self


def make_source_binding(
    *,
    run_id: str,
    run_span_id: str,
    assignment_id: str,
    attempt_id: str,
    owner_principal_id: str | None,
    selected_sinks: Sequence[str],
    source_identities: Mapping[str, TraceSourceIdentity],
    parent_created_at: Any | None = None,
) -> dict[str, Any]:
    sources = (
        ()
        if owner_principal_id is None
        else tuple(
            BoundTraceSource(
                **identity.model_dump(), backend=name, principal_scope_id=principal_trace_scope_id(owner_principal_id)
            )
            for name, identity in source_identities.items()
            if name in selected_sinks
        )
    )
    payload = TraceSourceBinding(
        run_id=run_id,
        run_span_id=run_span_id,
        assignment_id=assignment_id,
        attempt_id=attempt_id,
        selected_sinks=tuple(selected_sinks),
        sources=sources,
        # Normalized here rather than at the reader: the Driver holds an ISO string while the
        # sink envelope holds a MySQL DATETIME literal, and the two are cross-checked for
        # equality below. Storing the raw ISO form would report every agreement as a conflict.
        parent_created_at=None if parent_created_at is None else normalize_parent_created_at(parent_created_at),
    ).model_dump(mode='json')
    if payload.get('parent_created_at') is None:
        # Builds that predate the field forbid unknown keys, so a run without a partition key
        # must keep publishing the exact shape they already accept.
        payload.pop('parent_created_at', None)
    return payload


def source_binding(runtime_info: Mapping[str, Any] | None) -> TraceSourceBinding | None:
    trace = (runtime_info or {}).get('run_trace')
    if not isinstance(trace, Mapping) or 'source_binding' not in trace:
        return None
    try:
        return TraceSourceBinding.model_validate(trace['source_binding'])
    except ValidationError:
        raise DomainConflictError('trace_source_binding_invalid') from None


def validate_source_binding_update(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
    *,
    execution_id: str,
    assignment_id: str,
    attempt_id: str,
    owner_principal_id: str | None,
) -> None:
    current = source_binding(existing)
    update = source_binding(incoming)
    if current is not None and update is not None and current != update:
        raise DomainConflictError('trace_source_binding_immutable')
    binding = update or current
    if binding is None:
        return
    if (binding.run_id, binding.assignment_id, binding.attempt_id) != (execution_id, assignment_id, attempt_id):
        raise DomainConflictError('trace_source_binding_identity_mismatch')
    for source in binding.sources:
        if owner_principal_id is None or source.principal_scope_id != principal_trace_scope_id(owner_principal_id):
            raise DomainConflictError('trace_source_binding_scope_mismatch')
    for runtime in (existing, incoming):
        artifacts = (runtime or {}).get('_artifacts')
        envelope = artifacts.get('single') if isinstance(artifacts, Mapping) else None
        if isinstance(envelope, Mapping):
            validate_binding_envelope(binding, envelope)


def validate_binding_envelope(binding: TraceSourceBinding, envelope: Mapping[str, Any]) -> None:
    """Missing legacy fields are tolerated; explicit identity conflicts are not."""
    for key, expected in (('run_id', binding.run_id), ('run_span_id', binding.run_span_id), ('run_kind', 'single')):
        if key in envelope and envelope[key] != expected:
            raise DomainConflictError('trace_source_binding_artifact_mismatch')
    sinks = envelope.get('sinks')
    if sinks is None:
        return
    if not isinstance(sinks, list) or any(not isinstance(sink, Mapping) for sink in sinks):
        raise DomainConflictError('trace_source_binding_artifact_mismatch')
    names = [sink.get('name') for sink in sinks]
    if (
        any(not isinstance(name, str) for name in names)
        or len(names) != len(binding.selected_sinks)
        or set(names) != set(binding.selected_sinks)
    ):
        raise DomainConflictError('trace_source_binding_artifact_mismatch')
    by_name = {source.backend: source for source in binding.sources}
    for sink in sinks:
        metadata = sink.get('metadata')
        if not isinstance(metadata, Mapping):
            continue
        expected_values: dict[str, Any] = {
            'run_id': binding.run_id,
            'run_span_id': binding.run_span_id,
            'assignment_id': binding.assignment_id,
            'attempt_id': binding.attempt_id,
        }
        if binding.parent_created_at is not None:
            # Both carriers must agree: the reader trusts whichever it finds, and two
            # different keys mean one of them prunes to a partition holding none of the rows.
            # A binding without the key predates the field and is not a conflict.
            expected_values['parent_created_at'] = binding.parent_created_at
        if source := by_name.get(sink['name']):
            expected_values.update(source.model_dump(exclude={'backend'}))
        for key, expected in expected_values.items():
            value = metadata.get(key)
            if key == 'storage_scope' and value == 'batch_principal_v1':
                value = 'principal_v1'
            if value is not None and value != expected:
                raise DomainConflictError('trace_source_binding_artifact_mismatch')
