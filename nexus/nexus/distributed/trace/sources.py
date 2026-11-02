"""Resolve frozen execution metadata to explicitly configured read-only sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timezone
from typing import TYPE_CHECKING, Any

from nexus.core.registry import import_from_path
from nexus.distributed.errors import DomainConflictError
from nexus.distributed.trace.bindings import source_binding, validate_source_binding_update
from nexus.distributed.trace.models import (
    AssignmentTraceMetadata,
    AuthorizedTraceTarget,
    TraceReadError,
    TraceReaderConfig,
)
from nexus.runner.trace import principal_trace_scope_id


if TYPE_CHECKING:
    from nexus.distributed.config import DistributedConfig
    from nexus.runner.trace.tdsql.sink_source import TDSQLTraceSource


_TERMINAL = frozenset({'succeeded', 'failed', 'killed', 'lost', 'expired'})


@dataclass(slots=True)
class ResolvedTraceSource:
    reader: TraceReaderConfig
    principal_scope_id: str
    parent_created_at: str | None = None
    attempts: dict[str, tuple[str, int, bool] | None] = field(default_factory=dict)

    def scope(self, execution_id: str, run_span_id: str | None = None) -> dict[str, str]:
        value = {
            'storage_scope': 'principal_v1',
            'principal_scope_id': self.principal_scope_id,
            'run_id': execution_id,
        }
        if self.parent_created_at is not None:
            # Boundless prunes day partitions on this column; every read metadata carries it.
            value['parent_created_at'] = self.parent_created_at
        if run_span_id is not None:
            value['run_span_id'] = run_span_id
        return value


def _partition_key(
    reader: TraceReaderConfig,
    metadata: Mapping[str, Any],
    *,
    binding_key: str | None = None,
) -> str | None:
    """Resolve the Boundless partition key from the envelope, falling back to the binding.

    Classic TDSQL tables have no such column. On Boundless a missing value would silently
    turn every later query into a full-table scan, so an absent key is a resolution failure.

    ``_artifacts`` is only published when the run closes, so the envelope cannot serve a
    running Execution; the binding is reported with the running state and carries the same
    key. Both are accepted, and a disagreement is a conflict rather than a choice: one of
    the two would prune to a partition holding none of the rows.
    """
    from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT
    from nexus.runner.trace.tdsql.sink_source import parse_partition_key

    if reader.dialect != BOUNDLESS_DIALECT:
        return None
    try:
        envelope_key = parse_partition_key(metadata.get('parent_created_at'), required=binding_key is None)
    except ValueError:
        raise TraceReadError('trace_source_unresolved', 409) from None
    if envelope_key is None:
        return binding_key
    if binding_key is not None and binding_key != envelope_key:
        raise TraceReadError('trace_source_unresolved', 409)
    return envelope_key


@dataclass(frozen=True, slots=True)
class TraceSourceResolution:
    availability: str
    sources: tuple[ResolvedTraceSource, ...] = ()


class TraceSourceResolver:
    def __init__(self, config: DistributedConfig) -> None:
        self.config = config
        self.settings = config.trace_read
        self._raw_backends = {reader.backend for reader in self.settings.readers}
        for name, backend in config.run_trace_sinks.items():
            if name in {'artifact', 'local'}:
                continue
            if backend.source_identity is not None or backend.kind == 'tdsql':
                self._raw_backends.add(name)
            elif backend.source_class:
                try:
                    source_class = import_from_path(backend.source_class)
                    if callable(getattr(source_class, 'read_raw_event_page', None)) and callable(
                        getattr(source_class, 'list_raw_streams', None)
                    ):
                        self._raw_backends.add(name)
                except Exception:
                    # Bad reader registration is reported explicitly if selected.
                    self._raw_backends.add(name)

    def _reader(
        self,
        target: AuthorizedTraceTarget,
        backend: str,
        source_id: str,
        revision: str,
        *,
        cursor: bool = False,
    ) -> TraceReaderConfig:
        if backend not in self.settings.backend_allowlist:
            raise TraceReadError('trace_backend_read_disabled', 403)
        candidates = [
            reader
            for reader in self.settings.readers
            if reader.backend == backend
            and reader.source_id == source_id
            and reader.source_revision == revision
            and target.environment_id in reader.environment_ids
        ]
        if len(candidates) != 1:
            raise TraceReadError('trace_source_changed' if cursor else 'trace_source_unresolved', 409)
        reader = candidates[0]
        if reader.credential_ref not in self.config.secrets.trace_read_credentials:
            raise TraceReadError('trace_source_misconfigured', 503)
        return reader

    def from_cursor(self, target: AuthorizedTraceTarget, cursor: Mapping[str, Any]) -> ResolvedTraceSource:
        reader = self._reader(target, cursor['backend'], cursor['source_id'], cursor['source_revision'], cursor=True)
        # Paging skips resolution, so the signed cursor is the only carrier of the partition
        # key. A cursor minted against another dialect must be rejected, not silently widened.
        from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT

        parent_created_at = cursor.get('parent_created_at')
        if (reader.dialect == BOUNDLESS_DIALECT) != isinstance(parent_created_at, str):
            raise TraceReadError('invalid_trace_cursor', 400)
        return ResolvedTraceSource(
            reader, principal_trace_scope_id(target.owner_principal_id), parent_created_at=parent_created_at
        )

    def resolve(
        self,
        target: AuthorizedTraceTarget,
        assignments: Sequence[AssignmentTraceMetadata],
        *,
        backend: str | None = None,
    ) -> TraceSourceResolution:
        resolved: dict[str, ResolvedTraceSource] = {}
        waiting = False
        selected_any = False
        scope = principal_trace_scope_id(target.owner_principal_id)
        for assignment in assignments:
            try:
                binding = source_binding(assignment.runtime_info)
                if binding is not None:
                    validate_source_binding_update(
                        assignment.runtime_info,
                        None,
                        execution_id=target.execution_id,
                        assignment_id=assignment.assignment_id,
                        attempt_id=assignment.attempt_id,
                        owner_principal_id=target.owner_principal_id,
                    )
            except DomainConflictError:
                raise TraceReadError('trace_source_unresolved', 409) from None
            artifacts = (assignment.runtime_info or {}).get('_artifacts')
            envelope = artifacts.get('single') if isinstance(artifacts, Mapping) else None
            metadata_by_backend: dict[str, Mapping[str, Any]] = {}
            if isinstance(envelope, Mapping):
                for key, expected in (('run_id', target.execution_id), ('run_kind', 'single')):
                    if key in envelope and envelope[key] != expected:
                        raise TraceReadError('trace_source_unresolved', 409)
                for sink in envelope.get('sinks', []):
                    if not isinstance(sink, Mapping) or not isinstance(sink.get('name'), str):
                        raise TraceReadError('trace_source_unresolved', 409)
                    metadata = sink.get('metadata')
                    if not isinstance(metadata, Mapping) or sink['name'] in metadata_by_backend:
                        raise TraceReadError('trace_source_unresolved', 409)
                    metadata_by_backend[sink['name']] = metadata
            if binding is not None:
                selected = binding.selected_sinks
                identities = {source.backend: source for source in binding.sources}
                run_span = binding.run_span_id
            else:
                selected = tuple(metadata_by_backend)
                identities = {}
                run_span = envelope.get('run_span_id') if isinstance(envelope, Mapping) else None
                if not selected:
                    choice = target.driver_config.get('run_trace_backends')
                    names = choice.get('sinks') if isinstance(choice, Mapping) else None
                    if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
                        selected = tuple(names)
            if not selected:
                if target.execution_status not in _TERMINAL:
                    waiting = True
                    continue
                raise TraceReadError('trace_source_unresolved', 409)
            choice = target.driver_config.get('run_trace_backends')
            requested = choice.get('sinks') if isinstance(choice, Mapping) else None
            if requested is not None and binding is not None:
                assert isinstance(choice, Mapping)
                if not isinstance(requested, list) or any(not isinstance(name, str) for name in requested):
                    raise TraceReadError('trace_source_unresolved', 409)
                if not set(requested).issubset(selected):
                    raise TraceReadError('trace_source_unresolved', 409)
                extras = set(selected) - set(requested)
                if extras and not choice.get('append_artifact_fallback', False):
                    raise TraceReadError('trace_source_unresolved', 409)
                for name in extras:
                    configured = self.config.run_trace_sinks.get(name)
                    if name != 'artifact' and not (configured is not None and configured.required):
                        raise TraceReadError('trace_source_unresolved', 409)
            selected_any = True
            for selected_backend in selected:
                if backend is not None and selected_backend != backend:
                    continue
                name = selected_backend
                if name == 'artifact' or name not in self._raw_backends:
                    continue
                metadata = metadata_by_backend.get(name, {})
                storage_scope = metadata.get('storage_scope', 'principal_v1')
                if storage_scope not in {'principal_v1', 'batch_principal_v1'}:
                    raise TraceReadError('trace_backend_unsupported', 409)
                if metadata.get('principal_scope_id', scope) != scope:
                    raise TraceReadError('trace_source_unresolved', 409)
                for key, expected in (
                    ('run_id', target.execution_id),
                    ('assignment_id', assignment.assignment_id),
                    ('attempt_id', assignment.attempt_id),
                ):
                    if key in metadata and metadata[key] != expected:
                        raise TraceReadError('trace_source_unresolved', 409)
                if name in identities:
                    identity = identities[name]
                    reader = self._reader(target, name, identity.source_id, identity.source_revision)
                else:
                    # A new binding with an omitted identity means missing writer
                    # configuration, not permission to infer a default dataset.
                    if binding is not None:
                        raise TraceReadError('trace_source_unresolved', 409)
                    created_at = (
                        assignment.created_at.replace(tzinfo=timezone.utc)
                        if assignment.created_at.tzinfo is None
                        else assignment.created_at
                    )
                    candidates = [
                        reader
                        for reader in self.settings.readers
                        if reader.backend == name
                        and target.environment_id in reader.environment_ids
                        and reader.legacy_from is not None
                        and reader.legacy_until is not None
                        and reader.legacy_from <= created_at < reader.legacy_until
                    ]
                    if len(candidates) != 1:
                        raise TraceReadError('trace_source_unresolved', 409)
                    reader = self._reader(target, name, candidates[0].source_id, candidates[0].source_revision)
                partition_key = _partition_key(
                    reader, metadata, binding_key=binding.parent_created_at if binding is not None else None
                )
                previous = resolved.get(name)
                if previous is not None and (previous.reader != reader or previous.parent_created_at != partition_key):
                    # Retries of one execution share a parent, hence one partition key.
                    raise TraceReadError('trace_source_unresolved', 409)
                source = resolved.setdefault(name, ResolvedTraceSource(reader, scope, partition_key))
                if isinstance(run_span, str) and run_span:
                    attempt = (
                        assignment.attempt_id,
                        assignment.attempt_no,
                        assignment.assignment_id == target.current_assignment_id,
                    )
                    if run_span in source.attempts and source.attempts[run_span] != attempt:
                        source.attempts[run_span] = None
                    else:
                        source.attempts[run_span] = attempt
        if waiting or (not assignments and target.execution_status not in _TERMINAL):
            return TraceSourceResolution('waiting_source')
        if not resolved:
            raise TraceReadError('trace_backend_unsupported' if selected_any else 'trace_source_unresolved', 409)
        return TraceSourceResolution('readable', tuple(resolved.values()))

    def create_source(self, resolved: ResolvedTraceSource) -> TDSQLTraceSource:
        from nexus.runner.trace.tdsql.sink_source import TDSQLTraceSource

        reader = resolved.reader
        credential = self.config.secrets.trace_read_credentials.get(reader.credential_ref)
        if credential is None:
            raise TraceReadError('trace_source_misconfigured', 503)
        user, password = credential.read()
        return TDSQLTraceSource(
            host=reader.host,
            port=reader.port,
            db=reader.database,
            user=user,
            password=password,
            dialect=reader.dialect,
        )
