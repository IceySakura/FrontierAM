"""Distributed run-trace backend registration, selection, and source routing."""

from __future__ import annotations

import copy
import inspect
import logging

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

from nexus.core.registry import import_from_path
from nexus.distributed.artifacts.gateway import ApplicationGatewayArtifactTransport
from nexus.distributed.config import RunTraceBackendMetadata, RunTraceSinkConfig, run_trace_backend_catalog
from nexus.distributed.execution.runtime_info import normalize_runtime_info
from nexus.distributed.ports import AsyncCloseable
from nexus.distributed.trace.bindings import TraceSourceIdentity
from nexus.distributed.utils.identity import validate_run_trace_backend_name
from nexus.runner.live.trace import project_live_trace_records
from nexus.runner.trace import (
    RUN_KINDS,
    RUN_TRACE_METADATA_VERSION,
    RunTraceContext,
    RunTraceScope,
    RunTraceScopePreparer,
    RunTraceSink,
    RunTraceSource,
    RunTraceSourceUnavailable,
    RunTraceStreamEvent,
)


logger = logging.getLogger(__name__)

RUN_TRACE_RUNTIME_INFO_KEY = '_artifacts'
RUN_TRACE_LIVE_KEY = 'live'
RUN_TRACE_SINGLE_KEY = 'single'
LIVE_RESULT_POINTER_KIND = 'nexus_live_result'
RUN_RESULT_POINTER_KIND = 'nexus_run_trace_result'
RUN_TRACE_ENVELOPE_VERSION = RUN_TRACE_METADATA_VERSION
LOCAL_RUN_TRACE_SINK_CLASS = 'nexus.runner.trace.local::RunTraceLocalSink'
LOCAL_RUN_TRACE_SOURCE_CLASS = 'nexus.runner.trace.local::RunTraceLocalSource'
_RUN_TRACE_SINK_REQUEST_KEYS = frozenset({'execution_id', 'assignment_id', 'attempt_id'})


@dataclass(frozen=True, slots=True)
class RunTraceSinkRequest:
    execution_id: str
    assignment_id: str
    attempt_id: str
    artifact_scope_available: bool = True
    result_publish_deadline_at: datetime | None = None


RunTraceSinkFactory = Callable[[RunTraceSinkRequest], RunTraceSink]
RunTraceBackendAvailability = Callable[[RunTraceSinkRequest], bool]


def _always_available(request: RunTraceSinkRequest) -> bool:
    del request
    return True


def _never_available(request: RunTraceSinkRequest) -> bool:
    del request
    return False


class _UnavailableRunTraceSource(RunTraceSource):
    """Stands in for a backend this build could not construct.

    Reads raise ``RunTraceSourceUnavailable`` so multi-source readers fall through to a
    working backend instead of failing outright.
    """

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def _unavailable(self) -> RunTraceSourceUnavailable:
        return RunTraceSourceUnavailable(f'run trace backend {self.name!r} is unavailable: {self.reason}')

    async def read_events_after(
        self,
        metadata: Mapping[str, Any],
        cursor: int,
        *,
        limit: int | None = None,
    ) -> list[Any]:
        del metadata, cursor, limit
        raise self._unavailable()

    async def read_result(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        del metadata
        raise self._unavailable()


@dataclass(frozen=True, slots=True)
class RunTraceBackendRegistration:
    name: str
    sink_factory: RunTraceSinkFactory
    source: RunTraceSource
    system_managed: bool = False
    required: bool = False
    available: RunTraceBackendAvailability = _always_available
    unavailable_detail: str | None = None
    source_identity: TraceSourceIdentity | None = None


@dataclass(slots=True)
class ResolvedRunTraceBackends:
    """Named backend sinks selected for one run."""

    run_kind: str
    backend_sinks: dict[str, RunTraceSink]
    source_identities: dict[str, TraceSourceIdentity] = field(default_factory=dict)


class RunTraceBackendRegistry:
    """Trusted Instance Manager registry for dynamically configured backends."""

    def __init__(self, *, default_sink_names: Sequence[str] = ('local',)) -> None:
        self._registrations: dict[str, RunTraceBackendRegistration] = {}
        self.default_sink_names = tuple(default_sink_names)
        self._closeables: list[AsyncCloseable] = []
        self._closed = False
        self.artifact_transport: ApplicationGatewayArtifactTransport | None = None

    def register(self, registration: RunTraceBackendRegistration) -> None:
        name = validate_run_trace_backend_name(registration.name)
        if name in self._registrations:
            raise ValueError(f'run trace backend is already registered: {name}')
        if registration.required and not registration.system_managed:
            raise ValueError(f'required run trace backend must be system managed: {name}')
        self._registrations[name] = registration

    def validate(self) -> None:
        self.default_sink_names = tuple(
            dict.fromkeys((
                *self.default_sink_names,
                *(name for name, registration in self._registrations.items() if registration.required),
            ))
        )
        self._validate_sink_names(self.default_sink_names)

    def add_closeable(self, resource: AsyncCloseable) -> None:
        if self._closed:
            raise RuntimeError('run trace backend registry is closed')
        if all(existing is not resource for existing in self._closeables):
            self._closeables.append(resource)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for resource in reversed(self._closeables):
            try:
                await resource.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._closeables.clear()
        if first_error is not None:
            raise first_error

    def resolve(
        self,
        *,
        run_kind: str,
        driver_config: Mapping[str, Any],
        request: RunTraceSinkRequest,
    ) -> ResolvedRunTraceBackends:
        if run_kind not in RUN_KINDS:
            raise ValueError(f'unsupported run trace kind: {run_kind}')
        names = self._selected_sink_names(driver_config, request=request)
        return ResolvedRunTraceBackends(
            run_kind=run_kind,
            backend_sinks={name: self._registrations[name].sink_factory(request) for name in names},
            source_identities={
                name: identity for name in names if (identity := self._registrations[name].source_identity) is not None
            },
        )

    async def read_result(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        errors: list[Exception] = []
        for registration, metadata in self._sources(envelope):
            try:
                return await registration.source.read_result(metadata)
            except (OSError, RunTraceSourceUnavailable) as exc:
                errors.append(exc)
        if errors:
            raise FileNotFoundError('run result is unavailable from all trace sources') from errors[-1]
        raise FileNotFoundError('run trace artifact has no registered source')

    async def read_live_events_after(
        self,
        envelope: Mapping[str, Any],
        cursor: int,
        *,
        limit: int | None = None,
    ) -> list[RunTraceStreamEvent]:
        context = _context_from_envelope(envelope)
        errors: list[Exception] = []
        for registration, metadata in self._sources(envelope):
            try:
                records = await registration.source.read_events_after(metadata, 0)
                return project_live_trace_records(context, records, cursor=cursor, limit=limit)
            except (OSError, RunTraceSourceUnavailable) as exc:
                errors.append(exc)
        if errors:
            raise FileNotFoundError('live events are unavailable from all trace sources') from errors[-1]
        raise FileNotFoundError('run trace artifact has no registered source')

    def _selected_sink_names(
        self,
        driver_config: Mapping[str, Any],
        *,
        request: RunTraceSinkRequest,
    ) -> tuple[str, ...]:
        backend_config = driver_config.get('run_trace_backends')
        explicit_selection = False
        append_platform_required = False
        requested_names: tuple[str, ...] = ()
        if backend_config is None:
            names = self.default_sink_names
        elif not isinstance(backend_config, Mapping):
            raise ValueError('driver_config.run_trace_backends must be an object')
        else:
            append_platform_required = backend_config.get('append_artifact_fallback', False)
            if not isinstance(append_platform_required, bool):
                raise ValueError('driver_config.run_trace_backends.append_artifact_fallback must be a boolean')
            raw_names = backend_config.get('sinks')
            if raw_names is None:
                names = self.default_sink_names
            elif not isinstance(raw_names, list) or not all(isinstance(name, str) for name in raw_names):
                raise ValueError('driver_config.run_trace_backends.sinks must be a list of names')
            else:
                names = tuple(raw_names)
                requested_names = names
                explicit_selection = True
        if explicit_selection and append_platform_required:
            names = (
                *names,
                *(
                    name
                    for name, registration in self._registrations.items()
                    if registration.required and registration.available(request) and name not in names
                ),
            )
        self._validate_sink_names(names)
        unavailable_requested = [name for name in requested_names if not self._registrations[name].available(request)]
        if unavailable_requested:
            registration = self._registrations[unavailable_requested[0]]
            raise ValueError(
                registration.unavailable_detail
                or f'run trace backend is unavailable for this execution: {registration.name}'
            )
        names = tuple(name for name in names if self._registrations[name].available(request))
        self._validate_sink_names(names)
        return tuple(names)

    def _validate_sink_names(self, names: Sequence[str]) -> None:
        if not names:
            raise ValueError('driver_config.run_trace_backends.sinks requires at least one sink')
        if len(set(names)) != len(names):
            raise ValueError('driver_config.run_trace_backends.sinks contains duplicate names')
        unknown = [name for name in names if name not in self._registrations]
        if unknown:
            raise ValueError(f'unknown run trace sink: {unknown[0]}')

    def _sources(
        self,
        envelope: Mapping[str, Any],
    ) -> list[tuple[RunTraceBackendRegistration, Mapping[str, Any]]]:
        _context_from_envelope(envelope)
        raw_sinks = envelope.get('sinks')
        if not isinstance(raw_sinks, list) or not raw_sinks:
            raise ValueError('run trace artifact envelope has no sinks')
        result: list[tuple[RunTraceBackendRegistration, Mapping[str, Any]]] = []
        for item in raw_sinks:
            if not isinstance(item, Mapping):
                continue
            name = item.get('name')
            metadata = item.get('metadata')
            if not isinstance(name, str) or not isinstance(metadata, Mapping):
                continue
            registration = self._registrations.get(name)
            if registration is not None:
                result.append((registration, metadata))
        return result


def create_run_trace_backend_registry(
    config: Any,
    *,
    artifact_transport: ApplicationGatewayArtifactTransport | None = None,
) -> RunTraceBackendRegistry:
    from nexus.distributed.artifacts.gateway import (
        RunResultArtifactGateway,
        create_application_gateway_artifact_transport,
    )
    from nexus.distributed.artifacts.run_result import (
        ArtifactTraceSink,
        ArtifactTraceSource,
    )

    owns_artifact_transport = artifact_transport is None
    artifact_transport = artifact_transport or create_application_gateway_artifact_transport(config)
    artifact_enabled = artifact_transport is not None
    if artifact_enabled and 'artifact' in config.run_trace_sinks:
        raise ValueError("run trace backend name 'artifact' is reserved")
    metadata_by_name = {backend.name: backend for backend in run_trace_backend_catalog(config)}
    registry = RunTraceBackendRegistry(default_sink_names=config.default_run_trace_sinks)
    registry.artifact_transport = artifact_transport
    if owns_artifact_transport and artifact_transport is not None:
        registry.add_closeable(artifact_transport)
    configured: dict[str, dict[str, Any]] = {
        'local': {
            'sink_class': LOCAL_RUN_TRACE_SINK_CLASS,
            'source_class': LOCAL_RUN_TRACE_SOURCE_CLASS,
            'config': {'root': config.run_trace_artifact_root},
        }
    }
    for name, raw_config in config.run_trace_sinks.items():
        configured[name] = _merge_config(configured.get(name, {}), _configured_sink_values(name, raw_config))

    for name, values in configured.items():
        metadata = metadata_by_name[name]
        try:
            registration = _configured_registration(name, values, config=config, metadata=metadata)
        except Exception as exc:
            # One unusable backend must not stop the instance manager from starting. This
            # process runs submitted user code, so a backend configured after that code was
            # packaged would otherwise crash-loop every unrelated execution on the pool.
            #
            # Warning rather than error: across a deliberately mixed-version fleet, a build
            # that predates a backend is the expected case, not a fault of this process, and
            # logging it at error level would page someone on every such start. The loud
            # signal belongs at selection, where _unavailable_registration raises for the
            # runs that actually asked for this backend -- a genuine misconfiguration such
            # as a mistyped class path surfaces there, attributed to the run that hit it.
            logger.warning('run trace backend %r is unusable by this build: %s', name, exc, exc_info=True)
            registration = _unavailable_registration(name, metadata, exc)
        registry.register(registration)
    if artifact_transport is not None:
        client = RunResultArtifactGateway(artifact_transport)
        metadata = metadata_by_name['artifact']
        registry.register(
            RunTraceBackendRegistration(
                name='artifact',
                sink_factory=lambda request: ArtifactTraceSink(
                    client=client,
                    execution_id=request.execution_id,
                    assignment_id=request.assignment_id,
                    attempt_id=request.attempt_id,
                    max_result_size_bytes=config.artifact_run_result_max_size_bytes,
                    compression=config.artifact_run_result_compression,
                    max_decoded_size_bytes=config.artifact_run_result_max_decoded_size_bytes,
                    publish_timeout_seconds=config.artifact_run_result_publish_deadline.total_seconds(),
                    publish_deadline_at=request.result_publish_deadline_at,
                ),
                source=ArtifactTraceSource(
                    client=client, max_decoded_size_bytes=config.artifact_run_result_max_decoded_size_bytes
                ),
                system_managed=metadata.system_managed,
                required=metadata.required,
                available=lambda request: request.artifact_scope_available,
                unavailable_detail='artifact run trace backend requires execution allocation scope',
            )
        )
    registry.validate()
    return registry


async def prepare_configured_run_trace_backend_scope(
    config: Any,
    *,
    backend_name: str,
    scope_id: str,
) -> None:
    raw_config = config.run_trace_sinks.get(backend_name)
    if raw_config is None:
        raise ValueError(f'unknown run trace sink: {backend_name}')
    sink_config = RunTraceSinkConfig.model_validate(_configured_sink_values(backend_name, raw_config))
    sink_class_path = _required_config_value(backend_name, 'sinkClass', sink_config.sink_class)
    sink_class = _import_backend_class(backend_name, 'sink', sink_class_path, RunTraceSink)
    if not issubclass(sink_class, RunTraceScopePreparer):
        return

    common_config = dict(sink_config.config)
    _inject_secret_config(backend_name, config, sink_config, common_config)
    sink_kwargs = _merge_config(common_config, sink_config.sink_config)
    reserved_keys = _RUN_TRACE_SINK_REQUEST_KEYS.intersection(sink_kwargs)
    if reserved_keys:
        key = sorted(reserved_keys)[0]
        raise ValueError(f'run trace sink config for {backend_name!r} cannot set reserved key {key!r}')
    sink_kwargs.update({
        'execution_id': scope_id,
        'assignment_id': scope_id,
        'attempt_id': scope_id,
    })
    sink = _create_sink(backend_name, sink_class, sink_kwargs)
    await cast(RunTraceScopePreparer, sink).prepare_scope(RunTraceScope(run_kind='single', scope_id=scope_id))


def _context_from_envelope(envelope: Mapping[str, Any]) -> RunTraceContext:
    if envelope.get('version') != RUN_TRACE_ENVELOPE_VERSION:
        raise ValueError('unsupported run trace artifact envelope')
    run_id = envelope.get('run_id')
    run_span_id = envelope.get('run_span_id')
    run_kind = envelope.get('run_kind')
    if not isinstance(run_id, str) or not isinstance(run_span_id, str) or run_kind not in RUN_KINDS:
        raise ValueError('run trace artifact envelope has invalid run identity')
    return RunTraceContext(run_id=run_id, run_span_id=run_span_id, run_kind=cast(str, run_kind))


def _configured_sink_values(name: str, raw_config: Any) -> dict[str, Any]:
    if isinstance(raw_config, Mapping):
        model = RunTraceSinkConfig.model_validate(raw_config)
    elif isinstance(raw_config, RunTraceSinkConfig):
        model = raw_config
    else:
        raise TypeError(f'invalid run trace backend config for {name!r}')
    return model.model_dump(mode='python', exclude_unset=True)


def _inject_secret_config(
    name: str,
    config: Any,
    sink_config: RunTraceSinkConfig,
    common_config: dict[str, Any],
) -> None:
    for field_name, credential_ref in sink_config.secret_config.items():
        if any(
            field_name in configured
            for configured in (sink_config.config, sink_config.sink_config, sink_config.source_config)
        ):
            raise ValueError(f'run trace secret field {field_name!r} is also configured in cleartext for {name!r}')
        common_config[field_name] = config.read_run_trace_credential(credential_ref)


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _merge_config(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _required_config_value(name: str, field: str, value: str | None) -> str:
    if not value:
        raise ValueError(f'run trace sink config for {name!r} requires {field}')
    return value


def _import_backend_class(name: str, role: str, import_path: str, base: type[Any]) -> type[Any]:
    try:
        value = import_from_path(import_path)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f'failed to import run trace {role} class for {name!r}: {import_path}') from exc
    if not isinstance(value, type):
        raise TypeError(f'configured {role}Class for {name!r} is not a class: {import_path}')
    if not issubclass(value, base):
        raise TypeError(f'configured {role}Class for {name!r} must inherit {base.__name__}')
    return value


def _create_sink(name: str, sink_class: type[RunTraceSink], kwargs: Mapping[str, Any]) -> RunTraceSink:
    try:
        arguments = dict(kwargs)
        parameters = inspect.signature(sink_class).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            accepted = {
                key
                for key, parameter in parameters.items()
                if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
            }
            ignored = arguments.keys() - accepted
            if ignored:
                logger.warning(
                    'Run trace sink %r (%s) does not support config keys %s; '
                    "ignoring them and retaining this build's behavior",
                    name,
                    sink_class.__name__,
                    ', '.join(sorted(ignored)),
                )
                for key in ignored:
                    arguments.pop(key)
        return sink_class(**arguments)
    except Exception as exc:
        raise RuntimeError(f'failed to create run trace sink {name!r}') from exc


def _create_source(name: str, source_class: type[Any], kwargs: Mapping[str, Any]) -> RunTraceSource:
    try:
        return cast(RunTraceSource, source_class(**copy.deepcopy(dict(kwargs))))
    except Exception as exc:
        raise RuntimeError(f'failed to create run trace source {name!r}') from exc


def _configured_registration(
    name: str,
    values: Mapping[str, Any],
    *,
    config: Any,
    metadata: RunTraceBackendMetadata,
) -> RunTraceBackendRegistration:
    sink_config = RunTraceSinkConfig.model_validate(values)
    sink_class_path = _required_config_value(name, 'sinkClass', sink_config.sink_class)
    source_class_path = _required_config_value(name, 'sourceClass', sink_config.source_class)
    sink_class = _import_backend_class(name, 'sink', sink_class_path, RunTraceSink)
    source_class = _import_backend_class(name, 'source', source_class_path, RunTraceSource)
    common_config = dict(sink_config.config)
    _inject_secret_config(name, config, sink_config, common_config)
    sink_kwargs = _merge_config(common_config, sink_config.sink_config)
    source_kwargs = _merge_config(common_config, sink_config.source_config)
    reserved_keys = _RUN_TRACE_SINK_REQUEST_KEYS.intersection(sink_kwargs)
    if reserved_keys:
        key = sorted(reserved_keys)[0]
        raise ValueError(f'run trace sink config for {name!r} cannot set reserved key {key!r}')

    source = _create_source(name, source_class, source_kwargs)

    def create_sink(request: RunTraceSinkRequest) -> RunTraceSink:
        kwargs = copy.deepcopy(sink_kwargs)
        kwargs.update({
            'execution_id': request.execution_id,
            'assignment_id': request.assignment_id,
            'attempt_id': request.attempt_id,
        })
        return _create_sink(name, sink_class, kwargs)

    return RunTraceBackendRegistration(
        name=name,
        sink_factory=create_sink,
        source=source,
        source_identity=sink_config.source_identity or getattr(config, 'run_trace_source_identities', {}).get(name),
        system_managed=metadata.system_managed,
        required=metadata.required,
    )


def _unavailable_registration(
    name: str,
    metadata: RunTraceBackendMetadata,
    error: Exception,
) -> RunTraceBackendRegistration:
    """Keep a broken backend selectable-by-name so its failure stays attributable."""
    detail = f'run trace backend {name!r} is not usable by this build: {error}'

    def create_sink(request: RunTraceSinkRequest) -> RunTraceSink:
        del request
        raise RuntimeError(detail)

    return RunTraceBackendRegistration(
        name=name,
        sink_factory=create_sink,
        source=_UnavailableRunTraceSource(name, str(error)),
        system_managed=metadata.system_managed,
        required=metadata.required,
        available=_never_available,
        unavailable_detail=detail,
    )


def make_live_result_pointer(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {'_external': True, 'kind': LIVE_RESULT_POINTER_KIND, 'version': 2, 'metadata': dict(metadata)}


def make_run_result_pointer(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {'_external': True, 'kind': RUN_RESULT_POINTER_KIND, 'version': 1, 'metadata': dict(metadata)}


def is_live_result_pointer(value: Any) -> bool:
    return isinstance(value, dict) and value.get('_external') is True and value.get('kind') == LIVE_RESULT_POINTER_KIND


def is_run_result_pointer(value: Any) -> bool:
    return isinstance(value, dict) and value.get('_external') is True and value.get('kind') == RUN_RESULT_POINTER_KIND


def run_trace_metadata(runtime_info: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    raw = normalize_runtime_info(runtime_info).get(RUN_TRACE_RUNTIME_INFO_KEY)
    if not isinstance(raw, dict):
        return None
    metadata = raw.get(key)
    if not isinstance(metadata, dict):
        return None
    try:
        _context_from_envelope(metadata)
    except ValueError:
        return None
    if not isinstance(metadata.get('sinks'), list):
        return None
    return metadata


def live_trace_metadata(runtime_info: dict[str, Any] | None) -> dict[str, Any] | None:
    return run_trace_metadata(runtime_info, RUN_TRACE_LIVE_KEY)


def single_trace_metadata(runtime_info: dict[str, Any] | None) -> dict[str, Any] | None:
    return run_trace_metadata(runtime_info, RUN_TRACE_SINGLE_KEY)


__all__ = [
    'LIVE_RESULT_POINTER_KIND',
    'RUN_RESULT_POINTER_KIND',
    'RUN_TRACE_LIVE_KEY',
    'RUN_TRACE_RUNTIME_INFO_KEY',
    'RUN_TRACE_SINGLE_KEY',
    'ResolvedRunTraceBackends',
    'RunTraceBackendRegistration',
    'RunTraceBackendRegistry',
    'RunTraceSinkRequest',
    'create_run_trace_backend_registry',
    'prepare_configured_run_trace_backend_scope',
    'is_live_result_pointer',
    'is_run_result_pointer',
    'live_trace_metadata',
    'make_live_result_pointer',
    'make_run_result_pointer',
    'run_trace_metadata',
    'single_trace_metadata',
]
