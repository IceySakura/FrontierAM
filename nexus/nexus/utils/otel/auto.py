from __future__ import annotations

import importlib
import os
import sys
import threading

from collections.abc import Callable
from logging import getLogger
from typing import Any


logger = getLogger(__name__)

_LOCK = threading.Lock()
_initialized = False
_missing_dependency_logged = False
_runtime_available: bool | None = None

_HTTP_INSTRUMENTORS = ('fastapi', 'asgi')
_FASTAPI_EXCLUDE_SPANS_ENV = 'NEXUS_OTEL_FASTAPI_EXCLUDE_SPANS'
_USER_DISABLED_INSTRUMENTORS: set[str] = set()


def is_requested() -> bool:
    return os.getenv('NEXUS_OTEL_ENABLED', '').strip().lower() == 'true'


def is_enabled() -> bool:
    return is_requested() and _runtime_available is not False


def _load_official_initializer() -> Callable[..., Any] | None:
    try:
        import opentelemetry.distro  # noqa: F401

        from opentelemetry.instrumentation.auto_instrumentation import initialize
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith('opentelemetry'):
            raise
        return None
    return initialize


def _has_configured_tracer_provider() -> bool:
    try:
        from opentelemetry import trace
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith('opentelemetry'):
            raise
        return False
    return not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)


def _official_auto_instrumentation_is_running() -> bool:
    if 'opentelemetry.instrumentation.auto_instrumentation.sitecustomize' in sys.modules:
        return True

    # ``opentelemetry-instrument`` prepends the auto-instrumentation directory
    # to ``PYTHONPATH``, so Python imports the official hook as the top-level
    # ``sitecustomize`` module rather than under its package-qualified name.
    sitecustomize = sys.modules.get('sitecustomize')
    origin = getattr(getattr(sitecustomize, '__spec__', None), 'origin', None) or getattr(sitecustomize, '__file__', '')
    normalized = os.path.normcase(os.path.normpath(str(origin)))
    suffix = os.path.join('opentelemetry', 'instrumentation', 'auto_instrumentation', 'sitecustomize.py')
    return normalized.endswith(suffix) or normalized.endswith(f'{suffix}c')


def _instrument_nexus() -> None:
    from importlib.metadata import entry_points

    disabled = {name.strip() for name in os.getenv('OTEL_PYTHON_DISABLED_INSTRUMENTATIONS', '').split(',')}
    if '*' in disabled:
        return
    for entry_point in entry_points(group='opentelemetry_instrumentor'):
        if entry_point.value.startswith('nexus.utils.otel.') and entry_point.name not in disabled:
            instrumentor = entry_point.load()()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()


def instrumentor_options() -> dict[str, dict[str, Any]]:
    """Instrumentor options from ``NEXUS_OTEL_<NAME>_*``; empty by default, which keeps every span."""
    raw = os.getenv(_FASTAPI_EXCLUDE_SPANS_ENV, '')
    exclude_spans = [item.strip() for item in raw.split(',') if item.strip()]
    return {'fastapi': {'exclude_spans': exclude_spans}} if exclude_spans else {}


def _disable_auto_http_instrumentors() -> None:
    names = [name.strip() for name in os.getenv('OTEL_PYTHON_DISABLED_INSTRUMENTATIONS', '').split(',') if name.strip()]
    _USER_DISABLED_INSTRUMENTORS.update(names)
    for name in _HTTP_INSTRUMENTORS:
        if name not in names:
            names.append(name)
    os.environ['OTEL_PYTHON_DISABLED_INSTRUMENTATIONS'] = ','.join(names)


def _has_live_instrumented_apps(module: Any) -> bool:
    apps = getattr(getattr(module, '_InstrumentedFastAPI', None), '_instrumented_fastapi_apps', None)
    # Unknown bookkeeping counts as live: uninstrumenting would strip a serving app's middleware.
    return apps is None or bool(apps)


def _install_configured_instrumentors(configured: dict[str, dict[str, Any]]) -> None:
    """Install the configured instrumentors, replacing an option-less install only while no app exists."""
    if not configured:
        return

    from importlib.metadata import entry_points

    for entry_point in entry_points(group='opentelemetry_instrumentor'):
        options = configured.get(entry_point.name)
        if options is None or entry_point.name in _USER_DISABLED_INSTRUMENTORS:
            continue

        instrumentor = entry_point.load()()
        if instrumentor.is_instrumented_by_opentelemetry:
            if _has_live_instrumented_apps(importlib.import_module(entry_point.module)):
                logger.debug('%s already guards live apps; keeping the current spans', entry_point.name)
                continue
            instrumentor.uninstrument()

        try:
            instrumentor.instrument(**options)
        except TypeError:
            logger.info('%s does not support the configured options; keeping default spans', entry_point.name)
            instrumentor.instrument()


def initialize_if_enabled() -> None:
    global _initialized, _missing_dependency_logged, _runtime_available

    if not is_requested() or _initialized:
        return

    with _LOCK:
        if _initialized:
            return

        os.environ.setdefault('OTEL_PYTHON_DISTRO', 'distro')
        os.environ.setdefault('OTEL_PYTHON_CONFIGURATOR', 'configurator')
        os.environ.setdefault('OTEL_TRACES_EXPORTER', 'otlp_proto_http')
        os.environ.setdefault('OTEL_EXPORTER_OTLP_PROTOCOL', 'http/protobuf')
        os.environ.setdefault('OTEL_METRICS_EXPORTER', 'none')
        os.environ.setdefault('OTEL_LOGS_EXPORTER', 'none')
        os.environ['OTEL_PROPAGATORS'] = 'tracecontext'
        os.environ.setdefault('OTEL_PYTHON_AIOHTTP_SERVER_EXCLUDED_URLS', '.*/v1/health')
        os.environ.setdefault('NEXUS_OTEL_PROCESS_ROLE', 'host')
        os.environ.setdefault('OTEL_SERVICE_NAME', 'nexus-host')

        initialize = _load_official_initializer()
        if initialize is None:
            if not _missing_dependency_logged:
                logger.warning('Nexus OTel is enabled but the optional OTel dependencies are not installed')
                _missing_dependency_logged = True
            _runtime_available = False
            _initialized = True
            return

        _runtime_available = True
        configured = instrumentor_options()
        if configured:
            # exclude_spans has no env var, so the distro must not install its option-less copy.
            _disable_auto_http_instrumentors()
        if _has_configured_tracer_provider():
            if not _official_auto_instrumentation_is_running():
                _instrument_nexus()
            _install_configured_instrumentors(configured)
            _initialized = True
            return

        initialize(swallow_exceptions=False)
        _install_configured_instrumentors(configured)
        _initialized = True
