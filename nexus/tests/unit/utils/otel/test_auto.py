from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def _load_auto():
    spec = importlib.util.find_spec('nexus.utils.otel.auto')
    assert spec is not None, 'nexus.utils.otel.auto is not implemented'
    return importlib.reload(importlib.import_module('nexus.utils.otel.auto'))


@pytest.fixture(autouse=True)
def _clear_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        'NEXUS_OTEL_ENABLED',
        'OTEL_PYTHON_DISTRO',
        'OTEL_PYTHON_CONFIGURATOR',
        'OTEL_TRACES_EXPORTER',
        'OTEL_EXPORTER_OTLP_PROTOCOL',
        'OTEL_METRICS_EXPORTER',
        'OTEL_LOGS_EXPORTER',
        'OTEL_PROPAGATORS',
        'OTEL_PYTHON_AIOHTTP_SERVER_EXCLUDED_URLS',
        'NEXUS_OTEL_PROCESS_ROLE',
        'OTEL_SERVICE_NAME',
    ):
        monkeypatch.delenv(key, raising=False)


def test_disabled_initialization_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    auto = _load_auto()
    auto_instrumentation = importlib.import_module('opentelemetry.instrumentation.auto_instrumentation')
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(auto_instrumentation, 'initialize', lambda **kwargs: calls.append(kwargs))

    auto.initialize_if_enabled()

    assert calls == []
    assert 'OTEL_PYTHON_DISTRO' not in os.environ


def test_disabled_package_import_does_not_load_otel() -> None:
    env = dict(os.environ)
    env.pop('NEXUS_OTEL_ENABLED', None)
    project_root = Path(__file__).parents[4]

    completed = subprocess.run(
        [
            sys.executable,
            '-c',
            'import nexus, sys; print(any(name.startswith("opentelemetry") for name in sys.modules))',
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stdout.strip() == 'False'


def test_socketio_is_loaded_only_when_a_broadcaster_is_created() -> None:
    script = textwrap.dedent(
        """
        import sys
        from unittest.mock import AsyncMock
        from nexus.trace import SocketIOSpanBroadcaster

        assert 'socketio' not in sys.modules
        SocketIOSpanBroadcaster(AsyncMock())
        assert 'socketio' in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, '-c', script],
        cwd=Path(__file__).parents[4],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_external_auto_instrumentation_does_not_initialize_nexus_twice() -> None:
    command = Path(sys.executable).with_name('opentelemetry-instrument')
    if not command.exists():
        pytest.skip('opentelemetry-instrument is not installed')
    env = dict(os.environ)
    env.update(
        {
            'NEXUS_OTEL_ENABLED': 'true',
            'OTEL_TRACES_EXPORTER': 'none',
            'OTEL_METRICS_EXPORTER': 'none',
            'OTEL_LOGS_EXPORTER': 'none',
        }
    )

    completed = subprocess.run(
        [str(command), sys.executable, '-c', 'import nexus'],
        cwd=Path(__file__).parents[4],
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )

    output = completed.stdout + completed.stderr
    assert 'Overriding of current' not in output
    assert 'Attempting to instrument while already instrumented' not in output


def test_enabled_initialization_uses_official_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    auto = _load_auto()
    auto_instrumentation = importlib.import_module('opentelemetry.instrumentation.auto_instrumentation')
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(auto, '_has_configured_tracer_provider', lambda: False)
    monkeypatch.setattr(auto_instrumentation, 'initialize', lambda **kwargs: calls.append(kwargs))

    auto.initialize_if_enabled()

    assert calls == [{'swallow_exceptions': False}]
    assert os.environ['OTEL_PYTHON_DISTRO'] == 'distro'
    assert os.environ['OTEL_PYTHON_CONFIGURATOR'] == 'configurator'
    assert os.environ['OTEL_TRACES_EXPORTER'] == 'otlp_proto_http'
    assert os.environ['OTEL_EXPORTER_OTLP_PROTOCOL'] == 'http/protobuf'
    assert os.environ['OTEL_METRICS_EXPORTER'] == 'none'
    assert os.environ['OTEL_LOGS_EXPORTER'] == 'none'
    assert os.environ['OTEL_PROPAGATORS'] == 'tracecontext'
    assert os.environ['OTEL_PYTHON_AIOHTTP_SERVER_EXCLUDED_URLS'] == '.*/v1/health'
    assert os.environ['NEXUS_OTEL_PROCESS_ROLE'] == 'host'
    assert os.environ['OTEL_SERVICE_NAME'] == 'nexus-host'


def test_enabled_initialization_forces_tracecontext_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_PROPAGATORS', 'baggage,tracecontext')
    auto = _load_auto()
    auto_instrumentation = importlib.import_module('opentelemetry.instrumentation.auto_instrumentation')
    monkeypatch.setattr(auto, '_has_configured_tracer_provider', lambda: False)
    monkeypatch.setattr(auto_instrumentation, 'initialize', lambda **_kwargs: None)

    auto.initialize_if_enabled()

    assert os.environ['OTEL_PROPAGATORS'] == 'tracecontext'


def test_enabled_initialization_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    auto = _load_auto()
    auto_instrumentation = importlib.import_module('opentelemetry.instrumentation.auto_instrumentation')
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(auto, '_has_configured_tracer_provider', lambda: False)
    monkeypatch.setattr(auto_instrumentation, 'initialize', lambda **kwargs: calls.append(kwargs))

    auto.initialize_if_enabled()
    auto.initialize_if_enabled()

    assert calls == [{'swallow_exceptions': False}]


def test_enabled_initialization_does_not_install_json_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    auto = _load_auto()
    auto_instrumentation = importlib.import_module('opentelemetry.instrumentation.auto_instrumentation')
    calls: list[str] = []
    monkeypatch.setattr(auto, '_has_configured_tracer_provider', lambda: False)
    monkeypatch.setattr(auto_instrumentation, 'initialize', lambda **_kwargs: calls.append('initialize'))

    auto.initialize_if_enabled()

    assert calls == ['initialize']


def test_enabled_initialization_is_thread_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    auto = _load_auto()
    auto_instrumentation = importlib.import_module('opentelemetry.instrumentation.auto_instrumentation')
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(auto, '_has_configured_tracer_provider', lambda: False)
    monkeypatch.setattr(auto_instrumentation, 'initialize', lambda **kwargs: calls.append(kwargs))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _index: auto.initialize_if_enabled(), range(32)))

    assert calls == [{'swallow_exceptions': False}]


def test_partial_otel_install_is_reported_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    auto = _load_auto()
    monkeypatch.setattr(auto, '_load_official_initializer', lambda: None)

    auto.initialize_if_enabled()

    assert 'optional OTel dependencies are not installed' in caplog.text
    assert auto.is_enabled() is False

    from nexus.trace import SpanController, span_context
    from nexus.utils.otel.transport import transport_span

    previous_controller = SpanController._instance
    SpanController._instance = None
    caplog.clear()
    try:
        with span_context('missing-otel'):
            with transport_span('missing-otel.transport') as span:
                assert span is None
    finally:
        SpanController._instance = previous_controller

    assert not [record for record in caplog.records if record.name.startswith('nexus.utils.otel')]


def test_external_auto_instrumentation_skips_nested_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    auto = _load_auto()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(auto, '_has_configured_tracer_provider', lambda: True)
    monkeypatch.setattr(auto, '_official_auto_instrumentation_is_running', lambda: True)
    monkeypatch.setattr(auto, '_load_official_initializer', lambda: lambda **kwargs: calls.append(kwargs))

    auto.initialize_if_enabled()

    assert calls == []
    assert auto.is_enabled() is True


def test_programmatic_provider_still_installs_nexus_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    auto = _load_auto()
    calls: list[str] = []
    monkeypatch.setattr(auto, '_has_configured_tracer_provider', lambda: True)
    monkeypatch.setattr(auto, '_official_auto_instrumentation_is_running', lambda: False)
    monkeypatch.setattr(auto, '_instrument_nexus', lambda: calls.append('nexus'))
    monkeypatch.setattr(auto, '_load_official_initializer', lambda: lambda **_kwargs: calls.append('initialize'))

    auto.initialize_if_enabled()

    assert calls == ['nexus']


def test_nexus_package_calls_early_initializer(monkeypatch: pytest.MonkeyPatch) -> None:
    import nexus

    auto = _load_auto()
    calls: list[None] = []
    monkeypatch.setattr(auto, 'initialize_if_enabled', lambda: calls.append(None))

    importlib.reload(nexus)

    assert calls == [None]
