from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor

import pytest

from opentelemetry.sdk.trace import TracerProvider


TRACEPARENT = '00-11111111111111111111111111111111-2222222222222222-01'


def _load_subprocess():
    spec = importlib.util.find_spec('nexus.utils.otel.subprocess')
    assert spec is not None, 'nexus.utils.otel.subprocess is not implemented'
    return importlib.import_module('nexus.utils.otel.subprocess')


def test_context_env_is_per_call_and_does_not_mutate_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'authorization=collector-secret')
    monkeypatch.setenv('OTEL_SERVICE_NAME', 'nexus-host')
    monkeypatch.setenv('GLOBAL_VALUE', 'kept')
    module = _load_subprocess()
    monkeypatch.setattr(module, 'inject_carrier', lambda: {'traceparent': TRACEPARENT, 'tracestate': 'vendor=value'})
    before = dict(os.environ)

    child_env = module.inject_current_context_env({'EXPLICIT_VALUE': 'kept', 'TRACEPARENT': 'stale'})

    assert child_env == {
        'EXPLICIT_VALUE': 'kept',
        'TRACEPARENT': TRACEPARENT,
        'TRACESTATE': 'vendor=value',
    }
    assert dict(os.environ) == before


def test_random_subprocess_receives_only_carrier_not_managed_otel_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    monkeypatch.setattr(module, 'inject_carrier', lambda: {'traceparent': TRACEPARENT})

    child_env = module.inject_current_context_env(
        {
            'KEEP': 'value',
            'NEXUS_OTEL_ENABLED': 'true',
            'OTEL_EXPORTER_OTLP_HEADERS': 'authorization=collector-secret',
            'OTEL_BSP_MAX_QUEUE_SIZE': '4096',
        }
    )

    assert child_env == {'KEEP': 'value', 'TRACEPARENT': TRACEPARENT}


def test_managed_subprocess_can_preserve_official_otel_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    monkeypatch.setattr(module, 'inject_carrier', lambda: {'traceparent': TRACEPARENT})
    managed_env = {
        'NEXUS_OTEL_ENABLED': 'true',
        'OTEL_EXPORTER_OTLP_HEADERS': 'authorization=collector-secret',
        'OTEL_SERVICE_NAME': 'nexus-runtime-worker',
    }

    with module.preserve_managed_otel_environment():
        child_env = module.inject_current_context_env(managed_env)

    assert child_env == {**managed_env, 'TRACEPARENT': TRACEPARENT}


def test_disabled_context_env_removes_stale_carrier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)

    child_env = _load_subprocess().inject_current_context_env({'KEEP': 'value', 'TRACEPARENT': 'stale'})

    assert child_env == {'KEEP': 'value'}


def test_popen_instrumentor_injects_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    monkeypatch.setattr(module, 'inject_carrier', lambda: {'traceparent': TRACEPARENT})
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    original_init = subprocess.Popen.__init__

    instrumentor.instrument(skip_dep_check=True)
    try:
        completed = subprocess.run(
            [sys.executable, '-c', 'import os; print(os.environ.get("TRACEPARENT", ""))'],
            capture_output=True,
            check=True,
            env={'PATH': os.environ.get('PATH', '')},
            text=True,
        )
        assert completed.stdout.strip() == TRACEPARENT
        assert subprocess.Popen.__init__ is not original_init
    finally:
        instrumentor.uninstrument()

    assert subprocess.Popen.__init__ is original_init


def test_recreating_singleton_does_not_lose_originals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    original_init = subprocess.Popen.__init__

    instrumentor.instrument(skip_dep_check=True)
    try:
        assert module.NexusSubprocessInstrumentor() is instrumentor
    finally:
        instrumentor.uninstrument()

    assert subprocess.Popen.__init__ is original_init


@pytest.mark.asyncio
async def test_asyncio_create_subprocess_exec_injects_carrier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    monkeypatch.setattr(module, 'inject_carrier', lambda: {'traceparent': TRACEPARENT})
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    instrumentor.instrument(skip_dep_check=True)
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-c',
            'import os; print(os.environ.get("TRACEPARENT", ""))',
            env={'PATH': os.environ.get('PATH', '')},
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        assert process.returncode == 0
        assert stdout.decode().strip() == TRACEPARENT
    finally:
        instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_asyncio_create_subprocess_shell_injects_carrier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    monkeypatch.setattr(module, 'inject_carrier', lambda: {'traceparent': TRACEPARENT})
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    instrumentor.instrument(skip_dep_check=True)
    try:
        process = await asyncio.create_subprocess_shell(
            f'{sys.executable} -c \'import os; print(os.environ.get("TRACEPARENT", ""))\'',
            env={'PATH': os.environ.get('PATH', '')},
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        assert process.returncode == 0
        assert stdout.decode().strip() == TRACEPARENT
    finally:
        instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_context_env_uses_each_task_current_span(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    provider = TracerProvider(shutdown_on_exit=False)
    tracer = provider.get_tracer(__name__)

    async def capture(name: str) -> tuple[str, str]:
        with tracer.start_as_current_span(name) as span:
            await asyncio.sleep(0)
            traceparent = module.inject_current_context_env({})['TRACEPARENT']
            return traceparent, f'{span.get_span_context().span_id:016x}'

    try:
        first, second = await asyncio.gather(capture('first'), capture('second'))
    finally:
        provider.shutdown()

    assert first[0].split('-')[2] == first[1]
    assert second[0].split('-')[2] == second[1]
    assert first[0] != second[0]


def test_popen_injection_failure_uses_original_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()

    def fail_injection(_env) -> None:
        raise RuntimeError('boom')

    monkeypatch.setattr(module, 'inject_current_context_env', fail_injection)

    instrumentor.instrument(skip_dep_check=True)
    try:
        completed = subprocess.run(
            [sys.executable, '-c', 'print("business-result")'],
            capture_output=True,
            check=True,
            env={'PATH': os.environ.get('PATH', '')},
            text=True,
        )
    finally:
        instrumentor.uninstrument()

    assert completed.stdout.strip() == 'business-result'


def test_popen_original_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    calls: list[tuple[object, ...]] = []

    def failing_init(_instance, *args, **_kwargs) -> None:
        calls.append(args)
        raise ValueError('original failed')

    monkeypatch.setattr(subprocess.Popen, '__init__', failing_init)
    instrumentor.instrument(skip_dep_check=True)
    try:
        positional = (['command'], -1, None, None, None, None, None, None, True, False, None, {'PATH': ''})
        with pytest.raises(ValueError, match='original failed'):
            subprocess.Popen.__init__(object(), *positional)
    finally:
        instrumentor.uninstrument()

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_asyncio_shell_preserves_native_call_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def create_shell(cmd, *args, **kwargs):
        calls.append((cmd, args, kwargs))
        return 'process'

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', create_shell)
    instrumentor.instrument(skip_dep_check=True)
    try:
        positional = await asyncio.create_subprocess_shell('first', 'stdin', env={'KEEP': 'value'})
        keyword = await asyncio.create_subprocess_shell(cmd='second', env={'KEEP': 'value'})
    finally:
        instrumentor.uninstrument()

    assert positional == keyword == 'process'
    assert calls[0][0:2] == ('first', ('stdin',))
    assert calls[1][0:2] == ('second', ())
    assert calls[0][2]['env']['KEEP'] == 'value'
    assert calls[1][2]['env']['KEEP'] == 'value'


def test_disabled_instrumentor_does_not_patch_or_mark_instrumented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'false')
    module = _load_subprocess()
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    originals = (
        subprocess.Popen.__init__,
        asyncio.create_subprocess_exec,
        asyncio.create_subprocess_shell,
    )

    instrumentor.instrument(skip_dep_check=True)

    assert instrumentor.is_instrumented_by_opentelemetry is False
    assert originals == (
        subprocess.Popen.__init__,
        asyncio.create_subprocess_exec,
        asyncio.create_subprocess_shell,
    )


def test_concurrent_instrumentation_restores_exact_originals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    module = _load_subprocess()
    instrumentor = module.NexusSubprocessInstrumentor()
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()
    originals = (
        subprocess.Popen.__init__,
        asyncio.create_subprocess_exec,
        asyncio.create_subprocess_shell,
    )
    original_instrument = instrumentor._instrument
    start = threading.Event()

    def slow_instrument(**kwargs) -> None:
        time.sleep(0.02)
        original_instrument(**kwargs)

    def install(_index: int) -> None:
        start.wait()
        instrumentor.instrument(skip_dep_check=True)

    monkeypatch.setattr(instrumentor, '_instrument', slow_instrument)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(install, index) for index in range(8)]
        start.set()
        for future in futures:
            future.result()
    instrumentor.uninstrument()

    assert originals == (
        subprocess.Popen.__init__,
        asyncio.create_subprocess_exec,
        asyncio.create_subprocess_shell,
    )
