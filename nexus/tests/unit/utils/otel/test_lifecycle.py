from __future__ import annotations

import asyncio
import importlib

import pytest


def _load_lifecycle():
    spec = importlib.util.find_spec('nexus.utils.otel.lifecycle')
    assert spec is not None, 'nexus.utils.otel.lifecycle is not implemented'
    return importlib.import_module('nexus.utils.otel.lifecycle')


class _Provider:
    def __init__(self) -> None:
        self.flush_timeouts: list[int] = []
        self.shutdown_calls = 0

    def force_flush(self, timeout_millis: int) -> bool:
        self.flush_timeouts.append(timeout_millis)
        return True

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_force_flush_is_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)

    assert _load_lifecycle().force_flush() is True


def test_lifecycle_delegates_to_global_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = _Provider()
    trace = importlib.import_module('opentelemetry.trace')
    monkeypatch.setattr(trace, 'get_tracer_provider', lambda: provider)
    lifecycle = _load_lifecycle()

    assert lifecycle.force_flush(1234) is True
    lifecycle.shutdown()

    assert provider.flush_timeouts == [1234]
    assert provider.shutdown_calls == 1


def test_lifecycle_is_noop_when_optional_dependency_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    lifecycle = _load_lifecycle()
    monkeypatch.setattr(lifecycle, '_get_tracer_provider', lambda: None)

    assert lifecycle.force_flush() is True
    lifecycle.shutdown()


def test_force_flush_best_effort_never_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    lifecycle = _load_lifecycle()

    def fail(*, timeout_millis: int) -> bool:
        assert timeout_millis == 123
        raise RuntimeError('collector unavailable')

    monkeypatch.setattr(lifecycle, 'force_flush', fail)

    assert lifecycle.force_flush_best_effort(timeout_millis=123) is False
    assert 'OpenTelemetry force flush failed' in caplog.text


def test_force_flush_best_effort_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lifecycle = _load_lifecycle()
    monkeypatch.setattr(lifecycle, 'force_flush', lambda *, timeout_millis: False)

    assert lifecycle.force_flush_best_effort() is False
    assert 'OpenTelemetry force flush timed out' in caplog.text


@pytest.mark.asyncio
async def test_async_force_flush_best_effort_offloads_blocking_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = _load_lifecycle()
    calls: list[tuple[object, tuple, dict]] = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, 'to_thread', fake_to_thread)
    monkeypatch.setattr(lifecycle, 'force_flush_best_effort', lambda *, timeout_millis: timeout_millis == 123)

    assert await lifecycle.async_force_flush_best_effort(timeout_millis=123) is True
    assert calls == [(lifecycle.force_flush_best_effort, (), {'timeout_millis': 123})]
