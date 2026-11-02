from __future__ import annotations

import asyncio
import socket
import sys

from types import SimpleNamespace

import pytest

from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution.executors.asyncio_executor import AsyncioExecutor
from nexus.distributed.execution.executors.thread_executor import ThreadExecutor
from nexus.distributed.observability.event_loop import (
    LoopAiomonitorConfig,
    monitor_event_loop,
    start_aiomonitor_for_loop,
    start_event_loop_monitor,
)


class FakeMonitor:
    def __init__(self, calls: list[dict]) -> None:
        self.calls = calls
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeEventLoopMonitor:
    def __init__(self) -> None:
        self.start_calls: list[dict] = []
        self.stopped = False

    def start(self, interval: float, **kwargs) -> None:
        self.start_calls.append({'interval': interval, **kwargs})

    def stop(self) -> None:
        self.stopped = True


def install_fake_aiomonitor(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def start_monitor(loop, **kwargs):
        monitor = FakeMonitor(calls)
        calls.append({'loop': loop, 'kwargs': kwargs, 'monitor': monitor})
        return monitor

    monkeypatch.setitem(sys.modules, 'aiomonitor', SimpleNamespace(start_monitor=start_monitor))
    return calls


def test_distributed_config_enables_aiomonitor_by_default_and_supports_env_override(monkeypatch):
    assert DistributedConfig().aiomonitor_enabled is True
    assert DistributedConfig().aiomonitor_host == '127.0.0.1'
    assert DistributedConfig().aiomonitor_base_port == 50100
    assert DistributedConfig().aiomonitor_webui_base_port == 50200
    assert DistributedConfig().aiomonitor_console_base_port == 50300

    monkeypatch.setenv('NEXUS_DIST_AIOMONITOR_ENABLED', '0')
    monkeypatch.setenv('NEXUS_DIST_AIOMONITOR_HOST', '0.0.0.0')
    monkeypatch.setenv('NEXUS_DIST_AIOMONITOR_BASE_PORT', '60100')
    monkeypatch.setenv('NEXUS_DIST_AIOMONITOR_WEBUI_BASE_PORT', '60200')
    monkeypatch.setenv('NEXUS_DIST_AIOMONITOR_CONSOLE_BASE_PORT', '60300')

    config = DistributedConfig.from_env()
    assert config.aiomonitor_enabled is False
    assert config.aiomonitor_host == '0.0.0.0'
    assert config.aiomonitor_base_port == 60100
    assert config.aiomonitor_webui_base_port == 60200
    assert config.aiomonitor_console_base_port == 60300


async def test_start_aiomonitor_for_loop_uses_configured_ports_and_closes(monkeypatch):
    calls = install_fake_aiomonitor(monkeypatch)
    loop = asyncio.get_running_loop()
    config = LoopAiomonitorConfig(
        enabled=True,
        host='127.0.0.1',
        base_port=61000,
        webui_base_port=62000,
        console_base_port=63000,
    )

    handle = start_aiomonitor_for_loop(loop, config=config, name='test-loop', port_offset=7)

    assert handle is not None
    assert calls == [
        {
            'loop': loop,
            'kwargs': {
                'host': '127.0.0.1',
                'port': 61007,
                'webui_port': 62007,
                'console_port': 63007,
                'locals': {'loop_name': 'test-loop'},
            },
            'monitor': calls[0]['monitor'],
        }
    ]
    handle.close()
    assert calls[0]['monitor'].closed is True


async def test_start_aiomonitor_for_loop_skips_occupied_port_without_calling_library(monkeypatch):
    calls = install_fake_aiomonitor(monkeypatch)
    loop = asyncio.get_running_loop()
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        occupied.listen()
        occupied_port = occupied.getsockname()[1]
        handle = start_aiomonitor_for_loop(
            loop,
            config=LoopAiomonitorConfig(
                enabled=True,
                host='127.0.0.1',
                base_port=occupied_port,
                webui_base_port=62000,
                console_base_port=63000,
            ),
            name='occupied-loop',
        )

    assert handle is None
    assert calls == []


async def test_start_event_loop_monitor_reuses_shared_lag_monitor(monkeypatch):
    from nexus.distributed.observability import event_loop as event_loop_module

    monitor = FakeEventLoopMonitor()
    monkeypatch.setattr(event_loop_module, 'EventLoopMonitor', lambda: monitor)

    started = start_event_loop_monitor('scheduler:scheduler-1')

    assert started is monitor
    assert monitor.start_calls == [
        {
            'interval': 0.1,
            'watchdog_interval': 1.0,
            'lag_threshold': 1.0,
            'window_seconds': 5.0,
            'on_alert': monitor.start_calls[0]['on_alert'],
            'on_sample': monitor.start_calls[0]['on_sample'],
        }
    ]


async def test_start_event_loop_monitor_escalates_one_continuous_stall_at_threshold(monkeypatch):
    from nexus.distributed.observability import event_loop as event_loop_module

    monitor = FakeEventLoopMonitor()
    escalations: list[float] = []
    monkeypatch.setattr(event_loop_module, 'EventLoopMonitor', lambda: monitor)

    start_event_loop_monitor(
        'instance-manager:im-1',
        stall_timeout=180.0,
        on_stall=escalations.append,
    )
    on_alert = monitor.start_calls[0]['on_alert']
    stats = {'max': 0.0, 'avg': 0.0, 'samples': 0, 'window_seconds': 5.0}

    on_alert('stall', 179.9, stats)
    on_alert('stall', 180.0, stats)
    on_alert('stall', 181.0, stats)

    assert escalations == [180.0]


async def test_start_event_loop_monitor_arms_stall_escalation_before_logging(monkeypatch):
    from nexus.distributed.observability import event_loop as event_loop_module

    monitor = FakeEventLoopMonitor()
    escalations: list[float] = []
    monkeypatch.setattr(event_loop_module, 'EventLoopMonitor', lambda: monitor)
    monkeypatch.setattr(
        event_loop_module,
        '_log_event_loop_alert',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('logging is blocked')),
    )

    start_event_loop_monitor(
        'instance-manager:im-logging-blocked',
        stall_timeout=180.0,
        on_stall=escalations.append,
    )
    on_alert = monitor.start_calls[0]['on_alert']
    stats = {'max': 0.0, 'avg': 0.0, 'samples': 0, 'window_seconds': 5.0}

    on_alert('stall', 179.9, stats)
    with pytest.raises(RuntimeError, match='logging is blocked'):
        on_alert('stall', 180.0, stats)

    assert escalations == [180.0]


async def test_monitor_event_loop_continues_when_monitor_cannot_start(monkeypatch, caplog):
    from nexus.distributed.observability import event_loop as event_loop_module

    class BrokenEventLoopMonitor:
        def start(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError('thread unavailable')

    monkeypatch.setattr(event_loop_module, 'EventLoopMonitor', BrokenEventLoopMonitor)

    result = await monitor_event_loop('scheduler:scheduler-1', asyncio.sleep(0, result='completed'))

    assert result == 'completed'
    assert 'Failed to start event loop monitor for scheduler:scheduler-1' in caplog.text


async def test_monitor_event_loop_stops_monitor_when_operation_fails(monkeypatch):
    from nexus.distributed.observability import event_loop as event_loop_module

    monitor = FakeEventLoopMonitor()
    monkeypatch.setattr(event_loop_module, 'start_event_loop_monitor', lambda name: monitor)

    async def fail() -> None:
        raise RuntimeError('service failed')

    with pytest.raises(RuntimeError, match='service failed'):
        await monitor_event_loop('reconciler', fail())

    assert monitor.stopped is True


async def test_asyncio_executor_starts_aiomonitor_on_child_loop(monkeypatch):
    calls = install_fake_aiomonitor(monkeypatch)
    executor = AsyncioExecutor(
        aiomonitor_config=LoopAiomonitorConfig(
            enabled=True,
            host='127.0.0.1',
            base_port=61000,
            webui_base_port=62000,
            console_base_port=63000,
        )
    )

    await executor.start()
    try:
        assert len(calls) == 1
        assert calls[0]['loop'] is not asyncio.get_running_loop()
        assert calls[0]['kwargs']['port'] == 61001
        assert calls[0]['kwargs']['webui_port'] == 62001
        assert calls[0]['kwargs']['console_port'] == 63001
        assert calls[0]['kwargs']['locals'] == {'loop_name': 'asyncio-executor'}
    finally:
        await executor.close()

    assert calls[0]['monitor'].closed is True


async def test_thread_executor_starts_aiomonitor_for_worker_loop_and_closes(monkeypatch):
    calls = install_fake_aiomonitor(monkeypatch)
    executor = ThreadExecutor(
        strategy='per_execution',
        max_threads=1,
        aiomonitor_config=LoopAiomonitorConfig(
            enabled=True,
            host='127.0.0.1',
            base_port=61000,
            webui_base_port=62000,
            console_base_port=63000,
        ),
    )
    await executor.start()
    try:
        from tests.unit.distributed.executors.test_thread_executor import ThreadProbeDriver

        driver = ThreadProbeDriver()
        await executor.start_assignment(
            assignment_id='assignment-1',
            execution_id='execution-1',
            driver=driver,
            spec={},
            config={},
            reporter=object(),
        )
        assert await asyncio.to_thread(driver.started.wait, 2)
        assert len(calls) == 1
        assert calls[0]['loop'] is not asyncio.get_running_loop()
        assert calls[0]['kwargs']['port'] == 61011
        assert calls[0]['kwargs']['webui_port'] == 62011
        assert calls[0]['kwargs']['console_port'] == 63011
        assert calls[0]['kwargs']['locals'] == {'loop_name': 'thread-worker-1'}

        await executor.stop_assignment('assignment-1', timeout=0.01)
        assert await asyncio.to_thread(driver.cancelled.wait, 2)
    finally:
        await executor.close()

    assert calls[0]['monitor'].closed is True
