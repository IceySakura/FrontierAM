from __future__ import annotations

import asyncio
import logging
import socket

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from nexus.utils.eventloop_monitor import EventLoopMonitor
from nexus.utils.platform_metrics import record


logger = logging.getLogger(__name__)


_T = TypeVar('_T')
_LAG_PROBE_INTERVAL_SECONDS = 0.1
_LAG_WATCHDOG_INTERVAL_SECONDS = 1.0
_LAG_THRESHOLD_SECONDS = 1.0
_LAG_WINDOW_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class LoopAiomonitorConfig:
    enabled: bool = True
    host: str = '127.0.0.1'
    base_port: int = 50100
    webui_base_port: int = 50200
    console_base_port: int = 50300

    @classmethod
    def from_distributed_config(cls, config: Any) -> LoopAiomonitorConfig:
        return cls(
            enabled=config.aiomonitor_enabled,
            host=config.aiomonitor_host,
            base_port=config.aiomonitor_base_port,
            webui_base_port=config.aiomonitor_webui_base_port,
            console_base_port=config.aiomonitor_console_base_port,
        )


@dataclass(slots=True)
class AiomonitorHandle:
    name: str
    host: str
    port: int
    webui_port: int
    console_port: int
    _monitor: Any
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._monitor, 'close', None)
        if close is None:
            return
        try:
            close()
        except Exception:
            logger.exception('Failed to close aiomonitor for %s', self.name)


def start_event_loop_monitor(
    name: str,
    *,
    stall_timeout: float | None = None,
    on_stall: Callable[[float], None] | None = None,
) -> EventLoopMonitor | None:
    if not name:
        raise ValueError('event loop monitor name is required')
    if (stall_timeout is None) != (on_stall is None):
        raise ValueError('stall_timeout and on_stall must be configured together')
    if stall_timeout is not None and stall_timeout <= 0:
        raise ValueError('stall_timeout must be positive')
    monitor = EventLoopMonitor()
    escalated = False

    def handle_alert(reason: str, value: float, stats: dict[str, Any]) -> None:
        nonlocal escalated
        if reason != 'stall' or stall_timeout is None or on_stall is None:
            _log_event_loop_alert(name, reason, value, stats)
            return
        if value < stall_timeout or escalated:
            return
        escalated = True
        on_stall(value)
        _log_event_loop_alert(name, reason, value, stats)

    try:
        monitor.start(
            interval=_LAG_PROBE_INTERVAL_SECONDS,
            watchdog_interval=_LAG_WATCHDOG_INTERVAL_SECONDS,
            lag_threshold=_LAG_THRESHOLD_SECONDS,
            window_seconds=_LAG_WINDOW_SECONDS,
            on_alert=handle_alert,
            on_sample=lambda lag: record('event_loop_lag', lag, component=name.split(':', 1)[0]),
        )
    except Exception:
        logger.exception('Failed to start event loop monitor for %s', name)
        return None
    return monitor


async def monitor_event_loop(
    name: str,
    operation: Awaitable[_T],
    *,
    stall_timeout: float | None = None,
    on_stall: Callable[[float], None] | None = None,
) -> _T:
    monitor = (
        start_event_loop_monitor(name)
        if stall_timeout is None and on_stall is None
        else start_event_loop_monitor(name, stall_timeout=stall_timeout, on_stall=on_stall)
    )
    try:
        return await operation
    finally:
        if monitor is not None:
            monitor.stop()


def _log_event_loop_alert(name: str, reason: str, value: float, stats: dict[str, Any]) -> None:
    logger.warning(
        'Distributed event loop lag: name=%s reason=%s lag_seconds=%.3f '
        'window_max_seconds=%.3f window_avg_seconds=%.3f samples=%s window_seconds=%.1f',
        name,
        reason,
        value,
        stats.get('max', 0.0),
        stats.get('avg', 0.0),
        stats.get('samples', 0),
        stats.get('window_seconds', 0.0),
    )


def start_aiomonitor_for_loop(
    loop: asyncio.AbstractEventLoop,
    *,
    config: LoopAiomonitorConfig,
    name: str,
    port_offset: int = 0,
) -> AiomonitorHandle | None:
    if not config.enabled:
        return None

    port = config.base_port + port_offset
    webui_port = config.webui_base_port + port_offset
    console_port = config.console_base_port + port_offset
    ports = (port, webui_port, console_port)
    if not _ports_available(config.host, ports):
        logger.warning(
            'Skipping aiomonitor for %s because a configured port is unavailable on %s: %s',
            name,
            config.host,
            ports,
        )
        return None
    try:
        import aiomonitor
    except ImportError:
        logger.warning('aiomonitor is enabled for %s but the aiomonitor package is not installed', name)
        return None

    try:
        monitor = aiomonitor.start_monitor(
            loop,
            host=config.host,
            port=port,
            webui_port=webui_port,
            console_port=console_port,
            locals={'loop_name': name},
        )
    except Exception:
        logger.exception(
            'Failed to start aiomonitor for %s on %s:%s/%s/%s',
            name,
            config.host,
            port,
            webui_port,
            console_port,
        )
        return None

    logger.info(
        'Started aiomonitor for %s on %s:%s webui=%s console=%s',
        name,
        config.host,
        port,
        webui_port,
        console_port,
    )
    return AiomonitorHandle(
        name=name,
        host=config.host,
        port=port,
        webui_port=webui_port,
        console_port=console_port,
        _monitor=monitor,
    )


def _ports_available(host: str, ports: tuple[int, ...]) -> bool:
    if len(set(ports)) != len(ports):
        return False
    reservations: list[socket.socket] = []
    try:
        for port in ports:
            reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reservations.append(reservation)
            reservation.bind((host, port))
            reservation.listen(1)
        return True
    except (OSError, OverflowError):
        return False
    finally:
        for reservation in reservations:
            reservation.close()
