"""The Engine.IO idle-heartbeat abort is logged as INFO."""

from __future__ import annotations

import logging

from nexus.runtime.client.remote import _EngineIOHeartbeatFilter


def test_filter_is_installed_on_the_engineio_client_logger() -> None:
    engineio_logger = logging.getLogger('engineio.client')
    assert any(isinstance(item, _EngineIOHeartbeatFilter) for item in engineio_logger.filters)


def test_empty_packet_queue_is_logged_as_info() -> None:
    record = logging.LogRecord(
        name='engineio.client',
        level=logging.ERROR,
        pathname='async_client.py',
        lineno=639,
        msg='packet queue is empty, aborting',
        args=(),
        exc_info=None,
    )

    assert _EngineIOHeartbeatFilter().filter(record) is True
    assert record.levelno == logging.INFO
    assert record.levelname == 'INFO'


def test_other_engineio_errors_stay_errors() -> None:
    record = logging.LogRecord(
        name='engineio.client',
        level=logging.ERROR,
        pathname='async_client.py',
        lineno=1,
        msg='aiohttp package not installed',
        args=(),
        exc_info=None,
    )

    assert _EngineIOHeartbeatFilter().filter(record) is True
    assert record.levelno == logging.ERROR
    assert record.levelname == 'ERROR'
