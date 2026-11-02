"""Tests for logging setup and the runtime server's DEBUG log level."""

from __future__ import annotations

import builtins
import importlib.util
import logging

from nexus.utils.log import setup_logging


def test_setup_logging_stream_handler_debug_when_app_py_passes_debug() -> None:
    """The way app.py calls setup_logging must result in a DEBUG stdout stream handler.

    Replicates the exact call in nexus/runtime/server/app.py:204 (minus the socketio server,
    which is orthogonal to the stdout stream handler under test).
    """
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    saved_level = root_logger.level

    try:
        setup_logging(log_level=logging.DEBUG)

        stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert stream_handlers, 'setup_logging() should attach a stdout stream handler'
        assert stream_handlers[0].level == logging.DEBUG
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in saved_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(saved_level)


def test_setup_logging_uses_plain_stream_handler_when_tqdm_is_unavailable(monkeypatch) -> None:
    """Runtime startup must verify tqdm is importable instead of trusting its module spec."""

    class StubSocketIOServer:
        async def emit(self, _event_name, _payload):
            return None

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == 'tqdm' or name.startswith('tqdm.'):
            raise ModuleNotFoundError("No module named 'tqdm'", name='tqdm')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'find_spec', lambda _name: object())
    monkeypatch.setattr(builtins, '__import__', guarded_import)

    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    saved_level = root_logger.level

    try:
        setup_logging(socketio_server=StubSocketIOServer(), log_level=logging.DEBUG)  # type: ignore[arg-type]

        stream_handlers = [handler for handler in root_logger.handlers if isinstance(handler, logging.StreamHandler)]
        assert len(stream_handlers) == 1
        assert type(stream_handlers[0]) is logging.StreamHandler
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in saved_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(saved_level)


def test_app_py_calls_setup_logging_with_debug_log_level() -> None:
    """The app.py module must invoke setup_logging with log_level=logging.DEBUG.

    Reads the source directly (via importlib, without importing the module) so this stays
    a compile-time-ish guard: if someone silently drops the log_level argument, this fails
    without needing to boot the server. Importing nexus.runtime.server.app for real triggers
    module-level side effects (starting the socketio/ASGI app setup) that hang in a test process.
    """
    from pathlib import Path

    spec = importlib.util.find_spec('nexus.runtime.server.app')
    assert spec is not None
    assert spec.origin is not None
    source = Path(spec.origin).read_text(encoding='utf-8')
    assert 'setup_logging(socketio_server=sio, log_level=logging.DEBUG)' in source, (
        'nexus/runtime/server/app.py must call setup_logging with log_level=logging.DEBUG so the '
        'sandbox container stdout defaults to DEBUG level.'
    )
