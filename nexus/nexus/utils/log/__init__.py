"""Logging utilities for the nexus application.

This module provides:
- setup_logging: Main function to configure logging for the application
- Custom handlers: TqdmStreamHandler, SocketIOHandler
- Custom formatters: TrimFormatter
- Utility functions: overwrite_log, finish_overwrite
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import warnings

from logging.handlers import TimedRotatingFileHandler
from typing import TYPE_CHECKING

from nexus.utils.log.formatters import TrimFormatter
from nexus.utils.log.handlers import SocketIOHandler, TqdmStreamHandler
from nexus.utils.log.utils import finish_overwrite, overwrite_log


if TYPE_CHECKING:
    import socketio


def setup_logging(
    compat_tqdm=True,
    debug_file=False,
    log_level=logging.INFO,
    trim_threshold=(2000, 2000),
    socketio_server: socketio.AsyncServer | None = None,
) -> SocketIOHandler | None:
    """
    Setup logging for the application. This function should be called on startup only once.

    Args:
        compat_tqdm: Avoid messy log output when tqdm is used.
        debug_file: Whether to save logs to a file.
        log_level: Logging level for console output.
        trim_threshold: Tuple of (start, end) characters to keep when trimming long messages.
        socketio_server: Optional Socket.IO server to enable WebSocket log broadcasting.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = TrimFormatter(trim_threshold)

    if debug_file:
        os.makedirs('debug/logs', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = TimedRotatingFileHandler(
            f'debug/logs/{timestamp}_{os.getpid()}.log', when='midnight', encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

    use_tqdm_handler = False
    if compat_tqdm:
        try:
            from tqdm import tqdm
        except ModuleNotFoundError as exc:
            if exc.name != 'tqdm':
                raise
        else:
            use_tqdm_handler = tqdm is not None
    stream_handler = TqdmStreamHandler(sys.stdout) if use_tqdm_handler else logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(stream_handler)
    if debug_file:
        root_logger.addHandler(file_handler)

    # Setup Socket.IO logging handler if provided
    sio_handler: SocketIOHandler | None = None
    if socketio_server is not None:
        sio_handler = SocketIOHandler(socketio_server, event_name='log')
        sio_handler.setFormatter(formatter)
        sio_handler.setLevel(logging.DEBUG)  # Capture all logs
        root_logger.addHandler(sio_handler)
        logging.getLogger(__name__).debug('Socket.IO logging handler initialized')

    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai._base_client').setLevel(logging.WARNING)
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)
    logging.getLogger('databus').setLevel(logging.INFO)
    logging.getLogger('aiosqlite').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('socketio.server').setLevel(logging.WARNING)
    logging.getLogger('socketio.client').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)

    # Ignore Pydantic serialization warnings from litellm response objects
    warnings.filterwarnings('ignore', message='Pydantic serializer warnings')
    # Ignore litellm's atexit cleanup warning
    warnings.filterwarnings(
        'ignore', message="coroutine 'close_litellm_async_clients' was never awaited", category=RuntimeWarning
    )
    return sio_handler


__all__ = [
    'setup_logging',
    'TqdmStreamHandler',
    'SocketIOHandler',
    'TrimFormatter',
    'overwrite_log',
    'finish_overwrite',
]
