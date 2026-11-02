"""Logging formatters for the application."""

from __future__ import annotations

import copy
import json
import logging

from collections.abc import Callable, Collection, Mapping
from datetime import datetime, timezone
from traceback import format_exception
from typing import Any


_LOG_RECORD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {'message', 'asctime'}


class TrimFormatter(logging.Formatter):
    """Formatter that trims long log messages to prevent output bloat."""

    _inner = ' [... (truncated log)] '

    def __init__(self, trim_threshold, show_lineno=False):
        if show_lineno:
            super().__init__(
                '%(asctime)s | %(levelname)s | %(name)s(L%(lineno)d) | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        else:
            super().__init__(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        self._trim_threshold = trim_threshold

    def format(self, record):
        trimmed_record = copy.copy(record)
        trimmed_record.msg = self._trim(str(record.msg))
        return super().format(trimmed_record)

    def _trim(self, msg):
        if not self._trim_threshold:
            return msg
        start, end = self._trim_threshold
        # Both 0 means "no trimming"; otherwise trim once the message exceeds the budget.
        if start <= 0 and end <= 0:
            return msg
        if len(msg) <= start + end:
            return msg
        msg_start = msg[:start] if start > 0 else ''
        msg_end = msg[-end:] if end > 0 else ''
        return msg_start + self._inner + msg_end


class JsonLineFormatter(logging.Formatter):
    def __init__(
        self,
        *,
        component: str,
        identity: str | None = None,
        context_provider: Callable[[], Mapping[str, Any]] | None = None,
        top_level_extra_fields: Collection[str] = (),
    ) -> None:
        super().__init__()
        self.component = component
        self.identity = identity
        self.context_provider = context_provider
        self.top_level_extra_fields = set(top_level_extra_fields)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            'timestamp': _record_timestamp(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'component': self.component,
            'pathname': record.pathname,
            'lineno': record.lineno,
            'func_name': record.funcName,
            'process': record.process,
            'thread': record.thread,
        }
        if self.identity is not None:
            payload['identity'] = self.identity

        if self.context_provider is not None:
            for key, value in self.context_provider().items():
                if key in self.top_level_extra_fields:
                    payload[key] = _json_safe(value)

        extra_payload: dict[str, object] = {}
        for key, value in _record_extra(record).items():
            safe_value = _json_safe(value)
            if key in self.top_level_extra_fields:
                payload[key] = safe_value
            else:
                extra_payload[key] = safe_value
        if extra_payload:
            payload['extra'] = extra_payload

        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            payload['exception'] = {
                'type': exc_type.__name__ if exc_type else None,
                'message': str(exc_value) if exc_value is not None else '',
                'traceback': ''.join(format_exception(exc_type, exc_value, exc_traceback)),
            }

        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def _record_timestamp(record: logging.LogRecord) -> str:
    return (
        datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    )


def _record_extra(record: logging.LogRecord) -> dict[str, object]:
    result = {}
    for key, value in record.__dict__.items():
        if key.startswith('_') or key in _LOG_RECORD_FIELDS:
            continue
        result[key] = value
    return result


def _json_safe(value):
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
