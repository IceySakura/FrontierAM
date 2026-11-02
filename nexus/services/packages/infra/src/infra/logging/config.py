from __future__ import annotations

import json
import logging
import sys

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


_log_context: ContextVar[dict[str, Any] | None] = ContextVar('infra_log_context', default=None)
_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {'message', 'asctime'}

# 按小时滚动，默认保留 168 份（7 天）。
_DEFAULT_BACKUP_COUNT = 168


def get_log_context() -> dict[str, Any]:
    return dict(_log_context.get() or {})


@contextmanager
def bind_log_context(**fields: Any) -> Generator[None]:
    current = get_log_context()
    token = _log_context.set({**current, **{key: value for key, value in fields.items() if value is not None}})
    try:
        yield
    finally:
        _log_context.reset(token)


class JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec='milliseconds'),
            'level': record.levelname,
            'service': self.service_name,
            'logger': record.name,
            'message': record.getMessage(),
        }
        payload.update(get_log_context())
        payload.update(trace_context())
        extra = {
            key: _json_safe(value)
            for key, value in record.__dict__.items()
            if key not in _STANDARD_FIELDS and not key.startswith('_')
        }
        if extra:
            payload['extra'] = extra
        if record.exc_info:
            payload['exception'] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))


def configure_logging(
    service_name: str,
    *,
    level: str | int = logging.INFO,
    json_output: bool = True,
    log_file: str | Path | None = None,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> None:
    """配置根 logger：始终输出到 stdout，`log_file` 非空时额外滚动落盘。

    双写而非只落盘：保留 stdout 才能继续用 `kubectl logs`，
    文件则便于进容器 `tail -f` / `grep` 排查历史问题。

    按小时滚动，切分文件带 `.YYYY-MM-DD_HH` 后缀，便于按时间点定位问题；
    最多保留 `backup_count` 份，默认 168 份即 7 天。

    注意：按时间滚动不封顶单文件大小，突发流量下单小时的文件可能很大。
    磁盘紧张时应调小 `backup_count`，或在部署侧限制卷容量。

    落盘失败（目录不可写、磁盘满）不影响 stdout 输出与服务运行。
    """
    resolved_level = _log_level(level)
    formatter = JsonFormatter(service_name) if json_output else _text_formatter()

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(resolved_level)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = _build_file_handler(log_file, backup_count=backup_count)
        if file_handler is not None:
            file_handler.setFormatter(formatter)
            file_handler.setLevel(resolved_level)
            root.addHandler(file_handler)

    # uvicorn 自带独立 handler，会绕过 root 的 JSON formatter。清空其 handler 并开启
    # propagate，使访问/错误日志统一走上面的 formatter，保持日志格式一致。
    for logger_name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    for logger_name in ('httpcore', 'httpx'):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _build_file_handler(
    log_file: str | Path,
    *,
    backup_count: int,
) -> logging.Handler | None:
    """构建按小时滚动的文件 handler。失败时返回 None 并告警，不阻断服务启动。"""
    path = Path(log_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = TimedRotatingFileHandler(
            path,
            when='H',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8',
            delay=True,
            utc=False,
        )
        # 默认后缀为 .YYYY-MM-DD_HH-MM-SS，按小时滚动只需到小时
        handler.suffix = '%Y-%m-%d_%H'
        return handler
    except OSError:
        # 此时 stdout handler 已就绪，告警能被看到。
        logging.getLogger(__name__).warning('failed to open log file %s, file logging disabled', path, exc_info=True)
        return None


def _text_formatter() -> logging.Formatter:
    return logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def _log_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    resolved = logging.getLevelName(normalized)
    if not isinstance(resolved, int):
        raise ValueError(f'invalid log level: {level!r}')
    return resolved


def trace_context() -> dict[str, str]:
    """当前 OTel span 的 trace_id / span_id，无有效 span 时返回空 dict。"""
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
    except (ImportError, RuntimeError):
        return {}
    if not span_context.is_valid:
        return {}
    return {
        'trace_id': format(span_context.trace_id, '032x'),
        'span_id': format(span_context.span_id, '016x'),
    }


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
