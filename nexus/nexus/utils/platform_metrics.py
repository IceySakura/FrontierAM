"""Small, optional platform measurements at operation boundaries."""

from __future__ import annotations

import asyncio
import time

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar


class Recorder(Protocol):
    def record(self, name: str, value: float = 1, attributes: Mapping[str, str] | None = None) -> None: ...


_recorder: ContextVar[Recorder | None] = ContextVar('nexus_platform_metrics', default=None)
_attributes: ContextVar[Mapping[str, str] | None] = ContextVar('nexus_platform_metric_attributes', default=None)
_process_recorder: Recorder | None = None


def set_process_recorder(recorder: Recorder | None) -> None:
    global _process_recorder
    _process_recorder = recorder


def process_recorder() -> Recorder | None:
    return _process_recorder


def current_recorder() -> Recorder | None:
    return _recorder.get() or _process_recorder


def bind_metric_attributes(**attributes: str) -> None:
    if current_recorder() is not None:
        _attributes.set({**(_attributes.get() or {}), **attributes})


def current_metric_attributes() -> dict[str, str]:
    return dict(_attributes.get() or {})


@contextmanager
def metric_scope(recorder: Recorder | None = None, **attributes: str) -> Iterator[None]:
    recorder_token = _recorder.set(recorder or current_recorder())
    attributes_token = _attributes.set({**(_attributes.get() or {}), **attributes})
    try:
        yield
    finally:
        _attributes.reset(attributes_token)
        _recorder.reset(recorder_token)


def record(name: str, value: float = 1, /, **attributes: str) -> None:
    recorder = current_recorder()
    if recorder is not None:
        # Monitoring must not replace the outcome of the operation being measured.
        with suppress(Exception):
            recorder.record(name, value, {**(_attributes.get() or {}), **attributes})


@dataclass
class OperationResult:
    outcome: str = 'success'
    reason: str = 'none'


@contextmanager
def measure_operation(
    stage: str, *, operation: str = '', family: str = 'stage', timeout_seconds: float | None = None, **attributes: str
) -> Iterator[OperationResult]:
    started = time.perf_counter()
    result = OperationResult()
    finish_active = None
    recorder = current_recorder()
    if recorder is not None:
        with suppress(Exception):
            begin = getattr(recorder, 'begin_operation', None)
            if begin is not None:
                finish_active = begin(
                    stage,
                    timeout_seconds=timeout_seconds,
                    attributes={**(_attributes.get() or {}), **attributes, 'operation': operation or stage},
                )
    try:
        yield result
    except asyncio.CancelledError:
        result.outcome = 'cancelled'
        raise
    except Exception as exc:
        if result.outcome == 'success':
            result.outcome = (
                'timeout' if isinstance(exc, TimeoutError) or getattr(exc, 'status', None) == 504 else 'error'
            )
        if result.reason == 'none':
            result.reason = type(exc).__name__
        if isinstance(exc, ValueError) and str(exc) in {
            'artifact_identity_mismatch',
            'artifact_remote_sha256_mismatch',
            'artifact_remote_size_mismatch',
            'artifact_object_uri_mismatch',
        }:
            record('data_integrity_errors', source='artifact_store', reason=str(exc))
        raise
    finally:
        if finish_active is not None:
            with suppress(Exception):
                finish_active()
        dimensions = dict(
            attributes, stage=stage, operation=operation or stage, outcome=result.outcome, reason=result.reason
        )
        record(f'{family}_operations', **dimensions)
        record(
            f'{family}_failures',
            int(result.outcome not in {'success', 'cancelled', 'expected_unavailable'}),
            **dimensions,
        )
        duration_metric = 'terminal_report_distribution' if family == 'terminal' else f'{family}_duration_distribution'
        record(duration_metric, max(0, time.perf_counter() - started), **dimensions)


_P = ParamSpec('_P')
_T = TypeVar('_T')


def measured(stage: str, *, family: str = 'stage', timeout_seconds: float | None = None, **attributes: str):
    def decorate(function: Callable[_P, Awaitable[_T]]) -> Callable[_P, Awaitable[_T]]:
        @wraps(function)
        async def invoke(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            with measure_operation(stage, family=family, timeout_seconds=timeout_seconds, **attributes):
                return await function(*args, **kwargs)

        return invoke

    return decorate


@contextmanager
def measure_database(operation: str, *, source: str, acquire: bool = False):
    started = time.perf_counter()
    outcome = 'success'
    try:
        yield
    except BaseException as exc:
        outcome = (
            'cancelled'
            if isinstance(exc, asyncio.CancelledError)
            else 'timeout'
            if isinstance(exc, TimeoutError)
            else 'error'
        )
        record('database_errors', source=source, operation=operation, outcome=outcome)
        raise
    finally:
        metric = 'database_acquisition_distribution' if acquire else 'database_sql_duration_distribution'
        record('database_operations', source=source, operation=operation, outcome=outcome)
        if outcome == 'success':
            record('database_errors', 0, source=source, operation=operation, outcome=outcome)
        record(metric, max(0, time.perf_counter() - started), source=source, operation=operation, outcome=outcome)
