"""Live interactive runner state manager."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import tempfile
import traceback
import uuid

from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nexus.core import ExecutionContext, UserInputRequest
from nexus.runner.live.models import (
    CreateRunRequest,
    CreateRunResponse,
    LiveConfig,
    RunInfo,
    RunStatus,
    RunSummary,
    RuntimeInfo,
    SubmitTurnRequest,
    SubmitTurnResponse,
    request_cache_key,
)
from nexus.runner.live.store import SQLiteLiveRunStore
from nexus.runner.live.trace import (
    LiveRunTraceView,
    project_live_trace_records,
)
from nexus.runner.single import RunHookError, RunHooks, RunInput, RunOutput
from nexus.runner.single import run as run_single
from nexus.runner.trace import (
    RunTraceContext,
    RunTraceInMemoryJournal,
    RunTraceLocalSink,
    RunTraceLocalSource,
    RunTraceProcessor,
    RunTraceSession,
    RunTraceSink,
    ensure_run_trace_processor,
)
from nexus.runtime import Runtime, RuntimeProvider
from nexus.runtime.trace_drain import drain_runtime_trace


logger = logging.getLogger(__name__)
_TERMINAL_RUN_STATUSES = {'closed', 'failed', 'killed'}


def _sink_metadata(envelope: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for item in envelope.get('sinks') or []:
        if isinstance(item, Mapping) and item.get('name') == name and isinstance(item.get('metadata'), Mapping):
            return item['metadata']
    return None


def _run_trace_context_from_metadata(envelope: Mapping[str, Any]) -> RunTraceContext:
    return RunTraceContext(
        run_id=str(envelope.get('run_id') or ''),
        run_span_id=str(envelope.get('run_span_id') or ''),
        run_kind=str(envelope.get('run_kind') or 'live'),
    )


class LiveRunError(Exception):
    """HTTP-friendly live-run error."""

    def __init__(self, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class _TurnSubmission:
    turn_id: str
    text: str | None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        old_value = merged.get(key)
        if isinstance(old_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(old_value, value)
        else:
            merged[key] = value
    return merged


def _resolve_openai_config_file(request: CreateRunRequest) -> CreateRunRequest:
    resolved = request.model_copy(deep=True)
    if not resolved.openai_config_path:
        return resolved
    config_path = Path(resolved.openai_config_path).expanduser()
    try:
        source = RunInput.from_file(config_path)
    except FileNotFoundError as exc:
        raise LiveRunError(f'openai config file not found: {config_path}', status=400) from exc
    except Exception as exc:
        raise LiveRunError(f'invalid openai config file {config_path}: {exc}', status=400) from exc
    if not source.openai_clients:
        raise LiveRunError(f'openai config file has no openai_clients: {config_path}', status=400)
    resolved.openai_clients = _deep_merge(source.openai_clients, resolved.openai_clients)
    resolved.openai_config_path = None
    return resolved


def _request_cache_key(request: CreateRunRequest) -> str | None:
    return request_cache_key(request)


def _require_request_cache_key(request: CreateRunRequest) -> str:
    try:
        cache_key = _request_cache_key(request)
    except ValueError as exc:
        raise LiveRunError(str(exc), status=400) from exc
    if cache_key is None:
        raise LiveRunError('live.cache.key_paths is required to create a cached live run', status=400)
    return cache_key


def _runtime_id_from_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    hostname = urlparse(base_url).hostname
    if not hostname or not hostname.startswith('nexus-'):
        return None
    return hostname.split('.', 1)[0]


class LiveRun:
    """One long-lived interactive single-run wrapper."""

    def __init__(
        self,
        request: CreateRunRequest,
        *,
        cache_on_first_idle: bool = False,
        on_cache_ready: Callable[[LiveRun], Awaitable[None] | None] | None = None,
        on_terminal: Callable[[LiveRun], Any | Awaitable[Any]] | None = None,
        trace_backend_sinks: Mapping[str, RunTraceSink] | None = None,
        trace_metadata: dict[str, Any] | None = None,
        run_trace_processor: RunTraceProcessor | None = None,
    ) -> None:
        self.request = request
        self.run_id = request.run_id or f'run_{uuid.uuid4().hex[:12]}'
        self.cache_key = _request_cache_key(request)
        self.live = self._resolve_live_config(request)
        self.status: RunStatus = 'starting'
        self.error: str | None = None
        self._cache_on_first_idle = cache_on_first_idle
        self._cache_ready_announced = False
        self._on_cache_ready = on_cache_ready
        self._on_terminal = on_terminal
        self._trace_processor = run_trace_processor or ensure_run_trace_processor()
        self._default_trace_root: Path | None = None
        if trace_backend_sinks is None:
            self._default_trace_root = Path(tempfile.mkdtemp(prefix='nexus-live-trace-'))
            trace_backend_sinks = {
                'local': RunTraceLocalSink(
                    root=self._default_trace_root,
                    execution_id=self.run_id,
                    assignment_id='local',
                    attempt_id='local',
                )
            }
        self._trace_records = RunTraceInMemoryJournal()
        self._live_trace_view = LiveRunTraceView()
        self._trace_session = RunTraceSession(
            run_id=self.run_id,
            run_kind='live',
            processor=self._trace_processor,
            backend_sinks=trace_backend_sinks,
            internal_sinks=[self._trace_records, self._live_trace_view],
            metadata=trace_metadata,
        )
        self.output: RunOutput | None = None

        self.ctx: ExecutionContext | None = None
        self.runtime_provider: RuntimeProvider | None = None
        self.runtime: Runtime | None = None
        self._runtime_event_unsubscribes: list[Any] = []
        self._input_queue: asyncio.Queue[_TurnSubmission] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._closed = False
        self._turn_index = 0
        self.current_turn_id: str | None = None
        self._started_event = asyncio.Event()
        self._terminal_task: asyncio.Task[None] | None = None
        self._terminal_notified = False
        self._terminal_trace_error: Exception | None = None
        self._run_output_handled = False
        self._accept_runtime_events = True
        self._runtime_events_inflight = 0
        self._runtime_events_drained = asyncio.Event()
        self._runtime_events_drained.set()

    @property
    def events(self) -> LiveRunTraceView:
        return self._live_trace_view

    @property
    def trace_records(self) -> RunTraceInMemoryJournal:
        return self._trace_records

    @property
    def run_span_id(self) -> str:
        return self._trace_session.run_span_id

    def trace_metadata(self) -> dict[str, Any]:
        try:
            return self._trace_session.artifact_metadata()
        except RuntimeError:
            result = self._trace_session.result_artifact_metadata()
            if result:
                return result
            raise

    @property
    def runtime_base_url(self) -> str | None:
        if self.runtime is None:
            return None
        return self.runtime.base_url

    @staticmethod
    def _resolve_live_config(request: CreateRunRequest) -> LiveConfig:
        return request.live.model_copy(deep=True)

    async def start(self, *, wait_until_ready: bool = True) -> None:
        """Start one background single.run invocation."""
        await self._trace_session.start()
        try:
            await self._emit('run.status', run_id=self.run_id, status='starting')
            self._load_openai_config_file()
            run_input = self._build_run_input()
        except BaseException:
            with contextlib.suppress(Exception):
                await self._trace_session.close()
            raise

        self._task = asyncio.create_task(self._run_single(run_input), name=f'nexus-live-run-{self.run_id}')
        if wait_until_ready:
            await self._started_event.wait()

    def _load_openai_config_file(self) -> None:
        self.request = _resolve_openai_config_file(self.request)
        self.cache_key = _request_cache_key(self.request)

    def _build_run_input(self) -> RunInput:
        return RunInput(
            imports=list(self.request.imports),
            task_name=self.request.task_name,
            task_config=dict(self.request.task_config),
            tool_configs=dict(self.request.tool_configs),
            run_id=self.run_id,
            runtime_provider_name=self.request.runtime_provider_name,
            runtime_provider_config=dict(self.request.runtime_provider_config),
            runtime_configs=dict(self.request.runtime_configs),
            agent_name=self.request.agent_name,
            agent_config=dict(self.request.agent_config),
            openai_clients=dict(self.request.openai_clients),
            processors=self.request.processors,
            gateway_trajectory_observers=self.request.gateway_trajectory_observers,
            runner=self.request.runner,
            workflow_states=self.request.workflow_states,
        )

    @staticmethod
    def create_request_from_run_input(run_input: RunInput, *, live: LiveConfig | None = None) -> CreateRunRequest:
        """Convert a normal RunInput into the live API request shape."""
        return CreateRunRequest(
            run_id=run_input.run_id,
            imports=list(run_input.imports),
            openai_config_path=None,
            runtime_provider_name=run_input.runtime_provider_name,
            runtime_provider_config=dict(run_input.runtime_provider_config),
            runtime_configs=dict(run_input.runtime_configs),
            openai_clients=dict(run_input.openai_clients),
            tool_configs=dict(run_input.tool_configs),
            task_name=run_input.task_name,
            task_config=dict(run_input.task_config),
            agent_name=run_input.agent_name,
            agent_config=dict(run_input.agent_config),
            processors=run_input.processors,
            gateway_trajectory_observers=run_input.gateway_trajectory_observers,
            runner=run_input.runner,
            workflow_states=run_input.workflow_states,
            live=live or LiveConfig(),
        )

    async def _set_status(self, status: RunStatus, **extra: Any) -> None:
        if status in _TERMINAL_RUN_STATUSES:
            self._accept_runtime_events = False
            await self._unsubscribe_runtime_events()
            await self._runtime_events_drained.wait()
        self.status = status
        await self._emit('run.status', run_id=self.run_id, status=status, **extra)
        if status in _TERMINAL_RUN_STATUSES:
            await asyncio.shield(self._ensure_terminal_task())

    def _ensure_terminal_task(self) -> asyncio.Task[None]:
        if self._terminal_task is None:
            self._terminal_task = asyncio.create_task(
                self._finalize_terminal_status(),
                name=f'nexus-live-terminal-{self.run_id}',
            )
        return self._terminal_task

    async def _finalize_terminal_status(self) -> None:
        try:
            await self._trace_session.publish_result(
                status=self.status,
                error={'type': 'RunFailed', 'message': self.error} if self.error else None,
            )
            await self._trace_session.close()
        except Exception as exc:
            self._terminal_trace_error = exc
            self.status = 'failed'
            self.error = f'{type(exc).__name__}: {exc}'
            await self._notify_terminal()
            raise
        await self._notify_terminal()

    async def _notify_terminal(self) -> None:
        if self._terminal_notified:
            return
        self._terminal_notified = True
        if self._on_terminal is not None:
            result = self._on_terminal(self)
            if inspect.isawaitable(result):
                await result

    async def wait_trace_closed(self) -> None:
        if self._terminal_task is not None or self.status in _TERMINAL_RUN_STATUSES:
            await asyncio.shield(self._ensure_terminal_task())
            return
        await self._trace_session.close()

    async def _emit(self, event_type: str, **data: Any) -> None:
        await self._trace_session.emit(event_type, **data)

    async def submit_turn(self, request: SubmitTurnRequest) -> SubmitTurnResponse:
        """Submit the next user message while the run is idle."""
        if self._closed or self.status in ('killing', 'closed', 'failed', 'killed'):
            raise LiveRunError(f'run {self.run_id} is not accepting turns: {self.status}', status=409)
        if self.status != 'idle':
            raise LiveRunError(f'run {self.run_id} is not idle: {self.status}', status=409)

        self._turn_index += 1
        turn_id = f'turn_{self._turn_index}'
        submission = _TurnSubmission(turn_id=turn_id, text=request.text)
        await self._emit('turn.submitted', run_id=self.run_id, turn_id=turn_id, text=request.text)
        await self._input_queue.put(submission)
        return SubmitTurnResponse(run_id=self.run_id, turn_id=turn_id, status=self.status)

    async def _run_single(self, run_input: RunInput) -> None:
        hooks = RunHooks(
            agent_input_overrides={'get_user_input': self._get_user_input},
            on_context_ready=self._handle_context_ready,
            on_run_output=self._handle_run_output,
            on_context_closing=self._handle_context_closing,
        )
        try:
            with self._trace_session.activate():
                output = await run_single(run_input, hooks=hooks)
            hooks.raise_if_failed()
            if not self._run_output_handled:
                await self._handle_run_output(self.ctx, output)
            final_output = self.output or output
            self._closed = True
            await self._set_status('closed' if final_output.success else 'failed')
        except asyncio.CancelledError:
            raise
        except RunHookError as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            logger.error('Live run %s trace hook failed: %s', self.run_id, exc)
            await self._finalize_trace_pipeline_failure(exc)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            logger.exception('Live run %s failed', self.run_id)
            if self._terminal_trace_error is not None:
                return
            try:
                await self._emit(
                    'run.error',
                    run_id=self.run_id,
                    error=self.error,
                    traceback=traceback.format_exc(),
                )
                self._closed = True
                await self._set_status('failed')
            except Exception as trace_error:
                logger.error('Live run %s trace pipeline failed: %s', self.run_id, trace_error)
                await self._finalize_trace_pipeline_failure(trace_error)
        finally:
            self._started_event.set()
            await self._unsubscribe_runtime_events()

    async def _finalize_trace_pipeline_failure(self, error: Exception) -> None:
        self._closed = True
        self.status = 'failed'
        self._accept_runtime_events = False
        if self._terminal_trace_error is None:
            self._terminal_trace_error = error
        await self._unsubscribe_runtime_events()
        await self._runtime_events_drained.wait()
        # Best-effort: surface the real failure cause as a terminal `run.status` event before
        # closing the trace. This lets the result artifact record `failed` with the underlying
        # error and lets the driver report the true cause, instead of the opaque
        # "run trace closed without terminal run.status" -> ResultPersistenceFailed.
        # If the trace pipeline itself is already broken (e.g. emit raises), we swallow the
        # error and fall back to closing the trace without a terminal status as before.
        with contextlib.suppress(Exception):
            await self._emit(
                'run.status',
                run_id=self.run_id,
                status='failed',
                error=self.error or f'{type(error).__name__}: {error}',
                traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)),
            )
        terminal_task = self._ensure_terminal_task()
        with contextlib.suppress(Exception):
            await asyncio.shield(terminal_task)

    async def _handle_context_ready(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx
        self.runtime_provider = ctx.runtime_provider
        if ctx.runtime_provider is not None:
            self.runtime = await ctx.get_runtime()
            with contextlib.suppress(Exception):
                await self.runtime.ws_connect()
            self._subscribe_runtime_events(self.runtime)
        await self._emit(
            'runtime.ready',
            run_id=self.run_id,
            runtimes={name: info.model_dump(mode='json') for name, info in self._runtime_infos().items()},
        )
        if self.status == 'starting':
            if self.current_turn_id:
                await self._set_status('running', turn_id=self.current_turn_id)
                await self._emit('turn.started', run_id=self.run_id, turn_id=self.current_turn_id)
            else:
                await self._set_status('running')
        self._started_event.set()

    def _subscribe_runtime_events(self, runtime: Runtime) -> None:
        for event_name, handler in (('agent_event', self._handle_runtime_agent_event),):
            with contextlib.suppress(Exception):
                self._runtime_event_unsubscribes.append(runtime.on(event_name, handler))

    async def _unsubscribe_runtime_events(self) -> None:
        while self._runtime_event_unsubscribes:
            unsubscribe = self._runtime_event_unsubscribes.pop()
            try:
                unsubscribe()
            except Exception:
                logger.debug('Failed to unsubscribe runtime event listener', exc_info=True)

    async def _get_user_input(self, request: UserInputRequest | str | None) -> str | None:
        request_payload: dict[str, Any] | None
        if isinstance(request, UserInputRequest):
            request_payload = request.model_dump(mode='json')
        elif isinstance(request, str):
            request_payload = {'hint': request}
        else:
            request_payload = None

        await self._emit(
            'input.required',
            run_id=self.run_id,
            turn_id=self.current_turn_id,
            request=request_payload,
        )
        if self._cache_on_first_idle and not self._cache_ready_announced:
            self._cache_ready_announced = True
            await self._set_status('cached')
            if self._on_cache_ready is not None:
                result = self._on_cache_ready(self)
                if inspect.isawaitable(result):
                    await result
        else:
            await self._set_status('idle')

        submission = await self._input_queue.get()
        if submission.text is None or not submission.text.strip():
            return None
        await self._start_turn(submission.turn_id)
        return submission.text

    async def _start_turn(self, turn_id: str) -> None:
        self.current_turn_id = turn_id
        await self._set_status('running', turn_id=turn_id)
        await self._emit('turn.started', run_id=self.run_id, turn_id=turn_id)

    async def _handle_run_output(self, _ctx: ExecutionContext | None, output: RunOutput) -> None:
        if self._run_output_handled:
            return
        output_payload = await asyncio.to_thread(output.model_dump, mode='json')
        self._trace_session.capture_output(output_payload)
        await self._drain_runtime_traces(_ctx)
        self._run_output_handled = True
        self.output = output
        await self._emit(
            'run.output',
            run_id=self.run_id,
            success=output.success,
            output=output_payload,
        )

    async def _handle_context_closing(self, ctx: ExecutionContext) -> None:
        await self._drain_runtime_traces(ctx)

    async def _drain_runtime_traces(self, ctx: ExecutionContext | None = None) -> None:
        runtimes = self._active_runtimes()
        if ctx is not None and ctx.runtime_provider is not None:
            runtimes.update(ctx.runtime_provider.get_active_runtimes())
        drained_runtime_ids: set[int] = set()
        for runtime_name, runtime in runtimes.items():
            runtime_key = id(runtime)
            if runtime_key in drained_runtime_ids:
                continue
            drain = getattr(runtime, 'drain_trace', None)
            if drain is not None:
                await drain_runtime_trace(
                    runtime,
                    error_policy=self.request.runner.trace_drain_error_policy,
                    run_id=self.run_id,
                    runtime_name=runtime_name,
                )
            drained_runtime_ids.add(runtime_key)

    def get_runtime(self, name: str) -> Runtime:
        runtimes = self._active_runtimes()
        runtime = runtimes.get(name)
        if runtime is None:
            raise LiveRunError(f'run {self.run_id} has no runtime named {name!r}: {self.status}', status=404)
        if not getattr(runtime, 'base_url', None):
            raise LiveRunError(f'run {self.run_id} runtime {name!r} is not proxyable', status=409)
        return runtime

    async def kill(self) -> None:
        """Forcefully kill the run and its single.run task."""
        if self._closed or self.status in ('closed', 'failed', 'killed'):
            self._closed = True
            task = self._task
            if task is not None and task is not asyncio.current_task() and not task.done():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(task)
            with contextlib.suppress(Exception):
                await self.wait_trace_closed()
            return
        self._closed = True
        await self._set_status('killing')
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.status in _TERMINAL_RUN_STATUSES:
            return
        await self._unsubscribe_runtime_events()
        await self._emit('run.killed', run_id=self.run_id)
        await self._set_status('killed')

    async def _handle_runtime_agent_event(self, data: dict[str, Any]) -> None:
        if not self._accept_runtime_events:
            return
        self._runtime_events_inflight += 1
        self._runtime_events_drained.clear()
        payload = dict(data)
        payload.setdefault('source', 'agent_event')
        try:
            await self._emit('agent.event', run_id=self.run_id, event=payload)
        finally:
            self._runtime_events_inflight -= 1
            if self._runtime_events_inflight == 0:
                self._runtime_events_drained.set()

    def _active_runtimes(self) -> dict[str, Runtime]:
        runtimes: dict[str, Runtime] = {}
        if self.runtime_provider is not None:
            try:
                runtimes.update(self.runtime_provider.get_active_runtimes())
            except Exception:
                logger.debug('Failed to inspect active runtimes', exc_info=True)
        if self.runtime is not None:
            runtimes.setdefault('main', self.runtime)
        return runtimes

    def _runtime_infos(self) -> dict[str, RuntimeInfo]:
        runtimes = self._active_runtimes()
        return {
            name: RuntimeInfo(
                name=name,
                runtime_id=self._runtime_id(name, runtime),
                base_url=getattr(runtime, 'base_url', None),
                proxy_url=f'/v1/runs/{self.run_id}/runtimes/{name}/proxy',
                debug_gui_url=f'/v1/runs/{self.run_id}/runtimes/{name}/proxy/debug/gui',
            )
            for name, runtime in runtimes.items()
        }

    def _runtime_id(self, name: str, runtime: Runtime) -> str | None:
        runtime_id = getattr(runtime, 'runtime_id', None) or getattr(runtime, 'id', None)
        if runtime_id:
            return str(runtime_id)
        runtime_ids = getattr(self.runtime_provider, '_runtime_ids', None)
        if isinstance(runtime_ids, dict):
            runtime_id = runtime_ids.get(name)
            if runtime_id:
                return str(runtime_id)
        return _runtime_id_from_base_url(getattr(runtime, 'base_url', None))

    def get_info(self) -> RunInfo:
        return RunInfo(
            run_id=self.run_id,
            status=self.status,
            events_cursor=self.events.latest_id,
            current_turn_id=self.current_turn_id,
            cache_key=self.cache_key,
            runtime_base_url=self.runtime_base_url,
            runtimes=self._runtime_infos(),
            error=self.error,
            output=self.output.model_dump(mode='json') if self.output is not None else None,
        )

    def get_summary(self) -> RunSummary:
        return RunSummary(
            run_id=self.run_id,
            status=self.status,
            events_cursor=self.events.latest_id,
            current_turn_id=self.current_turn_id,
            cache_key=self.cache_key,
            runtime_base_url=self.runtime_base_url,
            error=self.error,
        )


class LiveRunManager:
    """Manage in-memory live runs for one service process."""

    def __init__(
        self,
        *,
        store: SQLiteLiveRunStore | None = None,
        storage_path: str | Path | None = None,
        trace_storage_path: str | Path | None = None,
        terminal_run_retention_seconds: float | None = 60.0,
    ) -> None:
        if store is not None and storage_path is not None:
            raise ValueError('pass either store or storage_path, not both')
        self.runs: dict[str, LiveRun] = {}
        self.store = store or (SQLiteLiveRunStore(storage_path) if storage_path is not None else None)
        if trace_storage_path is not None and self.store is None:
            raise ValueError('trace_storage_path requires live run storage')
        store_path = self.store.path if self.store is not None else None
        self.trace_storage_path = (
            Path(trace_storage_path)
            if trace_storage_path is not None
            else store_path.parent / f'{store_path.stem}-traces'
            if store_path is not None
            else None
        )
        self._trace_source = (
            RunTraceLocalSource(root=self.trace_storage_path) if self.trace_storage_path is not None else None
        )
        self._cache_runs: dict[str, deque[str]] = defaultdict(deque)
        self._terminal_run_retention_seconds = terminal_run_retention_seconds
        self._terminal_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.store is not None:
            await self.store.start()

    async def create_run(self, request: CreateRunRequest) -> CreateRunResponse:
        request = _resolve_openai_config_file(request)
        try:
            cache_key = _request_cache_key(request)
        except ValueError as exc:
            raise LiveRunError(str(exc), status=400) from exc
        async with self._lock:
            if request.run_id is None and cache_key is not None:
                cached = self._pop_cached_run(cache_key)
                if cached is not None:
                    await cached._set_status('idle')
                    return CreateRunResponse(
                        run_id=cached.run_id,
                        status=cached.status,
                        events_cursor=cached.events.latest_id,
                        reused_cache=True,
                        cache_key=cached.cache_key,
                    )
            run = await self._register_new_run(request, cache_on_first_idle=False)
        try:
            await run.start(wait_until_ready=False)
        except Exception:
            await self._delete_unstarted_run(run)
            raise
        return CreateRunResponse(
            run_id=run.run_id,
            status=run.status,
            events_cursor=run.events.latest_id,
            cache_key=run.cache_key,
        )

    async def create_cache_run(self, request: CreateRunRequest) -> CreateRunResponse:
        request = _resolve_openai_config_file(request)
        _require_request_cache_key(request)
        async with self._lock:
            run = await self._register_new_run(
                request,
                cache_on_first_idle=True,
                on_cache_ready=self._register_cached_run,
            )
        try:
            await run.start(wait_until_ready=False)
        except Exception:
            await self._delete_unstarted_run(run)
            raise
        return CreateRunResponse(
            run_id=run.run_id,
            status=run.status,
            events_cursor=run.events.latest_id,
            cache_key=run.cache_key,
        )

    async def _register_new_run(
        self,
        request: CreateRunRequest,
        *,
        cache_on_first_idle: bool,
        on_cache_ready: Callable[[LiveRun], Awaitable[None] | None] | None = None,
    ) -> LiveRun:
        run_id = request.run_id or f'run_{uuid.uuid4().hex[:12]}'
        if run_id in self.runs:
            raise LiveRunError(f'run already exists: {run_id}', status=409)
        if self.store is not None and await self.store.has_run(run_id):
            raise LiveRunError(f'run already exists: {run_id}', status=409)
        if request.run_id != run_id:
            request = request.model_copy(update={'run_id': run_id})
        run = LiveRun(
            request,
            cache_on_first_idle=cache_on_first_idle,
            on_cache_ready=on_cache_ready,
            on_terminal=self._handle_terminal_run,
            trace_backend_sinks=self._local_trace_backends(run_id),
            trace_metadata={'storage': 'local-live'} if self.trace_storage_path is not None else None,
        )
        self.runs[run.run_id] = run
        if self.store is not None:
            await self.store.create_run(run.run_id, run.request)
        return run

    async def _delete_unstarted_run(self, run: LiveRun) -> None:
        async with self._lock:
            self.runs.pop(run.run_id, None)
            self._remove_cached_run(run)
            if self.store is not None:
                await self.store.delete_run(run.run_id)

    def _local_trace_backends(self, run_id: str) -> dict[str, RunTraceSink]:
        if self.trace_storage_path is None:
            return {
                'local': RunTraceLocalSink(
                    root=Path(tempfile.mkdtemp(prefix='nexus-live-trace-')),
                    execution_id=run_id,
                    assignment_id='local',
                    attempt_id='local',
                )
            }
        sink = RunTraceLocalSink(
            root=self.trace_storage_path,
            execution_id=run_id,
            assignment_id='local',
            attempt_id='local',
        )
        return {'local': sink}

    async def _register_cached_run(self, run: LiveRun) -> None:
        if run.cache_key is None:
            return
        async with self._lock:
            if run.run_id in self.runs and run.status == 'cached':
                queue = self._cache_runs[run.cache_key]
                if run.run_id not in queue:
                    queue.append(run.run_id)

    def _pop_cached_run(self, cache_key: str) -> LiveRun | None:
        queue = self._cache_runs.get(cache_key)
        while queue:
            run_id = queue.popleft()
            run = self.runs.get(run_id)
            if run is not None and run.status == 'cached':
                return run
        for run in self.runs.values():
            if run.cache_key == cache_key and run.status == 'cached':
                self._remove_cached_run(run)
                return run
        return None

    def _remove_cached_run(self, run: LiveRun) -> None:
        if run.cache_key is None:
            return
        queue = self._cache_runs.get(run.cache_key)
        if not queue:
            return
        self._cache_runs[run.cache_key] = deque(run_id for run_id in queue if run_id != run.run_id)

    async def _handle_terminal_run(self, run: LiveRun) -> None:
        if run.status not in _TERMINAL_RUN_STATUSES:
            return
        if self.store is not None:
            try:
                await self.store.save_run_info(run.get_info(), trace_metadata=run.trace_metadata())
            except Exception:
                logger.exception('Failed to persist terminal live run metadata: run_id=%s', run.run_id)
                return
        self._schedule_terminal_cleanup(run)

    def _schedule_terminal_cleanup(self, run: LiveRun) -> None:
        if self.store is None:
            return
        delay = self._terminal_run_retention_seconds
        if delay is None or delay < 0 or run.status not in _TERMINAL_RUN_STATUSES:
            return
        old_task = self._terminal_cleanup_tasks.pop(run.run_id, None)
        if old_task is not None:
            old_task.cancel()
        task = asyncio.create_task(
            self._cleanup_terminal_run_after_delay(run.run_id, run, delay),
            name=f'nexus-live-cleanup-{run.run_id}',
        )
        self._terminal_cleanup_tasks[run.run_id] = task

    async def _cleanup_terminal_run_after_delay(self, run_id: str, run: LiveRun, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            async with self._lock:
                current = self.runs.get(run_id)
                if current is not run or current.status not in _TERMINAL_RUN_STATUSES:
                    return
                self._remove_cached_run(current)
                self.runs.pop(run_id, None)
        except asyncio.CancelledError:
            raise
        finally:
            current_task = asyncio.current_task()
            async with self._lock:
                if self._terminal_cleanup_tasks.get(run_id) is current_task:
                    self._terminal_cleanup_tasks.pop(run_id, None)

    def get_run(self, run_id: str) -> LiveRun:
        run = self.runs.get(run_id)
        if run is None:
            raise LiveRunError(f'run not found: {run_id}', status=404)
        return run

    async def get_run_events_after(self, run_id: str, cursor: int, *, limit: int | None = None) -> list[Any]:
        run = self.runs.get(run_id)
        if run is not None:
            return run.events.snapshot_after(cursor, limit=limit)
        if self.store is not None and await self.store.has_run(run_id):
            metadata = await self.store.get_trace_metadata(run_id)
            if metadata is None or self._trace_source is None:
                return []
            sink_metadata = _sink_metadata(metadata, 'local')
            if sink_metadata is None:
                return []
            records = await self._trace_source.read_events_after(sink_metadata, 0)
            context = _run_trace_context_from_metadata(metadata)
            return project_live_trace_records(context, records, cursor=cursor, limit=limit)
        raise LiveRunError(f'run not found: {run_id}', status=404)

    async def get_run_info(self, run_id: str) -> RunInfo:
        run = self.runs.get(run_id)
        if run is not None:
            return run.get_info()
        if self.store is not None:
            info = await self.store.get_run_info(run_id)
            if info is not None:
                return info
        raise LiveRunError(f'run not found: {run_id}', status=404)

    async def get_run_request(self, run_id: str) -> dict[str, Any]:
        run = self.runs.get(run_id)
        if run is not None:
            return run.request.model_dump(mode='json')
        if self.store is not None:
            request = await self.store.get_run_request(run_id)
            if request is not None:
                return request
        raise LiveRunError(f'run not found: {run_id}', status=404)

    async def list_run_infos(
        self,
        *,
        limit: int = 100,
        statuses: Sequence[str] | None = None,
        cache_key: str | None = None,
        query: str | None = None,
    ) -> list[RunInfo]:
        status_values = set(statuses or [])
        if self.store is not None:
            infos = await self.store.list_run_infos(limit=limit, statuses=statuses, cache_key=cache_key, query=query)
        else:
            infos = []
        infos_by_id = {info.run_id: info for info in infos}
        for run in self.runs.values():
            info = run.get_info()
            if status_values and info.status not in status_values:
                continue
            if cache_key and info.cache_key != cache_key:
                continue
            if query:
                lowered_query = query.lower()
                if lowered_query not in info.run_id.lower() and (
                    info.cache_key is None or lowered_query not in info.cache_key.lower()
                ):
                    continue
            infos_by_id[info.run_id] = info
        infos = list(infos_by_id.values())
        if status_values:
            infos = [info for info in infos if info.status in status_values]
        if cache_key:
            infos = [info for info in infos if info.cache_key == cache_key]
        if query:
            lowered_query = query.lower()
            infos = [
                info
                for info in infos
                if lowered_query in info.run_id.lower()
                or (info.cache_key is not None and lowered_query in info.cache_key.lower())
            ]
        return infos[:limit]

    async def list_run_summaries(
        self,
        *,
        limit: int = 100,
        statuses: Sequence[str] | None = None,
        cache_key: str | None = None,
        query: str | None = None,
    ) -> list[RunSummary]:
        status_values = set(statuses or [])
        if self.store is not None:
            summaries = await self.store.list_run_summaries(
                limit=limit, statuses=statuses, cache_key=cache_key, query=query
            )
        else:
            summaries = []
        summaries_by_id = {summary.run_id: summary for summary in summaries}
        for run in self.runs.values():
            summary = run.get_summary()
            if status_values and summary.status not in status_values:
                continue
            if cache_key and summary.cache_key != cache_key:
                continue
            if query:
                lowered_query = query.lower()
                if lowered_query not in summary.run_id.lower() and (
                    summary.cache_key is None or lowered_query not in summary.cache_key.lower()
                ):
                    continue
            summaries_by_id[summary.run_id] = summary
        summaries = list(summaries_by_id.values())
        if status_values:
            summaries = [summary for summary in summaries if summary.status in status_values]
        if cache_key:
            summaries = [summary for summary in summaries if summary.cache_key == cache_key]
        if query:
            lowered_query = query.lower()
            summaries = [
                summary
                for summary in summaries
                if lowered_query in summary.run_id.lower()
                or (summary.cache_key is not None and lowered_query in summary.cache_key.lower())
            ]
        return summaries[:limit]

    async def kill_run(self, run_id: str) -> RunInfo:
        run = self.get_run(run_id)
        await run.kill()
        self._remove_cached_run(run)
        return run.get_info()

    async def close_all(self) -> None:
        await asyncio.gather(*(run.kill() for run in list(self.runs.values())), return_exceptions=True)

    async def close(self) -> None:
        await self.close_all()
        cleanup_tasks = list(self._terminal_cleanup_tasks.values())
        for task in cleanup_tasks:
            task.cancel()
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self._terminal_cleanup_tasks.clear()
        if self.store is not None:
            await self.store.close()
