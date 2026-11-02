from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import traceback

from datetime import datetime
from typing import Any

from nexus.core.request_admission import RequestAdmission, bind_request_admission
from nexus.distributed.artifacts.run_result import ResultPersistenceError, result_persistence_error
from nexus.distributed.execution.executors.base import (
    ExecutionHandle,
    ExecutionReporter,
    LiveExecutionDriver,
    StopDisposition,
)
from nexus.distributed.execution.trace_backends import (
    RUN_TRACE_LIVE_KEY,
    RUN_TRACE_RUNTIME_INFO_KEY,
    RunTraceBackendRegistry,
    RunTraceSinkRequest,
    make_live_result_pointer,
)
from nexus.runner.live.manager import LiveRun
from nexus.runner.live.models import CreateRunRequest, LiveConfig, SubmitTurnRequest
from nexus.runner.live.sse import sse_bytes
from nexus.runner.single import RunInput
from nexus.runner.trace import iter_run_events, principal_trace_scope_id


logger = logging.getLogger(__name__)


class NexusLiveDriver(LiveExecutionDriver):
    spec_type = 'nexus_live'
    stop_disposition = StopDisposition.IMMEDIATE

    def __init__(
        self,
        *,
        execution_id: str,
        assignment_id: str,
        internal_base_url: str,
        trace_registry: RunTraceBackendRegistry,
        attempt_id: str | None = None,
        owner_principal_id: str | None = None,
        artifact_scope_available: bool = True,
        result_publish_deadline_at: datetime | None = None,
        execution_created_at: datetime | None = None,
        request_admission: RequestAdmission | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.assignment_id = assignment_id
        self.attempt_id = attempt_id or assignment_id
        self.owner_principal_id = owner_principal_id
        # Live runs never have a batch parent, so the run's own created_at is the
        # partition key for the Boundless dialect.
        self.execution_created_at = execution_created_at
        self.internal_base_url = internal_base_url.rstrip('/')
        self.trace_registry = trace_registry
        self.artifact_scope_available = artifact_scope_available
        self.result_publish_deadline_at = result_publish_deadline_at
        self.request_admission = request_admission
        self._live_run: LiveRun | None = None

    async def start(
        self,
        handle: ExecutionHandle,
        spec: dict[str, Any],
        config: dict[str, Any],
        reporter: ExecutionReporter,
    ) -> None:
        try:
            request = self._build_request(spec, config)
            trace_metadata = self._trace_context_metadata()
        except Exception as exc:
            await reporter.fail({
                'type': 'InvalidExecutionSpec',
                'message': str(exc),
                'traceback': traceback.format_exc(),
            })
            return
        try:
            trace_backends = self.trace_registry.resolve(
                run_kind='live',
                driver_config=config,
                request=RunTraceSinkRequest(
                    execution_id=self.execution_id,
                    assignment_id=self.assignment_id,
                    attempt_id=self.attempt_id,
                    artifact_scope_available=self.artifact_scope_available,
                    result_publish_deadline_at=self.result_publish_deadline_at,
                ),
            )
        except Exception as exc:
            await reporter.fail({'type': 'InvalidDriverConfig', 'message': str(exc)})
            return
        live_run = LiveRun(
            request,
            trace_backend_sinks=trace_backends.backend_sinks,
            trace_metadata=trace_metadata,
        )
        self._live_run = live_run
        try:
            with bind_request_admission(
                self.request_admission,
                execution_id=self.execution_id,
                attempt_id=self.attempt_id,
            ):
                await live_run.start(wait_until_ready=True)
                await self._watch_events(live_run, reporter)
        except asyncio.CancelledError:
            try:
                await live_run.kill()
            except Exception:
                logger.exception('Failed to kill live run during driver cancellation')
            raise
        except Exception as exc:
            with contextlib.suppress(Exception):
                await live_run.kill()
            with contextlib.suppress(Exception):
                await self._wait_live_trace_closed(live_run)
            with contextlib.suppress(Exception):
                await self._publish_live_trace_metadata(reporter)
            persistence_error = result_persistence_error(exc)
            if persistence_error is not None:
                error = {
                    'type': ResultPersistenceError.error_type,
                    'message': str(persistence_error),
                }
            else:
                error = {
                    'type': type(exc).__name__,
                    'message': str(exc),
                    'traceback': traceback.format_exc(),
                }
            result = None
            with contextlib.suppress(Exception):
                result = self._result_payload(live_run, None)
            await reporter.fail(error, result=result)

    def _trace_context_metadata(self) -> dict[str, Any]:
        metadata = {
            'execution_id': self.execution_id,
            'assignment_id': self.assignment_id,
            'attempt_id': self.attempt_id,
        }
        if self.owner_principal_id is not None:
            metadata.update({
                'storage_scope': 'principal_v1',
                'principal_scope_id': principal_trace_scope_id(self.owner_principal_id),
            })
            if self.execution_created_at is not None:
                metadata['parent_created_at'] = self.execution_created_at.isoformat()
        return metadata

    async def submit_command(self, handle: ExecutionHandle, command_type: str, payload: dict[str, Any]) -> None:
        live_run = self._require_live_run()
        if command_type != 'input.submit':
            raise RuntimeError(f'nexus_live does not accept command {command_type}')
        with bind_request_admission(
            self.request_admission,
            execution_id=self.execution_id,
            attempt_id=self.attempt_id,
            cancel_on_exit=False,
        ):
            await live_run.submit_turn(SubmitTurnRequest(text=str(payload.get('text') or '')))

    async def stop(self, handle: ExecutionHandle) -> None:
        live_run = self._live_run
        if live_run is not None:
            await live_run.kill()

    async def collect_sse_events_after(self, cursor: int, *, timeout: float = 15.0) -> list[tuple[int, bytes]]:
        live_run = self._require_live_run()
        events = await live_run.events.wait_after(cursor, timeout=timeout)
        return [(event.id, sse_bytes(event)) for event in events]

    async def get_runtime_base_url(self, runtime_name: str) -> str:
        live_run = self._require_live_run()
        runtime = live_run.get_runtime(runtime_name)
        if not runtime.base_url:
            raise RuntimeError('runtime_not_ready')
        return runtime.base_url.rstrip('/')

    async def get_execution_info(self, handle: ExecutionHandle) -> dict[str, Any]:
        live_run = self._require_live_run()
        info = live_run.get_info()
        return {
            'assignment_id': handle.assignment_id,
            'execution_id': handle.execution_id,
            'attempt_id': self.attempt_id,
            'driver': {'spec_type': self.spec_type, 'instance_mode': 'asyncio'},
            'runtime_info': {
                'runtimes': {name: runtime.model_dump(mode='json') for name, runtime in info.runtimes.items()}
            },
            'live': {
                'status': info.status,
                'events_cursor': info.events_cursor,
                'current_turn_id': info.current_turn_id,
            },
        }

    def _build_request(self, spec: dict[str, Any], config: dict[str, Any]):
        raw = copy.deepcopy(spec.get('run_input', spec))
        live_config_raw = spec.get('live')
        run_input = RunInput.from_dict(raw)
        run_input.run_id = self.execution_id
        run_input.runner = run_input.runner.model_copy(
            update={'hold_signal_handlers': False, 'early_output_path': None}
        )
        runtime_configs = self._runtime_configs_with_aiomonitor(run_input.runtime_configs, config)
        live_config = LiveConfig.model_validate(live_config_raw) if live_config_raw else LiveConfig()
        return CreateRunRequest(
            run_id=run_input.run_id,
            imports=list(run_input.imports),
            openai_config_path=None,
            runtime_provider_name=run_input.runtime_provider_name,
            runtime_provider_config=dict(run_input.runtime_provider_config),
            runtime_configs=runtime_configs,
            openai_clients=dict(run_input.openai_clients),
            tool_configs=dict(run_input.tool_configs),
            task_name=run_input.task_name,
            task_config=dict(run_input.task_config),
            agent_name=run_input.agent_name,
            agent_config=dict(run_input.agent_config),
            processors=run_input.processors,
            runner=run_input.runner,
            live=live_config,
        )

    def _runtime_configs_with_aiomonitor(
        self,
        runtime_configs: dict[str, dict[str, Any]],
        config: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        runtime_configs = copy.deepcopy(runtime_configs)
        aiomonitor_config = config.get('runtime_aiomonitor') if isinstance(config, dict) else None
        if isinstance(aiomonitor_config, dict) and aiomonitor_config.get('enabled') is False:
            return runtime_configs
        for runtime_config in runtime_configs.values():
            environment = runtime_config.setdefault('environment', {})
            if not isinstance(environment, dict):
                continue
            environment.setdefault('NEXUS_RUNTIME_AIOMONITOR_ENABLED', '1')
            environment.setdefault('NEXUS_RUNTIME_AIOMONITOR_HOST', '127.0.0.1')
            environment.setdefault('NEXUS_RUNTIME_AIOMONITOR_BASE_PORT', '50100')
            environment.setdefault('NEXUS_RUNTIME_AIOMONITOR_WEBUI_BASE_PORT', '50200')
            environment.setdefault('NEXUS_RUNTIME_AIOMONITOR_CONSOLE_BASE_PORT', '50300')
        return runtime_configs

    async def _watch_events(self, live_run: LiveRun, reporter: Any) -> None:
        cursor = 0
        pending_result: dict[str, Any] | None = None
        pending_error: dict[str, Any] | None = None
        pending_killed = False
        while True:
            records = await live_run.trace_records.wait_after(cursor, timeout=15.0)
            if not records:
                if getattr(live_run, 'status', None) == 'failed':
                    await self._report_failed_live_run(live_run, reporter, pending_error)
                    return
                if pending_killed and getattr(live_run, 'status', None) == 'killed':
                    await self._wait_live_trace_closed(live_run, result_required=False)
                    await self._publish_live_trace_metadata(reporter)
                    await reporter.mark_killed()
                    return
                continue
            for record in records:
                cursor = record.record_id
                for event in iter_run_events(record, live_run.run_span_id):
                    await self._handle_run_event(event, reporter)
                    if event.type == 'run.output':
                        output = event.data.get('output', event.data)
                        pending_result = output if isinstance(output, dict) else {'output': output}
                        if pending_result.get('success') is False:
                            raw_error = pending_result.get('error')
                            pending_error = (
                                raw_error
                                if isinstance(raw_error, dict)
                                else {'type': 'RunFailed', 'message': 'live run failed'}
                            )
                    elif event.type == 'run.error':
                        pending_error = event.data
                    elif event.type == 'run.killed':
                        pending_killed = True

                    terminal = await self._maybe_finish_from_event(
                        event,
                        live_run,
                        reporter,
                        pending_result=pending_result,
                        pending_error=pending_error,
                    )
                    if terminal:
                        return

            if getattr(live_run, 'status', None) == 'failed':
                await self._report_failed_live_run(live_run, reporter, pending_error)
                return

    async def _handle_run_event(
        self,
        event: Any,
        reporter: ExecutionReporter,
    ) -> None:
        if event.type == 'runtime.ready':
            runtime_info = event.data.get('runtimes') or {}
            await reporter.mark_running(
                endpoint_url=self.internal_base_url,
                endpoint_ready=True,
                runtime_info=runtime_info,
            )
        elif event.type == 'input.required':
            await reporter.mark_waiting_input(event.data)
        elif event.type == 'run.status' and event.data.get('status') == 'running':
            await reporter.mark_running(endpoint_url=self.internal_base_url, endpoint_ready=True, runtime_info={})

    async def _maybe_finish_from_event(
        self,
        event: Any,
        live_run: LiveRun,
        reporter: ExecutionReporter,
        *,
        pending_result: dict[str, Any] | None,
        pending_error: dict[str, Any] | None,
    ) -> bool:
        if event.type == 'run.error':
            await self._wait_live_trace_closed(live_run)
            await self._publish_live_trace_metadata(reporter)
            result = self._result_payload(live_run, pending_result)
            await reporter.fail(pending_error or event.data, result=result)
            return True
        if event.type != 'run.status':
            return False
        status = event.data.get('status')
        if status == 'killed':
            await self._wait_live_trace_closed(live_run)
            await self._publish_live_trace_metadata(reporter)
            await reporter.mark_killed()
            return True
        if status not in {'closed', 'failed'}:
            return False
        await self._wait_live_trace_closed(live_run)
        await self._publish_live_trace_metadata(reporter)
        if status == 'failed' or pending_error is not None:
            await reporter.fail(
                pending_error
                or {
                    'type': 'RunFailed',
                    'message': event.data.get('error') or f'live run finished with status {status}',
                },
                result=self._result_payload(live_run, pending_result),
            )
        elif pending_result is None:
            await reporter.fail({'type': 'MissingRunOutput', 'message': 'live run closed without output'})
        else:
            result = self._result_payload(live_run, pending_result)
            if result is None:
                raise AssertionError('live result payload unexpectedly missing')
            await reporter.complete(result)
        return True

    def _require_live_run(self) -> LiveRun:
        if self._live_run is None:
            raise RuntimeError('live run is not started')
        return self._live_run

    async def _publish_live_trace_metadata(self, reporter: ExecutionReporter) -> None:
        live_run = self._live_run
        if live_run is None:
            return
        await reporter.update_runtime_info({
            RUN_TRACE_RUNTIME_INFO_KEY: {
                RUN_TRACE_LIVE_KEY: self._live_trace_envelope(live_run),
            }
        })

    async def _wait_live_trace_closed(self, live_run: LiveRun, *, result_required: bool = True) -> None:
        await live_run.wait_trace_closed()

    async def _report_failed_live_run(
        self,
        live_run: LiveRun,
        reporter: ExecutionReporter,
        error: dict[str, Any] | None,
    ) -> None:
        await self._wait_live_trace_closed(live_run, result_required=False)
        await self._publish_live_trace_metadata(reporter)
        resolved = error
        if resolved is None:
            # Fall back to the underlying failure recorded on the live run (e.g. the trace
            # drain / Socket.IO error) instead of the generic "live run failed", so the
            # execution result surfaces the real cause.
            live_error = getattr(live_run, 'error', None)
            if live_error:
                resolved = {'type': 'RunFailed', 'message': live_error}
        await reporter.fail(
            resolved or {'type': 'RunFailed', 'message': 'live run failed'},
            result=self._result_payload(live_run, None),
        )

    def _live_trace_envelope(self, live_run: LiveRun) -> dict[str, Any]:
        return live_run.trace_metadata()

    def _result_payload(self, live_run: LiveRun, result: dict[str, Any] | None) -> dict[str, Any] | None:
        if result is None:
            if self.artifact_scope_available:
                with contextlib.suppress(RuntimeError):
                    metadata = self._live_trace_envelope(live_run)
                    if any(
                        item.get('name') == 'artifact' and item.get('metadata') for item in metadata.get('sinks', [])
                    ):
                        return make_live_result_pointer(metadata)
            return None
        if not self.artifact_scope_available:
            return copy.deepcopy(result)
        return make_live_result_pointer(self._live_trace_envelope(live_run))
