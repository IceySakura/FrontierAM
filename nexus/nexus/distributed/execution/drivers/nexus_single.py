from __future__ import annotations

import asyncio
import contextlib
import copy
import traceback

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from nexus.core.request_admission import RequestAdmission, bind_request_admission
from nexus.dataset.access import DataAccessContext, bind_data_access
from nexus.distributed.artifacts.run_result import ResultPersistenceError, result_persistence_error
from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    ExecutionHandle,
    ExecutionReporter,
    StopDisposition,
)
from nexus.distributed.execution.trace_backends import (
    RUN_TRACE_RUNTIME_INFO_KEY,
    RUN_TRACE_SINGLE_KEY,
    RunTraceBackendRegistry,
    RunTraceSinkRequest,
    make_run_result_pointer,
)
from nexus.distributed.trace.bindings import make_source_binding
from nexus.runner.single import RunHooks, RunInput, RunOutput
from nexus.runner.single import run as run_single
from nexus.runner.trace import (
    RunTraceSession,
    RunTraceSinkError,
    ensure_run_trace_processor,
    principal_trace_scope_id,
)
from nexus.runtime.trace_drain import drain_runtime_trace


class NexusSingleDriver(ExecutionDriver):
    spec_type = 'nexus_single'
    stop_disposition = StopDisposition.IMMEDIATE

    def __init__(
        self,
        *,
        execution_id: str,
        assignment_id: str,
        trace_registry: RunTraceBackendRegistry,
        attempt_id: str | None = None,
        owner_principal_id: str | None = None,
        parent_execution_id: str | None = None,
        internal_base_url: str | None = None,
        artifact_scope_available: bool = True,
        result_publish_deadline_at: datetime | None = None,
        execution_created_at: datetime | None = None,
        data_credentials_loader: Callable[[], Awaitable[dict[str, Any]]] | None = None,
        request_admission: RequestAdmission | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.assignment_id = assignment_id
        self.attempt_id = attempt_id or assignment_id
        self.owner_principal_id = owner_principal_id
        self.parent_execution_id = parent_execution_id
        # Partition key fallback for non-batch principal traces on the Boundless dialect;
        # batch children instead inherit the parent's created_at through spec['batch'].
        self.execution_created_at = execution_created_at
        self.internal_base_url = internal_base_url.rstrip('/') if internal_base_url else None
        self.artifact_scope_available = artifact_scope_available
        self.result_publish_deadline_at = result_publish_deadline_at
        self.trace_registry = trace_registry
        self.data_credentials_loader = data_credentials_loader
        self.request_admission = request_admission
        self._data_credentials: dict[str, Any] | None = None
        self._trace_session: RunTraceSession | None = None
        self._start_task: asyncio.Task[None] | None = None

    async def prepare_data_credentials(self) -> None:
        """Load on the Manager loop before transferring a driver to a worker thread."""
        self._data_credentials = await self.data_credentials_loader() if self.data_credentials_loader else {}

    async def start(
        self,
        handle: ExecutionHandle,
        spec: dict[str, Any],
        config: dict[str, Any],
        reporter: ExecutionReporter,
    ) -> None:
        self._start_task = asyncio.current_task()
        try:
            run_input = self._build_run_input(spec)
            trace_metadata = self._trace_context_metadata(spec)
        except Exception as exc:
            await reporter.fail({
                'type': 'InvalidExecutionSpec',
                'message': str(exc),
                'traceback': traceback.format_exc(),
            })
            return
        try:
            trace_backends = self.trace_registry.resolve(
                run_kind='single',
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
        trace_session = RunTraceSession(
            run_id=self.execution_id,
            run_kind='single',
            processor=ensure_run_trace_processor(),
            backend_sinks=trace_backends.backend_sinks,
            metadata=trace_metadata,
        )
        self._trace_session = trace_session
        output_emitted = False
        trace_closed = False
        captured_output: dict[str, Any] | None = None

        async def drain_runtime_traces(ctx: Any) -> None:
            runtime_provider = getattr(ctx, 'runtime_provider', None)
            if runtime_provider is None:
                return
            drained_runtime_ids: set[int] = set()
            for runtime_name, runtime in runtime_provider.get_active_runtimes().items():
                runtime_key = id(runtime)
                if runtime_key in drained_runtime_ids:
                    continue
                await drain_runtime_trace(
                    runtime,
                    error_policy=run_input.runner.trace_drain_error_policy,
                    run_id=self.execution_id,
                    runtime_name=runtime_name,
                )
                drained_runtime_ids.add(runtime_key)

        async def capture_output(output: RunOutput) -> dict[str, Any]:
            nonlocal captured_output
            captured_output = await asyncio.to_thread(output.model_dump, mode='json')
            trace_session.capture_output(captured_output)
            return captured_output

        async def emit_run_output(ctx: Any, output: RunOutput) -> None:
            nonlocal output_emitted
            payload = await capture_output(output)
            await drain_runtime_traces(ctx)
            await trace_session.emit(
                'run.output',
                run_id=self.execution_id,
                success=output.success,
                output=payload,
            )
            output_emitted = True

        async def close_trace() -> None:
            nonlocal trace_closed
            if trace_closed:
                return
            await trace_session.close()
            trace_closed = True

        try:
            await trace_session.start()
            binding = make_source_binding(
                run_id=self.execution_id,
                run_span_id=trace_session.run_span_id,
                assignment_id=self.assignment_id,
                attempt_id=self.attempt_id,
                owner_principal_id=self.owner_principal_id,
                selected_sinks=tuple(trace_backends.backend_sinks),
                source_identities=trace_backends.source_identities,
                # Readers prune Boundless partitions on this key and cannot derive it. Only
                # the artifact envelope used to carry it, and that is published at close, so
                # a running Execution was unreadable; report it with the running state.
                parent_created_at=trace_metadata.get('parent_created_at'),
            )
            await reporter.mark_running(
                endpoint_url=self.internal_base_url,
                endpoint_ready=False,
                runtime_info={'run_trace': {'source_binding': binding}},
            )
            hooks = RunHooks(
                on_run_output=emit_run_output,
                on_context_closing=drain_runtime_traces,
            )
            if self._data_credentials is None:
                await self.prepare_data_credentials()
            credentials = self._data_credentials or {}
            self._data_credentials = None
            async with DataAccessContext(credentials=credentials) as access:
                with (
                    trace_session.activate(),
                    bind_data_access(access),
                    bind_request_admission(
                        self.request_admission,
                        execution_id=self.execution_id,
                        attempt_id=self.attempt_id,
                    ),
                ):
                    output = await run_single(run_input, hooks=hooks)
                if captured_output is None:
                    await capture_output(output)
            hooks.raise_if_failed()
            if not output_emitted:
                await emit_run_output(None, output)
            await trace_session.publish_result(
                status='closed' if output.success else 'failed',
                error=captured_output.get('error') if captured_output else None,
            )
            await trace_session.emit(
                'run.status',
                run_id=self.execution_id,
                status='closed' if output.success else 'failed',
            )
            await close_trace()
        except asyncio.CancelledError:
            if not trace_closed:
                with contextlib.suppress(Exception):
                    await trace_session.emit('run.status', run_id=self.execution_id, status='killed')
                with contextlib.suppress(Exception):
                    await close_trace()
            with contextlib.suppress(Exception):
                await self._publish_trace_metadata(reporter)
            raise
        except Exception as exc:
            persistence_error = result_persistence_error(exc)
            if persistence_error is not None:
                error = {'type': ResultPersistenceError.error_type, 'message': str(persistence_error)}
            elif isinstance(exc, RunTraceSinkError):
                error = {'type': 'TraceArtifactUnavailable', 'message': str(exc)}
            else:
                error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            if captured_output is not None:
                try:
                    await trace_session.publish_result(status='failed', error=error)
                except Exception as publish_error:
                    error = {
                        'type': ResultPersistenceError.error_type,
                        'message': str(publish_error),
                        'traceback': traceback.format_exc(),
                        'original_error': error,
                    }
            if not trace_closed:
                with contextlib.suppress(Exception):
                    await trace_session.emit('run.error', run_id=self.execution_id, **error)
                    await trace_session.emit('run.status', run_id=self.execution_id, status='failed')
                with contextlib.suppress(Exception):
                    await close_trace()
            result_metadata = trace_session.result_artifact_metadata()
            metadata = self._result_metadata()
            result = None
            if metadata and (result_metadata or (output_emitted and trace_closed)):
                result = make_run_result_pointer(metadata)
            await reporter.fail(error, result=result, runtime_info=self._trace_runtime_info(metadata))
            return

        runtime_info = self._trace_runtime_info()
        if output.success:
            await reporter.complete(self._result_payload(output), runtime_info=runtime_info)
        else:
            error = output.error.model_dump(mode='json') if hasattr(output.error, 'model_dump') else output.error
            await reporter.fail(
                error or {'type': 'RunFailed', 'message': 'single run failed'},
                result=self._result_payload(output),
                runtime_info=runtime_info,
            )

    async def submit_command(self, handle: ExecutionHandle, command_type: str, payload: dict[str, Any]) -> None:
        raise RuntimeError(f'nexus_single does not accept command {command_type}')

    def _result_payload(self, output: RunOutput) -> dict[str, Any]:
        if not self.artifact_scope_available:
            return output.model_dump(mode='json')
        return make_run_result_pointer(self._result_metadata())

    def _trace_context_metadata(self, spec: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            'execution_id': self.execution_id,
            'assignment_id': self.assignment_id,
            'attempt_id': self.attempt_id,
        }
        batch = spec.get('batch')
        if self.parent_execution_id is None:
            if batch is not None:
                raise ValueError('batch child metadata requires a persisted parent_execution_id')
            if self.owner_principal_id is not None:
                metadata.update({
                    'storage_scope': 'principal_v1',
                    'principal_scope_id': principal_trace_scope_id(self.owner_principal_id),
                })
                if self.execution_created_at is not None:
                    metadata['parent_created_at'] = self.execution_created_at.isoformat()
            return metadata
        if not isinstance(batch, dict):
            raise ValueError('batch child metadata must be an object')
        if self.owner_principal_id is None:
            raise ValueError('batch child trace scope requires a persisted owner_principal_id')
        spec_parent_execution_id = batch.get('parent_execution_id')
        if not isinstance(spec_parent_execution_id, str) or not spec_parent_execution_id:
            raise ValueError('batch child parent_execution_id must be a non-empty string')
        if spec_parent_execution_id != self.parent_execution_id:
            raise ValueError('batch child parent_execution_id does not match persisted parent_execution_id')
        metadata.update({
            'storage_scope': 'principal_v1',
            'principal_scope_id': principal_trace_scope_id(self.owner_principal_id),
            'parent_execution_id': self.parent_execution_id,
        })
        # Every child of one parent must share the parent's timestamp so they land in the
        # same time partition, which is what makes DROP PARTITION a valid cleanup for a
        # finished parent. Falling back to the child's own created_at would scatter them.
        parent_created_at = batch.get('parent_created_at')
        if parent_created_at is not None:
            if not isinstance(parent_created_at, str) or not parent_created_at:
                raise ValueError('batch child parent_created_at must be a non-empty string')
            metadata['parent_created_at'] = parent_created_at
        return metadata

    async def stop(self, handle: ExecutionHandle) -> None:
        task = self._start_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    def _build_run_input(self, spec: dict[str, Any]) -> RunInput:
        raw = copy.deepcopy(spec.get('run_input', spec))
        run_input = RunInput.from_dict(raw)
        run_input.run_id = self.execution_id
        run_input.runner = run_input.runner.model_copy(update={'hold_after_task': None, 'early_output_path': None})
        return run_input

    async def _publish_trace_metadata(self, reporter: ExecutionReporter) -> None:
        runtime_info = self._trace_runtime_info()
        if runtime_info is None:
            return
        await reporter.update_runtime_info(runtime_info)

    def _trace_runtime_info(self, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        metadata = self._result_metadata() if metadata is None else metadata
        # A result-only envelope must not replace the registered trace sources.
        if not metadata or metadata.get('trace_complete') is False:
            return None
        return {
            RUN_TRACE_RUNTIME_INFO_KEY: {
                RUN_TRACE_SINGLE_KEY: metadata,
            }
        }

    def _result_metadata(self) -> dict[str, Any]:
        if self._trace_session is None:
            return {}
        try:
            return self._trace_session.artifact_metadata()
        except RuntimeError:
            return self._trace_session.result_artifact_metadata()
