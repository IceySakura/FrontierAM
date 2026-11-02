from __future__ import annotations

import asyncio
import contextlib
import traceback

from typing import Any

from nexus.distributed.batch.ports import BatchControl
from nexus.distributed.execution.executors.base import (
    ExecutionDriver,
    ExecutionHandle,
    ExecutionReporter,
    StopDisposition,
)
from nexus.distributed.execution.trace_backends import RunTraceBackendRegistry


class NexusBatchDriver(ExecutionDriver):
    spec_type = 'nexus_batch'
    stop_disposition = StopDisposition.DEFERRED

    def __init__(
        self,
        *,
        execution_id: str,
        attempt_id: str,
        owner_principal_id: str,
        batch_control: BatchControl,
        trace_registry: RunTraceBackendRegistry,
        reconcile_poll_interval: float = 0.5,
    ) -> None:
        if reconcile_poll_interval <= 0:
            raise ValueError('batch reconcile poll interval must be positive')
        self.execution_id = execution_id
        self.attempt_id = attempt_id
        self.batch_control = batch_control
        self.trace_registry = trace_registry
        self.reconcile_poll_interval = reconcile_poll_interval
        self._stopping = asyncio.Event()

    async def start(
        self,
        handle: ExecutionHandle,
        spec: dict[str, Any],
        config: dict[str, Any],
        reporter: ExecutionReporter,
    ) -> None:
        del handle
        try:
            batch = spec['batch']
            await reporter.mark_running(
                endpoint_url=None,
                endpoint_ready=False,
                runtime_info={'batch': {'generation_status': 'materializing'}},
            )
            await self.batch_control.ensure_prepared(
                self.execution_id,
                parent_attempt_id=self.attempt_id,
                bridge_timeout=_generation_bridge_timeout(spec),
            )
            await reporter.update_runtime_info({'batch': {'generation_status': 'complete'}})
            while not self._stopping.is_set():
                finalized = await self.batch_control.reconcile_once(
                    self.execution_id,
                    max_inflight=int(batch.get('max_inflight', 200)),
                    submit_batch_size=int(batch.get('submit_batch_size', 50)),
                    parent_attempt_id=self.attempt_id,
                )
                if finalized:
                    return
                await asyncio.sleep(self.reconcile_poll_interval)
        except asyncio.CancelledError:
            with contextlib.suppress(ValueError):
                await self.batch_control.stop_children(self.execution_id, parent_attempt_id=self.attempt_id)
            raise
        except Exception as exc:
            await reporter.fail({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        finally:
            with contextlib.suppress(Exception):
                await self.batch_control.release_parent(self.execution_id)

    async def submit_command(self, handle: ExecutionHandle, command_type: str, payload: dict[str, Any]) -> None:
        del handle, payload
        raise RuntimeError(f'nexus_batch does not accept command {command_type}')

    async def stop(self, handle: ExecutionHandle) -> None:
        del handle
        self._stopping.set()
        with contextlib.suppress(ValueError):
            await self.batch_control.stop_children(self.execution_id, parent_attempt_id=self.attempt_id)


def _generation_bridge_timeout(spec: dict[str, Any]) -> float:
    raw_limits = spec.get('generation_limits')
    limits = raw_limits if isinstance(raw_limits, dict) else {}
    timeout = limits.get('timeout_seconds', 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError('invalid_batch_generation_limit:timeout_seconds')
    return float(timeout) + 2.0
