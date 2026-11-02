"""Finalize a Gateway session through the standard span processor lifecycle."""

from __future__ import annotations

from typing import Any

from nexus.gateway.nexus_run_tracker import NexusRunTracker
from nexus.gateway.session_manager import session_manager
from nexus.gateway.trajectory import Trajectory, dump_trajectory_with_payloads
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import span_processor_registry
from nexus.trace.types import Span


@span_processor_registry.register('gateway_trajectory')
class GatewayTrajectoryProcessor(NexusRunTracker):
    """Return one canonical Gateway trajectory as a processor result.

    Session teardown needs the loop (draining publications, reading the record store),
    so it runs in ``afinalize``; ``finalize`` only settles the synchronous facts the
    span stream collected. The exported result is self-contained: node payloads are
    inlined and the store records are released, because the store's lifetime ends with
    the run while the result outlives it.
    """

    def __init__(self, *, trajectory_id: str) -> None:
        if not trajectory_id:
            raise ValueError('GatewayTrajectoryProcessor requires trajectory_id')
        super().__init__(trajectory_id=trajectory_id)
        self._terminal_messages: list[dict[str, Any]] | None = None
        self._task_output: dict[str, Any] | None = None
        self._task_failed = False
        self._result: dict[str, Any] | None = None

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        super().on_end(span, event)
        span_type = span.attributes.get('type')
        output = span.attributes.get('outputs')

        if span_type == 'agent' and isinstance(output, dict):
            messages = output.get('main_trajectory')
            if isinstance(messages, list):
                self._terminal_messages = messages
        elif span_type == 'task':
            if isinstance(output, dict):
                self._task_output = output
            if span.status.code == 'ERROR':
                self._task_failed = True

    def finalize(self) -> None:
        trajectory_id = self._trajectory_id
        if trajectory_id is None:
            raise ValueError('trajectory_id must not be None before finalizing.')
        if session_manager.has_session(trajectory_id):
            self._import_terminal_observations(self._terminal_messages)

    async def afinalize(self) -> None:
        """End the session, inline its payloads, and release the store records.

        Idempotent: ``end_session`` drops the session, so a second call (the run scope's
        cancellation path after an in-band call) finds nothing and returns.
        """
        trajectory_id = self._trajectory_id
        if trajectory_id is None:
            raise ValueError('trajectory_id must not be None before finalizing.')
        session = session_manager.get_session(trajectory_id)
        if session is None:
            # Already finalized, or the run never opened a session.
            return

        store = session.record_store
        # Unconditional: the scaffold has exited, so no run outcome leaves a consumer behind.
        await session.abort_in_flight()
        await session.finish_pending_publications()

        # NOTE(petersheng): same order as nexus.runner.hytuner._resolve_termination_reason: an
        # agent-reported reason wins, then the Gateway's own record of a length-limited last call.
        if self.termination_reason is not None:
            termination_reason = self.termination_reason
        elif session.last_main_call_truncated:
            termination_reason = 'TRUNCATION'
        elif self._task_failed:
            termination_reason = 'UNHANDLED_ERROR'
        else:
            termination_reason = 'ENV_DONE'
        task_output = self._task_output or {}
        extra = task_output.get('extra_fields')
        trajectory = session_manager.end_session(trajectory_id, termination_reason=termination_reason)
        refs = self._payload_refs(trajectory)
        try:
            trajectory = self.finalize_trajectory(
                trajectory,
                trajectory_id,
                termination_reason=termination_reason,
                final_reward=task_output.get('score'),
                extra=extra if isinstance(extra, dict) else None,
            )
            if trajectory is None:
                raise RuntimeError('finalize_trajectory returned None, cannot generate result')
            payloads = await store.get_many(list(refs.values())) if refs else []
            self._result = dump_trajectory_with_payloads(trajectory, dict(zip(refs, payloads, strict=True)))
        finally:
            if refs:
                await store.release_many(list(refs.values()))

    @staticmethod
    def _payload_refs(trajectory: Trajectory) -> dict[int, Any]:
        return {
            node.node_id: node.token_account.payload_ref
            for node in trajectory.iter_nodes()
            if node.token_account is not None and node.token_account.payload_ref is not None
        }

    def get_result(self) -> dict[str, Any] | None:
        return self._result
