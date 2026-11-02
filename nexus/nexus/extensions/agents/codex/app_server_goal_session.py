"""Native goal execution over an app-server thread."""

from __future__ import annotations

import asyncio

from contextlib import ExitStack
from typing import Any

from .app_server_protocol import AppServerClient
from .app_server_session import AppServerSession, AppServerSessionResult


class AppServerGoalSession(AppServerSession):
    def __init__(self, client: AppServerClient, *, token_budget: int | None = None) -> None:
        super().__init__(client)
        self.token_budget = token_budget
        self.goal: dict[str, Any] | None = None

    async def run(self, task: str) -> AppServerSessionResult:
        """Let Codex continue turns until its goal stops or a turn fails."""
        if self.thread_id is None:
            raise RuntimeError('Start a thread before running a goal')
        if self._completion is not None or self._turn_active:
            raise RuntimeError('A turn is already running')
        self.goal = None
        self.turn_id = None
        last_turn: dict[str, Any] | None = None
        completion = self._completion = asyncio.get_running_loop().create_future()

        def finish_if_idle() -> None:
            if self._turn_active or completion.done():
                return
            if last_turn is not None and last_turn.get('status') != 'completed':
                completion.set_result(last_turn)
            elif (
                self.goal is not None
                and self.goal.get('status') != 'active'
                and (self.goal.get('status') != 'complete' or last_turn is not None)
            ):
                completion.set_result(last_turn or {'status': 'interrupted'})

        async def on_started(params: dict[str, Any]) -> None:
            nonlocal last_turn
            if params.get('threadId') == self.thread_id:
                last_turn = None

        async def on_completed(params: dict[str, Any]) -> None:
            nonlocal last_turn
            if params.get('threadId') == self.thread_id and params['turn']['id'] == self.turn_id:
                last_turn = params['turn']
                finish_if_idle()

        async def on_updated(params: dict[str, Any]) -> None:
            if params.get('threadId') == self.thread_id:
                self.goal = params['goal']
                finish_if_idle()

        async def on_cleared(params: dict[str, Any]) -> None:
            if params.get('threadId') == self.thread_id:
                self.goal = None
                if not completion.done():
                    completion.set_result({'status': 'failed', 'error': 'Codex goal was cleared'})

        params: dict[str, Any] = {'threadId': self.thread_id, 'objective': task, 'status': 'active'}
        if self.token_budget is not None:
            params['tokenBudget'] = self.token_budget
        try:
            with ExitStack() as handlers:
                handlers.callback(self.client.register_notification_handler('turn/started', on_started))
                handlers.callback(self.client.register_notification_handler('turn/completed', on_completed))
                handlers.callback(self.client.register_notification_handler('thread/goal/updated', on_updated))
                handlers.callback(self.client.register_notification_handler('thread/goal/cleared', on_cleared))
                response = await self.client.request('thread/goal/set', params)
                # Notifications can advance the goal before its set response arrives.
                if self.goal is None and not completion.done():
                    self.goal = response['goal']
                finish_if_idle()
                turn = await completion
                goal_status = (self.goal or {}).get('status')
                success = turn.get('status') == 'completed' and goal_status == 'complete'
                error = '' if success else str(turn.get('error') or f'Codex goal ended with status {goal_status}')
                return AppServerSessionResult(success=success, error=error)
        finally:
            self._completion = None
            if not completion.done():
                completion.cancel()
            elif not completion.cancelled():
                completion.exception()

    async def interrupt(self) -> None:
        if self.goal is not None and self.goal.get('status') == 'active':
            await self.client.request('thread/goal/set', {'threadId': self.thread_id, 'status': 'paused'}, timeout=5)
        await super().interrupt()
