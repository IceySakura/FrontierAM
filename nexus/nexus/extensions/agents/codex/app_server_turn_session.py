"""Single-turn execution over an app-server thread."""

from __future__ import annotations

import asyncio

from contextlib import ExitStack
from typing import Any

from .app_server_session import AppServerSession, AppServerSessionResult


class AppServerTurnSession(AppServerSession):
    async def run(self, task: str) -> AppServerSessionResult:
        if self.thread_id is None:
            raise RuntimeError('Start a thread before running a turn')
        if self._completion is not None or self._turn_active:
            raise RuntimeError('A turn is already running')
        self.turn_id = None
        completion = self._completion = asyncio.get_running_loop().create_future()

        async def on_completed(params: dict[str, Any]) -> None:
            if (
                params.get('threadId') == self.thread_id
                and params['turn']['id'] == self.turn_id
                and not completion.done()
            ):
                completion.set_result(params['turn'])

        try:
            with ExitStack() as handlers:
                handlers.callback(self.client.register_notification_handler('turn/completed', on_completed))
                response = await self.client.request(
                    'turn/start', {'threadId': self.thread_id, 'input': [{'type': 'text', 'text': task}]}
                )
                self.turn_id = response['turn']['id']
                # Completion may have arrived before the start response.
                if not completion.done():
                    self._turn_active = True
                turn = await completion
                success = turn.get('status') == 'completed'
                error = (
                    '' if success else str(turn.get('error') or f'Codex turn ended with status {turn.get("status")}')
                )
                return AppServerSessionResult(success=success, error=error)
        finally:
            self._completion = None
            if not completion.done():
                completion.cancel()
            elif not completion.cancelled():
                # A transport failure can also fail the pending turn/start RPC.
                completion.exception()
