"""Thread and turn RPCs for a Codex app-server connection."""

from __future__ import annotations

import asyncio

from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from .app_server_protocol import AppServerClient


@dataclass(frozen=True)
class AppServerSessionResult:
    success: bool
    error: str = ''


class AppServerSession(ABC):
    """Manage a thread and its current turn independently of the execution mode."""

    def __init__(self, client: AppServerClient) -> None:
        self.client = client
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self._turn_active = False
        self._completion: asyncio.Future[dict[str, Any]] | None = None
        self._handlers = ExitStack()

    async def __aenter__(self) -> AppServerSession:
        try:
            self._handlers.callback(self.client.register_notification_handler('turn/started', self._on_turn_started))
            self._handlers.callback(
                self.client.register_notification_handler('turn/completed', self._on_turn_completed)
            )
            self._handlers.callback(self.client.register_error_handler(self._on_error))
            await self.client.request('initialize', {'clientInfo': {'name': 'nexus', 'title': 'Nexus', 'version': '1'}})
            await self.client.notify('initialized', {})
        except BaseException:
            self._handlers.close()
            raise
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self._handlers.close()

    async def start_thread(self, *, working_dir: str, model: str | None, developer_instructions: str | None) -> None:
        params: dict[str, Any] = {'cwd': working_dir, 'approvalPolicy': 'never', 'sandbox': 'danger-full-access'}
        if model:
            params['model'] = model
        if developer_instructions:
            params['developerInstructions'] = developer_instructions
        response = await self.client.request('thread/start', params)
        self.thread_id = response['thread']['id']

    @abstractmethod
    async def run(self, task: str) -> AppServerSessionResult:
        """Run a task using this session's execution policy."""

    async def _on_turn_started(self, params: dict[str, Any]) -> None:
        if params.get('threadId') == self.thread_id:
            self.turn_id = params['turn']['id']
            self._turn_active = True

    async def _on_turn_completed(self, params: dict[str, Any]) -> None:
        if params.get('threadId') != self.thread_id:
            return
        turn = params['turn']
        if self.turn_id is None or turn['id'] == self.turn_id:
            self.turn_id = turn['id']
            self._turn_active = False

    async def _on_error(self, error: Exception) -> None:
        if self._completion is not None and not self._completion.done():
            self._completion.set_exception(error)

    async def interrupt(self) -> None:
        if self._turn_active and self.turn_id is not None:
            await self.client.request('turn/interrupt', {'threadId': self.thread_id, 'turnId': self.turn_id}, timeout=5)
