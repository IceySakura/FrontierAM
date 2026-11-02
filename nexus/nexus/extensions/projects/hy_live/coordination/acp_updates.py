"""Host-side receiver for the ACP ``session/update`` frames a runtime pushes while a turn runs.

The runtime host relay emits ``acp_session_update`` ``{channel_id, turn_id, seq, update}`` for
every ACP update of a prompt sent with ``relay_turn_id`` (``installed_common/host_rpc.py``). A
backend binds the turn's channel and id to the task it works for before the turn starts and
unbinds it when the turn returns; each matching frame in between goes as is to the update sink.
``unbind`` reports how many frames arrived, which the backend compares with the ``relay_frames``
the turn result reports to decide whether to replay its events.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import weakref

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from nexus.extensions.agents.installed_common.host_rpc import ACP_SESSION_UPDATE_EVENT


logger = logging.getLogger(__name__)

TaskUpdateSink = Callable[[str, str, dict[str, Any]], None]
"""Receives ``(owner_id, task_id, update)`` with the raw ACP ``update`` dict."""


@dataclass
class _TurnBinding:
    turn_id: str
    owner_id: str
    task_id: str
    received: int = 0
    """Frames of this turn delivered while its channel was bound."""
    arrived: asyncio.Event = field(default_factory=asyncio.Event)


class AcpUpdateRouter:
    """Route pushed ACP updates to the task whose turn currently owns their channel."""

    def __init__(self, update_sink: TaskUpdateSink) -> None:
        self.update_sink = update_sink
        self._bindings: dict[str, _TurnBinding] = {}
        self._runtime_ref: weakref.ReferenceType[Any] | None = None

    def listen(self, runtime: Any) -> None:
        """Subscribe to the runtime's pushed updates; later calls for the same runtime are no-ops."""
        if self._runtime_ref is not None and self._runtime_ref() is runtime:
            return
        runtime.on(ACP_SESSION_UPDATE_EVENT, cast(Any, self.dispatch))
        self._runtime_ref = weakref.ref(runtime)

    def bind(self, channel_id: str, *, turn_id: str, owner_id: str, task_id: str) -> None:
        """Send this channel's frames of ``turn_id`` to one task until ``unbind``."""
        if not channel_id or not turn_id:
            raise ValueError('channel_id and turn_id must not be empty')
        self._bindings[channel_id] = _TurnBinding(turn_id=turn_id, owner_id=owner_id, task_id=task_id)

    def retarget(self, channel_id: str, *, task_id: str) -> bool:
        """Send the bound turn's later frames to ``task_id``, keeping its turn id and the frames counted so far.

        A steer hands a running turn to the Run it opened; the frames the turn already pushed stay where
        they were recorded.

        Returns:
            False when the channel has no bound turn.
        """
        binding = self._bindings.get(channel_id)
        if binding is None:
            return False
        binding.task_id = task_id
        return True

    async def wait_for_frames(self, channel_id: str, frames: int, *, timeout: float) -> None:
        """Wait up to ``timeout`` seconds until the bound turn has received ``frames`` frames."""
        binding = self._bindings.get(channel_id)
        if binding is None:
            return
        deadline = asyncio.get_running_loop().time() + timeout
        while binding.received < frames:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            binding.arrived.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(binding.arrived.wait(), timeout=remaining)

    def unbind(self, channel_id: str) -> int | None:
        """Stop routing this channel and return how many frames the push delivered for the turn, if it was bound."""
        binding = self._bindings.pop(channel_id, None)
        return binding.received if binding is not None else None

    def dispatch(self, data: Any) -> None:
        """Hand one pushed frame to the task bound to its channel and turn."""
        if not isinstance(data, dict):
            return
        channel_id = data.get('channel_id')
        update = data.get('update')
        if not isinstance(channel_id, str) or not isinstance(update, dict):
            return
        binding = self._bindings.get(channel_id)
        # NOTE(wallyshi): After a Host budget timeout the sandbox CLI keeps settling the abandoned prompt
        # while the next task's turn is already bound to the same session; those frames carry the old id.
        if binding is None or data.get('turn_id') != binding.turn_id:
            return
        binding.received += 1
        binding.arrived.set()
        self.update_sink(binding.owner_id, binding.task_id, update)

    def replay(self, owner_id: str, task_id: str, events: list[Any]) -> None:
        """Hand a finished turn's own ``session/update`` params to the update sink, in order."""
        for event in events:
            update = event.get('update') if isinstance(event, dict) else None
            if isinstance(update, dict):
                self.update_sink(owner_id, task_id, update)


__all__ = ['AcpUpdateRouter']
