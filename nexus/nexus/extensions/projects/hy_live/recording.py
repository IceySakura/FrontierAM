"""Record what hylive exchanged with the outside: every event fed to it and every command it gave.

Each line of ``events.jsonl`` is the session config (``dir=config``), an event (``dir=in``) or a command
(``dir=out``), in the order the runner took them. ``hylive.replay`` replays such a file.
"""

from __future__ import annotations

import itertools
import json
import logging
import os

from pathlib import Path
from typing import Any

from hylive import commands, events
from hylive.commands import COMMAND_ADAPTER
from hylive.events import EVENT_ADAPTER


logger = logging.getLogger(__name__)


class EventLog:
    """Append hylive's events and commands to one JSONL file as they happen."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self._stream = os.fdopen(descriptor, 'w', encoding='utf-8')
        self._seq = itertools.count(1)

    def event(self, event: events.Event) -> None:
        self._write('in', EVENT_ADAPTER.dump_python(event, mode='json'))

    def command(self, command: commands.Command) -> None:
        self._write('out', COMMAND_ADAPTER.dump_python(command, mode='json'))

    def config(self, config: dict[str, Any]) -> None:
        """The session config, written first so a replay starts from the same session and draws the same ids."""
        self._write('config', {'type': 'session_config', 'value': config})

    def close(self) -> None:
        self._stream.close()

    def _write(self, direction: str, record: dict[str, Any]) -> None:
        if self._stream.closed:
            return
        line = {'seq': next(self._seq), 'dir': direction, **record}
        try:
            self._stream.write(json.dumps(line, ensure_ascii=False, default=str) + '\n')
            self._stream.flush()
        except Exception:  # noqa: BLE001
            logger.exception('Failed to record an hy-live event; recording continues')


__all__ = ['EventLog']
