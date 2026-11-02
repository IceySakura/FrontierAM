from __future__ import annotations

import json
import os
import stat

from pathlib import Path

from hylive import commands, events
from hylive.commands import COMMAND_ADAPTER
from hylive.events import EVENT_ADAPTER

from nexus.extensions.projects.hy_live.recording import EventLog


def test_the_log_writes_config_events_and_commands_in_order_and_only_for_its_owner(tmp_path: Path) -> None:
    path = tmp_path / 'run' / 'events.jsonl'
    log = EventLog(path)

    log.config({'owner_id': 'o'})
    log.event(events.UserMessage(at=1.0, content='你好'))
    log.command(commands.Speak(say_id='say-1', text='你好呀', source='user'))
    log.close()
    log.event(events.UserFinished(at=2.0))

    lines = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert [(line['seq'], line['dir'], line['type']) for line in lines] == [
        (1, 'config', 'session_config'),
        (2, 'in', 'user_message'),
        (3, 'out', 'speak'),
    ], 'nothing is written once the log is closed'
    EVENT_ADAPTER.validate_python({key: value for key, value in lines[1].items() if key not in {'seq', 'dir'}})
    COMMAND_ADAPTER.validate_python({key: value for key, value in lines[2].items() if key not in {'seq', 'dir'}})
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600, 'a recording holds the conversation'
