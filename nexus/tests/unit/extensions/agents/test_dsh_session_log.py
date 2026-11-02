from __future__ import annotations

import json

from pathlib import Path

import pytest

from nexus.extensions.agents.dsh.session_log import DshSessionLog, DshSessionLogError, summarize_turn


def _write_records(path: Path, records: list[dict], *, partial: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(record) + '\n' for record in records) + partial
    path.write_text(payload)


@pytest.mark.asyncio
async def test_wait_for_turn_end_ignores_partial_last_line(tmp_path: Path) -> None:
    path = tmp_path / 'workspace' / 'session-1' / 'session.jsonl'
    records = [
        {'type': 'session', 'version': 0, 'id': 'session-1', 'cwd': '/workspace'},
        {'type': 'turn/start', 'seq': 0, 'data': {'turn': 1}},
        {
            'type': 'assistant/message',
            'seq': 1,
            'data': {'message': {'content': [{'type': 'text', 'text': 'done'}]}},
        },
        {'type': 'turn/end', 'seq': 2, 'data': {'turn': 1, 'reason': {'kind': 'completed'}}},
    ]
    _write_records(path, records, partial='{"type":"future"')
    log = await DshSessionLog.wait_for_session(
        root=tmp_path,
        session_id='session-1',
        cwd='/workspace',
        timeout=0.2,
    )

    events = await log.wait_for_turn_end(timeout=0.2)
    assert [event['seq'] for event in events] == [0, 1, 2]
    assert summarize_turn(events)['output'] == 'done'
    assert summarize_turn(events)['success'] is True


def test_session_log_rejects_discontinuous_sequence(tmp_path: Path) -> None:
    path = tmp_path / 'session.jsonl'
    _write_records(
        path,
        [
            {'type': 'session', 'version': 0, 'id': 'session-1', 'cwd': '/workspace'},
            {'type': 'turn/start', 'seq': 0, 'data': {'turn': 1}},
            {'type': 'turn/end', 'seq': 2, 'data': {'turn': 1, 'reason': {'kind': 'completed'}}},
        ],
    )
    log = DshSessionLog(path=path, session_id='session-1', cwd='/workspace')

    with pytest.raises(DshSessionLogError, match='discontinuous'):
        log._new_events()


def test_summarize_turn_treats_non_completed_reason_as_failure() -> None:
    summary = summarize_turn(
        [{'type': 'turn/end', 'seq': 0, 'data': {'turn': 1, 'reason': {'kind': 'max-tokens'}}}],
        acp_text='partial',
    )

    assert summary['success'] is False
    assert summary['status'] == 'max-tokens'
    assert summary['output'] == 'partial'
