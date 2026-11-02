from __future__ import annotations

import json
import os
import sqlite3

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class _LedgerCase:
    name: str
    db_env: str
    provider: str
    protocol: str
    call_count: int
    run_id: str


_CASES = (
    _LedgerCase(
        name='gpt_responses_osworld',
        db_env='NEXUS_CUA_LEDGER_GPT_RESPONSES_OSWORLD_DB',
        provider='openai',
        protocol='responses',
        call_count=3,
        run_id='libreoffice_calc__357ef137-7eeb-4c80-a3bb-0951f26a8aff',
    ),
    _LedgerCase(
        name='claude_opus_osworld',
        db_env='NEXUS_CUA_LEDGER_CLAUDE_OPUS_OSWORLD_DB',
        provider='litellm',
        protocol='anthropic',
        call_count=10,
        run_id='libreoffice_calc__357ef137-7eeb-4c80-a3bb-0951f26a8aff',
    ),
    _LedgerCase(
        name='claude_code_cuagym',
        db_env='NEXUS_CUA_LEDGER_CLAUDE_CODE_CUAGYM_DB',
        provider='litellm',
        protocol='anthropic',
        call_count=11,
        run_id='d8f85475-87c1-5d7e-b9ad-7a10e59d57fd',
    ),
)


def _load_single_run(db_path: Path) -> tuple[str, str, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    with sqlite3.connect(f'{db_path.as_uri()}?mode=ro&immutable=1', uri=True) as connection:
        assert connection.execute('pragma integrity_check').fetchone() == ('ok',)
        rows = connection.execute('select run_id, status, task_output, processor_results from runs').fetchall()
        trace_events = [
            json.loads(value)
            for (value,) in connection.execute('select event_data from run_trace_events order by time_unix_nano, id')
        ]

    assert len(rows) == 1
    run_id, status, task_output_json, processor_results_json = rows[0]
    assert isinstance(task_output_json, str)
    assert isinstance(processor_results_json, str)
    return run_id, status, json.loads(task_output_json), json.loads(processor_results_json), trace_events


@pytest.mark.e2e
@pytest.mark.parametrize('case', _CASES, ids=lambda case: case.name)
def test_historical_cua_run_has_complete_provider_call_trace(case: _LedgerCase) -> None:
    raw_path = os.environ.get(case.db_env)
    if not raw_path:
        pytest.skip(f'{case.db_env} is required for the historical CUA ledger E2E')
    assert raw_path is not None
    db_path = Path(raw_path).expanduser().resolve()
    assert db_path.is_file(), f'historical CUA SQLite database does not exist: {db_path}'

    run_id, status, task_output, processor_results, trace_events = _load_single_run(db_path)
    assert run_id == case.run_id
    assert status == 'success'
    assert task_output['score'] == 1.0

    chat = processor_results['chat']
    trajectories = chat['trajectories']
    assert len(trajectories) == case.call_count

    provider_events = [
        event for event in trace_events if isinstance((event.get('attributes') or {}).get('provider_call'), dict)
    ]
    assert len(provider_events) == case.call_count
    span_ids = [event['span_id'] for event in provider_events]
    assert all(isinstance(span_id, str) and span_id for span_id in span_ids)
    assert len(set(span_ids)) == len(span_ids), 'provider calls must not be duplicated in the raw trace'

    for event in provider_events:
        provider_io = event['attributes']['provider_call']
        assert provider_io['provider'] == case.provider
        assert provider_io['protocol'] == case.protocol
        assert provider_io['model']
        assert provider_io['endpoint']
        assert provider_io['request'] is not None
        assert provider_io['response'] is not None
        assert provider_io['attempts']
        assert provider_io['attempts'][-1]['status'] == 'success'
