import json
import os
import sqlite3
import tempfile

from pathlib import Path
from typing import Any

import pytest

from nexus.runner.batch import BatchConfig
from nexus.runner.batch.runner import run_batch
from tests.e2e._llm_account import load_sqlite_model_account
from tests.e2e._main_agent_batch_fixture import _CASE_ENV, _CLIENT_NAMES
from tests.e2e.test_main_agent_text_thinking import (
    _ACCOUNT_DB_ENV,
    _ACCOUNT_RUN_ID_ENV,
    _CASES,
    _PROTOCOL_FINAL,
    _assert_all_history_replayed,
)


_OUTPUT_ENV = 'NEXUS_MAIN_AGENT_BATCH_E2E_OUTPUT'


def _load_batch_evidence(db_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            'select run_id, status, config, task_output, agent_output, processor_results from runs'
        ).fetchone()
        trace_events = [
            value
            for (value,) in connection.execute('select event_data from run_trace_events order by time_unix_nano, id')
        ]
    finally:
        connection.close()
    assert row is not None
    run_id, status, config, task_output, agent_output, processor_results = row
    return {
        'run_id': run_id,
        'status': status,
        'config': json.loads(config),
        'task_output': json.loads(task_output),
        'agent_output': json.loads(agent_output),
        'processor_results': json.loads(processor_results),
        'trace_events': trace_events,
    }


def _provider_calls_from_trace_events(events: list[str]) -> list[dict[str, Any]]:
    calls = []
    for raw_event in events:
        event = json.loads(raw_event)
        attributes = event.get('attributes') or {}
        provider_call = attributes.get('provider_call')
        if isinstance(provider_call, dict):
            calls.append(provider_call)
    return calls


async def _drain_litellm_logging_worker(case_name: str) -> None:
    if not case_name.startswith('litellm_'):
        return
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    await GLOBAL_LOGGING_WORKER.flush()
    await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.e2e
@pytest.mark.llm
@pytest.mark.parametrize('case', _CASES, ids=lambda case: case.name)
async def test_main_agent_four_round_trace_persists_to_fresh_batch_sqlite(
    case: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not os.environ.get(_ACCOUNT_DB_ENV):
        pytest.skip(f'{_ACCOUNT_DB_ENV} is required for the live batch persistence E2E')

    output_root = Path(os.environ.get(_OUTPUT_ENV, tmp_path)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    case_dir = Path(tempfile.mkdtemp(prefix=f'{case.name}-', dir=output_root))
    generator_path = case_dir / 'generator.py'
    db_path = case_dir / 'batch.db'
    generator_path.write_text(
        'from tests.e2e._main_agent_batch_fixture import batch_item_from_environment\n\n'
        'def generate_configs():\n'
        '    yield batch_item_from_environment()\n',
        encoding='utf-8',
    )
    monkeypatch.setenv(_CASE_ENV, case.name)

    try:
        summary = await run_batch(
            generator_path=generator_path,
            db_path=db_path,
            batch_config=BatchConfig(
                max_concurrent=1,
                fail_fast=True,
                enable_trace_db=True,
                enable_log_capture=False,
                log_dual_write=False,
            ),
        )
    finally:
        await _drain_litellm_logging_worker(case.name)

    assert summary.total == 1
    assert summary.success == 1
    assert summary.failed == 0

    evidence = _load_batch_evidence(db_path)
    assert evidence['run_id'] == f'main-agent-{case.name}'
    assert evidence['status'] == 'success'
    assert evidence['config']['openai_clients']['main']['name'] == _CLIENT_NAMES[case.name]
    assert evidence['task_output'] == {'final_answer': _PROTOCOL_FINAL, 'iterations': 4}
    assert evidence['agent_output']['exit_status'] == 'finished'
    assert evidence['agent_output']['final_answer'] == _PROTOCOL_FINAL
    assert evidence['agent_output']['iterations'] == 4

    chat = evidence['processor_results']['chat']
    assert chat['trajectories']
    assert 'calls' not in chat
    trace_provider_calls = _provider_calls_from_trace_events(evidence['trace_events'])
    if case.name.startswith('litellm_'):
        assert trace_provider_calls == []
        messages = chat['trajectories'][0]['messages']
        assistant_call_ids = {
            tool_call['id']
            for message in messages
            if message['role'] == 'assistant'
            for tool_call in message.get('tool_calls') or []
        }
        tool_result_ids = {message.get('tool_call_id') for message in messages if message['role'] == 'tool'}
        assert len(assistant_call_ids) == 3
        assert tool_result_ids == assistant_call_ids
    else:
        assert len(trace_provider_calls) == 4
        assert [call['provider'] for call in trace_provider_calls] == [case.provider] * 4
        assert [call['protocol'] for call in trace_provider_calls] == [case.protocol] * 4
        assert all(call['attempts'][-1]['status'] == 'success' for call in trace_provider_calls)
        assert all(call.get('request') and call.get('response') for call in trace_provider_calls)
        _assert_all_history_replayed(
            case,
            trace_provider_calls,
            require_thinking=case.protocol != 'anthropic',
        )

    account = load_sqlite_model_account(
        os.environ[_ACCOUNT_DB_ENV],
        run_id=os.environ.get(_ACCOUNT_RUN_ID_ENV),
    )
    credential = account.credential
    assert credential not in json.dumps(evidence['task_output'], ensure_ascii=False)
    assert credential not in json.dumps(evidence['agent_output'], ensure_ascii=False)
    assert credential not in json.dumps(evidence['processor_results'], ensure_ascii=False)
    assert all(credential not in event for event in evidence['trace_events'])
