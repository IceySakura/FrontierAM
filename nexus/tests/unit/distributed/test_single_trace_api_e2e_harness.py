from __future__ import annotations

import copy

from types import SimpleNamespace

import pytest

from tests.e2e.distributed.run_single_trace_api import (
    Acceptance,
    accepted_execution_ids,
    normalized_page,
    submission_log_tail,
    validate_target_job,
    wait_export_checkpoint,
)
from tests.unit.distributed.test_trace_read_service import event


def encoded(record):
    return {**record, **{name: str(record[name]) for name in ('id', 'record_id', 'time_unix_nano')}}


async def test_export_checkpoint_wait_allows_cold_cloud_startup(tmp_path, monkeypatch):
    from tests.e2e.distributed import run_single_trace_api as harness

    checkpoint = tmp_path / 'checkpoint.json'
    elapsed = 0

    async def sleep(_):
        nonlocal elapsed
        elapsed += 60
        if elapsed == 240:
            checkpoint.write_text('{"committed_records": 1}')

    monkeypatch.setattr(harness.time, 'monotonic', lambda: elapsed)
    monkeypatch.setattr(harness.asyncio, 'sleep', sleep)
    value = await wait_export_checkpoint(SimpleNamespace(returncode=None), checkpoint, 1, timeout_seconds=1800)
    assert value['committed_records'] == 1
    assert elapsed == 240


async def test_export_checkpoint_wait_does_not_hide_early_process_failure(tmp_path):
    with pytest.raises(AssertionError, match='exited before checkpoint: 1'):
        await wait_export_checkpoint(SimpleNamespace(returncode=1), tmp_path / 'missing.json', 1)


def test_protocol_client_validates_whole_page_before_committing_projection():
    committed = {1: event(1)}
    before = copy.deepcopy(committed)
    broken = encoded(event(5))
    broken['run_span_id'] = 'another-stream'
    with pytest.raises(AssertionError):
        normalized_page(
            {'items': [encoded(event(3)), broken]}, execution_id='exec', run_span_id='span', committed=committed
        )
    assert committed == before


def test_protocol_client_preserves_precision_and_accepts_identical_replay():
    rows = normalized_page(
        {'items': [encoded(event(1)), encoded(event(3))]},
        execution_id='exec',
        run_span_id='span',
        committed={1: event(1)},
    )
    assert rows == [event(1), event(3)]


def test_preview_protocol_preserves_omission_identity_and_rejects_inconsistent_body():
    raw = encoded(event(1))
    raw.update(event_data=None, event_data_bytes=20 * 1024 * 1024, event_data_omitted=True)
    rows = normalized_page(
        {'items': [raw]}, execution_id='exec', run_span_id='span', committed={}, preview=True
    )
    assert rows == [{**event(1), 'event_data': None}]
    raw['event_data_omitted'] = False
    with pytest.raises(AssertionError):
        normalized_page({'items': [raw]}, execution_id='exec', run_span_id='span', committed={}, preview=True)


@pytest.mark.parametrize('records', [[event(1), event(1)], [event(3), event(1)]])
def test_protocol_client_rejects_duplicate_or_reversed_records(records):
    with pytest.raises(AssertionError):
        normalized_page(
            {'items': [encoded(row) for row in records]}, execution_id='exec', run_span_id='span', committed={}
        )


def test_submission_ownership_survives_non_json_output_and_partial_trailing_message():
    execution = 'exe_' + 'a' * 32
    assert accepted_execution_ids('log\n{"execution_id":"' + execution + '"}\n{unfinished') == [execution]
    assert accepted_execution_ids('{"execution_id":"not-an-execution"}') == []


def test_submission_diagnostics_hide_credentials_and_signed_urls(tmp_path):
    path = tmp_path / 'stderr'
    path.write_text('Error: upload failed\nsecret-value\nhttps://example.test/?token=hidden\n' + 'A' * 48)
    text = submission_log_tail(path, ['secret-value'])
    assert 'Error: upload failed' in text
    assert 'secret-value' not in text
    assert 'hidden' not in text
    assert 'A' * 48 not in text


def test_acceptance_source_preserves_locked_workspace_members(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    for name in ('pyproject.toml', 'uv.lock', 'README.md'):
        (project / name).write_text(name)
    member = project / 'services/packages/infra/pyproject.toml'
    member.parent.mkdir(parents=True)
    member.write_text('[project]\nname = "infra"\n')
    probe = tmp_path / 'probe.py'
    probe.write_text('# fixture\n')
    acceptance = Acceptance(SimpleNamespace(output_dir=tmp_path / 'output', project_root=project, probe=probe))
    acceptance.prepare()
    assert (acceptance.source / member.relative_to(project)).read_text() == member.read_text()
    assert (acceptance.source / 'uv.lock').read_text() == 'uv.lock'


def test_database_acceptance_refuses_host_side_execution(monkeypatch):
    monkeypatch.delenv('NEXUS_TRACE_API_E2E_CLUSTER_ID', raising=False)
    monkeypatch.delenv('POD_NAMESPACE', raising=False)
    with pytest.raises(RuntimeError, match='authorized runner-test cluster Job'):
        validate_target_job()


async def test_cleanup_reports_unresolved_submission_even_when_no_ids_were_observed(tmp_path):
    acceptance = Acceptance(SimpleNamespace(output_dir=tmp_path))
    acceptance.submissions['single'] = {'status': 'unresolved', 'idempotency_key': acceptance.marker}
    await acceptance.cleanup()
    assert acceptance.report['cleanup_verified'] is False


async def test_failure_keeps_request_evidence_before_cleanup_changes_last_request(tmp_path, monkeypatch):
    token = tmp_path / 'token'
    token.write_text('test-token')
    acceptance = Acceptance(SimpleNamespace(output_dir=tmp_path / 'output', token_file=token))
    failed_request = {'path': '/trace-streams', 'status': 503, 'error': 'trace_source_misconfigured'}

    async def request(*args, **kwargs):
        return {'principal_id': 'owner'}, 1

    async def submit(*args):
        acceptance.requests.append(failed_request)
        raise RuntimeError('test failure')

    async def cleanup():
        acceptance.requests.append({'path': '/allocations', 'status': 200})
        acceptance.report['cleanup_verified'] = True

    monkeypatch.setattr(acceptance, 'prepare', lambda: None)
    monkeypatch.setattr(acceptance, 'request', request)
    monkeypatch.setattr(acceptance, 'submit', submit)
    monkeypatch.setattr(acceptance, 'cleanup', cleanup)
    with pytest.raises(RuntimeError, match='test failure'):
        await acceptance.run()
    assert acceptance.report['failure']['last_request'] == failed_request
    assert acceptance.report['failure']['function'] == 'submit'
    assert acceptance.report['cleanup_verified'] is True


async def test_cleanup_discovers_retry_children_after_parent_stops(tmp_path, monkeypatch):
    acceptance = Acceptance(SimpleNamespace(output_dir=tmp_path))
    parent, child = 'exe_' + 'a' * 32, 'exe_' + 'b' * 32
    acceptance.batches.append(parent)
    acceptance.own(parent)
    acceptance.submissions['batch'] = {'status': 'accepted', 'execution_id': parent}
    stopped = []

    async def execution(identity):
        return {'status': 'running' if identity not in stopped else 'killed'}

    async def request(path, **kwargs):
        assert kwargs['method'] == 'POST'
        stopped.append(path.split('/')[3])
        return {}, 2

    async def released(identity, **kwargs):
        assert identity in stopped
        return {'execution_status': 'killed', 'allocations': []}

    async def discover(identity):
        assert identity in stopped
        acceptance.own(child)

    monkeypatch.setattr(acceptance, 'execution', execution)
    monkeypatch.setattr(acceptance, 'request', request)
    monkeypatch.setattr(acceptance, 'released', released)
    monkeypatch.setattr(acceptance, 'discover_children', discover)
    await acceptance.cleanup()
    assert stopped == [parent, child]
    assert acceptance.report['cleanup_verified'] is True
