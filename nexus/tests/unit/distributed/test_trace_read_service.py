from __future__ import annotations

import asyncio

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from nexus.distributed.config import DistributedConfig
from nexus.distributed.trace.bindings import make_source_binding
from nexus.distributed.trace.cursor import TraceCursorCodec, decode_stream_id, stream_id
from nexus.distributed.trace.models import (
    AssignmentTraceMetadata,
    AuthorizedTraceTarget,
    TraceReadError,
    TraceSourceIdentity,
    json_bytes,
)
from nexus.distributed.trace.service import TraceReadService
from nexus.runner.trace.tdsql.sink_source import (
    RawTraceEventPage,
    RawTraceEventTooLargeError,
    RawTraceRecordInvalidError,
)


def configuration(*, backends=('tdsql',), **budgets):
    return DistributedConfig(
        trace_read={
            'enabled': True,
            'backend_allowlist': list(backends),
            **budgets,
            'readers': [
                {
                    'backend': backend,
                    'source_id': f'dataset-{backend}',
                    'source_revision': '1',
                    'environment_ids': ['env-1'],
                    'host': 'reader.invalid',
                    'database': 'traces',
                    'credential_ref': 'reader',
                }
                for backend in backends
            ],
        },
        run_trace_sinks={backend: {'kind': 'tdsql'} for backend in backends},
        secrets={
            'trace_cursor_keys': {'default': 'k' * 32},
            'trace_read_credentials': {'reader': {'username': 'only-select', 'password': 'secret'}},
        },
    )


def target(**kwargs):
    return replace(AuthorizedTraceTarget('owner', 'exec', 'owner', 'env-1', 'running', 'asn', {}), **kwargs)


def metadata(*, backends=('tdsql',), span='span', assignment_id='asn', attempt_no=1, identities=True):
    binding = make_source_binding(
        run_id='exec',
        run_span_id=span,
        assignment_id=assignment_id,
        attempt_id=f'attempt-{attempt_no}',
        owner_principal_id='owner',
        selected_sinks=(*backends, 'artifact'),
        source_identities={
            name: TraceSourceIdentity(source_id=f'dataset-{name}', source_revision='1') for name in backends
        }
        if identities
        else {},
    )
    return AssignmentTraceMetadata(
        assignment_id,
        f'attempt-{attempt_no}',
        attempt_no,
        'active',
        datetime.now(timezone.utc),
        {'run_trace': {'source_binding': binding}},
    )


def event(record_id, *, span='span', payload=None):
    return {
        'id': 2**53 + record_id,
        'run_id': 'exec',
        'run_span_id': span,
        'parent_execution_id': None,
        'record_id': record_id,
        'event_type': 'update',
        'time_unix_nano': 1789286400000000000 + record_id,
        'trace_id': 'trace',
        'span_id': 'tool',
        'parent_span_id': None,
        'event_data': payload if payload is not None else ' {"message":"轨迹 ✓", "n":9007199254740993} ',
        'created_at': '2026-09-13T08:00:00.123456',
    }


class Source:
    def __init__(self, records=()):
        self.records = list(records)
        self.calls = []
        self.closed = False

    async def list_raw_streams(self, metadata, *, after, limit):
        self.calls.append(('list', metadata, after))
        spans = sorted({row['run_span_id'] for row in self.records}, key=str.lower)
        return [span for span in spans if after is None or span.lower() > after.lower()][:limit]

    async def raw_stream_exists(self, metadata):
        self.calls.append(('exists', metadata))
        return any(row['run_span_id'] == metadata['run_span_id'] for row in self.records)

    async def read_raw_event_page(
        self, metadata, after_record_id, *, limit, max_event_bytes, accept_record, omit_oversized_event_data=False
    ):
        self.calls.append(('events', metadata, after_record_id))
        rows = [
            row
            for row in self.records
            if row['run_span_id'] == metadata['run_span_id'] and row['record_id'] > after_record_id
        ]
        items = []
        for row in rows:
            row = dict(row)
            if len(items) == limit:
                return RawTraceEventPage(items, True)
            try:
                size = len((row['event_data'] or '').encode())
                oversized = size > max_event_bytes
                if oversized and not omit_oversized_event_data:
                    raise RawTraceEventTooLargeError(row['record_id'], size)
                if omit_oversized_event_data:
                    row.update(event_data_bytes=size, event_data_omitted=oversized)
                    if oversized:
                        row['event_data'] = None
                if not accept_record(row):
                    return RawTraceEventPage(items, True)
            except (RawTraceEventTooLargeError, RawTraceRecordInvalidError):
                if items:
                    return RawTraceEventPage(items, True)
                raise
            items.append(row)
        return RawTraceEventPage(items, False)


async def loader():
    return [metadata()]


def service(monkeypatch, *, config=None, records=()):
    instance = TraceReadService(config or configuration())
    source = Source(records)
    monkeypatch.setattr(instance.resolver, 'create_source', lambda resolved: source)
    return instance, source


async def test_preview_omits_large_body_and_both_modes_reuse_signed_positions(monkeypatch):
    large = event(2, payload=json_bytes({'text': '中' * 5000}).decode())
    api, source = service(
        monkeypatch,
        config=configuration(page_target_bytes=4096, max_event_bytes=4096),
        records=[event(1), large, event(3)],
    )
    first = await api.read_events(target(), stream(), load_metadata=loader, limit=1)
    with pytest.raises(TraceReadError, match='trace_event_too_large'):
        await api.read_events(target(), stream(), load_metadata=loader, cursor=first['next_cursor'])
    preview = await api.read_events(
        target(), stream(), load_metadata=loader, cursor=first['next_cursor'], mode='preview', limit=1
    )
    omitted = preview['items'][0]
    assert preview['mode'] == 'preview'
    assert omitted['record_id'] == '2'
    assert omitted['event_data'] is None
    assert omitted['event_data_omitted'] is True
    assert omitted['event_data_bytes'] == len(large['event_data'].encode())
    assert len(omitted) == 14
    assert preview['has_more'] is True
    tail = await api.read_events(target(), stream(), load_metadata=loader, cursor=preview['next_cursor'])
    assert tail['mode'] == 'raw'
    assert [r['record_id'] for r in tail['items']] == ['3']
    assert len(tail['items'][0]) == 12
    assert source.records[1] == large


async def test_preview_handles_encoded_growth_and_a_page_of_omitted_records(monkeypatch):
    payload = json_bytes({'quotes': '"' * 1200}).decode()
    assert len(payload.encode()) < 4096
    api, _ = service(
        monkeypatch,
        config=configuration(page_target_bytes=4096, max_event_bytes=4096),
        records=[event(1, payload=payload), event(2, payload=json_bytes({'value': 'x' * 6000}).decode())],
    )
    with pytest.raises(TraceReadError, match='trace_event_too_large'):
        await api.read_events(target(), stream(), load_metadata=loader)
    result = await api.read_events(target(), stream(), load_metadata=loader, mode='preview')
    assert [row['record_id'] for row in result['items']] == ['1', '2']
    assert all(row['event_data'] is None and row['event_data_omitted'] for row in result['items'])
    assert result['items'][0]['event_data_bytes'] == len(payload.encode())
    assert result['has_more'] is False
    empty = await api.read_events(
        target(), stream(), load_metadata=loader, cursor=result['next_cursor'], mode='preview'
    )
    assert empty['items'] == []


async def test_preview_small_body_is_exact_and_threshold_change_keeps_cursor(monkeypatch):
    large = event(2, payload=json_bytes({'text': 'x' * 6000}).decode())
    api, _ = service(
        monkeypatch,
        config=configuration(page_target_bytes=4096, max_event_bytes=4096),
        records=[event(1), large],
    )
    first = await api.read_events(target(), stream(), load_metadata=loader, mode='preview', limit=1)
    assert first['items'][0]['event_data'] == event(1)['event_data']
    assert first['items'][0]['event_data_omitted'] is False
    api.config = api.config.model_copy(update={'max_event_bytes': 8192})
    tail = await api.read_events(target(), stream(), load_metadata=loader, cursor=first['next_cursor'], mode='preview')
    assert tail['items'][0]['event_data'] == large['event_data']
    assert tail['items'][0]['event_data_omitted'] is False


@pytest.mark.parametrize('mode', ['', 'RAW', 'local', 'unknown'])
async def test_invalid_read_mode_does_not_query(monkeypatch, mode):
    api, source = service(monkeypatch)
    with pytest.raises(TraceReadError, match='invalid_request'):
        await api.read_events(target(), stream(), load_metadata=loader, mode=mode)
    assert not source.calls


def stream(*, span='span', backend='tdsql', revision='1'):
    return stream_id(
        backend=backend, source_id=f'dataset-{backend}', source_revision=revision, run_id='exec', run_span_id=span
    )


async def test_follow_gap_empty_page_late_terminal_event_and_replay(monkeypatch):
    api, source = service(monkeypatch, records=[event(1), event(3), event(9)])
    first = await api.read_events(target(), stream(), load_metadata=loader, limit=2)
    assert [row['record_id'] for row in first['items']] == ['1', '3']
    assert first['has_more'] is True
    assert first['items'][0]['id'] == str(2**53 + 1)
    assert first['items'][0]['event_data'] == event(1)['event_data']
    assert first['items'][0]['time_unix_nano'] == '1789286400000000001'
    assert first['items'][0]['parent_span_id'] is None
    second = await api.read_events(target(), stream(), load_metadata=loader, cursor=first['next_cursor'])
    assert [row['record_id'] for row in second['items']] == ['9']
    assert second['has_more'] is False
    assert 'complete' not in second
    empty = await api.read_events(target(), stream(), load_metadata=loader, cursor=second['next_cursor'])
    assert empty['items'] == []
    source.records.append(event(12))
    late = await api.read_events(
        target(execution_status='succeeded'), stream(), load_metadata=loader, cursor=empty['next_cursor']
    )
    replay = await api.read_events(target(), stream(), load_metadata=loader, cursor=empty['next_cursor'])
    assert (
        late['items']
        == replay['items']
        == [{**event(12), 'record_id': '12', 'id': str(2**53 + 12), 'time_unix_nano': '1789286400000000012'}]
    )


async def test_discovery_does_not_omit_later_lower_sorting_stream(monkeypatch):
    api, source = service(monkeypatch, records=[event(1, span='B'), event(1, span='Z')])
    first = await api.list_streams(target(), load_metadata=loader, limit=1)
    assert first['scan_complete'] is False
    source.records.append(event(1, span='a'))
    rest = await api.list_streams(target(), load_metadata=loader, cursor=first['next_cursor'], limit=1)
    assert [item['run_span_id'] for item in rest['items']] == ['Z']
    fresh = await api.list_streams(target(), load_metadata=loader)
    assert [item['run_span_id'] for item in fresh['items']] == ['a', 'B', 'Z']
    assert fresh['next_cursor'] is None


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('principal_id', 'reader'),
        ('owner_principal_id', 'other'),
        ('environment_id', 'env-other'),
        ('execution_id', 'exec-other'),
    ],
)
async def test_cursor_is_not_transferable_between_authorized_scopes(monkeypatch, field, value):
    api, source = service(monkeypatch, records=[event(1)])
    page = await api.read_events(target(), stream(), load_metadata=loader)
    calls = len(source.calls)
    with pytest.raises(TraceReadError, match='invalid_trace_cursor'):
        await api.read_events(target(**{field: value}), stream(), load_metadata=loader, cursor=page['next_cursor'])
    assert len(source.calls) == calls


async def test_cursor_purpose_stream_mismatch_and_tampering_do_not_query(monkeypatch):
    api, source = service(monkeypatch, records=[event(1), event(1, span='z')])
    discovery = await api.list_streams(target(), load_metadata=loader, limit=1)
    page = await api.read_events(target(), stream(), load_metadata=loader)
    calls = len(source.calls)
    for token, selected in [
        (discovery['next_cursor'], stream()),
        (page['next_cursor'], stream(span='z')),
        (page['next_cursor'] + 'x', stream()),
    ]:
        with pytest.raises(TraceReadError, match='invalid_trace_cursor'):
            await api.read_events(target(), selected, load_metadata=loader, cursor=token)
    assert len(source.calls) == calls


def test_shared_keyring_expiry_rotation_and_password_independent_stream_id():
    now = [1000]
    codec = TraceCursorCodec({'old': 'a' * 32}, current_key_id='old', ttl_seconds=10, clock=lambda: now[0])
    scope = {
        'principal': 'owner',
        'owner': 'owner',
        'environment': 'env-1',
        'execution': 'exec',
        'backend': 'tdsql',
        'source_id': 'dataset-tdsql',
        'source_revision': '1',
        'span': 'span',
    }
    token = codec.issue(scope, purpose='events', position='9007199254740993')
    replica = TraceCursorCodec({'old': 'a' * 32, 'new': 'b' * 32}, current_key_id='new', clock=lambda: now[0])
    assert replica.decode(token, purpose='events', target=target())['position'] == '9007199254740993'
    assert decode_stream_id(stream(), execution_id='exec')['source_revision'] == '1'
    now[0] += 10
    with pytest.raises(TraceReadError, match='trace_cursor_expired') as error:
        replica.decode(token, purpose='events', target=target())
    assert error.value.status == 410


async def test_multiple_backends_require_selection_and_artifact_is_excluded(monkeypatch):
    api, _ = service(monkeypatch, config=configuration(backends=('tdsql', 'tdsql-hyway')))

    async def both():
        return [metadata(backends=('tdsql', 'tdsql-hyway'))]

    result = await api.list_streams(target(), load_metadata=both)
    assert result['availability'] == 'select_backend'
    assert [b['name'] for b in result['backends']] == ['tdsql', 'tdsql-hyway']
    result = await api.list_streams(target(), load_metadata=both, backend='tdsql-hyway')
    assert result['backend'] == 'tdsql-hyway'
    assert result['availability'] == 'readable'


@pytest.mark.parametrize(
    ('case', 'code'),
    [
        ('identity', 'trace_source_unresolved'),
        ('reader', 'trace_source_misconfigured'),
        ('allowlist', 'trace_backend_read_disabled'),
        ('local', 'trace_backend_unsupported'),
    ],
)
async def test_missing_mapping_or_read_permission_is_not_an_empty_success(monkeypatch, case, code):
    config = configuration()
    if case == 'reader':
        config.secrets.trace_read_credentials.clear()
    if case == 'allowlist':
        config.trace_read = config.trace_read.model_copy(update={'backend_allowlist': ()})
    api, source = service(monkeypatch, config=config)

    async def load():
        return [metadata(backends=('local',) if case == 'local' else ('tdsql',), identities=case != 'identity')]

    with pytest.raises(TraceReadError, match=code):
        await api.list_streams(target(), load_metadata=load)
    assert source.calls == []


async def test_legacy_running_waits_and_terminal_unresolved_never_guesses_default(monkeypatch):
    api, source = service(monkeypatch)

    async def legacy():
        return [replace(metadata(), runtime_info={})]

    response = await api.list_streams(target(), load_metadata=legacy)
    assert response['availability'] == 'waiting_source'
    assert response['scan_complete'] is False
    with pytest.raises(TraceReadError, match='trace_source_unresolved'):
        await api.list_streams(target(execution_status='succeeded'), load_metadata=legacy)
    assert source.calls == []


async def test_large_record_is_lossless_single_page_and_hard_limit_does_not_skip(monkeypatch):
    big = '轨迹' * (1024 * 400)
    api, _ = service(monkeypatch, records=[event(1, payload=big), event(2)])
    page = await api.read_events(target(), stream(), load_metadata=loader)
    assert len(json_bytes(page)) > 2 * 1024 * 1024
    assert [row['record_id'] for row in page['items']] == ['1']
    assert page['has_more']
    assert page['items'][0]['event_data'] == big
    api, _ = service(
        monkeypatch,
        config=configuration(max_event_bytes=2048, page_target_bytes=1024),
        records=[event(1), event(2, payload='x' * 3000), event(3)],
    )
    page = await api.read_events(target(), stream(), load_metadata=loader)
    assert [row['record_id'] for row in page['items']] == ['1']
    for _ in range(2):
        with pytest.raises(TraceReadError, match='trace_event_too_large') as error:
            await api.read_events(target(), stream(), load_metadata=loader, cursor=page['next_cursor'])
        assert error.value.status == 413


async def test_unknown_stream_is_distinct_from_empty_page(monkeypatch):
    api, _ = service(monkeypatch)
    with pytest.raises(TraceReadError, match='trace_stream_not_found'):
        await api.read_events(target(), stream(), load_metadata=loader)


async def test_busy_timeout_and_cancel_release_budget_after_source_cleanup(monkeypatch):
    api, source = service(
        monkeypatch, config=configuration(max_concurrent_queries=1, max_principal_queries=1, query_timeout_seconds=0.3)
    )
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            source.closed = True

    monkeypatch.setattr(source, 'list_raw_streams', blocked)
    pending = asyncio.create_task(api.list_streams(target(), load_metadata=loader))
    await entered.wait()
    with pytest.raises(TraceReadError, match='trace_read_busy'):
        await api.list_streams(target(), load_metadata=loader)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert source.closed
    assert api.admission._active == 0
    assert api.admission._principals == {}
    source.closed = False
    with pytest.raises(TraceReadError, match='trace_source_unavailable'):
        await api.list_streams(target(), load_metadata=loader)
    assert source.closed
    assert api.admission._active == 0


async def test_cursor_reads_skip_metadata_rescan_and_keep_exact_source(monkeypatch):
    api, _ = service(monkeypatch, records=[event(1)])
    page = await api.read_events(target(), stream(), load_metadata=loader)

    async def forbidden():
        raise AssertionError('cursor page must not rescan assignment history')

    result = await api.read_events(target(), stream(), load_metadata=forbidden, cursor=page['next_cursor'])
    assert result['items'] == []
    config = configuration()
    config.trace_read = config.trace_read.model_copy(update={'readers': ()})
    changed, _ = service(monkeypatch, config=config)
    with pytest.raises(TraceReadError, match='trace_source_changed'):
        await changed.read_events(target(), stream(), load_metadata=forbidden, cursor=page['next_cursor'])


async def test_old_terminal_metadata_requires_explicit_historical_window(monkeypatch):
    from datetime import timedelta

    old = metadata()
    legacy = replace(
        old,
        runtime_info={
            '_artifacts': {
                'single': {
                    'run_id': 'exec',
                    'run_span_id': 'span',
                    'run_kind': 'single',
                    'sinks': [
                        {'name': 'tdsql', 'metadata': {'storage_scope': 'batch_principal_v1'}},
                        {'name': 'artifact', 'metadata': {}},
                    ],
                }
            }
        },
    )

    async def load():
        return [legacy]

    config = configuration()
    api, source = service(monkeypatch, config=config)
    with pytest.raises(TraceReadError, match='trace_source_unresolved'):
        await api.list_streams(target(execution_status='succeeded'), load_metadata=load)
    assert source.calls == []
    reader = config.trace_read.readers[0].model_copy(
        update={
            'legacy_from': old.created_at - timedelta(days=1),
            'legacy_until': old.created_at + timedelta(days=1),
        }
    )
    config.trace_read = config.trace_read.model_copy(update={'readers': (reader,)})
    api, _ = service(monkeypatch, config=config, records=[event(1)])
    result = await api.list_streams(target(execution_status='succeeded'), load_metadata=load)
    assert result['availability'] == 'readable'
    assert result['items'][0]['attempt_no'] == 1


async def test_same_named_backend_historical_versions_cannot_be_silently_combined(monkeypatch):
    config = configuration()
    reader = config.trace_read.readers[0]
    config.trace_read = config.trace_read.model_copy(
        update={'readers': (reader, reader.model_copy(update={'source_revision': '2'}))}
    )
    first = metadata()
    second = metadata(assignment_id='asn-2', span='later', attempt_no=2)
    second.runtime_info['run_trace']['source_binding']['sources'][0]['source_revision'] = '2'

    async def load():
        return [first, second]

    api, source = service(monkeypatch, config=config)
    with pytest.raises(TraceReadError, match='trace_source_unresolved'):
        await api.list_streams(target(), load_metadata=load)
    assert source.calls == []


async def test_persisted_selection_conflict_is_not_resolved_by_guessing(monkeypatch):
    api, source = service(monkeypatch)
    with pytest.raises(TraceReadError, match='trace_source_unresolved'):
        await api.list_streams(
            target(driver_config={'run_trace_backends': {'sinks': ['tdsql-hyway']}}), load_metadata=loader
        )
    assert source.calls == []


async def test_encoding_hard_limit_checks_escaped_bytes_and_preserves_prefix(monkeypatch):
    api, _ = service(
        monkeypatch,
        config=configuration(page_target_bytes=1024, max_event_bytes=2048),
        records=[event(1), event(3, payload='\x00' * 400), event(5)],
    )
    first = await api.read_events(target(), stream(), load_metadata=loader)
    assert [r['record_id'] for r in first['items']] == ['1']
    with pytest.raises(TraceReadError, match='trace_event_too_large'):
        await api.read_events(target(), stream(), load_metadata=loader, cursor=first['next_cursor'])


async def test_full_json_response_budget_includes_envelope_and_cursor(monkeypatch):
    api, _ = service(monkeypatch, config=configuration(page_target_bytes=2048), records=[event(i) for i in range(1, 8)])
    page = await api.read_events(target(), stream(), load_metadata=loader)
    assert len(page['items']) < 7
    assert len(json_bytes(page)) <= 2048
    assert page['has_more']


async def test_metadata_scan_limit_fails_instead_of_reporting_partial_candidates(monkeypatch):
    api, source = service(monkeypatch, config=configuration(max_assignments=1))

    async def load():
        return [metadata(), metadata(assignment_id='asn-2', span='later', attempt_no=2)]

    with pytest.raises(TraceReadError, match='trace_metadata_scan_incomplete'):
        await api.list_streams(target(), load_metadata=load)
    assert source.calls == []


async def test_unconfirmed_cleanup_stops_further_reads_instead_of_reusing_capacity(monkeypatch):
    from nexus.runner.trace.tdsql.sink_source import RawTraceCleanupError

    api, source = service(monkeypatch)

    async def unsafe(*args, **kwargs):
        raise RawTraceCleanupError('unconfirmed server query')

    monkeypatch.setattr(source, 'list_raw_streams', unsafe)
    with pytest.raises(TraceReadError, match='trace_source_misconfigured'):
        await api.list_streams(target(), load_metadata=loader)

    async def should_not_query(*args, **kwargs):
        pytest.fail('failed cleanup must block new queries')

    monkeypatch.setattr(source, 'list_raw_streams', should_not_query)
    with pytest.raises(TraceReadError, match='trace_source_misconfigured'):
        await api.list_streams(target(), load_metadata=loader)
    assert api.admission._active == 0


async def test_explicit_backend_selection_does_not_fall_back_or_require_other_reader(monkeypatch):
    config = configuration(backends=('tdsql', 'tdsql-hyway'))
    config.trace_read = config.trace_read.model_copy(update={'backend_allowlist': ('tdsql',)})
    api, _ = service(monkeypatch, config=config)

    async def both():
        return [metadata(backends=('tdsql', 'tdsql-hyway'))]

    with pytest.raises(TraceReadError, match='trace_backend_read_disabled'):
        await api.list_streams(target(), load_metadata=both)
    selected = await api.list_streams(target(), load_metadata=both, backend='tdsql')
    assert selected['backend'] == 'tdsql'
    with pytest.raises(TraceReadError, match='trace_backend_read_disabled'):
        await api.list_streams(target(), load_metadata=both, backend='tdsql-hyway')
