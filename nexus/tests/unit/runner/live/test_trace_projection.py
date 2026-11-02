from __future__ import annotations

from typing import Any

import pytest

from nexus.runner.trace import RunTraceContext
from nexus.trace.records import SpanEventRecord


def _record(
    record_id: int,
    event_type: str,
    *,
    events: list[dict[str, Any]] | None = None,
) -> SpanEventRecord:
    event_data: dict[str, Any] = {'type': event_type.upper(), 'span_id': 'run-span-1'}
    if event_type == 'start':
        event_data.update(
            {
                'name': 'run.live',
                'attributes': {'run_id': 'run-1', 'run_kind': 'live'},
                'events': [],
                'status': {'code': 'UNSET', 'message': None},
                'parent_span_id': '',
                'trace_id': 'trace-1',
            }
        )
    elif event_type == 'update':
        event_data['events'] = events or []
    return SpanEventRecord(
        record_id=record_id,
        event_type=event_type,
        time_unix_nano=record_id * 1_000_000_000,
        trace_id='trace-1',
        span_id='run-span-1',
        parent_span_id='',
        event_data=event_data,
    )


def _run_event(event_type: str, data: dict[str, Any], *, second: int) -> dict[str, Any]:
    return {
        'time_unix_nano': second * 1_000_000_000,
        'name': 'run.event',
        'attributes': {'event_type': event_type, 'event_data': data},
    }


@pytest.mark.asyncio
async def test_live_view_projects_raw_records_with_an_independent_cursor() -> None:
    from nexus.runner.live.trace import LiveRunTraceView

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    view = LiveRunTraceView(max_events=10)
    await view.start(context)
    await view.on_event(context, _record(1, 'start'))
    await view.on_event(
        context,
        _record(
            2,
            'update',
            events=[
                _run_event('run.status', {'status': 'running'}, second=2),
                _run_event('input.required', {'prompt': 'next'}, second=3),
            ],
        ),
    )

    events = view.snapshot_after(0)
    assert [event.id for event in events] == [1, 2]
    assert [event.type for event in events] == ['run.status', 'input.required']
    assert view.latest_id == 2


@pytest.mark.asyncio
async def test_active_view_and_closed_replay_use_the_same_projection_engine() -> None:
    from nexus.runner.live.trace import LiveRunTraceView, project_live_trace_records

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    records = [
        _record(1, 'start'),
        _record(2, 'update', events=[_run_event('run.status', {'status': 'idle'}, second=2)]),
        _record(3, 'end'),
    ]
    view = LiveRunTraceView(max_events=10)
    await view.start(context)
    for record in records:
        await view.on_event(context, record)
    await view.close(context)

    assert view.snapshot_after(0) == project_live_trace_records(context, records)


@pytest.mark.asyncio
async def test_live_view_reports_an_evicted_projection_cursor() -> None:
    from nexus.runner.live.trace import LiveRunTraceView
    from nexus.runner.trace import RunTraceCursorExpired

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    view = LiveRunTraceView(max_events=1)
    await view.start(context)
    await view.on_event(context, _record(1, 'start'))
    await view.on_event(
        context,
        _record(
            2,
            'update',
            events=[
                _run_event('run.status', {'status': 'running'}, second=2),
                _run_event('input.required', {'prompt': 'next'}, second=3),
            ],
        ),
    )

    with pytest.raises(RunTraceCursorExpired):
        view.snapshot_after(0)


def _completion_start_record(record_id: int, attributes: dict[str, Any]) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type='start',
        time_unix_nano=record_id * 1_000_000_000,
        trace_id='trace-1',
        span_id='span-oc-1',
        parent_span_id='run-span-1',
        event_data={
            'type': 'START',
            'span_id': 'span-oc-1',
            'name': 'openai_completion',
            'attributes': attributes,
            'events': [],
            'status': {'code': 'UNSET', 'message': None},
            'parent_span_id': 'run-span-1',
            'trace_id': 'trace-1',
        },
    )


def test_projector_truncates_oversized_trace_event_strings_without_mutating_raw_record() -> None:
    from nexus.runner.live.trace import LiveRunTraceProjector, RunTraceProjectionEngine

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    huge = 'x' * 5000
    attributes = {'gen_ai.prompt': huge, 'small': 'kept'}
    record = _completion_start_record(1, attributes)

    engine = RunTraceProjectionEngine(context, LiveRunTraceProjector(max_string_chars=100))
    events = engine.feed(record)

    assert len(events) == 1
    projected = events[0].data['event']['event']['attributes']
    assert projected['small'] == 'kept'
    assert projected['gen_ai.prompt'].startswith('x' * 100)
    assert projected['gen_ai.prompt'].endswith('[truncated 4900 chars]')
    assert len(projected['gen_ai.prompt']) < len(huge)
    # The raw record (the tdsql journal source of truth) is never mutated.
    assert record.event_data['attributes']['gen_ai.prompt'] == huge


def test_projector_truncation_can_be_disabled() -> None:
    from nexus.runner.live.trace import LiveRunTraceProjector, RunTraceProjectionEngine

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    huge = 'y' * 5000
    record = _completion_start_record(1, {'gen_ai.prompt': huge})

    engine = RunTraceProjectionEngine(context, LiveRunTraceProjector(max_string_chars=0))
    events = engine.feed(record)

    assert events[0].data['event']['event']['attributes']['gen_ai.prompt'] == huge


def test_projector_default_max_string_chars_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.runner.live.trace import DEFAULT_LIVE_TRACE_MAX_STRING_CHARS, LiveRunTraceProjector

    monkeypatch.delenv('NEXUS_LIVE_TRACE_MAX_STRING_CHARS', raising=False)
    assert LiveRunTraceProjector().max_string_chars == DEFAULT_LIVE_TRACE_MAX_STRING_CHARS

    monkeypatch.setenv('NEXUS_LIVE_TRACE_MAX_STRING_CHARS', '2048')
    assert LiveRunTraceProjector().max_string_chars == 2048

    # Explicit constructor argument wins; zero/negative disable truncation.
    assert LiveRunTraceProjector(max_string_chars=64).max_string_chars == 64
    monkeypatch.setenv('NEXUS_LIVE_TRACE_MAX_STRING_CHARS', '0')
    assert LiveRunTraceProjector().max_string_chars == 0
    monkeypatch.setenv('NEXUS_LIVE_TRACE_MAX_STRING_CHARS', 'not-a-number')
    assert LiveRunTraceProjector().max_string_chars == DEFAULT_LIVE_TRACE_MAX_STRING_CHARS


def _completion_update_record(record_id: int) -> SpanEventRecord:
    return SpanEventRecord(
        record_id=record_id,
        event_type='update',
        time_unix_nano=record_id * 1_000_000_000,
        trace_id='trace-1',
        span_id='span-oc-1',
        parent_span_id='run-span-1',
        event_data={
            'type': 'UPDATE',
            'span_id': 'span-oc-1',
            'events': [
                {
                    'time_unix_nano': record_id * 1_000_000_000,
                    'name': 'chat.completion.delta',
                    'attributes': {'delta': 'x'},
                }
            ],
        },
    )


def test_streaming_updates_do_not_deep_copy_retained_span_attributes() -> None:
    """Streaming deltas must not deep-copy the (image-sized) span input on every update.

    Regression guard: an ``openai_completion`` span carries the full request ``inputs``
    (an inlined image can be several MB). Deep-copying that payload once per streaming
    delta saturates the shared executor loop and starves the live SSE ``collect`` path,
    so a run finishes with artifacts while the consumer sees no events at all. The
    payload should be deep-copied at most once (at span start), never on updates.
    """
    from nexus.runner.live.trace import RunTraceProjectionEngine

    deep_copies = 0

    class _CountingStr(str):
        def __deepcopy__(self, memo: dict) -> _CountingStr:
            nonlocal deep_copies
            deep_copies += 1
            return self

    context = RunTraceContext(run_id='run-1', run_span_id='run-span-1', run_kind='live')
    engine = RunTraceProjectionEngine(context)

    engine.feed(_completion_start_record(1, {'inputs': {'image': _CountingStr('IMG')}}))
    # The span start still snapshots the retained span once.
    assert deep_copies == 1

    for record_id in range(2, 52):  # 50 streaming deltas
        events = engine.feed(_completion_update_record(record_id))
        assert len(events) == 1
        assert events[0].type == 'trace.event'

    # 50 streaming deltas must not add a single deep copy of the image-bearing input.
    assert deep_copies == 1
