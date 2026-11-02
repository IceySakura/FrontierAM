"""Local controller lifecycle checks; no Docker, Codex binary or model service."""

from __future__ import annotations

import asyncio
import json
import signal

import pytest

from nexus.extensions.tasks.frontieram_offline.agent import (
    _capture,
    _classify_stop,
    _drain_readers,
    _settle_cleanup,
)


def classify(reason='codex_process_exited', **overrides):
    values = {
        'final_seen': True,
        'returncode': 0,
        'errors': [],
        'error_events': [],
        'deadline': 100.0,
        'frozen': True,
    }
    values.update(overrides)
    return _classify_stop(reason, **values)


@pytest.mark.asyncio
async def test_process_exit_before_buffered_final_is_classified_after_drain(tmp_path):
    """Reproduce FIRST_COMPLETED winning while the stdout task has not run."""
    stream = asyncio.StreamReader()
    stream.feed_data(b'{"type":"turn.completed"}\n')
    stream.feed_eof()
    finished = asyncio.Event()
    errors = []
    reader = asyncio.create_task(_capture(stream, tmp_path / 'events', finished, errors, events=True))
    assert not finished.is_set()
    assert classify(final_seen=finished.is_set()) == ('codex_process_exited', False)

    await _drain_readers([reader], errors)

    assert finished.is_set()
    assert not errors
    assert classify(final_seen=finished.is_set(), errors=errors) == ('turn_completed', True)
    assert json.loads((tmp_path / 'events').read_text()) == {'type': 'turn.completed'}


@pytest.mark.asyncio
@pytest.mark.parametrize('payload', [b'not-json\n', b'[]\n', b'null\n', b'"text"\n'])
async def test_malformed_controller_events_stop_and_cannot_be_deadline_success(tmp_path, payload):
    stream = asyncio.StreamReader()
    stream.feed_data(payload)
    stream.feed_eof()
    finished = asyncio.Event()
    errors = []

    await _capture(stream, tmp_path / 'events', finished, errors, events=True)

    assert finished.is_set()
    assert errors == ['controller_event_stream_error']
    assert classify('development_deadline', errors=errors) == ('development_deadline', False)


@pytest.mark.asyncio
async def test_stream_line_over_limit_is_observed_instead_of_losing_reader(tmp_path):
    stream = asyncio.StreamReader(limit=8)
    stream.feed_data(b'{"type":"turn.completed"}\n')
    finished = asyncio.Event()
    errors = []

    await _capture(stream, tmp_path / 'events', finished, errors, events=True)

    assert finished.is_set()
    assert errors == ['controller_event_stream_error']


@pytest.mark.asyncio
async def test_stderr_io_error_is_observed_and_stops_controller(tmp_path):
    stream = asyncio.StreamReader()
    stream.set_exception(OSError('fixture I/O failure'))
    finished = asyncio.Event()
    errors = []

    await _capture(stream, tmp_path / 'stderr', finished, errors, events=False)

    assert finished.is_set()
    assert errors == ['controller_stderr_stream_error']


@pytest.mark.asyncio
async def test_inherited_stream_without_eof_has_bounded_drain(tmp_path):
    stream = asyncio.StreamReader()
    stream.feed_data(b'{"type":"turn.completed"}\n')
    finished = asyncio.Event()
    errors = []
    reader = asyncio.create_task(_capture(stream, tmp_path / 'events', finished, errors, events=True))
    await asyncio.wait_for(finished.wait(), timeout=1)

    await asyncio.wait_for(_drain_readers([reader], errors, timeout=0.01), timeout=1)

    assert reader.cancelled()
    assert errors == ['controller_log_drain_timeout']
    assert classify(errors=errors) == ('codex_process_exited', False)


@pytest.mark.asyncio
async def test_capture_cancellation_is_propagated(tmp_path):
    finished = asyncio.Event()
    errors = []
    reader = asyncio.create_task(_capture(asyncio.StreamReader(), tmp_path / 'events', finished, errors, events=True))
    await asyncio.sleep(0)
    reader.cancel()

    with pytest.raises(asyncio.CancelledError):
        await reader

    assert not finished.is_set()
    assert not errors
    with pytest.raises(asyncio.CancelledError):
        await _drain_readers([reader], errors)


@pytest.mark.asyncio
async def test_drain_owner_cancellation_is_not_swallowed(tmp_path):
    errors = []
    reader = asyncio.create_task(
        _capture(asyncio.StreamReader(), tmp_path / 'events', asyncio.Event(), errors, events=True)
    )
    drain = asyncio.create_task(_drain_readers([reader], errors))
    await asyncio.sleep(0)
    drain.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert not errors
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup_raises', [False, True])
async def test_owner_cancellation_waits_for_cleanup_then_propagates(cleanup_raises):
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    async def cleanup():
        started.set()
        await release.wait()
        completed.set()
        if cleanup_raises:
            raise RuntimeError('fixture cleanup failure')

    owner = asyncio.create_task(_settle_cleanup(cleanup()))
    await asyncio.wait_for(started.wait(), timeout=1)
    owner.cancel()
    await asyncio.sleep(0)
    assert not owner.done()
    assert not completed.is_set()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, timeout=1)
    assert completed.is_set()


@pytest.mark.asyncio
async def test_cleanup_errors_propagate_without_cancellation():
    async def cleanup():
        raise RuntimeError('fixture cleanup failure')

    with pytest.raises(RuntimeError, match='fixture cleanup failure'):
        await _settle_cleanup(cleanup())


@pytest.mark.parametrize('returncode', [1, -signal.SIGTERM, None])
def test_final_event_does_not_hide_controller_failure(returncode):
    assert classify(returncode=returncode) == ('codex_process_exited', False)


def test_drained_late_final_does_not_move_deadline_boundary():
    assert classify('development_deadline') == ('development_deadline', True)
    assert classify('development_deadline', frozen=False) == ('development_deadline', False)


@pytest.mark.parametrize('category', ['deadline', 'stopped', 'cancelled'])
def test_verified_deadline_can_accept_codex_error_exit(category):
    assert classify(
        'development_deadline',
        returncode=1,
        errors=['codex_turn_failed', category],
        error_events=[{'category': category, 'at_monotonic': 100.001}],
    ) == ('development_deadline', True)


@pytest.mark.parametrize(
    ('errors', 'events'),
    [
        (['deadline'], [{'category': 'deadline', 'at_monotonic': 90.0}]),
        (['deadline'], [{'category': 'deadline'}]),
        (['deadline'], []),
        (['codex_turn_failed'], []),
        ([], []),
        (['upstream_http'], [{'category': 'upstream_http', 'at_monotonic': 100.0}]),
        (
            ['controller_event_stream_error', 'deadline'],
            [{'category': 'deadline', 'at_monotonic': 100.0}],
        ),
    ],
)
def test_deadline_does_not_mask_earlier_or_unverified_errors(errors, events):
    assert classify('development_deadline', returncode=1, errors=errors, error_events=events) == (
        'development_deadline',
        False,
    )


def test_unfrozen_source_is_never_eligible():
    assert classify(frozen=False) == ('turn_completed', False)


def test_runner_cancel_is_never_eligible_even_with_buffered_final():
    assert classify('runner_cancelled') == ('runner_cancelled', False)
