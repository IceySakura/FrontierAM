import asyncio
import logging
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.agents import installed_common
from nexus.extensions.agents.installed_common import multiturn
from nexus.extensions.agents.installed_common.stream_trace import StreamTraceBuffer
from nexus.trace import SpanProcessor, get_current_span, processor_context, span_context


class _CaptureSpans(SpanProcessor):
    def __init__(self):
        self.ended = []

    def on_end(self, span, _event):
        self.ended.append(span)


@pytest.mark.asyncio
async def test_close_failure_preserves_default_policy_for_other_agents(monkeypatch):
    async def call(target, **kwargs):
        if target == 'close':
            raise TimeoutError('close failed')
        return {'success': True, 'output': 'done', 'exit_code': 0, 'events': [], 'status': 'completed'}

    monkeypatch.setattr(multiturn, 'call_runtime_agent', call)
    result = await multiturn.run_installed_multiturn(
        SimpleNamespace(session_id='test', user_prompt='go', max_turns=1, close_on_finish=True),
        agent_key='test',
        agent_label='Test',
        session_namespace='test',
        session_turn_target='turn',
        close_session_target='close',
        runtime_kwargs={},
    )
    assert result['success'] is False
    assert 'close failed' in result['stderr']


class _FakeStdin:
    def __init__(self):
        self.closed = False
        self.writes: list[bytes] = []

    def write(self, data):
        self.writes.append(data)

    async def drain(self):
        return None

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class _ErrorStream:
    def __init__(self, message: str = 'stream exploded'):
        self.message = message

    async def readline(self):
        raise ValueError(self.message)


class _BlockingStream:
    async def readline(self):
        await asyncio.Event().wait()
        return b''


class _FakeProc:
    def __init__(self, *, stdout=None, stderr=None, wait_blocks: bool = False):
        self.stdin = _FakeStdin()
        self.stdout = stdout or _BlockingStream()
        self.stderr = stderr or _BlockingStream()
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.wait_blocks = wait_blocks

    async def wait(self):
        if self.wait_blocks:
            await asyncio.Event().wait()
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeProxy:
    instances: list['_FakeProxy'] = []

    def __init__(self, **_kwargs):
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def test_cli_stream_reader_limit_defaults_to_16mb():
    assert installed_common.DEFAULT_CLI_STREAM_READER_LIMIT == 16 * 1024 * 1024


def test_record_trace_event_merges_adjacent_partial_text(monkeypatch):
    captured: list[tuple[str, list]] = []

    monkeypatch.setattr('nexus.trace.get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr(
        'nexus.trace.update_span',
        lambda span_id, *, events: captured.append((span_id, events)),
    )
    session = SimpleNamespace(agent_key='test_agent', session_id='session-1', cwd='/workspace')

    for text in ('hello', ' ', 'world'):
        multiturn._record_trace_event(
            session,
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {'type': 'text_delta', 'text': text},
                },
                'session_id': 'session-1',
            },
        )
    multiturn._record_trace_event(session, {'type': 'result', 'session_id': 'session-1'})

    assert [span_id for span_id, _events in captured] == ['span-1', 'span-1']
    merged = captured[0][1][0]
    assert merged.attributes['event_type'] == 'stream_event'
    assert merged.attributes['event']['event']['delta']['text'] == 'hello world'
    assert captured[1][1][0].attributes['event_type'] == 'result'


def test_record_trace_event_bounds_merged_partial_text(monkeypatch):
    captured: list[list] = []

    monkeypatch.setattr('nexus.trace.get_current_span_id', lambda: 'span-1')
    monkeypatch.setattr('nexus.trace.update_span', lambda _span_id, *, events: captured.append(events))
    session = SimpleNamespace(agent_key='test_agent', session_id='session-1', cwd='/workspace')

    for text in ('a' * (40 * 1024), 'b' * (40 * 1024)):
        multiturn._record_trace_event(
            session,
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {'type': 'text_delta', 'text': text},
                },
            },
        )

    assert len(captured) == 1
    assert captured[0][0].attributes['event']['event']['delta']['text'] == 'a' * (40 * 1024)


@pytest.mark.asyncio
async def test_installed_multiturn_traces_each_runtime_call_without_covering_user_wait(monkeypatch):
    runtime_turns: list[dict] = []
    input_span_names: list[str | None] = []

    async def session_turn_target(**_kwargs):
        raise AssertionError('call_runtime_agent should be used')

    async def close_session_target(**_kwargs):
        raise AssertionError('close_on_finish is disabled')

    async def fake_call_runtime_agent(target, /, **kwargs):
        assert target is session_turn_target
        span = get_current_span()
        assert span is not None
        runtime_turns.append({'span': span.model_copy(deep=True), 'kwargs': kwargs})
        return {
            'success': True,
            'session_id': 'session-1',
            'output': f'turn {len(runtime_turns)}',
            'stderr': '',
            'exit_code': 0,
            'events': [],
        }

    async def get_user_input(_request):
        span = get_current_span()
        input_span_names.append(span.name if span is not None else None)
        return 'second prompt' if len(input_span_names) == 1 else None

    monkeypatch.setattr(multiturn, 'call_runtime_agent', fake_call_runtime_agent)
    input_data = SimpleNamespace(
        session_id='session-1',
        user_prompt='first prompt',
        max_turns=3,
        close_on_finish=False,
        get_user_input=get_user_input,
    )
    processor = _CaptureSpans()

    with processor_context([processor]):
        with span_context('agent.test') as agent_span:
            result = await multiturn.run_installed_multiturn(
                input_data,
                agent_key='test_agent',
                agent_label='Test Agent',
                session_namespace='test-agent-session',
                session_turn_target=session_turn_target,
                close_session_target=close_session_target,
                runtime_kwargs={},
                llm_channel=True,
            )

    turn_spans = [span for span in processor.ended if span.name == 'agent.turn']
    assert result['turns'] == 2
    assert [item['span'].span_id for item in runtime_turns] == [span.span_id for span in turn_spans]
    assert all('turn_index' not in item['kwargs'] for item in runtime_turns)
    assert all(item['kwargs']['no_thinking_request_args'] is None for item in runtime_turns)
    assert all(item['kwargs']['haiku_model'] is None for item in runtime_turns)
    assert all(item['kwargs']['channel_id'] == result['session_id'] for item in runtime_turns)
    assert all(item['kwargs']['persistent_channel'] is True for item in runtime_turns)
    assert all(item['kwargs']['disable_invoke_wait_timeout'] is False for item in runtime_turns)
    runtime_trace_contexts = [item['kwargs']['runtime_trace_context'] for item in runtime_turns]
    assert [context.parent_span_id for context in runtime_trace_contexts] == [agent_span.span_id, agent_span.span_id]
    assert [context.trace_id for context in runtime_trace_contexts] == [
        agent_span.span.trace_id,
        agent_span.span.trace_id,
    ]
    assert [span.parent_span_id for span in turn_spans] == [agent_span.span_id, agent_span.span_id]
    assert [span.attributes['turn_index'] for span in turn_spans] == [1, 2]
    assert all(span.attributes['type'] == 'agent_turn' for span in turn_spans)
    assert all(span.attributes['agent'] == 'test_agent' for span in turn_spans)
    assert all(span.attributes['session_id'] == result['session_id'] for span in turn_spans)
    assert all(span.status.code == 'OK' for span in turn_spans)
    assert input_span_names == ['agent.test', 'agent.test']


@pytest.mark.asyncio
@pytest.mark.parametrize(('turn_frames', 'expected'), [([2, 3], 5), ([None, None], 'absent')])
async def test_installed_multiturn_sums_the_frames_each_turn_relayed(monkeypatch, turn_frames, expected):
    async def fake_call_runtime_agent(_target, /, **_kwargs):
        frames = turn_frames.pop(0)
        return {'success': True, 'output': '', 'stderr': '', 'exit_code': 0, 'events': [], 'relay_frames': frames}

    async def get_user_input(_request):
        return 'next' if turn_frames else None

    monkeypatch.setattr(multiturn, 'call_runtime_agent', fake_call_runtime_agent)
    result = await multiturn.run_installed_multiturn(
        SimpleNamespace(
            session_id='session-1',
            user_prompt='first',
            max_turns=3,
            close_on_finish=False,
            get_user_input=get_user_input,
        ),
        agent_key='test_agent',
        agent_label='Test Agent',
        session_namespace='test-agent-session',
        session_turn_target='unused',
        close_session_target='unused',
        runtime_kwargs={},
    )

    # Other agents build strict output models from this dict, so a run with no relay leaves the key out.
    assert result.get('relay_frames', 'absent') == expected


@pytest.mark.asyncio
async def test_start_session_closes_process_when_initialize_fails(monkeypatch, tmp_path):
    proc = _FakeProc()
    subprocess_kwargs: dict = {}

    async def fake_install_cli(**_kwargs):
        return Path('/tmp/fake-agent')

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        subprocess_kwargs.update(_kwargs)
        return proc

    async def fail_initialize(self, **_kwargs):
        raise RuntimeError('initialize failed')

    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setitem(
        sys.modules,
        'nexus.runtime.server.services.anthropic_proxy',
        SimpleNamespace(AnthropicProxy=_FakeProxy),
    )
    monkeypatch.setattr('nexus.utils.shell.get_os_env_without_ld_library_path', lambda: {})
    monkeypatch.setattr(multiturn.asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
    monkeypatch.setattr(multiturn._StreamJsonSession, 'initialize', fail_initialize)
    _FakeProxy.instances.clear()

    with pytest.raises(RuntimeError, match='initialize failed'):
        await multiturn._start_session(
            agent_key='test_agent',
            agent_label='Test Agent',
            session_id='session-1',
            working_dir=str(tmp_path),
            timeout=10,
            install_cli=fake_install_cli,
            install_dir='/tmp/fake-install',
            install_kwargs=None,
            path_prepend=None,
            extra_env=None,
            extra_cli_args=None,
            config_file_patches=None,
            pre_start_scripts=None,
            conversion_log_level='INFO',
            thinking_preset='hunyuan',
            include_partial_messages=False,
            include_hook_events=False,
            enable_user_input_hooks=True,
            record_trace_events=True,
            command_builder=lambda *_args: ['/tmp/fake-agent'],
        )

    assert proc.stdin.closed is True
    assert proc.returncode == 0
    assert _FakeProxy.instances
    assert _FakeProxy.instances[-1].stopped is True
    assert subprocess_kwargs['limit'] == installed_common.DEFAULT_CLI_STREAM_READER_LIMIT


@pytest.mark.asyncio
async def test_stdout_reader_error_is_reported_without_waiting_for_process():
    proc = _FakeProc(stdout=_ErrorStream('line too long'), wait_blocks=True)
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=proc,
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )

    await asyncio.wait_for(multiturn._read_stdout(session), timeout=0.2)

    event = await asyncio.wait_for(session.events.get(), timeout=0.2)
    assert event['type'] == '_stdout_read_error'
    assert 'line too long' in event['error']
    assert proc.terminated is True


@pytest.mark.asyncio
async def test_send_user_message_fails_fast_on_stdout_reader_error():
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    await session.events.put({'type': '_stdout_read_error', 'error': 'line too long'})

    with pytest.raises(RuntimeError, match='stdout reader failed: line too long'):
        await session.send_user_message('hello', timeout=1)


@pytest.mark.asyncio
async def test_session_turn_flushes_pending_stream_trace_when_turn_fails(monkeypatch):
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    session._stream_trace_buffer = StreamTraceBuffer(
        pending_span_id='span-1',
        pending_event={'type': 'stream_event', 'event': {'delta': {'text': 'final'}}},
        pending_key=('session-1', None, 0),
        pending_text_bytes=5,
    )
    captured: list[str] = []

    async def fail_turn(*_args, **_kwargs):
        raise RuntimeError('turn failed')

    monkeypatch.setattr(session, 'send_user_message', fail_turn)
    monkeypatch.setattr('nexus.trace.update_span', lambda span_id, *, events: captured.append(span_id))
    multiturn._SESSIONS[('test_agent', 'session-1')] = session
    try:
        result = await multiturn.run_installed_cli_session_turn(
            agent_key='test_agent',
            agent_label='Test Agent',
            session_id='session-1',
            prompt='hello',
            install_cli=lambda **_kwargs: None,
            install_dir='/tmp/not-used',
        )
    finally:
        multiturn._SESSIONS.pop(('test_agent', 'session-1'), None)

    assert result['success'] is False
    assert captured == ['span-1']


@pytest.mark.asyncio
async def test_stderr_reader_error_is_recorded_and_reported():
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(stderr=_ErrorStream('stderr exploded')),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )

    await multiturn._read_stderr(session)

    assert session.stderr_lines[-1] == 'stderr reader failed: ValueError: stderr exploded'
    event = await asyncio.wait_for(session.events.get(), timeout=0.2)
    assert event['type'] == '_stderr_read_error'


def test_short_hash_is_stable_and_distinguishes_turns():
    # Identical input -> identical hash; different input -> different hash. This is what
    # lets log readers spot an off-by-one turn (its output_hash equals the previous turn's).
    assert multiturn._short_hash('turn-1 output') == multiturn._short_hash('turn-1 output')
    assert multiturn._short_hash('turn-1 output') != multiturn._short_hash('turn-2 output')
    assert multiturn._short_hash('') == 'empty'
    # Non-string values are hashed via canonical JSON (key order independent).
    assert multiturn._short_hash({'a': 1, 'b': 2}) == multiturn._short_hash({'b': 2, 'a': 1})


@pytest.mark.asyncio
async def test_send_user_message_drains_stale_events_and_returns_fresh_result(caplog):
    """Regression for the multi-turn off-by-one: leftover events from a previous turn are
    drained *before* the new prompt is sent, so this turn returns its own fresh artifact
    rather than the previous turn's stale one.
    """
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    # Leftover events from the previous turn sitting in the shared queue.
    await session.events.put({
        'type': 'assistant',
        'message': {'role': 'assistant', 'content': 'PREV TURN artifact'},
        'session_id': 'session-1',
    })
    await session.events.put({'type': 'result', 'result': 'PREV TURN artifact', 'session_id': 'session-1'})

    async def deliver_fresh_after_prompt_sent():
        # The fresh turn's events only arrive once the new prompt has actually been written,
        # i.e. after the stale leftovers were drained. We wait past the post-write drain
        # quiet window so the fresh events cannot be misclassified as stale leftovers.
        for _ in range(200):
            if any(b'turn-2 prompt' in write for write in session.proc.stdin.writes):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(multiturn._TURN_POST_WRITE_QUIET_WINDOW * 3)
        await session.events.put({
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': 'FRESH turn-2 artifact'},
            'session_id': 'session-1',
        })
        await session.events.put({'type': 'result', 'result': 'FRESH turn-2 artifact', 'session_id': 'session-1'})

    logger_name = 'nexus.extensions.agents.installed_common.multiturn'
    with caplog.at_level(logging.WARNING, logger=logger_name):
        producer = asyncio.create_task(deliver_fresh_after_prompt_sent())
        result = await session.send_user_message('turn-2 prompt', timeout=5)
        await producer

    # Off-by-one is fixed: this turn returns its OWN artifact, not the previous turn's.
    assert result['output'] == 'FRESH turn-2 artifact'
    assert result['success'] is True
    assert any(b'turn-2 prompt' in write for write in session.proc.stdin.writes)
    # The stale leftovers were fully consumed by the drain (queue empty before fresh events).
    warnings = ' '.join(record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING)
    assert 'DRAINED' in warnings
    assert 'saw_result=True' in warnings


@pytest.mark.asyncio
async def test_drain_stale_events_clears_queue_and_reports_counts():
    """`_drain_stale_events` removes every leftover event and reports whether a result was seen."""
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    await session.events.put({'type': 'assistant', 'message': {'role': 'assistant', 'content': 'x'}})
    await session.events.put({'type': 'stream_event', 'data': {}})
    await session.events.put({'type': 'result', 'result': 'x', 'session_id': 'session-1'})

    drained, saw_result = await session._drain_stale_events()

    assert drained == 3
    assert saw_result is True
    assert session.events.empty()


@pytest.mark.asyncio
async def test_drain_stale_events_noop_on_empty_queue():
    """Draining an already-empty queue is a fast no-op (healthy turn path adds no latency)."""
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )

    drained, saw_result = await session._drain_stale_events()

    assert drained == 0
    assert saw_result is False
    assert session.events.empty()


@pytest.mark.asyncio
async def test_drain_stale_events_preserves_failure_sentinel():
    """A reader failure sentinel must survive the drain so the read loop can still raise it."""
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    await session.events.put({'type': 'assistant', 'message': {'role': 'assistant', 'content': 'stale'}})
    await session.events.put({'type': '_stdout_read_error', 'error': 'line too long'})

    drained, saw_result = await session._drain_stale_events()

    # Ordinary leftover discarded, but the sentinel is re-queued (drain stops at it).
    assert drained == 1
    assert saw_result is False
    remaining = session.events.get_nowait()
    assert remaining['type'] == '_stdout_read_error'


@pytest.mark.asyncio
async def test_send_user_message_no_stale_warning_on_clean_queue(caplog):
    """A healthy turn (empty queue, events arrive after the prompt) must not warn."""
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )

    async def deliver_events_after_send():
        # Simulate the CLI producing a fresh result only after some processing latency.
        await asyncio.sleep(0.6)
        await session.events.put({
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': 'FRESH artifact'},
            'session_id': 'session-1',
        })
        await session.events.put({'type': 'result', 'result': 'FRESH artifact', 'session_id': 'session-1'})

    logger_name = 'nexus.extensions.agents.installed_common.multiturn'
    with caplog.at_level(logging.WARNING, logger=logger_name):
        producer = asyncio.create_task(deliver_events_after_send())
        result = await session.send_user_message('turn-1 prompt', timeout=5)
        await producer

    assert result['output'] == 'FRESH artifact'
    warnings = ' '.join(record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING)
    assert 'STALE' not in warnings
    assert 'FAST-RETURN' not in warnings


@pytest.mark.asyncio
async def test_send_user_message_post_write_drain_blocks_off_by_one_from_write_drain_race(caplog):
    """Regression for CR on fix/proxy-live-multiturn: when ``qsize() == 0`` at pre-write
    time, a trailing event from the previous turn can still land in the queue *during*
    ``await stdin.drain()`` (which yields the event loop). Without the post-write drain
    the read loop would consume that stale event as this turn's fresh reply. Verify the
    post-write drain drops it and ``send_user_message`` waits for the genuine fresh events.
    """
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    assert session.events.qsize() == 0  # precondition: no pre-write stale events

    # Reproduce the race per the CR: the queue is EMPTY at pre-write time, and the stale
    # events from the previous turn only reach ``self.events`` during ``stdin.drain()``.
    stale_leftovers = [
        {
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': 'PREV TURN artifact'},
            'session_id': 'session-1',
        },
        {'type': 'result', 'result': 'PREV TURN artifact', 'session_id': 'session-1'},
    ]

    original_drain = session.proc.stdin.drain
    injected: dict[str, bool] = {'done': False}

    async def drain_that_enqueues_stale_leftover():
        # Model the observed race: the stdout reader task runs while stdin.drain()
        # yields the event loop and pushes a trailing event from the previous turn.
        await original_drain()
        if not injected['done']:
            injected['done'] = True
            for event in stale_leftovers:
                await session.events.put(event)

    session.proc.stdin.drain = drain_that_enqueues_stale_leftover

    async def deliver_fresh_after_post_drain_window():
        # Wait until the prompt has been written AND past the post-write quiet window, so
        # the post-write drain has already emptied the stale leftovers and returned.
        for _ in range(500):
            if any(b'turn-2 prompt' in write for write in session.proc.stdin.writes):
                break
            await asyncio.sleep(0.01)
        # Sleep past the post-write drain window so the fresh events are never inside it.
        await asyncio.sleep(multiturn._TURN_POST_WRITE_QUIET_WINDOW * 4)
        await session.events.put({
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': 'FRESH turn-2 artifact'},
            'session_id': 'session-1',
        })
        await session.events.put({'type': 'result', 'result': 'FRESH turn-2 artifact', 'session_id': 'session-1'})

    logger_name = 'nexus.extensions.agents.installed_common.multiturn'
    with caplog.at_level(logging.WARNING, logger=logger_name):
        producer = asyncio.create_task(deliver_fresh_after_post_drain_window())
        result = await session.send_user_message('turn-2 prompt', timeout=5)
        await producer

    # Post-write drain blocked the off-by-one: this turn returns its OWN artifact, not the
    # stale one that arrived during stdin.drain().
    assert result['output'] == 'FRESH turn-2 artifact'
    assert result['success'] is True
    assert injected['done'] is True  # the race actually happened
    # Diagnostic: the guard fired and logged that it dropped a stale ``result``.
    warnings = ' '.join(record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING)
    assert 'POST-WRITE-DRAINED' in warnings
    assert 'saw_result=True' in warnings


@pytest.mark.asyncio
async def test_drain_post_write_stale_events_drops_leftovers_but_preserves_sentinel():
    """The post-write drain must clear ordinary leftover events yet keep reader-failure
    sentinels in the queue so the read loop can still raise them."""
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )
    await session.events.put({'type': 'assistant', 'message': {'role': 'assistant', 'content': 'stale'}})
    await session.events.put({'type': 'result', 'result': 'stale', 'session_id': 'session-1'})
    await session.events.put({'type': '_stdout_read_error', 'error': 'line too long'})

    drained, saw_result = await session._drain_post_write_stale_events()

    assert drained == 2
    assert saw_result is True
    remaining = session.events.get_nowait()
    assert remaining['type'] == '_stdout_read_error'
    assert session.events.empty()


@pytest.mark.asyncio
async def test_drain_post_write_stale_events_noop_on_empty_queue_is_bounded():
    """No leftover -> drain reports zero and returns promptly (bounded by the quiet window)."""
    session = multiturn._StreamJsonSession(
        agent_key='test_agent',
        agent_label='Test Agent',
        session_id='session-1',
        proc=_FakeProc(),
        proxy=_FakeProxy(),
        cwd='/workspace',
        started_at=0,
    )

    started = asyncio.get_running_loop().time()
    drained, saw_result = await session._drain_post_write_stale_events()
    elapsed = asyncio.get_running_loop().time() - started

    assert drained == 0
    assert saw_result is False
    # Post-write drain uses a single short sleep; must not exceed a small multiple of the
    # quiet window (guards against accidental regression to unbounded loops).
    assert elapsed < multiturn._TURN_POST_WRITE_QUIET_WINDOW * 4
