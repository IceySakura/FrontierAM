"""The hy-live Langfuse tracker files each model call and tool call under the layer that caused it."""

from __future__ import annotations

import json

from types import SimpleNamespace
from typing import Any

import pytest


pytest.importorskip('langfuse', reason='the Langfuse tracker imports the langfuse SDK')

from nexus.extensions.projects.hy_live.langfuse_trace import HyLiveLangfuseTracker  # noqa: E402
from nexus.extensions.projects.hy_live.trace_spans import (  # noqa: E402
    brain_turn_attributes,
    midbrain_request_attributes,
)
from nexus.trace.events import SpanEndEvent  # noqa: E402
from nexus.trace.processors.langfuse_tracker import _AgentContext  # noqa: E402
from nexus.trace.types import Span, Status  # noqa: E402


class FakeObservation:
    def __init__(self, name: str = '', as_type: str = 'span', **kwargs: Any) -> None:
        self.name = name
        self.as_type = as_type
        self.kwargs = kwargs
        self.children: list[FakeObservation] = []
        self.ended = False
        self.end_time: int | None = None
        self._otel_span = SimpleNamespace(_start_time=None)

    def start_observation(self, *, name: str, as_type: str, **kwargs: Any) -> FakeObservation:
        child = FakeObservation(name, as_type, **kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> FakeObservation:
        self.name = kwargs.get('name', self.name)
        self.kwargs.update(kwargs)
        return self

    def end(self, end_time: int | None = None) -> FakeObservation:
        self.ended = True
        self.end_time = end_time
        return self

    def child(self, prefix: str) -> FakeObservation:
        matches = [child for child in self.children if child.name.startswith(prefix)]
        assert len(matches) == 1, (
            f'expected one child starting with {prefix!r} under {self.name!r}, got {[c.name for c in self.children]}'
        )
        return matches[0]


def _tracker(**kwargs: Any) -> tuple[HyLiveLangfuseTracker, FakeObservation]:
    tracker = HyLiveLangfuseTracker(
        public_key='pk', secret_key='sk', host='http://langfuse', flush_on_finalize=False, **kwargs
    )
    trace = FakeObservation('trace')
    agent = trace.start_observation(name='Agent', as_type='span')
    tracker._initialized = True
    tracker._langfuse = SimpleNamespace(create_score=lambda **_: None, flush=lambda: None)
    tracker._trace = trace
    tracker._agent_span = agent
    tracker._ctx = _AgentContext(parent_span=agent)
    tracker._ctx_stack = [tracker._ctx]
    return tracker, trace


_CLOCK = [1_000_000_000_000]


def _completion(
    messages: list[dict[str, Any]],
    *,
    request_meta: dict[str, Any] | None = None,
    text: str = '',
    tool_calls: list[dict[str, Any]] | None = None,
    attributes: dict[str, Any] | None = None,
    tools: bool = True,
    cancelled: bool = False,
) -> Span:
    _CLOCK[0] += 1_000_000_000
    message: dict[str, Any] = {'role': 'assistant', 'content': text}
    if tool_calls:
        message['tool_calls'] = tool_calls
    span_attributes: dict[str, Any] = {
        **(attributes or {}),
        '_client_key': 'main',
        'inputs': {
            'messages': messages,
            'request_meta': request_meta,
            'kwargs': {'tools': [{'type': 'function', 'function': {'name': 'x'}}]} if tools else {},
        },
    }
    if not cancelled:
        span_attributes['outputs'] = {
            'model': 'm',
            'choices': [{'message': message}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12},
        }
    return Span(
        trace_id='t',
        span_id=f's{_CLOCK[0]}',
        parent_span_id='',
        name='openai_completion',
        start_time_unix_nano=_CLOCK[0] - 500_000_000,
        end_time_unix_nano=_CLOCK[0],
        attributes=span_attributes,
        status=Status(code='ERROR', message='CancelledError') if cancelled else Status(code='OK'),
    )


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': call_id,
        'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)},
    }


def _tool_mark(layer: str, call: dict[str, Any], result: Any, *, extra: dict[str, Any] | None = None) -> Span:
    _CLOCK[0] += 1_000_000
    return Span(
        trace_id='t',
        span_id=f'tool{_CLOCK[0]}',
        parent_span_id='',
        name='hy_live_tool',
        start_time_unix_nano=_CLOCK[0],
        end_time_unix_nano=_CLOCK[0],
        attributes={
            **(extra or {}),
            'hy_live.layer': layer,
            'hy_live.tool_call_id': call['id'],
            'hy_live.tool_name': call['function']['name'],
            'hy_live.arguments': call['function']['arguments'],
            'hy_live.result': result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
            'hy_live.started_at': _CLOCK[0] / 1e9,
            'hy_live.ended_at': _CLOCK[0] / 1e9,
        },
    )


def _end(tracker: HyLiveLangfuseTracker, span: Span) -> None:
    tracker.on_end(span, SpanEndEvent(span_id=span.span_id))


def test_layers_nest_by_cause_from_user_turn_to_codebuddy_and_back() -> None:
    tracker, trace = _tracker()
    delegate = _call('c1', 'delegate', {'text': '写个贪吃蛇'})
    user_turn = [{'role': 'system', 'content': 'cerebellum'}, {'role': 'user', 'content': '写个贪吃蛇'}]
    _end(
        tracker,
        _completion(
            user_turn,
            request_meta={'invocation_kind': 'hy_live_cerebellum', 'source': 'turn', 'turn_step': 0},
            tool_calls=[delegate],
        ),
    )
    _end(tracker, _tool_mark('cerebellum', delegate, {'status': 'submitted'}, extra={'hy_live.request_id': 'r1'}))

    request = midbrain_request_attributes(request_id='r1', source='delegate', text='写个贪吃蛇')
    task_create = _call('m1', 'TaskCreate', {'subject': '贪吃蛇', 'description': '写个贪吃蛇'})
    midbrain_input = [{'role': 'system', 'content': 'midbrain'}, {'role': 'user', 'content': '小脑调用了 delegate'}]
    _end(
        tracker,
        _completion(
            midbrain_input,
            request_meta={'invocation_kind': 'hy_live_midbrain', 'source': 'delegate'},
            tool_calls=[task_create],
            attributes=request,
        ),
    )
    accepted = {'status': 'accepted', 'task_id': 'a', 'subject': '贪吃蛇', 'task_status': 'initializing'}
    _end(tracker, _tool_mark('midbrain', task_create, accepted, extra=request))
    reply = _call('m2', 'reply', {'status': 'accepted', 'say': '已受理'})
    _end(
        tracker,
        _completion(
            midbrain_input,
            request_meta={'invocation_kind': 'hy_live_midbrain', 'source': 'delegate'},
            tool_calls=[reply],
            attributes=request,
        ),
    )
    _end(tracker, _tool_mark('midbrain', reply, {'ok': True}, extra=request))

    acknowledge = user_turn + [
        {'role': 'assistant', 'content': ''},
        {'role': 'tool', 'content': '{"status": "submitted"}'},
    ]
    _end(
        tracker,
        _completion(
            acknowledge,
            request_meta={'invocation_kind': 'hy_live_cerebellum', 'source': 'acknowledge', 'turn_step': 0},
            text='我去写个贪吃蛇',
        ),
    )

    brain = brain_turn_attributes(operation_id='task/a/run/b', session_id='sess', prompt='写个贪吃蛇')
    write = _call('b1', 'Write', {'file_path': 'snake.html'})
    brain_input = [{'role': 'system', 'content': 'codebuddy'}, {'role': 'user', 'content': '写个贪吃蛇'}]
    _end(tracker, _completion(brain_input, tool_calls=[write], attributes=brain))
    _end(
        tracker,
        _completion(
            brain_input + [{'role': 'tool', 'tool_call_id': 'b1', 'content': 'written'}],
            text='写好了',
            attributes=brain,
        ),
    )

    conclude = midbrain_request_attributes(request_id='r2', source='工作结束', path='task/a/run/b')
    _end(
        tracker,
        _completion(
            midbrain_input,
            request_meta={'invocation_kind': 'hy_live_midbrain', 'source': '工作结束'},
            tool_calls=[_call('m3', 'reply', {'status': 'ok'})],
            attributes=conclude,
        ),
    )
    notice = '[通知]\n这条消息不是用户说的。\n不要调用工具。\n\n有一项工作做完了。\n工作：贪吃蛇'
    _end(
        tracker,
        _completion(
            acknowledge + [{'role': 'user', 'content': notice}],
            request_meta={'invocation_kind': 'hy_live_cerebellum', 'source': 'observe', 'turn_step': 0},
            text='贪吃蛇做好了',
        ),
    )
    _end(
        tracker,
        _completion(
            [{'role': 'system', 'content': 'user'}, {'role': 'user', 'content': '{}'}],
            request_meta={'invocation_kind': 'hy_live_user_simulator'},
            text='[DONE]',
            tools=False,
        ),
    )
    task_output = {
        'prompt': '写个贪吃蛇',
        'summary': '贪吃蛇做好了',
        'score': None,
        'policy_messages': acknowledge,
        'tasks': [{'task_id': 'task/a', 'objective': '写个贪吃蛇', 'status': 'done', 'activity': []}],
    }
    _end(
        tracker,
        Span(
            trace_id='t',
            span_id='task',
            parent_span_id='',
            name='hy-live',
            start_time_unix_nano=1,
            attributes={'type': 'task', 'outputs': task_output},
        ),
    )
    tracker.finalize()

    cerebellum = trace.child('小脑 Cerebellum')
    first = cerebellum.child('小脑 1 · 用户：写个贪吃蛇 → delegate')
    tool = first.child('Tool: delegate')
    assert tool.kwargs['output'] == '{"status": "submitted"}'
    request_span = tool.child('中脑 · delegate：写个贪吃蛇')
    assert [child.name for child in request_span.children] == ['中脑 1 → TaskCreate', '中脑 2 → reply']
    turn = request_span.child('中脑 1').child('Tool: TaskCreate').child('大脑 · 执行：贪吃蛇')
    assert [child.name for child in turn.children[:2]] == ['大脑 1 → Write', '大脑 2']
    assert turn.child('大脑 1').child('Tool: Write').kwargs['output'] == 'written'
    assert turn.child('中脑 · 工作结束：贪吃蛇').children[0].name == '中脑 3 → reply', (
        'The midbrain summary of a finished turn hangs under that turn'
    )
    assert cerebellum.child('小脑 2').name == '小脑 2 · 承接'
    assert cerebellum.child('小脑 3').name == '小脑 3 · 通知：有一项工作做完了。'
    assert trace.child('用户模拟器').child('用户 1').name == '用户 1 → [DONE]'
    assert all(child.ended for child in trace.children), 'Every layer container is closed at finalize'
    assert trace.ended
    assert first._otel_span._start_time is not None, 'A generation keeps its model call start time'
    assert first.end_time is not None
    prompt_spans = [child for child in cerebellum.children if child.as_type == 'span']
    assert [child.name for child in prompt_spans] == ['System Prompt', 'User Prompt', 'Tools Definition']
    assert all(child._otel_span._start_time < first._otel_span._start_time for child in prompt_spans), (
        'The prompt summaries sort before the first cerebellum call'
    )
    task_result = trace.kwargs['metadata']['task_result']
    assert task_result['summary'] == '贪吃蛇做好了'
    assert task_result['tasks'] == [{'task_id': 'task/a', 'objective': '写个贪吃蛇', 'status': 'done'}]
    assert 'policy_messages' not in task_result, 'The conversation is the trace itself, not trace metadata'


def test_a_cut_short_user_simulator_call_is_marked_unfinished() -> None:
    tracker, trace = _tracker()

    _end(
        tracker,
        _completion(
            [{'role': 'user', 'content': '{}'}],
            request_meta={'invocation_kind': 'hy_live_user_simulator'},
            tools=False,
            cancelled=True,
        ),
    )

    generation = trace.child('用户模拟器').child('用户 1')
    assert generation.name == '用户 1（未完成）'
    assert generation.kwargs['level'] == 'WARNING'
    assert generation.kwargs['status_message'] == 'CancelledError'


def test_a_receipt_notice_is_labelled_by_its_status_and_words() -> None:
    tracker, trace = _tracker()
    notice = '[通知]\n这条消息不是用户说的。\n\nstatus=failed\n账本里没有贷款还款试算这项工作'

    _end(
        tracker,
        _completion(
            [{'role': 'system', 'content': 'c'}, {'role': 'user', 'content': notice}],
            request_meta={'invocation_kind': 'hy_live_cerebellum', 'source': 'observe', 'turn_step': 0},
            text='没找到',
        ),
    )

    assert trace.child('小脑 Cerebellum').child('小脑 1').name == '小脑 1 · 通知·failed：账本里没有贷款还款试算这项工作'


def test_trace_is_named_after_the_episode_index() -> None:
    tracker, _ = _tracker(trace_metadata={'index': 'frontbrain_abc', 'trajectory_id': 't-1'})

    assert tracker._trace_name == 'hy-live · frontbrain_abc'


def test_codebuddy_turn_without_a_known_cause_goes_to_the_background_container() -> None:
    tracker, trace = _tracker()
    brain = brain_turn_attributes(operation_id='task/x/run/y', session_id='sess', prompt='做点事')

    _end(
        tracker,
        _completion(
            [{'role': 'user', 'content': '做点事'}],
            tool_calls=[_call('b1', 'Bash', {'command': 'ls'})],
            attributes=brain,
        ),
    )
    tracker.finalize()

    turn = trace.child('大脑（后台）').child('大脑 · 执行：做点事')
    assert (
        turn.child('大脑 1 → Bash').child('Tool: Bash').kwargs['output']
        == '(turn ended before this result was recorded)'
    )


def test_tool_less_codebuddy_call_is_not_a_step() -> None:
    tracker, trace = _tracker()
    brain = brain_turn_attributes(operation_id='task/x/run/y', session_id='sess', prompt='做点事')

    _end(tracker, _completion([{'role': 'user', 'content': '起个标题'}], text='标题', attributes=brain, tools=False))
    _end(tracker, _completion([{'role': 'user', 'content': '做点事'}], text='好了', attributes=brain))

    turn = trace.child('大脑（后台）').child('大脑 · 执行：做点事')
    assert [child.name for child in turn.children] == ['大脑 · 辅助调用', '大脑 1']


def test_backend_question_turn_hangs_under_its_ask_backend_call() -> None:
    tracker, trace = _tracker()
    request = midbrain_request_attributes(request_id='', source='question', text='进度如何')
    ask = _call('m1', 'ask_backend', {'path': 'task/a', 'question': '现在做到哪了'})
    _end(
        tracker,
        _completion(
            [{'role': 'user', 'content': 'q'}],
            request_meta={'invocation_kind': 'hy_live_midbrain', 'source': 'question'},
            tool_calls=[ask],
            attributes=request,
        ),
    )
    _end(
        tracker,
        _tool_mark(
            'midbrain',
            ask,
            {'status': 'accepted', 'run_path': 'task/a/run/b', 'title': '贪吃蛇', 'question': '现在做到哪了'},
        ),
    )

    brain = brain_turn_attributes(operation_id='question:abc', session_id='sess', prompt='现在做到哪了')
    _end(tracker, _completion([{'role': 'user', 'content': '现在做到哪了'}], text='写了一半', attributes=brain))

    ask_tool = trace.child('中脑（后台）').child('中脑 · question').child('中脑 1').child('Tool: ask_backend')
    assert ask_tool.child('大脑 · 追问：贪吃蛇').children[0].name == '大脑 1'
