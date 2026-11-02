"""Render one hy-live episode in Langfuse as its layers: cerebellum, midbrain, CodeBuddy and the user.

nexus's ``langfuse_tracker`` draws one agent: the first tool-bearing system prompt is the main
agent and every other call becomes one of its steps, with tool results read off the next call.
A hy-live episode runs four models at once, so that view interleaves them in one line and pins
each tool result on whichever call came next. This tracker keeps the base trace, its Taiji
metadata and upload, and files every call by who made it::

    Trace
    ├── 小脑 Cerebellum                         every policy call, in order
    │     ├── 小脑 1 · 用户：写个贪吃蛇 → delegate
    │     │     └── Tool: delegate              {"status": "submitted"}
    │     │           └── 中脑 · delegate       the midbrain request the call queued
    │     │                 ├── 中脑 1 → TaskCreate
    │     │                 │     └── Tool: TaskCreate
    │     │                 │           └── 大脑 · 贪吃蛇        the CodeBuddy turn it started
    │     │                 │                 └── 大脑 1 → Write
    │     │                 └── 中脑 2 → reply
    │     ├── 小脑 2 · 承接
    │     └── 小脑 3 · 通知：有一项工作做完了。
    ├── 用户模拟器 User Simulator
    └── 中脑（后台） / 大脑（后台）              work no visible call caused

A midbrain request that concludes a CodeBuddy turn hangs under that turn. Generations keep the
model call's own start and end time, so Langfuse's timeline shows how the layers overlap.

Selected by HyTuner through ``rollout.agent.nexus_langfuse_processor=hy_live_langfuse_tracker``.
"""

from __future__ import annotations

import contextlib
import json
import logging

from dataclasses import dataclass, field
from typing import Any

from hylive.midbrain_loop import CONCLUDE_SOURCE
from hylive.notices import RECEIPT_STATUS_PREFIX

from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import span_processor_registry
from nexus.trace.processors.langfuse_tracker import (
    LangfuseTrackerProcessor,
    _convert_usage,
    _extract_model_parameters,
    _extract_numeric_score,
    _extract_text_content,
    _extract_tools_summary,
    _parse_response,
    _set_span_start_time,
    _suppressed_otel_detach_log,
    _truncate_message,
)
from nexus.trace.types import Span

from .trace_spans import (
    BRAIN_LAYER,
    CEREBELLUM_INVOCATION,
    CEREBELLUM_LAYER,
    LAYER_ATTRIBUTE,
    MIDBRAIN_INVOCATION,
    MIDBRAIN_LAYER,
    OPERATION_ATTRIBUTE,
    PROMPT_ATTRIBUTE,
    REQUEST_ATTRIBUTE,
    REQUEST_PATH_ATTRIBUTE,
    REQUEST_SOURCE_ATTRIBUTE,
    REQUEST_TEXT_ATTRIBUTE,
    TOOL_SPAN_NAME,
    USER_SIMULATOR_INVOCATION,
)


logger = logging.getLogger(__name__)

CEREBELLUM_NAME = '小脑 Cerebellum'
USER_SIMULATOR_NAME = '用户模拟器 User Simulator'
MIDBRAIN_BACKGROUND_NAME = '中脑（后台）'
BRAIN_BACKGROUND_NAME = '大脑（后台）'
OTHER_CALLS_NAME = '其他模型调用'
_LABEL_CHARS = 24
_QUESTION_OPERATION_PREFIX = 'question:'
_UNANSWERED_TOOL_OUTPUT = '(turn ended before this result was recorded)'
_PROMPT_SPAN_LEAD_NS = 3_000_000
"""How far before the first cerebellum call its System Prompt, User Prompt and Tools spans start."""


def _task_key(operation: str) -> str:
    """``task/<id>`` of a turn's operation (``task/<id>/run/<run>[#n]``): the midbrain only ever sees Task ids."""
    parts = operation.split('#', 1)[0].split('/')
    return '/'.join(parts[:2]) if len(parts) >= 2 and parts[0] == 'task' else operation


def _short(text: Any, limit: int = _LABEL_CHARS) -> str:
    flat = ' '.join(str(text or '').split())
    return flat if len(flat) <= limit else flat[: limit - 1] + '…'


def _ns(seconds: Any) -> int | None:
    return int(float(seconds) * 1e9) if isinstance(seconds, (int, float)) else None


def _json_or_raw(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _injected_label(content: str) -> str:
    """``通知：<first line of the fact>`` for an injected ``[通知]`` / ``[失败]`` message.

    A midbrain receipt's fact starts with ``status=<status>``; its label carries the status and the
    line after it instead.
    """
    head = content.split('\n', 1)[0].strip('[]')
    fact = content.split('\n\n', 1)[1] if '\n\n' in content else ''
    lines = [line for line in fact.strip().split('\n') if line.strip()]
    if not lines:
        return head
    if lines[0].startswith(RECEIPT_STATUS_PREFIX):
        status = lines[0].removeprefix(RECEIPT_STATUS_PREFIX)
        return f'{head}·{status}：{_short(lines[1] if len(lines) > 1 else "", 20)}'
    return f'{head}：{_short(lines[0], 20)}'


def _task_summary(task_output: dict[str, Any] | None) -> dict[str, Any] | None:
    """The part of the hy-live task output worth keeping on the trace; the conversation is the trace itself."""
    if not task_output:
        return None
    return {
        'prompt': task_output.get('prompt'),
        'summary': task_output.get('summary'),
        'score': task_output.get('score'),
        'metadata': task_output.get('metadata'),
        'tasks': [
            {key: task.get(key) for key in ('task_id', 'objective', 'status')}
            for task in task_output.get('tasks') or []
            if isinstance(task, dict)
        ],
    }


@dataclass
class _Container:
    """One Langfuse span holding a run of generations, ended at its last generation's end."""

    span: Any
    last_end_ns: int | None = None
    step: int = 0
    prev_generation: Any = None
    prev_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ended: bool = False

    def close(self) -> None:
        if not self.ended:
            self.ended = True
            self.span.end(end_time=self.last_end_ns)


@span_processor_registry.register('hy_live_langfuse_tracker')
class HyLiveLangfuseTracker(LangfuseTrackerProcessor):
    """Langfuse tracker that draws a hy-live episode as its cerebellum, midbrain, CodeBuddy and user layers."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not kwargs.get('trace_name'):
            # Every trajectory of a run shares the run's default name; the episode's data index tells them apart.
            episode = self._trace_metadata.get('index') or self._trace_metadata.get('trajectory_id')
            if episode:
                self._trace_name = f'hy-live · {episode}'
        self._named_cerebellum = False
        self._generations_by_call: dict[tuple[str, str], Any] = {}
        self._tool_by_request: dict[str, Any] = {}
        self._tool_by_operation: dict[str, Any] = {}
        self._title_by_operation: dict[str, str] = {}
        self._tool_by_question: list[tuple[str, Any, str]] = []
        self._midbrain_requests: dict[str, _Container] = {}
        self._open_midbrain_request: _Container | None = None
        self._midbrain_step = 0
        self._brain_turns: dict[str, _Container] = {}
        self._backgrounds: dict[str, _Container] = {}
        self._layer_usage: dict[str, dict[str, int]] = {}

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.attributes.get('type') == 'task':
            super().on_end(span, event)
            return
        if span.name not in (TOOL_SPAN_NAME, 'openai_completion') or not self._ensure_init():
            return
        try:
            if span.name == TOOL_SPAN_NAME:
                self._on_tool(span)
            else:
                self._on_completion(span)
        except Exception:
            logger.warning('hy-live Langfuse rendering failed for span %s', span.name, exc_info=True)

    # ---- model calls ----

    def _on_completion(self, span: Span) -> None:
        inputs = span.attributes.get('inputs') or {}
        messages = inputs.get('messages') or []
        if not messages:
            return
        raw_request_meta = inputs.get('request_meta')
        request_meta: dict[str, Any] = raw_request_meta if isinstance(raw_request_meta, dict) else {}
        invocation = request_meta.get('invocation_kind')
        if invocation == CEREBELLUM_INVOCATION:
            self._on_cerebellum(span, inputs, messages, request_meta)
        elif invocation == MIDBRAIN_INVOCATION:
            self._on_midbrain(span, inputs, messages)
        elif invocation == USER_SIMULATOR_INVOCATION:
            container = self._background(USER_SIMULATOR_NAME, span)
            container.step += 1
            text = _parse_response(span.attributes.get('outputs'))[0]
            label = f'用户 {container.step} → {_short(text)}' if text else f'用户 {container.step}'
            self._generation(container, span, inputs, messages, 'user_simulator', label)
        elif span.attributes.get(LAYER_ATTRIBUTE) == BRAIN_LAYER:
            self._on_brain(span, inputs, messages)
        else:
            container = self._background(OTHER_CALLS_NAME, span)
            container.step += 1
            self._generation(container, span, inputs, messages, 'other', f'调用 {container.step}')

    def _on_cerebellum(
        self, span: Span, inputs: dict[str, Any], messages: list[dict[str, Any]], request_meta: dict[str, Any]
    ) -> None:
        ctx = self._ctx_stack[0]
        if not self._named_cerebellum:
            self._named_cerebellum = True
            self._agent_span.update(name=CEREBELLUM_NAME)
            self._tools_summary = _extract_tools_summary(inputs)
            self._model_parameters = _extract_model_parameters(inputs)
            self._cerebellum_prompt_spans(span, inputs, messages)
        ctx.step += 1
        self._global_step += 1
        tool_calls = _parse_response(span.attributes.get('outputs'))[2]
        label = f'小脑 {ctx.step} · {self._cerebellum_trigger(request_meta, messages)}'
        if tool_calls:
            label += f' → {", ".join(call["name"] for call in tool_calls)}'
        cerebellum = _Container(span=self._agent_span)
        generation, text = self._generation(
            cerebellum, span, inputs, messages, CEREBELLUM_LAYER, label, request_meta=request_meta
        )
        if text:
            ctx.last_output = text
        model = (span.attributes.get('outputs') or {}).get('model') or inputs.get('model')
        if model:
            self._model_name = str(model)
        for call in tool_calls:
            self._generations_by_call[(CEREBELLUM_LAYER, str(call.get('id') or ''))] = generation

    def _cerebellum_prompt_spans(self, span: Span, inputs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        """System Prompt, User Prompt and Tools Definition, timed just before the first cerebellum call."""
        system = next((m.get('content') for m in messages if m.get('role') == 'system'), '')
        user = next((m.get('content') for m in messages if m.get('role') == 'user'), '')
        tools = (inputs.get('kwargs') or {}).get('tools') or []
        first_ns = span.start_time_unix_nano
        parts: list[tuple[str, dict[str, Any]]] = [
            ('System Prompt', {'content': self._trunc_content(_extract_text_content(system))}),
            ('User Prompt', {'task': self._trunc_content(_extract_text_content(user))}),
        ]
        if tools:
            parts.append(('Tools Definition', {'tools': tools}))
        for offset, (name, payload) in enumerate(parts):
            prompt_span = self._agent_span.start_observation(name=name, as_type='span', input=payload)
            if first_ns:
                _set_span_start_time(prompt_span, first_ns - _PROMPT_SPAN_LEAD_NS + offset)
            prompt_span.end(end_time=first_ns - _PROMPT_SPAN_LEAD_NS + offset + 1 if first_ns else None)
        self._trace.update(input={'task': self._trunc_content(_extract_text_content(user))})

    @staticmethod
    def _cerebellum_trigger(request_meta: dict[str, Any], messages: list[dict[str, Any]]) -> str:
        source = request_meta.get('source')
        last = messages[-1] if messages else {}
        content = last.get('content')
        if source == 'turn' and request_meta.get('turn_step', 0) == 0:
            spoken = content if isinstance(content, str) else '（语音）'
            return f'用户：{_short(spoken)}'
        if source == 'turn':
            return '继续'
        if source == 'acknowledge':
            return '承接'
        if source in ('observe', 'failure') and isinstance(content, str):
            return _injected_label(content)
        return str(source or '调用')

    def _on_midbrain(self, span: Span, inputs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        request_id = str(span.attributes.get(REQUEST_ATTRIBUTE) or '')
        container = self._midbrain_requests.get(request_id) if request_id else None
        if container is None:
            container = self._open_midbrain(span, request_id)
        self._midbrain_step += 1
        tool_calls = _parse_response(span.attributes.get('outputs'))[2]
        label = f'中脑 {self._midbrain_step}'
        if tool_calls:
            label += f' → {", ".join(call["name"] for call in tool_calls)}'
        generation, _ = self._generation(container, span, inputs, messages, MIDBRAIN_LAYER, label)
        for call in tool_calls:
            self._generations_by_call[(MIDBRAIN_LAYER, str(call.get('id') or ''))] = generation

    def _open_midbrain(self, span: Span, request_id: str) -> _Container:
        # The midbrain serves one request at a time, so a new request means the previous one is done.
        if self._open_midbrain_request is not None:
            self._open_midbrain_request.close()
        source = str(span.attributes.get(REQUEST_SOURCE_ATTRIBUTE) or '')
        text = str(span.attributes.get(REQUEST_TEXT_ATTRIBUTE) or '')
        path = str(span.attributes.get(REQUEST_PATH_ATTRIBUTE) or '')
        parent = self._tool_by_request.pop(request_id, None)
        if parent is None and source == CONCLUDE_SOURCE and path in self._brain_turns:
            parent = self._brain_turns[path].span
        if parent is None:
            parent = self._background(MIDBRAIN_BACKGROUND_NAME, span).span
        if source == CONCLUDE_SOURCE:
            name = f'中脑 · {CONCLUDE_SOURCE}：{_short(self._title_by_operation.get(_task_key(path)) or path)}'
        elif text:
            name = f'中脑 · {source}：{_short(text)}'
        else:
            name = f'中脑 · {source}'
        observation = parent.start_observation(
            name=name,
            as_type='agent',
            input={'source': source, 'text': text, 'path': path},
            metadata={'layer': MIDBRAIN_LAYER, 'request_id': request_id},
        )
        if span.start_time_unix_nano:
            _set_span_start_time(observation, span.start_time_unix_nano)
        container = _Container(span=observation)
        if request_id:
            self._midbrain_requests[request_id] = container
        self._open_midbrain_request = container
        return container

    def _on_brain(self, span: Span, inputs: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        operation = str(span.attributes.get(OPERATION_ATTRIBUTE) or '')
        turn = self._brain_turns.get(operation)
        if turn is None:
            turn = self._open_brain_turn(span, operation)
        tool_calls = _parse_response(span.attributes.get('outputs'))[2]
        if not (inputs.get('kwargs') or {}).get('tools'):
            # A tool-less helper call (e.g. a title) is not one of the agent's steps.
            self._generation(turn, span, inputs, messages, BRAIN_LAYER, '大脑 · 辅助调用')
            return
        self._attach_brain_results(turn, messages)
        turn.step += 1
        label = f'大脑 {turn.step}'
        if tool_calls:
            label += f' → {", ".join(call["name"] for call in tool_calls[:4])}{"…" if len(tool_calls) > 4 else ""}'
        generation, _ = self._generation(turn, span, inputs, messages, BRAIN_LAYER, label)
        turn.prev_generation = generation
        turn.prev_tool_calls = tool_calls

    def _open_brain_turn(self, span: Span, operation: str) -> _Container:
        prompt = str(span.attributes.get(PROMPT_ATTRIBUTE) or '')
        parent = self._tool_by_operation.get(_task_key(operation))
        title = self._title_by_operation.get(_task_key(operation), '')
        if parent is None and operation.startswith(_QUESTION_OPERATION_PREFIX):
            for question, observation, question_title in reversed(self._tool_by_question):
                if question and question == prompt:
                    parent, title = observation, question_title
                    break
        if parent is None:
            parent = self._background(BRAIN_BACKGROUND_NAME, span).span
        kind = '追问' if operation.startswith(_QUESTION_OPERATION_PREFIX) else '执行'
        observation = parent.start_observation(
            name=f'大脑 · {kind}：{_short(title or prompt)}',
            as_type='agent',
            input={'prompt': prompt},
            metadata={
                'layer': BRAIN_LAYER,
                'operation_id': operation,
                'session_id': span.attributes.get('hy_live.session_id'),
            },
        )
        if span.start_time_unix_nano:
            _set_span_start_time(observation, span.start_time_unix_nano)
        turn = _Container(span=observation)
        self._brain_turns[operation] = turn
        return turn

    def _attach_brain_results(self, turn: _Container, messages: list[dict[str, Any]]) -> None:
        if turn.prev_generation is None or not turn.prev_tool_calls:
            return
        results = {
            message.get('tool_call_id', ''): _extract_text_content(message.get('content', ''))
            for message in messages
            if message.get('role') == 'tool'
        }
        for call in turn.prev_tool_calls:
            self._brain_tool(turn.prev_generation, call, results.get(call.get('id', ''), ''))
        turn.prev_generation, turn.prev_tool_calls = None, []

    def _brain_tool(self, generation: Any, call: dict[str, Any], output: str) -> None:
        arguments = call.get('arguments')
        tool = generation.start_observation(
            name=f'Tool: {call["name"]}',
            as_type='tool',
            input={key: self._trunc_tool(value) for key, value in arguments.items()}
            if isinstance(arguments, dict)
            else {'raw': self._trunc_tool(arguments)},
            output=self._trunc_tool(output),
            metadata={'layer': BRAIN_LAYER, 'tool_call_id': call.get('id', '')},
        )
        tool.end()

    def _generation(
        self,
        container: _Container,
        span: Span,
        inputs: dict[str, Any],
        messages: list[dict[str, Any]],
        layer: str,
        label: str,
        *,
        request_meta: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        """Upload one model call under ``container`` with its own start and end time.

        A call that ended without a response, e.g. a user simulator call a notice cut short, is
        labelled unfinished and marked as a warning.
        """
        outputs = span.attributes.get('outputs')
        text, thinking, tool_calls, usage, _, _ = _parse_response(outputs)
        unfinished = outputs is None or span.status.code == 'ERROR'
        if unfinished:
            label = f'{label}（未完成）'
        gen_input: dict[str, Any] = {
            'messages': [_truncate_message(m, self._max_content_chars, self._max_tool_chars) for m in messages],
            'total_messages': len(messages),
        }
        tools = (inputs.get('kwargs') or {}).get('tools')
        if tools:
            gen_input['tools'] = tools
        gen_output: dict[str, Any] = {}
        if thinking:
            gen_output['thinking'] = self._trunc_content(thinking)
        if text:
            gen_output['text'] = self._trunc_content(text)
        if tool_calls:
            gen_output['tool_calls'] = [{'name': call['name'], 'args': call['arguments']} for call in tool_calls]
        lf_usage = _convert_usage(usage)
        layer_usage = self._layer_usage.setdefault(layer, {'input': 0, 'output': 0, 'total': 0, 'cached_tokens': 0})
        for key in layer_usage:
            layer_usage[key] += lf_usage.get(key, 0)
            self._total_usage[key] += lf_usage.get(key, 0)
        metadata: dict[str, Any] = {
            **self._episode_metadata_fields(),
            'layer': layer,
            'client': span.attributes.get('_client_key'),
            'tool_names': [call['name'] for call in tool_calls],
            'source': 'realtime',
        }
        if request_meta:
            metadata['invocation'] = {key: value for key, value in request_meta.items() if key != 'invocation_kind'}
        if span.start_time_unix_nano and span.end_time_unix_nano:
            metadata['duration_s'] = round((span.end_time_unix_nano - span.start_time_unix_nano) / 1e9, 2)
        model = (outputs or {}).get('model') if isinstance(outputs, dict) else None
        generation = container.span.start_observation(
            name=label,
            as_type='generation',
            model=str(model or inputs.get('model') or ''),
            model_parameters=_extract_model_parameters(inputs) or None,
            input=gen_input,
            output=gen_output,
            usage_details=lf_usage if lf_usage.get('total', 0) > 0 else None,
            metadata=metadata,
            level='WARNING' if unfinished else None,
            status_message=(span.status.message or 'ended without a response') if unfinished else None,
        )
        if span.start_time_unix_nano:
            _set_span_start_time(generation, span.start_time_unix_nano)
        generation.end(end_time=span.end_time_unix_nano or None)
        container.last_end_ns = span.end_time_unix_nano or container.last_end_ns
        return generation, text

    def _background(self, name: str, span: Span) -> _Container:
        container = self._backgrounds.get(name)
        if container is None:
            observation = self._trace.start_observation(name=name, as_type='agent', metadata={'source': 'realtime'})
            if span.start_time_unix_nano:
                _set_span_start_time(observation, span.start_time_unix_nano)
            container = _Container(span=observation)
            self._backgrounds[name] = container
        return container

    # ---- tool calls ----

    def _on_tool(self, span: Span) -> None:
        attributes = span.attributes
        layer = str(attributes.get(LAYER_ATTRIBUTE) or '')
        name = str(attributes.get('hy_live.tool_name') or '')
        call_id = str(attributes.get('hy_live.tool_call_id') or '')
        request_id = str(attributes.get(REQUEST_ATTRIBUTE) or '')
        generation = self._generations_by_call.pop((layer, call_id), None)
        if generation is None:
            # A call whose generation was never seen still shows up, on its layer's own container.
            generation = (
                self._agent_span if layer == CEREBELLUM_LAYER else self._background(MIDBRAIN_BACKGROUND_NAME, span).span
            )
        arguments = _json_or_raw(attributes.get('hy_live.arguments'))
        result = str(attributes.get('hy_live.result') or '')
        started_ns, ended_ns = _ns(attributes.get('hy_live.started_at')), _ns(attributes.get('hy_live.ended_at'))
        tool = generation.start_observation(
            name=f'Tool: {name}',
            as_type='tool',
            input={key: self._trunc_tool(value) for key, value in arguments.items()}
            if isinstance(arguments, dict)
            else {'raw': self._trunc_tool(arguments)},
            output=self._trunc_tool(result),
            metadata={'layer': layer, 'tool_call_id': call_id, 'request_id': request_id, 'source': 'realtime'},
        )
        if started_ns:
            _set_span_start_time(tool, started_ns)
        tool.end(end_time=ended_ns)
        if layer == CEREBELLUM_LAYER and request_id:
            self._tool_by_request[request_id] = tool
        if layer == MIDBRAIN_LAYER:
            self._remember_operation(name, arguments, _json_or_raw(result), tool)

    def _remember_operation(self, name: str, arguments: Any, result: Any, tool: Any) -> None:
        """Keep the ledger calls that start CodeBuddy turns, so each turn hangs under its call."""
        if not isinstance(result, dict) or result.get('status') != 'accepted':
            return
        if name in ('TaskCreate', 'TaskUpdate') and result.get('task_id'):
            # A TaskUpdate steers the same Run, so the turns after it hang under the latest call.
            task_path = f'task/{result["task_id"]}'
            self._tool_by_operation[task_path] = tool
            self._title_by_operation[task_path] = str(result.get('subject') or '')
        elif name == 'ask_backend':
            question = result.get('question') or (arguments.get('question') if isinstance(arguments, dict) else '')
            self._tool_by_question.append((str(question or ''), tool, str(result.get('title') or '')))

    # ---- episode end ----

    def finalize(self) -> None:
        if not self._langfuse or not self._trace:
            return
        try:
            for turn in self._brain_turns.values():
                if turn.prev_generation is not None:
                    for call in turn.prev_tool_calls:
                        self._brain_tool(turn.prev_generation, call, _UNANSWERED_TOOL_OUTPUT)
                turn.close()
            for container in [*self._midbrain_requests.values(), *self._backgrounds.values()]:
                container.close()
            if self._open_midbrain_request is not None:
                self._open_midbrain_request.close()

            ctx = self._ctx_stack[0]
            score_value = _extract_numeric_score(self._task_output)
            summary = {
                'cerebellum_steps': ctx.step,
                'midbrain_steps': self._midbrain_step,
                'brain_turns': len(self._brain_turns),
                'layer_usage': self._layer_usage,
                'total_usage': self._total_usage,
            }
            self._agent_span.update(
                output={'last_output': ctx.last_output[:2000], 'total_steps': ctx.step},
                metadata={
                    'model': self._model_name,
                    'model_parameters': self._model_parameters,
                    'tools': self._tools_summary,
                },
            )
            self._agent_span.end()
            task_metadata = {'task_result': _task_summary(self._task_output)} if self._task_output else {}
            if score_value is not None:
                task_metadata['score'] = score_value
            self._trace.update(
                output={'last_output': ctx.last_output[:2000], **summary},
                metadata={**self._trace_metadata, 'model': self._model_name, **summary, **task_metadata},
            )
            self._trace.end()
            if score_value is not None:
                try:
                    self._langfuse.create_score(
                        trace_id=self._trace_id, name='episode_reward', value=score_value, comment=self._score_comment()
                    )
                except Exception:
                    logger.warning('Failed to upload Langfuse score', exc_info=True)
            if self._propagate_ctx:
                with contextlib.suppress(Exception), _suppressed_otel_detach_log():
                    self._propagate_ctx.__exit__(None, None, None)
            if self._flush_on_finalize:
                self._langfuse.flush()
            logger.info('hy-live Langfuse trace finalized: trace_id=%s %s', self._trace_id, summary)
        except Exception:
            logger.warning('hy-live Langfuse finalize failed', exc_info=True)


__all__ = ['HyLiveLangfuseTracker']
