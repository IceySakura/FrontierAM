from nexus.core.llm_stream_events import ChatCompletionStreamEventEmitter, LLMStreamEmitter
from nexus.trace import processor_context, span_context
from nexus.trace.events import SpanUpdateEvent
from nexus.trace.processors import SpanProcessor


class RecordingProcessor(SpanProcessor):
    def __init__(self) -> None:
        self.events = []

    def on_update(self, _span, event) -> None:
        if isinstance(event, SpanUpdateEvent):
            self.events.extend(event.events or [])


def event_names(processor: RecordingProcessor) -> list[str]:
    return [event.name for event in processor.events]


def test_stream_emitter_adds_common_attributes_and_truncates_large_delta() -> None:
    processor = RecordingProcessor()

    with processor_context([processor]):
        with span_context('openai_completion'):
            emitter = LLMStreamEmitter(provider='litellm', model='gpt-test', client_key='main', max_delta_chars=4)
            emitter.emit('chat.completion.delta', delta='hello world', **{'choice.index': 0, 'content.index': 0})

    assert event_names(processor) == ['chat.completion.delta']
    attrs = processor.events[0].attributes
    assert attrs['llm.sequence'] == 1
    assert attrs['llm.provider'] == 'litellm'
    assert attrs['llm.model'] == 'gpt-test'
    assert attrs['llm.client_key'] == 'main'
    assert attrs['llm.choice.index'] == 0
    assert attrs['llm.content.index'] == 0
    assert attrs['delta'] == 'hell'
    assert attrs['delta_truncated'] is True
    assert attrs['delta_original_length'] == 11


def test_chat_completion_stream_event_emitter_emits_chat_completion_events() -> None:
    processor = RecordingProcessor()

    with processor_context([processor]):
        with span_context('openai_completion'):
            emitter = LLMStreamEmitter(provider='litellm', model='gpt-test')
            event_emitter = ChatCompletionStreamEventEmitter(emitter=emitter)
            event_emitter.consume_chunk(
                {
                    'id': 'chatcmpl-test',
                    'created': 123,
                    'model': 'gpt-test',
                    'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hello '}}],
                }
            )
            event_emitter.consume_chunk(
                {
                    'id': 'chatcmpl-test',
                    'created': 123,
                    'model': 'gpt-test',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {
                                'content': 'world',
                                'reasoning_content': 'thought',
                                'tool_calls': [
                                    {
                                        'index': 0,
                                        'id': 'call_1',
                                        'type': 'function',
                                        'function': {'name': 'write_file', 'arguments': '{"path"'},
                                    }
                                ],
                            },
                        }
                    ],
                }
            )
            event_emitter.consume_chunk(
                {
                    'id': 'chatcmpl-test',
                    'created': 123,
                    'model': 'gpt-test',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': ':"/tmp/a"}'}}]},
                            'finish_reason': 'tool_calls',
                        }
                    ],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 5, 'total_tokens': 8},
                }
            )
            event_emitter.finish(
                {
                    'id': 'chatcmpl-test',
                    'created': 123,
                    'model': 'gpt-test',
                    'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {'role': 'assistant'}}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 5, 'total_tokens': 8},
                }
            )

    assert event_names(processor) == [
        'chat.completion.created',
        'chat.completion.delta',
        'chat.completion.delta',
        'chat.completion.delta',
        'chat.completion.reasoning.delta',
        'chat.completion.tool_call.delta',
        'chat.completion.tool_call.delta',
        'chat.completion.completed',
    ]
    role_attrs = processor.events[1].attributes
    assert role_attrs['llm.delta.role'] == 'assistant'
    assert 'delta' not in role_attrs
    tool_attrs = processor.events[-3].attributes
    assert tool_attrs['llm.tool_call.id'] == 'call_1'
    assert tool_attrs['llm.tool_call.function.name'] == 'write_file'
    assert tool_attrs['llm.tool_call.index'] == 0
    assert tool_attrs['llm.choice.index'] == 0
    completed_attrs = processor.events[-1].attributes
    assert completed_attrs['response_id'] == 'chatcmpl-test'
    assert completed_attrs['finish_reason'] == 'tool_calls'
    assert completed_attrs['usage.prompt_tokens'] == 3
    assert completed_attrs['usage.completion_tokens'] == 5
    assert completed_attrs['usage.total_tokens'] == 8


def test_chat_completion_stream_event_emitter_emits_role_delta_separately() -> None:
    processor = RecordingProcessor()

    with processor_context([processor]):
        with span_context('openai_completion'):
            emitter = LLMStreamEmitter(provider='litellm', model='gpt-test')
            event_emitter = ChatCompletionStreamEventEmitter(emitter=emitter)
            event_emitter.consume_chunk(
                {
                    'id': 'chatcmpl-test',
                    'created': 123,
                    'model': 'gpt-test',
                    'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hello'}}],
                }
            )

    assert event_names(processor) == [
        'chat.completion.created',
        'chat.completion.delta',
        'chat.completion.delta',
    ]
    role_attrs = processor.events[1].attributes
    assert role_attrs['llm.choice.index'] == 0
    assert role_attrs['llm.delta.role'] == 'assistant'
    assert 'delta' not in role_attrs
    content_attrs = processor.events[2].attributes
    assert content_attrs['delta'] == 'hello'


def test_chat_completion_stream_event_emitter_emits_failed_event() -> None:
    processor = RecordingProcessor()

    with processor_context([processor]):
        with span_context('openai_completion'):
            emitter = LLMStreamEmitter(provider='litellm', model='gpt-test')
            event_emitter = ChatCompletionStreamEventEmitter(emitter=emitter)
            event_emitter.fail(RuntimeError('boom'))

    assert event_names(processor) == ['chat.completion.failed']
    assert processor.events[0].attributes['error.type'] == 'RuntimeError'
    assert processor.events[0].attributes['error.message'] == 'boom'
