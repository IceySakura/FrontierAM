"""Regression: a gateway may split one assistant turn across several choices.

Observed on gpt-6-astra via the aihub azure Responses passthrough (2026-09-07): the
gateway returned choices[0] = preamble text with finish_reason='stop' and choices[1] =
the real tool_calls with finish_reason='tool_calls'. The converter used to take
choices[0] blindly, so every tool call was dropped, the agent saw a lone sentence and
finished the episode with zero tool use (codearena smoke produced only .gitignore).
"""

from nexus.runtime.server.services.anthropic_proxy.converter import openai_to_anthropic


def _split_choice_response() -> dict:
    """Verbatim shape of the observed gpt-6-astra response (trimmed)."""
    return {
        'id': 'chatcmpl-ce2a6f2c',
        'model': 'gpt-6-astra',
        'object': 'chat.completion',
        'choices': [
            {
                'finish_reason': 'stop',
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': '我会制作一个橙色主视觉的 B2B 采购列表页。\n',
                    'tool_calls': None,
                },
            },
            {
                'finish_reason': 'tool_calls',
                'index': 1,
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call_GW7u7gDAXZu52KvoYkYv8yDr',
                            'type': 'function',
                            'index': 0,
                            'function': {
                                'name': 'grep_files',
                                'arguments': '{"pattern":".","path":"."}',
                            },
                        }
                    ],
                },
            },
        ],
        'usage': {'prompt_tokens': 1863, 'completion_tokens': 105},
    }


def test_tool_calls_from_later_choice_are_not_dropped():
    result = openai_to_anthropic(_split_choice_response(), model='gpt-6-astra')

    tool_uses = [block for block in result.content if getattr(block, 'type', None) == 'tool_use']
    assert len(tool_uses) == 1, f'tool_use block lost: {result.content}'
    assert tool_uses[0].name == 'grep_files'
    assert tool_uses[0].id == 'call_GW7u7gDAXZu52KvoYkYv8yDr'

    # The agent must be told to keep going, not that the turn ended.
    assert result.stop_reason == 'tool_use'


def test_leading_preamble_text_is_preserved():
    result = openai_to_anthropic(_split_choice_response(), model='gpt-6-astra')

    texts = [block.text for block in result.content if getattr(block, 'type', None) == 'text']
    assert any('B2B' in text for text in texts), f'preamble text lost: {result.content}'


def test_single_choice_with_tool_calls_is_unchanged():
    """The common single-choice path must keep behaving exactly as before."""
    response = {
        'id': 'chatcmpl-single',
        'model': 'gpt-5.6-sol',
        'choices': [
            {
                'finish_reason': 'tool_calls',
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'checking',
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {'name': 'read_file', 'arguments': '{}'},
                        }
                    ],
                },
            }
        ],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
    }

    result = openai_to_anthropic(response, model='gpt-5.6-sol')

    tool_uses = [block for block in result.content if getattr(block, 'type', None) == 'tool_use']
    assert len(tool_uses) == 1
    assert tool_uses[0].name == 'read_file'
    assert result.stop_reason == 'tool_use'


def test_plain_text_single_choice_still_ends_turn():
    """No tool calls anywhere -> the turn genuinely ended."""
    response = {
        'id': 'chatcmpl-text',
        'model': 'gpt-6-astra',
        'choices': [
            {
                'finish_reason': 'stop',
                'index': 0,
                'message': {'role': 'assistant', 'content': 'done', 'tool_calls': None},
            }
        ],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 1},
    }

    result = openai_to_anthropic(response, model='gpt-6-astra')

    assert result.stop_reason == 'end_turn'
    assert [block.text for block in result.content if getattr(block, 'type', None) == 'text'] == ['done']


def test_truncated_tool_arguments_are_logged_not_swallowed(caplog):
    """A truncated argument payload must surface loudly.

    gpt-6-astra once returned a 78k-char create_file payload whose JSON was cut off
    mid-string. The converter degraded it to `{}`, so the agent issued create_file()
    with no path/content, got "Path cannot be project root", and retried 5 times.
    """
    import logging

    response = {
        'id': 'chatcmpl-trunc',
        'model': 'gpt-6-astra',
        'choices': [
            {
                'finish_reason': 'tool_calls',
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': 'call_trunc',
                            'type': 'function',
                            'function': {
                                'name': 'create_file',
                                # Unterminated string - exactly the observed failure.
                                'arguments': '{"path":"index.html","content":"<!DOCTYPE html>',
                            },
                        }
                    ],
                },
            }
        ],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 97730},
    }

    with caplog.at_level(logging.ERROR):
        result = openai_to_anthropic(response, model='gpt-6-astra')

    messages = [record.getMessage() for record in caplog.records]
    assert any('unparsable arguments' in message for message in messages), f'no error logged: {messages}'

    tool_uses = [block for block in result.content if getattr(block, 'type', None) == 'tool_use']
    assert len(tool_uses) == 1
    assert tool_uses[0].name == 'create_file'
