import copy

import pytest

from nexus.runtime.providers.macosworld_devcloud.bridge_guest.protocol import responses_to_chat


@pytest.mark.parametrize('detail', ['original', 'high', 'low', 'auto'])
def test_chat_backend_lifts_tool_images_after_all_tool_results(detail):
    request = {
        'reasoning': {'effort': 'high'},
        'input': [
            {'type': 'function_call', 'call_id': 'c1', 'name': 'view_image', 'arguments': '{}'},
            {'type': 'function_call', 'call_id': 'c2', 'name': 'exec_command', 'arguments': '{}'},
            {
                'type': 'function_call_output',
                'call_id': 'c1',
                'output': [{'type': 'input_image', 'image_url': 'data:image/png;base64,AAAA', 'detail': detail}],
            },
            {'type': 'function_call_output', 'call_id': 'c2', 'output': 'done'},
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'next'}]},
        ],
    }
    original = copy.deepcopy(request)
    result = responses_to_chat(request)
    messages = result['messages']
    assert [m['role'] for m in messages] == ['assistant', 'tool', 'tool', 'user', 'assistant']
    assert [m['tool_call_id'] for m in messages if m['role'] == 'tool'] == ['c1', 'c2']
    assert all(isinstance(m['content'], str) for m in messages if m['role'] == 'tool')
    image = messages[3]['content'][0]['image_url']
    assert image['url'] == 'data:image/png;base64,AAAA'
    assert image.get('detail') == (None if detail == 'original' else detail)
    assert result['reasoning_effort'] == 'high'
    assert request == original
