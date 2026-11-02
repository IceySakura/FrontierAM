"""Multimodal payload construction for the reward system client."""

import pytest

from nexus.utils.reward_system import (
    build_user_content,
    redact_messages_for_log,
    resolve_strip_multimodal_config,
)


DATA_URI = 'data:image/png;base64,' + 'A' * 400


def test_no_images_keeps_the_plain_text_payload():
    assert build_user_content('what is x?', None) == 'what is x?'
    assert build_user_content('what is x?', []) == 'what is x?'


def test_images_precede_the_question_text():
    content = build_user_content('what is x?', [DATA_URI, 'https://example.com/b.png'])

    assert content == [
        {'type': 'image_url', 'image_url': {'url': DATA_URI}},
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/b.png'}},
        {'type': 'text', 'text': 'what is x?'},
    ]


def test_an_image_only_row_sends_no_blank_text_part():
    # Part of the VLM corpus asks the question inside the picture and carries no
    # text; a blank text part is rejected by some vision endpoints.
    assert build_user_content('', [DATA_URI]) == [{'type': 'image_url', 'image_url': {'url': DATA_URI}}]


def test_markers_place_the_images_for_the_judge_too():
    # The judge used to receive the markers as literal text while the actor had
    # already replaced them, so the two read different questions.
    content = build_user_content('compare <image> with <image>', [DATA_URI, 'https://example.com/b.png'])

    assert content == [
        {'type': 'text', 'text': 'compare'},
        {'type': 'image_url', 'image_url': {'url': DATA_URI}},
        {'type': 'text', 'text': 'with'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/b.png'}},
    ]


def test_a_marker_only_question_reaches_the_judge_as_bare_images():
    assert build_user_content('<image><image>', [DATA_URI, DATA_URI]) == [
        {'type': 'image_url', 'image_url': {'url': DATA_URI}},
        {'type': 'image_url', 'image_url': {'url': DATA_URI}},
    ]


def test_local_image_path_is_rejected():
    # The reward service runs elsewhere and cannot read the caller's filesystem.
    with pytest.raises(ValueError, match='data URI or an HTTP'):
        build_user_content('what is x?', ['/tmp/jupyter_work/input_0.png'])


@pytest.mark.parametrize(
    ('configured', 'expected'),
    [
        (None, (False, set())),
        (True, (True, set())),
        (False, (False, set())),
        ([], (False, set())),
        (['TextOnly-Router-0', 'MMOCR-OCRRuleScoring-0'], (False, {'TextOnly-Router-0', 'MMOCR-OCRRuleScoring-0'})),
        ([1, 2], (False, set())),
        ('TextOnly-Router-0', (False, set())),
    ],
)
def test_strip_multimodal_config_resolution(configured, expected):
    assert resolve_strip_multimodal_config(configured) == expected


def test_log_redaction_replaces_image_payloads_with_a_size_marker():
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': DATA_URI}},
                {'type': 'text', 'text': 'what is x?'},
            ],
        },
        {'role': 'assistant', 'content': 'x = 2'},
    ]

    redacted = redact_messages_for_log(messages)

    logged_url = redacted[0]['content'][0]['image_url']['url']
    assert logged_url.endswith(f'...<{len(DATA_URI)} chars>')
    assert len(logged_url) < 120
    assert redacted[0]['content'][1] == {'type': 'text', 'text': 'what is x?'}
    assert redacted[1] == {'role': 'assistant', 'content': 'x = 2'}
    # The request payload itself must keep the full image.
    assert messages[0]['content'][0]['image_url']['url'] == DATA_URI


def test_log_redaction_keeps_short_image_urls_readable():
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://e.co/a.png'}}]}]

    assert redact_messages_for_log(messages) == messages


@pytest.mark.parametrize('part', [
    {'type': 'file', 'file': {'file_data': DATA_URI, 'filename': 'input.pdf'}},
    {'type': 'file', 'file': {'url': DATA_URI}},
    {'type': 'video_url', 'video_url': {'url': DATA_URI}},
    {'type': 'video', 'video': DATA_URI},
    {'type': 'image', 'image': DATA_URI},
    {'type': 'image_url', 'image_url': DATA_URI},
])
async def test_reward_request_log_redacts_media_without_changing_request(part, caplog, monkeypatch):
    import copy
    import logging

    from nexus.utils.reward_system import RewardSystemClient

    messages = [{'role': 'user', 'content': [part]}, {'role': 'assistant', 'content': 'Answer'}]
    original = copy.deepcopy(messages)
    captured = []

    async def request(data):
        captured.append(data)
        return {'data': {'reward': {'error_code': 0, 'final_score': 1}}}

    client = RewardSystemClient(user_token='test-token')
    monkeypatch.setattr(client, '_request', request)
    with caplog.at_level(logging.DEBUG, logger='nexus.utils.reward_system'):
        assert await client.get_score(messages, '', 'test-route-0') == 1
    assert DATA_URI not in caplog.text
    assert f'<{len(DATA_URI)} chars>' in caplog.text
    assert captured[0]['messages'] == original == messages


@pytest.mark.parametrize('log_level', ['INFO', 'DEBUG'])
async def test_reward_request_logs_are_bounded_without_truncating_payload(log_level, caplog, monkeypatch):
    import copy
    import logging

    from nexus.utils.reward_system import RewardSystemClient

    text = 'trajectory-payload-' * 1000
    messages = [
        {'role': 'user', 'content': 'Question'},
        {
            'role': 'assistant',
            'tool_calls': [{'id': 'call', 'function': {'name': 'tool', 'arguments': text}}],
        },
        {'role': 'tool', 'tool_call_id': 'call', 'content': [{'type': 'text', 'text': text}]},
        {'role': 'assistant', 'content': text},
    ]
    original = copy.deepcopy(messages)
    captured = []

    async def request(data):
        captured.append(data)
        return {'data': {'reward': {'error_code': 0, 'final_score': 1}}}

    client = RewardSystemClient(user_token='test-token')
    monkeypatch.setattr(client, '_request', request)
    with caplog.at_level(getattr(logging, log_level), logger='nexus.utils.reward_system'):
        assert await client.get_score(messages, '', 'test-route-0', log_summary=True) == 1

    records = [record for record in caplog.records if record.name == 'nexus.utils.reward_system']
    assert any('message_count=4' in record.getMessage() for record in records) == (log_level == 'DEBUG')
    assert all(len(record.getMessage()) < 4500 for record in records)
    assert all('trajectory-payload-' not in record.getMessage() for record in records if record.levelno >= logging.INFO)
    if log_level == 'DEBUG':
        assert 'trajectory-payload-' in caplog.text
        assert 'chars>' in caplog.text
    assert captured[0]['messages'] == original == messages
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger='nexus.utils.reward_system'):
        assert await client.get_score(messages, '', 'test-route-0') == 1
    assert text in caplog.text
    assert 'message_count=' not in caplog.text
    assert captured[1]['messages'] == original == messages
