"""Unit tests for routing search_stateful requests through the ATS POST cache."""

import hashlib
import json

from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexus.extensions.tools.search_stateful.backend import (
    ATS_CACHE_KEY_HEADER,
    GoogleApiBackend,
    ats_cache_key,
)


ATS = 'http://ats.example:3080'


def _response(payload, x_cache='miss'):
    response = Mock()
    response.status_code = 200
    response.text = json.dumps(payload)
    response.headers = {'X-Cache': x_cache}
    response.json = Mock(return_value=payload)
    return response


def _backend(ats_proxy=ATS):
    return GoogleApiBackend(user='u', apikey='k', use_cache=True, ats_proxy=ats_proxy)


def test_ats_cache_key_matches_sorted_compact_payload():
    expected = hashlib.sha256(
        b'{"body":{"num":10,"q":"x"},"extra":{},"method":"POST","path":"/search","provider":"serper"}'
    ).hexdigest()
    assert ats_cache_key('/search', {'q': 'x', 'num': 10}, 'serper', {}) == expected


def test_ats_proxy_disables_file_cache():
    backend = _backend()
    assert backend.use_cache is False
    assert backend.search_cache is None
    assert backend.visit_cache is None


@pytest.mark.asyncio
async def test_search_sends_proxy_and_cache_key():
    hit = _response({'organic': [{'title': 't', 'link': 'https://a', 'snippet': 's'}]}, 'hit-fresh')
    with patch(
        'nexus.extensions.tools.search_stateful.backend.http_client.post', new=AsyncMock(return_value=hit)
    ) as post:
        page = await _backend().search('x', 10)

    assert post.await_count == 1
    kwargs = post.await_args.kwargs
    assert kwargs['proxy'] == ATS
    assert kwargs['headers'][ATS_CACHE_KEY_HEADER] == ats_cache_key('/search', {'q': 'x', 'num': 10}, 'serper', {})
    assert [r.url for r in page.search_results] == ['https://a']


@pytest.mark.asyncio
async def test_empty_search_retries_once_under_derived_key():
    empty = _response({'organic': []}, 'hit-fresh')
    full = _response({'organic': [{'title': 't', 'link': 'https://a', 'snippet': 's'}]})
    with patch(
        'nexus.extensions.tools.search_stateful.backend.http_client.post', new=AsyncMock(side_effect=[empty, full])
    ) as post:
        page = await _backend().search('x', 10)

    keys = [call.kwargs['headers'][ATS_CACHE_KEY_HEADER] for call in post.await_args_list]
    assert keys == [
        ats_cache_key('/search', {'q': 'x', 'num': 10}, 'serper', {}),
        ats_cache_key('/search', {'q': 'x', 'num': 10}, 'serper', {'retry': 1}),
    ]
    assert len(page.search_results) == 1


@pytest.mark.asyncio
async def test_empty_visit_retries_once_under_derived_key():
    empty = _response({'data': {'text': '', 'title': 'p'}}, 'hit-fresh')
    full = _response({'data': {'text': 'body', 'title': 'p'}})
    with patch(
        'nexus.extensions.tools.search_stateful.backend.http_client.post', new=AsyncMock(side_effect=[empty, full])
    ) as post:
        page = await _backend().fetch('https://a')

    extras = [{'X-Return-Format': 'text'}, {'X-Return-Format': 'text', 'retry': 1}]
    keys = [call.kwargs['headers'][ATS_CACHE_KEY_HEADER] for call in post.await_args_list]
    assert keys == [ats_cache_key('/api/jina_ai/search', {'url': 'https://a'}, 'aws_jina', e) for e in extras]
    assert page.text == 'body'


@pytest.mark.asyncio
async def test_without_ats_proxy_requests_go_direct():
    full = _response({'data': {'text': 'body', 'title': 'p'}})
    with patch(
        'nexus.extensions.tools.search_stateful.backend.http_client.post', new=AsyncMock(return_value=full)
    ) as post:
        await GoogleApiBackend(user='u', apikey='k', use_cache=False, ats_proxy='').fetch('https://a')

    kwargs = post.await_args.kwargs
    assert 'proxy' not in kwargs
    assert ATS_CACHE_KEY_HEADER not in kwargs['headers']
