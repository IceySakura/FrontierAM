"""Monkey-patch BFCL's WebSearchAPI to use an internal search forwarder.

BFCL's default WebSearchAPI.search_engine_query uses SerpAPI (GoogleSearch),
which requires a SERPAPI_API_KEY and external network access. In our environment
we route searches through an internal forwarder service instead.

This module patches WebSearchAPI.search_engine_query at import time.
The patch is applied once in evaluator.py when the bfclv4 package is loaded.
"""

import base64
import json
import logging
import uuid

import requests


logger = logging.getLogger(__name__)

FORWARDER_URL = 'https://lizhi-assistant.testsite.woa.com/trpc.search_qa.common.ForwarderProxyHttp/Forward'
FORWARDER_API_KEY = '9d8e339b2e1984904d6a9a9a303f5b80b33cc451'


def _region_to_serper_gl(region: str | None) -> str:
    """Convert BFCL region code to Google Serper gl parameter."""
    if not region:
        return 'cn'
    region_prefix = region.split('-', 1)[0].strip().lower()
    if region_prefix in {'wt', ''}:
        return 'cn'
    return region_prefix


def _normalize_search_results(results: list, max_results: int | None, show_snippet: bool) -> list:
    """Normalize search results to BFCL's expected format."""
    output = []
    limit = max_results if max_results is not None else len(results)
    for result in results[:limit]:
        normalized = {
            'title': result.get('title', ''),
            'href': result.get('link') or result.get('href', ''),
        }
        if show_snippet:
            normalized['body'] = result.get('snippet') or result.get('body', '')
        output.append(normalized)
    return output


def _search_via_forwarder(self, keywords: str, max_results: int | None = 10, region: str | None = 'wt-wt') -> list:
    """Replacement for WebSearchAPI.search_engine_query using internal forwarder."""
    serper_payload = {
        'q': keywords,
        'page': 1,
        'gl': _region_to_serper_gl(region),
    }
    encoded_payload = base64.b64encode(json.dumps(serper_payload, ensure_ascii=False).encode('utf-8')).decode('utf-8')
    request_payload = {
        'header': {
            'session_id': '2111',
            'trace_id': uuid.uuid4().hex,
        },
        'forward_target': 'google_serper',
        'forward_header': {
            'X-API-KEY': FORWARDER_API_KEY,
            'Content-Type': 'application/json',
        },
        'forward_payload': encoded_payload,
    }

    try:
        response = requests.post(
            FORWARDER_URL,
            json=request_payload,
            headers={'Content-Type': 'application/json'},
            timeout=20,
        )
        response.raise_for_status()
        response_payload = response.json()
    except Exception as e:
        return {'error': f'Forwarder request failed: {e}'}  # ty: ignore[invalid-return-type]

    forward_rsp_data = response_payload.get('forward_rsp_data')
    if forward_rsp_data is None and isinstance(response_payload.get('data'), dict):
        forward_rsp_data = response_payload['data'].get('forward_rsp_data')
    if forward_rsp_data is None:
        return {'error': 'Forwarder response missing forward_rsp_data.'}  # ty: ignore[invalid-return-type]

    try:
        decoded_rsp_data = base64.b64decode(forward_rsp_data).decode('utf-8')
        search_results = json.loads(decoded_rsp_data)
    except Exception as e:
        return {'error': f'Invalid forward_rsp_data payload: {e}'}  # ty: ignore[invalid-return-type]

    organic_results = search_results.get('organic') or search_results.get('organic_results')
    if not organic_results:
        return {'error': 'Failed to retrieve search results from forwarder.'}  # ty: ignore[invalid-return-type]

    show_snippet = getattr(self, 'show_snippet', True)
    return _normalize_search_results(organic_results, max_results, show_snippet)


# Copy the docstring from the original for BFCL's function doc generation
_search_via_forwarder.__doc__ = """
This function queries the search engine for the provided keywords and region.

Args:
    keywords (str): The keywords to search for.
    max_results (int, optional): The maximum number of search results to return. Defaults to 10.
    region (str, optional): The region to search in. Defaults to "wt-wt".

Returns:
    list: A list of search result dictionaries, each containing:
    - 'title' (str): The title of the search result.
    - 'href' (str): The URL of the search result.
    - 'body' (str): A brief description or snippet from the search result.
"""


def apply_web_search_patch():
    """Patch WebSearchAPI.search_engine_query to use internal forwarder."""
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI

    WebSearchAPI.search_engine_query = _search_via_forwarder  # ty: ignore[invalid-assignment]
    logger.debug('Patched WebSearchAPI.search_engine_query → internal forwarder')
