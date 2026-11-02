"""Version-routing + swimlane coverage for the yb_one_agent web tools.

Locks in that ``web_fetch`` is routed to the ci-545 ``v3.0-generate`` tool_proxy
deployment while the other search tools keep the default ``work_agent`` version
(i.e. the routing does not bleed across tools).
"""

from __future__ import annotations

import os

import pytest

from nexus.core import ToolContext
from nexus.extensions.agents.yb_one_agent import tools as yb_tools


def _ctx(**tool_state) -> ToolContext:
    return ToolContext(tool_configs={}, tool_state=tool_state)


def _text(resp) -> str:
    return resp.output[0]['text']


class _FakeResp:
    """Minimal stand-in for a successful ``requests`` response."""

    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            'status': {'code': 0},
            'results': [{'status': {'code': 0}, 'for_model': [{'type': 'text', 'content': 'ok'}]}],
        }


def _capture_post(captured: dict):
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002 - mirror requests.post
        captured['url'] = url
        captured['payload'] = json
        return _FakeResp()

    return fake_post


class TestWebFetchVersionRouting:
    async def test_web_fetch_dispatches_v3_generate(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(yb_tools.requests, 'post', _capture_post(captured))

        resp = await yb_tools.web_fetch_tool(
            yb_tools.WebFetchInput(url='https://mp.weixin.qq.com/s/abc'),
            _ctx(),
        )

        call = captured['payload']['calls'][0]
        assert call['name'] == 'web_fetch'
        assert call['version'] == 'v3.0-generate'
        # url + empty prompt → server-side passthrough (raw / WeChat Markdown).
        assert call['arguments'] == {'url': 'https://mp.weixin.qq.com/s/abc', 'prompt': ''}
        assert _text(resp) == 'ok'

    async def test_web_search_keeps_work_agent_version(self, monkeypatch):
        # web_fetch's v3.0-generate routing must not bleed into the other tools.
        captured: dict = {}
        monkeypatch.setattr(yb_tools.requests, 'post', _capture_post(captured))

        await yb_tools.web_search_tool(
            yb_tools.WebSearchInput(description='d', query=['q']),
            _ctx(),
        )

        assert captured['payload']['calls'][0]['name'] == 'web_search'
        assert captured['payload']['calls'][0]['version'] == 'work_agent'


class TestDefaultSwimlane:
    def test_default_url_points_to_ci545(self):
        # Without a SEARCH_TOOLS_API_URL override the default swimlane targets ci-545.
        if os.environ.get('SEARCH_TOOLS_API_URL'):
            pytest.skip('SEARCH_TOOLS_API_URL overridden in this environment')
        assert '21.90.150.183:8003' in yb_tools._WORK_AGENT_TOOL_PROXY_URL

    def test_web_fetch_version_constant(self):
        assert yb_tools._WEB_FETCH_TOOL_VERSION == 'v3.0-generate'
