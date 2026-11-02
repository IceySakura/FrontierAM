from __future__ import annotations

import pytest

from nexus.extensions.harbor_bridge.browser import (
    BROWSER_SESSION_ENV,
    T3BrowserConfig,
    browser_agent_kwargs,
    browser_from_task,
    require_browser_session_opened,
    verifier_env_with_browser_session,
)


def test_browser_declaration_defaults_and_agent_kwargs() -> None:
    config = {'environment': {'browser': {'start_url': 'https://example.com/task'}}}

    browser = browser_from_task(config)

    assert browser == T3BrowserConfig(
        start_url='https://example.com/task',
        headless=True,
        viewport_width=1280,
        viewport_height=720,
        nav_timeout_ms=10000,
        runtime_name='main',
    )
    assert browser_agent_kwargs(config) == {'browser': browser.model_dump(mode='json')}


@pytest.mark.parametrize(
    'section',
    [
        {'start_url': 'relative/path'},
        {'start_url': 'ftp://example.com/task'},
        {'start_url': 'https://example.com', 'runtime_name': 'verifier'},
        {'start_url': 'https://example.com', 'viewport_width': 0},
        {'start_url': 'https://example.com', 'unknown': True},
    ],
)
def test_browser_declaration_is_strict(section: dict) -> None:
    with pytest.raises(ValueError, match='Invalid environment.browser'):
        browser_from_task({'environment': {'browser': section}})


def test_browser_helpers_leave_non_browser_tasks_untouched() -> None:
    env = {'JUDGE_MODEL': 'test'}

    assert browser_from_task({'environment': {}}) is None
    assert browser_agent_kwargs({'environment': {}}) == {}
    assert require_browser_session_opened(None, {}) is None
    assert verifier_env_with_browser_session(env, browser=None, tool_state={}) == env


def test_browser_session_is_required_and_exposed_to_verifier() -> None:
    browser = {'start_url': 'https://example.com/task'}
    tool_state = {'browser_session_id': ' session-1 '}

    assert require_browser_session_opened(browser, tool_state) == 'session-1'
    assert verifier_env_with_browser_session(
        {'JUDGE_MODEL': 'test'},
        browser=browser,
        tool_state=tool_state,
    ) == {
        'JUDGE_MODEL': 'test',
        BROWSER_SESSION_ENV: 'session-1',
    }


@pytest.mark.parametrize('tool_state', [{}, {'browser_session_id': ''}, {'browser_session_id': 123}])
def test_browser_task_fails_when_agent_did_not_publish_a_session(tool_state: dict) -> None:
    with pytest.raises(ValueError, match='no Agent opened a browser session'):
        require_browser_session_opened({'start_url': 'https://example.com'}, tool_state)


def test_browser_session_env_is_reserved_even_for_non_browser_tasks() -> None:
    with pytest.raises(ValueError, match='reserved by Harbor Bridge'):
        verifier_env_with_browser_session(
            {BROWSER_SESSION_ENV: 'task-controlled-session'},
            browser=None,
            tool_state={},
        )
