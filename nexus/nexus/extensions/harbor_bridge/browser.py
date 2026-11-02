"""Browser capability declared by a T3-Harbor task and owned by its Agent."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


BROWSER_AGENT_FIELD = 'browser'
BROWSER_SESSION_STATE_KEY = 'browser_session_id'
BROWSER_SESSION_ENV = 'T3_BROWSER_SESSION_ID'


class T3BrowserConfig(BaseModel):
    """Strict browser declaration forwarded to a browser-capable Agent."""

    model_config = ConfigDict(extra='forbid')

    start_url: str
    headless: bool = True
    viewport_width: int = Field(default=1280, gt=0)
    viewport_height: int = Field(default=720, gt=0)
    nav_timeout_ms: int = Field(default=10000, gt=0)
    runtime_name: Literal['main'] = 'main'

    @field_validator('start_url')
    @classmethod
    def _validate_start_url(cls, value: str) -> str:
        url = value.strip()
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc:
            raise ValueError(f'start_url must be an absolute http(s) URL, got {value!r}')
        return url


def browser_from_task(config: dict[str, Any]) -> T3BrowserConfig | None:
    """Read ``[environment.browser]`` from the raw T3 declaration."""
    environment = config.get('environment', {}) or {}
    if not isinstance(environment, dict):
        raise ValueError('[environment] must be a table')
    section = environment.get('browser')
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError('environment.browser must be a table')
    try:
        return T3BrowserConfig.model_validate(section)
    except ValueError as exc:
        raise ValueError(f'Invalid environment.browser: {exc}') from exc


def browser_agent_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build the Agent kwarg carrying the task's browser declaration."""
    browser = browser_from_task(config)
    return {BROWSER_AGENT_FIELD: browser.model_dump(mode='json')} if browser is not None else {}


def require_browser_session_opened(browser: dict[str, Any] | None, tool_state: dict[str, Any]) -> str | None:
    """Require a non-empty session ID when the task declared a browser."""
    if not browser:
        return None
    session_id = tool_state.get(BROWSER_SESSION_STATE_KEY)
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    raise ValueError(
        'task.toml declares [environment.browser] but no Agent opened a browser session '
        f'({BROWSER_SESSION_STATE_KEY!r} missing from tool_state); select a browser-capable '
        'Agent such as browser_gym_agent'
    )


def verifier_env_with_browser_session(
    verifier_env: dict[str, str],
    *,
    browser: dict[str, Any] | None,
    tool_state: dict[str, Any],
) -> dict[str, str]:
    """Expose the Agent's live browser session to a shared Verifier."""
    if BROWSER_SESSION_ENV in verifier_env:
        raise ValueError(f'{BROWSER_SESSION_ENV} is reserved by Harbor Bridge and cannot be set by the task')
    session_id = require_browser_session_opened(browser, tool_state)
    if session_id is None:
        return verifier_env
    return {**verifier_env, BROWSER_SESSION_ENV: session_id}


__all__ = [
    'BROWSER_AGENT_FIELD',
    'BROWSER_SESSION_ENV',
    'BROWSER_SESSION_STATE_KEY',
    'T3BrowserConfig',
    'browser_agent_kwargs',
    'browser_from_task',
    'require_browser_session_opened',
    'verifier_env_with_browser_session',
]
