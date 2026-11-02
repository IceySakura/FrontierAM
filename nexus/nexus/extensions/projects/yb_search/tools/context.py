"""Pydantic models for the ``tool_proxy_context_in`` YAML config block.

These mirror a subset of the tool_proxy ``context_in`` JSON fields that are
static per agent-run (read from YAML).
Runtime-injected fields (round / raw_query / history / conversation_id*
that change per iteration) are **not** in YAML; the dispatcher fills them
before constructing the PB request.

*Note*: ``SessionCtx.conversation_id`` / ``task_id`` are YAML-configurable
because they identify the run end-to-end; ``round`` / ``raw_query`` /
``history`` are per-iteration and filled by the dispatcher.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class UserCtx(BaseModel):
    """User-level context (immutable across an agent run).

    ``location`` / ``wx_openid`` are task-level fields: the agent merges
    them from ``ctx.task_config['user_context']`` before building the
    dispatcher, so YAML can leave them blank. See the yb-search
    tool-proxy manual for the recommended layout.
    """

    model_config = ConfigDict(extra='forbid')

    user_id: str = ''
    client_version: str = ''
    wx_openid: str = ''
    location: dict[str, Any] | None = None


class SessionCtx(BaseModel):
    """Session-level context (static identifiers).

    ``round`` / ``raw_query`` / ``history`` / mutable session fields
    (``cursors`` / ``visited_urls`` / ``sandbox_state``) are filled at runtime
    and deliberately **not** exposed here.
    """

    model_config = ConfigDict(extra='forbid')

    conversation_id: str = ''
    task_id: str = ''


class CallerCtx(BaseModel):
    """Caller-level context (experiment/routing metadata)."""

    model_config = ConfigDict(extra='forbid')

    call_source: str = ''
    model_name: str = ''
    exp: str = ''
    is_illustrated: bool = False


class ToolProxyContextIn(BaseModel):
    """Top-level ``tool_proxy_context_in`` YAML block."""

    model_config = ConfigDict(extra='forbid')

    user: UserCtx = UserCtx()
    session: SessionCtx = SessionCtx()
    caller: CallerCtx = CallerCtx()
