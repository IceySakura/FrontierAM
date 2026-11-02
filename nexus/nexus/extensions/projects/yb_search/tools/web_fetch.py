"""``web_fetch@v1.0`` tool definition.

Simplified v1 (see design doc § 8.1): we emit **no** ``context_in.extra``
JSON payload — the offline training scenario does not need any tool-specific
extras. Add them back by overriding ``build_extra`` once the online scenario
requires reproducing them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from nexus.extensions.projects.yb_search.tools.base import YbTool
from nexus.extensions.projects.yb_search.tools.registry import (
    yb_tool_registry,
)


class _WebFetchParams(BaseModel):
    """Parameters for ``web_fetch``.

    Server-side handler expects ``arguments.url`` to be a single string
    (``Value.string_value``). Passing a ``ListValue`` triggers ``[40001]
    url is required`` because the handler's type assertion to string
    yields ``""``. Batch fetching requires multiple ``web_fetch`` tool
    calls in the same batch, not a list inside one call.
    """

    url: str = Field(
        ...,
        min_length=1,
        description='Target URL to fetch. Issue multiple web_fetch calls if multiple URLs are needed.',
    )

    @field_validator('url', mode='before')
    @classmethod
    def _coerce_list_to_single(cls, v: Any) -> Any:
        """LLMs occasionally emit ``url=["..."]``; take the first element so we don't
        round-trip a ``ListValue`` to the server (which would 40001)."""
        if isinstance(v, list):
            if not v:
                raise ValueError('url list is empty')
            return v[0]
        return v


@yb_tool_registry.register
class WebFetchTool(YbTool):
    name = 'web_fetch'
    version = 'v1.0'
    description = (
        '网页抓取工具，输入一个目标 url（string），抓取该网页正文并返回'
        '标题、正文等结构化内容。 需要抓取多个 url 时请发起多次 web_fetch 调用。'
    )
    Params = _WebFetchParams
