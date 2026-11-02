"""``picture_search@v1.0`` tool definition.

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


class _PictureSearchParams(BaseModel):
    """Parameters for ``picture_search`` (aligned with online prompt_search)."""

    query: list[str] = Field(
        ...,
        min_length=1,
        description='One or more image-search queries. The online service ranks them jointly.',
    )

    @field_validator('query', mode='before')
    @classmethod
    def _coerce_single_string(cls, v: Any) -> Any:
        """Accept ``str`` or ``list[str]``; coerce bare strings to a one-element list."""
        if isinstance(v, str):
            return [v]
        return v


@yb_tool_registry.register
class PictureSearchTool(YbTool):
    name = 'picture_search'
    version = 'v1.0'
    description = '图片检索工具，输入一个或多个 query（list of string），在海量图片库中做语义匹配并返回图片结果列表。'
    Params = _PictureSearchParams
