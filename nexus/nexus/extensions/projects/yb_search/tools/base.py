"""Base class for Yb tool definitions.

Each subclass declares:
  - ``name`` / ``version`` (ClassVar)
  - ``Params`` (Pydantic BaseModel) used for both runtime validation
    and OpenAI function-schema generation
  - optional ``build_extra`` to construct a tool-specific JSON extra
    object (returns None when the tool has no tool-level extras)
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel


if TYPE_CHECKING:
    from nexus.extensions.projects.yb_search.tool_proxy_dispatcher import (
        _SessionState,
    )
    from nexus.extensions.projects.yb_search.tools.context import (
        ToolProxyContextIn,
    )


class YbTool(ABC):
    """Per-tool definition consumed by :class:`ToolProxyDispatcher`."""

    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str] = ''
    Params: ClassVar[type[BaseModel]]

    def build_extra(
        self,
        params: BaseModel,  # noqa: ARG002 - subclasses override
        context: ToolProxyContextIn,  # noqa: ARG002
        session_state: _SessionState,  # noqa: ARG002
    ) -> Any | None:
        """Return a ``context_in.extra`` JSON object, or ``None`` if the tool has none.

        Subclasses override this when the tool requires tool-specific input
        (e.g. ``web_search`` emitting ``PromptSearchIn``).
        """
        return None

    @classmethod
    def openai_schema(cls) -> dict[str, Any]:
        """Return an OpenAI function-schema entry for this tool."""
        return {
            'type': 'function',
            'function': {
                'name': cls.name,
                'version': cls.version,
                'description': cls.description,
                'parameters': cls.Params.model_json_schema(),
            },
        }
