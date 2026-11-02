"""Nexus Runtime dispatch adapter for the seven general-context tools."""

from __future__ import annotations

import json

from typing import Any

from nexus.core import ToolContext, ToolExecutor, ToolNameResolutionError, require_context, tool_registry
from nexus.core.tool.types import ToolSearchConfig


def execution_tool_state(state: dict[str, Any]) -> dict[str, Any]:
    """Exclude host-only audit history without mutating the task's state.

    MCPToolManager continues recording in ExecutionContext.tool_state for task
    evaluators. Tools receive only execution state; full results remain in the
    Agent conversation and trace outputs.
    """
    return {key: value for key, value in state.items() if key != '__mcp_call_records__'}


class _ContextAgentToolExecutor(ToolExecutor):
    """Keep accumulated MCP audit records out of per-call snapshots and RPCs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tool_name_resolution_error_count = 0
        super().__init__(*args, **kwargs)

    def _resolve_among(self, tool_name: str, candidate_names: list[str]) -> str:
        try:
            return super()._resolve_among(tool_name, candidate_names)
        except ToolNameResolutionError:
            # Count before bridge tools turn this exception into a normal
            # error observation. The counter belongs to this rollout executor.
            self.tool_name_resolution_error_count += 1
            raise

    def _tool_info_for_canonical_name(self, canonical_name: str, include_schema: bool) -> dict[str, Any]:
        info = super()._tool_info_for_canonical_name(canonical_name, include_schema=True)
        schema = info['input_schema']
        info['required'] = schema.get('required', [])
        if not include_schema:
            del info['input_schema']
        return info

    def _build_tool_context(self, *, include_executor: bool = False) -> ToolContext:
        ctx = require_context()
        tool_configs = ctx.tool_configs.copy()
        if self._tool_search_enabled:
            tool_configs['_deferred_tool_names'] = self._deferred_canonical_tool_names()
        # Filter before ToolContext construction: its immutable state conversion
        # otherwise copies the accumulated nested history even if removed later.
        return ToolContext(
            run_id=ctx.run_id,
            executor=self if include_executor else None,
            tool_configs=tool_configs,
            tool_state=execution_tool_state(ctx.tool_state),
            trace_id=ctx.trace_id,
        )


def tool_response_to_text(response: Any) -> str:
    """Flatten a ToolResponse's output content parts into a single string."""
    parts = getattr(response, 'output', None) or []
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            if (
                part.get('type') == 'text'
                and part.get('text') is not None
                or 'text' in part
                and part['text'] is not None
            ):
                chunks.append(str(part['text']))
            else:
                chunks.append(json.dumps(part, ensure_ascii=False))
        else:
            chunks.append(str(part))
    return '\n'.join(chunks)


class ContextGeneralToolExecutor:
    """Dispatch the seven model-visible tools through the configured runtime."""

    def __init__(
        self,
        *,
        enabled_tool_names: list[str],
        allowed_mcp_tool_names: list[str] | None = None,
        tool_search_config: ToolSearchConfig | dict[str, Any] | None = None,
    ) -> None:
        self.enabled_tool_names = list(enabled_tool_names)
        self.allowed_mcp_tool_names = list(allowed_mcp_tool_names or [])

        from . import runtime_tools  # noqa: F401  — register RUNTIME tools

        self._runtime = _ContextAgentToolExecutor(
            [tool_registry.get(runtime_tools.registry_id(name)) for name in self.enabled_tool_names],
            allowed_tool_names=[
                *(runtime_tools.registry_id(name) for name in self.enabled_tool_names),
                *self.allowed_mcp_tool_names,
            ],
            tool_search_config=tool_search_config,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return self._runtime.get_tool_schemas()

    @property
    def tool_name_resolution_error_count(self) -> int:
        return self._runtime.tool_name_resolution_error_count

    def build_tool_search_reminder(self) -> str | None:
        return self._runtime.build_tool_search_reminder()

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        response = await self._runtime.call(name, arguments)
        return tool_response_to_text(response)


__all__ = ['ContextGeneralToolExecutor', 'tool_response_to_text']
