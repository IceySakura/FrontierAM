"""macOSWorld native Codex selection; the shared Codex loop stays unchanged."""

from typing import Any, Literal

from nexus.core import agent_registry
from nexus.extensions.agents.codex.agent import (
    CodexNativeBridgeInput,
    CodexOutput,
    run_codex_native_bridge_with_adapter,
)
from nexus.extensions.agents.codex.json_events import _terminal_error_message


class MacOSWorldCodexInput(CodexNativeBridgeInput):
    api_protocol: Literal['responses', 'responses-chat'] = 'responses'


def _format_terminal_error(events: list[dict[str, Any]], stderr: str) -> str:
    detail = _terminal_error_message(events)
    return f'Codex terminal error: {detail[:4000]}\n{stderr}' if detail else stderr


@agent_registry.register('macosworld_codex_native_bridge')
async def run_macosworld_codex(input_data: MacOSWorldCodexInput) -> CodexOutput:
    return await run_codex_native_bridge_with_adapter(
        input_data,
        api_protocol=input_data.api_protocol,
        format_terminal_error=_format_terminal_error,
    )
