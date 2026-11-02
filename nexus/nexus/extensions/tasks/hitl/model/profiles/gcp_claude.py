"""GCP (Vertex) Claude worker profile.

Claude served via Google Vertex through the ``vertex_anthropic`` openai client.
Unlike the litellm-backed profiles, request_args are in Vertex/Anthropic native
shape (anthropic_version, max_tokens, thinking, output_config,
drop_forced_tool_choice_with_thinking) — the vertex_anthropic client reads these
directly. Thinking is on by default; the no-think path drops the thinking /
output_config blocks.

The client (vertex_anthropic) is declared here because it is bound to the
request_args shape. The thinking_preset (bedrock-claude, required so the proxy
round-trips thinking-block signatures) is a CC-harness concern the generator
applies via its profile-to-preset table.

Connection info (model string, base_url, api_key) stays env-driven.
"""

from typing import Any

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, register
from nexus.extensions.tasks.hitl.model.profiles._anthropic import thinking_block


# Vertex passthrough API version (matches the working gcp generator).
_ANTHROPIC_VERSION = 'vertex-2023-10-16'
# Fallback output budget (think + no-think). This only takes effect when the
# incoming request omits max_tokens — Claude Code always sends its own (governed
# by CLAUDE_CODE_MAX_OUTPUT_TOKENS), which overrides this value at the client
# request-arg merge. Overridable per-build via the max_tokens kwarg.
_MAX_TOKENS = 64000


def build(
    *,
    reasoning_effort: str = 'xhigh',
    thinking_type: str = 'adaptive',
    anthropic_version: str = _ANTHROPIC_VERSION,
    max_tokens: int = _MAX_TOKENS,
    **_ignored: Any,
) -> ModelProtocol:
    base: dict[str, Any] = {
        'anthropic_version': anthropic_version,
        'max_tokens': max_tokens,
        'drop_forced_tool_choice_with_thinking': True,
    }

    think = dict(base)
    thinking = thinking_block(thinking_type)
    if thinking:
        think['thinking'] = thinking
    if reasoning_effort:
        think['output_config'] = {'effort': reasoning_effort}

    # No-think path: same base, no thinking / output_config blocks.
    no_think = dict(base)

    return ModelProtocol(
        # The ``vertex_anthropic`` alias rather than ``vertex``: the alias injects
        # protocol=anthropic, which the plain ``vertex`` client requires in
        # client_args and raises without, and a profile sets request_args only.
        client_name='vertex_anthropic',
        usage_profile='bedrock_claude',
        cache_control_strategy='claude_code',
        think_request_args=think,
        no_think_request_args=no_think,
        thinking_on_think_args=thinking_type != 'disabled',
    )


register('gcp_claude', build)
