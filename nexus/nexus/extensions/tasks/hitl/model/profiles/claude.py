"""Claude (Bedrock) worker profile.

Thinking is on by default: the thinking block is emitted whenever ``thinking_type``
is set (it defaults to ``adaptive``), and reasoning effort is added when provided.
The thinking block carries a fixed max_tokens budget (extended thinking consumes
output budget). ``thinking_type='disabled'`` sends the explicit disable, which is
not the same request as omitting the field. Prompt caching uses Claude-Code-style
message tagging by default. The non-thinking path is empty (omitting the thinking
block is how Claude does not think).

Whether the model is asked to think is decided here by ``thinking_type`` alone, not
by the CC-harness ``thinking_preset``: that preset is a proxy thinking-block encoding
concern the generator resolves from its own profile-to-preset table. gcp_claude
behaves the same way.
"""

from typing import Any

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, register
from nexus.extensions.tasks.hitl.model.profiles._anthropic import thinking_block


# Fallback output budget for the extended-thinking path. This only takes effect
# when the incoming request omits max_tokens — Claude Code always sends its own
# (governed by CLAUDE_CODE_MAX_OUTPUT_TOKENS), which overrides this value at the
# client request-arg merge. Matches gcp_claude.
_THINKING_MAX_TOKENS = 64000


def build(
    *,
    reasoning_effort: str = 'xhigh',
    thinking_type: str = 'adaptive',
    cache_strategy: str = 'claude_code',
    **_ignored: Any,
) -> ModelProtocol:
    think: dict[str, Any] = {}
    if reasoning_effort:
        think['output_config'] = {'effort': reasoning_effort}
    # Thinking on by default; gated only on thinking_type (intrinsic), not the preset.
    thinking = thinking_block(thinking_type)
    if thinking:
        think['thinking'] = thinking
        # The budget belongs to extended thinking, so it rides with the block.
        if thinking['type'] != 'disabled':
            think['max_tokens'] = _THINKING_MAX_TOKENS

    return ModelProtocol(
        usage_profile='bedrock_claude',
        cache_control_strategy=cache_strategy,
        think_request_args=think,
        no_think_request_args={},
        thinking_on_think_args=thinking_type != 'disabled',
    )


register('bedrock_claude', build)
