"""GPT worker profile over the OpenAI Responses API.

Reached through the native ``openai_responses`` client rather than litellm: the
request args below are the Responses shape the client sends as-is.

Thinking is controlled by ``reasoning.effort`` alone -- there is no thinking block
on this protocol, so ``thinking_type`` does not apply. The non-thinking path
repeats every arg because the override replaces the request args rather than
merging into them, and ``store`` has to stay off on every call: this gateway is
stateless, so a stored turn is answered with ``rs_ not found`` on the next one.
"""

from typing import Any

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, register


# Responses names the output cap ``max_output_tokens``. It wins over the max_tokens
# Claude Code sends, which the client drops when this is already set.
_MAX_OUTPUT_TOKENS = 128000

DEFAULT_REASONING_MODE = 'standard'
"""What ``reasoning.mode`` carries. Empty omits the field."""


def build(
    *,
    reasoning_effort: str = 'max',
    reasoning_mode: str = DEFAULT_REASONING_MODE,
    max_output_tokens: int = _MAX_OUTPUT_TOKENS,
    **_ignored: Any,
) -> ModelProtocol:
    def _args(effort: str) -> dict[str, Any]:
        reasoning: dict[str, Any] = {'effort': effort, 'summary': 'detailed'}
        if reasoning_mode:
            reasoning['mode'] = reasoning_mode
        return {
            'store': False,
            'max_output_tokens': max_output_tokens,
            'reasoning': reasoning,
            'include': ['reasoning.encrypted_content'],
        }

    return ModelProtocol(
        client_name='openai_responses',
        usage_profile='default',
        cache_control_strategy='',
        think_request_args=_args(reasoning_effort),
        no_think_request_args=_args('none'),
    )


register('gpt_responses', build)
