"""HY3 (Hunyuan 3.0) worker profile.

HY3 toggles thinking through ``chat_template_kwargs`` in ``extra_body``, and it
THINKS BY DEFAULT — to not think it must explicitly send reasoning_effort=
'no_think'. So the non-thinking path is a fully populated request_args set, not
empty; an empty one would fall back to the thinking args and HY3 would keep
thinking on cheap calls.

Multiturn cache routing: HY3's inference proxy uses consistent-hash routing keyed
by a SessionID header, so all turns of one worker session land on the same
backend (which holds that session's KV cache). The header must ride on EVERY
request, so it is repeated in both the thinking and non-thinking arg sets. The
generator supplies the SessionID (one uuid per worker session) — it is dynamic,
so it never lives in this file.

Usage is reported in the standard OpenAI shape, so the default usage parser
applies. Caching is server-side (no client message tagging).
"""

from typing import Any

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, register


_THINK_TEMPLATE_KWARGS = {'reasoning_effort': 'high', 'interleaved_thinking': True}
_NO_THINK_TEMPLATE_KWARGS = {'reasoning_effort': 'no_think', 'interleaved_thinking': False}


def build(
    *,
    session_id: str = '',
    temperature: float = 1.0,
    **_ignored: Any,
) -> ModelProtocol:
    # Consistent-hash routing header, ridden on every request to maximize
    # multiturn cache hits. Omitted entirely when no SessionID is supplied.
    extra_headers: dict[str, Any] = {}
    if session_id:
        extra_headers = {'SessionID': session_id, 'StrategyType': 'ConsistentHash'}

    def _args(template_kwargs: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {
            'temperature': temperature,
            'extra_body': {'chat_template_kwargs': dict(template_kwargs)},
        }
        if extra_headers:
            args['extra_headers'] = dict(extra_headers)
        return args

    return ModelProtocol(
        usage_profile='default',
        think_request_args=_args(_THINK_TEMPLATE_KWARGS),
        no_think_request_args=_args(_NO_THINK_TEMPLATE_KWARGS),
    )


register('hy3', build)
