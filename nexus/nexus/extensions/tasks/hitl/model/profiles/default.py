"""Legacy ``default`` profile — an alias of ``openai_basic``.

What the generators select when ``HITL_WORKER_CLIENT_PROFILE`` is unset, since that
is the env var's own default. Adds nothing beyond usage logging; prompt caching is
off unless the generator supplies a cache strategy from env
(HITL_WORKER_CACHE_STRATEGY).

The protocol is identical to ``openai_basic``, which is the name to use for new runs
and which describes how the three plain-OpenAI profiles relate.
"""

from typing import Any

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, register


def build(**_ignored: Any) -> ModelProtocol:
    return ModelProtocol(usage_profile='default')


register('default', build)
