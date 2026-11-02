"""Basic OpenAI-compatible profile — the default fallback.

The one to name for any model reached over plain OpenAI chat, whatever its
family: no provider-specific request args, and prompt caching is server-side /
transparent so the client does no message tagging. ``resolve_profile`` returns
this when an unknown / empty profile name is requested.

Two other profiles are the same protocol under a different name. ``default`` is an
exact alias. ``naci_glm`` differs only in the ``usage_profile`` it stamps, which
records provenance on the usage export; the counters read are identical
(``trace/usage/profiles.py``). Name that one to mark a batch as having come over the
GLM route.
"""

from typing import Any

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, register


def build(**_ignored: Any) -> ModelProtocol:
    return ModelProtocol(usage_profile='default')


register('openai_basic', build)
