"""GLM (naci) worker profile.

The ``openai_basic`` protocol with ``usage_profile='naci_glm'``: no
provider-specific request args and no client-side caching. The naci_glm usage parser
reads the same counters as the default one, so naming this profile marks a batch's
provenance on the usage export rather than changing how it is measured.
"""

from typing import Any

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, register


def build(**_ignored: Any) -> ModelProtocol:
    return ModelProtocol(usage_profile='naci_glm')


register('naci_glm', build)
