"""Shared HITL worker-model layer.

A registry of profile *build functions* (one file per model family under
``profiles/``) returning a :class:`ModelProtocol`, plus pure helpers that stamp a
protocol onto an OpenAI client config. The model layer reads NO environment — the
generator owns all env access and dynamic values (model strings, SessionID, ...).

Adding a model means adding a profile file; recipes and helpers stay untouched.
"""

from nexus.extensions.tasks.hitl.model.base import (
    ModelProtocol,
    base_request_args,
    known_profiles,
    register,
    resolve_profile,
)
from nexus.extensions.tasks.hitl.model.builders import (
    apply_protocol_to_client,
    build_llm_client,
)


__all__ = [
    'ModelProtocol',
    'apply_protocol_to_client',
    'base_request_args',
    'build_llm_client',
    'known_profiles',
    'register',
    'resolve_profile',
]
