"""Worker-model protocol registry for HITL recipes.

A *protocol* describes how to talk to one model's API: the request args for the
thinking and non-thinking paths, the prompt-cache strategy, and which usage
parser to use. It is pure description — it contains NO connection info (model
string, base_url, api_key) and reads NO environment. The generator owns all env
access and dynamic values (e.g. a SessionID), passing them into the profile
build function; the build function returns a :class:`ModelProtocol`.

The two request-arg sets are independent rather than base plus delta: the
non-thinking path is a full request_args override at the litellm layer, so anything
that must ride on every request (e.g. a routing header) has to appear in BOTH sets.

One file per model family lives under ``profiles/``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


UsageProfileName = Literal['default', 'bedrock_claude', 'naci_glm']
"""Usage parser a protocol selects; the names ``trace/usage/profiles.py`` registers."""

ClientName = Literal['litellm', 'vertex_anthropic', 'openai_responses']
"""Registered openai client backend a protocol is bound to."""


@dataclass(frozen=True)
class ModelProtocol:
    """How to talk to one model's API. No model string, no env, no CC harness knobs.

    Fields:
      client_name            Registered openai client backend for the worker. Bound
                             to the request_args shape (vertex_anthropic expects the
                             Anthropic-native shape below), so it lives with the
                             protocol rather than in a separate table.
      usage_profile          Token/cost parser.
      cache_control_strategy Client-side prompt-cache strategy ('' = none / server-side
                             transparent). e.g. 'claude_code' tags messages for
                             Anthropic prompt caching. The generator may override
                             from env and supplies the ttl.
      think_request_args     request_args for the main (thinking) path. Must not
                             include the model string (the generator fills it).
      no_think_request_args  request_args for the non-thinking path. A full
                             override, so it must repeat anything that has to ride
                             on every request (headers). Model string excluded.
      thinking_on_think_args whether ``think_request_args`` asks the model to think.
                             A profile whose knobs can turn thinking off within that
                             set says so here. It describes that set alone; pair it
                             with which set the run selected.
    """

    client_name: ClientName = 'litellm'
    usage_profile: UsageProfileName = 'default'
    cache_control_strategy: str = ''
    think_request_args: dict[str, Any] = field(default_factory=dict)
    no_think_request_args: dict[str, Any] = field(default_factory=dict)
    thinking_on_think_args: bool = True


def base_request_args(protocol: ModelProtocol, *, use_thinking: bool) -> dict[str, Any]:
    """The request args a client carries, of the protocol's two sets.

    Whether the returned set asks the model to think is ``thinking_on_think_args``
    for the thinking set, and no for the other.
    """
    return protocol.think_request_args if use_thinking else protocol.no_think_request_args


# A profile build function: (**generator-supplied kwargs) -> ModelProtocol.
ProfileBuilder = Callable[..., ModelProtocol]

_REGISTRY: dict[str, ProfileBuilder] = {}

# Profile used when an unknown / empty name is requested. The basic OpenAI
# profile registers under this name at import time.
_DEFAULT_PROFILE_NAME = 'openai_basic'


def register(name: str, builder: ProfileBuilder) -> ProfileBuilder:
    """Register a profile build function under ``name``. Returns it unchanged."""
    _REGISTRY[name] = builder
    return builder


def resolve_profile(name: str) -> ProfileBuilder:
    """Look up a profile build function, falling back to the basic profile."""
    builder = _REGISTRY.get(name)
    if builder is not None:
        return builder
    return _REGISTRY[_DEFAULT_PROFILE_NAME]


def known_profiles() -> list[str]:
    """Names of all registered profiles (for diagnostics / logging)."""
    return sorted(_REGISTRY)
