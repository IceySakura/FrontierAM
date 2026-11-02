"""How one provider family reports token usage.

Providers agree on what the counters mean and disagree on where they are. So a spec
is a table of candidate key paths per normalized counter: the first path holding a
non-zero value wins, and 0 is the answer only when every path that resolves holds 0.
``cache_read_input_tokens`` (Anthropic-native, what litellm passes through) is
declared before ``prompt_tokens_details.cached_tokens`` (OpenAI shape, what the
vertex client normalizes to), but both are read, because a provider filling one
leaves the other at its ``int = 0`` default rather than absent
(``core/types/usage.py``).

``derive`` is the escape hatch for a provider whose numbers need computing rather
than reading. None uses it: the only such case in this codebase is Anthropic's
three-bucket input, and the vertex client folds that into ``prompt_tokens`` before
the response reaches here (``openai_clients/vertex.py``). A provider that reports
something genuinely underivable from key lookups gets a function instead of a row.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal


UsageProfile = Literal['default', 'bedrock_claude', 'naci_glm']
"""Which provider family a set of numbers came from.

Selects a spec, and is recorded on the export so a reader can tell how a label's
numbers were read. Every profile reads every key it knows, so naming the wrong one
no longer loses a counter -- it only mislabels the provenance.
"""

DEFAULT_PROFILE: UsageProfile = 'default'

# Counters every spec resolves. `input`/`output` are required; a response missing
# either is not usable usage at all.
COUNTERS = ('input', 'output', 'total', 'cache_read', 'cache_creation', 'reasoning')

KeyPaths = Mapping[str, Sequence[str]]
"""Counter name -> candidate paths into the usage block, in priority order.

A path is dot-separated: ``prompt_tokens_details.cached_tokens`` reads the nested
dict. The first path holding a non-zero integer wins; every path is read before
settling for 0.
"""

Derive = Callable[[Mapping[str, object]], Mapping[str, int]]
"""Compute counters a lookup cannot reach, given the raw usage block.

Returns whatever subset it can; the returned values override looked-up ones.
"""


@dataclass(frozen=True)
class UsageSpec:
    """Where one provider family puts each counter."""

    keys: KeyPaths = field(default_factory=dict)
    derive: Derive | None = None


# Every key name any provider in use reports a counter under. Ordered
# native-first: a client that normalizes to OpenAI shape fills both, and the
# native key is the one a pass-through client leaves untouched.
_COMMON_KEYS: KeyPaths = {
    'input': ('prompt_tokens',),
    'output': ('completion_tokens',),
    'total': ('total_tokens',),
    'cache_read': (
        'cache_read_input_tokens',
        'prompt_tokens_details.cached_tokens',
    ),
    'cache_creation': (
        'cache_creation_input_tokens',
        'prompt_tokens_details.cache_creation_tokens',
        # openai_responses names the same counter this third way.
        'prompt_tokens_details.cache_write_tokens',
    ),
    'reasoning': (
        'completion_tokens_details.reasoning_tokens',
        'output_tokens_details.thinking_tokens',
    ),
}


# One table, because the three families differ in nothing but key names -- which
# the candidate lists already cover. They stay distinct names so the export says
# which provider produced a label's numbers.
SPECS: Mapping[UsageProfile, UsageSpec] = {
    'default': UsageSpec(keys=_COMMON_KEYS),
    'bedrock_claude': UsageSpec(keys=_COMMON_KEYS),
    'naci_glm': UsageSpec(keys=_COMMON_KEYS),
}


def spec_for(profile: str) -> UsageSpec:
    """The spec for ``profile``, falling back to the default one.

    An unknown name reads the same counters rather than raising: the profile
    selects provenance, and refusing a typo here would fail a run over a label.
    """
    return SPECS.get(profile, SPECS[DEFAULT_PROFILE])  # type: ignore[arg-type]


def known_profiles() -> list[str]:
    return sorted(SPECS)


__all__ = [
    'COUNTERS',
    'DEFAULT_PROFILE',
    'SPECS',
    'Derive',
    'KeyPaths',
    'UsageProfile',
    'UsageSpec',
    'known_profiles',
    'spec_for',
]
