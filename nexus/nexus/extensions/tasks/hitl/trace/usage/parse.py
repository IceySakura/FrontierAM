"""Reading a provider's usage block into normalized counters."""

import logging

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from nexus.extensions.tasks.hitl.trace.usage.profiles import (
    COUNTERS,
    DEFAULT_PROFILE,
    UsageProfile,
    UsageSpec,
    spec_for,
)


logger = logging.getLogger(__name__)

PATH_SEPARATOR = '.'


@dataclass(frozen=True)
class UsageTrace:
    """One call's counters, in the same units whichever provider reported them.

    ``valid`` is False for a response carrying no usable usage block. Such a call is
    counted as invalid rather than as zero tokens, so a run cannot look cheap because
    its responses were unparseable.
    """

    profile: UsageProfile
    valid: bool
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def billed_input_tokens(self) -> int:
        """Input tokens excluding what came from cache.

        Cache reads are priced well below fresh input, so a comparison of prompt
        sizes wants this and a cost estimate wants the three separately.
        """
        return max(0, self.input_tokens - self.cache_read_tokens)


def parse_usage(profile: str, response_dump: Mapping[str, object]) -> UsageTrace:
    """Read the usage block out of a completion response dump.

    ``input`` and ``output`` are required: a block missing either is reported invalid
    rather than as a zero-token call. ``total`` falls back to their sum, which is what
    a provider that omits it means.
    """
    # The membership test is the narrowing; a type checker reads the branch as str.
    resolved: UsageProfile = (  # ty: ignore[invalid-assignment]
        profile if profile in ('default', 'bedrock_claude', 'naci_glm') else DEFAULT_PROFILE
    )
    usage = response_dump.get('usage')
    if not isinstance(usage, dict):
        return UsageTrace(profile=resolved, valid=False)

    counters = _read_counters(cast(Mapping[str, object], usage), spec_for(profile))
    if counters.get('input') is None or counters.get('output') is None:
        return UsageTrace(profile=resolved, valid=False)

    input_tokens = counters['input'] or 0
    output_tokens = counters['output'] or 0
    total = counters.get('total')
    return UsageTrace(
        profile=resolved,
        valid=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total if total is not None else input_tokens + output_tokens,
        cache_read_tokens=counters.get('cache_read') or 0,
        cache_creation_tokens=counters.get('cache_creation') or 0,
        reasoning_tokens=counters.get('reasoning') or 0,
    )


def _read_counters(usage: Mapping[str, object], spec: UsageSpec) -> dict[str, int | None]:
    """Resolve every counter the spec knows, then let ``derive`` override."""
    counters: dict[str, int | None] = {
        counter: _first_present(usage, spec.keys.get(counter, ())) for counter in COUNTERS
    }
    if spec.derive is None:
        return counters

    try:
        derived = spec.derive(usage)
    except Exception as exc:  # noqa: BLE001 - usage is diagnostic; a bad spec must not fail the call
        logger.warning('usage derive failed: %s', exc)
        return counters
    for counter, value in derived.items():
        if isinstance(value, int) and not isinstance(value, bool):
            counters[counter] = value
    return counters


def _first_present(usage: Mapping[str, object], paths: object) -> int | None:
    """The first path resolving to a non-zero integer.

    Falls back to 0 when every path that resolves holds 0, and to None when no path
    resolves at all. Zero and absent stay distinct: ``parse_usage`` reads None as an
    unusable usage block, while a zero counter is a reading.
    """
    if not isinstance(paths, (tuple, list)):
        return None
    zero_seen = False
    for path in paths:
        value = _at_path(usage, str(path))
        if value is None:
            continue
        if value:
            return value
        zero_seen = True
    return 0 if zero_seen else None


def _at_path(usage: Mapping[str, object], path: str) -> int | None:
    """Follow a dotted path, returning an int or None.

    None for a missing key, a non-integer, or a bool -- ``isinstance(True, int)`` is
    True, and a flag read as 1 token would silently enter the totals.
    """
    current: object = usage
    for segment in path.split(PATH_SEPARATOR):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    if isinstance(current, bool) or not isinstance(current, int):
        return None
    return current


def model_of(response_dump: Mapping[str, object]) -> str:
    """The model named in a response dump, or empty."""
    model = response_dump.get('model')
    return model if isinstance(model, str) else ''


__all__ = ['PATH_SEPARATOR', 'UsageTrace', 'model_of', 'parse_usage']
