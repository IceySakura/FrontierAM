from __future__ import annotations

from collections.abc import Iterable


DEFAULT_SCOPE = 'default'
RUN_SCOPE = 'run'
VALID_PROCESSOR_SCOPES = frozenset({DEFAULT_SCOPE, RUN_SCOPE})


def validate_processor_scope(processor_scope: str) -> str:
    if processor_scope not in VALID_PROCESSOR_SCOPES:
        expected = ', '.join(sorted(VALID_PROCESSOR_SCOPES))
        raise ValueError(f'unknown processor_scope {processor_scope!r}; expected one of: {expected}')
    return processor_scope


def validate_processor_scopes(processor_scopes: Iterable[str]) -> frozenset[str]:
    scopes = frozenset(processor_scopes)
    unknown = scopes - VALID_PROCESSOR_SCOPES
    if unknown:
        expected = ', '.join(sorted(VALID_PROCESSOR_SCOPES))
        unknown_values = ', '.join(sorted(unknown))
        raise ValueError(f'unknown processor_scope values: {unknown_values}; expected one of: {expected}')
    return scopes
