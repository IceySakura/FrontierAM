from __future__ import annotations

from collections.abc import Iterable

from nexus.distributed.errors import DomainValidationError


CANONICAL_USER_SCOPES = frozenset({
    'artifact:write',
    'execution:command',
    'execution:read',
    'execution:stop',
    'submission:batch',
    'submission:live',
    'submission:single',
})


def normalize_environment_grant_scopes(scopes: Iterable[str] | None) -> frozenset[str]:
    values = CANONICAL_USER_SCOPES if scopes is None else frozenset(scopes)
    if not values <= CANONICAL_USER_SCOPES:
        raise DomainValidationError('invalid_environment_grant_scopes')
    return frozenset(values)


__all__ = ['CANONICAL_USER_SCOPES', 'normalize_environment_grant_scopes']
