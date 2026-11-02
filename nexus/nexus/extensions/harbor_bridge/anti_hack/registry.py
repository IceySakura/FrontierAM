"""Registry for trusted Harbor anti-hack trajectory strategies."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from nexus.extensions.harbor_bridge.anti_hack.models import AntiHackContext, AntiHackResult


AntiHackStrategy = Callable[[AntiHackContext], AntiHackResult | Awaitable[AntiHackResult]]

_STRATEGIES: dict[str, AntiHackStrategy] = {}


def register_trajectory_strategy(name: str):
    """Register a built-in strategy under a stable serialized name."""
    normalized = name.strip()
    if not normalized:
        raise ValueError('Anti-hack trajectory strategy name must be non-empty')

    def decorator(strategy: AntiHackStrategy) -> AntiHackStrategy:
        if normalized in _STRATEGIES:
            raise ValueError(f'Anti-hack trajectory strategy already registered: {normalized}')
        _STRATEGIES[normalized] = strategy
        return strategy

    return decorator


def get_trajectory_strategy(name: str) -> AntiHackStrategy:
    """Return one trusted strategy or reject a stale/untrusted name."""
    try:
        return _STRATEGIES[name]
    except KeyError as exc:
        available = ', '.join(sorted(_STRATEGIES)) or '<none>'
        raise ValueError(f'Unknown anti-hack trajectory strategy {name!r}; available: {available}') from exc


__all__ = ['AntiHackStrategy', 'get_trajectory_strategy', 'register_trajectory_strategy']
