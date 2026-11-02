"""Strategy loading and fail-open execution for Harbor trajectory checks."""

from __future__ import annotations

import inspect
import logging

from nexus.extensions.harbor_bridge.anti_hack.models import AntiHackContext, AntiHackResult
from nexus.extensions.harbor_bridge.anti_hack.registry import get_trajectory_strategy


logger = logging.getLogger(__name__)


def _load_builtin_strategies() -> None:
    # Import for registration side effects. Keep this local so importing the
    # Harbor task does not eagerly import anti-hack detector dependencies.
    from nexus.extensions.harbor_bridge.anti_hack import strategies as _strategies  # noqa: F401


def validate_trajectory_strategy(name: str) -> None:
    """Reject an unknown strategy before allocating a Runtime."""
    _load_builtin_strategies()
    get_trajectory_strategy(name)


async def run_trajectory_strategy(name: str, context: AntiHackContext) -> AntiHackResult:
    """Run one strategy without allowing detector failure to replace task reward."""
    _load_builtin_strategies()
    strategy = get_trajectory_strategy(name)
    try:
        result = strategy(context)
        if inspect.isawaitable(result):
            result = await result
        return AntiHackResult.model_validate(result)
    except Exception as exc:
        logger.exception('Harbor anti-hack trajectory strategy %s failed', name)
        return AntiHackResult(
            status='error',
            error=f'{type(exc).__name__}: {exc}',
        )


__all__ = ['run_trajectory_strategy', 'validate_trajectory_strategy']
