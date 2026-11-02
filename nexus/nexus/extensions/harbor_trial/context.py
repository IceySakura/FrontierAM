"""Run-local dependencies injected into Harbor adapter classes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from nexus.core import AgentRunOutput, ExecutionContext


@dataclass(frozen=True)
class HarborTrialBinding:
    """Nexus-owned capabilities consumed by one in-process Harbor Trial."""

    execution_context: ExecutionContext
    run_agent: Callable[..., Awaitable[AgentRunOutput]]
    runtime_name: str = 'main'


_current_binding: ContextVar[HarborTrialBinding | None] = ContextVar('harbor_trial_binding', default=None)


def require_harbor_trial_binding() -> HarborTrialBinding:
    binding = _current_binding.get()
    if binding is None:
        raise RuntimeError('Nexus Harbor adapters must be created inside harbor_trial_binding()')
    return binding


@asynccontextmanager
async def harbor_trial_binding(binding: HarborTrialBinding) -> AsyncIterator[None]:
    token = _current_binding.set(binding)
    try:
        yield
    finally:
        _current_binding.reset(token)


__all__ = ['HarborTrialBinding', 'harbor_trial_binding', 'require_harbor_trial_binding']
