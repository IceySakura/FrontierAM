"""What the chained SWE recipe records about a run.

Registered as ``swe_chain``, which a run names in ``agent_config['hooks']``.
"""

import logging

from nexus.extensions.tasks.hitl.agents.agent_with_user import RunFinish, RunStart
from nexus.extensions.tasks.hitl.user_agent.hooks import HookEvent, HookOutcome, run_hook_registry


logger = logging.getLogger(__name__)


HOOK_NAME = 'swe_chain'
"""The name a run puts in ``agent_config['hooks']``."""


@run_hook_registry.register(HOOK_NAME)
class SWEChainHook:
    """Tags this recipe's log lines with the instance it is running."""

    events = (RunStart, RunFinish)

    def __init__(self, instance_id: str = '') -> None:
        self.instance_id = instance_id

    async def on_event(self, event: HookEvent) -> HookOutcome | None:
        if isinstance(event, RunStart):
            logger.info('[swe_chain_v2] [%s] requirements=%d', self.instance_id, event.state.queue.total)
            return None
        if not isinstance(event, RunFinish):
            return None
        logger.info('[swe_chain_v2] [%s] rounds=%d', self.instance_id, event.state.worker_rounds)
        return HookOutcome(results={HOOK_NAME: {'instance_id': self.instance_id, 'rounds': event.state.worker_rounds}})

    async def on_failure(self, event: HookEvent, error: Exception) -> bool:
        """A line this recipe failed to log is not worth the run it was logging."""
        return False


__all__ = ['HOOK_NAME', 'SWEChainHook']
