"""Post-iteration hooks for the mm_agent task.

Approach A for force-finish: when the agent is about to enter its *last*
iteration, append :data:`FORCE_FINISH_PROMPT` as a user message so the next
LLM call is forced to converge to a final answer without tool calls.
"""

from __future__ import annotations

import logging

from nexus.extensions.agents.main.agent import PostIterationHookContext, PostIterationHookResult
from nexus.extensions.tasks.mm_agent.prompts import FORCE_FINISH_PROMPT


logger = logging.getLogger(__name__)


async def force_finish_hook(ctx, hook_context: PostIterationHookContext) -> PostIterationHookResult | None:
    """Append FORCE_FINISH_PROMPT as a user message just before the last iteration.

    ``main_agent`` invokes this hook after tool execution, i.e. *before* the
    next LLM call. We inject the force-finish prompt exactly when the
    just-completed iteration is the penultimate one, so that the final LLM
    call sees the prompt and converges to a final answer.
    """
    del ctx
    if hook_context.max_iterations <= 0:
        return None

    if hook_context.iteration != hook_context.max_iterations - 1:
        return None

    logger.warning(
        '[mm_agent] Reached iteration %d/%d, injecting FORCE_FINISH_PROMPT',
        hook_context.iteration,
        hook_context.max_iterations,
    )
    return PostIterationHookResult(
        extra_messages=[{'role': 'user', 'content': FORCE_FINISH_PROMPT}],
    )
