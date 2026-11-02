"""A parallel run: several worker sessions at once, checks the launch script names.

Importing this module registers the task, the agent, and the two span processors the
generator configures. The chat tracker is upstream's own, resolved lazily.
"""

from nexus.extensions.tasks.hitl.recipe.parallel.agent import (
    ParallelSend,
    ParallelStep,
    run_parallel_agent,
)
from nexus.extensions.tasks.hitl.recipe.parallel.task import ParallelCheck, run_parallel_task
from nexus.extensions.tasks.hitl.trace.langfuse import MultiAgentLangfuseProcessor
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor


__all__ = [
    'HITLUsageProcessor',
    'MultiAgentLangfuseProcessor',
    'ParallelCheck',
    'ParallelSend',
    'ParallelStep',
    'run_parallel_agent',
    'run_parallel_task',
]
