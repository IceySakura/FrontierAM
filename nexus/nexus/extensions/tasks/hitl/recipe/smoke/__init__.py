"""A smoke run: one conversation, checks the launch script names.

Importing this module registers the task, the agent, and the two span processors the
generator configures. The chat tracker is upstream's own, resolved lazily.
"""

from nexus.extensions.tasks.hitl.recipe.smoke.agent import (
    Injection,
    ScriptedTurn,
    run_smoke_agent,
)
from nexus.extensions.tasks.hitl.recipe.smoke.task import SmokeCheck, run_smoke_task
from nexus.extensions.tasks.hitl.trace.langfuse import MultiAgentLangfuseProcessor
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor


__all__ = [
    'HITLUsageProcessor',
    'Injection',
    'MultiAgentLangfuseProcessor',
    'ScriptedTurn',
    'SmokeCheck',
    'run_smoke_agent',
    'run_smoke_task',
]
