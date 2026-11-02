"""The chained SWE recipe built on the user_agent components.

Importing this module registers the task, the run hook and the two span processors the
generator names, which is what its ``imports`` entry relies on. A processor named in
the run config but never imported is not in the registry when the runner resolves
it. The chat tracker is upstream's own, resolved lazily without an import entry.

The agent is ``hitl/agent_with_user``; what this recipe adds to a run is
:class:`~.hook.SWEChainHook`.
"""

from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.hook import SWEChainHook
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.task import (
    SWEChainV2TaskInput,
    run_swe_chain_v2_task,
)
from nexus.extensions.tasks.hitl.trace.langfuse import MultiAgentLangfuseProcessor
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor


__all__ = [
    'HITLUsageProcessor',
    'MultiAgentLangfuseProcessor',
    'SWEChainHook',
    'SWEChainV2TaskInput',
    'run_swe_chain_v2_task',
]
