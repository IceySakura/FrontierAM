"""The user simulator on the stock ``frontend_task``.

Importing this module registers the two span processors the generator configures. The
chat tracker is upstream's own, resolved lazily. The persona is a built-in one.
"""

from nexus.extensions.tasks.hitl.trace.langfuse import MultiAgentLangfuseProcessor
from nexus.extensions.tasks.hitl.trace.usage import HITLUsageProcessor


__all__ = ['HITLUsageProcessor', 'MultiAgentLangfuseProcessor']
