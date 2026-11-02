"""Uploading a HITL run's LLM calls to Langfuse, one trace per run.

Three layers:

* :mod:`.attribution` -- which agent a call belongs to;
* :mod:`.upload_dedup` -- uploading each message once per trace rather than once per
  later request;
* :mod:`.processor` -- the span processor, holding one Langfuse span per agent.
"""

from nexus.extensions.tasks.hitl.trace.langfuse.attribution import (
    LangfuseAgentRecord,
    record_for,
    tool_call_ids,
    user_fingerprints,
)
from nexus.extensions.tasks.hitl.trace.langfuse.processor import (
    SWAPPED_ATTRS,
    MultiAgentLangfuseProcessor,
)
from nexus.extensions.tasks.hitl.trace.langfuse.upload_dedup import (
    DIGEST_SIZE,
    DeduplicatingSpan,
    MessageLedger,
    deduplicate,
    fingerprint,
)


__all__ = [
    'DIGEST_SIZE',
    'SWAPPED_ATTRS',
    'DeduplicatingSpan',
    'LangfuseAgentRecord',
    'MessageLedger',
    'MultiAgentLangfuseProcessor',
    'deduplicate',
    'fingerprint',
    'record_for',
    'tool_call_ids',
    'user_fingerprints',
]
