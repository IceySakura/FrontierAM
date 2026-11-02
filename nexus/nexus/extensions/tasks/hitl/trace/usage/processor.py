"""Run-level token accounting, attributed by the per-request trace label.

Reads the ``openai_completion`` span the tracer already emits for every client
call, so each call is counted wherever it was made: the user agent's turns, its
compaction summaries, and each worker's requests. The counters are held at run level
rather than on anything the run tears down, so a run whose agent never returned still
exports what it spent.

Grouped by ``_chat_phase``, the label
:mod:`~nexus.extensions.tasks.hitl.trace.labels` puts on the span and the same
field ``ChatTrackerProcessor`` groups trajectories by, rather than by ``_client_key``
as the upstream ``token_counter`` does -- HITL runs the user and every worker through
one client, which one bucket per client key would merge.
"""

import logging

from collections.abc import Mapping
from typing import Any, cast

from nexus.extensions.tasks.hitl.trace.labels import UNATTRIBUTED
from nexus.extensions.tasks.hitl.trace.usage.parse import model_of
from nexus.extensions.tasks.hitl.trace.usage.profiles import DEFAULT_PROFILE, UsageProfile
from nexus.extensions.tasks.hitl.trace.usage.tracker import UsageTracker, combine
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.types import Span


logger = logging.getLogger(__name__)

COMPLETION_SPAN = 'openai_completion'
LABEL_ATTRIBUTE = '_chat_phase'
CLIENT_ATTRIBUTE = '_client_key'


@span_processor_registry.register('hitl_usage')
class HITLUsageProcessor(SpanProcessor):
    """Accumulate token usage per trace label for one run.

    ``profiles`` maps ``_client_key`` to the profile for that client;
    ``default_profile`` covers a client not named there. A profile records which
    provider family produced a label's numbers -- every profile reads every key any
    of them uses, so a wrong name mislabels provenance rather than losing a counter.
    Configured from a recipe generator::

        processors:
          usage:
            name: hitl_usage
            args:
              profiles: {worker: bedrock_claude}
              default_profile: default
    """

    def __init__(
        self,
        profiles: Mapping[str, str] | None = None,
        default_profile: UsageProfile = DEFAULT_PROFILE,
    ) -> None:
        self.profiles: dict[str, UsageProfile] = {
            key: cast(UsageProfile, value) for key, value in (profiles or {}).items()
        }
        self.default_profile: UsageProfile = default_profile
        self._by_label: dict[str, UsageTracker] = {}
        self._profiles_seen: dict[str, str] = {}

    def profile_for(self, client_key: object) -> UsageProfile:
        """The profile for the client that made a call."""
        if isinstance(client_key, str):
            return self.profiles.get(client_key, self.default_profile)
        return self.default_profile

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        del event
        if span.name != COMPLETION_SPAN:
            return

        outputs = span.attributes.get('outputs')
        if not isinstance(outputs, dict):
            return

        label = span.attributes.get(LABEL_ATTRIBUTE)
        label = label if isinstance(label, str) and label else UNATTRIBUTED
        profile = self.profile_for(span.attributes.get(CLIENT_ATTRIBUTE))

        # Recorded so a reader can tell which profile produced a label's numbers,
        # and see it if one label somehow spans two clients.
        self._profiles_seen[label] = profile

        tracker = self._by_label.setdefault(label, UsageTracker(profile=profile))
        tracker.record(outputs, model=_span_model(span), profile=profile, messages=_span_messages(span))

    def get_result(self) -> dict[str, Any]:
        """Per-label totals, and the run's own totals across every label."""
        by_label = {label: tracker.summary() for label, tracker in self._by_label.items()}
        return {
            'by_label': by_label,
            'totals': combine(by_label.values()),
            'profiles': dict(self._profiles_seen),
            'default_profile': self.default_profile,
        }


def _span_model(span: Span) -> str:
    """The model this call used, from the response and then the request."""
    for attribute in ('outputs', 'inputs'):
        value = span.attributes.get(attribute)
        if isinstance(value, dict) and (model := model_of(value)):
            return model
    return ''


def _span_messages(span: Span) -> int:
    """How many messages this request carried, or 0 when the span does not say.

    The same count is derivable from the chat tracker's export, but that block runs to
    megabytes against this processor's kilobyte, so a reader after the shape of a
    conversation would have to parse all of it. The span already carries the list.
    """
    inputs = span.attributes.get('inputs')
    if not isinstance(inputs, dict):
        return 0
    messages = inputs.get('messages')
    return len(messages) if isinstance(messages, list) else 0


__all__ = [
    'CLIENT_ATTRIBUTE',
    'COMPLETION_SPAN',
    'LABEL_ATTRIBUTE',
    'HITLUsageProcessor',
]
