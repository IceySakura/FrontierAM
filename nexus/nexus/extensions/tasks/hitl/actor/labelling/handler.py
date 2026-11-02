"""LLM request handler that attributes each request to its session.

The runtime's ``llm_request`` stream is shared by every session in a sandbox and
its envelope carries no session identity. This handler takes the identity from
the request itself: the sandbox stamps a label onto ``openai_params`` (see
:mod:`~nexus.extensions.tasks.hitl.actor.labelling.proxy`), and this handler
strips it back off and holds it for the duration of the completion call, so the
span carries it and one run-level ``ChatTrackerProcessor`` can separate every
conversation.

Token accounting is not done here. The same label reaches the span, so
``HITLUsageProcessor`` counts every call at run level -- including the host user
agent's, which never passes through this handler.

Overrides ``LLMRequestHandler._produce_response``, which sits inside the replay
guard, so the labelled call inherits deduplication, the rsio ack/heartbeat protocol
and the error-code path on failure.
"""

import logging

from typing import Any

from nexus.extensions.tasks.hitl.trace import labels as labels_module
from nexus.extensions.tasks.hitl.trace.labels import (
    UNATTRIBUTED,
    assert_no_label_fields,
    labelled,
    strip_label,
)
from nexus.runtime.server.services.llm_handler import LLMRequestHandler


logger = logging.getLogger(__name__)


class LabelledLLMRequestHandler(LLMRequestHandler):
    """Handler that attributes every sandbox LLM call to its session."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # A stock OpenAICompatibleClient has no _chat_phase, so without this its
        # spans carry no label and every trajectory exports unattributed.
        labels_module.enable_labels(self.openai_client)
        self._label_by_request: dict[str, str] = {}
        self._unattributed_requests = 0

    @property
    def unattributed_requests(self) -> int:
        """Sandbox requests that arrived without a session label.

        Counted rather than raised: this handler serves requests from inside a
        sandbox, where every one is supposed to be stamped by the proxy, so a
        non-zero count here means the stamping path was bypassed. That is worth
        seeing but is not a reason to abort a run mid-turn.

        Unlike ``labels.unattributed_count``, which also counts host-side calls
        made outside any :func:`~...labels.labelled` scope and where a non-zero
        value is normal.
        """
        return self._unattributed_requests

    def label_for(self, request_id: str) -> str | None:
        """The label this handler used for ``request_id``, if it saw it."""
        return self._label_by_request.get(request_id)

    async def _produce_response(
        self,
        request_id: str,
        data: dict[str, Any],
        *,
        rsio_mode: bool = True,
    ) -> dict[str, Any]:
        """Strip the label, then run the base call under it.

        Nothing may be awaited between entering ``labelled`` and the base call:
        ``@observe`` snapshots the label synchronously before awaiting, which is
        what lets one client instance serve concurrent sessions.
        """
        label, params = strip_label(data)

        # sanitize_openai_params filters by a fixed denylist, so an unstripped
        # key reaches the provider as a completion parameter.
        assert_no_label_fields(params)

        if label is None:
            self._unattributed_requests += 1
            logger.warning(
                'llm_request %s carries no session label; its trajectory will be recorded as %s',
                request_id,
                UNATTRIBUTED,
            )

        self._label_by_request[request_id] = label or UNATTRIBUTED

        with labelled(self.openai_client, label):
            return await super()._produce_response(request_id, params, rsio_mode=rsio_mode)


__all__ = ['LabelledLLMRequestHandler']
