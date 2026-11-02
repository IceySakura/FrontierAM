"""Per-session attribution for a shared ``llm_request`` stream.

The two ends of one mechanism. A sandbox holds several worker sessions whose LLM
calls all arrive on the runtime's single ``llm_request`` channel, and the envelope
carries no session identity:

* :mod:`.proxy` runs in the sandbox and stamps the session's label onto
  ``openai_params``;
* :mod:`.handler` runs on the host, strips it back off before the call reaches the
  provider, and holds it for the duration of the completion so the span carries it.

One run-level ``ChatTrackerProcessor`` then separates every conversation. The
label vocabulary and the read-back side live in
:mod:`nexus.extensions.tasks.hitl.trace.labels`.
"""

from nexus.extensions.tasks.hitl.actor.labelling.handler import LabelledLLMRequestHandler
from nexus.extensions.tasks.hitl.actor.labelling.proxy import (
    attach_label,
    build_labelled_proxy_class,
)


__all__ = [
    'LabelledLLMRequestHandler',
    'attach_label',
    'build_labelled_proxy_class',
]
