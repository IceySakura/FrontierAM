"""Judge-scoped transport deduplication; independent of model HTTP retries."""

import sys
import time

from typing import Any

from nexus.runtime.server.rpc_replay_guard import RPCReplayGuard
from nexus.runtime.server.services.llm_handler import LLMRequestHandler


class JudgeLLMRequestHandler(LLMRequestHandler):
    """Retain same-ID outcomes for the whole bounded Judge invocation.

    The shared handler exposes no replay policy constructor argument. Replace its
    private guard once, before registering, without changing the shared defaults.
    No capacity eviction during this invocation: evicting a completed outcome
    could execute a delayed resend twice. The Task owns this handler, unregisters
    it at Judge cleanup, and releases the cache with the handler's lifetime.
    """

    def __init__(self, *, judge_deadline: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.replay_retention_seconds = max(0.0, judge_deadline - time.monotonic()) + 60.0
        self._guard = RPCReplayGuard(
            ttl_seconds=self.replay_retention_seconds,
            max_entries=sys.maxsize,
        )
