"""Request identity checks on the sandbox side of the model bridge."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .gateway_proxy import WorkBuddyOpenAIProxy
from .metadata_proxy import WorkBuddyMetadataProxy
from .prompt_identity import PromptIdentityError, WorkBuddyPromptGuard


class WorkBuddyProxy(WorkBuddyOpenAIProxy):
    """WorkBuddyOpenAIProxy plus the rollout prompt check.

    The check runs on the request WorkBuddy sent, before the gateway proxy pins
    template kwargs or stamps identity metadata onto it.
    """

    def __init__(self, *args: Any, verify_prompt: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_guard = WorkBuddyPromptGuard() if verify_prompt else None

    def _prepare_request_body(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.prompt_guard is not None:
            try:
                self.prompt_guard.check(body)
            except PromptIdentityError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return super()._prepare_request_body(body)

    def _inspect_response_payload(self, payload: dict[str, Any]) -> None:
        super()._inspect_response_payload(payload)
        if self.prompt_guard is not None:
            self.prompt_guard.inspect_response(payload)


class WorkBuddyMetadataCaptureProxy(WorkBuddyProxy, WorkBuddyMetadataProxy):
    """WorkBuddyProxy that also saves raw request envelopes (opt-in, see WorkBuddyMetadataProxy)."""


__all__ = ['WorkBuddyMetadataCaptureProxy', 'WorkBuddyProxy']
