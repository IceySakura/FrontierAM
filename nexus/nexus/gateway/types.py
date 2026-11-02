"""Core data structures for the Gateway module."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pydantic import BaseModel, ConfigDict, model_validator


TOKEN_ID_DTYPE: type[np.int32] = np.int32
LOGPROB_DTYPE: type[np.float32] = np.float32


@dataclass(frozen=True)
class AliasCandidate:
    """A sampled reply judged against its chain's first resend, accepted when ``rejection`` is None."""

    node_id: int
    identity: Hashable
    rejection: str | None = None


@dataclass(frozen=True, eq=False)
class FrozenPrefix:
    """The chain's frozen physical stream as per-message spans and token IDs.

    Spans and IDs are exactly as recorded at append time and are never re-derived
    or re-encoded. Semantic message matching authorizes reuse; a fresh rendering
    supplies only the continuation and may not replace this stream. Internal span
    boundaries carry no authority: only the concatenation is load-bearing, and it
    ends at the last sampled token.

    ``raw_message_token_ids`` retains the engine's spans where media widened the training
    ones; an empty tuple means the engine saw exactly the training stream.
    Token arrays are borrowed from sealed accounts and must not be mutated by renderers.

    frozen_routing_length counts the leading expanded positions whose routing prefix Nodes already
    own, so this request's router replay capture begins right after them.

    ``node_ids`` are the Nodes the stream was spliced from. Recording attaches the
    request's first ``count`` messages to exactly these Nodes instead of matching again.

    ``alias_candidates`` are judged by selection but applied by recording, so read-only callers
    have no side effects.
    """

    count: int
    text: str
    templated_messages: tuple[str, ...]
    message_token_ids: tuple[np.ndarray, ...]
    raw_message_token_ids: tuple[np.ndarray, ...] = ()
    frozen_routing_length: int = 0
    node_ids: tuple[int, ...] = ()
    alias_candidates: tuple[AliasCandidate, ...] = ()

    @property
    def rollout_message_token_ids(self) -> tuple[np.ndarray, ...]:
        """Per-message spans as the rollout engine saw them (raw where media widened them)."""
        return self.raw_message_token_ids or self.message_token_ids


class BackendResult(BaseModel):
    """Unified backend return type.

    Token-level backends (vLLM replica) fill token_ids/log_probs/routed_experts.
    Message-level backends (OpenAI API) fill message/usage.
    """

    # Token-level output (vLLM replica fills these)
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None
    routed_experts: Any | None = None
    stop_reason: str | None = None
    wire_text: str | None = None
    inference_time: float | None = None
    # Message-level output (API fills these)
    message: dict | None = None
    usage: dict | None = None
    id: str | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choice_logprobs: Any | None = None
    choice_provider_specific_fields: dict[str, Any] | None = None
    # Shared
    extra_fields: dict[str, Any] = {}

    @model_validator(mode='after')
    def validate_token_facts(self) -> BackendResult:
        if self.token_ids is None:
            return self
        token_count = len(self.token_ids)
        if self.log_probs is not None and len(self.log_probs) != token_count:
            raise ValueError(f'log_probs length {len(self.log_probs)} != token_ids length {token_count}')
        indices = self.extra_fields.get('resp_topk_logprob_indice')
        values = self.extra_fields.get('resp_topk_logprob_value')
        if (indices is None) != (values is None):
            raise ValueError('top-k indices and values must be present together')
        if indices is not None and values is not None:
            if len(indices) != token_count or len(values) != token_count:
                raise ValueError('top-k rows must align with token_ids')
            for row, (index_row, value_row) in enumerate(zip(indices, values, strict=True)):
                if len(index_row) != len(value_row):
                    raise ValueError(f'top-k row {row} width mismatch')
        return self


class PromptRender(BaseModel):
    """Rendered prompt partitioned by message in the training and raw engine views.

    ``token_ids`` and ``message_token_ids`` hold the training view, the way
    ``TokenAccount.token_ids`` does. ``raw_message_token_ids`` holds the engine's spans
    when media expansion made the two differ, and is None when the engine saw exactly the
    training stream.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    token_ids: np.ndarray
    templated_messages: list[str]
    message_token_ids: list[np.ndarray]
    raw_message_token_ids: list[np.ndarray] | None = None
    generation_prompt: str
    generation_prompt_ids: np.ndarray
    message_media_counts: list[dict[str, int]] | None = None
    """Per-message image counts, partitioned the way ``message_token_ids`` is."""

    @property
    def rollout_message_token_ids(self) -> list[np.ndarray]:
        """Per-message spans as the rollout engine saw them (raw where media widened them)."""
        return self.raw_message_token_ids if self.raw_message_token_ids is not None else self.message_token_ids

    @property
    def rollout_token_ids(self) -> np.ndarray:
        """The whole prompt as the rollout engine saw it, generation prompt included."""
        if self.raw_message_token_ids is None:
            return self.token_ids
        return np.concatenate((*self.raw_message_token_ids, self.generation_prompt_ids))

    @model_validator(mode='after')
    def validate_partition(self) -> PromptRender:
        if len(self.templated_messages) != len(self.message_token_ids):
            raise ValueError('templated message and token counts differ')
        if ''.join(self.templated_messages) + self.generation_prompt != self.text:
            raise ValueError('templated messages do not reconstruct the rendered prompt')
        token_ids = np.concatenate((*self.message_token_ids, self.generation_prompt_ids))
        if not np.array_equal(token_ids, self.token_ids):
            raise ValueError('templated message tokens do not reconstruct the prompt token IDs')
        if self.message_media_counts is not None and len(self.message_media_counts) != len(self.message_token_ids):
            raise ValueError('message media counts differ in length from token partitions')
        return self


@dataclass(frozen=True)
class RequestMultiModal:
    """The media one request carries, in the two forms its consumers need.

    ``multi_modal_data`` holds the source media the engine expands itself, keyed by modality;
    ``multi_modal_items`` holds the per-item processor tensors the training path reads, keyed
    the same way. Both stay out of ``PromptRender``, which is wire data.
    """

    multi_modal_data: dict[str, Any] = field(default_factory=dict)
    multi_modal_items: dict[str, list[dict[str, Any]]] | None = None


class ProcessorResult(BaseModel):
    """Return type of Processor.create_completion(). Transparent to session layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: dict  # OpenAI-format assistant message
    # Token-level (only filled by token path)
    token_ids: list[int] | None = None
    prompt_ids: np.ndarray | None = None
    raw_prompt_ids: np.ndarray | None = None
    log_probs: list[float] | None = None
    routed_experts: Any | None = None
    # NOTE(arlenxu): tensors, so this stays in-memory/Ray form and never reaches the
    # JSON response payload.
    multi_modal_items: dict[str, list[dict[str, Any]]] | None = None
    stop_reason: str | None = None
    completion_text: str | None = None
    prompt_render: PromptRender | None = None
    inference_time: float | None = None
    wall_inference_time: float | None = None
    # Message-level (only filled by message path)
    usage: dict | None = None
    id: str | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choice_logprobs: Any | None = None
    choice_provider_specific_fields: dict[str, Any] | None = None
    # Shared
    extra_fields: dict[str, Any] = {}


class GatewayCompletionResult(BaseModel):
    """External return type of create_completion."""

    message: dict
    token_ids: list[int] | None = None
    log_probs: list[float] | None = None
    routed_experts: Any | None = None
    stop_reason: str | None = None
    usage: dict | None = None
    id: str | None = None
    created: int | None = None
    model: str | None = None
    object: str | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choice_logprobs: Any | None = None
    choice_provider_specific_fields: dict[str, Any] | None = None
    extra_fields: dict[str, Any] = {}
    recorded: bool = True
    inference_time: float | None = None
    wall_inference_time: float | None = None
