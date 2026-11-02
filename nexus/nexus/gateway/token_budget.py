"""Context-aware token budget accounting for one Gateway session."""

from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from typing import Any

import numpy as np

from nexus.core import ContextLengthExceededError
from nexus.gateway.types import TOKEN_ID_DTYPE
from nexus.utils.token_budget_config import TokenBudgetComponent, parse_token_budget_include


@dataclass(frozen=True)
class _PromptDelta:
    added_observation_tokens: int
    rebase_context: bool


class TokenBudgetState:
    """Track trajectory and context-local token consumption.

    The session charges classified tool continuations before generation. Explicit
    prompt updates may also account for strict external extensions. Committing a
    generation only advances the prompt baseline and charges its sampled output.
    Model capacity is supplied per request and is not part of the budget state.
    """

    def __init__(
        self,
        *,
        token_budget: int | None = None,
        token_budget_include: list[str] | None = None,
    ) -> None:
        self._token_budget = token_budget
        self._token_budget_components = parse_token_budget_include(token_budget_include)
        self._initial_prompt_lens: dict[Any, int] = {}
        self._token_consumed = 0
        self._segment_consumed_tokens: dict[Any, int] = {}
        self._accounted_prompt_ids: dict[Any, np.ndarray] = {}
        self._accounted_tool_observation_suffixes: set[tuple[Any, bytes]] = set()

    def prepare_generation(
        self,
        context_key: Any,
        raw_prompt_ids: np.ndarray,
        prompt_ids: np.ndarray,
        request_kwargs: dict[str, Any],
        uses_default_max_tokens: bool,
        *,
        max_model_len: int | None = None,
    ) -> dict[str, Any]:
        """Bound generation by every configured request, context, and trajectory limit.

        Default max_tokens covers prompt growth within the current context;
        explicit max_tokens bounds only this request. Both modes enforce the
        selected model's remaining capacity and the shared trajectory budget.
        """
        # Budget checks are read-only: compare against the last successfully committed stream.
        delta = self._inspect_prompt(context_key, raw_prompt_ids)
        initial_prompt_len = self._initial_prompt_lens.get(context_key, len(prompt_ids))
        if delta.rebase_context:
            # A rewritten prompt is only a candidate baseline until committed.
            initial_prompt_len = len(prompt_ids)

        params = dict(request_kwargs)
        limits: list[int] = []
        configured_max_tokens = params.get('max_tokens')
        if configured_max_tokens is not None:
            request_limit = int(configured_max_tokens)
            if uses_default_max_tokens:
                request_limit -= len(prompt_ids) - initial_prompt_len
            limits.append(request_limit)
        if max_model_len is not None:
            limits.append(max_model_len - len(prompt_ids))
        if self._token_budget is not None:
            limits.append(self._token_budget - self._token_consumed)
        if not limits:
            return params

        remaining = min(limits)
        if remaining <= 0:
            raise ContextLengthExceededError(
                f'Gateway context length exceeded: max_tokens={params.get("max_tokens")}, '
                f'initial_prompt_len={initial_prompt_len}, prompt_ids={len(prompt_ids)}'
            )
        params['max_tokens'] = remaining
        return params

    def account_tool_observation_suffix(
        self,
        context_key: Any,
        raw_prompt_ids: np.ndarray,
        messages: list[dict],
        suffix_token_count: int,
    ) -> None:
        """Charge a classified tool append without committing a candidate prompt baseline."""
        previous_ids = self._accounted_prompt_ids.get(context_key, np.empty(0, dtype=TOKEN_ID_DTYPE))
        if len(previous_ids) == 0 or self._common_prefix_length(previous_ids, raw_prompt_ids) != len(previous_ids):
            return
        # Match Legacy's full-history identity: manager updates and repeated failed
        # requests share a charge, while identical results in later rounds do not.
        serialized_history = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode()
        signature = (context_key, hashlib.sha256(serialized_history).digest())
        if signature in self._accounted_tool_observation_suffixes:
            return
        self._accounted_tool_observation_suffixes.add(signature)
        if TokenBudgetComponent.TOOL_OBSERVATION in self._token_budget_components:
            self._token_consumed += suffix_token_count
        self._segment_consumed_tokens[context_key] = (
            self._segment_consumed_tokens.get(context_key, 0) + suffix_token_count
        )

    def update_prompt(
        self,
        context_key: Any,
        raw_prompt_ids: np.ndarray,
        prompt_ids: np.ndarray,
        *,
        consume: bool = True,
    ) -> None:
        delta = self._inspect_prompt(context_key, raw_prompt_ids)
        if delta.rebase_context:
            self._initial_prompt_lens.pop(context_key, None)
        added_observation_tokens = delta.added_observation_tokens if consume else 0
        if TokenBudgetComponent.TOOL_OBSERVATION in self._token_budget_components:
            self._token_consumed += added_observation_tokens
        self._segment_consumed_tokens[context_key] = (
            self._segment_consumed_tokens.get(context_key, 0) + added_observation_tokens
        )
        self._accounted_prompt_ids[context_key] = np.array(raw_prompt_ids, dtype=TOKEN_ID_DTYPE)
        self._initial_prompt_lens.setdefault(context_key, len(prompt_ids))

    def record_generation(
        self,
        context_key: Any,
        raw_prompt_ids: np.ndarray,
        prompt_ids: np.ndarray,
        generated_ids: list[int],
        *,
        output_component: TokenBudgetComponent = TokenBudgetComponent.AGENT_OUTPUT,
    ) -> None:
        # Observations were charged before dispatch; only commit the prompt baseline here.
        self.update_prompt(context_key, raw_prompt_ids, prompt_ids, consume=False)
        if output_component in self._token_budget_components:
            self._token_consumed += len(generated_ids)
        self._segment_consumed_tokens[context_key] = self._segment_consumed_tokens.get(context_key, 0) + len(
            generated_ids
        )
        self._accounted_prompt_ids[context_key] = np.concatenate((
            self._accounted_prompt_ids[context_key],
            np.asarray(generated_ids, dtype=TOKEN_ID_DTYPE),
        ))

    def segment_consumed_tokens(self, context_key: Any) -> int:
        return self._segment_consumed_tokens.get(context_key, 0)

    def set_segment_consumed_tokens(self, context_key: Any, value: int) -> None:
        self._segment_consumed_tokens[context_key] = value

    def _inspect_prompt(self, context_key: Any, raw_prompt_ids: np.ndarray) -> _PromptDelta:
        previous_ids = self._accounted_prompt_ids.get(context_key, np.empty(0, dtype=TOKEN_ID_DTYPE))
        common_prefix_length = self._common_prefix_length(previous_ids, raw_prompt_ids)
        extends_previous = len(previous_ids) > 0 and common_prefix_length == len(previous_ids)
        return _PromptDelta(
            added_observation_tokens=len(raw_prompt_ids) - common_prefix_length if extends_previous else 0,
            rebase_context=len(previous_ids) > 0 and not extends_previous,
        )

    @staticmethod
    def _common_prefix_length(previous_ids: np.ndarray, prompt_ids: np.ndarray) -> int:
        shared_length = min(len(previous_ids), len(prompt_ids))
        mismatches = np.flatnonzero(previous_ids[:shared_length] != prompt_ids[:shared_length])
        return int(mismatches[0]) if mismatches.size else shared_length
