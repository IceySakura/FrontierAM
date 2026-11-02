"""HyTuner OpenAI-compatible client for RL training.

This client bridges Nexus agents to HyTuner's AsyncLLMServerManager,
enabling RL training of Nexus agents without modifying agent code.
"""

# /// script
# dependencies = ["numpy"]
#
# [project]
# name = "hytuner"
# authors = [
#     { name = "tarschen" },
#     { name = "luxyhliu" },
#     { name = "kumohu" },
#     { name = "petersheng" },
# ]
#
# [tool.nexus]
# openai_clients = { hytuner = "nexus.extensions.openai_clients.hytuner" }
# ///

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
import uuid

from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple, cast

import numpy as np

from pydantic import BaseModel, ConfigDict

from nexus.core import (
    UNPARSED_TOOL_CALL_FIELD,
    ContextLengthExceededError,
    ExtendedCompletionTokensDetails,
    OpenAICompatibleClient,
    OpenAIConfig,
    TokenCountResult,
    as_tokenizer_panic_error,
    openai_client_registry,
)
from nexus.core.context import require_context
from nexus.core.types import (
    ExtendedChatCompletion,
    ExtendedChoice,
    ExtendedMessage,
    ExtendedPromptTokensDetails,
    ExtendedUsage,
)
from nexus.trace import get_current_span_id, update_span
from nexus.utils.cache import cacheable
from nexus.utils.timeout import RUNNER_TIMEOUT_PAUSE_GROUP_KEY, ExecutionTimeoutController, async_timeout
from nexus.utils.timeout import capture_execution_timeouts as _capture_execution_timeouts
from nexus.utils.token_budget_config import TokenBudgetComponent, parse_token_budget_include
from nexus.utils.tool_call_ids import ToolCallIdGenerator


logger = logging.getLogger(__name__)


def _token_output_inference_time(token_output: Any, fallback: float) -> float:
    """Return one generate call's reported inference time, or its wall-clock fallback."""
    extra_fields = getattr(token_output, 'extra_fields', None) or {}
    raw_value = extra_fields.get('_inference_time')
    if raw_value is None:
        return max(0.0, fallback)
    try:
        inference_time = float(raw_value)
    except (TypeError, ValueError):
        return max(0.0, fallback)
    if not np.isfinite(inference_time) or inference_time < 0.0:
        return max(0.0, fallback)
    return inference_time


def _find_token_subsequence(token_ids: list[int], marker_ids: list[int]) -> int | None:
    """Return the first token offset of ``marker_ids`` in ``token_ids``."""
    if not marker_ids or len(marker_ids) > len(token_ids):
        return None
    last_start = len(token_ids) - len(marker_ids)
    for start in range(last_start + 1):
        if token_ids[start : start + len(marker_ids)] == marker_ids:
            return start
    return None


def _split_completion_token_usage(
    token_ids: list[int],
    think_end_ids: list[int],
    *,
    has_reasoning_boundary: bool,
    has_unclosed_reasoning: bool,
) -> tuple[int | None, int | None]:
    """Split generated tokens into additive reasoning and message buckets."""
    if has_unclosed_reasoning:
        return len(token_ids), 0
    if not has_reasoning_boundary:
        return 0, len(token_ids)

    end_start = _find_token_subsequence(token_ids, think_end_ids)
    if end_start is None:
        return None, None
    reasoning_tokens = end_start + len(think_end_ids)
    return reasoning_tokens, len(token_ids) - reasoning_tokens


def _resolve_effective_response_limit(
    *,
    sampling_params: dict[str, Any],
    max_tokens_budget: int | None,
    max_model_len: int | None,
    initial_prompt_len: int,
    sampling_params_enforced: bool,
    per_call_sampling_has_max_tokens: bool,
    request_meta: Any,
) -> int | None:
    """Return the effective whole-segment response limit exposed to HyTuner.

    A self-summary is a standalone one-request segment, so its request budget is
    also its segment budget. Enforced and other per-call limits on ordinary
    conversation turns do not cap the accumulated multi-turn segment; in that
    case the model window remains the only whole-segment limit.
    """
    model_response_limit = None if max_model_len is None else max_model_len - initial_prompt_len
    invocation_kind = request_meta.get('invocation_kind') if isinstance(request_meta, dict) else None
    if invocation_kind == 'summary_inference':
        requested_response_limit = max_tokens_budget
    elif sampling_params_enforced or per_call_sampling_has_max_tokens:
        requested_response_limit = None
    else:
        requested_response_limit = sampling_params.get('max_tokens')

    limits = [limit for limit in (model_response_limit, requested_response_limit) if limit is not None]
    if any(isinstance(limit, bool) or not isinstance(limit, int) for limit in limits):
        raise TypeError(f'effective response limits must be integers or null, got {limits!r}')
    return min(limits) if limits else None


class _PausedDeadline(NamedTuple):
    """A task deadline suspended by :func:`_pause_task_timeout`."""

    # An ``asyncio.Timeout``; kept untyped because ``Context.timeout_handle`` is ``Any``.
    handle: Any
    deadline: float


_ABORT_TIMEOUT_PAUSE_GROUP_KEY = RUNNER_TIMEOUT_PAUSE_GROUP_KEY


def _pause_task_timeout(request_id: str, *, reject_expired: bool = False) -> _PausedDeadline | None:
    """Suspend the enclosing ``asyncio.timeout`` deadline, returning what to restore.

    ``reschedule`` raises once the deadline has already fired. That RuntimeError must
    not escape: it would reach ``Timeout.__aexit__`` ahead of the cancellation the
    deadline just requested, and only a ``CancelledError`` arriving there gets
    converted into a ``TimeoutError``. Anything else leaves the cancellation to
    surface bare, outside the timeout scope, where nothing handles it.

    Returning ``None`` means the deadline was left untouched, so the caller must not
    try to restore it.
    """
    try:
        timeout_handle = require_context().timeout_handle
        if timeout_handle is None:
            return None
        original_deadline = timeout_handle.when()
        if original_deadline is None:
            return None
        if reject_expired and original_deadline <= asyncio.get_running_loop().time():
            return None
        timeout_handle.reschedule(None)
    except Exception as e:
        logger.warning(f'async timeout pause failed, request_id:{request_id}, err_msg:{e}')
        return None
    return _PausedDeadline(timeout_handle, original_deadline)


def _restore_task_timeout(paused: _PausedDeadline | None, wait_duration: float, request_id: str) -> None:
    """Re-arm a deadline suspended by :func:`_pause_task_timeout`, pushed out by the wait."""
    if paused is None:
        return
    try:
        paused.handle.reschedule(paused.deadline + wait_duration)
    except Exception as e:
        logger.warning(
            f'async timeout resume failed, the task now runs without a deadline, request_id:{request_id}, err_msg:{e}'
        )


async def _update_runtime_command_timeout(timeout_control_id: str, *, pause: bool, request_id: str) -> None:
    context = require_context()
    runtime = await context.get_runtime()
    action = 'pause' if pause else 'resume'
    target = f'nexus.runtime.executors.command::{action}_command_timeout'
    timeout_seconds = context.runtime_configs.get('main', {}).get('sandbox_timer_rpc_timeout_s')
    if timeout_seconds is None:
        # Callers that do not configure a control timeout keep the runtime request timeout.
        updated = await runtime.invoke(target, timeout_control_id, venv_id=runtime.default_venv_id)
    else:
        try:
            async with async_timeout(timeout_seconds):
                updated = await runtime.invoke(
                    target, timeout_control_id, venv_id=runtime.default_venv_id, request_timeout=timeout_seconds
                )
        # async-timeout raises asyncio.TimeoutError, which is not the builtin TimeoutError on Python 3.10.
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                f'Sandbox timer RPC did not finish within {timeout_seconds}s; action={action} '
                f'timeout_control_id={timeout_control_id!r} request_id={request_id}'
            ) from error
    if updated is not True:
        raise RuntimeError(
            f'Runtime command timeout {timeout_control_id!r} could not {action}: expected True, got {updated!r}; '
            f'request_id={request_id}; venv_id={runtime.default_venv_id!r}. '
            'Verify timer registration and host/runtime code alignment.'
        )


class _AbortTimeoutPauseGroup:
    """Pause shared task/command deadlines from the first waiter to the last.

    Model callbacks share their ExecutionContext. Cancelling one callback must
    not resume the deadlines while another is still parked on the abort gate.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._holder_count = 0
        self._paused_deadline: _PausedDeadline | None = None
        self._pause_started_at: float | None = None
        self._timeout_control_id: str | None = None
        self.track_shared_pause = False
        self._pause_intervals: list[tuple[float, float]] = []

    async def acquire(self, request_id: str) -> None:
        async with self._lock:
            if self._holder_count > 0:
                self._holder_count += 1
                return

            self._paused_deadline = (
                _pause_task_timeout(request_id, reject_expired=True)
                if self.track_shared_pause
                else _pause_task_timeout(request_id)
            )
            self._pause_started_at = time.perf_counter()
            self._timeout_control_id = require_context().runtime_command_timeout_id
            self._holder_count = 1
            try:
                if self._timeout_control_id is not None:
                    await _update_runtime_command_timeout(self._timeout_control_id, pause=True, request_id=request_id)
            except BaseException:
                # A failed/cancelled RPC may already have paused the remote timer.
                try:
                    await self._restore_locked(request_id)
                except Exception:
                    # Callers classify the pause failure, so a failed compensation must not replace it.
                    logger.warning('Timer pause compensation failed; request_id=%s', request_id, exc_info=True)
                raise

    async def release(self, request_id: str) -> None:
        async with self._lock:
            if self._holder_count <= 0:
                raise RuntimeError(f'abort timeout pause lease underflow; request_id={request_id}')
            self._holder_count -= 1
            if self._holder_count == 0:
                await self._restore_locked(request_id)

    async def _restore_locked(self, request_id: str) -> None:
        try:
            if self._timeout_control_id is not None:
                await _update_runtime_command_timeout(self._timeout_control_id, pause=False, request_id=request_id)
        finally:
            assert self._pause_started_at is not None
            pause_ended_at = time.perf_counter()
            if self.track_shared_pause:
                self._pause_intervals.append((self._pause_started_at, pause_ended_at))
            _restore_task_timeout(self._paused_deadline, pause_ended_at - self._pause_started_at, request_id)
            self._holder_count = 0
            self._paused_deadline = None
            self._pause_started_at = None
            self._timeout_control_id = None

    def get_paused_time(self, started_at: float) -> float:
        """Return the shared pause union within one completion's lifetime."""
        duration = sum(end - max(start, started_at) for start, end in self._pause_intervals if end > started_at)
        if self._pause_started_at is not None:
            duration += max(0.0, time.perf_counter() - max(self._pause_started_at, started_at))
        return duration

    def credit_timeout(self, duration: float) -> None:
        """Retain response compensation until the shared deadline resumes."""
        if self._paused_deadline is not None:
            self._paused_deadline = self._paused_deadline._replace(deadline=self._paused_deadline.deadline + duration)


def capture_hytuner_execution_timeouts(
    request_id: str, *, include_inference: bool = True, track_shared_pause: bool = False
) -> ExecutionTimeoutController | None:
    """Capture Nexus execution budgets for HyTuner generation.

    Returns:
        A controller for the active Nexus execution context, or None when no
        ExecutionContext is active. Standalone inference scopes are not captured
        without an ExecutionContext; use nexus.utils.timeout.capture_execution_timeouts
        for inference-only control.
    """
    from nexus.core.context import _current_context

    if _current_context.get() is None:
        return None
    context = require_context()
    group = context.runtime_shared.get_optional(_ABORT_TIMEOUT_PAUSE_GROUP_KEY, _AbortTimeoutPauseGroup)
    if group is None:
        group = _AbortTimeoutPauseGroup()
        context.runtime_shared.set(_ABORT_TIMEOUT_PAUSE_GROUP_KEY, group)
    if track_shared_pause:
        group.track_shared_pause = True
    return _capture_execution_timeouts(request_id, include_inference=include_inference, shared_budget=group)


async def _settle_abort_timeout_update(
    group: _AbortTimeoutPauseGroup, *, acquire: bool, request_id: str
) -> asyncio.CancelledError | None:
    """Finish a pause/resume transition before propagating caller cancellation."""
    update_task = asyncio.create_task(group.acquire(request_id) if acquire else group.release(request_id))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(update_task)
            return cancellation
        except asyncio.CancelledError as error:
            if update_task.cancelled():
                raise
            if cancellation is None:
                cancellation = error


async def _wait_for_abort_resume(abort_event: asyncio.Event, request_id: str) -> float:
    """Wait for rollout resume without consuming the shared execution budgets."""
    if abort_event.is_set():
        return 0.0

    context = require_context()
    group = context.runtime_shared.get_optional(_ABORT_TIMEOUT_PAUSE_GROUP_KEY, _AbortTimeoutPauseGroup)
    if group is None:
        group = _AbortTimeoutPauseGroup()
        context.runtime_shared.set(_ABORT_TIMEOUT_PAUSE_GROUP_KEY, group)

    wait_start = time.perf_counter()
    acquired = False
    try:
        cancellation = await _settle_abort_timeout_update(group, acquire=True, request_id=request_id)
        acquired = True
        if cancellation is not None:
            raise cancellation
        await abort_event.wait()
    finally:
        if acquired:
            cancellation = await _settle_abort_timeout_update(group, acquire=False, request_id=request_id)
            if cancellation is not None:
                raise cancellation
    return time.perf_counter() - wait_start


def _is_expected_rpc_abort(error: Exception, abort_event: asyncio.Event | None) -> bool:
    if abort_event is None or abort_event.is_set():
        return False
    try:
        from ray.exceptions import TaskCancelledError
    except ImportError:
        return False
    return isinstance(error, TaskCancelledError)


class ChatCompletionRequest(BaseModel):
    """Minimal request context for tool parsing in Nexus."""

    model_config = ConfigDict(extra='ignore')
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


@openai_client_registry.register('hytuner')
class HyTunerClient(OpenAICompatibleClient):
    """OpenAI-compatible client backed by HyTuner's AsyncLLMServerManager.

    This client:
    1. Converts messages to token ids using the tokenizer
    2. Calls server_manager.generate() to get model outputs with logprobs
    3. Converts token outputs back to ChatCompletion format
    4. Records token-level info in span attributes for TrainingTrajectoryProcessor

    The client uses prefix matching for incremental tokenization: on each call
    it compares the incoming messages with the previously seen messages, reuses
    cached tokens for the common prefix, and only tokenizes the new suffix.
    This works transparently across compaction boundaries — when compaction
    shortens the message list, the prefix match naturally detects the change
    and retokenizes from the divergence point.

    Configuration via OpenAIConfig:
        config = OpenAIConfig(
            client_args={
                'tokenizer': tokenizer,
                'processor': processor,  # optional, for multimodal models
                'server_manager': server_manager,
                'tool_parser': tool_parser,
                'apply_chat_template_kwargs': {},
                'strict_prefix_check': False,  # optional: reject compaction/history rewrites
            },
            request_args={
                'sampling_params': {'temperature': 1.0, 'top_p': 1.0},
            }
        )
        client = HyTunerClient(config)

        # Use like any OpenAI client
        response = await client.create_completion(messages=[...], tools=[...])
    """

    supports_request_meta = True

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self._client_key = 'main'
        self.tokenizer = config.client_args.get('tokenizer')
        self.processor = config.client_args.get('processor')
        self.server_manager = config.client_args.get('server_manager')
        self.tool_parser = config.client_args.get('tool_parser')
        self.template_parser = config.client_args.get('template_parser')
        self.apply_chat_template_kwargs = config.client_args.get('apply_chat_template_kwargs', {})
        self.max_model_len = config.client_args.get('max_model_len', None)
        self.process_vision_info_fn = config.client_args.get('process_vision_info_fn')
        self.strict_prefix_check = bool(config.client_args.get('strict_prefix_check', False))
        # Abort/Resume control
        # In fact, it's a reversed event, i.e. when the event is cleared, the agent loop is aborted.
        self.abort_event = config.client_args.get('abort_event')
        self.wait_for_abort_resume: Callable[[asyncio.Event, str], Awaitable[float]] = config.client_args.get(
            'wait_for_abort_resume', _wait_for_abort_resume
        )

        self.sampling_params = config.request_args.get('sampling_params', {})

        self.request_id = str(uuid.uuid4())
        self.last_messages: list[dict] = []
        self.last_tokens: list[int] = []
        self.last_expanded_tokens: list[int] = []  # Tracks expanded (vision-token-expanded) token history
        self._last_rendered_text: str | None = None
        # Recovery is common on protocol replay; report it only once per client.
        self._logged_prefix_recovery = False
        self._last_effective_response_limit: int | None = None
        self._last_num_images: int = 0
        self._last_num_videos: int = 0
        self._last_num_audios: int = 0
        self._initial_prompt_len: int | None = None
        self._tool_call_ids = ToolCallIdGenerator()
        self._training_data_observer: Any | None = None

        self.priority = config.client_args.get('priority', 0)

        # Reasoning (think) tokens — read from tool_parser which knows the model's convention.
        self._think_start_token = getattr(self.tool_parser, 'think_start_token', '<think>')
        self._think_end_token = getattr(self.tool_parser, 'think_end_token', '</think>')

        # Track time spent waiting for abort event to resume
        self.abort_wait_time: float = 0.0
        # Wall-clock time around successful generate calls, before excluding
        # scheduler/queue waiting reported by the inference service.
        self.inference_time: float = 0.0
        self.true_inference_time: float = 0.0
        # compaction related
        self.token_budget = config.client_args.get('token_budget', None)
        self.token_budget_components = parse_token_budget_include(config.client_args.get('token_budget_include'))
        self._accounted_tool_observation_suffixes: set[bytes] = set()
        self.token_consumed = 0
        self.curr_segment_consumed_tokens = 0
        self.enable_rollout_routing_replay = bool(config.client_args.get('enable_rollout_routing_replay', False))
        self._routed_experts: np.ndarray | None = None

        # Routing is deliberately opt-in for auxiliary models: an unknown model must stay on
        # the main path so its token-level data is not silently dropped from the RL trajectory.
        auxiliary_model_names = config.client_args.get('auxiliary_model_names', [])
        if isinstance(auxiliary_model_names, str):
            raise TypeError('auxiliary_model_names must be a collection of model names, not a string')
        self.auxiliary_model_names: set[str] = {
            self._normalize_routing_model_name(name) for name in auxiliary_model_names
        }

    def set_training_data_observer(self, observer: Any | None) -> None:
        """Bind a per-run token recorder used when nested Runtime spans arrive late."""
        self._training_data_observer = observer

    def fork_for_agent(self, *, agent_id: str, parent_agent_id: str | None) -> HyTunerClient:
        """Create an isolated conversation client sharing the inference runtime."""
        del parent_agent_id
        child = type(self)(self.config)
        child._client_key = f'{self._client_key}:{agent_id}'
        child._training_data_observer = self._training_data_observer
        return child

    @staticmethod
    def _normalize_tool_call_arguments(messages: list[dict]) -> None:
        """Parse valid JSON tool arguments in-place, preserving malformed strings."""
        for m in messages:
            if m.get('tool_calls'):
                for tool in m['tool_calls']:
                    if isinstance(tool['function']['arguments'], str):
                        try:
                            tool['function']['arguments'] = json.loads(tool['function']['arguments'])
                        except json.JSONDecodeError:
                            continue

    @staticmethod
    def _normalize_routing_model_name(model: Any) -> str:
        if not isinstance(model, str) or not model.strip():
            raise TypeError('auxiliary model names must be non-empty strings')
        return model.strip().rsplit('/', 1)[-1].lower()

    def _is_auxiliary_model(self, model: Any) -> bool:
        if not isinstance(model, str) or not model.strip():
            return False
        normalized_model = self._normalize_routing_model_name(model)
        is_claude_side_model = 'claude' in normalized_model and (
            'sonnet' in normalized_model or 'haiku' in normalized_model
        )
        return is_claude_side_model or normalized_model in self.auxiliary_model_names

    @staticmethod
    def _messages_with_normalized_tool_calls(messages: list[dict]) -> list[dict]:
        """Non-mutating variant of ``_normalize_tool_call_arguments``."""

        def _norm_tool_call(tool: dict) -> dict:
            args = tool['function']['arguments']
            if not isinstance(args, str):
                return tool
            try:
                normalized_args = json.loads(args)
            except json.JSONDecodeError:
                return tool
            return {**tool, 'function': {**tool['function'], 'arguments': normalized_args}}

        return [
            {**m, 'tool_calls': [_norm_tool_call(tc) for tc in m['tool_calls']]} if m.get('tool_calls') else m
            for m in messages
        ]

    @staticmethod
    def _normalize_message_for_compare(msg: dict[str, Any]) -> dict[str, Any]:
        """Normalize a message for conservative prefix comparisons."""
        normalized: dict[str, Any] = {}
        for k, v in msg.items():
            if k in ('raw_response', 'raw_tool'):
                continue
            if v is None or v == '' or v == [] or v == {}:
                continue
            normalized[k] = v
        return normalized

    @staticmethod
    def _messages_equal(msg1: dict, msg2: dict) -> bool:
        """Compare two messages for equality, ignoring transient fields."""
        return HyTunerClient._normalize_message_for_compare(msg1) == HyTunerClient._normalize_message_for_compare(msg2)

    @staticmethod
    def _find_common_prefix(messages: list[dict], last_messages: list[dict] | None) -> int:
        """Find the number of leading messages that are identical between the two lists."""
        if not last_messages:
            return 0
        prefix_len = 0
        for old_msg, new_msg in zip(last_messages, messages, strict=False):
            if not HyTunerClient._messages_equal(old_msg, new_msg):
                break
            prefix_len += 1
        return prefix_len

    async def _canonicalize_render_equivalent_prefix(
        self,
        messages: list[dict],
        tools: list[dict[str, Any]] | None,
        last_message_count: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[dict] | None:
        """Recover the token cache when a replayed history differs only in how it is serialized.

        Agents that reach us through a protocol proxy (Codex over the Responses API, Claude Code
        over the Messages API) do not echo assistant turns back verbatim: they decompose each turn
        into structured items and rebuild a message from them. The result carries extra fields and
        rejoined text, so ``_messages_equal`` rejects it even when nothing about the conversation
        changed. Falling back to a full retokenize then re-encodes the model's own output, which
        BPE merges differently from the sampled ids, and every downstream consumer of those ids
        silently shifts.

        Only the rendered text decides tokenization, so rendering is the honest test. When the
        replayed history renders to exactly what we cached, swap in the cached messages and keep
        the sampled token ids.

        Returns the messages to use, or None when the histories really do differ.
        """
        if self._last_rendered_text is None or len(messages) < last_message_count:
            return None

        replayed_text = await asyncio.to_thread(
            self.apply_chat_template,
            messages[:last_message_count],
            tools,
            add_generation_prompt=False,
            is_first_msg=False,
            chat_template_kwargs=chat_template_kwargs,
        )
        if replayed_text != self._last_rendered_text:
            return None

        if not self._logged_prefix_recovery:
            self._logged_prefix_recovery = True
            logger.debug(
                'HyTunerClient request_id=%s: replayed history is not field-identical but renders '
                'identically; reusing the sampled token ids instead of retokenizing.',
                self.request_id,
            )
        return list(self.last_messages) + list(messages[last_message_count:])

    def _log_prefix_transition(self, messages: list[dict], rendered_text: str, **details: Any) -> None:
        """Log bounded, content-free evidence only when a prefix or budget changes."""
        message_diffs = []
        for index, (cached, incoming) in enumerate(zip(self.last_messages, messages, strict=False)):
            if self._messages_equal(cached, incoming):
                continue
            cached_fields = self._normalize_message_for_compare(cached)
            incoming_fields = self._normalize_message_for_compare(incoming)
            differing = sorted(
                key
                for key in set(cached_fields) | set(incoming_fields)
                if cached_fields.get(key) != incoming_fields.get(key)
            )
            message_diffs.append({'index': index, 'fields': differing})
            if len(message_diffs) == 3:
                break
        previous_text = self._last_rendered_text
        first_render_diff = None
        if previous_text is not None and not rendered_text.startswith(previous_text):
            first_render_diff = next(
                (
                    index
                    for index, (old, new) in enumerate(zip(previous_text, rendered_text, strict=False))
                    if old != new
                ),
                min(len(previous_text), len(rendered_text)),
            )
        logger.warning(
            'HyTunerClient prefix/budget transition: %s',
            json.dumps({
                'request_id': self.request_id,
                'completion_span_id': get_current_span_id(),
                'phase': 'budget_check',
                'cached_messages': len(self.last_messages),
                'incoming_messages': len(messages),
                'message_diffs': message_diffs,
                'first_render_diff': first_render_diff,
                **details,
            }),
        )

    async def update_token_consumption(self, messages, **kwargs):
        self._normalize_tool_call_arguments(messages)
        prefix_len = self._find_common_prefix(messages, self.last_messages)

        tools = kwargs.get('tools')
        first_part = await asyncio.to_thread(
            self.apply_chat_template,
            messages[:prefix_len],
            tools,
            add_generation_prompt=False,
            is_first_msg=False,
        )
        full = await asyncio.to_thread(
            self.apply_chat_template,
            messages,
            tools,
            is_first_msg=False,
        )
        if first_part != full[: len(first_part)]:
            raise RuntimeError('Failed to parse new part of messages for consistent tokens')
        new_text = full[len(first_part) :]
        new_tokens = await self._encode_prompt_text(
            new_text,
            operation='token-consumption prompt',
            message_count=len(messages),
        )
        self._account_tool_observation_suffix(
            messages,
            prefix_len,
            new_tokens,
            is_append=bool(self.last_messages) and prefix_len == len(self.last_messages),
        )

    def _account_tool_observation_suffix(
        self,
        messages: list[dict],
        prefix_len: int,
        new_tokens: list[int],
        *,
        is_append: bool,
    ) -> None:
        """Charge an appended tool-result suffix once while always advancing the segment counter."""
        appended_messages = messages[prefix_len:]
        if (
            not is_append
            or not appended_messages
            or any(message.get('role') != 'tool' for message in appended_messages)
        ):
            return

        # Use the full history as the idempotency key: update_token_consumption()
        # and _create_completion() may observe the same append, while identical
        # tool results produced in two different rounds must still be charged twice.
        serialized_history = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode()
        signature = hashlib.sha256(serialized_history).digest()
        if signature in self._accounted_tool_observation_suffixes:
            return
        self._accounted_tool_observation_suffixes.add(signature)

        if TokenBudgetComponent.TOOL_OBSERVATION in self.token_budget_components:
            self.token_consumed += len(new_tokens)
        self.curr_segment_consumed_tokens += len(new_tokens)

    async def _encode_multi_modal_ids(
        self,
        raw_text: str,
        *,
        images: Any = None,
        videos: Any = None,
        audios: Any = None,
    ) -> list[int]:
        """Expand multi-modal placeholders into the training-side id view.

        Keep this aligned with the multi-modal id encoding on the HyTuner training side.
        """
        assert self.processor is not None, '_encode_multi_modal_ids requires a processor'
        processor = self.processor

        processor_videos = list(videos) if videos else None
        video_metadata = None
        if processor_videos and isinstance(processor_videos[0], tuple):
            video_values, video_metadata = zip(*processor_videos, strict=False)
            processor_videos = list(video_values)
            video_metadata = list(video_metadata)

        processor_inputs: dict[str, Any] = {
            'text': [raw_text],
            'add_special_tokens': False,
            'return_tensors': 'pt',
        }
        if video_metadata is not None:
            processor_inputs['video_metadata'] = video_metadata
            processor_inputs['do_sample_frames'] = False
        if images:
            processor_inputs['images'] = images
        if processor_videos:
            processor_inputs['videos'] = processor_videos
        if audios:
            processor_inputs['audio'] = audios

        def _run() -> list[int]:
            model_inputs = processor(**processor_inputs)
            return model_inputs.pop('input_ids').squeeze(0).tolist()

        return await asyncio.to_thread(_run)

    # @observe('openai_auxiliary_completion', self_attrs=['_client_key', '_chat_phase'])
    # @cacheable(tags=['llm', 'completion'], ignore_args=['self'])
    async def _auxiliary_completion(
        self,
        messages,
        model=None,
        request_args_override: dict[str, Any] | None = None,
        **kwargs,
    ) -> ExtendedChatCompletion:
        """Serve a non-main-model request without mutating the main RL trajectory."""
        sampling_keys = (
            'temperature',
            'top_p',
            'top_k',
            'max_tokens',
            'min_tokens',
            'stop',
            'stop_token_ids',
            'seed',
            'frequency_penalty',
            'presence_penalty',
            'repetition_penalty',
        )

        def _extract_sampling_params(args: dict[str, Any]) -> dict[str, Any]:
            nested = args.get('sampling_params')
            if nested is not None and not isinstance(nested, dict):
                raise TypeError('sampling_params must be a dict or null')
            result = dict(nested or {})
            for key in sampling_keys:
                if key in args and args[key] is not None:
                    result[key] = args[key]
            return result

        base_request_args = copy.deepcopy(
            request_args_override if request_args_override is not None else self.config.request_args
        )
        per_call_sampling_params = _extract_sampling_params(kwargs)
        base_sampling_params = _extract_sampling_params(base_request_args)
        merged_sampling_params = {
            **base_sampling_params,
            **per_call_sampling_params,
        }
        request_args = {
            **base_request_args,
            **kwargs,
            'model': model,
            'messages': messages,
            'sampling_params': merged_sampling_params,
        }
        for key in sampling_keys:
            request_args.pop(key, None)

        effective_request_args = self._resolve_effective_request_args(request_args)
        messages = copy.deepcopy(effective_request_args.pop('messages'))
        effective_request_args.pop('model', None)
        sampling_params = _extract_sampling_params(effective_request_args)
        effective_request_args['sampling_params'] = sampling_params
        for key in sampling_keys:
            effective_request_args.pop(key, None)
        kwargs = effective_request_args

        override_chat_template_kwargs = kwargs.pop('chat_template_kwargs', None)
        if override_chat_template_kwargs is not None and not isinstance(override_chat_template_kwargs, dict):
            raise TypeError('chat_template_kwargs must be a dict or null')
        chat_template_kwargs = (
            override_chat_template_kwargs
            if override_chat_template_kwargs is not None
            else ({} if request_args_override is not None else self.apply_chat_template_kwargs)
        )

        tools = kwargs.get('tools')
        request: Any = ChatCompletionRequest(tools=tools, chat_template_kwargs=chat_template_kwargs)
        messages = self._messages_with_normalized_tool_calls(messages)
        allow_retry_for_tool_call = kwargs.get('allow_retry_for_tool_call', False)

        if request_args_override is not None:
            full_text = await asyncio.to_thread(
                self.tokenizer.apply_chat_template,  # type: ignore[possibly-missing-attribute]
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
                is_training=True,
                **chat_template_kwargs,
            )
        else:
            full_text = await asyncio.to_thread(
                self.apply_chat_template,
                messages,
                tools,
                is_first_msg=True,
            )
        prompt_ids = await asyncio.to_thread(
            self.tokenizer.encode,  # type: ignore[possibly-missing-attribute]
            full_text,
            add_special_tokens=False,
        )

        images, videos, audios = None, None, None
        if self.processor is not None and self.process_vision_info_fn is not None:
            multi_modal_data = await self.process_vision_info_fn(messages)
            images = multi_modal_data.get('images')
            videos = multi_modal_data.get('videos')
            audios = multi_modal_data.get('audios')

        expanded_prompt_ids = prompt_ids
        if images or videos or audios:
            expanded_prompt_ids = await self._encode_multi_modal_ids(
                full_text,
                images=images,
                videos=videos,
                audios=audios,
            )

        initial_prompt_len = len(expanded_prompt_ids)
        # self.token_budget is the main trajectory's compaction/RL budget. Auxiliary requests
        # are stateless and must only be limited by the model context and resolved request args.
        if self.max_model_len is not None:
            context_max_tokens = self.max_model_len - initial_prompt_len
            if context_max_tokens <= 0:
                raise ContextLengthExceededError(
                    f'HyTuner auxiliary context length exceeded: max_model_len={self.max_model_len}, '
                    f'initial_prompt_len={initial_prompt_len}, prompt_ids={len(prompt_ids)}',
                )
            existing_max_tokens = sampling_params.get('max_tokens')
            if existing_max_tokens is None:
                sampling_params['max_tokens'] = context_max_tokens
            else:
                sampling_params['max_tokens'] = min(context_max_tokens, existing_max_tokens)

        max_tokens_budget = sampling_params.get('max_tokens')
        if max_tokens_budget is not None and max_tokens_budget <= 0:
            raise ContextLengthExceededError(
                f'HyTuner auxiliary context length exceeded: max_tokens={max_tokens_budget}, '
                f'max_model_len={self.max_model_len}, initial_prompt_len={initial_prompt_len}, '
                f'prompt_ids={len(prompt_ids)}',
            )

        all_token_ids: list[int] = []
        all_log_probs: list[float] = []
        all_resp_topk_logprob_indice: list[int] = []
        all_resp_topk_logprob_value: list[int] = []
        curr_prompt_ids = list(prompt_ids)
        token_output = None
        had_abort = False
        completion_inference_time = 0.0
        completion_abort_wait_time = 0.0

        while True:
            if self.abort_event is not None:
                completion_abort_wait_time += await self.wait_for_abort_resume(self.abort_event, self.request_id)

            curr_sampling_params = copy.deepcopy(sampling_params)
            if max_tokens_budget is not None:
                max_tokens_this_turn = max_tokens_budget - len(all_token_ids)
                if max_tokens_this_turn <= 0:
                    break
                curr_sampling_params['max_tokens'] = max_tokens_this_turn

            t_gen_start = time.perf_counter()
            try:
                token_output = await self._cached_generate(
                    prompt_ids=curr_prompt_ids,
                    sampling_params=curr_sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    priority=self.priority,
                )
            except Exception as error:
                if not _is_expected_rpc_abort(error, self.abort_event):
                    raise
                had_abort = True
                logger.info(
                    'HyTuner auxiliary RPC cancelled at a partial-rollout boundary; '
                    'waiting for resume before retrying, request_id=%s',
                    self.request_id,
                )
                continue

            wall_inference_time = time.perf_counter() - t_gen_start
            completion_inference_time += _token_output_inference_time(token_output, wall_inference_time)

            if token_output.token_ids:
                all_token_ids.extend(token_output.token_ids)
                curr_prompt_ids.extend(token_output.token_ids)
                if token_output.log_probs:
                    all_log_probs.extend(token_output.log_probs)
                elif all_log_probs:
                    all_log_probs.extend([0.0] * len(token_output.token_ids))
                    print(
                        '[Warning] log_probs is None for this round, but previous rounds had log_probs. '
                        'Padding with 0.0 to keep alignment.'
                    )
                if (
                    hasattr(token_output, 'extra_fields')
                    and token_output.extra_fields.get('resp_topk_logprob_indice') is not None
                ):
                    all_resp_topk_logprob_indice.extend(token_output.extra_fields['resp_topk_logprob_indice'])
                    all_resp_topk_logprob_value.extend(token_output.extra_fields['resp_topk_logprob_value'])

            if token_output.stop_reason != 'aborted':
                break
            had_abort = True

        if had_abort:
            token_output.token_ids = all_token_ids
            token_output.log_probs = all_log_probs if all_log_probs else None
            if (
                hasattr(token_output, 'extra_fields')
                and token_output.extra_fields.get('resp_topk_logprob_indice') is not None
            ):
                token_output.extra_fields['resp_topk_logprob_indice'] = all_resp_topk_logprob_indice
                token_output.extra_fields['resp_topk_logprob_value'] = all_resp_topk_logprob_value

        assert token_output is not None
        assert self.tokenizer is not None

        if not token_output.token_ids:
            logger.warning(
                f'HyTuner auxiliary request for model={model}: empty token_ids, stop_reason={token_output.stop_reason}'
            )
            completion = self.get_empty_completion(model=model)
            completion.provider_specific_fields = {
                'inference_time': completion_inference_time,
                '_runner_timeout_credited': completion_abort_wait_time,
            }
            return completion

        raw_response = await asyncio.to_thread(
            self.tokenizer.decode,
            token_output.token_ids,
            skip_special_tokens=True,
        )
        raw_tool, others, tool_calls = await self._parse_tool_calls(token_output.token_ids, request=request)
        completion = self.get_empty_completion(model=model)

        reasoning_effort = (request.chat_template_kwargs or {}).get('reasoning_effort')
        # NOTE(yufeihuang): Unconfigured callers retain the mainline marker-based parsing contract.
        thinking = (
            reasoning_effort != 'no_think'
            if reasoning_effort is not None
            else self._think_end_token in others or others.lstrip().startswith(self._think_start_token)
        )
        if thinking and self._think_end_token in others:
            reasoning_content, content = others.split(self._think_end_token, 1)
            if reasoning_content.startswith(self._think_start_token):
                reasoning_content = reasoning_content[len(self._think_start_token) :]
            reasoning_content = reasoning_content or None
        elif thinking:
            reasoning_content = others.removeprefix(self._think_start_token) or None
            content = ''
        else:
            reasoning_content = None
            content = others

        if max_tokens_budget is not None and len(all_token_ids) >= max_tokens_budget:
            completion.choices[0].finish_reason = 'length'
        else:
            completion.choices[0].finish_reason = 'tool_calls' if tool_calls else 'stop'
        if not tool_calls:
            self._flag_unparsed_tool_call(completion.choices[0], raw_tool)
        message = ExtendedMessage(
            role='assistant',
            reasoning_content=reasoning_content,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            raw_response=raw_response,
            raw_tool=raw_tool,
        )
        completion.choices[0].message = message
        completion.usage = ExtendedUsage(
            prompt_tokens=len(expanded_prompt_ids),
            completion_tokens=len(token_output.token_ids),
            total_tokens=len(expanded_prompt_ids) + len(token_output.token_ids),
            prompt_tokens_details=ExtendedPromptTokensDetails(cached_tokens=0),
        )
        completion.provider_specific_fields = {
            **(completion.provider_specific_fields or {}),
            'inference_time': completion_inference_time,
            '_runner_timeout_credited': completion_abort_wait_time,
        }

        if allow_retry_for_tool_call and not self.is_correct_for_tool_call(message):
            return completion
        return completion

    async def _create_completion(self, messages, model=None, **kwargs) -> ExtendedChatCompletion:
        """Create a chat completion using HyTuner's server manager.

        Args:
            messages: List of chat messages
            model: Model name (ignored, uses HyTuner's configured model)
            **kwargs: Additional arguments:
                - tools: Tool schemas for function calling
                - sampling_params: Override default sampling params
                - chat_template_kwargs: Override chat-template params for this call
                - count_toward_token_budget: Whether generated tokens update the
                  main-agent trajectory counters (default True)

        Returns:
            ExtendedChatCompletion with the model's response
        """
        request_args_override = kwargs.pop('_request_args_override', None)
        if self._is_auxiliary_model(model):
            return await self._auxiliary_completion(
                messages,
                model=model,
                request_args_override=request_args_override,
                **kwargs,
            )
        per_call_sampling_params = kwargs.get('sampling_params')
        if per_call_sampling_params is not None and not isinstance(per_call_sampling_params, dict):
            raise TypeError('sampling_params must be a dict or null')
        per_call_sampling_has_max_tokens = (
            isinstance(per_call_sampling_params, dict) and 'max_tokens' in per_call_sampling_params
        )
        merged_sampling_params = {
            **self.sampling_params,
            **(per_call_sampling_params or {}),
        }
        request_args = {**kwargs, 'model': model, 'messages': messages}
        if self.sampling_params or per_call_sampling_params is not None:
            request_args['sampling_params'] = merged_sampling_params
        effective_request_args = self._resolve_effective_request_args(request_args)
        messages = copy.deepcopy(effective_request_args.pop('messages'))
        effective_request_args.pop('model', None)
        kwargs = effective_request_args

        tools = kwargs.get('tools')
        call_chat_template_kwargs = kwargs.pop('chat_template_kwargs', None)
        if call_chat_template_kwargs is not None and not isinstance(call_chat_template_kwargs, dict):
            raise TypeError('chat_template_kwargs must be a dict or null')
        effective_chat_template_kwargs = {
            **self.apply_chat_template_kwargs,
            **(call_chat_template_kwargs or {}),
        }
        request: Any = ChatCompletionRequest(tools=tools, chat_template_kwargs=effective_chat_template_kwargs)
        self._normalize_tool_call_arguments(messages)

        allow_retry_for_tool_call = kwargs.get('allow_retry_for_tool_call', False)
        count_toward_token_budget = bool(kwargs.get('count_toward_token_budget', True))
        request_meta = kwargs.get('request_meta')
        invocation_kind = request_meta.get('invocation_kind') if isinstance(request_meta, dict) else None
        force_full_retokenization = invocation_kind in ('summary_inference', 'compaction_resume')

        # Prefix matching for incremental tokenization:
        # Compare incoming messages with cached last_messages to find the
        # common prefix, then only tokenize the new suffix.
        prefix_len = self._find_common_prefix(messages, self.last_messages)
        last_message_count = len(self.last_messages)
        if force_full_retokenization:
            prefix_len = 0
        # Set when the prefix was recovered by rendering, so the render below is not repeated.
        recovered_prefix_text: str | None = None
        if last_message_count > 0 and prefix_len < last_message_count and not force_full_retokenization:
            # Field comparison is stricter than tokenization needs. Before giving up the token
            # cache, check whether the history still renders to the same text.
            canonical_messages = await self._canonicalize_render_equivalent_prefix(
                messages,
                tools,
                last_message_count,
                effective_chat_template_kwargs,
            )
            if canonical_messages is not None:
                messages = canonical_messages
                prefix_len = last_message_count
                recovered_prefix_text = self._last_rendered_text
        # Equal-length continuations also reuse sampled history; only a new generation
        # prompt (if any) needs encoding. Message count alone does not invalidate a prefix.
        is_append = last_message_count > 0 and prefix_len == last_message_count
        if self.strict_prefix_check and last_message_count > 0:
            if prefix_len != last_message_count:
                raise RuntimeError(
                    'HyTunerClient strict_prefix_check failed: incoming messages changed previously cached history '
                    f'(matched_prefix_messages={prefix_len}, cached_messages={last_message_count}, '
                    f'incoming_messages={len(messages)}). Disable strict_prefix_check or disable compaction/history rewriting.'
                )
            is_append = True

        if is_append:
            # Normal multi-turn append or equal-length replay: all old messages match.
            first_part_text = recovered_prefix_text
            if first_part_text is None:
                first_part_text = await asyncio.to_thread(
                    self.apply_chat_template,
                    messages[:prefix_len],
                    tools,
                    add_generation_prompt=False,
                    is_first_msg=False,
                    chat_template_kwargs=effective_chat_template_kwargs,
                )
            if first_part_text != self._last_rendered_text:
                raise RuntimeError('HyTunerClient rendering inputs changed mid-trajectory; cached tokens invalid')
            cached_tokens = self.last_tokens
            cached_expanded_tokens = self.last_expanded_tokens
            cached_num_images = self._last_num_images
            cached_num_videos = self._last_num_videos
            cached_num_audios = self._last_num_audios
        else:
            # First call, messages fully changed, or compaction — full tokenize
            first_part_text = ''
            cached_tokens = []
            cached_expanded_tokens = []
            cached_num_images = 0
            cached_num_videos = 0
            cached_num_audios = 0

        full_text = await asyncio.to_thread(
            self.apply_chat_template,
            messages,
            tools,
            is_first_msg=not is_append,
            chat_template_kwargs=effective_chat_template_kwargs,
        )
        if not full_text.startswith(first_part_text):
            raise RuntimeError(
                'Failed to parse new part of messages for consistent text. '
                f'First part ({len(first_part_text)} chars) is not a prefix of full ({len(full_text)} chars).'
            )
        new_text = full_text[len(first_part_text) :]
        new_tokens = await self._encode_prompt_text(
            new_text,
            operation='request prompt',
            message_count=len(messages),
        )
        prompt_ids = cached_tokens + new_tokens

        # Detect prefix change → new segment boundary, reset budget baseline
        prefix_shortened = (
            last_message_count > 0 and self._initial_prompt_len is not None and prefix_len < last_message_count
        )
        previous_initial_prompt_len = self._initial_prompt_len
        # TODO(lux): if summarize compaction fails, may cause self._initial_prompt_len mismatch
        if prefix_shortened:
            self._initial_prompt_len = None  # Will be set below

        images, videos, audios = None, None, None
        if self.processor is not None and self.process_vision_info_fn is not None:
            multi_modal_data = await self.process_vision_info_fn(messages)
            images = multi_modal_data.get('images')
            videos = multi_modal_data.get('videos')
            audios = multi_modal_data.get('audios')
        # Compute expanded prompt ids for training data recording and token count.
        expanded_prompt_ids = prompt_ids
        new_expanded_tokens = new_tokens
        if images or videos or audios:
            # Slice out only the newly added images/videos/audios using saved counts from last turn.
            new_images = images[cached_num_images:] if images else None
            new_videos = videos[cached_num_videos:] if videos else None
            new_audios = audios[cached_num_audios:] if audios else None
            new_expanded_tokens = await self._encode_multi_modal_ids(
                new_text,
                images=new_images,
                videos=new_videos,
                audios=new_audios,
            )
            expanded_prompt_ids = cached_expanded_tokens + new_expanded_tokens

        self._account_tool_observation_suffix(
            messages,
            prefix_len,
            new_expanded_tokens,
            is_append=is_append,
        )

        if self._initial_prompt_len is None:
            self._initial_prompt_len = len(expanded_prompt_ids)
        sampling_params = {**kwargs.get('sampling_params', {})}
        sampling_params_enforced = 'sampling_params' in self._resolve_effective_request_args({})
        # If the trainer passes `max_model_len` (its max_seq_length), set `max_tokens`
        # to `max_model_len - len(prompt_ids)` to better utilize the available token budget.
        if self.max_model_len is not None and not sampling_params_enforced:
            budget_max_tokens = self.max_model_len - self._initial_prompt_len
            existing_max_tokens = sampling_params.get('max_tokens')
            if existing_max_tokens is None:
                sampling_params['max_tokens'] = budget_max_tokens
            else:
                sampling_params['max_tokens'] = min(budget_max_tokens, existing_max_tokens)
        curr_sampling_params = copy.deepcopy(sampling_params)
        should_compute_remaining = not sampling_params_enforced and not per_call_sampling_has_max_tokens

        # Compute the max_tokens limit for this completion call
        # This is the "budget" we have for generating tokens in this call
        max_tokens_budget: int | None = None
        if should_compute_remaining and 'max_tokens' in sampling_params:
            used_tokens = len(expanded_prompt_ids) - self._initial_prompt_len
            max_tokens = sampling_params.get('max_tokens')
            if max_tokens is not None:
                remaining_tokens = max_tokens - used_tokens
                if count_toward_token_budget and self.token_budget is not None:
                    remaining_tokens = min(remaining_tokens, self.token_budget - self.token_consumed)
                max_tokens_budget = remaining_tokens
        elif 'max_tokens' in sampling_params:
            max_tokens_budget = sampling_params['max_tokens']

        effective_response_limit = _resolve_effective_response_limit(
            sampling_params=sampling_params,
            max_tokens_budget=max_tokens_budget,
            max_model_len=self.max_model_len,
            initial_prompt_len=self._initial_prompt_len,
            sampling_params_enforced=sampling_params_enforced,
            per_call_sampling_has_max_tokens=per_call_sampling_has_max_tokens,
            request_meta=kwargs.get('request_meta'),
        )

        if last_message_count > 0 and (
            prefix_shortened or effective_response_limit != self._last_effective_response_limit
        ):
            request_meta = kwargs.get('request_meta')
            self._log_prefix_transition(
                messages,
                full_text,
                trigger='prefix_reset' if prefix_shortened else 'effective_limit_change',
                matched_prefix_messages=prefix_len,
                invocation_kind=request_meta.get('invocation_kind') if isinstance(request_meta, dict) else None,
                old_initial_prompt_len=previous_initial_prompt_len,
                new_initial_prompt_len=self._initial_prompt_len,
                old_effective_response_limit=self._last_effective_response_limit,
                new_effective_response_limit=effective_response_limit,
                max_model_len=self.max_model_len,
                sampling_max_tokens=sampling_params.get('max_tokens'),
                call_max_tokens=max_tokens_budget,
                sampling_params_enforced=sampling_params_enforced,
                per_call_max_tokens=per_call_sampling_has_max_tokens,
            )

        # Report prefix/budget changes even if this attempt is rejected before generation.
        if should_compute_remaining and max_tokens_budget is not None and max_tokens_budget <= 0:
            raise ContextLengthExceededError(
                f'HyTuner context length exceeded: max_tokens={sampling_params["max_tokens"]}, '
                f'used_tokens={len(expanded_prompt_ids) - self._initial_prompt_len}, '
                f'initial_prompt_len={self._initial_prompt_len}, prompt_ids={len(prompt_ids)}',
            )

        # TODO(petersheng): add priority to the request
        all_token_ids: list[int] = []
        all_log_probs: list[float] = []
        all_resp_topk_logprob_indice: list[int] = []
        all_resp_topk_logprob_value: list[int] = []
        curr_prompt_ids = list(prompt_ids)
        token_output = None
        had_abort = False
        completion_inference_time = 0.0
        completion_wall_inference_time = 0.0
        completion_abort_wait_time = 0.0
        # R3 offsets live in vLLM's image-EXPANDED space (from expanded_prompt_ids, never
        # curr_prompt_ids). Accumulate locally, commit only on success (np.concatenate makes a new
        # array); a non-append prompt is a new expanded stream/segment, so capture restarts at 0.
        routed_experts = self._routed_experts if is_append else None
        if routed_experts is not None and int(routed_experts.shape[0]) != len(cached_expanded_tokens) - 1:
            # Rows must cover exactly the cached expanded prefix minus its final (unprocessed)
            # token; anything else would silently misalign replay against the current stream.
            raise ValueError(
                f'R3 accumulated rows do not cover the cached expanded prefix: '
                f'rows={int(routed_experts.shape[0])}, expected={len(cached_expanded_tokens) - 1}, '
                f'request_id={self.request_id}, trajectory_id={getattr(self, "trajectory_id", None)}, '
                f'prefix_len={prefix_len}, last_message_count={last_message_count}, '
                f'incoming_message_count={len(messages)}, is_append={is_append}, '
                f'prefix_shortened={prefix_shortened}'
            )

        while True:
            if self.abort_event is not None:
                wait_time = await self.wait_for_abort_resume(self.abort_event, self.request_id)
                self.abort_wait_time += wait_time
                completion_abort_wait_time += wait_time

            # Compute remaining tokens for this turn from the original budget
            curr_sampling_params = copy.deepcopy(sampling_params)
            if max_tokens_budget is not None:
                max_tokens_this_turn = max_tokens_budget - len(all_token_ids)
                if max_tokens_this_turn <= 0:
                    break
                curr_sampling_params['max_tokens'] = max_tokens_this_turn
            if self.enable_rollout_routing_replay:
                curr_sampling_params['routed_experts_prompt_start'] = (
                    0 if routed_experts is None else int(routed_experts.shape[0])
                )
                r3_prompt_start = curr_sampling_params['routed_experts_prompt_start']
                # routed_experts_prompt_start indexes vLLM's image-EXPANDED positions, not the unexpanded
                # curr_prompt_ids. Use the expanded length to match vLLM's num_prompt_tokens assert on
                # multimodal turns; +all_token_ids counts this turn's generated tokens (1 position each).
                expanded_prompt_len = len(expanded_prompt_ids) + len(all_token_ids)
                if not (0 <= r3_prompt_start < expanded_prompt_len):
                    persisted_rows = None if self._routed_experts is None else int(self._routed_experts.shape[0])
                    trajectory_id = getattr(self, 'trajectory_id', None)
                    raise ValueError(
                        f'R3 routed_experts_prompt_start invalid before send: '
                        f'prompt_start={r3_prompt_start}, expanded_prompt_len={expanded_prompt_len}, '
                        f'prompt_len={len(curr_prompt_ids)}, request_id={self.request_id}, '
                        f'trajectory_id={trajectory_id}, previous_prompt_len={len(self.last_tokens)}, '
                        f'prefix_len={prefix_len}, last_message_count={last_message_count}, '
                        f'incoming_message_count={len(messages)}, is_append={is_append}, '
                        f'initial_prompt_len={self._initial_prompt_len}, '
                        f'prefix_shortened={prefix_shortened}, had_abort={had_abort}, '
                        f'persisted_rows={persisted_rows}'
                    )

            t_gen_start = time.perf_counter()
            try:
                token_output = await self._cached_generate(
                    prompt_ids=curr_prompt_ids,
                    sampling_params=curr_sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    priority=self.priority,
                )
            except Exception as error:
                if not _is_expected_rpc_abort(error, self.abort_event):
                    raise
                had_abort = True
                logger.info(
                    'HyTuner RPC cancelled at a partial-rollout boundary; '
                    'waiting for resume before retrying, request_id=%s',
                    self.request_id,
                )
                continue
            wall_inference_time = time.perf_counter() - t_gen_start
            completion_wall_inference_time += wall_inference_time
            completion_inference_time += _token_output_inference_time(token_output, wall_inference_time)

            if token_output.token_ids:
                all_token_ids.extend(token_output.token_ids)
                curr_prompt_ids.extend(token_output.token_ids)
                if token_output.log_probs:
                    all_log_probs.extend(token_output.log_probs)
                elif all_log_probs:
                    # Previous rounds had log_probs but this round doesn't — pad with 0.0 to keep alignment
                    all_log_probs.extend([0.0] * len(token_output.token_ids))
                    print(
                        '[Warning] log_probs is None for this round, but previous rounds had log_probs. \
                            Padding with 0.0 to keep alignment.'
                    )
                if (
                    hasattr(token_output, 'extra_fields')
                    and token_output.extra_fields.get('resp_topk_logprob_indice') is not None
                ):
                    all_resp_topk_logprob_indice.extend(token_output.extra_fields['resp_topk_logprob_indice'])
                    all_resp_topk_logprob_value.extend(token_output.extra_fields['resp_topk_logprob_value'])

                if self.enable_rollout_routing_replay:
                    # vLLM returns only this turn's new positions (requested via
                    # routed_experts_prompt_start = positions seen so far).
                    delta = getattr(token_output, 'routed_experts', None)
                    if delta is None:
                        raise ValueError(
                            'R3 routed_experts append protocol expected token_output.routed_experts; '
                            'got None (is routed experts capture enabled on the server?).'
                        )
                    delta = np.asarray(delta)

                    # Expanded space, so this also holds on multimodal turns: it is the only check
                    # that proves the client-side expansion still matches what vLLM processed,
                    # before a stale offset can trip the engine assert on the next turn.
                    r3_prompt_start = curr_sampling_params['routed_experts_prompt_start']
                    expected_rows = len(expanded_prompt_ids) + len(all_token_ids) - r3_prompt_start - 1
                    if delta.shape[0] != expected_rows:
                        raise ValueError(
                            f'R3 routed_experts delta length mismatch: got {delta.shape[0]} rows, '
                            f'expected {expected_rows}. request_id={self.request_id}, '
                            f'trajectory_id={getattr(self, "trajectory_id", None)}, '
                            f'prompt_start={r3_prompt_start}, '
                            f'expanded_prompt_len_incl_new_tokens={len(expanded_prompt_ids) + len(all_token_ids)}, '
                            f'prompt_len_incl_new_tokens={len(curr_prompt_ids)}, '
                            f'new_tokens={len(token_output.token_ids)}, stop_reason={token_output.stop_reason}, '
                            f'accumulated_rows={0 if routed_experts is None else int(routed_experts.shape[0])}, '
                            f'had_abort={had_abort}'
                        )
                    routed_experts = (
                        delta if routed_experts is None else np.concatenate((routed_experts, delta), axis=0)
                    )

            if token_output.stop_reason != 'aborted':
                break

            had_abort = True

        if had_abort:
            token_output.token_ids = all_token_ids
            token_output.log_probs = all_log_probs if all_log_probs else None
            if (
                hasattr(token_output, 'extra_fields')
                and token_output.extra_fields.get('resp_topk_logprob_indice') is not None
            ):
                token_output.extra_fields['resp_topk_logprob_indice'] = all_resp_topk_logprob_indice
                token_output.extra_fields['resp_topk_logprob_value'] = all_resp_topk_logprob_value

        assert token_output is not None
        assert self.tokenizer is not None
        self.inference_time += completion_wall_inference_time
        self.true_inference_time += completion_inference_time

        if self.enable_rollout_routing_replay:
            token_output.routed_experts = routed_experts

        # Handle empty token output
        if not token_output.token_ids:
            logger.warning(
                f'HyTuner request_id={self.request_id}: empty token_ids, stop_reason={token_output.stop_reason}'
            )
            completion = self.get_empty_completion(model=model)
            completion.provider_specific_fields = {
                'inference_time': completion_inference_time,
                '_runner_timeout_credited': completion_abort_wait_time,
            }
            return completion

        output_length_exceeded = token_output.stop_reason == 'length' or (
            max_tokens_budget is not None and len(all_token_ids) >= max_tokens_budget
        )
        raw_response = await asyncio.to_thread(
            self.tokenizer.decode,
            token_output.token_ids,
            skip_special_tokens=True,
        )
        try:
            raw_tool, others, tool_calls = await self._parse_tool_calls(token_output.token_ids, request=request)
        except Exception:
            if not output_length_exceeded:
                raise
            logger.warning('Tool-call parsing failed after output reached max_tokens; preserving raw partial response')
            raw_tool, others, tool_calls = '', raw_response, []

        completion = self.get_empty_completion(model=model)

        # Split others into reasoning_content and content by the think end token
        # so that chat templates (e.g. Hunyuan) can render them as separate fields.
        reasoning_effort = (request.chat_template_kwargs or {}).get('reasoning_effort')
        # NOTE(yufeihuang): Unconfigured callers retain the mainline marker-based parsing contract.
        thinking = (
            reasoning_effort != 'no_think'
            if reasoning_effort is not None
            else self._think_end_token in others or others.lstrip().startswith(self._think_start_token)
        )
        if thinking and self._think_end_token in others:
            reasoning_content, content = others.split(self._think_end_token, 1)
            # Strip leading think start tag if present (extract_tool_calls may prepend it)
            if reasoning_content.startswith(self._think_start_token):
                reasoning_content = reasoning_content[len(self._think_start_token) :]
            reasoning_content = reasoning_content or None
        elif thinking:
            reasoning_content = others.removeprefix(self._think_start_token) or None
            content = ''
        else:
            reasoning_content = None
            content = others

        # A generation that consumed its whole budget was cut by max_tokens, so the turn it
        # produced may well be incomplete. Report that as 'length' rather than letting it pass
        # for a natural stop: callers (mini_swe_agent, terminus2, swe_setup_agent) key off this
        # to tell "the model emitted something unparseable" apart from "we ran out of budget".
        completion.choices[0].finish_reason = (
            'length' if output_length_exceeded else ('tool_calls' if tool_calls else 'stop')
        )
        if not tool_calls:
            self._flag_unparsed_tool_call(completion.choices[0], raw_tool)
        message = ExtendedMessage(
            role='assistant',
            reasoning_content=reasoning_content,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            raw_response=raw_response,  # for cases like agts which need raw response for reward
            raw_tool=raw_tool,
        )
        completion.choices[0].message = message
        has_reasoning_boundary = thinking and self._think_end_token in others
        has_unclosed_reasoning = thinking and not has_reasoning_boundary
        think_end_ids = self.tokenizer.encode(self._think_end_token, add_special_tokens=False)
        reasoning_tokens, content_tokens = _split_completion_token_usage(
            token_output.token_ids,
            think_end_ids,
            has_reasoning_boundary=has_reasoning_boundary,
            has_unclosed_reasoning=has_unclosed_reasoning,
        )
        if reasoning_tokens is None:
            logger.warning(
                'HyTuner could not find the reasoning end-token sequence in generated token ids; '
                'completion token details will be omitted.'
            )
        completion.usage = ExtendedUsage(
            prompt_tokens=len(expanded_prompt_ids),
            completion_tokens=len(token_output.token_ids),
            total_tokens=len(expanded_prompt_ids) + len(token_output.token_ids),
            completion_tokens_details=ExtendedCompletionTokensDetails(
                reasoning_tokens=reasoning_tokens,
                text_tokens=content_tokens,
            ),
            prompt_tokens_details=ExtendedPromptTokensDetails(
                cached_tokens=len(cached_expanded_tokens),
            ),
        )
        completion.provider_specific_fields = {
            **(completion.provider_specific_fields or {}),
            'inference_time': completion_inference_time,
            '_runner_timeout_credited': completion_abort_wait_time,
        }

        if allow_retry_for_tool_call and not self.is_correct_for_tool_call(message):
            return completion

        self.last_messages = messages + [message.model_dump(exclude_none=True)]
        self._last_effective_response_limit = effective_response_limit
        self._normalize_tool_call_arguments(self.last_messages[-1:])
        self.last_tokens = prompt_ids + token_output.token_ids
        self.last_expanded_tokens = expanded_prompt_ids + token_output.token_ids
        self._last_rendered_text = await asyncio.to_thread(
            self.apply_chat_template,
            self.last_messages,
            tools,
            add_generation_prompt=False,
            is_first_msg=False,
            chat_template_kwargs=effective_chat_template_kwargs,
        )
        self._last_num_images = len(images) if images else 0
        self._last_num_videos = len(videos) if videos else 0
        self._last_num_audios = len(audios) if audios else 0
        if count_toward_token_budget:
            request_meta = kwargs.get('request_meta')
            invocation_kind = request_meta.get('invocation_kind') if isinstance(request_meta, dict) else None
            output_component = (
                TokenBudgetComponent.COMPACTION_OUTPUT
                if invocation_kind == 'summary_inference'
                else TokenBudgetComponent.AGENT_OUTPUT
            )
            if output_component in self.token_budget_components:
                self.token_consumed += len(token_output.token_ids)
            self.curr_segment_consumed_tokens += len(token_output.token_ids)

        # Build token-level training data for span update
        extra_fields = None
        if (
            hasattr(token_output, 'extra_fields')
            and token_output.extra_fields.get('resp_topk_logprob_indice') is not None
        ):
            extra_fields = {
                'resp_topk_logprob_indice': token_output.extra_fields['resp_topk_logprob_indice'],
                'resp_topk_logprob_value': token_output.extra_fields['resp_topk_logprob_value'],
            }
        training_data = {
            'request_id': self.request_id,
            'completion_span_id': get_current_span_id(),
            'prompt_ids': expanded_prompt_ids,
            'raw_prompt_ids': prompt_ids,
            'response_ids': token_output.token_ids,
            'response_logprobs': token_output.log_probs,
            'routed_experts': getattr(token_output, 'routed_experts', None),
            'messages': self.last_messages,
            'abort_wait_time': self.abort_wait_time,
            'llm_inference_time': self.inference_time,
            'true_llm_inference_time': self.true_inference_time,
            'effective_response_limit': effective_response_limit,
            # prefix_shortened: harness rewrote / compaction shortened the cached prefix this turn.
            # Content-aware, catches plateau/equal-length rewrites effective_response_limit misses.
            'prefix_shortened': prefix_shortened,
            'extra_fields': extra_fields,
        }

        # Update current span with token-level training data
        span_id = get_current_span_id()
        if span_id:
            span_attrs: dict[str, Any] = {
                '_hytuner_training_data': training_data,
            }
            update_span(span_id, attributes=span_attrs)
        observer = self._training_data_observer
        if observer is not None:
            observer.record_training_data(training_data)

        if self.enable_rollout_routing_replay:
            self._routed_experts = routed_experts
        return completion

    @cacheable(tags=['llm', 'generate'], ignore_args=['self'])
    async def _cached_generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        priority: int = 0,
    ) -> Any:
        """Cached wrapper for server_manager.generate.

        This method is cached with nested=True, so even if the outer
        create_completion fails after generate succeeds, the generate
        result will be cached and reused on retry.
        """
        server_manager = cast(Any, self.server_manager)
        if server_manager is None:
            raise RuntimeError('HyTuner client requires client_args.server_manager')

        generate = getattr(server_manager, 'generate', None)
        if generate is None:
            raise RuntimeError('HyTuner server_manager does not expose generate')

        generate_kwargs: dict[str, Any] = {
            'request_id': self.request_id,
            'prompt_ids': prompt_ids,
            'sampling_params': sampling_params,
            'image_data': image_data,
            'priority': priority,
        }
        if audio_data is not None:
            generate_kwargs['audio_data'] = audio_data
        if video_data is not None:
            generate_kwargs['video_data'] = video_data

        started = time.perf_counter()
        token_output = await generate(**generate_kwargs)
        inference_time = _token_output_inference_time(token_output, time.perf_counter() - started)
        extra_fields = dict(getattr(token_output, 'extra_fields', None) or {})
        extra_fields['_inference_time'] = inference_time
        token_output.extra_fields = extra_fields
        return token_output

    async def _count_tokens(
        self,
        *,
        messages,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> TokenCountResult:
        """Count tokens for ``messages`` + ``tools``, reusing this client's cache to speed up
        tokenization (especially for multimodal inputs). This is read-only: the cache is never
        (and must not be) mutated here.
        """
        if not add_generation_prompt:
            raise NotImplementedError('HyTunerClient._count_tokens requires add_generation_prompt=True')

        messages = self._messages_with_normalized_tool_calls(messages)

        # Read-only prefix lookup: only valid when the entire cached
        # ``last_messages`` matches the incoming prefix
        prefix_len = self._find_common_prefix(messages, self.last_messages)
        last_message_count = len(self.last_messages)
        can_reuse_cache = (
            last_message_count > 0 and prefix_len == last_message_count and self._last_rendered_text is not None
        )

        if can_reuse_cache:
            first_part_text = await asyncio.to_thread(
                self.apply_chat_template,
                messages[:prefix_len],
                tools,
                add_generation_prompt=False,
                is_first_msg=False,
            )
            if first_part_text != self._last_rendered_text:
                can_reuse_cache = False

        if can_reuse_cache:
            cached_tokens = self.last_tokens
            cached_expanded_tokens = self.last_expanded_tokens
        else:
            first_part_text = ''
            cached_tokens = []
            cached_expanded_tokens = []
            prefix_len = 0

        full_text = await asyncio.to_thread(
            self.apply_chat_template,
            messages,
            tools,
            add_generation_prompt=True,
            is_first_msg=not can_reuse_cache,
        )
        if not full_text.startswith(first_part_text):
            raise RuntimeError(
                'HyTunerClient.count_tokens: cached prefix render is not a prefix of full render. '
                f'first_part ({len(first_part_text)} chars) != prefix of full ({len(full_text)} chars).'
            )

        new_text = full_text[len(first_part_text) :]

        # Text-only client: encode the new suffix and add to cached prompt ids.
        if self.processor is None or self.process_vision_info_fn is None:
            new_tokens = await self._encode_prompt_text(
                new_text,
                operation='token-count prompt',
                message_count=len(messages),
            )
            return TokenCountResult(input_tokens=len(cached_tokens) + len(new_tokens))

        suffix = messages[prefix_len:]
        if not suffix:
            new_images, new_videos, new_audios = None, None, None
        else:
            multi_modal_data = await self.process_vision_info_fn(suffix)
            new_images = multi_modal_data.get('images')
            new_videos = multi_modal_data.get('videos')
            new_audios = multi_modal_data.get('audios')

        if not (new_images or new_videos or new_audios):
            new_tokens = await self._encode_prompt_text(
                new_text,
                operation='token-count prompt',
                message_count=len(messages),
            )
            return TokenCountResult(input_tokens=len(cached_expanded_tokens) + len(new_tokens))

        if new_videos:
            raise RuntimeError('Video input is not yet supported by HyTunerClient.count_tokens')

        new_expanded_tokens = await self._encode_multi_modal_ids(
            new_text,
            images=new_images,
            videos=new_videos,
            audios=new_audios,
        )
        return TokenCountResult(input_tokens=len(cached_expanded_tokens) + len(new_expanded_tokens))

    async def _encode_prompt_text(self, text: str, *, operation: str, message_count: int) -> list[int]:
        """Encode a rendered prompt while containing Rust tokenizer panics."""
        try:
            return await asyncio.to_thread(
                self.tokenizer.encode,  # type: ignore[possibly-missing-attribute]
                text,
                add_special_tokens=False,
            )
        except BaseException as error:
            tokenizer_error = as_tokenizer_panic_error(error, operation=operation)
            if tokenizer_error is None:
                raise
            logger.exception(
                'HyTuner prompt tokenization failed '
                '(operation=%s, request_id=%s, message_count=%d, content_chars=%d, rendered_text=%r)',
                operation,
                self.request_id,
                message_count,
                len(text),
                text,
            )
            raise tokenizer_error from error

    async def _parse_tool_calls(
        self,
        token_ids: list[int],
        request: ChatCompletionRequest | None = None,
    ) -> tuple[str, str, list[Any]]:
        # Parse tool calls and get raw output, content, and tool calls
        tool_parser = cast(Any, self.tool_parser)
        if tool_parser is None:
            raise RuntimeError('HyTuner client requires client_args.tool_parser')
        extract_tool_calls = getattr(tool_parser, 'extract_tool_calls', None)
        if extract_tool_calls is None:
            raise RuntimeError('HyTuner tool_parser does not expose extract_tool_calls')
        raw_output, others, tool_calls = await extract_tool_calls(token_ids, request=request)
        # The id must be non-empty and unique for the whole trajectory: consumers key tool
        # results by it, so a blank id drops every observation and a per-turn counter would
        # make turn N's result overwrite turn N-1's. Chat templates never render it, so the
        # value is free to choose and does not affect tokenization.
        return (
            raw_output,
            others,
            [
                {'id': self._tool_call_ids.next_id(), 'function': tool_call.model_dump(), 'type': 'function'}
                for tool_call in tool_calls
            ],
        )

    def _flag_unparsed_tool_call(self, choice: ExtendedChoice, raw_tool: str) -> None:
        """Mark a choice whose completion began a tool call the parser could not extract.

        Mirrors the gateway processor: the parser's tool-call start marker in its raw output
        (special tokens kept), with no tool calls extracted, is the evidence. A parser that does
        not name its marker leaves this check off.
        """
        marker = getattr(self.tool_parser, 'tool_calls_start_token', None)
        if isinstance(marker, str) and marker and marker in raw_tool:
            choice.provider_specific_fields = {
                **(choice.provider_specific_fields or {}),
                UNPARSED_TOOL_CALL_FIELD: True,
            }

    def apply_chat_template(
        self,
        messages,
        tools,
        add_generation_prompt=True,
        is_first_msg=None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        effective_chat_template_kwargs = (
            self.apply_chat_template_kwargs if chat_template_kwargs is None else chat_template_kwargs
        )
        if self.template_parser:
            # Use custom template parser if available
            if is_first_msg is None:
                is_first_msg = len(self.last_messages) == 0
            return self.template_parser.parse(
                messages,
                add_generation_prompt=add_generation_prompt,
                is_first_msg=is_first_msg,
                tools=tools,
                **effective_chat_template_kwargs,
            )

        return self.tokenizer.apply_chat_template(  # type: ignore[possibly-missing-attribute]
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            is_training=True,
            **effective_chat_template_kwargs,
        )

    def get_empty_completion(self, model: str | None = None):
        return ExtendedChatCompletion(
            id=f'chatcmpl-{uuid.uuid4()}',
            model=model or 'hytuner',
            object='chat.completion',
            created=int(time.time()),
            choices=[
                ExtendedChoice(
                    index=0,
                    message=ExtendedMessage(
                        role='assistant',
                        content='',
                        tool_calls=[],
                        raw_response='',
                        raw_tool='',
                    ),
                    finish_reason='stop',
                    logprobs=None,
                )
            ],
            usage=None,
        )

    def is_correct_for_tool_call(self, message: ExtendedMessage) -> bool:
        """Check the format for tool call arguments."""
        if not message.tool_calls:
            return False
        for tool_call in message.tool_calls:
            try:
                json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                return False
        return True
