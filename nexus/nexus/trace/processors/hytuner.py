# /// script
# dependencies = ["bashlex>=0.18"]
#
# [project]
# name = "hytuner"
# authors = [
#     { name = "luxyhliu" },
#     { name = "gussfang" },
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# span_processors = { hytuner = "nexus.trace.processors.hytuner" }
# ///

from __future__ import annotations

import asyncio
import fcntl
import functools
import json
import logging
import os
import socket
import time
import uuid

from pathlib import Path
from typing import Any

from nexus.trace.events import SpanEndEvent, SpanStartEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.processors.hytuner_segment_shaping import (
    SegmentShapingClassifier,
    normalize_segment_shaping_config,
)
from nexus.trace.types import Span
from nexus.utils.tool_analyzer import (
    build_turn_stats_from_messages,
    inject_tool_call_into_extra,
    summarize_tool_call_stats,
)


logger = logging.getLogger(__name__)

# Span types that contribute to the agent_timeline.
_TIMELINE_TYPES = {'task_phase', 'agent_inference', 'agent_observation', 'agent_compaction'}
_EFFECTIVE_RESPONSE_LIMIT_FIELD = 'effective_response_limit'


def _outbound_address() -> str:
    """Address the kernel would route from, or '' if there is none.

    Asks the routing table without sending a packet, and never consults /etc/hosts.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('8.8.8.8', 80))
            return probe.getsockname()[0]
    except OSError:
        return ''


@functools.lru_cache(maxsize=1)
def _node_dump_stem() -> str:
    """Filename stem identifying this node, stable for the lifetime of the process.

    Two nodes sharing one dump file interleave partial lines, because flock only
    serializes writers on a single machine. Resolving the hostname cannot tell nodes
    apart: containers routinely map it onto the loopback line of /etc/hosts, which
    silently collapses the whole cluster onto one file.
    """
    # LOCAL_IP is injected per node by Taiji; NODE_IP is its equivalent elsewhere.
    for address in (os.environ.get('LOCAL_IP'), os.environ.get('NODE_IP'), _outbound_address()):
        if address and not address.startswith('127.'):
            return address.replace('.', '-')
    # Uniqueness beats a tidy name here: two writers sharing a file corrupt each other.
    logger.warning('HyTuner: no routable node address found, falling back to a per-process dump file')
    return f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}'.replace('.', '-')


def _read_effective_response_limit(training_data: dict) -> tuple[bool, int | None]:
    if _EFFECTIVE_RESPONSE_LIMIT_FIELD not in training_data:
        return False, None
    value = training_data[_EFFECTIVE_RESPONSE_LIMIT_FIELD]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError(f'{_EFFECTIVE_RESPONSE_LIMIT_FIELD} must be an integer or null, got {value!r}')
    return True, value


# Per-tool failure detection.
#
# Default judgement falls back to a loose substring check ('Error' / 'error' in content)
# which catches Python tracebacks, generic HTTP errors, etc. — but it produces many
# false positives for tools whose successful output naturally contains the word
# "error" (e.g. visit's LLM-extracted webpage summaries discussing "trial and error",
# "standard error", "terror" as a substring, etc.).
#
# For tools listed below we use an explicit whitelist of failure markers instead.

# visit tool — see nexus/extensions/tools/web_search/visit.py
# Substring match (NOT startswith): single-url failure path returns
#   f'Read page for {url}. {VISIT_ERROR_TAG} ...'  (visit.py:400)
# so the marker can appear in the middle. Multi-url joined results are split by
#   '\n=======\n'  (visit.py:441)
# so a marker for any failed url also lives mid-string.
_VISIT_FAILURE_TAGS = (
    '[visit error]',  # jina readpage 4xx/5xx + huggingface block (visit.py:39)
    '[visit llm error]',  # summary LLM full fallback (visit.py:40)
    '[visit unknown response format]',  # summary LLM bad payload (visit.py:41)
    '[visit] Failed',  # legacy generic failure marker, kept for back-compat (visit.py:43)
)

# visit failure stage classification — see _classify_visit_failure below.
_VISIT_SUMMARY_LLM_TAGS = ('[visit llm error]', '[visit unknown response format]')
_VISIT_READPAGE_TAGS = ('[visit error]', '[visit] Failed')


def _is_tool_failure(name: str, content: str) -> bool:
    """Decide whether a tool message indicates failure.

    For the visit tool, only the explicit failure markers count — a successful
    visit returns LLM-extracted webpage text that frequently contains the word
    "error" without indicating a tool failure. For other tools we keep the
    loose 'Error/error' substring check; tighten per-tool here as needed.
    """
    if name == 'visit':
        return any(tag in content for tag in _VISIT_FAILURE_TAGS)
    return ('Error' in content) or ('error' in content)


def _classify_visit_failure(content: str) -> str:
    """Bucket a failed visit tool message into 'readpage' or 'summary_llm'.

    Order matters: summary_llm tags are checked first because the LLM-fallback
    content (visit.py:_build_jina_fallback_content / call_llm_server return path)
    carries [visit llm error] / [visit unknown response format] which uniquely
    identify the summary stage. [visit error] is reserved for jina readpage
    failures (4xx/5xx + huggingface block + entry-level read errors).
    """
    if any(tag in content for tag in _VISIT_SUMMARY_LLM_TAGS):
        return 'summary_llm'
    if any(tag in content for tag in _VISIT_READPAGE_TAGS):
        return 'readpage'
    # Should not reach here when _is_tool_failure returned True; bucket defensively.
    return 'readpage'


@span_processor_registry.register('hytuner')
class HytunerProcessor(SpanProcessor):
    """Processor that collects training trajectory data for HyTuner.

    Monitors openai_completion spans and accumulates:
    - prompt_ids: Initial prompt tokens (from first LLM call)
    - response_ids: All response tokens (LLM generations + tool responses)
    - response_mask: 1 for LLM tokens, 0 for tool/environment tokens
    - response_logprobs: Log probabilities for each token

    The output format is compatible with HyTuner's AgentLoopOutput.

    Context Management / Segments:
    When context_management is enabled, a long trajectory may be split into
    multiple segments at context-overflow boundaries.  Each segment is a
    self-contained (prompt_ids, response_ids, response_mask, response_logprobs)
    tuple.  Prior segments are stored in ``self.segments`` and the *current*
    (latest) segment lives in the usual instance attributes.

    Deferred Flush Pattern:
    Instead of immediately processing openai_completion spans, the processor
    buffers the training data and waits for the parent span to end:
    - agent_inference → flush buffer as normal conversation turn
    - agent_compaction (compacted=True) → finalize current segment, then:
      - if buffer contains an applied self-summary, record it as a standalone segment
      - reset for next segment
    This naturally separates summary inference traces from conversation turns
    without any explicit notification from the client or agent.
    """

    def __init__(
        self,
        file_path: str,
        trajectory_id: str,
        progress_state: dict | None = None,
        segment_shaping_config: dict[str, Any] | None = None,
    ):
        """Initialize the processor."""
        # Accumulated trajectory data
        self.file_path = file_path
        self.trajectory_id = trajectory_id
        self._progress_state = progress_state
        self.prompt_ids: list[int] = []
        self.raw_input_ids: list[int] = []  # Trajectory ids without multimodal placeholder expansion
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.routed_experts: Any = None  # Accumulated routed_experts for R3 support (np.ndarray)
        self.messages: list[dict] = []
        self.num_turns: int = 0
        self._first_prompt_captured = False
        self.effective_response_limit: int | None = None
        self._effective_response_limit_captured = False
        # Track time spent waiting for abort event to resume
        self.abort_wait_time: float = 0.0
        self.termination_reason: str | None = None
        # Set once the client reports a mid-segment prefix_shortened (harness rewrote the cached prefix); the
        # whole trajectory is then masked (UNHANDLED_ERROR) and later turns / span reasons no longer overwrite it.
        self._mask_unhandled_prefix_reset = False
        # Set only when a task reported one termination reason in place of another.
        self.cleared_termination_reason: str | None = None
        self.llm_inference_time: float = 0.0
        self.true_llm_inference_time: float = 0.0
        self.observation_time: float = 0.0
        self.reward_time: float = 0.0
        self.sandbox_init_time: float = 0.0
        self.init_compose_time: float = 0.0
        self.init_prepare_directories_time: float = 0.0
        self.init_boot_time: float = 0.0
        self.handoff_artifact_collect_time: float = 0.0
        self.handoff_release_main_time: float = 0.0
        self.handoff_sandbox_init_time: float = 0.0
        self.handoff_env_setup_time: float = 0.0
        self.scoring_check_score_time: float = 0.0
        self.runtime_id = None
        self.runtime_ids: list[str] = []
        self.eval_runtime_id = None
        self.request_id = None
        self.resp_topk_logprob_indice: list[list[int]] = []
        self.resp_topk_logprob_value: list[list[float]] = []
        self.agent_timeline: list[dict] = []
        # Per-turn tool call statistics (parse/exec totals + per-command
        # category records). Populated incrementally in ``_append_turn``.
        self.tool_call_stats: list[dict] = []
        self._tool_stats_cache_key: tuple[int, int] | None = None

        # Context management: prior segments (each is a dict with prompt_ids, response_ids, etc.)
        self.segments: list[dict] = []

        # Task identity: recorded from the task entity span (type='task')
        self.task_name: str | None = None

        # Optional submission patch for trajectory dumps.
        self.patch_text: str | None = None

        # Deferred flush: buffer the latest openai_completion training_data
        # until the parent span (agent_inference or agent_compaction) ends.
        self._buffered_completion: dict | None = None
        # Nested Runtime Agents can deliver token data directly before their
        # spans are replayed on the Host. Track span ids so the direct path and
        # the normal span callback remain idempotent.
        self._recorded_completion_span_ids: set[str] = set()

        # --- Segment reward shaping producer state ---
        # Emits raw per-turn shaping values for HyTuner's trainer-side consumer.
        # Disabled by default, so baseline runs do not receive extra fields.
        shaping_config = normalize_segment_shaping_config(segment_shaping_config)
        self._segment_shaping = SegmentShapingClassifier(shaping_config)
        self._shaping_enabled: bool = self._segment_shaping.enabled
        self.per_turn_shaping_rewards: list[float] = []
        self.per_turn_response_end_indices: list[int] = []
        self._pending_boundary: int | None = None
        self.shaping_trailing_tool_call_turns: int = 0
        self.shaping_multi_segment_skipped: bool = False

    def _finalize_current_segment(self) -> None:
        """Save the current accumulated trajectory as a segment and reset state.

        Called when a segment boundary is detected (context management truncation).
        Only creates a segment if there is actual data accumulated.
        """
        if not self._first_prompt_captured or not self.response_ids:
            # Nothing meaningful to save
            return

        self._finalize_pending_shaping_turn()
        shaping_extra_fields = self._build_current_segment_shaping_extra_fields()
        segment = {
            'prompt_ids': list(self.prompt_ids),
            'raw_input_ids': list(self.raw_input_ids),
            'response_ids': list(self.response_ids),
            'response_mask': list(self.response_mask),
            'response_logprobs': list(self.response_logprobs) if self.response_logprobs else None,
            'num_turns': self.num_turns,
            'segment_type': 'conversation',
            'messages': list(self.messages),
        }
        if self.routed_experts is not None:
            segment['routed_experts'] = self.routed_experts
        segment_extra_fields = dict(shaping_extra_fields or {})
        if self._effective_response_limit_captured:
            segment_extra_fields[_EFFECTIVE_RESPONSE_LIMIT_FIELD] = self.effective_response_limit
        if segment_extra_fields:
            segment['extra_fields'] = segment_extra_fields
        self.segments.append(segment)

        # Reset accumulation state for the next segment
        self.prompt_ids = []
        self.raw_input_ids = []
        self.response_ids = []
        self.response_mask = []
        self.response_logprobs = []
        self.routed_experts = None
        self.num_turns = 0
        self._first_prompt_captured = False
        self.effective_response_limit = None
        self._effective_response_limit_captured = False
        self.messages = []
        # New segment starts with fresh messages; invalidate the tool-stats
        # cache so the next reporting call recomputes from the empty list.
        self._tool_stats_cache_key = None
        self.tool_call_stats = []
        self._reset_current_segment_shaping_state()

    def _record_standalone_segment(self, training_data: dict) -> None:
        """Record a single completion as an independent segment.

        Unlike _append_turn which incrementally builds multi-turn segments
        (interleaving tool messages at mask=0 and LLM responses at mask=1),
        a standalone segment is one-shot: one prompt, one response, all mask=1.
        """
        prompt_ids = list(training_data.get('prompt_ids') or [])
        raw_prompt_ids = list(training_data.get('raw_prompt_ids') or [])
        response_ids = list(training_data.get('response_ids') or [])
        if not prompt_ids or not response_ids:
            return

        response_logprobs = training_data.get('response_logprobs')
        if response_logprobs is None:
            response_logprobs = [0.0] * len(response_ids)
        else:
            response_logprobs = list(response_logprobs)
            # Align length
            if len(response_logprobs) < len(response_ids):
                response_logprobs.extend([0.0] * (len(response_ids) - len(response_logprobs)))
            elif len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]

        segment = {
            'prompt_ids': prompt_ids,
            'raw_input_ids': raw_prompt_ids + response_ids,
            'response_ids': response_ids,
            'response_mask': [1] * len(response_ids),
            'response_logprobs': response_logprobs,
            'num_turns': 1,
            'segment_type': 'summary_inference',
            'messages': list(training_data.get('messages') or []),
        }
        if training_data.get('routed_experts') is not None:
            segment['routed_experts'] = training_data['routed_experts']
        segment_extra_fields = self._empty_segment_shaping_extra_fields() if self._shaping_enabled else {}
        has_effective_response_limit, effective_response_limit = _read_effective_response_limit(training_data)
        if has_effective_response_limit:
            segment_extra_fields[_EFFECTIVE_RESPONSE_LIMIT_FIELD] = effective_response_limit
        if segment_extra_fields:
            segment['extra_fields'] = segment_extra_fields
        self.segments.append(segment)

    def _flush_buffered_as_turn(self) -> None:
        """Flush the buffered completion as a normal conversation turn."""
        if self._buffered_completion is not None:
            self._append_turn(self._buffered_completion)
            self._buffered_completion = None

    def record_training_data(self, training_data: dict[str, Any]) -> None:
        """Buffer authoritative Host-side token data without requiring span replay."""
        completion_span_id = training_data.get('completion_span_id')
        if isinstance(completion_span_id, str) and completion_span_id:
            if completion_span_id in self._recorded_completion_span_ids:
                return
            self._recorded_completion_span_ids.add(completion_span_id)

        # Preserve the existing deferred-flush semantics. Parent
        # agent_inference/agent_compaction spans still decide how the current
        # buffer is classified; a following completion flushes the prior turn.
        if self._buffered_completion is not None:
            self._flush_buffered_as_turn()
        self._buffered_completion = training_data

    # --- Segment reward shaping helpers (producer side) ----------------------
    @staticmethod
    def _build_tool_call_id_to_name(messages_window: list[dict]) -> dict[str, str]:
        """Build tool_call_id -> tool_name from assistant messages."""
        mapping: dict[str, str] = {}
        for msg in messages_window:
            if not isinstance(msg, dict) or msg.get('role') != 'assistant':
                continue
            for tc in msg.get('tool_calls') or []:
                tc_id = tc.get('id', '')
                if not tc_id:
                    continue
                func = tc.get('function', {})
                mapping[tc_id] = func.get('name', 'unknown')
        return mapping

    @staticmethod
    def _empty_segment_shaping_extra_fields() -> dict:
        return {
            'per_turn_shaping_rewards': [],
            'per_turn_response_end_indices': [],
            'shaping_trailing_tool_call_turns': 0,
            'shaping_multi_segment_skipped': False,
        }

    def _finalize_pending_shaping_turn(self) -> None:
        """Close a pending assistant turn that never received a later tool-result flush.

        At a compaction boundary, the processor does not have a stable next
        completion where tool messages can be attributed to the previous
        assistant turn. Treat that boundary as a trailing turn with zero shaping
        rather than inventing a tool-error penalty.
        """
        if not self._shaping_enabled or self._pending_boundary is None:
            return

        if self.messages and isinstance(self.messages[-1], dict):
            last = self.messages[-1]
            if last.get('role') == 'assistant' and last.get('tool_calls'):
                self.shaping_trailing_tool_call_turns += 1
        self.per_turn_shaping_rewards.append(0.0)
        self.per_turn_response_end_indices.append(self._pending_boundary)
        self._pending_boundary = None
        self._segment_shaping.reset_state()

    def _build_current_segment_shaping_extra_fields(self) -> dict | None:
        if not self._shaping_enabled:
            return None
        return {
            'per_turn_shaping_rewards': list(self.per_turn_shaping_rewards),
            'per_turn_response_end_indices': list(self.per_turn_response_end_indices),
            'shaping_trailing_tool_call_turns': int(self.shaping_trailing_tool_call_turns),
            'shaping_multi_segment_skipped': bool(self.shaping_multi_segment_skipped),
        }

    def _reset_current_segment_shaping_state(self) -> None:
        if not self._shaping_enabled:
            return
        self.per_turn_shaping_rewards = []
        self.per_turn_response_end_indices = []
        self._pending_boundary = None
        self._segment_shaping.reset_state()
        self.shaping_trailing_tool_call_turns = 0

    def _shaping_flush_prior_turn(self, training_data: dict) -> None:
        """Attribute newly arrived tool messages to the previous assistant turn."""
        new_messages = training_data.get('messages') or []
        if len(new_messages) <= len(self.messages):
            if self._pending_boundary is not None:
                self.per_turn_shaping_rewards.append(0.0)
                self.per_turn_response_end_indices.append(self._pending_boundary)
                self._pending_boundary = None
                self._segment_shaping.reset_state()
            return

        window = new_messages[len(self.messages) : -1]
        new_tool_msgs = [m for m in window if isinstance(m, dict) and m.get('role') == 'tool']

        if self._pending_boundary is None:
            if new_tool_msgs:
                self._segment_shaping.reset_state()
            return

        tc_id_to_name: dict[str, str] = {}
        if self._segment_shaping.requires_tool_call_mapping:
            tc_id_to_name = self._build_tool_call_id_to_name(new_messages[:-1])
        turn_reward = self._segment_shaping.aggregate_turn(new_tool_msgs, tc_id_to_name)

        self.per_turn_shaping_rewards.append(turn_reward)
        self.per_turn_response_end_indices.append(self._pending_boundary)
        self._pending_boundary = None

    def _append_turn(self, training_data: dict) -> None:
        """Append a completion's training data as a conversation turn."""
        if self._mask_unhandled_prefix_reset:
            return
        if self._shaping_enabled:
            self._shaping_flush_prior_turn(training_data)

        # A mid-segment prefix_shortened desyncs append-only offsets, so mask. Checked before the limit raise:
        # a rewrite usually moves the limit too, and that raise runs on the harness's request path, failing
        # every later request. Guard on _first_prompt_captured (skip a segment's own first turn).
        if self._first_prompt_captured and training_data.get('prefix_shortened'):
            self._mask_unhandled_prefix_reset = True
            self.termination_reason = 'UNHANDLED_ERROR'
            logger.warning(
                'HytunerProcessor: prefix_shortened within one training segment; masking '
                'trajectory_id=%s request_id=%s completion_span_id=%s as UNHANDLED_ERROR.',
                self.trajectory_id,
                training_data.get('request_id'),
                training_data.get('completion_span_id'),
            )
            return

        has_effective_response_limit, effective_response_limit = _read_effective_response_limit(training_data)
        if has_effective_response_limit:
            if self._effective_response_limit_captured and self.effective_response_limit != effective_response_limit:
                raise ValueError(
                    'effective_response_limit changed within one training segment: '
                    f'{self.effective_response_limit!r} -> {effective_response_limit!r}; '
                    f'trajectory_id={self.trajectory_id!r}, request_id={training_data.get("request_id")!r}, '
                    f'completion_span_id={training_data.get("completion_span_id")!r}'
                )
            self.effective_response_limit = effective_response_limit
            self._effective_response_limit_captured = True

        extra_fields = training_data.get('extra_fields')
        include_topk_logprob_result = False
        if extra_fields is not None:
            check_key = [
                'resp_topk_logprob_indice',
                'resp_topk_logprob_value',
            ]
            for key in check_key:
                assert key in extra_fields, (
                    f'key:{key} not found in extra_fields, '
                    + 'check third_party/nexus/nexus/extensions/openai_clients/hytuner.py function _create_completion'
                )
            if (
                len(extra_fields['resp_topk_logprob_indice']) > 0
                and extra_fields['resp_topk_logprob_indice'][0] is not None
            ):
                include_topk_logprob_result = True

        # Capture first prompt (only once per segment)
        if not self._first_prompt_captured:
            self.prompt_ids = list(training_data['prompt_ids'])
            if include_topk_logprob_result and extra_fields is not None:
                self.topk = len(extra_fields['resp_topk_logprob_indice'][0])
            self._first_prompt_captured = True

        # append new prompt part
        new_prompt_part = training_data['prompt_ids'][len(self.prompt_ids) + len(self.response_ids) :]
        self.response_ids.extend(new_prompt_part)
        self.response_mask.extend([0 for _ in new_prompt_part])
        self.response_logprobs.extend([0.0 for _ in new_prompt_part])
        if include_topk_logprob_result:
            pad_list = [0 for _ in range(self.topk)]
            self.resp_topk_logprob_indice.extend([pad_list for _ in new_prompt_part])
            pad_list = [0.0 for _ in range(self.topk)]
            self.resp_topk_logprob_value.extend([pad_list for _ in new_prompt_part])

        # append new response part
        response_ids = list(training_data['response_ids'])
        self.response_ids.extend(response_ids)
        self.response_mask.extend([1 for _ in response_ids])

        self.raw_input_ids = list(training_data['raw_prompt_ids']) + response_ids

        # ``routed_experts`` is the whole-segment routing accumulated by the
        # client; each completion carries the latest (superset) array.
        if training_data.get('routed_experts') is not None:
            self.routed_experts = training_data['routed_experts']
        if include_topk_logprob_result and extra_fields is not None:
            resp_topk_logprob_indice = list(extra_fields['resp_topk_logprob_indice'])
            resp_topk_logprob_value = list(extra_fields['resp_topk_logprob_value'])
            if len(resp_topk_logprob_indice) < len(response_ids):
                pad_size = len(response_ids) - len(resp_topk_logprob_indice)
                resp_topk_logprob_indice.extend([0 for _ in range(self.topk)] * pad_size)
                resp_topk_logprob_value.extend([0.0 for _ in range(self.topk)] * pad_size)
            elif len(resp_topk_logprob_indice) > len(response_ids):
                chunk_size = len(response_ids)
                resp_topk_logprob_indice = resp_topk_logprob_indice[:chunk_size]
                resp_topk_logprob_value = resp_topk_logprob_value[:chunk_size]
            self.resp_topk_logprob_indice.extend(resp_topk_logprob_indice)
            self.resp_topk_logprob_value.extend(resp_topk_logprob_value)

        response_logprobs = training_data.get('response_logprobs')
        if response_logprobs is None:
            response_logprobs = [0.0 for _ in response_ids]
        else:
            response_logprobs = list(response_logprobs)
            if len(response_logprobs) < len(response_ids):
                response_logprobs.extend([0.0] * (len(response_ids) - len(response_logprobs)))
            elif len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
        self.response_logprobs.extend(response_logprobs)

        # Capture LLM inference time from training data
        llm_inference_time = training_data.get('llm_inference_time')
        if llm_inference_time is not None:
            self.llm_inference_time = llm_inference_time
        true_llm_inference_time = training_data.get('true_llm_inference_time')
        if true_llm_inference_time is not None:
            self.true_llm_inference_time = true_llm_inference_time
        elif llm_inference_time is not None:
            # Backward compatibility for traces emitted before the two metrics
            # were split: the sole value represented the best known inference time.
            self.true_llm_inference_time = llm_inference_time

        # capture termination_reason if present
        if 'termination_reason' in training_data:
            self.termination_reason = training_data['termination_reason']

        if len(self.response_ids) != len(self.response_mask) or len(self.response_ids) != len(self.response_logprobs):
            raise ValueError(
                f'HyTunerProcessor length mismatch: response_ids={len(self.response_ids)} response_mask={len(self.response_mask)} response_logprobs={len(self.response_logprobs)}'
            )

        # assign the messages directly
        self.messages = training_data['messages']

        # NOTE: tool_call_stats is intentionally NOT updated per-turn.
        # hytuner's OpenAI client emits training_data['messages'] without the
        # current turn's tool/user response messages (they are appended by the
        # agent loop *before* the next LLM call). A per-turn computation here
        # therefore consistently misses the executor output and over-reports
        # exec_failure. Plan B: compute everything once at reporting time
        # (``_recompute_tool_call_stats``) when the full message history is
        # already settled.

        # accumulate abort wait time
        abort_wait_time = training_data.get('abort_wait_time', 0.0)
        if abort_wait_time:
            self.abort_wait_time = abort_wait_time  # Use latest value (cumulative in client)

        # capture request_id
        if 'request_id' in training_data:
            self.request_id = training_data['request_id']

        if self._shaping_enabled:
            self._pending_boundary = len(self.response_ids) - 1 if response_ids else -1

        self.num_turns += 1

    def on_start(self, span: Span, event: SpanStartEvent) -> None:
        """Record the task name from the task entity span."""
        if span.attributes.get('type') == 'task':
            self.task_name = span.attributes.get('name')

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.name == 'openai_completion':
            training_data = span.attributes.get('_hytuner_training_data')
            if training_data:
                # The parent span still determines turn/compaction semantics.
                # This also deduplicates a direct HyTunerClient notification.
                self.record_training_data(training_data)
            return

        span_type = span.attributes.get('type')

        # Handle agent_inference: flush buffer as normal conversation turn
        if span_type == 'agent_inference':
            self._flush_buffered_as_turn()

        # Handle compaction span. V2 trains a self-summary completion only when
        # that exact summary was applied. Other strategies retain the legacy
        # buffering behavior for compatibility.
        elif span_type == 'agent_compaction':
            compacted = span.attributes.get('compacted') is True
            requested_strategy = span.attributes.get('requested_strategy')
            if requested_strategy == 'self_summarize_v2':
                if compacted:
                    self._finalize_current_segment()
                    if (
                        self._buffered_completion is not None
                        and span.attributes.get('effective_strategy') == 'self_summarize_v2'
                    ):
                        self._record_standalone_segment(self._buffered_completion)
                self._buffered_completion = None
            elif compacted:
                self._finalize_current_segment()
                if self._buffered_completion is not None:
                    self._record_standalone_segment(self._buffered_completion)
                    self._buffered_completion = None

        # Timeline recording
        if span_type in _TIMELINE_TYPES and span.end_time_unix_nano is not None:
            # Skip timeline recording for compaction spans that did not
            # actually compact (compacted=False) to avoid noise.
            if span_type == 'agent_compaction' and not span.attributes.get('compacted'):
                pass  # fall through without appending to timeline
            else:
                # Derive segment type/name from span attributes
                seg_type = span_type
                if span_type == 'agent_observation':
                    seg_name = span.attributes.get('tool_name', 'observation')
                elif span_type == 'agent_inference':
                    seg_name = 'gen'
                elif span_type == 'agent_compaction':
                    seg_name = 'compaction'
                elif span_type == 'task_phase':
                    seg_type = self.task_name
                    seg_name = span.attributes.get('phase', span.name)
                else:
                    seg_name = span.name

                segment: dict = {
                    'type': seg_type,
                    'name': seg_name,
                    'start_time': span.start_time_unix_nano / 1e9,
                    'end_time': span.end_time_unix_nano / 1e9,
                }
                tool_args_json = span.attributes.get('hytuner.tool_args')
                if tool_args_json is not None:
                    if isinstance(tool_args_json, str):
                        try:
                            segment['tool_args'] = json.loads(tool_args_json)
                        except json.JSONDecodeError:
                            segment['tool_args'] = tool_args_json
                    else:
                        segment['tool_args'] = tool_args_json
                if 'hytuner.tool_success' in span.attributes:
                    segment['success'] = span.attributes['hytuner.tool_success']
                if 'hytuner.tool_error' in span.attributes:
                    segment['error'] = span.attributes['hytuner.tool_error']
                if 'hycode.span_transfer' in span.attributes:
                    segment['span_transfer'] = span.attributes['hycode.span_transfer']
                self.agent_timeline.append(segment)

        # Read submission patches reported through Nexus tracing.
        if 'hytuner.patch_text' in span.attributes:
            self.patch_text = span.attributes['hytuner.patch_text'] or ''

        # Check for termination_reason and other metrics from non-LLM spans
        termination_reason = span.attributes.get('hytuner.termination_reason') or span.attributes.get(
            'agent.termination_reason'
        )
        if termination_reason and not self._mask_unhandled_prefix_reset:
            self.termination_reason = termination_reason

        # The reason a task replaced, when it reported one termination reason in place of another.
        # Kept beside termination_reason so the dump still says what actually happened.
        cleared_termination_reason = span.attributes.get('hytuner.cleared_termination_reason')
        if cleared_termination_reason:
            self.cleared_termination_reason = cleared_termination_reason

        observation_time = span.attributes.get('hytuner.observation_time') or span.attributes.get(
            'agent.observation_time'
        )
        if observation_time is not None:
            self.observation_time += float(observation_time)

        reward_time = span.attributes.get('hytuner.reward_time')
        if reward_time is not None:
            self.reward_time += float(reward_time)

        # Init subphase durations come from span timestamps; unentered phases remain zero.
        if span_type == 'task_phase' and span.end_time_unix_nano is not None:
            duration = (span.end_time_unix_nano - span.start_time_unix_nano) / 1e9
            if span.name == 'task.init.get_runtime':
                self.sandbox_init_time += duration
            elif span.name == 'task.init.compose':
                self.init_compose_time += duration
            elif span.name == 'task.init.prepare_directories':
                self.init_prepare_directories_time += duration
            elif span.name == 'task.init.boot':
                self.init_boot_time += duration
            elif span.name == 'task.handoff.artifact_collect_time':
                self.handoff_artifact_collect_time += duration
            elif span.name == 'task.handoff.release_main_time':
                self.handoff_release_main_time += duration
            elif span.name == 'task.handoff.sandbox_init_time':
                self.handoff_sandbox_init_time += duration
            elif span.name == 'task.handoff.env_setup_time':
                self.handoff_env_setup_time += duration
            elif span.name == 'task.scoring.check_score_time':
                self.scoring_check_score_time += duration

        runtime_id = span.attributes.get('hytuner.runtime_id')
        if runtime_id:
            self.runtime_id = runtime_id
            if self._progress_state is not None:
                self._progress_state['runtime_id'] = runtime_id
            if runtime_id not in self.runtime_ids:
                self.runtime_ids.append(runtime_id)

        eval_runtime_id = span.attributes.get('hytuner.eval_runtime_id')
        if eval_runtime_id:
            self.eval_runtime_id = eval_runtime_id
            if self._progress_state is not None:
                self._progress_state['eval_runtime_id'] = eval_runtime_id

    def _parse_tool_stats(self) -> dict[str, dict[str, int]]:
        """Parse tool call statistics from accumulated messages.

        Returns:
            A dict mapping tool names to their stats:
            {tool_name: {"total": N, "success": N, "failure": N,
                         "failure_readpage": N, "failure_summary_llm": N}}

            ``failure_readpage`` and ``failure_summary_llm`` are emitted for the
            ``visit`` tool only and split ``failure`` into the two stages of its
            internal pipeline (jina_readpage HTTP fetch vs. LLM summary call).
            For other tools they remain 0.
        """
        stats: dict[str, dict[str, int]] = {}
        tool_call_map: dict[str, str] = {}  # {tool_call_id: tool_name}

        for msg in self.messages:
            if not isinstance(msg, dict):
                continue
            # Extract tool calls from assistant messages
            if msg.get('role') == 'assistant':
                for tc in msg.get('tool_calls') or []:
                    func = tc.get('function', {})
                    name = func.get('name', 'unknown')
                    tc_id = tc.get('id', '')
                    tool_call_map[tc_id] = name
                    if name not in stats:
                        stats[name] = {
                            'total': 0,
                            'success': 0,
                            'failure': 0,
                            'failure_readpage': 0,
                            'failure_summary_llm': 0,
                        }
                    stats[name]['total'] += 1
            # Match tool results to tool calls
            if msg.get('role') == 'tool':
                tc_id = msg.get('tool_call_id', '')
                name = tool_call_map.get(tc_id, msg.get('name', 'unknown'))
                if name not in stats:
                    stats[name] = {
                        'total': 0,
                        'success': 0,
                        'failure': 0,
                        'failure_readpage': 0,
                        'failure_summary_llm': 0,
                    }
                content = str(msg.get('content', ''))
                if _is_tool_failure(name, content):
                    stats[name]['failure'] += 1
                    if name == 'visit':
                        stage = _classify_visit_failure(content)
                        stats[name][f'failure_{stage}'] += 1
                else:
                    stats[name]['success'] += 1
        return stats

    def _recompute_tool_call_stats(self) -> None:
        """Recompute per-turn tool-call statistics from the FULL message history.

        Plan B: rather than incrementally accumulating per-turn stats during
        ``_append_turn`` (which racy w.r.t. the agent loop's tool-message
        injection ordering), we scan ``self.messages`` once at reporting time.

        The trajectory is partitioned into per-assistant windows. Each window
        starts at one assistant message and stops just before the next
        assistant message; the contents in between (tool / user messages) are
        the executor responses for that assistant. Each window is fed to
        ``build_turn_stats_from_messages`` which then dispatches to the SWE
        or Terminus2 branch and returns one per-turn record.

        The total scan time is logged at INFO so we can monitor the cost of
        the analytics path.
        """
        t0 = time.perf_counter()
        messages_seq = self.messages if isinstance(self.messages, list) else []
        cache_key = (id(messages_seq), len(messages_seq))
        if self._tool_stats_cache_key == cache_key:
            return
        new_stats: list[dict] = []
        gating_rejected = False
        compute_failed = False
        try:
            messages = messages_seq
            # Find every assistant index in order; their windows are
            # [idx_k, idx_{k+1}) (last one extends to len(messages)).
            assistant_indices: list[int] = [
                i for i, m in enumerate(messages) if isinstance(m, dict) and m.get('role') == 'assistant'
            ]
            for k, start in enumerate(assistant_indices):
                end = assistant_indices[k + 1] if k + 1 < len(assistant_indices) else len(messages)
                window = messages[start:end]
                turn_stats = build_turn_stats_from_messages(window, task_name=self.task_name)
                if turn_stats is None:
                    # Plan A gating rejected this task; abandon the whole
                    # trajectory's tool stats so downstream writers omit the
                    # field entirely (matching the previous semantics).
                    gating_rejected = True
                    new_stats = []
                    break
                new_stats.append(turn_stats)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug('HyTuner: _recompute_tool_call_stats failed: %s', e)
            new_stats = []
            compute_failed = True

        if gating_rejected:
            self.tool_call_stats = []
        else:
            self.tool_call_stats = new_stats

        # Only memoize a clean (non-exceptional) result so a transient bug
        # doesn't get pinned for the rest of the trajectory.
        if not compute_failed:
            self._tool_stats_cache_key = cache_key
        else:
            self._tool_stats_cache_key = None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_turns = len(self.tool_call_stats)
        n_messages = len(self.messages) if isinstance(self.messages, list) else 0
        print(
            f'HyTuner: _recompute_tool_call_stats took {elapsed_ms:.2f}ms '
            f'(turns={n_turns}, n_messages={n_messages}, gating_rejected={gating_rejected})'
        )

    def get_result(self) -> dict[str, list[int] | list[float] | list[str] | list[dict] | int | str | None | dict]:
        # Flush any remaining buffered completion (covers the last LLM call
        # for agents that never emit agent_inference spans).
        self._flush_buffered_as_turn()
        # Plan B: derive tool_call_stats from the full settled message history
        # right before reporting. Cheap (microseconds — milliseconds) and
        # avoids the per-turn race with the agent loop's tool-message append.
        self._recompute_tool_call_stats()

        self._finalize_pending_shaping_turn()

        if self.resp_topk_logprob_indice:
            extra_fields = {
                'resp_topk_logprob_indice': self.resp_topk_logprob_indice,
                'resp_topk_logprob_value': self.resp_topk_logprob_value,
            }
        else:
            extra_fields = None

        if self._shaping_enabled:
            if extra_fields is None:
                extra_fields = {}
            shaping_extra_fields = self._build_current_segment_shaping_extra_fields()
            if shaping_extra_fields is not None:
                extra_fields.update(shaping_extra_fields)

        if self._effective_response_limit_captured:
            if extra_fields is None:
                extra_fields = {}
            extra_fields[_EFFECTIVE_RESPONSE_LIMIT_FIELD] = self.effective_response_limit

        result = {
            'prompt_ids': self.prompt_ids,
            'raw_input_ids': self.raw_input_ids,
            'response_ids': self.response_ids,
            'response_mask': self.response_mask,
            'response_logprobs': self.response_logprobs if self.response_logprobs else None,
            'routed_experts': self.routed_experts,
            'messages': self.messages,
            'num_turns': self.num_turns,
            'termination_reason': self.termination_reason,
            'unhandled_prefix_reset': self._mask_unhandled_prefix_reset,
            'tool_stats': self._parse_tool_stats(),
            'runtime_id': self.runtime_id,
            'runtime_ids': self.runtime_ids,
            'request_id': self.request_id,
            'eval_runtime_id': self.eval_runtime_id if self.eval_runtime_id else 'unknown',
            'metrics': {
                'abort_wait_time': self.abort_wait_time,
                'llm_inference_time': self.llm_inference_time,
                'true_llm_inference_time': self.true_llm_inference_time,
                'observation_time': self.observation_time,
                'reward_time': self.reward_time,
                'sandbox_init_time': self.sandbox_init_time,
                'init_compose_time': self.init_compose_time,
                'init_prepare_directories_time': self.init_prepare_directories_time,
                'init_boot_time': self.init_boot_time,
                'handoff_artifact_collect_time': self.handoff_artifact_collect_time,
                'handoff_release_main_time': self.handoff_release_main_time,
                'handoff_sandbox_init_time': self.handoff_sandbox_init_time,
                'handoff_env_setup_time': self.handoff_env_setup_time,
                'scoring_check_score_time': self.scoring_check_score_time,
            },
            'extra_fields': inject_tool_call_into_extra(extra_fields, self.tool_call_stats),
            'agent_timeline': self.agent_timeline,
        }
        # Include prior segments from context management (if any)
        if self.segments:
            result['segments'] = self.segments
        return result

    def get_dump_result(self) -> dict:
        """
        Collect and return the core runtime information of the trajectory in a
        serializable dictionary format.

        This method is mainly used for dumping or logging intermediate/final results.
        Args:
            messages: the recorded message history
            num_turns: the total number of interaction turns
            trajectory_id: a unique identifier for the trajectory (empty string if not set)
        """
        # Plan B: recompute tool stats from the full message history before
        # reporting so the dumped tool_call_stats / tool_call_summary cover
        # the whole trajectory consistently.
        self._recompute_tool_call_stats()
        dump = {
            'messages': self.messages,
            'num_turns': self.num_turns,
            'trajectory_id': self.trajectory_id if self.trajectory_id else '',
            'termination_reason': self.termination_reason,
            'runtime_id': self.runtime_id if self.runtime_id else 'unknown',
            'runtime_ids': self.runtime_ids,
            'eval_runtime_id': self.eval_runtime_id if self.eval_runtime_id else 'unknown',
            'metrics': {
                'abort_wait_time': self.abort_wait_time,
                'llm_inference_time': self.llm_inference_time,
                'true_llm_inference_time': self.true_llm_inference_time,
                'observation_time': self.observation_time,
                'reward_time': self.reward_time,
                'sandbox_init_time': self.sandbox_init_time,
                'init_compose_time': self.init_compose_time,
                'init_prepare_directories_time': self.init_prepare_directories_time,
                'init_boot_time': self.init_boot_time,
                'handoff_artifact_collect_time': self.handoff_artifact_collect_time,
                'handoff_release_main_time': self.handoff_release_main_time,
                'handoff_sandbox_init_time': self.handoff_sandbox_init_time,
                'handoff_env_setup_time': self.handoff_env_setup_time,
                'scoring_check_score_time': self.scoring_check_score_time,
            },
        }
        if self.cleared_termination_reason is not None:
            # Present only on a rollout whose task reported one reason in place of another, so its
            # absence is the normal case and its presence is the thing worth counting.
            dump['cleared_termination_reason'] = self.cleared_termination_reason
        if self.patch_text is not None:
            dump['patch_text'] = self.patch_text
        if self.tool_call_stats:
            dump['tool_call_stats'] = list(self.tool_call_stats)
            dump['tool_call_summary'] = summarize_tool_call_stats(self.tool_call_stats)
        return dump

    def snapshot_result(self) -> dict:
        """Get a read-only snapshot of current trajectory data.

        Unlike get_result(), this does NOT flush buffered completions.
        It does, however, refresh ``tool_call_stats`` from the current
        ``self.messages`` so the reported analytics stay consistent with the
        Plan B model (one-shot scan at reporting time).
        """
        self._recompute_tool_call_stats()
        if self.resp_topk_logprob_indice:
            extra_fields = {
                'resp_topk_logprob_indice': self.resp_topk_logprob_indice,
                'resp_topk_logprob_value': self.resp_topk_logprob_value,
            }
        else:
            extra_fields = None
        if self._effective_response_limit_captured:
            if extra_fields is None:
                extra_fields = {}
            extra_fields[_EFFECTIVE_RESPONSE_LIMIT_FIELD] = self.effective_response_limit

        return {
            'prompt_ids': list(self.prompt_ids),
            'raw_input_ids': list(self.raw_input_ids),
            'response_ids': list(self.response_ids),
            'response_mask': list(self.response_mask),
            'response_logprobs': list(self.response_logprobs) if self.response_logprobs else None,
            'routed_experts': self.routed_experts,
            'messages': list(self.messages),
            'num_turns': self.num_turns,
            'termination_reason': self.termination_reason,
            'tool_stats': self._parse_tool_stats(),
            'runtime_id': self.runtime_id,
            'request_id': self.request_id,
            'eval_runtime_id': self.eval_runtime_id if self.eval_runtime_id else 'unknown',
            'metrics': {
                'abort_wait_time': self.abort_wait_time,
                'llm_inference_time': self.llm_inference_time,
                'observation_time': self.observation_time,
                'reward_time': self.reward_time,
                'sandbox_init_time': self.sandbox_init_time,
                'init_compose_time': self.init_compose_time,
                'init_prepare_directories_time': self.init_prepare_directories_time,
                'init_boot_time': self.init_boot_time,
                'handoff_artifact_collect_time': self.handoff_artifact_collect_time,
                'handoff_release_main_time': self.handoff_release_main_time,
                'handoff_sandbox_init_time': self.handoff_sandbox_init_time,
                'handoff_env_setup_time': self.handoff_env_setup_time,
                'scoring_check_score_time': self.scoring_check_score_time,
            },
            'extra_fields': inject_tool_call_into_extra(extra_fields, self.tool_call_stats),
            'agent_timeline': list(self.agent_timeline),
            'segments': [dict(seg) for seg in self.segments],
        }

    async def afinalize(self) -> None:
        """Compute tool-call statistics and write the trajectory dump in a worker thread.

        Both are synchronous and can take seconds on a long trajectory (one bashlex parse per
        shell command, then flock and fsync on a node-shared file). Run on the event loop they
        stalled every other trajectory of the AgentLoopWorker and the rollout controller's
        status RPCs. ``single.run`` awaits this after ``processor_context`` has unregistered the
        processor and drained span dispatch, so nothing mutates the trajectory concurrently.

        The last completion of an agent that emits no ``agent_inference`` span is still buffered
        here. It is flushed first, on the loop like every other turn mutation, so the statistics
        and the dump cover the final turn and ``get_result`` finds them already computed.
        """
        self._flush_buffered_as_turn()
        try:
            await asyncio.to_thread(self._finalize_off_loop)
        except Exception:
            # Same contract as processor_context gave finalize(): a failed dump is logged and
            # never fails the trajectory it describes.
            logger.exception('HyTuner: finalizing trajectory %s failed', self.trajectory_id)

    def _finalize_off_loop(self) -> None:
        self._recompute_tool_call_stats()
        # Write the trajectory to file, if path not provided, skip
        if self.file_path is None:
            return

        step_id = int(self.trajectory_id.split('_')[0]) if self.trajectory_id else 0
        output_file_path = os.path.join(
            self.file_path, 'chat_completions', f'global_step_{step_id}', f'{_node_dump_stem()}.jsonl'
        )

        self.write_trajectory_to_file(output_file_path, self.get_dump_result())

    def write_trajectory_to_file(self, file_path: str | Path, content: dict, mode: str = 'a') -> None:
        """Process-safe and coroutine-safe file writing.

        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            mode: File open mode ('a' for append, 'w' for write)
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file with appropriate flags
        fd = os.open(str(file_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            # Acquire exclusive lock - this blocks until lock is available
            # This works across processes AND coroutines
            fcntl.flock(fd, fcntl.LOCK_EX)

            try:
                # Handle write mode
                if mode == 'w':
                    os.ftruncate(fd, 0)
                    os.lseek(fd, 0, os.SEEK_SET)
                else:
                    # Append mode - seek to end
                    os.lseek(fd, 0, os.SEEK_END)

                # Write content
                os.write(fd, json.dumps(content, ensure_ascii=False).encode('utf-8') + b'\n')

                # Ensure data is written to disk
                os.fsync(fd)
            except Exception:
                logger.exception(
                    'Failed to dump message to file (trajectory_id=%s, mode=%s, content_type=%s, content_len=%s)',
                    self.trajectory_id,
                    mode,
                    type(content).__name__,
                    len(content) if hasattr(content, '__len__') else 'N/A',
                )
            finally:
                # Release file lock
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
