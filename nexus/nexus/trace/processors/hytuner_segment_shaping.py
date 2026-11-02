from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any


SHAPING_ERROR_TAXONOMIES = {'coarse', 'controllable_v1'}
SHAPING_POLICIES = {'legacy', 'tool_error_turn_penalty_only'}

FORMAT_ERROR_PHRASES = (
    'json decode',
    'json parse',
    'invalid json',
    'malformed json',
    'parse arguments',
    'parse tool',
    'format error',
)
INVALID_PARAM_PHRASES = (
    'empty code',
    'empty query',
    'empty pattern',
    'query cannot be empty',
    'missing required',
    'missing parameter',
    'invalid parameter',
    'invalid params',
    'invalid argument',
    'invalid arguments',
    'invalid input',
    'validation error',
    'no browser state found',
    'cannot run find on search results page',
    'url is required',
    'invalid url',
    'invalid link id',
    'no pages to access',
    'available cursor indices',
    'invalid location parameter',
    '`cursor` should be an integer',
)
OOM_PHRASES = (
    'out of memory',
    'cuda out of memory',
    'memoryerror',
    'oom',
)
TRANSPORT_PHRASES = (
    'api request timed out',
    'api client error',
    'api request failed',
    'failed to parse api response',
    'http error',
    'invoke: job',
    'call_tool failed',
    'status=',
    'remote runtime unavailable',
    'runtime unavailable',
    'runtime not available',
    'runtime is not connected',
    'kernel not started',
    'failed to start kernel',
    'failed to initialize kernel',
    'connection refused',
    'connection reset',
    'connection aborted',
    'timed out',
    'timeout',
    'service unavailable',
    'temporarily unavailable',
    'gateway timeout',
    'bad gateway',
    'too many requests',
    'rate limit',
    'remote end closed',
    'network error',
)
SEARCH_NO_RESULT_PHRASES = (
    'no results',
    'no result',
    '0 results',
    'zero results',
    'nothing found',
)
OPEN_FETCH_PHRASES = (
    'failed to fetch url',
    'fetch error',
    'could not be resolved',
    'http error',
    'status code',
    '404',
    '403',
    '500',
    'ssl',
    'certificate',
)


@dataclass(frozen=True)
class SegmentShapingConfig:
    mode: str | None
    policy: str
    weights: dict[str, float]
    success_weights: dict[str, float]
    error_taxonomy: str
    escalation_factor: float
    tool_error_turn_penalty: float
    enabled: bool


def canonical_tool_name_for_reward(tool_name: str | None) -> str:
    """Map concrete tool names to stable reward-policy names."""
    name = (tool_name or '').strip().lower()
    if not name:
        return 'unknown'

    leaf = name.rsplit('/', 1)[-1]
    if name == 'search_stateful':
        return 'search'
    if leaf in {'search', 'open', 'find'} and (name == leaf or name.startswith('search_stateful/')):
        return leaf

    if leaf == 'python' or name.endswith('python_stateful') or 'code_interpreter' in name or 'jupyter' in name:
        return 'python'

    return leaf or name


def normalize_segment_shaping_config(
    segment_shaping_config: dict[str, Any] | None = None,
    *,
    shaping_mode: str | None = None,
    shaping_policy: str = 'legacy',
    shaping_weights: dict[str, float] | None = None,
    shaping_success_weights: dict[str, float] | None = None,
    shaping_error_taxonomy: str = 'coarse',
    shaping_escalation_factor: float = 0.0,
    tool_error_turn_penalty: float = -0.02,
) -> SegmentShapingConfig:
    """Normalize runner dict config and legacy processor kwargs into one config."""
    config = segment_shaping_config
    if config is not None:
        shaping_mode = config.get('mode', config.get('shaping_mode', shaping_mode))
        shaping_policy = config.get('policy', config.get('shaping_policy', shaping_policy))
        shaping_weights = config.get('weights', config.get('shaping_weights', shaping_weights))
        shaping_success_weights = config.get(
            'success_weights', config.get('shaping_success_weights', shaping_success_weights)
        )
        shaping_error_taxonomy = config.get(
            'error_taxonomy', config.get('shaping_error_taxonomy', shaping_error_taxonomy)
        )
        shaping_escalation_factor = config.get(
            'escalation_factor', config.get('shaping_escalation_factor', shaping_escalation_factor)
        )
        tool_error_turn_penalty = config.get('tool_error_turn_penalty', tool_error_turn_penalty)

    policy = str(shaping_policy or 'legacy')
    if policy not in SHAPING_POLICIES:
        allowed = ', '.join(sorted(SHAPING_POLICIES))
        raise ValueError(f'Unknown shaping_policy: {policy!r}. Only {allowed} are supported.')
    if policy == 'tool_error_turn_penalty_only' and shaping_mode != 'segment':
        raise ValueError("shaping_policy='tool_error_turn_penalty_only' requires shaping_mode='segment'.")

    weights = dict(shaping_weights or {})
    success_weights = {
        canonical_tool_name_for_reward(k): float(v) for k, v in dict(shaping_success_weights or {}).items()
    }
    escalation_factor = float(shaping_escalation_factor or 0.0)
    tool_error_penalty = float(tool_error_turn_penalty)
    if policy == 'tool_error_turn_penalty_only':
        if weights:
            raise ValueError("shaping_policy='tool_error_turn_penalty_only' requires empty shaping_weights.")
        if success_weights:
            raise ValueError("shaping_policy='tool_error_turn_penalty_only' requires empty shaping_success_weights.")
        if escalation_factor != 0.0:
            raise ValueError("shaping_policy='tool_error_turn_penalty_only' requires shaping_escalation_factor=0.0.")
        if tool_error_penalty > 0.0:
            raise ValueError("tool_error_turn_penalty must be <= 0 for shaping_policy='tool_error_turn_penalty_only'.")

    error_taxonomy = str(shaping_error_taxonomy or 'coarse')
    if error_taxonomy not in SHAPING_ERROR_TAXONOMIES:
        allowed = ', '.join(sorted(SHAPING_ERROR_TAXONOMIES))
        raise ValueError(f'Unknown shaping_error_taxonomy: {error_taxonomy!r}. Only {allowed} are supported.')

    enabled = (shaping_mode == 'segment') and (
        policy == 'tool_error_turn_penalty_only' or bool(weights or success_weights)
    )
    return SegmentShapingConfig(
        mode=shaping_mode,
        policy=policy,
        weights=weights,
        success_weights=success_weights,
        error_taxonomy=error_taxonomy,
        escalation_factor=escalation_factor,
        tool_error_turn_penalty=tool_error_penalty,
        enabled=enabled,
    )


class SegmentShapingClassifier:
    """Classify and aggregate per-turn tool shaping rewards for HyTuner."""

    def __init__(self, config: SegmentShapingConfig):
        self.config = config
        self._consecutive_event_key: tuple[str, str] | None = None
        self._consecutive_count: int = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def requires_tool_call_mapping(self) -> bool:
        return self.config.enabled and self.config.policy != 'tool_error_turn_penalty_only'

    def reset_state(self) -> None:
        self._consecutive_event_key = None
        self._consecutive_count = 0

    def aggregate_turn(self, tool_messages: list[dict], tool_call_id_to_name: dict[str, str]) -> float:
        if self.config.policy == 'tool_error_turn_penalty_only':
            self.reset_state()
            return self.aggregate_tool_error_turn_penalty(tool_messages)

        events: list[tuple[str, str]] = []
        for tool_msg in tool_messages:
            tc_id = tool_msg.get('tool_call_id', '') if isinstance(tool_msg, dict) else ''
            tool_name = tool_call_id_to_name.get(tc_id, 'unknown')
            content = tool_msg.get('content', '') if isinstance(tool_msg, dict) else ''
            event_type = self.classify_tool_message(tool_name, content)
            events.append((tool_name, event_type))

        turn_reward = self._aggregate_turn_events(events)
        if not events:
            self.reset_state()
        return turn_reward

    def classify_tool_message(self, tool_name: str | None, content: Any) -> str:
        if self.config.error_taxonomy == 'coarse':
            return self._classify_tool_message_coarse(content)
        return self._classify_tool_message_controllable_v1(tool_name, content)

    @staticmethod
    def _classify_tool_message_coarse(content: Any) -> str:
        text = tool_message_prefix_text(content)
        if 'Error' in text or 'error' in text:
            return 'tool_runtime_error'
        return 'success'

    def _classify_tool_message_controllable_v1(self, tool_name: str | None, content: Any) -> str:
        """Classify Nexus Tool error messages into controllability buckets."""
        text = tool_message_prefix_text(content).strip()
        if not text.lower().startswith('tool error:'):
            return 'success'

        body = text.split(':', 1)[1].strip()
        lower = body.lower()
        canonical_tool_name = canonical_tool_name_for_reward(tool_name)

        if _contains_any(lower, OOM_PHRASES):
            return 'oom'
        if canonical_tool_name == 'python' and _has_python_code_execution_context(lower):
            return 'code_runtime_error'
        if _contains_any(lower, FORMAT_ERROR_PHRASES):
            return 'format_error'
        if _contains_any(lower, INVALID_PARAM_PHRASES):
            return 'invalid_params'
        if _is_unknown_tool_error(lower) or canonical_tool_name == 'unknown':
            return 'unknown_tool'
        if canonical_tool_name == 'search' and _contains_any(lower, SEARCH_NO_RESULT_PHRASES):
            return 'search_no_results'
        if canonical_tool_name == 'open' and _contains_any(lower, OPEN_FETCH_PHRASES):
            return 'open_fetch_error'
        if _contains_any(lower, TRANSPORT_PHRASES):
            return 'transport_error'
        if canonical_tool_name == 'python' and _has_python_exception_name(lower):
            return 'code_runtime_error'
        if canonical_tool_name == 'python':
            return 'code_runtime_error'
        if canonical_tool_name == 'search':
            return 'search_error'
        if canonical_tool_name == 'open':
            return 'open_fetch_error'
        return 'tool_runtime_error'

    def _aggregate_turn_events(self, events: list[tuple[str, str]]) -> float:
        turn_total = 0.0
        for tool_name, event_type in events:
            if event_type == 'success':
                canonical_tool_name = canonical_tool_name_for_reward(tool_name)
                weight = self.config.success_weights.get(canonical_tool_name, self.config.weights.get(event_type, 0.0))
            else:
                weight = self.config.weights.get(event_type, 0.0)

            current_key = (tool_name, event_type)
            if event_type == 'success':
                self.reset_state()
            elif current_key == self._consecutive_event_key:
                self._consecutive_count += 1
            else:
                self._consecutive_count = 0
                self._consecutive_event_key = current_key
            multiplier = 1.0 + self.config.escalation_factor * self._consecutive_count
            turn_total += weight * multiplier
        return turn_total

    def aggregate_tool_error_turn_penalty(self, tool_messages: list[dict]) -> float:
        has_tool_error = any(
            tool_message_prefix_text(tool_msg.get('content', '')).lower().startswith('tool error:')
            for tool_msg in tool_messages
        )
        return self.config.tool_error_turn_penalty if has_tool_error else 0.0


def tool_message_prefix_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                return str(part.get('text', ''))
    return str(content or '')


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_unknown_tool_error(text: str) -> bool:
    if 'unknown tool' in text or 'tool not found' in text:
        return True
    return bool(
        'not found. available' in text and re.search(r"\b(?:mcp\s+)?tool\s+['\"`][^'\"`]+['\"`]\s+not found\b", text)
    )


def _has_python_code_execution_context(text: str) -> bool:
    return 'error executing code' in text or 'traceback (most recent call last)' in text


def _has_python_exception_name(text: str) -> bool:
    return bool(
        re.search(
            r'\b(?:syntax|name|type|value|runtime|index|key|zero\s*division|'
            r'attribute|import|module\s*not\s*found|time\s*limit\s*exceeded)error\b',
            text,
        )
    )
