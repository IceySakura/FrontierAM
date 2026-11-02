"""Deterministic evaluators for WebArena-Verified.

This module provides deterministic evaluation without LLM-as-judge:
- ValueNormalizer: Schema-driven value normalization
- ValueComparator: Recursive structural comparison
- Evaluators: AgentResponseEvaluator, NetworkEventEvaluator, etc.

Key features matching WebArena-Verified:
- Type-aware normalization (duration, location-name, etc.)
- Unordered array comparison
- JSON key order independent object comparison
- Tolerance-based numeric/coordinate comparison

Reference: WebArena-Verified evaluation system
https://github.com/ServiceNow/webarena-verified
"""

from __future__ import annotations

import contextlib
import json
import logging
import re

from typing import Any

from .data_types import (
    TYPE_REGISTRY,
    NormalizedBoolean,
    NormalizedNumber,
    NormalizedString,
    NormalizedType,
)


logger = logging.getLogger(__name__)


class EvalResult:
    """Result of an evaluation check."""

    def __init__(self, success: bool, message: str = '', details: dict | None = None):
        self.success = success
        self.message = message
        self.details = details or {}

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = '✓' if self.success else '✗'
        return f'EvalResult({status}, {self.message!r})'


class ValueNormalizer:
    """Schema-driven value normalizer.

    Normalizes values according to JSON Schema type definitions,
    supporting alternatives and nested structures.
    """

    def normalize(
        self, value: Any, schema: dict | None = None, strict: bool = True, **kwargs: Any
    ) -> NormalizedType | tuple | dict | None:
        """Normalize a value according to schema.

        Args:
            value: Raw value to normalize
            schema: JSON Schema defining the type
            strict: If True, raise on normalization errors

        Returns:
            Normalized value (NormalizedType, tuple for arrays, dict for objects)
        """
        if value is None:
            return None

        if schema is None:
            # Infer type from value
            if isinstance(value, bool):
                return NormalizedBoolean(value)
            if isinstance(value, (int, float)):
                return NormalizedNumber(value)
            if isinstance(value, (list, tuple)):
                return tuple(self.normalize(item, None, strict) for item in value)
            if isinstance(value, dict):
                return {k: self.normalize(v, None, strict) for k, v in value.items()}
            return NormalizedString(value)

        # Extract type from schema
        schema_type = schema.get('type', 'string')
        format_type = schema.get('format')

        # Format takes precedence
        type_name = format_type or schema_type

        if schema_type == 'array':
            return self._normalize_array(value, schema, strict, **kwargs)
        if schema_type == 'object':
            return self._normalize_object(value, schema, strict, **kwargs)
        return self._normalize_simple(value, type_name, strict, **kwargs)

    def _normalize_simple(self, value: Any, type_name: str, strict: bool, **kwargs: Any) -> NormalizedType:
        """Normalize a simple (non-container) value."""
        type_class = TYPE_REGISTRY.get(type_name, NormalizedString)
        try:
            return type_class(value, **kwargs)
        except Exception as e:
            if strict:
                raise ValueError(f'Failed to normalize {value!r} as {type_name}: {e}') from e
            logger.warning(f'Normalization failed for {value!r}: {e}')
            return NormalizedString(value)

    def _normalize_array(self, value: Any, schema: dict, strict: bool, **kwargs: Any) -> tuple:
        """Normalize an array value."""
        import json

        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)

        if not isinstance(value, (list, tuple)):
            if strict:
                raise ValueError(f'Expected array, got {type(value).__name__}')
            value = [value]

        items_schema = schema.get('items', {})
        normalized = []

        for item in value:
            normalized.append(self.normalize(item, items_schema, strict, **kwargs))

        return tuple(normalized)

    def _normalize_object(self, value: Any, schema: dict, strict: bool, **kwargs: Any) -> dict:
        """Normalize an object value."""
        import json

        if isinstance(value, str):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)

        if not isinstance(value, dict):
            if strict:
                raise ValueError(f'Expected object, got {type(value).__name__}')
            return {'value': self.normalize(value, None, strict, **kwargs)}

        properties = schema.get('properties', {})
        normalized = {}

        for key, val in value.items():
            prop_schema = properties.get(key)
            normalized[key] = self.normalize(val, prop_schema, strict, **kwargs)

        return normalized


class ValueComparator:
    """Recursive structural value comparator.

    Compares normalized values with support for:
    - Nested objects and arrays
    - Ordered/unordered array comparison
    - Alternative values
    - Ignored keys
    """

    def compare(
        self,
        actual: Any,
        expected: Any,
        ordered: bool = False,
        ignore_extra_keys: bool = False,
        ignored_keys: set | None = None,
    ) -> EvalResult:
        """Compare actual value against expected.

        Args:
            actual: Actual value from agent
            expected: Expected value (may contain alternatives)
            ordered: Whether array order matters
            ignore_extra_keys: Allow extra keys in actual objects
            ignored_keys: Keys to skip during object comparison

        Returns:
            EvalResult with success status and details
        """
        return self._compare_recursive(
            actual=actual,
            expected=expected,
            ordered=ordered,
            ignore_extra_keys=ignore_extra_keys,
            ignored_keys=ignored_keys or set(),
            path='value',
        )

    def _compare_recursive(
        self,
        actual: Any,
        expected: Any,
        ordered: bool,
        ignore_extra_keys: bool,
        ignored_keys: set,
        path: str,
    ) -> EvalResult:
        """Recursively compare values."""
        # Handle None
        if expected is None and actual is None:
            return EvalResult(True, f'{path}: both None')
        if expected is None or actual is None:
            return EvalResult(False, f'{path}: expected {expected}, got {actual}')

        # Handle NormalizedType
        if isinstance(expected, NormalizedType):
            if expected == actual:
                return EvalResult(True, f'{path}: match')
            return EvalResult(
                False,
                f'{path}: expected {expected.normalized}, got {actual}',
                {'expected': expected.normalized, 'actual': actual},
            )

        # Handle dict
        if isinstance(expected, dict):
            return self._compare_objects(actual, expected, ordered, ignore_extra_keys, ignored_keys, path)

        # Handle array
        if isinstance(expected, (list, tuple)):
            return self._compare_arrays(actual, expected, ordered, ignore_extra_keys, ignored_keys, path)

        # Direct comparison
        if actual == expected:
            return EvalResult(True, f'{path}: direct match')

        # Try regex match if expected looks like a regex pattern
        if (
            isinstance(expected, str)
            and isinstance(actual, str)
            and (expected.startswith('^') or expected.endswith('$'))
        ):
            try:
                if re.match(expected, actual, re.IGNORECASE):
                    return EvalResult(True, f'{path}: regex match')
            except re.error:
                pass  # Invalid regex, continue with other comparisons

        # Try normalized comparison
        try:
            if NormalizedString(actual) == NormalizedString(expected):
                return EvalResult(True, f'{path}: normalized string match')
        except Exception:
            pass

        return EvalResult(False, f'{path}: expected {expected}, got {actual}', {'expected': expected, 'actual': actual})

    def _compare_objects(
        self,
        actual: Any,
        expected: dict,
        ordered: bool,
        ignore_extra_keys: bool,
        ignored_keys: set,
        path: str,
    ) -> EvalResult:
        """Compare object/dict values."""
        if not isinstance(actual, dict):
            return EvalResult(False, f'{path}: expected object, got {type(actual).__name__}')

        # Check for missing keys
        expected_keys = set(expected.keys()) - ignored_keys
        actual_keys = set(actual.keys()) - ignored_keys

        missing_keys = expected_keys - actual_keys
        if missing_keys:
            return EvalResult(False, f'{path}: missing keys {missing_keys}', {'missing_keys': list(missing_keys)})

        # Check for extra keys (unless ignored)
        if not ignore_extra_keys:
            extra_keys = actual_keys - expected_keys
            if extra_keys:
                return EvalResult(False, f'{path}: extra keys {extra_keys}', {'extra_keys': list(extra_keys)})

        # Compare each key
        for key in expected_keys:
            result = self._compare_recursive(
                actual=actual.get(key),
                expected=expected[key],
                ordered=ordered,
                ignore_extra_keys=ignore_extra_keys,
                ignored_keys=ignored_keys,
                path=f'{path}.{key}',
            )
            if not result.success:
                return result

        return EvalResult(True, f'{path}: object match')

    def _compare_arrays(
        self,
        actual: Any,
        expected: tuple | list,
        ordered: bool,
        ignore_extra_keys: bool,
        ignored_keys: set,
        path: str,
    ) -> EvalResult:
        """Compare array/list values."""
        if not isinstance(actual, (list, tuple)):
            return EvalResult(False, f'{path}: expected array, got {type(actual).__name__}')

        if len(actual) != len(expected):
            return EvalResult(
                False,
                f'{path}: length mismatch (expected {len(expected)}, got {len(actual)})',
                {'expected_length': len(expected), 'actual_length': len(actual)},
            )

        if ordered:
            return self._compare_arrays_ordered(actual, expected, ordered, ignore_extra_keys, ignored_keys, path)
        return self._compare_arrays_unordered(actual, expected, ordered, ignore_extra_keys, ignored_keys, path)

    def _compare_arrays_ordered(
        self,
        actual: list | tuple,
        expected: list | tuple,
        ordered: bool,
        ignore_extra_keys: bool,
        ignored_keys: set,
        path: str,
    ) -> EvalResult:
        """Compare arrays in order."""
        for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
            result = self._compare_recursive(
                actual=a,
                expected=e,
                ordered=ordered,
                ignore_extra_keys=ignore_extra_keys,
                ignored_keys=ignored_keys,
                path=f'{path}[{i}]',
            )
            if not result.success:
                return result

        return EvalResult(True, f'{path}: ordered array match')

    def _compare_arrays_unordered(
        self,
        actual: list | tuple,
        expected: list | tuple,
        ordered: bool,
        ignore_extra_keys: bool,
        ignored_keys: set,
        path: str,
    ) -> EvalResult:
        """Compare arrays ignoring order (greedy matching)."""
        actual_matched = [False] * len(actual)

        for exp_idx, exp_val in enumerate(expected):
            matched = False
            for act_idx, act_val in enumerate(actual):
                if actual_matched[act_idx]:
                    continue

                result = self._compare_recursive(
                    actual=act_val,
                    expected=exp_val,
                    ordered=ordered,
                    ignore_extra_keys=ignore_extra_keys,
                    ignored_keys=ignored_keys,
                    path=f'{path}[{exp_idx}]',
                )

                if result.success:
                    actual_matched[act_idx] = True
                    matched = True
                    break

            if not matched:
                return EvalResult(
                    False, f'{path}: could not find match for expected[{exp_idx}]', {'unmatched_expected': exp_val}
                )

        return EvalResult(True, f'{path}: unordered array match')


# =============================================================================
# WebArena-Verified Evaluators
# =============================================================================


class AgentResponseEvaluator:
    """Evaluates agent's response data against expected values.

    Uses deterministic comparison instead of LLM-as-judge.
    Key features:
    - Type-aware normalization based on results_schema
    - JSON key order independent comparison
    - Unordered array comparison (unless ordered=True)
    - Tolerance-based comparison for numeric/time/distance values
    """

    def __init__(self):
        self.normalizer = ValueNormalizer()
        self.comparator = ValueComparator()

    def evaluate(
        self,
        agent_response: dict | str | None,
        eval_config: dict,
    ) -> EvalResult:
        """Evaluate agent response.

        Args:
            agent_response: Agent's response (dict with retrieved_data, or string)
            eval_config: Evaluation configuration with expected values and schema

        Returns:
            EvalResult with score and details
        """
        expected = eval_config.get('expected', {})
        expected_status = expected.get('status', 'SUCCESS')
        expected_data = expected.get('retrieved_data')
        results_schema = eval_config.get('results_schema')

        # Parse agent response - handle string that might be JSON
        agent_response = self._parse_agent_response(agent_response)

        if agent_response is None:
            agent_response = {}

        # Check task_type and status
        actual_status = agent_response.get('status', 'SUCCESS')

        # Handle NOT_FOUND_ERROR and other status checks
        if expected_status != actual_status:
            if expected_status in ('NOT_FOUND_ERROR', 'ACTION_NOT_ALLOWED_ERROR'):
                # Expected "not achievable" / "not found" / "not allowed" task
                # Check if agent response indicates nothing was found or task is not achievable
                if self._is_not_achievable_response(agent_response):
                    return EvalResult(True, 'Correctly identified as not achievable')
                # Empty array also means "not found"
                actual_data = self._extract_actual_data(agent_response)
                if actual_data == [] or actual_data == '[]' or actual_data is None:
                    return EvalResult(True, 'Empty result indicates not found/not allowed')
            # Don't fail just on status mismatch if we have data to compare
            if expected_data is None:
                return EvalResult(False, f'Status mismatch: expected {expected_status}, got {actual_status}')

        # If no expected data, just status match is enough
        if expected_data is None:
            return EvalResult(True, 'Status matched, no data verification required')

        # Get actual data - try multiple sources
        actual_data = self._extract_actual_data(agent_response)

        if actual_data is None:
            return EvalResult(False, 'No response data found', {'expected': expected_data, 'actual': None})

        # If expected is array and actual is string, try to parse text as array
        if isinstance(expected_data, list) and isinstance(actual_data, str):
            parsed_array = self._try_parse_text_as_array(actual_data, expected_data)
            if parsed_array is not None:
                actual_data = parsed_array
            elif len(expected_data) == 1:
                # Single-element expected array: wrap actual string as single-element list
                # so that regex patterns like ['^#?\\s*0*170$'] can match '000000170'
                actual_data = [actual_data]

        # Normalize values using schema
        try:
            normalized_expected = self.normalizer.normalize(expected_data, results_schema, strict=True)
            normalized_actual = self.normalizer.normalize(actual_data, results_schema, strict=False)
        except Exception as e:
            logger.warning(f'Normalization error: {e}, falling back to direct comparison')
            normalized_expected = expected_data
            normalized_actual = actual_data

        # Compare
        ordered = eval_config.get('ordered', False)
        result = self.comparator.compare(
            actual=normalized_actual,
            expected=normalized_expected,
            ordered=ordered,
            ignore_extra_keys=True,
        )

        # Add debug info
        if not result.success:
            result.details['expected_data'] = str(expected_data)[:200]
            result.details['actual_data'] = str(actual_data)[:200]

        return result

    def _parse_agent_response(self, response: dict | str | None) -> dict | None:
        """Parse agent response, handling JSON strings."""
        if response is None:
            return None

        if isinstance(response, dict):
            return response

        if isinstance(response, str):
            response = response.strip()

            # Try to parse as JSON
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    return parsed
                # If it's a list or other type, wrap it
                return {'retrieved_data': parsed}
            except json.JSONDecodeError:
                pass

            # It's a plain string response
            return {'retrieved_data': response}

        return {'retrieved_data': response}

    def _extract_actual_data(self, response: dict) -> Any:
        """Extract actual data from agent response, trying multiple fields."""
        # Try common fields
        for field in ['retrieved_data', 'data', 'result', 'answer', 'message', 'response']:
            if field in response and response[field] is not None:
                data = response[field]
                # Try to parse if it's a JSON string
                if isinstance(data, str):
                    data = data.strip()
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError:
                        pass
                return data

        # If response has other keys, maybe it IS the data
        non_meta_keys = [k for k in response if k not in ('status', 'task_type', 'error')]
        if non_meta_keys:
            # Return the response itself as data (without status/task_type)
            return {k: response[k] for k in non_meta_keys}

        return None

    def _try_parse_text_as_array(self, text: str, expected_array: list) -> list | None:
        """Try to parse text as array when expected is an array.

        Handles common text formats:
        - Comma-separated: "a, b, c"
        - Newline-separated: "a\\nb\\nc"
        - Arrow-separated: "a → b → c"
        - Numbered list: "1. a\\n2. b\\n3. c"
        """
        if not isinstance(text, str) or not isinstance(expected_array, list):
            return None

        text = text.strip()
        if not text:
            return None

        # Try JSON first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Detect delimiter by checking what appears in text

        # Try arrow separator (for routes/sequences)
        if ' → ' in text or ' -> ' in text:
            parts = re.split(r'\s*[→\->]+\s*', text)
            # Clean up parts - remove extra info in parentheses at the end
            cleaned = []
            for p in parts:
                # Remove trailing info like "(Total travel time: 5 hours)"
                p = re.sub(r'\s*\([^)]*\)\s*$', '', p).strip()
                if p:
                    cleaned.append(p)
            if len(cleaned) == len(expected_array):
                return cleaned

        # Try newline separator
        if '\n' in text:
            lines = text.strip().split('\n')
            # Clean up numbered list format
            cleaned = []
            for line in lines:
                line = line.strip()
                # Remove numbered prefix like "1. " or "1) "
                line = re.sub(r'^[\d]+[.)]\s*', '', line)
                if line:
                    cleaned.append(line)
            if len(cleaned) == len(expected_array):
                return cleaned

        # Try comma separator (most common)
        if ',' in text:
            parts = [p.strip() for p in text.split(',')]
            if len(parts) == len(expected_array):
                return parts

        return None

    def _is_not_achievable_response(self, response: dict | str) -> bool:
        """Check if response indicates task is not achievable."""
        if isinstance(response, str):
            text = response.lower()
        else:
            text = str(response.get('message', '')).lower()
            text += ' ' + str(response.get('retrieved_data', '')).lower()
            text += ' ' + str(response.get('error', '')).lower()

        not_achievable_patterns = [
            r'\bn/?a\b',
            r'not achievable',
            r'cannot be completed',
            r'impossible',
            r'not possible',
            r'unable to',
            r'no results',
            r'not found',
            r'does not exist',
            r'couldn\'t find',
            r'could not find',
        ]

        return any(re.search(pattern, text) for pattern in not_achievable_patterns)


class StringMatchEvaluator:
    """Classic string matching evaluator (for backward compatibility).

    Supports exact_match, must_include, and fuzzy patterns.
    """

    def __init__(self):
        self.normalizer = ValueNormalizer()

    def evaluate(
        self,
        answer: str,
        eval_config: dict,
    ) -> EvalResult:
        """Evaluate using string matching.

        Args:
            answer: Agent's answer string
            eval_config: Evaluation config with reference_answers

        Returns:
            EvalResult with score
        """
        reference_answers = eval_config.get('reference_answers', {})

        if not reference_answers:
            # Use new-style expected data
            expected = eval_config.get('expected', {})
            expected_data = expected.get('retrieved_data')
            if expected_data is not None:
                # Convert to must_include style
                if isinstance(expected_data, list):
                    reference_answers = {'must_include': [str(x) for x in expected_data]}
                else:
                    reference_answers = {'must_include': [str(expected_data)]}

        answer_clean = self._clean_answer(answer)

        for approach, value in reference_answers.items():
            if approach == 'exact_match':
                if self._clean_answer(value) != answer_clean:
                    return EvalResult(False, f'Exact match failed: expected {value}')

            elif approach == 'must_include':
                if not isinstance(value, list):
                    value = [value]
                for must_val in value:
                    if self._clean_answer(must_val) not in answer_clean:
                        return EvalResult(False, f"Must include '{must_val}' not found")

            elif approach == 'fuzzy_match':
                # For verified version, use deterministic string similarity
                if not isinstance(value, list):
                    value = [value]
                for ref in value:
                    if not self._fuzzy_match(answer_clean, self._clean_answer(ref)):
                        return EvalResult(False, f'Fuzzy match failed for {ref}')

        return EvalResult(True, 'All string checks passed')

    def _clean_answer(self, answer: Any) -> str:
        """Clean and normalize answer string."""
        answer = str(answer).strip()
        if (answer.startswith("'") and answer.endswith("'")) or (answer.startswith('"') and answer.endswith('"')):
            answer = answer[1:-1]
        return answer.lower()

    def _fuzzy_match(self, actual: str, expected: str) -> bool:
        """Deterministic fuzzy matching based on token overlap."""
        actual_tokens = set(re.findall(r'\b\w+\b', actual))
        expected_tokens = set(re.findall(r'\b\w+\b', expected))

        if not expected_tokens:
            return True

        # Check if most expected tokens are in actual
        overlap = len(actual_tokens & expected_tokens)
        threshold = 0.8 * len(expected_tokens)

        return overlap >= threshold


class UrlMatchEvaluator:
    """URL matching evaluator."""

    def evaluate(
        self,
        current_url: str,
        eval_config: dict,
    ) -> EvalResult:
        """Evaluate by checking current URL.

        Args:
            current_url: Current page URL
            eval_config: Config with reference_url

        Returns:
            EvalResult
        """
        from urllib.parse import parse_qs, urlparse

        reference_url = eval_config.get('reference_url', '')
        matching_rule = eval_config.get('url_note', 'GOLD in PRED')

        if not reference_url:
            return EvalResult(True, 'No URL verification required')

        def clean_url(url: str) -> str:
            return str(url).rstrip('/')

        pred = clean_url(current_url)
        ref_urls = [clean_url(u) for u in reference_url.split(' |OR| ')]

        if matching_rule == 'GOLD in PRED':
            for ref_url in ref_urls:
                ref_parsed = urlparse(ref_url)
                pred_parsed = urlparse(pred)

                # Check base path
                ref_path = (ref_parsed.netloc + ref_parsed.path).rstrip('/')
                pred_path = (pred_parsed.netloc + pred_parsed.path).rstrip('/')

                if ref_path in pred_path:
                    # Check query params
                    ref_query = parse_qs(ref_parsed.query)
                    pred_query = parse_qs(pred_parsed.query)

                    query_match = all(
                        any(pv in pred_query.get(k, []) for pv in ref_query.get(k, [])) for k in ref_query
                    )

                    if query_match:
                        return EvalResult(True, f'URL matched: {ref_url}')

        return EvalResult(False, f'URL mismatch: expected one of {ref_urls}, got {pred}')


class NetworkEventEvaluator:
    """Network event evaluator.

    Checks if the agent performed expected network operations during task execution.
    This is the primary evaluator for action-based tasks in WebArena-Verified.

    Supports:
    - URL matching (single or multiple with OR relationship)
    - HTTP method verification (GET/POST/PUT/DELETE/PATCH)
    - POST data verification with JSONPath support
    - Regex patterns (^...$)
    - Ignored query parameter patterns

    Example eval_config:
    {
        "evaluator": "NetworkEventEvaluator",
        "expected": {
            "http_method": "POST",
            "url": "__GITLAB__/primer/design/notes",
            "post_data": {"$.note.note": "Thanks, working on reviews"}
        }
    }
    """

    def evaluate(
        self,
        network_events: list,  # list[NetworkEvent] from task.py
        eval_config: dict,
    ) -> EvalResult:
        """Evaluate network events (visited URLs with method and POST data).

        Args:
            network_events: List of NetworkEvent objects with method, url, and post_data
            eval_config: Configuration with expected URLs, methods, and POST data patterns

        Returns:
            EvalResult with success status
        """
        expected = eval_config.get('expected', {})

        # Extract expected values
        expected_urls = expected.get('url', [])
        expected_method = expected.get('http_method', '').upper()
        expected_post_data = expected.get('post_data', {})
        expected_query_params = expected.get('query_params', {})  # NEW: query params validation

        # Normalize expected URLs to list
        if isinstance(expected_urls, str):
            expected_urls = [expected_urls]

        if not expected_urls:
            return EvalResult(True, 'No URL verification required')

        # Get ignored query params patterns
        ignored_patterns = eval_config.get('ignored_query_params_patterns') or []

        # Convert legacy format (list of URLs) to NetworkEvent-like objects if needed
        normalized_events = []
        for event in network_events:
            if hasattr(event, 'method'):
                normalized_events.append(event)
            elif isinstance(event, str):
                # Legacy: just a URL string
                class LegacyEvent:
                    def __init__(self, url: str):
                        self.method = 'GET'
                        self.url = url
                        self.post_data = None

                normalized_events.append(LegacyEvent(event))
            elif isinstance(event, dict):
                # Dict format
                class DictEvent:
                    def __init__(self, d: dict):
                        self.method = d.get('method', 'GET').upper()
                        self.url = d.get('url', '')
                        self.post_data = d.get('post_data')

                normalized_events.append(DictEvent(event))

        # Check if any network event matches the expected criteria
        for expected_url in expected_urls:
            match_result = self._find_matching_event(
                expected_url=expected_url,
                expected_method=expected_method,
                expected_post_data=expected_post_data,
                expected_query_params=expected_query_params,  # NEW
                network_events=normalized_events,
                ignored_patterns=ignored_patterns,
            )
            if match_result.success:
                return match_result

        # Build detailed failure message
        event_summary = [f'{e.method} {e.url}' for e in normalized_events[:10]]
        return EvalResult(
            False,
            'No network event matched expected criteria',
            {
                'expected_urls': expected_urls,
                'expected_method': expected_method or 'any',
                'expected_post_data': expected_post_data or 'none',
                'expected_query_params': expected_query_params or 'none',
                'actual_events': event_summary,
            },
        )

    def _find_matching_event(
        self,
        expected_url: str,
        expected_method: str,
        expected_post_data: dict,
        expected_query_params: dict,  # NEW: query params to validate
        network_events: list,
        ignored_patterns: list[str],
    ) -> EvalResult:
        """Find a network event matching all criteria."""
        for event in network_events:
            # Check URL match (path only, query params checked separately)
            if not self._url_matches(expected_url, event.url, ignored_patterns):
                continue

            # Check HTTP method if specified
            if expected_method and event.method != expected_method:
                continue

            # Check query params if specified
            if expected_query_params and not self._query_params_match(
                expected_query_params, event.url, ignored_patterns
            ):
                continue

            # Check POST data if specified
            if expected_post_data and not self._post_data_matches(expected_post_data, event.post_data):
                continue

            return EvalResult(
                True,
                f'Network event matched: {event.method} {event.url}',
                {'matched_event': f'{event.method} {event.url}'},
            )

        return EvalResult(False, 'No matching event found')

    def _query_params_match(
        self,
        expected_params: dict,
        actual_url: str,
        ignored_patterns: list[str] | None,
    ) -> bool:
        """Check if URL contains expected query parameters.

        Args:
            expected_params: Dict of expected query params, e.g. {"assignee_username": ["byteblaze"]}
                Values can be:
                - Plain strings: exact match (case-insensitive, whitespace-trimmed)
                - Regex patterns (^...$): regex match against actual value
                - Regex with empty alternative (e.g. ^(all|)$): param is optional
            actual_url: The actual URL to check
            ignored_patterns: Patterns for params to ignore

        Returns:
            True if all expected params are present with matching values

        Note:
            Values are compared after stripping whitespace to handle edge cases
            where ground truth may have leading/trailing spaces.
        """
        from urllib.parse import parse_qs, urlparse

        if not expected_params:
            return True

        if ignored_patterns is None:
            ignored_patterns = []

        actual_parsed = urlparse(actual_url)
        actual_query = parse_qs(actual_parsed.query)

        for key, expected_values in expected_params.items():
            # Skip ignored params
            if self._should_ignore_param(key, ignored_patterns):
                continue

            # Normalize expected values to list
            if isinstance(expected_values, str):
                expected_values = [expected_values]

            # Check if any expected value allows empty/missing (regex with empty alternative)
            allows_empty = False
            for ev in expected_values:
                if isinstance(ev, str) and ev.startswith('^') and ev.endswith('$'):
                    # It's a regex, check if it matches empty string
                    try:
                        if re.match(ev, ''):
                            allows_empty = True
                            break
                    except re.error:
                        pass

            # Check if key exists
            if key not in actual_query:
                # If the regex allows empty string, missing param is OK
                if allows_empty:
                    continue
                return False

            # Get actual values and strip whitespace
            actual_values = actual_query[key]
            actual_values_stripped = [v.strip() for v in actual_values]

            # Check if any expected value matches any actual value
            matched = False
            for ev in expected_values:
                ev_str = str(ev).strip()

                # Check if it's a regex pattern
                if ev_str.startswith('^') and ev_str.endswith('$'):
                    # Try regex match
                    try:
                        for av in actual_values_stripped:
                            if re.match(ev_str, av):
                                matched = True
                                break
                    except re.error:
                        # Invalid regex, fall back to exact match
                        if ev_str in actual_values_stripped:
                            matched = True
                else:
                    # Plain string comparison
                    if ev_str in actual_values_stripped:
                        matched = True

                if matched:
                    break

            if not matched:
                return False

        return True

    def _should_ignore_param(self, param: str, ignored_patterns: list[str]) -> bool:
        """Check if a parameter should be ignored based on patterns."""
        for pattern in ignored_patterns:
            try:
                if re.match(f'^{pattern}$', param):
                    return True
            except re.error:
                if pattern == param:
                    return True
        return False

    def _url_matches(
        self,
        expected_url: str,
        actual_url: str,
        ignored_patterns: list[str] | None,
    ) -> bool:
        """Check if expected URL matches actual URL."""
        from urllib.parse import parse_qs, urlparse

        # Ensure ignored_patterns is not None
        if ignored_patterns is None:
            ignored_patterns = []

        # Check if it's a regex pattern
        is_regex = expected_url.startswith('^') and expected_url.endswith('$')

        if is_regex:
            try:
                if re.match(expected_url, actual_url):
                    return True
            except re.error:
                pass
            return False

        # Handle placeholder URLs (e.g., __GITLAB__/dashboard/todos)
        placeholder_pattern = r'^__[A-Z_]+__'
        if re.match(placeholder_pattern, expected_url):
            # Extract path after placeholder
            placeholder_match = re.match(r'^__[A-Z_]+__(.*)', expected_url)
            if placeholder_match:
                expected_path = placeholder_match.group(1).rstrip('/')
                actual_parsed = urlparse(actual_url)
                actual_path = actual_parsed.path.rstrip('/')

                # Check if actual URL ends with the expected path
                if actual_path.endswith(expected_path) or expected_path in actual_path:
                    # Also check query params if expected has them
                    if '?' in expected_url:
                        expected_query_str = expected_url.split('?', 1)[1]
                        expected_query = parse_qs(expected_query_str)
                        actual_query = parse_qs(actual_parsed.query)
                        for key, _values in expected_query.items():
                            if key not in actual_query:
                                return False
                    return True
            return False

        expected_parsed = urlparse(expected_url)
        actual_parsed = urlparse(actual_url)

        # Compare scheme and netloc (case insensitive)
        if expected_parsed.scheme and expected_parsed.scheme.lower() != actual_parsed.scheme.lower():
            return False
        if expected_parsed.netloc and expected_parsed.netloc.lower() != actual_parsed.netloc.lower():
            return False

        # Compare paths (strip trailing slash)
        expected_path = expected_parsed.path.rstrip('/')
        actual_path = actual_parsed.path.rstrip('/')
        if expected_path != actual_path and expected_path not in actual_path:
            return False

        # If ignoring all query params, we're done
        if '.*' in ignored_patterns:
            return True

        # Compare query params (ignoring specified patterns)
        expected_query = parse_qs(expected_parsed.query)
        actual_query = parse_qs(actual_parsed.query)

        def should_ignore(param: str) -> bool:
            for pattern in ignored_patterns:
                try:
                    if re.match(f'^{pattern}$', param):
                        return True
                except re.error:
                    if pattern == param:
                        return True
            return False

        # Check that all expected query params are present in actual
        for key, values in expected_query.items():
            if should_ignore(key):
                continue
            if key not in actual_query:
                return False
            if not any(v in actual_query[key] for v in values):
                return False

        return True

    def _post_data_matches(
        self,
        expected_post_data: dict,
        actual_post_data: dict | str | None,
    ) -> bool:
        """Check if actual POST data matches expected patterns.

        Supports JSONPath patterns like:
        - {"$.note.note": "expected value"}
        - {"$.field": "value"}
        - {"field": "value"} (direct field match)

        Also supports:
        - JSON data
        - Multipart form data (boundary-based)
        - URL-encoded form data
        """
        if not expected_post_data:
            return True

        if actual_post_data is None:
            return False

        # Parse actual_post_data if it's a string
        if isinstance(actual_post_data, str):
            # Try JSON first
            try:
                actual_post_data = json.loads(actual_post_data)
            except json.JSONDecodeError:
                # Try to parse as multipart form data
                parsed_multipart = self._parse_multipart_form_data(actual_post_data)
                if parsed_multipart:
                    actual_post_data = parsed_multipart
                else:
                    # Try to parse as URL-encoded form data
                    parsed_urlencoded = self._parse_urlencoded_form_data(actual_post_data)
                    if parsed_urlencoded:
                        actual_post_data = parsed_urlencoded
                    else:
                        # Not parseable, check for direct string match
                        if len(expected_post_data) == 1:
                            expected_value = list(expected_post_data.values())[0]
                            return str(expected_value).lower() in actual_post_data.lower()
                        return False

        if not isinstance(actual_post_data, dict):
            return False

        # Check each expected field
        for key, expected_value in expected_post_data.items():
            actual_value = self._get_jsonpath_value(key, actual_post_data)

            if actual_value is None:
                return False

            # Compare values (case-insensitive string comparison)
            if not self._values_match(expected_value, actual_value):
                return False

        return True

    def _parse_multipart_form_data(self, data: str) -> dict | None:
        """Parse multipart form data string into a dictionary.

        Handles format like:
        ------WebKitFormBoundaryXXX
        Content-Disposition: form-data; name="field_name"

        field_value
        ------WebKitFormBoundaryXXX--

        Returns:
            Dict of field names to values, or None if not multipart
        """
        if not data:
            return None

        # Check if this looks like multipart form data
        # Common patterns: ------WebKitFormBoundary, --boundary, etc.
        boundary_patterns = [
            r'^----+WebKitFormBoundary\w+',
            r'^----+\w+',
            r'^--[\w-]+',
        ]

        boundary = None
        for pattern in boundary_patterns:
            match = re.match(pattern, data.strip(), re.MULTILINE)
            if match:
                boundary = match.group(0)
                break

        if not boundary:
            return None

        result = {}

        # Split by boundary
        parts = data.split(boundary)

        for part in parts:
            part = part.strip()
            if not part or part == '--':
                continue

            # Parse Content-Disposition header
            # Format: Content-Disposition: form-data; name="field_name"
            content_disp_match = re.search(
                r'Content-Disposition:\s*form-data;\s*name="([^"]+)"(?:;\s*filename="([^"]+)")?', part, re.IGNORECASE
            )

            if not content_disp_match:
                continue

            field_name = content_disp_match.group(1)
            # filename = content_disp_match.group(2)  # For file uploads, not used here

            # The value is after the headers (separated by double newline)
            # Skip headers and get the value
            header_end = part.find('\r\n\r\n')
            if header_end == -1:
                header_end = part.find('\n\n')

            if header_end != -1:
                value = part[header_end:].strip()
                # Remove trailing boundary marker if present
                if value.endswith('--'):
                    value = value[:-2].strip()
                result[field_name] = value

        return result if result else None

    def _parse_urlencoded_form_data(self, data: str) -> dict | None:
        """Parse URL-encoded form data string into a dictionary.

        Handles format like: field1=value1&field2=value2

        Returns:
            Dict of field names to values, or None if not URL-encoded
        """
        from urllib.parse import parse_qs

        if not data:
            return None

        # Check if this looks like URL-encoded data
        # Should contain = and possibly &, no newlines in field=value pairs
        if '=' not in data:
            return None

        # Skip if it looks like JSON
        stripped = data.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            return None

        try:
            # parse_qs returns lists, we want single values
            parsed = parse_qs(data, keep_blank_values=True)
            result = {}
            for key, values in parsed.items():
                # Take the first value if multiple
                result[key] = values[0] if values else ''
            return result if result else None
        except Exception:
            return None

    def _get_jsonpath_value(self, path: str, data: dict) -> Any:
        """Extract value from data using JSONPath-like syntax.

        Supports:
        - $.field.subfield - nested field access
        - field - direct field access (no $ prefix)
        """
        if not path:
            return data

        # Remove leading $. if present
        if path.startswith('$.'):
            path = path[2:]
        elif path.startswith('$'):
            path = path[1:]

        # Split by . and navigate
        parts = path.split('.')
        current = data

        for part in parts:
            if not part:
                continue

            # Handle array index like [0]
            array_match = re.match(r'^(\w+)\[(\d+)\]$', part)
            if array_match:
                field_name = array_match.group(1)
                index = int(array_match.group(2))
                if not isinstance(current, dict) or field_name not in current:
                    return None
                current = current[field_name]
                if not isinstance(current, (list, tuple)) or index >= len(current):
                    return None
                current = current[index]
            else:
                if not isinstance(current, dict) or part not in current:
                    return None
                current = current[part]

        return current

    def _values_match(self, expected: Any, actual: Any) -> bool:
        """Compare two values for matching.

        Handles:
        - Direct equality
        - Regex pattern matching (patterns starting with ^ or ending with $)
        - Case-insensitive string comparison
        - HTML content comparison (strips tags)
        - Numeric comparison
        - Substring matching
        """
        # Direct equality
        if expected == actual:
            return True

        # String comparison (case-insensitive)
        if isinstance(expected, str) and isinstance(actual, str):
            expected_lower = expected.lower()
            actual_lower = actual.lower()

            if expected_lower == actual_lower:
                return True

            # Check if expected is a regex pattern
            # Patterns typically start with ^ or end with $ or contain .*
            if expected.startswith('^') or expected.endswith('$') or '.*' in expected:
                try:
                    if re.match(expected, actual, re.IGNORECASE):
                        return True
                    # Also try matching against the original case
                    if re.match(expected, actual):
                        return True
                except re.error:
                    # Invalid regex, continue with other matching methods
                    pass

            # HTML content comparison - strip tags and compare
            expected_text = self._strip_html_tags(expected).lower()
            actual_text = self._strip_html_tags(actual).lower()

            if expected_text and actual_text:
                # Compare without HTML tags
                if expected_text == actual_text:
                    return True
                # Also try stripped version (handles whitespace differences)
                if expected_text.strip() == actual_text.strip():
                    return True

            # Check if expected is contained in actual
            if expected_lower in actual_lower:
                return True

            # Check if stripped expected is in stripped actual
            if expected_text.strip() in actual_text.strip():
                return True

        # Numeric comparison
        try:
            if float(expected) == float(actual):
                return True
        except (TypeError, ValueError):
            pass

        # String representation comparison
        return str(expected).lower() == str(actual).lower()

    def _strip_html_tags(self, text: str) -> str:
        """Strip HTML tags from text.

        Args:
            text: Text that may contain HTML tags

        Returns:
            Text with HTML tags removed
        """
        if not isinstance(text, str):
            return str(text)

        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        return re.sub(r'\s+', ' ', clean)


# =============================================================================
# Main evaluation function
# =============================================================================


async def evaluate_task_deterministic(
    task_config: dict,
    agent_answer: str | dict | None,
    page: Any = None,
    network_events: list | None = None,
    visited_urls: list[str] | None = None,  # Legacy parameter, kept for backward compatibility
) -> tuple[float, dict]:
    """Evaluate task using deterministic methods.

    Args:
        task_config: Task configuration with eval settings
        agent_answer: Agent's answer (string or structured response)
        page: Playwright page object (for HTML evaluation)
        network_events: List of NetworkEvent objects with method, url, and post_data
        visited_urls: Legacy parameter - list of URLs (deprecated, use network_events)

    Returns:
        Tuple of (score, details)
    """
    eval_config = task_config.get('eval', {})

    # Handle string-encoded JSON (common in webarena-verified dataset)
    if isinstance(eval_config, str):
        try:
            eval_config = json.loads(eval_config)
        except json.JSONDecodeError:
            eval_config = {}

    # Get evaluator list - prefer _original_evaluators (from generator.py conversion)
    # Fall back to treating eval_config as the list (for direct webarena-verified format)
    if isinstance(eval_config, dict) and '_original_evaluators' in eval_config:
        eval_list = eval_config['_original_evaluators']
    elif isinstance(eval_config, list):
        eval_list = eval_config
    else:
        eval_list = [eval_config]

    total_score = 1.0
    details = {'evaluators': []}

    # Convert legacy visited_urls to network_events if needed
    if network_events is None and visited_urls:
        # Create simple NetworkEvent-like objects from URLs
        class LegacyEvent:
            def __init__(self, url: str):
                self.method = 'GET'
                self.url = url
                self.post_data = None

        network_events = [LegacyEvent(url) for url in visited_urls]

    for eval_item in eval_list:
        evaluator_type = eval_item.get('evaluator', 'AgentResponseEvaluator')

        if evaluator_type == 'AgentResponseEvaluator':
            evaluator = AgentResponseEvaluator()
            # Convert agent_answer to expected format
            if isinstance(agent_answer, str):
                response = {'retrieved_data': agent_answer, 'status': 'SUCCESS'}
            else:
                response = agent_answer or {}
            result = evaluator.evaluate(response, eval_item)

        elif evaluator_type == 'StringMatchEvaluator':
            evaluator = StringMatchEvaluator()
            result = evaluator.evaluate(str(agent_answer or ''), eval_item)

        elif evaluator_type == 'URLMatchEvaluator':
            evaluator = UrlMatchEvaluator()
            current_url = page.url if page else ''
            result = evaluator.evaluate(current_url, eval_item)

        elif evaluator_type == 'NetworkEventEvaluator':
            evaluator = NetworkEventEvaluator()
            # Use provided network_events
            events = network_events or []

            # If no events from HAR, try to get current URL from page
            if page and hasattr(page, 'url') and not events:

                class PageEvent:
                    def __init__(self, url: str):
                        self.method = 'GET'
                        self.url = url
                        self.post_data = None

                events = [PageEvent(page.url)]

            # If still no events, try to extract from agent_answer (agent might have returned the URL)
            if not events and agent_answer:
                answer_str = str(agent_answer)
                # Look for URLs in the answer
                url_pattern = r'https?://[^\s<>"]+|http://[^\s<>"]+'
                found_urls = re.findall(url_pattern, answer_str)
                if found_urls:

                    class AnswerEvent:
                        def __init__(self, url: str):
                            self.method = 'GET'
                            self.url = url
                            self.post_data = None

                    events = [AnswerEvent(url) for url in found_urls]

            # NetworkEventEvaluator requires network events to properly evaluate
            # If no events are available, we cannot verify network events - this should fail
            # unless the task doesn't actually require URL/method/post_data verification
            if not events:
                # Check if this task actually requires verification
                expected = eval_item.get('expected', {})
                expected_urls = expected.get('url', [])
                expected_method = expected.get('http_method', '')
                expected_post_data = expected.get('post_data', {})

                if expected_urls or expected_method or expected_post_data:
                    # Task requires verification but we have no HAR data
                    result = EvalResult(
                        False,
                        'NetworkEventEvaluator failed: no network events available (HAR recording may not be enabled)',
                        {
                            'expected_urls': expected_urls,
                            'expected_method': expected_method,
                            'expected_post_data': expected_post_data,
                            'actual_events': [],
                        },
                    )
                else:
                    # No verification required
                    result = EvalResult(True, 'NetworkEventEvaluator: no verification required')
            else:
                result = evaluator.evaluate(events, eval_item)

        else:
            # Fallback to string match
            evaluator = StringMatchEvaluator()
            result = evaluator.evaluate(str(agent_answer or ''), eval_item)

        score = 1.0 if result.success else 0.0
        total_score *= score

        details['evaluators'].append({
            'type': evaluator_type,
            'success': result.success,
            'message': result.message,
        })

    return total_score, details


# Legacy evaluation functions for backward compatibility
async def eval_string_match(config: dict, answer: str) -> float:
    """Evaluate using string matching (legacy interface)."""
    evaluator = StringMatchEvaluator()
    result = evaluator.evaluate(answer, config.get('eval', {}))
    return 1.0 if result.success else 0.0


async def eval_url_match(config: dict, current_url: str) -> float:
    """Evaluate by checking URL (legacy interface)."""
    evaluator = UrlMatchEvaluator()
    result = evaluator.evaluate(current_url, config.get('eval', {}))
    return 1.0 if result.success else 0.0
