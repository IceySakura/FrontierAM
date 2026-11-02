"""JSON extraction helpers for model responses."""

import json as json_lib
import re

from typing import TypeVar, cast


JsonValue = dict[str, object] | list[object] | str | int | float | bool | None
JsonObject = dict[str, object]
JsonArray = list[object]

_T = TypeVar('_T', JsonObject, JsonArray)


class JSONExtractionError(ValueError):
    """Raised when no valid JSON value can be extracted from text."""


def extract_json(text: str) -> JsonValue:
    """Extract a JSON value from direct text, code fences, or embedded content."""
    value = _extract_json_value(text)
    return cast(JsonValue, value)


def extract_json_object(text: str) -> JsonObject:
    """Extract a JSON object from model output."""
    return _extract_typed_json(text, dict, 'object')


def extract_json_array(text: str) -> JsonArray:
    """Extract a JSON array from model output."""
    return _extract_typed_json(text, list, 'array')


def _extract_typed_json(text: str, expected_type: type[_T], label: str) -> _T:
    value = _extract_json_value(text, expected_type)
    if isinstance(value, expected_type):
        return value
    raise JSONExtractionError(f'Extracted JSON is not a {label}: {type(value).__name__}')


def _extract_json_value(text: str, expected: type | None = None) -> object:
    """The first candidate that parses, and that is ``expected`` when one is named.

    Text around a JSON value can hold something that parses as JSON itself -- a
    ``byte[]`` in prose is a valid empty array -- so a caller that knows the shape
    it needs passes it, and a candidate of another shape is skipped rather than
    returned.
    """
    text = text.strip()
    if not text:
        raise JSONExtractionError('Empty response')

    wrong_shape: object = _NO_VALUE

    def take(parsed: object) -> bool:
        nonlocal wrong_shape
        if parsed is _NO_VALUE:
            return False
        if expected is not None and not isinstance(parsed, expected):
            wrong_shape = parsed
            return False
        return True

    parsed = _try_parse(text)
    if take(parsed):
        return parsed

    for block in reversed(_extract_code_blocks(text)):
        parsed = _try_parse(block.strip())
        if take(parsed):
            return parsed

    for candidate in _balanced_json_candidates(text):
        parsed = _try_parse(candidate)
        if take(parsed):
            return parsed

    if wrong_shape is not _NO_VALUE:
        return wrong_shape
    raise JSONExtractionError(f'No valid JSON found in response ({len(text)} chars)')


class _NoValue:
    pass


_NO_VALUE = _NoValue()


def _try_parse(text: str) -> object:
    try:
        return json_lib.loads(text)
    except json_lib.JSONDecodeError:
        return _NO_VALUE


def _extract_code_blocks(text: str) -> list[str]:
    return re.findall(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL | re.IGNORECASE)


def _balanced_json_candidates(text: str) -> list[str]:
    starts = [i for i, char in enumerate(text) if char in '{[']
    candidates: list[str] = []

    for start in reversed(starts):
        opener = text[start]
        closer = '}' if opener == '{' else ']'
        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]

            if escape:
                escape = False
                continue
            if char == '\\' and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break

    return candidates
