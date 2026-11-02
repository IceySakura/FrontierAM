"""Tests for the JsonFieldScanner extensions used by the streaming transform."""

from __future__ import annotations

import json

import pytest

from nexus.utils.json_scan import JsonFallbackNeeded, JsonFieldScanner


def _scan(body: bytes, **kwargs) -> JsonFieldScanner:
    scanner = JsonFieldScanner(**kwargs)
    scanner.feed(body)
    scanner.finish()
    return scanner


def test_reject_non_finite_raises_when_enabled_and_passes_when_disabled():
    body = b'{"success": true, "score": NaN}'
    with pytest.raises(ValueError, match='NaN or Infinity'):
        _scan(body, reject_non_finite=True)
    _scan(body)  # default tolerates (legacy behaviour)


def test_empty_members_tracks_empty_containers_and_strings():
    scanner = _scan(b'{"a": {}, "b": [], "c": "", "d": [1], "e": "x", "f": null}')
    assert scanner.empty_members == {'a', 'b', 'c'}


def test_flat_object_fast_path_reports_emptiness():
    scanner = _scan(b'{"task_output": {}, "agent_output": {"k": 1}}')
    assert scanner.empty_members == {'task_output'}


def test_root_string_length_only_keeps_short_values_and_truncates_long_ones():
    short = _scan(b'{"error": null, "msg": "boom", "big": "' + b'x' * 5000 + b'"}', root_string_length_only=True)
    assert short.scalars['error'] is None
    assert short.scalars['msg'] == 'boom'
    assert 'big' not in short.scalars  # oversized: value not retained
    assert 'big' not in short.empty_members  # but it is non-empty
    assert short.spans['big'][1] > short.spans['big'][0]




def test_duplicate_and_limit_errors_are_fallback_class():
    with pytest.raises(JsonFallbackNeeded, match='duplicate'):
        _scan(b'{"a": 1, "a": 2}')
    with pytest.raises(JsonFallbackNeeded, match='nesting exceeds limit'):
        _scan(b'{"a": ' + b'[' * 40 + b']' * 40 + b'}', max_depth=10)
    # Oversized root numbers no longer fall back (they can never be null);
    # only oversized root KEYS keep the fallback (member identity needs them).


def test_oversized_root_number_no_longer_falls_back():
    body = b'{"count": ' + b'9' * 5000 + b', "note": "ok"}'
    scanner = _scan(body, reject_non_finite=True, root_string_length_only=True)
    assert 'count' in scanner.spans
    # 数字永不为 null：缺失即非 null，语义与截断字符串一致
    assert 'count' not in scanner.scalars
    assert scanner.scalars['note'] == 'ok'


def test_oversized_root_key_still_falls_back():
    big_key = 'k' * 5000
    body = json.dumps({big_key: 1}).encode()
    with pytest.raises(JsonFallbackNeeded, match='exceeds limit'):
        _scan(body, root_string_length_only=True)


def test_array_element_spans_string_aware_and_chunk_safe():
    from nexus.utils.json_scan import iter_array_element_spans

    body = b'[{"name": "run.event", "attributes": {"a": "]x\\"[}"}}, {"name": "other"}, 42, null, true]'
    for size in (None, 1, 3, 7, 100):
        chunks = [body] if size is None else [body[i : i + size] for i in range(0, len(body), size)]
        spans = list(iter_array_element_spans(iter(chunks), 0))
        elems = [body[a:b] for a, b in spans]
        assert len(spans) == 5
        assert elems[0] == b'{"name": "run.event", "attributes": {"a": "]x\\"[}"}}'
        assert elems[1:] == [b'{"name": "other"}', b'42', b'null', b'true']


def test_array_element_spans_empty_array():
    from nexus.utils.json_scan import iter_array_element_spans

    assert list(iter_array_element_spans(iter([b'[]']), 0)) == []
