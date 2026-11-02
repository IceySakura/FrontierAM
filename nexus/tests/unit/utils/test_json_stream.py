import json

import pytest

from nexus.utils.json_stream import bounded_json_text, iter_json_bytes


@pytest.mark.parametrize('ensure_ascii', [True, False])
def test_stream_json_matches_stdlib_for_large_strings_and_nested_values(ensure_ascii):
    value = {'b': [None, True, 3.5, '中文🙂\n\\"' * 10000], 'a': {'empty': []}}
    chunks = list(iter_json_bytes(value, ensure_ascii=ensure_ascii))
    assert max(map(len, chunks)) < 256 * 1024
    assert b''.join(chunks) == json.dumps(
        value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')


def test_bounded_json_counts_utf8_bytes_after_limit_without_retaining_output():
    value = {'text': '截图' * 10000}
    text, size = bounded_json_text(value, limit=100)
    assert text is None
    assert size == len(b''.join(iter_json_bytes(value, ensure_ascii=True, sort_keys=False)))
    text, same_size = bounded_json_text(value, limit=size)
    assert same_size == size
    assert json.loads(text) == value


def test_stream_json_rejects_circular_references():
    value = []
    value.append(value)
    with pytest.raises(ValueError, match='Circular reference'):
        list(iter_json_bytes(value))


def test_stream_json_preserves_stdlib_scalar_key_compatibility():
    value = {None: 'none', 3: 'integer', False: 'false', 1.5: 'float'}
    assert b''.join(iter_json_bytes(value, sort_keys=False)) == json.dumps(value, separators=(',', ':')).encode('utf-8')


@pytest.mark.parametrize('ensure_ascii', [True, False])
@pytest.mark.parametrize('sort_keys', [True, False])
@pytest.mark.parametrize('separators', [(',', ':'), (', ', ': ')])
def test_stream_json_fast_scalars_preserve_stdlib_format(ensure_ascii, sort_keys, separators):
    from enum import IntEnum

    class Number(IntEnum):
        ONE = 1

    class CustomInt(int):
        def __repr__(self):
            return 'not JSON'

    value = {
        'z': [None, True, False, Number.ONE, CustomInt(7), -(10**200), -0.0, 1e200],
        'a': ['\x00\t\n\\"中文🙂', '', 'x' * (16 * 1024), 'x' * (16 * 1024 + 1)],
        'nonfinite': [float('nan'), float('inf'), -float('inf')],
    }
    options = {'ensure_ascii': ensure_ascii, 'sort_keys': sort_keys, 'separators': separators}
    assert b''.join(iter_json_bytes(value, **options)) == json.dumps(value, **options).encode('utf-8')
    with pytest.raises(ValueError, match='Out of range float'):
        list(iter_json_bytes(value, allow_nan=False, **options))


def test_object_members_parses_members_from_spans():
    from io import BytesIO

    from nexus.utils.json_scan import object_members

    body = json.dumps(
        {'success': True, 'count': 3, 'note': None, 'big': {'items': ['中文\U0001f600' * 1000]}, 'ratio': 0.5},
        ensure_ascii=False,
    ).encode()
    members = object_members(BytesIO(body))
    assert members == {
        'success': True,
        'count': 3,
        'note': None,
        'big': {'items': ['中文\U0001f600' * 1000]},
        'ratio': 0.5,
    }


def test_object_members_rejects_non_object_and_invalid_json():
    from io import BytesIO

    from nexus.utils.json_scan import object_members

    with pytest.raises(ValueError, match='must be an object'):
        object_members(BytesIO(b'[1, 2]'))
    with pytest.raises(ValueError, match='truncated'):
        object_members(BytesIO(b'{"success": '))


def test_object_members_falls_back_to_full_parse_on_size_limits():
    from io import BytesIO

    from nexus.utils.json_scan import object_members

    deep = {'success': True, 'nested': json.loads('[' * 40 + ']' * 40)}
    assert object_members(BytesIO(json.dumps(deep).encode()), max_depth=10) == deep

    wide = {'success': True, 'note': 'x' * 10_000}
    assert object_members(BytesIO(json.dumps(wide).encode()), metadata_limit=64) == wide
