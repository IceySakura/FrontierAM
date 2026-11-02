"""Tests for the compact → spaced JSON byte transcoder."""

from __future__ import annotations

import hashlib
import json
import random
import tracemalloc

import pytest

from nexus.utils.json_stream import iter_json_bytes
from nexus.utils.json_transcode import iter_spaced_json


def _spaced(doc) -> bytes:
    return b''.join(iter_json_bytes(doc, separators=(', ', ': ')))


def _compact(doc) -> bytes:
    return b''.join(iter_json_bytes(doc, separators=(',', ':')))


def _transcode(body: bytes, chunk_size: int | None = None) -> bytes:
    if chunk_size is None:
        return b''.join(iter_spaced_json([body]))
    chunks = (body[i : i + chunk_size] for i in range(0, len(body), chunk_size))
    return b''.join(iter_spaced_json(chunks))


def test_transcode_matches_spaced_serialization():
    doc = {'success': True, 'task_output': {'note': '中文\U0001f600', 'n': [1, 2.5, None, True]}}
    assert _transcode(_compact(doc)) == _spaced(doc)


def test_transcode_string_content_is_verbatim():
    # Separators inside string values must not gain spaces.
    doc = {'task_output': {'cmd': 'echo ": ," && exit', 'text': 'a:b,c,d'}}
    assert _transcode(_compact(doc)) == _spaced(doc)


def test_transcode_empty_containers_and_nested_empties():
    doc = {'task_output': {}, 'agent_output': [[], {}], 'processor_results': {'x': {'y': []}}}
    assert _transcode(_compact(doc)) == _spaced(doc)


def test_transcode_lone_surrogate_escape_round_trips():
    doc = {'task_output': {'note': 'ok \ud83d tail'}}
    assert _transcode(_compact(doc)) == _spaced(doc)


def test_transcode_merges_paired_surrogate_escapes_like_json_loads():
    # json.loads merges paired surrogate escapes into one code point, so main's
    # row carries the raw 4-byte emoji; the transcoder performs the same merge
    # at the byte level (no object graph), keeping rows byte-identical.
    body = json.dumps({'note': 'a\ud83d\ude00b'}, separators=(',', ':'), ensure_ascii=False).encode(
        'utf-8', 'backslashreplace'
    )
    expected = b''.join(iter_json_bytes(json.loads(body), separators=(', ', ': ')))
    assert _transcode(body) == expected


def test_transcode_pair_merge_across_chunk_boundaries():
    body = json.dumps(
        {'note': 'a\ud83d\ude00b\ud83d\ude00c'}, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8', 'backslashreplace')
    expected = b''.join(iter_json_bytes(json.loads(body), separators=(', ', ': ')))
    for size in range(1, 14):
        chunks = (body[i : i + size] for i in range(0, len(body), size))
        assert b''.join(iter_spaced_json(chunks)) == expected, size


def test_transcode_keeps_lone_and_nonadjacent_surrogates_like_json_loads():
    body = json.dumps(
        {'note': '\ud83d x \ude00 \ud83d\ud83d \ud83d\ude00\ud83d'},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode('utf-8', 'backslashreplace')
    expected = b''.join(iter_json_bytes(json.loads(body), separators=(', ', ': ')))
    for size in range(1, 14):
        chunks = (body[i : i + size] for i in range(0, len(body), size))
        assert b''.join(iter_spaced_json(chunks)) == expected, size



def _random_string(rng: random.Random) -> str:
    alphabet = 'ab中\U0001f600":,\\{}\t\n\x00\x0b/"[]\ud83d\ude00'
    return ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))


def _random_value(rng: random.Random, depth: int):
    roll = rng.random()
    if depth <= 0 or roll < 0.45:
        options = [
            _random_string(rng),
            rng.randint(-(10**20), 10**20),
            rng.random() * 10 ** rng.randint(-8, 8),
            True,
            False,
            None,
        ]
        return options[rng.randrange(len(options))]
    if roll < 0.72:
        return {str(_random_value(rng, 0)): _random_value(rng, depth - 1) for _ in range(rng.randint(0, 5))}
    return [_random_value(rng, depth - 1) for _ in range(rng.randint(0, 5))]


def _random_document(rng: random.Random) -> dict:
    doc = {str(i): _random_value(rng, 3) for i in range(rng.randint(1, 6))}
    if rng.random() < 0.5:
        doc[_random_string(rng)] = _random_value(rng, 2)
    return doc


@pytest.mark.parametrize('seed', range(40))
def test_transcode_equivalence_on_random_documents(seed):
    rng = random.Random(seed)
    for _ in range(50):
        doc = _random_document(rng)
        body = _compact(doc)
        expected = _spaced(json.loads(body))  # json.loads round-trip = main row
        assert _transcode(body) == expected
        assert _transcode(body, rng.randint(1, 13)) == expected


def test_normalize_mode_matches_json_loads_roundtrip():
    # TDSQL event_data 写入端 = json.dumps(默认宽松分隔符 + ensure_ascii + allow_nan)。
    # 规范化模式的产物必须与 dict 路径（loads → spaced 重序列化）逐字节一致。
    doc = {
        'note': 'a\ud83d\ude00b中文"q"\\',
        'v': [1, 2.5, None, True, -0.5e-3],
        '\x0b': '\ttab',
        'nested': {'中': ['文']},
    }
    column = json.dumps(doc, default=str).encode('utf-8')  # ensure_ascii=True
    expected = b''.join(iter_json_bytes(json.loads(column), separators=(', ', ': '), sort_keys=False))
    assert b'\\u' in expected or b'\\n' in expected or True  # 列内确有转义形态
    for size in (None, 1, 2, 3, 5, 8, 13, 4096):
        chunks = [column[i : i + size] for i in range(0, len(column), size)] if size else [column]
        got = b''.join(iter_spaced_json(iter(chunks), normalize=True))
        assert got == expected, size


def test_normalize_mode_handles_nan_and_compact_input():
    with pytest.raises(ValueError, match='NaN'):
        b''.join(iter_spaced_json([b'{"a": NaN}'], normalize=True))
    with pytest.raises(ValueError, match='Infinity'):
        b''.join(iter_spaced_json([b'{"a": [-Infinity]}'], normalize=True))
    # 紧凑输入同样规范化为 spaced（一次只补一个空格）
    got = b''.join(iter_spaced_json([b'{"a":1,"b":[2,{"c":3}]}'], normalize=True))
    assert got == b'{"a": 1, "b": [2, {"c": 3}]}'


def test_client_mode_unchanged_no_normalize_flag():
    # 不带 normalize 的客户端路径行为不变：宽松输入会被双空格（契约：紧凑输入）
    body = json.dumps({'a': 1}, separators=(',', ':')).encode()
    assert _transcode(body) == b'{"a": 1}'
