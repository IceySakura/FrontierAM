"""Encode JSON in bounded chunks, including individual large string values."""

from __future__ import annotations

import json

from collections.abc import Callable, Iterator, Mapping
from typing import Any


def iter_json_bytes(
    value: Any,
    *,
    ensure_ascii: bool = False,
    sort_keys: bool = True,
    default: Callable[[Any], Any] | None = None,
    allow_nan: bool = True,
    separators: tuple[str, str] = (',', ':'),
) -> Iterator[bytes]:
    active: set[int] = set()
    quote = json.encoder.encode_basestring_ascii if ensure_ascii else json.encoder.encode_basestring

    def encode(item: Any) -> Iterator[bytes]:
        if isinstance(item, str):
            # backslashreplace: lone surrogates (truncated multi-byte input) would
            # otherwise raise; they encode to \udXXX, a valid JSON escape that
            # json.loads round-trips to the same string.
            if len(item) <= 16 * 1024:
                yield quote(item).encode('utf-8', 'backslashreplace')
                return
            yield b'"'
            for start in range(0, len(item), 16 * 1024):
                yield quote(item[start : start + 16 * 1024])[1:-1].encode('utf-8', 'backslashreplace')
            yield b'"'
        elif isinstance(item, (Mapping, list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError('Circular reference detected')
            active.add(identity)
            try:
                mapping = isinstance(item, Mapping)
                yield b'{' if mapping else b'['
                values = sorted(item.items()) if mapping and sort_keys else item.items() if mapping else enumerate(item)
                first = True
                for key, child in values:
                    if not first:
                        yield separators[0].encode('ascii')
                    first = False
                    if mapping:
                        if not isinstance(key, str):
                            if key is None or isinstance(key, (bool, int, float)):
                                key = json.dumps(key, allow_nan=allow_nan)
                            else:
                                raise TypeError('JSON object keys must be strings or JSON scalars')
                        yield from encode(key)
                        yield separators[1].encode('ascii')
                    yield from encode(child)
                yield b'}' if mapping else b']'
            finally:
                active.remove(identity)
        elif item is None:
            yield b'null'
        elif isinstance(item, bool):
            yield b'true' if item else b'false'
        elif isinstance(item, int):
            yield int.__repr__(item).encode('ascii')
        elif isinstance(item, float):
            yield json.dumps(item, allow_nan=allow_nan).encode('ascii')
        elif default is not None:
            yield from encode(default(item))
        else:
            raise TypeError(f'Object of type {type(item).__name__} is not JSON serializable')

    yield from encode(value)


def bounded_json_text(value: Any, limit: int) -> tuple[str | None, int]:
    """Count the entire UTF-8 encoding, retaining at most ``limit`` bytes."""
    parts: list[bytes] = []
    size = 0
    for part in iter_json_bytes(value, ensure_ascii=True, sort_keys=False, default=str):
        size += len(part)
        if size <= limit:
            parts.append(part)
        elif parts:
            parts.clear()
    return (b''.join(parts).decode('utf-8') if size <= limit else None), size
