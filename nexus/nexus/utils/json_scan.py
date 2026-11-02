"""Validate JSON incrementally without materializing strings or containers.

Top-level field spans can be copied verbatim, including arbitrarily large strings.
Only top-level keys and scalar metadata are retained, with explicit size bounds.
"""

from __future__ import annotations

import codecs
import json
import re

from collections.abc import Iterable, Iterator
from typing import Any, BinaryIO, cast


_WS = b' \t\n\r'


_STRING_SPECIAL = re.compile(rb'["\\\x00-\x1f]')
_SPACE = re.compile(rb'[^ \t\r\n]')
_PRIMITIVE_END = re.compile(rb'[ \t\r\n,\]}]')
_NUMBER = re.compile(rb'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z')

# Common flat objects can be validated by the C regex engine. The 4 KiB match
# window bounds regex backtracking and excludes large strings/nested containers.
_SIMPLE_STRING = rb'"[^"\\\x00-\x1f]*"'
_SIMPLE_NUMBER = rb'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
_SIMPLE_VALUE = rb'(?:' + _SIMPLE_STRING + rb'|' + _SIMPLE_NUMBER + rb'|true|false|null|NaN|-?Infinity)'
_SIMPLE_MEMBER = _SIMPLE_STRING + rb'[ \t\r\n]*:[ \t\r\n]*' + _SIMPLE_VALUE
_FLAT_OBJECT = re.compile(
    rb'\{[ \t\r\n]*(?:' + _SIMPLE_MEMBER + rb'(?:[ \t\r\n]*,[ \t\r\n]*' + _SIMPLE_MEMBER + rb')*)?[ \t\r\n]*\}'
)
# Same shape, but NaN/Infinity are not accepted as values: lets the flat fast
# path stay enabled when the caller rejects non-finite tokens.
_SIMPLE_MEMBER_STRICT = _SIMPLE_STRING + rb'[ \t\r\n]*:[ \t\r\n]*' + _SIMPLE_VALUE.replace(rb'|NaN|-?Infinity', rb'')
_FLAT_OBJECT_NO_NAN = re.compile(
    rb'\{[ \t\r\n]*(?:'
    + _SIMPLE_MEMBER_STRICT
    + rb'(?:[ \t\r\n]*,[ \t\r\n]*'
    + _SIMPLE_MEMBER_STRICT
    + rb')*)?[ \t\r\n]*\}'
)


class JsonFallbackNeeded(ValueError):  # noqa: N818 - the name reads as a sentence
    """Scanner rejected input that ``json.loads`` would have accepted.

    Callers that must reproduce ``json.loads`` semantics exactly (the export
    row-transform) treat this as "fall back to a full parse for this document";
    callers that merely validate (the result pipeline) can let it bubble as the
    plain ``ValueError`` it is.
    """


class JsonFieldScanner:
    """Strict JSON object scanner with absolute byte spans for its direct fields."""

    def __init__(
        self,
        *,
        max_depth: int = 1000,
        metadata_limit: int = 4096,
        reject_non_finite: bool = False,
        root_string_length_only: bool = False,
    ) -> None:
        self.offset = 0
        self.fields: dict[str, tuple[int, int | None, str]] = {}
        self.spans: dict[str, tuple[int, int]] = {}
        self.kinds: dict[str, str] = {}
        self.scalars: dict[str, Any] = {}
        self.empty_members: set[str] = set()
        self._stack: list[list[str]] = []
        self._mode = ''
        self._escape = 0
        self._token = bytearray()
        self._capture = False
        self._key = ''
        self._key_token = False
        self._nested_key = False
        self._root_scalar = False
        self._started = False
        self._done = False
        self._utf8 = codecs.getincrementaldecoder('utf-8')()
        self._max_depth = max_depth
        self._limit = metadata_limit
        self.reject_non_finite = reject_non_finite
        self.root_string_length_only = root_string_length_only
        self._string_occupied = False
        self._string_truncated = False

    def _append(self, value: bytes) -> None:
        if self._capture:
            if len(self._token) + len(value) > self._limit:
                if self.root_string_length_only and self._mode == 'string' and not self._key_token:
                    # Oversized root string: keep it out of RAM, remember that it
                    # is non-empty. The span still covers the whole member.
                    self._capture = False
                    self._string_occupied = True
                    self._string_truncated = True
                    return
                if self.root_string_length_only and not self._key_token and self._mode != 'string':
                    # Export's span-only mode can omit oversized scalar values.
                    # Default validation must reject them instead of skipping checks.
                    self._capture = False
                    self._string_truncated = True
                    return
                raise JsonFallbackNeeded('JSON metadata or number exceeds limit')
            self._token.extend(value)

    def _complete(self, end: int) -> None:
        if not self._stack:
            self._done = True
            return
        frame = self._stack[-1]
        frame[1] = 'comma'
        if len(self._stack) == 1:
            start, _, kind = self.fields[self._key]
            self.fields[self._key] = (start, end, kind)
            self.spans[self._key] = (start, end)
            self.kinds[self._key] = kind

    def _scalar_end(self, end: int) -> None:
        raw = bytes(self._token)
        if self._nested_key:
            self._stack[-1][1] = 'colon'
        elif self._key_token:
            key = json.loads(raw)
            if len(self.fields) >= 10000:
                raise ValueError('too many JSON envelope fields')
            if key in self.fields:
                raise JsonFallbackNeeded('duplicate JSON envelope field')
            self._key = key
            self._stack[-1][1] = 'colon'
        else:
            if self.reject_non_finite and raw in {b'NaN', b'Infinity', b'-Infinity'}:
                raise ValueError('JSON contains NaN or Infinity')
            if (
                self._mode == 'primitive'
                and not self._string_truncated  # truncated tokens were validated incrementally
                and raw not in {b'true', b'false', b'null', b'NaN', b'Infinity', b'-Infinity'}
                and not _NUMBER.fullmatch(raw)
            ):
                raise ValueError('invalid JSON scalar')
            if self._root_scalar:
                if raw == b'""':
                    self.empty_members.add(self._key)
                if not self._string_truncated:
                    self.scalars[self._key] = json.loads(raw)
            self._complete(end)
        self._mode = ''
        self._token.clear()
        self._string_truncated = False

    def feed(self, data: bytes) -> None:
        self._utf8.decode(data)
        i, n, base = 0, len(data), self.offset
        while i < n:
            if self._mode == 'string':
                if self._escape:
                    self._string_occupied = True
                    char = data[i]
                    if self._escape == 1:
                        if char == 117:
                            self._escape = 5
                        elif char in b'"\\/bfnrt':
                            self._escape = 0
                        else:
                            raise ValueError('invalid JSON escape')
                    else:
                        if char not in b'0123456789abcdefABCDEF':
                            raise ValueError('invalid JSON unicode escape')
                        self._escape -= 1
                        if self._escape == 1:
                            self._escape = 0
                    self._append(data[i : i + 1])
                    i += 1
                    continue
                match = _STRING_SPECIAL.search(data, i)
                stop = match.start() if match else n
                self._append(data[i:stop])
                if stop > i:
                    self._string_occupied = True
                i = stop
                if i == n:
                    break
                char = data[i]
                self._append(data[i : i + 1])
                i += 1
                if char == 34:
                    self._scalar_end(base + i)
                elif char == 92:
                    self._escape = 1
                else:
                    raise ValueError('unescaped JSON control character')
                continue
            if self._mode == 'primitive':
                match = _PRIMITIVE_END.search(data, i)
                stop = match.start() if match else n
                self._append(data[i:stop])
                i = stop
                if i == n:
                    break
                self._scalar_end(base + i)
                continue
            match = _SPACE.search(data, i)
            if match is None:
                break
            i = match.start()
            char = data[i]
            if self._done:
                raise ValueError('trailing JSON content')
            if not self._started:
                if char != 123:
                    raise ValueError('JSON envelope must be an object')
                self._started = True
                self._stack.append(['object', 'first_key'])
                i += 1
                continue
            kind, state = self._stack[-1]
            closing = 125 if kind == 'object' else 93
            if char == closing and state in {'first_key', 'first_value', 'comma'}:
                self._stack.pop()
                if len(self._stack) == 1 and state in {'first_key', 'first_value'}:
                    self.empty_members.add(self._key)
                i += 1
                self._complete(base + i)
                continue
            if state == 'comma':
                if char != 44:
                    raise ValueError('expected JSON comma')
                self._stack[-1][1] = 'key' if kind == 'object' else 'value'
                i += 1
                continue
            if state == 'colon':
                if char != 58:
                    raise ValueError('expected JSON colon')
                self._stack[-1][1] = 'value'
                i += 1
                continue
            is_key = state in {'key', 'first_key'}
            if is_key and char != 34:
                raise ValueError('expected JSON object key')
            root = len(self._stack) == 1
            if root and not is_key:
                field_kind = 'object' if char == 123 else 'array' if char == 91 else 'scalar'
                self.fields[self._key] = (base + i, None, field_kind)
            self._key_token = is_key and root
            self._root_scalar = root and not is_key
            if char in (123, 91):
                if len(self._stack) >= self._max_depth:
                    raise JsonFallbackNeeded('JSON nesting exceeds limit')
                # The flat fast path skips token-level checks: it is only used
                # when its regex already encodes the caller's checks (NaN
                # rejection via the strict pattern; surrogate escapes cannot
                # appear in a flat object since _SIMPLE_STRING excludes
                # backslashes).
                flat = None
                if char == 123:
                    pattern = _FLAT_OBJECT_NO_NAN if self.reject_non_finite else _FLAT_OBJECT
                    flat = pattern.match(data, i, min(n, i + 4096))
                if flat is not None:
                    i = flat.end()
                    if len(self._stack) == 1 and data[i - 2 : i] == b'{}':
                        self.empty_members.add(self._key)
                    self._complete(base + i)
                    continue
                self._stack.append(['object', 'first_key'] if char == 123 else ['array', 'first_value'])
                i += 1
            elif char == 34:
                self._mode = 'string'
                self._capture = root
                self._token.clear()
                self._string_occupied = False
                self._string_truncated = False
                self._append(b'"')
                # Nested keys must finish in the colon state too, without retaining their text.
                self._nested_key = is_key and not root
                i += 1
            else:
                self._mode = 'primitive'
                self._capture = True
                self._token.clear()
                self._nested_key = False
        self.offset += n

    def finish(self) -> None:
        self._utf8.decode(b'', final=True)
        if self._mode == 'primitive':
            self._scalar_end(self.offset)
        if not self._done or self._mode or self._stack:
            raise ValueError('truncated JSON document')


def object_members(
    stream: BinaryIO,
    *,
    chunk_bytes: int = 1 << 20,
    metadata_limit: int = 4096,
    max_depth: int = 1000,
) -> dict[str, Any]:
    """Incrementally validate a streamed JSON object and return its members.

    The stream is scanned once without materializing the document; each member
    is then parsed from its exact byte span, so peak memory is the accumulated
    values plus the largest single member instead of the whole document plus its
    parsed form. Scanner size bounds and other ``json.loads`` divergences
    (nesting/size limits, duplicate keys) fall back to a full ``json.loads`` of
    the stream, matching what an unbounded parse would have done.
    """
    scanner = JsonFieldScanner(metadata_limit=metadata_limit, max_depth=max_depth)
    try:
        while chunk := stream.read(chunk_bytes):
            scanner.feed(chunk)
        scanner.finish()
    except JsonFallbackNeeded:
        stream.seek(0)
        return cast(dict[str, Any], json.loads(stream.read()))
    members: dict[str, Any] = {}
    for name, (start, end, _) in scanner.fields.items():
        if end is None:  # unreachable after finish(); keeps the span type honest
            raise ValueError('incomplete JSON envelope field')
        stream.seek(start)
        members[name] = json.loads(stream.read(end - start))
    return members


def iter_array_element_spans(chunks: Iterable[bytes], start_offset: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` spans of the top-level elements of a JSON array.

    ``chunks`` must cover the array **including its opening ``[``**;
    ``start_offset`` is the stream offset of ``chunks[0][0]``. Tracking is
    string/escape aware, so brackets inside strings never confuse the depth
    counter. Spans are yielded incrementally as each element closes — the
    caller can stream-scan a huge element without materializing the array.
    """
    offset = start_offset
    depth = 0  # nesting inside the current element (objects/arrays)
    in_string = False
    escaped = False
    elem_start: int | None = None
    seen_open = False  # the array's own '[' has been consumed
    for chunk in chunks:
        i, n = 0, len(chunk)
        while i < n:
            byte = chunk[i]
            if not seen_open:
                if byte in _WS:
                    i += 1
                    offset += 1
                    continue
                seen_open = True  # caller feeds the full array span including '['
                if byte == 0x5B:
                    i += 1
                    offset += 1
                    continue
            if in_string:
                if escaped:
                    escaped = False
                elif byte == 0x5C:  # backslash
                    escaped = True
                elif byte == 0x22:  # quote closes the string
                    in_string = False
            elif byte == 0x22:
                in_string = True
            elif byte in b'{[':
                depth += 1
                if depth == 1 and elem_start is None:
                    elem_start = offset
            elif byte in b'}]':
                depth -= 1
                if depth == 0 and elem_start is not None:
                    yield (elem_start, offset + 1)
                    elem_start = None
                elif depth == -1 and elem_start is not None:
                    # Scalar last element closed by the array's own bracket.
                    yield (elem_start, offset)
                    elem_start = None
            elif byte == 0x2C and depth == 0:  # comma separates elements
                if elem_start is not None:
                    yield (elem_start, offset)
                    elem_start = None
            elif depth == 0 and elem_start is None and byte not in _WS:
                elem_start = offset  # scalar element (number/true/false/null)
            offset += 1
            i += 1
