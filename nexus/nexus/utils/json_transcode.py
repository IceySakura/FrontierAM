"""Transcode compact JSON bytes to ``(', ', ': ')`` separators without parsing.

The export row-transform copies member spans from a run_result artifact into an
output row. The artifact body and the output row are produced by the same
encoder family (``nexus.utils.json_stream.iter_json_bytes``), so their only
byte-level difference is separator spacing: the body is compact, the row uses
``(', ', ': ')``. This module converts between the two with a byte-level state
machine — strings are skipped (never decoded), structural commas/colons gain a
space, everything else passes through — so peak memory is one chunk either way.

Input must be strict JSON with no insignificant whitespace (both hold for
self-produced bodies; whitespace would be passed through and preserved
verbatim, not normalized).
"""

from __future__ import annotations

import re

from collections.abc import Iterable, Iterator


# Bytes that end an "outside string" run or change state there.
_OUTSIDE_SPECIAL = re.compile(rb'["{}:,\[\]]')
_OUTSIDE_SPECIAL_NAN = re.compile(rb'["{}:,\[\]]|NaN|Infinity|-Infinity')
_WS = b' \t\n\r'
# Bytes that end an "inside string" run: closing quote or escape introducer.
_IN_STRING_SPECIAL = re.compile(rb'["\\]')

_OUT, _IN_STRING, _ESCAPE, _UNICODE = range(4)


def iter_spaced_json(chunks: Iterable[bytes], chunk_size: int = 1 << 20, normalize: bool = False) -> Iterator[bytes]:
    """Yield ``(', ', ': ')``-separated bytes for an iterable of compact JSON bytes.

    Structural commas/colons (outside strings) gain one space; string bytes,
    escape sequences (including ``\\uXXXX`` split across chunk boundaries) and
    container delimiters pass through unchanged - with one exception: a *paired*
    surrogate escape (u D800-DBFF immediately followed by u DC00-DFFF) is
    re-emitted as the merged code point's raw UTF-8, matching what json.loads
    + re-serialization produces for it. Lone surrogate escapes pass through
    verbatim. Memory stays bounded: at most 12 bytes are held while a
    potential surrogate pair is being resolved.
    never buffered whole: memory is the current chunk in and out.
    """
    state = _OUT
    unicode_left = 0
    ws_skip = normalize  # after ', '/': ' structural whitespace is already emitted
    special = _OUTSIDE_SPECIAL_NAN if normalize else _OUTSIDE_SPECIAL
    pending_high = -1  # high-surrogate value held back, awaiting a low
    held = bytearray()  # the held escape bytes (<=6), flushed when no merge
    hexbuf = bytearray()
    esc = bytearray()  # bytes of the escape currently being consumed
    for chunk in chunks:
        if not chunk:
            continue
        # NaN authority is the scanner tee (chunk-safe at any depth); the
        # per-chunk token check below is defense-in-depth for direct callers.
        out = bytearray()
        i, n = 0, len(chunk)
        while i < n:
            if state == _OUT:
                match = special.search(chunk, i)
                stop = match.start() if match else n
                if stop > i:
                    if normalize and ws_skip:
                        j = i
                        while j < stop and chunk[j] in _WS:
                            j += 1
                        out += chunk[j:stop]
                        ws_skip = False
                    else:
                        out += chunk[i:stop]
                    i = stop
                if i >= n:
                    break
                char = chunk[i]
                if char == 0x22:  # quote opens a string
                    out += b'"'
                    state = _IN_STRING
                    ws_skip = False
                elif char == 0x2C:  # comma
                    out += b', '
                    ws_skip = normalize
                elif char == 0x3A:  # colon
                    out += b': '
                    ws_skip = normalize
                elif normalize and char in _WS:
                    i += 1
                    continue  # inter-token whitespace: output already spaced
                elif normalize and char in b'NI-':
                    token = chunk[i : i + 9]
                    if token.startswith((b'NaN', b'Infinity', b'-Infinity')):
                        raise ValueError('JSON contains NaN or Infinity')
                    out += chunk[i : i + 1]
                    i += 1
                    ws_skip = False
                    continue
                else:  # {} [] delimiters
                    out += chunk[i : i + 1]
                    ws_skip = False
                i += 1
            elif state == _IN_STRING:
                match = _IN_STRING_SPECIAL.search(chunk, i)
                stop = match.start() if match else n
                if stop > i:
                    if pending_high >= 0:
                        out += held  # plain text after a lone high: no merge
                        held.clear()
                        pending_high = -1
                    out += chunk[i:stop]
                    i = stop
                if i >= n:
                    break
                char = chunk[i]
                if char == 0x5C:  # backslash starts an escape; flush deferred
                    esc += bytes((char,))
                    state = _ESCAPE
                else:
                    if pending_high >= 0:
                        out += held  # lone high surrogate: escape stays as-is
                        held.clear()
                        pending_high = -1
                    out += chunk[i : i + 1]
                    if char == 0x22:
                        state = _OUT
                i += 1
            elif state == _ESCAPE:
                char = chunk[i]
                esc += chunk[i : i + 1]
                i += 1
                if char == 0x75:  # 'u' starts a unicode escape (in-string)
                    state = _UNICODE
                    unicode_left = 4
                    hexbuf.clear()
                else:
                    if pending_high >= 0:
                        out += held  # short escape after a lone high: no merge
                        held.clear()
                        pending_high = -1
                    if normalize and char == 0x2F:
                        out += b'/'  # loads('\\/') == '/' -> re-encode emits bare slash
                    else:
                        out += esc
                    esc.clear()
                    state = _IN_STRING
            else:  # _UNICODE: remaining hex digits of the unicode escape
                take = min(unicode_left, n - i)
                hexbuf += chunk[i : i + take]
                esc += chunk[i : i + take]
                i += take
                unicode_left -= take
                if unicode_left == 0:
                    value = int(hexbuf, 16)
                    if pending_high >= 0 and 0xDC00 <= value <= 0xDFFF:
                        # Paired escapes merge into one code point, re-encoded
                        # as raw UTF-8 exactly like json.loads does.
                        code = 0x10000 + ((pending_high - 0xD800) << 10) + (value - 0xDC00)
                        out += chr(code).encode('utf-8')
                        held.clear()
                        pending_high = -1
                    elif 0xD800 <= value <= 0xDBFF:
                        if pending_high >= 0:
                            out += held  # high followed by high: no merge
                        held.clear()
                        held += esc
                        pending_high = value
                    else:
                        if pending_high >= 0:
                            out += held  # separated: no merge
                            held.clear()
                        pending_high = -1
                        if normalize and value == 0x22:
                            out += b'\\"'  # decode-then-reencode yields the short escape
                        elif normalize and value == 0x5C:
                            out += b'\\\\'
                        elif normalize and value >= 0x20 and not 0xD800 <= value <= 0xDFFF:
                            out += chr(value).encode('utf-8')  # ensure_ascii column -> raw UTF-8
                        else:
                            out += esc  # control chars and lone surrogates round-trip verbatim
                    esc.clear()
                    state = _IN_STRING
        yield bytes(out)
