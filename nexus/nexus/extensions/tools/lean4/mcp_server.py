"""Lean4 Proof MCP Server — pure JSON-RPC 2.0 over stdio.

A standalone MCP server for Claude Code (or any MCP-capable client) that
exposes a single ``check_proof`` tool. The tool forwards a Lean 4 source
snippet (``cmd``) to a remote Lean 4 server via
``AsyncLean4Client.async_check_whole`` and returns the verification result.

Mirrors the layout of ``tools/search_stateful/mcp_server.py``:
  - implements the minimal MCP protocol subset by hand (newline-delimited
    JSON-RPC 2.0 over stdio, per the MCP spec);
  - all logging goes to stderr (stdout is reserved for the transport);
  - the Lean 4 client is constructed lazily in ``_init`` and reused for
    every tool call.

Env vars (set by the MCP client config):
  LEAN4_POLARIS_URL    — Lean 4 server URL
                         (default: ``lean4-server-test-v2.production.polaris:8080``)
  LEAN4_REPO_PATH      — Lean repo path on the server side
                         (default: ``/miniF2F-lean4``)
  LEAN4_TIMEOUT        — per-request timeout in seconds (default: 600)
  LEAN4_RETRY_NUM      — number of retries on transient errors (default: 3)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from async_lean4_client import AsyncLean4Client


# ---------------------------------------------------------------------------
# Logging — stderr only (stdout is the JSON-RPC transport)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='[mcp-lean4] %(levelname)s %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy client — constructed once in _init() and reused per call
# ---------------------------------------------------------------------------

_client: AsyncLean4Client | None = None
_default_timeout: float = 600.0

# asyncio.StreamReader defaults to a 64 KiB per-line limit. A single tools/call
# request carries the full Lean source `cmd` as one JSON line, which easily
# exceeds 64 KiB for large proofs — readline() then raises LimitOverrunError
# and crashes the server. Use a much larger limit (overridable via env).
_STREAM_READER_LIMIT: int = 1 * 1024 * 1024  # 1 MiB


def _get_client() -> AsyncLean4Client:
    if _client is None:
        raise RuntimeError('Lean4 client has not been initialised')
    return _client


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning('Invalid float for %s=%r, falling back to %s', name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning('Invalid int for %s=%r, falling back to %s', name, raw, default)
        return default


def _init() -> None:
    """Initialise the Lean 4 client from env vars."""
    global _client, _default_timeout

    from async_lean4_client import AsyncLean4Client

    polaris_url = os.environ.get(
        'LEAN4_POLARIS_URL',
        'http://lean4-server-test-v2.production.polaris:8080',
    )
    repo_path = os.environ.get('LEAN4_REPO_PATH', '/miniF2F-lean4')
    timeout = _env_float('LEAN4_TIMEOUT', 600.0)
    retry_num = _env_int('LEAN4_RETRY_NUM', 3)

    _client = AsyncLean4Client(
        polaris_url=polaris_url,
        retry_num=retry_num,
        repo_path=repo_path,
        timeout=timeout,
    )

    _default_timeout = timeout

    logger.debug(
        'Lean4 client initialised (url=%s, repo=%s, timeout=%.0fs, retries=%d)',
        polaris_url,
        repo_path,
        timeout,
        retry_num,
    )


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 over stdio — newline-delimited (MCP spec)
# ---------------------------------------------------------------------------


async def _read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one newline-delimited JSON-RPC message. Returns ``None`` on EOF."""
    while True:
        try:
            line = await reader.readline()
        except ValueError as exc:
            # Raised when a single line exceeds the StreamReader limit. asyncio
            # has already discarded the offending line from the buffer, so skip
            # it and keep the server alive instead of letting it crash.
            logger.warning('Skipping oversized JSON-RPC line: %s', exc)
            continue
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning('Skipping malformed JSON line: %s', exc)
            continue


def _write_message(msg: dict[str, Any]) -> None:
    """Write one newline-delimited JSON-RPC message to stdout."""
    line = json.dumps(msg, ensure_ascii=False, separators=(',', ':'))
    sys.stdout.write(line + '\n')
    sys.stdout.flush()


def _ok(request_id: Any, result: Any) -> dict[str, Any]:
    return {'jsonrpc': '2.0', 'id': request_id, 'result': result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        'name': 'check_proof',
        'description': (
            'Verify a Lean 4 proof (or any standalone Lean 4 snippet) against '
            'the configured Lean 4 server. Submit the FULL Lean source as `cmd` '
            '(imports, opens, set_options, theorem statement, and proof). The '
            'server compiles it and returns a JSON object with shape:\n'
            '  {\n'
            '    "is_ok": <bool>,                # true ⇔ no Lean errors\n'
            '    "proof_with_line_num": <str>,   # your `cmd` with line numbers;\n'
            '                                    # regions far from any reported\n'
            '                                    # position are elided as "..."\n'
            '    "repl_res": {\n'
            '      "messages": [                 # warnings / errors\n'
            '        {"severity": "warning"|"error"|"info",\n'
            '         "pos":    {"line": <int>, "column": <int>},\n'
            '         "endPos": {"line": <int>, "column": <int>},\n'
            '         "data": "<text>"}\n'
            '      ],\n'
            '      "suppressed": <str>           # present only when repeated\n'
            '                                    # errors on one line were folded\n'
            '    },\n'
            '    ...\n'
            '  }\n'
            '\n'
            'Use `is_ok` as the success signal. When false, read every entry of '
            '`repl_res.messages` (line/column point into your `cmd`) and revise '
            'the snippet accordingly. Warnings about unused variables / '
            'unreachable tactics do NOT make `is_ok` false on their own — only '
            'real errors or remaining `sorry`s do.\n'
            'Responses are compacted to protect the context budget: a very long '
            '`data` is clipped, repeated errors on the same line are folded into '
            '`suppressed`, and `apply?`/`hint` suggestion dumps are omitted — '
            'those tactics leave a `sorry` behind, so a proof built from them is '
            'rejected. Re-check a smaller snippet if you need more detail.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'cmd': {
                    'type': 'string',
                    'description': (
                        'Full Lean 4 source to type-check, including imports, opens, '
                        'set_options, the theorem statement, and the proof.'
                    ),
                },
            },
            'required': ['cmd'],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_content(text: str) -> list[dict[str, Any]]:
    """Wrap *text* in MCP ``TextContent`` format."""
    return [{'type': 'text', 'text': text}]


def _error_content(text: str) -> dict[str, Any]:
    """Return MCP call_tool error result."""
    return {'content': _text_content(text), 'isError': True}


def _ok_content(payload: Any) -> dict[str, Any]:
    """Return MCP call_tool success result, serialising *payload* as JSON text."""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return {'content': _text_content(text)}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def extract_fragment(res: Any, cmd: str | list[str]) -> Any:
    if not res:
        return res
    if isinstance(cmd, str):
        lines = cmd.split('\n')
    else:
        assert isinstance(cmd, list), f'bad cmd type for extract_fragment {cmd}'
        lines = cmd
    if isinstance(res, dict):
        pos = res.get('pos')
        end_pos = res.get('endPos')
        # The Lean REPL may emit ``pos``/``endPos`` as null (e.g. file-level
        # messages), so guard against None before subscripting ``['line']``.
        if isinstance(pos, dict) and isinstance(end_pos, dict) and 'line' in pos and 'line' in end_pos:
            start_line = max(pos['line'] - 1, 0)
            end_line = max(end_pos['line'], 0)
            if end_line <= start_line:
                end_line = start_line + 1
            fragment = '\n'.join(lines[start_line:end_line])
            res['text_fragment'] = fragment
            return res
        return {k: extract_fragment(v, lines) for k, v in res.items()}
    if isinstance(res, list):
        return [extract_fragment(v, lines) for v in res]
    return res


def add_line_num(cmd: str) -> str:
    if not cmd:
        return cmd
    lines = [f'line_{n + 1}: {line}' for n, line in enumerate(cmd.split('\n'))]
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Response compaction
#
# Measured over 17,211 real tool responses from the A3B-Lean4 run: the average
# payload was ~3.2 KB and the median rollout burned ~38 K tokens on tool
# observations alone, pushing 53-62% of rollouts into TRUNCATION -- where the
# proof is never submitted and the score is 0 regardless of proof quality. The
# passes below cut ~23% of the payload while keeping the signal the model acts
# on.
#
# Deliberately NOT touched:
#   * the JSON shape (keys / nesting) -- the system prompt documents it;
#   * `text_fragment` -- it ties an error to the offending source line;
#   * non-`Try this` `info` messages -- they carry `#eval` / `#check` output,
#     which the model uses to inspect terms, not just to verify a proof.
# --------------------------------------------------------------------------- #

# Cap for one message's `data`. Real error data has p90=366 and p95=547 chars,
# so 600 leaves ~96% of messages untouched while clipping the tail (largest
# observed: 277 KB of goal state).
MAX_MESSAGE_DATA_CHARS = 600

# Lean reports one error per branch, so `cases` / `interval_cases` can emit
# hundreds of near-identical failures at one source line (worst case observed:
# 120 errors on line 23, differing only in the case label). Keep a few examples
# per line plus a handful of distinct lines; summarise the rest.
MAX_ERRORS_PER_LINE = 2
MAX_ERROR_LINES = 6

# Radius, in lines, retained around every reported position in the echoed source.
PROOF_ECHO_CONTEXT_LINES = 8

_TRY_THIS_PREFIX = 'Try this:'


def _clip_data(message: Any) -> Any:
    """Return ``message`` with an over-long ``data`` field clipped."""
    if not isinstance(message, dict):
        return message
    data = str(message.get('data') or '')
    if len(data) <= MAX_MESSAGE_DATA_CHARS:
        return message
    clipped = dict(message)
    clipped['data'] = (
        f'{data[:MAX_MESSAGE_DATA_CHARS]}\n...(truncated, {len(data) - MAX_MESSAGE_DATA_CHARS} more chars)'
    )
    return clipped


def compact_messages(messages: Any) -> tuple[Any, str | None]:
    """Drop redundant messages and clip long ones.

    Returns ``(messages, summary)``; ``summary`` describes the suppressed errors
    (``None`` when nothing was suppressed) so the model still learns how many
    branches failed and where.
    """
    if not isinstance(messages, list):
        return messages, None

    errors: list[dict[str, Any]] = []
    others: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            others.append(message)
            continue
        if message.get('severity') == 'error':
            errors.append(message)
            continue
        # `apply?` / `hint` emit one info per candidate tactic. Measured adoption
        # in the following turn: 3 of 5,485 suggestions (0.05%), for ~23% of all
        # info bytes -- and both tactics leave `sorryAx`, so a proof built from
        # them is rejected anyway.
        if str(message.get('data') or '').startswith(_TRY_THIS_PREFIX):
            continue
        others.append(message)

    grouped: dict[Any, list[dict[str, Any]]] = {}
    for error in errors:
        position = error.get('pos')
        line = position.get('line') if isinstance(position, dict) else None
        grouped.setdefault(line, []).append(error)

    kept: list[dict[str, Any]] = []
    suppressed: dict[Any, int] = {}
    for index, (line, group) in enumerate(grouped.items()):
        if index < MAX_ERROR_LINES:
            kept.extend(group[:MAX_ERRORS_PER_LINE])
            extra = len(group) - MAX_ERRORS_PER_LINE
            if extra > 0:
                suppressed[line] = suppressed.get(line, 0) + extra
        else:
            suppressed[line] = suppressed.get(line, 0) + len(group)

    compacted: list[Any] = [_clip_data(m) for m in kept]
    compacted.extend(_clip_data(m) for m in others)

    summary = None
    if suppressed:
        worst = sorted(suppressed.items(), key=lambda kv: -kv[1])[:MAX_ERROR_LINES]
        detail = ', '.join(f'line {line}x{count}' for line, count in worst)
        summary = f'{sum(suppressed.values())} more error(s) suppressed ({detail})'
    return compacted, summary


def _reported_lines(messages: Any, num_lines: int) -> set[int]:
    """Source lines referenced by any message position."""
    lines: set[int] = set()
    if not isinstance(messages, list):
        return lines
    for message in messages:
        if not isinstance(message, dict):
            continue
        for key in ('pos', 'endPos'):
            position = message.get(key)
            if isinstance(position, dict):
                line = position.get('line')
                if isinstance(line, int) and 1 <= line <= num_lines:
                    lines.add(line)
    return lines


def echo_proof_with_line_num(cmd: str, messages: Any) -> str:
    """Echo ``cmd`` with line numbers, eliding regions far from any diagnostic.

    The line-numbered echo measurably helps the model locate and fix errors, so
    it is kept -- but only the neighbourhood of each reported position is useful.

    With no reported position there is nothing to point at, so the numbered echo
    carries no information the model does not already have (it just sent this
    exact source) and is dropped entirely. Observed cases: a clean check, a
    transport-level failure, and an upstream text-filter rejection such as
    "can not contains sorry|admit|stop" -- none of which reference a line.
    """
    if not cmd:
        return cmd

    lines = cmd.split('\n')
    total = len(lines)
    anchors = _reported_lines(messages, total)

    if not anchors:
        return ''

    keep: set[int] = set()
    for anchor in anchors:
        start = max(1, anchor - PROOF_ECHO_CONTEXT_LINES)
        end = min(total, anchor + PROOF_ECHO_CONTEXT_LINES)
        keep.update(range(start, end + 1))

    out: list[str] = []
    previous = 0
    for number in sorted(keep):
        # Mark every elided region, including one before the first kept line, so
        # the model can tell the echo is a excerpt rather than the whole file.
        if number > previous + 1:
            out.append('...')
        out.append(f'line_{number}: {lines[number - 1]}')
        previous = number
    if previous < total:
        out.append('...')
    return '\n'.join(out)


async def _do_check_proof(args: dict[str, Any]) -> dict[str, Any]:
    """Run ``async_check_whole(cmd)`` and return the JSON result."""
    cmd = args.get('cmd', '')
    if not isinstance(cmd, str) or not cmd.strip():
        return _error_content('Error: `cmd` must be a non-empty Lean 4 source string.')

    client = _get_client()

    try:
        raw_result = await client.async_check_whole(cmd=cmd)
    except Exception as exc:
        logger.exception('check_proof failed')
        return _error_content(f'Lean4 verification failed: {type(exc).__name__}: {exc}')

    if not isinstance(raw_result, dict):
        return _error_content(f'Unexpected response type from Lean4 server: {type(raw_result).__name__}')

    if raw_result.get('exception'):
        return _error_content(f'Lean4 verification failed: {raw_result["exception"]}')

    repl_res = extract_fragment(raw_result.get('repl_res', {}), cmd)
    is_ok = raw_result.get('is_ok', False)
    messages: Any = None
    if isinstance(repl_res, dict):
        messages, suppressed = compact_messages(repl_res.get('messages'))
        if messages is not None:
            repl_res['messages'] = messages
        if suppressed:
            repl_res['suppressed'] = suppressed
    payload = {
        'is_ok': is_ok,
        'repl_res': repl_res,
        'proof_with_line_num': echo_proof_with_line_num(cmd, messages),
    }
    return _ok_content(payload)


_TOOL_DISPATCH = {
    'check_proof': _do_check_proof,
}


# ---------------------------------------------------------------------------
# MCP protocol dispatch
# ---------------------------------------------------------------------------

SERVER_INFO = {
    'name': 'lean4-proof',
    'version': '0.1.0',
}

CAPABILITIES = {
    'tools': {},
}


async def _dispatch(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message. Returns a response dict, or ``None`` for notifications."""
    method = msg.get('method', '')
    msg_id = msg.get('id')
    params = msg.get('params', {})

    if msg_id is None:
        if method == 'notifications/initialized':
            logger.debug('Client sent notifications/initialized')
        elif method == 'notifications/cancelled':
            logger.info('Client cancelled request: %s', params.get('requestId'))
        else:
            logger.debug('Ignoring notification: %s', method)
        return None

    if method == 'initialize':
        return _ok(
            msg_id,
            {
                'protocolVersion': params.get('protocolVersion', '2024-11-05'),
                'capabilities': CAPABILITIES,
                'serverInfo': SERVER_INFO,
            },
        )

    if method == 'ping':
        return _ok(msg_id, {})

    if method == 'tools/list':
        return _ok(msg_id, {'tools': TOOL_SCHEMAS})

    if method == 'tools/call':
        tool_name = params.get('name', '')
        tool_args = params.get('arguments', {})
        handler = _TOOL_DISPATCH.get(tool_name)
        if handler is None:
            return _ok(msg_id, _error_content(f'Unknown tool: {tool_name}'))
        try:
            result = await handler(tool_args)
        except Exception as exc:
            logger.exception('Tool %s failed', tool_name)
            result = _error_content(f'Internal error: {exc}')
        return _ok(msg_id, result)

    if method in ('resources/list', 'resources/templates/list', 'prompts/list'):
        items_key = 'prompts' if 'prompts' in method else 'resources'
        return _ok(msg_id, {items_key: []})

    return _error(msg_id, -32601, f'Method not found: {method}')


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _main() -> None:
    try:
        _init()
    except Exception as exc:
        logger.exception('Lean4 MCP server initialisation failed: %s', exc)
        raise
    limit = _env_int('LEAN4_MCP_MAX_LINE_BYTES', _STREAM_READER_LIMIT)
    reader = asyncio.StreamReader(limit=limit)
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    logger.debug('MCP lean4-proof server ready')

    while True:
        msg = await _read_message(reader)
        if msg is None:
            logger.debug('EOF on stdin, shutting down')
            break
        logger.debug('← %s', msg.get('method', '?'))

        response = await _dispatch(msg)
        if response is not None:
            _write_message(response)
            logger.debug('→ id=%s', response.get('id'))


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt, BrokenPipeError):
        asyncio.run(_main())
