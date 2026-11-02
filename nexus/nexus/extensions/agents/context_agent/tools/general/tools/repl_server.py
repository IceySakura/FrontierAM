"""Persistent REPL server — runs inside the python tool subprocess.

Protocol (length-prefixed frames over stdin/stdout):

    request  : 8 ASCII hex digits = N, then N bytes of UTF-8 code
    response : 8 ASCII hex digits = M, then M bytes of UTF-8 JSON
               {"stdout": str, "stderr": str, "result": str|None}

Each request's code is ``exec``'d in a single persistent namespace, so
variables / imports / functions defined in one call are visible in the
next (Jupyter-like statefulness). If the final statement is an
expression, its ``repr`` is returned as ``result`` (like a REPL).

This module is intentionally dependency-free and is launched via
``python -u repl_server.py``.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import traceback


def _read_exact(stream, n: int) -> bytes | None:
    """Read exactly *n* bytes; return None on clean EOF before any byte."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None  # EOF (parent closed stdin) — shut down cleanly
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


def _read_frame(stream) -> bytes | None:
    header = _read_exact(stream, 8)
    if header is None:
        return None
    try:
        n = int(header.decode('ascii'), 16)
    except ValueError:
        return None
    return _read_exact(stream, n)


def _write_frame(stream, payload: bytes) -> None:
    stream.write(f'{len(payload):08x}'.encode('ascii'))
    stream.write(payload)
    stream.flush()


def _run(code: str, ns: dict) -> dict:
    out, err = io.StringIO(), io.StringIO()
    result_repr: str | None = None
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError:
        return {'stdout': '', 'stderr': traceback.format_exc(), 'result': None}

    # Split off a trailing bare expression so we can echo its value. The
    # isinstance check binds the statement first so its narrowed type survives
    # the pop (a type checker cannot narrow the return of ``body.pop()``).
    last_expr = None
    last_stmt = tree.body[-1] if tree.body else None
    if isinstance(last_stmt, ast.Expr):
        tree.body.pop()
        last_expr = ast.Expression(last_stmt.value)

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if tree.body:
                exec(compile(tree, '<python_tool>', 'exec'), ns)  # noqa: S102
            if last_expr is not None:
                value = eval(  # noqa: S307
                    compile(last_expr, '<python_tool>', 'eval'), ns
                )
                if value is not None:
                    result_repr = repr(value)
    except BaseException:  # noqa: BLE001 — surface any error text to the model
        err.write(traceback.format_exc())

    return {'stdout': out.getvalue(), 'stderr': err.getvalue(), 'result': result_repr}


def main() -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    ns: dict = {'__name__': '__main__'}
    while True:
        frame = _read_frame(stdin)
        if frame is None:
            break
        code = frame.decode('utf-8', 'replace')
        resp = _run(code, ns)
        _write_frame(stdout, json.dumps(resp, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    main()
