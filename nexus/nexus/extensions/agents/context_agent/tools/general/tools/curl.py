"""``curl`` — download a URL into the rollout workspace.

Returns only METADATA: HTTP status, content type, byte size, the saved path,
and either the detected binary format or the line count when the payload is
text. The body is never returned. That is the point of having this as a tool
rather than leaving it to ``bash``: a model that types ``curl <url>`` without
``-o`` dumps a multi-megabyte PDF straight into the conversation, and the
meta-first workflow the system prompt asks for needs the size and type *before*
anything is read.

Wraps the main image's ``curl`` binary rather than an HTTP library. If an
inherited proxy cannot be resolved, it retries once without proxy settings.
This matters for sandboxes using a provider-level egress allowlist: traffic is
already filtered transparently there, while a stale explicit proxy would send
the request to the proxy host instead of the allowlisted destination.
"""

from __future__ import annotations

import subprocess
import threading

from pathlib import Path

from ..fileio import FileReadError, _hint_for, sniff_kind
from ..processes import run_cancellable, run_command_group
from ..state import GeneralState
from ..subprocess_env import main_image_env, sandbox_command, subprocess_identity_args
from ..workspace import sanitize_filename


_FIELD_SEP = '\t'
_WRITE_OUT = _FIELD_SEP.join(['%{http_code}', '%{content_type}', '%{size_download}', '%{url_effective}'])

# curl exit codes worth translating into something actionable.
_EXIT_HINTS: dict[int, str] = {
    6: 'could not resolve host — check the URL spelling',
    7: 'could not connect to host',
    22: 'server returned an HTTP error',
    23: 'write error — the output path may not be writable',
    28: 'operation timed out',
    35: 'TLS handshake failed',
    56: 'failure receiving network data',
    63: 'maximum file size exceeded',
}


def _target_name(url: str, output: str | None) -> str:
    if output:
        return sanitize_filename(str(output).strip(), fallback='download')
    tail = url.split('?')[0].split('#')[0].rstrip('/').rsplit('/', 1)[-1]
    return sanitize_filename(tail, fallback='download')


def _run_curl(cmd: list[str], timeout: int, *, cancelled: threading.Event | None = None) -> tuple[int, str, str]:
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return run_command_group(
            sandbox_command(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cancelled=cancelled if cancelled is not None else threading.Event(),
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout + 15,
            # Match bash/python by resolving binaries from the main image and
            # keep endpoint credentials out of the child environment.
            env=main_image_env(),
            **subprocess_identity_args(),
        )

    try:
        proc = run(cmd)
        if proc.returncode == 5:
            # CURLE_COULDNT_RESOLVE_PROXY. In network-policy sandboxes the
            # destination may be directly allowlisted even though an inherited
            # sandbox proxy is unavailable. ``--disable`` must be curl's first
            # option to ignore curlrc files; ``--proxy ''`` also overrides all
            # proxy environment variables.
            proc = run([cmd[0], '--disable', '--proxy', '', *cmd[1:]])
    except subprocess.TimeoutExpired:
        return 28, '', f'curl exceeded {timeout}s'
    except Exception as exc:  # noqa: BLE001
        return -1, '', str(exc)
    return proc.returncode, proc.stdout, proc.stderr


async def curl(state: GeneralState, url: str, output: str | None = None) -> str:
    url = str(url or '').strip()
    if not url:
        return "(the 'url' argument is required)"

    if not url.lower().startswith(('http://', 'https://')):
        return (
            f'(curl only handles http(s) URLs; got: {url}\n'
            'If this is a local path, there is nothing to download — pass it '
            'straight to read_lines / grep / file_search, or inspect it with bash.)'
        )

    name = _target_name(url, output)
    dest = state.workspace.docs_dir / name
    rel = state.workspace.path_for(name)
    timeout = state.knobs.curl_timeout

    cmd = [
        'curl',
        '-sS',
        '-L',
        '--max-time',
        str(timeout),
        '--max-filesize',
        str(state.knobs.curl_max_bytes),
        '-o',
        str(dest),
        '-w',
        _WRITE_OUT,
        url,
    ]
    code, stdout, stderr = await run_cancellable(_run_curl, cmd, timeout)

    if code != 0:
        hint = _EXIT_HINTS.get(code, '')
        detail = (stderr or '').strip()
        parts = [f'[Error] curl failed (exit {code})']
        if hint:
            parts.append(f': {hint}')
        if detail:
            parts.append(f'\n{detail}')
        # A partial file is worse than none: it would read as valid text.
        try:
            if dest.exists():
                dest.unlink()
        except OSError:
            pass
        return ''.join(parts)

    fields = (stdout or '').strip().split(_FIELD_SEP)
    http_code = fields[0] if len(fields) > 0 else '?'
    content_type = fields[1] if len(fields) > 1 else ''
    size_download = fields[2] if len(fields) > 2 else '?'
    final_url = fields[3] if len(fields) > 3 else url

    lines = [
        f'Downloaded to: {rel}',
        f'HTTP status:   {http_code}',
        f'Content-Type:  {content_type or "(not reported)"}',
        f'Size:          {size_download} bytes',
    ]
    if final_url and final_url != url:
        lines.append(f'Redirected to: {final_url}')

    if http_code and not http_code.startswith('2'):
        lines.append('')
        lines.append(
            f'WARNING: status {http_code} is not a success code — the saved file '
            'is probably an error page, not the document.'
        )

    try:
        kind = sniff_kind(Path(dest))
    except FileReadError as exc:
        lines.append(f'\n(could not inspect the saved file: {exc})')
        return '\n'.join(lines)

    lines.append('')
    if kind is not None:
        lines.append(
            f'Detected format: {kind}. This is NOT readable as text — '
            f'read_lines / grep / file_search will refuse it.\n{_hint_for(kind)}'
        )
    else:
        try:
            n_lines = len(state.read_lines(Path(dest)))
            lines.append(f'Detected format: text, {n_lines} lines. Ready for read_lines / grep / file_search on {rel}.')
        except FileReadError as exc:
            lines.append(f'(saved, but not readable as text: {exc})')

    return '\n'.join(lines)
