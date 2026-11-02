"General-purpose Python execution, with images returned to the model."

import asyncio
import base64
import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading

from typing import Any

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel

from .image_tools import (
    IMAGE_KEY,
    RENDER_KEY,
    TOOL_RECORDS_KEY,
    _img_return_suffix,
    _tool_image_return_enabled,
)


logger = logging.getLogger(__name__)

WORKDIR_KEY = 'pptx_code_workdir'

for _var in (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_var, '1')

_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
_MAX_IMAGES_PER_CALL = 4

_MAX_STDOUT = 16000
_TRUNC_NOTE = (
    '\n\n[... output truncated: it exceeded {lim} characters, {n} characters '
    'were dropped. What you see above is INCOMPLETE -- do not treat it as the '
    'full answer. Re-run with fewer lines per call, or narrow what you print.]'
)

_WORKER_IDLE_ENV = 'PPTX_CODE_WORKER_IDLE'
_WORKER_IDLE_DEFAULT = 600.0

# Model code runs in a dedicated worker subprocess per work directory, not on
# a thread of this process. A thread cannot be stopped: a `while True` would
# keep redirect_stdout/stderr's process-wide swap in place forever (every
# later print and -- in the codex MCP server -- the JSON-RPC writer would be
# swallowed), concurrent calls would interleave their captures, and
# asyncio.run's exit path would block waiting for the stuck thread. A worker
# process has none of those failure modes: its sys.stdout swap, its PIL
# monkeypatch and its namespace all die with it when we kill() on timeout, it
# serialises naturally (one readline at a time), and it exits by itself when
# the parent's pipe closes or it idles too long.
#
# Protocol: one JSON request line on stdin, one JSON response line on stdout.
# A rogue model thread or a raw os.write(1, ...) in model code can corrupt the
# response; that is detected as an unreadable response and the worker is
# replaced, which contains the damage to that one call.
_WORKER_SRC = r"""
import contextlib
import json
import os
import select
import sys
import traceback


class _Cap:
    def __init__(self, cap, marker):
        self.cap = cap
        self.marker = marker
        self.parts = []
        self.n = 0
        self.marked = False

    def write(self, s):
        if self.n < self.cap:
            self.parts.append(s)
            self.n += len(s)
            if self.n >= self.cap and not self.marked:
                self.parts.append(self.marker)
                self.marked = True
        return len(s)

    def flush(self):
        pass

    def getvalue(self):
        return ''.join(self.parts)


def main():
    # `python -c SRC -- <workdir>` puts '--' itself in sys.argv, so the
    # workdir is the argument AFTER the separator, not sys.argv[1].
    work = sys.argv[sys.argv.index('--') + 1]
    pipe = sys.stdout
    targets = {'target.png', 'render.png'}
    state = {'read_target': False}
    ns = {'__name__': '__main__', 'WORK_DIR': work}
    builtin_open = open

    def _rel(p):
        if isinstance(p, str) and not os.path.isabs(p):
            return os.path.join(work, p)
        return p

    def _note(p):
        if isinstance(p, str) and os.path.basename(p) in targets:
            state['read_target'] = True

    def _open_in_work(path, *a, **kw):
        _note(path)
        return builtin_open(_rel(path), *a, **kw)

    ns['open'] = _open_in_work

    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        _img_open_orig = Image.open
        _img_save_orig = Image.Image.save

        def _img_open(fp, mode='r', formats=None):
            _note(fp)
            return _img_open_orig(_rel(fp), mode=mode, formats=formats)

        def _img_save(self, fp, format=None, **params):
            return _img_save_orig(self, _rel(fp), format=format, **params)

        Image.open = _img_open
        Image.Image.save = _img_save

    idle = os.environ.get('PPTX_CODE_WORKER_IDLE', '600')
    try:
        idle_s = max(1.0, float(idle))
    except ValueError:
        idle_s = 600.0

    stdin = sys.stdin
    while True:
        if not select.select([stdin], [], [], idle_s)[0]:
            return  # idle past the limit: the session moved on, stop lingering
        line = stdin.readline()
        if not line:
            return  # parent closed the pipe: exit with it, never outlive it
        try:
            code = json.loads(line)['code']
        except Exception:
            code = None
        out = _Cap(262144, '\n[... output capped at 262144 characters]')
        err = _Cap(65536, '\n[... stderr capped at 65536 characters]')
        failed = None
        state['read_target'] = False
        if code is None:
            failed = 'the request could not be parsed'
        else:
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    exec(compile(code, '<model_code>', 'exec'), ns)
            except BaseException:
                failed = traceback.format_exc(limit=6)
        resp = {
            'ok': failed is None,
            'stdout': out.getvalue(),
            'stderr': err.getvalue(),
            'failed': failed,
            'read_target': state['read_target'],
        }
        pipe.write(json.dumps(resp, ensure_ascii=False) + '\n')
        pipe.flush()


main()
"""

# One worker process per workdir, and one lock per workdir. Calls for the same
# workdir must not overlap: the worker is single-threaded by design, and two
# in-flight requests would interleave writes and read each other's responses.
# A threading.Lock (not asyncio.Lock) because the codex MCP server runs each
# request in a fresh asyncio.run() event loop -- asyncio primitives do not
# survive crossing loops, plain locks do.
_WORKERS: dict[str, subprocess.Popen[str]] = {}
_WORKER_LOCKS: dict[str, threading.Lock] = {}


def _clip(text: str, lim: int | None = None) -> str:
    lim = _MAX_STDOUT if lim is None else lim
    if text is None:
        return '(no output)'
    if len(text) <= lim:
        return text
    return text[:lim] + _TRUNC_NOTE.format(lim=lim, n=len(text) - lim)


_DEFAULT_TIMEOUT = 60


class CodeInput(RobustBaseModel):
    code: str = Field(description='Python code to execute.')
    timeout: float | None = Field(default=_DEFAULT_TIMEOUT, description='Seconds before the call is abandoned.')


class CodeInterpreterInput(RobustBaseModel):
    code: str = Field(description='Python code to execute.')
    language: str | None = Field(
        default=None,
        description=(
            'Accepted for compatibility with the SFT-side tool schema and ignored: '
            'this interpreter is Python, so naming a language cannot change what runs.'
        ),
    )
    timeout: float | None = Field(default=_DEFAULT_TIMEOUT, description='Seconds before the call is abandoned.')


def _image_to_data_url(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    mime = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
    }.get(ext, 'image/png')
    try:
        with open(path, 'rb') as handle:
            return f'data:{mime};base64,{base64.b64encode(handle.read()).decode("utf-8")}'
    except OSError:
        return None


def _ensure_workdir(context: ToolContext) -> tuple[str, bool]:
    work = context.tool_state.get(WORKDIR_KEY)
    if work and os.path.isdir(work):
        # An existing workdir still needs its seed images: both call paths
        # pre-create the directory (the built-in loop seeds task_input.working_dir,
        # the codex MCP server seeds its own), so "already exists" is the normal
        # case on the first call -- without re-seeding, Image.open("target.png")
        # from the tool description would be a FileNotFoundError.
        #
        # target.png is constant for the slide, so it is only backfilled when
        # missing. render.png is NOT constant: the tool description promises
        # "your most recent render", so it is overwritten from the current
        # RENDER_KEY image on every call -- an exists() skip would leave the
        # model comparing against the first-ever render forever.
        image = context.tool_state.get(IMAGE_KEY)
        if image is not None and hasattr(image, 'save') and not os.path.exists(os.path.join(work, 'target.png')):
            try:
                image.save(os.path.join(work, 'target.png'))
            except (OSError, ValueError):
                logger.warning('could not materialise %s into the code workdir', IMAGE_KEY)
        image = context.tool_state.get(RENDER_KEY)
        if image is not None and hasattr(image, 'save'):
            try:
                image.save(os.path.join(work, 'render.png'))
            except (OSError, ValueError):
                logger.warning('could not refresh render.png in the code workdir')
        return work, False
    work = tempfile.mkdtemp(prefix='pptx_code_')

    for key, filename in ((IMAGE_KEY, 'target.png'), (RENDER_KEY, 'render.png')):
        image = context.tool_state.get(key)
        if image is not None and hasattr(image, 'save'):
            try:
                image.save(os.path.join(work, filename))
            except (OSError, ValueError):
                logger.warning('could not materialise %s into the code workdir', key)
    return work, True


def _snapshot_images(work: str) -> set[str]:
    try:
        return {f for f in os.listdir(work) if f.lower().endswith(_IMAGE_EXTS)}
    except OSError:
        return set()


def _get_worker(work: str) -> subprocess.Popen[str]:
    proc = _WORKERS.get(work)
    if proc is not None and proc.poll() is None:
        return proc
    proc = subprocess.Popen(  # noqa: S603 - fixed argv: our own interpreter + embedded source
        [sys.executable, '-c', _WORKER_SRC, '--', work],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding='utf-8',
        errors='replace',
    )
    _WORKERS[work] = proc
    return proc


async def _drop_worker(work: str, proc: subprocess.Popen[str]) -> None:
    if _WORKERS.get(work) is proc:
        _WORKERS.pop(work, None)
    if proc.poll() is None:
        proc.kill()
        await asyncio.to_thread(proc.wait)
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()


def _write_line(stdin: Any, request: str) -> None:
    stdin.write(request)
    stdin.flush()


_CODE_TOOL_DETAIL = (
    'Run Python to inspect or measure the slide images, and to measure any image you produce.\n\n'
    'Use it for whatever the fixed tools do not cover: resize to compare at the same scale, '
    "crop an arbitrary region, sample a region's dominant colour, threshold or mask to "
    'separate a shape from its background, detect edges to find an exact rule position, '
    'or just read the image dimensions before estimating coordinates.\n\n'
    'Available files, by relative path:\n'
    '  target.png  - the slide you are reconstructing\n'
    '  render.png  - your most recent render, if you have called render_slide\n\n'
    '⚠️ These are NOT in os.getcwd(). Relative paths are resolved for you in '
    '`open` and in PIL Image.open/save, so `Image.open("target.png")` works — but '
    'os.listdir("."), glob and os.path.exists will not see them. The absolute '
    'directory is in the WORK_DIR variable if you need it. Do not go hunting '
    'through /tmp for these files; use the relative name or WORK_DIR.\n\n'
    'To keep an image, save it: `img.save("crop.png")`. Writing files is expected.\n'
    'Variables persist between calls, so you can build on earlier steps.\n\n'
    'Installed: PIL (Pillow), numpy. Nothing else can be assumed — check before importing.' + _img_return_suffix('code')
)

_CODE_INTERPRETER_DESCRIPTION = (
    'Run Python. Use it for arithmetic and for drawing your own artwork.\n\n' + _CODE_TOOL_DETAIL
)


@tool_registry.register(
    'visual/ppt_run_python',
    meta={
        'name': 'run_python',
        'description': _CODE_TOOL_DETAIL,
        'tool_type': ToolType.RUNTIME,
    },
)
async def run_python_tool(params: CodeInput, context: ToolContext) -> ToolResponse:
    code = (params.code or '').strip()
    if code.startswith('```'):
        lines = code.splitlines()
        code = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
    if not code.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: empty code.'}],
            meta={'success': False},
        )

    work, work_is_new = _ensure_workdir(context)
    before = _snapshot_images(work)

    lock = _WORKER_LOCKS.setdefault(work, threading.Lock())
    await asyncio.to_thread(lock.acquire)
    out_text = ''
    err_text = ''
    read_hit = False
    failed: str | None = None
    line: str | None = None
    try:
        proc = _get_worker(work)
        stdin, stdout = proc.stdin, proc.stdout
        assert stdin is not None  # spawned with PIPE above
        assert stdout is not None
        try:
            await asyncio.to_thread(_write_line, stdin, json.dumps({'code': code}, ensure_ascii=False) + '\n')
        except (BrokenPipeError, ValueError):
            await _drop_worker(work, proc)
            failed = 'the code worker was no longer alive to receive the code.'
        else:
            timeout = params.timeout if params.timeout and params.timeout > 0 else None
            try:
                line = await asyncio.wait_for(asyncio.to_thread(stdout.readline), timeout=timeout)
            except asyncio.TimeoutError:
                await _drop_worker(work, proc)
                failed = (
                    f'TimeoutError: the call exceeded {params.timeout} second(s) and the worker process '
                    'running it was killed, so variables from earlier calls are gone with it. Files '
                    'written before the timeout persist.'
                )
        if failed is None and not line:
            await _drop_worker(work, proc)
            failed = (
                'the code worker exited before answering (it crashed, closed stdin, or wrote straight '
                'to fd 1). No output was captured.'
            )
        if failed is None and line:  # `and line` also narrows str for json.loads
            try:
                resp = json.loads(line)
            except ValueError:
                await _drop_worker(work, proc)
                failed = 'the code worker returned an unreadable response (a raw write to fd 1 corrupts it).'
            else:
                out_text = resp.get('stdout') or ''
                err_text = resp.get('stderr') or ''
                read_hit = bool(resp.get('read_target'))
                worker_failed = resp.get('failed')
                if worker_failed:
                    failed = str(worker_failed)
    finally:
        lock.release()

    parts: list[dict[str, Any]] = []
    out_s = out_text.strip()
    err_s = err_text.strip()
    if out_s:
        parts.append({'type': 'text', 'text': f'stdout:\n{_clip(out_s)}'})
    if err_s:
        parts.append({'type': 'text', 'text': f'stderr:\n{_clip(err_s, 4000)}'})
    if failed:
        if failed.startswith('TimeoutError'):
            prefix = 'Timeout'
        elif failed.lstrip().startswith('Traceback'):
            prefix = 'Traceback'
        else:
            prefix = 'Error'
        parts.append({'type': 'text', 'text': f'{prefix}:\n{failed[-2000:]}'})

    new_images = sorted(_snapshot_images(work) - before)
    shown = 0
    for filename in new_images:
        if shown >= _MAX_IMAGES_PER_CALL:
            parts.append({
                'type': 'text',
                'text': f'({len(new_images) - shown} more image(s) written but not shown)',
            })
            break
        url = _image_to_data_url(os.path.join(work, filename))
        if url is None:
            continue
        parts.append({'type': 'text', 'text': f'\n{filename}:'})
        if _tool_image_return_enabled():
            parts.append({'type': 'image_url', 'image_url': {'url': url}})
        else:
            parts.append({
                'type': 'text',
                'text': (
                    f'[image saved ({filename}); it is NOT shown back to you as a '
                    f'picture -- print the numbers you need from it, or measure it '
                    f'with the measurement tools]'
                ),
            })
        shown += 1

    if not parts:
        parts.append({
            'type': 'text',
            'text': 'Code ran with no output. Print a value or save an image to see something.',
        })

    _existing = context.tool_state.get(TOOL_RECORDS_KEY)
    records = list(_existing) if isinstance(_existing, (list, tuple)) else []
    records.append({
        'tool': 'run_python',
        'success': failed is None,
        'n_images': shown,
        'saved': list(new_images),
        'read_target': read_hit,
        'code_chars': len(code),
    })
    updates: dict[str, Any] = {TOOL_RECORDS_KEY: records}
    if work_is_new:
        updates[WORKDIR_KEY] = work

    return ToolResponse(
        output=[*parts],
        meta={'success': failed is None, 'n_images': shown, 'workdir': work},
        context_updates=updates,
    )


@tool_registry.register(
    'visual/ppt_code_interpreter',
    meta={
        'name': 'code_interpreter',
        'description': _CODE_INTERPRETER_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def code_interpreter_tool(params: CodeInterpreterInput, context: ToolContext) -> ToolResponse:
    return await run_python_tool(CodeInput(code=params.code, timeout=params.timeout), context)
