"""Tests for the code_tool worker-process architecture.

These exist because the review round asked for non-hand-made verification of
the three follow-up issues: (1) a timed-out call must not leave the host's
stdout hijacked or the codex MCP server wedged, (2) vendored sys.path edits
must not leak (covered in test_font_tools), (3) render.png must track the most
recent render. They also cover code_interpreter, which had no tests at all.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time

from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pytest  # noqa: E402

from nexus.extensions.tasks.ppt_rubric.ppt_tools import code_tool as ct  # noqa: E402
from nexus.extensions.tasks.ppt_rubric.ppt_tools.code_tool import (  # noqa: E402
    WORKDIR_KEY,
    CodeInput,
    CodeInterpreterInput,
    code_interpreter_tool,
    run_python_tool,
)

from PIL import Image  # noqa: E402


def _context(*, seeded: bool = True, workdir: str | None = None) -> SimpleNamespace:
    state: dict = {}
    if seeded:
        state[ct.IMAGE_KEY] = Image.new('RGB', (24, 16), '#123456')
        state[ct.RENDER_KEY] = Image.new('RGB', (24, 16), '#654321')
    if workdir is not None:
        state[WORKDIR_KEY] = workdir
    return SimpleNamespace(tool_state=state)


def _texts(resp) -> str:
    return '\n'.join(p.get('text', '') for p in resp.output if p.get('type') == 'text')


def _pixel(path: Path) -> tuple:
    with Image.open(path) as img:
        return list(img.getdata())[0]


def _absorb(ctx, resp) -> None:
    """Mimic the harness absorbing context_updates back into tool_state.

    Both real consumers keep the workdir stable across calls: the codex MCP
    server calls _absorb(response.context_updates) after every tool call
    (ppt_mcp_server.py), and the built-in loop pre-seeds WORKDIR_KEY from
    task_input.working_dir before the first call (task.py). A bare
    SimpleNamespace does neither, so a test making a second call without this
    would let the tool lazily create a FRESH workdir -- silently defeating the
    persistence the test claims to check. That is not hypothetical: the first
    version of the persistence test passed with a NameError traceback because
    the worker's "line 104" frame satisfied `assert '10' in texts`.
    """
    ctx.tool_state.update(resp.context_updates)


@pytest.fixture(autouse=True)
def _reap_workers():
    yield
    for _work, proc in list(ct._WORKERS.items()):
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
    ct._WORKERS.clear()
    ct._WORKER_LOCKS.clear()


@pytest.mark.asyncio
async def test_output_and_variable_persistence():
    ctx = _context()
    r1 = await run_python_tool(CodeInput(code='print("hello")\nx = 5'), ctx)
    assert 'hello' in _texts(r1)
    assert r1.meta['success'] is True
    _absorb(ctx, r1)

    r2 = await run_python_tool(CodeInput(code='print(x * 2)'), ctx)
    # success must be asserted too: a NameError would leave '10' absent from
    # stdout, but substring checks can be satisfied by traceback line numbers.
    assert r2.meta['success'] is True, _texts(r2)
    assert '10' in _texts(r2)
    assert r2.meta['workdir'] == r1.meta['workdir']


@pytest.mark.asyncio
async def test_timeout_kills_worker_and_leaves_host_stdout_intact():
    ctx = _context()
    stdout_before = sys.stdout

    t0 = time.monotonic()
    r = await run_python_tool(CodeInput(code='import time\ntime.sleep(30)', timeout=1.0), ctx)
    elapsed = time.monotonic() - t0

    assert elapsed < 10, f'timeout did not cut the call short ({elapsed:.1f}s)'
    txt = _texts(r)
    assert 'TimeoutError' in txt
    assert 'killed' in txt
    # The review's core complaint: redirect_stdout used to leave sys.stdout
    # pointing at a StringIO forever. With the worker process, the host's
    # stdout object is never touched at all.
    assert sys.stdout is stdout_before
    # The poisoned worker must be gone so the next call cannot inherit it.
    assert r.meta['workdir'] not in ct._WORKERS
    _absorb(ctx, r)

    r2 = await run_python_tool(CodeInput(code='print("after")'), ctx)
    assert 'after' in _texts(r2)
    assert r2.meta['success'] is True


@pytest.mark.asyncio
async def test_same_workdir_calls_are_serialised():
    # Two calls share one pre-seeded workdir (as they do once the harness has
    # applied WORKDIR_KEY). Their captures must not interleave.
    work = tempfile.mkdtemp(prefix='pptx_code_test_')
    ctx = _context(workdir=work)
    code_a = 'import time\nfor i in range(8):\n    print("A%d" % i)\n    time.sleep(0.02)'
    code_b = 'import time\nfor i in range(8):\n    print("B%d" % i)\n    time.sleep(0.02)'

    ra, rb = await asyncio.gather(
        run_python_tool(CodeInput(code=code_a), ctx),
        run_python_tool(CodeInput(code=code_b), ctx),
    )

    ta, tb = _texts(ra), _texts(rb)
    assert 'B' not in ta
    assert 'A' not in tb
    assert ra.meta['success'] and rb.meta['success']


@pytest.mark.asyncio
async def test_render_png_tracks_the_most_recent_render():
    ctx = _context()
    r1 = await run_python_tool(CodeInput(code='print("one")'), ctx)
    work = Path(r1.meta['workdir'])
    assert _pixel(work / 'render.png') == (0x65, 0x43, 0x21)
    _absorb(ctx, r1)

    # render_slide produces a new render; the next call must show it, not the
    # first-ever one (the exists()-skip bug the review caught).
    ctx.tool_state[ct.RENDER_KEY] = Image.new('RGB', (24, 16), '#ffffff')
    r2 = await run_python_tool(CodeInput(code='print("two")'), ctx)
    assert r2.meta['workdir'] == str(work)
    assert _pixel(work / 'render.png') == (0xFF, 0xFF, 0xFF)
    # target.png is constant for the slide and must stay the seeded target.
    assert _pixel(work / 'target.png') == (0x12, 0x34, 0x56)


@pytest.mark.asyncio
async def test_existing_workdir_gets_missing_target_png():
    work = tempfile.mkdtemp(prefix='pptx_code_test_')
    ctx = _context(workdir=work)  # dir exists, no images in it

    r = await run_python_tool(CodeInput(code='print("x")'), ctx)

    assert (Path(work) / 'target.png').exists()
    assert _pixel(Path(work) / 'target.png') == (0x12, 0x34, 0x56)
    assert (Path(work) / 'render.png').exists()
    assert r.meta['success'] is True


@pytest.mark.asyncio
async def test_reading_target_png_is_recorded():
    ctx = _context()
    r = await run_python_tool(CodeInput(code='from PIL import Image\nprint(Image.open("target.png").size)'), ctx)
    assert r.meta['success'] is True
    rec = r.context_updates[ct.TOOL_RECORDS_KEY][-1]
    assert rec['read_target'] is True


@pytest.mark.asyncio
async def test_code_interpreter_delegates_and_ignores_language():
    ctx = _context()
    r = await code_interpreter_tool(CodeInterpreterInput(code='print("ci")\ny = 3', language='rust', timeout=30), ctx)
    assert 'ci' in _texts(r)
    assert r.meta['success'] is True
    _absorb(ctx, r)

    r2 = await code_interpreter_tool(CodeInterpreterInput(code='print(y)'), ctx)
    assert r2.meta['success'] is True, _texts(r2)
    assert '3' in _texts(r2)


@pytest.mark.asyncio
async def test_timeout_reported_for_code_interpreter_too():
    ctx = _context()
    r = await code_interpreter_tool(CodeInterpreterInput(code='while True:\n    pass', timeout=0.5), ctx)
    assert 'TimeoutError' in _texts(r)
    assert r.meta['success'] is False


@pytest.mark.asyncio
async def test_worker_exits_when_idle(monkeypatch):
    monkeypatch.setenv(ct._WORKER_IDLE_ENV, '1')
    ctx = _context()
    r = await run_python_tool(CodeInput(code='print("hi")'), ctx)
    proc = ct._WORKERS[r.meta['workdir']]
    # The worker must not outlive an abandoned session by more than the idle
    # limit -- this is what keeps the built-in runtime from accumulating one
    # live worker per finished task.
    await asyncio.to_thread(proc.wait, 10)
