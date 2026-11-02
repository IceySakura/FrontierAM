"""The user-memory file a run hands its worker.

Claude Code reads ``~/.claude/CLAUDE.md`` when a session starts, so what is placed
there reaches the worker's main agent *and* every subagent it spawns --
``--append-system-prompt`` reaches only the main agent. Three things therefore have
to hold, and none of them is visible in an export after the fact:

* the file is written before the worker's first turn, because Claude reads it at
  startup and a PTY session lives for the whole run;
* it is written once per sandbox rather than once per session, because every session
  in one container shares the home it lands in;
* a failed write stops the run, because a worker that never saw its instructions
  produces a run that looks fine and is not;
* the path comes from the manager whose harness will read it, so a harness reading
  somewhere else is given somewhere else.
"""

import asyncio

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.tasks.hitl.actor.memory import (
    CONFIG_KEY,
    memory_text,
    place_user_memory,
    write_user_memory,
)
from nexus.extensions.tasks.hitl.actor.print_claude_code.spec import MEMORY_PATH as CLAUDE_CODE_MEMORY_PATH


MEMORY = '## How I like to work\n\nSay what you verified and what you did not.\n'


class _Result:
    def __init__(self, path: str, *, success: bool = True, message: str = '') -> None:
        self.path = path
        self.success = success
        self.message = message


class _Response:
    def __init__(self, results: list[_Result]) -> None:
        self.results = results


class RecordingRuntime:
    """Records the order of uploads and sandbox invocations.

    Ordering is the point: a write that lands after the first turn is a write
    Claude never read.
    """

    default_venv_id = None

    def __init__(self, *, write_ok: bool = True) -> None:
        self.events: list[str] = []
        self.uploads: list[Any] = []
        self.written: list[str] = []
        self.paths: list[str] = []
        self.write_ok = write_ok

    async def upload_file(self, files: list[Any], **_kwargs: Any) -> _Response:
        self.events.append('upload_file')
        self.uploads.extend(files)
        return _Response(
            [_Result(item.path, success=self.upload_ok, message='' if self.upload_ok else 'no space') for item in files]
        )

    async def invoke(self, target: Any, /, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        name = getattr(target, '__name__', str(target))
        self.events.append(name)
        if name == 'write_user_memory':
            if not self.write_ok:
                raise RuntimeError('no space left on device')
            self.written.append(kwargs['text'])
            self.paths.append(kwargs['memory_path'])
            return {'path': f'/sandbox/home/{kwargs["memory_path"]}', 'chars': str(len(kwargs['text']))}
        return {'session_id': kwargs.get('session_id', ''), 'label': kwargs.get('label', ''), 'proxy_port': 9001}

    async def ws_connect(self) -> None:
        self.events.append('ws_connect')

    def on(self, event: str, _handler: Any) -> None:
        pass

    def off(self, event: str, _handler: Any = None) -> None:
        pass

    async def emit(self, event: str, payload: Any = None) -> None:
        pass


# -- what counts as having been given a memory file ------------------------


def test_a_run_given_none_writes_nothing() -> None:
    """The common case: no memory configured, so the sandbox is left alone."""
    runtime = RecordingRuntime()

    asyncio.run(place_user_memory(runtime, {}, memory_path=CLAUDE_CODE_MEMORY_PATH))

    assert runtime.written == []


@pytest.mark.parametrize('blank', ['', '   ', '\n\t\n'])
def test_a_blank_memory_is_treated_as_none(blank: str) -> None:
    """An all-whitespace file is one nobody finished writing."""
    runtime = RecordingRuntime()

    asyncio.run(place_user_memory(runtime, {CONFIG_KEY: blank}, memory_path=CLAUDE_CODE_MEMORY_PATH))

    assert runtime.written == []
    assert memory_text({CONFIG_KEY: blank}) == ''


def test_the_memory_reaches_the_sandbox_verbatim() -> None:
    runtime = RecordingRuntime()

    asyncio.run(place_user_memory(runtime, {CONFIG_KEY: MEMORY}, memory_path=CLAUDE_CODE_MEMORY_PATH))

    assert runtime.written == [MEMORY]


def test_the_path_follows_the_sandbox_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The hook settings are placed under the sandbox's own HOME, and this must match.

    A host-side path would write somewhere Claude never looks, and the write would
    still report success.
    """
    monkeypatch.setenv('HOME', str(tmp_path))

    result = write_user_memory(text=MEMORY, memory_path=CLAUDE_CODE_MEMORY_PATH)

    written = tmp_path / CLAUDE_CODE_MEMORY_PATH
    assert written.read_text(encoding='utf-8') == MEMORY
    assert result['path'] == str(written)


def test_a_failed_write_stops_the_run() -> None:
    """Continuing would produce a worker that silently never saw its instructions."""
    runtime = RecordingRuntime(write_ok=False)

    with pytest.raises(RuntimeError, match='could not place the user memory'):
        asyncio.run(place_user_memory(runtime, {CONFIG_KEY: MEMORY}, memory_path=CLAUDE_CODE_MEMORY_PATH))


# -- the path is the harness's to declare ----------------------------------


def test_a_harness_that_reads_elsewhere_is_written_elsewhere() -> None:
    """Nothing here decides on CLAUDE.md; a harness reading AGENTS.md says so."""
    runtime = RecordingRuntime()

    asyncio.run(place_user_memory(runtime, {CONFIG_KEY: MEMORY}, memory_path='AGENTS.md'))

    assert runtime.paths == ['AGENTS.md']


def test_a_harness_that_reads_nowhere_writes_nothing() -> None:
    """A harness with no such file leaves the sandbox alone rather than guessing."""
    runtime = RecordingRuntime()

    asyncio.run(place_user_memory(runtime, {CONFIG_KEY: MEMORY}, memory_path=''))

    assert runtime.written == []


def test_every_registered_backend_declares_where_its_harness_reads() -> None:
    """The path comes from the backend's own spec, not from one harness's default."""
    from nexus.extensions.tasks.hitl.actor.factory import backends, make_worker_manager
    from nexus.extensions.tasks.hitl.actor.print_codex.spec import MEMORY_PATH as CODEX_MEMORY_PATH

    declared = {
        'print_claude_code': CLAUDE_CODE_MEMORY_PATH,
        'pty_claude_code': CLAUDE_CODE_MEMORY_PATH,
        'print_codex': CODEX_MEMORY_PATH,
    }

    assert set(backends()) == set(declared)
    for backend in backends():
        assert make_worker_manager(backend).memory_path == declared[backend], backend


# -- the ordering, per backend --------------------------------------------


def test_the_print_backend_writes_it_before_any_turn() -> None:
    """Each print turn is its own Claude process, and every one reads the file."""
    from nexus.extensions.tasks.hitl.actor.print_claude_code.backend import LabelledPrintBackend

    runtime = RecordingRuntime()
    backend = LabelledPrintBackend(
        {CONFIG_KEY: MEMORY, 'working_dir': '/workspace'},
        memory_path=CLAUDE_CODE_MEMORY_PATH,
    )

    asyncio.run(backend.open_session(runtime, session_id='s1', label='worker/_/w', working_dir='/testbed'))

    assert 'write_user_memory' in runtime.events, 'the memory was never written'
    assert runtime.events.index('write_user_memory') < runtime.events.index('open_session')
    assert runtime.written == [MEMORY]
    assert runtime.paths == [CLAUDE_CODE_MEMORY_PATH]


def test_the_print_backend_writes_it_once_per_manager() -> None:
    """``ensure_env`` is idempotent; the write must not repeat per session."""
    from nexus.extensions.tasks.hitl.actor.print_claude_code.backend import LabelledPrintBackend

    runtime = RecordingRuntime()
    backend = LabelledPrintBackend({CONFIG_KEY: MEMORY}, memory_path=CLAUDE_CODE_MEMORY_PATH)

    asyncio.run(backend.open_session(runtime, session_id='s1', label='worker/_/a', working_dir='/testbed'))
    asyncio.run(backend.open_session(runtime, session_id='s2', label='worker/_/b', working_dir='/testbed'))

    assert runtime.events.count('write_user_memory') == 1


def _pty_manager(runtime: RecordingRuntime, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A PTY manager whose ``setup()`` reaches ``runtime`` instead of a sandbox."""
    from nexus.extensions.tasks.hitl.actor.pty_claude_code.manager import PtyClaudeCodeSessionManager

    async def get_runtime(name='main') -> RecordingRuntime:
        assert name == 'main'
        return runtime

    monkeypatch.setattr(
        'nexus.extensions.tasks.hitl.actor.pty_claude_code.manager.require_context',
        lambda: SimpleNamespace(get_runtime=get_runtime, get_openai_client=lambda _name='main': object()),
    )
    return PtyClaudeCodeSessionManager(default_config={CONFIG_KEY: MEMORY}, llm_handler=object())


def test_the_pty_backend_writes_it_before_any_session_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PTY session lives for the whole run, so after it opens is too late."""
    runtime = RecordingRuntime()
    manager = _pty_manager(runtime, monkeypatch)

    asyncio.run(manager.setup())
    asyncio.run(manager.start(name='worker'))

    assert 'write_user_memory' in runtime.events, 'the memory was never written'
    opened = next(index for index, name in enumerate(runtime.events) if name == 'open_session')
    assert runtime.events.index('write_user_memory') < opened
    assert runtime.written == [MEMORY]
    assert runtime.paths == [CLAUDE_CODE_MEMORY_PATH]


def test_the_pty_backend_writes_it_once_however_many_sessions_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """One home, one file: N sessions must not write it N times."""
    runtime = RecordingRuntime()
    manager = _pty_manager(runtime, monkeypatch)

    asyncio.run(manager.setup())
    for name in ('a', 'b', 'c'):
        asyncio.run(manager.start(name=name))

    assert runtime.events.count('write_user_memory') == 1
