import asyncio
import os
import signal

import pytest

from nexus.runtime.executors.terminal.pty_io import PtyProcess


@pytest.mark.asyncio
async def test_spawn_rejects_invalid_dimensions_before_opening_pty(monkeypatch):
    opened = False

    def fake_openpty():
        nonlocal opened
        opened = True
        raise AssertionError('openpty should not be called')

    monkeypatch.setattr('nexus.runtime.executors.terminal.pty_io.pty.openpty', fake_openpty)

    with pytest.raises(ValueError, match='between 1 and 65535'):
        await PtyProcess.spawn(
            argv=['/bin/true'],
            env={'PATH': '/usr/bin:/bin'},
            rows=-1,
            cols=40,
            on_data=lambda _data: None,
            on_exit=lambda: None,
        )

    assert opened is False


@pytest.mark.asyncio
async def test_spawn_closes_pty_descriptors_when_subprocess_creation_fails(monkeypatch):
    master, slave = os.openpty()
    monkeypatch.setattr('nexus.runtime.executors.terminal.pty_io.pty.openpty', lambda: (master, slave))

    async def fail_to_spawn(*_args, **_kwargs):
        raise RuntimeError('forced spawn failure')

    monkeypatch.setattr(
        'nexus.runtime.executors.terminal.pty_io.asyncio.create_subprocess_exec',
        fail_to_spawn,
    )

    with pytest.raises(RuntimeError, match='forced spawn failure'):
        await PtyProcess.spawn(
            argv=['/bin/true'],
            env={'PATH': '/usr/bin:/bin'},
            rows=10,
            cols=40,
            on_data=lambda _data: None,
            on_exit=lambda: None,
        )

    for fd in (master, slave):
        with pytest.raises(OSError, match='Bad file descriptor'):
            os.fstat(fd)


@pytest.mark.asyncio
async def test_spawn_echo_and_read():
    chunks: list[bytes] = []
    exited = asyncio.Event()
    proc = await PtyProcess.spawn(
        argv=['/bin/sh', '-c', 'printf hello'],
        env={'PATH': '/usr/bin:/bin'},
        rows=10,
        cols=40,
        on_data=chunks.append,
        on_exit=exited.set,
    )
    await asyncio.wait_for(exited.wait(), timeout=5)
    await asyncio.sleep(0.05)
    proc.close()
    assert b'hello' in b''.join(chunks)


@pytest.mark.asyncio
async def test_write_then_read_back():
    chunks: list[bytes] = []
    proc = await PtyProcess.spawn(
        argv=['/bin/cat'],
        env={'PATH': '/usr/bin:/bin'},
        rows=10,
        cols=40,
        on_data=chunks.append,
        on_exit=lambda: None,
    )
    await proc.write(b'ping\n')
    await asyncio.sleep(0.1)
    proc.killpg()
    proc.close()
    assert b'ping' in b''.join(chunks)


@pytest.mark.asyncio
async def test_terminate_descendants_continues_terminates_then_kills_survivors(monkeypatch):
    proc = object.__new__(PtyProcess)
    targets = iter((({201, 202}, {301}), ({202}, {301})))
    monkeypatch.setattr(proc, '_descendant_signal_targets', lambda: next(targets))

    sent: list[tuple[frozenset[int], frozenset[int], int]] = []
    monkeypatch.setattr(
        proc,
        '_signal_targets',
        lambda pgids, pids, sig: sent.append((frozenset(pgids), frozenset(pids), sig)),
    )

    result = await proc.terminate_descendants(grace_seconds=0)

    assert result == [201, 202]
    assert sent == [
        (frozenset({201, 202}), frozenset({301}), signal.SIGCONT),
        (frozenset({201, 202}), frozenset({301}), signal.SIGTERM),
        (frozenset({202}), frozenset({301}), signal.SIGKILL),
    ]


def test_descendant_signal_targets_excludes_shell_group_but_keeps_child_pids(monkeypatch):
    proc = object.__new__(PtyProcess)
    proc.proc = type('Proc', (), {'pid': 100})()
    proc.shell_pgid = 100

    children = [type('Child', (), {'pid': pid})() for pid in (101, 102, 103)]
    monkeypatch.setattr(
        'nexus.runtime.executors.terminal.pty_io.psutil.Process',
        lambda _pid: type('Parent', (), {'children': lambda self, recursive: children})(),
    )
    pgids = {101: 201, 102: 201, 103: 100}
    monkeypatch.setattr('nexus.runtime.executors.terminal.pty_io.os.getpgid', lambda pid: pgids[pid])

    assert proc._descendant_signal_targets() == ({201}, {103})
