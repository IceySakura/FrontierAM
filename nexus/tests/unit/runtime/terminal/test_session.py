import os
import subprocess
import uuid

import pytest

from nexus.runtime.executors.terminal.capture import CaptureStatus
from nexus.runtime.executors.terminal.session import TerminalSession


@pytest.mark.asyncio
async def test_start_uses_patched_bash_capture_env(monkeypatch, tmp_path):
    from nexus.runtime.executors.terminal import session as session_mod

    # The session env is a copy of os.environ, so drop any ambient LD_PRELOAD (some shells
    # export one, e.g. libcuda.so from ~/.bashrc). The assertion below is about the session
    # NOT introducing a preload of its own — capture is compiled into the patched bash.
    monkeypatch.delenv('LD_PRELOAD', raising=False)

    captured: dict[str, object] = {}

    class FakePty:
        def killpg(self):
            pass

        def close(self):
            pass

    async def fake_spawn(*, argv, env, rows, cols, on_data, on_exit):
        captured['argv'] = argv
        captured['env'] = env
        return FakePty()

    async def fake_status(_reader):
        return CaptureStatus(status='idle', command_id=0)

    monkeypatch.setattr(session_mod, 'locate_patched_bash', lambda: '/tmp/patched-bash')
    monkeypatch.setattr(session_mod.PtyProcess, 'spawn', fake_spawn)
    monkeypatch.setattr(session_mod.CaptureReader, 'read_status', fake_status)

    s = TerminalSession(session_id=f'unit-capture-env-{uuid.uuid4().hex}')
    await s.start()
    try:
        env = captured['env']
        argv = captured['argv']
        assert isinstance(env, dict)
        assert isinstance(argv, list)
        assert env['NEXUS_CAPTURE'] == '1'
        assert env['NEXUS_CAPTURE_DIR'] == s.output_dir
        assert 'LD_PRELOAD' not in env
        bootstrap = argv[argv.index('--rcfile') + 1]
        capture_file = tmp_path / 'capture-session'
        bootstrap_env = {**os.environ, 'BASH_ENV': '/dev/null', 'NEXUS_TEST_CAPTURE_FILE': str(capture_file)}
        subprocess.run(
            [
                '/bin/bash',
                '--noprofile',
                '--norc',
                '-c',
                'nexus_capture() { printf "%s" "$1" > "$NEXUS_TEST_CAPTURE_FILE"; }; . "$1"',
                'capture-test',
                bootstrap,
            ],
            env=bootstrap_env,
            check=True,
        )
        assert capture_file.read_text() == s.session_id
        # The same bootstrap must also work when capture is enabled only by environment.
        subprocess.run(['/bin/bash', '--noprofile', '--norc', bootstrap], env=bootstrap_env, check=True)
        assert not os.path.exists(os.path.join(s.output_dir, 'init.sh'))
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_start_failure_cleans_up(monkeypatch):
    from nexus.runtime.executors.terminal import session as session_mod

    class FakePty:
        def killpg(self):
            pass

        def close(self):
            pass

    async def fake_spawn(*, argv, env, rows, cols, on_data, on_exit):
        return FakePty()

    async def boom(_reader):
        raise RuntimeError('forced readiness failure')

    monkeypatch.setattr(session_mod, 'locate_patched_bash', lambda: '/tmp/patched-bash')
    monkeypatch.setattr(session_mod.PtyProcess, 'spawn', fake_spawn)
    monkeypatch.setattr(session_mod.CaptureReader, 'read_status', boom)

    s = TerminalSession(session_id=f'unit-start-fail-{uuid.uuid4().hex}')
    output_dir = s.output_dir
    with pytest.raises(RuntimeError, match='forced readiness failure'):
        await s.start()

    assert s._dead is True
    assert s._pty is None
    assert not os.path.exists(output_dir)
    assert s._screen._executor._shutdown
