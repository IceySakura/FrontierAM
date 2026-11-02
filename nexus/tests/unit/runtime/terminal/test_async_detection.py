"""bash-capture skip-capture detection for common background/async commands.

Commands with high-confidence async risk are not hijacked, so CommandInfo has
empty stdout/stderr and exit_code 0. Ordinary foreground commands must not be
misclassified, because that would drop their captured output.
"""

import asyncio

import pytest

from nexus.runtime.executors.terminal.session import TerminalSession


@pytest.mark.integration
@pytest.mark.asyncio
async def test_background_command_does_not_hang():
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('sleep 5 &', timeout=10)
        assert info.exit_code == 0
        info2 = await s.run_command('echo alive', timeout=10)
        assert 'alive' in info2.stdout
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_background_command_creates_empty_data_files():
    """Background commands are not hijacked, but still materialize empty
    stdout/stderr/output files so readers do not depend on FileNotFound fallback.
    """
    import os

    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('sleep 5 &', timeout=10)
        assert info.exit_code == 0
        for name in ('stdout', 'stderr', 'output'):
            path = os.path.join(s.output_dir, f'{name}_{info.command_id}')
            assert os.path.exists(path), f'missing {path}'
            assert os.path.getsize(path) == 0, f'expected empty {path}'
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_substitution_does_not_hang():
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('cat <(echo via-procsub)', timeout=10)
        assert info.exit_code == 0
        assert info.stdout == 'via-procsub\n'
        assert info.output == 'via-procsub\n'
        info2 = await s.run_command('echo still-alive', timeout=10)
        assert 'still-alive' in info2.stdout
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('command', 'want_stdout', 'want_stderr'),
    [
        ("echo '&'", '&\n', ''),
        ("echo '<(literal)'", '<(literal)\n', ''),
        ("echo 'x=<(literal)'", 'x=<(literal)\n', ''),
        ("echo err >&2", '', 'err\n'),
        ("echo file &>/tmp/nexus-bc-redirect-test; cat /tmp/nexus-bc-redirect-test", 'file\n', ''),
        ("echo a && echo b", 'a\nb\n', ''),
    ],
)
async def test_foreground_commands_are_not_misclassified_as_background(
    command, want_stdout, want_stderr
):
    """False positives are not allowed: ordinary foreground commands must
    still be captured even when their text contains &, <(, or redirection forms.
    """
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command(command, timeout=10)
        assert info.exit_code == 0
        assert info.stdout == want_stdout
        assert info.stderr == want_stderr
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('command', 'want_stdout'),
    [
        ('sleep 0.1 & echo after-bg', 'after-bg\n'),
        ('echo before-bg; sleep 0.1 &', 'before-bg\n'),
        ('(sleep 0.1 &) ; echo after-sub-bg', 'after-sub-bg\n'),
        ('if false; then sleep 0.1 & fi; echo after-false-branch', 'after-false-branch\n'),
        ('if true; then sleep 0.1 & echo after-if-bg; fi', 'after-if-bg\n'),
    ],
)
async def test_mixed_background_and_foreground_commands_are_captured(command, want_stdout):
    """A submitted command is foreground if it has foreground work.

    The async detector must not skip capture just because some descendant node
    contains `&`; doing so drops the foreground part of mixed command lists.
    """
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command(command, timeout=10)
        assert info.exit_code == 0
        assert info.stdout == want_stdout
        assert want_stdout in info.output
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_output_no_loss():
    """~580KB output must be captured without loss (concurrent reader drains
    the pipe during execution; no 64KB deadlock)."""
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('seq 1 100000', timeout=30)
        assert info.exit_code == 0
        lines = info.stdout.strip().split('\n')
        assert lines[0] == '1'
        assert lines[-1] == '100000'
        assert len(lines) == 100000
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_arithmetic_expansion_error_finalizes_command_and_session_recovers():
    """Bash expansion errors can jump to top-level; capture must still finalize."""
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('echo BEFORE_ARITH; echo $((1/0)); echo AFTER_ARITH', timeout=10)
        assert info.end_time is not None
        assert info.exit_code != 0
        assert info.stdout == 'BEFORE_ARITH\n'
        assert 'division by 0' in info.stderr

        info2 = await s.run_command('echo AFTER_ARITH_RECOVERED', timeout=10)
        assert info2.stdout == 'AFTER_ARITH_RECOVERED\n'
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ctrl_c_finalizes_running_command_and_session_recovers():
    """SIGINT from the PTY must not leave capture status stuck running."""
    s = TerminalSession()
    await s.start()
    try:
        command_id = await s.start_command('echo SIG_START; sleep 30; echo SIG_END')
        await asyncio.sleep(0.3)
        await s.send_keys('C-c')

        info = None
        for _ in range(50):
            info = await s.get_command_info(command_id)
            if info is not None and info.end_time is not None:
                break
            await asyncio.sleep(0.1)

        assert info is not None
        assert info.end_time is not None
        assert info.exit_code != 0
        assert info.stdout == 'SIG_START\n'

        info2 = await s.run_command('echo AFTER_SIGINT_RECOVERED', timeout=10)
        assert info2.stdout == 'AFTER_SIGINT_RECOVERED\n'
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ctrl_z_finalizes_capture_and_terminate_cleans_resumed_job():
    """Stopped jobs release the prompt and keep appending output after resume."""
    import psutil

    s = TerminalSession()
    await s.start()
    try:
        command = (
            "python -c 'import signal, time; "
            "signal.signal(signal.SIGCONT, lambda *_: print(\"AFTER\", flush=True)); "
            "print(\"BEFORE\", flush=True); time.sleep(30)'"
        )
        command_id = await s.start_command(command)
        await asyncio.sleep(0.3)
        await s.send_keys('C-z')

        info = None
        for _ in range(50):
            info = await s.get_command_info(command_id)
            if info is not None and info.end_time is not None:
                break
            await asyncio.sleep(0.1)

        assert info is not None
        assert info.exit_code == 128 + 20  # Linux SIGTSTP
        assert info.stdout == 'BEFORE\n'

        recovered = await s.run_command('echo AFTER_STOP_RECOVERED', timeout=10)
        assert recovered.stdout == 'AFTER_STOP_RECOVERED\n'

        await s.run_command('bg %1', timeout=10)
        resumed = None
        for _ in range(30):
            resumed = await s.get_command_info(command_id)
            if resumed is not None and resumed.stdout == 'BEFORE\nAFTER\n':
                break
            await asyncio.sleep(0.1)
        assert resumed is not None
        assert resumed.stdout == 'BEFORE\nAFTER\n'

        jobs = await s.run_command('jobs -p', timeout=10)
        job_pids = [int(pid) for pid in jobs.stdout.split()]
        assert job_pids
        await s.terminate_session_processes()
        assert all(not psutil.pid_exists(pid) for pid in job_pids)

        alive = await s.run_command('echo SHELL_ALIVE', timeout=10)
        assert alive.stdout == 'SHELL_ALIVE\n'
    finally:
        await s.close()
