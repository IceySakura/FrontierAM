"""Multi-line commands are aggregated by batch (no { } wrapping), so capture_pane
shows the agent's original text. Needs the patched bash (NEXUS_BASH_BIN)."""

import asyncio

import pytest

from nexus.runtime.executors.terminal.session import TerminalSession


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiline_aggregates_to_one_commandinfo():
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('echo A\necho B\necho C', timeout=10)
        assert info.exit_code == 0
        assert info.stdout.strip().splitlines() == ['A', 'B', 'C']
        assert info.command == 'echo A\necho B\necho C'   # original text, not { }
        # next command still works after a multi-line batch
        info2 = await s.run_command('echo next', timeout=10)
        assert 'next' in info2.stdout
    finally:
        await s.close()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_capture_pane_has_no_brace_wrapping():
    s = TerminalSession()
    await s.start()
    try:
        await s.run_command('echo 123\necho 456', timeout=10)
        pane = await s.capture_pane()
        # The core win: screen shows the agent's lines, not { ...; } / > prompts.
        assert '{ echo' not in pane
        assert '\n> ' not in pane and not pane.startswith('> ')
        assert '123' in pane and '456' in pane
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiline_with_sleep_does_not_finish_early():
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('echo A\nsleep 0.3\necho C', timeout=10)
        assert info.exit_code == 0
        assert info.stdout.strip().splitlines() == ['A', 'C']
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiline_mid_failure_continues_and_takes_last_exit():
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('echo A\nfalse\necho C', timeout=10)
        # all three run; exit code is the LAST command's (echo C -> 0)
        assert info.exit_code == 0
        assert 'A' in info.stdout and 'C' in info.stdout
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_query_path_aggregates():
    """Mimic codeact: start_command then poll query until end_time is set."""
    s = TerminalSession()
    await s.start()
    try:
        first_id = await s.start_command('echo A\necho B\necho C')
        assert first_id != 0
        info = None
        for _ in range(200):
            info = await s.get_command_info(first_id)
            assert info is not None
            if info.end_time is not None:
                break
            await asyncio.sleep(0.02)
        assert info is not None and info.end_time is not None
        assert info.command_id == first_id
        assert info.stdout.strip().splitlines() == ['A', 'B', 'C']
        assert info.exit_code == 0
    finally:
        await s.close()
