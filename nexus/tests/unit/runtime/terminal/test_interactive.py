import asyncio

import pytest

from nexus.runtime.executors.terminal.session import TerminalSession


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interactive_python_repl():
    s = TerminalSession()
    await s.start()
    try:
        await s.send_keys(['python3 -q', 'Enter'])
        await asyncio.sleep(0.5)
        await s.send_keys(['print(6*7)', 'Enter'])
        await asyncio.sleep(0.5)
        pane = await s.capture_pane()
        assert '42' in pane
        await s.send_keys(['exit()', 'Enter'])
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ctrl_c_interrupts_running_command():
    s = TerminalSession()
    await s.start()
    try:
        await s.send_keys(['sleep 30', 'Enter'])
        await asyncio.sleep(0.3)
        await s.send_keys('C-c')
        await asyncio.sleep(0.3)
        info = await s.run_command('echo back', timeout=10)
        assert 'back' in info.stdout
    finally:
        await s.close()
