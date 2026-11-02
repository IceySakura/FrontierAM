import pytest

from nexus.runtime.executors.terminal.session import TerminalSession


@pytest.mark.integration
@pytest.mark.asyncio
async def test_exit_code_immune_to_prompt_command_tampering():
    s = TerminalSession()
    await s.start()
    try:
        await s.run_command('unset PROMPT_COMMAND; trap - DEBUG', timeout=10)
        info = await s.run_command('false', timeout=10)
        assert info.exit_code == 1
        info2 = await s.run_command('true', timeout=10)
        assert info2.exit_code == 0
    finally:
        await s.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiline_command_runs_as_one():
    s = TerminalSession()
    await s.start()
    try:
        info = await s.run_command('a=1\nb=2\necho $((a+b))', timeout=10)
        assert '3' in info.stdout
        assert info.exit_code == 0
    finally:
        await s.close()
