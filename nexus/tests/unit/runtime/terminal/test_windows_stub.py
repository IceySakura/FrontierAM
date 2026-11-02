import pytest

from nexus.utils.platform import is_windows


pytestmark = pytest.mark.skipif(not is_windows(), reason='Windows-only terminal stub')


async def test_terminal_session_executors_fail_explicitly_on_windows() -> None:
    from nexus.runtime.executors.terminal import (
        capture_pane,
        create_terminal_session,
        destroy_terminal_session,
        query_command_status,
        run_command_in_session,
        send_keys_to_session,
        start_command_in_session,
        terminate_session_processes,
    )

    executors = [
        capture_pane,
        create_terminal_session,
        destroy_terminal_session,
        query_command_status,
        run_command_in_session,
        send_keys_to_session,
        start_command_in_session,
        terminate_session_processes,
    ]
    for executor in executors:
        with pytest.raises(NotImplementedError, match='not implemented on Windows'):
            await executor(object())
