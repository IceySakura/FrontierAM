import asyncio
import json
import os
import signal
import sys

import pytest

from nexus.extensions.agents.workbuddy import runtime_setup


@pytest.mark.asyncio
async def test_oversized_json_line_does_not_kill_the_output_pump() -> None:
    """A single --json ACP frame can blow past the default 64 KiB reader limit."""
    script = 'print("x" * (128 * 1024)); print("after the big line")'
    cmd = [sys.executable, '-c', script]

    exit_code, stdout, _stderr = await runtime_setup._run_workbuddy_streaming(
        cmd=cmd,
        cwd=os.getcwd(),
        env={},
        timeout=60,
    )

    assert exit_code == 0
    # The big line survives (16 MiB limit) and the pump keeps going afterwards.
    assert len(stdout) > 128 * 1024
    assert 'after the big line' in stdout


@pytest.mark.asyncio
async def test_timeout_kills_the_whole_process_group() -> None:
    """wb spawns a daemon; killing only the node parent would orphan it."""
    script = (
        'import os, subprocess, sys, time\n'
        'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])\n'
        'print(child.pid, flush=True)\n'
        'time.sleep(120)\n'
    )
    exit_code, stdout, stderr = await runtime_setup._run_workbuddy_streaming(
        cmd=[sys.executable, '-c', script],
        cwd=os.getcwd(),
        env={},
        timeout=2,
    )

    assert exit_code == -1
    assert '[timeout]' in stderr

    child_pid = int(stdout.strip().splitlines()[0])
    for _ in range(50):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail(f'grandchild {child_pid} survived the timeout kill')


@pytest.mark.asyncio
async def test_stdin_is_closed_so_a_promptless_wb_cannot_hang() -> None:
    """Without a prompt wb reads stdin; an inherited stdin would hang until timeout."""
    script = 'import sys; data = sys.stdin.read(); print(f"read {len(data)} bytes")'
    exit_code, stdout, _stderr = await runtime_setup._run_workbuddy_streaming(
        cmd=[sys.executable, '-c', script],
        cwd=os.getcwd(),
        env={},
        timeout=10,
    )
    assert exit_code == 0
    assert 'read 0 bytes' in stdout


@pytest.mark.asyncio
async def test_streaming_returns_json_lines_parseable_by_the_acp_parser() -> None:
    turn = {'stopReason': 'end_turn', '_meta': {'codebuddy.ai/outcome': 'SUCCESS'}}
    update = {
        'sessionId': 's',
        'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': 'pong'}},
    }
    script = f'print({json.dumps(json.dumps(update))}); print({json.dumps(json.dumps(turn))})'

    _exit_code, stdout, _stderr = await runtime_setup._run_workbuddy_streaming(
        cmd=[sys.executable, '-c', script],
        cwd=os.getcwd(),
        env={},
        timeout=30,
    )

    updates, results = runtime_setup.parse_wb_json_lines(stdout)
    assert runtime_setup.assistant_text(updates) == 'pong'
    assert runtime_setup.task_outcome(results[-1]) == 'SUCCESS'
