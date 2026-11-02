"""Kilo and OpenCode receive the task on stdin, not as a positional argument.

Both CLIs share the same ``run`` handler lineage (kilocode is a fork of opencode),
whose message assembler wraps every positional holding a space in literal quote
bytes and escapes the quotes inside it:

    let message = [...args.message, ...(args["--"] || [])]
      .map((arg) => (arg.includes(" ") ? `"${arg.replace(/"/g, '\\"')}"` : arg))
      .join(" ")

A whole SWE problem statement is one such positional, so passing it that way put a
mangled copy of the task in front of the model -- ``#[error("...")]`` reached it as
``#[error(\\"...\\")]``.  The stdin path in the same handler is taken verbatim.
"""

import sys

import pytest

from nexus.extensions.agents.kilo.runtime_setup import _run_kilo_streaming
from nexus.extensions.agents.opencode.runtime_setup import _run_opencode_streaming


ECHO_STDIN = [sys.executable, '-c', 'import sys; sys.stdout.write(sys.stdin.read())']
COUNT_STDIN = [sys.executable, '-c', 'import sys; sys.stdout.write(str(len(sys.stdin.read())))']

# a task the old positional path would have mangled: spaces trigger the wrap, inner quotes get escaped
TASK = '\n'.join(
    [
        'Fix the derive macro so this compiles:',
        '',
        '#[derive(Error, Debug)]',
        '#[error("bad field {0}", r#fn)]',
        'pub struct E(String);',
    ]
)


@pytest.mark.asyncio
async def test_kilo_gets_the_task_on_stdin_verbatim(tmp_path) -> None:
    exit_code, stdout, _ = await _run_kilo_streaming(
        cmd=ECHO_STDIN,
        task=TASK,
        cwd=str(tmp_path),
        env={},
        timeout=30,
    )

    assert exit_code == 0
    assert stdout == TASK
    assert not stdout.startswith('"')
    assert '\\"' not in stdout


@pytest.mark.asyncio
async def test_opencode_gets_the_task_on_stdin_verbatim(tmp_path) -> None:
    exit_code, stdout, _ = await _run_opencode_streaming(
        cmd=ECHO_STDIN,
        task=TASK,
        cwd=str(tmp_path),
        env={},
        timeout=30,
    )

    assert exit_code == 0
    assert stdout == TASK
    assert not stdout.startswith('"')
    assert '\\"' not in stdout


@pytest.mark.asyncio
async def test_stdin_is_closed_so_the_cli_stops_waiting_for_input(tmp_path) -> None:
    """Both CLIs read stdin to EOF before starting the session, so an unclosed pipe would hang the run."""
    exit_code, stdout, _ = await _run_kilo_streaming(
        cmd=COUNT_STDIN,
        task=TASK,
        cwd=str(tmp_path),
        env={},
        timeout=30,
    )

    assert exit_code == 0
    assert stdout == str(len(TASK))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'payload',
    [
        'trailing backslash \\',
        'a NUL \x00 byte',  # would have raised ValueError as an argv element
        'CJK 中文 emoji 🙂 combining é',
        'shell metachars $(id) `id` ${HOME} | & ; > <',
        'quotes " \' and an escaped \\" pair',
        '\n\n leading and trailing whitespace \t\n',
    ],
    ids=['backslash', 'nul', 'unicode', 'metachars', 'quotes', 'whitespace'],
)
async def test_any_payload_survives_the_handover(tmp_path, payload: str) -> None:
    """No shell and no argv sit between us and the CLI, so the bytes are whatever we wrote."""
    _, stdout, _ = await _run_kilo_streaming(cmd=COUNT_STDIN, task=payload, cwd=str(tmp_path), env={}, timeout=30)

    assert stdout == str(len(payload))


@pytest.mark.asyncio
async def test_a_task_beyond_the_pipe_buffer_still_gets_through(tmp_path) -> None:
    """A 1 MiB task exceeds both the 64 KiB pipe buffer and the 128 KiB per-argument limit the old positional had."""
    payload = 'x' * (1024 * 1024)

    exit_code, stdout, _ = await _run_kilo_streaming(
        cmd=COUNT_STDIN, task=payload, cwd=str(tmp_path), env={}, timeout=60
    )

    assert exit_code == 0
    assert stdout == str(len(payload))


@pytest.mark.asyncio
async def test_a_cli_that_exits_before_reading_reports_its_own_failure(tmp_path) -> None:
    """A broken pipe must not mask the exit code and stderr that say why the CLI died."""
    exit_code, _, stderr = await _run_kilo_streaming(
        cmd=[sys.executable, '-c', 'import sys; sys.stderr.write("boom"); sys.exit(3)'],
        task='x' * (1024 * 1024),
        cwd=str(tmp_path),
        env={},
        timeout=30,
    )

    assert exit_code == 3
    assert 'boom' in stderr
    assert 'stdin write failed' not in stderr
