"""Regression tests for run_command's Exception path.

The except-branch used ``traceback.print_tb`` (which returns ``None`` and
prints the traceback to sys.stderr) when building the error message. When
combined with an exception whose ``str(e)`` is empty, the message that
reached the caller ended in a literal ``| None`` and carried no useful
signal about what actually failed — leaving nothing to debug.

These tests pin the fix: on any Exception, the returned ``stderr`` must
include the exception type name and the formatted traceback, and must NOT
contain the ``| None`` artifact.
"""

import asyncio
import base64
import shlex

from unittest.mock import patch

import pytest

from nexus.runtime.executors.command import EXECUTABLE_BASH, _command_for_execution_user, _powershell_args, run_command
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandRequest
from nexus.utils.platform import is_windows


async def _raise_file_not_found(*args, **kwargs):
    raise FileNotFoundError(2, "No such file or directory: '/nonexistent'")


async def _raise_bare_exception(*args, **kwargs):
    # str(e) is empty — the shape that most easily reaches the empty-message branch.
    raise Exception()


async def _raise_timeout(*args, **kwargs):
    raise asyncio.TimeoutError()


async def test_run_command_exception_includes_type_and_traceback() -> None:
    """Errors with a message must surface both the type name and the stack."""
    with patch(
        'nexus.runtime.executors.command._create_command_process',
        side_effect=_raise_file_not_found,
    ):
        response = await run_command(RunCommandRequest(command='echo hi', timeout=5))

    assert response.status == CommandRunStatus.Error
    stderr = response.stderr or ''

    # Type name is present so callers know WHAT failed.
    assert 'FileNotFoundError' in stderr
    # Original message is preserved.
    assert "No such file or directory: '/nonexistent'" in stderr
    # A real traceback body is present (Python-standard header).
    assert 'Traceback (most recent call last):' in stderr
    # The old bug artifact is gone.
    assert '| None' not in stderr


async def test_run_command_exception_with_empty_message_still_useful() -> None:
    """Exception with empty str(e): message must still name the type + carry a stack."""
    with patch(
        'nexus.runtime.executors.command._create_command_process',
        side_effect=_raise_bare_exception,
    ):
        response = await run_command(RunCommandRequest(command='echo hi', timeout=5))

    assert response.status == CommandRunStatus.Error
    stderr = response.stderr or ''

    # Even with an empty message, the type name distinguishes the failure mode.
    assert 'Exception' in stderr
    assert 'Traceback (most recent call last):' in stderr
    # Regression guard: the old branch produced a trailing `| None` and no stack.
    assert '| None' not in stderr


async def test_run_command_timeout_error_names_the_type() -> None:
    """asyncio.TimeoutError also has empty str(e); type name must appear."""
    with patch(
        'nexus.runtime.executors.command._create_command_process',
        side_effect=_raise_timeout,
    ):
        response = await run_command(RunCommandRequest(command='echo hi', timeout=5))

    assert response.status == CommandRunStatus.Error
    stderr = response.stderr or ''

    # In Python 3.11+ asyncio.TimeoutError IS TimeoutError; either name is fine.
    assert 'TimeoutError' in stderr
    assert '| None' not in stderr


async def test_run_command_success_path_still_works() -> None:
    """Sanity check: the exception path fix must not regress the happy path."""
    response = await run_command(RunCommandRequest(command='echo hello', timeout=5))

    assert response.status == CommandRunStatus.Finished
    assert response.return_code == 0
    assert (response.stdout or '').strip() == 'hello'


def test_powershell_command_is_utf16_encoded() -> None:
    args = _powershell_args("Write-Output '你好'")
    encoded = args[args.index('-EncodedCommand') + 1]
    decoded = base64.b64decode(encoded).decode('utf-16-le')

    assert "Write-Output '你好'" in decoded
    assert "$ProgressPreference = 'SilentlyContinue'" in decoded


@pytest.mark.skipif(is_windows(), reason='Unix execution-user switching requires pwd')
def test_command_for_execution_user_quotes_user_and_command() -> None:
    command = "printf '%s' dangerous"
    with patch('pwd.getpwuid') as getpwuid:
        getpwuid.return_value = type('User', (), {'pw_name': 'root'})()
        wrapped = _command_for_execution_user(command, 'agent user')

    expected = f'su {shlex.quote("agent user")} -s {shlex.quote(EXECUTABLE_BASH)} -c {shlex.quote(command)}'
    assert wrapped == expected


def test_windows_execution_user_is_explicitly_unsupported() -> None:
    with patch('nexus.runtime.executors.command.is_windows', return_value=True):
        assert _command_for_execution_user('echo hello', None) == 'echo hello'
        with pytest.raises(NotImplementedError, match='execution_user is not supported on Windows'):
            _command_for_execution_user('echo hello', 'agent')
