from __future__ import annotations

from nexus.extensions.agents.claude_code.native_bridge.adapters.shell import (
    build_claude_code_command,
    build_claude_code_env,
    build_pre_start_command,
)


def test_build_claude_code_env_points_to_local_bridge() -> None:
    env = build_claude_code_env(local_bridge_base_url='http://127.0.0.1:8080', extra_env={'A': 'B'})

    assert env['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:8080'
    assert env['ANTHROPIC_API_KEY'] == 'dummy-key-for-proxy'
    assert env['CLAUDE_ENABLE_STREAM_WATCHDOG'] == '0'
    assert env['A'] == 'B'


def test_build_claude_code_env_allows_stream_watchdog_override() -> None:
    env = build_claude_code_env(
        local_bridge_base_url='http://127.0.0.1:8080',
        extra_env={'CLAUDE_ENABLE_STREAM_WATCHDOG': '1'},
    )

    assert env['CLAUDE_ENABLE_STREAM_WATCHDOG'] == '1'


def test_build_claude_code_command_uses_print_and_task() -> None:
    cmd = build_claude_code_command('/tmp/claude', task='create answer.txt', extra_cli_args=['--output-format', 'json'])

    assert cmd[:2] == ['/tmp/claude', '--dangerously-skip-permissions']
    assert '--print' in cmd
    assert cmd[-2:] == ['--', 'create answer.txt']


def test_build_pre_start_command_bash_bakes_cwd_and_env() -> None:
    cmd = build_pre_start_command('bash', 'echo hi', cwd='/tmp/work', env={'FOO': 'bar'})

    assert 'mkdir -p /tmp/work' in cmd
    assert 'cd /tmp/work' in cmd
    assert 'FOO=bar' in cmd
    assert 'bash -c' in cmd


def test_build_pre_start_command_python_uses_python3() -> None:
    cmd = build_pre_start_command('python', 'print(1)', cwd='/tmp/work', env={})

    assert 'python3 -c' in cmd


def test_build_pre_start_command_rejects_unknown_interpreter() -> None:
    import pytest

    with pytest.raises(ValueError, match='Unsupported pre-start interpreter'):
        build_pre_start_command('ruby', 'puts 1', cwd='/tmp', env={})
