import asyncio
import os
import signal
import sys
import time

from pathlib import Path

import pytest

from nexus.extensions.agents.workbuddy import runtime_setup
from nexus.extensions.agents.workbuddy.runtime_setup import (
    build_models_config,
    build_thinking_settings,
    build_workbuddy_argv,
    build_workbuddy_env,
    cli_model_flag,
)


def test_models_config_points_at_proxy() -> None:
    config = build_models_config('http://127.0.0.1:4321/v1/chat/completions')
    model = config['models'][0]
    assert model['id'] == 'general-model'
    assert model['url'] == 'http://127.0.0.1:4321/v1/chat/completions'
    assert model['apiKey'] == 'dummy-key-for-proxy'
    assert model['supportsToolCall'] is True
    assert model['supportsReasoning'] is False
    assert config['availableModels'] == ['general-model']


def test_models_config_declares_reasoning_on_request() -> None:
    config = build_models_config('http://127.0.0.1:4321/v1/chat/completions', supports_reasoning=True)
    assert config['models'][0]['supportsReasoning'] is True


def test_thinking_settings_carry_effort_only_when_set() -> None:
    assert build_thinking_settings('high') == {'alwaysThinkingEnabled': True, 'reasoningEffort': 'high'}
    assert build_thinking_settings(None) == {'alwaysThinkingEnabled': True}


def test_cli_model_flag_prefixes_only_published_releases() -> None:
    assert cli_model_flag('general-model', '5.9.0') == 'custom-local:general-model'
    assert cli_model_flag('general-model', '5.11.0-hotfix.202610272104') == 'custom-local:general-model'
    assert cli_model_flag('general-model', '5.7.0') == 'general-model'
    assert cli_model_flag('custom-local:general-model', '5.9.0') == 'custom-local:general-model'


def test_argv_keeps_prompt_last() -> None:
    argv = build_workbuddy_argv('/nix/node', '/opt/wb/package/bin/wb.js', 'fix the bug')
    assert argv[:4] == ['/nix/node', '/opt/wb/package/bin/wb.js', '--model', 'general-model']
    assert argv[4:8] == ['--preset', 'code', '--effort', 'xhigh']
    assert '--json' in argv
    assert argv[-1] == 'fix the bug'
    assert '--mode' not in argv


def test_argv_omits_optional_flags() -> None:
    argv = build_workbuddy_argv(
        '/nix/node',
        '/opt/wb.js',
        'hello',
        effort=None,
        preset=None,
        mode='ask',
        json_output=False,
        extra_cli_args=['--resume', 'last'],
    )
    assert '--effort' not in argv
    assert '--preset' not in argv
    assert '--json' not in argv
    assert argv[-5:] == ['--mode', 'ask', '--resume', 'last', 'hello']


def test_env_routes_codebuddy_compat_vars() -> None:
    env = build_workbuddy_env(
        home='/root',
        install_dir='/opt/workbuddy',
        proxy_base_url='http://127.0.0.1:9/v1',
        api_key='dummy-key-for-proxy',
        config_dir='/root/.workbuddy',
        extra_env={'OPENAI_TIMEOUT': '120'},
    )
    assert env['CODEBUDDY_BASE_URL'] == 'http://127.0.0.1:9/v1'
    assert env['CODEBUDDY_API_KEY'] == 'dummy-key-for-proxy'
    assert env['WORKBUDDY_CONFIG_DIR'] == '/root/.workbuddy'
    assert env['IS_SANDBOX'] == '1'
    assert env['OPENAI_TIMEOUT'] == '120'
    assert '/opt/workbuddy/package/bin' in env['PATH']


def test_env_ships_ca_defaults_that_stay_overridable() -> None:
    env = build_workbuddy_env(
        home='/root',
        install_dir='/opt/workbuddy',
        proxy_base_url='http://127.0.0.1:9/v1',
        api_key='dummy-key-for-proxy',
        config_dir='/root/.workbuddy',
    )
    assert env['SSL_CERT_FILE'] == '/etc/ssl/certs/ca-certificates.crt'
    assert env['NODE_EXTRA_CA_CERTS'] == '/etc/ssl/certs/ca-certificates.crt'
    assert env['SSL_CERT_DIR'] == '/etc/ssl/certs'

    overridden = build_workbuddy_env(
        home='/root',
        install_dir='/opt/workbuddy',
        proxy_base_url='http://127.0.0.1:9/v1',
        api_key='dummy-key-for-proxy',
        config_dir='/root/.workbuddy',
        extra_env={'SSL_CERT_FILE': '/custom/ca.pem'},
    )
    assert overridden['SSL_CERT_FILE'] == '/custom/ca.pem'


def test_env_refuses_to_let_extra_env_hijack_routing() -> None:
    env = build_workbuddy_env(
        home='/root',
        install_dir='/opt/workbuddy',
        proxy_base_url='http://127.0.0.1:9/v1',
        api_key='dummy-key-for-proxy',
        config_dir='/root/.workbuddy',
        extra_env={
            'CODEBUDDY_BASE_URL': 'http://evil.example/v1',
            'CODEBUDDY_API_KEY': 'stolen',
            'HOME': '/tmp/elsewhere',
            'WORKBUDDY_CONFIG_DIR': '/tmp/elsewhere',
            'PATH': '/nowhere',
            'OPENAI_TIMEOUT': '120',
        },
    )
    assert env['CODEBUDDY_BASE_URL'] == 'http://127.0.0.1:9/v1'
    assert env['CODEBUDDY_API_KEY'] == 'dummy-key-for-proxy'
    assert env['HOME'] == '/root'
    assert env['WORKBUDDY_CONFIG_DIR'] == '/root/.workbuddy'
    assert '/opt/workbuddy/package/bin' in env['PATH']
    # Non-protected keys still pass through.
    assert env['OPENAI_TIMEOUT'] == '120'


# A stand-in for wb: it starts a background child that inherits stdout/stderr, like wb's sidecar
# and MCP servers do, and records the child's pid so the test can check it was cleaned up.
_FAKE_WB_PREAMBLE = """
import subprocess, sys, time
child = subprocess.Popen(['sleep', '300'])
open(sys.argv[1], 'w').write(str(child.pid))
print('working', flush=True)
"""


def _alive(pid: int) -> bool:
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != 'Z'


@pytest.fixture
def fast_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_setup, '_SHUTDOWN_GRACE_SECONDS', 0.3)
    monkeypatch.setattr(runtime_setup, '_POLL_INTERVAL_SECONDS', 0.05)
    monkeypatch.setattr(runtime_setup, '_TERMINATE_GRACE_SECONDS', 1.0)
    monkeypatch.setattr(runtime_setup, '_DRAIN_TIMEOUT_SECONDS', 2.0)


async def _run_fake_wb(tmp_path: Path, body: str, timeout: float) -> tuple[int, str, str, int, float]:
    pid_file = tmp_path / 'child.pid'
    started = time.monotonic()
    code, stdout, stderr = await runtime_setup._run_workbuddy_streaming(
        [sys.executable, '-c', _FAKE_WB_PREAMBLE + body, str(pid_file)],
        str(tmp_path),
        {},
        timeout=timeout,
    )
    return code, stdout, stderr, int(pid_file.read_text()), time.monotonic() - started


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_lifecycle')
async def test_run_stops_wb_that_lingers_after_shutdown(tmp_path: Path) -> None:
    body = (
        "print('2026-09-22T22:05:23.183Z INFO  [wb:daemon-app-server] stopped {}', file=sys.stderr, flush=True)\n"
        'time.sleep(300)\n'
    )
    code, stdout, stderr, child_pid, elapsed = await _run_fake_wb(tmp_path, body, timeout=60)
    assert elapsed < 15
    assert code == 0
    assert 'working' in stdout
    assert '[timeout]' not in stderr
    assert not _alive(child_pid)


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_lifecycle')
async def test_run_returns_when_wb_exits_but_a_child_holds_the_pipes(tmp_path: Path) -> None:
    code, stdout, _, child_pid, elapsed = await _run_fake_wb(tmp_path, 'sys.exit(3)\n', timeout=60)
    assert elapsed < 15
    assert code == 3
    assert 'working' in stdout
    assert not _alive(child_pid)


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_lifecycle')
async def test_run_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    code, _, stderr, child_pid, elapsed = await _run_fake_wb(tmp_path, 'time.sleep(300)\n', timeout=1.0)
    assert elapsed < 15
    assert code == -1
    assert stderr.endswith('[timeout]')
    assert not _alive(child_pid)


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_lifecycle')
async def test_prompt_verification_failure_stops_wb_without_waiting_for_task_timeout(tmp_path: Path) -> None:
    pid_file = tmp_path / 'child.pid'
    started = time.monotonic()
    code, _, stderr = await runtime_setup._run_workbuddy_streaming(
        [sys.executable, '-c', _FAKE_WB_PREAMBLE + 'time.sleep(300)\n', str(pid_file)],
        str(tmp_path),
        {},
        timeout=60,
        should_abort=pid_file.exists,
    )
    assert time.monotonic() - started < 15
    assert code == -1
    assert 'prompt identity verification failed' in stderr
    assert not _alive(int(pid_file.read_text()))


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_lifecycle')
async def test_cancel_during_wait_stops_the_process_group(tmp_path: Path) -> None:
    """Cancelling the poll sleep must still kill wb and the children holding its pipes."""
    pid_file = tmp_path / 'child.pid'
    run = asyncio.create_task(
        runtime_setup._run_workbuddy_streaming(
            [sys.executable, '-c', _FAKE_WB_PREAMBLE + 'time.sleep(300)\n', str(pid_file)],
            str(tmp_path),
            {},
            timeout=60,
        )
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not pid_file.exists():
        await asyncio.sleep(0.05)
    if not pid_file.exists():
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        pytest.fail('fake wb did not record its child pid')

    child_pid = int(pid_file.read_text())
    assert _alive(child_pid)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    for _ in range(50):
        if not _alive(child_pid):
            return
        await asyncio.sleep(0.1)
    os.kill(child_pid, signal.SIGKILL)
    pytest.fail(f'child {child_pid} survived cancellation')


@pytest.mark.asyncio
@pytest.mark.usefixtures('fast_lifecycle')
async def test_run_keeps_stdout_lines_longer_than_asyncio_default_limit(tmp_path: Path) -> None:
    body = "sys.stdout.write('x' * 300000 + '\\n')\nsys.stdout.flush()\n"
    code, stdout, _, _, _ = await _run_fake_wb(tmp_path, body, timeout=60)
    assert code == 0
    assert 'x' * 300000 in stdout
