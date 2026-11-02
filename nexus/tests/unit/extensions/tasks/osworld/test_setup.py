"""OSWorld setup regressions."""

from __future__ import annotations

import base64
import json

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.os_world.execution.guest_network import (
    OSWORLD_PROXY_SERVER,
    guest_proxy_env,
    proxy_setup_failure_message,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(('initial_url', 'closed'), [('about:blank', True), ('chrome://newtab/', False)])
async def test_open_chrome_tabs_closes_only_blank_tabs(monkeypatch, initial_url, closed) -> None:
    from nexus.extensions.tasks.os_world.execution import setup

    class FakePage:
        def __init__(self, url: str) -> None:
            self.url = url
            self.closed = False

        async def goto(self, url: str, **kwargs: object) -> None:
            _ = kwargs
            self.url = url

        def is_closed(self) -> bool:
            return self.closed

        async def close(self) -> None:
            self.closed = True

    class FakeContext:
        def __init__(self) -> None:
            self.original = FakePage(initial_url)
            self.pages = [self.original]

        async def new_page(self) -> FakePage:
            page = FakePage('about:blank')
            self.pages.append(page)
            return page

    context = FakeContext()
    browser = SimpleNamespace(contexts=[context])

    async def fake_run_setup(runtime: object, operation: object) -> None:
        _ = runtime
        await operation(browser)

    monkeypatch.setattr(setup, '_run_chrome_cdp_setup', fake_run_setup)

    await setup._open_chrome_tabs(object(), ['https://example.com/one', 'https://example.com/two'])

    assert context.original.closed is closed
    assert [page.url for page in context.pages[1:]] == [
        'https://example.com/one',
        'https://example.com/two',
    ]
    assert all(not page.closed for page in context.pages[1:])


@pytest.mark.asyncio
async def test_open_chrome_tabs_keeps_original_page_when_no_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.tasks.os_world.execution import setup

    page = SimpleNamespace(closed=False, url='https://example.com/existing')
    page.is_closed = lambda: page.closed

    async def close_page() -> None:
        page.closed = True

    page.close = close_page
    context = SimpleNamespace(pages=[page])
    browser = SimpleNamespace(contexts=[context])

    async def fake_run_setup(runtime: object, operation: object) -> None:
        _ = runtime
        await operation(browser)

    monkeypatch.setattr(setup, '_run_chrome_cdp_setup', fake_run_setup)

    await setup._open_chrome_tabs(object(), [])

    assert page.closed is False


def test_guest_proxy_env_is_local_and_bypasses_bridge() -> None:
    env = guest_proxy_env({})
    for key in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
        assert env[key] == OSWORLD_PROXY_SERVER
    assert env['no_proxy'] == env['NO_PROXY'] == 'localhost,127.0.0.1,::1,[::1]'


def test_guest_proxy_env_preserves_explicit_bypasses_without_mutating_input() -> None:
    original = {
        'NO_PROXY': '  service.internal,localhost,second.internal ',
        'no_proxy': 'service.internal,,127.0.0.1',
        'http_proxy': 'http://host-proxy.invalid:3128',
        'ALL_PROXY': 'socks5://host-proxy.invalid:1080',
        'API_TIMEOUT_MS': '600000',
    }
    before = original.copy()
    env = guest_proxy_env(original)
    assert original == before
    assert env['NO_PROXY'] == env['no_proxy'] == 'localhost,127.0.0.1,::1,[::1],service.internal,second.internal'
    assert env['ALL_PROXY'] == env['http_proxy'] == OSWORLD_PROXY_SERVER
    assert 'API_TIMEOUT_MS' not in env
    assert not any('host-proxy' in value for value in env.values())


def test_proxy_failure_summary_is_useful_without_copying_sensitive_output() -> None:
    message = proxy_setup_failure_message(
        {
            'stage': 'install',
            'upload': {'success': True, 'status_code': 200},
            'return_code': 127,
            'command': 'echo private-command',
            'stdout': 'private-stdout',
            'stderr': 'http://private-user:private-password@proxy.invalid:9999 tinyproxy: not found',
        }
    )
    assert 'stage=install' in message
    assert 'upload_success=True' in message
    assert 'upload_status_code=200' in message
    assert 'command_return_code=127' in message
    assert 'tinyproxy: not found' in message
    assert 'private-' not in message
    assert 'proxy.invalid' not in message


def test_proxy_failure_summary_does_not_trust_unexpected_field_types() -> None:
    message = proxy_setup_failure_message(
        {
            'stage': 'private-stage',
            'upload': 'private-upload',
            'return_code': 'private-code',
        }
    )
    assert 'stage=unknown' in message
    assert 'command_return_code=None' in message
    assert 'private-' not in message


@pytest.mark.asyncio
async def test_os_world_setup_runner_executes_download_launch_and_open(tmp_path, monkeypatch):
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    commands: list[str] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return {'return_code': 0, 'stdout': 'ok', 'stderr': ''}

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    async def fake_sleep(seconds):
        commands.append(f'sleep {seconds}')

    monkeypatch.setattr('nexus.extensions.tasks.os_world.execution.setup.asyncio.sleep', fake_sleep)

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'download',
                'parameters': {'files': [{'url': 'https://example.test/a.txt', 'path': '/tmp/a.txt'}]},
            },
            {
                'type': 'download',
                'parameters': {'files': [{'url': 'https://example.test/eval.sh', 'path': 'eval.sh'}]},
            },
            {'type': 'launch', 'parameters': {'command': ['xterm', '-title', 'Demo']}},
            {'type': 'open', 'parameters': {'path': '/tmp/a.txt'}},
            {'type': 'sleep', 'parameters': {'seconds': 0.25}},
        ],
    )

    assert [item['type'] for item in results] == ['download', 'download', 'launch', 'open', 'sleep']
    assert 'curl -L --retry 3 -o /tmp/a.txt' in commands[0]
    assert 'https://example.test/a.txt' in commands[0]
    assert 'mkdir -p . && curl -L --retry 3 -o eval.sh' in commands[1]
    assert 'https://example.test/eval.sh' in commands[1]
    assert 'nohup xterm -title Demo' in commands[2]
    assert 'xdg-open /tmp/a.txt' in commands[3]
    assert 'runuser -u user' in commands[0]
    assert 'HOME=/home/user' in commands[0]
    assert commands[4] == 'sleep 0.25'


@pytest.mark.asyncio
async def test_os_world_setup_download_can_skip_user_wrapper():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    commands: list[str] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'download',
                'parameters': {'files': [{'url': 'https://example.test/a.txt', 'path': '/tmp/a.txt'}]},
            },
        ],
        user_command_fn=lambda command: command,
    )

    assert results[0]['success'] is True
    assert len(commands) == 1
    assert 'curl -L --retry 3 -o /tmp/a.txt' in commands[0]
    assert 'runuser -u user' not in commands[0]


def test_os_world_setup_download_honors_host_proxy_environment():
    # Training hosts reach Hugging Face only through the environment proxy; a direct client
    # silently leaves setup assets missing from the sandbox.
    from nexus.extensions.tasks.os_world.execution import setup as osworld_setup
    from nexus.utils.async_http import http_client_with_trust_env

    assert osworld_setup.http_client is http_client_with_trust_env


@pytest.mark.asyncio
async def test_os_world_setup_download_uses_host_cache_and_runtime_upload(tmp_path, monkeypatch):
    from nexus.extensions.tasks.os_world.execution import setup as osworld_setup

    fetches: list[str] = []
    uploads = []

    async def fake_download_url_bytes(url: str) -> bytes:
        fetches.append(url)
        return b'cached asset'

    class FakeRuntime:
        async def upload_file(self, files, overwrite=True):
            uploads.append((files, overwrite))
            return {'results': [{'path': file.path, 'success': True, 'message': 'ok'} for file in files]}

        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError(f'setup download should not run VM curl: {command}')

    monkeypatch.setattr(osworld_setup, '_download_url_bytes', fake_download_url_bytes)

    results = await osworld_setup.run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'download',
                'parameters': {'files': [{'url': 'https://example.test/a.txt', 'path': '/tmp/a.txt'}]},
            },
            {
                'type': 'download',
                'parameters': {'files': [{'url': 'https://example.test/a.txt', 'path': '/tmp/a.txt'}]},
            },
        ],
        cache_dir=tmp_path,
    )

    assert [item['success'] for item in results] == [True, True]
    assert fetches == ['https://example.test/a.txt']
    assert len(uploads) == 2
    assert uploads[0][1] is True
    assert uploads[0][0][0].path == '/tmp/a.txt'
    assert uploads[0][0][0].encoding == 'base64'
    assert base64.b64decode(uploads[0][0][0].content) == b'cached asset'
    assert results[0]['files'][0]['cache_hit'] is False
    assert results[1]['files'][0]['cache_hit'] is True


@pytest.mark.asyncio
async def test_os_world_setup_download_prefers_setup_upload_file(tmp_path, monkeypatch):
    from nexus.extensions.tasks.os_world.execution import setup as osworld_setup

    async def fake_download_url_bytes(url: str) -> bytes:
        return b'streamed asset'

    uploads: list[tuple[Path, str]] = []

    class FakeRuntime:
        async def setup_upload_file(self, local_path, remote_path):
            uploads.append((Path(local_path), remote_path))
            return SimpleNamespace(status_code=200, text='File Uploaded')

        async def upload_file(self, files, overwrite=True):
            raise AssertionError('setup download should prefer setup_upload_file')

    monkeypatch.setattr(osworld_setup, '_download_url_bytes', fake_download_url_bytes)

    results = await osworld_setup.run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'download',
                'parameters': {'files': [{'url': 'https://example.test/asset.txt', 'path': '/tmp/asset.txt'}]},
            },
        ],
        cache_dir=tmp_path,
    )

    assert results[0]['success'] is True
    assert len(uploads) == 1
    assert uploads[0][0].read_bytes() == b'streamed asset'
    assert uploads[0][1] == '/tmp/asset.txt'


@pytest.mark.asyncio
async def test_os_world_setup_runner_uses_google_chrome_without_chromium_compatibility(monkeypatch):
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    commands: list[str] = []
    opened: list[str] = []

    async def fake_open_tabs(runtime, urls):
        opened.extend(urls)

    monkeypatch.setattr('nexus.extensions.tasks.os_world.execution.setup._open_chrome_tabs', fake_open_tabs)

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    await run_setup_actions(
        FakeRuntime(),
        [
            {'type': 'launch', 'parameters': {'command': ['google-chrome', '--remote-debugging-port=1337']}},
            {'type': 'execute', 'parameters': {'command': ['pkill', 'chrome']}},
            {'type': 'chrome_open_tabs', 'parameters': {'urls_to_open': ['https://example.test']}},
        ],
    )

    assert 'nohup google-chrome --remote-debugging-port=1337' in commands[0]
    assert 'pkill chrome' in commands[1]
    assert 'pkill -f' not in commands[1]
    assert len(commands) == 2
    assert opened == ['https://example.test']
    assert not any('chromium' in command for command in commands)


@pytest.mark.asyncio
async def test_os_world_setup_launch_adds_proxy_only_to_chrome_launch(monkeypatch):
    from nexus.extensions.tasks.os_world.execution.setup import OSWORLD_PROXY_SERVER_ARG, run_setup_actions

    commands: list[str] = []
    opened: list[str] = []

    async def fake_open_tabs(runtime, urls):
        opened.extend(urls)

    monkeypatch.setattr('nexus.extensions.tasks.os_world.execution.setup._open_chrome_tabs', fake_open_tabs)

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    await run_setup_actions(
        FakeRuntime(),
        [
            {'type': 'launch', 'parameters': {'command': ['google-chrome', '--remote-debugging-port=1337']}},
            {'type': 'launch', 'parameters': {'command': ['xterm', '-title', 'Demo']}},
            {'type': 'chrome_open_tabs', 'parameters': {'urls_to_open': ['https://example.test']}},
        ],
        user_command_fn=lambda command: command,
        use_proxy=True,
    )

    assert OSWORLD_PROXY_SERVER_ARG in commands[0]
    assert OSWORLD_PROXY_SERVER_ARG not in commands[1]
    assert 'nohup xterm -title Demo' in commands[1]
    assert len(commands) == 2
    assert opened == ['https://example.test']


@pytest.mark.asyncio
async def test_os_world_proxy_setup_uploads_config_and_redacts_results(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import run_proxy_setup

    proxy_config = tmp_path / 'proxy.json'
    proxy_config.write_text(
        json.dumps(
            [
                {
                    'host': 'proxy.example.test',
                    'port': 18888,
                    'username': 'dummy-user',
                    'password': 'dummy-password',
                    'protocol': 'http',
                }
            ]
        ),
        encoding='utf-8',
    )
    uploaded: dict[str, object] = {}
    commands: list[str] = []

    class FakeRuntime:
        async def setup_upload_bytes(self, remote_path, content, *, filename='file'):
            uploaded['remote_path'] = remote_path
            uploaded['content'] = content.decode('utf-8')
            uploaded['filename'] = filename
            return SimpleNamespace(status_code=200, text='ok')

        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='started', stderr='')

    result = await run_proxy_setup(
        FakeRuntime(),
        proxy_config_file=proxy_config,
        user_command_fn=lambda command: command,
        client_password='dummy-client-password',
    )

    dumped = json.dumps(result)
    assert result['success'] is True
    assert result['stage'] == 'ready'
    assert result['proxy_server'] == 'http://127.0.0.1:18888'
    assert uploaded['remote_path'] == '/tmp/tinyproxy.conf'
    assert uploaded['filename'] == 'tinyproxy.conf'
    assert 'Upstream http dummy-user:dummy-password@proxy.example.test:18888' in uploaded['content']

    # Commands run in order: install, keep-alive launch (shell fallback), bashrc, probe.
    install_command, launch_command, bashrc_command, probe_command = commands
    assert 'apt-get install -y tinyproxy' in install_command
    assert 'systemctl disable tinyproxy' in install_command
    assert 'pkill -x tinyproxy' in install_command
    assert 'pkill -f' not in install_command
    assert 'dummy-client-password' in install_command
    assert 'tinyproxy -c /tmp/tinyproxy.conf -d' in launch_command
    assert 'setsid' in launch_command
    assert 'export http_proxy=http://127.0.0.1:18888' in bashrc_command
    assert '-x http://127.0.0.1:18888' in probe_command
    assert 'curl' in probe_command

    # Neither the resolved client password nor any proxy credential leaks into the result.
    for text in (launch_command, bashrc_command, probe_command):
        assert 'dummy-client-password' not in text
    for text in (*commands, dumped):
        assert 'dummy-password' not in text
        assert 'dummy-user' not in text
        assert 'proxy.example.test' not in text
    assert 'dummy-client-password' not in dumped
    assert '{CLIENT_PASSWORD}' in result['install']['command']


@pytest.mark.asyncio
async def test_os_world_proxy_setup_prefers_setup_launch_keep_alive(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import run_proxy_setup

    proxy_config = tmp_path / 'proxy.json'
    proxy_config.write_text(
        json.dumps([{'host': 'proxy.example.test', 'port': 18888, 'protocol': 'http'}]),
        encoding='utf-8',
    )
    launched: dict[str, object] = {}
    commands: list[str] = []

    class FakeRuntime:
        async def setup_upload_bytes(self, remote_path, content, *, filename='file'):
            return SimpleNamespace(status_code=200, text='ok')

        async def setup_launch(self, command, *, shell=False):
            launched['command'] = command
            launched['shell'] = shell
            return SimpleNamespace(status_code=200, text='tinyproxy launched successfully')

        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='proxy-probe-ok 200', stderr='')

    result = await run_proxy_setup(
        FakeRuntime(),
        proxy_config_file=proxy_config,
        user_command_fn=lambda command: command,
        client_password='dummy-client-password',
    )

    assert result['success'] is True
    assert launched['command'] == ['tinyproxy', '-c', '/tmp/tinyproxy.conf', '-d']
    assert launched['shell'] is False
    assert result['launch']['launch_method'] == 'setup_launch'
    assert result['launch']['success'] is True
    # No tinyproxy start command runs through run_command when the launch endpoint is used.
    assert not any('setsid tinyproxy' in command for command in commands)
    # Install, bashrc, probe still go through run_command.
    assert any('apt-get install -y tinyproxy' in command for command in commands)
    assert any('-x http://127.0.0.1:18888' in command for command in commands)


@pytest.mark.asyncio
async def test_os_world_proxy_setup_fails_when_probe_fails(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import run_proxy_setup

    proxy_config = tmp_path / 'proxy.json'
    proxy_config.write_text(
        json.dumps([{'host': 'proxy.example.test', 'port': 18888, 'protocol': 'http'}]),
        encoding='utf-8',
    )

    class FakeRuntime:
        async def setup_upload_bytes(self, remote_path, content, *, filename='file'):
            return SimpleNamespace(status_code=200, text='ok')

        async def run_command(self, command, timeout=60, cwd=None):
            if '-x http://127.0.0.1:18888' in command:
                return SimpleNamespace(return_code=1, stdout='proxy-probe-failed last_code=000', stderr='')
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

    result = await run_proxy_setup(
        FakeRuntime(),
        proxy_config_file=proxy_config,
        user_command_fn=lambda command: command,
        client_password='dummy-client-password',
    )

    assert result['success'] is False
    assert result['stage'] == 'probe'
    assert result['return_code'] == 1
    assert 'proxy-probe-failed' in result['stdout']


@pytest.mark.asyncio
async def test_os_world_proxy_setup_fails_when_install_fails(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import run_proxy_setup

    proxy_config = tmp_path / 'proxy.json'
    proxy_config.write_text(
        json.dumps([{'host': 'proxy.example.test', 'port': 18888, 'protocol': 'http'}]),
        encoding='utf-8',
    )
    commands: list[str] = []

    class FakeRuntime:
        async def setup_upload_bytes(self, remote_path, content, *, filename='file'):
            return SimpleNamespace(status_code=200, text='ok')

        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            if 'apt-get install' in command:
                return SimpleNamespace(return_code=100, stdout='', stderr='apt failed')
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

    result = await run_proxy_setup(
        FakeRuntime(),
        proxy_config_file=proxy_config,
        user_command_fn=lambda command: command,
        client_password='dummy-client-password',
    )

    assert result['success'] is False
    assert result['stage'] == 'install'
    assert result['return_code'] == 100
    # Setup stops after install failure: no probe command runs.
    assert not any('-x http://127.0.0.1:18888' in command for command in commands)


@pytest.mark.asyncio
async def test_os_world_proxy_setup_fails_when_upload_fails(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import run_proxy_setup

    proxy_config = tmp_path / 'proxy.json'
    proxy_config.write_text(
        json.dumps([{'host': 'proxy.example.test', 'port': 18888, 'protocol': 'http'}]),
        encoding='utf-8',
    )
    ran = False

    class FakeRuntime:
        async def setup_upload_bytes(self, remote_path, content, *, filename='file'):
            return SimpleNamespace(status_code=500, text='upload failed')

        async def run_command(self, command, timeout=60, cwd=None):
            nonlocal ran
            ran = True
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

    result = await run_proxy_setup(
        FakeRuntime(),
        proxy_config_file=proxy_config,
        user_command_fn=lambda command: command,
    )

    assert result['success'] is False
    assert result['upload']['success'] is False
    assert ran is False


def test_load_proxy_config_supports_config_shapes(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import _load_proxy_config

    list_config = tmp_path / 'list.json'
    list_config.write_text(
        json.dumps(
            [
                {'host': 'first.example.test', 'port': 1080, 'protocol': 'socks5'},
                {'host': 'second.example.test', 'port': 3128},
            ]
        ),
        encoding='utf-8',
    )
    first = _load_proxy_config(list_config)
    assert first == {
        'host': 'first.example.test',
        'port': 1080,
        'username': None,
        'password': None,
        'protocol': 'socks5',
    }

    dict_config = tmp_path / 'dict.json'
    dict_config.write_text(
        json.dumps({'host': 'solo.example.test', 'port': 8080, 'username': 'u', 'password': 'p'}),
        encoding='utf-8',
    )
    assert _load_proxy_config(dict_config)['host'] == 'solo.example.test'

    proxies_config = tmp_path / 'proxies.json'
    proxies_config.write_text(
        json.dumps({'proxies': [{'host': 'pool.example.test', 'port': 9999}]}),
        encoding='utf-8',
    )
    assert _load_proxy_config(proxies_config)['host'] == 'pool.example.test'
    assert _load_proxy_config(proxies_config)['protocol'] == 'http'


def test_load_proxy_config_rejects_unusable_configs(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import _load_proxy_config

    empty_config = tmp_path / 'empty.json'
    empty_config.write_text(json.dumps([{'host': '', 'port': None}]), encoding='utf-8')
    with pytest.raises(ValueError, match='usable proxy entry'):
        _load_proxy_config(empty_config)

    missing_config = tmp_path / 'missing.json'
    with pytest.raises(ValueError, match='Failed to load'):
        _load_proxy_config(missing_config)


@pytest.mark.asyncio
async def test_os_world_setup_runner_updates_browse_history_without_screen_size():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    calls: list[tuple[str, int]] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            calls.append((command, timeout))
            return SimpleNamespace(return_code=0, stdout='{"history_count": 1}', stderr='')

        async def gui_get_screen_size(self):
            raise AssertionError('update_browse_history should not request screen size')

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'update_browse_history',
                'parameters': {
                    'history': [
                        {
                            'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                            'title': 'Rick Astley - Never Gonna Give You Up',
                            'visit_time_from_now_in_seconds': 3600,
                        }
                    ]
                },
            }
        ],
    )

    assert results[0]['type'] == 'update_browse_history'
    assert results[0]['success'] is True
    assert results[0]['stdout'] == '{"history_count": 1}'
    assert len(calls) == 1
    command, timeout = calls[0]
    assert timeout == 300
    assert 'runuser -u user' in command
    assert 'history_empty.sqlite' in command
    assert 'base64.b64decode' in command
    assert 'INSERT INTO urls' in command
    assert 'google-chrome' in command
    assert 'History' in command


@pytest.mark.asyncio
async def test_os_world_setup_activate_window_prefers_native_endpoint_without_screen_size():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    activate_calls: list[tuple[str, bool, bool]] = []

    class FakeRuntime:
        async def setup_activate_window(self, window_name, *, strict=False, by_class=False):
            activate_calls.append((window_name, strict, by_class))
            return SimpleNamespace(status_code=200, text='')

        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError(f'activate_window should prefer native setup endpoint: {command}')

        async def gui_get_screen_size(self):
            raise AssertionError('activate_window should not request screen size')

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'activate_window',
                'parameters': {'window_name': 'Message Filters', 'strict': True, 'by_class': True},
            }
        ],
    )

    assert activate_calls == [('Message Filters', True, True)]
    assert results == [
        {
            'type': 'activate_window',
            'window_name': 'Message Filters',
            'strict': True,
            'by_class': True,
            'success': True,
            'status_code': 200,
            'stdout': '',
            'stderr': '',
        }
    ]


@pytest.mark.asyncio
async def test_os_world_setup_activate_window_reports_native_failure_without_fallback():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    class FakeRuntime:
        async def setup_activate_window(self, window_name, *, strict=False, by_class=False):
            return SimpleNamespace(status_code=500, text='activate failed')

        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError('native activate_window failure should not run fallback activate')

    results = await run_setup_actions(
        FakeRuntime(),
        [{'type': 'activate_window', 'parameters': {'window_name': 'Message Filters'}}],
    )

    assert results[0]['success'] is False
    assert results[0]['status_code'] == 500
    assert results[0]['stderr'] == 'activate failed'


@pytest.mark.asyncio
async def test_os_world_setup_activate_window_falls_back_to_wmctrl_flags():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    commands: list[tuple[str, int]] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append((command, timeout))
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def gui_get_screen_size(self):
            raise AssertionError('activate_window should not request screen size')

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'activate_window',
                'parameters': {'window_name': 'Message Filters', 'strict': True, 'by_class': True},
            }
        ],
        user_command_fn=lambda command: command,
    )

    assert commands == [("wmctrl -xFa 'Message Filters'", 30)]
    assert results[0]['success'] is True
    assert results[0]['window_name'] == 'Message Filters'
    assert results[0]['strict'] is True
    assert results[0]['by_class'] is True


@pytest.mark.asyncio
async def test_os_world_setup_activate_window_requires_window_name():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError('activate_window with missing window_name should not run commands')

    with pytest.raises(ValueError, match='activate_window requires window_name'):
        await run_setup_actions(FakeRuntime(), [{'type': 'activate_window', 'parameters': {'window_name': ''}}])


@pytest.mark.asyncio
async def test_os_world_setup_close_window_prefers_native_endpoint_without_screen_size():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    close_calls: list[tuple[str, bool, bool]] = []

    class FakeRuntime:
        async def setup_close_window(self, window_name, *, strict=False, by_class=False):
            close_calls.append((window_name, strict, by_class))
            return SimpleNamespace(status_code=200, text='')

        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError(f'close_window should prefer native setup endpoint: {command}')

        async def gui_get_screen_size(self):
            raise AssertionError('close_window should not request screen size')

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'close_window',
                'parameters': {'window_name': 'Message Filters', 'strict': True, 'by_class': True},
            }
        ],
    )

    assert close_calls == [('Message Filters', True, True)]
    assert results == [
        {
            'type': 'close_window',
            'window_name': 'Message Filters',
            'strict': True,
            'by_class': True,
            'success': True,
            'status_code': 200,
            'stdout': '',
            'stderr': '',
        }
    ]


@pytest.mark.asyncio
async def test_os_world_setup_close_window_reports_native_failure_without_fallback():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    class FakeRuntime:
        async def setup_close_window(self, window_name, *, strict=False, by_class=False):
            return SimpleNamespace(status_code=500, text='close failed')

        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError('native close_window failure should not run fallback close')

    results = await run_setup_actions(
        FakeRuntime(),
        [{'type': 'close_window', 'parameters': {'window_name': 'Message Filters'}}],
    )

    assert results[0]['success'] is False
    assert results[0]['status_code'] == 500
    assert results[0]['stderr'] == 'close failed'


@pytest.mark.asyncio
async def test_os_world_setup_close_window_falls_back_to_wmctrl_flags():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    commands: list[tuple[str, int]] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append((command, timeout))
            return SimpleNamespace(return_code=0, stdout='', stderr='')

        async def gui_get_screen_size(self):
            raise AssertionError('close_window should not request screen size')

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'close_window',
                'parameters': {'window_name': 'Message Filters', 'strict': True, 'by_class': True},
            }
        ],
        user_command_fn=lambda command: command,
    )

    assert commands == [("wmctrl -xFc 'Message Filters'", 30)]
    assert results[0]['success'] is True
    assert results[0]['window_name'] == 'Message Filters'
    assert results[0]['strict'] is True
    assert results[0]['by_class'] is True


@pytest.mark.asyncio
async def test_os_world_setup_close_window_requires_window_name():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            raise AssertionError('close_window with missing window_name should not run commands')

    with pytest.raises(ValueError, match='close_window requires window_name'):
        await run_setup_actions(FakeRuntime(), [{'type': 'close_window', 'parameters': {'window_name': ''}}])


@pytest.mark.asyncio
async def test_os_world_setup_runner_reports_failed_commands():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            return SimpleNamespace(return_code=7, stdout='', stderr='boom')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    results = await run_setup_actions(
        FakeRuntime(),
        [{'type': 'execute', 'parameters': {'command': 'false'}}],
    )

    assert results[0]['success'] is False
    assert results[0]['return_code'] == 7
    assert results[0]['stderr'] == 'boom'


@pytest.mark.asyncio
async def test_os_world_setup_execute_replaces_client_password_without_returning_secret():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    commands: list[str] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    client_password = 'unit-test-password'
    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'execute',
                'parameters': {'command': 'echo {CLIENT_PASSWORD} | sudo -S true'},
            }
        ],
        user_command_fn=lambda command: command,
        client_password=client_password,
    )

    assert commands == [f'echo {client_password} | sudo -S true']
    assert results[0]['success'] is True
    assert results[0]['command'] == 'echo {CLIENT_PASSWORD} | sudo -S true'
    assert client_password not in json.dumps(results)


@pytest.mark.asyncio
async def test_os_world_setup_command_replaces_list_item_client_password_and_screen_placeholders():
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    commands: list[str] = []

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            commands.append(command)
            return SimpleNamespace(return_code=0, stdout='ok', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1920, height=1080)

    client_password = 'list-password'
    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'command',
                'parameters': {
                    'command': [
                        'python3',
                        '-c',
                        'print("{CLIENT_PASSWORD}", {SCREEN_WIDTH}, {SCREEN_HEIGHT_HALF})',
                    ],
                },
            }
        ],
        user_command_fn=lambda command: command,
        client_password=client_password,
    )

    assert len(commands) == 1
    assert client_password in commands[0]
    assert '1920' in commands[0]
    assert '540' in commands[0]
    assert '{CLIENT_PASSWORD}' not in commands[0]
    assert '{SCREEN_WIDTH' not in commands[0]
    assert results[0]['success'] is True
    assert client_password not in results[0]['command']
    assert '{CLIENT_PASSWORD}' in results[0]['command']
    assert '1920' in results[0]['command']
    assert '540' in results[0]['command']
    assert '{SCREEN_WIDTH' not in results[0]['command']
    assert client_password not in json.dumps(results)


@pytest.mark.asyncio
async def test_os_world_setup_execute_writes_stdout_and_stderr_cache_files(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            return SimpleNamespace(return_code=0, stdout='cached stdout', stderr='cached stderr')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'execute',
                'parameters': {
                    'command': 'printf cache',
                    'stdout': 'setup.out',
                    'stderr': 'setup.err',
                },
            }
        ],
        cache_dir=tmp_path,
    )

    assert results[0]['success'] is True
    assert results[0]['attempts'] == 1
    assert (tmp_path / 'setup.out').read_text(encoding='utf-8') == 'cached stdout'
    assert (tmp_path / 'setup.err').read_text(encoding='utf-8') == 'cached stderr'


@pytest.mark.asyncio
async def test_os_world_setup_execute_retries_until_stdout_or_stderr_match(tmp_path, monkeypatch):
    from nexus.extensions.tasks.os_world.execution import setup as osworld_setup

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    responses = [
        SimpleNamespace(return_code=0, stdout='not yet', stderr=''),
        SimpleNamespace(return_code=0, stdout='still not stdout', stderr='ready now'),
    ]

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            return responses.pop(0)

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    monkeypatch.setattr(osworld_setup.asyncio, 'sleep', fake_sleep)

    results = await osworld_setup.run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'execute',
                'parameters': {
                    'command': 'wait-for-ready',
                    'stdout': 'until.out',
                    'stderr': 'until.err',
                    'until': {'stdout': 'done', 'stderr': 'ready'},
                },
            }
        ],
        cache_dir=tmp_path,
    )

    assert results[0]['success'] is True
    assert results[0]['attempts'] == 2
    assert sleeps == [0.3]
    assert (tmp_path / 'until.out').read_text(encoding='utf-8') == 'still not stdout'
    assert (tmp_path / 'until.err').read_text(encoding='utf-8') == 'ready now'


@pytest.mark.asyncio
async def test_os_world_setup_command_reuses_execute_until_returncode_path(tmp_path):
    from nexus.extensions.tasks.os_world.execution.setup import run_setup_actions

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            return SimpleNamespace(return_code=7, stdout='matched returncode', stderr='')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    results = await run_setup_actions(
        FakeRuntime(),
        [
            {
                'type': 'command',
                'parameters': {
                    'command': 'exit 7',
                    'stdout': 'command.out',
                    'until': {'returncode': 7},
                },
            }
        ],
        cache_dir=tmp_path,
    )

    assert results[0]['type'] == 'command'
    assert results[0]['return_code'] == 7
    assert results[0]['success'] is False
    assert results[0]['attempts'] == 1
    assert (tmp_path / 'command.out').read_text(encoding='utf-8') == 'matched returncode'


@pytest.mark.asyncio
async def test_os_world_setup_execute_stops_after_five_runtime_invocation_failures(monkeypatch):
    from nexus.extensions.tasks.os_world.execution import setup as osworld_setup

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    class FakeRuntime:
        async def run_command(self, command, timeout=60, cwd=None):
            raise RuntimeError('transport unavailable')

        async def gui_get_screen_size(self):
            return SimpleNamespace(success=True, width=1280, height=800)

    monkeypatch.setattr(osworld_setup.asyncio, 'sleep', fake_sleep)

    results = await osworld_setup.run_setup_actions(
        FakeRuntime(),
        [{'type': 'execute', 'parameters': {'command': 'wait', 'until': {'stdout': 'ready'}}}],
    )

    assert results[0]['success'] is False
    assert results[0]['return_code'] is None
    assert results[0]['attempts'] == 5
    assert results[0]['error'] == 'RuntimeError: transport unavailable'
    assert sleeps == [0.3, 0.3, 0.3, 0.3]
