import asyncio
import base64
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from nexus.extensions.tasks.os_world_v2.adapter.transport import (
    Transport,
    install_requests_transport,
    local_gitlab_origin,
)


@pytest.mark.parametrize('value', ['localhost', 'http://user:secret@localhost', 'http://localhost?token=x'])
def test_gitlab_requires_absolute_url_without_credentials(value):
    with pytest.raises(ValueError, match='GITLAB_URL'):
        local_gitlab_origin({'GITLAB_URL': value})


def test_remote_gitlab_is_not_routed_into_guest():
    assert local_gitlab_origin({'GITLAB_URL': 'https://gitlab.example.test'}) is None
    assert local_gitlab_origin({}) is None
    assert local_gitlab_origin({'GITLAB_URL': 'http://localhost'}) == ('http', 'localhost', 80)
    assert local_gitlab_origin({'GITLAB_PAGES_URL': 'http://pages.localhost'}, 'GITLAB_PAGES_URL') == (
        'http',
        'pages.localhost',
        80,
    )


def test_worker_error_redacts_configured_secrets(monkeypatch):
    from nexus.extensions.tasks.os_world_v2.adapter.worker import safe_error_message

    monkeypatch.setenv('GITLAB_PRIVATE_TOKEN', 'unit-sensitive-value')
    message = safe_error_message(RuntimeError('PRIVATE-TOKEN: unit-sensitive-value rejected'))
    assert 'unit-sensitive-value' not in message
    assert '[REDACTED]' in message
    assert len(safe_error_message(RuntimeError('x' * 8000))) <= 4000


@pytest.mark.asyncio
async def test_only_configured_gitlab_origin_is_allowed():
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                return_code=0,
                stdout='NEXUS_WEBSITE_RESPONSE='
                + json.dumps(
                    {
                        'status': 200,
                        'headers': {},
                        'body': base64.b64encode(b'{"id":1}').decode(),
                    }
                ),
            )
        )
    )
    async with Transport(runtime, [], 'localhost', gitlab_origin=('http', 'localhost', 80)) as proxy:
        async with aiohttp.ClientSession() as session:
            for url, expected in [
                ('http://localhost/api/v4/user', 200),
                ('http://localhost:5000/execute', 400),
                ('https://localhost/api/v4/user', 400),
                ('http://127.0.0.1/api/v4/user', 400),
            ]:
                async with session.post(
                    f'http://{proxy.host}:5000/_nexus/website',
                    json={
                        'url': url,
                        'method': 'GET',
                        'headers': {},
                        'body': '',
                    },
                ) as response:
                    assert response.status == expected
    runtime.run_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_pages_requires_explicit_exact_origin():
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                return_code=0,
                stdout='NEXUS_WEBSITE_RESPONSE='
                + json.dumps(
                    {
                        'status': 200,
                        'headers': {},
                        'body': base64.b64encode(b'published').decode(),
                    }
                ),
            )
        )
    )
    async with Transport(
        runtime,
        [],
        'localhost',
        gitlab_origin=('http', 'localhost', 80),
        gitlab_pages_origin=('http', 'pages.localhost', 80),
    ) as proxy:
        async with aiohttp.ClientSession() as session:
            for url, expected in [
                ('http://pages.localhost/user/project/', 200),
                ('http://pages.localhost:5000/', 400),
                ('http://arbitrary.pages.localhost/', 400),
            ]:
                async with session.post(
                    f'http://{proxy.host}:5000/_nexus/website',
                    json={
                        'url': url,
                        'method': 'GET',
                        'headers': {},
                        'body': '',
                    },
                ) as response:
                    assert response.status == expected


@pytest.mark.asyncio
async def test_gitlab_readiness_waits_for_boot_service(monkeypatch):
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import wait_local_gitlab

    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout='LoadState=loaded\nActiveState=activating\nResult=success'),
                SimpleNamespace(return_code=0, stdout='LoadState=loaded\nActiveState=active\nResult=success'),
                SimpleNamespace(return_code=0, stdout='NEXUS_GITLAB_READY'),
            ]
        )
    )
    monkeypatch.setattr('nexus.extensions.tasks.os_world_v2.adapter.adapter.asyncio.sleep', AsyncMock())
    await wait_local_gitlab(runtime, 30, initial_wait=0)
    assert runtime.run_command.await_count == 3


@pytest.mark.asyncio
async def test_gitlab_readiness_rejects_failed_service():
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import wait_local_gitlab

    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                return_code=0,
                stdout='LoadState=loaded\nActiveState=failed\nResult=exit-code',
            )
        )
    )
    with pytest.raises(RuntimeError, match='GitLab startup logs'):
        await wait_local_gitlab(runtime, 30, initial_wait=0)


@pytest.mark.asyncio
async def test_gitlab_sleeps_ten_minutes_before_first_probe(monkeypatch):
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import wait_local_gitlab

    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout='ActiveState=active\nResult=success'),
                SimpleNamespace(return_code=0, stdout='NEXUS_GITLAB_READY'),
            ]
        )
    )

    async def sleep(seconds):
        assert seconds == 600
        runtime.run_command.assert_not_awaited()

    monkeypatch.setattr('nexus.extensions.tasks.os_world_v2.adapter.adapter.asyncio.sleep', sleep)
    await wait_local_gitlab(runtime, 1800)
    assert runtime.run_command.await_count == 2


@pytest.mark.asyncio
async def test_gitlab_service_active_but_http_probe_fails_keeps_waiting(monkeypatch):
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import wait_local_gitlab

    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout='ActiveState=active\nResult=success'),
                SimpleNamespace(return_code=1, stdout=''),
                SimpleNamespace(return_code=0, stdout='ActiveState=active\nResult=success'),
                SimpleNamespace(return_code=0, stdout='NEXUS_GITLAB_READY'),
            ]
        )
    )
    sleep = AsyncMock()
    monkeypatch.setattr('nexus.extensions.tasks.os_world_v2.adapter.adapter.asyncio.sleep', sleep)
    await wait_local_gitlab(runtime, 30, initial_wait=0)
    assert runtime.run_command.await_count == 4
    assert [call.args[0] for call in sleep.await_args_list] == [0, 5]


@pytest.mark.asyncio
async def test_gitlab_missing_service_does_not_skip_readiness():
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import wait_local_gitlab

    runtime = SimpleNamespace(
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                return_code=0,
                stdout='LoadState=not-found\nActiveState=inactive',
            )
        )
    )
    with pytest.raises(TimeoutError, match='before the deadline'):
        await wait_local_gitlab(runtime, 0.01, initial_wait=0)


@pytest.mark.asyncio
async def test_gitlab_deadline_must_leave_time_after_initial_wait():
    from nexus.extensions.tasks.os_world_v2.adapter.adapter import wait_local_gitlab

    with pytest.raises(ValueError, match='must exceed'):
        await wait_local_gitlab(SimpleNamespace(), 600)


@pytest.mark.asyncio
async def test_worker_gitlab_api_uses_guest_and_preserves_auth(monkeypatch):
    import requests

    calls = []

    async def command(**kwargs):
        calls.append(kwargs.get('stdin', kwargs['command']))
        return SimpleNamespace(
            return_code=0,
            stdout='NEXUS_WEBSITE_RESPONSE='
            + json.dumps(
                {
                    'status': 201,
                    'headers': {'Content-Type': 'application/json'},
                    'body': base64.b64encode(b'{"id":42}').decode(),
                }
            ),
        )

    # Ensure this worker-only patch is restored after the unit test.
    monkeypatch.setattr(requests.Session, 'send', requests.Session.send)
    async with Transport(
        SimpleNamespace(run_command=command),
        [],
        'localhost',
        gitlab_origin=('http', 'localhost', 80),
    ) as proxy:
        install_requests_transport(proxy.host, [], 'localhost', [None], gitlab_origin=('http', 'localhost', 80))

        def request():
            session = requests.Session()
            session.trust_env = False
            return session.post(
                'http://localhost/api/v4/users',
                headers={'PRIVATE-TOKEN': 'unit-test-only'},
                json={'username': 'fixture'},
                timeout=5,
            )

        response = await asyncio.to_thread(request)
        assert response.status_code == 201
        assert response.json() == {'id': 42}
    assert 'PRIVATE-TOKEN' in calls[0]
    assert 'unit-test-only' in calls[0]
    assert 'http://localhost/api/v4/users' in calls[0]
