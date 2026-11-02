from __future__ import annotations

from typing import Any

import pytest

from nexus.core.context import ExecutionContext
from nexus.runtime import InvalidPersistentRuntimeError, RuntimeProvider


class _Runtime:
    pass


class _Provider(RuntimeProvider):
    def __init__(self, events: list[str], *, acquire_error: BaseException | None = None) -> None:
        super().__init__()
        self.events = events
        self.acquire_error = acquire_error
        self.acquire_params: dict[str, Any] | None = None

    async def acquire(self, name: str, **params: Any) -> Any:
        self.acquire_params = self.get_params(name, **params)
        if self.acquire_error is not None:
            raise self.acquire_error
        runtime = _Runtime()
        self._active_runtimes[name] = runtime  # type: ignore[assignment]
        return runtime

    async def _release_impl(self, name: str) -> None:
        self.events.append('runtime-release')
        self._active_runtimes.pop(name, None)


class _Account:
    def __init__(self, events: list[str], credentials: dict[str, Any]) -> None:
        self.events = events
        self.credentials = credentials
        self.invalidated_with: str | None = None
        self.released = False

    async def invalidate(self, reason: str = '') -> None:
        self.invalidated_with = reason
        self.events.append('account-invalidate')

    async def release(self) -> None:
        if not self.released:
            self.released = True
            self.events.append('account-release')


class _AccountManager:
    def __init__(self, account: _Account) -> None:
        self.account = account
        self.acquire_args: tuple[str, int | None, str, list[str] | None] | None = None

    async def acquire(
        self,
        pool: str,
        *,
        ttl_seconds: int | None = None,
        reservation_token: str = '',
        tags: list[str] | None = None,
    ) -> Any:
        self.acquire_args = (pool, ttl_seconds, reservation_token, tags)
        return self.account

    async def release_all(self) -> None:
        await self.account.release()

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runtime_pool_passes_credentials_as_one_provider_argument_and_releases_on_exit() -> None:
    events: list[str] = []
    provider = _Provider(events)
    account = _Account(events, {'instance_id': 'sandbox-123'})
    manager = _AccountManager(account)
    context = ExecutionContext(
        runtime_provider=provider,
        runtime_configs={
            'main': {
                'runtime_pool': {
                    'name': 'persistent-ags-osworld',
                    'lease_ttl_seconds': 4200,
                    'tags': ['chrome', 'vscode'],
                },
            }
        },
    )
    context._account_pool_manager = manager  # type: ignore[assignment]

    async with context:
        await context.get_runtime()

    assert manager.acquire_args == ('persistent-ags-osworld', 4200, '', ['chrome', 'vscode'])
    assert provider.acquire_params == {'runtime_credential': {'instance_id': 'sandbox-123'}}
    assert events == ['account-release', 'runtime-release']


class _InvalidRuntimeSubclassError(InvalidPersistentRuntimeError):
    pass


@pytest.mark.asyncio
async def test_runtime_pool_invalidates_and_reraises_explicit_invalid_runtime_error() -> None:
    events: list[str] = []
    error = _InvalidRuntimeSubclassError('persistent sandbox is gone')
    provider = _Provider(events, acquire_error=error)
    account = _Account(events, {'instance_id': 'sandbox-gone'})
    manager = _AccountManager(account)
    context = ExecutionContext(
        runtime_provider=provider,
        runtime_configs={'main': {'runtime_pool': {'name': 'persistent-ags-osworld'}}},
    )
    context._account_pool_manager = manager  # type: ignore[assignment]

    with pytest.raises(_InvalidRuntimeSubclassError, match='persistent sandbox is gone'):
        await context.get_runtime()

    assert account.invalidated_with == 'persistent sandbox is gone'
    assert account.released is False
    assert events == ['account-invalidate']


@pytest.mark.asyncio
async def test_runtime_pool_releases_and_reraises_other_acquire_errors() -> None:
    events: list[str] = []
    provider = _Provider(events, acquire_error=TimeoutError('temporary control-plane timeout'))
    account = _Account(events, {'instance_ip': '10.0.0.42'})
    manager = _AccountManager(account)
    context = ExecutionContext(
        runtime_provider=provider,
        runtime_configs={'main': {'runtime_pool': {'name': 'persistent-devcloud-macos'}}},
    )
    context._account_pool_manager = manager  # type: ignore[assignment]

    with pytest.raises(TimeoutError, match='temporary control-plane timeout'):
        await context.get_runtime()

    assert account.invalidated_with is None
    assert account.released is True
    assert events == ['account-release']


@pytest.mark.asyncio
async def test_runtime_pool_post_acquire_error_does_not_release_provider_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    provider = _Provider(events)
    account = _Account(events, {'instance_id': 'sandbox-123'})
    context = ExecutionContext(
        runtime_provider=provider,
        runtime_configs={
            'main': {
                'runtime_pool': {'name': 'persistent-ags-osworld'},
                'worker_venv': True,
            }
        },
    )
    context._account_pool_manager = _AccountManager(account)  # type: ignore[assignment]

    async def fail_setup_worker_venv(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('worker venv setup failed')

    monkeypatch.setattr(ExecutionContext, '_setup_worker_venv', fail_setup_worker_venv)

    with pytest.raises(RuntimeError, match='worker venv setup failed'):
        await context.get_runtime()

    assert events == ['account-release']
    assert 'main' in provider._active_runtimes


@pytest.mark.asyncio
async def test_runtime_credential_can_be_configured_without_runtime_pool() -> None:
    events: list[str] = []
    provider = _Provider(events)
    provider.configure('main', runtime_credential={'instance_id': 'standalone-sandbox'})
    context = ExecutionContext(runtime_provider=provider)

    await context.get_runtime()

    assert provider.acquire_params == {'runtime_credential': {'instance_id': 'standalone-sandbox'}}


@pytest.mark.asyncio
async def test_release_runtime_releases_account_after_provider() -> None:
    events: list[str] = []
    provider = _Provider(events)
    account = _Account(events, {'instance_id': 'sandbox-123'})
    context = ExecutionContext(
        runtime_provider=provider,
        runtime_configs={'main': {'runtime_pool': {'name': 'persistent-ags-osworld'}}},
    )
    context._account_pool_manager = _AccountManager(account)  # type: ignore[assignment]

    await context.get_runtime()

    assert await context.release_runtime() is True
    assert events == ['runtime-release', 'account-release']
    assert 'main' not in context._runtimes
    assert 'main' not in context._runtime_accounts


@pytest.mark.asyncio
async def test_suppressed_release_keeps_runtime_account_until_forced() -> None:
    events: list[str] = []
    provider = _Provider(events)
    account = _Account(events, {'instance_id': 'sandbox-123'})
    context = ExecutionContext(
        runtime_provider=provider,
        runtime_configs={'main': {'runtime_pool': {'name': 'persistent-ags-osworld'}}},
    )
    context._account_pool_manager = _AccountManager(account)  # type: ignore[assignment]
    await context.get_runtime()
    provider._suppress_release = True

    assert await context.release_runtime() is False
    assert events == []
    assert 'main' in context._runtimes
    assert 'main' in context._runtime_accounts

    assert await context.release_runtime(force=True) is True
    assert events == ['runtime-release', 'account-release']
