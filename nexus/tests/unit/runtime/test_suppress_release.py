"""Tests for RuntimeProvider suppress release mechanism."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.providers.base import RuntimeProvider


class ConcreteProvider(RuntimeProvider):
    """Test provider with trackable release calls."""

    def __init__(self) -> None:
        super().__init__()
        self.release_impl_calls: list[str] = []
        self.release_all_impl_called: bool = False

    async def acquire(self, name: str, **params: Any) -> Any:
        runtime = MagicMock()
        runtime.base_url = f'http://fake:32000/{name}'
        self._active_runtimes[name] = runtime
        return runtime

    async def _release_impl(self, name: str) -> None:
        self.release_impl_calls.append(name)
        self._active_runtimes.pop(name, None)

    async def _release_all_impl(self) -> None:
        self.release_all_impl_called = True
        names = list(self._active_runtimes.keys())
        for name in names:
            await self._release_impl(name)


@pytest.fixture
def provider() -> ConcreteProvider:
    return ConcreteProvider()


class TestSuppressReleaseDefault:
    """When _suppress_release is False (default), release works normally."""

    @pytest.mark.asyncio
    async def test_release_calls_impl(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        await provider.release('main')
        assert provider.release_impl_calls == ['main']
        assert 'main' not in provider._active_runtimes

    @pytest.mark.asyncio
    async def test_release_all_calls_impl(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        await provider.acquire('eval')
        await provider.release_all()
        assert set(provider.release_impl_calls) == {'main', 'eval'}

    @pytest.mark.asyncio
    async def test_suppress_flag_defaults_false(self, provider: ConcreteProvider) -> None:
        assert provider._suppress_release is False


class TestSuppressReleaseEnabled:
    """When _suppress_release is True, release calls become no-ops."""

    @pytest.mark.asyncio
    async def test_release_suppressed(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        provider._suppress_release = True
        await provider.release('main')
        assert provider.release_impl_calls == []
        assert 'main' in provider._active_runtimes

    @pytest.mark.asyncio
    async def test_release_all_suppressed(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        await provider.acquire('eval')
        provider._suppress_release = True
        await provider.release_all()
        assert provider.release_impl_calls == []
        assert 'main' in provider._active_runtimes
        assert 'eval' in provider._active_runtimes

    @pytest.mark.asyncio
    async def test_release_works_after_suppress_disabled(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        provider._suppress_release = True
        await provider.release('main')
        assert provider.release_impl_calls == []

        provider._suppress_release = False
        await provider.release('main')
        assert provider.release_impl_calls == ['main']

    @pytest.mark.asyncio
    async def test_release_all_works_after_suppress_disabled(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        await provider.acquire('eval')
        provider._suppress_release = True
        await provider.release_all()
        assert provider.release_impl_calls == []

        provider._suppress_release = False
        await provider.release_all()
        assert set(provider.release_impl_calls) == {'main', 'eval'}


class TestReleaseAllImplOverride:
    """Subclass with custom _release_all_impl is also suppressed."""

    @pytest.mark.asyncio
    async def test_custom_release_all_impl_suppressed(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        provider._suppress_release = True
        await provider.release_all()
        assert provider.release_all_impl_called is False

    @pytest.mark.asyncio
    async def test_custom_release_all_impl_called_when_not_suppressed(self, provider: ConcreteProvider) -> None:
        await provider.acquire('main')
        await provider.release_all()
        assert provider.release_all_impl_called is True
