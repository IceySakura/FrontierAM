from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.providers import gongfeng as gf
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


_validate = GongfengRuntimeProvider._validate_cap_add


def _make_gongfeng_provider(**overrides) -> GongfengRuntimeProvider:
    return GongfengRuntimeProvider(
        api_base_url='http://gongfeng.example.com',
        token='unit-test-token',
        image='unit-test/image:latest',
        print_port_mappings=False,
        **overrides,
    )


# --- _validate_cap_add: input handling ------------------------------------------------------------


def test_none_returns_none():
    assert _validate(None) is None


def test_empty_list_returns_none():
    # Empty list == "clear / not configured": nothing to send.
    assert _validate([]) is None


@pytest.mark.parametrize('falsy', ['', {}, 0])
def test_falsy_non_list_raises_instead_of_being_swallowed(falsy):
    # Regression: `if not cap_add` used to run BEFORE the type check, so these falsy non-list
    # values were silently treated as "unset" instead of being rejected.
    with pytest.raises(ValueError, match='cap_add must be a list'):
        _validate(falsy)


def test_non_list_truthy_raises():
    with pytest.raises(ValueError, match='cap_add must be a list'):
        _validate('NET_ADMIN')


# --- _validate_cap_add: per-item name validation --------------------------------------------------


def test_valid_names_pass_and_are_stripped():
    assert _validate(['NET_ADMIN', '  SYS_PTRACE  ']) == ['NET_ADMIN', 'SYS_PTRACE']


def test_duplicates_are_deduped():
    assert _validate(['NET_ADMIN', 'NET_ADMIN']) == ['NET_ADMIN']


@pytest.mark.parametrize('bad', ['NET-ADMIN', 'NET ADMIN', '123', 'net_admin', '_FOO', '1CAP'])
def test_malformed_names_raise(bad):
    # Hyphen / space / pure-digit / lowercase / leading underscore or digit must all be rejected.
    # `cap == cap.upper()` alone let the first three through.
    with pytest.raises(ValueError, match=r'cap_add\[0\]'):
        _validate([bad])


def test_cap_prefix_rejected_with_dedicated_message():
    # 'CAP_NET_ADMIN' matches [A-Z][A-Z0-9_]*, so the CAP_ prefix needs its own earlier check.
    with pytest.raises(ValueError, match='must not carry the CAP_ prefix'):
        _validate(['CAP_NET_ADMIN'])


def test_non_string_entry_raises():
    with pytest.raises(ValueError, match=r'cap_add\[1\] must be a non-empty string'):
        _validate(['NET_ADMIN', 123])


def test_blank_entry_raises():
    with pytest.raises(ValueError, match=r'cap_add\[0\] must be a non-empty string'):
        _validate(['   '])


# --- provider / acquire wiring --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_default_cap_add_reaches_start_payload():
    provider = _make_gongfeng_provider(cap_add=['NET_ADMIN'])
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', cap_add=provider.cap_add)

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['cap_add'] == ['NET_ADMIN']


@pytest.mark.asyncio
async def test_cap_add_absent_from_payload_by_default():
    provider = _make_gongfeng_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img')

    _endpoint, payload = provider._call_api.await_args.args
    assert 'cap_add' not in payload


@pytest.mark.asyncio
async def test_acquire_override_replaces_provider_default():
    provider = _make_gongfeng_provider(cap_add=['NET_ADMIN'])
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    await provider.acquire('main', cap_add=['SYS_PTRACE'])

    assert provider._start_runtime.await_args.kwargs['cap_add'] == ['SYS_PTRACE']


@pytest.mark.asyncio
async def test_acquire_empty_list_clears_provider_default():
    provider = _make_gongfeng_provider(cap_add=['NET_ADMIN'])
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]

    await provider.acquire('main', cap_add=[])

    # [] normalizes to None -> not forwarded as a cap_add list to the backend.
    assert provider._start_runtime.await_args.kwargs['cap_add'] is None


@pytest.mark.asyncio
async def test_acquire_rejects_malformed_cap_add():
    provider = _make_gongfeng_provider()

    with pytest.raises(ValueError, match=r'cap_add\[0\]'):
        await provider.acquire('main', cap_add=['net_admin'])


def test_provider_init_rejects_malformed_cap_add():
    with pytest.raises(ValueError, match=r'cap_add\[0\]'):
        _make_gongfeng_provider(cap_add=['NET-ADMIN'])
