from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.providers import gongfeng as gf
from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


_validate = GongfengRuntimeProvider._validate_ttl


def _make_gongfeng_provider(**overrides) -> GongfengRuntimeProvider:
    return GongfengRuntimeProvider(
        api_base_url='http://gongfeng.example.com',
        token='unit-test-token',
        image='unit-test/image:latest',
        print_port_mappings=False,
        **overrides,
    )


# --- _validate_ttl: input handling ----------------------------------------------------------------


def test_none_returns_none():
    assert _validate(None) is None


@pytest.mark.parametrize('blank', ['', '   '])
def test_blank_returns_none(blank):
    # Empty == "clear / not configured": nothing to send, the backend applies its own default.
    assert _validate(blank) is None


@pytest.mark.parametrize('not_a_string', [12, 12.0, ['12h'], {'ttl': '12h'}])
def test_non_string_raises(not_a_string):
    with pytest.raises(ValueError, match='ttl must be a Go duration string'):
        _validate(not_a_string)


# --- _validate_ttl: Go duration grammar -----------------------------------------------------------


@pytest.mark.parametrize('value', ['30m', '12h', '168h', '1h30m', '90s', '1.5h'])
def test_valid_durations_pass(value):
    assert _validate(value) == value


@pytest.mark.parametrize('value', ['500ms', '1000000us', '1ns'])
def test_sub_second_units_are_rejected(value):
    # Go's ParseDuration accepts these, but a sub-second lifetime can never produce a usable
    # sandbox, so they are rejected here instead of being forwarded.
    with pytest.raises(ValueError, match='sub-second units are not accepted'):
        _validate(value)


def test_surrounding_whitespace_is_stripped():
    assert _validate('  12h  ') == '12h'


@pytest.mark.parametrize('value', ['7d', '1d12h', '2 days', '1w', '1y'])
def test_day_week_year_units_get_a_dedicated_hint(value):
    # Go duration has no unit above hours; the generic grammar error wouldn't tell the caller
    # to rewrite 7 days as 168h.
    with pytest.raises(ValueError, match='does not support'):
        _validate(value)


@pytest.mark.parametrize('value', ['12', 'abc', '12 h', '12hh', 'h', '12h30', '--1h'])
def test_malformed_durations_raise(value):
    with pytest.raises(ValueError, match='must be a Go duration string made of'):
        _validate(value)


@pytest.mark.parametrize('value', ['0s', '0h', '-1h', '-30m'])
def test_non_positive_durations_raise(value):
    with pytest.raises(ValueError, match='must be positive'):
        _validate(value)


@pytest.mark.parametrize('value', ['169h', '10080m1s', '2h168h'])
def test_over_the_backend_maximum_raises(value):
    with pytest.raises(ValueError, match='must not exceed 168h'):
        _validate(value)


def test_exactly_the_maximum_passes():
    # 168h is the backend's inclusive upper bound (MaxSuanliSandboxTTL).
    assert _validate('168h') == '168h'
    assert gf.MAX_TTL_SECONDS == 168 * 3600


# --- provider / acquire wiring --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_default_ttl_reaches_start_payload():
    provider = _make_gongfeng_provider(ttl='12h')
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', ttl=provider.ttl)

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['ttl'] == '12h'


@pytest.mark.asyncio
async def test_ttl_absent_from_payload_by_default():
    # Not sending the field is what lets the backend tell "caller didn't ask" apart from
    # "caller explicitly asked for 12h" — the two differ in routing and expiry accounting.
    provider = _make_gongfeng_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img')

    _endpoint, payload = provider._call_api.await_args.args
    assert 'ttl' not in payload


def _stub_acquire_path(provider: GongfengRuntimeProvider) -> None:
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_acquire_override_replaces_provider_default():
    provider = _make_gongfeng_provider(ttl='12h')
    _stub_acquire_path(provider)

    await provider.acquire('main', ttl='30m')

    assert provider._start_runtime.await_args.kwargs['ttl'] == '30m'


@pytest.mark.asyncio
async def test_acquire_empty_string_clears_provider_default():
    provider = _make_gongfeng_provider(ttl='12h')
    _stub_acquire_path(provider)

    await provider.acquire('main', ttl='')

    # '' normalizes to None -> field omitted, so the backend applies its own default.
    assert provider._start_runtime.await_args.kwargs['ttl'] is None


@pytest.mark.asyncio
async def test_acquire_inherits_provider_default_when_unset():
    provider = _make_gongfeng_provider(ttl='12h')
    _stub_acquire_path(provider)

    await provider.acquire('main')

    assert provider._start_runtime.await_args.kwargs['ttl'] == '12h'


@pytest.mark.asyncio
async def test_acquire_rejects_malformed_ttl():
    provider = _make_gongfeng_provider()

    with pytest.raises(ValueError, match='does not support'):
        await provider.acquire('main', ttl='7d')


def test_provider_init_rejects_malformed_ttl():
    with pytest.raises(ValueError, match='must not exceed 168h'):
        _make_gongfeng_provider(ttl='169h')
