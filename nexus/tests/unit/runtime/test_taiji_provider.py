from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.providers import gongfeng as gf
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.runtime.providers.taiji import (
    BACKEND_PROVIDER_NAME,
    TaijiRuntimeProvider,
)


VALID_RESOURCE_SPEC = {'cpu': 4, 'memory': 16, 'gpu': 1, 'gpu_card_type': 'H20'}
VALID_APP_GROUP = 'TaiJi_HYAide_T3_Nexus_Sandbox'
VALID_APP_GROUP_LOCATION = 'gy'
APP_GROUP_REQUIRED = 'taiji_app_group and taiji_app_group_location are required'
PASSTHROUGH_PARAMS = (
    'enable_dind',
    'runtime_disable_aslr',
    'cap_add',
    'ttl',
    'need_transparent_proxy',
    'network_policy',
)
EFFECTIVE_VALUES = {
    'enable_dind': True,
    'runtime_disable_aslr': True,
    'cap_add': ['NET_ADMIN'],
    'ttl': '2h',
    'need_transparent_proxy': ['http'],
    'network_policy': {'default_action': 'deny'},
}
CLEARED_VALUES = {
    'enable_dind': 'false',
    'runtime_disable_aslr': 'false',
    'cap_add': [],
    'ttl': '',
    'need_transparent_proxy': None,
    'network_policy': {},
}


def _make_provider(**overrides) -> TaijiRuntimeProvider:
    kwargs = {
        'api_base_url': 'http://sandbox.example.com',
        'token': 'unit-test-token',
        'image': 'unit-test/image:latest',
        'print_port_mappings': False,
        'taiji_app_group': VALID_APP_GROUP,
        'taiji_app_group_location': VALID_APP_GROUP_LOCATION,
    }
    kwargs.update(overrides)
    return TaijiRuntimeProvider(**kwargs)


def _stub_acquire(provider: TaijiRuntimeProvider) -> None:
    provider._start_runtime = AsyncMock(  # type: ignore[method-assign]
        return_value=('rid', 'http://runtime', {}, gf.RUNTIME_STATUS_SCHEDULED)
    )
    fake_runtime = MagicMock()
    fake_runtime.port_mappings = {}
    provider._create_runtime = MagicMock(return_value=fake_runtime)  # type: ignore[method-assign]
    provider._start_watchdog = MagicMock()  # type: ignore[method-assign]


def test_value_tables_cover_every_passthrough_param():
    assert set(EFFECTIVE_VALUES) == set(CLEARED_VALUES) == set(PASSTHROUGH_PARAMS)


# --- runtime_provider pinning ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_and_resume_raise_not_implemented() -> None:
    provider = _make_provider()
    provider._call_api = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(NotImplementedError, match='does not support pause'):
        await provider.pause()
    with pytest.raises(NotImplementedError, match='does not support resume'):
        await provider.resume('main')

    provider._call_api.assert_not_awaited()


def test_runtime_provider_is_pinned_to_taiji():
    assert _make_provider().runtime_provider == BACKEND_PROVIDER_NAME


def test_init_rejects_conflicting_runtime_provider():
    with pytest.raises(ValueError, match='pins runtime_provider'):
        _make_provider(runtime_provider='gongfeng')


def test_configure_rejects_conflicting_runtime_provider():
    provider = _make_provider()

    with pytest.raises(ValueError, match='pins runtime_provider'):
        provider.configure('main', runtime_provider='gongfeng')


@pytest.mark.asyncio
async def test_acquire_rejects_conflicting_runtime_provider():
    provider = _make_provider()

    with pytest.raises(ValueError, match='pins runtime_provider'):
        await provider.acquire('main', runtime_provider='gongfeng')


def test_init_accepts_redundant_runtime_provider():
    assert _make_provider(runtime_provider=BACKEND_PROVIDER_NAME).runtime_provider == BACKEND_PROVIDER_NAME


def test_api_base_url_stays_positional_like_the_parent():
    provider = TaijiRuntimeProvider(
        'http://sandbox.example.com',
        token='unit-test-token',
        taiji_app_group=VALID_APP_GROUP,
        taiji_app_group_location=VALID_APP_GROUP_LOCATION,
    )

    assert provider.api_base_url == 'http://sandbox.example.com'
    assert provider.runtime_provider == BACKEND_PROVIDER_NAME


def test_token_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path):
    key_file = tmp_path / 'gongfeng' / 'key'
    monkeypatch.setattr(gf, 'GONGFENG_AUTH_KEY_FILE', key_file)
    monkeypatch.delenv('GONGFENG_TOKEN', raising=False)

    with pytest.raises(ValueError, match='token is required'):
        TaijiRuntimeProvider(
            'http://sandbox.example.com',
            print_port_mappings=False,
            taiji_app_group=VALID_APP_GROUP,
            taiji_app_group_location=VALID_APP_GROUP_LOCATION,
        )


def test_dummy_token_inits_without_quota_fields():
    provider = TaijiRuntimeProvider(
        'http://sandbox.example.com',
        token='unit-test-token',
        print_port_mappings=False,
        taiji_app_group=VALID_APP_GROUP,
        taiji_app_group_location=VALID_APP_GROUP_LOCATION,
    )

    assert provider.token == 'unit-test-token'


# --- capability passthrough -----------------------------------------------------------------------


@pytest.mark.parametrize('param', PASSTHROUGH_PARAMS)
def test_init_accepts_passthrough_param(param):
    provider = _make_provider(**{param: EFFECTIVE_VALUES[param]})
    provider.configure('main', **{param: EFFECTIVE_VALUES[param]})


@pytest.mark.parametrize('param', PASSTHROUGH_PARAMS)
def test_cleared_values_are_still_accepted(param):
    provider = _make_provider(**{param: CLEARED_VALUES[param]})
    provider.configure('main', **{param: CLEARED_VALUES[param]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('kwargs', 'payload_key', 'payload_value'),
    [
        ({'enable_dind': True}, 'enable_dind', True),
        ({'runtime_disable_aslr': True}, 'runtime_disable_aslr', True),
        ({'cap_add': ['NET_ADMIN']}, 'cap_add', ['NET_ADMIN']),
        ({'ttl': '2h'}, 'ttl', '2h'),
        ({'need_transparent_proxy': ['http']}, 'need_transparent_proxy', ['http']),
        (
            {'network_policy': {'default_action': 'deny'}},
            'network_policy',
            {'default_action': 'deny'},
        ),
    ],
)
async def test_start_payload_forwards_passthrough_param(kwargs, payload_key, payload_value):
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', **kwargs)

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['runtime_provider'] == BACKEND_PROVIDER_NAME
    assert payload[payload_key] == payload_value


@pytest.mark.asyncio
async def test_start_payload_omits_cleared_passthrough_params():
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img')

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['runtime_provider'] == BACKEND_PROVIDER_NAME
    assert [param for param in PASSTHROUGH_PARAMS if param in payload] == []
    assert 'taiji_app_group' not in payload
    assert 'taiji_app_group_location' not in payload


@pytest.mark.asyncio
async def test_acquire_forwards_passthrough_params_to_start():
    provider = _make_provider()
    _stub_acquire(provider)

    await provider.acquire(
        'main',
        resource_spec=VALID_RESOURCE_SPEC,
        enable_dind=True,
        ttl='2h',
        cap_add=['SYS_PTRACE'],
    )

    kwargs = provider._start_runtime.await_args.kwargs
    assert kwargs['enable_dind'] is True
    assert kwargs['ttl'] == '2h'
    assert kwargs['cap_add'] == ['SYS_PTRACE']
    assert kwargs['resource_spec']['gpu_card_type'] == 'H20'


# --- ttl 48h cap ----------------------------------------------------------------------------------


def test_init_accepts_ttl_48h():
    assert _make_provider(ttl='48h').ttl == '48h'


def test_init_rejects_ttl_above_48h():
    with pytest.raises(ValueError, match='taiji ttl must not exceed 48h'):
        _make_provider(ttl='49h')


def test_init_rejects_ttl_168h():
    with pytest.raises(ValueError, match='taiji ttl must not exceed 48h'):
        _make_provider(ttl='168h')


@pytest.mark.asyncio
async def test_start_payload_carries_ttl_48h():
    provider = _make_provider()
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', ttl='48h')

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['ttl'] == '48h'


# --- application group pair -----------------------------------------------------------------------


def test_init_rejects_missing_app_group_pair():
    with pytest.raises(ValueError, match=APP_GROUP_REQUIRED):
        TaijiRuntimeProvider(
            api_base_url='http://sandbox.example.com',
            token='unit-test-token',
            image='unit-test/image:latest',
            print_port_mappings=False,
        )


def test_init_rejects_empty_app_group_pair():
    with pytest.raises(ValueError, match=APP_GROUP_REQUIRED):
        _make_provider(taiji_app_group='', taiji_app_group_location='')


def test_init_rejects_app_group_without_location():
    with pytest.raises(ValueError, match=APP_GROUP_REQUIRED):
        _make_provider(taiji_app_group=VALID_APP_GROUP, taiji_app_group_location='')


def test_init_rejects_location_without_app_group():
    with pytest.raises(ValueError, match=APP_GROUP_REQUIRED):
        _make_provider(taiji_app_group='', taiji_app_group_location=VALID_APP_GROUP_LOCATION)


def test_init_accepts_app_group_pair():
    provider = _make_provider(
        taiji_app_group=VALID_APP_GROUP,
        taiji_app_group_location=VALID_APP_GROUP_LOCATION,
    )
    assert provider.taiji_app_group == VALID_APP_GROUP
    assert provider.taiji_app_group_location == VALID_APP_GROUP_LOCATION


def test_configure_rejects_one_sided_app_group():
    provider = _make_provider()
    with pytest.raises(ValueError, match=APP_GROUP_REQUIRED):
        provider.configure('main', taiji_app_group=VALID_APP_GROUP)


@pytest.mark.asyncio
async def test_acquire_rejects_one_sided_app_group():
    provider = _make_provider()
    with pytest.raises(ValueError, match=APP_GROUP_REQUIRED):
        await provider.acquire('main', resource_spec=VALID_RESOURCE_SPEC, taiji_app_group_location='zw')


@pytest.mark.asyncio
async def test_start_payload_carries_app_group_pair():
    provider = _make_provider(
        taiji_app_group='TaiJi_HYAide_T3_Nexus_Sandbox',
        taiji_app_group_location='gy',
    )
    provider._call_api = AsyncMock(  # type: ignore[method-assign]
        return_value={'code': 0, 'data': {'runtime_id': 'rid', 'url': 'http://runtime', 'port_mappings': {}}}
    )

    await provider._start_runtime(image='img', extra_payload=provider._start_payload_extras({}))

    _endpoint, payload = provider._call_api.await_args.args
    assert payload['taiji_app_group'] == 'TaiJi_HYAide_T3_Nexus_Sandbox'
    assert payload['taiji_app_group_location'] == 'gy'


@pytest.mark.asyncio
async def test_acquire_overrides_provider_app_group_pair():
    provider = _make_provider(
        taiji_app_group='TaiJi_Default',
        taiji_app_group_location='gy',
    )
    _stub_acquire(provider)

    await provider.acquire(
        'main',
        resource_spec=VALID_RESOURCE_SPEC,
        taiji_app_group='TaiJi_HYAide_T3_Nexus_Sandbox',
        taiji_app_group_location='zw',
    )

    extras = provider._start_runtime.await_args.kwargs['extra_payload']
    assert extras == {
        'taiji_app_group': 'TaiJi_HYAide_T3_Nexus_Sandbox',
        'taiji_app_group_location': 'zw',
    }


@pytest.mark.asyncio
async def test_acquire_sends_provider_app_group_pair():
    provider = _make_provider()
    _stub_acquire(provider)

    await provider.acquire('main', resource_spec=VALID_RESOURCE_SPEC)

    assert provider._start_runtime.await_args.kwargs['extra_payload'] == {
        'taiji_app_group': VALID_APP_GROUP,
        'taiji_app_group_location': VALID_APP_GROUP_LOCATION,
    }


# --- registration / GPU card type -----------------------------------------------------------------


def test_registered_under_taiji_name():
    assert runtime_provider_registry.get('taiji') is TaijiRuntimeProvider


def test_init_accepts_valid_gpu_card_type():
    provider = _make_provider(resource_spec=VALID_RESOURCE_SPEC)
    assert provider.resource_spec['gpu_card_type'] == 'H20'


def test_init_rejects_missing_gpu_card_type_when_spec_present():
    with pytest.raises(ValueError, match='gpu_card_type'):
        _make_provider(resource_spec={'cpu': 4, 'memory': 16, 'gpu': 1})


def test_init_passes_through_unknown_gpu_card_type():
    # 卡型可用性由沙盒后端按平台配置裁定，nexus 只透传
    provider = _make_provider(resource_spec={'gpu': 1, 'gpu_card_type': 'H100'})
    assert provider.resource_spec['gpu_card_type'] == 'H100'


def test_init_passes_through_lowercase_gpu_card_type():
    provider = _make_provider(resource_spec={'gpu': 1, 'gpu_card_type': 'h20'})
    assert provider.resource_spec['gpu_card_type'] == 'h20'


def test_configure_rejects_missing_gpu_card_type():
    provider = _make_provider()
    with pytest.raises(ValueError, match='gpu_card_type'):
        provider.configure('main', resource_spec={'gpu': 1})


@pytest.mark.asyncio
async def test_acquire_rejects_missing_gpu_card_type():
    provider = _make_provider()
    with pytest.raises(ValueError, match='gpu_card_type'):
        await provider.acquire('main')


@pytest.mark.asyncio
async def test_acquire_accepts_gpu_card_type_from_provider_default():
    provider = _make_provider(resource_spec=VALID_RESOURCE_SPEC)
    _stub_acquire(provider)

    await provider.acquire('main')

    assert provider._start_runtime.await_args.kwargs['resource_spec']['gpu_card_type'] == 'H20'
