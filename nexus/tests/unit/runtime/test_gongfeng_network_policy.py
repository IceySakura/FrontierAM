from typing import Any
from unittest.mock import AsyncMock

import pytest

from nexus.runtime.providers.gongfeng import GongfengRuntimeProvider


_validate = GongfengRuntimeProvider._validate_network_policy


def _make_provider() -> GongfengRuntimeProvider:
    return GongfengRuntimeProvider(
        api_base_url='http://gongfeng.example.com',
        token='unit-test-token',
        image='unit-test/image:latest',
        print_port_mappings=False,
    )


def _provider_with_runtime(name: str = 'main', runtime_id: str = 'nexus-abc') -> GongfengRuntimeProvider:
    provider = _make_provider()
    provider._runtime_ids[name] = runtime_id
    provider._call_api = AsyncMock(return_value={'code': 0, 'msg': 'ok'})  # type: ignore[method-assign]
    return provider


def _update_payload(provider: GongfengRuntimeProvider) -> dict[str, Any]:
    endpoint, payload = provider._call_api.await_args.args  # type: ignore[attr-defined]
    assert endpoint == '/sandbox/v3/runtime/update'
    return payload


def test_default_action_omitted_falls_back_to_allow():
    # MR 760 regression: the spec/docstring promise default_action defaults to 'allow' when omitted,
    # so the minimal egress-only policy must validate instead of raising.
    result = _validate({'egress': [{'action': 'allow', 'target': '*.tencent.com'}]})

    assert result == {'default_action': 'allow', 'egress': [{'action': 'allow', 'target': '*.tencent.com'}]}


def test_explicit_default_action_is_preserved():
    result = _validate(
        {
            'default_action': 'deny',
            'egress': [{'action': 'allow', 'target': '*.woa.com'}],
        }
    )

    assert result == {'default_action': 'deny', 'egress': [{'action': 'allow', 'target': '*.woa.com'}]}


def test_explicit_invalid_default_action_still_raises():
    with pytest.raises(ValueError, match='default_action'):
        _validate({'default_action': 'foo', 'egress': []})


def test_invalid_egress_action_raises():
    with pytest.raises(ValueError, match=r'egress\[0\]\.action'):
        _validate({'default_action': 'deny', 'egress': [{'action': 'maybe', 'target': '*.woa.com'}]})


def test_empty_egress_target_raises():
    with pytest.raises(ValueError, match=r'egress\[0\]\.target'):
        _validate({'default_action': 'deny', 'egress': [{'action': 'allow', 'target': '   '}]})


def test_none_and_empty_return_none():
    assert _validate(None) is None
    assert _validate({}) is None


def test_stray_keys_are_dropped():
    result = _validate({'default_action': 'allow', 'egress': [], 'extra': 1})

    # exact-equality already proves the stray 'extra' key was dropped.
    assert result == {'default_action': 'allow', 'egress': []}


def test_proxy_rules_are_rejected_on_the_create_path():
    # The computing platform's create API does not take proxy rules: rejecting beats dropping,
    # which would read as "I set the rule and it did nothing".
    with pytest.raises(ValueError, match='update_runtime'):
        _validate({'default_action': 'deny', 'proxy': [{'action': 'direct', 'target': '10.0.0.0/8'}]})


def test_proxy_rules_are_normalized_when_allowed():
    result = _validate(
        {
            'default_action': 'deny',
            'egress': [{'action': 'allow', 'target': '*.pypi.org'}],
            'proxy': [
                {'action': 'direct', 'target': '10.0.0.0/8'},
                {
                    'action': 'proxy',
                    'target': '0.0.0.0/0',
                    'to': {'http': 'http://1.2.3.4:3180', 'default': 'socks5://1.2.3.4:3280', 'https': ''},
                },
            ],
        },
        allow_proxy=True,
    )

    assert result == {
        'default_action': 'deny',
        'egress': [{'action': 'allow', 'target': '*.pypi.org'}],
        'proxy': [
            {'action': 'direct', 'target': '10.0.0.0/8'},
            {
                'action': 'proxy',
                'target': '0.0.0.0/0',
                # the empty 'https' upstream is dropped rather than sent as an empty string
                'to': {'http': 'http://1.2.3.4:3180', 'default': 'socks5://1.2.3.4:3280'},
            },
        ],
    }


def test_policy_without_proxy_rules_keeps_the_old_shape():
    # Guards the start-path payload: no 'proxy' key appears for callers that never used one.
    assert _validate({'default_action': 'deny'}, allow_proxy=True) == {'default_action': 'deny', 'egress': []}


def test_invalid_proxy_action_raises():
    with pytest.raises(ValueError, match=r'proxy\[0\]\.action'):
        _validate({'proxy': [{'action': 'forward', 'target': 'example.com'}]}, allow_proxy=True)


def test_empty_proxy_target_raises():
    with pytest.raises(ValueError, match=r'proxy\[0\]\.target'):
        _validate({'proxy': [{'action': 'direct', 'target': '  '}]}, allow_proxy=True)


def test_direct_rule_with_upstream_raises():
    with pytest.raises(ValueError, match='must be empty when action is "direct"'):
        _validate(
            {'proxy': [{'action': 'direct', 'target': 'example.com', 'to': {'http': 'http://1.2.3.4:3180'}}]},
            allow_proxy=True,
        )


def test_proxy_rule_without_upstream_raises():
    with pytest.raises(ValueError, match=r'proxy\[0\]\.to requires at least one'):
        _validate({'proxy': [{'action': 'proxy', 'target': 'example.com', 'to': {}}]}, allow_proxy=True)


@pytest.mark.asyncio
async def test_update_runtime_sends_the_resolved_runtime_id_and_policy():
    provider = _provider_with_runtime(runtime_id='nexus-0869b9aa')

    await provider.update_runtime(
        'main',
        network_policy={
            'default_action': 'deny',
            'egress': [{'action': 'allow', 'target': '*.pypi.org'}],
            'proxy': [{'action': 'proxy', 'target': '0.0.0.0/0', 'to': {'http': 'http://1.2.3.4:3180'}}],
        },
    )

    assert _update_payload(provider) == {
        'runtime_id': 'nexus-0869b9aa',
        'network_policy': {
            'default_action': 'deny',
            'egress': [{'action': 'allow', 'target': '*.pypi.org'}],
            'proxy': [{'action': 'proxy', 'target': '0.0.0.0/0', 'to': {'http': 'http://1.2.3.4:3180'}}],
        },
    }


@pytest.mark.asyncio
async def test_update_runtime_without_any_facet_raises_before_calling_the_api():
    provider = _provider_with_runtime()

    # {} means "facet not supplied"; resetting the policy is spelled {'default_action': 'allow'}.
    for policy in (None, {}):
        with pytest.raises(ValueError, match='at least one facet'):
            await provider.update_runtime('main', network_policy=policy)

    provider._call_api.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_update_runtime_on_unknown_name_raises():
    provider = _provider_with_runtime(name='main')

    with pytest.raises(ValueError, match='No active runtime'):
        await provider.update_runtime('eval', network_policy={'default_action': 'allow'})

    provider._call_api.assert_not_awaited()  # type: ignore[attr-defined]
