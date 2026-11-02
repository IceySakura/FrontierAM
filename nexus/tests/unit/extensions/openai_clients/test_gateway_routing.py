import pytest

from nexus.extensions.openai_clients.gateway_routing import merge_gateway_routing


def test_merge_gateway_routing_preserves_credential_and_existing_cache_task_id() -> None:
    result = merge_gateway_routing(
        'app-id:app-key?cache_task_id=task%2Fone&provider=old&timeout=10',
        {
            'provider': 'google',
            'model': 'fable 5',
            'api': 'claude_api',
            'timeout': 600,
        },
    )

    assert result == (
        'app-id:app-key?cache_task_id=task%2Fone&provider=google&timeout=600&model=fable+5&api=claude_api'
    )
    assert result.count('provider=') == 1
    assert result.count('timeout=') == 1


def test_merge_gateway_routing_accepts_structured_cache_task_id() -> None:
    result = merge_gateway_routing(
        'app-id:app-key',
        {'provider': 'google', 'cache_task_id': 'task/one'},
    )

    assert result == 'app-id:app-key?provider=google&cache_task_id=task%2Fone'


def test_merge_gateway_routing_empty_mapping_returns_original_key_exactly() -> None:
    api_key = 'credential?provider=one&provider=two'

    assert merge_gateway_routing(api_key, {}) is api_key


@pytest.mark.parametrize('api_key', ['', '   ', None])
def test_merge_gateway_routing_rejects_empty_api_key(api_key) -> None:
    with pytest.raises(ValueError, match='api_key'):
        merge_gateway_routing(api_key, {'provider': 'google'})


def test_merge_gateway_routing_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match='mapping'):
        merge_gateway_routing('credential', [('provider', 'google')])


@pytest.mark.parametrize('value', [None, True, 1.5, ['google'], {'name': 'google'}])
def test_merge_gateway_routing_rejects_non_scalar_value(value) -> None:
    with pytest.raises(TypeError, match='string or integer'):
        merge_gateway_routing('credential', {'provider': value})


def test_merge_gateway_routing_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match='Unsupported'):
        merge_gateway_routing('credential', {'secret': 'value'})


def test_merge_gateway_routing_rejects_non_string_parameter_name() -> None:
    with pytest.raises(TypeError, match='parameter names'):
        merge_gateway_routing('credential', {1: 'google'})
