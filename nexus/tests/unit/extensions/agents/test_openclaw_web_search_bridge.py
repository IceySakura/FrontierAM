import pytest

from nexus.extensions.agents.openclaw.web_search_bridge import (
    DEFAULT_PUBLIC_SERPER_URL,
    SerperBridgeConfig,
    UnsupportedBraveQueryError,
    _sanitize_detail,
    build_serper_request,
    map_serper_to_brave,
    resolve_serper_bridge_config,
)


def test_map_serper_response_to_brave_results() -> None:
    mapped = map_serper_to_brave(
        {
            'organic': [
                {
                    'title': 'Nexus runtime',
                    'link': 'https://example.com/runtime',
                    'snippet': 'Runtime docs.',
                    'date': '2 days ago',
                },
                {
                    'title': 'Missing link is skipped',
                    'snippet': 'No URL here.',
                },
            ]
        }
    )

    assert mapped == {
        'web': {
            'results': [
                {
                    'title': 'Nexus runtime',
                    'url': 'https://example.com/runtime',
                    'description': 'Runtime docs.',
                    'age': '2 days ago',
                }
            ]
        }
    }


def test_build_public_serper_request_uses_x_api_key() -> None:
    config = SerperBridgeConfig(api_key='public-key', api_type='official')

    request = build_serper_request(query='openclaw', count=7, config=config)

    assert request.url == DEFAULT_PUBLIC_SERPER_URL
    assert request.headers == {'Content-Type': 'application/json', 'X-API-KEY': 'public-key'}
    assert request.body == {'q': 'openclaw', 'num': 7}


def test_build_distillation_serper_request_uses_gateway_authorization() -> None:
    config = SerperBridgeConfig(
        api_key='app-id:app-key',
        api_type='distillation',
        search_url='http://search-gateway.local/search',
    )

    request = build_serper_request(query='nexus bridge', count=99, config=config)

    assert request.url == 'http://search-gateway.local/search'
    assert request.headers == {
        'Authorization': 'Bearer app-id:app-key?provider=serper',
        'Content-Type': 'application/json',
    }
    assert request.body == {'q': 'nexus bridge', 'num': 10}


def test_build_serper_request_maps_brave_locale_and_freshness() -> None:
    config = SerperBridgeConfig(api_key='public-key', api_type='official')

    request = build_serper_request(
        query='openclaw recent',
        count='3',
        config=config,
        search_lang='zh',
        ui_lang='zh-CN',
        country='CN',
        freshness='pw',
    )

    assert request.body == {'q': 'openclaw recent', 'num': 3, 'hl': 'zh', 'gl': 'CN', 'tbs': 'qdr:w'}


def test_build_serper_request_maps_brave_date_range_freshness() -> None:
    config = SerperBridgeConfig(api_key='public-key', api_type='official')

    request = build_serper_request(
        query='openclaw date range',
        count='3',
        config=config,
        freshness='2026-06-01to2026-06-04',
    )

    assert request.body == {
        'q': 'openclaw date range',
        'num': 3,
        'tbs': 'cdr:1,cd_min:6/1/2026,cd_max:6/4/2026',
    }


def test_build_serper_request_rejects_unsupported_brave_query_params() -> None:
    config = SerperBridgeConfig(api_key='public-key', api_type='official')

    with pytest.raises(UnsupportedBraveQueryError, match='unsupported Brave freshness value'):
        build_serper_request(query='openclaw', count=3, config=config, freshness='last_30_minutes')

    with pytest.raises(UnsupportedBraveQueryError, match='cannot be mapped to one Serper hl'):
        build_serper_request(query='openclaw', count=3, config=config, search_lang='zh', ui_lang='en-US')


def test_resolve_config_derives_key_from_distillation_app_credentials() -> None:
    config = resolve_serper_bridge_config(
        {
            'SERPER_API_TYPE': 'distillation',
            'SEARCH_SERPER_URL': 'http://search-gateway.local/search',
            'SEARCH_SERPER_APP_ID': 'app-id',
            'SEARCH_SERPER_APP_KEY': 'app-key',
        }
    )

    assert config is not None
    assert config.api_type == 'distillation'
    assert config.api_key == 'app-id:app-key'
    assert config.search_url == 'http://search-gateway.local/search'


def test_resolve_config_requires_explicit_distillation_search_url() -> None:
    assert resolve_serper_bridge_config({'SERPER_API_TYPE': 'distillation', 'SERPER_API_KEY': 'app-id:app-key'}) is None


def test_build_distillation_serper_request_requires_search_url() -> None:
    config = SerperBridgeConfig(api_key='app-id:app-key', api_type='distillation')

    with pytest.raises(ValueError, match='SEARCH_SERPER_URL'):
        build_serper_request(query='nexus bridge', count=3, config=config)


def test_resolve_config_returns_none_without_credentials() -> None:
    assert resolve_serper_bridge_config({}) is None


def test_sanitize_detail_redacts_full_key_and_distillation_parts() -> None:
    config = SerperBridgeConfig(api_key='app-id:app-key', api_type='distillation')

    sanitized = _sanitize_detail(
        'Bearer app-id:app-key?provider=serper failed for app-id with secret app-key',
        config,
    )

    assert 'app-id' not in sanitized
    assert 'app-key' not in sanitized
    assert sanitized == 'Bearer [redacted] failed for [redacted] with secret [redacted]'
