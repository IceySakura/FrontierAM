"""Runtime/client config helpers for stem_mas benchmark generation."""

from __future__ import annotations

from typing import Any

from nexus.extensions.tasks.stem_mas.benchmark.eval.config_models import (
    CacheMetricsConfig,
    OpenAIClientSpecConfig,
)
from nexus.extensions.tasks.stem_mas.benchmark.eval.providers import (
    BenchmarkModelProvider,
    parse_model_provider,
    usage_spec_for_model_provider,
)


class BenchmarkRuntimeConfig:
    CACHE_SESSION_HEADER = 'SessionID'
    CACHE_STRATEGY_HEADER = 'StrategyType'
    HYA_CLIENT_IMPORT = 'nexus.extensions.tasks.stem_mas.benchmark.eval.openai_clients'
    CACHE_METRICS_IMPORT = 'nexus.extensions.tasks.stem_mas.benchmark.eval.cache_metrics'
    HYA_OPENAPI_CLIENT = 'stem_mas_hya_openapi'
    CACHE_METRICS_PROCESSOR = 'stem_mas_benchmark_cache_metrics'
    CACHE_METRICS_KEY = 'cache_metrics'
    MODEL_PROVIDER_KEY = 'model_provider'
    FORBIDDEN_TOP_LEVEL_KEYS = (
        'cache_provider',
        'provider_specs',
        'platform_profile',
        'request_profile',
        'model_providers',
        'provider_runtimes',
    )

    @classmethod
    def ensure_cache_metrics_processor(
        cls,
        config: dict,
        *,
        cache_metrics_config: CacheMetricsConfig | None = None,
        client_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        model_provider = cls.model_provider_from_config(config)
        if model_provider is not None:
            cls.apply_model_provider(config, model_provider)
            config.pop(cls.MODEL_PROVIDER_KEY, None)

        cls.require_runtime_import(config, cls.CACHE_METRICS_IMPORT, reason='benchmark cache metrics')
        processors = config.setdefault('processors', {})
        if not isinstance(processors, dict):
            raise ValueError('processors must be a dict when benchmark cache metrics is enabled')
        resolved_cache_metrics_config = cache_metrics_config or cls.cache_metrics_config(config)

        processor_config = processors.setdefault(
            cls.CACHE_METRICS_KEY,
            {
                'name': cls.CACHE_METRICS_PROCESSOR,
                'args': {},
            },
        )
        if not isinstance(processor_config, dict):
            raise ValueError(f'processors.{cls.CACHE_METRICS_KEY} must be a dict')
        args = processor_config.setdefault('args', {})
        if not isinstance(args, dict):
            raise ValueError(f'processors.{cls.CACHE_METRICS_KEY}.args must be a dict')
        args.setdefault(
            'client_metadata',
            client_metadata
            if client_metadata is not None
            else cls.openai_client_metadata(
                config,
                resolved_cache_metrics_config,
                main_model_provider=model_provider,
            ),
        )

    @classmethod
    def model_provider_from_config(cls, config: dict) -> BenchmarkModelProvider | None:
        cls.validate_forbidden_config(config)
        raw_config = config.get(cls.MODEL_PROVIDER_KEY)
        if raw_config is None:
            return None
        return parse_model_provider(raw_config)

    @classmethod
    def apply_model_provider(
        cls,
        config: dict,
        model_provider: BenchmarkModelProvider,
    ) -> None:
        openai_clients = config.setdefault('openai_clients', {})
        if not isinstance(openai_clients, dict):
            raise ValueError('openai_clients must be a dict')
        openai_clients['main'] = model_provider.to_openai_client()

        if model_provider.client_name == cls.HYA_OPENAPI_CLIENT:
            cls.require_runtime_import(config, cls.HYA_CLIENT_IMPORT, reason='stem_mas HYA OpenAPI client')

    @classmethod
    def require_runtime_import(cls, config: dict, import_path: str, *, reason: str) -> None:
        imports = config.setdefault('imports', [])
        if not isinstance(imports, list):
            raise ValueError(f'imports must be a list when {reason} is enabled')
        if import_path not in imports:
            imports.append(import_path)

    @classmethod
    def cache_metrics_config(cls, config: dict) -> CacheMetricsConfig:
        raw_config = config.get(cls.CACHE_METRICS_KEY)
        if raw_config is None:
            return CacheMetricsConfig()
        return CacheMetricsConfig.model_validate(raw_config)

    @classmethod
    def validate_forbidden_config(cls, config: dict) -> None:
        for key in cls.FORBIDDEN_TOP_LEVEL_KEYS:
            if key in config:
                raise ValueError(f'top-level {key} is not supported')

    @classmethod
    def openai_client_metadata(
        cls,
        config: dict,
        cache_metrics_config: CacheMetricsConfig | None = None,
        *,
        main_model_provider: BenchmarkModelProvider | None = None,
    ) -> dict[str, dict[str, Any]]:
        cls.validate_forbidden_config(config)
        openai_clients = config.get('openai_clients')
        if openai_clients is None:
            return {}
        if not isinstance(openai_clients, dict):
            raise ValueError('openai_clients must be a dict')

        resolved_cache_metrics_config = cache_metrics_config or cls.cache_metrics_config(config)
        client_overrides = resolved_cache_metrics_config.client_overrides

        result: dict[str, dict[str, Any]] = {}
        for client_key, client_spec in openai_clients.items():
            client = OpenAIClientSpecConfig.model_validate(client_spec)
            client_key_text = str(client_key)
            name = client.name
            provider = name
            model = client.config.request_args.get('model') or client.config.client_args.get('model') or ''
            model_text = str(model) if model is not None else ''
            if main_model_provider is not None and client_key_text == 'main':
                if client_key_text in client_overrides:
                    raise ValueError('cache_metrics.client_overrides.main must be omitted when model_provider is set')
                model_provider = main_model_provider.name
                usage_spec = main_model_provider.usage_spec()
            else:
                cache_override = client_overrides.get(client_key_text)
                if cache_override is None:
                    raise ValueError(f'cache_metrics.client_overrides.{client_key_text}.model_provider is required')
                model_provider = cache_override.model_provider
                usage_spec = usage_spec_for_model_provider(model_provider)
            cache_routing = cls.cache_routing_from_request_args(client.config.request_args)
            result[client_key_text] = {
                'name': name,
                'provider': provider,
                'model': model_text,
                'model_provider': model_provider,
                'usage_spec': usage_spec.to_config(),
                'cache_session_id': cache_routing.get('session_id', ''),
                'cache_strategy': cache_routing.get('strategy', ''),
            }
        return result

    @classmethod
    def cache_routing_from_request_args(cls, request_args: dict[str, Any]) -> dict[str, str]:
        extra_headers = request_args.get('extra_headers')
        if not isinstance(extra_headers, dict):
            return {'session_id': '', 'strategy': ''}
        return {
            'session_id': cls.header_value(extra_headers, cls.CACHE_SESSION_HEADER),
            'strategy': cls.header_value(extra_headers, cls.CACHE_STRATEGY_HEADER),
        }

    @staticmethod
    def header_value(headers: dict[str, Any], name: str) -> str:
        for key, value in headers.items():
            if str(key).lower() == name.lower():
                return str(value or '')
        return ''


__all__ = ['BenchmarkRuntimeConfig']
