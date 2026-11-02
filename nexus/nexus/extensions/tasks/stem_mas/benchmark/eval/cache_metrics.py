# /// script
# dependencies = []
#
# [tool.nexus]
# span_processors = { stem_mas_benchmark_cache_metrics = "nexus.extensions.tasks.stem_mas.benchmark.eval.cache_metrics" }
# ///

"""Prompt-cache usage metrics for stem_mas benchmark runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexus.extensions.tasks.stem_mas.benchmark.eval.providers import UsageSpec, selector_text
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import SpanProcessor, span_processor_registry
from nexus.trace.tracer import update_span
from nexus.trace.types import Span


class CacheUsageExtraction(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_metric_present: bool = False
    cache_hit_provable: bool = False
    cache_metric_paths: dict[str, str | None] = Field(default_factory=dict)
    provider_spec: str = ''


class CacheClientMetadata(BaseModel):
    """Validated runtime metadata derived from the benchmark RunPart."""

    model_config = ConfigDict(extra='forbid')

    name: str = ''
    provider: str = ''
    model_provider: str
    model: str = ''
    usage_spec: UsageSpec
    cache_session_id: str = ''
    cache_strategy: str = ''

    @field_validator('model_provider', mode='before')
    @classmethod
    def normalize_model_provider(cls, value: Any) -> str:
        model_provider = selector_text(value)
        if not model_provider:
            raise ValueError('model_provider is required')
        return model_provider

    @model_validator(mode='after')
    def validate_usage_spec_name(self) -> CacheClientMetadata:
        if self.usage_spec.name != self.model_provider:
            raise ValueError('usage_spec.name must match model_provider')
        return self


class CacheUsageExtractor:
    """Extracts cache usage fields from a provider-specific usage payload."""

    def __init__(self, spec: UsageSpec) -> None:
        self.spec = spec

    def extract(self, usage: dict[str, Any]) -> CacheUsageExtraction:
        cached_value, cached_path = self._first_int(usage, self.spec.cached_tokens)
        miss_value, miss_path = self._first_int(usage, self.spec.cache_miss_tokens)
        prompt_tokens = self._first_int(usage, self.spec.prompt_tokens)[0]
        completion_tokens = self._first_int(usage, self.spec.completion_tokens)[0]
        total_tokens = self._first_int(usage, self.spec.total_tokens)[0] or prompt_tokens + completion_tokens
        cached_tokens = cached_value or 0
        cache_miss_tokens = miss_value or 0

        return CacheUsageExtraction(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_miss_tokens=cache_miss_tokens,
            cache_metric_present=cached_path is not None or miss_path is not None,
            cache_hit_provable=cached_tokens > 0,
            cache_metric_paths={
                'cached_tokens': self._path_text(cached_path),
                'cache_miss_tokens': self._path_text(miss_path),
            },
            provider_spec=self.spec.name,
        )

    @classmethod
    def _first_int(cls, data: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> tuple[int, tuple[str, ...] | None]:
        for path in paths:
            value = cls._nested(data, path)
            if value is not None:
                return cls._strict_int(value, path), path
        return 0, None

    @staticmethod
    def _nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = data
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @classmethod
    def _strict_int(cls, value: Any, path: tuple[str, ...]) -> int:
        if isinstance(value, bool):
            raise ValueError(f'usage field {cls._path_text(path)} must be an integer, got bool')
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(f'usage field {cls._path_text(path)} must be an integer, got {value!r}')
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                return int(text)
        raise ValueError(f'usage field {cls._path_text(path)} must be an integer, got {value!r}')

    @staticmethod
    def _path_text(path: tuple[str, ...] | None) -> str | None:
        if path is None:
            return None
        return '.'.join(path)


class CacheMetricCall(BaseModel):
    trace_id: str = ''
    span_id: str = ''
    client_key: str
    provider: str = ''
    model_provider: str = ''
    model: str = ''
    cache_session_id: str = ''
    cache_strategy: str = ''
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_miss_tokens: int = 0
    non_cached_prompt_tokens: int = 0
    cache_metric_present: bool = False
    cache_hit_provable: bool = False
    cache_metric_paths: dict[str, str | None] = Field(default_factory=dict)
    model_provider_spec: str = ''


class CacheMetricSummary(BaseModel):
    provider: str = ''
    model_provider: str = ''
    model: str = ''
    cache_session_ids: list[str] = Field(default_factory=list)
    cache_strategies: list[str] = Field(default_factory=list)
    llm_call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_miss_tokens: int = 0
    non_cached_prompt_tokens: int = 0
    cache_metric_present_count: int = 0
    cache_hit_provable_count: int = 0
    cache_metric_present: bool = False
    cache_hit_provable: bool = False


@span_processor_registry.register('stem_mas_benchmark_cache_metrics')
class StemMasBenchmarkCacheMetricsProcessor(SpanProcessor):
    """Collect cache-token evidence from benchmark LLM spans."""

    def __init__(
        self,
        client_metadata: dict[str, dict[str, Any]] | None = None,
        emit_trace_update: bool = True,
    ) -> None:
        self.client_metadata = self._parse_client_metadata(client_metadata)
        self.emit_trace_update = emit_trace_update
        self.calls: list[dict[str, Any]] = []
        self.by_client: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _parse_client_metadata(
        client_metadata: dict[str, dict[str, Any]] | None,
    ) -> dict[str, CacheClientMetadata]:
        if client_metadata is None:
            return {}
        if not isinstance(client_metadata, dict):
            raise ValueError('cache metrics client_metadata must be a dict')
        return {
            str(client_key): CacheClientMetadata.model_validate(metadata)
            for client_key, metadata in client_metadata.items()
        }

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        if span.name != 'openai_completion':
            return

        outputs = span.attributes.get('outputs')
        if not isinstance(outputs, dict):
            return

        usage = outputs.get('usage')
        if not isinstance(usage, dict):
            usage = {}

        client_key = str(span.attributes.get('_client_key') or '_unknown_client_')
        metadata = self.client_metadata.get(client_key)
        if metadata is None:
            raise ValueError(f'cache metrics missing client_metadata for {client_key!r}')
        model = _model_from_span(span, outputs, metadata.model)
        provider = metadata.provider or metadata.name
        extraction = CacheUsageExtractor(metadata.usage_spec).extract(usage)
        cache_routing = _cache_routing_from_span(span)
        cache_session_id = cache_routing.get('session_id') or metadata.cache_session_id
        cache_strategy = cache_routing.get('strategy') or metadata.cache_strategy

        call = CacheMetricCall(
            trace_id=span.trace_id,
            span_id=span.span_id,
            client_key=client_key,
            provider=provider,
            model_provider=metadata.model_provider,
            model=model,
            cache_session_id=cache_session_id,
            cache_strategy=cache_strategy,
            prompt_tokens=extraction.prompt_tokens,
            completion_tokens=extraction.completion_tokens,
            total_tokens=extraction.total_tokens,
            cached_tokens=extraction.cached_tokens,
            cache_miss_tokens=extraction.cache_miss_tokens,
            non_cached_prompt_tokens=max(extraction.prompt_tokens - extraction.cached_tokens, 0),
            cache_metric_present=extraction.cache_metric_present,
            cache_hit_provable=extraction.cache_hit_provable,
            cache_metric_paths=extraction.cache_metric_paths,
            model_provider_spec=extraction.provider_spec,
        ).model_dump()
        self.calls.append(call)
        self._add_to_client_summary(call)
        self._emit_trace_update(span, call)

    def _add_to_client_summary(self, call: dict[str, Any]) -> None:
        client_key = call['client_key']
        summary = self.by_client.setdefault(
            client_key,
            CacheMetricSummary(
                provider=call['provider'],
                model_provider=call['model_provider'],
                model=call['model'],
            ).model_dump(),
        )
        if call['model'] and not summary.get('model'):
            summary['model'] = call['model']
        if call['provider'] and not summary.get('provider'):
            summary['provider'] = call['provider']
        if call['model_provider'] and not summary.get('model_provider'):
            summary['model_provider'] = call['model_provider']

        for summary_key, call_key in (
            ('cache_session_ids', 'cache_session_id'),
            ('cache_strategies', 'cache_strategy'),
        ):
            value = str(call.get(call_key) or '')
            if value and value not in summary[summary_key]:
                summary[summary_key].append(value)

        summary['llm_call_count'] += 1
        for key in (
            'prompt_tokens',
            'completion_tokens',
            'total_tokens',
            'cached_tokens',
            'cache_miss_tokens',
            'non_cached_prompt_tokens',
        ):
            summary[key] += int(call.get(key) or 0)
        if call['cache_metric_present']:
            summary['cache_metric_present_count'] += 1
            summary['cache_metric_present'] = True
        if call['cache_hit_provable']:
            summary['cache_hit_provable_count'] += 1
            summary['cache_hit_provable'] = True

    def _emit_trace_update(self, span: Span, call: dict[str, Any]) -> None:
        if not self.emit_trace_update:
            return
        update_span(
            span.span_id,
            attributes={
                'cache_metrics': call,
                'cache_hit_provable': call['cache_hit_provable'],
                'cache_metric_present': call['cache_metric_present'],
                'cached_tokens': call['cached_tokens'],
                'cache_session_id': call['cache_session_id'],
                'cache_strategy': call['cache_strategy'],
            },
        )

    def get_result(self) -> dict[str, Any]:
        return {
            'by_client': self.by_client,
            'calls': self.calls,
        }


def _model_from_span(span: Span, outputs: dict[str, Any], metadata_model: str) -> str:
    output_model = outputs.get('model')
    if output_model:
        return str(output_model)

    inputs = span.attributes.get('inputs')
    if isinstance(inputs, dict):
        input_model = inputs.get('model')
        if input_model:
            return str(input_model)
        kwargs = inputs.get('kwargs')
        if isinstance(kwargs, dict) and kwargs.get('model'):
            return str(kwargs['model'])

    return metadata_model


def _cache_routing_from_span(span: Span) -> dict[str, str]:
    inputs = span.attributes.get('inputs')
    if not isinstance(inputs, dict):
        return {'session_id': '', 'strategy': ''}

    candidates: list[dict[str, Any]] = []
    kwargs = inputs.get('kwargs')
    if isinstance(kwargs, dict):
        candidates.append(kwargs)
        extra_headers = kwargs.get('extra_headers')
        if isinstance(extra_headers, dict):
            candidates.append(extra_headers)
    extra_headers = inputs.get('extra_headers')
    if isinstance(extra_headers, dict):
        candidates.append(extra_headers)

    session_id = ''
    strategy = ''
    for headers in candidates:
        session_id = session_id or _header_value(headers, 'SessionID')
        strategy = strategy or _header_value(headers, 'StrategyType')
    return {'session_id': session_id, 'strategy': strategy}


def _header_value(headers: dict[str, Any], name: str) -> str:
    for key, value in headers.items():
        if str(key).lower() == name.lower():
            return str(value or '')
    return ''


__all__ = [
    'CacheClientMetadata',
    'CacheMetricCall',
    'CacheMetricSummary',
    'CacheUsageExtractor',
    'CacheUsageExtraction',
    'StemMasBenchmarkCacheMetricsProcessor',
    'UsageSpec',
]
