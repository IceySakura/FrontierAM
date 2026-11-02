"""Model-provider contracts for stem_mas benchmark runs."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def selector_text(value: Any) -> str:
    return str(value or '').strip().lower()


def require_text(value: Any, field_name: str) -> str:
    text = str(value or '').strip()
    if not text:
        raise ValueError(f'{field_name} is required')
    return text


def _usage_paths(*paths: str) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(part.strip() for part in path.split('.')) for path in paths)


class UsageSpec(BaseModel):
    """Provider-specific usage field paths."""

    model_config = ConfigDict(extra='forbid')
    path_fields: ClassVar[tuple[str, ...]] = (
        'cached_tokens',
        'cache_miss_tokens',
        'prompt_tokens',
        'completion_tokens',
        'total_tokens',
    )

    name: str = ''
    cached_tokens: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    cache_miss_tokens: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    prompt_tokens: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    completion_tokens: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)
    total_tokens: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)

    @field_validator(*path_fields, mode='before')
    @classmethod
    def parse_paths(cls, value: Any) -> tuple[tuple[str, ...], ...]:
        if value is None:
            return ()
        if isinstance(value, tuple) and all(
            isinstance(path, tuple) and all(isinstance(part, str) for part in path) for path in value
        ):
            return value
        if not isinstance(value, (list, tuple)) or not all(isinstance(path, str) for path in value):
            raise ValueError('usage paths must be lists of dot-path strings')

        paths = []
        for path in value:
            parts = tuple(part.strip() for part in path.split('.'))
            if not parts or any(not part for part in parts):
                raise ValueError('usage path entries must be non-empty dot paths')
            paths.append(parts)
        return tuple(paths)

    @field_validator('name', mode='before')
    @classmethod
    def normalize_name(cls, value: Any) -> str:
        return selector_text(value)

    def to_config(self) -> dict[str, Any]:
        config = self.model_dump(mode='json')
        for field_name in self.path_fields:
            config[field_name] = ['.'.join(path) for path in getattr(self, field_name)]
        return config


class ReasoningSpec(BaseModel):
    """Reasoning-generation intent before provider-specific request mapping."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    effort: str | None = None

    @field_validator('effort', mode='before')
    @classmethod
    def normalize_effort(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    @model_validator(mode='after')
    def validate_effort(self) -> ReasoningSpec:
        if self.enabled and not self.effort:
            raise ValueError('reasoning.effort is required when reasoning.enabled=true')
        if not self.enabled and self.effort:
            raise ValueError('reasoning.effort must be omitted when reasoning.enabled=false')
        return self


class GenerationSpec(BaseModel):
    """Generation knobs accepted by stem_mas benchmark model providers."""

    model_config = ConfigDict(extra='forbid')

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = Field(default=4096, gt=0)
    top_k: int | None = Field(default=None, gt=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    reasoning: ReasoningSpec = Field(default_factory=ReasoningSpec)

    def to_request_args(self) -> dict[str, Any]:
        request_args: dict[str, Any] = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens,
        }
        for key in ('top_k', 'min_p', 'repetition_penalty'):
            value = getattr(self, key)
            if value is not None:
                request_args[key] = value
        return request_args


class HyaOpenAPIContract(BaseModel):
    """Request/response contract for the stem_mas HYA OpenAPI client."""

    model_config = ConfigDict(extra='forbid')

    provider_name: str
    reasoning: ReasoningSpec = Field(default_factory=ReasoningSpec)
    max_tokens_cap: int | None = Field(default=None, gt=0)
    response_reasoning_field: str = 'reasoning'
    normalize_reasoning_to: str = 'reasoning_content'
    require_final_content: bool = True
    default_extra_body: dict[str, Any] = Field(
        default_factory=lambda: {
            'openai_infer': True,
            'stream': False,
            'add_generation_prompt': True,
        }
    )

    @field_validator('provider_name', mode='before')
    @classmethod
    def normalize_provider_name(cls, value: Any) -> str:
        text = selector_text(value)
        if not text:
            raise ValueError('hya_openapi_contract.provider_name is required')
        return text

    @field_validator('response_reasoning_field', 'normalize_reasoning_to', mode='before')
    @classmethod
    def validate_response_fields(cls, value: Any) -> str:
        text = str(value or '').strip()
        if not text:
            raise ValueError('HYA response field names must be non-empty')
        return text

    @field_validator('default_extra_body', mode='before')
    @classmethod
    def validate_default_extra_body(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('hya_openapi_contract.default_extra_body must be a dict')
        return dict(value)


OPENAI_USAGE_SPEC = UsageSpec(
    name='openai',
    cached_tokens=_usage_paths(
        'prompt_tokens_details.cached_tokens',
        'input_tokens_details.cached_tokens',
    ),
    cache_miss_tokens=_usage_paths(
        'prompt_tokens_details.cache_miss_tokens',
        'input_tokens_details.cache_miss_tokens',
    ),
    prompt_tokens=_usage_paths('prompt_tokens', 'input_tokens'),
    completion_tokens=_usage_paths('completion_tokens', 'output_tokens'),
    total_tokens=_usage_paths('total_tokens'),
)
DEEPSEEK_USAGE_SPEC = UsageSpec(
    name='deepseek',
    cached_tokens=_usage_paths(
        'prompt_cache_hit_tokens',
        'prompt_tokens_details.cached_tokens',
    ),
    cache_miss_tokens=_usage_paths(
        'prompt_cache_miss_tokens',
        'prompt_tokens_details.cache_miss_tokens',
    ),
    prompt_tokens=_usage_paths('prompt_tokens'),
    completion_tokens=_usage_paths('completion_tokens'),
    total_tokens=_usage_paths('total_tokens'),
)
TAIJI_USAGE_SPEC = UsageSpec(
    name='taiji',
    cached_tokens=_usage_paths('cost_info.cache_hit_tokens'),
    cache_miss_tokens=_usage_paths('cost_info.cache_miss_tokens'),
    prompt_tokens=_usage_paths('prompt_tokens', 'input_tokens'),
    completion_tokens=_usage_paths('completion_tokens', 'output_tokens'),
    total_tokens=_usage_paths('total_tokens'),
)
ZHIPU_USAGE_SPEC = UsageSpec(
    name='zhipu',
    cached_tokens=_usage_paths('prompt_tokens_details.cached_tokens'),
    prompt_tokens=_usage_paths('prompt_tokens'),
    completion_tokens=_usage_paths('completion_tokens'),
    total_tokens=_usage_paths('total_tokens'),
)
ANTHROPIC_USAGE_SPEC = UsageSpec(
    name='anthropic',
    cached_tokens=_usage_paths('cache_read_input_tokens'),
    cache_miss_tokens=_usage_paths('cache_creation_input_tokens'),
    prompt_tokens=_usage_paths('input_tokens'),
    completion_tokens=_usage_paths('output_tokens'),
)


class BenchmarkModelProvider(BaseModel):
    """User-facing aggregate root for one benchmark model provider."""

    model_config = ConfigDict(extra='forbid')

    name: str
    model: str
    timeout: float = Field(default=900.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)

    client_name: ClassVar[str] = 'openai'
    cache_usage_spec: ClassVar[UsageSpec]

    @field_validator('name', mode='before')
    @classmethod
    def normalize_name(cls, value: Any) -> str:
        return selector_text(value)

    @field_validator('model', mode='before')
    @classmethod
    def validate_model(cls, value: Any) -> str:
        return require_text(value, 'model_provider.model')

    @classmethod
    def usage_spec(cls) -> UsageSpec:
        return cls.cache_usage_spec

    def to_openai_client(self) -> dict[str, Any]:  # pragma: no cover - implemented by subclasses.
        raise NotImplementedError


class DeepSeekFlashOneStop(BenchmarkModelProvider):
    """DeepSeek Flash served by the one-stop HYA OpenAPI/vLLM platform."""

    name: Literal['deepseek_flash_one_stop'] = 'deepseek_flash_one_stop'
    service: str
    authorization: str
    wsid: str = '10103'
    max_tokens_cap: int | None = Field(default=None, gt=0)
    timeout: float = Field(default=36000.0, gt=0.0)
    max_retries: int = Field(default=25, ge=0)
    client_name: ClassVar[str] = 'stem_mas_hya_openapi'
    cache_usage_spec: ClassVar[UsageSpec] = OPENAI_USAGE_SPEC.model_copy(update={'name': 'deepseek_flash_one_stop'})

    @field_validator('service', 'authorization', 'wsid', mode='before')
    @classmethod
    def validate_required_text(cls, value: Any, info: Any) -> str:
        return require_text(value, f'model_provider.{info.field_name}')

    def hya_openapi_contract(self) -> HyaOpenAPIContract:
        return HyaOpenAPIContract(
            provider_name=self.name,
            reasoning=self.generation.reasoning,
            max_tokens_cap=self.max_tokens_cap,
            response_reasoning_field='reasoning',
        )

    def to_openai_client(self) -> dict[str, Any]:
        return {
            'name': self.client_name,
            'config': {
                'client_args': {
                    'model': self.model,
                    'service': self.service,
                    'authorization': self.authorization,
                    'wsid': self.wsid,
                    'timeout': self.timeout,
                    'max_retries': self.max_retries,
                    'hya_openapi_contract': self.hya_openapi_contract().model_dump(mode='json'),
                },
                'request_args': self.generation.to_request_args(),
            },
        }


class DeepSeekFlashDistillation(BenchmarkModelProvider):
    """DeepSeek Flash routed through the distillation OpenAI-compatible platform."""

    name: Literal['deepseek_flash_distillation'] = 'deepseek_flash_distillation'
    api_key: str
    base_url: str
    default_headers: dict[str, str] = Field(default_factory=lambda: {'X-route-env': 'response'})
    cache_usage_spec: ClassVar[UsageSpec] = DEEPSEEK_USAGE_SPEC.model_copy(
        update={'name': 'deepseek_flash_distillation'}
    )

    @field_validator('api_key', 'base_url', mode='before')
    @classmethod
    def validate_required_text(cls, value: Any, info: Any) -> str:
        return require_text(value, f'model_provider.{info.field_name}')

    @field_validator('default_headers', mode='before')
    @classmethod
    def validate_default_headers(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('model_provider.default_headers must be a dict')
        return {str(key): str(item) for key, item in value.items()}

    @model_validator(mode='after')
    def validate_generation(self) -> DeepSeekFlashDistillation:
        if self.generation.reasoning.enabled:
            raise ValueError('DeepSeekFlashDistillation does not support configurable reasoning')
        return self

    def to_openai_client(self) -> dict[str, Any]:
        return {
            'name': self.client_name,
            'config': {
                'client_args': {
                    'api_key': self.api_key,
                    'base_url': self.base_url,
                    'default_headers': self.default_headers,
                    'timeout': self.timeout,
                    'max_retries': self.max_retries,
                },
                'request_args': {
                    'model': self.model,
                    **self.generation.to_request_args(),
                },
            },
        }


MODEL_PROVIDER_CLASSES: dict[str, type[BenchmarkModelProvider]] = {}
REGISTERED_USAGE_SPECS: dict[str, UsageSpec] = {}


def register_model_provider(provider_cls: type[BenchmarkModelProvider]) -> type[BenchmarkModelProvider]:
    provider_name = selector_text(provider_cls.model_fields['name'].default)
    if not provider_name:
        raise ValueError(f'{provider_cls.__name__}.name default is required for provider registration')
    if provider_name in MODEL_PROVIDER_CLASSES:
        raise ValueError(f'duplicate model_provider registration: {provider_name!r}')
    usage_spec = provider_cls.usage_spec()
    if usage_spec.name != provider_name:
        raise ValueError(f'{provider_cls.__name__}.cache_usage_spec.name must match {provider_name!r}')
    MODEL_PROVIDER_CLASSES[provider_name] = provider_cls
    REGISTERED_USAGE_SPECS[provider_name] = usage_spec
    return provider_cls


for _provider_cls in (DeepSeekFlashOneStop, DeepSeekFlashDistillation):
    register_model_provider(_provider_cls)


def _builtin_usage_specs() -> dict[str, UsageSpec]:
    """Usage specs for explicit non-aggregate providers used by auxiliary benchmark clients."""

    return {
        spec.name: spec
        for spec in (
            OPENAI_USAGE_SPEC,
            TAIJI_USAGE_SPEC,
            DEEPSEEK_USAGE_SPEC,
            ZHIPU_USAGE_SPEC,
            ANTHROPIC_USAGE_SPEC,
        )
    }


def _usage_specs_by_model_provider() -> dict[str, UsageSpec]:
    specs = _builtin_usage_specs()
    for provider_name, usage_spec in REGISTERED_USAGE_SPECS.items():
        if provider_name in specs:
            raise ValueError(f'model_provider {provider_name!r} conflicts with a built-in usage spec')
        specs[provider_name] = usage_spec
    return specs


def usage_spec_for_model_provider(model_provider: str) -> UsageSpec:
    normalized = selector_text(model_provider)
    spec = _usage_specs_by_model_provider().get(normalized)
    if spec is None:
        raise ValueError(f'Unknown model_provider {model_provider!r}; register a BenchmarkModelProvider subclass')
    return spec


def parse_model_provider(value: Any) -> BenchmarkModelProvider:
    if not isinstance(value, dict):
        raise ValueError('model_provider must be a dict')
    provider_name = selector_text(value.get('name'))
    if not provider_name:
        raise ValueError('model_provider.name is required')
    provider_cls = MODEL_PROVIDER_CLASSES.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f'Unknown model_provider {provider_name!r}; register a BenchmarkModelProvider subclass in providers.py'
        )
    return provider_cls.model_validate(value)


__all__ = [
    'BenchmarkModelProvider',
    'DeepSeekFlashDistillation',
    'DeepSeekFlashOneStop',
    'GenerationSpec',
    'HyaOpenAPIContract',
    'ReasoningSpec',
    'UsageSpec',
    'parse_model_provider',
    'register_model_provider',
    'selector_text',
    'usage_spec_for_model_provider',
]
