"""Single-model Compose YAML loading and per-judge model bindings."""

from __future__ import annotations

import hashlib
import json
import os
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.core import Registry, openai_client_registry
from nexus.core.openai_client import OpenAIConfig
from nexus.runner.compose import deep_merge
from nexus.runner.compose.model import RawRunPart, UseSpec
from nexus.runner.compose.resolver import resolve_non_data_part
from nexus.runner.single import RunInput


MODEL_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'llm'
_FILE_NAME = re.compile(r'[a-z0-9][a-z0-9.-]*-aihub-2\.0\.yaml')
_ENV_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_PLACEHOLDER = re.compile(r'<[A-Za-z_][A-Za-z0-9_-]*>')
_SECRET = re.compile(r'api[_-]?key|authorization|password|secret|access[_-]?token', re.I)


class ProfileError(ValueError):
    """Configuration error with credential-safe diagnostics."""


def _cache_task_id_override() -> str:
    value = os.environ.get('JUDGE_CACHE_TASK_ID', '').strip()
    if _PLACEHOLDER.search(value) or '${' in value:
        raise ProfileError('JUDGE_CACHE_TASK_ID must not contain unresolved placeholders')
    return value


def _validate_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProfileError('Model configuration keys must be strings')
            if key.endswith('__env'):
                if not isinstance(child, str) or not _ENV_NAME.fullmatch(child):
                    raise ProfileError('Environment references must contain variable names')
            elif _SECRET.search(key):
                raise ProfileError('Secrets must use __env references, not literal values')
            _validate_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _validate_secrets(child)
    elif isinstance(value, str) and _PLACEHOLDER.search(value):
        raise ProfileError('Model configuration contains an unresolved placeholder')


def _resolve_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key.endswith('__env'):
                target = key[:-5]
                if not target or target in value:
                    raise ProfileError('Conflicting literal and environment-reference fields')
                secret = os.environ.get(child)
                if not secret or not secret.strip() or _PLACEHOLDER.search(secret):
                    raise ProfileError(f'Missing or placeholder environment variable: {child}')
                result[target] = secret
            else:
                result[key] = _resolve_secrets(child)
        return result
    if isinstance(value, list):
        return [_resolve_secrets(child) for child in value]
    return value


@dataclass(frozen=True)
class ModelConfig:
    filename: str
    source: str
    client_spec: dict[str, Any] = field(repr=False)
    agent_config: dict[str, Any] = field(repr=False)
    merge: dict[str, str] = field(repr=False)
    imports: tuple[str, ...]
    config_sha256: str

    @property
    def model(self) -> str:
        config = self.client_spec['config']
        return config.get('enforced_request_args', {}).get('model') or config['request_args']['model']

    def validate_environment(self) -> None:
        _resolve_secrets(self.client_spec)
        _resolve_secrets(self.agent_config)

    def build_client(self, purpose: str, *, cache_task_id: str | None = None):
        self.validate_environment()
        for module in self.imports:
            Registry.import_module(module)
        spec = _resolve_secrets(self.client_spec)
        if cache_task_id is not None and spec['name'] == 'litellm':
            spec['config']['client_args']['cache_task_id'] = cache_task_id
        try:
            client = openai_client_registry.get(spec['name'])(OpenAIConfig(**spec['config']))
        except Exception:
            raise ProfileError(f'Cannot construct model client: {self.filename}') from None
        client._client_key = purpose
        return client

    def apply_agent(self, base):
        """Apply model-specific agent settings without replacing task resources."""
        from nexus.extensions.agents.claude_code.agent import ClaudeCodeInput

        body = deep_merge(
            {'agent_config': base.model_dump()}, {'agent_config': _resolve_secrets(self.agent_config)}, self.merge
        )
        return ClaudeCodeInput.model_validate(body['agent_config'])


def load_model(filename: str) -> ModelConfig:
    if not isinstance(filename, str) or not _FILE_NAME.fullmatch(filename):
        raise ProfileError('Model must name a bundled *-aihub-2.0.yaml file, not a path')
    path = MODEL_DIR / filename
    if path.resolve().parent != MODEL_DIR.resolve() or not path.is_file():
        raise ProfileError(f'Unknown model YAML: {filename}')
    try:
        part = resolve_non_data_part(RawRunPart(kind='use', value=UseSpec(name=str(path), params={}), source=str(path)))
        if set(part.body) - {'imports', 'agent_config', 'openai_clients'}:
            raise ProfileError('Model YAML may only contain imports, agent_config and openai_clients')
        if set(part.body.get('openai_clients', {})) != {'main'}:
            raise ProfileError('Single-model YAML must contain exactly openai_clients.main')
        _validate_secrets(part.body)
        parsed = RunInput.from_dict(deep_merge({'task_name': 'artifact_model_config'}, part.body, part.merge))
        config = parsed.openai_clients['main']
        _validate_secrets(config)
        _validate_secrets(parsed.agent_config)
        args = config['config']['client_args']
        if 'api_key__env' not in args:
            raise ProfileError('Model YAML must declare client_args.api_key__env')
        for control in ('cache_task_id_from_run_id', 'nexus_trace_id_from_run_id'):
            if control in args:
                raise ProfileError('Run-derived client controls are not supported in Judge model YAML')
        from urllib.parse import urlsplit

        url = urlsplit(args['base_url'])
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or _SECRET.search(url.query):
            raise ProfileError('Model endpoint must not contain credentials')
        model = (
            config['config'].get('enforced_request_args', {}).get('model') or config['config']['request_args']['model']
        )
        if not isinstance(model, str) or not model.strip() or '${' in model:
            raise ProfileError('Model YAML must resolve to a model name')
        allowed_agent = {
            'thinking_preset',
            'conversion_log_level',
            'no_thinking_request_args',
            'no_thinking_request_args_mode',
            'haiku_model',
            'extra_env',
        }
        if set(parsed.agent_config) - allowed_agent:
            raise ProfileError('Model agent_config must not override task runtime, tools or execution budgets')
        canonical = json.dumps(
            {'client': config, 'agent': parsed.agent_config, 'imports': parsed.imports, 'merge': part.merge},
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
    except ProfileError:
        raise
    except Exception:
        raise ProfileError(f'Cannot resolve model YAML: {filename} (configuration values omitted)') from None
    return ModelConfig(
        filename,
        part.source,
        config,
        parsed.agent_config,
        dict(part.merge),
        tuple(parsed.imports),
        hashlib.sha256(canonical.encode()).hexdigest(),
    )


@dataclass(frozen=True)
class JudgeModels:
    bindings: dict[str, ModelConfig]

    @property
    def models(self) -> dict[str, str]:
        return {purpose: config.model for purpose, config in self.bindings.items()}

    def validate_environment(self) -> None:
        _cache_task_id_override()
        for config in self.bindings.values():
            config.validate_environment()

    def build_clients(self, *, instance_id: str) -> dict[str, Any]:
        """Bind clients to the data instance, stable across runs and request turns.

        An explicit experiment override may replace the instance-derived ID.
        Neither the source model configuration nor the environment is mutated.
        """
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ProfileError('A non-empty data instance_id is required to bind Judge clients')
        if _PLACEHOLDER.search(instance_id) or '${' in instance_id:
            raise ProfileError('Data instance_id must not contain unresolved placeholders')
        self.validate_environment()
        cache_task_id = _cache_task_id_override() or instance_id.strip()
        return {
            purpose: config.build_client(purpose, cache_task_id=cache_task_id)
            for purpose, config in self.bindings.items()
        }

    def provenance(self) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for path in sorted(root.rglob('*.py')):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        return {
            'models': self.models,
            'bindings': {
                purpose: {'file': config.filename, 'source': config.source, 'config_sha256': config.config_sha256}
                for purpose, config in self.bindings.items()
            },
            'judge_code_sha256': digest.hexdigest(),
        }


def select_models(judge_name: str, overrides: dict[str, str] | None = None) -> JudgeModels:
    from nexus.extensions.tasks.artifact.judges.registry import get_judge

    spec = get_judge(judge_name)
    bindings = dict(spec.model_files)
    if set(bindings) != set(spec.requires):
        raise ProfileError('Judge must bind one model file for every required purpose')
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - bindings.keys():
            raise ProfileError('Model overrides must reference declared Judge roles: ' + ', '.join(sorted(bindings)))
        bindings.update(overrides)
    result = JudgeModels({purpose: load_model(filename) for purpose, filename in bindings.items()})
    result.validate_environment()
    return result


def main() -> None:
    import sys

    from nexus.extensions.tasks.artifact.harbor_entry import resolve_judge_models

    try:
        selected = resolve_judge_models()
        print(json.dumps(selected.provenance(), ensure_ascii=False))
    except Exception as exc:
        print(str(exc) if isinstance(exc, ProfileError) else 'Judge model preflight failed', file=sys.stderr)
        sys.exit(1)
