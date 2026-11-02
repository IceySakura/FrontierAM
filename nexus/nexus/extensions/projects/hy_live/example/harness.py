"""Build a Gongfeng RunInput from a frozen example data JSON."""

from __future__ import annotations

import json

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus.extensions.projects.hy_live.example.cases import Backend, ExampleCase
from nexus.extensions.projects.hy_live.example.prepare_scripted_user_data import build_scripted_user_row
from nexus.extensions.projects.hy_live.web_search import WebSearchProvider
from nexus.runner.single import RunInput


DEFAULT_API_BASE = 'http://api.aihub.woa.com/standard/v1'
DEFAULT_MODEL = 'deepseek-v4.1-flash-expires-on-0910'
DEFAULT_SANDBOX_API = 'http://hyrl-sandbox.prod.woa.com:8052'
# Tag source of truth: docs/impl/runtime_image.md.
DEFAULT_RUNTIME_IMAGE = (
    'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.21-hylive-992fb888-wb8da3c28fa1-py312-slim'
)
# The workspace image is the task environment and the runtime image only donates /nix; one tag for
# both would couple their warm-up lanes.
DEFAULT_WORKSPACE_IMAGE = 'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:claude-code-hle'
# Both need a runtime image whose host relay knows the key; any other image rejects it and the
# CodeBuddy session never starts (docs/impl/runtime_image.md).
DEFAULT_RELAY_UPDATES = True
DEFAULT_CLIENT_QUESTIONS = True
DEFAULT_BACKEND: Backend = 'workbuddy'
DEFAULT_WORKBUDDY_FORK_MODE = 'quick'
"""Mode of a WorkBuddy follow-up branch, so it answers without tools next to the running turn."""
# NOTE(wallyshi): These name the CodeBuddy npm package; a WorkBuddy session installs its own package
# and never reads them, so they are dropped rather than shown as if they configured wb.
_CODEBUDDY_ONLY_KEYS = frozenset({'version', 'npm_registry'})


def load_case_data(case: ExampleCase) -> dict[str, Any]:
    payload = json.loads(case.data_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Example data must be a JSON object: {case.data_path}')
    if payload.get('case') != case.name:
        raise ValueError(f'Example data case {payload.get("case")!r} does not match {case.name!r}')
    return payload


def _merge_mapping(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_mapping(current, value)
        else:
            merged[key] = value
    return merged


def _select_workbuddy(
    task_config: dict[str, Any], *, package_path: str | None, package_sha256: str | None
) -> dict[str, Any]:
    """Run the case on WorkBuddy: ``backend_adapter='workbuddy'`` and one ``work_agent_configs.workbuddy``.

    Case data is written for CodeBuddy unless it is WorkBuddy-only, so the ``codebuddy`` config moves
    under ``workbuddy`` first and a ``workbuddy`` config of the case is laid over it. The package the
    Host uploads into the sandbox comes from the command line, never from case data. No package
    path uses the runtime image's preinstalled CLI.
    """
    configs = dict(task_config.get('work_agent_configs') or {})
    shared = {
        key: value for key, value in (configs.pop('codebuddy', None) or {}).items() if key not in _CODEBUDDY_ONLY_KEYS
    }
    workbuddy_config = _merge_mapping(shared, configs.pop('workbuddy', None) or {})
    package: dict[str, Any] = {'package_path': package_path}
    if package_sha256:
        package['package_sha256'] = package_sha256
    workbuddy_config['workbuddy'] = {
        'fork_mode': DEFAULT_WORKBUDDY_FORK_MODE,
        **(workbuddy_config.get('workbuddy') or {}),
        **package,
    }
    configs['workbuddy'] = workbuddy_config
    return {**task_config, 'backend_adapter': 'workbuddy', 'work_agent_configs': configs}


@dataclass(frozen=True)
class LayerSampling:
    """Request args of one hy-live model layer; an empty ``reasoning_effort`` is not sent."""

    reasoning_effort: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096


CEREBELLUM_SAMPLING = LayerSampling(reasoning_effort='none')
"""The cerebellum speaks with thinking off."""
MIDBRAIN_SAMPLING = LayerSampling(reasoning_effort='low')
"""The midbrain keeps light thinking."""


def _client_spec(
    api_key_env: str,
    base_url: str,
    model: str,
    reasoning_effort: str,
    temperature: float,
    *,
    top_p: float | None = None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    request_args: dict[str, Any] = {'model': model}
    if reasoning_effort:
        request_args['reasoning_effort'] = reasoning_effort
    request_args['temperature'] = temperature
    if top_p is not None:
        request_args['top_p'] = top_p
    request_args['max_completion_tokens'] = max_tokens
    return {
        'name': 'openai',
        'config': {
            'client_args': {
                'api_key__env': api_key_env,
                'base_url': base_url,
                'timeout': 600,
                # Model gateways may rate-limit parallel cases; the SDK's backoff rides out the 429s.
                'max_retries': 8,
            },
            'request_args': request_args,
            'enforced_request_args': {'model': model, 'stream': False},
        },
    }


def _layer_client_spec(api_key_env: str, base_url: str, model: str, sampling: LayerSampling) -> dict[str, Any]:
    return _client_spec(
        api_key_env,
        base_url,
        model,
        sampling.reasoning_effort,
        sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
    )


def build_run_input(
    case: ExampleCase,
    *,
    run_id: str,
    api_base_url: str,
    model: str,
    reasoning_effort: str,
    temperature: float,
    sandbox_api_url: str,
    runtime_image: str,
    workspace_image: str,
    timeout: float,
    worker_venv: bool = False,
    sandbox_purpose: str | None = None,
    relay_updates: bool = DEFAULT_RELAY_UPDATES,
    client_questions: bool = DEFAULT_CLIENT_QUESTIONS,
    cerebellum_sampling: LayerSampling = CEREBELLUM_SAMPLING,
    midbrain_sampling: LayerSampling = MIDBRAIN_SAMPLING,
    web_search_provider: WebSearchProvider | None = None,
    backend: Backend = DEFAULT_BACKEND,
    workbuddy_package: str | None = None,
    workbuddy_package_sha256: str | None = None,
    pass_env: Sequence[str] = (),
) -> RunInput:
    """Overlay frozen example data onto the hy-live scripted row, then attach Gongfeng runtime.

    The cerebellum (``main``) and midbrain (``midbrain``) get their own clients and sampling;
    ``reasoning_effort`` and ``temperature`` apply to the working agent (``external``) and the
    user simulator. ``web_search_provider`` overrides the case's own choice; None keeps it.
    ``client_questions`` is always written explicitly, so the hy-live default never reaches an
    image that predates it. ``backend='workbuddy'`` runs the case on ``wb --acp`` with the Host
    package ``workbuddy_package``, when supplied, or the runtime image's preinstalled CLI. The
    package and its checksum are ignored on CodeBuddy. Each ``pass_env`` name reaches the working agent
    as ``extra_env`` ``<NAME>__env``, which the runner resolves from this host's environment, so the
    value never enters the run data.

    Raises:
        ValueError: The case needs ``client_questions`` but it is off, the case is not written for
            ``backend``, or a supplied WorkBuddy package file does not exist.
    """
    if 'client_questions' in case.runtime_features and not client_questions:
        raise ValueError(
            f'Example {case.name!r} needs a runtime image whose host relay knows client_questions; '
            'pass --client-questions with such an image (docs/impl/runtime_image.md)'
        )
    if backend not in case.backends:
        raise ValueError(
            f'Example {case.name!r} runs only on --backend {"/".join(sorted(case.backends))}, got --backend {backend}'
        )
    package_path: str | None = None
    if backend == 'workbuddy' and workbuddy_package:
        package_file = Path(workbuddy_package).expanduser()
        if not package_file.is_file():
            raise ValueError(f'WorkBuddy package {workbuddy_package!r} is not a file on this host')
        package_path = str(package_file.resolve())
    case_data = load_case_data(case)
    # NOTE(wallyshi): Frozen cases share a CodeBuddy base before the selected backend is applied.
    row = build_scripted_user_row(agent_loop='nexus_agent', backend='codebuddy')
    nexus_config = json.loads(row['nexus'])
    task_config = nexus_config['task_config']
    overlay = case_data.get('task_config')
    if isinstance(overlay, dict):
        task_config = _merge_mapping(task_config, overlay)
    task_config['backend_adapter'] = backend
    if backend == 'workbuddy':
        task_config = _select_workbuddy(task_config, package_path=package_path, package_sha256=workbuddy_package_sha256)
    else:
        configs = task_config.get('work_agent_configs') or {}
        task_config['work_agent_configs'] = {key: value for key, value in configs.items() if key != 'workbuddy'}
    if relay_updates:
        task_config = _merge_mapping(task_config, {'work_agent_configs': {backend: {'relay_updates': True}}})
    task_config = _merge_mapping(task_config, {'work_agent_configs': {backend: {'client_questions': client_questions}}})
    if pass_env:
        env_refs = {f'{name}__env': name for name in pass_env}
        task_config = _merge_mapping(task_config, {'work_agent_configs': {backend: {'extra_env': env_refs}}})
    if web_search_provider is not None:
        task_config['web_search_provider'] = web_search_provider
    prompt = str(case_data['prompt'])
    user_simulator = case_data['user_simulator']
    if not isinstance(user_simulator, dict):
        raise ValueError(f'Example {case.name!r} user_simulator must be an object')
    task_config['user_simulator'] = user_simulator
    task_config['prompt'] = prompt
    task_config.setdefault('midbrain_client_name', 'midbrain')
    nexus_config['task_config'] = task_config
    nexus_config['openai_clients'] = {
        'main': _layer_client_spec('LIVE_AGENT_POLICY_API_KEY', api_base_url, model, cerebellum_sampling),
        'midbrain': _layer_client_spec('LIVE_AGENT_EXTERNAL_API_KEY', api_base_url, model, midbrain_sampling),
        'external': _client_spec(
            'LIVE_AGENT_EXTERNAL_API_KEY',
            api_base_url,
            model,
            reasoning_effort,
            temperature,
        ),
        # Frozen user simulator; must not share the trainable policy client.
        'user_simulator': _client_spec(
            'LIVE_AGENT_EXTERNAL_API_KEY',
            api_base_url,
            model,
            reasoning_effort,
            temperature,
        ),
    }
    runtime_provider_config: dict[str, Any] = {
        'api_base_url': sandbox_api_url,
        'token__env': 'GONGFENG_RUNTIME_TOKEN',
        'runtime_image': runtime_image,
        'python_version': '3.12',
        'runtime_provider': 'gongfeng',
        'resource_spec': {'cpu': 2, 'memory': 8},
        'request_timeout': 7200,
        'rpc_protocol': 'http',
    }
    # A purpose-scoped sandbox token needs the purpose on every create request.
    if sandbox_purpose:
        runtime_provider_config['purpose'] = sandbox_purpose
    nexus_config.update({
        'run_id': run_id,
        'runtime_provider_name': 'gongfeng',
        'runtime_provider_config': runtime_provider_config,
        'runtime_configs': {
            'main': {
                'image': workspace_image,
                'working_dir': '/workspace',
                'worker_venv': worker_venv,
                # Keep the platform-injected proxy variables; CodeBuddy's web tools need them.
                'environment': {
                    'UV_CACHE_DIR': '/tmp/nexus-uv-cache',
                    'UV_PYTHON_DOWNLOADS': 'never',
                },
            }
        },
        'runner': {'timeout': timeout},
        'processors': {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}},
    })
    return RunInput.model_validate(nexus_config)
