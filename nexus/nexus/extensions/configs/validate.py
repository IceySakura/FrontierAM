"""Check ``configs/{agents,llms,runtime}/*.yaml`` against the schema in README.md.

    python -m nexus.extensions.configs.validate                    # every file
    python -m nexus.extensions.configs.validate agents/kilo.yaml   # selected files
    python -m nexus.extensions.configs.validate --grid             # also merge every agent x llm x runtime
    python -m nexus.extensions.configs.validate --strict           # a file checked only in part counts as a failure

Exit status is 1 when any error is found. Without ``--strict`` a file whose plugin needs a missing third-party
package (``litellm``, ``langfuse``, ...) is only checked in part and reported as SKIP, and the run can still
end ``files: N/N`` green; use ``--strict`` before launching anything, so green means every key was checked.

The point is to catch keys that nexus would silently drop: agent, task and RunInput models ignore unknown
fields, so a typo in ``agent_config`` or ``task_config`` never raises at run time.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys

from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).parent
AXES = ('agents', 'llms', 'runtime')
ENVELOPE = {'body', 'merge', 'source'}
BODY_KEYS = {
    'agents': {'imports', 'agent_name', 'agent_config', 'agent_config_by_llm'},
    'llms': {'imports', 'openai_clients', 'llm'},
}
# Keys the loader folds and strips before the config ever becomes a RunInput.
LOADER_KEYS = {'llm', 'agent_config_by_llm'}
# Facts an llm file may state about its model. The scaffold decides what to do with them.
# `use_azure_provider` is a fact about this specific wire, not about the model family: within
# family=gpt, the aihub2 'standard' endpoint and the real Azure-backed endpoints disagree on it.
# `wrap_plaintext_reasoning_content` is per-wire for the same reason: it tracks whether the wire
# delivers native reasoning_items (openai_responses does, every other client does not), which is
# a property of the protocol rather than of the family.
LLM_FACT_KEYS = {
    'family',
    'anthropic_model',
    'no_thinking_request_args',
    'use_azure_provider',
    'wrap_plaintext_reasoning_content',
    'context_window',
    'auto_compact_token_limit',
    'reasoning_effort',
}
ENV_DEFAULT_RE = re.compile(r'^\$\{oc\.env:[A-Z0-9_]+,([^}]+)\}$')


def _aihub_provider_errors(client: Any) -> list[str]:
    """An AIHub wire must pin its vendor; unpinned, AIHub 2.0 picks one per request."""
    args = (client.get('config') or {}).get('client_args') if isinstance(client, dict) else None
    if not isinstance(args, dict):
        return []
    url = ' '.join(str(args.get(k) or '') for k in ('base_url', 'endpoint'))
    if 'api.aihub.woa.com' not in url:
        return []
    routing = args.get('gateway_routing')
    if 'provider=' in str(args.get('api_key') or '') or (isinstance(routing, dict) and routing.get('provider')):
        return []
    return [
        f'{url.strip()} 没钉厂商：api_key 加 ?provider=<厂商> 或写 client_args.gateway_routing.provider，'
        '否则 AIHub 2.0 每个请求自己挑厂商'
    ]


def _literal(value: Any) -> Any:
    """``${oc.env:X,default}`` -> ``default``; anything else unchanged."""
    match = ENV_DEFAULT_RE.match(value) if isinstance(value, str) else None
    return match.group(1) if match else value


def _provider_params(name: str) -> set[str]:
    """Named constructor parameters of a runtime provider. Providers take ``**kwargs``, so typos are swallowed."""
    from nexus.runtime.providers.registry import runtime_provider_registry

    importlib.import_module(f'nexus.runtime.providers.{name}')
    params = inspect.signature(runtime_provider_registry.get(name).__init__).parameters.values()
    return {p.name for p in params if p.name != 'self' and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def _import_all(modules: list[str]) -> tuple[str | None, str | None]:
    """Import every plugin module. Returns (error, skip reason).

    A missing third-party package (``langfuse``, ``litellm``, ...) means this interpreter lacks an
    optional dependency, not that the yaml is wrong, so it is reported as a skip instead of a failure.
    """
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split('.')[0] != 'nexus':
                return None, f'缺第三方依赖 {exc.name}，跳过 {module} 的键检查'
            return f'import {module} 失败: {type(exc).__name__}: {exc}', None
        except Exception as exc:  # noqa: BLE001 - report any import failure as a config error
            return f'import {module} 失败: {type(exc).__name__}: {exc}', None
    return None, None


def _input_fields(registry: Any, name: str) -> set[str]:
    from nexus.runner.single import _get_input_type

    return set(_get_input_type(registry.get(name)).model_fields)


def check_file(path: Path, skips: list[str] | None = None) -> list[str]:
    """Return error messages for one yaml file; append any skipped key checks to ``skips``."""
    from nexus.extensions.configs.load_config import LLM_FAMILIES
    from nexus.runner.single import RunInput, agent_registry, openai_client_registry, task_registry

    axis = path.parent.name
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        return [f'yaml 解析失败: {exc}']
    errors = []

    if set(data) != ENVELOPE:
        errors.append(f'顶层键应为 body/merge/source，实际是 {sorted(data)}')
    body = data.get('body') or {}
    if not isinstance(data.get('source'), str):
        errors.append('source 必须是字符串')
    if body.get('imports') and (data.get('merge') or {}).get('imports') != 'append_unique':
        errors.append('有 body.imports 时 merge.imports 必须是 append_unique')

    unknown_top = set(body) - set(RunInput.model_fields) - LOADER_KEYS
    if unknown_top:
        errors.append(f'body 里有 RunInput 不认识的键（会被丢弃）: {sorted(unknown_top)}')
    if axis in BODY_KEYS and set(body) - BODY_KEYS[axis]:
        errors.append(f'{axis} 只能写 {sorted(BODY_KEYS[axis])}，多了 {sorted(set(body) - BODY_KEYS[axis])}')

    failed, skipped = _import_all(body.get('imports') or [])
    if failed:
        return [*errors, failed]
    registry_ok = not skipped
    if skipped and skips is not None:
        skips.append(f'{path.name}: {skipped}')

    if axis == 'agents':
        name = body.get('agent_name')
        sections = body.get('agent_config_by_llm') or {}
        if unknown_family := sorted(set(sections) - set(LLM_FAMILIES)):
            errors.append(f'agent_config_by_llm 里有未知的模型族 {unknown_family}，已知: {list(LLM_FAMILIES)}')
        if not name:
            errors.append('缺 agent_name')
        elif registry_ok:
            try:
                fields = _input_fields(agent_registry, name)
            except Exception as exc:  # noqa: BLE001
                errors.append(f'agent {name!r} 没有注册（imports 写对了吗）: {exc}')
            else:
                unknown = set(body.get('agent_config') or {}) - fields
                if unknown:
                    errors.append(f'agent_config 里有 {name} 不认识的键（会被静默丢弃）: {sorted(unknown)}')
                # A section writes into agent_config, so the same rule applies to every family.
                for family, keys in sections.items():
                    if unknown := set(keys or {}) - fields:
                        errors.append(
                            f'agent_config_by_llm.{family} 里有 {name} 不认识的键（会被静默丢弃）: {sorted(unknown)}'
                        )

    elif axis == 'llms':
        facts = body.get('llm')
        if not isinstance(facts, dict):
            errors.append('缺 llm（模型事实），llm 文件不再直接写 agent_config')
        else:
            if facts.get('family') not in LLM_FAMILIES:
                errors.append(f'llm.family 必须是 {list(LLM_FAMILIES)} 之一，实际是 {facts.get("family")!r}')
            if unknown := sorted(set(facts) - LLM_FACT_KEYS):
                errors.append(f'llm 只能写这几个事实 {sorted(LLM_FACT_KEYS)}，多了 {unknown}')
        clients = body.get('openai_clients') or {}
        if 'main' not in clients:
            errors.append('缺 openai_clients.main')
        for key, client in clients.items():
            errors.extend(f'openai_clients.{key}: {e}' for e in _aihub_provider_errors(client))
        for key, client in clients.items() if registry_ok else ():
            try:
                openai_client_registry.get(client.get('name'))
            except Exception as exc:  # noqa: BLE001
                errors.append(f'openai_clients.{key}.name={client.get("name")!r} 没有注册: {exc}')

    elif axis == 'runtime':
        for key in ('task_name', 'runtime_provider_name'):
            if key not in body:
                errors.append(f'缺 {key}')
        name = body.get('task_name')
        if name and registry_ok:
            try:
                fields = _input_fields(task_registry, name)
            except Exception as exc:  # noqa: BLE001
                errors.append(f'task {name!r} 没有注册: {exc}')
            else:
                unknown = set(body.get('task_config') or {}) - fields
                if unknown:
                    errors.append(f'task_config 里有 {name} 不认识的键（会被静默丢弃）: {sorted(unknown)}')
        provider = _literal(body.get('runtime_provider_name'))
        if isinstance(provider, str) and provider and registry_ok:
            try:
                params = _provider_params(provider)
            except Exception as exc:  # noqa: BLE001
                errors.append(f'runtime provider {provider!r} 找不到: {exc}')
            else:
                keys = {k.removesuffix('__env') for k in body.get('runtime_provider_config') or {}}
                if unknown := keys - params:
                    errors.append(
                        f'runtime_provider_config 里有 {provider} 不认识的键（会被静默丢弃）: {sorted(unknown)}'
                    )
    return errors


def check_grid() -> list[str]:
    """Merge every agent x llm x runtime through load_config, the same path a generator takes."""
    from nexus.extensions.configs.load_config import load_config

    names = {axis: sorted(p.stem for p in (HERE / axis).glob('*.yaml')) for axis in AXES}
    saved = {k: os.environ.get(k) for k in ('AGENT_CONFIG', 'LLM_CONFIG', 'RUNTIME_CONFIG')}
    errors = []
    try:
        for runtime in names['runtime']:
            for agent in names['agents']:
                for llm in names['llms']:
                    os.environ.update(RUNTIME_CONFIG=runtime, AGENT_CONFIG=agent, LLM_CONFIG=llm)
                    try:
                        merged = load_config()
                        missing = [k for k in ('task_name', 'agent_name', 'openai_clients') if k not in merged]
                        if missing:
                            errors.append(f'{runtime} + {agent} + {llm}: 合并后缺 {missing}')
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f'{runtime} + {agent} + {llm}: {type(exc).__name__}: {exc}')
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    total = len(names['runtime']) * len(names['agents']) * len(names['llms'])
    print(f'grid: {total - len(errors)}/{total} 个组合合并成功')
    return errors


def main(argv: list[str]) -> int:
    targets = [a for a in argv if not a.startswith('--')]
    if targets:
        paths = [HERE / t for t in targets]
    else:
        paths = [p for axis in AXES for p in sorted((HERE / axis).glob('*.yaml'))]
    failed = 0
    skips: list[str] = []
    for path in paths:
        errors = check_file(path, skips)
        rel = path.relative_to(HERE) if path.is_relative_to(HERE) else path
        if errors:
            failed += 1
            print(f'FAIL {rel}')
            for err in errors:
                print(f'     {err}')
    # Decide everything that counts as a failure before printing the tally, so the tally cannot
    # say "47/47 通过" on the line above a FAIL. --strict counts each half-checked file, to match
    # the per-file granularity used above.
    strict_skips = skips if '--strict' in argv else []
    failed += len(strict_skips)
    print(f'files: {len(paths) - failed}/{len(paths)} 通过')
    for skip in skips:
        print(f'SKIP {skip}')
    if skips:
        print(f'{len(skips)} 份文件只查了外壳：当前解释器缺可选依赖，装上后再跑一遍才算全查过')
        if strict_skips:
            print('FAIL --strict: 以上文件没有查完，装好依赖再跑')
    if '--grid' in argv:
        grid_errors = check_grid()
        for err in grid_errors:
            print(f'FAIL {err}')
        failed += bool(grid_errors)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
