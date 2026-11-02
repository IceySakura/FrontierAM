#!/usr/bin/env python3
"""Frontend task batch generator for `nexus run-batch`.

This generator reads a local frontend dataset in JSONL format and yields `BatchItem`s,
one per instance, suitable for running `frontend_task` at scale.

Usage:
  uv run nexus run-batch -g ./workspace/frontend/generator.py -d ./outputs/frontend_batch.db --max-concurrent 5

Environment variables:
  - FRONTEND_DATASET_PATH: path to JSONL dataset
    (default: /nexus/tests/e2e/data/frontend/frontend-fix-v6.jsonl)
  - FRONTEND_INSTANCE_IDS: comma-separated list of specific instance_ids to run (default: all)
  - FRONTEND_SKIP: number of instances to skip before starting (default: 0)
  - FRONTEND_LIMIT: max instances to yield (default: no limit)
  - FRONTEND_RUN_ORACLE: "1"/"true" to run without agent (default: false)
  - FRONTEND_SKIP_EVAL: "1"/"true" to skip patch eval (default: false)
  - FRONTEND_AGENT_NAME: agent name (default: claude_code; ignored if run_oracle=true)
  - FRONTEND_RUNTIME_PROVIDER_NAME: suanli/gongfeng/... (default: gongfeng)
  - FRONTEND_RUNTIME_TOKEN: runtime provider token
  - FRONTEND_RUNTIME_API_BASE_URL: runtime base URL
  - FRONTEND_RUNTIME_CONTAINER_IMAGE: sandbox container image
  - FRONTEND_RUNTIME_IMAGE: nexus runtime image
  - FRONTEND_WORKING_DIR: agent working directory (default: /workspace)
  - CLAUDE_PLUGINS: optional comma-separated plugin names; can be overridden by dataset extra_env
  - LLM_PROVIDER: "hy3" (default) or "opus"
  - ENABLE_THINKING: "1"/"true" to enable thinking mode (default: false)
  - FRONTEND_DISABLE_HOOKS: "1"/"true" to disable PreToolUse hooks for ExitPlanMode/AskUserQuestion (default: false)
  - SKILL_NAME: skill name to prepend as "/<skill>" trigger before prompt (default: empty, no trigger);
                can be overridden by dataset extra_env
  - FRONTEND_SKILLS_DIR: comma-separated local skill directories to sync into sandbox ~/.claude/skills/
                         Claude Code automatically registers each directory as a slash command by name:
                           directory frontend_codesign  → command /frontend_codesign
                           directory frontend-codesign  → command /frontend-codesign
                           directory frontendCodesign   → command /frontendCodesign
                         Directory name IS the command name, no extra mapping or config needed.
                         Each directory must exist (error if not found).
  - FRONTEND_ENABLE_SEARCH: "1"/"true" to enable WebSearch MCP and disable built-in WebSearch tool (default: false)
  - FRONTEND_SEARCH_MCP_URL: SSE URL for WebSearch MCP server (default: https://mcp-api.tencent-cloud.com/sse/7bb631b3c03b41f2)
  - OPENAPI_API_KEY / OPENAPI_BASE_URL / OPENAPI_MODEL_NAME: model client config
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import uuid

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.extensions.tasks.swe.constants import DATASET_SWE_BENCH, SWEbenchInstance
from nexus.runner.single import RunInput


# ---------------------------------------------------------------------------
# Claude Code plugin offline installation (runs inside runtime container).
# Reads plugin files from local ~/.claude/plugins/marketplaces/ and generates
# a self-contained Python script to write them into the container.
# No network needed. Requires local `claude plugin install` to have been run once.
# ---------------------------------------------------------------------------


def _parse_comma_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, (list, tuple, set)) else str(value).split(',')
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _instance_extra_env(inst: SWEbenchInstance) -> dict[str, Any]:
    extra_env = inst.get('extra_env') or {}
    return extra_env if isinstance(extra_env, dict) else {}


def _extra_env_value(extra_env: dict[str, Any], name: str, default: Any = '') -> Any:
    return extra_env.get(name, default)


# Comma-separated list via env var, e.g. CLAUDE_PLUGINS="frontend-design,skill-creator"
# Default: no plugins installed. Per-instance extra_env["CLAUDE_PLUGINS"] can override it.
CLAUDE_PLUGINS_TO_INSTALL: list[str] = _parse_comma_list(os.getenv('CLAUDE_PLUGINS', ''))

_MARKETPLACE_NAME = 'claude-plugins-official'
_MARKETPLACE_ROOT = Path(os.path.expanduser('~/.claude/plugins/marketplaces')) / _MARKETPLACE_NAME


def _load_local_plugin(plugin_name: str) -> dict[str, Any]:
    import subprocess as _sp

    plugin_dir = _MARKETPLACE_ROOT / 'plugins' / plugin_name
    if not plugin_dir.is_dir():
        raise FileNotFoundError(
            f'Plugin "{plugin_name}" not found at {plugin_dir}. '
            f'Run: claude plugin marketplace add anthropics/{_MARKETPLACE_NAME} && '
            f'claude plugin install {plugin_name}'
        )

    files: dict[str, str] = {}
    for fp in sorted(plugin_dir.rglob('*')):
        if fp.is_file():
            with contextlib.suppress(UnicodeDecodeError):
                files[str(fp.relative_to(plugin_dir))] = fp.read_text('utf-8')

    if '.claude-plugin/plugin.json' not in files:
        raise FileNotFoundError(f'Plugin "{plugin_name}" missing .claude-plugin/plugin.json')

    git_dir = _MARKETPLACE_ROOT / '.git'
    if git_dir.is_dir():
        r = _sp.run(['git', 'rev-parse', 'HEAD'], cwd=_MARKETPLACE_ROOT, capture_output=True, text=True)
        git_sha = r.stdout.strip() if r.returncode == 0 else 'unknown'
    else:
        git_sha = 'unknown'

    return {'files': files, 'version': git_sha[:12], 'gitCommitSha': git_sha}


def _load_marketplace_json() -> str:
    """Read marketplace.json from the local marketplace clone."""
    mp_json = _MARKETPLACE_ROOT / '.claude-plugin' / 'marketplace.json'
    if not mp_json.exists():
        raise FileNotFoundError(
            f'marketplace.json not found at {mp_json}. '
            f'Run: claude plugin marketplace add anthropics/{_MARKETPLACE_NAME}'
        )
    return mp_json.read_text('utf-8')


def _build_plugin_install_script(plugins: list[str]) -> str:
    import json as _json
    import subprocess as _sp

    if (_MARKETPLACE_ROOT / '.git').is_dir():
        with contextlib.suppress(Exception):
            _sp.run(['git', 'pull', '--ff-only'], cwd=_MARKETPLACE_ROOT, capture_output=True, timeout=30)

    plugin_data = {}
    for name in plugins:
        plugin_data[f'{name}@{_MARKETPLACE_NAME}'] = _load_local_plugin(name)

    marketplace_json_content = _load_marketplace_json()

    lines = [
        'import os, json',
        'home = os.environ.get("HOME", "/root")',
        'plugins_dir = os.path.join(home, ".claude", "plugins")',
        'os.makedirs(plugins_dir, exist_ok=True)',
        '',
        'installed = {"version": 2, "plugins": {}}',
        'marketplace_meta = {',
        f'    {_MARKETPLACE_NAME!r}: {{',
        f'        "source": {{"source": "github", "repo": "anthropics/{_MARKETPLACE_NAME}"}},',
        f'        "installLocation": os.path.join(plugins_dir, "marketplaces", {_MARKETPLACE_NAME!r}),',
        '        "lastUpdated": "2026-04-23T16:06:43.701Z",',
        '    }',
        '}',
    ]

    for key, data in plugin_data.items():
        version = data['version']
        git_sha = data['gitCommitSha']
        marketplace = key.split('@')[-1]
        pname = key.split('@')[0]

        lines.append('')
        lines.append(f'cache_dir = os.path.join(plugins_dir, "cache", {marketplace!r}, {pname!r}, {version!r})')

        for rel_path, content in data['files'].items():
            lines.append(f'fp = os.path.join(cache_dir, {rel_path!r})')
            lines.append('os.makedirs(os.path.dirname(fp), exist_ok=True)')
            lines.append(f'open(fp, "w").write({_json.dumps(content)})')

        lines.append(f'installed["plugins"][{key!r}] = [{{')
        lines.append('    "scope": "user",')
        lines.append('    "installPath": cache_dir,')
        lines.append(f'    "version": {version!r},')
        lines.append('    "installedAt": "2026-04-23T16:06:43.987Z",')
        lines.append('    "lastUpdated": "2026-04-23T16:06:43.987Z",')
        lines.append(f'    "gitCommitSha": {git_sha!r},')
        lines.append('}]')

    # Build enabledPlugins for settings.json
    enabled = dict.fromkeys(plugin_data, True)

    # Write marketplace.json into marketplaces directory
    lines.append('')
    lines.append(f'mp_dir = os.path.join(plugins_dir, "marketplaces", {_MARKETPLACE_NAME!r}, ".claude-plugin")')
    lines.append('os.makedirs(mp_dir, exist_ok=True)')
    lines.append(f'open(os.path.join(mp_dir, "marketplace.json"), "w").write({_json.dumps(marketplace_json_content)})')
    lines.append('')
    lines.append(
        'open(os.path.join(plugins_dir, "installed_plugins.json"), "w").write(json.dumps(installed, indent=2))'
    )
    lines.append(
        'open(os.path.join(plugins_dir, "known_marketplaces.json"), "w").write(json.dumps(marketplace_meta, indent=2))'
    )
    lines.append('')
    lines.append('settings_path = os.path.join(home, ".claude", "settings.json")')
    lines.append('settings = {}')
    lines.append('if os.path.exists(settings_path):')
    lines.append('    try: settings = json.loads(open(settings_path).read())')
    lines.append('    except: pass')
    lines.append(f'settings["enabledPlugins"] = {{**settings.get("enabledPlugins", {{}}), **{enabled!r}}}')
    lines.append(
        f'settings.setdefault("extraKnownMarketplaces", {{}})[{_MARKETPLACE_NAME!r}] = {{"source": {{"source": "github", "repo": "anthropics/{_MARKETPLACE_NAME}"}}}}'
    )
    lines.append('open(settings_path, "w").write(json.dumps(settings, indent=2))')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Skills injection: sync local skill directories into sandbox ~/.claude/skills/
# FRONTEND_SKILLS_DIR: comma-separated local paths, each must be a skill directory
# containing SKILL.md. Directory name becomes the skill command (e.g. frontend_codesign → /frontend_codesign).
# ---------------------------------------------------------------------------


def _parse_skills_dirs() -> list[Path]:
    """Parse FRONTEND_SKILLS_DIR env var into validated Path list."""
    raw = os.getenv('FRONTEND_SKILLS_DIR', '')
    if not raw:
        return []
    dirs = []
    for p in raw.split(','):
        p = p.strip()
        if not p:
            continue
        path = Path(p).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'FRONTEND_SKILLS_DIR: directory not found: {path}')
        dirs.append(path)
    return dirs


def _build_skills_install_script(skills_dirs: list[Path]) -> str:
    """Generate a Python script that writes skill files into ~/.claude/skills/ in the sandbox."""
    import base64 as _b64

    lines = [
        'import os, base64',
        'home = os.environ.get("HOME", "/root")',
        'skills_root = os.path.join(home, ".claude", "skills")',
        'os.makedirs(skills_root, exist_ok=True)',
    ]

    for skill_dir in skills_dirs:
        skill_name = skill_dir.name
        # Collect all files in the skill directory
        for fp in sorted(skill_dir.rglob('*')):
            if fp.is_file():
                try:
                    content_bytes = fp.read_bytes()
                except Exception:
                    continue
                rel_path = str(fp.relative_to(skill_dir))
                encoded = _b64.b64encode(content_bytes).decode()
                target = f'os.path.join(skills_root, {skill_name!r}, {rel_path!r})'
                lines.append(f'os.makedirs(os.path.dirname({target}), exist_ok=True)')
                lines.append(f'open({target}, "wb").write(base64.b64decode({encoded!r}))')

        lines.append(f'print("Skill installed: {skill_name}")')

    return '\n'.join(lines)


SKILLS_DIRS_TO_INSTALL: list[Path] = _parse_skills_dirs()


# Override resource_spec for specific instances
# Format: {'instance_id': {'cpu': N, 'memory': M}}
INSTANCE_RESOURCE_OVERRIDE = {
    # scikit-learn__scikit-learn-14710 需要进行大量的数学计算, 十分耗cpu
    'scikit-learn__scikit-learn-14710': {'cpu': 16, 'memory': 16},
}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str) -> int | None:
    v = os.getenv(name)
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _env_list(name: str) -> set[str] | None:
    """Parse comma-separated list from env var, return as set for O(1) lookup."""
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return set(items) if items else None


def _iter_instances(dataset_path: str) -> Iterator[SWEbenchInstance]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if path.suffix != '.jsonl':
        raise ValueError(f'Dataset must be .jsonl for batch mode: {dataset_path}')

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


DEFAULT_DATASET_PATH = 'root/swe-bench-no-hints.jsonl'


def _build_openai_clients_hy3(
    enable_thinking: bool = False, session_id: str | None = None
) -> dict[str, dict[str, Any]]:
    temperature = _env_float('OPENAPI_TEMP', 0.9)
    model_name = os.getenv('OPENAPI_MODEL_NAME', 'openai/HY3-dev0413')
    api_key = os.getenv('OPENAPI_API_KEY', 'EMPTY')
    base_url = os.getenv('OPENAPI_BASE_URL', 'http://infer-proxy-yb-test.production.polaris:8000/openapi')

    # print(f'[frontend_wthink] base_url={base_url} api_key={api_key} temperature={temperature} model_name={model_name}', flush=True)

    request_args: dict[str, Any] = {
        'model': model_name,
        'timeout': 1800.0,
        'max_retries': 0,
        'temperature': temperature,
        'top_p': 1.0,
        'top_k': -1,
        'max_tokens': 32768,
        'repetition_penalty': 1.0,
        'extra_body': {
            'chat_template_kwargs': {
                'reasoning_effort': {'effort': os.getenv('REASONING_EFFORT', 'high')},
                'interleaved_thinking': True,
            },
        },
    }

    if session_id:
        request_args['extra_headers'] = {
            'SessionID': session_id,
            'StrategyType': 'ConsistentHash',
        }

    if not enable_thinking:
        request_args['chat_template_kwargs'] = {'reasoning_effort': 'no_think'}
    return {
        'main': {
            'name': 'litellm',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                },
                'request_args': request_args,
            },
        }
    }


def _build_openai_clients_openai_opus(
    enable_thinking: bool = False, session_id: str | None = None
) -> dict[str, dict[str, Any]]:
    api_key = os.getenv('OPENAPI_API_KEY', 'EMPTY')
    base_url = os.getenv('OPENAPI_BASE_URL', 'http://trpc-gpt-eval.production.polaris:8080')

    model_name = os.getenv('OPENAPI_MODEL_NAME', 'bedrock/anthropic.claude-opus-4-6-v1')
    # print(f'[frontend_wthink] base_url={base_url} api_key={api_key} model_name={model_name}', flush=True)
    request_args: dict[str, Any] = {
        'model': model_name,
        'output_config': {'effort': os.getenv('REASONING_EFFORT', 'max')},
    }
    if enable_thinking:
        request_args['thinking'] = {'type': 'adaptive'}
    return {
        'main': {
            'name': 'litellm',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                    'cache_control_enabled': True,
                    'cache_control_strategy': 'claude_code',
                    'log_llm_usage': True,
                },
                'request_args': request_args,
            },
        }
    }


def _build_openai_clients_openai(
    enable_thinking: bool = False, session_id: str | None = None
) -> dict[str, dict[str, Any]]:
    """
    支持openai系列模型传入参数
    """
    api_key = os.getenv('OPENAPI_API_KEY', 'EMPTY')
    base_url = os.getenv('OPENAI_BASE_URL', 'http://llm-api.model-eval.woa.com/v1')
    model_name = os.getenv('OPENAPI_MODEL_NAME', 'openai/xxxx')
    # print(f'[frontend_wthink] base_url={base_url} api_key={api_key} model_name={model_name}', flush=True)

    extra_body: dict[str, Any] = {}
    if enable_thinking:
        extra_body['enable_thinking'] = True

    return {
        'main': {
            'name': 'litellm',
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': base_url,
                    'cache_control_enabled': True,
                },
                'request_args': {
                    'model': model_name,
                    'extra_body': extra_body,
                },
            },
        },
    }


_EXITPLAN_HOOK_CMD = (
    'echo \'{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{}}}\''
)

# AskUserQuestion hook: read script froo-located file, deploy to container at runtime.
_ASKUSER_HOOK_SCRIPT = (Path(__file__).parent / 'askuser_hook.py').read_text(encoding='utf-8')
_ASKUSER_HOOK_SCRIPT_PATH = '/tmp/_nexus_askuser_hook.py'
_ASKUSER_HOOK_CMD = f'python3 {_ASKUSER_HOOK_SCRIPT_PATH}'


def _build_askuser_hook_script_writer() -> str:
    """Generate a Python script that writes the AskUserQuestion hook to the container."""
    import base64 as _b64

    encoded = _b64.b64encode(_ASKUSER_HOOK_SCRIPT.encode()).decode()
    return (
        'import base64, pathlib\n'
        f'pathlib.Path({_ASKUSER_HOOK_SCRIPT_PATH!r}).write_text('
        f'base64.b64decode({encoded!r}).decode(), encoding="utf-8")\n'
    )


def _build_hook_patches() -> dict[str, Any]:
    """Build config_file_patches that inject PreToolUse hooks for non-interactive mode.

    In --print mode, ExitPlanMode and AskUserQuestion have requiresUserInteraction()=true
    which short-circuits before --allowedTools is checked.

    - ExitPlanMode: hook allows it (permissionDecision=allow).
    - AskUserQuestion: hook denies it but provides the auto-selected answer via
      permissionDecisionReason. This works because updatedInput.answers doesn't
      actually populate the tool result in --print mode.
    """
    patches: list[dict[str, Any]] = [
        {
            'path': '~/.claude/settings.json',
            'content': {
                'hooks': {
                    'PreToolUse': [
                        {
                            'matcher': 'ExitPlanMode',
                            'hooks': [{'type': 'command', 'command': _EXITPLAN_HOOK_CMD}],
                        },
                        {
                            'matcher': 'AskUserQuestion',
                            'hooks': [{'type': 'command', 'command': _ASKUSER_HOOK_CMD}],
                        },
                    ],
                },
            },
        },
    ]

    # Inject WebSearch MCP server config if enabled
    if _env_bool('FRONTEND_ENABLE_SEARCH'):
        mcp_url = os.getenv(
            'FRONTEND_SEARCH_MCP_URL',
            'https://mcp-api.tencent-cloud.com/sse/7bb631b3c03b41f2',
        )
        patches.append({
            'path': '~/.claude/mcp.json',
            'content': {
                'mcpServers': {
                    'WebSearchMCP': {
                        'type': 'sse',
                        'url': mcp_url,
                    },
                },
            },
        })

    return {'config_file_patches': patches}


def generate_configs() -> Iterator[BatchItem]:
    dataset_path = os.getenv('FRONTEND_DATASET_PATH', DEFAULT_DATASET_PATH)
    skip = _env_int('FRONTEND_SKIP') or 0
    limit = _env_int('FRONTEND_LIMIT')
    instance_ids_filter = _env_list('FRONTEND_INSTANCE_IDS')  # NEW: specific instance_ids to run
    run_oracle = _env_bool('FRONTEND_RUN_ORACLE', default=False)
    skip_eval = _env_bool('FRONTEND_SKIP_EVAL', default=False)
    runtime_provider_name = os.getenv('FRONTEND_RUNTIME_PROVIDER_NAME', 'gongfeng')

    agent_name = os.getenv('FRONTEND_AGENT_NAME', 'claude_code')

    # Build plugin install scripts lazily per plugin set because datasets can override
    # CLAUDE_PLUGINS per instance via extra_env.
    plugin_install_scripts: dict[tuple[str, ...], str | None] = {}

    skipped = 0
    produced = 0
    items: list[BatchItem] = []
    for inst in _iter_instances(dataset_path):
        instance_id = str(inst.get('instance_id', '') or '')
        if not instance_id:
            prompt = inst.get('prompt', '')
            if not prompt:
                continue
            instance_id = hashlib.md5(prompt.encode('utf-8')).hexdigest()

        # Filter by specific instance_ids if provided
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue

        # Skip instances until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        inst_dataset_type = str(inst.get('dataset_type') or DATASET_SWE_BENCH)

        extra_env = _instance_extra_env(inst)

        # Build prompt: problem_statement > prompt > text
        original_prompt = str(inst.get('problem_statement', '') or inst.get('prompt', '') or inst.get('text', ''))
        # Prepend skill trigger if SKILL_NAME is set (e.g. "frontend-design").
        # Per-instance extra_env["SKILL_NAME"] overrides the process env default.
        skill_name = str(_extra_env_value(extra_env, 'SKILL_NAME', os.getenv('SKILL_NAME', '')) or '').strip()
        final_prompt = f'/{skill_name} ' + original_prompt if skill_name else original_prompt

        plugins_to_install = _parse_comma_list(_extra_env_value(extra_env, 'CLAUDE_PLUGINS', CLAUDE_PLUGINS_TO_INSTALL))
        plugin_key = tuple(plugins_to_install)
        if plugin_key not in plugin_install_scripts:
            plugin_install_scripts[plugin_key] = (
                _build_plugin_install_script(plugins_to_install) if plugins_to_install else None
            )
        _plugin_install_script = plugin_install_scripts[plugin_key]

        task_imports = [
            'nexus.extensions.tasks.frontend',
            'nexus.extensions.agents.claude_code',
        ]
        task_name = 'frontend_task'
        task_config: dict[str, Any] = {
            'prompt': final_prompt,
        }
        if not run_oracle:
            task_config['dataset_path'] = dataset_path
            task_config['instance_id'] = instance_id
            task_config['run_oracle'] = False
            task_config['skip_infer_patch_eval'] = bool(skip_eval)

        config: dict[str, Any] = {
            'imports': task_imports,
            'task_name': task_name,
            'task_config': task_config,
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': {
                'token': os.getenv('FRONTEND_RUNTIME_TOKEN', 'EMPTY'),
                'api_base_url': os.getenv('FRONTEND_RUNTIME_API_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052'),
                'image': os.getenv(
                    'FRONTEND_RUNTIME_CONTAINER_IMAGE',
                    'mirrors.tencent.com/hunyuan_yanguan/webenv.claudecode.testset:v3',
                ),
                'runtime_image': os.getenv('FRONTEND_RUNTIME_IMAGE', 'current'),
                'request_timeout': 10000.0,
                'runtime_provider': runtime_provider_name,
                'resource_spec': {
                    'cpu': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('cpu', 16),
                    'memory': INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {}).get('memory', 32),
                },
            },
            'runtime_configs': {
                'main': {
                    'worker_venv': True,
                },
            },
            'processors': {
                'chat': {
                    'name': 'chat_tracker',
                    'args': {
                        'tools_mode': 'branch_last',
                    },
                },
            },
        }
        working_dir = os.getenv('FRONTEND_WORKING_DIR', '/workspace')
        if not run_oracle:
            config['agent_name'] = agent_name
            config['agent_config'] = {
                'pre_start_scripts': [
                    {
                        'code': f'mkdir -p {working_dir}',
                    },
                    *(
                        [
                            {
                                'interpreter': 'python',
                                'code': _plugin_install_script,
                            }
                        ]
                        if _plugin_install_script
                        else []
                    ),
                    *(
                        [
                            {
                                'interpreter': 'python',
                                'code': _build_skills_install_script(SKILLS_DIRS_TO_INSTALL),
                            }
                        ]
                        if SKILLS_DIRS_TO_INSTALL
                        else []
                    ),
                    *(
                        [
                            {
                                'interpreter': 'python',
                                'code': _build_askuser_hook_script_writer(),
                            }
                        ]
                        if not _env_bool('FRONTEND_DISABLE_HOOKS')
                        else []
                    ),
                ],
                'working_dir': working_dir,
                'version': '2.1.117',
                'extra_cli_args': [
                    '--print',
                    '--output-format',
                    'stream-json',
                    '--verbose',
                    '--disallowedTools',
                    'WebSearch',
                ],
                **(_build_hook_patches() if not _env_bool('FRONTEND_DISABLE_HOOKS') else {}),
                'extra_env': {
                    'ANTHROPIC_MODEL': 'claude-opus-4-7',
                    'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
                    'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
                    'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
                    'CLAUDE_CODE_SKIP_BEDROCK_AUTH': '1',
                    'ENABLE_TOOL_SEARCH': 'false',
                },
                'timeout': 10000,
                'thinking_preset': 'bedrock-claude',
                # converter 详细转换日志，便于排查格式问题
                'conversion_log_level': 'DEBUG',
                # 当 FRONTEND_LLM_ROUTER_ENABLE=1 时，启用 Claude Code 内部模型路由：
                # - no_thinking_request_args: 无 thinking 请求（title 生成、commit message 等）使用的模型
                # - haiku_model: haiku 轻量级任务路由到的模型
                **(
                    {
                        'no_thinking_request_args': {
                            'model': os.getenv(
                                'FRONTEND_NO_THINKING_MODEL', 'bedrock/api_aws_third_anthropic.claude-opus-4-7'
                            ),
                        },
                        'haiku_model': os.getenv(
                            'FRONTEND_HAIKU_MODEL', 'bedrock/anthropic.claude-haiku-4-5-20251001-v1:0'
                        ),
                    }
                    if _env_bool('FRONTEND_LLM_ROUTER_ENABLE')
                    else {}
                ),
            }
            # LLM_PROVIDER: "hy3" (default) or "opus"
            llm_provider = os.getenv('LLM_PROVIDER', 'hy3')
            enable_thinking = _env_bool('ENABLE_THINKING', default=False)
            if llm_provider == 'opus':
                config['openai_clients'] = _build_openai_clients_openai_opus(
                    enable_thinking=enable_thinking,
                    session_id=str(uuid.uuid4()),
                )
            elif llm_provider == 'openai':
                config['openai_clients'] = _build_openai_clients_openai(
                    enable_thinking=enable_thinking,
                    session_id=str(uuid.uuid4()),
                )
            else:
                config['openai_clients'] = _build_openai_clients_hy3(
                    enable_thinking=enable_thinking,
                    session_id=str(uuid.uuid4()),
                )
        else:
            config['agent_name'] = None
            config['openai_clients'] = {}

        # EVAL_TIMES: generate N BatchItems per instance, each with a fresh session_id
        eval_times = _env_int('EVAL_TIMES') or 1
        for _round in range(eval_times):
            if _round > 0 and not run_oracle:
                if llm_provider == 'opus':
                    config['openai_clients'] = _build_openai_clients_openai_opus(
                        enable_thinking=enable_thinking,
                        session_id=str(uuid.uuid4()),
                    )
                elif llm_provider == 'openai':
                    config['openai_clients'] = _build_openai_clients_openai(
                        enable_thinking=enable_thinking,
                        session_id=str(uuid.uuid4()),
                    )
                else:
                    config['openai_clients'] = _build_openai_clients_hy3(
                        enable_thinking=enable_thinking,
                        session_id=str(uuid.uuid4()),
                    )
            run_id = instance_id if eval_times == 1 else f'{instance_id}_r{_round}'
            items.append(
                BatchItem(
                    config=RunInput.model_validate(config),
                    run_id=run_id,
                    metadata={
                        'instance_id': instance_id,
                        'dataset_type': inst_dataset_type,
                        'prompt': original_prompt,
                    },
                )
            )

        produced += 1
        if limit is not None and produced >= limit:
            break

    # FRONTEND_SHUFFLE: randomize BatchItem order (default true)
    if _env_bool('FRONTEND_SHUFFLE', default=True):
        random.shuffle(items)

    yield from items


def get_total_count() -> int:
    dataset_path = os.getenv('FRONTEND_DATASET_PATH', DEFAULT_DATASET_PATH)
    skip = _env_int('FRONTEND_SKIP') or 0
    limit = _env_int('FRONTEND_LIMIT')
    instance_ids_filter = _env_list('FRONTEND_INSTANCE_IDS')  # NEW: specific instance_ids to run
    n = 0
    skipped = 0
    for inst in _iter_instances(dataset_path):
        instance_id = str(inst.get('instance_id', '') or '')
        if not instance_id:
            prompt = inst.get('prompt', '')
            if not prompt:
                continue
            instance_id = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        # Filter by specific instance_ids if provided
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue
        # Skip instances until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue
        n += 1
        if limit is not None and n >= limit:
            return limit
    return n


def get_batch_metadata() -> dict[str, Any]:
    dataset_path = os.getenv('FRONTEND_DATASET_PATH', DEFAULT_DATASET_PATH)
    instance_ids_filter = _env_list('FRONTEND_INSTANCE_IDS')
    return {
        'task_name': 'frontend_task',
        'dataset_path': dataset_path,
        'dataset_type_default': DATASET_SWE_BENCH,
        'run_oracle': _env_bool('FRONTEND_RUN_ORACLE', default=False),
        'instance_ids_filter': list(instance_ids_filter) if instance_ids_filter else None,
    }
