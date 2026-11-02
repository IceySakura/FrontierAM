#!/usr/bin/env python3
"""
Terminal-style task batch generator for `nexus run-batch`.

Reads a JSONL dataset with terminal tasks (data analysis, etc.) and yields BatchItems
that use `terminal_task` with Claude Code agent. Each task's environment/ data is
deployed to /workspace, and evaluation is done via tests/test.sh → reward.txt.

Input JSONL format (each line):
  {"instance_id": "da-L1-xxx", "prompt": "...", "task_dir": "/path/to/task/dir"}

Usage:
  uv run nexus run-batch -g ./nexus/extensions/tasks/frontend/generator_midtrain_terminal.py \
    -d ./outputs/terminal_batch.db --max-concurrent 5

Environment variables:
  - FRONTEND_DATASET_PATH: path to JSONL dataset
  - FRONTEND_INSTANCE_IDS: comma-separated list of specific instance_ids to run (default: all)
  - FRONTEND_SKIP: number of instances to skip (default: 0)
  - FRONTEND_LIMIT: max instances to yield (default: no limit)
  - FRONTEND_AGENT_TIMEOUT: agent execution timeout in seconds (default: 1800)
  - FRONTEND_TEST_TIMEOUT: test execution timeout in seconds (default: 600)
  - FRONTEND_RUN_ORACLE: "1"/"true" to run oracle solution (default: false)
  - FRONTEND_AGENT_NAME: agent name (default: claude_code)
  - FRONTEND_RUNTIME_PROVIDER_NAME: suanli/gongfeng/... (default: gongfeng)
  - FRONTEND_RUNTIME_TOKEN: runtime provider token
  - FRONTEND_RUNTIME_API_BASE_URL: runtime base URL
  - FRONTEND_RUNTIME_CONTAINER_IMAGE: sandbox container image
  - FRONTEND_RUNTIME_IMAGE: nexus runtime image
  - FRONTEND_WORKING_DIR: agent working directory (default: /workspace)
  - CLAUDE_PLUGINS: comma-separated plugin names
  - LLM_PROVIDER: "hy3" (default) or "opus" or "openai"
  - ENABLE_THINKING: "1"/"true" to enable thinking mode (default: false)
  - FRONTEND_DISABLE_HOOKS: "1"/"true" to disable PreToolUse hooks (default: false)
  - SKILL_NAME: skill name to prepend as "/<skill>" trigger before prompt
  - FRONTEND_SKILLS_DIR: comma-separated local skill directories to sync into sandbox
  - FRONTEND_ENABLE_SEARCH: "1"/"true" to enable WebSearch MCP (default: false)
  - FRONTEND_SEARCH_MCP_URL: SSE URL for WebSearch MCP server
  - OPENAPI_API_KEY / OPENAPI_BASE_URL / OPENAPI_MODEL_NAME: model client config
"""

from __future__ import annotations

import base64
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
from nexus.runner.single import RunInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_comma_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw_items = value if isinstance(value, (list, tuple, set)) else str(value).split(',')
    return [str(item).strip() for item in raw_items if str(item).strip()]


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
    v = os.getenv(name)
    if not v:
        return None
    items = [item.strip() for item in v.split(',') if item.strip()]
    return set(items) if items else None


def _iter_instances(dataset_path: str) -> Iterator[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Environment file deployment: task_dir/environment/ → /workspace
# Also deploys instruction.md → /instruction.md (for reward_request_agent.py)
# ---------------------------------------------------------------------------


def _build_environment_install_script(task_dir: str, target_dir: str = '/workspace') -> str:
    """Generate Python script to deploy environment files and instruction.md to container."""
    task_path = Path(task_dir)
    lines = [
        'import os, base64',
        f'target_dir = {target_dir!r}',
        'os.makedirs(target_dir, exist_ok=True)',
    ]

    # Deploy environment/ files to target_dir
    env_dir = task_path / 'environment'
    if env_dir.is_dir():
        for fp in sorted(env_dir.rglob('*')):
            if fp.is_file():
                try:
                    content_bytes = fp.read_bytes()
                except Exception:
                    continue
                rel_path = str(fp.relative_to(env_dir))
                encoded = base64.b64encode(content_bytes).decode()
                dest = f'os.path.join(target_dir, {rel_path!r})'
                lines.append(f'os.makedirs(os.path.dirname({dest}), exist_ok=True)')
                lines.append(f'open({dest}, "wb").write(base64.b64decode({encoded!r}))')
        lines.append(f'print("Deployed {env_dir.name}/ to " + target_dir)')

    # Deploy instruction.md to /instruction.md (reward_request_agent.py reads it from there)
    instruction_file = task_path / 'instruction.md'
    if instruction_file.is_file():
        encoded = base64.b64encode(instruction_file.read_bytes()).decode()
        lines.append(f'open("/instruction.md", "wb").write(base64.b64decode({encoded!r}))')
        lines.append('print("Deployed instruction.md to /instruction.md")')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Claude Code plugin offline installation
# ---------------------------------------------------------------------------

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
    mp_json = _MARKETPLACE_ROOT / '.claude-plugin' / 'marketplace.json'
    if not mp_json.exists():
        raise FileNotFoundError(
            f'marketplace.json not found at {mp_json}. '
            f'Run: claude plugin marketplace add anthropics/{_MARKETPLACE_NAME}'
        )
    return mp_json.read_text('utf-8')


def _build_plugin_install_script(plugins: list[str]) -> str:
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
            lines.append(f'open(fp, "w").write({json.dumps(content)})')

        lines.append(f'installed["plugins"][{key!r}] = [{{')
        lines.append('    "scope": "user",')
        lines.append('    "installPath": cache_dir,')
        lines.append(f'    "version": {version!r},')
        lines.append('    "installedAt": "2026-04-23T16:06:43.987Z",')
        lines.append('    "lastUpdated": "2026-04-23T16:06:43.987Z",')
        lines.append(f'    "gitCommitSha": {git_sha!r},')
        lines.append('}]')

    enabled = dict.fromkeys(plugin_data, True)

    lines.append('')
    lines.append(f'mp_dir = os.path.join(plugins_dir, "marketplaces", {_MARKETPLACE_NAME!r}, ".claude-plugin")')
    lines.append('os.makedirs(mp_dir, exist_ok=True)')
    lines.append(f'open(os.path.join(mp_dir, "marketplace.json"), "w").write({json.dumps(marketplace_json_content)})')
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
# Skills injection
# ---------------------------------------------------------------------------


def _parse_skills_dirs() -> list[Path]:
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
    lines = [
        'import os, base64',
        'home = os.environ.get("HOME", "/root")',
        'skills_root = os.path.join(home, ".claude", "skills")',
        'os.makedirs(skills_root, exist_ok=True)',
    ]

    for skill_dir in skills_dirs:
        skill_name = skill_dir.name
        for fp in sorted(skill_dir.rglob('*')):
            if fp.is_file():
                try:
                    content_bytes = fp.read_bytes()
                except Exception:
                    continue
                rel_path = str(fp.relative_to(skill_dir))
                encoded = base64.b64encode(content_bytes).decode()
                target = f'os.path.join(skills_root, {skill_name!r}, {rel_path!r})'
                lines.append(f'os.makedirs(os.path.dirname({target}), exist_ok=True)')
                lines.append(f'open({target}, "wb").write(base64.b64decode({encoded!r}))')
        lines.append(f'print("Skill installed: {skill_name}")')

    return '\n'.join(lines)


SKILLS_DIRS_TO_INSTALL: list[Path] = _parse_skills_dirs()


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

_EXITPLAN_HOOK_CMD = (
    'echo \'{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{}}}\''
)

_ASKUSER_HOOK_SCRIPT = (Path(__file__).parent / 'askuser_hook.py').read_text(encoding='utf-8')
_ASKUSER_HOOK_SCRIPT_PATH = '/tmp/_nexus_askuser_hook.py'
_ASKUSER_HOOK_CMD = f'python3 {_ASKUSER_HOOK_SCRIPT_PATH}'


def _build_askuser_hook_script_writer() -> str:
    encoded = base64.b64encode(_ASKUSER_HOOK_SCRIPT.encode()).decode()
    return (
        'import base64, pathlib\n'
        f'pathlib.Path({_ASKUSER_HOOK_SCRIPT_PATH!r}).write_text('
        f'base64.b64decode({encoded!r}).decode(), encoding="utf-8")\n'
    )


def _build_hook_patches() -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# OpenAI client builders
# ---------------------------------------------------------------------------


def _build_openai_clients_hy3(
    enable_thinking: bool = False, session_id: str | None = None
) -> dict[str, dict[str, Any]]:
    temperature = _env_float('OPENAPI_TEMP', 0.9)
    model_name = os.getenv('OPENAPI_MODEL_NAME', 'openai/HY3-dev0413')
    api_key = os.getenv('OPENAPI_API_KEY', 'EMPTY')
    base_url = os.getenv('OPENAPI_BASE_URL', 'http://infer-proxy-yb-test.production.polaris:8000/openapi')

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
    api_key = os.getenv('OPENAPI_API_KEY', 'EMPTY')
    base_url = os.getenv('OPENAI_BASE_URL', 'http://llm-api.model-eval.woa.com/v1')
    model_name = os.getenv('OPENAPI_MODEL_NAME', 'openai/xxxx')

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


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

DEFAULT_DATASET_PATH = (
    '/apdcephfs_zwfy3/share_302867151/maxluxchen/data/frontend_data/fitz_data/data_analysis/data/tasks.jsonl'
)


def generate_configs() -> Iterator[BatchItem]:
    dataset_path = os.getenv('FRONTEND_DATASET_PATH', DEFAULT_DATASET_PATH)
    skip = _env_int('FRONTEND_SKIP') or 0
    limit = _env_int('FRONTEND_LIMIT')
    instance_ids_filter = _env_list('FRONTEND_INSTANCE_IDS')
    runtime_provider_name = os.getenv('FRONTEND_RUNTIME_PROVIDER_NAME', 'gongfeng')
    agent_name = os.getenv('FRONTEND_AGENT_NAME', 'claude_code')
    working_dir = os.getenv('FRONTEND_WORKING_DIR', '/workspace')
    agent_timeout = int(os.getenv('FRONTEND_AGENT_TIMEOUT', '1800'))
    test_timeout = int(os.getenv('FRONTEND_TEST_TIMEOUT', '600'))

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

        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue

        if skipped < skip:
            skipped += 1
            continue

        task_dir = str(inst.get('task_dir', ''))
        if not task_dir:
            print(f'WARNING: instance {instance_id} has no task_dir, skipping')
            continue

        # Build prompt: append report filename requirement
        original_prompt = str(inst.get('prompt', ''))
        final_prompt = original_prompt.rstrip() + '\n\n结果html命名为report.html'

        # Prepend skill trigger if SKILL_NAME is set
        skill_name = str(os.getenv('SKILL_NAME', '') or '').strip()
        if skill_name:
            final_prompt = f'/{skill_name} ' + final_prompt

        # Plugin install script
        plugins_to_install = _parse_comma_list(os.getenv('CLAUDE_PLUGINS', ''))
        plugin_key = tuple(plugins_to_install)
        if plugin_key not in plugin_install_scripts:
            plugin_install_scripts[plugin_key] = (
                _build_plugin_install_script(plugins_to_install) if plugins_to_install else None
            )
        _plugin_install_script = plugin_install_scripts[plugin_key]

        # Environment deploy script (task_dir/environment/ → /workspace, instruction.md → /)
        env_install_script = _build_environment_install_script(task_dir, target_dir=working_dir)

        # Collect EVAL_JUDGE_* env vars → container JUDGE_* env vars for test.sh
        eval_env: dict[str, str] = {}
        for host_key, container_key in [
            ('EVAL_JUDGE_MODEL', 'JUDGE_MODEL'),
            ('EVAL_JUDGE_HOST', 'JUDGE_HOST'),
            ('EVAL_JUDGE_APP_ID', 'JUDGE_APP_ID'),
            ('EVAL_JUDGE_APP_KEY', 'JUDGE_APP_KEY'),
            ('EVAL_JUDGE_TIMEOUT', 'JUDGE_TIMEOUT'),
        ]:
            val = os.getenv(host_key, '')
            if val:
                eval_env[container_key] = val

        task_imports = [
            'nexus.extensions.tasks.frontend',
            'nexus.extensions.agents.claude_code',
        ]
        task_config: dict[str, Any] = {
            'prompt': final_prompt,
            'task_dir': task_dir,
            'test_timeout': test_timeout,
            'run_eval': True,
            'working_dir': working_dir,
            'eval_env': eval_env,
        }

        config: dict[str, Any] = {
            'imports': task_imports,
            'task_name': 'frontend_task',
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
                    'cpu': 16,
                    'memory': 32,
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

        config['agent_name'] = agent_name
        config['agent_config'] = {
            'pre_start_scripts': [
                {'code': f'mkdir -p {working_dir}'},
                {'interpreter': 'python', 'code': env_install_script},
                *([{'interpreter': 'python', 'code': _plugin_install_script}] if _plugin_install_script else []),
                *(
                    [{'interpreter': 'python', 'code': _build_skills_install_script(SKILLS_DIRS_TO_INSTALL)}]
                    if SKILLS_DIRS_TO_INSTALL
                    else []
                ),
                *(
                    [{'interpreter': 'python', 'code': _build_askuser_hook_script_writer()}]
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
            'timeout': agent_timeout,
            'thinking_preset': 'bedrock-claude',
            'conversion_log_level': 'DEBUG',
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

        # LLM provider
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
        # EVAL_TIMES: generate N BatchItems per instance
        eval_times = _env_int('EVAL_TIMES') or 1
        for _round in range(eval_times):
            if _round > 0:
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
                        'task_dir': task_dir,
                        'prompt': original_prompt,
                    },
                )
            )

        produced += 1
        if limit is not None and produced >= limit:
            break

    # Shuffle by default
    if _env_bool('FRONTEND_SHUFFLE', default=True):
        random.shuffle(items)

    yield from items


def get_total_count() -> int:
    dataset_path = os.getenv('FRONTEND_DATASET_PATH', DEFAULT_DATASET_PATH)
    skip = _env_int('FRONTEND_SKIP') or 0
    limit = _env_int('FRONTEND_LIMIT')
    instance_ids_filter = _env_list('FRONTEND_INSTANCE_IDS')
    n = 0
    skipped = 0
    for inst in _iter_instances(dataset_path):
        instance_id = str(inst.get('instance_id', '') or '')
        if not instance_id:
            prompt = inst.get('prompt', '')
            if not prompt:
                continue
            instance_id = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        if instance_ids_filter is not None and instance_id not in instance_ids_filter:
            continue
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
        'instance_ids_filter': list(instance_ids_filter) if instance_ids_filter else None,
    }
