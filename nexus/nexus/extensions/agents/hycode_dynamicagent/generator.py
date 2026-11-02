"""Batch generator for HyCode DynamicAgent Code Arena harnesses.

Usage:
  export OPENAI_API_KEY=...
  export OPENAI_BASE_URL=https://api.deepseek.com
  export OPENAI_MODEL=deepseek-chat
  export LANGFUSE_SECRET_KEY=...
  export LANGFUSE_PUBLIC_KEY=...
  export LANGFUSE_BASE_URL=http://21.91.27.43:3000

  HYCODE_DYNAMICAGENT_HARNESSES=code-arena-mcp,code-arena-react \
  HYCODE_DYNAMICAGENT_PROMPT='Create a 3D rendered Transformer architecture diagram.' \
  uv run nexus run-batch \
    -g ./nexus/extensions/agents/hycode_dynamicagent/generator.py \
    -d ./outputs/hycode_dynamicagent_code_arena.db \
    --max-concurrent 1
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


DEFAULT_LANGFUSE_BASE_URL = 'http://21.91.27.43:3000'
DEFAULT_PACKAGE_VERSION = 'latest'
DEFAULT_NPM_REGISTRY = 'https://mirrors.tencent.com/npm/'
DEFAULT_MODEL = 'deepseek-chat'
DEFAULT_BASE_URL = 'https://api.deepseek.com'
DEFAULT_PROMPT = (
    'Create a polished 3D rendered diagram of the Transformer model architecture. '
    'Show token embeddings, positional encoding, stacked encoder/decoder blocks, '
    'multi-head attention, feed-forward layers, residual connections, normalization, '
    'and output probabilities with clear labels and an interactive or visually rich presentation.'
)

SUPPORTED_HARNESSES = (
    'code-arena-mcp',
    'code-arena-html',
    'code-arena-react',
    'code-arena-nextjs-fullstack',
)
DEFAULT_HARNESSES = SUPPORTED_HARNESSES


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_json(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if not value:
        return default
    return json.loads(value)


def _env_list(name: str, default: tuple[str, ...]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return list(default)
    items = [item.strip() for item in value.split(',') if item.strip()]
    if items == ['all']:
        return list(SUPPORTED_HARNESSES)
    unknown = [item for item in items if item not in SUPPORTED_HARNESSES]
    if unknown:
        raise ValueError(f'Unsupported harness(es): {unknown}. Supported: {SUPPORTED_HARNESSES}')
    return items


def _required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = ' or '.join(names)
    raise ValueError(f'Missing required environment variable: {joined}')


def _react_bootstrap_script(working_dir: str = '/testbed') -> str:
    # Header carries the only dynamic bit (working_dir); body is a raw string so the
    # JSON/TSX braces don't collide with f-string formatting.
    return (
        f'set -euo pipefail\ncd {working_dir}\n'
        + r"""if [ ! -f package.json ]; then
  cat > package.json <<'JSON'
{"scripts":{"build":"vite build","dev":"vite --host 0.0.0.0"},"dependencies":{"@vitejs/plugin-react":"latest","vite":"latest","typescript":"latest","react":"latest","react-dom":"latest","lucide-react":"latest","three":"latest","@types/react":"latest","@types/react-dom":"latest"},"devDependencies":{}}
JSON
fi
mkdir -p src
cat > index.html <<'HTML'
<!doctype html><html><head><meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" /><title>Code Arena React</title></head><body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body></html>
HTML
cat > src/main.tsx <<'TS'
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './style.css'
createRoot(document.getElementById('root')!).render(<App />)
TS
cat > src/App.tsx <<'TS'
export default function App() {
  return <main className="min-h-screen bg-neutral-950 text-white">Ready</main>
}
TS
cat > src/style.css <<'CSS'
html,body,#root{margin:0;min-height:100%;font-family:Inter,ui-sans-serif,system-ui,sans-serif}
CSS
npm install --silent --no-audit --no-fund
"""
    )


def _nextjs_bootstrap_script(working_dir: str = '/testbed') -> str:
    return (
        f'set -euo pipefail\ncd {working_dir}\n'
        + r"""if [ ! -f package.json ]; then
  cat > package.json <<'JSON'
{"scripts":{"build":"next build","start":"next start -p 3000","dev":"next dev --hostname 0.0.0.0"},"dependencies":{"next":"latest","react":"latest","react-dom":"latest","typescript":"latest","@types/node":"latest","@types/react":"latest","@types/react-dom":"latest","lucide-react":"latest","three":"latest","drizzle-orm":"latest","pg":"latest"},"devDependencies":{}}
JSON
fi
mkdir -p src/app/api/health src/db
cat > tsconfig.json <<'JSON'
{"compilerOptions":{"target":"ES2017","lib":["dom","dom.iterable","esnext"],"allowJs":true,"skipLibCheck":true,"strict":false,"noEmit":true,"incremental":true,"esModuleInterop":true,"module":"esnext","moduleResolution":"bundler","resolveJsonModule":true,"isolatedModules":true,"jsx":"preserve","plugins":[{"name":"next"}],"baseUrl":".","paths":{"@/*":["./src/*"]}},"include":["next-env.d.ts","**/*.ts","**/*.tsx",".next/types/**/*.ts"],"exclude":["node_modules"]}
JSON
cat > src/app/layout.tsx <<'TS'
import './globals.css'
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>
}
TS
cat > src/app/page.tsx <<'TS'
export default function Page() {
  return <main className="min-h-screen bg-neutral-950 text-white">Ready</main>
}
TS
cat > src/app/globals.css <<'CSS'
html,body{margin:0;min-height:100%;font-family:Inter,ui-sans-serif,system-ui,sans-serif}
CSS
cat > src/app/api/health/route.ts <<'TS'
export function GET() {
  return Response.json({ ok: true })
}
TS
cat > src/db/index.ts <<'TS'
export const db = {}
TS
cat > src/db/schema.ts <<'TS'
export const schema = {}
TS
printf 'DATABASE_URL=postgresql://postgres:postgres@localhost:5432/app_db\n' > .env
npm install --silent --no-audit --no-fund
"""
    )


def _bootstrap_scripts_for(harness: str, working_dir: str = '/testbed') -> list[dict[str, Any]]:
    if not _env_bool('HYCODE_DYNAMICAGENT_BOOTSTRAP_TEMPLATES', True):
        return []
    if harness == 'code-arena-react':
        return [{'interpreter': 'bash', 'timeout': 1200.0, 'code': _react_bootstrap_script(working_dir)}]
    if harness == 'code-arena-nextjs-fullstack':
        return [{'interpreter': 'bash', 'timeout': 1800.0, 'code': _nextjs_bootstrap_script(working_dir)}]
    return []


def _extra_cli_args(harness: str) -> list[str]:
    args = [
        '--print',
        '--input-format',
        'text',
        '--output-format',
        'stream-json',
        '--verbose',
        '--no-session-persistence',
        '--permission-mode',
        'bypassPermissions',
    ]
    if not _env_bool('HYCODE_DYNAMICAGENT_ALLOW_NATIVE_WEB', False):
        args.extend(['--disallowedTools', 'WebSearch', 'WebFetch'])
    args.extend(['--agent', harness])
    extra = _env_json('HYCODE_DYNAMICAGENT_EXTRA_CLI_ARGS_JSON', [])
    if extra:
        args.extend(extra)
    return args


def _openai_clients() -> dict[str, dict[str, Any]]:
    api_key = _required_env('OPENAI_API_KEY')
    request_args: dict[str, Any] = {
        'model': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
        'max_tokens': _env_int('OPENAI_MAX_TOKENS', 4000) or 4000,
    }
    request_args.update(_env_json('OPENAI_REQUEST_ARGS_JSON', {}))
    return {
        'main': {
            'name': os.getenv('OPENAI_CLIENT_NAME', 'openai'),
            'config': {
                'client_args': {
                    'api_key': api_key,
                    'base_url': os.getenv('OPENAI_BASE_URL', DEFAULT_BASE_URL),
                    'max_retries': _env_int('OPENAI_MAX_RETRIES', 3) or 3,
                },
                'request_args': request_args,
            },
        },
    }


def _processors(harness: str, run_id: str) -> dict[str, Any]:
    processors: dict[str, Any] = {
        'chat': {
            'name': 'chat_tracker',
            'args': {'tools_mode': os.getenv('HYCODE_DYNAMICAGENT_TOOLS_MODE', 'branch_last')},
        },
    }
    if _env_bool('HYCODE_DYNAMICAGENT_DISABLE_LANGFUSE', False):
        return processors

    model = os.getenv('OPENAI_MODEL', DEFAULT_MODEL)
    experiment = os.getenv('HYCODE_DYNAMICAGENT_EXPERIMENT', 'hycode_dynamicagent_code_arena')
    processors['langfuse'] = {
        'name': 'langfuse_tracker',
        'args': {
            'public_key': _required_env('LANGFUSE_PUBLIC_KEY'),
            'secret_key': _required_env('LANGFUSE_SECRET_KEY'),
            'host': os.getenv('LANGFUSE_BASE_URL', DEFAULT_LANGFUSE_BASE_URL),
            'session_id': run_id,
            'trace_name': run_id,
            'tags': [
                'hycode_dynamicagent',
                'code_arena',
                harness,
                experiment,
                model,
            ],
            'trace_metadata': {
                'agent_name': 'hycode_dynamicagent',
                'harness': harness,
                'experiment': experiment,
                'model': model,
                'package_version': os.getenv('HYCODE_DYNAMICAGENT_PACKAGE_VERSION', DEFAULT_PACKAGE_VERSION),
                'prompt': os.getenv('HYCODE_DYNAMICAGENT_PROMPT', DEFAULT_PROMPT),
            },
        },
    }
    return processors


def _runtime_configs() -> dict[str, dict[str, Any]]:
    default = {
        'main': {
            'image': os.getenv('HYCODE_DYNAMICAGENT_RUNTIME_IMAGE', 'node:22-bookworm'),
            'worker_venv': True,
        },
    }
    return _env_json('HYCODE_DYNAMICAGENT_RUNTIME_CONFIGS_JSON', default)


def _runtime_provider_config() -> dict[str, Any]:
    return _env_json('HYCODE_DYNAMICAGENT_RUNTIME_PROVIDER_CONFIG_JSON', {})


def _agent_config(harness: str) -> dict[str, Any]:
    working_dir = os.getenv('HYCODE_DYNAMICAGENT_WORKING_DIR', '/testbed')
    extra_env = {
        'CODE_ARENA_ROOT': working_dir,
        'CLAUDE_CODE_USE_NATIVE_FILE_SEARCH': '1',
        'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
    }
    extra_env.update(_env_json('HYCODE_DYNAMICAGENT_EXTRA_ENV_JSON', {}))
    return {
        'working_dir': working_dir,
        'timeout': _env_int('HYCODE_DYNAMICAGENT_AGENT_TIMEOUT', 3600) or 3600,
        'version': os.getenv('HYCODE_DYNAMICAGENT_PACKAGE_VERSION', DEFAULT_PACKAGE_VERSION),
        'npm_registry': os.getenv('HYCODE_DYNAMICAGENT_NPM_REGISTRY', DEFAULT_NPM_REGISTRY),
        'user_prompt': os.getenv('HYCODE_DYNAMICAGENT_PROMPT', DEFAULT_PROMPT),
        'extra_env': extra_env,
        'extra_cli_args': _extra_cli_args(harness),
        'pre_start_scripts': _bootstrap_scripts_for(harness),
    }


def _build_config(harness: str, repeat_index: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    experiment = os.getenv('HYCODE_DYNAMICAGENT_EXPERIMENT', 'hycode_dynamicagent_code_arena')
    run_id_prefix = os.getenv('HYCODE_DYNAMICAGENT_RUN_ID_PREFIX', experiment)
    run_id = f'{run_id_prefix}__{harness}__r{repeat_index}'
    config = {
        'imports': [
            'nexus.extensions.tasks.identity',
            'nexus.extensions.agents.hycode_dynamicagent',
        ],
        'task_name': 'identity',
        'run_id': run_id,
        'runtime_provider_name': os.getenv('HYCODE_DYNAMICAGENT_RUNTIME_PROVIDER', 'docker'),
        'runtime_provider_config': _runtime_provider_config(),
        'runtime_configs': _runtime_configs(),
        'agent_name': 'hycode_dynamicagent',
        'agent_config': _agent_config(harness),
        'openai_clients': _openai_clients(),
        'processors': _processors(harness, run_id),
    }
    metadata = {
        'agent_name': 'hycode_dynamicagent',
        'harness': harness,
        'repeat_index': repeat_index,
        'experiment': experiment,
        'model': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
    }
    return run_id, config, metadata


def _selected_items() -> list[tuple[str, int]]:
    harnesses = _env_list('HYCODE_DYNAMICAGENT_HARNESSES', DEFAULT_HARNESSES)
    repeat = _env_int('HYCODE_DYNAMICAGENT_REPEAT', 1) or 1
    items = [(harness, index) for index in range(repeat) for harness in harnesses]
    limit = _env_int('HYCODE_DYNAMICAGENT_LIMIT')
    if limit is not None:
        items = items[:limit]
    return items


def generate_configs() -> Iterator[BatchItem]:
    for harness, repeat_index in _selected_items():
        run_id, config, metadata = _build_config(harness, repeat_index)
        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=run_id,
            metadata=metadata,
        )


def get_total_count() -> int:
    return len(_selected_items())


def get_batch_metadata() -> dict[str, Any]:
    return {
        'agent_name': 'hycode_dynamicagent',
        'harnesses': [harness for harness, _ in _selected_items()],
        'experiment': os.getenv('HYCODE_DYNAMICAGENT_EXPERIMENT', 'hycode_dynamicagent_code_arena'),
        'model': os.getenv('OPENAI_MODEL', DEFAULT_MODEL),
        'langfuse_enabled': not _env_bool('HYCODE_DYNAMICAGENT_DISABLE_LANGFUSE', False),
        'langfuse_host': os.getenv('LANGFUSE_BASE_URL', DEFAULT_LANGFUSE_BASE_URL),
        'package_version': os.getenv('HYCODE_DYNAMICAGENT_PACKAGE_VERSION', DEFAULT_PACKAGE_VERSION),
    }


if __name__ == '__main__':
    print(json.dumps(get_batch_metadata(), ensure_ascii=False, indent=2))
    print(f'total_count={get_total_count()}')
