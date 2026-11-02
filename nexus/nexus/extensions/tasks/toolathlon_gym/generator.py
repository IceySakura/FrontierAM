"""Batch config generator for toolathlon_gym RL training.

Builds cases directly from the toolathlon manifest without mcp_benchmark's
``decide_toolathlon`` filtering or ``localize_toolathlon_rendered`` command
overrides.  toolathlon_gym's MCP servers are self-contained (local
implementations backed by PostgreSQL), so they need neither API-key checks
nor command localization.

Environment variables:
    TOOLATHLON_GYM_CASE_IDS:       Comma-separated case IDs to include
    TOOLATHLON_GYM_SKIP:           Number of cases to skip (pagination)
    TOOLATHLON_GYM_LIMIT:          Maximum number of cases to generate
    TOOLATHLON_GYM_MAX_ITERATIONS:  Override max agent iterations per case
    TOOLATHLON_GYM_RUN_TIMEOUT:    Per-task timeout in seconds (default: 7200)

    Standard mcp_benchmark runtime env vars are also supported:
    MCP_BENCHMARK_RUNTIME_PROVIDER_NAME, MCP_BENCHMARK_MAIN_IMAGE,
    MCP_BENCHMARK_RUNTIME_BASE_URL, MCP_BENCHMARK_RUNTIME_TOKEN, etc.
"""

from __future__ import annotations

import json
import logging
import os
import socket

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import yaml

from nexus.core.tool.mcp import MCPClientConfig, MCPServerSpec, MCPTransportKind
from nexus.extensions.tasks.mcp_benchmark.constants import (
    BENCHMARK_IMAGE_HELPERS,
    BENCHMARK_IMAGE_NAME,
    BENCHMARK_WORK_ROOT,
    TOOLATHLON_DATASET_ROOT,
    TOOLATHLON_IMAGE_ROOT,
    TOOLATHLON_IMAGE_TASKS_ROOT,
    TOOLATHLON_IMAGE_VENV_BIN,
    TOOLATHLON_SHIMS_ROOT,
)
from nexus.extensions.tasks.mcp_benchmark.credentials import WorkspaceKeys
from nexus.extensions.tasks.mcp_benchmark.manifests import load_toolathlon_manifest_rows
from nexus.extensions.tasks.mcp_benchmark.mode import BenchmarkServerMode
from nexus.extensions.tasks.mcp_benchmark.server_specs import render_value
from nexus.extensions.tasks.mcp_benchmark.types import (
    BenchmarkCase,
    BenchmarkCommand,
    BenchmarkDataset,
    BenchmarkEvaluator,
)
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import ProcessorConfig, RunInput, RunnerConfig


logger = logging.getLogger(__name__)

# Server yaml file name overrides (server_name -> yaml stem)
_SERVER_FILE_NAMES: dict[str, str] = {
    'arxiv-latex': 'arxiv-latex-mcp',
    'fetch': 'npx-fetch',
    'rail_12306': '12306',
    'scholarly': 'scholarly_search',
    'youtube-transcript': 'youtube_transcript',
}


# ---------------------------------------------------------------------------
# Case building — reads toolathlon_gym YAMLs directly, no localize/filtering
# ---------------------------------------------------------------------------


def _load_server_specs_direct(
    server_names: list[str],
    *,
    runtime_name: str,
    template_values: dict[str, str],
) -> list[dict[str, Any]]:
    """Load MCP server specs from toolathlon_gym YAML configs.

    Unlike ``mcp_benchmark.server_specs.load_toolathlon_server_specs``, this
    function does **not** localize commands or reject unresolved placeholders.
    toolathlon_gym's servers are self-contained local implementations that
    work as-is with the YAML-defined commands.
    """
    config_dir = TOOLATHLON_DATASET_ROOT / 'configs' / 'mcp_servers'
    specs: list[dict[str, Any]] = []
    for server_name in server_names:
        file_name = _SERVER_FILE_NAMES.get(server_name, server_name)
        path = config_dir / f'{file_name}.yaml'
        if not path.exists():
            logger.warning('No YAML config for server %s at %s, skipping', server_name, path)
            continue
        data = yaml.safe_load(path.read_text())
        rendered = render_value(data, template_values)
        params = rendered.get('params', {})
        client = MCPClientConfig(
            transport=MCPTransportKind.STDIO,
            command=params.get('command', ''),
            args=params.get('args', []),
            env=params.get('env', {}) or {},
            cwd=params.get('cwd'),
            timeout_s=rendered.get('client_session_timeout_seconds'),
        )
        spec = MCPServerSpec(
            name=rendered.get('name', server_name),
            runtime_name=runtime_name,
            client=client,
        )
        spec_dict = spec.model_dump(mode='json', exclude_none=True)

        # Rewrite `uv run <path>/server.py` → `<path>/.venv/bin/python <path>/server.py`
        # The sandbox `uv run` may re-sync the venv and drop extra deps (e.g. psycopg2).
        # Using .venv/bin/python directly avoids this issue.
        client_dict = spec_dict.get('client', {})
        if client_dict.get('command') == 'uv' and len(client_dict.get('args', [])) >= 2:
            args = client_dict['args']
            if args[0] == 'run' and args[1].endswith('.py'):
                server_dir = args[1].rsplit('/', 1)[0] if '/' in args[1] else '.'
                client_dict['command'] = f'{server_dir}/.venv/bin/python'
                client_dict['args'] = args[1:]  # drop 'run', keep server.py path

        specs.append(spec_dict)
    return specs


def _case_root(case_id: str, server_mode: BenchmarkServerMode) -> str:
    return f'{BENCHMARK_WORK_ROOT}/toolathlon/{server_mode.value}/{case_id}'


def _agent_workspace(case_id: str, server_mode: BenchmarkServerMode) -> str:
    return f'{_case_root(case_id, server_mode)}/agent_workspace'


def _build_toolathlon_gym_case(
    row: dict[str, Any],
    keys: WorkspaceKeys,
    *,
    server_mode: BenchmarkServerMode,
) -> BenchmarkCase | None:
    """Build a BenchmarkCase from a manifest row — no decide_toolathlon filtering."""
    case_id = row['case_id']
    # Resolve task_dir robustly: manifest may store task_dir_rel as empty string
    # (None in JSON → ''), in which case `TOOLATHLON_DATASET_ROOT / ''` collapses
    # to the dataset root itself — a SILENT bug that causes the entire dataset
    # (hundreds of MB) to be tarred and uploaded as "case data", clobbering
    # case_root with the wrong tree (Dockerfile, local_servers/, …) and hiding
    # the real evaluation/ folder. Fall back to the canonical tasks/finalpool path.
    task_dir_rel = row.get('task_dir_rel') or ''
    if task_dir_rel:
        task_dir = TOOLATHLON_DATASET_ROOT / task_dir_rel
    else:
        task_dir = TOOLATHLON_DATASET_ROOT / 'tasks' / 'finalpool' / case_id
    if not task_dir.is_dir():
        raise RuntimeError(
            f'Cannot resolve task_dir for case {case_id!r}: tried {task_dir} '
            f'(task_dir_rel={task_dir_rel!r}). Dataset root: {TOOLATHLON_DATASET_ROOT}'
        )
    prompt = row['prompt']
    system_prompt = row.get('system_prompt')
    server_names = list(row['server_names'])

    agent_ws = _agent_workspace(case_id, server_mode)
    case_root = _case_root(case_id, server_mode)

    # Build template values for YAML rendering
    template_values = keys.toolathlon_template_values(
        task_dir=task_dir,
        case_root=case_root,
        agent_workspace=agent_ws,
        server_mode=server_mode,
    )
    # Add case_root as a template variable (needed by sqlite-tools-mcp YAML)
    template_values['case_root'] = case_root

    # Load server specs directly (no localize, no placeholder check)
    mcp_servers = _load_server_specs_direct(
        server_names,
        runtime_name='main',
        template_values=template_values,
    )

    # Bootstraps
    toolathlon_pythonpath = ':'.join([
        BENCHMARK_IMAGE_HELPERS,
        TOOLATHLON_SHIMS_ROOT,
        case_root,
        f'{TOOLATHLON_IMAGE_ROOT}/utils',
        TOOLATHLON_IMAGE_ROOT,
    ])
    toolathlon_env = {
        'PYTHONPATH': toolathlon_pythonpath,
        'NEXUS_TOOLATHLON_CASE_ROOT': case_root,
        'NEXUS_TOOLATHLON_DATASET_ROOT': TOOLATHLON_IMAGE_ROOT,
        'NEXUS_TOOLATHLON_EMAIL_STORE_PATH': f'{case_root}/.nexus_email_store.json',
        'NEXUS_TOOLATHLON_GOOGLE_MOCK_BASE_URL': 'http://127.0.0.1:10004',
        'NEXUS_TOOLATHLON_GOOGLE_STORE_PATH': f'{case_root}/.nexus_google_store.json',
        'NEXUS_TOOLATHLON_MOCK_ROOT': f'{case_root}/.nexus_mock',
        'NEXUS_TOOLATHLON_NOTION_STORE_PATH': f'{case_root}/.nexus_notion_store.json',
        'NEXUS_TOOLATHLON_CANVAS_STORE_PATH': f'{case_root}/.nexus_canvas_store.json',
        'NEXUS_TOOLATHLON_K8S_STATE_PATH': f'{case_root}/.nexus_k8s_state.json',
        'NEXUS_TOOLATHLON_K8S_ROOT': f'{case_root}/.nexus_k8s',
        'NEXUS_TOOLATHLON_SNOWFLAKE_DB_PATH': f'{case_root}/.nexus_snowflake.sqlite',
        'NEXUS_TOOLATHLON_WOOCOMMERCE_STORE_PATH': f'{case_root}/.nexus_woocommerce_store.json',
        'NEXUS_TOOLATHLON_WOOCOMMERCE_MEDIA_DIR': f'{case_root}/.nexus_woocommerce_media',
        'NEXUS_TOOLATHLON_SERVER_MODE': server_mode.value,
        'NEXUS_TOOLATHLON_TOKEN_VALUES_JSON': json.dumps(
            {k.removeprefix('token.'): v for k, v in template_values.items() if k.startswith('token.')},
            ensure_ascii=False,
        ),
        # PostgreSQL connection for sandbox (preprocess scripts may need these)
        'PG_HOST': '127.0.0.1',
        'PG_PORT': '5432',
        'PG_USER': 'eigent',
        'PG_PASSWORD': 'camel',
        'PG_DATABASE': 'toolathlon_gym',
        'PGHOST': '127.0.0.1',
        'PGPORT': '5432',
        'PGUSER': 'eigent',
        'PGPASSWORD': 'camel',
        'PGDATABASE': 'toolathlon_gym',
    }

    bootstraps: list[BenchmarkCommand] = [
        # Start PostgreSQL if available (toolathlon_gym training image)
        BenchmarkCommand(
            command='test -x /opt/toolathlon_gym_db/start_pg.sh && /opt/toolathlon_gym_db/start_pg.sh true || true',
            cwd='/tmp',
            env={},
            timeout_s=30.0,
        ),
        # Fix sent_log foreign key: init.sql.gz lacks ON DELETE CASCADE, causing
        # ForeignKeyViolation when preprocess scripts DELETE FROM email.messages.
        # Retry up to 5 times with 1s delay to handle PG still starting up.
        BenchmarkCommand(
            command=(
                f'{TOOLATHLON_IMAGE_VENV_BIN}/python -c "'
                'import psycopg2, os, time\n'
                'for attempt in range(5):\n'
                '    try:\n'
                "        conn = psycopg2.connect(host=os.environ.get('PGHOST','127.0.0.1'), "
                "database=os.environ.get('PGDATABASE','toolathlon_gym'), "
                "user=os.environ.get('PGUSER','eigent'), "
                "password=os.environ.get('PGPASSWORD','camel'))\n"
                '        conn.autocommit = True\n'
                '        cur = conn.cursor()\n'
                "        cur.execute('ALTER TABLE IF EXISTS email.sent_log DROP CONSTRAINT IF EXISTS sent_log_message_id_fkey')\n"
                "        cur.execute('ALTER TABLE IF EXISTS email.sent_log ADD CONSTRAINT sent_log_message_id_fkey "
                "FOREIGN KEY (message_id) REFERENCES email.messages(id) ON DELETE CASCADE')\n"
                '        conn.close()\n'
                '        break\n'
                '    except Exception as e:\n'
                '        if attempt < 4:\n'
                '            time.sleep(1)\n'
                '        else:\n'
                "            print(f'FK fix failed after retries: {e}')\n"
                '"'
            ),
            cwd='/tmp',
            env=toolathlon_env,
            timeout_s=30.0,
        ),
        BenchmarkCommand(
            command=f'{TOOLATHLON_IMAGE_VENV_BIN}/python {BENCHMARK_IMAGE_HELPERS}/toolathlon_prepare_case.py --case-root {case_root}',
            cwd=case_root,
            env=toolathlon_env,
            timeout_s=120.0,
        ),
    ]

    # --- Runtime fallback for MCP servers missing build artifacts in the image.
    #
    # Long-term fix lives in image/Dockerfile copy.toolathlon_gym (Group 2 now covers
    # 12306-mcp). This runtime `npm install` is a backward-compat safety net:
    #   * With an old image (e.g. toolathlon_gym:20260417v1) missing the 12306-mcp build,
    #     it does the install on first use of that sandbox.
    #   * With a freshly-built image that already has node_modules + build/index.js,
    #     the `test -d ... && test -f ...` short-circuits and this becomes a no-op.
    # Idempotent and cheap when everything is in place.
    node_servers_needing_install = {
        # server_name -> absolute server directory in the image
        'rail_12306': '/opt/mcp-benchmarks/local_servers/12306-mcp',
    }
    for server_name in server_names:
        server_dir = node_servers_needing_install.get(server_name)
        if not server_dir:
            continue
        bootstraps.append(
            BenchmarkCommand(
                command=(
                    f'cd {server_dir} && '
                    f'(test -d node_modules && test -f build/index.js) || '
                    f'(npm install --omit=dev --no-audit --no-fund --prefer-offline 2>&1 | tail -5 && '
                    f' (test -f build/index.js || npx tsc 2>&1 | tail -3))'
                ),
                cwd='/tmp',
                env={},
                timeout_s=300.0,
            )
        )

    # Runtime fallback for Python MCP servers needing `uv sync`.
    # sqlite-tools-mcp is bundled inside each case directory at case_root/mcp_server/
    # and needs its venv initialized on first run.
    _needs_sqlite_tools = any('mcp_server' in ' '.join(spec.get('client', {}).get('args', [])) for spec in mcp_servers)
    if _needs_sqlite_tools:
        bootstraps.append(
            BenchmarkCommand(
                command=(f'cd {case_root}/mcp_server && test -d .venv || uv sync 2>&1 | tail -5'),
                cwd='/tmp',
                env={},
                timeout_s=120.0,
            )
        )

    # Inject k8s env into server spec if needed
    for spec in mcp_servers:
        if spec.get('name') != 'k8s':
            continue
        client = spec.setdefault('client', {})
        env = dict(client.get('env', {}) or {})
        env.update({
            'NEXUS_TOOLATHLON_K8S_STATE_PATH': toolathlon_env['NEXUS_TOOLATHLON_K8S_STATE_PATH'],
            'NEXUS_TOOLATHLON_K8S_ROOT': toolathlon_env['NEXUS_TOOLATHLON_K8S_ROOT'],
        })
        client['env'] = env

    # Fix PostgreSQL connection params for sandbox environment.
    # The YAML configs use docker-compose values (host=postgres, user=postgres);
    # in gongfeng sandbox, PG runs locally with different credentials.
    # Some servers read PG_HOST/PG_USER, others read PGHOST/PGUSER from os.environ.
    # We inject both variants into ALL server envs so PG-backed servers work correctly.
    pg_env_inject = {
        'PG_HOST': '127.0.0.1',
        'PG_PORT': '5432',
        'PG_USER': 'eigent',
        'PG_PASSWORD': 'camel',
        'PG_DATABASE': 'toolathlon_gym',
        'PGHOST': '127.0.0.1',
        'PGPORT': '5432',
        'PGUSER': 'eigent',
        'PGPASSWORD': 'camel',
        'PGDATABASE': 'toolathlon_gym',
    }
    for spec in mcp_servers:
        client = spec.get('client', {})
        env = dict(client.get('env') or {})
        env.update(pg_env_inject)
        client['env'] = env

    # Preprocess
    preprocess = None
    preprocess_main = task_dir / 'preprocess' / 'main.py'
    preprocess_sh = task_dir / 'preprocess' / 'run.sh'
    if preprocess_main.exists():
        preprocess = BenchmarkCommand(
            command=f'{TOOLATHLON_IMAGE_VENV_BIN}/python -m preprocess.main',
            cwd=case_root,
            env=toolathlon_env,
            timeout_s=1200.0,
        )
    elif preprocess_sh.exists():
        preprocess = BenchmarkCommand(
            command=f'bash {case_root}/preprocess/run.sh',
            cwd=f'{case_root}/preprocess',
            env=toolathlon_env,
            timeout_s=1200.0,
        )

    # Evaluator
    evaluator = BenchmarkEvaluator(
        type='toolathlon_script',
        config={
            'command': (
                f'{TOOLATHLON_IMAGE_VENV_BIN}/python -m evaluation.main '
                f'--agent_workspace {case_root}/agent_workspace '
                f'--groundtruth_workspace {case_root}/groundtruth_workspace'
            ),
            'cwd': case_root,
            'env': toolathlon_env,
            'timeout_s': 1200.0,
        },
    )

    return BenchmarkCase(
        dataset=BenchmarkDataset.TOOLATHLON,
        case_id=case_id,
        server_mode=server_mode,
        prompt=prompt,
        system_prompt=system_prompt,
        mcp_servers=mcp_servers,
        max_iterations=100,
        runtime_name='main',
        case_source_dir=f'{TOOLATHLON_IMAGE_TASKS_ROOT}/{case_id}',
        initial_workspace_dir='initial_workspace' if (task_dir / 'initial_workspace').exists() else None,
        bootstraps=bootstraps,
        preprocess=preprocess,
        evaluator=evaluator,
        metadata={
            'source_path': str(task_dir),
            'server_names': server_names,
            'server_mode': server_mode.value,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


def _env_list(name: str) -> set[str] | None:
    raw = os.getenv(name)
    if not raw:
        return None
    items = {item.strip() for item in raw.split(',') if item.strip()}
    return items or None


def _build_runtime_settings() -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    """Build runtime provider config, mirroring mcp_benchmark's pattern."""
    provider = os.getenv('MCP_BENCHMARK_RUNTIME_PROVIDER_NAME', 'docker')
    main_image = os.getenv('MCP_BENCHMARK_MAIN_IMAGE', BENCHMARK_IMAGE_NAME)
    runtime_image = os.getenv('MCP_BENCHMARK_RUNTIME_IMAGE', 'current')
    request_timeout = float(os.getenv('MCP_BENCHMARK_RUNTIME_REQUEST_TIMEOUT', '1800'))
    runtime_cluster = (
        os.getenv('MCP_BENCHMARK_RUNTIME_PROVIDER_CLUSTER')
        or os.getenv('MCP_BENCHMARK_RUNTIME_CLUSTER')
        or os.getenv('MCP_BENCHMARK_GONGFENG_RUNTIME_PROVIDER')
    )
    runtime_cpu = _env_int('MCP_BENCHMARK_RUNTIME_CPU')
    runtime_memory = _env_int('MCP_BENCHMARK_RUNTIME_MEMORY')

    if provider == 'docker':
        provider_config: dict[str, Any] = {
            'runtime_image': runtime_image,
            'request_timeout': request_timeout,
        }
    else:
        provider_config = {
            'api_base_url': os.getenv('MCP_BENCHMARK_RUNTIME_BASE_URL', 'http://hyrl-sandbox.prod.woa.com:8052'),
            'token': os.getenv('MCP_BENCHMARK_RUNTIME_TOKEN', ''),
            'runtime_image': runtime_image,
            'request_timeout': request_timeout,
            'rpc_protocol': os.getenv('MCP_BENCHMARK_RPC_PROTOCOL', 'http'),
            'purpose': os.getenv('MCP_BENCHMARK_RUNTIME_PURPOSE', 'prod'),
        }
        if runtime_cluster:
            provider_config['runtime_provider'] = runtime_cluster
        resource_spec = {}
        if runtime_cpu is not None:
            resource_spec['cpu'] = runtime_cpu
        if runtime_memory is not None:
            resource_spec['memory'] = runtime_memory
        if resource_spec:
            provider_config['resource_spec'] = resource_spec

    runtime_configs = {
        'main': {
            'image': main_image,
            'working_dir': '/workspace',
        }
    }
    return provider, provider_config, runtime_configs


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """Build OpenAI client config from env vars."""
    base_url = os.getenv('MCP_BENCHMARK_OPENAI_BASE_URL', '')
    api_key = os.getenv('MCP_BENCHMARK_OPENAI_API_KEY', '')
    model = os.getenv('MCP_BENCHMARK_OPENAI_MODEL', '')
    name = os.getenv('MCP_BENCHMARK_OPENAI_CLIENTS_NAME', '')

    if not all([base_url, api_key, model]):
        return {}

    client_args: dict[str, Any] = {'base_url': base_url, 'api_key': api_key}
    wsid = os.getenv('MCP_BENCHMARK_OPENAI_WSID')
    if wsid:
        client_args['wsid'] = wsid

    return {
        'main': {
            'name': name,
            'config': {
                'client_args': client_args,
                'request_args': {'model': model},
            },
        }
    }


def _case_to_task_config(case: BenchmarkCase) -> dict[str, Any]:
    """Convert a BenchmarkCase to ToolathlonGymTaskInput-compatible config."""
    return {
        'case_id': case.case_id,
        'server_mode': case.server_mode.value,
        'prompt': case.prompt,
        'system_prompt': case.system_prompt,
        'mcp_servers': case.mcp_servers,
        'enabled_tools': case.enabled_tools,
        'max_iterations': case.max_iterations,
        'observation_time_timeout': 600,
        'tool_output_truncation': case.tool_output_truncation,
        'runtime_name': case.runtime_name,
        'case_source_dir': case.case_source_dir,
        'local_case_source_dir': case.metadata.get('source_path'),
        'initial_workspace_dir': case.initial_workspace_dir,
        'bootstraps': [cmd.model_dump(mode='json') for cmd in case.bootstraps],
        'preprocess': None if case.preprocess is None else case.preprocess.model_dump(mode='json'),
        'evaluator': case.evaluator.model_dump(mode='json'),
        'metadata': case.metadata,
    }


# ---------------------------------------------------------------------------
# Collection and filtering
# ---------------------------------------------------------------------------


def _collect_filtered_cases() -> list[BenchmarkCase]:
    """Collect all toolathlon_gym cases, building directly without API-key filtering."""
    server_mode = BenchmarkServerMode.MOCK
    keys = WorkspaceKeys()

    cases: list[BenchmarkCase] = []
    for row in load_toolathlon_manifest_rows():
        try:
            case = _build_toolathlon_gym_case(row, keys, server_mode=server_mode)
            if case is not None:
                cases.append(case)
        except Exception as exc:
            logger.warning('Skipping case %s: %s', row.get('case_id', '?'), exc)

    # Apply case ID filter
    case_ids_filter = _env_list('TOOLATHLON_GYM_CASE_IDS')
    if case_ids_filter is not None:
        cases = [c for c in cases if c.case_id in case_ids_filter]

    # Apply max_iterations override
    max_iterations_override = _env_int('TOOLATHLON_GYM_MAX_ITERATIONS')
    if max_iterations_override is not None:
        for c in cases:
            c.max_iterations = max_iterations_override

    # Pagination
    skip = _env_int('TOOLATHLON_GYM_SKIP', 0) or 0
    limit = _env_int('TOOLATHLON_GYM_LIMIT')
    cases = cases[skip:]
    if limit is not None:
        cases = cases[:limit]

    return cases


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_configs() -> Iterator[BatchItem]:
    """Generate batch items for RL training on Toolathlon tasks."""
    cases = _collect_filtered_cases()
    provider_name, provider_config, runtime_configs = _build_runtime_settings()
    openai_clients = _build_openai_clients()
    runner_timeout = float(os.getenv('TOOLATHLON_GYM_RUN_TIMEOUT', '7200'))

    for case in cases:
        run_id = f'toolathlon-gym-{case.case_id}'
        task_config = _case_to_task_config(case)

        config = RunInput(
            imports=['nexus.extensions.tasks.toolathlon_gym', 'nexus.extensions.agents.main'],
            task_name='toolathlon_gym',
            task_config=task_config,
            runtime_provider_name=provider_name,
            runtime_provider_config=provider_config,
            runtime_configs=runtime_configs,
            agent_name='main_agent',
            openai_clients=openai_clients,
            processors={'agent_metrics': ProcessorConfig(name='main_agent_metrics')},
            runner=RunnerConfig(timeout=runner_timeout),
        )
        yield BatchItem(
            config=config,
            run_id=run_id,
            metadata={
                'dataset': 'toolathlon',
                'case_id': case.case_id,
                'server_mode': case.server_mode.value,
                'source_path': case.metadata.get('source_path'),
                'server_names': case.metadata.get('server_names', []),
            },
        )


def get_total_count() -> int:
    """Return total number of cases after filtering."""
    return len(_collect_filtered_cases())


def get_batch_metadata() -> dict[str, Any]:
    """Return metadata about the current batch configuration."""
    cases = _collect_filtered_cases()
    return {
        'task_type': 'toolathlon_gym',
        'generated_count': len(cases),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'hostname': socket.gethostname(),
    }
