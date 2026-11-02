import asyncio
import copy
import json
import logging
import re
import shlex

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from nexus.core import require_context
from nexus.core.task import task_registry
from nexus.core.tool import ToolExecutor
from nexus.core.tool.mcp.models import ConnectedMCPTool
from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.executors.file import FileItem

from .cleanup import run_cleanups
from .constants import (
    BENCHMARK_CALL_RECORDS_KEY,
    BENCHMARK_LOCAL_SERVERS,
    BENCHMARK_WORK_ROOT,
    TOOLATHLON_IMAGE_ROOT,
    UNIVERSE_EVAL_HELPER,
)
from .evaluators import LLMJudgeEvaluator, ToolathlonRuntimeEvaluator, UniverseRuntimeEvaluator
from .server_specs import render_value
from .tools.python_execute import PYTHON_EXECUTE_WORKSPACE_KEY
from .types import (
    BenchmarkCommand,
    BenchmarkDataset,
    BenchmarkEvaluator,
    BenchmarkInputFile,
    BenchmarkService,
    MCPBenchmarkTaskInput,
    MCPBenchmarkTaskOutput,
)


logger = logging.getLogger(__name__)


DEFAULT_BENCHMARK_SYSTEM_PROMPT = (
    'You are operating inside a benchmark sandbox and must work autonomously. '
    'Use the available tools to inspect the environment, local files, repositories, and data before asking for help. '
    'Do not ask the user to provide local paths, repository locations, or environment details that can be discovered via tools. '
    'If the task appears to involve local data, proactively inspect accessible directories such as /data, /workspace, and the current workspace. '
    'When the task depends on exact facts, include the exact identifiers in the final answer as well, such as hashes, IDs, dates, counts, filenames, and quoted titles or messages when available. '
    'Base your final answer only on information you obtained from tool calls or directly from the prompt.'
)


def _case_root(task_input: MCPBenchmarkTaskInput) -> str:
    return str(
        PurePosixPath(BENCHMARK_WORK_ROOT)
        / task_input.dataset.value
        / task_input.server_mode.value
        / task_input.case_id
    )


def _agent_workspace(case_root: str) -> str:
    return str(PurePosixPath(case_root) / 'agent_workspace')


def _toolathlon_launch_time() -> str:
    # Use UTC: eval scripts run in UTC containers and parse this as local time.
    # If we emit CST here, the timestamp filter shifts +8h and misses all entries.
    return datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M:%S %A')


def _append_cli_arg(command: str, flag: str, value: str) -> str:
    if flag in command:
        return command
    return f'{command} {flag} {shlex.quote(value)}'


async def _write_toolathlon_eval_log(
    runtime, *, case_root: str, agent_output: Any, final_answer: str, launch_time: str
) -> str:
    log_path = str(PurePosixPath(case_root) / 'res_log.json')
    payload = {
        'status': 'success',
        'config': {'launch_time': launch_time},
        'messages': getattr(agent_output, 'main_trajectory', []),
        'final_answer': final_answer,
    }
    result = await runtime.upload_file([
        FileItem(path=log_path, content=json.dumps(payload, ensure_ascii=False), encoding='utf8')
    ])
    upload_result = result.results[0]
    if not upload_result.success:
        raise RuntimeError(f'failed to upload toolathlon eval log: {upload_result.message}')
    return log_path


def _case_file_path(case_root: str, path: str) -> str:
    pure_path = PurePosixPath(path)
    if pure_path.is_absolute():
        return str(pure_path)
    return str(PurePosixPath(case_root) / pure_path)


async def _run_checked_command(
    runtime, *, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout: float = 600.0
) -> None:
    result = await runtime.run_command(command=command, cwd=cwd, extra_env=env or {}, timeout=timeout)
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        raise RuntimeError(
            f'command failed: {command}\nstatus={result.status.value}\nreturn_code={result.return_code}\n'
            f'stdout={(result.stdout or "")[:4000]}\nstderr={(result.stderr or "")[:4000]}'
        )


def _runtime_config_for_service(service: BenchmarkService) -> dict[str, Any]:
    runtime_config = dict(service.runtime_config)
    runtime_config.setdefault('image', service.image)
    if service.exposed_ports:
        runtime_config.setdefault('exposed_ports', service.exposed_ports)
    if service.working_dir:
        runtime_config.setdefault('working_dir', service.working_dir)
    return runtime_config


async def _wait_for_service_ready(runtime, service: BenchmarkService) -> None:
    if service.ready_check is None:
        return

    deadline = asyncio.get_running_loop().time() + service.ready_timeout_s
    last_error: Exception | None = None

    while True:
        try:
            await _run_checked_command(
                runtime,
                command=service.ready_check.command,
                cwd=service.ready_check.cwd,
                env=service.ready_check.env,
                timeout=service.ready_check.timeout_s,
            )
            return
        except Exception as exc:
            last_error = exc
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(service.ready_poll_interval_s)

    raise RuntimeError(f'service {service.name} failed readiness check') from last_error


def _service_connection_values(service: BenchmarkService, runtime) -> dict[str, str]:
    values: dict[str, str] = {}
    for connection in service.connections:
        if connection.internal_port not in runtime.port_mappings:
            raise RuntimeError(
                f'service {service.name} missing port mapping for {connection.internal_port}; '
                f'available={sorted(runtime.port_mappings)}'
            )
        base_url = runtime.port_mappings[connection.internal_port]
        parsed = urlparse(base_url)
        host = parsed.hostname
        port = parsed.port
        if not host or port is None:
            raise RuntimeError(f'invalid port mapping for service {service.name}: {base_url}')
        values[connection.placeholder] = connection.value_template.format(
            base_url=base_url,
            host=host,
            port=port,
            internal_port=connection.internal_port,
            service_name=service.name,
            runtime_name=service.runtime_name,
        )
    return values


def _replace_placeholders(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        rendered = value
        for placeholder, replacement in replacements.items():
            rendered = rendered.replace(placeholder, replacement)
        return rendered
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, replacements) for key, item in value.items()}
    return value


def _resolve_input_files(
    input_files: list[BenchmarkInputFile], replacements: dict[str, str]
) -> list[BenchmarkInputFile]:
    payload = _replace_placeholders([item.model_dump(mode='json') for item in input_files], replacements)
    return [BenchmarkInputFile.model_validate(item) for item in payload]


def _resolve_commands(commands: list[BenchmarkCommand], replacements: dict[str, str]) -> list[BenchmarkCommand]:
    payload = _replace_placeholders([item.model_dump(mode='json') for item in commands], replacements)
    return [BenchmarkCommand.model_validate(item) for item in payload]


def _resolve_command(command: BenchmarkCommand | None, replacements: dict[str, str]) -> BenchmarkCommand | None:
    if command is None:
        return None
    return BenchmarkCommand.model_validate(_replace_placeholders(command.model_dump(mode='json'), replacements))


def _resolve_evaluator(evaluator: BenchmarkEvaluator, replacements: dict[str, str]) -> BenchmarkEvaluator:
    return BenchmarkEvaluator.model_validate(_replace_placeholders(evaluator.model_dump(mode='json'), replacements))


def _contains_template_placeholders(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False)
    return '${' in text or '{{' in text


def _ensure_no_template_placeholders(value: Any) -> None:
    if _contains_template_placeholders(value):
        raise ValueError(f'unresolved placeholders remain: {json.dumps(value, ensure_ascii=False)}')


def _register_toolathlon_google_map_aliases(ctx, *, dataset: BenchmarkDataset) -> list[str]:
    """Add compatibility aliases for Toolathlon Google Maps MCP tools.

    Some Toolathlon trajectories call ``google_map_maps_maps_*`` after seeing the
    ``google_map_maps_*`` namespace. Registering aliases here keeps the fix scoped
    to the benchmark task without touching shared tool execution code.
    """
    if dataset != BenchmarkDataset.TOOLATHLON:
        return []

    manager = getattr(ctx, '_mcp_manager', None)
    if manager is None:
        return []

    aliases: list[str] = []
    for full_name, tool in list(manager.tools.items()):
        if not full_name.startswith('google_map_maps_'):
            continue
        alias_name = full_name.replace('google_map_maps_', 'google_map_maps_maps_', 1)
        if alias_name in manager.tools:
            continue
        manager.tools[alias_name] = ConnectedMCPTool(
            full_name=alias_name,
            server_name=tool.server_name,
            tool_name=tool.tool_name,
            description=tool.description,
            input_schema=copy.deepcopy(tool.input_schema),
            server_key=tool.server_key,
        )
        ctx.mcp_tools[alias_name] = manager.tools[alias_name]
        aliases.append(alias_name)
    return aliases


async def _load_toolathlon_runtime_template_values(runtime, *, case_root: str) -> dict[str, str]:
    sentinel = '__NEXUS_TOOLATHLON_TEMPLATE_VALUES__='
    result = await runtime.run_command(
        command=(
            "python3 - <<'PY'\n"
            'import json\n'
            'import os\n'
            'import runpy\n'
            'import sys\n'
            'import types\n'
            'from pathlib import Path\n'
            '\n'
            'class Dict(dict):\n'
            '    def __getattr__(self, item):\n'
            '        try:\n'
            '            return self[item]\n'
            '        except KeyError as exc:\n'
            '            raise AttributeError(item) from exc\n'
            '\n'
            "module = types.ModuleType('addict')\n"
            'module.Dict = Dict\n'
            "sys.modules.setdefault('addict', module)\n"
            '\n'
            "token_path = Path(os.environ['CASE_ROOT']) / 'configs' / 'token_key_session.py'\n"
            'namespace = runpy.run_path(str(token_path))\n'
            "raw = namespace.get('all_token_key_session', {})\n"
            'try:\n'
            '    payload = dict(raw)\n'
            'except Exception:\n'
            '    payload = {}\n'
            'for key, value in namespace.items():\n'
            "    if key.startswith('_') or key == 'all_token_key_session':\n"
            '        continue\n'
            '    if callable(value) or isinstance(value, types.ModuleType):\n'
            '        continue\n'
            '    payload.setdefault(key, value)\n'
            'mapping = {}\n'
            'for key, value in payload.items():\n'
            '    if value is None:\n'
            '        continue\n'
            '    text = str(value).strip()\n'
            '    if not text:\n'
            '        continue\n'
            "    mapping[f'token.{key}'] = text\n"
            f'print({sentinel!r} + json.dumps(mapping, ensure_ascii=False))\n'
            'PY'
        ),
        cwd=case_root,
        extra_env={'CASE_ROOT': case_root},
        timeout=180.0,
    )
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        raise RuntimeError(
            'failed to load Toolathlon runtime token values\n'
            f'status={result.status.value}\nreturn_code={result.return_code}\n'
            f'stdout={(result.stdout or "")[:4000]}\nstderr={(result.stderr or "")[:4000]}'
        )

    for line in reversed((result.stdout or '').splitlines()):
        if line.startswith(sentinel):
            payload = json.loads(line[len(sentinel) :])
            if not isinstance(payload, dict):
                raise RuntimeError('Toolathlon runtime token payload is not an object')
            return {str(key): str(value) for key, value in payload.items()}

    raise RuntimeError(f'failed to parse Toolathlon runtime token values from stdout: {(result.stdout or "")[:4000]}')


async def _upload_input_files(runtime, *, case_root: str, input_files: list[BenchmarkInputFile]) -> None:
    if not input_files:
        return

    result = await runtime.upload_file(
        [
            FileItem(
                path=_case_file_path(case_root, item.path),
                content=item.content,
                encoding=item.encoding,
            )
            for item in input_files
        ],
        overwrite=True,
    )
    failures = [entry for entry in result.results if not entry.success]
    if failures:
        message = '\n'.join(f'{entry.path}: {entry.message}' for entry in failures[:20])
        raise RuntimeError(f'failed to upload benchmark input files:\n{message}')


async def _rewrite_case_text_files(runtime, *, case_root: str, replacements: dict[str, str]) -> None:
    if not replacements:
        return

    await _run_checked_command(
        runtime,
        command=(
            "python3 - <<'PY'\n"
            'import json\n'
            'import os\n'
            'from pathlib import Path\n'
            '\n'
            "case_root = Path(os.environ['CASE_ROOT'])\n"
            "replacements = json.loads(os.environ['TEXT_REPLACEMENTS'])\n"
            "suffixes = {'.py', '.sh', '.json', '.yaml', '.yml', '.txt', '.md', '.csv', '.html', '.js', '.ts'}\n"
            '\n'
            'for root, dirnames, filenames in os.walk(case_root, topdown=True, followlinks=False):\n'
            '    current = Path(root)\n'
            '    dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]\n'
            '    for filename in filenames:\n'
            '        path = current / filename\n'
            '        if path.is_symlink() or path.suffix.lower() not in suffixes:\n'
            '            continue\n'
            '        try:\n'
            "            original = path.read_text(encoding='utf-8')\n"
            '        except Exception:\n'
            '            continue\n'
            '        updated = original\n'
            '        for source, target in replacements.items():\n'
            '            updated = updated.replace(source, target)\n'
            '        if updated != original:\n'
            "            path.write_text(updated, encoding='utf-8')\n"
            'PY'
        ),
        cwd=case_root,
        env={
            'CASE_ROOT': case_root,
            'TEXT_REPLACEMENTS': json.dumps(replacements, ensure_ascii=False),
        },
        timeout=180.0,
    )


async def _setup_services(task_input: MCPBenchmarkTaskInput) -> dict[str, str]:
    if not task_input.services:
        return {}

    ctx = require_context()
    if ctx.runtime_provider is None:
        raise RuntimeError('benchmark services require a runtime provider')

    for service in task_input.services:
        ctx.runtime_provider.configure(service.runtime_name, **_runtime_config_for_service(service))

    replacements: dict[str, str] = {}
    for service in task_input.services:
        runtime = await ctx.get_runtime(service.runtime_name)
        runtime_replacements = _service_connection_values(service, runtime)
        if service.startup is not None:
            startup = _resolve_command(service.startup, runtime_replacements)
            assert startup is not None
            await _run_checked_command(
                runtime,
                command=startup.command,
                cwd=startup.cwd,
                env=startup.env,
                timeout=startup.timeout_s,
            )
        await _wait_for_service_ready(runtime, service)
        replacements.update(runtime_replacements)

    return replacements


async def _prepare_case(runtime, task_input: MCPBenchmarkTaskInput) -> tuple[str, str]:
    case_root = _case_root(task_input)
    agent_workspace = _agent_workspace(case_root)

    quoted_case_root = shlex.quote(case_root)
    quoted_agent_workspace = shlex.quote(agent_workspace)
    commands = [f'rm -rf {quoted_case_root}', f'mkdir -p {quoted_case_root} {quoted_agent_workspace}']

    if task_input.case_source_dir:
        quoted_source = shlex.quote(task_input.case_source_dir)
        commands.append(f'cp -a {quoted_source}/. {quoted_case_root}/')

    if task_input.dataset == BenchmarkDataset.TOOLATHLON:
        commands.extend([
            f'mkdir -p {quoted_case_root}/configs',
            f'mkdir -p {quoted_case_root}/configs/mcp_servers',
            f'ln -sfn {shlex.quote(f"{TOOLATHLON_IMAGE_ROOT}/deployment")} {quoted_case_root}/deployment',
            f'ln -sfn {shlex.quote(f"{TOOLATHLON_IMAGE_ROOT}/local_binary")} {quoted_case_root}/local_binary',
            f'ln -sfn {shlex.quote(BENCHMARK_LOCAL_SERVERS)} {quoted_case_root}/local_servers',
        ])

    if task_input.initial_workspace_dir:
        initial_workspace = shlex.quote(str(PurePosixPath(case_root) / task_input.initial_workspace_dir))
        commands.append(
            f'if [ -d {initial_workspace} ]; then cp -a {initial_workspace}/. {quoted_agent_workspace}/; fi'
        )

    await _run_checked_command(runtime, command=' && '.join(commands), timeout=600.0)
    return case_root, agent_workspace


def _build_agent_prompt(task_input: MCPBenchmarkTaskInput) -> str:
    prompt = task_input.prompt
    if task_input.output_format:
        prompt = (
            f'{prompt}\n\nReturn the final answer as JSON matching this schema/example:\n{task_input.output_format}'
        )
    return prompt


def _resolve_system_prompt(task_input: MCPBenchmarkTaskInput) -> str | list[dict[str, Any]]:
    return task_input.system_prompt if task_input.system_prompt is not None else DEFAULT_BENCHMARK_SYSTEM_PROMPT


async def _run_startup_tool_setup(task_input: MCPBenchmarkTaskInput, loaded_tool_names: list[str]) -> None:
    if task_input.dataset == BenchmarkDataset.ATLAS and 'desktop-commander_set_config_value' in loaded_tool_names:
        await ToolExecutor([]).call(
            'desktop-commander_set_config_value',
            {'key': 'allowedDirectories', 'value': ['/data']},
        )


async def _run_bootstraps(runtime, commands: list[BenchmarkCommand]) -> None:
    for command in commands:
        await _run_checked_command(
            runtime,
            command=command.command,
            cwd=command.cwd,
            env=command.env,
            timeout=command.timeout_s,
        )


async def _evaluate(
    task_input: MCPBenchmarkTaskInput,
    *,
    evaluator: BenchmarkEvaluator,
    runtime,
    final_answer: str,
    agent_workspace: str,
    launch_time: str | None,
    res_log_file: str | None,
) -> Any:
    evaluator_type = evaluator.type
    evaluator_config = dict(evaluator.config)

    if evaluator_type == 'atlas_llm_judge':
        return await LLMJudgeEvaluator().evaluate(final_answer=final_answer, config=evaluator_config)
    if evaluator_type == 'toolathlon_script':
        command = evaluator_config['command']
        if task_input.dataset == BenchmarkDataset.TOOLATHLON and launch_time is not None:
            command = _append_cli_arg(command, '--agent_workspace', agent_workspace)
            command = _append_cli_arg(command, '--launch_time', launch_time)
        if task_input.dataset == BenchmarkDataset.TOOLATHLON and res_log_file is not None:
            command = _append_cli_arg(command, '--res_log_file', res_log_file)
        return await ToolathlonRuntimeEvaluator().evaluate(
            runtime=runtime,
            command=command,
            cwd=evaluator_config.get('cwd'),
            env=dict(evaluator_config.get('env', {})),
            timeout_s=float(evaluator_config.get('timeout_s', 600.0)),
        )
    if evaluator_type == 'universe_official':
        input_payload = {
            'response': final_answer,
            'evaluators': evaluator_config['evaluators'],
            'context_env': evaluator_config.get('context_env', {}),
        }
        return await UniverseRuntimeEvaluator().evaluate(
            runtime=runtime,
            helper_script=UNIVERSE_EVAL_HELPER,
            input_payload=input_payload,
            work_dir=_case_root(task_input),
            timeout_s=float(evaluator_config.get('timeout_s', 300.0)),
        )
    raise ValueError(f'Unknown evaluator type: {evaluator_type}')


_CLEANUP_STALE_GCP_LOGS_SCRIPT = r"""
import json, os, sys
bucket_file = os.path.join(os.environ['CASE_ROOT'], 'groundtruth_workspace', 'log_bucket_name.txt')
if not os.path.exists(bucket_file):
    sys.exit(0)
with open(bucket_file) as f:
    correct_name = f.read().strip()
if not correct_name:
    sys.exit(0)
creds_path = os.path.join(os.environ['CASE_ROOT'], 'configs', 'gcp-service_account.keys.json')
if not os.path.exists(creds_path):
    sys.exit(0)
with open(creds_path) as f:
    project_id = json.load(f).get('project_id')
from google.oauth2 import service_account
from google.cloud import logging as glogging
credentials = service_account.Credentials.from_service_account_file(creds_path)
client = glogging.Client(project=project_id, credentials=credentials)
prefix = correct_name.rsplit('-', 1)[0]
seen = set()
for entry in client.list_entries(
    filter_=f'logName=~"projects/{project_id}/logs/{prefix}" AND NOT logName="projects/{project_id}/logs/{correct_name}"',
    page_size=50,
):
    log_name = entry.log_name.split('/')[-1] if '/' in entry.log_name else entry.log_name
    if log_name != correct_name and log_name.startswith(prefix) and log_name not in seen:
        seen.add(log_name)
        try:
            client.logger(log_name).delete()
            print(f'Deleted stale log: {log_name}')
        except Exception as e:
            print(f'Failed to delete {log_name}: {e}')
"""


async def _cleanup_stale_gcp_logs(runtime, *, case_root: str) -> None:
    """Delete stale GCP log entries so that agent's list_logs only returns the current bucket name."""
    script = _CLEANUP_STALE_GCP_LOGS_SCRIPT.strip()
    cmd = (
        f'CASE_ROOT={shlex.quote(case_root)}'
        ' GOOGLE_CLOUD_ENABLE_REST_TRANSPORT=true'
        ' GOOGLE_CLOUD_DISABLE_GRPC=true'
        ' SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt'
        f' /opt/mcp-benchmarks/datasets/toolathlon/.venv/bin/python -c {shlex.quote(script)}'
    )
    try:
        result = await runtime.run_command(command=cmd, timeout=30)
        if result.stdout:
            logger.debug('GCP log cleanup: %s', result.stdout.strip())
        if result.return_code != 0 and result.stderr:
            logger.warning('GCP log cleanup stderr: %s', result.stderr.strip()[:300])
    except Exception as exc:
        logger.warning('GCP log cleanup failed (non-fatal): %s', exc)


async def _refresh_gcp_allowed_buckets(runtime, *, case_root: str, mcp_servers: list[dict]) -> None:
    """Read updated bucket_name.txt and log_bucket_name.txt from the container and patch mcp_servers.

    After preprocess, these files contain the *current* active bucket names,
    which may differ from the stale values baked in at generator time.
    Patches both ``--allowed-buckets`` (storage) and ``--allowed-log-buckets`` (logging).
    """
    gw = case_root + '/groundtruth_workspace'
    refresh_map: dict[str, str] = {}
    for filename, flag in [
        ('bucket_name.txt', '--allowed-buckets'),
        ('log_bucket_name.txt', '--allowed-log-buckets'),
    ]:
        cmd = f'cat {shlex.quote(gw + "/" + filename)} 2>/dev/null'
        try:
            result = await runtime.run_command(command=cmd, timeout=10)
        except Exception:
            continue
        if result.return_code != 0 or not result.stdout:
            continue
        value = result.stdout.strip()
        if value:
            refresh_map[flag] = value

    if not refresh_map:
        return

    for spec in mcp_servers:
        spec_client = spec.get('client') or spec.get('params') or {}
        if spec.get('name') != 'google-cloud':
            continue
        args = spec_client.get('args', [])
        # After the sed-patch rewrite in datasets.py, the google-cloud server uses
        # bash -c "... --allowed-buckets <old_value> ... --allowed-log-buckets <old_value> ..."
        # as a single string in args[1].
        for flag, new_value in refresh_map.items():
            for i, arg in enumerate(args):
                if flag in str(arg):
                    args[i] = re.sub(
                        rf'({re.escape(flag)}\s+)\S+',
                        rf'\1{new_value}',
                        args[i],
                    )
                    logger.debug('Refreshed %s to %s', flag, new_value)
        return


@task_registry.register('mcp_benchmark')
async def mcp_benchmark_task(task_input: MCPBenchmarkTaskInput) -> MCPBenchmarkTaskOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime(task_input.runtime_name)
    case_root, agent_workspace = await _prepare_case(runtime, task_input)
    service_replacements: dict[str, str] = {}
    resolved_mcp_servers = copy.deepcopy(task_input.mcp_servers)
    resolved_input_files = _resolve_input_files(task_input.input_files, service_replacements)
    resolved_case_text_replacements = dict(task_input.case_text_replacements)
    resolved_bootstraps = _resolve_commands(task_input.bootstraps, service_replacements)
    resolved_preprocess = _resolve_command(task_input.preprocess, service_replacements)
    resolved_evaluator = _resolve_evaluator(task_input.evaluator, service_replacements)
    resolved_context_env = copy.deepcopy(task_input.context_env)

    final_answer = ''
    eval_result = None
    agent_output = None
    tool_names: list[str] = []
    cleanup_results: list[dict[str, Any]] = []
    task_error: Exception | None = None
    ctx.tool_state.pop(BENCHMARK_CALL_RECORDS_KEY, None)
    launch_time: str | None = _toolathlon_launch_time() if task_input.dataset == BenchmarkDataset.TOOLATHLON else None
    res_log_file: str | None = None

    try:
        service_replacements = await _setup_services(task_input)
        resolved_mcp_servers = _replace_placeholders(copy.deepcopy(task_input.mcp_servers), service_replacements)
        resolved_input_files = _resolve_input_files(task_input.input_files, service_replacements)
        resolved_case_text_replacements = _replace_placeholders(
            copy.deepcopy(task_input.case_text_replacements), service_replacements
        )
        resolved_bootstraps = _resolve_commands(task_input.bootstraps, service_replacements)
        resolved_preprocess = _resolve_command(task_input.preprocess, service_replacements)
        resolved_evaluator = _resolve_evaluator(task_input.evaluator, service_replacements)
        resolved_context_env = _replace_placeholders(copy.deepcopy(task_input.context_env), service_replacements)

        await _upload_input_files(runtime, case_root=case_root, input_files=resolved_input_files)
        await _rewrite_case_text_files(
            runtime,
            case_root=case_root,
            replacements=resolved_case_text_replacements,
        )

        if resolved_bootstraps:
            await _run_bootstraps(runtime, resolved_bootstraps)

        if resolved_preprocess is not None:
            preprocess_command = resolved_preprocess.command
            if task_input.dataset == BenchmarkDataset.TOOLATHLON and launch_time is not None:
                preprocess_command = _append_cli_arg(preprocess_command, '--agent_workspace', agent_workspace)
                preprocess_command = _append_cli_arg(preprocess_command, '--launch_time', launch_time)
            await _run_checked_command(
                runtime,
                command=preprocess_command,
                cwd=resolved_preprocess.cwd,
                env=resolved_preprocess.env,
                timeout=resolved_preprocess.timeout_s,
            )

        # After preprocess, clean stale GCP logs so agent's list_logs only sees the current bucket.
        # Applies to academic-warning / ab-testing / live-transactions which use log_bucket_name.txt.
        if task_input.dataset == BenchmarkDataset.TOOLATHLON:
            await _cleanup_stale_gcp_logs(runtime, case_root=case_root)
            await _refresh_gcp_allowed_buckets(runtime, case_root=case_root, mcp_servers=resolved_mcp_servers)

        if task_input.dataset == BenchmarkDataset.TOOLATHLON and _contains_template_placeholders(resolved_mcp_servers):
            runtime_template_values = await _load_toolathlon_runtime_template_values(runtime, case_root=case_root)
            runtime_template_values = _replace_placeholders(runtime_template_values, service_replacements)
            resolved_mcp_servers = render_value(resolved_mcp_servers, runtime_template_values)
            resolved_mcp_servers = _replace_placeholders(resolved_mcp_servers, service_replacements)

        _ensure_no_template_placeholders(resolved_mcp_servers)

        # Ensure per-call timeout is set on each MCP server to prevent indefinite hangs.
        for server in resolved_mcp_servers:
            client = server.get('client', server)
            client.setdefault('timeout_s', 300.0)

        loaded_tools = await ctx.load_mcp_tools(resolved_mcp_servers, clear_existing=True)
        loaded_tools.update({
            alias: ctx._mcp_manager.tools[alias]
            for alias in _register_toolathlon_google_map_aliases(ctx, dataset=task_input.dataset)
        })
        loaded_tool_names = sorted(loaded_tools)
        logger.debug('Loaded MCP tools for %s/%s: %s', task_input.dataset.value, task_input.case_id, loaded_tool_names)
        await _run_startup_tool_setup(task_input, loaded_tool_names)

        if task_input.enabled_tools:
            allowed = set(task_input.enabled_tools)
            tool_names = [name for name in loaded_tool_names if name in allowed]
        else:
            tool_names = loaded_tool_names

        if not tool_names:
            raise RuntimeError(f'No tools enabled for {task_input.dataset.value}/{task_input.case_id}')

        # Inject local-python_execute for Toolathlon so agent can run arbitrary Python in container
        if task_input.dataset == BenchmarkDataset.TOOLATHLON:
            ctx.tool_state[PYTHON_EXECUTE_WORKSPACE_KEY] = agent_workspace
            tool_names.append('local-python_execute')

        agent_kwargs: dict[str, Any] = {
            'user_prompt': _build_agent_prompt(task_input),
            'tools': tool_names,
            'max_iterations': task_input.max_iterations,
            'parallel_tool_calls': True,
        }
        if task_input.tool_output_truncation is not None:
            agent_kwargs['tool_output_truncation'] = task_input.tool_output_truncation
        agent_kwargs['system_prompt'] = _resolve_system_prompt(task_input)

        agent_output = await task_input.run_agent(**agent_kwargs)
        final_answer = getattr(agent_output, 'final_answer', '') or ''
        if task_input.dataset == BenchmarkDataset.TOOLATHLON and launch_time is not None:
            res_log_file = await _write_toolathlon_eval_log(
                runtime,
                case_root=case_root,
                agent_output=agent_output,
                final_answer=final_answer,
                launch_time=launch_time,
            )
        eval_result = await _evaluate(
            task_input,
            evaluator=resolved_evaluator,
            runtime=runtime,
            final_answer=final_answer,
            agent_workspace=agent_workspace,
            launch_time=launch_time,
            res_log_file=res_log_file,
        )
    except Exception as exc:
        task_error = exc
    finally:
        if task_input.cleanups:
            cleanup_results = await run_cleanups(
                cleanups=task_input.cleanups,
                tool_calls=ctx.tool_state.get(BENCHMARK_CALL_RECORDS_KEY, []),
                env=resolved_context_env,
            )
        ctx.tool_state.pop(BENCHMARK_CALL_RECORDS_KEY, None)

    if task_error is not None:
        raise task_error
    assert eval_result is not None
    assert agent_output is not None
    details = dict(eval_result.details)
    if cleanup_results:
        details['cleanup'] = cleanup_results

    return MCPBenchmarkTaskOutput(
        dataset=task_input.dataset,
        case_id=task_input.case_id,
        server_mode=task_input.server_mode,
        score=eval_result.score,
        passed=eval_result.passed,
        eval_details=details,
        main_trajectory=getattr(agent_output, 'main_trajectory', []),
        agent_final_answer=final_answer,
        loaded_tools=tool_names,
        agent_workspace=agent_workspace,
    )
