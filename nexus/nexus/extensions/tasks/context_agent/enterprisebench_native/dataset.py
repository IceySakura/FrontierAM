"""Turn a Syntara enterprise-office delivery into a Nexus-native dataset.

The delivery is 300 task directories carrying binary office documents and
per-service seed data — 1.8 GB that no JSONL row can hold. The dataset service
is the channel for that: each task becomes one item whose ``files/`` directory
is published as a COS blob, and the rollout pulls it back with
``ctx.task_files.download()`` instead of finding it on a mount that only exists
on the machine that built the dataset.

That is also what removes the per-task Docker image. A task's image differed
from its five siblings only by ``COPY`` layers of exactly this data plus two
``ENV`` lines (verified across all 300: ``agent-requirements.txt`` is
byte-identical everywhere, and two Dockerfiles taken from different worlds
differ only in ``WORLDBENCH_TOOL_SETS`` and ``ANCHOR_DATETIME``). Moving the
data into task_files and the two variables into ``boot_env`` leaves six shared
base images instead of three hundred task images.

What each item carries:

``files/``
    Only the runtime payload: the Judge prompt template, grading-context helper,
    optional oracle solution, and environment seed files. ``task.toml``,
    ``instruction.md``, authored ``judge.toml``, ``opencode.json``, and the
    authored wrapper are removed from the generated Taskfiles after conversion.

``body``
    The normalized Nexus ``RunInput``. The builder parses both ``task.toml``
    and ``tests/judge.toml`` once and stores their executable meaning here.

``metadata``
    The routing and reporting keys — company, world, tool sets, rubric size.
    Read by the eval class for per-slice reporting, and by a human deciding
    which slice to re-run.

Usage::

    python -m nexus.extensions.tasks.context_agent.enterprisebench_native.dataset \\
        --source /path/to/tencent_aug24_delivery \\
        --out /path/to/syntara_office_dataset \\
        --tasks /path/to/eval75_tasks.txt \\
        --image-prefix mirrors.tencent.com/hunyuan_yanguan/syntara \\
        --image-tag v1

Then publish it::

    nexus dataset publish <out> <namespace>/<name>@<version> --admin <user>
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import sys

from pathlib import Path
from typing import Any

import yaml

from nexus.extensions.tasks.context_agent.enterprisebench_native.verifier_config import (
    normalize_judge_config,
)


DEFAULT_AGENT_NAME = 'context_agent_general'
DEFAULT_CORE_TOOL_SET = 'core'


# The compose command is a one-line YAML flow sequence in every task of this
# delivery. Parsed with a regex rather than a YAML dependency, and validated
# against the service count so a shape change fails loudly instead of silently
# booting the wrong world.
_COMMAND_RE = re.compile(r'^\s*command:\s*\[(.+)\]\s*$', re.MULTILINE)
_SERVICE_RE = re.compile(r'^  ([A-Za-z0-9_-]+):\s*$', re.MULTILINE)
_FROM_RE = re.compile(r'^FROM\s+(\S+)', re.MULTILINE)
_ENV_TOOLSETS_RE = re.compile(r'WORLDBENCH_TOOL_SETS="([^"]*)"')
_ENV_ANCHOR_RE = re.compile(r'ANCHOR_DATETIME="([^"]*)"')

# The MCP proxy's port, moved off the delivery's own 8000.
#
# 8000 is not available in a Nexus sandbox: the runtime server binds it
# (deployment/runtime/start.sh DEFAULT_PORT=8000) and it gets there first,
# both on gongfeng and under the docker provider's default runtime_port. The
# proxy then never comes up, and because the runtime server answers /mcp with
# a 404 the failure reads as an MCP session that connects and is "terminated
# immediately" rather than as a port conflict. Nothing in the delivery pins
# 8000 -- it is a `start.sh --port` argument -- so the dataset moves it.
MCP_PORT = 8900

# What the delivery itself uses, and therefore what has to be rewritten.
DELIVERY_MCP_PORT = 8000

GRADED_WORKDIR = '/workdir'

# Where the task's own data lands, and where the world expects to read it.
#
# The task Dockerfile ``COPY``s each of these directories to TWO destinations —
# a live one the agent and the world mutate, and a pristine one kept for
# diffing at grade time. We upload only the pristine pair and let the boot
# command clone it, so a 48 MB task crosses the runtime channel once instead
# of twice.
PRISTINE_DATA_DIR = '/initial_data'
PRISTINE_WORKSPACE_DIR = '/initial_workspace'
LIVE_DATA_DIR = '/data'
SEED_COMMAND = (
    f'mkdir -p {PRISTINE_WORKSPACE_DIR} {LIVE_DATA_DIR} {GRADED_WORKDIR}'
    f' && cp -a {PRISTINE_DATA_DIR}/. {LIVE_DATA_DIR}/'
    f' && cp -a {PRISTINE_WORKSPACE_DIR}/. {GRADED_WORKDIR}/'
)

# A real MCP ``initialize`` handshake. The weaker probes are both wrong here:
# ``curl -sf`` never succeeds (a header-less GET to a streamable-http endpoint
# answers 406, and ``-f`` turns that into exit 22), and plain ``curl -s``
# accepts ANY http response — including the Nexus runtime server's own 404,
# which is how a five-second "ready" turned into an immediately terminated MCP
# session. Only a handshake proves the right server owns the port.
_INIT_PAYLOAD = (
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{'
    '"protocolVersion":"2025-06-18","capabilities":{},'
    '"clientInfo":{"name":"nexus-healthcheck","version":"0"}}}'
)


def mcp_url(port: int) -> str:
    return f'http://localhost:{port}/mcp'


def healthcheck_command(port: int) -> str:
    return (
        f'curl -s -X POST {mcp_url(port)} '
        f"-H 'Content-Type: application/json' "
        f"-H 'Accept: application/json, text/event-stream' "
        f"-d '{_INIT_PAYLOAD}' | grep -q '\"result\"'"
    )


def retarget_port(argv: list[str], port: int) -> list[str]:
    """Rewrite ``start.sh``'s ``--port`` argument, asserting it was there."""
    try:
        index = argv.index('--port')
    except ValueError as exc:
        raise ValueError(f'compose command has no --port to retarget: {argv}') from exc
    if index + 1 >= len(argv):
        raise ValueError(f'compose command ends after --port: {argv}')
    return [*argv[:index], '--port', str(port), *argv[index + 2 :]]


def parse_compose_command(compose_path: Path) -> list[str]:
    """Lift the single service's ``command`` out of a compose file.

    Raises when there is not exactly one service: everything downstream (one
    container, one boot command) depends on that being true.
    """
    text = compose_path.read_text(encoding='utf-8')
    services = _SERVICE_RE.findall(text)
    if len(services) != 1:
        raise ValueError(f'{compose_path}: expected exactly 1 compose service, found {services or "none"}')
    match = _COMMAND_RE.search(text)
    if not match:
        raise ValueError(f'{compose_path}: no single-line `command: [...]` found')
    argv = [item.strip().strip('"\'') for item in match.group(1).split(',')]
    argv = [part for part in argv if part]
    if not argv:
        raise ValueError(f'{compose_path}: empty command')
    return argv


def read_world(dockerfile: Path) -> str:
    """Return the base world (``suitecrm``, ``appflowy``, ...) a task builds on."""
    match = _FROM_RE.search(dockerfile.read_text(encoding='utf-8'))
    if not match:
        raise ValueError(f'{dockerfile}: no FROM line')
    # ``FROM syntara-suitecrm`` -> ``suitecrm``
    return match.group(1).split('-', 1)[-1]


def read_env(dockerfile: Path) -> tuple[str, str]:
    """Return ``(tool_sets, anchor_datetime)`` — the only per-task ENV values."""
    text = dockerfile.read_text(encoding='utf-8')
    tool_sets = _ENV_TOOLSETS_RE.search(text)
    anchor = _ENV_ANCHOR_RE.search(text)
    return (tool_sets.group(1) if tool_sets else ''), (anchor.group(1) if anchor else '')


def filter_tool_sets(tool_sets: str, disabled: set[str]) -> str:
    """Remove disabled world tool-set families while preserving separator style.

    A disabled name removes both the exact set and its underscore-prefixed
    variants. In particular, disabling ``core`` removes the delivery's actual
    ``core_read``, ``core_write``, and ``core_read_multimodal`` sets.
    """
    disabled = {tool_set.lower() for tool_set in disabled}
    enabled = [
        tool_set
        for tool_set in re.split(r'[\s,]+', tool_sets.strip())
        if not any(
            tool_set.lower() == disabled_set or tool_set.lower().startswith(f'{disabled_set}_')
            for disabled_set in disabled
        )
    ]
    separator = ',' if ',' in tool_sets else ' '
    return separator.join(enabled)


def filter_boot_tool_sets(argv: list[str], disabled: set[str]) -> list[str]:
    """Remove disabled tool-set families from a ``start.sh --tool-sets`` argv."""
    try:
        start = argv.index('--tool-sets') + 1
    except ValueError:
        return list(argv)

    end = start
    while end < len(argv) and not argv[end].startswith('--'):
        end += 1
    enabled = filter_tool_sets(' '.join(argv[start:end]), disabled).split()
    return [*argv[:start], *enabled, *argv[end:]]


def load_task_toml(task_toml: Path) -> dict[str, Any]:
    import tomllib

    with open(task_toml, 'rb') as fh:
        config = tomllib.load(fh)
    allowed_top = {'schema_version', 'task', 'agent', 'environment', 'verifier'}
    unexpected = sorted(set(config) - allowed_top)
    if unexpected:
        raise ValueError(f'{task_toml}: unsupported top-level sections: {unexpected}')
    environment = config.get('environment') or {}
    unexpected_environment = sorted(set(environment) - {'os', 'cpus', 'memory_mb', 'env', 'mcp_servers'})
    if unexpected_environment:
        raise ValueError(f'{task_toml}: unsupported environment fields: {unexpected_environment}')
    if str(environment.get('os') or 'linux').lower() != 'linux':
        raise ValueError(f'{task_toml}: EnterpriseBench Native supports Linux tasks only')
    verifier = config.get('verifier') or {}
    unexpected_verifier = sorted(set(verifier) - {'timeout_sec', 'env'})
    if unexpected_verifier:
        raise ValueError(f'{task_toml}: unsupported verifier fields: {unexpected_verifier}')
    if verifier.get('collect'):
        raise ValueError(f'{task_toml}: verifier.collect is not supported')
    unexpected_agent = sorted(set(config.get('agent') or {}) - {'timeout_sec'})
    if unexpected_agent:
        raise ValueError(f'{task_toml}: unsupported agent fields: {unexpected_agent}')
    return config


def normalize_mcp_servers(environment: dict[str, Any], *, port: int) -> list[dict[str, Any]]:
    """Convert authored MCP entries to Nexus's Runtime-local server format."""
    servers = environment.get('mcp_servers') or []
    if not isinstance(servers, list) or not servers:
        raise ValueError('task.toml environment.mcp_servers must be a non-empty list')
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(servers):
        if not isinstance(raw, dict):
            raise TypeError(f'environment.mcp_servers[{index}] must be a mapping')
        name = str(raw.get('name') or '').strip()
        transport = str(raw.get('transport') or 'streamable-http').strip()
        url = str(raw.get('url') or '').strip()
        if not name or not url:
            raise ValueError(f'environment.mcp_servers[{index}] requires name and url')
        if transport not in {'streamable-http', 'http'}:
            raise ValueError(f'Unsupported EnterpriseBench MCP transport: {transport!r}')
        url = url.replace(f'localhost:{DELIVERY_MCP_PORT}', f'localhost:{port}')
        normalized.append({
            'name': name,
            'runtime_name': 'main',
            'url': url,
            'nexus_managed': False,
        })
    return normalized


def load_judge_config(task_dir: Path) -> dict[str, Any]:
    """Parse the authored RewardKit configuration into a Nexus item field."""
    import tomllib

    judge_toml = task_dir / 'tests' / 'judge.toml'
    if not judge_toml.is_file():
        raise FileNotFoundError(f'{task_dir.name}: missing tests/judge.toml')
    try:
        with open(judge_toml, 'rb') as handle:
            authored = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f'{judge_toml}: invalid TOML: {exc}') from exc
    return normalize_judge_config({
        'judge': authored.get('judge') or {},
        'criteria': authored.get('criterion') or [],
        'scoring': authored.get('scoring') or {},
    })


def prepare_files_tree(files_dir: Path) -> None:
    """Remove converted config files, leaving only data needed inside the Runtime."""
    tests_dir = files_dir / 'tests'
    prompt = tests_dir / 'prompt_template.md'
    if not prompt.is_file():
        raise FileNotFoundError(f'{files_dir}: missing tests/prompt_template.md')
    fetch_context = tests_dir / 'fetch_grading_context.py'
    if not fetch_context.is_file():
        raise FileNotFoundError(f'{files_dir}: missing tests/fetch_grading_context.py')
    # Runtime execution replaces these from Nexus-owned sources. Removing the
    # authored copies makes accidental fallback impossible. Their source copies
    # remain available in the original delivery for auditing.
    for obsolete in ('judge.toml', 'opencode.json', 'rewardkit_opencode_agent.py', 'test.sh'):
        path = tests_dir / obsolete
        if path.exists():
            path.unlink()
    task_toml = files_dir / 'task.toml'
    if task_toml.exists():
        task_toml.unlink()
    instruction = files_dir / 'instruction.md'
    if instruction.exists():
        instruction.unlink()


def build_item(
    task_id: str,
    task_dir: Path,
    *,
    image: str,
    agent_name: str,
    task_name: str,
    max_tool_rounds: int,
    cpu: int,
    memory_gb: int,
    port: int,
    disabled_world_tool_sets: set[str] | None = None,
) -> dict[str, Any]:
    """Render one normalized Nexus item from an authored EnterpriseBench task."""
    environment = task_dir / 'environment'
    config = load_task_toml(task_dir / 'task.toml')
    docker_tool_sets, anchor = read_env(environment / 'Dockerfile')
    boot_argv = retarget_port(parse_compose_command(environment / 'docker-compose.yaml'), port)

    env_block = (config.get('environment') or {}).get('env') or {}
    original_tool_sets = docker_tool_sets or str(env_block.get('WORLDBENCH_TOOL_SETS', ''))
    disabled_world_tool_sets = {DEFAULT_CORE_TOOL_SET, *(disabled_world_tool_sets or set())}
    tool_sets = filter_tool_sets(original_tool_sets, disabled_world_tool_sets)
    boot_argv = filter_boot_tool_sets(boot_argv, disabled_world_tool_sets)
    resources = config.get('environment') or {}
    mcp_servers = normalize_mcp_servers(resources, port=port)
    instruction_path = task_dir / 'instruction.md'
    if not instruction_path.is_file():
        raise FileNotFoundError(f'{task_id}: missing instruction.md')
    instruction = instruction_path.read_text(encoding='utf-8')
    if not instruction.strip():
        raise ValueError(f'{task_id}: instruction.md is empty')
    judge_config = load_judge_config(task_dir)
    n_criteria = len(judge_config['criteria'])
    task_meta = config.get('task') or {}
    task_config: dict[str, Any] = {
        'instance_id': task_id,
        'benchmark_task_name': str(task_meta.get('name') or task_id),
        'task_description': str(task_meta.get('description') or ''),
        # Parsed at dataset-build time. Runtime execution never reads task.toml
        # or tests/judge.toml as configuration sources.
        'instruction': instruction,
        # Empty on purpose: the task directory travels as this item's files/
        # blob and is materialized by the Native task. A path here would only
        # resolve on the machine that built the dataset.
        'task_dir': '',
        # The graded directory. Nexus's verifier wrapper runs RewardKit with
        # ``--workspace /workdir``; left at the task's ``/app`` default the
        # agent works in one directory while the verifier reads another, which
        # scores 0 while looking perfectly healthy.
        'workdir': GRADED_WORKDIR,
        # Replaces the task Dockerfile's COPY layers. The uploads land the
        # pristine copies; the boot chain clones them into the live paths
        # BEFORE start.sh runs, which is load-bearing — the world restores
        # ``mysql.tar.zst`` and friends out of /data at startup, so seeding
        # afterwards would boot an empty company.
        'environment_uploads': [
            {'source': 'environment/initial_external_services', 'target': PRISTINE_DATA_DIR},
            {
                'source': 'environment/initial_workspace',
                'target': PRISTINE_WORKSPACE_DIR,
                # TaskFiles ZIPs contain files, not empty directory entries.
                # Preserve a valid empty workspace after materialization.
                'ensure_directory': True,
            },
        ],
        # The EnterpriseBench Native task enumerates Runtime-preloaded files and
        # exposes them through the standard Context Agent ./workspace manifest.
        'expose_workdir_files': True,
        'workdir_context_max_files': 500,
        'judge_client_name': 'judge',
        'judge_config': judge_config,
        'judge_mcp_url': mcp_url(port),
        'network_isolation_enabled': True,
        'verifier_network_isolation_enabled': True,
        'boot_command': f'{SEED_COMMAND} && exec ' + ' '.join(shlex.quote(part) for part in boot_argv),
        'boot_env': {
            'INPUTDIR': env_block.get('INPUTDIR', LIVE_DATA_DIR),
            'OUTPUTDIR': env_block.get('OUTPUTDIR', LIVE_DATA_DIR),
            'WORLDBENCH_TOOL_SETS': tool_sets,
            'ANCHOR_DATETIME': anchor,
        },
        'healthcheck_command': healthcheck_command(port),
        # Left empty on purpose: a GET-based fallback probe
        # from a URL, and a GET cannot tell the MCP proxy apart from anything
        # else that answers on this port.
        'healthcheck_url': '',
        'run_oracle': False,
    }

    return {
        'id': task_id,
        'files': {'type': 'directory', 'path': 'files'},
        'body': {
            'imports': [
                'nexus.extensions',
                'nexus.extensions.tasks.context_agent.enterprisebench_native.task',
            ],
            'task_name': task_name,
            'task_config': task_config,
            'agent_name': agent_name,
            'agent_config': {
                'max_tool_rounds': max_tool_rounds,
                # The common curl/bash/python/file tools come from Nexus's
                # context_agent_general implementation. The Docker world's
                # Core tool set is removed from WORLDBENCH_TOOL_SETS above, so
                # MCP contributes application tools only.
                'enabled_tools': None,
                # ~240 MCP tools, and Azure rejects a tools array longer than
                # 128, so they cannot all be sent. Deferring puts them behind a
                # search tool.
                'tool_search': {'enabled': True, 'defer_all_mcp': True},
                # runtime_name is the load-bearing part: Nexus starts a proxy
                # INSIDE this runtime, so ``localhost`` resolves in the sandbox
                # rather than on the host running Nexus.
                'mcp_servers': mcp_servers,
            },
            'runtime_configs': {
                'main': {
                    'image': image,
                    'working_dir': GRADED_WORKDIR,
                    # Runtime-side context_agent_general tools come from the
                    # current checkout rather than relying on the base runtime
                    # image to contain this exact Nexus revision.
                    'worker_venv': {'modules': []},
                    'resource_spec': {
                        'cpu': int(resources.get('cpus') or cpu),
                        # Enterprise worlds run the application stack, document
                        # tooling, Agent worker, and OpenCode Judge together.
                        # Do not let a smaller task.toml hint undercut the
                        # evaluator's 10 GB operational default.
                        'memory': max(memory_gb, int(resources.get('memory_mb') or 0) // 1024),
                    },
                },
            },
        },
        'metadata': {
            'task_id': task_id,
            'dataset_name': 'SyntaraOffice',
            'company': task_id.rsplit('_', 1)[0],
            'world': read_world(environment / 'Dockerfile'),
            'tool_sets': tool_sets,
            'anchor_datetime': anchor,
            'n_criteria': n_criteria,
            'judge_harness': 'opencode',
        },
    }


def select_tasks(tasks_root: Path, task_list: Path | None, limit: int | None) -> list[str]:
    available = sorted(p.name for p in tasks_root.iterdir() if p.is_dir())
    if task_list is not None:
        wanted = [
            line.strip()
            for line in task_list.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')
        ]
        missing = [name for name in wanted if name not in set(available)]
        if missing:
            raise SystemExit(f'{len(missing)} task(s) from {task_list} are not in {tasks_root}: {missing[:5]}')
        selected = wanted
    else:
        selected = available
    return selected[:limit] if limit else selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='Delivery root (the directory holding tasks/).')
    parser.add_argument('--out', required=True, help='Dataset directory to write.')
    parser.add_argument('--tasks', default=None, help='File listing task ids, one per line (e.g. eval75_tasks.txt).')
    parser.add_argument('--limit', type=int, default=None, help='Keep only the first N selected tasks.')
    parser.add_argument('--name', default='syntara-office', help='Dataset name written into dataset.yaml.')
    parser.add_argument('--split', default='test')
    parser.add_argument(
        '--image-prefix',
        default='mirrors.tencent.com/hunyuan_yanguan/syntara',
        help='Registry prefix; the world name is appended, e.g. <prefix>-suitecrm:<tag>.',
    )
    parser.add_argument('--image-tag', default='v1')
    parser.add_argument('--task-name', default='context_agent_enterprisebench_native_task')
    parser.add_argument('--agent-name', default=DEFAULT_AGENT_NAME)
    parser.add_argument(
        '--disable-world-tool-set',
        action='append',
        default=[],
        help=(
            'Remove this tool set from WORLDBENCH_TOOL_SETS before booting the world. '
            'Repeat to disable more sets. The core set is always disabled.'
        ),
    )
    parser.add_argument('--max-tool-rounds', type=int, default=999)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--memory', type=int, default=10, help='Memory in GB.')
    parser.add_argument(
        '--mcp-port',
        type=int,
        default=MCP_PORT,
        help=(
            f'Port the world MCP proxy binds inside the sandbox (default {MCP_PORT}). '
            f'Must not be {DELIVERY_MCP_PORT} — the Nexus runtime server owns that one.'
        ),
    )
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help=(
            'Point each item at the delivery in place instead of copying 1.8 GB. '
            'Runtime configuration still comes from the generated Nexus item.'
        ),
    )
    parser.add_argument('--force', action='store_true', help='Overwrite a non-empty --out.')
    args = parser.parse_args()

    if args.mcp_port == DELIVERY_MCP_PORT:
        raise SystemExit(
            f'--mcp-port {DELIVERY_MCP_PORT} collides with the Nexus runtime server, which binds it '
            'first; the proxy would never come up and the agent would dial the runtime server instead.'
        )

    source = Path(args.source).expanduser().resolve()
    tasks_root = source / 'tasks'
    if not tasks_root.is_dir():
        raise SystemExit(f'no tasks/ under {source}')

    out = Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f'{out} is not empty (pass --force to overwrite)')
        shutil.rmtree(out)

    selected = select_tasks(tasks_root, Path(args.tasks).expanduser() if args.tasks else None, args.limit)
    if not selected:
        raise SystemExit('no tasks selected')

    item_paths: list[str] = []
    worlds: dict[str, int] = {}
    for index, task_id in enumerate(selected, 1):
        task_dir = tasks_root / task_id
        world = read_world(task_dir / 'environment' / 'Dockerfile')
        worlds[world] = worlds.get(world, 0) + 1
        image = f'{args.image_prefix}-{world}:{args.image_tag}'

        item = build_item(
            task_id,
            task_dir,
            image=image,
            agent_name=args.agent_name,
            task_name=args.task_name,
            max_tool_rounds=args.max_tool_rounds,
            cpu=args.cpu,
            memory_gb=args.memory,
            port=args.mcp_port,
            disabled_world_tool_sets=set(args.disable_world_tool_set or []),
        )

        item_dir = out / 'items' / task_id
        item_dir.mkdir(parents=True, exist_ok=True)
        if args.no_copy:
            # Relative so the dataset directory stays movable alongside the
            # delivery; resolved against item.yaml's own parent by
            # DatasetItem.local_task_files_path().
            import os

            item['files']['path'] = os.path.relpath(task_dir, item_dir)
        else:
            # copy_function=shutil.copy, not the default copy2: copy2 replays
            # metadata, and on the shared filesystem this delivery lives on
            # that raises EPERM for some files. Only the bytes matter.
            files_dir = item_dir / 'files'
            shutil.copytree(task_dir, files_dir, dirs_exist_ok=True, copy_function=shutil.copy)
            prepare_files_tree(files_dir)

        with open(item_dir / 'item.yaml', 'w', encoding='utf-8') as fh:
            yaml.safe_dump(item, fh, sort_keys=False, allow_unicode=True)
        item_paths.append(f'items/{task_id}/item.yaml')
        print(f'[{index}/{len(selected)}] {task_id} ({world})', flush=True)

    dataset_yaml = {
        'schema_version': 1,
        'name': args.name,
        'authors': [],
        'maintainers': [],
        'default_split': args.split,
        'splits': {args.split: {'items': item_paths}},
    }
    with open(out / 'dataset.yaml', 'w', encoding='utf-8') as fh:
        yaml.safe_dump(dataset_yaml, fh, sort_keys=False, allow_unicode=True)

    print()
    print(f'{out}: {len(item_paths)} items')
    for world in sorted(worlds):
        print(f'  {args.image_prefix}-{world}:{args.image_tag}  x{worlds[world]}')
    print()
    print('Images to build and push (one per world, shared by every task in it):')
    for world in sorted(worlds):
        print(f'  {args.image_prefix}-{world}:{args.image_tag}')
    print()
    print('Publish with:')
    print(f'  nexus dataset publish {out} <namespace>/{args.name}@<version> --admin <user>')


if __name__ == '__main__':
    sys.exit(main())
