#!/usr/bin/env python3
"""SWE-Any rollout generator for ``nexus run-batch``.

This is an intentionally independent copy of the production SWE-Cosmos rollout generator:
SWE-Any keeps the same agent/runtime/cache behavior without importing another dataset's
generator. It reads Export's ``swe_any_instances.jsonl`` and dispatches every row through
``swe_task``; ``dataset_type='swe-any'`` selects the SWE-Any dataset handler.

The host LLM backend is selected by OPENAI_CLIENT_NAME: ``litellm`` (AWS/bedrock,
the default), ``vertex_anthropic`` (GCP passthrough), or ``openai`` (for example,
Taiji HY3-offline). All provider-specific defaults live in the provider env maps.

Usage:
  uv run nexus run-batch \
    -g nexus/extensions/tasks/swe/swe_any/generator.py \
    -d ./outputs/swe_any_rollout.db --max-concurrent 5

Environment variables:
  Most "default" values below are seeded by HARD_CODED_ENV via os.environ.setdefault,
  so an externally exported env always wins. Secrets (API key / token) are never
  hard-coded and must come from env.

  - SWE_DATASET: remote dataset path (e.g., "swe-public/SWE-bench_Verified"). When set, fetches instance list from dataset server and uses remote loading. Takes precedence over SWE_DATASET_PATH.
  - SWE_DATASET_SPLIT: split name for remote dataset (default: empty, uses default split)
  - SWE_DATASET_PATH: path to JSONL dataset (default: ./workspace/swe-bench/swe-bench.jsonl). Only used when SWE_DATASET is not set.
  - SWE_INCLUDE_INSTANCE_IDS: comma-separated whitelist of instance_ids to run (default: all). Each item is a literal instance_id or a path to a .json/.jsonl file from which instance_ids are extracted. Example: "django__django-10097,psf__requests-1724" or "/data/subset.jsonl"
  - SWE_EXCLUDE_INSTANCE_IDS: comma-separated blacklist of instance_ids to skip (default: none). Same item syntax as SWE_INCLUDE_INSTANCE_IDS; applied after the include filter. An instance_id listed in both raises an error.
  - SWE_SKIP: number of instances to skip before starting (default: 0)
  - SWE_LIMIT: max instances to yield (default: no limit)
  - SWE_RUN_ORACLE: "1"/"true" to use golden patch without agent (default: false)
  - SWE_EVAL_TIMEOUT_SECONDS: per-instance eval (grade) timeout in seconds (default: 2000)
  - SWE_WORKER_VENV: "1"/"true" to upload the current nexus wheel into the eval sandbox (default: true). Needed when the host has dataset handlers not baked into the runtime image.
  - SWE_RECORD_TRAJECTORY: "1"/"true" to record agent LLM trajectories via chat_tracker (default: true). Results land in each run's RunOutput.processor_results.chat.trajectories. Set "0" for very large sweeps.
  - SWE_TOOLS_MODE: tools mode for chat_tracker (default: branch_last)
  - SWE_ENABLE_ANTI_HACK_PROMPT: append the shared SWE anti-hack notice (default: false)
  - SWE_ENABLE_REWARD_HACKING_DETECTION: inspect the chat trajectory for answer lookup (default: false; requires SWE_RECORD_TRAJECTORY=true)
  - SWE_ENABLE_ANTI_HACK: relabel detected hacks as CODE_REWARD_HACKING (default: false)
  - SWE_HACK_REWARD_OVERRIDE: optional replacement reward for detected hacks; unset means mask-only
  - SWE_AGENT_NAME: agent name (default: swe_agent; ignored if run_oracle=true)
  - SWE_RUNTIME_BASE_URL: runtime base URL (default: http://hyrl-sandbox.prod.woa.com:8052)
  - SWE_RUNTIME_PROVIDER_NAME: docker/gongfeng/... (default: gongfeng)
  - SWE_RUNTIME_TAG: runtime tag to use (default: current)
  - SWE_RUNTIME_TIMEOUT: per-RPC HTTP timeout floor in seconds for the swe_agent path (default: 14400). The request_timeout is computed as max(agent_timeout, eval_timeout) + 600, where agent_timeout is SWE_RUNTIME_TIMEOUT for swe_agent and CLAUDE_CODE_TIMEOUT for claude_code (so claude_code ignores SWE_RUNTIME_TIMEOUT).
  - SWE_RUNTIME_CPU / SWE_RUNTIME_MEMORY: default runtime resources (default: 8 / 16)
  - SWE_RPC_PROTOCOL: runtime RPC protocol (default: http)
  - SWE_USE_NEW_CACHE_PROXY: "1"/"true" to route through the new cache proxy (default: true)
  - GONGFENG_RUNTIME_TOKEN: authentication token for Gongfeng sandbox API (required for gongfeng provider; falls back to GONGFENG_TOKEN)
  - GONGFENG_PURPOSE: purpose passed to the Gongfeng sandbox API (default: test)
  - SWE_ANY_BATCH_ID: optional Gongfeng batch id
  - OPENAI_CLIENT_NAME: host LLM client — "litellm" (AWS/bedrock, default) or "vertex_anthropic" (GCP passthrough). Picks the provider defaults below.
  - OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL: client config. OPENAI_MODEL is provider-prefixed for litellm (e.g. bedrock/anthropic.claude-opus-4-8) and a bare Anthropic name for vertex (e.g. claude-opus-4-8).
  - SWE_LLM_TIMEOUT / SWE_LLM_MAX_RETRIES: backend LLM client timeout (seconds) and max retries (default: 900 / 15)
  - MAX_TOKENS: max output tokens per request, excluding thinking (default: 16000)
  - SWE_THINKING_ENABLED: "1"/"true" to attach thinking + output_config to thinking requests (default: true).
  - SWE_THINKING_TYPE / SWE_THINKING_EFFORT: thinking type and effort (default: adaptive / xhigh)
  - SWE_THINKING_PRESET: anthropic proxy converter preset (default: bedrock-claude)
  - SWE_CACHE_CONTROL_ENABLED / SWE_CACHE_CONTROL_STRATEGY: Anthropic prompt caching (default: true / claude_code). Enabled when SWE_AGENT_NAME=claude_code.
  - SWE_LOG_LLM_USAGE: "1"/"true" to log token usage / cache hit rate (default: true)
  Note: each instance gets its own client config for claude models, whose api_key carries a per-run `cache_task_id` (32-hex md5 of time + pid + instance_id). The gateway partitions its prompt cache by that id, so it must stay unique per task; sharing one id across instances cross-wires their caches.
  vertex_anthropic only:
  - SWE_ANTHROPIC_VERSION: anthropic_version sent with every request (default: vertex-2023-10-16)
  - SWE_GCP_ENDPOINT: explicit vertex endpoint URL; overrides the base_url-derived endpoint (default: unset)
  - SWE_THINKING_BUDGET_TOKENS: explicit thinking token budget, added to the thinking block when set (default: unset)
  - SWE_DROP_FORCED_TOOL_CHOICE_WITH_THINKING: drop a forced tool_choice while thinking is on, since vertex rejects that combo (default: true)
  - SWE_TEMPERATURE / SWE_TOP_P / SWE_TOP_K: optional sampling params, passed through only when set (default: unset)
  - SWE_STOP_SEQUENCES: optional stop sequences, separated by "||" or "," (default: unset)
  - SWE_GCP_METADATA_JSON: optional JSON object sent as request metadata (default: unset)
  ---
  - CLAUDE_CODE_VERSION / CLAUDE_CODE_BINARY_URL / CLAUDE_CODE_BINARY_CHECKSUM / CLAUDE_CODE_WORKING_DIR / CLAUDE_CODE_TIMEOUT: used when SWE_AGENT_NAME=claude_code
  - ANTHROPIC_MODEL: model id the Claude Code CLI uses internally (default: claude-opus-4-8[1m]); the [1m] suffix only tells the CLI to budget a 1M context window. The host drops this model name and routes every request to OPENAI_MODEL instead (see llm_handler), so it never reaches the backend and is provider-agnostic; claude_code only
  - CLAUDE_CODE_DISALLOWED_TOOLS: comma-separated tools to disable (default: WebSearch,WebFetch); claude_code only
  - CLAUDE_CODE_EXTRA_CLI_ARGS: extra CLI args appended to the Claude Code invocation (shlex-split); claude_code only
  - CLAUDE_CODE_EXTRA_ENV_JSON: JSON object merged into the Claude Code extra_env; claude_code only
  - SWE_NO_THINKING_MODEL / SWE_NO_THINKING_MAX_TOKENS: override model / max_tokens for no-thinking requests (title, commit message, etc.); claude_code only
  - SWE_HAIKU_MODEL: backend model name Claude Code routes haiku requests to (must be a backend-routable name, not a CLI-style name); claude_code only
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import time

from collections.abc import Iterator
from pathlib import Path
from typing import Any, overload

from nexus import BatchItem
from nexus.extensions.tasks.swe.constants import DATASET_SWE_ANY, SWEbenchInstance
from nexus.runner.single import RunInput


# ---------------------------------------------------------------------------
# Hard-coded non-secret defaults.
#
# 仅作为「兜底默认值」集中放置，用 setdefault 而非 update：外部 env 仍然优先，这里只在它们没设置时生效。
# 注意：不要把任何密钥（API key / token）写进来——密钥仍走 env。
#
# 同一份 generator 支持三套后端，按 OPENAI_CLIENT_NAME 分流：
#   - litellm        -> AWS/bedrock 直连（默认）
#   - vertex_anthropic -> GCP vertex passthrough
#   - openai         -> OpenAI-compatible endpoint（默认指向 Taiji HY3-offline）
# provider 相关的默认值（client 名、模型名、anthropic_version 等）分别放进 _LITELLM_ENV / _VERTEX_ENV，其余共用 _SHARED_ENV。
# ---------------------------------------------------------------------------
_SHARED_ENV: dict[str, str] = {
    'CLAUDE_CODE_VERSION': '2.1.117',
    'CLAUDE_CODE_WORKING_DIR': '/testbed',
    'CLAUDE_CODE_TIMEOUT': '14400',
    'SWE_LLM_TIMEOUT': '900',
    'SWE_LLM_MAX_RETRIES': '15',
    'ANTHROPIC_MODEL': 'claude-opus-4-8[1m]',
    'OPENAI_BASE_URL': 'http://trpc-gpt-eval.production.polaris:8080',
    'MAX_TOKENS': '16000',
    'SWE_THINKING_ENABLED': 'true',
    'SWE_THINKING_TYPE': 'adaptive',
    'SWE_THINKING_EFFORT': 'xhigh',
    'SWE_THINKING_PRESET': 'bedrock-claude',
    'SWE_CACHE_CONTROL_ENABLED': 'true',
    'SWE_CACHE_CONTROL_STRATEGY': 'claude_code',
    'SWE_LOG_LLM_USAGE': 'true',
    'SWE_USE_NEW_CACHE_PROXY': 'true',
    'CLAUDE_CODE_DISALLOWED_TOOLS': 'WebSearch,WebFetch',
    'SWE_MAX_ITERATIONS': '300',  # max iterations for swe_agent
    'SWE_RUNTIME_TIMEOUT': '14400',
    'SWE_EVAL_TIMEOUT_SECONDS': '2000',
    'SWE_RUNTIME_PROVIDER_NAME': 'gongfeng',
    'SWE_RUNTIME_BASE_URL': 'http://hyrl-sandbox.prod.woa.com:8052',
    'GONGFENG_PURPOSE': 'test',
    'SWE_RUNTIME_TAG': 'current',
    'SWE_RUNTIME_CPU': '8',
    'SWE_RUNTIME_MEMORY': '16',
    'SWE_RPC_PROTOCOL': 'http',
    'SWE_WORKER_VENV': 'true',
    'SWE_RECORD_TRAJECTORY': 'true',
    'SWE_TOOLS_MODE': 'branch_last',
    'SWE_ENABLE_ANTI_HACK_PROMPT': 'false',
    'SWE_ENABLE_REWARD_HACKING_DETECTION': 'false',
    'SWE_ENABLE_ANTI_HACK': 'false',
    'SWE_RAISE_ON_RUNTIME_ERROR': 'false',
    'SWE_RUN_ORACLE': 'false',
}

# litellm/bedrock: 模型名带 provider 前缀；
_LITELLM_ENV: dict[str, str] = {
    'OPENAI_CLIENT_NAME': 'litellm',
    'OPENAI_MODEL': 'bedrock/anthropic.claude-opus-4-8',
}

# vertex_anthropic: 裸 Anthropic 模型名（client 用它拼 vertex endpoint）+ anthropic_version
_VERTEX_ENV: dict[str, str] = {
    'OPENAI_CLIENT_NAME': 'vertex_anthropic',
    'OPENAI_MODEL': 'claude-opus-4-8',
    'SWE_ANTHROPIC_VERSION': 'vertex-2023-10-16',
    'SWE_DROP_FORCED_TOOL_CHOICE_WITH_THINKING': 'true',
}

_OPENAI_ENV: dict[str, str] = {
    'OPENAI_CLIENT_NAME': 'openai',
    'OPENAI_MODEL': 'hy3-offline',
    'OPENAI_BASE_URL': 'http://api.taiji.woa.com/openapi/inner/v2',
    'SWE_THINKING_ENABLED': 'false',
    'SWE_CACHE_CONTROL_ENABLED': 'false',
}

_CLIENT_NAME = os.getenv('OPENAI_CLIENT_NAME', 'litellm')
_PROVIDER_ENV = (
    _VERTEX_ENV if _CLIENT_NAME == 'vertex_anthropic' else _OPENAI_ENV if _CLIENT_NAME == 'openai' else _LITELLM_ENV
)
HARD_CODED_ENV: dict[str, str] = {**_SHARED_ENV, **_PROVIDER_ENV}

for _k, _v in HARD_CODED_ENV.items():
    os.environ.setdefault(_k, _v)


def _use_vertex() -> bool:
    """True when the host LLM client is the GCP vertex_anthropic passthrough."""
    return os.environ['OPENAI_CLIENT_NAME'] == 'vertex_anthropic'


def _use_openai() -> bool:
    """True when the host LLM client uses a plain OpenAI-compatible endpoint."""
    return os.environ['OPENAI_CLIENT_NAME'] == 'openai'


DEFAULT_SWE_DATASET_PATH = './outputs/swe_any/swe_any_instances.jsonl'

# Override resource_spec for specific instances
# Format: {'instance_id': {'cpu': N, 'memory': M}}
INSTANCE_RESOURCE_OVERRIDE: dict[str, dict[str, int]] = {}


# ---------------------------------------------------------------------------
# Hooks for non-interactive mode (--print)
# ---------------------------------------------------------------------------

_EXITPLAN_HOOK_CMD = (
    'echo \'{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow","updatedInput":{}}}\''
)

_ASKUSER_HOOK_SCRIPT = (Path(__file__).parent / 'askuser_hook.py').read_text(encoding='utf-8')
_ASKUSER_HOOK_SCRIPT_PATH = '/tmp/_nexus_askuser_hook.py'
_ASKUSER_HOOK_CMD = f'python3 {_ASKUSER_HOOK_SCRIPT_PATH}'


def _build_askuser_hook_script_writer() -> str:
    """Return a *bash* one-liner that writes the askuser hook to the sandbox.

    Bash (not python) on purpose: some repo images alias ``python`` to Python 2, where ``import pathlib`` fails (``No module named pathlib``). The hook source is base64-encoded so it carries no shell/Python special chars and rides safely through the ``shlex.quote``'d ``bash -c`` channel; the sandbox decodes it with ``base64 -d`` (coreutils/busybox), independent of any python version.
    """
    import base64 as _b64

    encoded = _b64.b64encode(_ASKUSER_HOOK_SCRIPT.encode()).decode()
    return (
        f'mkdir -p "$(dirname {_ASKUSER_HOOK_SCRIPT_PATH})" && '
        f"printf '%s' '{encoded}' | base64 -d > {_ASKUSER_HOOK_SCRIPT_PATH}"
    )


def _build_hook_patches() -> dict[str, Any]:
    return {
        'config_file_patches': [
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
        ],
    }


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


@overload
def _env_int(name: str, default: int) -> int: ...


@overload
def _env_int(name: str, default: None = None) -> int | None: ...


def _env_int(name: str, default: int | None = None) -> int | None:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float | None = None) -> float | None:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_stop_sequences() -> list[str] | None:
    """Parse SWE_STOP_SEQUENCES into a list, splitting on '||' when present, else ',' (vertex only)."""
    raw = os.getenv('SWE_STOP_SEQUENCES')
    if not raw:
        return None
    sep = '||' if '||' in raw else ','
    items = [item for item in (part.strip() for part in raw.split(sep)) if item]
    return items or None


def _instance_id_of(obj: Any) -> str:
    """Extract an instance_id from a selector element: a bare string, or a dict's ``instance_id`` field."""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        return str(obj.get('instance_id', '') or '').strip()
    return ''


def _extract_instance_ids(path: Path) -> set[str]:
    """Pull instance_id values out of a ``.json`` (a list of strings/dicts, or a single dict) or ``.jsonl`` (one object per line) file."""
    ids: set[str] = set()
    if path.suffix.lower() == '.jsonl':
        with path.open('r', encoding='utf-8') as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                iid = _instance_id_of(json.loads(line))
                if not iid:
                    raise ValueError(f'{path}:{line_no} has no resolvable "instance_id".')
                ids.add(iid)
        return ids
    data = json.loads(path.read_text(encoding='utf-8'))
    items = data if isinstance(data, list) else [data]
    for idx, obj in enumerate(items):
        iid = _instance_id_of(obj)
        if not iid:
            raise ValueError(f'{path}[{idx}] has no resolvable "instance_id".')
        ids.add(iid)
    return ids


def _resolve_instance_selector(name: str) -> set[str]:
    """Parse ``SWE_INCLUDE_INSTANCE_IDS`` / ``SWE_EXCLUDE_INSTANCE_IDS`` into a set of instance_id values.

    The value is a comma-separated list of items; each item is either a literal ``instance_id`` or a path to a ``.json`` / ``.jsonl`` file from which instance_ids are extracted (see :func:`_extract_instance_ids`). Returns an empty set when unset/empty, which disables that filter.
    """
    raw = os.getenv(name, '').strip()
    if not raw:
        return set()
    ids: set[str] = set()
    for item in (x.strip() for x in raw.split(',')):
        if not item:
            continue
        lower = item.lower()
        if lower.endswith('.json') or lower.endswith('.jsonl'):
            path = Path(item)
            if not path.is_file():
                raise ValueError(f'{name}: selector file not found: {item}')
            ids.update(_extract_instance_ids(path))
        else:
            ids.add(item)
    return ids


def _selected_instance_filters() -> tuple[set[str], set[str]]:
    """Resolve the include/exclude instance_id sets, rejecting any instance_id claimed by both.

    Include is a whitelist (empty = allow all); exclude is a blacklist applied after include.
    """
    include = _resolve_instance_selector('SWE_INCLUDE_INSTANCE_IDS')
    exclude = _resolve_instance_selector('SWE_EXCLUDE_INSTANCE_IDS')
    conflict = include & exclude
    if conflict:
        raise ValueError(
            f'instance_id(s) listed in both SWE_INCLUDE_INSTANCE_IDS and SWE_EXCLUDE_INSTANCE_IDS: {sorted(conflict)}'
        )
    return include, exclude


def _is_instance_selected(instance_id: str, include: set[str], exclude: set[str]) -> bool:
    """Apply include (whitelist; empty = all) then exclude (blacklist)."""
    if include and instance_id not in include:
        return False
    return instance_id not in exclude


def _env_list(name: str) -> list[str]:
    """Parse comma-separated list from env var, preserving order."""
    v = os.getenv(name)
    if not v:
        return []
    return [item.strip() for item in v.split(',') if item.strip()]


def _iter_instances_from_file(dataset_path: str) -> Iterator[SWEbenchInstance]:
    """Iterate over instances from a local JSONL file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if path.suffix != '.jsonl':
        raise ValueError(f'Dataset must be .jsonl for batch mode: {dataset_path}')

    with path.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            dataset_type = instance.get('dataset_type')
            if dataset_type != DATASET_SWE_ANY:
                raise ValueError(
                    f'{dataset_path}:{line_number}: expected dataset_type={DATASET_SWE_ANY!r}, got {dataset_type!r}'
                )
            yield instance


def _iter_remote_instance_ids(dataset: str, split: str | None = None) -> Iterator[str]:
    """Iterate over instance IDs from the remote dataset server."""
    from nexus.utils.dataset_client import list_instances_sync

    instance_ids = list_instances_sync(dataset, split)
    yield from instance_ids


def _iter_instance_ids(
    remote_dataset: str | None, remote_split: str | None, dataset_path: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Iterate over (instance_id, metadata) tuples from remote or local source."""
    if remote_dataset:
        # Remote mode: iterate over instance IDs from server
        for instance_id in _iter_remote_instance_ids(remote_dataset, remote_split):
            if instance_id:
                yield (
                    instance_id,
                    {
                        'dataset': remote_dataset,
                        'split': remote_split or '',
                    },
                )
    else:
        # Local mode: iterate over instances from local file
        for inst in _iter_instances_from_file(dataset_path):
            instance_id = str(inst.get('instance_id', '') or '')
            if instance_id:
                yield (
                    instance_id,
                    {
                        'dataset_path': dataset_path,
                        'repo': inst.get('repo', ''),
                        'dataset_type': DATASET_SWE_ANY,
                        'language': inst.get('language', 'unknown'),
                    },
                )


def _build_request_args(*, include_thinking: bool) -> dict[str, Any]:
    """Build request_args sent to the backend on every request.

    按 provider 分流: vertex_anthropic 需要 anthropic_version / drop_forced_tool_choice，且 thinking 块不带 display；litellm/bedrock 用 provider 前缀模型名，thinking 带 display=summarized。include_thinking 区分两条路径: thinking 请求带上 thinking / output_config，no-thinking 请求 (title 生成、commit message 等) 只保留基础参数以节省 token。
    """
    # request_args: 每次请求都会传给后端 API
    args: dict[str, Any] = {
        # litellm 需 provider 前缀名；vertex 需裸 Anthropic 名（用于拼 endpoint）
        'model': os.environ['OPENAI_MODEL'],
        # 最大输出 token（不含 thinking）
        'max_tokens': _env_int('MAX_TOKENS', 16000),
    }

    if _use_vertex():
        args['anthropic_version'] = os.environ['SWE_ANTHROPIC_VERSION']
        # thinking 开启时 vertex 拒绝强制 tool_choice，置 true 让 client 自动丢弃
        args['drop_forced_tool_choice_with_thinking'] = _env_bool('SWE_DROP_FORCED_TOOL_CHOICE_WITH_THINKING', True)
        # 可选采样参数：仅在显式设置时透传
        if (temperature := _env_float('SWE_TEMPERATURE')) is not None:
            args['temperature'] = temperature
        if (top_p := _env_float('SWE_TOP_P')) is not None:
            args['top_p'] = top_p
        if (top_k := _env_int('SWE_TOP_K')) is not None:
            args['top_k'] = top_k
        if stop_sequences := _env_stop_sequences():
            args['stop'] = stop_sequences
        if metadata_raw := os.getenv('SWE_GCP_METADATA_JSON'):
            args['metadata'] = json.loads(metadata_raw)
    elif _use_openai() and (reasoning_effort := os.getenv('SWE_REASONING_EFFORT')):
        args['reasoning_effort'] = reasoning_effort

    if include_thinking and _env_bool('SWE_THINKING_ENABLED'):
        thinking: dict[str, Any] = {'type': os.environ['SWE_THINKING_TYPE']}
        if _use_vertex():
            # vertex: 可选显式 budget_tokens；不带 display
            if (budget_tokens := _env_int('SWE_THINKING_BUDGET_TOKENS')) is not None:
                thinking['budget_tokens'] = budget_tokens
        else:
            # litellm/bedrock: thinking 块带 display=summarized
            thinking['display'] = 'summarized'
        args['thinking'] = thinking
        args['output_config'] = {'effort': os.environ['SWE_THINKING_EFFORT']}

    return args


def _build_cache_task_id(instance_id: str) -> str:
    """生成本次 run 的 ``cache_task_id`` —— 32 位 md5 。

    网关按这个 id 隔离 prompt 缓存，所以它必须在任务维度全局唯一：两个 instance 共用一个 id 会让彼此的前缀缓存串台。 种子取 时间 + 进程号 + instance_id ，保证并发 shard 之间也不会 撞号（每个 shard 是独立进程，且 generator 只在生成阶段跑一次）。
    """
    seed = f'{time.time()}:{os.getpid()}:{instance_id}'
    return hashlib.md5(seed.encode('utf-8')).hexdigest()


def _build_openai_clients(instance_id: str) -> dict[str, dict[str, Any]]:
    client_name = os.environ['OPENAI_CLIENT_NAME']
    api_key = os.getenv('OPENAI_API_KEY', '<REDACTED_API_KEY>').strip()
    base_url = os.environ['OPENAI_BASE_URL']
    agent_name = os.getenv('SWE_AGENT_NAME', 'swe_agent')

    # cache_task_id 作为 query 挂在 api_key 上：Claude 网关 key，或 model-eval host 默认开。
    attach_cache = 'cache_task_id=' not in api_key and ('claude' in api_key or 'llm-api.model-eval.woa.com' in base_url)
    if attach_cache:
        cache_task_id = _build_cache_task_id(instance_id)
        separator = '&' if '?' in api_key else '?'
        api_key = f'{api_key}{separator}cache_task_id={cache_task_id}'

    # client_args: 连接层配置，仅在 nexus client 内部消费
    client_args: dict[str, Any] = {
        'api_key': api_key,
        'base_url': base_url,
        'timeout': _env_int('SWE_LLM_TIMEOUT'),
        'max_retries': _env_int('SWE_LLM_MAX_RETRIES'),
    }
    if not _use_openai():
        # LiteLLM/Vertex consume this Nexus-specific option; AsyncOpenAI rejects it.
        client_args['log_llm_usage'] = _env_bool('SWE_LOG_LLM_USAGE')

    if agent_name == 'claude_code':
        # 启用 Anthropic prompt caching
        client_args['cache_control_enabled'] = _env_bool('SWE_CACHE_CONTROL_ENABLED')
        # 'claude_code' 策略: 标记最后 1 条 user/tool 消息 + 最后一个 tool 定义
        # 适合 Claude Code 长对话模式（每轮只追加少量新内容，前缀高度稳定，cache 命中率 >95%）
        client_args['cache_control_strategy'] = os.environ['SWE_CACHE_CONTROL_STRATEGY']
    # 可选：直接指定 vertex endpoint（vertex 专属），否则 client 用 base_url 拼接
    if _use_vertex() and (endpoint := os.getenv('SWE_GCP_ENDPOINT')):
        client_args['endpoint'] = endpoint

    return {
        'main': {
            'name': client_name,
            'config': {
                'client_args': client_args,
                'request_args': _build_request_args(include_thinking=True),
            },
        }
    }


def _build_agent_config(agent_name: str) -> dict[str, Any]:
    if agent_name == 'claude_code':
        # 注入到 Claude Code 进程的环境变量
        extra_env: dict[str, str] = {
            # CLI 进程内部使用的模型标识（[1m] 仅让 CLI 按 1M 上下文预算，不影响后端）；
            # host 侧会丢弃此 model 名并统一改用 OPENAI_MODEL，故与 provider 无关。可用 ANTHROPIC_MODEL 覆盖
            'ANTHROPIC_MODEL': os.environ['ANTHROPIC_MODEL'],
            # 禁用 attribution header（减少不必要请求头）
            'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
            # 禁用遥测等非必要网络请求
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
            # 禁用实验性 beta 功能，避免不可控行为
            'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
        }
        # 允许通过 CLAUDE_CODE_EXTRA_ENV_JSON（JSON）追加/覆盖 extra_env
        extra_env_json = os.getenv('CLAUDE_CODE_EXTRA_ENV_JSON')
        if extra_env_json:
            extra_env.update(json.loads(extra_env_json))

        # Claude Code CLI 参数:
        #   --print: 非交互模式
        #   --output-format stream-json: JSON 流式输出，便于 nexus 解析 tool calls/thinking
        #   --verbose: 详细日志
        #   --disallowedTools: 禁用工具，默认 WebSearch/WebFetch，可用 CLAUDE_CODE_DISALLOWED_TOOLS 覆盖
        extra_cli_args = [
            '--print',
            '--output-format',
            'stream-json',
            '--verbose',
            '--disallowedTools',
            *_env_list('CLAUDE_CODE_DISALLOWED_TOOLS'),
        ]
        # 允许通过 CLAUDE_CODE_EXTRA_CLI_ARGS 追加额外 CLI 参数
        if os.getenv('CLAUDE_CODE_EXTRA_CLI_ARGS'):
            extra_cli_args.extend(shlex.split(os.environ['CLAUDE_CODE_EXTRA_CLI_ARGS']))

        agent_config: dict[str, Any] = {
            'working_dir': os.environ['CLAUDE_CODE_WORKING_DIR'],
            # agent 总超时（秒）。14400s = 4h，留给复杂 SWE 实例足够时间
            'timeout': _env_int('CLAUDE_CODE_TIMEOUT'),
            # Claude Code CLI 版本号（runtime 镜像中预装）
            'version': os.environ['CLAUDE_CODE_VERSION'],
            'pre_start_scripts': [
                {
                    'interpreter': 'bash',
                    'code': _build_askuser_hook_script_writer(),
                },
            ],
            'extra_cli_args': extra_cli_args,
            'extra_env': extra_env,
            # 告诉 anthropic proxy converter 使用 blocks 格式:
            #   请求方向: 从 assistant 消息读取 thinking_blocks 数组，保留 signature
            #   响应方向: 从 LLM 响应读取 thinking_blocks，构造 ThinkingContent
            #   Anthropic API 要求 thinking block 必须携带有效 signature
            'thinking_preset': os.environ['SWE_THINKING_PRESET'],
            **_build_hook_patches(),
        }

        # 当 Claude Code 发送不需要 thinking 的请求时（title 生成、commit message 等），用这组 request_args 覆盖默认配置，去掉 thinking 参数节省 token。可用 SWE_NO_THINKING_MODEL / SWE_NO_THINKING_MAX_TOKENS 单独覆盖。
        no_thinking_args = _build_request_args(include_thinking=False)
        no_thinking_model = os.getenv('SWE_NO_THINKING_MODEL')
        if no_thinking_model:
            no_thinking_args['model'] = no_thinking_model
        no_thinking_max_tokens = _env_int('SWE_NO_THINKING_MAX_TOKENS')
        if no_thinking_max_tokens is not None:
            no_thinking_args['max_tokens'] = no_thinking_max_tokens
        agent_config['no_thinking_request_args'] = no_thinking_args

        # Claude Code 请求 haiku 模型时（轻量级任务），路由到此模型。注意：值须为后端可路由的合法模型名（litellm 带 provider 前缀 / vertex 裸 Anthropic 名），非 CLI 风格名。
        haiku_model = os.getenv('SWE_HAIKU_MODEL')
        if haiku_model:
            agent_config['haiku_model'] = haiku_model

        binary_url = os.getenv('CLAUDE_CODE_BINARY_URL', '').strip()
        if binary_url:
            agent_config['binary_url'] = binary_url
        binary_checksum = os.getenv('CLAUDE_CODE_BINARY_CHECKSUM', '').strip()
        if binary_checksum:
            agent_config['binary_checksum'] = binary_checksum
        return agent_config

    return {'max_iterations': _env_int('SWE_MAX_ITERATIONS')}


def _build_runtime_provider_config(instance_id: str) -> dict[str, Any]:
    """Build runtime_provider_config. Only instance_id is passed (for per-instance resource overrides); everything else is read from env inside."""
    agent_name = os.getenv('SWE_AGENT_NAME', 'swe_agent')
    eval_timeout_seconds = _env_int('SWE_EVAL_TIMEOUT_SECONDS', 2000)
    # request_timeout（单次 RPC HTTP 超时）须包住最长的单次调用：
    # agent 会话（仅 claude_code 有时间超时；swe_agent 用步数限制，时间记 0）或 eval 阶段，二者取大再加 600s 余量。
    agent_timeout = (
        _env_int('CLAUDE_CODE_TIMEOUT', 14400)
        if agent_name == 'claude_code'
        else _env_int('SWE_RUNTIME_TIMEOUT', 14400)
    )
    runtime_timeout = max(agent_timeout, eval_timeout_seconds) + 600

    runtime_tag = os.environ['SWE_RUNTIME_TAG'].strip()
    runtime_image = (
        runtime_tag if runtime_tag == 'current' else f'mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{runtime_tag}'
    )

    override = INSTANCE_RESOURCE_OVERRIDE.get(instance_id, {})
    config = {
        'api_base_url': os.environ['SWE_RUNTIME_BASE_URL'],
        'token': os.getenv('GONGFENG_RUNTIME_TOKEN', os.getenv('GONGFENG_TOKEN', '')),
        'runtime_image': runtime_image,
        'request_timeout': runtime_timeout,
        'resource_spec': {
            'cpu': override.get('cpu', _env_int('SWE_RUNTIME_CPU')),
            'memory': override.get('memory', _env_int('SWE_RUNTIME_MEMORY')),
        },
        'rpc_protocol': os.environ['SWE_RPC_PROTOCOL'],
        'purpose': os.environ['GONGFENG_PURPOSE'],
    }
    batch_id = os.getenv('SWE_ANY_BATCH_ID')
    if batch_id:
        config['batch_id'] = batch_id
    return config


def _build_runtime_configs() -> dict[str, Any]:
    """Build per-stage runtime_configs (main/eval). All knobs read from env."""
    use_new_cache_proxy = _env_bool('SWE_USE_NEW_CACHE_PROXY')
    stage_config: dict[str, Any] = {'use_new_cache_proxy': use_new_cache_proxy}
    if _env_bool('SWE_WORKER_VENV'):
        stage_config['worker_venv'] = True
    return {'main': dict(stage_config), 'eval': dict(stage_config)}


def _build_processors() -> dict[str, Any]:
    """Build span processors and enforce the detector's trajectory dependency."""
    if _env_bool('SWE_ENABLE_REWARD_HACKING_DETECTION') and not _env_bool('SWE_RECORD_TRAJECTORY'):
        raise ValueError('SWE_ENABLE_REWARD_HACKING_DETECTION requires SWE_RECORD_TRAJECTORY=true')
    if not _env_bool('SWE_RECORD_TRAJECTORY'):
        return {}
    return {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': os.environ['SWE_TOOLS_MODE']}}}


def generate_configs() -> Iterator[BatchItem]:
    # Check for remote dataset first (takes precedence over local file)
    remote_dataset = os.getenv('SWE_DATASET', '') or None
    remote_split = os.getenv('SWE_DATASET_SPLIT', '') or None
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_SWE_DATASET_PATH)

    skip = _env_int('SWE_SKIP', 0)
    limit = _env_int('SWE_LIMIT')
    include_ids, exclude_ids = _selected_instance_filters()
    run_oracle = _env_bool('SWE_RUN_ORACLE')
    raise_on_runtime_error = _env_bool('SWE_RAISE_ON_RUNTIME_ERROR')
    enable_anti_hack_prompt = _env_bool('SWE_ENABLE_ANTI_HACK_PROMPT')
    enable_reward_hacking_detection = _env_bool('SWE_ENABLE_REWARD_HACKING_DETECTION')
    enable_anti_hack = _env_bool('SWE_ENABLE_ANTI_HACK')
    hack_reward_override = _env_float('SWE_HACK_REWARD_OVERRIDE')
    agent_name = os.getenv('SWE_AGENT_NAME', 'swe_agent')
    eval_timeout_seconds = _env_int('SWE_EVAL_TIMEOUT_SECONDS')
    runtime_provider_name = os.environ['SWE_RUNTIME_PROVIDER_NAME']

    skipped = 0
    produced = 0

    for instance_id, instance_meta in _iter_instance_ids(remote_dataset, remote_split, dataset_path):
        # Apply include (whitelist) / exclude (blacklist) filters
        if not _is_instance_selected(instance_id, include_ids, exclude_ids):
            continue

        # Skip instances until we reach the skip threshold
        if skipped < skip:
            skipped += 1
            continue

        # Build task_config based on source type
        if remote_dataset:
            task_config: dict[str, Any] = {
                'remote_dataset': remote_dataset,
                'split': remote_split or '',
                'instance_id': instance_id,
                'run_oracle': bool(run_oracle),
                'raise_on_runtime_error': bool(raise_on_runtime_error),
                'eval_timeout_seconds': eval_timeout_seconds,
                'enable_anti_hack_prompt': enable_anti_hack_prompt,
                'enable_reward_hacking_detection': enable_reward_hacking_detection,
                'enable_anti_hack': enable_anti_hack,
                'hack_reward_override': hack_reward_override,
            }
        else:
            task_config = {
                'dataset_path': dataset_path,
                'instance_id': instance_id,
                'run_oracle': bool(run_oracle),
                'raise_on_runtime_error': bool(raise_on_runtime_error),
                'eval_timeout_seconds': eval_timeout_seconds,
                'enable_anti_hack_prompt': enable_anti_hack_prompt,
                'enable_reward_hacking_detection': enable_reward_hacking_detection,
                'enable_anti_hack': enable_anti_hack,
                'hack_reward_override': hack_reward_override,
            }

        config: dict[str, Any] = {
            'imports': [
                'nexus.extensions.tasks.swe',
                'nexus.extensions.agents.swe_agent',
                'nexus.extensions.agents.claude_code',
            ],
            'task_name': 'swe_task',
            'task_config': task_config,
            'runtime_provider_name': runtime_provider_name,
            'runtime_provider_config': _build_runtime_provider_config(instance_id),
            'runtime_configs': _build_runtime_configs(),
            'processors': _build_processors(),
        }

        # If not running oracle, we need an agent + LLM config.
        if not run_oracle:
            config['agent_name'] = agent_name
            config['agent_config'] = _build_agent_config(agent_name)
            # 每个 instance 单独构建 client 配置：cache_task_id 必须逐 run 唯一，共用一份会串台。
            config['openai_clients'] = _build_openai_clients(instance_id)
        else:
            config['agent_name'] = None
            config['openai_clients'] = {}

        item_metadata: dict[str, Any] = {
            'instance_id': instance_id,
            'agent_name': agent_name,
            'model': os.environ['OPENAI_MODEL'],
            **instance_meta,
        }

        yield BatchItem(
            config=RunInput.model_validate(config),
            run_id=instance_id,
            metadata=item_metadata,
        )

        produced += 1
        if limit is not None and produced >= limit:
            break


def get_total_count() -> int:
    remote_dataset = os.getenv('SWE_DATASET', '') or None
    remote_split = os.getenv('SWE_DATASET_SPLIT', '') or None
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_SWE_DATASET_PATH)
    skip = _env_int('SWE_SKIP', 0)
    limit = _env_int('SWE_LIMIT')
    include_ids, exclude_ids = _selected_instance_filters()

    n = 0
    skipped = 0

    for instance_id, _ in _iter_instance_ids(remote_dataset, remote_split, dataset_path):
        # Apply include (whitelist) / exclude (blacklist) filters
        if not _is_instance_selected(instance_id, include_ids, exclude_ids):
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
    remote_dataset = os.getenv('SWE_DATASET', '') or None
    remote_split = os.getenv('SWE_DATASET_SPLIT', '') or None
    dataset_path = os.getenv('SWE_DATASET_PATH', DEFAULT_SWE_DATASET_PATH)
    include_ids, exclude_ids = _selected_instance_filters()

    metadata: dict[str, Any] = {
        'task_name': 'swe_task',
        'run_oracle': _env_bool('SWE_RUN_ORACLE'),
        'include_instance_ids': sorted(include_ids) if include_ids else None,
        'exclude_instance_ids': sorted(exclude_ids) if exclude_ids else None,
        'agent_name': os.getenv('SWE_AGENT_NAME', 'swe_agent'),
        'model': os.environ['OPENAI_MODEL'],
        'enable_anti_hack_prompt': _env_bool('SWE_ENABLE_ANTI_HACK_PROMPT'),
        'enable_reward_hacking_detection': _env_bool('SWE_ENABLE_REWARD_HACKING_DETECTION'),
        'enable_anti_hack': _env_bool('SWE_ENABLE_ANTI_HACK'),
        'hack_reward_override': _env_float('SWE_HACK_REWARD_OVERRIDE'),
    }

    if remote_dataset:
        metadata['dataset'] = remote_dataset
        metadata['split'] = remote_split or ''
    else:
        metadata['dataset_path'] = dataset_path
        metadata['dataset_type_default'] = DATASET_SWE_ANY

    return metadata
