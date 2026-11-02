"""local_one_agent_infer batch generator for nexus run-batch.

改编自 ``one_agent_infer.generator``：从 benchmark JSONL 生成 BatchItem，多轮 session
（同 ``session_id``）合并成一个 BatchItem 的 ``turns`` 列表。

本地模式差异：
- ``task_name`` 为 ``local_one_agent_infer``，``agent_name`` 为 ``local_one_agent``。
- 每个 item 注入 ``task_config.local_context``（``local_one_agent`` 的权限/沙箱/SP 渲染上下文，
  agent 从 ``task_config['local_context']`` 消费）。工作区路径仅配
  ``local_context.workspace.path``，task 与 agent 共用（见 ``resolve_workspace_path``）。

环境变量：
    YB_BENCH_PATH:   benchmark 根目录（含 jsonl 与 files/）
    YB_BENCH_JSONL:  benchmark jsonl 路径（可选，覆盖默认）
    YB_BENCH_CONFIG / LOCAL_BENCH_CONFIG:
        base config YAML 完整路径（可选）
    YB_BENCH_CONFIG_NAME / LOCAL_BENCH_CONFIG_NAME:
        config/ 下的文件名（可选，例如 ``ags_base.yaml``）；未设完整路径时生效，
        默认 ``base.yaml``
    YB_BENCH_LIMIT:  只取前 N 个 session（可选）
    LOCAL_WORKSPACE_DIR: 本地工作区根目录（默认 /workspace，写入 local_context.workspace.path）

AGS 路由：
    若 base config 顶层 ``runtime_provider_name == 'ags'``，则将 ``task_name`` 覆写为
    ``local_one_agent_infer_ags``（见 ``ags_task.py``），走远端 upload + runtime 初始化。

逐 item 运行时配置（jsonl 内配置，无需环境变量）：
    可在 record 里放一个可选的 ``local_config`` 对象，逐 session（取该 session 首条
    record）覆盖运行时环境。优先级 **record > 环境变量 > YAML 默认**——base config 在
    load 时已 ``resolve=True`` 把环境变量/默认值填入，record 里显式给出的字段再叠加覆盖，
    故不带 ``local_config`` 的存量 jsonl 行为完全不变（向后兼容）。支持字段：

        {
          "local_config": {
            "workspace_dir": "/workspace",           # 覆盖 local_context.workspace.path
            "permission_level": "readonly",          # readonly / basic
            "runtime": {"os": "windows", "arch": "x64", "shell": "powershell"},
            "env_mode": "warm",                      # cold / warm
            "max_iterations": 30
          }
        }
"""

from __future__ import annotations

import json
import logging
import os
import re

from collections import OrderedDict
from pathlib import Path

from omegaconf import OmegaConf

from nexus import BatchItem


logger = logging.getLogger(__name__)


# arch 别名归一化：bench 数据/环境变量里常见 x64 / amd64 / aarch64 等写法，
# 统一映射到 LocalRuntime.arch 接受的规范值（arm64 / x86_64），避免校验失败。
_ARCH_ALIASES = {
    'x64': 'x86_64',
    'amd64': 'x86_64',
    'x86-64': 'x86_64',
    'x86_64': 'x86_64',
    'aarch64': 'arm64',
    'arm64': 'arm64',
}


def _normalize_arch(arch: str) -> str:
    """把常见 arch 写法归一化到 LocalRuntime 接受的规范值；未知值原样返回。"""
    return _ARCH_ALIASES.get(str(arch).strip().lower(), arch)


def _benchmark_root(input_path: str | None = None) -> Path:
    """Resolve benchmark root directory."""
    if input_path:
        return Path(input_path)
    env = os.getenv('YB_BENCH_PATH')
    if env:
        return Path(env)
    raise FileNotFoundError('Benchmark root not found. Set YB_BENCH_PATH or pass --input-path.')


def _split_time_and_tz(time_str: str) -> tuple[str, str]:
    """Split a bench ``time`` string into ``(date_part, tz_name)``.

    Bench 记录里 time 已内嵌时区，如 ``'2026-03-12 Thursday (Asia/Shanghai, UTC+8)'``。
    SP 里 ``cur_date`` 期望纯日期，故拆出日期与 IANA 时区名分别下发。
    """
    if not time_str:
        return '', ''
    m = re.search(r'\s*\(([^,)]+)(?:,[^)]*)?\)\s*$', time_str)
    if not m:
        return time_str.strip(), ''
    date_part = time_str[: m.start()].strip()
    tz_name = m.group(1).strip()
    return date_part, tz_name


def _config_path(base_config_path: str | Path | None = None) -> Path:
    """Resolve config YAML path.

    优先级：显式 ``base_config_path`` > 完整路径环境变量 >
    ``config/`` 下文件名环境变量 > 默认 ``config/base.yaml``。
    """
    config_dir = Path(__file__).parent / 'config'

    if base_config_path:
        p = Path(base_config_path)
        if p.exists():
            return p
        # 允许只传文件名（相对 config/）
        named = config_dir / p.name
        if named.exists():
            return named
        raise FileNotFoundError(f'Config not found: {p}')

    env_path = os.getenv('YB_BENCH_CONFIG') or os.getenv('LOCAL_BENCH_CONFIG')
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        # 相对路径：相对 cwd 找不到时再试 config/
        named = config_dir / p.name
        if named.exists():
            return named
        raise FileNotFoundError(f'Config not found: {p}')

    env_name = os.getenv('YB_BENCH_CONFIG_NAME') or os.getenv('LOCAL_BENCH_CONFIG_NAME')
    if env_name:
        named = config_dir / Path(env_name).name
        if named.exists():
            return named
        raise FileNotFoundError(f'Config not found: {named}. Set YB_BENCH_CONFIG_NAME to a file under {config_dir}.')

    default = config_dir / 'base.yaml'
    if default.exists():
        return default
    raise FileNotFoundError(
        'Config not found. Set YB_BENCH_CONFIG / YB_BENCH_CONFIG_NAME=ags_base.yaml or create config/base.yaml.'
    )


_AGS_TASK_NAME = 'local_one_agent_infer_ags'


def _apply_ags_routing(config: dict) -> None:
    """``runtime_provider_name=ags`` 时路由到 ``ags_task`` 注册的 task。"""
    if config.get('runtime_provider_name') != 'ags':
        return
    prev = config.get('task_name')
    config['task_name'] = _AGS_TASK_NAME
    if prev and prev != _AGS_TASK_NAME:
        logger.debug(
            'runtime_provider_name=ags → task_name %r → %r',
            prev,
            _AGS_TASK_NAME,
        )


def _load_base_config(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _load_benchmark(root: Path) -> list[dict]:
    """Load benchmark JSONL file."""
    env_jsonl = os.getenv('YB_BENCH_JSONL')
    if env_jsonl:
        jsonl_path = Path(env_jsonl)
    else:
        # 默认取根目录下唯一的 *.jsonl（找不到/多个则报错，让用户显式指定）
        candidates = sorted(root.glob('*.jsonl'))
        if len(candidates) == 1:
            jsonl_path = candidates[0]
        elif not candidates:
            raise FileNotFoundError(f'No .jsonl found under {root}. Set YB_BENCH_JSONL explicitly.')
        else:
            raise FileNotFoundError(
                f'Multiple .jsonl under {root}: {[c.name for c in candidates]}. Set YB_BENCH_JSONL to pick one.'
            )

    if not jsonl_path.exists():
        raise FileNotFoundError(f'Benchmark file not found: {jsonl_path}')

    records = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL line {idx + 1}: {exc}') from exc
            records.append(data)
    return records


def _group_by_session(records: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group records by session_id, preserving order of first appearance."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in records:
        key = r.get('session_id') or r.get('task_id', '')
        groups.setdefault(key, []).append(r)

    result = []
    for key, items in groups.items():
        items.sort(key=lambda x: x.get('turn') or 1)
        result.append((key, items))
    return result


def _record_to_turn(record: dict) -> dict:
    """Convert a benchmark record to a turn dict for task_config.turns."""
    input_file_list = record.get('input_file_list', [])
    if isinstance(input_file_list, str):
        input_file_list = json.loads(input_file_list)

    turn = {
        'turn': record.get('turn') or 1,
        'task_id': record['task_id'],
        'query': record['query'],
        'input_file_list': input_file_list,
        'expected_output': record.get('expected_output', ''),
        'L1': record.get('L1', ''),
        'L2': record.get('L2', ''),
        'difficulty': record.get('difficulty', ''),
        'source_metadata': record.get('source_metadata'),
    }

    pre_parsed = record.get('pre_parsed_doc_blocks')
    if pre_parsed:
        turn['pre_parsed_doc_blocks'] = pre_parsed

    if 'session_doc_index_start' in record:
        turn['session_doc_index_start'] = record['session_doc_index_start']

    return turn


def _apply_local_config(
    local_config: dict,
    *,
    local_context: dict,
    agent_config: dict,
) -> None:
    """把 record 的 ``local_config`` 叠加到 local_context / agent_config 上。

    仅覆盖 record 里**显式给出**的字段（record > env > YAML 默认）；未给出的保持 base
    config 里已由环境变量/默认值 resolve 好的值。就地修改 ``local_context`` /
    ``agent_config``。工作区路径写入 ``local_context.workspace.path``（单一配置源）。
    """
    if not isinstance(local_config, dict):
        return

    workspace_path = local_config.get('workspace_dir')
    workspace_override = local_config.get('workspace')
    if isinstance(workspace_override, dict) and workspace_override.get('path'):
        workspace_path = workspace_override['path']
    if workspace_path:
        workspace = dict(local_context.get('workspace') or {})
        workspace['path'] = str(workspace_path)
        local_context['workspace'] = workspace

    if local_config.get('permission_level'):
        permission = dict(local_context.get('permission') or {})
        permission['level'] = local_config['permission_level']
        local_context['permission'] = permission

    runtime_override = local_config.get('runtime')
    if isinstance(runtime_override, dict) and runtime_override:
        runtime = dict(local_context.get('runtime') or {})
        for key in ('os', 'arch', 'shell'):
            if runtime_override.get(key):
                runtime[key] = runtime_override[key]
        if runtime.get('arch'):
            runtime['arch'] = _normalize_arch(runtime['arch'])
        local_context['runtime'] = runtime

    if local_config.get('env_mode'):
        env = dict(local_context.get('env') or {})
        env['mode'] = local_config['env_mode']
        local_context['env'] = env

    if local_config.get('max_iterations') is not None:
        agent_config['max_iterations'] = local_config['max_iterations']


class LocalOneAgentInferGenerator:
    """Yields BatchItem instances from a local-mode OneAgentBench benchmark."""

    def __init__(
        self,
        input_path: str | None = None,
        base_config_path: str | Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.benchmark_root = _benchmark_root(input_path)
        _resolved_config = _config_path(base_config_path)
        logger.debug(
            '[local_one_agent_infer] base_config_path=%r YB_BENCH_CONFIG=%r -> resolved config: %s',
            base_config_path,
            os.getenv('YB_BENCH_CONFIG'),
            _resolved_config,
        )
        self.base_config = _load_base_config(_resolved_config)
        _apply_ags_routing(self.base_config)
        env_limit = os.getenv('YB_BENCH_LIMIT')
        self.limit = int(env_limit) if env_limit else limit

    def __iter__(self):
        records = _load_benchmark(self.benchmark_root)
        sessions = _group_by_session(records)

        if self.limit is not None:
            sessions = sessions[: self.limit]

        default_task_config = dict(self.base_config.get('task_config') or {})

        for i, (session_key, items) in enumerate(sessions):
            config = dict(self.base_config)
            _apply_ags_routing(config)
            first = items[0]

            turns = [_record_to_turn(r) for r in items]

            cur_date, tz_name = _split_time_and_tz(first.get('time', ''))

            # session 级运行时配置：取首条 record 的可选 local_config，叠加到 base config
            # 已 resolve 的 local_context / agent_config 上（record > env > YAML 默认）。
            # local_context 模板来自 base config 的 task_config.local_context（agent 从
            # task_config['local_context'] 消费，见 local_one_agent/agent.py）。
            base_task_config = dict(default_task_config)
            agent_config = dict(config.get('agent_config') or {})
            local_context = dict(base_task_config.get('local_context') or {})
            _apply_local_config(
                first.get('local_config') or {},
                local_context=local_context,
                agent_config=agent_config,
            )

            # 最终兜底：base config（含 LOCAL_RUNTIME_ARCH 环境变量）里的 arch 也可能是
            # x64 等别名，统一归一化到 LocalRuntime 接受的规范值。
            runtime = dict(local_context.get('runtime') or {})
            if runtime.get('arch'):
                runtime['arch'] = _normalize_arch(runtime['arch'])
                local_context['runtime'] = runtime

            workspace = dict(local_context.get('workspace') or {})
            workspace.setdefault('scope', 'conversation')
            local_context['workspace'] = workspace

            # SP「用户信息」区渲染用：cur_date / user_location 下发到 local_context，
            # 供 local_one_agent 的 _render_system_prompt 注入模板。
            local_context['cur_date'] = first.get('time', cur_date)
            local_context['user_location'] = first.get('location', '')

            config['task_config'] = {
                'task_id': first['task_id'],
                'session_id': session_key,
                'turns': turns,
                'benchmark_root': str(self.benchmark_root),
                'timeout': 600 * len(turns),  # scale timeout with turn count
                # local_context：agent 从 task_config['local_context'] 读取完整的
                # 权限/沙箱/runtime/env/SP 渲染上下文；task 通过 resolve_workspace_path 读工作区。
                'local_context': local_context,
                # 溯源信息：task 只拿得到 task_config（拿不到 BatchItem.metadata），
                # 落盘 trajectory 时用它把 metadata / source_metadata 补全到导出格式。
                'trace_meta': {
                    'index': i,
                    'session_id': session_key,
                    'source_metadata': first.get('source_metadata'),
                },
            }

            config['agent_config'] = agent_config

            # meta（cur_date / timezone / user_location）——本地 SP 渲染可用。
            # local_one_agent 目前不消费该字段，保留以便 SP 未来引用或评测记录。
            meta = {
                k: v
                for k, v in {
                    'cur_date': cur_date,
                    'timezone': tz_name,
                    'user_location': first.get('location', ''),
                }.items()
                if v
            }
            if meta:
                config['task_config']['meta'] = meta

            yield BatchItem(
                config=config,  # type: ignore[invalid-argument-type]
                run_id=f'local_bench_{session_key}',
                metadata={
                    'index': i,
                    'session_id': session_key,
                    'task_id': first['task_id'],
                    'n_turns': len(turns),
                    'L1': first.get('L1', ''),
                    'L2': first.get('L2', ''),
                    'difficulty': first.get('difficulty', ''),
                },
            )

    @staticmethod
    def total_count() -> int:
        root = _benchmark_root(None)
        records = _load_benchmark(root)
        sessions = _group_by_session(records)
        total = len(sessions)
        env_limit = os.getenv('YB_BENCH_LIMIT')
        if env_limit:
            return min(total, int(env_limit))
        return total

    @staticmethod
    def metadata() -> dict:
        return {
            'description': 'local-mode OneAgentBench evaluation (multi-turn session support)',
            'experiment': 'local_one_agent_infer',
        }


# ── public API (expected by GeneratorLoader) ────────────────────────


def generate_configs(
    input_path: str | None = None,
    base_config_path: str | Path | None = None,
):
    yield from LocalOneAgentInferGenerator(
        input_path=input_path,
        base_config_path=base_config_path,
    )


def get_total_count() -> int:
    return LocalOneAgentInferGenerator.total_count()


def get_batch_metadata() -> dict:
    return LocalOneAgentInferGenerator.metadata()


__all__ = ['generate_configs', 'get_total_count', 'get_batch_metadata']
