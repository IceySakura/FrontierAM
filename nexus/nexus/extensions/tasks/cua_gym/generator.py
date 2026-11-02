"""CUA-Gym public train split batch generator for ``nexus run-batch``.

遍历官方 CUA-Gym public 训练集 task bundle，按需预筛后产出 BatchItem，
用统一镜像 + gongfeng provider 在 Nexus 上批量执行。

用法：
  nexus run-batch -g nexus/extensions/tasks/cua_gym/generator.py \
    -d /data/cua_runs/smoke.db --max-concurrent 4 --export-format jsonl

环境变量：
  数据集
  - CUA_GYM_TASK_ROOT       解压后的 bundle 根目录（默认 /data/cua_full）
  - CUA_GYM_ONLY_SUPPORTED  只保留 setup 全是 download/execute 的题（默认 true）
  - CUA_GYM_LIMIT           最多产出多少题（默认 10；空/0 表示不限）
  - CUA_GYM_SKIP            跳过前 N 题（默认 0）
  - CUA_GYM_TASK_IDS        逗号分隔，只跑指定 task_id（默认全部）
  - CUA_GYM_APP_TYPES       逗号分隔，只跑指定 app_type（默认全部）

  模型服务（Azure OpenAI 透传网关）
  - MODEL_SERVICE_API_BASE_URL  默认 http://api.model-eval.woa.com/openai/v1
  - MODEL_SERVICE_API_APP_ID
  - MODEL_SERVICE_API_APP_KEY
  - NEXUS_MODEL                 默认 gpt-5.5
  - MODEL_SERVICE_API_TIMEOUT   默认 60

  gongfeng runtime provider
  - GONGFENG_API_BASE_URL       默认 http://hyrl-sandbox.prod.woa.com:8052
  - GONGFENG_TOKEN
  - GONGFENG_RUNTIME_PROVIDER   默认 gongfeng
  - NEXUS_GONGFENG_RUNTIME_IMAGE 默认 current
  - NEXUS_GONGFENG_CUA_GYM_IMAGE 统一 session 镜像
        默认 mirrors.tencent.com/hunyuan_yanguan/nexus-cua-gym-session:latest

  agent / task
  - CUA_GYM_MAX_ITERATIONS      默认 80
  - CUA_GYM_IMAGE_HISTORY_LIMIT 默认 6
  - CUA_GYM_ACTION_SPACE        默认 pyautogui
  - CUA_GYM_UNSUPPORTED_POLICY  默认 fail
"""

from __future__ import annotations

import json
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus import BatchItem
from nexus.runner.single import RunInput


SUPPORTED_SETUP_ACTIONS = {'download', 'execute', 'command', 'launch', 'open', 'activate_window', 'sleep'}
DEFAULT_TASK_ROOT = '/data/cua_full'
DEFAULT_MODEL_BASE_URL = 'http://api.model-eval.woa.com/openai/v1'
DEFAULT_GONGFENG_BASE_URL = 'http://hyrl-sandbox.prod.woa.com:8052'
DEFAULT_IMAGE = 'mirrors.tencent.com/hunyuan_yanguan/nexus-cua-gym-session:latest'


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _env_int(name: str, default: int | None) -> int | None:
    v = os.getenv(name)
    if v is None or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_list(name: str) -> set[str] | None:
    v = os.getenv(name)
    if not v:
        return None
    items = {item.strip() for item in v.split(',') if item.strip()}
    return items or None


def _task_root() -> Path:
    return Path(os.getenv('CUA_GYM_TASK_ROOT', DEFAULT_TASK_ROOT))


def _only_supported(types: list[str]) -> bool:
    return bool(types) and all(t in SUPPORTED_SETUP_ACTIONS for t in types)


def _iter_selected() -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (task_id, task_json) for tasks passing all filters, in sorted order."""
    root = _task_root()
    if not root.is_dir():
        raise FileNotFoundError(f'CUA-Gym task root not found: {root}')

    only_supported = _env_bool('CUA_GYM_ONLY_SUPPORTED', True)
    skip = _env_int('CUA_GYM_SKIP', 0) or 0
    limit = _env_int('CUA_GYM_LIMIT', 10)
    id_filter = _env_list('CUA_GYM_TASK_IDS')
    app_filter = _env_list('CUA_GYM_APP_TYPES')

    skipped = 0
    produced = 0
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        task_json = task_dir / 'task.json'
        if not task_json.is_file():
            continue
        try:
            data = json.loads(task_json.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue

        task_id = str(data.get('task_id') or data.get('id') or task_dir.name)
        app_type = str(data.get('app_type') or data.get('application') or data.get('app') or '')
        types = [str(a.get('type', '')) for a in (data.get('config') or [])]

        if id_filter is not None and task_id not in id_filter:
            continue
        if app_filter is not None and app_type not in app_filter:
            continue
        if only_supported and not _only_supported(types):
            continue
        if skipped < skip:
            skipped += 1
            continue

        yield task_id, data
        produced += 1
        if limit and produced >= limit:
            return


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """构建模型客户端，按 MODEL_PROVIDER 切换 azure_openai(gpt) 或 aws_third(claude bedrock)。

    - azure_openai：openai 兼容 client，透传网关 /openai/v1，model 如 gpt-5.5
    - aws_third：litellm client，litellm 自动按 provider=aws_third 映射成 bedrock，
      model 形如 anthropic.claude-opus-4-8，base_url 指向 bedrock invoke 网关根
    """
    app_id = os.getenv('MODEL_SERVICE_API_APP_ID', '')
    app_key = os.getenv('MODEL_SERVICE_API_APP_KEY', '')
    provider = os.getenv('MODEL_PROVIDER', 'azure_openai')
    model = os.getenv('NEXUS_MODEL', 'gpt-5.5')
    timeout = os.getenv('MODEL_SERVICE_API_TIMEOUT', '120')

    if provider == 'aws_third':
        base_url = os.getenv('MODEL_SERVICE_API_BASE_URL', 'http://api.model-eval.woa.com')
        api_key = f'{app_id}:{app_key}?provider=aws_third&model={model}&timeout={timeout}'
        return {
            'main': {
                'name': 'litellm',
                'config': {
                    'client_args': {
                        'api_key': api_key,
                        'base_url': base_url,
                        # claude prompt caching：image_window 策略——断点打在首张图之前的
                        # 稳定文本前缀边界（而非滑动图片尾部）。滑动窗口每轮把旧图改写成占位
                        # 文本，若按 last_n 把断点打在图片之后，缓存前缀会在改写点分叉、保留的图
                        # 每轮重复写缓存（cache_creation 飙升、几乎无 read）。打在图片之前则让
                        # system+tools+文本历史稳定命中，图片尾部按普通输入计费（省去 1.25x 写）。
                        # 仅改 cache_control 元数据，不改消息内容，reward 不受影响。
                        'cache_control_enabled': True,
                        'cache_control_strategy': 'nexus.extensions.tasks.cua_gym.cache::image_window',
                    },
                    'request_args': {'model': model},
                },
            }
        }

    base_url = os.getenv('MODEL_SERVICE_API_BASE_URL', DEFAULT_MODEL_BASE_URL)
    api_key = f'{app_id}:{app_key}?provider=azure_openai&model={model}&timeout={timeout}'
    return {
        'main': {
            'name': 'openai',
            'config': {
                'client_args': {'api_key': api_key, 'base_url': base_url},
                'request_args': {'model': model},
            },
        }
    }


def _build_config(task_id: str) -> dict[str, Any]:
    action_space = os.getenv('CUA_GYM_ACTION_SPACE', 'pyautogui')
    unsupported_policy = os.getenv('CUA_GYM_UNSUPPORTED_POLICY', 'fail')
    max_iterations = _env_int('CUA_GYM_MAX_ITERATIONS', 80)
    image_history_limit = _env_int('CUA_GYM_IMAGE_HISTORY_LIMIT', 6)

    return {
        'imports': ['nexus.extensions.tasks.cua_gym'],
        'task_name': 'cua_gym',
        'task_config': {
            'task_root': str(_task_root()),
            'task_id': task_id,
            'run_setup': True,
            'run_evaluator': True,
            'unsupported_setup_policy': unsupported_policy,
            'require_done': True,
            'action_space': action_space,
            'mock_hub_mode': 'local',
            'mock_hub_root': os.getenv('CUA_GYM_HUB_ROOT', '/opt/cua-gym-hub'),
        },
        'runtime_provider_name': 'gongfeng',
        'runtime_provider_config': {
            'api_base_url': os.getenv('GONGFENG_API_BASE_URL', DEFAULT_GONGFENG_BASE_URL),
            'token': os.getenv('GONGFENG_TOKEN', ''),
            'runtime_image': os.getenv('NEXUS_GONGFENG_RUNTIME_IMAGE', 'current'),
            'runtime_provider': os.getenv('GONGFENG_RUNTIME_PROVIDER', 'gongfeng'),
            'resource_spec': {'cpu': 4, 'memory': 16},
            'queue_wait_max': _env_int('GONGFENG_QUEUE_WAIT_MAX', 1800) or 0,
            'print_port_mappings': True,
        },
        'runtime_configs': {
            'main': {
                'image': os.getenv('NEXUS_GONGFENG_CUA_GYM_IMAGE', DEFAULT_IMAGE),
                'exposed_ports': [5900, 6080, 18000, 18001, 18002],
                'environment': {
                    'GTK_A11Y': 'none',
                    'NEXUS_GUI_REQUIRE_SYSTEM_DBUS': '1',
                },
            }
        },
        'agent_name': 'main_agent',
        'agent_config': {
            'max_iterations': max_iterations,
            'parallel_tool_calls': False,
            'image_history_limit': image_history_limit,
            'tools': ['desktop/screenshot', 'desktop/pyautogui', 'desktop/fixed_marker'],
        },
        'openai_clients': _build_openai_clients(),
        'processors': {
            'chat': {'name': 'chat_tracker'},
            'agent_metrics': {'name': 'main_agent_metrics'},
        },
    }


def generate_configs() -> Iterator[BatchItem]:
    for task_id, data in _iter_selected():
        app_type = str(data.get('app_type') or data.get('application') or data.get('app') or '')
        types = [str(a.get('type', '')) for a in (data.get('config') or [])]
        yield BatchItem(
            config=RunInput.model_validate(_build_config(task_id)),
            run_id=task_id,
            metadata={
                'task_id': task_id,
                'app_type': app_type,
                'provider': 'gongfeng',
                'setup_action_types': types,
            },
        )


def get_total_count() -> int:
    return sum(1 for _ in _iter_selected())


def get_batch_metadata() -> dict[str, Any]:
    return {
        'task_name': 'cua_gym',
        'task_root': str(_task_root()),
        'only_supported': _env_bool('CUA_GYM_ONLY_SUPPORTED', True),
        'model': os.getenv('NEXUS_MODEL', 'gpt-5.5'),
        'image': os.getenv('NEXUS_GONGFENG_CUA_GYM_IMAGE', DEFAULT_IMAGE),
    }
