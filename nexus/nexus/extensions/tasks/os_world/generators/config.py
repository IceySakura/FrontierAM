"""Shared configuration for the six native AGS entrypoints."""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any

from nexus.core.dep_checker import check_module
from nexus.core.module_meta import parse_directory_module
from nexus.extensions.tasks.os_world.data import task_source


OSWorldBatchTask = task_source.OSWorldBatchTask
_env_int = task_source.env_int
_env_float = task_source.env_float
_env_bool = task_source.env_bool
EVALUATOR_HOST_DEP_CHECK_ENV = 'OSWORLD_VALIDATE_EVALUATOR_HOST_DEPS'
DEFAULT_AGS_INSTANCE_READY_TIMEOUT = 600.0
AGS_SECRET_ID_INTERPOLATION = '${oc.env:AGS_SECRET_ID}'
AGS_SECRET_KEY_INTERPOLATION = '${oc.env:AGS_SECRET_KEY}'
AGS_API_KEY_INTERPOLATION = '${oc.env:AGS_API_KEY}'
AGS_OSWORLD_TOOL_ID_INTERPOLATION = '${oc.env:AGS_OSWORLD_TOOL_ID}'
AGS_OSWORLD_TEMPLATE_INTERPOLATION = '${oc.env:AGS_OSWORLD_TEMPLATE}'
OSWORLD_PROXY_CONFIG_FILE_INTERPOLATION = '${oc.env:OSWORLD_PROXY_CONFIG_FILE}'


def _unsupported_evaluator_policy() -> str:
    value = os.getenv('OSWORLD_UNSUPPORTED_EVALUATOR_POLICY', 'fail')
    if value not in {'skip', 'fail'}:
        raise ValueError("OSWORLD_UNSUPPORTED_EVALUATOR_POLICY must be 'skip' or 'fail'")
    return value


def _proxy_config() -> tuple[bool, bool, str | None]:
    enable_proxy = _env_bool('OSWORLD_ENABLE_PROXY', False)
    force_proxy = _env_bool('OSWORLD_FORCE_PROXY', False)
    proxy_config_file = os.getenv('OSWORLD_PROXY_CONFIG_FILE', '').strip()
    if enable_proxy:
        if not proxy_config_file:
            raise ValueError('OSWORLD_ENABLE_PROXY=true requires OSWORLD_PROXY_CONFIG_FILE to be set')
        if not Path(proxy_config_file).exists():
            raise FileNotFoundError(f'OSWorld proxy config file not found: {proxy_config_file}')
    return enable_proxy, force_proxy, proxy_config_file or None


def _project_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'nexus').is_dir():
            return parent
    return path.parents[3]


def _missing_evaluator_host_deps() -> list[str]:
    module_root = Path(__file__).resolve().parents[1]
    meta = parse_directory_module(module_root, _project_root())
    if meta is None:
        raise RuntimeError(f'Unable to read OSWorld module metadata from {module_root / "pyproject.toml"}')
    return check_module(meta).missing_host


def _validate_evaluator_host_deps() -> None:
    if not _env_bool(EVALUATOR_HOST_DEP_CHECK_ENV, True):
        return

    missing = _missing_evaluator_host_deps()
    if not missing:
        return

    missing_lines = '\n'.join(f'  - {dep}' for dep in missing)
    raise RuntimeError(
        'OSWorld official evaluator host dependencies are missing. The batch generator sets '
        'task_config.run_evaluator=true, so official metric scoring needs the packages declared in '
        'nexus/extensions/tasks/os_world/pyproject.toml [project.optional-dependencies].host.\n'
        f'Missing packages:\n{missing_lines}\n'
        'Install them with:\n'
        '  uv run nexus modules install os_world\n'
        'or:\n'
        "  uv pip install -e 'nexus/extensions/tasks/os_world[host]'\n"
        f'Set {EVALUATOR_HOST_DEP_CHECK_ENV}=0 only for metadata/config debugging without official scoring.'
    )


def _pyautogui_execution_mode() -> str:
    mode = os.getenv('OSWORLD_PYAUTOGUI_EXECUTION_MODE')
    if mode is None or not mode.strip():
        mode = 'run_command'
    mode = mode.strip()
    if mode not in {'gui_pyautogui', 'run_command'}:
        raise ValueError("OSWORLD_PYAUTOGUI_EXECUTION_MODE must be 'gui_pyautogui' or 'run_command'")
    return mode


def gpt_cua_config(task: OSWorldBatchTask) -> dict[str, Any]:
    artifact_root = os.getenv('OSWORLD_ARTIFACT_DIR', '').strip()
    config = {
        'max_iterations': _env_int('OSWORLD_MAX_ITERATIONS', 50) or 50,
        'reasoning_effort': os.getenv('OSWORLD_REASONING_EFFORT', 'xhigh'),
        'client_password': os.getenv('OSWORLD_CLIENT_PASSWORD', 'password'),
        'sleep_after_execution': _env_float('OSWORLD_SLEEP_AFTER_EXECUTION', 0.5),
        'step_pause': _env_float('OSWORLD_STEP_PAUSE', 1.0),
        'pyautogui_execution_mode': _pyautogui_execution_mode(),
        'pyautogui_timeout': _env_float('OSWORLD_PYAUTOGUI_TIMEOUT', 30.0),
    }
    if artifact_root:
        config['artifact_dir'] = str(Path(artifact_root) / task.domain / task.task_id)
        config['record_screenshots'] = _env_bool('OSWORLD_RECORD_SCREENSHOTS', True)
    return config


def _ags_provision_mode() -> str:
    mode = os.getenv('OSWORLD_AGS_PROVISION_MODE', 'control_plane').strip().lower()
    if mode not in {'control_plane', 'template'}:
        raise ValueError("OSWORLD_AGS_PROVISION_MODE must be 'control_plane' or 'template'")
    return mode


def _runtime_provider_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        'domain': os.getenv('AGS_DOMAIN', 'ap-guangzhou.tencentags.com'),
        'interface_mode': os.getenv('OSWORLD_AGS_INTERFACE_MODE', 'nexus'),
        'osworld_cdp_proxy_mode': os.getenv('OSWORLD_AGS_CDP_PROXY_MODE', 'auto'),
        'request_timeout': _env_float('AGS_REQUEST_TIMEOUT', 1800.0),
        'instance_ready_timeout': _env_float(
            'OSWORLD_AGS_INSTANCE_READY_TIMEOUT',
            DEFAULT_AGS_INSTANCE_READY_TIMEOUT,
        ),
    }
    if _ags_provision_mode() == 'template':
        config['api_key'] = AGS_API_KEY_INTERPOLATION
    else:
        config.update({
            'secret_id': AGS_SECRET_ID_INTERPOLATION,
            'secret_key': AGS_SECRET_KEY_INTERPOLATION,
            'region': os.getenv('AGS_REGION', 'ap-guangzhou'),
        })
    return config


def _runtime_configs() -> dict[str, Any]:
    runtime_config: dict[str, Any] = {
        'runtime_kind': 'osworld',
        'timeout': _env_int('OSWORLD_SANDBOX_TIMEOUT', 2400) or 2400,
        'tunnel_port': _env_int('OSWORLD_TUNNEL_PORT', 32000) or 32000,
    }
    if _ags_provision_mode() == 'template':
        runtime_config['template'] = AGS_OSWORLD_TEMPLATE_INTERPOLATION
    else:
        runtime_config.update({
            'tool_type': 'osworld',
            'tool_id': AGS_OSWORLD_TOOL_ID_INTERPOLATION,
            'skip_tool_lookup': _env_bool('OSWORLD_AGS_SKIP_TOOL_LOOKUP', True),
        })
    return {'main': runtime_config}


def build_task_config(task: OSWorldBatchTask) -> dict[str, Any]:
    """Task and proxy settings shared by the native model/harness generators."""
    enable_proxy, force_proxy, _proxy_config_file = _proxy_config()
    task_config: dict[str, Any] = {
        'domain': task.domain,
        'task_id': task.task_id,
        'observation_type': 'screenshot',
        'action_space': 'pyautogui',
        'run_setup': True,
        'run_evaluator': True,
        'unsupported_evaluator_policy': _unsupported_evaluator_policy(),
    }
    if enable_proxy:
        task_config['enable_proxy'] = True
        task_config['force_proxy'] = force_proxy
        task_config['proxy_config_file'] = OSWORLD_PROXY_CONFIG_FILE_INTERPOLATION
    task_config['skip_gui_start'] = _env_bool('OSWORLD_SKIP_GUI_START', True)
    task_config['command_as_user'] = _env_bool('OSWORLD_COMMAND_AS_USER', False)
    return task_config


def claude_cua_config() -> dict[str, Any]:
    """Build only the CUA behavior; model routing belongs to the client."""
    return {
        'max_iterations': _env_int('OSWORLD_MAX_ITERATIONS', 100) or 100,
        'max_tokens': _env_int('OSWORLD_LLM_MAX_TOKENS', 4096) or 4096,
        'transport': 'client',
        'client_name': 'main',
        'api_style': os.getenv('OSWORLD_ANTHROPIC_API_STYLE', 'messages'),
        'bedrock_anthropic_version': os.getenv('OSWORLD_ANTHROPIC_BEDROCK_VERSION', 'bedrock-2023-05-31'),
        'screen_width': _env_int('OSWORLD_SCREEN_WIDTH', 1920) or 1920,
        'screen_height': _env_int('OSWORLD_SCREEN_HEIGHT', 1080) or 1080,
        'model_display_width': _env_int('ANTHROPIC_MODEL_DISPLAY_WIDTH', 1280) or 1280,
        'model_display_height': _env_int('ANTHROPIC_MODEL_DISPLAY_HEIGHT', 720) or 720,
        'only_n_most_recent_images': _env_int('OSWORLD_MAX_IMAGE_HISTORY', 8) or 8,
        'no_thinking': _env_bool('ANTHROPIC_NO_THINKING', False),
        'effort': os.getenv('ANTHROPIC_EFFORT', 'max'),
        'use_isp': _env_bool('ANTHROPIC_USE_ISP', False),
        'temperature': _env_float('OSWORLD_LLM_TEMPERATURE', 1.0),
        'client_password': os.getenv('OSWORLD_CLIENT_PASSWORD', ''),
        'sleep_after_execution': _env_float('OSWORLD_SLEEP_AFTER_EXECUTION', 3.0),
        'step_pause': _env_float('OSWORLD_STEP_PAUSE', 1.0),
        'pyautogui_execution_mode': _pyautogui_execution_mode(),
        'pyautogui_timeout': _env_float('OSWORLD_PYAUTOGUI_TIMEOUT', 30.0),
        'api_retry_times': _env_int('OSWORLD_ANTHROPIC_RETRY_TIMES', 5) or 5,
    }
