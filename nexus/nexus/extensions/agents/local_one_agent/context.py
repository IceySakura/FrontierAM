"""Data models for local_one_agent context."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class LocalWorkspace(BaseModel):
    """Session-level workspace config. Immutable after session creation."""

    path: str
    scope: Literal['conversation']


class LocalPermission(BaseModel):
    """Conversation-level permission. May change between rounds.

    两级权限（Part Ⅰ §1）：删除 ``full``，仅保留 ``readonly`` / ``basic``。
    ``previous_level`` 记录上一轮权限，用于让模型感知「从 X 切到 Y」的切换动作
    （现虽只有两级，后续多级时该记录作用更大）。 首轮为 ``None``。
    """

    level: Literal['readonly', 'basic']
    previous_level: Literal['readonly', 'basic'] | None = None


class LocalRuntime(BaseModel):
    """Session-level runtime environment. Fixed at session creation.

    ``os`` 只保留用户真实系统 ``macos`` / ``windows``。 训练在 Linux 容器里用
    bwrap 模拟 macos 沙箱（见 sandbox.py），``linux`` 不是用户可选 OS。
    """

    os: Literal['macos', 'windows']
    arch: Literal['arm64', 'x86_64']
    shell: Literal['zsh', 'bash', 'powershell', 'cmd']
    default_browser: str | None = None


class UserDecisionRule(BaseModel):
    """A single rule for simulating user decisions in training.

    ``decision`` 是训练侧 mock 值（非 wire 协议）：``confirmed`` 对应线上放行、
    ``rejected`` 对应线上 ``AuthAnswer=deny`` / for_model 的 ``USER_REJECTED``。
    """

    pattern: str
    decision: Literal['confirmed', 'rejected']


class PermissionEngineConfig(BaseModel):
    """Configuration for permission engine's user decision simulation."""

    user_decisions: list[UserDecisionRule] = []


class LocalEnv(BaseModel):
    """训练环境依赖预置（训推一致）。

    真实机器只保证 ``python`` + ``node`` **runtime**，library（Python=requirements.txt、
    Node=package.json）按需装。 用 ``mode`` 一个开关兼顾冷 / 暖启动：

    - ``cold``（默认）：只有 runtime、库未装。 清单文件（若已在 workspace 里）由模型自己
      ``pip install -r`` / ``npm install`` 拉起（走沙箱 install-escape 放网）。 冷启动信号，
      最贴近真实裸机。
    - ``warm``：task 启动前由 harness 预装（``provision``），模型进来 ``import`` 直接可用。
      模拟「用户已完成安装」。

    ``python_requirements`` / ``node_manifest``：**相对 workspace 的清单文件路径**（可选，
    不内联包名）。 仅 ``warm`` 时用于预装；``cold`` 不预装（清单作为 workspace 数据存在即可）。
    """

    mode: Literal['cold', 'warm'] = 'cold'
    python_requirements: str | None = None
    node_manifest: str | None = None
    # 仅 Windows/WAA 用：本地主机目录，provisioning 会把其内文件（保留相对结构）上传到 VM
    # 的 workspace，用于把清单（requirements.txt / package.json）等 workspace 数据 seed 进 VM，
    # 再由 warm 安装。 macos/linux（workspace 即本地）不用，缺省 None 不做任何事。
    local_seed_dir: str | None = None


def resolve_workspace_path(task_config: dict) -> str:
    """从 task_config 解析工作区绝对路径（单一配置源）。

    优先级：``local_context.workspace.path`` > 遗留 ``workspace_dir`` > ``/workspace``。
    ``local_one_agent`` 与 ``local_one_agent_infer`` 均通过此函数对齐路径，避免两处重复配置。
    """
    local_context = task_config.get('local_context') or {}
    workspace = local_context.get('workspace') or {}
    path = workspace.get('path')
    if path:
        return str(path)
    legacy = task_config.get('workspace_dir')
    if legacy:
        return str(legacy)
    return '/workspace'


class LocalContext(BaseModel):
    """Top-level context for local_one_agent. Separate from agent_config."""

    workspace: LocalWorkspace
    permission: LocalPermission
    runtime: LocalRuntime
    permission_engine: PermissionEngineConfig | None = None
    # skill 路径（Part Ⅰ §7）。load_skill / SP 的 skill_dir 都用它。
    # macos/linux 模拟：默认烤入镜像的路径，可不配（load_skill 回退到默认 skill 目录）。
    # windows：路径不同，需显式配置。
    skill_path: str | None = None
    # 环境依赖预置（cold/warm）。 默认 cold：只有 runtime、库按需装。
    env: LocalEnv = LocalEnv()
    # Windows OS 级沙箱（ybsbr）开关。 默认开:命令在 VM 内被 ybsbr 包装隔离（探测不到 ybsbr 时
    # 自动降级为仅权限引擎）。 置 False 可显式关闭沙箱（仅权限引擎）。 macos/linux 忽略此字段（走 bwrap）。
    windows_sandbox: bool = True
    # SP「用户信息」区渲染用：当前日期与用户位置（generator 逐 session 从 bench
    # 记录的 time / location 填充）。 缺省为空字符串，纯文本 SP 不受影响。
    cur_date: str = ''
    user_location: str = ''
