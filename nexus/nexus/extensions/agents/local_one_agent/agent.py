"""local_one_agent — agent entry point for local workspace scenarios.

Reuses main_agent's inference loop structure with LocalToolExecutor substituted in.
The key differences from main_agent:
1. Uses LocalToolExecutor (permission engine + sandbox + local execution)
2. Reads LocalContext from task_config['local_context']
3. Injects local context info into system prompt
"""

import asyncio
import json
import logging
import ntpath
import os
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from nexus.core import (
    ContentPartParam,
    ContextLengthExceededError,
    EntryNotFoundError,
    agent_registry,
    import_from_path,
    require_context,
    tool_registry,
    validate_history,
)
from nexus.core.tool import ToolArgumentValidationError, ToolNameResolutionError
from nexus.extensions.agents.local_one_agent.context import LocalContext, LocalPermission
from nexus.extensions.agents.local_one_agent.executor import LocalToolExecutor
from nexus.extensions.agents.local_one_agent.waa_backend import WaaBackend
from nexus.extensions.agents.local_one_agent.windows_sandbox import WindowsSandbox
from nexus.extensions.agents.main.agent import (
    MainAgentInput,
    MainAgentOutput,
    MainExitStatus,
)
from nexus.extensions.agents.main.utils import (
    extract_tool_output_content,
    extract_tool_output_text,
    fmt_msg,
    log_messages,
    maybe_truncate,
)
from nexus.extensions.agents.yb_one_agent.llm_errors import is_llm_http_error
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.executors.models.command import CommandRunStatus
from nexus.trace import get_current_span_id, span_context, update_span
from nexus.utils.timeout import InferenceTimeoutError, TimeBudget, await_completion_with_time_budget, with_timeout
from nexus.utils.tool_schema_overrides import load_tool_schema_overrides


logger = logging.getLogger(__name__)

# Skill tools: always registered on the executor (so load_skill-unlocked calls
# resolve), but excluded from the model ``tools`` parameter — same split as
# yb_one_agent ``_make_executor`` / ``build_tool_schemas``.
_SKILL_TOOL_NAMES: tuple[str, ...] = (
    'yb_one_agent/image_generation/generate',
    'yb_one_agent/task_schedule/create_task',
    'yb_one_agent/task_schedule/list_tasks',
    'yb_one_agent/task_schedule/update_task',
    'yb_one_agent/task_schedule/delete_task',
)
_SKILL_TOOL_NAME_SET: frozenset[str] = frozenset(_SKILL_TOOL_NAMES)


def _ensure_skill_tools_registered() -> None:
    """Register skill tools (real/mock), matching yb_one_agent.__init__."""
    if any(name in tool_registry for name in _SKILL_TOOL_NAMES):
        return

    from nexus.core import Registry

    image_real = os.environ.get('IMAGE_GEN_USE_REAL', '').lower() in ('1', 'true', 'yes')
    schedule_real = os.environ.get('TASK_SCHEDULE_USE_REAL', '').lower() in ('1', 'true', 'yes')
    Registry.import_module(
        'nexus.extensions.agents.yb_one_agent.skill_tools.image_generation.real_tools'
        if image_real
        else 'nexus.extensions.agents.yb_one_agent.skill_tools.image_generation.mock_tools'
    )
    Registry.import_module(
        'nexus.extensions.agents.yb_one_agent.skill_tools.task_schedule.real_tools'
        if schedule_real
        else 'nexus.extensions.agents.yb_one_agent.skill_tools.task_schedule.mock_tools'
    )


def _make_local_tool_executor(
    configured_tools: list[str],
    *,
    local_context: LocalContext,
    dynamic_tool_names: set[str],
) -> LocalToolExecutor:
    """Build LocalToolExecutor with configured tools + skill tools.

    Skill tools are merged in (yb_one_agent ``_make_executor`` style) so they remain
    callable after ``load_skill`` even when omitted from agent_config.tools.
    """
    _ensure_skill_tools_registered()
    tool_names = list(dict.fromkeys([*configured_tools, *_SKILL_TOOL_NAMES]))

    tool_funcs = []
    for tool_name in tool_names:
        if tool_name in dynamic_tool_names:
            continue
        try:
            tool_funcs.append(tool_registry.get(tool_name))
        except EntryNotFoundError as exc:
            raise EntryNotFoundError(
                f"Tool '{tool_name}' not found in registry or context-loaded MCP tools. "
                f'Context MCP tools: {sorted(dynamic_tool_names)}'
            ) from exc

    return LocalToolExecutor(
        tool_funcs,
        local_context=local_context,
        allowed_tool_names=tool_names,
    )


class LocalOneAgentInput(MainAgentInput):
    """Input for local_one_agent.

    Inherits ``history`` from ``MainAgentInput`` (``WithHistory``). Adds
    ``resume_from_history`` for resume-trajectory benches where the partial
    trajectory already ends at the continuation point (e.g. a tool result).
    """

    resume_from_history: bool = Field(
        default=False,
        description=(
            'When True, `history` already ends with the message the agent '
            'should continue from (e.g. a tool result), so NO new user message '
            'is appended — the loop resumes directly from `history`. '
            '`user_prompt` is ignored in this mode.'
        ),
    )

    # local_one_agent 覆盖 main 的默认值：默认容错继续。 一次幻觉出的工具参数 / 校验
    # 失败 / 执行异常，会作为 tool observation 喂回模型让其自我纠正，而不是终止整条
    # 轨迹（对 RL / 评测更友好）。 仍可用 ``tool_error_continue=False`` 或
    # ``NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE`` 环境变量显式关闭（env 优先级最高）。
    tool_error_continue: bool = Field(
        default=True,
        description=(
            'On an unrecoverable tool-call error (unparseable args, failed '
            'name/arg validation, or tool execution exception) feed the error '
            'back to the model as a tool observation and continue the loop, '
            'instead of aborting the run. Defaults to True for local_one_agent; '
            'overridable via the NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE env var.'
        ),
    )


def _render_system_prompt(system_prompt: str, local_ctx: LocalContext) -> str:
    """用 LocalContext 把 system_prompt 当 Jinja2 模板渲染（Part Ⅰ §2）。

    注入变量：``working_dir`` / ``local_runtime_os`` / ``local_runtime_arch`` /
    ``local_runtime_shell`` / ``local_permission_level`` / ``local_previous_permission_level`` /
    ``skill_dir`` / ``cur_date`` / ``user_location``。
    若 system_prompt 不含模板变量（纯文本），渲染结果即原文，不影响既有配置。
    """
    import jinja2

    working_dir = local_ctx.workspace.path
    skill_dir = local_ctx.skill_path or os.environ.get('YB_SKILL_PATH') or '/data/skills'
    variables = {
        'working_dir': working_dir,
        'local_runtime_os': local_ctx.runtime.os,
        'local_runtime_arch': local_ctx.runtime.arch,
        'local_runtime_shell': local_ctx.runtime.shell,
        'local_permission_level': local_ctx.permission.level,
        'local_previous_permission_level': local_ctx.permission.previous_level or '',
        'skill_dir': skill_dir,
        'cur_date': local_ctx.cur_date,
        'user_location': local_ctx.user_location,
    }
    try:
        template = jinja2.Template(system_prompt, undefined=jinja2.Undefined)
        return template.render(**variables)
    except jinja2.TemplateError as exc:
        logger.warning('system_prompt Jinja2 渲染失败，按原文使用: %s', exc)
        return system_prompt


# Unix-style（macos / bash|zsh）高风险示例
_PERMISSION_SWITCH_REMINDERS = {
    'readonly': (
        '当前用户切换到只读模式：你只能使用 READ 相关的工具（如读取文件、列目录、查看信息、检索内容等），'
        '不能创建、修改、删除文件，也不能执行任何有副作用的命令。'
        '即使加载了 skill，也只能使用其中的只读操作，不要按 skill 里的步骤去创建或编辑文件。'
    ),
    'basic': (
        '当前用户切换到编辑模式，现在可以读写文件、执行安全命令；'
        '超出工作区或高风险操作（如 rm -rf、git push）仍会触发用户确认或被拒绝。'
    ),
}


_INITIAL_MODE_REMINDERS = {
    'readonly': (
        '当前处于只读模式：你只能使用 READ 相关的工具（如读取文件、列目录、查看信息、检索内容等），'
        '不能创建、修改、删除文件，也不能执行任何有副作用的命令。'
        '即使加载了 skill，也只能使用其中的只读操作，不要按 skill 里的步骤去创建或编辑文件。'
    ),
    'basic': (
        '当前处于编辑模式，可以读写文件、执行安全命令；'
        '超出工作区或高风险操作（如 rm -rf、git push）仍会触发用户确认或被拒绝。'
    ),
}


# Windows PowerShell 风格高风险示例（``os == windows && shell == powershell``）
_PERMISSION_SWITCH_REMINDERS_POWERSHELL = {
    'readonly': (
        '当前用户切换到只读模式：你只能使用 READ 相关的工具（如读取文件、列目录、查看信息、检索内容等），'
        '不能创建、修改、删除文件，也不能执行任何有副作用的命令。'
        '即使加载了 skill，也只能使用其中的只读操作，不要按 skill 里的步骤去创建或编辑文件。'
    ),
    'basic': (
        '当前用户切换到编辑模式，现在可以读写文件、执行安全命令；'
        '超出工作区或高风险操作（如 Remove-Item -Recurse -Force、git push）仍会触发用户确认或被拒绝。'
    ),
}


_INITIAL_MODE_REMINDERS_POWERSHELL = {
    'readonly': (
        '当前处于只读模式：你只能使用 READ 相关的工具（如读取文件、列目录、查看信息、检索内容等），'
        '不能创建、修改、删除文件，也不能执行任何有副作用的命令。'
        '即使加载了 skill，也只能使用其中的只读操作，不要按 skill 里的步骤去创建或编辑文件。'
    ),
    'basic': (
        '当前处于编辑模式，可以读写文件、执行安全命令；'
        '超出工作区或高风险操作（如 Remove-Item -Recurse -Force、git push）仍会触发用户确认或被拒绝。'
    ),
}


def _wrap_system_reminder(text: str) -> str:
    """把一段文本包成 <system-reminder> 标签。

    对齐 CC / OpenCode / Codex 的范式：由 harness 注入的系统指令用该标签承载，
    出现在上下文末尾时权重高、模型指令跟随强。 通用 helper，后续 hook / tool result 可复用。
    """
    return f'<system-reminder>\n{text}\n</system-reminder>'


def _append_system_reminder(
    user_content: str | list[ContentPartParam],
    reminder: str | None,
) -> str | list[ContentPartParam]:
    """把 reminder 拼到 user 消息尾部（不改动原始入参）。

    - ``reminder`` 为 None → 原样返回
    - 字符串 content → 用空行拼接
    - 多模态 list content → 追加一个 text part（返回副本，不 mutate 原 list）
    """
    if reminder is None:
        return user_content
    if isinstance(user_content, str):
        return f'{user_content}\n\n{reminder}'
    return [*user_content, {'type': 'text', 'text': reminder}]


def _is_windows_powershell(local_ctx: LocalContext) -> bool:
    """``os == windows`` 且 ``shell == powershell`` 时使用 PowerShell 风格 reminder。"""
    return local_ctx.runtime.os == 'windows' and local_ctx.runtime.shell == 'powershell'


def _permission_switch_reminders_for(local_ctx: LocalContext) -> dict[str, str]:
    if _is_windows_powershell(local_ctx):
        return _PERMISSION_SWITCH_REMINDERS_POWERSHELL
    return _PERMISSION_SWITCH_REMINDERS


def _initial_mode_reminders_for(local_ctx: LocalContext) -> dict[str, str]:
    if _is_windows_powershell(local_ctx):
        return _INITIAL_MODE_REMINDERS_POWERSHELL
    return _INITIAL_MODE_REMINDERS


def _build_permission_switch_reminder(perm: LocalPermission, local_ctx: LocalContext) -> str | None:
    """权限相比上一轮发生切换时，返回对应提示文案；否则返回 None。"""
    if perm.previous_level is None or perm.previous_level == perm.level:
        return None
    return _permission_switch_reminders_for(local_ctx).get(perm.level)


def _build_initial_mode_reminder(perm: LocalPermission, local_ctx: LocalContext) -> str | None:
    """首轮（无历史权限）时，返回当前 Mode 的描述文案；否则返回 None。

    首个 user 消息也需要一个 system reminder 告知模型当前所处的 Mode，
    与后续的「权限切换」提示（``_build_permission_switch_reminder``）互补：
    前者描述初始状态，后者描述状态变更。
    """
    if perm.previous_level is not None:
        return None
    return _initial_mode_reminders_for(local_ctx).get(perm.level)


def _build_mode_reminder(perm: LocalPermission, local_ctx: LocalContext) -> str | None:
    """为首个 user 消息挑选合适的 Mode 提示：首轮给初始描述，后续给切换提示。

    ``os == windows && shell == powershell`` 时使用 PowerShell 风格高风险示例文案。
    """
    return _build_initial_mode_reminder(perm, local_ctx) or _build_permission_switch_reminder(perm, local_ctx)


def _resolve_ws_path(workspace_path: str, rel_or_abs: str) -> str:
    """把相对 workspace 的清单路径解析成绝对路径。"""
    return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(workspace_path, rel_or_abs)


async def _run_provision(cmd: list[str], cwd: str, label: str) -> None:
    """跑一条预置命令；失败只告警不中断（保持训练可继续，行为贴近真实）。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        tail = (out or b'').decode('utf-8', errors='replace')[-800:]
        if proc.returncode == 0:
            logger.debug('env warm 预置成功 [%s]: %s', label, ' '.join(cmd))
        else:
            logger.warning('env warm 预置失败 [%s] rc=%s: %s\n%s', label, proc.returncode, ' '.join(cmd), tail)
    except FileNotFoundError:
        logger.warning('env warm 预置跳过 [%s]：命令不可用 %s', label, cmd[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning('env warm 预置异常 [%s]: %s', label, exc)


async def _provision_env(local_ctx: LocalContext) -> None:
    """warm 模式下在 agent loop 前预装依赖，让模型 import 直接可用（训推一致）。

    - Python：全局 ``python3 -m pip install -r <requirements>``（模型的 ``python3`` 直接见到；
      不用 venv 以保持透明——真实训练每个 rollout 一个新容器，全局装无串味问题）。
    - Node：清单所在目录 ``npm install``（``node_modules`` 天然落 workspace）。
    cold 模式不预装（清单作为 workspace 数据存在即可，模型按需自行安装）。

    windows：走 WAA VM 预置（node runtime 上传 + skill 上传 + warm 装库），见 _provision_env_windows。
    """
    if local_ctx.runtime.os == 'windows':
        await _provision_env_windows(local_ctx)
        return

    env = local_ctx.env
    if env is None or env.mode != 'warm':
        return

    ws = local_ctx.workspace.path

    if env.python_requirements:
        req = _resolve_ws_path(ws, env.python_requirements)
        if os.path.isfile(req):
            await _run_provision(['python3', '-m', 'pip', 'install', '-r', req], cwd=ws, label='pip')
        else:
            logger.warning('env warm：python_requirements 不存在，跳过预装: %s', req)

    if env.node_manifest:
        manifest = _resolve_ws_path(ws, env.node_manifest)
        if os.path.isfile(manifest):
            await _run_provision(['npm', 'install'], cwd=os.path.dirname(manifest), label='npm')
        else:
            logger.warning('env warm：node_manifest 不存在，跳过预装: %s', manifest)


# ═══════════════ Windows / WAA provisioning ═══════════════

# node / python runtime 由 WAA VM 镜像预装（node 实测 v24.x 在系统 PATH），故不再上传/解压 runtime。


def _resolve_ws_path_win(workspace_path: str, rel_or_abs: str) -> str:
    """把相对 workspace 的清单路径解析成 VM 绝对路径（ntpath 语义）。"""
    raw = rel_or_abs.replace('/', '\\')
    return raw if ntpath.isabs(raw) else ntpath.join(workspace_path.replace('/', '\\'), raw)


async def _run_provision_vm(backend: WaaBackend, command: str, label: str) -> None:
    """在 VM 内跑一条预置命令（PowerShell）；失败只告警不中断。"""
    try:
        resp = await backend.run_command(command)
    except Exception as exc:  # noqa: BLE001  预置失败不阻断训练
        logger.warning('windows provisioning 异常 [%s]: %s', label, exc)
        return
    if resp.status == CommandRunStatus.Finished and resp.return_code in (0, None):
        logger.debug('windows provisioning 成功 [%s]', label)
    else:
        tail = (resp.stderr or resp.stdout or '')[-800:]
        logger.warning('windows provisioning 失败 [%s] status=%s rc=%s: %s', label, resp.status, resp.return_code, tail)


async def _provision_skill_windows(backend: WaaBackend, skill_path: str) -> None:
    r"""把本地烤入的 skill 目录逐文件上传到 VM 的 skill_path（保留相对结构）。

    当前 windows 中的 skill_path 路径为：
        * C:\Users\82015\AppData\Roaming\Tencent\Yuanbao\Global\workspace\skills\
        * %AppData%\Tencent\Yuanbao\Global\workspace\skills
    """
    # 将镜像内部的 skills 目录上传到 VM 的 skill_path 中
    source = Path('/data/skills/')
    if not source.is_dir():
        logger.warning('windows provisioning：本地 skill 源不存在，跳过: %s', source)
        return

    files = [p for p in source.rglob('*') if p.is_file()]
    if not files:
        logger.debug('windows provisioning：skill 源为空，跳过: %s', source)
        return

    uploaded = 0
    for f in files:
        rel = f.relative_to(source)
        vm_target = ntpath.join(skill_path, *rel.parts)
        try:
            await backend.upload_file(vm_target, f.read_bytes())
        except Exception as exc:  # noqa: BLE001  单文件失败不阻断其余
            logger.warning('windows provisioning：skill 上传失败 %s: %s', vm_target, exc)
        else:
            uploaded += 1
    logger.debug('windows provisioning：skill 上传完成 %d/%d → %s', uploaded, len(files), skill_path)


async def _seed_workspace_windows(backend: WaaBackend, local_seed_dir: str, workspace_path: str) -> None:
    """把本地 seed 目录内的文件（保留相对结构）上传到 VM workspace。"""
    source = Path(local_seed_dir)
    if not source.is_dir():
        logger.warning('windows provisioning：local_seed_dir 不存在，跳过 seed: %s', source)
        return

    files = [p for p in source.rglob('*') if p.is_file()]
    uploaded = 0
    for f in files:
        rel = f.relative_to(source)
        vm_target = ntpath.join(workspace_path.replace('/', '\\'), *rel.parts)
        try:
            await backend.upload_file(vm_target, f.read_bytes())
        except Exception as exc:  # noqa: BLE001  单文件失败不阻断其余
            logger.warning('windows provisioning：workspace seed 上传失败 %s: %s', vm_target, exc)
        else:
            uploaded += 1
    logger.debug('windows provisioning：workspace seed 上传完成 %d/%d → %s', uploaded, len(files), workspace_path)


async def _provision_ybsbr_windows(backend: WaaBackend, local_ctx: LocalContext) -> None:
    """预置 ybsbr 沙箱:建 init 目录 + 按 workspace/权限档生成并上传两个 profile（主/放网）。

    仅在 ``windows_sandbox`` 开启时做（默认开）。 profile 走 write_file（生成内容,非本地文件）。
    执行侧的可用性探测在 executor.（provisioning 只备料,不探测。） 失败只告警不中断。
    """
    if not local_ctx.windows_sandbox:
        return
    sandbox = WindowsSandbox(local_ctx.workspace, local_ctx.permission)
    try:
        for d in sandbox.init_dirs():
            await backend.create_folder(d)
        for vm_path, content in sandbox.iter_profiles():
            await backend.write_file(vm_path, content)
    except Exception as exc:  # noqa: BLE001  预置失败不阻断,executor 探测不到 profile 会自然降级
        logger.warning('windows provisioning：ybsbr profile 预置失败: %s', exc)
        return
    logger.debug('windows provisioning：ybsbr profile 已上传（主/放网）')


async def _provision_env_windows(local_ctx: LocalContext) -> None:
    """Windows/WAA 预置：skill 恒做；warm 时再在 VM 内装 Python/Node 库。

    node/python runtime 由 VM 镜像预装，无需上传。 skill 无论 cold/warm 都要（技能是运行前提）；
    pip/npm 库安装仅 warm。 命令恒走 PowerShell（WaaBackend 内部固定），与模型自身 shell 选择无关。
    """
    ctx = require_context()
    runtime = await ctx.get_runtime()
    backend = WaaBackend(runtime)

    # 全新 VM 上 workspace 目录未必存在，先建，否则工具 cwd 切换 / 写文件会失败。
    ws_path = local_ctx.workspace.path
    if ws_path:
        try:
            await backend.create_folder(ws_path)
        except Exception as exc:  # noqa: BLE001  建目录失败不阻断，交由后续命令暴露
            logger.warning('windows provisioning：workspace 目录创建失败 %s: %s', ws_path, exc)

    await _provision_ybsbr_windows(backend, local_ctx)

    if local_ctx.skill_path:
        await _provision_skill_windows(backend, local_ctx.skill_path)

    env = local_ctx.env
    # 把本地 seed 目录的文件上传到 VM workspace（清单等 workspace 数据 seed 进 VM），cold/warm 都做。
    if env is not None and env.local_seed_dir:
        await _seed_workspace_windows(backend, env.local_seed_dir, ws_path)

    if env is None or env.mode != 'warm':
        return

    ws = local_ctx.workspace.path
    if env.python_requirements:
        req = _resolve_ws_path_win(ws, env.python_requirements)
        await _run_provision_vm(backend, f"python -m pip install -r '{req}'", 'pip')

    if env.node_manifest:
        manifest = _resolve_ws_path_win(ws, env.node_manifest)
        manifest_dir = ntpath.dirname(manifest)
        await _run_provision_vm(backend, f"Set-Location '{manifest_dir}'; npm install", 'npm')


@dataclass
class _LoopState:
    iterations: int = 0
    tool_call_count: int = 0
    tool_error_count: int = 0


@dataclass
class _PreparedToolCall:
    id: str
    name: str
    args: Any


def _prepare_tool_calls(tool_calls: list[Any]) -> list[_PreparedToolCall] | None:
    prepared: list[_PreparedToolCall] = []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None
        prepared.append(_PreparedToolCall(id=tc.id, name=tc.function.name, args=args))
    return prepared


def _build_tool_message(tool_call_id: str, content: str | list[dict[str, Any]]) -> dict[str, Any]:
    return {'role': 'tool', 'tool_call_id': tool_call_id, 'content': content}


# ═══════════════ 工具错误可恢复：反馈而非中断（对齐 main/agent.py） ═══════════════
#
# 默认行为（历史）：任一工具调用的参数不合法 / 校验失败 / 执行异常，都会直接终止整条
# 轨迹（exit_status=tool_error）。 这对 RL / 评测很不友好——一次幻觉出的参数本应是
# 可自我纠正的小失误。 下面这组 helper 把错误包装成 tool observation 喂回模型，让循环
# 继续，行为与 main_agent 完全一致，由 ``tool_error_continue`` 配置 /
# ``NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE`` 环境变量控制（默认开：容错继续，可显式关闭）。

_TOOL_ERROR_CONTINUE_ENV = 'NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE'
_TRUE_VALUES = {'1', 'true', 'yes', 'y', 'on'}
_FALSE_VALUES = {'0', 'false', 'no', 'n', 'off'}

_INVALID_JSON_ARGS_OBSERVATION = (
    'Tool error: tool call arguments are not valid JSON and could not be parsed. '
    'Please resend the tool call with corrected JSON arguments.'
)

_SKIPPED_TOOL_CALL_OBSERVATION = (
    'Tool call was not executed because another tool call in the same assistant message '
    'had invalid arguments. Please resend the intended tool calls with corrected arguments.'
)

_SKIPPED_AFTER_ERROR_OBSERVATION = (
    'Tool error: this tool call was not executed because an earlier tool call in '
    'the same assistant message failed. Please resend it if still needed.'
)


def _resolve_tool_error_continue(agent_input: LocalOneAgentInput) -> bool:
    """是否把工具调用错误喂回模型继续（而非中断）。

    ``NEXUS_MAIN_AGENT_TOOL_ERROR_CONTINUE`` 环境变量若被设成可识别的真/假值，
    覆盖 ``tool_error_continue`` 配置字段；否则用配置字段（默认 ``True``）。
    与 main_agent 复用同一开关，一处配置同时作用于两个 agent。
    """
    raw = os.environ.get(_TOOL_ERROR_CONTINUE_ENV)
    if raw is not None:
        value = raw.strip().lower()
        if value in _TRUE_VALUES:
            return True
        if value in _FALSE_VALUES:
            return False
    return bool(agent_input.tool_error_continue)


def _tool_call_args_parseable(tool_calls: list[Any]) -> list[bool]:
    """逐个 tool_call 标记：args 为合法 JSON → True，否则 False。

    在「JSON 解析失败」路径（``_prepare_tool_calls`` 对整批返回 None）上用来把错误
    归因到真正畸形的那几个调用，而不是连累整批。
    """
    flags: list[bool] = []
    for tc in tool_calls:
        try:
            json.loads(tc.function.arguments)
            flags.append(True)
        except (json.JSONDecodeError, AttributeError, TypeError):
            flags.append(False)
    return flags


def _sanitize_unparseable_tool_args(msg: dict[str, Any]) -> None:
    """把 assistant 消息里每个 tool_call 的 ``arguments`` 就地改成 JSON-loadable。

    继续（continue）越过不可解析的参数错误时，那条 assistant 消息仍留在 history 里。
    下游客户端会用 ``json.loads`` 规整 tool-call 参数，非法字符串会让下一次推理崩溃，
    故把原始文本包进一个合法 JSON 对象，既保证可解析又保留原文供模型查看。
    """
    for tc in msg.get('tool_calls') or []:
        fn = tc.get('function') or {}
        args = fn.get('arguments')
        if isinstance(args, str):
            try:
                json.loads(args)
            except (json.JSONDecodeError, ValueError):
                fn['arguments'] = json.dumps({'_unparseable_arguments': args}, ensure_ascii=False)


def _build_tool_call_validation_messages(
    executor: LocalToolExecutor,
    tool_calls: list[_PreparedToolCall],
) -> tuple[list[dict[str, Any]] | None, int]:
    """工具名 / 参数不合法时返回 tool error observations；全部合法则返回 (None, 0)。

    执行前先做一遍非破坏性校验（``executor.validate_tool_call``）：畸形调用得到面向模型的
    错误说明，同批次其余合法调用得到「未执行」提示，保证 history 里每个 tool_call 都有
    一条对应的 tool 消息（well-formed），模型才能据此在下一轮改正。
    """
    if not executor.available_schema_names():
        return None, 0

    errors_by_call_id: dict[str, str] = {}
    for tc in tool_calls:
        try:
            executor.validate_tool_call(tc.name, tc.args)
        except (ToolNameResolutionError, ToolArgumentValidationError) as exc:
            errors_by_call_id[tc.id] = f'Tool error: {exc.message_for_model}'

    if not errors_by_call_id:
        return None, 0

    messages: list[dict[str, Any]] = [
        _build_tool_message(tc.id, errors_by_call_id.get(tc.id, _SKIPPED_TOOL_CALL_OBSERVATION)) for tc in tool_calls
    ]
    return messages, len(errors_by_call_id)


def _tool_error_observation_span(iteration: int):
    """可恢复的工具错误轮次用的 observation span。

    continue-on-tool-error 路径把错误喂回后跳过了正常的工具执行 observation span；
    用这个 span 包住其记账逻辑，让该轮次仍在 trace / metrics 里可见。
    """
    return span_context(
        'agent.observation',
        attributes={'type': 'agent_observation', 'iteration': iteration, 'tool_error': True},
    )


def _pad_missing_tool_messages(
    tool_messages: list[dict[str, Any]],
    prepared_tool_calls: list[_PreparedToolCall],
) -> list[dict[str, Any]]:
    """保证每个 tool_call 都有一条对应的 tool 响应消息。

    工具执行错误会短路 executor，导致同一 assistant 消息里靠后的 tool_call 没有响应。
    继续越过该错误时必须让 history 保持 well-formed（每个 tool_call 一条 tool 响应），
    故按位置补上占位的错误 observation。
    """
    if len(tool_messages) >= len(prepared_tool_calls):
        return tool_messages
    padded = list(tool_messages)
    for tc in prepared_tool_calls[len(tool_messages) :]:
        padded.append(_build_tool_message(tc.id, _SKIPPED_AFTER_ERROR_OBSERVATION))
    return padded


@agent_registry.register('local_one_agent')
async def run_local_one_agent(agent_input: LocalOneAgentInput) -> MainAgentOutput:
    ctx = require_context()
    client = ctx.openai_client

    if agent_input.mcp_servers:
        await ctx.load_mcp_tools(extra_servers=agent_input.mcp_servers)

    dynamic_tool_names = set(getattr(ctx, 'mcp_tools', {}).keys())

    local_ctx = LocalContext(**ctx.task_config.get('local_context', {}))
    logger.debug(
        'local_one_agent: workspace=%s, permission=%s, os=%s, env=%s',
        local_ctx.workspace.path,
        local_ctx.permission.level,
        local_ctx.runtime.os,
        local_ctx.env.mode,
    )

    # 环境依赖预置（warm 预装 / cold 不装），在进入 loop 前完成（训推一致）。
    await _provision_env(local_ctx)

    executor = _make_local_tool_executor(
        agent_input.tools,
        local_context=local_ctx,
        dynamic_tool_names=dynamic_tool_names,
    )

    if agent_input.tool_overrides:
        for override in agent_input.tool_overrides:
            executor.set_tool_override(
                override.tool_name,
                name=override.name,
                description=override.description,
                params=override.params,
            )

    # Builtin / direct tools only — skill tools stay on the executor but are
    # declared to the model via load_skill (additional_tools.json), not here.
    tool_schemas = executor.get_model_tool_schemas(exclude_canonical_names=_SKILL_TOOL_NAME_SET)

    resolved_overrides = load_tool_schema_overrides(agent_input.tool_schema_overrides)
    if resolved_overrides:
        tool_schemas = resolved_overrides

    hook_fn = None
    if agent_input.post_iteration_hook:
        hook_fn = import_from_path(agent_input.post_iteration_hook)

    rendered_system_prompt = _render_system_prompt(
        agent_input.system_prompt if isinstance(agent_input.system_prompt, str) else str(agent_input.system_prompt),
        local_ctx,
    )
    mode_text = _build_mode_reminder(local_ctx.permission, local_ctx)
    reminder = _wrap_system_reminder(mode_text) if mode_text else None

    # 多轮去重：同一 session 的每一轮都会重新 run_agent，并从共享的 ``ctx.task_config``
    # 重建一个全新的 LocalContext——``previous_level`` 因此永远拿不到上一轮的值，恒为 None，
    # 导致 mode reminder 每轮都被重复 append。 这里把本轮实际生效的 ``level`` 写回
    # ``ctx.task_config``，让**下一轮**重建 LocalContext 时把它读成 ``previous_level``，
    # 从而 ``_build_mode_reminder`` 只在首轮或权限真正切换时才产出提示。
    ctx.task_config.setdefault('local_context', {}).setdefault('permission', {})['previous_level'] = (
        local_ctx.permission.level
    )

    # Insert prior-turn / resume history between the system prompt and the
    # current user turn (multi-turn sessions, CI resume_trajectory benches).
    validate_history(agent_input.history)
    prior_history = agent_input.history or []
    history: list[dict[str, Any]] = [
        {'role': 'system', 'content': rendered_system_prompt},
        *prior_history,
    ]

    if agent_input.resume_from_history:
        # `history` already ends with the message to continue from (e.g. a
        # tool result); do NOT append a new user turn. user_prompt is ignored.
        if not agent_input.history:
            logger.warning(
                'resume_from_history=True but no history supplied; '
                'falling back to user_prompt as the first user message.'
            )
            user_content = _append_system_reminder(agent_input.user_prompt, reminder)
            history.append({'role': 'user', 'content': user_content})
    else:
        user_content = _append_system_reminder(agent_input.user_prompt, reminder)
        history.append({'role': 'user', 'content': user_content})

    log_messages(history, prefix='init ')

    state = _LoopState()
    exit_status = MainExitStatus.MAX_ITERATIONS.value
    final_answer: str | None = None
    llm_http_error: BaseException | None = None
    invalid_args_max_retries = agent_input.invalid_tool_args_retries
    invalid_arg_observation_retries = 0
    tool_error_continue = _resolve_tool_error_continue(agent_input)
    if tool_error_continue:
        logger.debug('local_one_agent: continue-on-tool-error enabled — 工具错误将喂回模型继续，而非中断轨迹')
    inference_budget = TimeBudget(agent_input.inference_time_timeout)
    observation_budget = TimeBudget(agent_input.observation_time_timeout)

    for iteration in range(1, agent_input.max_iterations + 1):
        state.iterations = iteration

        with span_context(
            'agent.iteration',
            attributes={'type': 'agent_iteration', 'iteration': iteration},
        ):
            valid_response = False
            message = None
            prepared_tool_calls: list[_PreparedToolCall] | None = None

            for retry in range(invalid_args_max_retries + 1):
                try:
                    with span_context(
                        'agent.inference',
                        attributes={'type': 'agent_inference', 'iteration': iteration, 'retry': retry},
                    ):
                        response = await await_completion_with_time_budget(
                            inference_budget,
                            lambda: client.create_completion(
                                messages=history,
                                tools=tool_schemas or None,
                            ),
                        )
                except ContextLengthExceededError:
                    exit_status = MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value
                    break
                except InferenceTimeoutError:
                    exit_status = MainExitStatus.INFERENCE_TIMEOUT.value
                    break
                except Exception as exc:
                    logger.exception('Inference call failed: %s', exc)
                    exit_status = MainExitStatus.INFERENCE_ERROR.value
                    if is_llm_http_error(exc):
                        llm_http_error = exc
                    break

                try:
                    message = response.choices[0].message
                except Exception as exc:
                    logger.exception('Invalid inference response shape: %s', exc)
                    exit_status = MainExitStatus.INFERENCE_ERROR.value
                    break

                if not message.tool_calls:
                    break

                prepared_tool_calls = _prepare_tool_calls(message.tool_calls)
                if prepared_tool_calls is None:
                    if retry < invalid_args_max_retries:
                        continue
                    break

                valid_response = True
                break

            if exit_status in {
                MainExitStatus.INFERENCE_TIMEOUT.value,
                MainExitStatus.CONTEXT_LENGTH_EXCEEDED.value,
                MainExitStatus.INFERENCE_ERROR.value,
            }:
                break

            assert message is not None
            msg_dict = message.model_dump(exclude_none=True)
            history.append(msg_dict)
            logger.debug('iter %d resp %s', iteration, fmt_msg(msg_dict))

            if message.content:
                final_answer = message.content

            if not message.tool_calls:
                exit_status = MainExitStatus.FINISHED.value
                break

            # JSON 解析失败（重试耗尽后，至少一个调用的 args 不是合法 JSON）。
            # 每个调用都计入 attempted，只把真正畸形的计入 error，保证
            # tool_error_count 精确且不超过 tool_call_count。
            if not valid_response:
                args_ok = _tool_call_args_parseable(message.tool_calls)
                state.tool_call_count += len(args_ok)
                state.tool_error_count += sum(1 for ok in args_ok if not ok)
                if tool_error_continue:
                    # 保持 assistant 消息 JSON-loadable，再逐个 tool_call 喂回 observation：
                    # 畸形的给 parse error，合法的兄弟调用给「已跳过」提示。 用 observation
                    # span 包住，让本轮仍在 trace / metrics 里可见（它跳过了下面的执行 span）。
                    _sanitize_unparseable_tool_args(history[-1])
                    with _tool_error_observation_span(iteration):
                        err_messages = [
                            _build_tool_message(
                                tc.id,
                                _SKIPPED_TOOL_CALL_OBSERVATION if ok else _INVALID_JSON_ARGS_OBSERVATION,
                            )
                            for tc, ok in zip(message.tool_calls, args_ok, strict=False)
                        ]
                        log_messages(err_messages, prefix=f'iter {iteration} obs ')
                        history.extend(err_messages)
                    continue
                exit_status = MainExitStatus.TOOL_ERROR.value
                break

            assert prepared_tool_calls is not None

            # 执行前先校验工具名 / 参数。 不合法的调用作为 tool observation 喂回，让模型
            # 自我纠正，而不是让整条轨迹因一次参数错误直接终止（本次修复的核心 bug）。
            validation_messages, validation_error_count = _build_tool_call_validation_messages(
                executor, prepared_tool_calls
            )
            if validation_messages is not None:
                state.tool_call_count += len(prepared_tool_calls)
                state.tool_error_count += validation_error_count
                log_messages(validation_messages, prefix=f'iter {iteration} obs ')
                history.extend(validation_messages)

                if invalid_arg_observation_retries < invalid_args_max_retries:
                    invalid_arg_observation_retries += 1
                    logger.warning(
                        'Invalid tool call arguments, feedback retry %d/%d',
                        invalid_arg_observation_retries,
                        invalid_args_max_retries,
                    )
                    continue

                if tool_error_continue:
                    # 校验错误 observation 已在上面追加。 用 observation span 让这一可恢复
                    # 错误轮次仍在 trace / metrics 里可见，然后继续，让模型改正工具名 / 参数。
                    with _tool_error_observation_span(iteration):
                        logger.warning(
                            'Invalid tool call; continue-on-tool-error enabled — feeding error back to the model'
                        )
                    continue

                exit_status = MainExitStatus.TOOL_ERROR.value
                break

            invalid_arg_observation_retries = 0

            t0 = time.monotonic()
            try:
                with span_context(
                    'agent.observation',
                    attributes={'type': 'agent_observation', 'iteration': iteration},
                ):
                    async with with_timeout(observation_budget.remaining()):
                        tool_messages, tc_exit = await _execute_tools_sequential(
                            executor, prepared_tool_calls, agent_input, state
                        )
                observation_budget.consume(time.monotonic() - t0)
            except (asyncio.TimeoutError, TimeoutError):
                observation_budget.consume(time.monotonic() - t0)
                exit_status = MainExitStatus.OBSERVATION_TIMEOUT.value
                break

            if hook_fn is not None and tc_exit is None:
                try:
                    hook_result = await hook_fn(ctx, tool_messages)
                except Exception:
                    logger.warning('Post-iteration hook failed', exc_info=True)
                    hook_result = None
                if hook_result is not None:
                    if hook_result.tool_messages is not None:
                        tool_messages = hook_result.tool_messages
                    if hook_result.extra_messages:
                        tool_messages = tool_messages + hook_result.extra_messages

            # continue-on-tool-error：把致命的工具执行错误转成可恢复的 observation，循环继续。
            tc_error_recoverable = tool_error_continue and tc_exit in {
                MainExitStatus.TOOL_ERROR.value,
                MainExitStatus.TOOL_EXECUTION_ERROR.value,
            }
            if tc_error_recoverable:
                tool_messages = _pad_missing_tool_messages(tool_messages, prepared_tool_calls)
                logger.warning('Tool execution error; continue-on-tool-error enabled — continuing loop')

            log_messages(tool_messages, prefix=f'iter {iteration} obs ')
            history.extend(tool_messages)

            if tc_exit is not None and not tc_error_recoverable:
                exit_status = tc_exit
                break

    span_id = get_current_span_id()
    if span_id:
        try:
            update_span(
                span_id,
                attributes={
                    'agent.exit_status': exit_status,
                    'agent.iterations': state.iterations,
                    'agent.tool_call_count': state.tool_call_count,
                    'agent.tool_error_count': state.tool_error_count,
                },
            )
        except Exception:
            logger.warning('Failed to update span attributes for agent', exc_info=True)

    if llm_http_error is not None:
        raise llm_http_error

    return MainAgentOutput(
        main_trajectory=history,
        exit_status=exit_status,
        iterations=state.iterations,
        final_answer=final_answer,
    )


async def _execute_tools_sequential(
    executor: LocalToolExecutor,
    tool_calls: list[_PreparedToolCall],
    agent_input: LocalOneAgentInput,
    state: _LoopState,
) -> tuple[list[dict[str, Any]], str | None]:

    messages: list[dict[str, Any]] = []
    termination_exit: str | None = None

    for tc in tool_calls:
        state.tool_call_count += 1

        try:
            resp = await executor.call(tc.name, tc.args)
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, TimeoutError):
            raise
        except RuntimeUnhealthyError as exc:
            state.tool_error_count += 1
            messages.append(_build_tool_message(tc.id, f'Tool error: {exc}'))
            return messages, MainExitStatus.RUNTIME_ERROR.value
        except (ToolNameResolutionError, ToolArgumentValidationError) as exc:
            # 工具名 / 参数不合法（正常情况下已被执行前的校验拦截；这里兜底）。
            state.tool_error_count += 1
            message_for_model = getattr(exc, 'message_for_model', str(exc))
            messages.append(_build_tool_message(tc.id, f'Tool error: {message_for_model}'))
            return messages, MainExitStatus.TOOL_ERROR.value
        except Exception as exc:
            state.tool_error_count += 1
            messages.append(_build_tool_message(tc.id, f'Tool error: {exc}'))
            return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value

        try:
            extract_fn = extract_tool_output_content if agent_input.multimodal_tool_output else extract_tool_output_text
            output_content = extract_fn(resp.output)
            if isinstance(output_content, str):
                output_content = maybe_truncate(output_content, agent_input.tool_output_truncation)
        except Exception as exc:
            state.tool_error_count += 1
            messages.append(_build_tool_message(tc.id, f'Tool error: {exc}'))
            return messages, MainExitStatus.TOOL_EXECUTION_ERROR.value

        messages.append(_build_tool_message(tc.id, output_content))

        if resp.meta.get('termination_requested') is True:
            termination_exit = MainExitStatus.FINISHED.value

    return messages, termination_exit
