"""aidata 任务入口 —— engine 各子模块之上的轻量编排器。

下面的 ``@task_registry.register('aidata')`` 函数刻意保持精简：所有非平凡步骤
都已下沉到专属的 engine 模块（case_spec / context / sandbox / mcp_catalog /
state / rubric）。 此处保留的仅是正常路径的执行序列以及可在步骤间中断的错误
处理逻辑。

执行序列：

    0. 解析 case.yaml → 填充 AidataTaskInput
    1. 准备沙箱布局（AGENT_DESKTOP + INTERNAL_ROOT）
    2. 上传随包发布的 mock MCP 服务器脚本
    3. 原样上传上下文文件（宿主 → 沙箱）
    4. 上传 case 资产并运行 case.bootstrap_script
    5. 加载 MCP 工具目录（默认项 + case 追加项）
    6. 组装 agent 用户消息（清单 + 任务）
    7. 运行 agent
    8. 关闭 MCP
    9. 解析 rubric（加载已提交的 rubric.yaml；必需）
   10. 下载 mock 状态并评估 rubric
   11. 返回 AidataTaskOutput

任一步骤发生错误时会以 ``error_message`` 提前返回，而不是抛出异常 ——
task_registry 基础设施据此将该次运行标记为“失败但可上报”（这样 nexus UI 仍能
拿到结构化结果）。

注：此处刻意不使用 ``from __future__ import annotations`` —— task_registry.register
装饰器使用 ``inspect.signature`` 在运行时校验输入参数是 ``TaskRunInput`` 的子类，
当注解是惰性字符串而非真实类对象时该校验会失败。
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
import uuid

from pathlib import Path
from typing import Any

from nexus.core import require_context, task_registry
from nexus.runtime.executors.file import FileItem
from nexus.trace import get_current_span_id, update_span
from nexus.utils.timeout import async_timeout

from .models import AidataTaskInput, AidataTaskOutput
from .paths import (
    AGENT_DESKTOP,
    AGENT_INPUTS_DIR,
    AGENT_OUTPUT_DIR,
    CASES_DIR,
    CASE_DIR_IN,
    CONFINE_DIR,
    CONFINE_DIR_IN,
    CONFINE_SH_IN,
    INTERNAL_ROOT,
    JUNIT_REPORT_PATH_IN,
    MOCK_SERVER_DATA_DIR_IN,
    SERVERS_DIR,
    SERVERS_DIR_IN,
    STATE_DIR_IN,
    TESTS_DIR_IN,
)
from .system_prompt import resolve_system_prompt
from .. import rubric as rubric_lib
from ..case.spec import (
    case_rubric_path,
    load_case_rubric,
    resolve_case,
)
from ..rubric.check_map import load_check_map
from ..rubric.junit import JunitParseError, JunitReport, parse_junit_xml
from ..rubric.reward import RewardScriptError, resolve_reward
from ..runtime import base_tools as _base_tools  # noqa: F401 - 导入以注册 aidata 本地工具。
from ..runtime.context import build_context_manifest, render_manifest
from ..runtime.mcp_catalog import build_specs, mock_env, normalize_host_env
from ..runtime.prompt_builder import discover_project_context
from ..runtime.sandbox import ensure_command_succeeded, run_bootstrap, upload_case_assets, upload_dir, upload_text_file
from ..runtime.services import apply_text_replacements, setup_services
from ..runtime.state import build_summary, collect_crontab_state, download_state, extract_final_answer


logger = logging.getLogger(__name__)
_MIN_MCP_CLOSE_TIMEOUT_S = 0.001

_HYTUNER_TERMINATION_ATTR = 'hytuner.termination_reason'
_TERMINATION_REWARD_ERROR = 'REWARD_ERROR'
_TERMINATION_ENV_ERROR = 'ENV_ERROR'
_TERMINATION_INITIALIZATION_ERROR = 'INITIALIZATION_ERROR'
_TERMINATION_TASK_TIMEOUT = 'TASK_TIMEOUT'
_TERMINATION_UNHANDLED_ERROR = 'UNHANDLED_ERROR'


# 向后兼容：某些较早的调用方从 ``engine.task`` 导入 AidataTaskInput / AidataTaskOutput。
# 在此处继续重导出这些名称。
__all__ = ['AidataTaskInput', 'AidataTaskOutput', 'build_agent_kwargs', 'run_aidata']


def build_agent_kwargs(
    task_input: AidataTaskInput,
    *,
    user_message: str,
    system_prompt: str,
    tools: list[str],
) -> dict[str, Any]:
    agent_kwargs: dict[str, Any] = {
        'user_prompt': user_message,
        'system_prompt': system_prompt,
        'tools': tools,
        'timeout': task_input.agent_timeout,
        'max_iterations': task_input.max_iterations,
    }
    if task_input.observation_time_timeout is not None:
        agent_kwargs['observation_time_timeout'] = task_input.observation_time_timeout
    return agent_kwargs


def _remaining_agent_timeout(task_input: AidataTaskInput, agent_timeout_started_at: float) -> float:
    return max(0.0, float(task_input.agent_timeout) - (time.monotonic() - agent_timeout_started_at))


async def _close_mcp_tools_with_agent_timeout(
    ctx: Any,
    *,
    task_input: AidataTaskInput,
    task_id: str,
    agent_timeout_started_at: float | None,
    propagate_timeout: bool,
) -> None:
    """关闭 MCP 工具，同时不让清理逻辑挤占 agent 超时预算。"""
    if agent_timeout_started_at is None:
        with contextlib.suppress(Exception):
            await ctx.close_mcp_tools()
        return

    remaining = _remaining_agent_timeout(task_input, agent_timeout_started_at)
    close_timeout = max(remaining, _MIN_MCP_CLOSE_TIMEOUT_S)

    try:
        async with async_timeout(close_timeout):
            await ctx.close_mcp_tools()
    except (asyncio.TimeoutError, TimeoutError) as exc:
        logger.warning(
            '[aidata] MCP close exceeded agent timeout budget task=%s timeout=%ss remaining=%.3fs',
            task_id,
            task_input.agent_timeout,
            remaining,
        )
        if propagate_timeout:
            raise TimeoutError(f'MCP close exceeded agent timeout budget after {task_input.agent_timeout}s') from exc
    except Exception as exc:
        logger.warning('[aidata] failed to close MCP tools task=%s: %s', task_id, exc)


def _safe_filename_part(value: str) -> str:
    """返回适用于宿主侧转储的、文件系统安全的文件名分量。"""
    safe = ''.join(ch if (ch.isalnum() or ch in {'-', '_', '.'}) else '_' for ch in value)
    return safe.strip('._') or 'unknown'


def _state_dump_enabled() -> bool:
    return os.environ.get('AIDATA_STATE_DUMP', '1').strip() != '0'


def _resolve_mock_now(task_input):
    """提取 case 的冻结时钟锚点（``MOCK_SERVER_NOW``）以便转发到 shell MCP 的
    libconfine faketime 钩子。

    case 通过在 sidecar 服务的 ``startup.env`` 中设置 ``MOCK_SERVER_NOW``（ISO-8601）
    来钉住确定性时钟 —— 该单个值同时驱动 mock 数据（经由 MockServer 的
    ``_common/clock.py``）以及在此处转发之后驱动 agent shell 的 ``date``。 返回
    找到的第一个此类值，若没有任何服务钉住时钟则返回 ``None``（常见情况 —— 此时
    不注入 faketime，挂钟保持原样）。
    """
    for service in getattr(task_input, 'services', None) or []:
        startup = getattr(service, 'startup', None)
        env = getattr(startup, 'env', None) if startup is not None else None
        if env:
            val = env.get('MOCK_SERVER_NOW')
            if val and val.strip():
                return val
    return None


def _report_hytuner_termination(termination_reason: str | None) -> None:
    """通过 HyTuner 的 span 通道上报 AIDATA 的非 agent 失败原因。"""
    if not termination_reason:
        return
    span_id = get_current_span_id()
    if not span_id:
        return
    try:
        update_span(span_id, attributes={_HYTUNER_TERMINATION_ATTR: termination_reason})
    except Exception as exc:
        logger.warning('[aidata] failed to report hytuner termination_reason=%s: %s', termination_reason, exc)


def _hytuner_extra_fields(
    termination_reason: str,
    *,
    failure_phase: str,
    failure_kind: str,
    score_valid: bool = False,
    detail: str | None = None,
    invalid_check_ids: list[str] | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        _HYTUNER_TERMINATION_ATTR: termination_reason,
        'termination_reason': termination_reason,
        'aidata.score_valid': score_valid,
        'aidata.failure_phase': failure_phase,
        'aidata.failure_kind': failure_kind,
    }
    if detail:
        extra['aidata.failure_detail'] = detail
    if invalid_check_ids:
        extra['aidata.invalid_check_ids'] = invalid_check_ids
    return extra


def _invalid_task_output(
    *,
    task_id: str,
    start: float,
    error_message: str,
    termination_reason: str,
    failure_phase: str,
    failure_kind: str,
    state_snapshots: dict[str, str] | None = None,
) -> AidataTaskOutput:
    _report_hytuner_termination(termination_reason)
    logger.warning(
        '[aidata] invalid sample task=%s reason=%s phase=%s kind=%s error=%s',
        task_id,
        termination_reason,
        failure_phase,
        failure_kind,
        error_message,
    )
    return AidataTaskOutput(
        task_id=task_id,
        score=0.0,
        error_message=error_message,
        state_snapshots=state_snapshots or {},
        extra_fields=_hytuner_extra_fields(
            termination_reason,
            failure_phase=failure_phase,
            failure_kind=failure_kind,
            detail=error_message,
        ),
        wall_time_s=time.monotonic() - start,
    )


def _termination_from_rubric_results(rubric_results: list[dict[str, Any]]) -> tuple[str, str, list[str]] | None:
    unavailable_ids = [
        str(row.get('id') or '') for row in rubric_results if str(row.get('verdict') or '').upper() == 'UNAVAILABLE'
    ]
    if unavailable_ids:
        return _TERMINATION_REWARD_ERROR, 'judge_unavailable', unavailable_ids

    authoring_error_ids = [
        str(row.get('id') or '')
        for row in rubric_results
        if str(row.get('verdict') or '').upper() in {'INVALID', 'UNKNOWN_ASSERT_TYPE', 'UNKNOWN_KIND'}
    ]
    if authoring_error_ids:
        return _TERMINATION_INITIALIZATION_ERROR, 'rubric_authoring_error', authoring_error_ids

    evaluator_error_ids = [
        str(row.get('id') or '')
        for row in rubric_results
        if str(row.get('verdict') or '').upper() in {'ERROR', 'NO_RUNTIME'}
    ]
    if evaluator_error_ids:
        return _TERMINATION_UNHANDLED_ERROR, 'rubric_evaluator_error', evaluator_error_ids

    return None


def _dump_state_to_file(state: dict[str, Any], task_id: str, case_id: str, *, phase: str = 'state') -> str:
    """将完整的 state 字典写入 JSON 文件以便运行后检查。

    通过环境变量控制：
      - ``AIDATA_STATE_DUMP=0``  → 跳过（默认：启用）
      - ``AIDATA_STATE_DUMP_DIR`` → 输出目录（默认：``/tmp/aidata-state/``）

    返回写入的路径，跳过或出错时返回空字符串。
    """
    if not _state_dump_enabled():
        return ''
    dump_dir = Path(os.environ.get('AIDATA_STATE_DUMP_DIR', '/tmp/aidata-state'))
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        label = _safe_filename_part(case_id or task_id or 'unknown')
        phase_label = _safe_filename_part(phase or 'state')
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = dump_dir / f'{phase_label}_{label}_{ts}.json'
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        logger.debug('[aidata] %s state dumped to %s (%d bytes)', phase_label, path, path.stat().st_size)
        return str(path)
    except Exception as exc:
        logger.warning('[aidata] state dump failed: %s', exc)
        return ''


def _compact_csv(values: list[str], *, max_items: int = 40, max_chars: int = 600) -> str:
    shown = values[:max_items]
    text = ','.join(shown)
    if len(values) > max_items:
        text = f'{text},+{len(values) - max_items}' if text else f'+{len(values) - max_items}'
    if len(text) > max_chars:
        text = f'{text[:max_chars]}...'
    return text or '-'


def _log_mcp_tools_summary(task_id: str, case_id: str, mcp_tools: dict[str, Any]) -> None:
    names = sorted(str(name) for name in mcp_tools)
    logger.warning(
        '[aidata] mcp task=%s case=%s count=%d tools=%s',
        task_id,
        case_id or '-',
        len(names),
        _compact_csv(names),
    )


def _rubric_status_char(row: dict[str, Any]) -> str:
    if row.get('passed'):
        return 'Y'
    verdict = str(row.get('verdict') or '').upper()
    if verdict == 'UNAVAILABLE':
        return 'U'
    if verdict in {'ERROR', 'NO_RUNTIME'}:
        return 'E'
    if verdict in {'INVALID', 'UNKNOWN_ASSERT_TYPE', 'UNKNOWN_KIND'}:
        return 'I'
    if verdict == 'MISSING':
        return 'M'
    return 'N'


def _compact_rubric_status(rubric_results: list[dict[str, Any]], *, max_chars: int = 240) -> str:
    status = ''.join(_rubric_status_char(row) for row in rubric_results)
    if len(status) > max_chars:
        status = f'{status[:max_chars]}...'
    return status or '-'


def _rubric_status_counts(status: str) -> str:
    if status == '-':
        return '-'
    return ','.join(f'{ch}={status.count(ch)}' for ch in 'YNMUEI' if status.count(ch))


def _log_rubric_summary(
    task_id: str,
    case_id: str,
    rubric_results: list[dict[str, Any]],
    *,
    termination_reason: str | None = None,
    failure_kind: str | None = None,
) -> None:
    status = _compact_rubric_status(rubric_results)
    logger.warning(
        '[aidata] rubric task=%s case=%s total=%d status=%s counts=%s invalid=%s/%s',
        task_id,
        case_id or '-',
        len(rubric_results),
        status,
        _rubric_status_counts(status),
        termination_reason or '-',
        failure_kind or '-',
    )


@task_registry.register('aidata')
async def run_aidata(task_input: AidataTaskInput) -> AidataTaskOutput:
    start = time.monotonic()
    ctx = require_context()

    # 0. 解析 case.yaml → task_input
    try:
        resolve_case(task_input)
    except Exception as exc:
        logger.error('[aidata] case resolution failed: %s', exc)
        return _invalid_task_output(
            task_id=task_input.task_id,
            start=start,
            error_message=f'case resolution failed: {exc}',
            termination_reason=_TERMINATION_INITIALIZATION_ERROR,
            failure_phase='case_resolution',
            failure_kind='case_config_error',
        )

    task_id = task_input.task_id or 'aidata'
    state_snapshots: dict[str, str] = {}
    normalize_host_env()
    logger.debug('[aidata] start task=%s case=%s', task_id, task_input.case_id or '(none)')

    if not task_input.user_prompt.strip():
        return _invalid_task_output(
            task_id=task_id,
            start=start,
            error_message='user_prompt is required (empty); set task_config.user_prompt or task_config.case_id',
            termination_reason=_TERMINATION_INITIALIZATION_ERROR,
            failure_phase='input_validation',
            failure_kind='missing_user_prompt',
        )

    # 每 case ``main`` 镜像覆盖（task.yaml::runtime.main.image）。
    # 在获取运行时之前重新配置 provider，以便需要
    # ``aidata-aio-agent:2.x`` 路径（或任何其他镜像布局）的历史 case
    # 能够运行在正确的容器中，而无需修改全局的
    # ``runtime.template.yaml``。
    if task_input.runtime_image and ctx.runtime_provider is not None:
        ctx.runtime_provider.configure('main', image=task_input.runtime_image)

    try:
        runtime = await ctx.get_runtime('main')
    except Exception as exc:
        return _invalid_task_output(
            task_id=task_id,
            start=start,
            error_message=f'runtime acquire failed: {exc}',
            termination_reason=_TERMINATION_ENV_ERROR,
            failure_phase='runtime_acquire',
            failure_kind='runtime_acquire_failed',
        )

    runtime_id = ctx.get_runtime_id('main')
    if runtime_id:
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

    # 辅助 sidecar 服务容器（在 case.yaml::services 中声明）。 完整契约
    # 见 engine/services.py。 在 bootstrap 与 MCP 工具加载之前完成，以便
    # 二者均可引用已解析的服务 URL。
    service_case_dir = (CASES_DIR / task_input.case_id) if task_input.case_id else None
    try:
        service_replacements = await setup_services(
            task_input.services,
            case_dir=service_case_dir,
            mock_env_vars=mock_env(task_input.user_prompt) if task_input.services else None,
        )
    except Exception as exc:
        logger.error('[aidata] service setup failed: %s', exc, exc_info=True)
        return _invalid_task_output(
            task_id=task_id,
            start=start,
            error_message=f'service setup failed: {exc}',
            termination_reason=_TERMINATION_ENV_ERROR,
            failure_phase='service_setup',
            failure_kind='service_setup_failed',
        )

    agent_timeout_started_at: float | None = None

    try:
        # 1. 准备工作区：擦除上一次运行 + 重建隐藏的内部根目录以及 agent 的桌面。
        #    skills 符号链接使得 harness 的 skill 探针（engine.prompt_builder）以及
        #    agent 自己的 shell 都能通过 desktop 找到 /workspace/skills。
        #
        #    MOCK_SERVER_DATA_DIR_IN 在这里无条件创建，即使暂无
        #    source='mock_server' 的随包 mock 已注册 —— 开销很小，且允许
        #    每 case 的 extra_mcp_servers 引用
        #    /opt/mock-server/<group>/<name>/ 下的脚本（例如 life/ride_app）
        #    这些脚本通过 mock_data_dir(<name>) 读取 MOCK_SERVER_DATA_DIR，
        #    从而开箱即用，无需 case 侧再做 mkdir。
        desktop_skills = f'{AGENT_DESKTOP}/skills'
        image_skills = '/workspace/skills'
        prep_cmd = (
            f'rm -rf {INTERNAL_ROOT} {AGENT_DESKTOP} && '
            f'mkdir -p {SERVERS_DIR_IN} {STATE_DIR_IN} {CASE_DIR_IN} '
            f'{MOCK_SERVER_DATA_DIR_IN} {AGENT_INPUTS_DIR} {AGENT_OUTPUT_DIR} && '
            f'chmod 700 {INTERNAL_ROOT} && '
            f'[ -d {image_skills} ] && ln -snf {image_skills} {desktop_skills} || true'
        )
        await runtime.run_command(prep_cmd, timeout=15)

        # 2. 上传 engine/servers/ 下的本地 MCP 服务器脚本。
        #    Office mocks 来自 mock-server sidecar 容器
        #    通过 HTTP MCP 提供，按 case 声明。
        await upload_dir(runtime, SERVERS_DIR, SERVERS_DIR_IN)

        # 2.5. 上传并编译 LD_PRELOAD confine 套件。 保存在 INTERNAL_ROOT
        #      之下（该目录本身在 deny 前缀列表中），使得 agent 无法篡改
        #      .so 来解除对自身的约束。 在沙箱中针对镜像的确切 glibc
        #      进行编译 —— 无需修改 Dockerfile 即可推广到多个运行时镜像。
        #      详见 mock_servers/confine/README.md。
        #
        #      默认 enforce 模式下，上传或编译失败会直接标记为环境初始化
        #      错误，避免 shell MCP 启动后每条命令都以 exit 99 失败。
        #      warn / off 模式仍允许开发镜像在没有 C 编译器时继续运行。
        try:
            await upload_dir(runtime, CONFINE_DIR, CONFINE_DIR_IN)
            confine_compile = await runtime.run_command(
                f'chmod +x {CONFINE_SH_IN} && bash {CONFINE_SH_IN}',
                timeout=30,
            )
            ensure_command_succeeded(confine_compile, 'confine kit compilation')
        except Exception as exc:
            confine_mode = (os.getenv('AIDATA_CONFINE_MODE', 'enforce') or 'enforce').strip().lower()
            if confine_mode == 'enforce':
                logger.error('[aidata] confine kit setup failed in enforce mode: %s', exc, exc_info=True)
                return _invalid_task_output(
                    task_id=task_id,
                    start=start,
                    error_message=f'confine kit setup failed: {exc}',
                    termination_reason=_TERMINATION_ENV_ERROR,
                    failure_phase='sandbox_setup',
                    failure_kind='confine_setup_failed',
                    state_snapshots=state_snapshots,
                )
            logger.warning('[aidata] confine kit setup failed: %s', exc)

        # 2.6. 探测 LibreOffice 是否可用（尽力而为，非阻塞）。
        #      若镜像缺少 LibreOffice，则在 system prompt 中补一行上下文，
        #      避免 agent 浪费迭代去调用 ``libreoffice``。
        lo_available: bool | None = None
        try:
            lo_probe = await runtime.run_command(
                'libreoffice --version 2>/dev/null || echo __LO_MISSING__',
                timeout=30,
            )
            lo_stdout = (getattr(lo_probe, 'stdout', '') or '').strip()
            if '__LO_MISSING__' in lo_stdout or not lo_stdout:
                logger.warning(
                    '[aidata] LibreOffice not found in sandbox — agent will fall back to Python-only document handling'
                )
                lo_available = False
            else:
                lo_available = True
                logger.debug('[aidata] LibreOffice available: %s', lo_stdout[:80])
        except Exception:
            logger.debug('[aidata] LibreOffice probe failed (non-fatal)')

        # 3. 将每个上下文文件原样上传到 agent desktop，让 agent 自己通过
        #    pdf-tools / xlsx skill / shell 去读取。 内容不会被内联进 prompt
        #    —— 清单只列出每个文件的路径 + 建议的读取工具（build_context_manifest /
        #    render_manifest）。 让 agent 真实调用工具去检查输入文件正是本
        #    benchmark 的核心目的。
        context_raw_root = AGENT_INPUTS_DIR
        for p in task_input.context_files:
            path = Path(p).expanduser()
            if path.exists() and path.is_file():
                b64 = base64.b64encode(path.read_bytes()).decode('ascii')
                await runtime.upload_file(
                    files=[FileItem(path=f'{context_raw_root}/{path.name}', content=b64, encoding='base64')],
                    overwrite=True,
                )
        manifest = build_context_manifest(task_input.context_files, context_raw_root)
        logger.debug('[aidata] context: %d files uploaded to agent desktop', len(task_input.context_files))

        # 4. 每 case 的资产 + bootstrap。 位于上下文上传与 MCP 加载之间，
        #    这样 bootstrap 既能引用上下文文件与种子数据，也能保证在 MCP
        #    客户端尝试连接时，由 bootstrap 启动的后端服务已经在监听。
        #    case 也可以在此处准备可选的 prompt 端文件，例如
        #    /workspace/system_prompt.md 与 /workspace/desktop/MEMORY.md。
        shipped = 0
        if task_input.case_id:
            case_dir_host = CASES_DIR / task_input.case_id
            if case_dir_host.is_dir():
                shipped = await upload_case_assets(runtime, case_dir_host, CASE_DIR_IN)

        if task_input.case_id and task_input.bootstrap_script:
            if shipped == 0:
                logger.warning(
                    '[aidata] bootstrap_script=%s set but no case assets were uploaded (script probably missing)',
                    task_input.bootstrap_script,
                )
            try:
                await run_bootstrap(
                    runtime,
                    task_input.bootstrap_script,
                    CASE_DIR_IN,
                    extra_env=service_replacements.env,
                )
            except Exception as exc:
                logger.error('[aidata] bootstrap failed: %s', exc)
                return _invalid_task_output(
                    task_id=task_id,
                    start=start,
                    error_message=f'bootstrap failed: {exc}',
                    termination_reason=_TERMINATION_ENV_ERROR,
                    failure_phase='bootstrap',
                    failure_kind='bootstrap_failed',
                )

        # 5. 加载 MCP 目录（默认项 + 每 case 追加项）。 服务占位符会
        #    在此处被替换，因此 spec 可以在 client.args / client.env 中
        #    嵌入 ``__AIDATA_FOO_URL__``；同时 engine/servers/ 下每一个
        #    随包 mock 都会继承这些解析后的 URL 作为环境变量（这样
        #    engine/servers/<pkg>/server.py 可以读取
        #    ``os.environ['<env_name>']`` 去代理位于兄弟 sidecar 服务
        #    容器中的 REST 后端）。
        # 每次运行的 UUID 用于 case 局部占位符（``__CASE_UUID__``）——
        # 供 stdio mocks 使用，让它们把独立 sqlite 目录放在 ``/tmp``
        # 之下，从而并发 episode 之间不会互相覆盖各自的 DB。 该替换在
        # 服务 URL 替换 *之后* 但在 ``build_specs`` *之前* 应用，从而
        # 保证 MCP 子进程的环境与之后 ``state_equals(sql:)`` 通过
        # ``state['_rubric_sqlite_path']`` 读取的路径一致（该值是在
        # 下文 ``download_state`` 之后由
        # ``task_input.rubric_sqlite_path`` 注入的）。
        case_uuid = uuid.uuid4().hex[:12]
        case_placeholders = {
            '__CASE_UUID__': case_uuid,
            '__AIDATA_CASE_UUID__': case_uuid,
        }
        resolved_extra_servers = apply_text_replacements(
            apply_text_replacements(task_input.extra_mcp_servers, service_replacements.text),
            case_placeholders,
        )
        resolved_user_prompt = apply_text_replacements(
            apply_text_replacements(task_input.user_prompt, service_replacements.text),
            case_placeholders,
        )
        resolved_rubric_sqlite_path = apply_text_replacements(
            apply_text_replacements(task_input.rubric_sqlite_path, service_replacements.text),
            case_placeholders,
        )
        resolved_dump_state = apply_text_replacements(
            apply_text_replacements(task_input.dump_state, service_replacements.text),
            case_placeholders,
        )
        await ctx.load_mcp_tools(
            extra_servers=build_specs(
                resolved_user_prompt,
                resolved_extra_servers,
                service_env=service_replacements.env,
                wx_agent_hub=task_input.wx_agent_hub,
                main_image=task_input.runtime_image,
                mock_now=_resolve_mock_now(task_input),
            ),
            clear_existing=True,
        )
        _log_mcp_tools_summary(task_id, task_input.case_id, dict(getattr(ctx, 'mcp_tools', {}) or {}))
        if _state_dump_enabled():
            before_agent_state = await download_state(
                runtime,
                services=task_input.services,
                dump_state_args=resolved_dump_state,
            )
            before_agent_path = _dump_state_to_file(
                before_agent_state,
                task_id,
                task_input.case_id,
                phase='before_agent',
            )
            if before_agent_path:
                state_snapshots['before_agent'] = before_agent_path

        # 6. 构建用户消息：清单 + 任务（内容不会内联）
        user_message_parts: list[str] = []
        manifest_text = render_manifest(manifest)
        if manifest_text:
            user_message_parts.append(manifest_text)
        user_message_parts.append('## 任务\n\n' + resolved_user_prompt.strip())
        user_message = '\n\n'.join(user_message_parts)

        # 7. 运行 agent。 system prompt 在此处懒解析（而非在 import 时），
        #    这样 case 的 bootstrap 脚本 —— 已在上文完成运行 —— 就有机会在
        #    /workspace/system_prompt.md 中放置一个覆盖版本，或准备可选的
        #    /workspace/desktop/MEMORY.md。 这两个文件都是尽力而为的，多数
        #    case 中默默缺席。
        #
        # ``main_agent`` 自身不会做任何探测，因此由 harness 组装固定的沙箱
        # 环境上下文，加上探测得到的工作区目录树 / SKILL.md frontmatter，
        # 再交给它。 ``main_agent`` 还要求显式提供 ``tools`` 列表 —— 即
        # 每一个刚被加载进 ``ctx.mcp_tools`` 的 MCP 工具，加上用于显式完成
        # 的 ``base/finish``。
        lo_context: list[str] = []
        if lo_available is False:
            lo_context = [
                'LibreOffice is NOT installed in this sandbox — use Python '
                'libraries (openpyxl, python-pptx) and Node packages '
                '(pptxgenjs, docx) for all document operations.'
            ]
        project_context = await discover_project_context(
            runtime,
            task_context_lines=lo_context,
        )
        resolved_prompt_text = await resolve_system_prompt(
            runtime,
            project_context=project_context,
            task_memory_enabled=True,
        )
        local_tool_names = ['base/finish']
        if task_input.ask_user and task_input.ask_user.get('enabled', True):
            ctx.tool_configs['base/ask_user'] = dict(task_input.ask_user)
            local_tool_names.append('base/ask_user')
        mcp_tool_names = [
            name for name in sorted(getattr(ctx, 'mcp_tools', {}).keys()) if not name.endswith('_dump_state')
        ]
        agent_tool_names = [
            *local_tool_names,
            *task_input.agent_tools,
            *mcp_tool_names,
        ]
        logger.debug(
            '[aidata] running agent, timeout=%ds tools=%d (local=%d agent=%d mcp=%d)',
            task_input.agent_timeout,
            len(agent_tool_names),
            len(local_tool_names),
            len(task_input.agent_tools),
            len(agent_tool_names) - len(local_tool_names) - len(task_input.agent_tools),
        )
        agent_kwargs = build_agent_kwargs(
            task_input,
            user_message=user_message,
            system_prompt=resolved_prompt_text,
            tools=agent_tool_names,
        )
        # ``initial_trajectory`` 会从 task.yaml 解析进 ``task_input``，但
        # 不会在此处转发给 ``run_agent`` —— 目前的规范 agent
        # （``main_agent``）尚未支持种子历史 kwarg，而本 engine 模块也
        # 不允许伸手进 agent 层。 在这项支持落地之前，
        # ``tool_called(scope: seed)`` 检查会返回 []。
        agent_timeout_started_at = time.monotonic()
        async with async_timeout(task_input.agent_timeout):
            agent_output = await task_input.run_agent(**agent_kwargs)
            trajectory = getattr(agent_output, 'main_trajectory', [])
            logger.info('[aidata] agent completed, %d messages in trajectory', len(trajectory))

            # 8. 在关闭 MCP 之前下载 state —— R3 的 dump_state 工具
            #    需要活跃会话才能回调 sidecar mock。
            after_agent_state = await download_state(
                runtime,
                services=task_input.services,
                dump_state_args=resolved_dump_state,
            )
            after_agent_path = _dump_state_to_file(
                after_agent_state,
                task_id,
                task_input.case_id,
                phase='after_agent',
            )
            if after_agent_path:
                state_snapshots['after_agent'] = after_agent_path

            state = dict(after_agent_state)
            crontab_entries = await collect_crontab_state(runtime)
            if crontab_entries:
                state['crontab_entries'] = crontab_entries
            state['trajectory'] = list(trajectory) if trajectory else []
            # ``_initial_trajectory_len`` 刻意保持未设置 —— 种子回合并未
            # 真正出现在 ``trajectory`` 中（见上文 ``run_agent`` 调用点），
            # 因此在此处上报其长度会错误切分
            # ``tool_called(scope: seed/live)``。
            state['agent_answer'] = extract_final_answer(state)

            # 8b. 在同一个 agent 超时预算内关闭 MCP，防止清理阶段在 agent
            # 之后再引入运行时客户端轮询超时。
            await _close_mcp_tools_with_agent_timeout(
                ctx,
                task_input=task_input,
                task_id=task_id,
                agent_timeout_started_at=agent_timeout_started_at,
                propagate_timeout=True,
            )

        # 9. 解析 rubric。 已提交的 cases/<id>/rubric.yaml 是必需的 ——
        #    没有 LLM 回退。 缺失的 case_id、缺失的 rubric 文件、无法读取或
        #    为空的 rubric 都是硬性错误：我们拒绝在伪造的 rubric 上评分。
        if not task_input.case_id:
            return _invalid_task_output(
                task_id=task_id,
                start=start,
                error_message='no case_id set — a committed cases/<id>/rubric.yaml is required for scoring',
                termination_reason=_TERMINATION_INITIALIZATION_ERROR,
                failure_phase='rubric_resolution',
                failure_kind='missing_case_id',
                state_snapshots=state_snapshots,
            )
        rubric_file = case_rubric_path(task_input.case_id)
        if not rubric_file.exists():
            return _invalid_task_output(
                task_id=task_id,
                start=start,
                error_message=f'no rubric.yaml for case {task_input.case_id!r}; hand-write {rubric_file}',
                termination_reason=_TERMINATION_INITIALIZATION_ERROR,
                failure_phase='rubric_resolution',
                failure_kind='missing_rubric',
                state_snapshots=state_snapshots,
            )
        try:
            rubric = load_case_rubric(task_input.case_id)
            logger.debug('[aidata] using committed rubric: %d checks (%s)', len(rubric.get('checks', [])), rubric_file)
        except Exception as exc:
            logger.error('[aidata] rubric load failed: %s', exc)
            return _invalid_task_output(
                task_id=task_id,
                start=start,
                error_message=f'rubric load failed: {exc}',
                termination_reason=_TERMINATION_INITIALIZATION_ERROR,
                failure_phase='rubric_resolution',
                failure_kind='rubric_load_failed',
                state_snapshots=state_snapshots,
            )

        # 10. 针对之前下载的 state 评估 rubric。

        # 暴露 case 声明的 sqlite 路径（占位符已解析），以便
        # ``state_equals(sql:…)`` 探针能以只读方式打开它。 该路径是
        # *沙箱内* 路径（典型值
        # ``/tmp/mock-run/<uuid>/online_shopping.db`` —— 由运行在
        # aio-agent 容器中的 stdio MCP 子进程写入）。 rubric 评估器
        # 本身运行在 harness 宿主上，但会通过 ``runtime.run_command``
        # 在 *沙箱内* 执行 sqlite 查询；见
        # :func:`engine.rubric._snapshot_sql_scalar_via_runtime` 的
        # 打开侧契约。 这样避免每次探测都把整个 DB 文件（online_shopping
        # 场景常达数 MB）传回宿主，也移除了此前预先下载到
        # ``/tmp/nexus-aidata-rubric-sqlite/`` 的临时方案。
        #
        # 若 mock 从未写过该文件（agent 在首次工具调用前崩溃，或
        # case 的 sql: 探针都属于愿望性质），沙箱内 python3 片段会把
        # sqlite 的 ``unable to open database file`` 错误暴露为每探针
        # 一个 ``MISSING`` 判定 —— 与
        # ``AidataTaskInput.rubric_sqlite_path`` 上记录的失败即拒契约一致。
        if resolved_rubric_sqlite_path:
            state['_rubric_sqlite_path'] = resolved_rubric_sqlite_path
        # 若 sqlite oracle 位于 sidecar 服务容器中，rubric 必须把它的
        # python3+sqlite3 探针分发到该运行时 —— main 容器无法看见 sidecar
        # 的文件系统（online_shopping 会在其 mock-server sidecar 内写入
        # /var/lib/mock-server/online_shopping/rollout.db）。
        # engine/rubric.py::_resolve_sql_runtime 借助
        # require_context().get_runtime(<service_name>) 拿到该运行时。
        if task_input.rubric_sqlite_service:
            state['_rubric_sqlite_service'] = task_input.rubric_sqlite_service

        # 10a. v4.2 测试协议：若 case 提供 `cases/<id>/run-test.sh`，则
        # 触发一次并解析它按契约必须留下的 junit 报告
        # /workspace/tests/report.junit.xml。 未提供 run-test.sh 的
        # case（迁移前的 8 个 case 全部如此）会完全跳过本块 —— rubric
        # 依然按之前流程被评估。
        junit_report: JunitReport | None = None
        junit_ctx = None
        junit_summary: dict[str, Any] = {}
        if task_input.case_id:
            junit_report, junit_ctx, junit_summary = await _maybe_run_case_tests(
                runtime,
                task_input.case_id,
                state,
            )

        # 10b. 把落盘产物暂存进 state，让 ``judge_claim`` 检查能通过
        # ``from: file:<path>`` 对 agent 持久化的文件评分。
        # 尽力而为：文件缺失时切片留空，judge 会报告 EMPTY_SOURCE 而非崩溃。
        await _read_sandbox_files(runtime, state, rubric)

        rubric_results = await rubric_lib.run_rubric_async(
            state,
            rubric,
            runtime=runtime,
            junit_ctx=junit_ctx,
        )
        score_summary = rubric_lib.compute_scores(rubric, rubric_results)
        state_summary = build_summary(state)
        # 默认仍由 ScorePolicy 折算 hard-gate 后的加权分。 若 rubric 配置了
        # config.reward.script，则脚本会在 checks 全部完成后接收结果 JSON，
        # 它的输出取代 ScorePolicy，成为最终 score。 hard gate 仍会在脚本
        # 输出之上封顶，脚本不能绕过硬门控。
        clamped_score = float(score_summary['score'])
        score_policy = rubric_lib.ScorePolicy.resolve(rubric)
        score = rubric_lib.finalize_score(clamped_score, score_policy)
        score_policy_out = score_policy.as_dict()
        passed = score_summary['passed']
        total = score_summary['total']
        dimension_scores = score_summary.get('dimension_scores') or {}
        rubric_version = str(rubric.get('version') or '2.0')
        rubric_termination = _termination_from_rubric_results(rubric_results)
        extra_fields: dict[str, Any] = {}
        if rubric_termination is not None:
            termination_reason, failure_kind, invalid_check_ids = rubric_termination
            _report_hytuner_termination(termination_reason)
            extra_fields = _hytuner_extra_fields(
                termination_reason,
                failure_phase='rubric_evaluation',
                failure_kind=failure_kind,
                score_valid=False,
                invalid_check_ids=invalid_check_ids,
            )
        else:
            try:
                reward_outcome = await resolve_reward(
                    rubric,
                    rubric_results,
                    score_summary,
                    runtime=runtime,
                    cwd=CASE_DIR_IN,
                    state_summary=state_summary,
                )
            except RewardScriptError as exc:
                termination_reason = _TERMINATION_REWARD_ERROR
                failure_kind = 'reward_script_error'
                detail = f'{exc.kind}: {exc}'
                score = 0.0
                clamped_score = 0.0
                raw_config = rubric.get('config') or {}
                raw_reward_cfg = raw_config.get('reward') if isinstance(raw_config, dict) else None
                reward_cfg = raw_reward_cfg if isinstance(raw_reward_cfg, dict) else {}
                score_policy_out = {
                    'mode': 'script',
                    'script': reward_cfg.get('script'),
                    'timeout_sec': reward_cfg.get('timeout_sec', 15.0),
                }
                _report_hytuner_termination(termination_reason)
                extra_fields = _hytuner_extra_fields(
                    termination_reason,
                    failure_phase='reward_calculation',
                    failure_kind=failure_kind,
                    score_valid=False,
                    detail=detail,
                )
                logger.warning('[aidata] reward script failed task=%s kind=%s error=%s', task_id, exc.kind, exc)
            else:
                score = reward_outcome.score
                clamped_score = reward_outcome.continuous_score
                score_policy_out = reward_outcome.policy
                termination_reason = None
                failure_kind = None
        _log_rubric_summary(
            task_id,
            task_input.case_id,
            rubric_results,
            termination_reason=termination_reason,
            failure_kind=failure_kind,
        )

        if score_summary.get('hard_gate_triggered'):
            logger.warning(
                '[aidata] HARD GATE failed task=%s gates=%s raw_score=%.3f weighted_capped_score=%.3f final_score=%.3f',
                task_id,
                score_summary.get('hard_gate_failed'),
                score_summary.get('raw_score', clamped_score),
                score_summary.get('score', 0.0),
                score,
            )

        if dimension_scores:
            dim_str = ' '.join(f'{d}={v:.3f}' for d, v in dimension_scores.items())
            logger.info(
                '[aidata] done task=%s version=%s score=%.3f cont=%.3f policy=%s (%d/%d) dims={ %s }',
                task_id,
                rubric_version,
                score,
                clamped_score,
                score_policy_out,
                passed,
                total,
                dim_str,
            )
        else:
            logger.info(
                '[aidata] done task=%s version=%s score=%.3f cont=%.3f policy=%s (%d/%d)',
                task_id,
                rubric_version,
                score,
                clamped_score,
                score_policy_out,
                passed,
                total,
            )

        meta_obj = task_input.case_metadata
        case_metadata_out = meta_obj.model_dump() if meta_obj is not None else {}

        state_dump_path = _dump_state_to_file(state, task_id, task_input.case_id)

        return AidataTaskOutput(
            task_id=task_id,
            score=score,
            continuous_score=clamped_score,
            score_policy=score_policy_out,
            rubric=rubric,
            rubric_results=rubric_results,
            rubric_version=rubric_version,
            dimension_scores=dimension_scores,
            junit_summary=junit_summary,
            case_metadata=case_metadata_out,
            state_summary=state_summary,
            state_dump_path=state_dump_path,
            state_snapshots=state_snapshots,
            extra_fields=extra_fields,
            wall_time_s=time.monotonic() - start,
        )

    except (asyncio.TimeoutError, TimeoutError):
        return _invalid_task_output(
            task_id=task_id,
            start=start,
            error_message=f'agent timeout after {task_input.agent_timeout}s',
            termination_reason=_TERMINATION_TASK_TIMEOUT,
            failure_phase='agent_run',
            failure_kind='agent_timeout',
            state_snapshots=state_snapshots,
        )
    except Exception as exc:
        logger.error('[aidata] failed: %s', exc, exc_info=True)
        return _invalid_task_output(
            task_id=task_id,
            start=start,
            error_message=f'task error: {exc}',
            termination_reason=_TERMINATION_UNHANDLED_ERROR,
            failure_phase='task',
            failure_kind='unhandled_exception',
            state_snapshots=state_snapshots,
        )
    finally:
        await _close_mcp_tools_with_agent_timeout(
            ctx,
            task_input=task_input,
            task_id=task_id,
            agent_timeout_started_at=agent_timeout_started_at,
            propagate_timeout=False,
        )


# junit verify 脚本从这里查找 agent 轨迹。 ``verify.py`` 会先探测
# ``$TRAJECTORY_PATH`` 再回退到此绝对路径，因此写到这里并同时导出
# 该环境变量能覆盖两条查找路径。
TRAJECTORY_PATH_IN = '/workspace/trajectory.jsonl'


async def _upload_trajectory_for_tests(runtime: Any, state: dict | None) -> bool:
    """将 ``state['trajectory']`` 序列化为 JSONL 并上传到沙箱，供 junit
    验证脚本读取。 写入成功时返回 True。 尽力而为：绝不抛出（即便
    轨迹无法暂存，测试执行也必须继续）。
    """
    if not state:
        return False
    trajectory = state.get('trajectory') or []
    if not trajectory:
        return False
    try:
        lines = []
        for msg in trajectory:
            # ensure_ascii=False 保持工具名与中文内容可被验证器的
            # 子串/正则检查匹配。
            lines.append(json.dumps(msg, ensure_ascii=False))
        content = '\n'.join(lines)
        await upload_text_file(runtime, TRAJECTORY_PATH_IN, content)
        logger.debug(
            '[aidata] staged trajectory for junit verifiers: %d msgs -> %s',
            len(trajectory),
            TRAJECTORY_PATH_IN,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning('[aidata] failed to stage trajectory for tests: %s', exc)
        return False


def _rubric_file_slices(rubric: dict | None) -> list[str]:
    """收集所有 rubric 检查项中被 ``from: file:<path>`` 选择器引用的沙箱
    绝对路径。 返回去重后的列表。
    """
    paths: list[str] = []
    seen: set[str] = set()
    for check in (rubric or {}).get('checks', []) or []:
        assert_cfg = check.get('assert') if isinstance(check, dict) else None
        if not isinstance(assert_cfg, dict):
            continue
        for key in ('from', 'source'):
            raw = assert_cfg.get(key)
            if isinstance(raw, str) and raw.startswith('file:'):
                path = raw.split(':', 1)[1].strip()
                if path and path not in seen:
                    seen.add(path)
                    paths.append(path)
    return paths


async def _read_sandbox_files(runtime: Any, state: dict, rubric: dict | None) -> None:
    """把 rubric 引用的每一个 ``from: file:<path>`` 落盘产物从沙箱读取到
    ``state['_sandbox_files'][path]``，让 ``judge_claim``（以及其他文本
    断言）可以对 agent 持久化的文件评分，而不是仅评分从轨迹提取的
    ``agent_answer``。

    尽力而为：每个读取失败都会被记录并跳过 —— 缺失的文件只是让切片
    留空（judge 会报告 EMPTY_SOURCE 而非让任务崩溃）。
    """
    paths = _rubric_file_slices(rubric)
    if not paths:
        return

    files: dict[str, str] = {}
    for path in paths:
        try:
            res = await runtime.run_command(f'cat {path}', timeout=15)
        except Exception as exc:  # noqa: BLE001
            logger.warning('[aidata] failed to read sandbox file %s: %s', path, exc)
            continue
        # 非零退出（文件不存在）依然会返回；此时 stdout 为空。
        text = getattr(res, 'stdout', '') or ''
        if text.strip():
            files[path] = text
            logger.debug('[aidata] staged sandbox file for rubric: %d chars (%s)', len(text), path)
        else:
            logger.info('[aidata] sandbox file empty/missing at %s', path)
    if files:
        state['_sandbox_files'] = files


async def _maybe_run_case_tests(
    runtime: Any, case_id: str, state: dict | None = None
) -> tuple[
    JunitReport | None,
    rubric_lib.JunitLookupContext | None,
    dict[str, Any],
]:
    """若 case 提供了 ``run-test.sh``，则上传并调用它，然后解析其
    junit 输出。 返回 ``(report, junit_ctx, summary)`` —— 若 case 未启用
    v4.2 测试协议，其中任一都可能是 ``None`` 或空。 从不抛出：失败时
    记录日志并返回一个占位 summary，从而让 rubric 依然运行（缺失
    junit 只会让每个 ``test_result`` 检查报告 MISSING / NO_JUNIT 而不是
    崩溃）。
    """
    case_dir_host = CASES_DIR / case_id
    # 两个可接受位置 —— 当同时存在时 case-root 优先于 tests/
    # （RUBRIC.md §1 “engine 会优先调它”）。 保留 tests/run-test.sh
    # 作为回退，允许 case 把 runner 与 pytest 资产放在一起。
    run_test_host = case_dir_host / 'run-test.sh'
    run_test_rel = 'run-test.sh'
    if not run_test_host.is_file():
        alt = case_dir_host / 'tests' / 'run-test.sh'
        if alt.is_file():
            run_test_host = alt
            run_test_rel = 'tests/run-test.sh'
        else:
            return None, None, {}

    run_test_in = f'{CASE_DIR_IN}/{run_test_rel}'
    logger.debug('[aidata] running %s (case=%s)', run_test_rel, case_id)

    # 显式上传 check_map.yaml 与 run-test.sh，即便 upload_case_assets
    # 已经上传了整个 case 目录；此操作幂等，可确保对没有 case 资产的
    # bootstrap 任务它们也存在。
    prep = f'mkdir -p {TESTS_DIR_IN} && chmod +x {run_test_in} 2>/dev/null || true'
    with contextlib.suppress(Exception):
        await runtime.run_command(prep, timeout=10)

    # 让 junit 验证器能拿到 agent 轨迹。 case 的 verify 脚本从文件里
    # 读取（先 ``$TRAJECTORY_PATH``，然后 ``/workspace/trajectory.jsonl``）；
    # 之前没有任何一处写入过它，因此即便 agent 明显调用了预期工具，
    # 每一个基于轨迹的检查也会报告 ``trajectory_present=False``。 把
    # ``state['trajectory']`` 作为 JSONL 落盘并通过环境变量指向它。
    traj_env_prefix = ''
    traj_written = await _upload_trajectory_for_tests(runtime, state)
    if traj_written:
        traj_env_prefix = f'TRAJECTORY_PATH={TRAJECTORY_PATH_IN} '

    try:
        result = await runtime.run_command(
            f'{traj_env_prefix}bash {run_test_in}',
            cwd=CASE_DIR_IN,
            timeout=600,
        )
    except Exception as exc:
        logger.error('[aidata] run-test.sh invocation failed: %s', exc)
        return None, None, {'error': f'run_command_failed: {exc}'}

    exit_code = getattr(result, 'return_code', None)
    if exit_code is None:
        exit_code = getattr(result, 'exit_code', None)
    stdout_head = (getattr(result, 'stdout', '') or '')[:400]
    stderr_head = (getattr(result, 'stderr', '') or '')[:400]
    logger.info('[aidata] run-test.sh exit=%s stdout[:400]=%r stderr[:400]=%r', exit_code, stdout_head, stderr_head)

    # 从沙箱抓取 junit.xml。 缺失时提供工程师友好的错误信息 —— 这是
    # 最常见的 “case 作者忘记接入测试协议” 失败模式。
    junit_text = await _download_file_text(runtime, JUNIT_REPORT_PATH_IN)
    if junit_text is None:
        logger.warning(
            '[aidata] %s missing after run-test.sh; test_result checks will '
            'surface MISSING. If this case uses v4.2 assert.type=test_result, '
            'ensure run-test.sh writes a junit XML to that path.',
            JUNIT_REPORT_PATH_IN,
        )
        return (
            None,
            None,
            {
                'error': 'junit_report_missing',
                'exit_code': exit_code,
            },
        )

    try:
        report = parse_junit_xml(junit_text)
    except JunitParseError as exc:
        logger.error('[aidata] junit parse failed: %s', exc)
        return (
            None,
            None,
            {
                'error': f'junit_parse_failed: {exc}',
                'exit_code': exit_code,
            },
        )

    check_map_host = case_dir_host / 'tests' / 'check_map.yaml'
    explicit_map: dict[str, str] = {}
    if check_map_host.exists():
        try:
            explicit_map = load_check_map(case_dir_host / 'tests')
        except Exception as exc:
            logger.error('[aidata] check_map.yaml parse failed: %s', exc)
            # 非致命：回退到基于命名约定的解析。

    junit_ctx = rubric_lib.JunitLookupContext(
        report=report,
        explicit_map=explicit_map,
    )
    summary = {
        'exit_code': exit_code,
        'total': report.total,
        'passed': report.passed,
        'failed': report.failed,
        'testcase_names': report.all_names()[:64],
    }
    logger.info('[aidata] junit report: %d total / %d passed / %d failed', report.total, report.passed, report.failed)
    return report, junit_ctx, summary


async def _download_file_text(runtime: Any, path: str) -> str | None:
    """尽力而为地把沙箱内某个文件抓回作为 UTF-8 文本。 遇到任何错误
    （文件缺失、权限、内容异常）都返回 ``None``。 用于把 junit 报告
    读回来供 rubric 阶段检查。
    """
    try:
        resp = await runtime.download_file(
            files=[{'path': path, 'encoding': 'base64'}],
        )
    except Exception as exc:
        logger.debug('[aidata] download_file %s failed: %s', path, exc)
        return None
    files = getattr(resp, 'files', None) or getattr(resp, 'results', None) or []
    if not files:
        return None
    content = getattr(files[0], 'content', None)
    if not content:
        return None
    try:
        return base64.b64decode(content).decode('utf-8', errors='replace')
    except Exception as exc:
        logger.debug('[aidata] %s decode failed: %s', path, exc)
        return None
