"""Navi agent 与 WindowsAgentArena 运行时的集成模块。

本模块实现了 Navi agent 的核心循环：观察(Observe) → 规划(Plan) → 执行(Act)，
通过截屏和无障碍树(a11y tree)获取环境状态，调用 LLM 进行决策，
再通过 WAA 的 computer API 执行操作。

使用标准 nexus function-calling 机制：ToolExecutor + OpenAI tools API。
"""

import asyncio
import contextlib
import json
import logging

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolExecutor,
    WithFinalAnswer,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.runtime.providers.ags.waa import require_raw_waa_runtime
from nexus.trace import get_current_span_id, span_context, update_span

from .observation import observe
from .prompt import PLANNING_SYSTEM_PROMPT, build_observation_message
from .utils import coerce_prompt, image_to_base64_png, image_to_data_url, resize_image


logger = logging.getLogger(__name__)

# Import tools module to trigger registration
Registry.import_module('nexus.extensions.agents.navi.tools')


def _is_non_retryable_llm_request_error(exc: Exception) -> bool:
    # 临时接入 GPT/Kimi 等 OpenAI-compatible 后端时，余额、鉴权、内容策略等错误重试不会恢复；
    # 这里提前识别并停止，避免全量 WAA 复现阶段无意义消耗额度。
    class_name = type(exc).__name__
    text = str(exc).lower()
    if class_name in {
        'BadRequestError',
        'AuthenticationError',
        'PermissionDeniedError',
        'NotFoundError',
        'UnprocessableEntityError',
    }:
        return True
    markers = (
        'content management policy',
        'content filter',
        'content_filter',
        'responsibleaipolicyviolation',
        'invalid request',
        'invalid_api_key',
        'authentication',
        'permission denied',
        'insufficient budget',
        'insufficient balance',
        'exceeded your current quota',
        'billing hard limit',
    )
    return any(marker in text for marker in markers)


class NaviAgentInput(AgentRunInput, WithUserPrompt):
    """Input for the Nexus Navi agent.

    The implementation intentionally keeps Navi's requested operating mode as
    defaults: ``obs_view=screen``, ``som_origin=a11y``, ``use_last_screen=True``
    and ``action_space=code_block``.
    """

    max_iterations: int = 15
    """Maximum observe-plan-act iterations before stopping."""

    sleep_after_execution: float = 0.5
    """Delay after code execution before the next observation."""

    initial_observation_delay: float = 2.0
    """Delay before the first observation, to let setup-launched apps finish rendering.

    The initial observation happens before the loop starts, so it is not covered by
    ``sleep_after_execution``. Without this wait, the first screenshot may capture a
    half-rendered window. This is common with Win11 File Explorer's WinUI/XAML
    toolbar: UIA controls can be ready before DWM has repainted the top chrome,
    leaving gray placeholder areas in the first screenshot."""

    tool_call_retries: int = 2
    """Number of retries when LLM does not produce a valid tool call."""

    observe_retries: int = 3
    """Number of retries when observation (screenshot + a11y tree) fails."""

    trace_dir: str | None = None
    """Directory for saving per-step observability data (screenshots, a11y, trajectory).
    If set, a subdirectory named by task_id (or run_id) will be created under this path."""

    task_id: str | None = None
    """Task ID passed from the WAA task, used for trace subdirectory naming."""


class NaviExitStatus(str, Enum):
    """Navi agent exit status — why the agent loop ended.

    Aligned with MainExitStatus pattern for consistency across agents.
    """

    DONE = 'done'  # agent 判断任务已完成
    FAIL = 'fail'  # agent 判断任务不可行
    MAX_ITERATIONS = 'max_iterations'  # 达到最大迭代次数
    OBSERVE_FAILED = 'observe_failed'  # 环境观察阶段异常
    CONTEXT_LENGTH_EXCEEDED = 'context_length_exceeded'  # LLM 上下文长度超限
    INFERENCE_TIMEOUT = 'inference_timeout'  # LLM 推理超时
    INFERENCE_ERROR = 'inference_error'  # LLM 推理调用异常
    TOOL_CALL_ERROR = 'tool_call_error'  # LLM 未产生有效 tool call（重试耗尽）
    ERROR = 'error'  # 未预期的运行时错误


class NaviAgentOutput(AgentRunOutput, WithFinalAnswer):
    """Navi agent 的执行结果。

    Attributes:
        exit_status: 退出状态，取值为 NaviExitStatus 枚举值。
        iterations: 实际执行的迭代次数。
        actions: 所有已执行动作的记录列表，每项包含 tool_name、tool_args、content。
        final_answer: agent 的最终结论/回答，可为 None。
    """

    exit_status: str = NaviExitStatus.MAX_ITERATIONS.value
    iterations: int = 0
    actions: list[dict] = []
    usage: dict[str, int] = {}


@dataclass(frozen=True)
class NaviActionRecord:
    """Planner 输出的单次动作记录（不可变）。"""

    tool_name: str
    """调用的 tool 名称。"""

    code: str
    """execute_code 的代码内容，或其他 tool 的关键参数。"""

    content: str
    """LLM 的分析文本（assistant message content）。"""


class NaviSessionState:
    """单次运行的 Navi 会话状态（可变）。

    保存在 Pydantic 模型之外，用于在迭代过程中维护截图历史、
    动作记录、记忆块等可变状态。
    """

    def __init__(self, user_prompt: str | list[Any]) -> None:
        self.instruction = coerce_prompt(user_prompt)
        self.memory: str = ''
        self.previous_actions: list[NaviActionRecord] = []
        self.exit_status = NaviExitStatus.MAX_ITERATIONS
        self.final_answer: str | None = None
        self.usage: dict[str, int] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }

    @property
    def memory_block(self) -> str:
        """格式化记忆块用于渲染到 prompt 中。"""
        if self.memory:
            return f'```memory\n{self.memory}\n```'
        return '```memory\n# empty memory block\n```'


@agent_registry.register('navi_agent')
async def run_navi_agent(agent_input: NaviAgentInput) -> NaviAgentOutput:
    """Run Navi as a Nexus agent against a WAA raw runtime.

    Each step mirrors the original WAA loop: observe screen and a11y tree,
    build an a11y-only SoM prompt, call LLM with tools, execute the returned
    tool call, then loop.
    """
    # ---- 初始化阶段 ----
    ctx = require_context()
    client = ctx.openai_client

    # 获取 WAA 原始运行时（支持从 nexus 风格 runtime 自动转换）
    runtime = require_raw_waa_runtime(await ctx.get_runtime())
    state = NaviSessionState(agent_input.user_prompt)
    iteration = 0

    # 可观测性：trace 目录
    trace_path: Path | None = None
    if agent_input.trace_dir:
        trace_name = agent_input.task_id or ctx.run_id or 'unknown'
        trace_path = Path(agent_input.trace_dir) / trace_name
        trace_path.mkdir(parents=True, exist_ok=True)
        logger.debug('Trace directory: %s', trace_path)

        # 保存 agent input
        _save_agent_input(trace_path, agent_input)

    # 创建 ToolExecutor
    executor = ToolExecutor([
        tool_registry.get('navi/execute_code'),
        tool_registry.get('navi/wait'),
        tool_registry.get('navi/done'),
        tool_registry.get('navi/fail'),
    ])
    tool_schemas = executor.get_tool_schemas()

    logger.debug(
        'Navi agent started — max_iterations=%d, sleep_after_execution=%.2f',
        agent_input.max_iterations,
        agent_input.sleep_after_execution,
    )

    # 对话历史（累积模式）
    history: list[dict[str, Any]] = [
        {'role': 'system', 'content': PLANNING_SYSTEM_PROMPT},
    ]

    # ---- 初始观察（循环前）----
    # 首次观察前等待，让 setup 启动的应用完成渲染，避免抓到半渲染画面。
    if agent_input.initial_observation_delay > 0:
        await asyncio.sleep(agent_input.initial_observation_delay)

    initial_obs = None
    for obs_attempt in range(agent_input.observe_retries + 1):
        try:
            initial_obs = await observe(runtime)
            break
        except Exception as e:
            if obs_attempt < agent_input.observe_retries:
                logger.warning(
                    'Initial observation failed, retry %d/%d: %s',
                    obs_attempt + 1,
                    agent_input.observe_retries,
                    e,
                )
                await asyncio.sleep(2.0)
            else:
                logger.exception('Initial observation failed after %d retries', agent_input.observe_retries)
                state.exit_status = NaviExitStatus.OBSERVE_FAILED

    if initial_obs is None:
        return NaviAgentOutput(
            exit_status=state.exit_status.value,
            iterations=0,
            final_answer=state.final_answer,
            actions=[],
            usage=state.usage,
        )

    # 保存初始 trace
    if trace_path:
        await _save_step_observation(trace_path, 0, initial_obs)

    # 构建初始 user message
    annotated_url = image_to_data_url(resize_image(initial_obs.annotated_screenshot))
    obs_text = build_observation_message(
        instruction=state.instruction,
        window_title=initial_obs.fg_window_title,
        window_names=initial_obs.fg_window_names,
        clipboard_content=initial_obs.computer_clipboard,
        a11y_entities=initial_obs.a11y_entities,
        screen_size=initial_obs.screenshot.size,
    )
    history.append({
        'role': 'user',
        'content': [
            {'type': 'text', 'text': obs_text},
            {'type': 'image_url', 'image_url': {'url': annotated_url}},
        ],
    })

    # 当前观察（用于 update_computer）
    current_obs = initial_obs

    try:
        for step in range(1, agent_input.max_iterations + 1):
            iteration = step
            logger.debug('[iter %d/%d] Step started', iteration, agent_input.max_iterations)

            with span_context(
                'agent.iteration',
                attributes={
                    'type': 'agent_iteration',
                    'iteration': iteration,
                },
            ):
                # ---- LLM 调用 ----
                tool_call_result = await _call_llm_with_tools(
                    client=client,
                    messages=history,
                    tool_schemas=tool_schemas,
                    retries=agent_input.tool_call_retries,
                    state=state,
                    iteration=iteration,
                    max_iterations=agent_input.max_iterations,
                )

                if tool_call_result is None:
                    break

                tool_name, tool_args, assistant_content, assistant_msg = tool_call_result
                logger.debug(
                    '[iter %d/%d] Tool call — name=%s, args=%s',
                    iteration,
                    agent_input.max_iterations,
                    tool_name,
                    json.dumps(tool_args, ensure_ascii=False)[:200],
                )

                # 追加 assistant message 到历史
                history.append(assistant_msg)
                tool_call_id = assistant_msg.get('tool_calls', [{}])[0].get('id', '')

                # ---- 同步 computer 状态并执行 tool ----
                with span_context(
                    'agent.observation',
                    attributes={
                        'type': 'agent_observation',
                        'iteration': iteration,
                    },
                ):
                    await runtime.update_computer(
                        rects=[
                            [
                                int(ent['shape']['x']),
                                int(ent['shape']['y']),
                                int(ent['shape']['x'] + ent['shape']['width']),
                                int(ent['shape']['y'] + ent['shape']['height']),
                            ]
                            for ent in current_obs.a11y_entities
                        ],
                        window_rect=[0, 0, current_obs.screenshot.width, current_obs.screenshot.height],
                        screenshot=image_to_base64_png(current_obs.screenshot),
                        scale=(1.0, 1.0),
                        clipboard_content=current_obs.computer_clipboard,
                        swap_ctrl_alt=False,
                    )

                    # 执行 tool
                    execution_error: str | None = None
                    try:
                        await executor.call(tool_name, tool_args)
                    except Exception as exc:
                        logger.exception('[iter %d/%d] Tool execution failed', iteration, agent_input.max_iterations)
                        error_code = tool_args.get('code', str(tool_args))
                        execution_error = f'EXECUTION ERROR: {exc}'
                        state.previous_actions.append(
                            NaviActionRecord(
                                tool_name=tool_name,
                                code=f'{error_code}\n# {execution_error}',
                                content=assistant_content,
                            )
                        )

                # ---- 处理 tool 响应 ----
                if execution_error is None:
                    # 更新 memory
                    new_memory = ctx.tool_state.get('memory')
                    if new_memory is not None:
                        state.memory = new_memory

                    # 记录动作
                    code_or_summary = tool_args.get('code') or tool_args.get('summary') or tool_args.get('reason') or ''
                    state.previous_actions.append(
                        NaviActionRecord(
                            tool_name=tool_name,
                            code=code_or_summary,
                            content=assistant_content,
                        )
                    )

                # 检查退出条件
                exit_status = ctx.tool_state.get('exit_status')
                if exit_status == 'done':
                    state.exit_status = NaviExitStatus.DONE
                    state.final_answer = 'DONE'
                    logger.info('[iter %d/%d] Agent completed task', iteration, agent_input.max_iterations)
                    history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': 'Task completed.'})
                    break
                if exit_status == 'fail':
                    state.exit_status = NaviExitStatus.FAIL
                    state.final_answer = 'FAIL'
                    logger.info('[iter %d/%d] Agent marked task as infeasible', iteration, agent_input.max_iterations)
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call_id,
                        'content': 'Task marked as infeasible.',
                    })
                    break

                # execute_code 和 wait 后的延时
                if tool_name in ('execute_code', 'wait'):
                    await asyncio.sleep(agent_input.sleep_after_execution)

                # ---- Observe（执行后获取新屏幕状态）----
                next_obs = None
                for obs_attempt in range(agent_input.observe_retries + 1):
                    try:
                        next_obs = await observe(runtime)
                        break
                    except Exception as e:
                        if obs_attempt < agent_input.observe_retries:
                            logger.warning(
                                '[iter %d/%d] Post-action observation failed, retry %d/%d: %s',
                                iteration,
                                agent_input.max_iterations,
                                obs_attempt + 1,
                                agent_input.observe_retries,
                                e,
                            )
                            await asyncio.sleep(2.0)
                        else:
                            logger.exception(
                                '[iter %d/%d] Post-action observation failed after retries',
                                iteration,
                                agent_input.max_iterations,
                            )

                if next_obs is None:
                    # 观察失败，追加简单 tool response 并退出
                    history.append({'role': 'tool', 'tool_call_id': tool_call_id, 'content': 'Observation failed.'})
                    state.exit_status = NaviExitStatus.OBSERVE_FAILED
                    break

                # 保存 trace
                if trace_path:
                    await _save_step_observation(trace_path, iteration, next_obs)

                # 构建 tool response（执行结果 + 新观察）
                next_annotated_url = image_to_data_url(resize_image(next_obs.annotated_screenshot))
                next_obs_text = build_observation_message(
                    window_title=next_obs.fg_window_title,
                    window_names=next_obs.fg_window_names,
                    clipboard_content=next_obs.computer_clipboard,
                    a11y_entities=next_obs.a11y_entities,
                    screen_size=next_obs.screenshot.size,
                )

                result_prefix = execution_error or 'Action executed successfully.'
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call_id,
                    'content': [
                        {'type': 'text', 'text': f'{result_prefix}\n\nUpdated screen observation:\n\n{next_obs_text}'},
                        {'type': 'image_url', 'image_url': {'url': next_annotated_url}},
                    ],
                })

                # 更新当前观察
                current_obs = next_obs

                # ---- 保存 trace（trajectory）----
                if trace_path:
                    await _save_step_trajectory(trace_path, iteration, history)

    except Exception:
        state.exit_status = NaviExitStatus.ERROR
        logger.exception('Navi agent loop encountered an unexpected error')
    finally:
        span_id = get_current_span_id()
        if span_id:
            try:
                update_span(
                    span_id,
                    attributes={
                        'agent.exit_status': state.exit_status.value,
                        'agent.iterations': iteration,
                    },
                )
            except Exception:
                logger.warning('Failed to update span attributes for Navi agent', exc_info=True)

    logger.info(
        'Navi agent finished — exit_status=%s, iterations=%d',
        state.exit_status.value,
        iteration,
    )

    return NaviAgentOutput(
        exit_status=state.exit_status.value,
        iterations=iteration,
        final_answer=state.final_answer,
        actions=[
            {'tool_name': record.tool_name, 'code': record.code, 'content': record.content}
            for record in state.previous_actions
        ],
        usage=state.usage,
    )


async def _call_llm_with_tools(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict],
    retries: int,
    state: NaviSessionState,
    iteration: int,
    max_iterations: int,
) -> tuple[str, dict, str, dict] | None:
    """调用 LLM 并提取 tool call，失败时重试。

    Returns:
        ``(tool_name, tool_args, assistant_content, assistant_msg_dict)`` 或 ``None``（失败时设置 state.exit_status）。
    """
    for retry in range(retries + 1):
        with span_context(
            'agent.inference',
            attributes={
                'type': 'agent_inference',
                'iteration': iteration,
                'retry': retry,
            },
        ):
            try:
                response = await client.create_completion(messages=messages, tools=tool_schemas)
                # 记录 token 用量用于全量跑分成本核算；不参与任务评分逻辑。
                _add_completion_usage(state.usage, getattr(response, 'usage', None))
            except ContextLengthExceededError:
                state.exit_status = NaviExitStatus.CONTEXT_LENGTH_EXCEEDED
                logger.warning('[iter %d/%d] Context length exceeded — stopping', iteration, max_iterations)
                return None
            except (asyncio.TimeoutError, TimeoutError):
                state.exit_status = NaviExitStatus.INFERENCE_TIMEOUT
                logger.warning('[iter %d/%d] Inference timed out — stopping', iteration, max_iterations)
                return None
            except Exception as exc:
                state.exit_status = NaviExitStatus.INFERENCE_ERROR
                if _is_non_retryable_llm_request_error(exc):
                    # 非瞬时 LLM 请求错误直接结束；继续 retry 只会把明确的配置/额度问题伪装成长时间卡顿。
                    logger.warning(
                        '[iter %d/%d] Inference request rejected by provider policy or request validation — stopping',
                        iteration,
                        max_iterations,
                    )
                    return None
                logger.exception('[iter %d/%d] Inference call failed: %s', iteration, max_iterations, exc)
                return None

        message = response.choices[0].message
        assistant_content = str(message.content or '')

        logger.debug(
            '[iter %d/%d][retry %d] LLM content: %s',
            iteration,
            max_iterations,
            retry,
            assistant_content[:300],
        )

        # 检查是否产生了 tool_calls
        if not message.tool_calls:
            if retry < retries:
                logger.warning(
                    '[iter %d/%d] No tool calls in response, retry %d/%d',
                    iteration,
                    max_iterations,
                    retry + 1,
                    retries,
                )
                continue
            logger.warning(
                '[iter %d/%d] No tool calls after %d retries — stopping',
                iteration,
                max_iterations,
                retries,
            )
            state.exit_status = NaviExitStatus.TOOL_CALL_ERROR
            return None

        # 解析第一个 tool call
        tool_call = message.tool_calls[0]
        try:
            tool_args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            if retry < retries:
                logger.warning(
                    '[iter %d/%d] Invalid JSON in tool arguments, retry %d/%d',
                    iteration,
                    max_iterations,
                    retry + 1,
                    retries,
                )
                continue
            logger.warning(
                '[iter %d/%d] Invalid JSON in tool arguments after %d retries — stopping',
                iteration,
                max_iterations,
                retries,
            )
            state.exit_status = NaviExitStatus.TOOL_CALL_ERROR
            return None

        return tool_call.function.name, tool_args, assistant_content, message.model_dump(exclude_none=True)

    # Should not reach here, but just in case
    state.exit_status = NaviExitStatus.TOOL_CALL_ERROR
    return None


def _add_completion_usage(total: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _save_agent_input(trace_path: Path, agent_input: NaviAgentInput) -> None:
    """保存 agent input 配置到 trace 目录（同步，仅在初始化时调用一次）。"""
    try:
        data = agent_input.model_dump(exclude={'trace_dir'})
        (trace_path / 'agent_input.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        logger.warning('Failed to save agent_input to trace', exc_info=True)


async def _save_step_observation(trace_path: Path, iteration: int, obs: Any) -> None:
    """保存单步观察数据到 trace 目录（异步）。"""

    def _write():
        step_dir = trace_path / f'step_{iteration}'
        step_dir.mkdir(parents=True, exist_ok=True)

        obs.screenshot.save(step_dir / 'screenshot.png')
        obs.annotated_screenshot.save(step_dir / 'annotated.png')

        (step_dir / 'a11y_tree.xml').write_text(obs.a11y_tree, encoding='utf-8')

        meta = {
            'window_title': obs.fg_window_title,
            'window_names': obs.fg_window_names,
            'clipboard': obs.computer_clipboard,
            'a11y_entities_count': len(obs.a11y_entities),
            'screen_size': list(obs.screenshot.size),
        }
        (step_dir / 'observation.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    try:
        await asyncio.to_thread(_write)
    except Exception:
        logger.warning('Failed to save trace observation for step %d', iteration, exc_info=True)


async def _save_step_trajectory(
    trace_path: Path,
    iteration: int,
    history: list[dict[str, Any]],
) -> None:
    """保存当前完整 history 到 trace 目录（异步），去掉图片内容。"""

    def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """深拷贝 messages，将 image_url 替换为占位符，将 tool_calls arguments 解析为对象。"""
        stripped = []
        for msg in messages:
            new_msg = dict(msg)
            # 处理 content 中的图片
            content = msg.get('content')
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'image_url':
                        new_content.append({'type': 'image_url', 'image_url': {'url': '<image_omitted>'}})
                    else:
                        new_content.append(part)
                new_msg['content'] = new_content
            # 处理 tool_calls 中的 arguments（JSON string → object）
            tool_calls = msg.get('tool_calls')
            if isinstance(tool_calls, list):
                new_tool_calls = []
                for tc in tool_calls:
                    new_tc = dict(tc)
                    func = tc.get('function')
                    if isinstance(func, dict) and isinstance(func.get('arguments'), str):
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            new_tc['function'] = {**func, 'arguments': json.loads(func['arguments'])}
                    new_tool_calls.append(new_tc)
                new_msg['tool_calls'] = new_tool_calls
            stripped.append(new_msg)
        return stripped

    def _write():
        step_dir = trace_path / f'step_{iteration}'
        step_dir.mkdir(parents=True, exist_ok=True)

        stripped_history = _strip_images(history)
        (step_dir / 'trajectory.json').write_text(
            json.dumps(stripped_history, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    try:
        await asyncio.to_thread(_write)
    except Exception:
        logger.warning('Failed to save trace trajectory for step %d', iteration, exc_info=True)
