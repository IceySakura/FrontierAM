"""
Yb OneAgent 任务定义模块。

定义 Yb OneAgent 任务的输入、输出和 reward 逻辑。
"""

import logging
import time
import uuid

from functools import partial
from typing import Any

from pydantic import Field

from nexus.core import task_registry
from nexus.core.task import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore
from nexus.extensions.projects.yb_search.log_utils import log_trace
from nexus.trace import get_current_span_id, update_span

from .reward.total_reward import compute_score_async


_log_task = partial(log_trace, 'TASK')


class YbOneAgentTaskInput(TaskRunInput, WithAgentRunner):
    """AGTS 任务输入。"""

    prompt: list[dict] = Field(description='对话消息列表')
    ground_truth: str = Field(default='', description='参考答案')
    reward_style: str = Field(default='llm-judge', description='奖励计算方式')
    data_source: str = Field(default='agts_closed_qa', description='数据来源标识')
    extra_info: dict = Field(default_factory=dict, description='额外信息')
    run_oracle: bool = Field(default=False, description='是否运行 Oracle')
    agent_name: str = Field(default='yb_one_agent_toolproxy', description='Agent 名称')
    max_iterations: int = Field(default=2, description='最大迭代次数')
    format_type: str = Field(default='deepseek', description='工具格式类型')
    traj_id: str = Field(default='', description='轨迹唯一标识，由 HyTuner 传入，用于日志追踪')


class YbOneAgentTaskOutput(TaskRunOutput, WithScore):
    """AGTS 任务输出。"""

    response: str = Field(default='', description='模型完整响应')
    final_answer: str = Field(default='', description='提取的最终答案')
    judgment: str = Field(default='', description='评判结果')
    tool_call_count: int = Field(default=0, description='工具调用次数')
    extra_fields: dict = Field(default_factory=dict, description='传递给 HyTuner 的额外字段')


@task_registry.register('yb_one_agent_toolproxy')
async def run_yb_one_agent_toolproxy_task(task_input: YbOneAgentTaskInput) -> YbOneAgentTaskOutput:
    """
    运行 Yb OneAgent 任务。

    流程：
    1. 从 prompt 提取用户问题
    2. 运行 Agent 获取回答
    3. 组装必要字段（reward 由 HyTuner 侧计算）
    4. 返回结果

    Args:
        task_input: 任务输入

    Returns:
        任务输出
    """
    # 使用 HyTuner 传入的 traj_id，或生成一个短 uuid 作为 fallback
    traj_id = task_input.traj_id or uuid.uuid4().hex[:8]
    _log_task(traj_id, 'TASK_START', f'data_source={task_input.data_source}')

    extra_info = dict(task_input.extra_info)
    extra_info['prompt'] = task_input.prompt
    fmt = task_input.format_type.lower()
    if 'yb' in fmt or 'hunyuan' in fmt:
        extra_info['model_type'] = 'yb'
    elif 'deepseek' in fmt:
        extra_info['model_type'] = 'ds'
    else:
        extra_info['model_type'] = 'qwen'
    extra_info['traj_id'] = traj_id  # 传递给 reward 计算

    # hunyuan_reward 数据是单轮评估，强制 max_iterations=1
    if 'hunyuan' in task_input.data_source:
        task_input.max_iterations = 1

    logger = logging.getLogger(__name__)

    # 默认值，agent 异常时使用
    response = ''
    final_answer = None
    tool_call_count = 0
    tool_call_list = []
    tool_result_list = []
    web_len = 0
    image_len = 0
    video_len = 0
    mini_len = 0
    num_turns = 0
    response_str = ''
    tool_timeout_count = 0
    tool_error_count = 0
    tool_status_counts: dict[str, int] = {}
    max_queries_per_turn: dict[str, int] = {}
    empty_tool_output_reasons: list[dict] = []
    observation_time = 0.0
    tool_rate_limit_wait_time = 0.0
    tool_rpc_time = 0.0
    agent_run_failed = False

    if task_input.run_oracle:
        _log_task(traj_id, 'RUN_ORACLE')
        response = task_input.ground_truth
        final_answer = task_input.ground_truth
        response_str = task_input.ground_truth
    else:
        try:
            _log_task(traj_id, 'AGENT_RUN_START')
            agent_output = await task_input.run_agent(
                messages=task_input.prompt,
                extra_info=extra_info,
                max_iterations=task_input.max_iterations,
                format_type=task_input.format_type,
                agent_id=traj_id,  # 传递给 agent 用于日志
            )
            _log_task(traj_id, 'AGENT_RUN_DONE')

            response = getattr(agent_output, 'response_str', '')
            tool_call_list = getattr(agent_output, 'tool_call_list', [])
            tool_result_list = getattr(agent_output, 'tool_result_list', [])

            web_len = int(getattr(agent_output, 'web_len', 0) or 0)
            image_len = int(getattr(agent_output, 'image_len', 0) or 0)
            video_len = int(getattr(agent_output, 'video_len', 0) or 0)
            mini_len = int(getattr(agent_output, 'mini_len', 0) or 0)
            num_turns = int(getattr(agent_output, 'num_turns', 0) or 0)
            tool_timeout_count = int(getattr(agent_output, 'tool_timeout_count', 0) or 0)
            tool_error_count = int(getattr(agent_output, 'tool_error_count', 0) or 0)
            tool_status_counts = dict(getattr(agent_output, 'tool_status_counts', {}) or {})
            max_queries_per_turn = dict(getattr(agent_output, 'max_queries_per_turn', {}) or {})
            empty_tool_output_reasons = list(getattr(agent_output, 'empty_tool_output_reasons', []) or [])
            observation_time = float(getattr(agent_output, 'observation_time', 0.0) or 0.0)
            tool_rate_limit_wait_time = float(getattr(agent_output, 'tool_rate_limit_wait_time', 0.0) or 0.0)
            tool_rpc_time = float(getattr(agent_output, 'tool_rpc_time', 0.0) or 0.0)

            final_answer = getattr(agent_output, 'final_answer', None)
            tool_call_count = getattr(agent_output, 'tool_call_count', 0)
            response_str = getattr(agent_output, 'response_str', '')
        except Exception as e:
            logger.error(f'Agent run failed for {traj_id}: {type(e).__name__}: {e}')
            _log_task(traj_id, 'AGENT_RUN_FAILED', f'{type(e).__name__}: {e}')
            agent_run_failed = True

    # [对齐 baseline] 将 agent 返回的字段补充到 extra_info 中，供 reward 计算使用
    extra_info['tool_call_list'] = tool_call_list
    extra_info['tool_result_list'] = tool_result_list
    extra_info['web_len'] = web_len
    extra_info['image_len'] = image_len
    extra_info['video_len'] = video_len
    extra_info['mini_len'] = mini_len
    extra_info['tool_call_count'] = tool_call_count
    extra_info['num_turns'] = num_turns
    extra_info['tool_timeout_count'] = tool_timeout_count
    extra_info['tool_error_count'] = tool_error_count
    extra_info['tool_status_counts'] = tool_status_counts
    extra_info['max_queries_per_turn'] = max_queries_per_turn
    extra_info['tool_rate_limit_wait_time'] = tool_rate_limit_wait_time
    extra_info['tool_rpc_time'] = tool_rpc_time

    _log_task(traj_id, 'REWARD_CALC_START')
    reward_start = time.time()
    reward_error_count = 0
    reward_timeout_count = 0
    reward_details: dict[str, Any]
    try:
        reward_details = await compute_score_async(
            solution_str=response_str,
            ground_truth=task_input.ground_truth,
            extra_info=extra_info,
            data_source=task_input.data_source,
        )
    except Exception as e:
        logger.error(f'Reward compute failed for {traj_id}: {type(e).__name__}: {e}')
        _log_task(traj_id, 'REWARD_CALC_FAILED', f'{type(e).__name__}: {e}')
        reward_details = {
            'score': -200.0,
            'judgment': '',
            'reward_error_count': 1,
            'reward_timeout_count': 1 if 'timeout' in type(e).__name__.lower() else 0,
            'reward_error_sources': [f'compute_score_async:{type(e).__name__}'],
        }
    score = float(reward_details.get('score', 0.0) or 0.0)
    judgment = str(reward_details.get('judgment', '') or '')
    reward_error_count = int(reward_details.get('reward_error_count', 0) or 0)
    reward_timeout_count = int(reward_details.get('reward_timeout_count', 0) or 0)
    reward_error_sources = reward_details.get('reward_error_sources', [])
    reward_time = time.time() - reward_start
    reward_details['reward_time'] = reward_time
    reward_details.setdefault('reward_rate_limit_wait_time', 0.0)

    format_error_count = int(tool_status_counts.get('format_error', 0) or 0)
    timeout_count = int(tool_status_counts.get('timeout', 0) or 0)
    infra_error_count = sum(
        int(tool_status_counts.get(category, 0) or 0)
        for category in (
            'error',
            'transport_error',
            'config_error',
            'tool_not_found',
            'rate_limited',
            'partial_success',
        )
    )

    termination_reason = 'ENV_DONE'
    if agent_run_failed:
        score = -200.0
        reward_details['score'] = score
        termination_reason = 'UNHANDLED_ERROR'
    elif reward_error_count > 0:
        score = -200.0
        reward_details['score'] = score
        termination_reason = 'REWARD_ERROR'
    elif timeout_count > 0 or tool_timeout_count > 0:
        score = -200.0
        reward_details['score'] = score
        termination_reason = 'ENV_TIMEOUT'
    elif format_error_count > 0:
        score = -1.0
        reward_details['score'] = score
        termination_reason = 'FORMAT_ERROR'
    elif infra_error_count > 0 or tool_error_count > 0:
        score = -200.0
        reward_details['score'] = score
        termination_reason = 'ENV_ERROR'

    done_detail = f'score={score}, judgment={judgment}, {reward_time=:.2f}, termination={termination_reason}'
    if tool_error_count or tool_timeout_count:
        done_detail += f', tool_errors={tool_error_count}, tool_timeouts={tool_timeout_count}'
    if tool_status_counts:
        done_detail += f', tool_status={tool_status_counts}'
    if reward_error_sources:
        done_detail += f', reward_errors={reward_error_sources}'
    _log_task(traj_id, 'TASK_DONE', done_detail)

    reward_details['termination_reason'] = termination_reason
    reward_details['tool_timeout_count'] = tool_timeout_count
    reward_details['tool_error_count'] = tool_error_count
    reward_details['tool_rate_limit_wait_time'] = tool_rate_limit_wait_time
    reward_details['tool_rpc_time'] = tool_rpc_time
    reward_details['empty_tool_output_count'] = len(empty_tool_output_reasons)
    reward_details['empty_tool_output_reasons'] = empty_tool_output_reasons

    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                'hytuner.reward_time': reward_time,
                'hytuner.observation_time': observation_time,
                'hytuner.termination_reason': termination_reason,
            },
        )

    return YbOneAgentTaskOutput(
        score=score,
        response=response,
        final_answer=final_answer or '',
        judgment=judgment,
        tool_call_count=tool_call_count,
        extra_fields={
            'reward_extra_info': reward_details,
            'tool_call_counts': tool_call_count,
            'tool_timeout_count': tool_timeout_count,
            'tool_error_count': tool_error_count,
            'tool_status_counts': tool_status_counts,
            'max_queries_per_turn': max_queries_per_turn,
            'tool_rate_limit_wait_time': tool_rate_limit_wait_time,
            'tool_rpc_time': tool_rpc_time,
            'reward_error_count': reward_error_count,
            'reward_timeout_count': reward_timeout_count,
            'termination_reason': termination_reason,
        },
    )
